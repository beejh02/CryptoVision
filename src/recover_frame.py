import argparse
from pathlib import Path

import cv2

from src.config import MASTER_KEY_PATH
from src.sidecar_crypto import decrypt_roi_png, load_or_create_master_key
from src.sidecar_writer import read_roi_records_for_frame


def overlay_roi(
    frame_bgr,
    roi_bgr,
    bbox: list[int],
):
    x1, y1, x2, y2 = bbox

    target_w = x2 - x1
    target_h = y2 - y1

    if target_w <= 0 or target_h <= 0:
        return

    if roi_bgr.shape[1] != target_w or roi_bgr.shape[0] != target_h:
        roi_bgr = cv2.resize(roi_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    frame_bgr[y1:y2, x1:x2] = roi_bgr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--public-video", type=str, required=True)
    parser.add_argument("--sidecar", type=str, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    master_key = load_or_create_master_key(MASTER_KEY_PATH)

    cap = cv2.VideoCapture(args.public_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open public video: {args.public_video}")

    target_index = args.frame_id - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame_id={args.frame_id}")

    records = read_roi_records_for_frame(args.sidecar, args.frame_id)

    for record in records:
        aad = {
            "version": record["version"],
            "frame_id": record["frame_id"],
            "det_index": record["det_index"],
            "class_name": record["class_name"],
            "bbox": record["bbox"],
            "conf": record["conf"],
            "roi_height": record["roi_height"],
            "roi_width": record["roi_width"],
            "channels": record["channels"],
        }

        roi = decrypt_roi_png(
            encrypted=record["encrypted"],
            master_key=master_key,
            aad=aad,
        )

        overlay_roi(frame, roi, record["bbox"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), frame)

    print(f"Recovered frame saved: {out_path}")
    print(f"Recovered ROI count: {len(records)}")


if __name__ == "__main__":
    main()
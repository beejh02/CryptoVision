import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import DEFAULT_CODEC, MASTER_KEY_PATH
from src.recover_frame import overlay_roi
from src.sidecar_crypto import decrypt_roi_png, load_or_create_master_key
from src.sidecar_writer import read_sidecar_records
from src.video_io import create_video_writer


def group_roi_records_by_frame(sidecar_path: str):
    records = read_sidecar_records(sidecar_path)

    grouped = defaultdict(list)

    for record in records:
        if record.get("type") != "roi":
            continue

        grouped[int(record["frame_id"])].append(record)

    return grouped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--public-video", type=str, required=True)
    parser.add_argument("--sidecar", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    master_key = load_or_create_master_key(MASTER_KEY_PATH)

    grouped_records = group_roi_records_by_frame(args.sidecar)

    cap = cv2.VideoCapture(args.public_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open public video: {args.public_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = create_video_writer(
        output_path=str(out_path),
        width=width,
        height=height,
        fps=fps,
        codec=DEFAULT_CODEC,
    )

    frame_id = 0

    with tqdm(total=total_frames, desc="Recovering video") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1

            records = grouped_records.get(frame_id, [])

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

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()

    print(f"Recovered video saved: {out_path}")


if __name__ == "__main__":
    main()
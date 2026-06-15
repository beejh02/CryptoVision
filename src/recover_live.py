import argparse
import time
from pathlib import Path

import cv2

from src.config import (
    DEFAULT_CODEC,
    DEFAULT_CONF_THRES,
    DEFAULT_FACE_MARGIN_RATIO,
    DEFAULT_FPS,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU_THRES,
    DEFAULT_PUBLIC_MASK_MODE,
    DEFAULT_SESSION_NAME,
    MASTER_KEY_PATH,
    RUNS_DIR,
    YOLO26_FACE_WEIGHT,
)
from src.privacy_policy import get_policy_for_class
from src.public_renderer import crop_roi, expand_and_clip_bbox, render_public_frame
from src.sidecar_crypto import encrypt_roi_png, load_or_create_master_key
from src.sidecar_writer import SidecarWriter
from src.video_io import create_video_writer, open_capture
from src.yolo26_face_detector import YOLO26FaceDetector


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--csi", action="store_true")

    parser.add_argument("--weights", type=str, default=str(YOLO26_FACE_WEIGHT))
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--session", type=str, default=DEFAULT_SESSION_NAME)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRES)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRES)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)

    parser.add_argument("--mask", type=str, default=DEFAULT_PUBLIC_MASK_MODE, choices=["black", "blur", "pixelate"])
    parser.add_argument("--margin", type=float, default=DEFAULT_FACE_MARGIN_RATIO)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)

    return parser


def main():
    args = build_argparser().parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO26n-face weight not found: {weights_path}\n"
            f"Place your yolo26n-face.pt at: {YOLO26_FACE_WEIGHT}"
        )

    session_dir = RUNS_DIR / args.session
    public_dir = session_dir / "public"
    private_dir = session_dir / "private"

    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    public_video_path = public_dir / "public.mp4"
    sidecar_path = private_dir / "private.sidecar.jsonl"

    master_key = load_or_create_master_key(MASTER_KEY_PATH)

    detector = YOLO26FaceDetector(
        weight_path=str(weights_path),
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    cap = open_capture(
        source=args.source,
        csi=args.csi,
        width=args.width,
        height=args.height,
        fps=int(args.fps),
    )

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame.")

    frame_h, frame_w = first_frame.shape[:2]

    writer = create_video_writer(
        output_path=str(public_video_path),
        width=frame_w,
        height=frame_h,
        fps=args.fps,
        codec=DEFAULT_CODEC,
    )

    frame_id = 0
    start_time = time.time()
    last_fps = 0.0

    with SidecarWriter(sidecar_path) as sidecar:
        sidecar.write_header(
            {
                "version": "1.0",
                "project": "EdgeGuard-ROI",
                "detector": "YOLO26n-face",
                "algorithm": "ChaCha20-Poly1305",
                "public_video": str(public_video_path.name),
                "created_at": time.time(),
                "frame_width": frame_w,
                "frame_height": frame_h,
                "fps": args.fps,
                "mask_mode": args.mask,
            }
        )

        pending_frame = first_frame

        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            frame_id += 1
            t0 = time.time()

            detections = detector.detect(frame)

            masked_bboxes = []
            stored_count = 0

            for det_index, det in enumerate(detections):
                policy = get_policy_for_class(det.class_name)
                if policy is None:
                    continue

                bbox = expand_and_clip_bbox(
                    bbox=det.bbox,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    margin_ratio=args.margin,
                )

                if bbox is None:
                    continue

                masked_bboxes.append((bbox, args.mask))

                if policy.store_original and policy.encrypt_original:
                    roi = crop_roi(frame, bbox)

                    aad = {
                        "version": "1.0",
                        "frame_id": frame_id,
                        "det_index": det_index,
                        "class_name": det.class_name,
                        "bbox": bbox,
                        "conf": round(det.conf, 6),
                        "roi_height": int(roi.shape[0]),
                        "roi_width": int(roi.shape[1]),
                        "channels": int(roi.shape[2]),
                    }

                    encrypted = encrypt_roi_png(
                        roi_bgr=roi,
                        master_key=master_key,
                        aad=aad,
                    )

                    sidecar.write_roi_record(
                        {
                            **aad,
                            "encrypted": encrypted,
                        }
                    )

                    stored_count += 1

            public_frame = render_public_frame(frame, masked_bboxes)
            writer.write(public_frame)

            dt = time.time() - t0
            fps_now = 1.0 / dt if dt > 0 else 0.0
            last_fps = fps_now if last_fps == 0 else last_fps * 0.9 + fps_now * 0.1

            if args.display:
                preview = public_frame.copy()
                cv2.putText(
                    preview,
                    f"YOLO26n-face | frame={frame_id} | faces={stored_count} | FPS={last_fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("EdgeGuard-ROI Public Preview", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0.0

    print("Done.")
    print(f"Frames: {frame_id}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Public video: {public_video_path}")
    print(f"Private sidecar: {sidecar_path}")
    print(f"Master key: {MASTER_KEY_PATH}")


if __name__ == "__main__":
    main()
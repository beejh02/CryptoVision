from typing import Tuple

import cv2
import numpy as np


def expand_and_clip_bbox(
    bbox: list[int],
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> list[int] | None:
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(frame_width, x2 + mx)
    y2 = min(frame_height, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def crop_roi(frame_bgr: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame_bgr[y1:y2, x1:x2].copy()


def apply_black_mask(frame_bgr: np.ndarray, bbox: list[int]) -> None:
    x1, y1, x2, y2 = bbox
    frame_bgr[y1:y2, x1:x2] = 0


def apply_blur_mask(frame_bgr: np.ndarray, bbox: list[int]) -> None:
    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return

    h, w = roi.shape[:2]
    k = max(15, ((min(w, h) // 5) | 1))
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame_bgr[y1:y2, x1:x2] = blurred


def apply_pixelate_mask(frame_bgr: np.ndarray, bbox: list[int], block_size: int = 12) -> None:
    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return

    h, w = roi.shape[:2]

    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    frame_bgr[y1:y2, x1:x2] = pixelated


def render_public_frame(
    original_frame_bgr: np.ndarray,
    masked_bboxes: list[Tuple[list[int], str]],
) -> np.ndarray:
    """
    original frame은 건드리지 않고 public frame만 생성한다.

    masked_bboxes:
        [
            ([x1, y1, x2, y2], "black"),
            ([x1, y1, x2, y2], "blur"),
        ]
    """
    public_frame = original_frame_bgr.copy()

    for bbox, action in masked_bboxes:
        if action == "black":
            apply_black_mask(public_frame, bbox)
        elif action == "blur":
            apply_blur_mask(public_frame, bbox)
        elif action == "pixelate":
            apply_pixelate_mask(public_frame, bbox)
        else:
            apply_black_mask(public_frame, bbox)

    return public_frame


def draw_debug_boxes(
    frame_bgr: np.ndarray,
    bboxes: list[list[int]],
    label: str = "face",
) -> np.ndarray:
    debug = frame_bgr.copy()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return debug
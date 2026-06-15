from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: list[int]          # [x1, y1, x2, y2]
    conf: float
    class_id: int
    class_name: str


class YOLO26FaceDetector:
    """
    YOLO26n-face 전용 detector.

    입력:
        frame_bgr: OpenCV BGR 이미지

    출력:
        Detection 리스트
    """

    def __init__(
        self,
        weight_path: str,
        device: str = "0",
        imgsz: int = 640,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
    ):
        self.weight_path = weight_path
        self.device = device
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = YOLO(weight_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []

        names = result.names if result.names else {}

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            conf = float(box.conf[0].detach().cpu().item())
            cls_id = int(box.cls[0].detach().cpu().item())

            x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]

            # face 전용 모델이라 names가 비어있거나 이상해도 face로 고정
            class_name = names.get(cls_id, "face")
            if class_name != "face":
                class_name = "face"

            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    conf=conf,
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

        return detections
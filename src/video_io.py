import cv2


def parse_video_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def build_csi_gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1"
    )


def open_capture(
    source: str,
    csi: bool = False,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
):
    if csi:
        pipeline = build_csi_gstreamer_pipeline(width, height, fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(parse_video_source(source))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source.")

    return cap


def create_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str = "mp4v",
):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    return writer
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyPolicy:
    class_name: str
    public_action: str
    store_original: bool
    encrypt_original: bool
    margin_ratio: float


# YOLO26n-face 전용 정책
FACE_POLICY = PrivacyPolicy(
    class_name="face",
    public_action="black",
    store_original=True,
    encrypt_original=True,
    margin_ratio=0.08,
)


def get_policy_for_class(class_name: str) -> PrivacyPolicy | None:
    """
    현재 버전은 YOLO26n-face 전용.
    class_name이 face가 아니면 처리하지 않는다.
    """
    if class_name == "face":
        return FACE_POLICY

    return None
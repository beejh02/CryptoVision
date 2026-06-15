import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes


def load_or_create_master_key(key_path: str | Path) -> bytes:
    """
    32-byte master key를 hex로 저장한다.
    이 파일은 절대 GitHub에 올리면 안 된다.
    """
    key_path = Path(key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        key_hex = key_path.read_text(encoding="utf-8").strip()
        key = bytes.fromhex(key_hex)

        if len(key) != 32:
            raise ValueError("master.key must be 32 bytes.")

        return key

    key = get_random_bytes(32)
    key_path.write_text(key.hex(), encoding="utf-8")

    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass

    return key


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """
    AEAD AAD용 canonical JSON.
    같은 데이터면 항상 같은 byte sequence가 나오도록 정렬한다.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def encode_roi_image_png(roi_bgr: np.ndarray) -> bytes:
    """
    원본 ROI를 PNG로 인코딩한 뒤 암호화한다.
    raw bytes보다 sidecar 크기를 줄이고, 복구 시 shape 관리가 쉬워진다.
    PNG는 lossless라 복구 가능하다.
    """
    import cv2

    ok, encoded = cv2.imencode(".png", roi_bgr)

    if not ok:
        raise RuntimeError("Failed to encode ROI as PNG.")

    return encoded.tobytes()


def decode_roi_image_png(png_bytes: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    roi = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if roi is None:
        raise RuntimeError("Failed to decode ROI PNG.")

    return roi


def encrypt_roi_png(
    roi_bgr: np.ndarray,
    master_key: bytes,
    aad: dict[str, Any],
) -> dict[str, str]:
    """
    ROI 이미지를 PNG bytes로 만든 뒤 ChaCha20-Poly1305로 암호화한다.
    """
    plaintext = encode_roi_image_png(roi_bgr)

    nonce = get_random_bytes(12)

    cipher = ChaCha20_Poly1305.new(
        key=master_key,
        nonce=nonce,
    )

    aad_bytes = canonical_json_bytes(aad)
    cipher.update(aad_bytes)

    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return {
        "nonce": b64e(nonce),
        "ciphertext": b64e(ciphertext),
        "tag": b64e(tag),
    }


def decrypt_roi_png(
    encrypted: dict[str, str],
    master_key: bytes,
    aad: dict[str, Any],
) -> np.ndarray:
    nonce = b64d(encrypted["nonce"])
    ciphertext = b64d(encrypted["ciphertext"])
    tag = b64d(encrypted["tag"])

    cipher = ChaCha20_Poly1305.new(
        key=master_key,
        nonce=nonce,
    )

    aad_bytes = canonical_json_bytes(aad)
    cipher.update(aad_bytes)

    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    return decode_roi_image_png(plaintext)
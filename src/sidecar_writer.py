import json
from pathlib import Path
from typing import Any


class SidecarWriter:
    """
    JSONL sidecar writer.
    한 줄에 하나의 암호화 ROI record를 저장한다.
    """

    def __init__(self, sidecar_path: str | Path):
        self.sidecar_path = Path(sidecar_path)
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.sidecar_path.open("w", encoding="utf-8")

    def write_header(self, header: dict[str, Any]) -> None:
        record = {
            "type": "header",
            **header,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def write_roi_record(self, record: dict[str, Any]) -> None:
        record = {
            "type": "roi",
            **record,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.file.flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def read_sidecar_records(sidecar_path: str | Path) -> list[dict[str, Any]]:
    sidecar_path = Path(sidecar_path)

    records = []

    with sidecar_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def read_roi_records_for_frame(
    sidecar_path: str | Path,
    frame_id: int,
) -> list[dict[str, Any]]:
    records = read_sidecar_records(sidecar_path)

    result = []

    for record in records:
        if record.get("type") != "roi":
            continue

        if int(record["frame_id"]) == int(frame_id):
            result.append(record)

    return result
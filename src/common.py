from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AnnotationRecord:
    image_path: Path
    json_path: Path
    prefix: str
    boxes: list[list[float]]
    width: int
    height: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_prefix(name: str) -> str:
    return Path(name).stem.split("-", 1)[0]


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def safe_open_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if is_image_path(path))


def load_annotation_record(json_path: Path) -> AnnotationRecord:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw_image_path = Path(str(raw["imagePath"]).replace("\\", "/"))
    candidates = [
        json_path.parent / raw_image_path,
        json_path.parent / raw_image_path.name,
    ]
    image_path = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            image_path = candidate
            break
    boxes: list[list[float]] = []
    for shape in raw.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
    return AnnotationRecord(
        image_path=image_path,
        json_path=json_path,
        prefix=image_prefix(image_path.name),
        boxes=boxes,
        width=int(raw["imageWidth"]),
        height=int(raw["imageHeight"]),
    )


def load_detection_records(data_dir: Path) -> list[AnnotationRecord]:
    records = []
    for json_path in sorted(data_dir.glob("*.json")):
        record = load_annotation_record(json_path)
        if record.image_path.exists() and record.boxes:
            records.append(record)
    return records


def grouped_holdout_query_names(query_dir: Path, count_per_prefix: int = 2) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for path in list_images(query_dir):
        grouped.setdefault(image_prefix(path.name), []).append(path.name)
    holdout = []
    for prefix in sorted(grouped):
        holdout.extend(sorted(grouped[prefix])[:count_per_prefix])
    return holdout


def split_records_by_prefix(
    records: Iterable[AnnotationRecord],
    holdout_names: set[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[AnnotationRecord], list[AnnotationRecord], list[AnnotationRecord]]:
    holdout: list[AnnotationRecord] = []
    grouped: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        if record.image_path.name in holdout_names:
            holdout.append(record)
        else:
            grouped.setdefault(record.prefix, []).append(record)

    rng = random.Random(seed)
    train: list[AnnotationRecord] = []
    val: list[AnnotationRecord] = []
    for prefix in sorted(grouped):
        items = grouped[prefix][:]
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio))
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    return train, val, sorted(holdout, key=lambda record: record.image_path.name)

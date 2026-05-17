from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from src.config import Config

_SAM_SOURCES = {"sam_auto", "sam_point", "roboflow"}
_OCCLUSION_SUBDIRS = ("rendah", "sedang", "tinggi")


class OcclusionLevel(Enum):
    RENDAH = "rendah"   # oklusi < 30%
    SEDANG = "sedang"   # oklusi 30–60%
    TINGGI = "tinggi"   # oklusi > 60%


@dataclass
class ImageEntry:
    index: int
    filename: str
    filepath: str
    width: int
    height: int
    annotated: bool
    auto_labeled: bool
    annotation_path: str
    split: str | None


@dataclass
class Polygon:
    instance_id: int
    class_id: int
    class_name: str
    points: list[tuple[float, float]]
    color: list[int]
    source: str
    confidence: float
    occlusion_level: OcclusionLevel | None = None
    occlusion_ratio: float | None = None
    occlusion_manual: bool = False
    is_confirmed: bool = False


@dataclass
class AnnotationProgress:
    total: int
    annotated: int
    manual: int
    auto_labeled: int
    unreviewed_auto: int


@dataclass
class DatasetSplit:
    train: list[ImageEntry]
    val: list[ImageEntry]
    test: list[ImageEntry]
    seed: int
    created_at: str


class ImageManager:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._images: list[ImageEntry] = []
        self._current_idx: int = 0
        self._session_saved: set[int] = set()

    def load_folder(self, path: str) -> int:
        folder = Path(path)
        extensions = set(self._config.image_extensions)

        files = sorted(
            [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions],
            key=lambda f: f.name,
        )

        annotations_dir = Path(self._config.paths.annotations)
        self._images = []
        self._session_saved = set()

        for idx, filepath in enumerate(files):
            stem = filepath.stem

            with Image.open(filepath) as img:
                width, height = img.size

            annotated = False
            auto_labeled = False
            annotation_path = self._find_annotation(annotations_dir, stem)
            if annotation_path is None:
                annotation_path = annotations_dir / f"{stem}.json"

            if annotation_path.exists():
                try:
                    with open(annotation_path, encoding="utf-8") as f:
                        data = json.load(f)
                    raw_polygons = data.get("polygons", [])
                    if raw_polygons:
                        annotated = True
                        auto_labeled = all(p.get("source", "") in _SAM_SOURCES for p in raw_polygons)
                except (json.JSONDecodeError, KeyError):
                    pass

            self._images.append(
                ImageEntry(
                    index=idx,
                    filename=filepath.name,
                    filepath=str(filepath),
                    width=width,
                    height=height,
                    annotated=annotated,
                    auto_labeled=auto_labeled,
                    annotation_path=str(annotation_path),
                    split=None,
                )
            )

        self._current_idx = 0
        return len(self._images)

    def current_image(self) -> ImageEntry:
        return self._images[self._current_idx]

    def next(self) -> ImageEntry:
        self._current_idx = (self._current_idx + 1) % len(self._images)
        return self._images[self._current_idx]

    def prev(self) -> ImageEntry:
        self._current_idx = (self._current_idx - 1) % len(self._images)
        return self._images[self._current_idx]

    def go_to(self, index: int) -> ImageEntry:
        if not 0 <= index < len(self._images):
            raise IndexError(f"Index {index} out of range (0–{len(self._images) - 1})")
        self._current_idx = index
        return self._images[self._current_idx]

    def get_all(self) -> list[ImageEntry]:
        return list(self._images)

    def get_progress(self) -> AnnotationProgress:
        total = len(self._images)
        annotated = sum(1 for e in self._images if e.annotated)
        auto_labeled_count = sum(1 for e in self._images if e.auto_labeled)
        manual = sum(1 for e in self._images if e.annotated and not e.auto_labeled)
        unreviewed_auto = sum(
            1 for e in self._images
            if e.auto_labeled and e.index not in self._session_saved
        )
        return AnnotationProgress(
            total=total,
            annotated=annotated,
            manual=manual,
            auto_labeled=auto_labeled_count,
            unreviewed_auto=unreviewed_auto,
        )

    def save_annotation(self, entry: ImageEntry, polygons: list[Polygon]) -> None:
        base_dir = Path(self._config.paths.annotations)
        subfolder = self._occlusion_subfolder(polygons)
        save_dir = base_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(entry.filepath).stem
        save_path = save_dir / f"{stem}.json"

        data = {
            "filename": entry.filename,
            "width": entry.width,
            "height": entry.height,
            "polygons": [
                {
                    "instance_id": p.instance_id,
                    "class_id": p.class_id,
                    "class_name": p.class_name,
                    "points": [list(pt) for pt in p.points],
                    "color": p.color,
                    "source": p.source,
                    "confidence": p.confidence,
                    "occlusion_level": p.occlusion_level.value if p.occlusion_level else None,
                    "occlusion_ratio": p.occlusion_ratio,
                    "occlusion_manual": p.occlusion_manual,
                    "is_confirmed": p.is_confirmed,
                }
                for p in polygons
            ],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        entry.annotation_path = str(save_path)
        entry.annotated = True
        self._session_saved.add(entry.index)

    def load_annotation(self, entry: ImageEntry) -> list[Polygon]:
        base_dir = Path(self._config.paths.annotations)
        stem = Path(entry.filepath).stem
        path = self._find_annotation(base_dir, stem)
        if path is None:
            return []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for p in data.get("polygons", []):
            occlusion_level = None
            raw_level = p.get("occlusion_level")
            if raw_level:
                try:
                    occlusion_level = OcclusionLevel(raw_level)
                except ValueError:
                    pass
            result.append(Polygon(
                instance_id=p["instance_id"],
                class_id=p["class_id"],
                class_name=p["class_name"],
                points=[tuple(pt) for pt in p["points"]],
                color=p["color"],
                source=p["source"],
                confidence=p["confidence"],
                occlusion_level=occlusion_level,
                occlusion_ratio=p.get("occlusion_ratio"),
                occlusion_manual=p.get("occlusion_manual", False),
                is_confirmed=p.get("is_confirmed", False),
            ))
        return result

    def _find_annotation(self, base_dir: Path, stem: str) -> Path | None:
        """Search occlusion subfolders then flat dir for an annotation file."""
        for subdir in _OCCLUSION_SUBDIRS:
            candidate = base_dir / subdir / f"{stem}.json"
            if candidate.exists():
                return candidate
        flat = base_dir / f"{stem}.json"
        return flat if flat.exists() else None

    def _occlusion_subfolder(self, polygons: list[Polygon]) -> str:
        levels = {p.occlusion_level for p in polygons if p.occlusion_level is not None}
        if OcclusionLevel.TINGGI in levels:
            return "tinggi"
        if OcclusionLevel.SEDANG in levels:
            return "sedang"
        return "rendah"

    def is_annotated(self, entry: ImageEntry) -> bool:
        return entry.annotated

    def generate_split(self) -> DatasetSplit:
        split_path = Path(self._config.paths.annotations) / "split.json"

        if split_path.exists():
            return self._load_split_file(split_path)

        seed = self._config.dataset_split.seed
        train_ratio = self._config.dataset_split.train
        val_ratio = self._config.dataset_split.val

        filenames = [e.filename for e in self._images]
        rng = random.Random(seed)
        rng.shuffle(filenames)

        n = len(filenames)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)

        split_data = {
            "seed": seed,
            "created_at": date.today().isoformat(),
            "train": filenames[:n_train],
            "val": filenames[n_train:n_train + n_val],
            "test": filenames[n_train + n_val:],
        }

        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        return self._build_dataset_split(split_data)

    def get_split(self) -> DatasetSplit | None:
        split_path = Path(self._config.paths.annotations) / "split.json"
        if not split_path.exists():
            return None
        return self._load_split_file(split_path)

    def _load_split_file(self, path: Path) -> DatasetSplit:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self._build_dataset_split(data)

    def _build_dataset_split(self, data: dict) -> DatasetSplit:
        name_to_entry = {e.filename: e for e in self._images}

        train = [name_to_entry[n] for n in data["train"] if n in name_to_entry]
        val = [name_to_entry[n] for n in data["val"] if n in name_to_entry]
        test = [name_to_entry[n] for n in data["test"] if n in name_to_entry]

        for entry in train:
            entry.split = "train"
        for entry in val:
            entry.split = "val"
        for entry in test:
            entry.split = "test"

        return DatasetSplit(
            train=train,
            val=val,
            test=test,
            seed=data["seed"],
            created_at=data["created_at"],
        )

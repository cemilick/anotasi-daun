"""Tests for src/image_manager.py"""
import json
import types
from pathlib import Path

import pytest
from PIL import Image

from src.image_manager import (
    AnnotationProgress,
    DatasetSplit,
    ImageEntry,
    ImageManager,
    Polygon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, extensions=None, split=None):
    """Build a minimal Config-like object backed by tmp_path."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]
    if split is None:
        split = {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42}

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    paths_ns = types.SimpleNamespace(annotations=annotations_dir)
    split_ns = types.SimpleNamespace(**split)

    cfg = types.SimpleNamespace(
        image_extensions=extensions,
        paths=paths_ns,
        dataset_split=split_ns,
    )
    return cfg


def _make_image(path: Path, width: int = 100, height: int = 80) -> Path:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path)
    return path


def _write_annotation(annotations_dir: Path, stem: str, polygons: list[dict]) -> Path:
    data = {
        "filename": f"{stem}.jpg",
        "width": 100,
        "height": 80,
        "polygons": polygons,
    }
    path = annotations_dir / f"{stem}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _manual_polygon(instance_id: int = 1) -> dict:
    return {
        "instance_id": instance_id,
        "class_id": 1,
        "class_name": "daun",
        "points": [[10, 20], [30, 20], [30, 40]],
        "color": [255, 0, 0],
        "source": "manual",
        "confidence": 1.0,
    }


def _sam_polygon(instance_id: int = 1, source: str = "sam_auto") -> dict:
    return {
        "instance_id": instance_id,
        "class_id": 1,
        "class_name": "daun",
        "points": [[5, 5], [50, 5], [50, 50]],
        "color": [0, 255, 0],
        "source": source,
        "confidence": 0.92,
    }


# ---------------------------------------------------------------------------
# Task 7.1 — load_folder
# ---------------------------------------------------------------------------

class TestLoadFolder:
    def test_returns_correct_count(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for name in ["a.jpg", "b.png", "c.jpeg"]:
            _make_image(images_dir / name)

        mgr = ImageManager(cfg)
        count = mgr.load_folder(str(images_dir))
        assert count == 3

    def test_reads_dimensions(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg", width=200, height=150)

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        entry = mgr.current_image()

        assert entry.width == 200
        assert entry.height == 150

    def test_sorted_by_filename(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            _make_image(images_dir / name)

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        names = [e.filename for e in mgr.get_all()]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]

    def test_annotated_flag_set_when_json_exists(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg")
        _write_annotation(tmp_path / "annotations", "leaf", [_manual_polygon()])

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        entry = mgr.current_image()

        assert entry.annotated is True
        assert entry.auto_labeled is False

    def test_auto_labeled_flag_set_for_sam_only(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg")
        _write_annotation(tmp_path / "annotations", "leaf", [_sam_polygon()])

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        entry = mgr.current_image()

        assert entry.annotated is True
        assert entry.auto_labeled is True

    def test_empty_folder_returns_zero(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mgr = ImageManager(cfg)
        count = mgr.load_folder(str(images_dir))
        assert count == 0

    def test_ignores_non_image_files(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg")
        (images_dir / "readme.txt").write_text("hello")

        mgr = ImageManager(cfg)
        count = mgr.load_folder(str(images_dir))
        assert count == 1


# ---------------------------------------------------------------------------
# Task 7.2 — Navigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def _setup(self, tmp_path, count: int = 4):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(count):
            _make_image(images_dir / f"img_{i:02d}.jpg")

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        return mgr

    def test_next_advances_index(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        entry = mgr.next()
        assert entry.index == 1

    def test_next_wraps_from_last_to_first(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        mgr.go_to(3)
        entry = mgr.next()
        assert entry.index == 0

    def test_prev_wraps_from_first_to_last(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        entry = mgr.prev()
        assert entry.index == 3

    def test_prev_goes_backward(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        mgr.go_to(2)
        entry = mgr.prev()
        assert entry.index == 1

    def test_go_to_specific_index(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        entry = mgr.go_to(2)
        assert entry.index == 2

    def test_go_to_invalid_raises(self, tmp_path):
        mgr = self._setup(tmp_path, 4)
        with pytest.raises(IndexError):
            mgr.go_to(10)


# ---------------------------------------------------------------------------
# Task 7.3 — save/load annotation round-trip
# ---------------------------------------------------------------------------

class TestAnnotationPersistence:
    def _setup(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg")

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        return mgr

    def test_save_creates_json_file(self, tmp_path):
        mgr = self._setup(tmp_path)
        entry = mgr.current_image()
        polygon = Polygon(
            instance_id=1, class_id=1, class_name="daun",
            points=[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0)],
            color=[255, 0, 0], source="manual", confidence=1.0,
        )
        mgr.save_annotation(entry, [polygon])
        assert Path(entry.annotation_path).exists()

    def test_is_annotated_true_after_save(self, tmp_path):
        mgr = self._setup(tmp_path)
        entry = mgr.current_image()
        assert mgr.is_annotated(entry) is False

        polygon = Polygon(
            instance_id=1, class_id=1, class_name="daun",
            points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            color=[0, 255, 0], source="manual", confidence=1.0,
        )
        mgr.save_annotation(entry, [polygon])
        assert mgr.is_annotated(entry) is True

    def test_load_annotation_returns_empty_when_no_file(self, tmp_path):
        mgr = self._setup(tmp_path)
        entry = mgr.current_image()
        result = mgr.load_annotation(entry)
        assert result == []

    def test_round_trip_manual_polygon(self, tmp_path):
        mgr = self._setup(tmp_path)
        entry = mgr.current_image()
        original = Polygon(
            instance_id=7, class_id=1, class_name="daun",
            points=[(1.5, 2.5), (3.0, 4.0)],
            color=[255, 82, 82], source="manual", confidence=1.0,
        )
        mgr.save_annotation(entry, [original])
        loaded = mgr.load_annotation(entry)

        assert len(loaded) == 1
        p = loaded[0]
        assert p.instance_id == 7
        assert p.source == "manual"
        assert p.confidence == 1.0
        assert p.points == [(1.5, 2.5), (3.0, 4.0)]
        assert p.color == [255, 82, 82]

    def test_round_trip_sam_polygon(self, tmp_path):
        mgr = self._setup(tmp_path)
        entry = mgr.current_image()
        original = Polygon(
            instance_id=2, class_id=1, class_name="daun",
            points=[(5.0, 5.0), (50.0, 10.0), (45.0, 55.0)],
            color=[0, 255, 0], source="sam_auto", confidence=0.87,
        )
        mgr.save_annotation(entry, [original])
        loaded = mgr.load_annotation(entry)

        assert len(loaded) == 1
        p = loaded[0]
        assert p.source == "sam_auto"
        assert abs(p.confidence - 0.87) < 1e-9

    def test_save_creates_annotations_dir_if_missing(self, tmp_path):
        cfg = _make_config(tmp_path)
        # remove the pre-created annotations dir
        ann_dir = tmp_path / "annotations"
        for f in ann_dir.iterdir():
            f.unlink()
        ann_dir.rmdir()

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _make_image(images_dir / "leaf.jpg")

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        entry = mgr.current_image()
        polygon = Polygon(
            instance_id=1, class_id=1, class_name="daun",
            points=[(0.0, 0.0)], color=[0, 0, 255],
            source="manual", confidence=1.0,
        )
        mgr.save_annotation(entry, [polygon])
        assert Path(entry.annotation_path).exists()


# ---------------------------------------------------------------------------
# Task 7.4 — get_progress
# ---------------------------------------------------------------------------

class TestGetProgress:
    def _setup_mixed(self, tmp_path):
        """
        10 images:
          - 4 manual annotations
          - 3 SAM-only annotations (not re-saved this session)
          - 1 SAM annotation (re-saved this session)
          - 2 unannotated
        """
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"

        for i in range(10):
            _make_image(images_dir / f"img_{i:02d}.jpg")

        # 4 manual
        for i in range(4):
            _write_annotation(ann_dir, f"img_{i:02d}", [_manual_polygon()])

        # 4 SAM-only
        for i in range(4, 8):
            _write_annotation(ann_dir, f"img_{i:02d}", [_sam_polygon()])

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))

        # re-save img_07 this session (confirms the SAM annotation)
        entry = mgr.go_to(7)
        sam_poly = Polygon(
            instance_id=1, class_id=1, class_name="daun",
            points=[(5.0, 5.0), (50.0, 5.0), (50.0, 50.0)],
            color=[0, 255, 0], source="sam_auto", confidence=0.92,
        )
        mgr.save_annotation(entry, [sam_poly])

        return mgr

    def test_progress_totals(self, tmp_path):
        mgr = self._setup_mixed(tmp_path)
        progress = mgr.get_progress()

        assert progress.total == 10
        assert progress.annotated == 8
        assert progress.manual == 4
        assert progress.auto_labeled == 4
        assert progress.unreviewed_auto == 3

    def test_progress_all_zeros_when_empty(self, tmp_path):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        p = mgr.get_progress()

        assert p.total == 0
        assert p.annotated == 0
        assert p.manual == 0
        assert p.auto_labeled == 0
        assert p.unreviewed_auto == 0


# ---------------------------------------------------------------------------
# Task 7.5 — generate_split
# ---------------------------------------------------------------------------

class TestGenerateSplit:
    def _setup(self, tmp_path, n: int = 10):
        cfg = _make_config(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(n):
            _make_image(images_dir / f"img_{i:02d}.jpg")

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        return mgr, cfg

    def test_no_overlap_between_splits(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 10)
        split = mgr.generate_split()

        train_names = {e.filename for e in split.train}
        val_names = {e.filename for e in split.val}
        test_names = {e.filename for e in split.test}

        assert train_names & val_names == set()
        assert train_names & test_names == set()
        assert val_names & test_names == set()

    def test_all_images_covered(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 10)
        split = mgr.generate_split()

        all_names = {e.filename for e in mgr.get_all()}
        split_names = (
            {e.filename for e in split.train}
            | {e.filename for e in split.val}
            | {e.filename for e in split.test}
        )
        assert split_names == all_names

    def test_approximate_ratios(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 100)
        split = mgr.generate_split()

        assert len(split.train) == 70
        assert len(split.val) == 20
        assert len(split.test) == 10

    def test_idempotent_second_call(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 10)
        split1 = mgr.generate_split()
        split2 = mgr.generate_split()

        assert [e.filename for e in split1.train] == [e.filename for e in split2.train]
        assert [e.filename for e in split1.val] == [e.filename for e in split2.val]
        assert [e.filename for e in split1.test] == [e.filename for e in split2.test]

    def test_split_json_written(self, tmp_path):
        mgr, cfg = self._setup(tmp_path, 10)
        mgr.generate_split()
        split_path = Path(cfg.paths.annotations) / "split.json"
        assert split_path.exists()

    def test_get_split_returns_none_before_generate(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 5)
        assert mgr.get_split() is None

    def test_get_split_loads_after_generate(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 10)
        mgr.generate_split()
        split = mgr.get_split()
        assert split is not None
        assert isinstance(split, DatasetSplit)

    def test_entry_split_field_updated(self, tmp_path):
        mgr, _ = self._setup(tmp_path, 10)
        mgr.generate_split()
        for entry in mgr.get_all():
            assert entry.split in {"train", "val", "test"}

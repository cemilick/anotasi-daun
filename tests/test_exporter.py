"""Tests for src/exporter.py"""
import json
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.exporter import Exporter, compute_area_shoelace, compute_bbox, polygon_to_rle
from src.image_manager import ImageEntry, OcclusionLevel, Polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(classes=None):
    if classes is None:
        classes = [{"id": 1, "name": "daun_kelengkeh_itoh", "supercategory": "plant"}]
    cfg = types.SimpleNamespace(classes=classes)
    return cfg


def _make_entry(
    index: int,
    filename: str = "img.jpg",
    width: int = 200,
    height: int = 150,
    split: str | None = None,
    annotated: bool = True,
) -> ImageEntry:
    return ImageEntry(
        index=index,
        filename=filename,
        filepath=f"/tmp/{filename}",
        width=width,
        height=height,
        annotated=annotated,
        auto_labeled=False,
        annotation_path="",
        split=split,
    )


def _make_polygon(
    points=None,
    source: str = "manual",
    confidence: float = 0.9,
    occlusion_level: OcclusionLevel | None = None,
    occlusion_ratio: float | None = None,
) -> Polygon:
    if points is None:
        points = [(10.0, 20.0), (50.0, 10.0), (60.0, 50.0), (20.0, 60.0)]
    return Polygon(
        instance_id=1,
        class_id=1,
        class_name="daun_kelengkeh_itoh",
        points=points,
        color=[255, 0, 0],
        source=source,
        confidence=confidence,
        occlusion_level=occlusion_level,
        occlusion_ratio=occlusion_ratio,
    )


class _MockImageManager:
    """Minimal ImageManager substitute for testing."""

    def __init__(self, entries: list[ImageEntry], annotation_map: dict[int, list[Polygon]]):
        self._entries = entries
        self._annotation_map = annotation_map  # index -> polygons

    def get_all(self) -> list[ImageEntry]:
        return list(self._entries)

    def load_annotation(self, entry: ImageEntry) -> list[Polygon]:
        return list(self._annotation_map.get(entry.index, []))


# ---------------------------------------------------------------------------
# Unit: utilities
# ---------------------------------------------------------------------------

class TestComputeBbox:
    def test_returns_xmin_ymin_width_height(self):
        points = [(100.0, 200.0), (150.0, 180.0), (200.0, 220.0), (160.0, 260.0)]
        bbox = compute_bbox(points)
        assert bbox == [100.0, 180.0, 100.0, 80.0]

    def test_single_rect(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
        x, y, w, h = compute_bbox(points)
        assert x == 0.0 and y == 0.0 and w == 10.0 and h == 5.0


class TestComputeAreaShoelace:
    def test_rectangle(self):
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
        assert compute_area_shoelace(points) == pytest.approx(50.0)

    def test_triangle(self):
        points = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        assert compute_area_shoelace(points) == pytest.approx(6.0)

    def test_area_less_than_bbox_for_nonrect(self):
        # Diamond: clearly less than its bounding box
        points = [(5.0, 0.0), (10.0, 5.0), (5.0, 10.0), (0.0, 5.0)]
        area = compute_area_shoelace(points)
        bbox = compute_bbox(points)
        assert area < bbox[2] * bbox[3]

    def test_degenerate_less_than_3_points(self):
        assert compute_area_shoelace([]) == 0.0
        assert compute_area_shoelace([(0.0, 0.0), (1.0, 1.0)]) == 0.0


class TestPolygonToRle:
    def test_returns_dict_with_counts_and_size(self):
        pytest.importorskip("cv2")
        pytest.importorskip("pycocotools")
        points = [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]
        rle = polygon_to_rle(points, height=100, width=80)
        assert isinstance(rle, dict)
        assert "counts" in rle and "size" in rle
        assert isinstance(rle["counts"], str)
        assert rle["size"] == [100, 80]


# ---------------------------------------------------------------------------
# Integration: COCO JSON
# ---------------------------------------------------------------------------

class TestExportCoco:
    def test_output_readable_by_pycocotools(self, tmp_path):
        pytest.importorskip("cv2")
        pycocotools = pytest.importorskip("pycocotools")
        from pycocotools.coco import COCO

        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "daun_001.jpg", width=200, height=150, split="train")
        poly = _make_polygon()
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "annotations.json"
        exporter.export_coco(mgr, str(out))

        coco = COCO(str(out))
        assert len(coco.imgs) == 1
        assert len(coco.anns) == 1

    def test_segmentation_is_rle_dict(self, tmp_path):
        pytest.importorskip("cv2")
        pytest.importorskip("pycocotools")
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon()
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=True)

        data = json.loads(out.read_text(encoding="utf-8"))
        seg = data["annotations"][0]["segmentation"]
        assert isinstance(seg, dict)
        assert "counts" in seg and isinstance(seg["counts"], str)
        assert "size" in seg and len(seg["size"]) == 2

    def test_segmentation_polygon_always_present(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon()
        mgr = _MockImageManager([entry], {0: [poly]})

        # use_rle=False avoids cv2 dependency; polygon fallback must still be present
        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "segmentation_polygon" in data["annotations"][0]

    def test_bbox_format_coco(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon(points=[(100.0, 200.0), (150.0, 180.0), (200.0, 220.0), (160.0, 260.0)])
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        bbox = data["annotations"][0]["bbox"]
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(100.0)
        assert bbox[1] == pytest.approx(180.0)
        assert bbox[2] == pytest.approx(100.0)
        assert bbox[3] == pytest.approx(80.0)

    def test_area_shoelace_not_bbox_area(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=200)
        # Diamond polygon — area should be < bbox_w * bbox_h
        poly = _make_polygon(points=[(50.0, 0.0), (100.0, 50.0), (50.0, 100.0), (0.0, 50.0)])
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        ann = data["annotations"][0]
        bbox = ann["bbox"]
        assert ann["area"] < bbox[2] * bbox[3]

    def test_unannotated_images_skipped(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        e1 = _make_entry(0, "img1.jpg", annotated=True)
        e2 = _make_entry(1, "img2.jpg", annotated=False)
        poly = _make_polygon()
        mgr = _MockImageManager([e1, e2], {0: [poly], 1: []})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == "img1.jpg"

    def test_split_filter(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        e_train = _make_entry(0, "train.jpg", split="train")
        e_val = _make_entry(1, "val.jpg", split="val")
        poly = _make_polygon()
        mgr = _MockImageManager([e_train, e_val], {0: [poly], 1: [poly]})

        out = tmp_path / "train_only.json"
        exporter.export_coco(mgr, str(out), split="train", use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == "train.jpg"

    def test_source_and_confidence_stored(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon(source="sam_point", confidence=0.92)
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        ann = data["annotations"][0]
        assert ann["source"] == "sam_point"
        assert ann["confidence"] == pytest.approx(0.92)

    def test_occlusion_level_null_when_none(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon(occlusion_level=None, occlusion_ratio=None)
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        ann = data["annotations"][0]
        assert ann["occlusion_level"] is None
        assert ann["occlusion_ratio"] is None

    def test_occlusion_fields_stored_when_set(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon(
            occlusion_level=OcclusionLevel.SEDANG,
            occlusion_ratio=0.43,
        )
        mgr = _MockImageManager([entry], {0: [poly]})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        ann = data["annotations"][0]
        assert ann["occlusion_level"] == "sedang"
        assert ann["occlusion_ratio"] == pytest.approx(0.43)

    def test_annotation_ids_globally_unique(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entries = [_make_entry(i, f"img{i}.jpg", width=200, height=200) for i in range(5)]
        poly = _make_polygon()
        mgr = _MockImageManager(entries, {i: [poly, poly, poly] for i in range(5)})

        out = tmp_path / "out.json"
        exporter.export_coco(mgr, str(out), use_rle=False)
        data = json.loads(out.read_text(encoding="utf-8"))
        ids = [a["id"] for a in data["annotations"]]
        assert len(ids) == len(set(ids)) == 15


# ---------------------------------------------------------------------------
# Integration: COCO splits
# ---------------------------------------------------------------------------

class TestExportCocoSplits:
    def _export_splits_no_rle(self, exporter, mgr, tmp_path):
        """Helper: export splits with use_rle=False to avoid cv2 dependency."""
        out = Path(tmp_path)
        result: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            file_path = out / f"{split_name}.json"
            result[split_name] = exporter.export_coco(
                mgr, str(file_path), split=split_name, use_rle=False
            )
        return result

    def test_three_files_generated(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entries = [
            _make_entry(0, "t.jpg", split="train"),
            _make_entry(1, "v.jpg", split="val"),
            _make_entry(2, "te.jpg", split="test"),
        ]
        poly = _make_polygon()
        mgr = _MockImageManager(entries, {0: [poly], 1: [poly], 2: [poly]})

        result = self._export_splits_no_rle(exporter, mgr, tmp_path)

        assert set(result.keys()) == {"train", "val", "test"}
        for key in ("train", "val", "test"):
            assert Path(result[key]).exists()

    def test_no_image_overlap_between_splits(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entries = [
            _make_entry(0, "t1.jpg", split="train"),
            _make_entry(1, "t2.jpg", split="train"),
            _make_entry(2, "v.jpg", split="val"),
            _make_entry(3, "te.jpg", split="test"),
        ]
        poly = _make_polygon()
        mgr = _MockImageManager(entries, {i: [poly] for i in range(4)})

        result = self._export_splits_no_rle(exporter, mgr, tmp_path)

        all_ids: list[set] = []
        for path in result.values():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            all_ids.append({img["id"] for img in data["images"]})

        for i, s1 in enumerate(all_ids):
            for j, s2 in enumerate(all_ids):
                if i != j:
                    assert s1.isdisjoint(s2), "Image IDs overlap between splits"

    def test_empty_split_generates_valid_json(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "t.jpg", split="train")
        poly = _make_polygon()
        mgr = _MockImageManager([entry], {0: [poly]})

        result = self._export_splits_no_rle(exporter, mgr, tmp_path)

        for split_name in ("val", "test"):
            data = json.loads(Path(result[split_name]).read_text(encoding="utf-8"))
            assert data["images"] == []
            assert data["annotations"] == []

    def test_return_value_contains_paths(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        mgr = _MockImageManager([], {})
        result = self._export_splits_no_rle(exporter, mgr, tmp_path)
        assert "train" in result and "val" in result and "test" in result


# ---------------------------------------------------------------------------
# Integration: Pascal VOC XML
# ---------------------------------------------------------------------------

class TestExportVocXml:
    def test_one_xml_per_annotated_image(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entries = [_make_entry(i, f"img{i}.jpg") for i in range(3)]
        poly = _make_polygon()
        mgr = _MockImageManager(entries, {0: [poly], 1: [poly], 2: [poly]})

        written = exporter.export_voc_xml(mgr, str(tmp_path))
        assert len(written) == 3

    def test_xml_parseable_by_etree(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "daun_001.jpg", width=3024, height=4032)
        poly = _make_polygon(source="sam_point", confidence=0.92)
        mgr = _MockImageManager([entry], {0: [poly]})

        written = exporter.export_voc_xml(mgr, str(tmp_path))
        assert len(written) == 1
        tree = ET.parse(written[0])
        assert tree.getroot().tag == "annotation"

    def test_required_elements_present(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "daun_001.jpg", width=3024, height=4032)
        poly = _make_polygon()
        mgr = _MockImageManager([entry], {0: [poly]})

        written = exporter.export_voc_xml(mgr, str(tmp_path))
        root = ET.parse(written[0]).getroot()

        assert root.findtext("filename") == "daun_001.jpg"
        assert root.findtext("size/width") == "3024"
        assert root.findtext("size/height") == "4032"
        assert root.findtext("size/depth") == "3"
        assert root.findtext("segmented") == "1"

    def test_object_bndbox_and_polygon(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=200)
        poly = _make_polygon(
            points=[(100.0, 200.0), (150.0, 180.0), (200.0, 220.0), (160.0, 260.0)],
            source="sam_point",
            confidence=0.92,
        )
        mgr = _MockImageManager([entry], {0: [poly]})

        written = exporter.export_voc_xml(mgr, str(tmp_path))
        root = ET.parse(written[0]).getroot()
        obj = root.find("object")

        assert obj.findtext("bndbox/xmin") == "100"
        assert obj.findtext("bndbox/ymin") == "180"
        assert obj.findtext("bndbox/xmax") == "200"
        assert obj.findtext("bndbox/ymax") == "260"
        assert obj.findtext("source") == "sam_point"
        assert obj.findtext("confidence") == "0.92"
        assert obj.find("polygon") is not None

    def test_unannotated_images_produce_no_xml(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "empty.jpg", annotated=False)
        mgr = _MockImageManager([entry], {0: []})

        written = exporter.export_voc_xml(mgr, str(tmp_path))
        assert written == []
        assert not (tmp_path / "empty.xml").exists()

    def test_split_filter_voc(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        e_train = _make_entry(0, "train.jpg", split="train")
        e_val = _make_entry(1, "val.jpg", split="val")
        poly = _make_polygon()
        mgr = _MockImageManager([e_train, e_val], {0: [poly], 1: [poly]})

        written = exporter.export_voc_xml(mgr, str(tmp_path), split="val")
        assert len(written) == 1
        assert "val.xml" in written[0]


# ---------------------------------------------------------------------------
# Integration: export_single_coco
# ---------------------------------------------------------------------------

class TestExportSingleCoco:
    def test_returns_dict_without_writing_file(self, tmp_path):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon()

        result = exporter.export_single_coco(entry, [poly], use_rle=False)

        assert isinstance(result, dict)
        assert "images" in result and "annotations" in result
        assert len(list(tmp_path.iterdir())) == 0

    def test_annotation_id_start_respected(self):
        cfg = _make_config()
        exporter = Exporter(cfg)
        entry = _make_entry(0, "img.jpg", width=200, height=150)
        poly = _make_polygon()

        result = exporter.export_single_coco(entry, [poly], annotation_id_start=42, use_rle=False)
        assert result["annotations"][0]["id"] == 42

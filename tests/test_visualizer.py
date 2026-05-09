"""Tests for src/visualizer.py"""
import json
import sys
import types
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

from PIL import Image  # noqa: E402 (PIL already listed in requirements)

from src.image_manager import ImageEntry, ImageManager, Polygon  # noqa: E402
from src.visualizer import Visualizer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(mask_opacity: float = 0.45):
    canvas_ns = types.SimpleNamespace(mask_opacity=mask_opacity)
    return types.SimpleNamespace(canvas=canvas_ns)


def _make_image_file(path: Path, width: int = 200, height: int = 150) -> Path:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(str(path))
    return path


def _poly(
    instance_id: int = 1,
    source: str = "manual",
    confidence: float = 1.0,
    color: list[int] | None = None,
    points: list[tuple[float, float]] | None = None,
) -> Polygon:
    return Polygon(
        instance_id=instance_id,
        class_id=1,
        class_name="daun",
        points=points or [(20.0, 20.0), (80.0, 20.0), (80.0, 60.0), (20.0, 60.0)],
        color=color or [200, 100, 50],
        source=source,
        confidence=confidence,
    )


def _make_config_with_paths(tmp_path: Path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    paths_ns = types.SimpleNamespace(annotations=annotations_dir)
    split_ns = types.SimpleNamespace(train=0.7, val=0.2, test=0.1, seed=42)
    return types.SimpleNamespace(
        image_extensions=[".jpg", ".jpeg", ".png"],
        paths=paths_ns,
        dataset_split=split_ns,
        canvas=types.SimpleNamespace(mask_opacity=0.45),
    )


# ---------------------------------------------------------------------------
# Task 5.1 — render() with polygons: original still visible under mask
# ---------------------------------------------------------------------------

class TestRenderWithPolygons:
    def test_returns_ndarray(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path)
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [_poly()])
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=200, height=150)
        original = cv2.imread(str(img_path))
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [_poly()])
        assert result.shape == original.shape

    def test_image_still_visible_under_mask(self, tmp_path):
        """With alpha=0.45 the original pixel values contribute 55% — result
        should differ from a fully-opaque fill but not equal the raw image either."""
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=200, height=150)
        original = cv2.imread(str(img_path))
        viz = Visualizer(_make_config(mask_opacity=0.45))
        result = viz.render(str(img_path), [_poly()])
        # result must differ from pure original (mask was applied)
        assert not np.array_equal(result, original)
        # But the background area outside polygon is unchanged
        bg_orig = original[0, 0]
        bg_res = result[0, 0]
        assert np.array_equal(bg_orig, bg_res)

    def test_multiple_polygons_all_filled(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=300, height=200)
        polys = [
            _poly(1, points=[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]),
            _poly(2, points=[(100.0, 100.0), (150.0, 100.0), (150.0, 150.0), (100.0, 150.0)]),
        ]
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), polys)
        assert result is not None
        assert result.shape[:2] == (200, 300)


# ---------------------------------------------------------------------------
# Task 5.2 — highlight_auto=True: dashed yellow border + SAM badge
# ---------------------------------------------------------------------------

class TestHighlightAuto:
    def test_sam_polygon_gets_yellow_pixels(self, tmp_path):
        """Yellow border pixels (B≈0, G≈220, R≈255) should appear in result
        when highlight_auto=True and polygon is not manual."""
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=300, height=200)
        sam_poly = _poly(1, source="sam_auto", confidence=0.92)
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [sam_poly], highlight_auto=True)
        # Check for pixels that are clearly yellow-ish (high R, high G, low B)
        r, g, b = result[:, :, 2], result[:, :, 1], result[:, :, 0]
        yellow_mask = (r > 200) & (g > 180) & (b < 50)
        assert yellow_mask.any(), "Expected yellow border pixels for SAM polygon"

    def test_manual_polygon_no_yellow_border(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=300, height=200)
        manual_poly = _poly(1, source="manual")
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [manual_poly], highlight_auto=True)
        r, g, b = result[:, :, 2], result[:, :, 1], result[:, :, 0]
        yellow_mask = (r > 200) & (g > 180) & (b < 50)
        assert not yellow_mask.any(), "Manual polygon should not have yellow border"

    def test_no_highlight_when_flag_false(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=300, height=200)
        sam_poly = _poly(1, source="sam_auto", confidence=0.92)
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [sam_poly], highlight_auto=False)
        r, g, b = result[:, :, 2], result[:, :, 1], result[:, :, 0]
        yellow_mask = (r > 200) & (g > 180) & (b < 50)
        assert not yellow_mask.any()


# ---------------------------------------------------------------------------
# Task 5.3 — render() with empty polygon list returns original unchanged
# ---------------------------------------------------------------------------

class TestRenderEmptyPolygons:
    def test_empty_polygons_returns_original(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=200, height=150)
        original = cv2.imread(str(img_path))
        viz = Visualizer(_make_config())
        result = viz.render(str(img_path), [])
        assert np.array_equal(result, original)


# ---------------------------------------------------------------------------
# Task 5.4 — render_thumbnail() returns valid QPixmap within max_size
# ---------------------------------------------------------------------------

class TestRenderThumbnail:
    @pytest.fixture(autouse=True)
    def _qapp(self):
        """Ensure a QApplication exists for QPixmap creation."""
        from PyQt5.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        yield app

    def test_returns_qpixmap(self, tmp_path):
        from PyQt5.QtGui import QPixmap as _QPixmap
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=400, height=300)
        viz = Visualizer(_make_config())
        result = viz.render_thumbnail(str(img_path), [])
        assert isinstance(result, _QPixmap)
        assert not result.isNull()

    def test_thumbnail_within_max_size(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=600, height=400)
        viz = Visualizer(_make_config())
        max_size = (150, 100)
        result = viz.render_thumbnail(str(img_path), [], max_size=max_size)
        assert result.width() <= max_size[0]
        assert result.height() <= max_size[1]

    def test_thumbnail_preserves_aspect_ratio(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=400, height=200)
        viz = Visualizer(_make_config())
        result = viz.render_thumbnail(str(img_path), [], max_size=(200, 200))
        ratio = result.width() / result.height()
        assert abs(ratio - 2.0) < 0.1

    def test_small_image_not_upscaled(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=100, height=80)
        viz = Visualizer(_make_config())
        result = viz.render_thumbnail(str(img_path), [], max_size=(300, 300))
        assert result.width() <= 300
        assert result.height() <= 300


# ---------------------------------------------------------------------------
# Task 5.5 — render_comparison() output is 2× wide
# ---------------------------------------------------------------------------

class TestRenderComparison:
    def test_output_width_is_double(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=200, height=150)
        before = [_poly(1, source="sam_auto", confidence=0.9)]
        after = [_poly(1, source="manual")]
        viz = Visualizer(_make_config())
        result = viz.render_comparison(str(img_path), before, after)
        assert result.shape[1] == 400  # 2 × 200
        assert result.shape[0] == 150

    def test_output_height_matches_original(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path, width=160, height=120)
        viz = Visualizer(_make_config())
        result = viz.render_comparison(str(img_path), [], [])
        assert result.shape[0] == 120
        assert result.shape[1] == 320

    def test_returns_ndarray(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        _make_image_file(img_path)
        viz = Visualizer(_make_config())
        result = viz.render_comparison(str(img_path), [], [])
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Task 5.6 — export_all(): callback called N times, JPG quality 95 saved
# ---------------------------------------------------------------------------

class TestExportAll:
    def _make_manager_with_annotated(self, tmp_path: Path, count: int):
        cfg = _make_config_with_paths(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_dir = tmp_path / "annotations"

        entries = []
        for i in range(count):
            name = f"leaf_{i:02d}.jpg"
            img_path = images_dir / name
            _make_image_file(img_path)

            poly_data = {
                "instance_id": 1, "class_id": 1, "class_name": "daun",
                "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "color": [200, 100, 50], "source": "manual", "confidence": 1.0,
            }
            ann = ann_dir / f"leaf_{i:02d}.json"
            ann.write_text(
                json.dumps({
                    "filename": name, "width": 200, "height": 150,
                    "polygons": [poly_data],
                }),
                encoding="utf-8",
            )

        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        return mgr, cfg

    def test_callback_called_once_per_image(self, tmp_path):
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 3)
        viz = Visualizer(cfg)
        calls = []
        output_dir = str(tmp_path / "output")
        viz.export_all(mgr, output_dir, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[2] == (3, 3)

    def test_no_callback_does_not_raise(self, tmp_path):
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 2)
        viz = Visualizer(cfg)
        output_dir = str(tmp_path / "output")
        viz.export_all(mgr, output_dir, progress_callback=None)

    def test_output_files_created(self, tmp_path):
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 2)
        viz = Visualizer(cfg)
        output_dir = tmp_path / "output"
        viz.export_all(mgr, str(output_dir))
        assert (output_dir / "leaf_00_annotated.jpg").exists()
        assert (output_dir / "leaf_01_annotated.jpg").exists()

    def test_output_filename_convention(self, tmp_path):
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 1)
        viz = Visualizer(cfg)
        output_dir = tmp_path / "output"
        paths = viz.export_all(mgr, str(output_dir))
        assert paths[0].endswith("leaf_00_annotated.jpg")

    def test_saved_as_jpg_quality_95(self, tmp_path):
        """Verify files are valid JPEG (can be re-read by OpenCV)."""
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 1)
        viz = Visualizer(cfg)
        output_dir = tmp_path / "output"
        paths = viz.export_all(mgr, str(output_dir))
        img = cv2.imread(paths[0])
        assert img is not None
        assert img.shape[0] == 150
        assert img.shape[1] == 200

    def test_returns_list_of_paths(self, tmp_path):
        mgr, cfg = self._make_manager_with_annotated(tmp_path, 2)
        viz = Visualizer(cfg)
        paths = viz.export_all(mgr, str(tmp_path / "output"))
        assert len(paths) == 2
        for p in paths:
            assert isinstance(p, str)

    def test_unannotated_images_skipped(self, tmp_path):
        cfg = _make_config_with_paths(tmp_path)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for name in ["a.jpg", "b.jpg"]:
            _make_image_file(images_dir / name)
        # only 'a' has annotation
        ann = tmp_path / "annotations" / "a.json"
        ann.write_text(
            json.dumps({
                "filename": "a.jpg", "width": 200, "height": 150,
                "polygons": [{
                    "instance_id": 1, "class_id": 1, "class_name": "daun",
                    "points": [[10, 10], [50, 10], [50, 50]],
                    "color": [200, 100, 50], "source": "manual", "confidence": 1.0,
                }],
            }),
            encoding="utf-8",
        )
        mgr = ImageManager(cfg)
        mgr.load_folder(str(images_dir))
        viz = Visualizer(cfg)
        paths = viz.export_all(mgr, str(tmp_path / "output"))
        assert len(paths) == 1
        assert "a_annotated" in paths[0]

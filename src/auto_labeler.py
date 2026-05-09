from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor

from src.image_manager import ImageEntry, OcclusionLevel, Polygon

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger(__name__)

# Leave one core free for the UI thread
_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_polygon(
    binary_mask: np.ndarray, epsilon: float = 2.0
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, epsilon, closed=True)
    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return points if len(points) >= 3 else []


def _rasterize_mask(args: tuple) -> np.ndarray:
    """Rasterize one polygon into a binary mask (runs in thread pool)."""
    poly, h, w = args
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly.points:
        pts = np.array(poly.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _compute_overlap(args: tuple) -> float:
    """Overlap ratio for one polygon against union of all others (runs in thread pool)."""
    own_mask, prefix_union, suffix_union = args
    own_area = int(own_mask.sum())
    if own_area == 0:
        return 0.0
    union_others = np.bitwise_or(prefix_union, suffix_union)
    return int(np.bitwise_and(own_mask, union_others).sum()) / own_area


def compute_occlusion_levels(
    polygons: list[Polygon], image_shape: tuple[int, int]
) -> list[Polygon]:
    if not polygons:
        return polygons
    h, w = image_shape
    n = len(polygons)

    # --- Phase 1: rasterize all polygon masks in parallel ---
    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        masks = list(ex.map(_rasterize_mask, [(p, h, w) for p in polygons]))

    if n == 1:
        polygons[0].occlusion_level = OcclusionLevel.RENDAH
        polygons[0].occlusion_ratio = 0.0
        return polygons

    # --- Phase 2: prefix-suffix union — O(N) instead of O(N²) ---
    # prefix[i] = union of masks[0 .. i-1]
    # suffix[i] = union of masks[i+1 .. n-1]
    prefix = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    suffix = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    for i in range(1, n):
        np.bitwise_or(prefix[i - 1], masks[i - 1], out=prefix[i])
    for i in range(n - 2, -1, -1):
        np.bitwise_or(suffix[i + 1], masks[i + 1], out=suffix[i])

    # --- Phase 3: compute overlap ratios in parallel ---
    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        ratios = list(ex.map(
            _compute_overlap,
            [(masks[i], prefix[i], suffix[i]) for i in range(n)],
        ))

    for poly, ratio in zip(polygons, ratios):
        poly.occlusion_ratio = ratio
        if ratio < 0.30:
            poly.occlusion_level = OcclusionLevel.RENDAH
        elif ratio <= 0.60:
            poly.occlusion_level = OcclusionLevel.SEDANG
        else:
            poly.occlusion_level = OcclusionLevel.TINGGI

    return polygons


def _process_contour(args: tuple) -> tuple[list[tuple[float, float]], float] | None:
    """Simplify one contour and return (scaled_pts, orig_area) or None (runs in thread pool)."""
    contour, epsilon, scale, min_area_scaled = args
    area = cv2.contourArea(contour)
    if area < min_area_scaled:
        return None
    arc = cv2.arcLength(contour, True)
    eps = max(epsilon * scale, arc * 0.005)
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = [(float(p[0][0]) / scale, float(p[0][1]) / scale) for p in approx]
    if len(pts) < 3:
        return None
    return pts, area / (scale ** 2)


def _polygon_color(instance_id: int) -> list[int]:
    hue = (instance_id * 137) % 360
    c = QColor.fromHsv(hue, 200, 230)
    return [c.red(), c.green(), c.blue()]


def _class_info(config: Config) -> tuple[int, str]:
    try:
        cls = config.classes[0]
        return cls["id"], cls["name"]
    except (AttributeError, IndexError, KeyError):
        return 1, "daun_kelengkeh_itoh"


_WORK_SIZE = 1024  # max dimension for HSV processing; contours scaled back up after


def _get_plant_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Binary mask of green/plant regions via HSV thresholding.

    Internally downscales to _WORK_SIZE to keep morphological ops fast.
    Returns (mask_at_work_scale, scale_factor) so contours can be rescaled.
    """
    h, w = image_rgb.shape[:2]
    scale = min(1.0, _WORK_SIZE / max(h, w))
    if scale < 1.0:
        ww, wh = int(w * scale), int(h * scale)
        small = cv2.resize(image_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
    else:
        small = image_rgb
        wh, ww = h, w

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([20, 35, 30]), np.array([90, 255, 255]))

    k = max(3, min(wh, ww) // 60)
    k = k if k % 2 == 1 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, scale


def _grabcut(image_rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Run GrabCut with rect=(x, y, w, h); return foreground binary mask."""
    h, w = image_rgb.shape[:2]
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image_bgr, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# AutoLabeler  (GrabCut + HSV — no model download required)
# ---------------------------------------------------------------------------

class AutoLabeler:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._loaded = False

    def load_model(self) -> None:
        """No external model needed — GrabCut is built into OpenCV."""
        log.info("AutoLabeler: menggunakan GrabCut + HSV (tidak perlu download model)")
        self._loaded = True
        log.info("AutoLabeler: siap digunakan")

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # predict_from_point — GrabCut with auto-sized box around click
    # ------------------------------------------------------------------

    def predict_from_point(
        self,
        image: np.ndarray,
        point: tuple[float, float],
        negative_points: list[tuple[float, float]] | None = None,
        instance_id_start: int = 1,
    ) -> list[Polygon]:
        if not self._loaded:
            log.warning("predict_from_point dipanggil sebelum load_model()")
            return []
        h, w = image.shape[:2]
        cx, cy = int(point[0]), int(point[1])
        pad = max(40, min(h, w) // 7)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(w, cx + pad), min(h, cy + pad)
        log.debug("GrabCut point (%.0f, %.0f) → box (%d,%d,%d,%d)", cx, cy, x1, y1, x2, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            log.warning("Box terlalu kecil, skip")
            return []
        result = self._grabcut_to_polygons(
            image, x1, y1, x2 - x1, y2 - y1, instance_id_start, source="sam_point"
        )
        log.info("predict_from_point: %d polygon dihasilkan", len(result))
        return result

    # ------------------------------------------------------------------
    # predict_from_box — GrabCut inside user-drawn box
    # ------------------------------------------------------------------

    def predict_from_box(
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float],
        instance_id_start: int = 1,
    ) -> list[Polygon]:
        if not self._loaded:
            log.warning("predict_from_box dipanggil sebelum load_model()")
            return []
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(box[0])), max(0, int(box[1])),
            min(w, int(box[2])), min(h, int(box[3])),
        )
        bw, bh = x2 - x1, y2 - y1
        log.debug("GrabCut box (%d,%d) ukuran %dx%d", x1, y1, bw, bh)
        if bw < 10 or bh < 10:
            log.warning("Box terlalu kecil (%dx%d), skip", bw, bh)
            return []
        result = self._grabcut_to_polygons(
            image, x1, y1, bw, bh, instance_id_start, source="sam_point"
        )
        log.info("predict_from_box: %d polygon dihasilkan", len(result))
        return result

    # ------------------------------------------------------------------
    # predict_automatic — HSV color detection + contour extraction
    # ------------------------------------------------------------------

    def predict_automatic(
        self,
        image: np.ndarray,
        instance_id_start: int = 1,
        progress_callback: callable = None,
    ) -> list[Polygon]:
        if not self._loaded:
            log.warning("predict_automatic dipanggil sebelum load_model()")
            return []

        h, w = image.shape[:2]
        log.info("predict_automatic: mulai pada gambar %dx%d", w, h)

        min_area = getattr(
            getattr(self._config, "sam", None), "min_polygon_area_px", 500
        )
        epsilon = getattr(
            getattr(self._config, "sam", None), "polygon_simplify_epsilon", 2.0
        )

        log.info("[1/4] Membuat HSV mask (gambar %dx%d → kerja maks %dpx)...", w, h, _WORK_SIZE)
        mask, scale = _get_plant_mask(image)
        # min_area is in original-image pixels; convert to work-scale pixels
        min_area_scaled = min_area * (scale ** 2)
        green_px = int(mask.sum() / 255)
        work_px = mask.shape[0] * mask.shape[1]
        log.info(
            "[2/4] HSV mask selesai: %d piksel hijau (%.1f%% gambar, skala=%.2f)",
            green_px, 100 * green_px / work_px, scale,
        )

        log.info("[3/4] Mencari kontur dari mask...")
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        log.info(
            "      %d kontur ditemukan, filter area >= %.0f px² (skala kerja)",
            len(contours), min_area_scaled,
        )

        class_id, class_name = _class_info(self._config)
        polygons: list[Polygon] = []
        iid = instance_id_start
        total = len(contours)
        skipped = 0

        log.info("      Memproses %d kontur secara paralel (%d worker)...", total, _CPU_WORKERS)
        args_list = [(c, epsilon, scale, min_area_scaled) for c in contours]

        # Process all contours in parallel; results are in the same order as contours
        with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
            results = list(ex.map(_process_contour, args_list))

        for idx, result in enumerate(results):
            if progress_callback:
                progress_callback(idx + 1, total)
            if result is None:
                skipped += 1
                continue
            pts, orig_area = result
            log.info(
                "      + polygon #%d: area=%.0f px², %d titik",
                iid - instance_id_start + 1, orig_area, len(pts),
            )
            polygons.append(Polygon(
                instance_id=iid,
                class_id=class_id,
                class_name=class_name,
                points=pts,
                color=_polygon_color(iid),
                source="sam_auto",
                confidence=0.90,
            ))
            iid += 1

        log.info(
            "[4/4] Kontur selesai: %d polygon, %d dilewati (area terlalu kecil)",
            len(polygons), skipped,
        )

        if polygons:
            log.info("      Menghitung tingkat oklusi untuk %d polygon...", len(polygons))
            compute_occlusion_levels(polygons, image.shape[:2])
            log.info("      Oklusi selesai")

        return polygons

    def predict_automatic_batch(
        self,
        image_entries: list[ImageEntry],
        confidence_threshold: float,
        progress_callback: callable = None,
    ) -> dict[str, list[Polygon]]:
        result: dict[str, list[Polygon]] = {}
        for entry in image_entries:
            img = cv2.imread(entry.filepath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            polygons = self.predict_automatic(img_rgb)
            polygons = [p for p in polygons if p.confidence >= confidence_threshold]
            result[entry.filename] = polygons
            if progress_callback:
                progress_callback(entry, polygons)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _grabcut_to_polygons(
        self,
        image: np.ndarray,
        x: int, y: int, bw: int, bh: int,
        instance_id_start: int,
        source: str,
    ) -> list[Polygon]:
        epsilon = getattr(
            getattr(self._config, "sam", None), "polygon_simplify_epsilon", 2.0
        )
        fg_mask = _grabcut(image, (x, y, bw, bh))
        pts = mask_to_polygon(fg_mask, epsilon)
        if not pts:
            return []
        class_id, class_name = _class_info(self._config)
        iid = instance_id_start
        return [Polygon(
            instance_id=iid,
            class_id=class_id,
            class_name=class_name,
            points=pts,
            color=_polygon_color(iid),
            source=source,
            confidence=0.85,
        )]


# ---------------------------------------------------------------------------
# AutoLabelWorker
# ---------------------------------------------------------------------------

class AutoLabelWorker(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, filename
    result_ready = pyqtSignal(str, list)   # filename, list[Polygon]
    finished = pyqtSignal(int)             # total polygons generated
    error = pyqtSignal(str)               # error message

    def __init__(
        self,
        auto_labeler: AutoLabeler,
        entries: list[ImageEntry],
        confidence_threshold: float,
    ) -> None:
        super().__init__()
        self._auto_labeler = auto_labeler
        self._entries = entries
        self._confidence = confidence_threshold
        self._stop_event = threading.Event()

    def run(self) -> None:
        total = len(self._entries)
        total_polygons = 0
        log.info("AutoLabelWorker: mulai batch %d gambar (confidence >= %.2f)",
                 total, self._confidence)
        for idx, entry in enumerate(self._entries, start=1):
            if self._stop_event.is_set():
                log.info("AutoLabelWorker: dihentikan setelah %d/%d gambar", idx - 1, total)
                break
            log.info("[%d/%d] Memproses: %s", idx, total, entry.filename)
            try:
                img = cv2.imread(entry.filepath)
                if img is None:
                    log.warning("  Gagal membaca gambar: %s", entry.filepath)
                    self.progress.emit(idx, total, entry.filename)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                polygons = self._auto_labeler.predict_automatic(
                    img_rgb, instance_id_start=total_polygons + 1
                )
                polygons = [p for p in polygons if p.confidence >= self._confidence]
                total_polygons += len(polygons)
                log.info("  → %d polygon diterima untuk %s", len(polygons), entry.filename)
                self.result_ready.emit(entry.filename, polygons)
                self.progress.emit(idx, total, entry.filename)
            except Exception as exc:
                log.error("  Error pada %s: %s", entry.filename, exc, exc_info=True)
                self.error.emit(f"{entry.filename}: {exc}")
        log.info("AutoLabelWorker selesai: total %d polygon dari %d gambar",
                 total_polygons, total)
        self.finished.emit(total_polygons)

    def stop(self) -> None:
        self._stop_event.set()

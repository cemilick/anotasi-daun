from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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
    poly, h, w = args
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly.points:
        pts = np.array(poly.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _compute_overlap(args: tuple) -> float:
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

    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        masks = list(ex.map(_rasterize_mask, [(p, h, w) for p in polygons]))

    if n == 1:
        polygons[0].occlusion_level = OcclusionLevel.RENDAH
        polygons[0].occlusion_ratio = 0.0
        return polygons

    prefix = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    suffix = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    for i in range(1, n):
        np.bitwise_or(prefix[i - 1], masks[i - 1], out=prefix[i])
    for i in range(n - 2, -1, -1):
        np.bitwise_or(suffix[i + 1], masks[i + 1], out=suffix[i])

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


# ---------------------------------------------------------------------------
# AutoLabeler  (MobileSAM)
# ---------------------------------------------------------------------------

class AutoLabeler:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._loaded = False
        self._predictor = None
        self._generator = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        checkpoint = Path(str(self._config.paths.sam_checkpoint))
        if not checkpoint.exists():
            log.warning("AutoLabeler: checkpoint tidak ditemukan: %s", checkpoint)
            log.warning("AutoLabeler: download dari https://github.com/ChaoningZhang/MobileSAM")
            log.warning("AutoLabeler: letakkan file mobile_sam.pt di folder: %s", checkpoint.parent)
            return

        try:
            from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except ImportError:
            log.error("AutoLabeler: mobile-sam belum terinstall — jalankan: pip install mobile-sam")
            return

        try:
            sam_cfg = getattr(self._config, "sam", None)
            model_type = getattr(sam_cfg, "model_type", "vit_t")
            device = getattr(sam_cfg, "device", "cpu")

            log.info("AutoLabeler: memuat MobileSAM (model_type=%s, device=%s) ...", model_type, device)
            log.info("AutoLabeler: checkpoint: %s", checkpoint)

            import torch
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
            try:
                sam.to(device=device)
            except RuntimeError:
                log.warning("AutoLabeler: gagal ke device '%s', fallback ke CPU", device)
                sam.to(device="cpu")
                device = "cpu"
            sam.eval()

            self._predictor = SamPredictor(sam)
            self._generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=getattr(sam_cfg, "points_per_side", 32),
                pred_iou_thresh=getattr(sam_cfg, "pred_iou_thresh", 0.88),
                stability_score_thresh=getattr(sam_cfg, "stability_score_thresh", 0.95),
                min_mask_region_area=getattr(sam_cfg, "min_polygon_area_px", 500),
            )

            self._loaded = True
            log.info("AutoLabeler: MobileSAM siap digunakan (device=%s)", device)

        except Exception as exc:
            log.error("AutoLabeler: gagal memuat model — %s", exc, exc_info=True)
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # predict_from_point
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
        log.info("predict_from_point: titik (%.0f, %.0f) pada gambar %dx%d", point[0], point[1], w, h)

        coords = [[point[0], point[1]]]
        labels = [1]
        if negative_points:
            for np_ in negative_points:
                coords.append([np_[0], np_[1]])
                labels.append(0)

        try:
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(coords, dtype=float),
                point_labels=np.array(labels, dtype=int),
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            epsilon = getattr(getattr(self._config, "sam", None), "polygon_simplify_epsilon", 2.0)
            pts = mask_to_polygon(masks[best_idx], epsilon)
            if not pts:
                log.info("predict_from_point: tidak ada polygon valid")
                return []
            class_id, class_name = _class_info(self._config)
            confidence = float(scores[best_idx])
            log.info("predict_from_point: 1 polygon, confidence=%.3f", confidence)
            return [Polygon(
                instance_id=instance_id_start,
                class_id=class_id,
                class_name=class_name,
                points=pts,
                color=_polygon_color(instance_id_start),
                source="sam_point",
                confidence=confidence,
            )]
        except Exception as exc:
            log.error("predict_from_point: error — %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # predict_from_box
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
        log.info("predict_from_box: box (%d,%d,%d,%d) pada gambar %dx%d", x1, y1, x2, y2, w, h)

        try:
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                box=np.array([x1, y1, x2, y2], dtype=float),
                multimask_output=False,
            )
            if not len(masks):
                log.info("predict_from_box: tidak ada mask dihasilkan")
                return []
            epsilon = getattr(getattr(self._config, "sam", None), "polygon_simplify_epsilon", 2.0)
            pts = mask_to_polygon(masks[0], epsilon)
            if not pts:
                log.info("predict_from_box: mask tidak bisa dikonversi ke polygon")
                return []
            class_id, class_name = _class_info(self._config)
            confidence = float(scores[0])
            log.info("predict_from_box: 1 polygon, confidence=%.3f", confidence)
            return [Polygon(
                instance_id=instance_id_start,
                class_id=class_id,
                class_name=class_name,
                points=pts,
                color=_polygon_color(instance_id_start),
                source="sam_point",
                confidence=confidence,
            )]
        except Exception as exc:
            log.error("predict_from_box: error — %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # predict_automatic
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

        sam_cfg = getattr(self._config, "sam", None)
        min_area = getattr(sam_cfg, "min_polygon_area_px", 500)
        epsilon = getattr(sam_cfg, "polygon_simplify_epsilon", 2.0)

        try:
            log.info("[1/3] MobileSAM generate masks (ini bisa makan beberapa detik)...")
            mask_dicts = self._generator.generate(image)
            mask_dicts.sort(key=lambda m: m["area"], reverse=True)
            total = len(mask_dicts)
            log.info("[2/3] %d mask ditemukan, filter area >= %d px²", total, min_area)

            class_id, class_name = _class_info(self._config)
            polygons: list[Polygon] = []
            iid = instance_id_start
            skipped = 0

            for idx, m in enumerate(mask_dicts):
                if m["area"] < min_area:
                    skipped += 1
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    continue

                pts = mask_to_polygon(m["segmentation"], epsilon)
                if not pts:
                    skipped += 1
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    continue

                confidence = float(m.get("predicted_iou", 0.9))
                log.info(
                    "      + polygon #%d: area=%d px², %d titik, iou=%.3f",
                    iid - instance_id_start + 1, m["area"], len(pts), confidence,
                )
                polygons.append(Polygon(
                    instance_id=iid,
                    class_id=class_id,
                    class_name=class_name,
                    points=pts,
                    color=_polygon_color(iid),
                    source="sam_auto",
                    confidence=confidence,
                ))
                iid += 1

                if progress_callback:
                    progress_callback(idx + 1, total)

            log.info(
                "[3/3] Selesai: %d polygon dihasilkan, %d dilewati",
                len(polygons), skipped,
            )

            if polygons:
                log.info("      Menghitung oklusi untuk %d polygon...", len(polygons))
                compute_occlusion_levels(polygons, (h, w))
                log.info("      Oklusi selesai")

            return polygons

        except Exception as exc:
            log.error("predict_automatic: error — %s", exc, exc_info=True)
            return []

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

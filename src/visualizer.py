from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

from src.image_manager import ImageManager, Polygon

if TYPE_CHECKING:
    from src.config import Config

_SAM_BORDER_BGR = (0, 220, 255)  # [255, 220, 0] RGB → BGR
_DASH_LEN = 8


class Visualizer:
    def __init__(self, config: Config) -> None:
        self._config = config

    def render(
        self,
        image_path: str,
        polygons: list[Polygon],
        show_labels: bool = True,
        show_border: bool = True,
        highlight_auto: bool = False,
    ) -> np.ndarray:
        """Return gambar BGR numpy array dengan overlay mask."""
        image = cv2.imread(image_path)
        if not polygons:
            return image

        alpha = getattr(getattr(self._config, "canvas", None), "mask_opacity", 0.45)
        overlay = image.copy()

        for polygon in polygons:
            pts = _to_pts(polygon.points)
            cv2.fillPoly(overlay, [pts], _bgr(polygon.color))

        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        for polygon in polygons:
            pts = _to_pts(polygon.points)
            is_auto = polygon.source != "manual"

            if show_border:
                if highlight_auto and is_auto:
                    _draw_dashed_poly(blended, pts, _SAM_BORDER_BGR, 2)
                else:
                    cv2.polylines(blended, [pts], True, _bgr(polygon.color), 2)

            if highlight_auto and is_auto:
                _draw_sam_badge(blended, pts, polygon.confidence)

            if show_labels:
                _draw_label(blended, pts, polygon.instance_id)

        return blended

    def save(
        self,
        image_path: str,
        polygons: list[Polygon],
        output_path: str,
        show_labels: bool = True,
        highlight_auto: bool = False,
    ) -> str:
        """Render dan simpan ke output_path. Return path file."""
        rendered = self.render(
            image_path, polygons, show_labels=show_labels, highlight_auto=highlight_auto
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return output_path

    def render_thumbnail(
        self,
        image_path: str,
        polygons: list[Polygon],
        max_size: tuple[int, int] = (300, 300),
    ) -> QPixmap:
        """Return QPixmap thumbnail untuk panel samping UI.

        MUST be called from the PyQt5 main thread.
        """
        rendered = self.render(image_path, polygons)
        h, w = rendered.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            rendered = cv2.resize(
                rendered,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def render_comparison(
        self,
        image_path: str,
        polygons_before: list[Polygon],
        polygons_after: list[Polygon],
    ) -> np.ndarray:
        """Side-by-side: kiri = SAM raw, kanan = setelah koreksi manual."""
        left = self.render(image_path, polygons_before)
        right = self.render(image_path, polygons_after)
        return np.hstack([left, right])

    def export_all(
        self,
        image_manager: ImageManager,
        output_dir: str,
        progress_callback: callable = None,
    ) -> list[str]:
        """Render dan simpan semua gambar yang sudah dianotasi."""
        entries = [e for e in image_manager.get_all() if e.annotated]
        total = len(entries)
        saved: list[str] = []
        for idx, entry in enumerate(entries, start=1):
            polygons = image_manager.load_annotation(entry)
            stem = Path(entry.filepath).stem
            out = str(Path(output_dir) / f"{stem}_annotated.jpg")
            saved.append(self.save(entry.filepath, polygons, out))
            if progress_callback is not None:
                progress_callback(idx, total)
        return saved


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bgr(color: list[int]) -> tuple[int, int, int]:
    return (color[2], color[1], color[0])


def _to_pts(points: list[tuple[float, float]]) -> np.ndarray:
    return np.array(points, dtype=np.int32)


def _draw_dashed_poly(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        draw = True
        while pos < length:
            end = min(pos + _DASH_LEN, length)
            if draw:
                a = (int(x1 + ux * pos), int(y1 + uy * pos))
                b = (int(x1 + ux * end), int(y1 + uy * end))
                cv2.line(img, a, b, color, thickness)
            pos = end
            draw = not draw


def _draw_sam_badge(
    img: np.ndarray,
    pts: np.ndarray,
    confidence: float,
) -> None:
    min_x = int(pts[:, 0].min())
    min_y = int(pts[:, 1].min())
    text = f"SAM {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    tx, ty = min_x + 2, min_y + th + 2
    cv2.rectangle(img, (tx - 1, ty - th - 1), (tx + tw + 1, ty + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, scale, _SAM_BORDER_BGR, thick, cv2.LINE_AA)


def _draw_label(img: np.ndarray, pts: np.ndarray, instance_id: int) -> None:
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    text = f"#{instance_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    tx = cx - tw // 2
    ty = cy + th // 2
    # Semi-transparent white background via alpha blend
    patch = img.copy()
    cv2.rectangle(
        patch,
        (tx - 2, ty - th - 2),
        (tx + tw + 2, ty + baseline + 2),
        (255, 255, 255),
        -1,
    )
    cv2.addWeighted(patch, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

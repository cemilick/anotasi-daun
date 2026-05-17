from __future__ import annotations

import copy
from enum import Enum
from typing import TYPE_CHECKING

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QLabel, QWidget

from src.image_manager import ImageEntry, OcclusionLevel, Polygon

if TYPE_CHECKING:
    from src.config import Config


class CanvasMode(Enum):
    DRAW = "draw"
    SELECT = "select"
    SAM_POINT = "sam_point"
    SAM_BOX = "sam_box"
    VIEW = "view"


def compute_occlusion_levels(polygons: list[Polygon]) -> None:
    """Delegate to the real implementation in auto_labeler."""
    try:
        from src.auto_labeler import compute_occlusion_levels as _real
        # Need image_shape — use a large canvas as proxy when not available
        # Determine bounds from polygon points
        all_pts = [pt for p in polygons if not p.occlusion_manual for pt in p.points]
        if not all_pts:
            return
        max_x = int(max(pt[0] for pt in all_pts)) + 1
        max_y = int(max(pt[1] for pt in all_pts)) + 1
        _real([p for p in polygons if not p.occlusion_manual], (max_y, max_x))
    except Exception:
        # Fallback: mark everything RENDAH so canvas never blinks endlessly
        for p in polygons:
            if not p.occlusion_manual:
                p.occlusion_level = OcclusionLevel.RENDAH
                p.occlusion_ratio = 0.0


class AnnotationCanvas(QWidget):
    annotation_changed = pyqtSignal(list)
    status_message = pyqtSignal(str)
    point_clicked = pyqtSignal(float, float)
    point_refine_clicked = pyqtSignal(int, float, float, list)  # instance_id, x, y, prompt_points
    box_selected = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2 (image coords)
    has_unsaved_changes = pyqtSignal(bool)  # emit True when there are unsaved changes

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._mode: CanvasMode = CanvasMode.VIEW
        self._polygons: list[Polygon] = []
        self._pending_polygons: list[Polygon] = []
        self._entry: ImageEntry | None = None
        self._pixmap: QPixmap | None = None

        # Draw mode state
        self._active_points: list[tuple[float, float]] = []
        self._mouse_pos: tuple[float, float] | None = None

        # Select mode state
        self._selected_id: int | None = None
        self._drag_vertex: tuple[int, int] | None = None   # (polygon_idx, point_idx)
        self._drag_pre_snapshot: tuple | None = None        # snapshot before drag starts

        # SAM_BOX drag state
        self._box_start: tuple[float, float] | None = None
        self._box_end: tuple[float, float] | None = None

        # SAM_POINT refinement state
        self._active_refinement_id: int | None = None  # ID polygon yang sedang di-refine
        self._refinement_prompt_points: list[tuple[float, float]] = []  # Titik prompt untuk refinement

        # Pan state (middle-click or Ctrl+drag)
        self._panning: bool = False
        self._pan_start_mouse: tuple[float, float] | None = None

        # Zoom / pan
        self._zoom_scale: float = 1.0
        self._pan_offset: list[float] = [0.0, 0.0]

        # Undo stack (stores deep-copies of (_polygons, _pending_polygons))
        self._undo_stack: list[tuple[list[Polygon], list[Polygon]]] = []

        # Blink timer for polygons whose occlusion_level is None
        self._blink_visible: bool = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(800)
        self._blink_timer.timeout.connect(self._on_blink)
        self._blink_timer.start()

        # Toast overlay
        self._toast = QLabel(self)
        self._toast.setStyleSheet(
            "background-color: rgba(0,0,0,180); color: white;"
            " padding: 6px 12px; border-radius: 6px; font-size: 11px;"
        )
        self._toast.setAlignment(Qt.AlignCenter)
        self._toast.hide()

        # Loading spinner
        self._spinner = QLabel("⏳ SAM…", self)
        self._spinner.setStyleSheet(
            "background-color: rgba(0,0,0,160); color: white;"
            " padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        self._spinner.hide()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_image(self, entry: ImageEntry, polygons: list[Polygon]) -> None:
        self._entry = entry
        self._polygons = list(polygons)
        self._pending_polygons = []
        self._active_points = []
        self._selected_id = None
        self._drag_vertex = None
        self._drag_pre_snapshot = None
        self._undo_stack = []
        self._active_refinement_id = None
        self._refinement_prompt_points = []
        self._pixmap = QPixmap(entry.filepath)
        self.reset_zoom()

    def get_polygons(self) -> list[Polygon]:
        return [p for p in self._polygons if p.occlusion_level is not None]

    def set_mode(self, mode: CanvasMode) -> None:
        self._mode = mode
        self._active_points = []
        self._selected_id = None
        self._drag_vertex = None
        self._drag_pre_snapshot = None
        self._box_start = None
        self._box_end = None
        # Clear refinement state when switching modes
        self._active_refinement_id = None
        self._refinement_prompt_points = []
        self.update()

    def inject_polygons(self, polygons: list[Polygon]) -> None:
        self._push_undo()
        self._pending_polygons.extend(polygons)
        self._trigger_occlusion_recompute()
        self.update()

    def accept_all_auto(self) -> None:
        if not self._pending_polygons:
            return
        self._push_undo()
        for p in self._pending_polygons:
            p.is_confirmed = True
        self._polygons.extend(self._pending_polygons)
        self._pending_polygons = []
        self._trigger_occlusion_recompute()
        self.annotation_changed.emit(self.get_polygons())

    def reject_auto_polygon(self, instance_id: int) -> None:
        self._push_undo()
        self._pending_polygons = [
            p for p in self._pending_polygons if p.instance_id != instance_id
        ]
        self.update()

    def delete_selected(self) -> None:
        if self._selected_id is None:
            return
        self._push_undo()
        self._polygons = [p for p in self._polygons if p.instance_id != self._selected_id]
        self._selected_id = None
        self._trigger_occlusion_recompute()
        self.annotation_changed.emit(self.get_polygons())

    def clear_all(self) -> None:
        self._push_undo()
        self._polygons = []
        self._pending_polygons = []
        self._active_refinement_id = None
        self._refinement_prompt_points = []
        self.annotation_changed.emit(self.get_polygons())
        self.update()

    def is_refinement_active(self) -> bool:
        """Check if there's an active polygon being refined."""
        return self._active_refinement_id is not None

    def get_active_refinement_id(self) -> int | None:
        """Get the ID of the polygon currently being refined."""
        return self._active_refinement_id

    def start_refinement(self, instance_id: int, first_point: tuple[float, float]) -> None:
        """Start refining an existing polygon with the first prompt point."""
        self._active_refinement_id = instance_id
        self._refinement_prompt_points = [first_point]
        self._selected_id = instance_id
        self.has_unsaved_changes.emit(True)  # Mark as unsaved (refinement started)
        self.update()

    def finish_refinement(self) -> None:
        """Finish the current refinement session (called after Done)."""
        self._active_refinement_id = None
        self._refinement_prompt_points = []
        self._selected_id = None
        self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        prev_polygons, prev_pending = self._undo_stack.pop()
        old_final = self.get_polygons()
        self._polygons = prev_polygons
        self._pending_polygons = prev_pending
        self.update()
        if self.get_polygons() != old_final:
            self.annotation_changed.emit(self.get_polygons())

    def zoom_in(self) -> None:
        self._zoom_scale = min(10.0, self._zoom_scale * 1.1)
        self.update()

    def zoom_out(self) -> None:
        self._zoom_scale = max(0.1, self._zoom_scale / 1.1)
        self.update()

    def reset_zoom(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._zoom_scale = 1.0
            self._pan_offset = [0.0, 0.0]
            self.update()
            return
        w = self.width() or 1
        h = self.height() or 1
        pw = self._pixmap.width() or 1
        ph = self._pixmap.height() or 1
        self._zoom_scale = min(w / pw, h / ph)
        img_w = pw * self._zoom_scale
        img_h = ph * self._zoom_scale
        self._pan_offset = [(w - img_w) / 2.0, (h - img_h) / 2.0]
        self.update()

    def set_loading(self, active: bool) -> None:
        if active:
            self._spinner.adjustSize()
            self._spinner.move(self.width() - self._spinner.width() - 8, 8)
            self._spinner.show()
        else:
            self._spinner.hide()

    # ------------------------------------------------------------------ #
    # Coordinate transforms                                                #
    # ------------------------------------------------------------------ #

    def _img_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self._zoom_scale + self._pan_offset[0], y * self._zoom_scale + self._pan_offset[1]

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self._pan_offset[0]) / self._zoom_scale, (cy - self._pan_offset[1]) / self._zoom_scale

    # ------------------------------------------------------------------ #
    # Undo helpers                                                         #
    # ------------------------------------------------------------------ #

    def _push_undo(self) -> None:
        snapshot = (copy.deepcopy(self._polygons), copy.deepcopy(self._pending_polygons))
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    # ------------------------------------------------------------------ #
    # Occlusion                                                            #
    # ------------------------------------------------------------------ #

    def _trigger_occlusion_recompute(self) -> None:
        compute_occlusion_levels(self._polygons + self._pending_polygons)
        self.update()

    # ------------------------------------------------------------------ #
    # Instance utilities                                                   #
    # ------------------------------------------------------------------ #

    def _next_instance_id(self) -> int:
        all_polys = self._polygons + self._pending_polygons
        return max((p.instance_id for p in all_polys), default=0) + 1

    def _polygon_color(self, instance_id: int) -> QColor:
        hue = (instance_id * 137) % 360
        return QColor.fromHsv(hue, 200, 230)

    def _point_in_polygon(self, px: float, py: float, points: list[tuple[float, float]]) -> bool:
        n = len(points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = points[i]
            xj, yj = points[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------ #
    # Blink timer                                                          #
    # ------------------------------------------------------------------ #

    def _on_blink(self) -> None:
        all_polys = self._polygons + self._pending_polygons
        if any(p.occlusion_level is None for p in all_polys):
            self._blink_visible = not self._blink_visible
            self.update()

    # ------------------------------------------------------------------ #
    # Toast                                                                #
    # ------------------------------------------------------------------ #

    def _show_toast(self, message: str) -> None:
        self._toast.setText(message)
        self._toast.adjustSize()
        margin = 12
        self._toast.move(margin, self.height() - self._toast.height() - margin)
        self._toast.show()
        QTimer.singleShot(2000, self._toast.hide)

    # ------------------------------------------------------------------ #
    # Paint                                                                #
    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)

        if self._pixmap and not self._pixmap.isNull():
            x = int(self._pan_offset[0])
            y = int(self._pan_offset[1])
            w = int(self._pixmap.width() * self._zoom_scale)
            h = int(self._pixmap.height() * self._zoom_scale)
            painter.drawPixmap(x, y, w, h, self._pixmap)

        self._draw_polygons(painter)
        self._draw_active_polygon(painter)
        self._draw_sam_box_preview(painter)
        self._draw_refinement_points(painter)

    def _draw_polygons(self, painter: QPainter) -> None:
        for p in self._polygons + self._pending_polygons:
            is_pending = p in self._pending_polygons
            is_selected = (p.instance_id == self._selected_id and not is_pending)
            is_refining = (p.instance_id == self._active_refinement_id)
            color = self._polygon_color(p.instance_id)

            points_canvas = [self._img_to_canvas(x, y) for x, y in p.points]
            poly = QPolygonF()
            for cx, cy in points_canvas:
                poly.append(QPointF(cx, cy))

            # Fill
            fill_color = QColor(color)
            fill_color.setAlpha(60)
            painter.setBrush(QBrush(fill_color))

            # Border
            if is_refining:
                # Active refinement - cyan border
                painter.setPen(QPen(QColor(0, 255, 255), 3, Qt.SolidLine))
            elif is_selected:
                painter.setPen(QPen(Qt.white, 3, Qt.SolidLine))
            elif is_pending:
                painter.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
            elif p.occlusion_level is None:
                if self._blink_visible:
                    painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.SolidLine))
                else:
                    painter.setPen(Qt.NoPen)
            else:
                painter.setPen(QPen(color, 2, Qt.SolidLine))

            painter.drawPolygon(poly)

            # SAM badge
            if is_pending and points_canvas:
                min_x = min(c[0] for c in points_canvas)
                min_y = min(c[1] for c in points_canvas)
                painter.setPen(QPen(QColor(255, 220, 0)))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(int(min_x) + 2, int(min_y) + 12, "SAM")

            # Drag handles for selected polygon
            if is_selected:
                painter.setPen(QPen(Qt.white, 1))
                painter.setBrush(QBrush(Qt.white))
                for cx, cy in points_canvas:
                    painter.drawRect(int(cx) - 3, int(cy) - 3, 6, 6)

            # Centroid label
            self._draw_centroid_label(painter, p, points_canvas)

    def _draw_centroid_label(
        self, painter: QPainter, p: Polygon, points_canvas: list[tuple[float, float]]
    ) -> None:
        if not points_canvas or p.occlusion_level is None:
            return
        cx = sum(c[0] for c in points_canvas) / len(points_canvas)
        cy = sum(c[1] for c in points_canvas) / len(points_canvas)

        level_char = {"rendah": "R", "sedang": "S", "tinggi": "T"}.get(
            p.occlusion_level.value, "?"
        )
        pct = int((p.occlusion_ratio or 0.0) * 100)
        if p.occlusion_manual:
            label = f"#{p.instance_id} [{level_char}\U0001f512]"
        else:
            label = f"#{p.instance_id} [{level_char} {pct}%]"

        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(int(cx), int(cy), label)

    def _draw_active_polygon(self, painter: QPainter) -> None:
        if self._mode != CanvasMode.DRAW or not self._active_points:
            return
        points_canvas = [self._img_to_canvas(x, y) for x, y in self._active_points]
        pen = QPen(Qt.green, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        for i in range(len(points_canvas) - 1):
            x1, y1 = points_canvas[i]
            x2, y2 = points_canvas[i + 1]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Vertex dots
        painter.setBrush(QBrush(Qt.green))
        for cx, cy in points_canvas:
            painter.drawEllipse(int(cx) - 3, int(cy) - 3, 6, 6)

        # Preview line to mouse
        if self._mouse_pos and points_canvas:
            mx, my = self._img_to_canvas(*self._mouse_pos)
            lx, ly = points_canvas[-1]
            painter.setPen(QPen(Qt.green, 1, Qt.DashLine))
            painter.drawLine(int(lx), int(ly), int(mx), int(my))

    def _draw_sam_box_preview(self, painter: QPainter) -> None:
        if self._mode != CanvasMode.SAM_BOX or not self._box_start or not self._box_end:
            return
        x1, y1 = self._img_to_canvas(*self._box_start)
        x2, y2 = self._img_to_canvas(*self._box_end)
        painter.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        left = int(min(x1, x2))
        top = int(min(y1, y2))
        w = int(abs(x2 - x1))
        h = int(abs(y2 - y1))
        painter.drawRect(left, top, w, h)

    def _draw_refinement_points(self, painter: QPainter) -> None:
        """Draw prompt points for active refinement polygon."""
        if not self._refinement_prompt_points:
            return

        for idx, (px, py) in enumerate(self._refinement_prompt_points):
            cx, cy = self._img_to_canvas(px, py)

            # Draw point marker
            if idx == 0:
                # First point - larger, different color
                color = QColor(0, 255, 0)  # Green
                radius = 8
            else:
                # Subsequent points
                color = QColor(100, 255, 100)  # Light green
                radius = 6

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, 2))
            painter.drawEllipse(int(cx) - radius, int(cy) - radius, radius * 2, radius * 2)

            # Draw number
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(int(cx) + radius + 2, int(cy) + radius // 2, str(idx + 1))

    # ------------------------------------------------------------------ #
    # Mouse events                                                         #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.pos()
        cx, cy = float(pos.x()), float(pos.y())
        ix, iy = self._canvas_to_img(cx, cy)

        # Pan: middle-click or Ctrl+left-click (any mode)
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        ):
            self._panning = True
            self._pan_start_mouse = (cx, cy)
            return

        if self._mode == CanvasMode.DRAW:
            if event.button() == Qt.LeftButton:
                self._active_points.append((ix, iy))
                self.has_unsaved_changes.emit(True)  # Mark as unsaved
                self.status_message.emit(f"Titik: {len(self._active_points)}")
                self.update()
            elif event.button() == Qt.RightButton and self._active_points:
                self._active_points.pop()
                self.has_unsaved_changes.emit(len(self._active_points) > 0)  # Still unsaved if points remain
                self.status_message.emit(f"Titik: {len(self._active_points)}")
                self.update()

        elif self._mode == CanvasMode.SELECT:
            if event.button() == Qt.LeftButton:
                # Check vertex drag handles first (if a polygon is selected)
                if self._selected_id is not None:
                    for pidx, p in enumerate(self._polygons):
                        if p.instance_id == self._selected_id:
                            for vidx, (px, py) in enumerate(p.points):
                                pcx, pcy = self._img_to_canvas(px, py)
                                if abs(pcx - cx) <= 6 and abs(pcy - cy) <= 6:
                                    self._drag_pre_snapshot = (
                                        copy.deepcopy(self._polygons),
                                        copy.deepcopy(self._pending_polygons),
                                    )
                                    self._drag_vertex = (pidx, vidx)
                                    return
                # Hit-test polygons (top-most first)
                hit = None
                for p in reversed(self._polygons):
                    if self._point_in_polygon(ix, iy, p.points):
                        hit = p.instance_id
                        break
                self._selected_id = hit
                self.update()

        elif self._mode == CanvasMode.SAM_POINT:
            if event.button() == Qt.LeftButton:
                # Check if clicking inside the active refinement polygon
                if self._active_refinement_id is not None:
                    for p in self._polygons:
                        if p.instance_id == self._active_refinement_id:
                            if self._point_in_polygon(ix, iy, p.points):
                                # Refine existing polygon with multi-point prompt
                                self._refinement_prompt_points.append((ix, iy))
                                self.point_refine_clicked.emit(
                                    self._active_refinement_id,
                                    ix, iy,
                                    list(self._refinement_prompt_points)
                                )
                                self.status_message.emit(f"Refine polygon #{p.instance_id}: {len(self._refinement_prompt_points)} prompt points")
                                return
                # Check if clicking inside any existing polygon to start refinement
                hit = None
                for p in reversed(self._polygons):
                    if self._point_in_polygon(ix, iy, p.points):
                        hit = p.instance_id
                        break
                if hit is not None:
                    # Start refining this polygon
                    self.start_refinement(hit, (ix, iy))
                    self.point_refine_clicked.emit(hit, ix, iy, [(ix, iy)])
                    self.status_message.emit(f"Refine polygon #{hit}: 1 prompt point (tekan Done untuk final)")
                    self.update()
                else:
                    # Create new polygon
                    self._active_refinement_id = None
                    self._refinement_prompt_points = []
                    self.point_clicked.emit(ix, iy)

        elif self._mode == CanvasMode.SAM_BOX:
            if event.button() == Qt.LeftButton:
                self._box_start = (ix, iy)
                self._box_end = (ix, iy)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.pos()
        cx, cy = float(pos.x()), float(pos.y())
        ix, iy = self._canvas_to_img(cx, cy)
        self._mouse_pos = (ix, iy)

        if self._panning and self._pan_start_mouse:
            dx = cx - self._pan_start_mouse[0]
            dy = cy - self._pan_start_mouse[1]
            self._pan_offset[0] += dx
            self._pan_offset[1] += dy
            self._pan_start_mouse = (cx, cy)
            self.update()
            return

        if self._mode == CanvasMode.DRAW:
            self.update()

        elif self._mode == CanvasMode.SELECT and self._drag_vertex is not None:
            pidx, vidx = self._drag_vertex
            self._polygons[pidx].points[vidx] = (ix, iy)
            self.update()

        elif self._mode == CanvasMode.SAM_BOX and self._box_start is not None:
            self._box_end = (ix, iy)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        pos = event.pos()
        cx, cy = float(pos.x()), float(pos.y())

        if self._panning:
            self._panning = False
            self._pan_start_mouse = None
            return

        if self._mode == CanvasMode.SELECT and self._drag_vertex is not None:
            # Push pre-drag snapshot so undo restores state before drag
            if self._drag_pre_snapshot is not None:
                self._undo_stack.append(self._drag_pre_snapshot)
                if len(self._undo_stack) > 30:
                    self._undo_stack.pop(0)
                self._drag_pre_snapshot = None
            self._drag_vertex = None
            self._trigger_occlusion_recompute()
            self.annotation_changed.emit(self.get_polygons())

        elif self._mode == CanvasMode.SAM_BOX and self._box_start is not None:
            ix, iy = self._canvas_to_img(cx, cy)
            ix1, iy1 = self._box_start
            self.box_selected.emit(
                min(ix1, ix), min(iy1, iy),
                max(ix1, ix), max(iy1, iy),
            )
            self._box_start = None
            self._box_end = None
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._mode != CanvasMode.DRAW or event.button() != Qt.LeftButton:
            return
        # Qt fires mousePressEvent before mouseDoubleClickEvent for the 2nd click.
        # Remove the spurious point added by that press.
        if self._active_points:
            self._active_points.pop()

        if len(self._active_points) < 3:
            self.status_message.emit("Min 3 titik untuk menutup polygon")
            self.update()
            return

        new_id = self._next_instance_id()
        color = self._polygon_color(new_id)

        try:
            class_id = self._config.classes[0]["id"]
            class_name = self._config.classes[0]["name"]
        except (AttributeError, IndexError, KeyError):
            class_id = 1
            class_name = "unknown"

        polygon = Polygon(
            instance_id=new_id,
            class_id=class_id,
            class_name=class_name,
            points=list(self._active_points),
            color=[color.red(), color.green(), color.blue()],
            source="manual",
            confidence=1.0,
        )
        self._active_points = []
        self._push_undo()
        self._polygons.append(polygon)
        self._trigger_occlusion_recompute()
        self.annotation_changed.emit(self.get_polygons())
        self.has_unsaved_changes.emit(True)  # Mark as unsaved (polygon created)

        level = polygon.occlusion_level
        pct = int((polygon.occlusion_ratio or 0.0) * 100)
        level_name = level.value.capitalize() if level else "Belum dihitung"
        self._show_toast(f"Oklusi: {level_name} ({pct}%)")

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.angleDelta().y() > 0:
            self._zoom_scale = min(10.0, self._zoom_scale * 1.1)
        else:
            self._zoom_scale = max(0.1, self._zoom_scale / 1.1)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()

        if self._mode == CanvasMode.DRAW and key == Qt.Key_Escape:
            self._active_points = []
            self.update()
            return

        if self._mode == CanvasMode.SELECT:
            if key == Qt.Key_Delete:
                self.delete_selected()
                return

            if self._selected_id is not None:
                _level_map = {
                    Qt.Key_1: OcclusionLevel.RENDAH,
                    Qt.Key_2: OcclusionLevel.SEDANG,
                    Qt.Key_3: OcclusionLevel.TINGGI,
                }
                if key in _level_map:
                    for p in self._polygons:
                        if p.instance_id == self._selected_id:
                            p.occlusion_level = _level_map[key]
                            p.occlusion_manual = True
                    self.update()
                    return
                if key == Qt.Key_0:
                    for p in self._polygons:
                        if p.instance_id == self._selected_id:
                            p.occlusion_manual = False
                    self._trigger_occlusion_recompute()
                    return

        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key == Qt.Key_Minus:
            self.zoom_out()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap and not self._pixmap.isNull():
            self.reset_zoom()

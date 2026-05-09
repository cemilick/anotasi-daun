## 1. Data Model Extensions

- [x] 1.1 Add `OcclusionLevel` enum (`RENDAH`, `SEDANG`, `TINGGI`) to `src/annotation_engine.py`
- [x] 1.2 Add `CanvasMode` enum (`DRAW`, `SELECT`, `SAM_POINT`, `SAM_BOX`, `VIEW`) to `src/annotation_engine.py`
- [x] 1.3 Extend `Polygon` dataclass in `src/image_manager.py` with fields: `occlusion_level: OcclusionLevel | None = None`, `occlusion_ratio: float | None = None`, `occlusion_manual: bool = False`, `is_confirmed: bool = False`
- [x] 1.4 Update `ImageManager.save_annotation()` to route JSON to `annotations/rendah/`, `annotations/sedang/`, or `annotations/tinggi/` based on highest `occlusion_level` in the polygon list
- [x] 1.5 Update `ImageManager.load_annotation()` to search all three occlusion subfolders when locating an existing annotation file
- [x] 1.6 Update JSON serialization/deserialization in `ImageManager` to round-trip all four new Polygon fields (`occlusion_level`, `occlusion_ratio`, `occlusion_manual`, `is_confirmed`)

## 2. AnnotationCanvas Core Setup

- [x] 2.1 Create `src/annotation_engine.py` with `AnnotationCanvas(QWidget)` skeleton: `__init__`, signals (`annotation_changed`, `status_message`, `point_clicked`), and `_polygons`/`_pending_polygons` lists
- [x] 2.2 Implement `load_image(entry, polygons)`: store image, set `_polygons`, reset active drawing state, call `reset_zoom()`, trigger `repaint()`
- [x] 2.3 Implement coordinate transform helpers: `_img_to_canvas(x, y) -> (cx, cy)` and `_canvas_to_img(cx, cy) -> (x, y)` using current zoom scale and pan offset
- [x] 2.4 Implement `paintEvent`: draw image scaled by zoom+pan, then call `_draw_polygons()` and `_draw_active_polygon()`
- [x] 2.5 Implement `_draw_polygons()`: iterate `_polygons` and `_pending_polygons`, dispatch to correct render style per polygon status
- [x] 2.6 Implement `set_mode(mode: CanvasMode)`: update `_mode`, reset transient state (active polygon, drag state), call `update()`
- [x] 2.7 Implement `get_polygons()`: return `[p for p in _polygons if p.occlusion_level is not None]`

## 3. Draw Mode

- [x] 3.1 Implement `mousePressEvent` for DRAW: left-click appends point to `_active_points`; right-click removes last point; emit `status_message` with current point count
- [x] 3.2 Implement `mouseDoubleClickEvent` for DRAW: if `len(_active_points) >= 3`, close polygon — create `Polygon`, append to `_polygons`, push undo snapshot, call `_trigger_occlusion_recompute()`, emit `annotation_changed`, show toast
- [x] 3.3 Implement `keyPressEvent` for `Escape` in DRAW mode: clear `_active_points` and call `update()`
- [x] 3.4 Implement `_draw_active_polygon()`: draw lines between `_active_points` and a line from last point to current mouse position as a preview

## 4. Select Mode

- [x] 4.1 Implement `mousePressEvent` for SELECT: hit-test `_polygons` to find clicked polygon (point-in-polygon check), set `_selected_id`; click on empty area clears selection
- [x] 4.2 Implement drag handle hit-test: on `mousePressEvent` in SELECT, if click is within 6px (canvas space) of a vertex, set `_drag_vertex = (polygon_idx, point_idx)`
- [x] 4.3 Implement `mouseMoveEvent` for SELECT with active drag: update `Polygon.points[point_idx]` to `_canvas_to_img(mouse_pos)` and call `update()`
- [x] 4.4 Implement `mouseReleaseEvent` for SELECT drag: push undo snapshot, call `_trigger_occlusion_recompute()`, emit `annotation_changed`, clear `_drag_vertex`
- [x] 4.5 Implement `keyPressEvent` for `Delete` in SELECT: call `delete_selected()`
- [x] 4.6 Implement `delete_selected()`: remove selected polygon from `_polygons`, push undo snapshot, call `_trigger_occlusion_recompute()`, emit `annotation_changed`
- [x] 4.7 Implement `keyPressEvent` for `1`/`2`/`3` in SELECT: set `occlusion_level` and `occlusion_manual = True` on selected polygon, call `update()`
- [x] 4.8 Implement `keyPressEvent` for `0` in SELECT: set `occlusion_manual = False`, call `_trigger_occlusion_recompute()`
- [x] 4.9 Implement `_draw_polygons()` rendering for selected polygon: white 3px solid border + 6x6 filled squares at each vertex

## 5. SAM Modes

- [x] 5.1 Implement `mousePressEvent` for SAM_POINT: left-click converts to image coords via `_canvas_to_img()` and emits `point_clicked(x, y)`
- [x] 5.2 Implement `mousePressEvent` and `mouseMoveEvent` for SAM_BOX: track drag start/end; `mouseReleaseEvent` converts to image coords and emits `box_selected` signal (or stores for external query)
- [x] 5.3 Implement `_draw_sam_box_preview()`: draw dashed rectangle from drag start to current mouse position during SAM_BOX drag

## 6. Inject, Accept, Reject

- [x] 6.1 Implement `inject_polygons(polygons)`: push undo snapshot, append to `_pending_polygons`, call `_trigger_occlusion_recompute()`, call `update()` (do NOT emit `annotation_changed`)
- [x] 6.2 Implement `accept_all_auto()`: push undo snapshot, set `is_confirmed = True` on all pending, move all from `_pending_polygons` to `_polygons`, call `_trigger_occlusion_recompute()`, emit `annotation_changed`
- [x] 6.3 Implement `reject_auto_polygon(instance_id)`: push undo snapshot, remove matching polygon from `_pending_polygons`, call `update()` (do NOT emit `annotation_changed`)
- [x] 6.4 Implement `accept_all_auto` button hook: wire to Enter key or toolbar button (in spec-06 scope — skip here, just ensure method is public)

## 7. Occlusion Rendering and Computation

- [x] 7.1 Implement `_trigger_occlusion_recompute()`: call `compute_occlusion_levels(_polygons + _pending_polygons)` from `auto_labeler.py`; skip polygons where `occlusion_manual = True`; update `occlusion_level` and `occlusion_ratio` on remaining polygons
- [x] 7.2 Add stub `compute_occlusion_levels(polygons)` that returns all `OcclusionLevel.RENDAH` with `occlusion_ratio = 0.0`, to be replaced by spec-07 implementation
- [x] 7.3 Implement blink QTimer (800ms interval): toggle `_blink_visible` flag, call `update()` if any polygon in `_polygons + _pending_polygons` has `occlusion_level = None`
- [x] 7.4 Implement border rendering for `occlusion_level = None`: orange border drawn only when `_blink_visible = True`
- [x] 7.5 Implement centroid label rendering: compute centroid of polygon points, draw `#<id> [R <pct>%]`/`[S ...]`/`[T ...]` at centroid; use `🔒` suffix when `occlusion_manual = True`
- [x] 7.6 Implement toast notification: `QLabel` overlay in canvas corner, shown after polygon closure with text `"Oklusi: Rendah (12%)"`, hidden via `QTimer.singleShot(2000, toast.hide)`

## 8. Loading Spinner

- [x] 8.1 Add `set_loading(active: bool)` method: show/hide a spinner `QLabel` overlay in the top-right corner of the canvas
- [x] 8.2 Implement spinner as animated dots or static "⏳" label (no external gif dependency); shown above the image layer via absolute positioning in the widget

## 9. Undo Stack

- [x] 9.1 Implement `_push_undo()`: deep-copy `(_polygons, _pending_polygons)` and append to `_undo_stack`; if `len > 30`, pop the oldest entry
- [x] 9.2 Implement `undo()`: if `_undo_stack` is non-empty, pop last snapshot, restore `_polygons` and `_pending_polygons`, call `update()`, emit `annotation_changed` if `_polygons` changed

## 10. Zoom and Pan

- [x] 10.1 Implement `wheelEvent`: increment/decrement `_zoom_scale` by 10% per scroll step; clamp to range [0.1, 10.0]; call `update()`
- [x] 10.2 Implement `keyPressEvent` for `+`/`-`: call `zoom_in()` / `zoom_out()`; implement `zoom_in()` and `zoom_out()` as ±10% scale change
- [x] 10.3 Implement middle-click drag pan: on `mousePressEvent` with middle button, start pan; `mouseMoveEvent` updates `_pan_offset`; call `update()`
- [x] 10.4 Implement Ctrl+drag pan: same as middle-click but triggered when Ctrl is held during left-button drag
- [x] 10.5 Implement `reset_zoom()`: compute scale to fit image in widget (min of width ratio, height ratio), center image, call `update()`

## 11. Utility Methods

- [x] 11.1 Implement `clear_all()`: push undo snapshot, clear `_polygons` and `_pending_polygons`, emit `annotation_changed`, call `update()`
- [x] 11.2 Implement `_next_instance_id()`: return `max(p.instance_id for p in _polygons + _pending_polygons) + 1` or `1` if empty
- [x] 11.3 Implement `_polygon_color(instance_id)`: return deterministic `QColor` per `instance_id` using HSV hue cycling (e.g., `hue = (instance_id * 137) % 360`)
- [x] 11.4 Implement `_point_in_polygon(px, py, points)`: ray-casting algorithm in image coordinates for SELECT hit-test

## ADDED Requirements

### Requirement: Canvas renders image with polygon overlays
`AnnotationCanvas` SHALL display the loaded `ImageEntry` image scaled to fit the widget, and render all polygons in `_polygons` and `_pending_polygons` on top of the image using coordinate transforms from image pixel space to widget space.

#### Scenario: Image loaded with existing polygons
- **WHEN** `load_image(entry, polygons)` is called with a non-empty polygon list
- **THEN** the canvas repaints showing the image and all polygons rendered at correct positions

#### Scenario: Empty polygon list
- **WHEN** `load_image(entry, [])` is called
- **THEN** the canvas shows only the image with no polygon overlays

### Requirement: Draw mode polygon creation
In `CanvasMode.DRAW`, `AnnotationCanvas` SHALL add a point to the active polygon on each left-click, remove the last point on right-click, close the polygon on double-click (if at least 3 points), and cancel the active polygon on `Escape`. A closed polygon is added to `_polygons` and `annotation_changed` is emitted.

#### Scenario: Draw polygon with three clicks and double-click
- **WHEN** user left-clicks 3 times then double-clicks in DRAW mode
- **THEN** a new `Polygon` with 3 points appears in `get_polygons()` and `annotation_changed` is emitted

#### Scenario: Right-click removes last point
- **WHEN** user has 2 points in active polygon and right-clicks
- **THEN** the active polygon has 1 point and `get_polygons()` is unchanged

#### Scenario: Double-click with fewer than 3 points does not close
- **WHEN** user has 2 points and double-clicks
- **THEN** the polygon is NOT closed and stays in the active drawing state

#### Scenario: Escape cancels active polygon
- **WHEN** user has 1 or more points in active polygon and presses `Escape`
- **THEN** the active polygon is cleared and `get_polygons()` is unchanged

### Requirement: Select mode polygon editing
In `CanvasMode.SELECT`, `AnnotationCanvas` SHALL highlight the clicked polygon with a white 3px border and 6x6 drag handles on each vertex. Dragging a vertex SHALL update the polygon's `points` in place. Pressing `Delete` SHALL remove the selected polygon. Clicking an empty area SHALL deselect.

#### Scenario: Click selects polygon
- **WHEN** user clicks inside a polygon in SELECT mode
- **THEN** that polygon is highlighted with a white border and drag handles

#### Scenario: Drag vertex moves point
- **WHEN** user drags a vertex handle to a new position in SELECT mode
- **THEN** the polygon's corresponding point is updated to the new coordinates in image pixel space

#### Scenario: Delete removes selected polygon
- **WHEN** a polygon is selected and user presses `Delete`
- **THEN** the polygon is removed from `_polygons`, `annotation_changed` is emitted, and `compute_occlusion_levels` is called

#### Scenario: Click empty area deselects
- **WHEN** user clicks on an area with no polygon in SELECT mode
- **THEN** no polygon is highlighted and selection is cleared

### Requirement: SAM_POINT mode emits click signal
In `CanvasMode.SAM_POINT`, `AnnotationCanvas` SHALL emit `point_clicked(x, y)` with image pixel coordinates on left-click, where `x` and `y` are the actual image coordinates (not widget coordinates).

#### Scenario: Left-click emits point_clicked
- **WHEN** user left-clicks in SAM_POINT mode
- **THEN** `point_clicked` signal is emitted with coordinates corresponding to the image pixel at that position

#### Scenario: Coordinates are in image space not widget space
- **WHEN** user clicks at widget position (200, 150) and the image is displayed at 0.5x zoom offset (50, 30)
- **THEN** emitted coordinates reflect the image pixel position, not (200, 150)

### Requirement: SAM_BOX mode emits bounding box
In `CanvasMode.SAM_BOX`, `AnnotationCanvas` SHALL track a drag gesture and emit a bounding box in image pixel coordinates when the drag ends. The drag rectangle SHALL be shown as a dashed overlay during dragging.

#### Scenario: Drag creates bounding box
- **WHEN** user drags from point A to point B in SAM_BOX mode
- **THEN** a dashed rectangle is drawn during the drag and the bounding box in image coordinates is available after mouse release

### Requirement: Inject polygons from AutoLabeler
`inject_polygons(polygons)` SHALL add the given polygons to `_pending_polygons` with their rendering showing a yellow dashed border and a "SAM" badge. Injected polygons SHALL NOT appear in `get_polygons()`. After inject, `compute_occlusion_levels` SHALL be called for the combined set of `_polygons + _pending_polygons`.

#### Scenario: Injected polygons not in get_polygons
- **WHEN** `inject_polygons([p1, p2])` is called
- **THEN** `get_polygons()` does not contain `p1` or `p2`

#### Scenario: Injected polygons visible with yellow dashed border
- **WHEN** `inject_polygons([p1])` is called
- **THEN** the canvas repaints showing `p1` with a yellow dashed border and "SAM" badge

#### Scenario: Occlusion recomputed after inject
- **WHEN** `inject_polygons([p1])` is called
- **THEN** `compute_occlusion_levels` is called with `_polygons + _pending_polygons`

### Requirement: Accept and reject SAM polygons
`accept_all_auto()` SHALL move all polygons from `_pending_polygons` to `_polygons`, set `is_confirmed = True` on each, emit `annotation_changed`, and call `compute_occlusion_levels`. `reject_auto_polygon(instance_id)` SHALL remove the matching polygon from `_pending_polygons` without affecting `_polygons`.

#### Scenario: accept_all_auto moves pending to final
- **WHEN** `_pending_polygons` has 2 polygons and `accept_all_auto()` is called
- **THEN** `get_polygons()` contains those 2 polygons with `is_confirmed = True` and `_pending_polygons` is empty

#### Scenario: reject_auto_polygon removes from pending
- **WHEN** `_pending_polygons` has polygon with `instance_id=5` and `reject_auto_polygon(5)` is called
- **THEN** the polygon is no longer in `_pending_polygons` and `_polygons` is unchanged

### Requirement: Polygon rendering by status
`AnnotationCanvas` SHALL render each polygon with distinct visual styles:
- Manual polygon (`source="manual"`, confirmed): semi-transparent fill + solid 2px border in instance color
- Pending SAM polygon (`is_confirmed=False`): semi-transparent fill + yellow dashed border + "SAM" badge at top-left of bounding box
- Selected polygon (SELECT mode): same fill + white 3px solid border + 6x6 drag handle squares
- Polygon with `occlusion_level = None`: same fill + orange border that toggles visibility every 800ms

#### Scenario: Manual polygon has solid border
- **WHEN** a manual polygon is rendered
- **THEN** it uses a solid 2px border in its instance color

#### Scenario: Pending SAM polygon has yellow dashed border
- **WHEN** a pending SAM polygon is rendered
- **THEN** it uses a yellow dashed border and shows a "SAM" text badge

#### Scenario: Uncomputed occlusion triggers blinking border
- **WHEN** a polygon has `occlusion_level = None`
- **THEN** its border alternates between visible and invisible at 800ms intervals

### Requirement: Centroid label with occlusion badge
`AnnotationCanvas` SHALL render a text label at each polygon's centroid showing `#<id> [R <pct>%]`, `#<id> [S <pct>%]`, or `#<id> [T <pct>%]` based on `occlusion_level` and `occlusion_ratio`. Polygons with `occlusion_manual = True` SHALL show a lock symbol: `#<id> [R🔒]`.

#### Scenario: Label shows level and percentage
- **WHEN** a polygon has `occlusion_level = OcclusionLevel.SEDANG` and `occlusion_ratio = 0.45`
- **THEN** its centroid label reads `#<id> [S 45%]`

#### Scenario: Manual override shows lock badge
- **WHEN** a polygon has `occlusion_manual = True` and `occlusion_level = OcclusionLevel.RENDAH`
- **THEN** its centroid label reads `#<id> [R🔒]`

### Requirement: Occlusion auto-recomputed on polygon mutation
`compute_occlusion_levels()` SHALL be called automatically after: polygon closed in DRAW mode, `inject_polygons()` called, polygon deleted in SELECT mode, polygon vertex dragged (on mouse release). Polygons with `occlusion_manual = True` SHALL be excluded from recomputation (their values are preserved).

#### Scenario: Recomputed after polygon closed
- **WHEN** a polygon is closed by double-click in DRAW mode
- **THEN** `compute_occlusion_levels` is called and `occlusion_level` is set on the new polygon before `paintEvent`

#### Scenario: Manual override preserved during recompute
- **WHEN** `compute_occlusion_levels` is called and a polygon has `occlusion_manual = True`
- **THEN** that polygon's `occlusion_level` and `occlusion_ratio` are not modified

### Requirement: Occlusion manual override via keyboard
In `CanvasMode.SELECT` with a polygon selected, pressing `1`, `2`, or `3` SHALL set `occlusion_level` to `RENDAH`, `SEDANG`, or `TINGGI` respectively and set `occlusion_manual = True`. Pressing `0` SHALL set `occlusion_manual = False` and trigger `compute_occlusion_levels` to restore the computed value.

#### Scenario: Key 2 sets SEDANG with manual flag
- **WHEN** a polygon is selected and user presses `2`
- **THEN** `occlusion_level = OcclusionLevel.SEDANG` and `occlusion_manual = True`

#### Scenario: Key 0 resets override
- **WHEN** a polygon with `occlusion_manual = True` is selected and user presses `0`
- **THEN** `occlusion_manual = False` and `compute_occlusion_levels` is called

### Requirement: Toast notification after polygon closed
After a polygon is closed in DRAW mode, `AnnotationCanvas` SHALL display a 2-second toast overlay in the canvas corner showing the computed occlusion level and percentage, e.g., `"Oklusi: Rendah (12%)"`.

#### Scenario: Toast appears with correct info
- **WHEN** a polygon is closed and `occlusion_level = OcclusionLevel.RENDAH` with `occlusion_ratio = 0.12`
- **THEN** a toast showing `"Oklusi: Rendah (12%)"` is visible for 2 seconds then disappears

### Requirement: Loading spinner during SAM processing
`AnnotationCanvas` SHALL display a spinner overlay at the canvas corner while `set_loading(True)` is active, indicating SAM inference is in progress. The spinner is hidden when `set_loading(False)` is called.

#### Scenario: Spinner visible during loading
- **WHEN** `set_loading(True)` is called
- **THEN** a spinner indicator is visible on the canvas

#### Scenario: Spinner hidden after loading
- **WHEN** `set_loading(False)` is called
- **THEN** the spinner is no longer visible

### Requirement: Undo stack with 30-step limit
`AnnotationCanvas` SHALL maintain an undo stack of up to 30 states. `undo()` SHALL restore the previous state of `_polygons` and `_pending_polygons`. Actions that push to undo stack: close polygon (DRAW), delete polygon (SELECT), drag vertex (mouse release), `inject_polygons`, `accept_all_auto`, `reject_auto_polygon`, `clear_all`.

#### Scenario: Undo reverses polygon closure
- **WHEN** a polygon is closed in DRAW mode and `undo()` is called
- **THEN** the closed polygon is removed from `_polygons` and the canvas returns to the state before closure

#### Scenario: Undo reverses inject
- **WHEN** `inject_polygons([p])` is called and `undo()` is called
- **THEN** `_pending_polygons` is empty again

#### Scenario: Undo stack limited to 30 steps
- **WHEN** 31 state-mutating actions are performed
- **THEN** only the most recent 30 states are undoable; the oldest is discarded

### Requirement: Zoom and pan
`AnnotationCanvas` SHALL support zoom via scroll wheel and `+`/`-` keys, and pan via middle-click drag or `Ctrl+drag`. `zoom_in()`, `zoom_out()`, and `reset_zoom()` (fit-to-canvas) SHALL be callable programmatically. Zoom SHALL not modify stored polygon coordinates.

#### Scenario: Zoom does not change polygon coordinates
- **WHEN** `zoom_in()` is called 5 times
- **THEN** `get_polygons()[0].points` values are unchanged

#### Scenario: reset_zoom fits image to widget
- **WHEN** `reset_zoom()` is called
- **THEN** the image is scaled and centered to fit the widget dimensions

### Requirement: get_polygons returns only confirmed polygons with occlusion computed
`get_polygons()` SHALL return only polygons in `_polygons` (not `_pending_polygons`) where `occlusion_level is not None`.

#### Scenario: Pending polygons excluded
- **WHEN** `_pending_polygons` has 2 items and `_polygons` has 3 items
- **THEN** `get_polygons()` returns exactly 3 items

#### Scenario: Polygons with None occlusion excluded
- **WHEN** a polygon in `_polygons` has `occlusion_level = None`
- **THEN** `get_polygons()` does not include that polygon

### Requirement: annotation_changed emitted only on final polygon list change
`annotation_changed` SHALL be emitted with the current `get_polygons()` result only when `_polygons` changes (polygon added, removed, vertex moved, or accepted from pending). It SHALL NOT be emitted when `_pending_polygons` changes alone.

#### Scenario: annotation_changed not emitted on inject
- **WHEN** `inject_polygons([p])` is called
- **THEN** `annotation_changed` signal is NOT emitted

#### Scenario: annotation_changed emitted on accept_all_auto
- **WHEN** `accept_all_auto()` is called with non-empty `_pending_polygons`
- **THEN** `annotation_changed` is emitted once with the updated polygon list

### Requirement: Annotation JSON saved to occlusion subfolder
`AnnotationCanvas` SHALL determine the output subfolder for the annotation JSON based on the highest occlusion level present in `_polygons`: `annotations/tinggi/` if any polygon is `TINGGI`, else `annotations/sedang/` if any is `SEDANG`, else `annotations/rendah/`.

#### Scenario: Highest occlusion determines subfolder
- **WHEN** polygons include RENDAH and SEDANG instances
- **THEN** the annotation is saved under `annotations/sedang/<basename>.json`

#### Scenario: All RENDAH goes to rendah subfolder
- **WHEN** all polygons have `occlusion_level = OcclusionLevel.RENDAH`
- **THEN** the annotation is saved under `annotations/rendah/<basename>.json`

## 1. Scaffold MainWindow

- [x] 1.1 Create `src/main_window.py` with `MainWindow(QMainWindow)` class accepting `Config`
- [x] 1.2 Instantiate sub-modules in `__init__`: ImageManager, AnnotationCanvas, AutoLabeler, Visualizer, Exporter
- [x] 1.3 Set up main layout: canvas center, right panel `QWidget`, status bar

## 2. Toolbar

- [x] 2.1 Build first toolbar row: Open Folder, Save, separator, Auto Label, Auto All, separator, Draw, Select
- [x] 2.2 Build second toolbar row: Export COCO Splits, Export XML, Export Visualized, separator, Zoom+, Zoom−, Fit
- [x] 2.3 Wire each toolbar action to its corresponding MainWindow method

## 3. Right Panel — Dataset Stats

- [x] 3.1 Add dataset stats section: total images label, annotated/total progress bar, auto-labeled count label
- [x] 3.2 Implement `_update_stats()` to refresh stats whenever annotations or image set change

## 4. Right Panel — SAM Controls

- [x] 4.1 Add SAM Controls group box with [Point], [Box], [Auto] buttons
- [x] 4.2 Add confidence slider (0.50–0.99, default 0.75, step 0.01) with numeric label
- [x] 4.3 Add [✓ Accept All] and [✗ Reject All] buttons
- [x] 4.4 Disable SAM controls when `AutoLabeler.is_ready()` is False; show "SAM loading…" in status bar

## 5. Right Panel — Instance List

- [x] 5.1 Build scrollable `QListWidget` for instance list in right panel
- [x] 5.2 Implement `_update_instance_list(polygons)` to populate rows with color swatch, ID, source, confidence badge, ⚠️ for SAM pending
- [x] 5.3 Connect row click to select the corresponding polygon on canvas
- [x] 5.4 Add × delete button per row; connect to remove polygon from canvas
- [x] 5.5 Cap display at 50 rows with "… and N more" overflow label

## 6. Right Panel — Thumbnail Preview

- [x] 6.1 Add a `QLabel` thumbnail area at the bottom of the right panel
- [x] 6.2 Update thumbnail whenever annotations change on the active image

## 7. Status Bar

- [x] 7.1 Implement `_update_status_bar()` updating filename, image index, annotation count, active mode, zoom level
- [x] 7.2 Call `_update_status_bar()` on every navigation, mode change, zoom change, and annotation change

## 8. Keyboard Shortcuts

- [x] 8.1 Register `QShortcut` for N (next), P (prev), S (save)
- [x] 8.2 Register `QShortcut` for A (SAM_POINT), B (SAM_BOX), D (DRAW), V (SELECT)
- [x] 8.3 Register `QShortcut` for Enter (accept all), Escape (reject / cancel), Delete (delete selected)
- [x] 8.4 Register `QShortcut` for Ctrl+Z (undo), + (zoom in), - (zoom out), F (fit)

## 9. Image Navigation & Auto-Save

- [x] 9.1 Implement `_load_image_to_canvas(entry)`: save current accepted polygons before loading new image
- [x] 9.2 Implement `next_image()` and `prev_image()` using ImageManager; guard boundaries
- [x] 9.3 Show "Tersimpan ✓" in status bar for 2 s after auto-save

## 10. SAM Point & Box Integration

- [x] 10.1 Implement `_on_point_clicked(x, y)`: forward to AutoLabeler; on result call `_on_sam_result()`
- [x] 10.2 Implement `_on_sam_result(polygons)`: filter by confidence slider, inject as pending polygons into canvas
- [x] 10.3 Implement `trigger_sam_point()` and `trigger_sam_box()`: switch canvas mode, update SAM controls UI
- [x] 10.4 Implement `accept_all_auto()`: set all pending polygons to accepted, refresh instance list
- [x] 10.5 Implement `reject_selected_auto()` (Reject All): remove all pending polygons, refresh instance list

## 11. Auto-Label Active Image

- [x] 11.1 Implement `trigger_sam_auto_current()`: run AutoLabeler full-image in background, inject results as pending

## 12. Auto-Label All (Batch)

- [x] 12.1 Implement `trigger_sam_auto_all()`: show confirmation dialog with unannotated count and confidence slider
- [x] 12.2 Create `AutoLabelWorker(QThread)` that processes images sequentially, emitting progress signals
- [x] 12.3 Connect worker progress to `QProgressDialog`; wire Cancel button to worker interrupt flag
- [x] 12.4 On completion show summary dialog: "X gambar diproses, Y polygon digenerate"

## 13. Export Actions

- [x] 13.1 Implement `export_coco_splits()`: call Exporter, show summary dialog with per-split image counts
- [x] 13.2 Implement `export_voc()`: call Exporter, confirm output path to user
- [x] 13.3 Implement `export_visualized()`: call Visualizer render, confirm output path to user

## 14. Application Entry Point

- [x] 14.1 Create `main.py` that loads `Config`, instantiates `MainWindow`, and starts the Qt event loop

## 15. Acceptance Criteria Verification

- [x] 15.1 Open Folder loads images and updates progress bar
- [x] 15.2 SAM_POINT click triggers SAM and shows yellow-border pending polygon
- [x] 15.3 Accept All moves pending polygons to final list and updates thumbnail
- [x] 15.4 Auto All runs in background thread (UI stays responsive) with progress dialog
- [x] 15.5 Instance list shows "SAM 0.92 ⚠️" badge for pending SAM polygons
- [x] 15.6 Export COCO Splits produces 3 JSON files and shows split summary
- [x] 15.7 Auto-save on navigation excludes pending SAM polygons
- [x] 15.8 All keyboard shortcuts from spec table work correctly
- [x] 15.9 Status bar reflects current mode, image, annotation count, and zoom at all times

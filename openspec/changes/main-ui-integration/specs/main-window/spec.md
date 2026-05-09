## ADDED Requirements

### Requirement: Application launches with MainWindow
The system SHALL provide a `MainWindow(QMainWindow)` in `src/main_window.py` that accepts a `Config` object, initialises all sub-modules (ImageManager, AnnotationCanvas, AutoLabeler, Visualizer, Exporter), and shows a fully composed GUI on startup.

#### Scenario: Successful launch
- **WHEN** the application entry point is executed with a valid `Config`
- **THEN** a window opens showing an empty canvas, an inactive toolbar, a right panel with zeroed stats, and a status bar

### Requirement: Open Folder loads dataset
The system SHALL allow the user to open a folder via toolbar or keyboard shortcut, causing ImageManager to scan it and the UI to display the first image with an updated progress bar.

#### Scenario: Valid folder selected
- **WHEN** user clicks Open Folder and selects a directory containing image files
- **THEN** the canvas shows the first image, the dataset stat shows total image count, and the progress bar reflects annotated vs total

#### Scenario: Empty folder selected
- **WHEN** user selects a folder with no recognisable images
- **THEN** canvas remains empty and status bar shows "No images found"

### Requirement: Save current annotation
The system SHALL save the active image's accepted polygons to persistent storage when the user presses Save (toolbar or `S` key), excluding any pending SAM polygons.

#### Scenario: Save with accepted polygons
- **WHEN** user presses S
- **THEN** accepted polygons are written to storage and status bar briefly shows "Tersimpan ✓"

#### Scenario: Pending SAM polygons not saved
- **WHEN** user presses S while SAM pending polygons exist on canvas
- **THEN** only accepted polygons are saved; pending polygons remain on canvas unchanged

### Requirement: Navigate between images with auto-save
The system SHALL navigate to the next or previous image when the user presses N or P (or toolbar arrows), automatically saving accepted polygons for the current image before switching.

#### Scenario: Navigate forward
- **WHEN** user presses N (or clicks ▶)
- **THEN** current accepted polygons are saved, canvas loads the next image, and the status bar updates to show the new filename and position

#### Scenario: Navigate at boundary
- **WHEN** user presses N on the last image
- **THEN** nothing changes and status bar shows "Gambar terakhir"

### Requirement: SAM Point mode
The system SHALL enter SAM_POINT mode when user clicks [Point] button or presses A, causing each subsequent canvas click to forward (x, y) to AutoLabeler and inject the returned polygon as a pending polygon with a yellow border.

#### Scenario: Point click triggers SAM
- **WHEN** mode is SAM_POINT and user clicks on canvas
- **THEN** AutoLabeler receives the click coordinates and returns a polygon that appears on canvas with yellow border and pending status

#### Scenario: SAM not ready
- **WHEN** user tries to enter SAM_POINT mode before AutoLabeler is initialised
- **THEN** the button is disabled and status bar shows "SAM loading…"

### Requirement: SAM Box mode
The system SHALL enter SAM_BOX mode when user clicks [Box] button or presses B, causing a drag on the canvas to define a bounding box that is forwarded to AutoLabeler, with the result injected as a pending polygon.

#### Scenario: Box drag triggers SAM
- **WHEN** mode is SAM_BOX and user drags on canvas
- **THEN** AutoLabeler receives the bounding box and returns a polygon that appears with yellow border and pending status

### Requirement: Accept all pending SAM polygons
The system SHALL move all pending SAM polygons to the accepted list when user clicks [✓ Accept All] or presses Enter.

#### Scenario: Accept all
- **WHEN** pending SAM polygons exist and user presses Enter
- **THEN** all pending polygons become accepted, their border color changes from yellow to the normal annotation color, and the instance list updates

### Requirement: Reject all pending SAM polygons
The system SHALL discard all pending SAM polygons when user clicks [✗ Reject All] or presses Escape.

#### Scenario: Reject all
- **WHEN** pending SAM polygons exist and user presses Escape
- **THEN** all pending polygons are removed from canvas and the instance list updates

### Requirement: Confidence threshold slider
The system SHALL provide a slider (range 0.50–0.99, default 0.75) in the SAM Controls panel that filters out SAM results below the set confidence before they are injected as pending polygons.

#### Scenario: Low-confidence result filtered
- **WHEN** AutoLabeler returns a polygon with confidence below the slider value
- **THEN** the polygon is not added to canvas

### Requirement: Auto-label active image
The system SHALL run full-image automatic SAM segmentation on the current image when user clicks [✨ Auto Label] or presses the toolbar button, injecting all results above the confidence threshold as pending polygons.

#### Scenario: Auto label current image
- **WHEN** user triggers Auto Label on an image
- **THEN** AutoLabeler runs in the background and all qualifying polygons appear as pending with yellow borders upon completion

### Requirement: Auto-label all images with progress dialog
The system SHALL display a dialog and run batch auto-labeling in a background thread when user clicks [⚡ Auto All], keeping the UI responsive and allowing cancellation.

#### Scenario: Batch starts
- **WHEN** user clicks Auto All and confirms the dialog
- **THEN** a QProgressDialog appears, a background QThread processes images one by one, and the dialog updates after each image

#### Scenario: Batch cancelled
- **WHEN** user clicks Cancel in the progress dialog
- **THEN** the worker stops after finishing the current image and a summary shows how many were processed

#### Scenario: Batch completes
- **WHEN** all images are processed
- **THEN** a summary dialog shows "X gambar diproses, Y polygon digenerate" and pending polygons await review per image

### Requirement: Instance list panel
The system SHALL show a scrollable list of annotation instances for the current image in the right panel, each row showing a color swatch, ID, source label, and — for SAM polygons — a confidence badge and ⚠️ review indicator.

#### Scenario: Instance displayed with SAM badge
- **WHEN** a SAM polygon with confidence 0.92 is in the accepted list
- **THEN** the row shows "SAM 0.92 ⚠️"

#### Scenario: Click row selects instance
- **WHEN** user clicks an instance row
- **THEN** the corresponding polygon is selected on canvas

#### Scenario: Delete button removes instance
- **WHEN** user clicks × on an instance row
- **THEN** the polygon is removed from canvas and the list updates

### Requirement: Status bar
The system SHALL keep a status bar at the bottom of the window that at all times shows the current filename, image index, annotation count, active mode, and zoom level.

#### Scenario: Status bar updates on navigation
- **WHEN** user navigates to a new image
- **THEN** status bar shows updated filename, position (e.g. "2/12"), and annotation count

#### Scenario: Status bar shows SAM mode
- **WHEN** mode is SAM_POINT or SAM_BOX
- **THEN** status bar mode segment shows "SAM_POINT" or "SAM_BOX" respectively

### Requirement: Keyboard shortcuts
The system SHALL support all shortcuts defined in the spec: N, P, S, A, B, D, V, Enter, Escape, Delete, Ctrl+Z, +, -, F.

#### Scenario: All shortcuts functional
- **WHEN** any key in the shortcut table is pressed
- **THEN** the corresponding action executes as specified

### Requirement: Zoom controls
The system SHALL provide zoom-in, zoom-out, and fit-to-window controls via toolbar buttons and + / - / F keys.

#### Scenario: Zoom in
- **WHEN** user presses + or clicks Zoom+
- **THEN** canvas zoom level increases and status bar updates

#### Scenario: Fit to window
- **WHEN** user presses F or clicks Fit
- **THEN** canvas scales to show the full image within the available area

### Requirement: Export actions
The system SHALL trigger the appropriate Exporter method when the user clicks Export COCO Splits, Export XML, or Export Visualized in the toolbar.

#### Scenario: Export COCO Splits
- **WHEN** user clicks Export COCO Splits
- **THEN** Exporter generates train/val/test JSON files in output/coco/ and a summary dialog shows image counts per split

#### Scenario: Export Visualized
- **WHEN** user clicks Export Visualized
- **THEN** Visualizer renders all annotated images to output/visualized/

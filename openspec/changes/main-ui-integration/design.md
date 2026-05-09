## Context

All core modules (ImageManager, AnnotationCanvas, AutoLabeler, Visualizer, Exporter) are implemented independently. There is no application shell that wires them into a launchable GUI. This design covers `MainWindow(QMainWindow)` in `src/main_window.py` — the final integration layer that every other module feeds into.

The app targets PyQt5. SAM inference can be slow (1–3 s per image); batch operations must never freeze the UI.

## Goals / Non-Goals

**Goals:**
- Single `MainWindow` class that composes all modules via their public APIs
- Two-row toolbar, right-side panel (stats + SAM controls + instance list + thumbnail), bottom status bar
- Full keyboard shortcut table from spec-06
- Auto-save on image navigation (pending SAM polygons excluded)
- Batch Auto-Label All running in a `QThread` with a progress dialog
- SAM point/box click forwarded from canvas signals to AutoLabeler; results injected back as pending polygons

**Non-Goals:**
- Dark mode, live training, or import from external formats (LabelMe/CVAT)
- Changing any existing module's internal implementation

## Decisions

**D1 — Right panel as a plain `QWidget` with `QVBoxLayout` (not `QDockWidget`)**
Using a fixed right panel keeps layout predictable for a single-user desktop tool. Floating dock windows add complexity with no benefit here. If resizability is needed later, a `QSplitter` can be dropped in around the canvas.

**D2 — Background thread via `QThread` + `pyqtSignal` for Auto-Label All**
`QThread` with a worker object is the idiomatic PyQt5 pattern. A `QProgressDialog` with `cancel()` connected to the worker's interrupt flag gives the user a cancellation path. Alternative (blocking main thread with `QApplication.processEvents`) was rejected because it risks deadlocks if SAM emits signals.

**D3 — Pending SAM polygons stored on `AnnotationCanvas`, not `MainWindow`**
The canvas already owns all polygon state. "Pending" is just a flag on each `Polygon` (e.g., `Polygon.source == "sam_pending"`). Accept/Reject operations change that flag; auto-save skips pending polygons by filtering on the flag. This keeps MainWindow stateless w.r.t. polygon data.

**D4 — Keyboard shortcuts via `QShortcut` objects created in `__init__`**
Centralised shortcut creation is easier to audit than scattered `keyPressEvent` overrides. Each shortcut maps directly to an existing method.

**D5 — Auto-save implemented in `_load_image_to_canvas()`**
Before loading a new image the method saves the current image's accepted polygons. This single chokepoint means both N/P keys and thumbnail clicks auto-save without duplicating logic.

## Risks / Trade-offs

- **SAM model not loaded when user clicks Point/Box** → Auto-label controls are disabled until `AutoLabeler.is_ready()` returns `True`; status bar shows "SAM loading…" during initialisation.
- **Batch Auto-Label All on a large dataset can take minutes** → The QProgressDialog has a Cancel button that sets a threading.Event on the worker; worker checks it between images.
- **Instance list grows large** → Capped at showing 50 rows; a "… and N more" label replaces overflow. This avoids layout thrashing on datasets with hundreds of instances.
- **Auto-save loses data if the app crashes mid-navigation** → Acceptable for v1; a crash-recovery checkpoint is out of scope.

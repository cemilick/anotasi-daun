## Why

The annotation pipeline has all core modules built (ImageManager, AnnotationCanvas, AutoLabeler, Visualizer, Exporter) but no unified application shell to connect them. Users need a single launchable PyQt5 window that wires every module together with toolbar, side panels, keyboard shortcuts, and SAM auto-labeling controls.

## What Changes

- Introduce `src/main_window.py` — `MainWindow(QMainWindow)` that hosts all existing modules
- Add a two-row toolbar: file/navigation, SAM auto-label triggers, mode toggles, export actions, and zoom controls
- Add a right-side panel with dataset stats/progress bar, SAM controls (Point/Box/Auto, confidence slider, Accept/Reject), instance list with badges, and thumbnail preview
- Add a bottom status bar showing current image name, annotation count, active mode, and zoom level
- Implement full keyboard shortcut set (N/P/S/A/B/D/V/Enter/Escape/Delete/Ctrl+Z/+/-/F)
- Implement auto-save on image navigation (pending SAM polygons excluded)
- Implement batch Auto-Label All dialog with background thread processing and progress feedback

## Capabilities

### New Capabilities

- `main-window`: The top-level `MainWindow` application shell integrating ImageManager, AnnotationCanvas, AutoLabeler, Visualizer, and Exporter into a fully functional PyQt5 annotation application with toolbar, SAM controls panel, instance list panel, status bar, keyboard shortcuts, and auto-save behavior.

### Modified Capabilities

## Impact

- New file: `src/main_window.py`
- New entry point: `main.py` (or equivalent launcher)
- Depends on: `src/image_manager.py`, `src/annotation_canvas.py`, `src/auto_labeler.py`, `src/visualizer.py`, `src/exporter.py`, `src/config.py`
- No breaking changes to existing modules — MainWindow consumes their public APIs

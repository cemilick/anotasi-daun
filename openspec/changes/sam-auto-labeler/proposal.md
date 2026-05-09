## Why

The annotation tool has a stub `src/auto_labeler.py` (created for spec-06 UI integration) that exposes the correct public API but performs no real inference — SAM model loading and all predict methods return empty lists. Full SAM-based auto-labeling must now be implemented so users can actually generate polygon annotations from point clicks, box drags, and full-image automatic segmentation.

## What Changes

- Replace the stub `AutoLabeler` body with a complete SAM ViT-H/L/B implementation:
  - `load_model()` — loads SAM checkpoint to CUDA/CPU; auto-falls back to CPU if CUDA unavailable
  - `predict_from_point()` — point-prompt segmentation; picks best of 3 multi-mask candidates
  - `predict_from_box()` — box-prompt segmentation; returns polygon within dragged bbox
  - `predict_automatic()` — full-image SamAutomaticMaskGenerator; filters by confidence + min area; computes occlusion levels
  - `predict_automatic_batch()` — convenience wrapper over multiple entries
- Replace the stub `compute_occlusion_levels()` with the full mask-rasterization + overlap-ratio implementation
- Replace the stub `mask_to_polygon()` with the cv2 contour + Douglas-Peucker simplification implementation
- `AutoLabelWorker(QThread)` already has its full implementation from spec-06; verify signals and graceful stop work correctly

## Capabilities

### New Capabilities

- `sam-inference`: Full SAM inference pipeline — model loading, point/box/automatic prediction, mask-to-polygon conversion, and occlusion level computation — implemented in `src/auto_labeler.py`.

### Modified Capabilities

## Impact

- `src/auto_labeler.py` — complete rewrite of AutoLabeler method bodies (stubs → real SAM calls); AutoLabelWorker body already correct
- New runtime dependency: `segment-anything>=1.0`, `torch>=2.0.0`, `torchvision>=0.15.0` (already in `requirements.txt`)
- `src/annotation_engine.py` — `compute_occlusion_levels` stub can be replaced by the real implementation from `auto_labeler.py` (or left as-is since the canvas calls it only as a fallback)
- No breaking changes to `MainWindow` or any other module — they already call the correct API

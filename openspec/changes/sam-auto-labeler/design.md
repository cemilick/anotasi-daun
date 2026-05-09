## Context

`src/auto_labeler.py` was created as part of spec-06 (MainWindow integration) with stub implementations: `load_model()` returns immediately without loading, all predict methods return `[]`, and `compute_occlusion_levels` / `mask_to_polygon` have placeholder implementations. The file structure, class hierarchy, and public API are correct and are consumed by `MainWindow` already. This change fills in the method bodies only — no API surface changes.

SAM (Segment Anything Model) from Meta AI is the inference backend. The ViT-H checkpoint (~2.4 GB) is the default; ViT-L and ViT-B are also supported via `config.sam.model_type`. Inference runs on CUDA if available, CPU otherwise. Polygon output feeds directly into `MainWindow` → `AnnotationCanvas` via the existing signal chain.

## Goals / Non-Goals

**Goals:**
- `load_model()` loads the SAM checkpoint to the configured device; auto-falls back to CPU if CUDA is unavailable
- `predict_from_point()` runs SAM with a single point prompt; returns the best of 3 multi-mask candidates as a `Polygon`
- `predict_from_box()` runs SAM with a bounding-box prompt; returns the single best mask as a `Polygon`
- `predict_automatic()` runs `SamAutomaticMaskGenerator`; filters by confidence threshold and min area; computes occlusion levels on the full result set
- `compute_occlusion_levels()` correctly computes per-polygon overlap ratios using rasterized masks and classifies into RENDAH/SEDANG/TINGGI
- `mask_to_polygon()` converts a binary mask to a simplified polygon via `cv2.findContours` + Douglas-Peucker
- `AutoLabelWorker` stops gracefully when `stop()` is called between images

**Non-Goals:**
- Fine-tuning SAM on the dataset
- SAM 2 or SAM-HQ support
- Concurrent multi-image inference (single-threaded worker is sufficient)
- Negative point prompts UI (API supports it, but UI doesn't expose it yet)

## Decisions

**D1 — Keep all SAM state in AutoLabeler (predictor + generator as instance variables)**
`SamPredictor.set_image()` caches image embeddings. Storing the predictor on the instance avoids re-initialising it per call, enabling embedding reuse if the same image is predicted multiple times (point + box on the same frame). Alternative (re-init each call) is simpler but wastes 1–3 s per predict call on embedding computation.

**D2 — pick best multi-mask by `iou_prediction` score (argmax)**
SAM's `multimask_output=True` returns 3 masks of different scales. The IoU prediction score is the best single-signal proxy for mask quality without requiring ground truth. Alternatives (area heuristics, stability score) are less reliable.

**D3 — compute_occlusion_levels uses rasterized binary masks (NumPy bitwise ops)**
Polygon-polygon geometric intersection (Shapely) would be more exact but adds a heavy dependency and is slower for large polygons. Rasterizing to boolean arrays at image resolution and using `np.bitwise_and` is exact enough for annotation purposes (error < 1 pixel) and uses already-available cv2/numpy.

**D4 — mask_to_polygon uses cv2.findContours on the largest external contour only**
Multiple disconnected blobs in a single SAM mask are unusual for leaf segmentation (leaves are mostly convex). Taking the largest contour discards small holes/artefacts without needing multi-polygon support in the data model.

**D5 — AutoLabelWorker checks stop_event between images, not within predict_automatic**
Checking inside predict_automatic would require threading-aware changes to the generator loop (which is opaque SAM internals). Checking between images gives a coarser but safe and non-invasive stop boundary.

## Risks / Trade-offs

- **[OOM on CUDA with ViT-H]** → `load_model()` catches `RuntimeError` and retries on CPU; status reported via `is_loaded()`
- **[SAM checkpoint missing]** → `load_model()` checks path existence before loading; `is_loaded()` returns `False`; SAM controls in UI remain disabled (handled by MainWindow)
- **[Very slow on CPU (20–60 s/image)]** → Acceptable for v1; Auto All batches use worker thread so UI stays responsive
- **[Large images cause GPU OOM during embedding]** → Not mitigated in v1; users should resize images > 4096px before labeling (out of scope)
- **[compute_occlusion_levels is O(N²) in mask count]** → For typical leaf images (< 50 polygons) this is negligible; would need optimisation beyond ~200 polygons

## Migration Plan

The stub in `auto_labeler.py` can be replaced method-by-method. The public API is identical, so `MainWindow` requires no changes. If the SAM checkpoint is not yet downloaded, `is_loaded()` stays False and the UI gracefully shows disabled SAM controls — no user-visible regression.

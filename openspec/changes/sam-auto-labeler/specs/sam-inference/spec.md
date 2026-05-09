## ADDED Requirements

### Requirement: Model loads to correct device with fallback
`AutoLabeler.load_model()` SHALL load the SAM checkpoint specified in `config.paths.sam_checkpoint` to the device in `config.sam.device`. If the device is `"cuda"` but CUDA is unavailable, it SHALL silently fall back to `"cpu"`. After a successful load `is_loaded()` SHALL return `True`; if the checkpoint file does not exist or loading fails, `is_loaded()` SHALL return `False`.

#### Scenario: Successful load on GPU
- **WHEN** `load_model()` is called with a valid checkpoint path and CUDA is available
- **THEN** `is_loaded()` returns `True` and the predictor is bound to the CUDA device

#### Scenario: CPU fallback when CUDA unavailable
- **WHEN** config specifies `device: "cuda"` but `torch.cuda.is_available()` returns `False`
- **THEN** `load_model()` loads the model on CPU and `is_loaded()` returns `True`

#### Scenario: Missing checkpoint
- **WHEN** the checkpoint file path does not exist on disk
- **THEN** `load_model()` returns without raising and `is_loaded()` returns `False`

### Requirement: Point-prompt prediction returns best polygon
`AutoLabeler.predict_from_point()` SHALL accept an RGB numpy image and a `(x, y)` point, invoke SAM with `multimask_output=True` to obtain 3 candidate masks, select the candidate with the highest `iou_prediction` score, convert it to a polygon via `mask_to_polygon()`, and return a single-element `list[Polygon]` with `source="sam_point"` and `confidence` equal to the winning IoU score.

#### Scenario: Valid point on leaf returns polygon
- **WHEN** a point inside a leaf region is passed to `predict_from_point()`
- **THEN** a list with one `Polygon` is returned, with at least 3 points and `source == "sam_point"`

#### Scenario: Confidence reflects IoU score
- **WHEN** `predict_from_point()` succeeds
- **THEN** `polygon.confidence` equals the `iou_prediction` score of the winning mask (0.0–1.0)

#### Scenario: Empty polygon from degenerate mask returns empty list
- **WHEN** the winning mask produces fewer than 3 contour points after simplification
- **THEN** `predict_from_point()` returns `[]`

#### Scenario: Model not loaded returns empty list
- **WHEN** `predict_from_point()` is called before `load_model()`
- **THEN** it returns `[]` without raising

### Requirement: Box-prompt prediction returns polygon within bbox
`AutoLabeler.predict_from_box()` SHALL accept an RGB numpy image and a `(x_min, y_min, x_max, y_max)` tuple, invoke SAM with the box as the prompt, convert the resulting mask to a polygon, and return a single-element `list[Polygon]` with `source="sam_point"` and `confidence` equal to the predicted IoU score.

#### Scenario: Valid box around leaf returns polygon
- **WHEN** a tight bounding box around a leaf is passed to `predict_from_box()`
- **THEN** a list with one `Polygon` is returned that fits within or overlaps the box region

#### Scenario: Model not loaded returns empty list
- **WHEN** `predict_from_box()` is called before `load_model()`
- **THEN** it returns `[]` without raising

### Requirement: Automatic segmentation filters by confidence and area
`AutoLabeler.predict_automatic()` SHALL run `SamAutomaticMaskGenerator` (configured from `config.sam`) on the full image, filter out masks with `predicted_iou` below the threshold passed (or the config default) and masks with area below `config.sam.min_polygon_area_px`, sort results by area descending, convert each to a `Polygon` with `source="sam_auto"`, and compute occlusion levels for the full result set before returning.

#### Scenario: All returned polygons meet confidence threshold
- **WHEN** `predict_automatic()` is called with `confidence_threshold=0.75`
- **THEN** every polygon in the returned list has `confidence >= 0.75`

#### Scenario: Small noise masks are filtered
- **WHEN** a mask has area < `config.sam.min_polygon_area_px`
- **THEN** it is not included in the returned list

#### Scenario: Occlusion levels are set on all returned polygons
- **WHEN** `predict_automatic()` returns a non-empty list
- **THEN** every polygon has `occlusion_level` that is not `None`

#### Scenario: Source is sam_auto
- **WHEN** `predict_automatic()` succeeds
- **THEN** every returned polygon has `source == "sam_auto"`

#### Scenario: Model not loaded returns empty list
- **WHEN** `predict_automatic()` is called before `load_model()`
- **THEN** it returns `[]` without raising

### Requirement: mask_to_polygon converts binary mask to simplified polygon
`mask_to_polygon(binary_mask, epsilon)` SHALL find the largest external contour in the binary mask, apply `cv2.approxPolyDP` with the given `epsilon` (Douglas-Peucker simplification), and return the resulting `list[tuple[float, float]]`. If no contour is found or fewer than 3 points remain after simplification, it SHALL return `[]`.

#### Scenario: Valid mask returns polygon with at least 3 points
- **WHEN** a binary mask with a clear leaf-shaped blob is passed
- **THEN** the returned list has at least 3 `(float, float)` tuples

#### Scenario: Empty mask returns empty list
- **WHEN** a binary mask with all zeros is passed
- **THEN** `mask_to_polygon()` returns `[]`

#### Scenario: Simplification respects epsilon
- **WHEN** a larger epsilon is used
- **THEN** the returned polygon has fewer points than with a smaller epsilon

### Requirement: compute_occlusion_levels classifies overlap correctly
`compute_occlusion_levels(polygons, image_shape)` SHALL rasterize each polygon, compute the ratio of each polygon's area that overlaps with the union of all other polygons, and set `occlusion_level` and `occlusion_ratio` on each polygon: ratio < 0.30 → `RENDAH`, 0.30–0.60 → `SEDANG`, > 0.60 → `TINGGI`. A single polygon or a zero-area polygon SHALL always receive `RENDAH` with `occlusion_ratio = 0.0`.

#### Scenario: Non-overlapping polygons are RENDAH
- **WHEN** two polygons with no spatial overlap are passed
- **THEN** both receive `occlusion_level == OcclusionLevel.RENDAH` and `occlusion_ratio == 0.0`

#### Scenario: Fully covered polygon is TINGGI
- **WHEN** a small polygon is entirely contained within a larger polygon
- **THEN** the small polygon receives `occlusion_level == OcclusionLevel.TINGGI`

#### Scenario: Single polygon is RENDAH
- **WHEN** a list containing exactly one polygon is passed
- **THEN** it receives `occlusion_level == OcclusionLevel.RENDAH` and `occlusion_ratio == 0.0`

#### Scenario: Partial overlap is SEDANG
- **WHEN** two polygons overlap by approximately 45% of one polygon's area
- **THEN** that polygon receives `occlusion_level == OcclusionLevel.SEDANG`

### Requirement: AutoLabelWorker stops gracefully between images
`AutoLabelWorker.stop()` SHALL set an internal flag that causes the `run()` loop to exit after the current image finishes processing. It SHALL NOT raise an exception and SHALL NOT interrupt mid-inference. The `finished` signal SHALL still be emitted with the count of polygons generated before the stop.

#### Scenario: Stop mid-batch exits after current image
- **WHEN** `stop()` is called while the worker is processing image N of M
- **THEN** the worker finishes image N, does not process N+1 through M, and emits `finished` with the total polygon count so far

#### Scenario: Worker emits progress for each completed image
- **WHEN** the worker processes images successfully
- **THEN** `progress` signal is emitted after each image with `(current, total, filename)`

#### Scenario: Worker emits result_ready for each completed image
- **WHEN** the worker processes an image and produces polygons
- **THEN** `result_ready` signal is emitted with `(filename, list[Polygon])`

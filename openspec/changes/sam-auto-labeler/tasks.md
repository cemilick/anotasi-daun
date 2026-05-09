## 1. Utilities — mask_to_polygon

- [ ] 1.1 Implement `mask_to_polygon(binary_mask, epsilon)`: run `cv2.findContours` with `RETR_EXTERNAL` + `CHAIN_APPROX_SIMPLE`, take largest contour by area
- [ ] 1.2 Apply `cv2.approxPolyDP(contour, epsilon, closed=True)` and extract `(float, float)` point list
- [ ] 1.3 Return `[]` if no contours found or fewer than 3 points after simplification
- [ ] 1.4 Write unit test: valid mask → ≥ 3 points; empty mask → `[]`; larger epsilon → fewer points

## 2. Utilities — compute_occlusion_levels

- [ ] 2.1 Implement rasterization loop: for each polygon, `cv2.fillPoly` a zeros mask at `image_shape`
- [ ] 2.2 For each polygon compute overlap ratio: `bitwise_and(own_mask, union_of_others).sum() / own_area`
- [ ] 2.3 Assign `occlusion_level` based on thresholds (< 0.30 → RENDAH, ≤ 0.60 → SEDANG, > 0.60 → TINGGI) and set `occlusion_ratio`
- [ ] 2.4 Handle edge cases: single polygon → RENDAH; zero-area polygon → RENDAH with ratio 0.0
- [ ] 2.5 Write unit tests: non-overlapping → both RENDAH; fully contained small poly → TINGGI; single poly → RENDAH

## 3. AutoLabeler — Model Loading

- [ ] 3.1 In `load_model()`, import `sam_model_registry`, `SamPredictor`, `SamAutomaticMaskGenerator` from `segment_anything`
- [ ] 3.2 Check checkpoint path exists; return early (no raise) if missing
- [ ] 3.3 Load `sam_model_registry[model_type](checkpoint=checkpoint)` and call `sam.to(device=device)`
- [ ] 3.4 Wrap CUDA-specific logic: catch `RuntimeError` on `.to("cuda")` and retry on `"cpu"`
- [ ] 3.5 Assign `self._predictor = SamPredictor(sam)` and `self._generator = SamAutomaticMaskGenerator(sam, ...)` using config params
- [ ] 3.6 Set `self._loaded = True` only on full success; catch all exceptions → `self._loaded = False`

## 4. AutoLabeler — predict_from_point

- [ ] 4.1 Guard: return `[]` if `not self._loaded`
- [ ] 4.2 Call `self._predictor.set_image(image)` (RGB numpy array)
- [ ] 4.3 Build `point_coords` and `point_labels` arrays; include optional `negative_points` with label 0
- [ ] 4.4 Call `self._predictor.predict(point_coords=..., point_labels=..., multimask_output=True)` → `masks, scores, _`
- [ ] 4.5 Select `best_idx = np.argmax(scores)`; call `mask_to_polygon(masks[best_idx], epsilon)` → points
- [ ] 4.6 Return `[]` if points is empty; otherwise build and return `[Polygon(..., source="sam_point", confidence=float(scores[best_idx]))]`

## 5. AutoLabeler — predict_from_box

- [ ] 5.1 Guard: return `[]` if `not self._loaded`
- [ ] 5.2 Call `self._predictor.set_image(image)`
- [ ] 5.3 Call `self._predictor.predict(box=np.array([x1,y1,x2,y2]), multimask_output=False)` → `masks, scores, _`
- [ ] 5.4 Return `[]` if no masks; call `mask_to_polygon(masks[0], epsilon)` → points
- [ ] 5.5 Return `[]` if points is empty; otherwise build and return `[Polygon(..., source="sam_point", confidence=float(scores[0]))]`

## 6. AutoLabeler — predict_automatic

- [ ] 6.1 Guard: return `[]` if `not self._loaded`
- [ ] 6.2 Call `self._generator.generate(image)` → list of mask dicts; sort by `"area"` descending
- [ ] 6.3 Filter: skip masks with `area < config.sam.min_polygon_area_px`
- [ ] 6.4 For each remaining mask: call `mask_to_polygon(mask["segmentation"], epsilon)` → points; skip if empty
- [ ] 6.5 Build `Polygon` with `source="sam_auto"`, `confidence=mask["predicted_iou"]`, auto-assigned `instance_id` starting from `instance_id_start`
- [ ] 6.6 After collecting all polygons: call `compute_occlusion_levels(polygons, image.shape[:2])`
- [ ] 6.7 Call `progress_callback(idx, total)` after each mask if provided
- [ ] 6.8 Return the filtered, sorted, occlusion-annotated polygon list

## 7. AutoLabelWorker — Verify and Harden

- [ ] 7.1 Confirm `stop_event` is checked at the top of each loop iteration (after-image, not mid-inference)
- [ ] 7.2 Ensure `finished` signal is emitted even when stop is triggered mid-batch
- [ ] 7.3 Ensure `error` signal emits per-image error string and loop continues (one bad image does not abort the batch)
- [ ] 7.4 Add `instance_id_start` continuity: accumulate `total_polygons` so IDs don't collide across images in a batch

## 8. Integration — annotation_engine.py stub replacement

- [ ] 8.1 Update `compute_occlusion_levels` stub in `annotation_engine.py` to call the real implementation from `auto_labeler.py` (import and delegate)
- [ ] 8.2 Verify canvas `_trigger_occlusion_recompute()` still works correctly after the delegation

## 9. Tests

- [ ] 9.1 Write test for `mask_to_polygon` covering: valid blob, empty mask, epsilon effect
- [ ] 9.2 Write test for `compute_occlusion_levels` covering: no overlap → RENDAH, full containment → TINGGI, single polygon → RENDAH, partial overlap → SEDANG
- [ ] 9.3 Write smoke test for `AutoLabeler.load_model()` that handles missing checkpoint gracefully (`is_loaded() == False`)
- [ ] 9.4 Write mock-based test for `predict_from_point` that stubs `SamPredictor.predict` and verifies polygon construction and confidence assignment
- [ ] 9.5 Write mock-based test for `predict_automatic` that stubs `SamAutomaticMaskGenerator.generate` and verifies filtering, sorting, and occlusion level assignment
- [ ] 9.6 Write test for `AutoLabelWorker.stop()`: confirm batch halts after current image and `finished` is emitted

## 10. Acceptance Criteria Verification

- [ ] 10.1 `load_model()` loads SAM ViT-H to CUDA/CPU without OOM (manual test with checkpoint present)
- [ ] 10.2 `predict_from_point()` returns a valid polygon (≥ 3 points) for a click on a leaf region
- [ ] 10.3 `predict_from_box()` returns a polygon covering the dragged box area
- [ ] 10.4 `predict_automatic()` returns polygons all with `confidence >= threshold` and no area < `min_polygon_area_px`
- [ ] 10.5 All SAM polygons have correct `source` field (`"sam_point"` / `"sam_auto"`)
- [ ] 10.6 `AutoLabelWorker` runs in a QThread — UI does not freeze during batch
- [ ] 10.7 `AutoLabelWorker.stop()` halts gracefully; `finished` is still emitted
- [ ] 10.8 `predict_automatic()` polygons have non-None `occlusion_level` on return

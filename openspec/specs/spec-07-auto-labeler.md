# Spec 07 — Auto-Labeler (SAM-Based Automatic Annotation)

## Goal

Modul auto-labeling berbasis Segment Anything Model (SAM) dari Meta AI untuk menghasilkan polygon segmentasi daun secara otomatis — baik via klik titik, drag box, maupun full-image automatic segmentation.

## Background

SAM (ViT-H checkpoint) dapat mensegmentasi objek apa pun hanya dari satu titik klik atau bounding box. Untuk dataset daun kelengkeh itoh, ini sangat efektif karena daun memiliki warna dan tekstur yang kontras terhadap latar. Polygon hasil SAM disimpan dengan `source = "sam_point"` / `"sam_auto"` dan perlu dikonfirmasi user sebelum masuk ke anotasi final.

## Model SAM

| Variant | Checkpoint | Akurasi | Kecepatan |
|---------|-----------|---------|-----------|
| ViT-H (default) | `sam_vit_h_4b8939.pth` (~2.4GB) | Tertinggi | ~2s/gambar (GPU) |
| ViT-L | `sam_vit_l_0b3195.pth` (~1.2GB) | Tinggi | ~1s/gambar |
| ViT-B | `sam_vit_b_01ec64.pth` (~375MB) | Cukup | ~0.5s/gambar |

Download: `https://github.com/facebookresearch/segment-anything#model-checkpoints`

## Module: `src/auto_labeler.py`

### Class: `AutoLabeler`

```python
class AutoLabeler:
    def __init__(self, config: Config): ...

    def load_model(self) -> None: ...
    """Load SAM checkpoint ke GPU/CPU. Panggil sekali saat startup."""

    def is_loaded(self) -> bool: ...

    def predict_from_point(
        self,
        image: np.ndarray,
        point: tuple[float, float],
        negative_points: list[tuple[float, float]] = None,
        instance_id_start: int = 1,
    ) -> list[Polygon]:
        """
        Segmentasi dari satu titik klik positif (+ opsional negative points).
        SAM mengembalikan 3 kandidat mask; pilih yang confidence tertinggi.
        Return list[Polygon] (biasanya 1 polygon).
        """
        ...

    def predict_from_box(
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float],  # x_min, y_min, x_max, y_max
        instance_id_start: int = 1,
    ) -> list[Polygon]:
        """
        Segmentasi dari bounding box yang di-drag user.
        Return list[Polygon] (biasanya 1 polygon).
        """
        ...

    def predict_automatic(
        self,
        image: np.ndarray,
        instance_id_start: int = 1,
        progress_callback: callable = None,
    ) -> list[Polygon]:
        """
        Full-image automatic segmentation menggunakan SamAutomaticMaskGenerator.
        Filter hasil berdasarkan confidence threshold dari config.
        Return list[Polygon] semua objek yang terdeteksi.
        """
        ...

    def predict_automatic_batch(
        self,
        image_entries: list[ImageEntry],
        confidence_threshold: float,
        progress_callback: callable = None,
    ) -> dict[str, list[Polygon]]:
        """
        Batch auto-label beberapa gambar.
        Return dict {filename: list[Polygon]}.
        Gunakan QThread agar UI tidak freeze.
        """
        ...
```

### Class: `AutoLabelWorker(QThread)`

```python
class AutoLabelWorker(QThread):
    """Worker thread untuk batch auto-labeling agar UI tidak freeze."""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    result_ready = pyqtSignal(str, list)  # filename, list[Polygon]
    finished = pyqtSignal(int)            # total polygons generated
    error = pyqtSignal(str)              # error message

    def __init__(self, auto_labeler: AutoLabeler, entries: list[ImageEntry],
                 confidence_threshold: float): ...
    def run(self) -> None: ...
    def stop(self) -> None: ...           # graceful stop
```

## Alur Prediksi SAM

### Mode: Point Prompt

```
User klik (x, y) di canvas
→ canvas.point_clicked.emit(x, y)
→ MainWindow._on_point_clicked(x, y)
→ AutoLabeler.predict_from_point(image_np, (x, y))
   → sam_predictor.set_image(image_np)
   → sam_predictor.predict(
         point_coords=[[x, y]],
         point_labels=[1],          # 1 = foreground
         multimask_output=True      # dapat 3 kandidat
     )
   → pilih mask dengan iou_prediction tertinggi
   → mask → contour via cv2.findContours
   → simplify contour ke polygon (Douglas-Peucker, epsilon=2.0)
   → buat Polygon(source="sam_point", confidence=iou_score)
→ MainWindow._on_sam_result([polygon])
→ canvas.inject_polygons([polygon])
```

### Mode: Box Prompt

```
User drag box (x1,y1) → (x2,y2)
→ AutoLabeler.predict_from_box(image_np, (x1,y1,x2,y2))
   → sam_predictor.predict(box=np.array([[x1,y1,x2,y2]]))
   → proses sama seperti point prompt
```

### Mode: Automatic (Full Image)

```
AutoLabeler.predict_automatic(image_np)
→ mask_generator = SamAutomaticMaskGenerator(
      model=sam,
      points_per_side=config.sam.points_per_side,      # 32
      pred_iou_thresh=config.sam.pred_iou_thresh,      # 0.88
      stability_score_thresh=config.sam.stability_score_thresh,  # 0.95
  )
→ masks = mask_generator.generate(image_np)
→ filter: mask["predicted_iou"] >= confidence_threshold
→ filter: area polygon > min_area (500 px²) untuk hilangkan noise
→ sort by area descending (daun besar dulu)
→ convert setiap mask["segmentation"] → polygon via contour
→ compute_occlusion_levels(polygons, image_np.shape[:2])  ← otomatis set occlusion_level
→ return list[Polygon] dengan source="sam_auto", occlusion_level sudah terisi
```

### Konversi Mask → Polygon

```python
def mask_to_polygon(binary_mask: np.ndarray, epsilon: float = 2.0) -> list[tuple]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    # Ambil contour terbesar
    largest = max(contours, key=cv2.contourArea)
    # Douglas-Peucker simplification
    approx = cv2.approxPolyDP(largest, epsilon, closed=True)
    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return points if len(points) >= 3 else []
```

### Auto-Komputasi Tingkat Oklusi

```python
def compute_occlusion_levels(
    polygons: list[Polygon],
    image_shape: tuple[int, int],  # (height, width)
) -> list[Polygon]:
    """
    Hitung tingkat oklusi setiap polygon berdasarkan tumpang tindih antar polygon.

    Untuk setiap polygon A:
        occlusion_ratio = area(A ∩ union(semua polygon lain)) / area(A)

    Threshold:
        < 0.30  → OcclusionLevel.RENDAH
        0.30–0.60 → OcclusionLevel.SEDANG
        > 0.60  → OcclusionLevel.TINGGI

    Tidak ada dependency baru — hanya np + cv2.fillPoly.
    """
    h, w = image_shape

    # Rasterize setiap polygon ke binary mask
    masks = []
    for poly in polygons:
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(poly.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)

    for i, poly in enumerate(polygons):
        own_mask = masks[i]
        own_area = int(own_mask.sum())

        if own_area == 0 or len(polygons) == 1:
            poly.occlusion_level = OcclusionLevel.RENDAH
            poly.occlusion_ratio = 0.0
            continue

        # Union semua polygon lain, lalu intersect dengan polygon ini
        others = np.zeros((h, w), dtype=np.uint8)
        for j, other_mask in enumerate(masks):
            if i != j:
                others = np.bitwise_or(others, other_mask)

        overlap_area = int(np.bitwise_and(own_mask, others).sum())
        ratio = overlap_area / own_area
        poly.occlusion_ratio = ratio  # simpan untuk ditampilkan di UI

        if ratio < 0.30:
            poly.occlusion_level = OcclusionLevel.RENDAH
        elif ratio <= 0.60:
            poly.occlusion_level = OcclusionLevel.SEDANG
        else:
            poly.occlusion_level = OcclusionLevel.TINGGI

    return polygons
```

**Kapan dipanggil:**

| Skenario | Kapan | Polygon yang dihitung |
|----------|-------|----------------------|
| `predict_automatic()` | Setelah semua polygon selesai dibuat | Seluruh list baru |
| `predict_from_point()` / `predict_from_box()` | Setelah inject, sebelum dikembalikan | Polygon baru + semua polygon existing di canvas |
| Polygon manual dihapus/diubah | Dipicu dari `annotation_engine` | Seluruh polygon yang tersisa di canvas |

`compute_occlusion_levels()` tersedia sebagai fungsi modul (`src/auto_labeler.py`) dan dapat dipanggil dari luar modul oleh `AnnotationCanvas`.

## Konfigurasi SAM (dari `config.json`)

```json
"sam": {
  "model_type": "vit_h",
  "device": "cuda",
  "points_per_side": 32,
  "pred_iou_thresh": 0.88,
  "stability_score_thresh": 0.95,
  "auto_confidence_threshold": 0.75,
  "min_polygon_area_px": 500,
  "polygon_simplify_epsilon": 2.0
}
```

## Acceptance Criteria

- [ ] `load_model()` berhasil load SAM ViT-H ke CUDA/CPU tanpa OOM
- [ ] `predict_from_point()` mengembalikan polygon valid (min 3 titik) untuk klik pada area daun
- [ ] `predict_from_box()` mengembalikan polygon yang mencakup area dalam box
- [ ] `predict_automatic()` mengembalikan list polygon, semua dengan `confidence >= threshold`
- [ ] Polygon hasil SAM memiliki `source = "sam_point"` / `"sam_auto"` yang benar
- [ ] Polygon dengan area < `min_polygon_area_px` difilter dari output automatic
- [ ] `AutoLabelWorker` berjalan di QThread — UI tidak freeze selama proses
- [ ] `AutoLabelWorker.stop()` menghentikan batch dengan graceful (tidak crash)
- [ ] `progress_callback` dipanggil tiap gambar selesai di batch mode
- [ ] Fallback ke `"cpu"` otomatis jika CUDA tidak tersedia
- [ ] `predict_automatic()` mengembalikan polygon dengan `occlusion_level` sudah terisi (tidak `None`)
- [ ] `predict_from_point()` / `predict_from_box()` menerima `existing_polygons` dan memanggil `compute_occlusion_levels` untuk seluruh set (existing + baru)
- [ ] `compute_occlusion_levels()` dengan 1 polygon selalu mengembalikan `OcclusionLevel.RENDAH`
- [ ] `compute_occlusion_levels()` menghasilkan `occlusion_ratio` yang benar: polygon kecil yang sepenuhnya di bawah polygon besar → TINGGI; polygon besar yang hanya sedikit tertutup → RENDAH

## Dependency

```
segment-anything>=1.0
torch>=2.0.0
torchvision>=0.15.0
```

## Out of Scope

- Fine-tuning SAM pada dataset daun (pakai pretrained SAM)
- SAM 2 / SAM-HQ (bisa ditambah di iterasi berikutnya)
- Integrasi CVAT / LabelMe eksternal API

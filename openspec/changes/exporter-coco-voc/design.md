## Context

`src/exporter.py` adalah jembatan antara Phase 1 (auto-labeling dengan SAM) dan Phase 2 (training Mask-RCNN + ASPP via Spec 08). Saat ini anotasi disimpan sebagai objek `Polygon` di dalam `ImageManager`, namun tidak ada mekanisme untuk mengkonversinya ke format yang dapat dikonsumsi oleh framework training standar.

Output utama adalah COCO JSON dengan RLE mask — format yang kompatibel dengan `pycocotools.coco.COCO` dan `torchvision.datasets.CocoDetection`. Pascal VOC XML disertakan sebagai format sekunder untuk kebutuhan tooling yang belum mendukung COCO.

Constraint utama: `pycocotools` harus menjadi dependensi Phase 1 (bukan hanya Phase 2), karena RLE encoding dibutuhkan di sini.

## Goals / Non-Goals

**Goals:**
- Export COCO JSON dengan segmentation RLE (primary) dan polygon fallback
- Export Pascal VOC XML per-gambar
- Dataset split ke `train.json` / `val.json` / `test.json` tanpa overlap
- Menyertakan metadata: `source`, `confidence`, `occlusion_level`, `occlusion_ratio`
- Gambar belum dianotasi dilewati dengan aman (tanpa error)

**Non-Goals:**
- Export ke YOLO TXT format
- Upload ke platform eksternal (CVAT, Roboflow)
- Augmentasi data (Spec 08)
- Evaluasi metrik (mAP, IoU) — domain Spec 08

## Decisions

### 1. RLE via `pycocotools.mask.encode()` + `cv2.fillPoly()`

**Keputusan**: Konversi polygon → binary mask via `cv2.fillPoly()`, lalu encode ke RLE via `pycocotools.mask.encode(np.asfortranarray(mask))`.

**Rationale**: Ini adalah pendekatan standar yang digunakan komunitas COCO dan kompatibel langsung dengan `pycocotools`. `cv2.fillPoly()` sudah menjadi dependensi project (OpenCV), sehingga tidak menambah dependensi baru selain `pycocotools`.

**Alternatif dipertimbangkan**: Shapely rasterization — lebih lambat dan menambah dependensi baru tanpa keuntungan akurasi.

### 2. Simpan `segmentation_polygon` sebagai fallback

**Keputusan**: Setiap annotation COCO menyertakan field `segmentation_polygon` (flat list `[x1,y1,x2,y2,...]`) di samping RLE.

**Rationale**: RLE tidak reversible ke koordinat polygon secara eksak. Menyimpan polygon asli memungkinkan downstream tools memilih format yang diinginkan tanpa re-rasterisasi.

**Alternatif dipertimbangkan**: Hanya simpan RLE — ditolak karena menghilangkan informasi polygon yang dibutuhkan untuk debugging dan visualisasi.

### 3. Area via Shoelace formula (bukan bbox area)

**Keputusan**: Area dihitung dari koordinat polygon menggunakan Shoelace formula.

**Rationale**: Bbox area memberikan over-estimasi besar untuk polygon yang tidak rectangular (daun). COCO standard merekomendasikan area segmentation mask; Shoelace adalah aproksimasi yang baik tanpa biaya rasterisasi.

### 4. Global annotation ID auto-increment

**Keputusan**: ID annotation di-increment secara global di seluruh dataset (bukan per-gambar).

**Rationale**: Keharusan standar COCO — annotation ID harus unik secara global di seluruh file JSON. Implementasi melalui counter yang dipass antar-gambar saat iterasi.

### 5. `Exporter` stateless, menerima `ImageManager` sebagai input

**Keputusan**: `Exporter` tidak menyimpan state antar-call. Setiap method menerima `ImageManager` sebagai argumen.

**Rationale**: Memudahkan testing (tidak perlu setup state) dan memungkinkan penggunaan di berbagai konteks (GUI callback, CLI, script). `Config` digunakan hanya untuk path default dan metadata.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| `pycocotools` sulit diinstall di Windows (native C extension) | Dokumentasikan di README; pertimbangkan `pycocotools-windows` sebagai alternatif |
| RLE `counts` harus di-decode dari bytes ke string untuk JSON serialization | Selalu panggil `rle["counts"].decode("utf-8")` setelah encode |
| Gambar dengan polygon kosong (belum dianotasi) menyebabkan error | Guard kondisi: lewati gambar jika `len(polygons) == 0` |
| Split ratio tidak dikonfigurasi — jika `ImageEntry.split` tidak diset | Fallback: lewati gambar tanpa split assignment, atau assign default "train" |
| Format RLE `size` adalah `[height, width]` (bukan `[width, height]`) | Ikuti konvensi COCO: `size: [height, width]` |

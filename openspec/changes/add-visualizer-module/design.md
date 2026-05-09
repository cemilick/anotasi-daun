## Context

Project ini adalah tool anotasi daun (`labeling-daun-itoh`) yang menggunakan PyQt5 sebagai UI dan OpenCV untuk image processing. Anotasi disimpan sebagai polygon per-instance dengan warna unik tiap daun. Saat ini belum ada cara untuk merender hasil anotasi sebagai gambar overlay yang bisa dilihat pengguna atau disimpan ke disk.

Modul `Visualizer` akan menjadi modul rendering murni — ia membaca data polygon yang sudah ada (`Polygon`) dan gambar asli, lalu menghasilkan output visual. Tidak ada state yang disimpan di dalam objek `Visualizer` selain `config`.

## Goals / Non-Goals

**Goals:**
- Render overlay mask semi-transparan per-instance menggunakan `cv2.addWeighted`
- Highlight visual untuk polygon auto-labeled SAM (border kuning putus-putus + badge)
- Ekspor thumbnail `QPixmap` untuk panel samping PyQt5
- Side-by-side comparison untuk QC dataset
- Batch export semua gambar teranotasi dengan progress callback

**Non-Goals:**
- Semantic segmentation (satu warna semua instance)
- Rendering video/GIF
- Format output selain JPG/PNG
- Menyimpan state anotasi (itu tugas `AnnotationEngine`)

## Decisions

### 1. Rendering dengan `cv2.addWeighted` (bukan per-pixel alpha blending manual)

`cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)` adalah cara standar OpenCV untuk blending dua frame. Alternatif seperti manual numpy blending (`image * (1-a) + mask * a`) memberikan kontrol lebih, tetapi tidak lebih cepat dan lebih verbose.

**Dipilih**: `cv2.addWeighted` karena sudah dioptimasi di level C dan hasilnya identik.

### 2. Layer overlay dibuat dari copy gambar asli, bukan canvas kosong

Semua polygon di-fill ke satu layer `overlay` (copy gambar), lalu di-blend sekali. Alternatif: blend satu polygon per iterasi. Cara ini lebih efisien karena `addWeighted` dipanggil satu kali, bukan N kali.

**Dipilih**: satu overlay layer, blend sekali.

### 3. Border putus-putus diimplementasi manual (bukan `cv2.polylines` dengan flag)

OpenCV tidak mendukung dashed polylines secara native. Implementasi: iterasi segment antar titik polygon, gambar segmen setiap 8px, skip 8px berikutnya.

**Dipilih**: implementasi manual segment-by-segment karena tidak ada API OpenCV yang lebih simpel.

### 4. `render_thumbnail` mengembalikan `QPixmap` (bukan `np.ndarray`)

Panel UI PyQt5 membutuhkan `QPixmap`. Konversi: BGR → RGB → `QImage` → `QPixmap`. Dilakukan di dalam `render_thumbnail` agar caller tidak perlu tahu detail konversi.

### 5. `export_all` memanggil `image_manager.get_annotated_images()` untuk mendapat daftar gambar

`ImageManager` sudah mengelola daftar gambar dan status anotasinya. `Visualizer` tidak menyimpan daftar gambar sendiri — prinsip single responsibility.

## Risks / Trade-offs

- **[Risk] Performa pada gambar resolusi tinggi (>20MP)**: `cv2.fillPoly` dan copy array besar bisa lambat. → Mitigation: thumbnail di-resize dulu sebelum render jika max_size kecil.
- **[Risk] QPixmap hanya bisa dibuat di main thread PyQt5**: Jika `render_thumbnail` dipanggil dari background thread, akan crash. → Mitigation: dokumentasikan bahwa `render_thumbnail` harus dipanggil dari main thread; `export_all` hanya menggunakan numpy/OpenCV (aman di thread).
- **[Trade-off] Satu opacity global dari config**: Semua mask menggunakan `config.canvas.mask_opacity`. Tidak ada per-polygon opacity. Ini cukup untuk kebutuhan saat ini dan menghindari kompleksitas UI tambahan.

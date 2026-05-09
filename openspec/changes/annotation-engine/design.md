## Context

Proyek labeling daun sudah memiliki `ImageManager` (spec-02) sebagai state management dataset. Spec ini menambahkan canvas interaktif berbasis PyQt5 yang menjadi antarmuka utama untuk menggambar dan mengedit anotasi polygon. Canvas harus mendukung dua alur kerja paralel: penggambaran manual klik-per-titik dan penerimaan/penolakan polygon yang di-generate SAM dari `AutoLabeler` (spec-07). Polygon yang dihasilkan keduanya akhirnya menggunakan dataclass `Polygon` yang diperluas dari spec-02.

## Goals / Non-Goals

**Goals:**
- Widget `AnnotationCanvas(QWidget)` yang merender gambar beserta polygon di atasnya dengan transform zoom/pan
- Pemisahan bersih antara polygon pending (SAM, belum dikonfirmasi) dan polygon final (tersimpan)
- Komputasi oklusi otomatis setelah setiap mutasi polygon, tanpa input user
- Undo stack berbasis snapshot yang andal sampai 30 langkah
- Koordinat polygon selalu disimpan dalam ruang pixel asli gambar (bukan widget)

**Non-Goals:**
- Inferensi SAM (spec-07) — canvas hanya emit signal dan menerima hasil inject
- Export ke format COCO (spec-05)
- Bounding box annotation (hanya polygon)
- Multi-gambar canvas atau side-by-side comparison

## Decisions

### 1. Koordinat dalam ruang pixel asli, transform hanya saat render

**Pilihan**: Semua `Polygon.points` disimpan dalam koordinat pixel gambar asli. `paintEvent` menghitung transform canvas ↔ image secara real-time berdasarkan offset dan skala zoom.

**Alternatif**: Simpan dalam koordinat widget, transformasi balik saat simpan.

**Alasan**: Koordinat asli adalah sumber kebenaran. Zoom/pan tidak pernah mengubah nilai yang tersimpan — hanya viewport yang bergerak. Ini menyederhanakan undo (snapshot koordinat selalu valid) dan mencegah akumulasi floating-point error saat zoom in/out berulang.

### 2. Dua list terpisah: `_polygons` (final) dan `_pending_polygons` (SAM)

**Pilihan**: Polygon SAM yang belum dikonfirmasi disimpan di `_pending_polygons`. `get_polygons()` hanya mengembalikan `_polygons`. `inject_polygons()` memasukkan ke `_pending_polygons`, bukan ke `_polygons`.

**Alternatif**: Satu list dengan field `is_confirmed: bool` sebagai filter.

**Alasan**: Pemisahan fisik mencegah polygon pending secara tidak sengaja masuk ke output atau disimpan ke disk. `annotation_changed` signal hanya di-emit saat `_polygons` berubah, bukan saat pending berubah. Undo untuk inject cukup clear `_pending_polygons`.

### 3. Undo stack berbasis full-snapshot polygon list

**Pilihan**: Setiap state-mutating action push salinan `(_polygons, _pending_polygons)` ke stack sebelum mutasi. `undo()` pop dan restore.

**Alternatif**: Command pattern dengan inverse operations (lebih kompleks).

**Alasan**: Dataset polygon per gambar kecil (umumnya < 50 instance), sehingga deep copy murah. Full snapshot menghindari bug inversion logic. Limit 30 langkah mencegah memory bloat.

### 4. QTimer untuk blink dan toast

**Pilihan**: Satu `QTimer` (interval 800ms) men-toggle `_blink_visible` flag untuk rendering border oranye. Toast menggunakan `QLabel` overlay yang di-hide via `QTimer.singleShot(2000, ...)`.

**Alternatif**: Thread terpisah atau `QAnimation`.

**Alasan**: QTimer terintegrasi dengan event loop Qt, thread-safe, dan tidak memerlukan dependensi tambahan. Single timer untuk semua blink cukup karena semua polygon blink serentak.

### 5. Oklusi dipanggil synchronous setelah setiap mutasi

**Pilihan**: Panggil `compute_occlusion_levels(polygons)` langsung setelah polygon ditambah/dihapus/digeser, sebelum `update()` dipanggil.

**Alternatif**: Jalankan di thread background, update async.

**Alasan**: Komputasi oklusi berbasis intersection area polygon adalah operasi geometri yang cepat (< 10ms untuk < 50 polygon). Sinkronous menjamin konsistensi: saat `paintEvent` dipanggil, nilai oklusi selalu up-to-date. Polygon `occlusion_level = None` hanya terjadi sesaat sebelum fungsi dipanggil; border oranye berkedip sebagai indikator transisi.

### 6. Drag handle titik polygon: threshold 6px

**Pilihan**: Hit-test titik polygon dalam mode SELECT menggunakan radius 6px di ruang canvas (bukan gambar). Jika klik dalam radius 6px dari titik mana pun, aktifkan drag mode untuk titik itu.

**Alternatif**: Hit-test di ruang gambar (invariant terhadap zoom).

**Alasan**: UX lebih konsisten — target klik tetap 6x6 pixel di layar terlepas dari zoom level. Threshold di ruang canvas membuat high-zoom edit lebih presisi dan low-zoom tetap usable.

### 7. Subfolder output annotations/ berdasarkan oklusi tertinggi per gambar

**Pilihan**: `save_annotation()` (dipanggil oleh `ImageManager`) menentukan subfolder (`rendah/`, `sedang/`, `tinggi/`) berdasarkan oklusi level tertinggi di antara semua polygon final gambar tersebut.

**Alternatif**: Simpan di satu folder, tambahkan field metadata di JSON.

**Alasan**: Subfolder memudahkan sampling dataset per tingkat oklusi saat training. Logika "tertinggi yang ada" menjamin tidak ada gambar dengan oklusi signifikan yang terlewat di folder rendah.

## Risks / Trade-offs

- **Lag saat banyak polygon digeser cepat** → `compute_occlusion_levels` dipanggil setiap `mouseMoveEvent` saat drag titik. Mitigasi: debounce dengan panggilan hanya saat `mouseReleaseEvent` jika profiling menunjukkan lag.
- **`_pending_polygons` tidak masuk undo snapshot** → Jika user inject, undo, lalu inject lagi, pending list reset. Ini by-design: SAM inject adalah operasi atomik. Undo hanya membatalkan inject seluruhnya.
- **Koordinat float vs int** → Polygon points disimpan sebagai `float` untuk presisi. Saat zoom tinggi dan drag, sub-pixel precision dijaga. Export ke COCO membulatkan ke integer saat diperlukan (di exporter, bukan canvas).

## Open Questions

- Apakah `compute_occlusion_levels()` dari spec-07 sudah tersedia saat spec-03 diimplementasikan, atau perlu stub/mock sementara? → Asumsikan stub yang mengembalikan `RENDAH` untuk semua polygon selama spec-07 belum selesai.

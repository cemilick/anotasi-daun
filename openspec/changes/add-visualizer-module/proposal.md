## Why

Setelah proses anotasi selesai, pengguna membutuhkan cara untuk memvisualisasikan dan memverifikasi hasil anotasi sebelum digunakan untuk training — saat ini tidak ada modul yang dapat menghasilkan overlay mask berwarna per-instance di atas foto asli. Modul ini juga mendukung quality check dengan membandingkan polygon SAM otomatis versus koreksi manual.

## What Changes

- Tambah modul baru `src/visualizer.py` dengan class `Visualizer`
- Mendukung rendering overlay mask semi-transparan per-instance (setiap polygon warna berbeda)
- Mendukung border putus-putus + badge `SAM` untuk polygon auto-labeled
- Mendukung ekspor thumbnail sebagai `QPixmap` untuk ditampilkan di panel UI PyQt5
- Mendukung side-by-side comparison: SAM raw vs setelah koreksi manual
- Mendukung batch export semua gambar teranotasi dengan progress callback

## Capabilities

### New Capabilities
- `visualizer`: Modul rendering overlay mask berwarna per-instance di atas gambar asli; mendukung highlight polygon SAM, label centroid, thumbnail QPixmap, side-by-side comparison, dan batch export.

### Modified Capabilities
<!-- Tidak ada perubahan requirements pada capability yang sudah ada -->

## Impact

- **Baru**: `src/visualizer.py` — class `Visualizer` dengan metode `render`, `save`, `render_thumbnail`, `render_comparison`, `export_all`
- **Dependensi**: `opencv-python`, `numpy`, `PyQt5` (sudah ada di project), `Config`, `Polygon`, `ImageManager` dari modul lain
- **Output**: file JPG kualitas 95 di `output/visualized/<nama>_annotated.jpg`
- **Tidak ada breaking change** — modul baru, tidak memodifikasi modul yang sudah ada

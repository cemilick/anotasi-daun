## Why

Aplikasi labeling daun membutuhkan canvas interaktif tempat pengguna dapat menggambar polygon per-instance secara manual maupun mengkonfirmasi/mengedit hasil auto-labeling dari SAM. Tanpa modul ini, tidak ada mekanisme input anotasi visual yang terhubung ke sistem penyimpanan dan auto-labeler.

## What Changes

- Tambah modul baru `src/annotation_engine.py` dengan class `AnnotationCanvas(QWidget)` sebagai widget canvas utama
- Definisi `CanvasMode` enum untuk lima mode interaksi: DRAW, SELECT, SAM_POINT, SAM_BOX, VIEW
- Definisi `OcclusionLevel` enum: RENDAH / SEDANG / TINGGI berdasarkan persentase area tertutup
- Perluasan dataclass `Polygon` (dari image-manager) dengan field oklusi: `occlusion_level`, `occlusion_ratio`, `occlusion_manual`, `is_confirmed`
- Sistem rendering polygon berbeda per status: manual solid, SAM pending border kuning putus-putus, selected border putih tebal, oklusi belum dihitung border oranye berkedip
- Komputasi oklusi otomatis dipicu setiap kali polygon ditambah, dihapus, atau digeser
- Undo stack maksimal 30 langkah
- Zoom/pan dengan scroll mouse dan middle-click drag
- Output JSON anotasi per gambar ke subfolder `annotations/rendah/`, `annotations/sedang/`, atau `annotations/tinggi/` berdasarkan level oklusi tertinggi dalam gambar

## Capabilities

### New Capabilities
- `annotation-canvas`: Widget canvas interaktif PyQt5 — menggambar polygon manual (klik per titik, double-klik tutup), memilih/mengedit titik (drag handle), menerima inject polygon SAM (border kuning putus-putus), accept/reject polygon SAM, shortcut override oklusi (1/2/3/0), label centroid badge, zoom/pan, dan undo stack

### Modified Capabilities
- `image-manager`: Perluasan `Polygon` dataclass dengan field oklusi (`occlusion_level`, `occlusion_ratio`, `occlusion_manual`, `is_confirmed`) dan enum baru `OcclusionLevel`

## Impact

- **Baru**: `src/annotation_engine.py` — modul utama canvas anotasi
- **Modifikasi**: `src/image_manager.py` — Polygon dataclass diperluas dengan field oklusi
- **Bergantung pada**: `Config` dari spec-01, `ImageEntry` dan `Polygon` dari spec-02, `compute_occlusion_levels()` dari spec-07 (auto_labeler)
- **Digunakan oleh**: spec-06 (Main UI menampilkan canvas dan menghubungkan toolbar mode)
- **Dependensi eksternal**: PyQt5 (QWidget, QPainter, pyqtSignal), tidak ada library baru

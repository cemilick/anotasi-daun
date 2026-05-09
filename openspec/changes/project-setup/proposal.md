## Why

Project labeling-daun-itoh belum memiliki struktur apapun. Kita perlu fondasi Phase 1 (auto-labeling tool): direktori standar, konfigurasi terpusat, dan entry point agar modul-modul berikutnya (anotasi, SAM auto-label, ekspor COCO) dapat dibangun di atas pondasi yang konsisten.

## What Changes

- Buat seluruh struktur direktori Phase 1: `images/`, `annotations/` (beserta subfolder `rendah/`, `sedang/`, `tinggi/`), `output/coco/`, `output/voc/`, `output/visualized/`, `sam_weights/`, `src/`
- Tambah `config.json` sebagai file konfigurasi tunggal (paths, SAM params, canvas, dataset split)
- Tambah `src/config.py` — singleton `Config` dengan akses hierarkis dan auto-fallback CUDA→CPU
- Tambah `requirements.txt` (Phase 1): Pillow, OpenCV, NumPy, PyQt5, segment-anything, torch, torchvision, **pycocotools**, pyyaml
- Tambah `main.py` sebagai entry point Phase 1 dengan argumen CLI opsional (`--images`, `--sam-checkpoint`, `--cpu`)
- Folder dan file Phase 2 (`training/`, `requirements-train.txt`, `checkpoints/`) di-scaffold tapi **tidak diimplementasikan** di spec ini

## Capabilities

### New Capabilities

- `project-config`: Singleton `Config` yang load `config.json` satu kali; akses hierarkis (dot-notation), konversi path ke `pathlib.Path`, auto-fallback device CUDA→CPU, method `force_cpu()`
- `cli-entry-point`: `main.py` yang membuat semua folder Phase 1 otomatis (termasuk `annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/`), mem-parse CLI args, dan menginisialisasi `QApplication` + `MainWindow`

### Modified Capabilities

<!-- Tidak ada — ini setup awal proyek -->

## Impact

- **Seluruh modul `src/`**: bergantung pada `Config.get()` untuk paths dan setting
- **Dependensi Phase 1**: PyQt5, OpenCV, Pillow, segment-anything, torch, torchvision, pycocotools (untuk RLE di Exporter), pyyaml
- **Dependensi Phase 2** (terpisah, `requirements-train.txt`): albumentations, tensorboard, tqdm
- **Tidak ada breaking change** — proyek baru

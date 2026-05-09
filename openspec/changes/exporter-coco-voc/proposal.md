## Why

Phase 1 (auto-labeling) membutuhkan output yang dapat dikonsumsi langsung oleh Phase 2 (training Mask-RCNN + ASPP). Saat ini belum ada modul yang mengkonversi anotasi polygon internal ke format standar COCO JSON (dengan RLE mask) dan Pascal VOC XML, sehingga pipeline antara kedua fase terputus.

## What Changes

- Tambah modul baru `src/exporter.py` dengan class `Exporter`
- Implementasi export COCO JSON dengan RLE mask via `pycocotools` (kompatibel dengan `pycocotools.coco.COCO` dan `torchvision.datasets.CocoDetection`)
- Implementasi export Pascal VOC XML per-gambar
- Implementasi dataset split otomatis ke `train.json`, `val.json`, `test.json` tanpa overlap antar-split
- Tambah field metadata anotasi: `source`, `confidence`, `occlusion_level`, `occlusion_ratio`

## Capabilities

### New Capabilities

- `coco-export`: Export seluruh anotasi ke COCO JSON format dengan RLE segmentation mask, polygon fallback, bbox, area (Shoelace), dan metadata okluasi; mendukung filter per-split
- `voc-export`: Export anotasi per-gambar ke Pascal VOC XML dengan bounding box dan polygon koordinat
- `dataset-split-export`: Export tiga file split (train/val/test) sekaligus dengan jaminan tidak ada overlap gambar antar-split

### Modified Capabilities

<!-- Tidak ada spec yang perlu dimodifikasi. -->

## Impact

- **Kode baru**: `src/exporter.py`
- **Dependensi baru**: `pycocotools` (Phase 1, bukan hanya Phase 2) — dibutuhkan untuk `mask.encode()`
- **Dependensi runtime**: `cv2.fillPoly` untuk konversi polygon → binary mask → RLE
- **Input**: `ImageManager`, `ImageEntry`, `Polygon` dari modul yang sudah ada
- **Output**: file JSON dan XML di direktori `output/coco/` dan `output/voc/`
- **Downstream**: Spec 08 (Mask-RCNN + ASPP) mengonsumsi file JSON yang dihasilkan modul ini

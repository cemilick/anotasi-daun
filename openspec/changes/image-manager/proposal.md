## Why

Aplikasi labeling daun membutuhkan modul terpusat untuk memuat gambar dari folder, menavigasi antar gambar, menyimpan/memuat anotasi per gambar, dan membagi dataset ke split train/val/test sebelum export ke format COCO. Tanpa modul ini, tidak ada state management yang konsisten antara navigasi UI, annotation engine, dan exporter.

## What Changes

- Tambah modul baru `src/image_manager.py` dengan class `ImageManager` sebagai pusat pengelolaan dataset
- Definisi dataclass `ImageEntry` untuk merepresentasikan setiap gambar beserta metadata anotasinya
- Definisi dataclass `Polygon` untuk struktur data anotasi (manual maupun SAM-generated)
- Definisi dataclass `AnnotationProgress` untuk tracking progress anotasi keseluruhan dataset
- Definisi dataclass `DatasetSplit` untuk hasil pembagian train/val/test
- Format file anotasi JSON di `annotations/<basename>.json` per gambar
- Format file split JSON di `annotations/split.json` untuk dataset split

## Capabilities

### New Capabilities
- `image-manager`: Modul pengelolaan dataset — memuat gambar dari folder, navigasi (next/prev/go_to dengan wrap-around), persistensi anotasi (save/load JSON), tracking progress anotasi, dan generate/load dataset split (70/20/10)

### Modified Capabilities
<!-- tidak ada capability yang sudah ada yang berubah requirement-nya -->

## Impact

- **Baru**: `src/image_manager.py` — modul utama
- **Baru**: Direktori `annotations/` — menyimpan file `.json` per gambar dan `split.json`
- **Bergantung pada**: `Config` dari spec-01 (ekstensi file yang didukung, rasio split dataset)
- **Digunakan oleh**: spec-03 (annotation engine membaca/menyimpan Polygon), spec-05 (exporter membaca ImageEntry dan DatasetSplit), spec-06 (UI navigasi memanggil next/prev/go_to)
- **Dependensi eksternal**: Pillow (membaca dimensi gambar), Python standard library (json, os, random)

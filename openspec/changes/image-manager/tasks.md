## 1. Dataclass Definitions

- [x] 1.1 Definisikan dataclass `ImageEntry` dengan semua field (index, filename, filepath, width, height, annotated, auto_labeled, annotation_path, split)
- [x] 1.2 Definisikan dataclass `Polygon` dengan semua field (instance_id, class_id, class_name, points, color, source, confidence)
- [x] 1.3 Definisikan dataclass `AnnotationProgress` dengan field total, annotated, manual, auto_labeled, unreviewed_auto
- [x] 1.4 Definisikan dataclass `DatasetSplit` dengan field train, val, test, seed, created_at

## 2. ImageManager — Inisialisasi dan Load Folder

- [x] 2.1 Implementasi `__init__` yang menerima `Config` dan inisialisasi state internal (image list, current index)
- [x] 2.2 Implementasi `load_folder`: scan direktori untuk ekstensi dari config, sort by filename
- [x] 2.3 Tambahkan pembacaan dimensi gambar via Pillow untuk setiap file di `load_folder`
- [x] 2.4 Cek keberadaan file anotasi saat `load_folder` untuk set `annotated` dan `auto_labeled` di setiap `ImageEntry`
- [x] 2.5 Implementasi `get_all()` yang mengembalikan salinan list `ImageEntry`

## 3. Navigasi

- [x] 3.1 Implementasi `current_image()` yang mengembalikan `ImageEntry` pada indeks aktif
- [x] 3.2 Implementasi `next()` dengan wrap-around (indeks terakhir → 0)
- [x] 3.3 Implementasi `prev()` dengan wrap-around (indeks 0 → terakhir)
- [x] 3.4 Implementasi `go_to(index)` dengan validasi indeks

## 4. Persistensi Anotasi

- [x] 4.1 Implementasi `save_annotation`: serialisasi list Polygon ke JSON, tulis ke `annotations/<basename>.json`, update `entry.annotated = True` dan `entry.auto_labeled = False`
- [x] 4.2 Implementasi `load_annotation`: baca JSON dari `annotations/<basename>.json`, deserialkan ke list Polygon; kembalikan list kosong jika file tidak ada
- [x] 4.3 Implementasi `is_annotated(entry)`: kembalikan `entry.annotated`
- [x] 4.4 Pastikan direktori `annotations/` dibuat otomatis jika belum ada saat `save_annotation`

## 5. Progress Tracking

- [x] 5.1 Implementasi `get_progress()`: hitung total, annotated, manual, auto_labeled dari state `ImageEntry`
- [x] 5.2 Implementasi logika `unreviewed_auto`: gambar dengan `auto_labeled = True` yang belum di-save ulang oleh user

## 6. Dataset Split

- [x] 6.1 Implementasi `generate_split()`: cek apakah `annotations/split.json` sudah ada; jika ada, load dan kembalikan
- [x] 6.2 Implementasi logika generate split baru: shuffle dengan seed dari config, potong sesuai rasio 70/20/10
- [x] 6.3 Serialisasi `DatasetSplit` ke `annotations/split.json` setelah generate
- [x] 6.4 Implementasi `get_split()`: load `split.json` jika ada; kembalikan `None` jika belum ada
- [x] 6.5 Update field `split` pada setiap `ImageEntry` setelah split di-generate atau di-load

## 7. Pengujian

- [x] 7.1 Tulis test untuk `load_folder`: verifikasi jumlah gambar, dimensi, dan flag `annotated` saat file JSON ada
- [x] 7.2 Tulis test navigasi: next/prev wrap-around, go_to valid index
- [x] 7.3 Tulis test save/load annotation: round-trip Polygon termasuk source sam_auto, nilai confidence
- [x] 7.4 Tulis test `get_progress`: skenario mixed dataset (manual + SAM + unreviewed)
- [x] 7.5 Tulis test `generate_split`: no overlap, rasio mendekati 70/20/10, idempotent pada pemanggilan kedua

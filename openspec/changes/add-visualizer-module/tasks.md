## 1. Scaffold & Setup

- [x] 1.1 Buat file `src/visualizer.py` dengan class `Visualizer` dan konstruktor `__init__(self, config: Config)`
- [x] 1.2 Tambahkan import yang dibutuhkan: `cv2`, `numpy`, `QPixmap`, `QImage`, `Config`, `Polygon`, `ImageManager`

## 2. Core Render

- [x] 2.1 Implementasi `render()`: baca gambar asli dengan `cv2.imread`, buat layer overlay (copy gambar), fill setiap polygon dengan `cv2.fillPoly`, blend dengan `cv2.addWeighted` menggunakan `config.canvas.mask_opacity`
- [x] 2.2 Implementasi rendering border solid untuk polygon manual: `cv2.polylines` warna sama dengan mask, tebal 2px
- [x] 2.3 Implementasi border putus-putus untuk polygon SAM: iterasi segmen antar titik, gambar 8px skip 8px, warna `[255, 220, 0]`, tebal 2px
- [x] 2.4 Implementasi label `#<id>` di centroid: hitung centroid dari points, gambar background kotak putih semi-transparan, tulis teks dengan `cv2.FONT_HERSHEY_SIMPLEX` scale 0.5
- [x] 2.5 Implementasi badge `SAM <confidence>` di pojok kiri atas bounding box polygon SAM saat `highlight_auto=True`

## 3. Save & Export

- [x] 3.1 Implementasi `save()`: panggil `render()`, simpan hasil ke `output_path` dengan `cv2.imwrite` parameter JPG kualitas 95, return `output_path`
- [x] 3.2 Implementasi `export_all()`: iterasi gambar teranotasi dari `image_manager`, panggil `save()` per gambar dengan nama output `<nama>_annotated.jpg`, panggil `progress_callback(current, total)` setelah tiap gambar (handle `None` callback)

## 4. Thumbnail & Comparison

- [x] 4.1 Implementasi `render_thumbnail()`: panggil `render()`, resize hasil ke dalam `max_size` dengan mempertahankan aspek rasio (`cv2.resize`), konversi BGR→RGB→`QImage`→`QPixmap`, return `QPixmap`
- [x] 4.2 Implementasi `render_comparison()`: render `polygons_before` dan `polygons_after` masing-masing, gabungkan side-by-side dengan `np.hstack`, return array hasil

## 5. Verifikasi Acceptance Criteria

- [x] 5.1 Test `render()` dengan beberapa polygon — pastikan foto asli terlihat di balik mask
- [x] 5.2 Test `highlight_auto=True` — verifikasi border kuning putus-putus dan badge SAM muncul
- [x] 5.3 Test `render()` dengan polygon kosong — pastikan mengembalikan gambar asli tanpa modifikasi
- [x] 5.4 Test `render_thumbnail()` — verifikasi mengembalikan `QPixmap` valid dalam batas max_size
- [x] 5.5 Test `render_comparison()` — verifikasi output lebar 2× gambar asli
- [x] 5.6 Test `export_all()` — verifikasi progress_callback dipanggil N kali dan file JPG kualitas 95 tersimpan

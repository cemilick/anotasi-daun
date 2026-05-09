## ADDED Requirements

### Requirement: Render overlay mask per-instance
Sistem SHALL menghasilkan gambar BGR numpy array dengan setiap polygon diisi warna semi-transparan dari `polygon.color` dengan opacity `config.canvas.mask_opacity` (default 0.45), di-blend di atas gambar asli menggunakan `cv2.addWeighted`.

#### Scenario: Render dengan beberapa polygon
- **WHEN** `render()` dipanggil dengan daftar polygon yang valid dan gambar asli yang ada
- **THEN** sistem mengembalikan numpy array BGR di mana setiap polygon terisi warna semi-transparan dan foto asli masih terlihat di balik mask

#### Scenario: Render gambar tanpa anotasi
- **WHEN** `render()` dipanggil dengan daftar polygon kosong
- **THEN** sistem mengembalikan gambar asli tanpa modifikasi

### Requirement: Border polygon sesuai sumber anotasi
Sistem SHALL menggambar border pada setiap polygon: polygon manual mendapat border solid tebal 2px dengan warna yang sama dengan mask; polygon SAM (saat `highlight_auto=True`) mendapat border putus-putus kuning `[255, 220, 0]` tebal 2px (gambar 8px, skip 8px).

#### Scenario: Border solid polygon manual
- **WHEN** `render()` dipanggil dengan polygon bersumber `"manual"` dan `highlight_auto=True`
- **THEN** polygon tersebut memiliki border solid berwarna sama dengan mask, bukan border kuning putus-putus

#### Scenario: Border kuning putus-putus polygon SAM
- **WHEN** `render()` dipanggil dengan polygon bersumber bukan `"manual"` dan `highlight_auto=True`
- **THEN** polygon tersebut memiliki border kuning putus-putus

### Requirement: Badge SAM dan confidence score
Sistem SHALL menampilkan label kecil `SAM <confidence>` (contoh: `SAM 0.92`) di pojok kiri atas bounding box polygon SAM ketika `highlight_auto=True`.

#### Scenario: Badge ditampilkan untuk polygon SAM
- **WHEN** `render()` dipanggil dengan `highlight_auto=True` dan polygon memiliki `source != "manual"`
- **THEN** teks `SAM <score>` muncul di pojok kiri atas polygon tersebut

#### Scenario: Badge tidak ditampilkan saat highlight_auto=False
- **WHEN** `render()` dipanggil dengan `highlight_auto=False`
- **THEN** tidak ada badge SAM yang ditampilkan pada polygon manapun

### Requirement: Label instance di centroid
Sistem SHALL menampilkan teks `#<id>` di centroid setiap polygon saat `show_labels=True`, dengan background kotak putih semi-transparan, menggunakan `cv2.FONT_HERSHEY_SIMPLEX` scale 0.5.

#### Scenario: Label ditampilkan saat show_labels aktif
- **WHEN** `render()` dipanggil dengan `show_labels=True`
- **THEN** setiap polygon menampilkan label `#<instance_id>` di titik centroid-nya

#### Scenario: Label tidak ditampilkan saat show_labels nonaktif
- **WHEN** `render()` dipanggil dengan `show_labels=False`
- **THEN** tidak ada label teks yang ditampilkan

### Requirement: Simpan visualisasi ke disk
Sistem SHALL menyimpan hasil render ke `output_path` sebagai JPG dengan kualitas 95 dan mengembalikan path file yang tersimpan.

#### Scenario: Simpan berhasil
- **WHEN** `save()` dipanggil dengan path gambar, daftar polygon, dan output_path yang valid
- **THEN** file JPG tersimpan di output_path dengan kualitas 95 dan fungsi mengembalikan output_path

### Requirement: Render thumbnail QPixmap
Sistem SHALL menghasilkan `QPixmap` thumbnail yang di-resize ke dalam batas `max_size` (default 300×300) dengan mempertahankan aspek rasio, untuk ditampilkan di panel samping UI PyQt5.

#### Scenario: Thumbnail dalam batas ukuran
- **WHEN** `render_thumbnail()` dipanggil dengan gambar berukuran lebih besar dari max_size
- **THEN** QPixmap yang dikembalikan memiliki dimensi yang tidak melebihi max_size di kedua sisi

#### Scenario: Thumbnail adalah QPixmap valid
- **WHEN** `render_thumbnail()` dipanggil dengan parameter valid
- **THEN** nilai yang dikembalikan adalah instance `QPixmap` yang tidak null dan bisa ditampilkan di widget PyQt5

### Requirement: Side-by-side comparison
Sistem SHALL menghasilkan gambar numpy array dengan lebar 2× gambar asli: sisi kiri menampilkan `polygons_before`, sisi kanan menampilkan `polygons_after`.

#### Scenario: Output lebar dua kali gambar asli
- **WHEN** `render_comparison()` dipanggil dengan dua daftar polygon
- **THEN** array yang dikembalikan memiliki lebar = 2 × lebar gambar asli dan tinggi = tinggi gambar asli

### Requirement: Batch export dengan progress callback
Sistem SHALL mengiterasi semua gambar teranotasi dari `image_manager`, merender dan menyimpan tiap gambar, serta memanggil `progress_callback(current, total)` setelah setiap gambar selesai diproses.

#### Scenario: Progress callback dipanggil setiap gambar
- **WHEN** `export_all()` dipanggil dengan `progress_callback` yang valid dan ada N gambar teranotasi
- **THEN** callback dipanggil tepat N kali dengan nilai `current` yang meningkat dari 1 hingga N

#### Scenario: Export tanpa callback tidak error
- **WHEN** `export_all()` dipanggil tanpa `progress_callback` (None)
- **THEN** proses berjalan normal tanpa exception

### Requirement: Nama file output standar
Sistem SHALL menyimpan file hasil export batch dengan nama `<nama_file_asli>_annotated.jpg` di dalam direktori `output_dir`.

#### Scenario: Nama file mengikuti konvensi
- **WHEN** `export_all()` memproses gambar bernama `daun_001.jpg`
- **THEN** file output tersimpan sebagai `<output_dir>/daun_001_annotated.jpg`

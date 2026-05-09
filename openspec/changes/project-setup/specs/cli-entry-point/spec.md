## ADDED Requirements

### Requirement: Scaffold semua folder Phase 1 otomatis
Sistem SHALL membuat semua direktori Phase 1 yang diperlukan secara otomatis saat `main.py` dijalankan, jika belum ada. Direktori yang dibuat: `images/`, `annotations/`, `annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/`, `output/coco/`, `output/voc/`, `output/visualized/`, `sam_weights/`.

#### Scenario: Folder belum ada
- **WHEN** `main.py` dijalankan di direktori proyek baru
- **THEN** sistem membuat semua folder Phase 1 (termasuk tiga subfolder `annotations/`) tanpa error

#### Scenario: Folder sudah ada
- **WHEN** `main.py` dijalankan dan semua folder sudah ada
- **THEN** sistem tidak raise error dan tidak mengubah konten folder yang sudah ada

### Requirement: Argumen CLI --images
Sistem SHALL mendukung argumen `--images <path>` yang meng-override nilai `config.paths.images` untuk sesi tersebut.

#### Scenario: Override folder gambar
- **WHEN** pengguna menjalankan `python main.py --images /custom/images`
- **THEN** aplikasi menggunakan `/custom/images` sebagai sumber gambar, bukan nilai default dari `config.json`

### Requirement: Argumen CLI --sam-checkpoint
Sistem SHALL mendukung argumen `--sam-checkpoint <path>` yang meng-override nilai `config.paths.sam_checkpoint` untuk sesi tersebut.

#### Scenario: Override path SAM weights
- **WHEN** pengguna menjalankan `python main.py --sam-checkpoint /other/sam.pth`
- **THEN** aplikasi menggunakan path tersebut saat menginisialisasi SAM

### Requirement: Argumen CLI --cpu
Sistem SHALL mendukung flag `--cpu` yang memaksa SAM berjalan di CPU terlepas dari nilai `config.sam.device`.

#### Scenario: Paksa CPU via flag
- **WHEN** pengguna menjalankan `python main.py --cpu`
- **THEN** `Config.get().sam.device` bernilai `"cpu"` dan tidak ada usaha inisialisasi CUDA

### Requirement: Inisialisasi QApplication dan MainWindow
Sistem SHALL membuat instance `QApplication` dan `MainWindow`, menampilkan window, dan memasuki event loop Qt.

#### Scenario: Startup normal
- **WHEN** `main.py` dijalankan tanpa argumen
- **THEN** aplikasi membuat `QApplication`, menginisialisasi `MainWindow`, menampilkan window, dan memasuki event loop tanpa error

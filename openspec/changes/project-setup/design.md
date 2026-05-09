## Context

Proyek baru tanpa kode existing. Pipeline dibagi dua fase independen: **Phase 1** (auto-labeling tool: GUI PyQt5 + SAM) dan **Phase 2** (training Mask-RCNN+ASPP). Keduanya berbagi `config.json` yang sama. Spec ini hanya membangun fondasi Phase 1. Phase 2 dikerjakan di waktu terpisah dan hanya mengonsumsi COCO JSON output dari Phase 1.

Constraint utama:
- SAM membutuhkan PyTorch; jika CUDA tidak tersedia, harus bisa berjalan di CPU tanpa error
- `config.json` adalah single source of truth — tidak boleh ada nilai hardcoded di modul lain
- `pycocotools` masuk `requirements.txt` (Phase 1) karena Exporter (Spec 05) butuh RLE encoding — bukan hanya untuk evaluasi training
- Folder oklusi (`annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/`) dibuat saat startup karena Annotation Engine (Spec 03) langsung menulis ke sana

## Goals / Non-Goals

**Goals:**
- Struktur direktori Phase 1 lengkap, dibuat otomatis saat `main.py` pertama kali dijalankan
- `Config` singleton yang load `config.json` satu kali, tersedia global, dengan auto-fallback CUDA→CPU
- `main.py` yang mem-parse CLI args, membuat folder, dan menginisialisasi aplikasi
- Dua file requirements yang terpisah: `requirements.txt` (Phase 1) dan `requirements-train.txt` (Phase 2 — hanya tambahan)

**Non-Goals:**
- Implementasi UI `MainWindow` (Spec 06)
- Scaffold folder Phase 2 (`training/`, `checkpoints/`) — akan dibuat saat Phase 2 dimulai
- Download otomatis SAM weights
- Hot-reload `config.json` saat aplikasi berjalan
- Multi-user / concurrent config access

## Decisions

### 1. Singleton `Config` via class variable + classmethod `get()`

**Keputusan**: Lazy initialization via `_instance = None` — dibuat saat `get()` pertama dipanggil.

**Alternatif**: module-level global, dependency injection.

**Rationale**: `classmethod` mudah diakses dari semua modul tanpa import tambahan; `Config._instance = None` cukup untuk reset saat testing. Lebih bersih daripada global, lebih sederhana daripada DI untuk single-process app.

### 2. CUDA auto-fallback di `Config.get()`

**Keputusan**: Jika `sam.device == "cuda"` tapi torch tidak tersedia atau `cuda.is_available()` False → ubah ke `"cpu"` dalam instance (tidak memodifikasi file).

**Rationale**: Fail-safe by default. Import torch dalam `try/except` agar tidak crash jika torch belum terinstall saat `Config` diload.

### 3. `pycocotools` ada di `requirements.txt` bukan `requirements-train.txt`

**Keputusan**: Pindahkan `pycocotools` ke Phase 1 requirements.

**Rationale**: Exporter (Spec 05) menggunakan `pycocotools.mask.encode()` untuk RLE — ini terjadi di Phase 1 saat user mengekspor anotasi. User Phase 1 tidak boleh terkena error import hanya karena belum install `requirements-train.txt`.

### 4. `pathlib.Path` untuk semua path di `Config`

**Keputusan**: Semua nilai `config.paths` di-wrap sebagai `pathlib.Path` relatif terhadap root proyek saat Config dimuat.

**Rationale**: Cross-platform (Windows/Linux), operator `/` untuk join path, konsisten di seluruh codebase.

### 5. Folder oklusi dibuat di `main.py`

**Keputusan**: `annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/` dibuat bersama folder lain saat startup.

**Rationale**: Annotation Engine (Spec 03) langsung menulis JSON ke subfolder ini setelah user menyimpan anotasi — harus sudah ada sebelum aksi pertama.

## Risks / Trade-offs

- **[Risk] torch tidak terinstall saat `Config` dimuat** → Mitigation: wrap `import torch` dan `cuda.is_available()` dalam `try/except`; default ke `"cpu"` jika gagal
- **[Risk] `config.json` tidak ditemukan** → Mitigation: raise `FileNotFoundError` dengan pesan yang menyertakan path absolut
- **[Risk] Konflik versi antara `requirements.txt` dan `requirements-train.txt`** → Mitigation: `requirements-train.txt` tidak mengulang torch/torchvision/pycocotools; hanya menambah albumentations, tensorboard, tqdm

## Migration Plan

Tidak diperlukan — proyek baru. Langkah setup:
1. `git clone` / buat folder proyek
2. `pip install -r requirements.txt`
3. Download SAM weights ke `sam_weights/`
4. `python main.py`

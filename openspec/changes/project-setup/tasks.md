## 1. File Konfigurasi & Requirements

- [x] 1.1 Buat `config.json` di root proyek dengan semua section: `project_name`, `classes`, `instance_colors`, `image_extensions`, `paths` (termasuk `annotations_rendah/sedang/tinggi`), `sam`, `canvas`, `dataset_split`
- [x] 1.2 Buat `requirements.txt` (Phase 1): Pillow, opencv-python, numpy, PyQt5, segment-anything, torch, torchvision, pycocotools, pyyaml — dengan pin versi minimum
- [x] 1.3 Buat `requirements-train.txt` (Phase 2): albumentations, tensorboard, tqdm — tanpa mengulang torch/pycocotools
- [x] 1.4 Buat `src/__init__.py` (kosong)

## 2. Config Singleton (`src/config.py`)

- [x] 2.1 Buat class `Config` dengan `_instance = None` sebagai class variable
- [x] 2.2 Implementasi classmethod `get()`: lazy-load `config.json` dari root proyek; kembalikan instance yang sama di pemanggilan berikutnya
- [x] 2.3 Implementasi akses hierarkis via objek namespace — semua key di `config.json` dapat diakses dengan dot-notation
- [x] 2.4 Konversi semua nilai dalam `config.paths` ke `pathlib.Path` relatif terhadap root proyek (termasuk `annotations_rendah`, `annotations_sedang`, `annotations_tinggi`)
- [x] 2.5 Implementasi CUDA auto-fallback dalam `get()`: wrap `import torch` dan `cuda.is_available()` dalam `try/except`; jika gagal atau CUDA tidak ada, set `sam.device = "cpu"` tanpa memodifikasi file
- [x] 2.6 Implementasi classmethod `force_cpu()`: set `sam.device = "cpu"` pada instance aktif

## 3. Entry Point (`main.py`)

- [x] 3.1 Buat `main.py` dengan `argparse`: flag `--images <path>`, `--sam-checkpoint <path>`, `--cpu`
- [x] 3.2 Load `Config.get()` lalu terapkan override CLI: `--images` → update `config.paths.images`, `--sam-checkpoint` → update `config.paths.sam_checkpoint`, `--cpu` → panggil `Config.force_cpu()`
- [x] 3.3 Buat semua folder Phase 1 dengan `Path.mkdir(parents=True, exist_ok=True)`: `images/`, `annotations/`, `annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/`, `output/coco/`, `output/voc/`, `output/visualized/`, `sam_weights/`
- [x] 3.4 Inisialisasi `QApplication` dan `MainWindow`, tampilkan window, jalankan `sys.exit(app.exec_())`

## 4. Verifikasi

- [x] 4.1 Jalankan `python main.py --cpu` di direktori kosong; pastikan semua 9 folder terbuat tanpa error
- [x] 4.2 Verifikasi `Config.get() is Config.get()` → `True` (singleton)
- [x] 4.3 Verifikasi `Config.get().sam.device == "cpu"` di mesin tanpa GPU
- [x] 4.4 Verifikasi `Config.get().paths.annotations_rendah` mengembalikan `pathlib.Path`
- [x] 4.5 Jalankan `pip install -r requirements.txt` di environment bersih; pastikan tidak ada konflik versi

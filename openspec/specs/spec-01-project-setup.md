# Spec 01 — Project Setup & Struktur Direktori

## Goal

Inisialisasi project anotasi + training pipeline daun kelengkeh itoh: struktur folder, dependensi Python (termasuk SAM untuk auto-labeling dan PyTorch untuk Mask-RCNN+ASPP), konfigurasi, dan entry point aplikasi.

## Background

Dataset terdiri dari foto daun kelengkeh itoh. Pipeline lengkap:
1. **Anotasi** — manual polygon + auto-labeling via SAM (Segment Anything Model)
2. **Export** — COCO JSON (instance segmentation, kompatibel Mask-RCNN)
3. **Training** — Mask-RCNN dengan ASPP module untuk meningkatkan kualitas mask

## Fase Implementasi

Proyek dibagi menjadi dua fase yang **sepenuhnya independen**:

| Fase | Cakupan | Spec | Entry Point |
|------|---------|------|-------------|
| **Phase 1 — Auto-Labeling Tool** | Anotasi manual + SAM auto-label + ekspor COCO/VOC | Spec 01–07 | `python main.py` |
| **Phase 2 — Training Pipeline** | Mask-RCNN+ASPP training + evaluasi mAP | Spec 08 | `python training/train.py` |

Phase 2 hanya membutuhkan file COCO JSON yang dihasilkan Phase 1 (`output/coco/*.json`). Tidak ada ketergantungan pada modul `src/`.

## Struktur Direktori

```
labeling-daun-itoh/
│
│  ── Phase 1: Auto-Labeling Tool ──────────────────────────────────
├── images/                      # Foto dataset (input)
├── annotations/                 # Anotasi per gambar (.json internal)
│   ├── rendah/                  # Gambar dengan oklusi dominan < 30%
│   ├── sedang/                  # Gambar dengan oklusi dominan 30–60%
│   └── tinggi/                  # Gambar dengan oklusi dominan > 60%
├── output/
│   ├── coco/
│   │   ├── annotations.json     # Full dataset COCO (input untuk Phase 2)
│   │   ├── train.json           # Split training
│   │   ├── val.json             # Split validasi
│   │   └── test.json            # Split test
│   ├── voc/                     # Pascal VOC XML per gambar
│   └── visualized/              # Gambar hasil overlay mask berwarna
├── src/
│   ├── __init__.py
│   ├── config.py                # Singleton Config loader
│   ├── image_manager.py         # Spec 02
│   ├── annotation_engine.py     # Spec 03
│   ├── auto_labeler.py          # Spec 07 (SAM-based auto-labeling)
│   ├── visualizer.py            # Spec 04
│   ├── exporter.py              # Spec 05
│   └── main_window.py           # Spec 06
├── sam_weights/                 # Folder weight SAM (gitignore)
├── config.json
├── requirements.txt             # Dependensi Phase 1
└── main.py                      # Entry point Phase 1
│
│  ── Phase 2: Training Pipeline (implementasi terpisah) ───────────
├── training/
│   ├── model.py                 # Spec 08 (Mask-RCNN + ASPP)
│   ├── dataset.py               # Spec 08 (COCO dataset loader)
│   ├── train.py                 # Spec 08 (training loop)
│   ├── evaluate.py              # Spec 08 (evaluasi mAP)
│   └── config_train.yaml        # Hyperparameter training
├── checkpoints/                 # Checkpoint model training (gitignore)
└── requirements-train.txt       # Dependensi Phase 2
```

## File: `requirements.txt` (Phase 1 — Auto-Labeling Tool)

```
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0
PyQt5>=5.15.0
segment-anything>=1.0          # SAM Meta AI
torch>=2.0.0                   # CPU/GPU untuk SAM inference
torchvision>=0.15.0
pycocotools>=2.0.6             # RLE mask encoding untuk COCO export (Spec 05)
pyyaml>=6.0
```

## File: `requirements-train.txt` (Phase 2 — Training Mask-RCNN)

```
# Semua dependensi Phase 1 sudah mencakup torch, torchvision, pycocotools
# File ini hanya menambah dependensi eksklusif training:
albumentations>=1.3.0          # Augmentasi data
tensorboard>=2.13.0            # Logging training
tqdm>=4.65.0
```

## File: `config.json`

```json
{
  "project_name": "Daun Kelengkeh Itoh",
  "classes": [
    { "id": 1, "name": "daun_kelengkeh_itoh", "supercategory": "plant" }
  ],
  "instance_colors": [
    [255, 82, 82], [82, 255, 82], [82, 82, 255],
    [255, 255, 82], [255, 82, 255], [82, 255, 255],
    [255, 165, 0],  [128, 0, 128]
  ],
  "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
  "paths": {
    "images": "images",
    "annotations": "annotations",
    "annotations_rendah": "annotations/rendah",
    "annotations_sedang": "annotations/sedang",
    "annotations_tinggi": "annotations/tinggi",
    "output_coco": "output/coco",
    "output_voc": "output/voc",
    "output_visualized": "output/visualized",
    "sam_checkpoint": "sam_weights/sam_vit_h_4b8939.pth"
  },
  "sam": {
    "model_type": "vit_h",
    "device": "cuda",
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "auto_confidence_threshold": 0.75
  },
  "canvas": {
    "default_width": 1200,
    "default_height": 800,
    "zoom_step": 0.1,
    "mask_opacity": 0.45
  },
  "dataset_split": {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
    "seed": 42
  }
}
```

## File: `src/config.py`

- Singleton class `Config` yang load `config.json` sekali
- Accessible dari semua modul via `Config.get()`
- Method `Config.get().sam.device` → auto-fallback ke `"cpu"` jika CUDA tidak tersedia

## File: `main.py`

- Entry point aplikasi anotasi
- Load `Config`, buat folder output jika belum ada
- Inisialisasi `QApplication` + `MainWindow`
- Argument CLI opsional:
  - `--images <path>` — override folder gambar
  - `--sam-checkpoint <path>` — override path weight SAM
  - `--cpu` — paksa SAM berjalan di CPU

## Acceptance Criteria

- [ ] Semua folder dibuat otomatis saat `main.py` dijalankan pertama kali (termasuk `annotations/rendah/`, `annotations/sedang/`, `annotations/tinggi/`)
- [ ] `Config.get()` mengembalikan instance singleton yang konsisten
- [ ] Auto-fallback ke `"cpu"` jika `config.sam.device = "cuda"` tapi CUDA tidak tersedia
- [ ] `python main.py --cpu` memaksa SAM berjalan di CPU tanpa error
- [ ] `pip install -r requirements.txt` berjalan tanpa konflik
- [ ] `pip install -r requirements-train.txt` berjalan tanpa konflik

## Out of Scope

- Implementasi UI (Spec 06)
- Download otomatis SAM weights (user download manual)

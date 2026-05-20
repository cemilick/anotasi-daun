# Labeling Daun Itoh — Prop-DeOccNet

Proyek anotasi dataset dan training model segmentasi instance daun kelengkeh itoh yang tumpang tindih (partially occluded). Terdiri dari dua phase:

- **Phase 1**: Auto-labeling tool (PyQt5 + MobileSAM + Roboflow) → `src/`
- **Phase 2**: Model training Prop-DeOccNet (Mask-RCNN + ASPP + Boundary Head) → `training/`

---

## Phase 1 — Auto-Labeling Tool

### Setup

```bash
pip install -r requirements.txt
python main.py
```

### Fitur

- Auto-label dengan MobileSAM (point + auto)
- Auto-label dengan Roboflow API
- Klasifikasi oklusi otomatis: rendah / sedang / tinggi
- Override oklusi manual via UI panel
- Export COCO JSON dengan stratified split 70/10/20

---

## Phase 2 — Prop-DeOccNet Training

### Prerequisites

- CUDA GPU (direkomendasikan ≥8 GB VRAM)
- Dataset sudah dianotasi dan diekspor via Phase 1

### Setup Hardware

| Hardware | Langkah install torch |
|---|---|
| NVIDIA GPU (CUDA) | `pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121` |
| AMD/Intel iGPU (DirectML, Windows) | `pip install torch==2.2.2 torchvision==0.17.2` lalu `pip install torch-directml` |
| CPU only | `pip install torch==2.2.2 torchvision==0.17.2` |

```bash
# Lanjutkan dengan:
pip install -r requirements-train.txt
```

### Export Dataset dari Phase 1

Buka Phase 1 tool → Menu Export → COCO JSON  
Output: `output/coco/train.json`, `val.json`, `test.json`

### Training

```bash
# Training penuh (GPU)
python -m training.train

# Testing lokal — CPU/AMD iGPU, 3 epoch, image 256px, ResNet-50
python -m training.train training/config_local.yaml

# Force device tertentu
python -m training.train --device cpu
python -m training.train --device directml
```

Konfigurasi lengkap di `training/config_train.yaml`:

| Parameter | Default | Keterangan |
|---|---|---|
| backbone | resnet101 | ResNet-50 atau ResNet-101 |
| aspp_rates | [6,12,18,24] | Atrous rates ASPP |
| use_boundary_head | true | Aktifkan Boundary Attention Head |
| image_size | 512 | Ukuran input (px) |
| epochs | 50 | Jumlah epoch |
| batch_size | 2 | Batch size training |
| optimizer | adam | Adam (lr=1e-4) |
| mosaic_prob | 0.3 | Probabilitas Mosaic Augmentation |

### Resume Training

```yaml
# Di config_train.yaml:
resume: checkpoints/epoch_025.pth
```

### Evaluasi

```bash
python -c "
from training.evaluate import evaluate
from training.model import PropDeOccNet
from training.dataset import DaunDataset, collate_fn, get_val_transforms
from torch.utils.data import DataLoader
import torch, yaml

cfg = yaml.safe_load(open('training/config_train.yaml'))
model = PropDeOccNet(num_classes=2)
model.load_state_dict(torch.load('checkpoints/best.pth')['model_state_dict'])
ds = DaunDataset(cfg['test_json'], cfg['images_dir'], get_val_transforms())
loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
print(evaluate(model, loader))
"
```

### Studi Ablasi (M0–M3)

```bash
python -m training.ablation
# Output: checkpoints/ablation_results.json
```

| Varian | Deskripsi |
|---|---|
| M0 | Baseline Mask-RCNN ResNet-101 |
| M1 | + ASPP [6,12,18,24] |
| M2 | + Boundary Attention Head |
| M3 | Full Prop-DeOccNet (ASPP + BAH) |

### TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Struktur Direktori

```
labeling-daun-itoh/
├── images/                    # Gambar input
├── annotations/               # Anotasi per oklusi level
│   ├── rendah/
│   ├── sedang/
│   ├── tinggi/
│   └── split.json             # Stratified split 70/10/20
├── output/coco/               # Export COCO JSON
├── src/                       # Phase 1: labeling tool
├── training/                  # Phase 2: Prop-DeOccNet
│   ├── model.py               # ASPPModule, BoundaryAttentionHead, PropDeOccNet
│   ├── dataset.py             # DaunDataset + mosaic augmentation
│   ├── train.py               # Training loop + combined loss
│   ├── evaluate.py            # mAP + BF Score evaluation
│   ├── ablation.py            # Studi ablasi M0–M3
│   └── config_train.yaml      # Hyperparameter
├── checkpoints/               # Model checkpoints (dibuat saat training)
├── runs/                      # TensorBoard logs (dibuat saat training)
├── requirements.txt           # Phase 1 dependencies
└── requirements-train.txt     # Phase 2 dependencies
```

---

## Metrik Evaluasi

| Metrik | Keterangan |
|---|---|
| **BF Score** | Boundary F1 Score — metrik primer, kualitas batas segmentasi |
| mAP @ 0.50:0.95 | COCO mAP standar |
| mAP_50 | mAP @ IoU threshold 0.50 |
| mAP_75 | mAP @ IoU threshold 0.75 |
| IoU mean | Rata-rata IoU per instance |

---

## Referensi

- Tesis: "Prop-DeOccNet: Pengembangan Arsitektur Deep Learning untuk Segmentasi Daun yang Tumpang Tindih Sebagian" — Dhea Anggita, UGM S2 Ilmu Komputer
- Spec: `openspec/specs/spec-08-maskrcnn-aspp.md`

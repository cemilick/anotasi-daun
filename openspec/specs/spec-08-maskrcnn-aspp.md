# Spec 08 — Mask-RCNN + ASPP (Model Training Pipeline)

> **Phase 2 — Implementasi Terpisah**
> Spec ini tidak bergantung pada kode `src/` dari Phase 1 (auto-labeling tool).
> Satu-satunya input dari Phase 1 adalah file COCO JSON di `output/coco/` yang dihasilkan oleh Spec 05 (Exporter).
>
> **Prasyarat sebelum mengerjakan spec ini:**
> - Phase 1 (Spec 01–07) sudah diimplementasikan
> - Dataset sudah dianotasi dan diekspor ke `output/coco/train.json`, `val.json`, `test.json`
> - Jalankan: `pip install -r requirements.txt -r requirements-train.txt`
> - Entry point Phase 2: `python training/train.py`

## Goal

Implementasi Mask-RCNN dengan modul ASPP (Atrous Spatial Pyramid Pooling) yang disisipkan pada mask head untuk meningkatkan kualitas segmentasi instance daun kelengkeh itoh, beserta training loop, evaluasi mAP, dan export model.

## Background

Mask-RCNN standar menggunakan 4-layer konvolusi sederhana di mask head. ASPP dari DeepLab menangkap konteks multi-skala dengan atrous convolution pada rate berbeda `[6, 12, 18]`, sangat efektif untuk objek daun yang memiliki variasi ukuran dan bentuk. ASPP disisipkan antara RoI Align dan convolutional mask head.

## Arsitektur

```
Input Image
    │
    ▼
ResNet-50/101 Backbone
    │
    ▼
FPN (Feature Pyramid Network) — P2, P3, P4, P5, P6
    │
    ├──► RPN (Region Proposal Network)
    │         │
    │         ▼ proposals
    │
    ▼
RoI Align (14×14)
    │
    ▼
┌─────────────────────────────────┐
│        ASPP Module              │  ← disisipkan di sini
│  ┌──────┐ ┌──────┐ ┌──────┐   │
│  │ 1×1  │ │3×3 r6│ │3×3r12│  │
│  │conv  │ │atrous│ │atrous│  │
│  └──────┘ └──────┘ └──────┘  │
│  ┌──────┐                     │
│  │3×3r18│  + Global Avg Pool  │
│  │atrous│                     │
│  └──────┘                     │
│         concat → 1×1 conv     │
└─────────────────────────────────┘
    │
    ▼
Mask Head (4× conv 3×3 + deconv + sigmoid)
    │
    ▼
Binary Mask per Instance (28×28 → resize ke bbox size)
    │
    ▼
Box Head → Class + BBox regression
```

## Struktur File Training

```
training/
├── model.py           # Mask-RCNN + ASPP architecture
├── dataset.py         # COCO dataset loader + augmentasi
├── train.py           # Training loop
├── evaluate.py        # Evaluasi mAP COCO
└── config_train.yaml  # Semua hyperparameter
```

---

## File: `training/model.py`

### Class: `ASPPModule(nn.Module)`

```python
class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Input: feature map dari RoI Align (B, C, 14, 14)
    Output: feature map (B, out_channels, 14, 14)
    """
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        atrous_rates: list[int] = [6, 12, 18],
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Branches:
        1. 1×1 conv (rate=1, context lokal)
        2. 3×3 atrous conv rate=6
        3. 3×3 atrous conv rate=12
        4. 3×3 atrous conv rate=18
        5. Global Average Pooling → upsample ke ukuran input
        Concat semua branches → 1×1 conv → BN → ReLU → output
        """
        ...
```

### Class: `MaskRCNNWithASPP(nn.Module)`

```python
class MaskRCNNWithASPP(nn.Module):
    """
    Mask-RCNN dengan ASPP module disisipkan setelah RoI Align,
    sebelum mask head, menggunakan torchvision sebagai base.
    """
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",       # "resnet50" | "resnet101"
        pretrained_backbone: bool = True,
        aspp_rates: list[int] = [6, 12, 18],
        aspp_out_channels: int = 256,
        trainable_backbone_layers: int = 3,
    ): ...

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> tuple[dict, list[dict]]:
        """
        Training: return (loss_dict, detections)
        Inference: return ({}, detections)
        """
        ...
```

#### Implementasi ASPP Hook

```python
# Cara integrasi: hook ke mask_roi_pool output sebelum mask_head
# Di __init__:
self.aspp = ASPPModule(in_channels=256, out_channels=256, atrous_rates=aspp_rates)

# Di forward:
# 1. Jalankan backbone + FPN + RPN seperti biasa
# 2. Jalankan roi_pool untuk mask features → shape (N, 256, 14, 14)
# 3. Lewatkan melalui ASPP: mask_features = self.aspp(mask_features)
# 4. Lanjutkan ke mask_head dan mask_predictor seperti biasa
```

---

## File: `training/dataset.py`

### Class: `DaunDataset(torch.utils.data.Dataset)`

```python
class DaunDataset(Dataset):
    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        transforms=None,
    ): ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """
        Return:
        - image: FloatTensor [3, H, W] normalized
        - target: {
            "boxes":   FloatTensor [N, 4] (x1,y1,x2,y2),
            "labels":  Int64Tensor [N],
            "masks":   BoolTensor  [N, H, W],  # dari RLE COCO
            "image_id": IntTensor [1],
            "area":    FloatTensor [N],
            "iscrowd": Int64Tensor [N],
          }
        """
        ...
```

### Augmentasi (via Albumentations)

```python
def get_train_transforms(image_size: int = 800):
    return A.Compose([
        A.RandomHorizontalFlip(p=0.5),
        A.RandomVerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.4),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05, p=0.5
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, value=0
        ),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
```

---

## File: `training/train.py`

### Fungsi Utama

```python
def train(config_path: str = "training/config_train.yaml") -> None:
    """
    Training loop lengkap dengan:
    - DataLoader train/val
    - Optimizer: SGD (momentum=0.9, weight_decay=0.0005)
    - LR Scheduler: MultiStepLR ([step1, step2])
    - Warmup: linear LR warmup di 500 iterasi pertama
    - TensorBoard logging: loss, mAP, lr
    - Checkpoint setiap epoch + best model (berdasarkan val mAP)
    - Resume dari checkpoint jika ada
    """
    ...
```

### Training Loop

```python
for epoch in range(start_epoch, config.epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()  # warmup phase
        # Log ke TensorBoard
    
    lr_scheduler_epoch.step()  # MultiStepLR
    val_map = evaluate(model, val_loader)
    save_checkpoint(model, optimizer, epoch, val_map)
```

---

## File: `training/evaluate.py`

```python
def evaluate(
    model: MaskRCNNWithASPP,
    data_loader: DataLoader,
    iou_type: str = "segm",    # "segm" untuk instance segmentation
    device: str = "cuda",
) -> dict:
    """
    Evaluasi menggunakan pycocotools.
    Return: {
        "mAP":      float,  # mAP @ IoU 0.50:0.95
        "mAP_50":   float,  # mAP @ IoU 0.50
        "mAP_75":   float,  # mAP @ IoU 0.75
        "mAP_small": float,
        "mAP_medium": float,
        "mAP_large": float,
        "mAR_100":  float,
    }
    """
    ...
```

---

## File: `training/config_train.yaml`

```yaml
# Data
train_json:   output/coco/train.json
val_json:     output/coco/val.json
test_json:    output/coco/test.json
images_dir:   images/
num_classes:  2                # 1 kelas + background

# Model
backbone:     resnet50         # resnet50 | resnet101
pretrained_backbone: true
aspp_rates:   [6, 12, 18]
aspp_out_channels: 256
trainable_backbone_layers: 3

# Training
epochs:       50
batch_size:   2
num_workers:  4
image_size:   800              # panjang sisi terpanjang

# Optimizer
optimizer:    sgd
lr:           0.005
momentum:     0.9
weight_decay: 0.0005

# LR Schedule
warmup_iters: 500
lr_steps:     [30, 40]         # epoch untuk turunkan LR
lr_gamma:     0.1

# Checkpoint
checkpoint_dir: checkpoints/
save_every:   5                # simpan tiap N epoch
resume:       null             # null | path ke checkpoint

# Logging
tensorboard_dir: runs/
log_every:    20               # log setiap N iterasi

# Oklusi (opsional — dari field occlusion_level di COCO JSON Phase 1)
# Jika diaktifkan, training hanya menggunakan subset berdasarkan level oklusi
# null = gunakan semua data
occlusion_filter: null         # null | "rendah" | "sedang" | "tinggi"
```

---

## Acceptance Criteria

### Model Architecture
- [ ] `ASPPModule.forward()` menghasilkan output shape `(B, 256, 14, 14)` sama dengan input
- [ ] Semua 5 branches ASPP (1×1, 3×3×3 atrous, GAP) terhubung dan di-concat
- [ ] `MaskRCNNWithASPP` dapat forward pass tanpa error (training dan inference mode)
- [ ] ASPP terintegrasi antara RoI Align dan mask head (bukan menggantikan backbone)

### Dataset
- [ ] `DaunDataset` membaca COCO JSON dan mengembalikan masks sebagai `BoolTensor [N, H, W]`
- [ ] Augmentasi diterapkan konsisten ke gambar, boxes, dan masks secara bersamaan
- [ ] Dataset dapat di-load dengan `DataLoader(num_workers=4)` tanpa deadlock

### Training
- [ ] Training loop berjalan minimal 1 epoch tanpa error
- [ ] Loss turun dari epoch 1 ke epoch 2 (sanity check)
- [ ] TensorBoard mencatat `loss_classifier`, `loss_box_reg`, `loss_mask`, `loss_objectness`, `loss_rpn_box_reg`
- [ ] Checkpoint disimpan setiap `save_every` epoch
- [ ] `resume: checkpoints/best.pth` melanjutkan training dari epoch yang tepat

### Evaluasi
- [ ] `evaluate()` mengembalikan mAP dict dengan semua key yang dispesifikasikan
- [ ] `mAP_50` ≥ 0.50 setelah training 50 epoch pada dataset yang sudah lengkap dianotasi
- [ ] Evaluasi berjalan tanpa error pada `test.json`

### Export
- [ ] Model dapat di-export ke TorchScript: `torch.jit.script(model)`
- [ ] `torch.save(model.state_dict(), "best_model.pth")` dan load kembali tanpa error

## Out of Scope

- Hyperparameter tuning otomatis
- Deployment / inference API
- Fine-tuning SAM dengan dataset ini
- Multi-GPU training (DataParallel/DDP)

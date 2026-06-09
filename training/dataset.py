from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


# ── Occlusion Utilities ────────────────────────────────────────────────────────
#
# Pendekatan: bbox overlap ratio (bukan mask overlap).
#
# Roboflow menganotasi hanya bagian *terlihat* dari setiap daun, sehingga mask
# tidak pernah tumpang tindih → mask-overlap selalu ≈ 0.
# Sebaliknya, bounding box mencakup seluruh extent daun (termasuk bagian
# tersembunyi), sehingga bbox dua daun yang saling menghalangi AKAN overlap.
#
# Rumus per-gambar:
#   occlusion_ratio = (Σ luas_bbox − luas_piksel_unik) / Σ luas_bbox
#
# Threshold default: None → auto persentil ke-33 & ke-67 dari distribusi
# training → distribusi ~seimbang antara rendah/sedang/tinggi.


def _compute_image_occlusion_bbox(anns: list[dict], img_h: int, img_w: int) -> float:
    """
    Hitung rasio oklusi gambar dari overlap bounding box.

    Rasterisasi semua bbox ke grid berukuran ≤512 px, lalu:
      ratio = (Σ luas_bbox − piksel_unik) / Σ luas_bbox

    Nilai 0 berarti tidak ada overlap bbox sama sekali.
    Nilai 1 berarti seluruh area bbox saling tumpang tindih (sangat padat).
    """
    if not anns:
        return 0.0

    scale = min(1.0, 512.0 / max(img_h, img_w, 1))
    gh = max(1, int(img_h * scale))
    gw = max(1, int(img_w * scale))

    coverage = np.zeros((gh, gw), dtype=np.int16)
    total_bbox_area = 0

    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        x1 = max(0, int(bx * scale))
        y1 = max(0, int(by * scale))
        x2 = min(gw, int((bx + bw) * scale) + 1)
        y2 = min(gh, int((by + bh) * scale) + 1)
        if x2 > x1 and y2 > y1:
            coverage[y1:y2, x1:x2] += 1
            total_bbox_area += (x2 - x1) * (y2 - y1)

    if total_bbox_area == 0:
        return 0.0

    unique_area = int(np.sum(coverage >= 1))
    overlap_area = total_bbox_area - unique_area
    return max(0.0, float(overlap_area) / total_bbox_area)


def _classify_occlusion(ratio: float, thresholds: tuple[float, float]) -> str:
    low_thresh, high_thresh = thresholds
    if ratio < low_thresh:
        return "rendah"
    elif ratio < high_thresh:
        return "sedang"
    return "tinggi"


def compute_dataset_occlusion_stats(
    coco_json_path: str,
    thresholds: tuple[float, float] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Hitung statistik oklusi per-gambar untuk dataset COCO.

    Metode: bbox overlap ratio (lihat _compute_image_occlusion_bbox).
    Tidak memerlukan decoding mask → jauh lebih cepat dari metode mask.

    Parameters
    ----------
    thresholds : (low, high) atau None
        Jika None, threshold dihitung otomatis dari persentil ke-33 dan ke-67
        distribusi rasio oklusi dataset ini → distribusi ~seimbang.

    Returns
    -------
    dict berisi:
        rendah, sedang, tinggi : int  — jumlah gambar per level
        total      : int
        thresholds : (float, float)  — threshold yang dipakai (auto atau fixed)
        per_image  : dict[image_id → {"level": str, "ratio": float}]
    """
    coco = COCO(coco_json_path)
    image_ids = sorted(coco.imgs.keys())

    if verbose:
        print(f"  Menghitung oklusi {len(image_ids)} gambar (bbox overlap) …", flush=True)

    # ── Hitung rasio per gambar ────────────────────────────────────────────────
    ratios: dict[int, float] = {}
    for image_id in image_ids:
        img_info = coco.imgs[image_id]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        ratios[image_id] = _compute_image_occlusion_bbox(
            anns, img_info["height"], img_info["width"]
        )

    # ── Tentukan threshold ─────────────────────────────────────────────────────
    vals = np.array(list(ratios.values()), dtype=np.float64)
    if thresholds is None:
        p33 = float(np.percentile(vals, 33))
        p67 = float(np.percentile(vals, 67))
        # Hindari threshold degenerate (semua nilai sama)
        if p33 >= p67:
            p33 = float(np.percentile(vals, 25))
            p67 = float(np.percentile(vals, 75))
        if p33 >= p67:
            med = float(np.median(vals))
            p33 = max(0.0, med * 0.667)
            p67 = med * 1.333
        thresholds = (p33, p67)

    # ── Klasifikasi ────────────────────────────────────────────────────────────
    counts: dict[str, int] = {"rendah": 0, "sedang": 0, "tinggi": 0}
    per_image: dict[int, dict] = {}
    for image_id, ratio in ratios.items():
        level = _classify_occlusion(ratio, thresholds)
        counts[level] += 1
        per_image[image_id] = {"level": level, "ratio": ratio}

    total = len(image_ids)
    counts["total"] = total
    counts["thresholds"] = thresholds
    counts["per_image"] = per_image

    if verbose and total > 0:
        low_t, high_t = thresholds
        print(f"  Metode       : bbox overlap ratio")
        print(f"  Threshold    : rendah < {low_t:.4f} ≤ sedang < {high_t:.4f} ≤ tinggi")
        print(f"  (auto dari persentil ke-33={low_t:.4f}, ke-67={high_t:.4f})")
        for level, label in [
            ("rendah", f"rendah (< {low_t:.4f})"),
            ("sedang", f"sedang ({low_t:.4f}–{high_t:.4f})"),
            ("tinggi", f"tinggi (≥ {high_t:.4f})"),
        ]:
            n = counts[level]
            print(f"    {label}: {n} gambar ({n / total:.1%})")
        print(f"    total: {total} gambar")
        # Mini histogram distribusi rasio
        print(f"  Distribusi rasio: "
              f"min={vals.min():.4f}  "
              f"p25={np.percentile(vals,25):.4f}  "
              f"median={np.median(vals):.4f}  "
              f"p75={np.percentile(vals,75):.4f}  "
              f"max={vals.max():.4f}")

    return counts


# ── Dataset ────────────────────────────────────────────────────────────────────

class DaunDataset(Dataset):
    """
    COCO-format dataset loader untuk instance segmentasi daun kelengkeng itoh.

    images_dir
        Path ke folder gambar. Jika ``None``, gambar diasumsikan berada di
        folder yang sama dengan file JSON (Roboflow style).

    occlusion_filter
        ``"rendah"``, ``"sedang"``, atau ``"tinggi"`` untuk memfilter gambar
        berdasarkan level oklusi. Oklusi dihitung dari bbox overlap saat init.

    occlusion_thresholds
        ``(low, high)`` untuk klasifikasi.  ``None`` → auto dari persentil ke-33
        & ke-67 distribusi dataset ini.
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str | None = None,
        transforms: Callable | None = None,
        mosaic_prob: float = 0.0,
        occlusion_filter: str | None = None,
        occlusion_thresholds: tuple[float, float] | None = None,
    ) -> None:
        self.coco = COCO(coco_json_path)
        self.images_dir = Path(images_dir) if images_dir else Path(coco_json_path).parent
        self.transforms = transforms
        self.mosaic_prob = mosaic_prob
        self._occlusion_cache: dict[int, dict] | None = None

        all_ids = sorted(self.coco.imgs.keys())

        if occlusion_filter is not None:
            valid_levels = {"rendah", "sedang", "tinggi"}
            if occlusion_filter not in valid_levels:
                raise ValueError(f"occlusion_filter harus salah satu dari {valid_levels}")
            stats = compute_dataset_occlusion_stats(
                coco_json_path, thresholds=occlusion_thresholds, verbose=False
            )
            self._occlusion_cache = stats["per_image"]
            self.occlusion_thresholds = stats["thresholds"]
            self.ids = [
                i for i in all_ids
                if self._occlusion_cache.get(i, {}).get("level") == occlusion_filter
            ]
            print(
                f"  occlusion_filter='{occlusion_filter}': "
                f"{len(self.ids)}/{len(all_ids)} gambar dipakai"
            )
        else:
            self.occlusion_thresholds = occlusion_thresholds
            self.ids = all_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            indices = [idx] + random.choices(range(len(self.ids)), k=3)
            return mosaic_collate(self, indices)
        return self._load_single(idx)

    def _load_single(self, idx: int) -> tuple[torch.Tensor, dict]:
        image_id = self.ids[idx]
        img_info = self.coco.imgs[image_id]
        img_path = self.images_dir / img_info["file_name"]

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks, boxes, labels, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            rle = self.coco.annToRLE(ann)
            mask = coco_mask.decode(rle).astype(bool)
            masks.append(mask)

            pos = np.where(mask)
            if len(pos[0]) == 0:
                continue
            x1, y1 = int(pos[1].min()), int(pos[0].min())
            x2, y2 = int(pos[1].max()), int(pos[0].max())
            boxes.append([x1, y1, x2, y2])
            labels.append(ann.get("category_id", 1))
            areas.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(masks) == 0:
            masks_arr = np.zeros((0, h, w), dtype=bool)
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            masks_arr = np.stack(masks, axis=0).astype(bool)
            boxes_arr = np.array(boxes, dtype=np.float32)

        if self.transforms is not None and len(masks_arr) > 0:
            image, boxes_arr, masks_arr = _apply_transforms(
                self.transforms, image, boxes_arr, masks_arr
            )
        elif self.transforms is not None:
            image, _, _ = _apply_transforms(
                self.transforms, image, boxes_arr, np.zeros((0, h, w), dtype=bool)
            )

        image_tensor = _to_tensor(image)

        # Oklusi dari cache (filter mode) atau hitung on-the-fly via bbox
        if self._occlusion_cache is not None and image_id in self._occlusion_cache:
            occlusion_ratio = self._occlusion_cache[image_id]["ratio"]
            occlusion_level = self._occlusion_cache[image_id]["level"]
        elif anns and self.occlusion_thresholds is not None:
            occlusion_ratio = _compute_image_occlusion_bbox(anns, h, w)
            occlusion_level = _classify_occlusion(occlusion_ratio, self.occlusion_thresholds)
        else:
            occlusion_ratio = _compute_image_occlusion_bbox(anns, h, w)
            occlusion_level = "unknown"

        target = {
            "boxes":    torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels":   torch.as_tensor(labels, dtype=torch.int64),
            "masks":    torch.as_tensor(masks_arr, dtype=torch.bool),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd":  torch.as_tensor(iscrowd, dtype=torch.int64),
            "occlusion_level": occlusion_level,
            "occlusion_ratio": torch.tensor(occlusion_ratio, dtype=torch.float32),
        }
        return image_tensor, target


def _apply_transforms(
    transforms: Callable,
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks_u8 = [masks[i].astype(np.uint8) for i in range(len(masks))]

    if len(boxes) == 0:
        result = transforms(image=image, bboxes=[], labels=[], masks=masks_u8)
        out_masks = (
            np.stack([m.astype(bool) for m in result["masks"]], axis=0)
            if result["masks"] else masks
        )
        return result["image"], boxes, out_masks

    result = transforms(
        image=image,
        bboxes=boxes.tolist(),
        labels=list(range(len(boxes))),
        masks=masks_u8,
    )
    out_image = result["image"]
    out_boxes = (
        np.array(result["bboxes"], dtype=np.float32)
        if result["bboxes"]
        else np.zeros((0, 4), dtype=np.float32)
    )
    out_masks = (
        np.stack([m.astype(bool) for m in result["masks"]], axis=0)
        if result["masks"]
        else np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
    )
    return out_image, out_boxes, out_masks


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    t = torch.from_numpy(image)
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    return t.float()


def get_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomScale(scale_limit=(-0.3, 0.3), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def mosaic_collate(
    dataset: DaunDataset,
    indices: list[int],
    image_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    half = image_size // 2
    mosaic_img = np.zeros((image_size, image_size, 3), dtype=np.float32)

    all_boxes, all_labels, all_masks = [], [], []
    all_areas, all_iscrowd = [], []
    image_id = None
    offsets = [(0, 0), (half, 0), (0, half), (half, half)]

    for i, idx in enumerate(indices[:4]):
        img_tensor, target = dataset._load_single(idx)
        img = img_tensor.permute(1, 2, 0).numpy()

        if image_id is None:
            image_id = target["image_id"]

        img_resized = cv2.resize(img, (half, half))
        x_off, y_off = offsets[i]
        mosaic_img[y_off:y_off + half, x_off:x_off + half] = img_resized

        orig_h, orig_w = img.shape[:2]
        scale_x, scale_y = half / orig_w, half / orig_h

        for box, mask, label, area, crowd in zip(
            target["boxes"].numpy(), target["masks"].numpy(),
            target["labels"].tolist(), target["area"].tolist(),
            target["iscrowd"].tolist(),
        ):
            x1c = max(0.0, min(box[0] * scale_x + x_off, image_size - 1))
            y1c = max(0.0, min(box[1] * scale_y + y_off, image_size - 1))
            x2c = max(0.0, min(box[2] * scale_x + x_off, image_size - 1))
            y2c = max(0.0, min(box[3] * scale_y + y_off, image_size - 1))

            orig_area = (box[2] - box[0]) * (box[3] - box[1])
            clipped = (x2c - x1c) * (y2c - y1c)
            if orig_area > 0 and clipped / orig_area < 0.2:
                continue
            if x2c <= x1c or y2c <= y1c:
                continue

            mask_r = cv2.resize(
                mask.astype(np.uint8), (half, half), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mos_mask = np.zeros((image_size, image_size), dtype=bool)
            mos_mask[y_off:y_off + half, x_off:x_off + half] = mask_r

            all_boxes.append([x1c, y1c, x2c, y2c])
            all_labels.append(label)
            all_masks.append(mos_mask)
            all_areas.append(float(clipped))
            all_iscrowd.append(crowd)

    mosaic_tensor = torch.from_numpy(mosaic_img.transpose(2, 0, 1)).float()

    if all_boxes:
        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        masks_t = torch.stack([torch.from_numpy(m) for m in all_masks]).bool()
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        masks_t = torch.zeros((0, image_size, image_size), dtype=torch.bool)

    return mosaic_tensor, {
        "boxes":    boxes_t,
        "labels":   torch.tensor(all_labels, dtype=torch.int64),
        "masks":    masks_t,
        "image_id": image_id if image_id is not None else torch.tensor([0]),
        "area":     torch.tensor(all_areas, dtype=torch.float32),
        "iscrowd":  torch.tensor(all_iscrowd, dtype=torch.int64),
        "occlusion_level": "mosaic",
        "occlusion_ratio": torch.tensor(0.0, dtype=torch.float32),
    }


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    return [item[0] for item in batch], [item[1] for item in batch]

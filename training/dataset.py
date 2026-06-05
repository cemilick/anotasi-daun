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

_DEFAULT_OCCLUSION_THRESHOLDS = (0.15, 0.40)  # (low_limit, high_limit)


def _compute_annotation_occlusion(mask_i: np.ndarray, all_masks: list[np.ndarray]) -> float:
    """Compute fraction of mask_i that is covered by any other mask in all_masks."""
    area_i = int(mask_i.sum())
    if area_i == 0 or len(all_masks) <= 1:
        return 0.0
    other_union = np.zeros(mask_i.shape, dtype=bool)
    for m in all_masks:
        if m is not mask_i:
            other_union |= m
    overlap = int((mask_i & other_union).sum())
    return overlap / area_i


def _classify_occlusion(ratio: float, thresholds: tuple[float, float] = _DEFAULT_OCCLUSION_THRESHOLDS) -> str:
    low_thresh, high_thresh = thresholds
    if ratio < low_thresh:
        return "rendah"
    elif ratio < high_thresh:
        return "sedang"
    return "tinggi"


def compute_dataset_occlusion_stats(
    coco_json_path: str,
    thresholds: tuple[float, float] = _DEFAULT_OCCLUSION_THRESHOLDS,
    verbose: bool = True,
) -> dict:
    """
    Compute per-image occlusion statistics for a COCO-format dataset.

    Occlusion is estimated from mask overlap — the fraction of each leaf mask that
    is covered by other leaf masks in the same image.  The image-level occlusion
    ratio is the mean of all per-annotation ratios in that image.

    Returns
    -------
    dict with keys:
        rendah  : int   – number of images classified as low-occlusion
        sedang  : int   – medium-occlusion image count
        tinggi  : int   – high-occlusion image count
        total   : int   – total images processed
        per_image : dict[image_id → {"level": str, "ratio": float}]
    Thresholds : rendah < low_thresh <= sedang < high_thresh <= tinggi
    """
    coco = COCO(coco_json_path)
    counts: dict[str, int] = {"rendah": 0, "sedang": 0, "tinggi": 0}
    per_image: dict[int, dict] = {}

    image_ids = sorted(coco.imgs.keys())
    if verbose:
        print(f"  Computing occlusion stats for {len(image_ids)} images …", flush=True)

    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            counts["rendah"] += 1
            per_image[image_id] = {"level": "rendah", "ratio": 0.0}
            continue

        # Decode all binary masks for this image
        masks: list[np.ndarray] = []
        for ann in anns:
            rle = coco.annToRLE(ann)
            masks.append(coco_mask.decode(rle).astype(bool))

        ratios = [_compute_annotation_occlusion(m, masks) for m in masks]
        img_ratio = float(np.mean(ratios))
        level = _classify_occlusion(img_ratio, thresholds)
        counts[level] += 1
        per_image[image_id] = {"level": level, "ratio": img_ratio}

    total = len(image_ids)
    counts["total"] = total

    if verbose and total > 0:
        low_t, high_t = thresholds
        print(f"  Occlusion thresholds: rendah < {low_t:.0%} ≤ sedang < {high_t:.0%} ≤ tinggi")
        for level, label in [
            ("rendah", f"rendah (< {low_t:.0%})"),
            ("sedang", f"sedang ({low_t:.0%}–{high_t:.0%})"),
            ("tinggi", f"tinggi (≥ {high_t:.0%})"),
        ]:
            n = counts[level]
            print(f"    {label}: {n} images ({n / total:.1%})")
        print(f"    total: {total} images")

    counts["per_image"] = per_image
    return counts


# ── Dataset ────────────────────────────────────────────────────────────────────

class DaunDataset(Dataset):
    """
    COCO-format dataset loader for daun kelengkeng itoh instance segmentation.

    images_dir
        Path to directory containing image files.  When ``None`` (default for
        Roboflow-style datasets), images are assumed to be in the same folder
        as the COCO JSON file.

    occlusion_filter
        When set to ``"rendah"``, ``"sedang"``, or ``"tinggi"``, only images
        whose computed occlusion level matches are included in the dataset.
        Occlusion is computed from mask overlap during ``__init__``.

    occlusion_thresholds
        Two-element tuple ``(low_thresh, high_thresh)`` used to classify the
        mean per-annotation occlusion ratio of an image.
        Default: ``(0.15, 0.40)``.
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str | None = None,
        transforms: Callable | None = None,
        mosaic_prob: float = 0.0,
        occlusion_filter: str | None = None,
        occlusion_thresholds: tuple[float, float] = _DEFAULT_OCCLUSION_THRESHOLDS,
    ) -> None:
        self.coco = COCO(coco_json_path)
        # Auto-detect images_dir from JSON parent (Roboflow style: images co-located with JSON)
        self.images_dir = Path(images_dir) if images_dir else Path(coco_json_path).parent
        self.transforms = transforms
        self.mosaic_prob = mosaic_prob
        self.occlusion_thresholds = occlusion_thresholds
        self._occlusion_cache: dict[int, dict] | None = None

        all_ids = sorted(self.coco.imgs.keys())

        if occlusion_filter is not None:
            valid_levels = {"rendah", "sedang", "tinggi"}
            if occlusion_filter not in valid_levels:
                raise ValueError(f"occlusion_filter must be one of {valid_levels}, got '{occlusion_filter}'")
            stats = compute_dataset_occlusion_stats(
                coco_json_path, thresholds=occlusion_thresholds, verbose=False
            )
            self._occlusion_cache = stats["per_image"]
            self.ids = [i for i in all_ids if self._occlusion_cache.get(i, {}).get("level") == occlusion_filter]
            print(
                f"  occlusion_filter='{occlusion_filter}': "
                f"{len(self.ids)}/{len(all_ids)} images retained"
            )
        else:
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

        masks = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            rle = self.coco.annToRLE(ann)
            mask = coco_mask.decode(rle).astype(bool)  # (H, W)
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
            image, _, _ = _apply_transforms(self.transforms, image, boxes_arr, np.zeros((0, h, w), dtype=bool))

        image_tensor = _to_tensor(image)

        # Compute image-level occlusion level (from cache if available, else on-the-fly)
        if self._occlusion_cache is not None and image_id in self._occlusion_cache:
            occlusion_level = self._occlusion_cache[image_id]["level"]
            occlusion_ratio = self._occlusion_cache[image_id]["ratio"]
        else:
            if len(masks) > 1:
                ratios = [_compute_annotation_occlusion(m, masks) for m in masks]
                occlusion_ratio = float(np.mean(ratios))
            else:
                occlusion_ratio = 0.0
            occlusion_level = _classify_occlusion(occlusion_ratio, self.occlusion_thresholds)

        target = {
            "boxes": torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks_arr, dtype=torch.bool),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
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
    """Apply albumentations transforms to image + boxes + masks together."""
    masks_u8 = [masks[i].astype(np.uint8) for i in range(len(masks))]

    if len(boxes) == 0:
        result = transforms(image=image, bboxes=[], labels=[], masks=masks_u8)
        out_masks = np.stack([m.astype(bool) for m in result["masks"]], axis=0) if result["masks"] else masks
        return result["image"], boxes, out_masks

    bbox_list = boxes.tolist()
    labels_list = list(range(len(boxes)))

    result = transforms(image=image, bboxes=bbox_list, labels=labels_list, masks=masks_u8)

    out_image = result["image"]
    out_boxes = np.array(result["bboxes"], dtype=np.float32) if result["bboxes"] else np.zeros((0, 4), dtype=np.float32)
    out_masks = (
        np.stack([m.astype(bool) for m in result["masks"]], axis=0)
        if result["masks"]
        else np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
    )
    return out_image, out_boxes, out_masks


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC numpy array to CHW float tensor. Handles already-normalized arrays."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    t = torch.from_numpy(image)
    if t.ndim == 3:
        t = t.permute(2, 0, 1)  # HWC → CHW
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
    """
    Combine 4 images into a 2×2 mosaic grid of size image_size×image_size.
    Adjusts bounding boxes and masks to mosaic coordinates.
    Filters boxes with visible area < 20% of original.
    """
    half = image_size // 2
    mosaic_img = np.zeros((image_size, image_size, 3), dtype=np.float32)

    all_boxes: list[list[float]] = []
    all_labels: list[int] = []
    all_masks: list[np.ndarray] = []
    all_areas: list[float] = []
    all_iscrowd: list[int] = []
    image_id = None

    offsets = [(0, 0), (half, 0), (0, half), (half, half)]  # (x_off, y_off)

    for i, idx in enumerate(indices[:4]):
        img_tensor, target = dataset._load_single(idx)
        img = img_tensor.permute(1, 2, 0).numpy()  # CHW → HWC

        if image_id is None:
            image_id = target["image_id"]

        img_resized = cv2.resize(img, (half, half))
        x_off, y_off = offsets[i]
        mosaic_img[y_off:y_off + half, x_off:x_off + half] = img_resized

        orig_h, orig_w = img.shape[:2]
        scale_x = half / orig_w
        scale_y = half / orig_h

        boxes = target["boxes"].numpy()
        masks = target["masks"].numpy()
        labels = target["labels"].tolist()
        areas = target["area"].tolist()
        iscrowd = target["iscrowd"].tolist()

        for j, (box, mask, label, area, crowd) in enumerate(zip(boxes, masks, labels, areas, iscrowd)):
            x1 = box[0] * scale_x + x_off
            y1 = box[1] * scale_y + y_off
            x2 = box[2] * scale_x + x_off
            y2 = box[3] * scale_y + y_off

            x1c = max(0.0, min(x1, image_size - 1))
            y1c = max(0.0, min(y1, image_size - 1))
            x2c = max(0.0, min(x2, image_size - 1))
            y2c = max(0.0, min(y2, image_size - 1))

            orig_area = (box[2] - box[0]) * (box[3] - box[1])
            clipped_area = (x2c - x1c) * (y2c - y1c)

            if orig_area > 0 and clipped_area / orig_area < 0.2:
                continue
            if x2c <= x1c or y2c <= y1c:
                continue

            mask_resized = cv2.resize(mask.astype(np.uint8), (half, half), interpolation=cv2.INTER_NEAREST).astype(bool)
            mosaic_mask = np.zeros((image_size, image_size), dtype=bool)
            mosaic_mask[y_off:y_off + half, x_off:x_off + half] = mask_resized

            all_boxes.append([x1c, y1c, x2c, y2c])
            all_labels.append(label)
            all_masks.append(mosaic_mask)
            all_areas.append(float(clipped_area))
            all_iscrowd.append(crowd)

    mosaic_tensor = torch.from_numpy(mosaic_img.transpose(2, 0, 1)).float()  # HWC → CHW

    if all_boxes:
        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        masks_t = torch.stack([torch.from_numpy(m) for m in all_masks]).bool()
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        masks_t = torch.zeros((0, image_size, image_size), dtype=torch.bool)

    target = {
        "boxes": boxes_t,
        "labels": torch.tensor(all_labels, dtype=torch.int64),
        "masks": masks_t,
        "image_id": image_id if image_id is not None else torch.tensor([0]),
        "area": torch.tensor(all_areas, dtype=torch.float32),
        "iscrowd": torch.tensor(all_iscrowd, dtype=torch.int64),
        "occlusion_level": "mosaic",
        "occlusion_ratio": torch.tensor(0.0, dtype=torch.float32),
    }
    return mosaic_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    """DataLoader collate function — keep images and targets as lists (required by Mask-RCNN)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

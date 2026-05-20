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


class DaunDataset(Dataset):
    """
    COCO-format dataset loader for daun kelengkeh itoh instance segmentation.
    Returns Z-score normalized FloatTensor images and full target dicts for Mask-RCNN.
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        transforms: Callable | None = None,
        mosaic_prob: float = 0.0,
    ) -> None:
        self.coco = COCO(coco_json_path)
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.mosaic_prob = mosaic_prob
        self.ids = sorted(self.coco.imgs.keys())

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

            # Bounding box from mask
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

        target = {
            "boxes": torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks_arr, dtype=torch.bool),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def _apply_transforms(
    transforms: Callable,
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply albumentations transforms to image + boxes + masks together."""
    # cv2 (used internally by albumentations) does not support bool dtype — convert to uint8.
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

        # Resize to half size
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
            # Scale box and translate to mosaic position
            x1 = box[0] * scale_x + x_off
            y1 = box[1] * scale_y + y_off
            x2 = box[2] * scale_x + x_off
            y2 = box[3] * scale_y + y_off

            # Clip to mosaic bounds
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

            # Resize mask to half size and place in mosaic
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
    }
    return mosaic_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    """DataLoader collate function — keep images and targets as lists (required by Mask-RCNN)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

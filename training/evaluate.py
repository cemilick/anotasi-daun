from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_bf_score(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    threshold: float = 0.02,
) -> float:
    """
    Boundary F1 Score (BF Score) — primary evaluation metric.

    For each (pred_mask, gt_mask) pair:
    1. Extract boundary pixels from pred and gt via distance transform
    2. Compute precision: fraction of pred boundary within `threshold * diag` of gt boundary
    3. Compute recall: fraction of gt boundary within threshold of pred boundary
    4. BF = 2 * P * R / (P + R + eps)

    Returns mean BF Score over all instances.
    threshold: fraction of image diagonal used as matching distance.
    """
    if not pred_masks or not gt_masks:
        return 0.0

    scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.astype(bool)
        gt = gt.astype(bool)

        diag = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
        max_dist = threshold * diag

        pred_boundary = _extract_boundary(pred)
        gt_boundary = _extract_boundary(gt)

        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            scores.append(1.0)
            continue
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            scores.append(0.0)
            continue

        # Distance from every pred boundary pixel to nearest gt boundary pixel
        gt_dist = distance_transform_edt(~gt_boundary)
        pred_dist = distance_transform_edt(~pred_boundary)

        precision = float((gt_dist[pred_boundary] <= max_dist).mean())
        recall = float((pred_dist[gt_boundary] <= max_dist).mean())

        bf = 2 * precision * recall / (precision + recall + 1e-7)
        scores.append(bf)

    return float(np.mean(scores)) if scores else 0.0


def _extract_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Boolean boundary mask: pixels at mask edge."""
    import cv2
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (dilated - eroded).astype(bool)


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = float((pred_mask & gt_mask).sum())
    union = float((pred_mask | gt_mask).sum())
    return intersection / (union + 1e-7)


def evaluate(
    model,
    data_loader: DataLoader,
    iou_type: str = "segm",
    device: str = "cuda",
) -> dict:
    """
    Evaluate PropDeOccNet using pycocotools + BF Score.

    Returns dict with: mAP, mAP_50, mAP_75, mAP_small, mAP_medium, mAP_large,
                       mAR_100, bf_score, iou_mean
    """
    was_training = model.training
    model.eval()

    coco_gt_data = _build_coco_gt(data_loader)
    coco_gt = _load_coco_from_dict(coco_gt_data)

    coco_dt_list = []
    all_pred_masks: list[np.ndarray] = []
    all_gt_masks: list[np.ndarray] = []
    iou_scores: list[float] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            _, detections = model(images)

            for det, target in zip(detections, targets):
                image_id = int(target["image_id"][0].item())
                gt_masks_np = target["masks"].numpy()  # (N, H, W)

                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy()
                masks = det["masks"].cpu().squeeze(1).numpy()  # (N, H, W)

                for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
                    mask_bin = (mask > 0.5).astype(np.uint8)

                    # RLE encode for pycocotools
                    from pycocotools import mask as coco_mask_util
                    rle = coco_mask_util.encode(np.asfortranarray(mask_bin))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    x1, y1, x2, y2 = box.tolist()
                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

                    # BF Score and IoU per instance (match to nearest GT)
                    for gt_mask in gt_masks_np:
                        all_pred_masks.append(mask_bin.astype(bool))
                        all_gt_masks.append(gt_mask.astype(bool))
                        iou_scores.append(_compute_iou(mask_bin.astype(bool), gt_mask.astype(bool)))

    # mAP via pycocotools
    map_results = _run_coco_eval(coco_gt, coco_dt_list, iou_type)

    # BF Score
    bf_score = compute_bf_score(all_pred_masks, all_gt_masks)

    # Restore training mode
    if was_training:
        model.train()

    return {
        "mAP": map_results.get("mAP", 0.0),
        "mAP_50": map_results.get("mAP_50", 0.0),
        "mAP_75": map_results.get("mAP_75", 0.0),
        "mAP_small": map_results.get("mAP_small", 0.0),
        "mAP_medium": map_results.get("mAP_medium", 0.0),
        "mAP_large": map_results.get("mAP_large", 0.0),
        "mAR_100": map_results.get("mAR_100", 0.0),
        "bf_score": bf_score,
        "iou_mean": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }


def _build_coco_gt(data_loader: DataLoader) -> dict:
    """Build COCO-format ground truth dict from DataLoader."""
    images = []
    annotations = []
    ann_id = 1

    from pycocotools import mask as coco_mask_util

    for _, targets in data_loader:
        for target in targets:
            image_id = int(target["image_id"][0].item())
            masks = target["masks"].numpy()
            labels = target["labels"].numpy()

            h, w = (masks.shape[1], masks.shape[2]) if len(masks) > 0 else (512, 512)
            images.append({"id": image_id, "height": h, "width": w, "file_name": f"{image_id}.jpg"})

            for mask, label in zip(masks, labels):
                mask_u8 = mask.astype(np.uint8)
                rle = coco_mask_util.encode(np.asfortranarray(mask_u8))
                rle["counts"] = rle["counts"].decode("utf-8")
                area = float(coco_mask_util.area(coco_mask_util.encode(np.asfortranarray(mask_u8))))
                bbox = coco_mask_util.toBbox(coco_mask_util.encode(np.asfortranarray(mask_u8))).tolist()
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "daun_kelengkeh_itoh", "supercategory": "plant"}],
    }


def _load_coco_from_dict(data: dict) -> COCO:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = f.name
    coco = COCO(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return coco


def _run_coco_eval(coco_gt: COCO, dt_list: list[dict], iou_type: str) -> dict:
    if not dt_list:
        return {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(dt_list, f)
        dt_path = f.name

    coco_dt = coco_gt.loadRes(dt_path)
    Path(dt_path).unlink(missing_ok=True)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "mAR_100": float(stats[8]),
    }

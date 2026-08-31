"""
Fungsi Kerugian (Loss Function) — Bab III Persamaan 3.4-3.7.

Dipisah dari train.py agar bisa dipakai baik oleh training/train.py (training
loop) maupun training/model.py (PropDeOccNet.forward, tempat loss ini benar-
benar dihitung & di-backward) tanpa circular import (model.py <-> train.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss per pixel. pred and target both (N,) or any flat shape."""
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.where(target == 1, torch.sigmoid(pred), 1 - torch.sigmoid(pred))
    focal_weight = alpha * ((1 - p_t) ** gamma)
    return (focal_weight * bce).mean()


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice loss for segmentation. pred is logits, target is binary."""
    pred_sig = torch.sigmoid(pred.float().view(-1))
    target_flat = target.float().view(-1)
    intersection = (pred_sig * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_sig.sum() + target_flat.sum() + smooth)


def boundary_loss(
    pred_boundary: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    BCE loss antara boundary map prediksi dan boundary yang diekstrak dari GT mask.
    Seluruh operasi dilakukan di GPU menggunakan F.max_pool2d
    (dilation ≡ MaxPool, erosion ≡ -MaxPool(-x)) — menghilangkan CPU-GPU transfer bottleneck.

    Args:
        pred_boundary: (N, 1, H, W) float probabilities dari BoundaryAttentionHead
        target_mask:   (N, H, W) binary GT mask
    """
    x = target_mask.float().unsqueeze(1)                      # (N, 1, H, W)
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded  = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    boundary_target = (dilated - eroded).clamp(0.0, 1.0)      # (N, 1, H, W)
    pred_flat   = pred_boundary.float().view(-1)
    target_flat = boundary_target.view(-1)
    return F.binary_cross_entropy(pred_flat.clamp(1e-7, 1 - 1e-7), target_flat)


def combined_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    pred_boundary: torch.Tensor | None = None,
    target_boundary: torch.Tensor | None = None,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L_total = w_focal*L_focal + w_dice*L_dice + w_boundary*L_boundary
    Returns (total_loss, component_dict).
    """
    if weights is None:
        weights = {"focal": 1.0, "dice": 1.0, "boundary": 1.0}

    l_focal = focal_loss(pred_mask, target_mask)
    l_dice = dice_loss(pred_mask, target_mask)

    l_boundary = torch.tensor(0.0, device=pred_mask.device)
    if pred_boundary is not None and target_boundary is not None:
        l_boundary = boundary_loss(pred_boundary, target_boundary)

    total = (
        weights.get("focal", 1.0) * l_focal
        + weights.get("dice", 1.0) * l_dice
        + weights.get("boundary", 1.0) * l_boundary
    )
    return total, {
        "loss_focal": l_focal.item(),
        "loss_dice": l_dice.item(),
        "loss_boundary": l_boundary.item(),
    }

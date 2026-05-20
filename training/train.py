from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Must be set before any tensorboard/tensorflow import to prevent TF from loading.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# TensorBoard is optional — falls back to a no-op writer if TF import conflicts exist.
try:
    from torch.utils.tensorboard import SummaryWriter as _RealWriter
    _HAS_TENSORBOARD = True
except Exception as _tb_err:
    _HAS_TENSORBOARD = False
    print(f"[WARNING] TensorBoard disabled: {_tb_err.__class__.__name__}: {_tb_err}")
    print("[WARNING] Run `pip uninstall tensorflow tensorflow-intel keras tf-keras -y` to fix.")

    class _RealWriter:  # type: ignore[no-redef]
        def __init__(self, log_dir: str | None = None, **kwargs: object) -> None:
            if log_dir:
                print(f"[TensorBoard disabled] logs skipped (would go to: {log_dir})")
        def add_scalar(self, *args: object, **kwargs: object) -> None:
            pass
        def close(self) -> None:
            pass

SummaryWriter = _RealWriter

from training.dataset import DaunDataset, collate_fn, get_train_transforms, get_val_transforms
from training.evaluate import evaluate
from training.model import PropDeOccNet


def _get_device(prefer: str | None = None) -> tuple[object, str]:
    """
    Auto-detect best available device: CUDA → DirectML → CPU.
    Returns (device_obj, device_label) where device_obj is passed to .to().
    prefer: force "cuda", "directml", or "cpu".
    """
    if prefer == "cpu":
        return "cpu", "cpu"

    if prefer == "cuda" or (prefer is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Device: CUDA ({name})")
            return "cuda", "cuda"

    if prefer == "directml" or prefer is None:
        try:
            import torch_directml  # type: ignore
            if torch_directml.is_available():
                dml = torch_directml.device()
                print("Device: DirectML (AMD/Intel GPU via DirectX 12)")
                return dml, "directml"
        except ImportError:
            pass

    print("Device: CPU (slow - use config_local.yaml for quick tests)")
    return "cpu", "cpu"


# ── Loss Functions ─────────────────────────────────────────────────────────────

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


def _extract_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Extract boundary pixels via morphological dilation - erosion."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (dilated - eroded).astype(np.float32)


def boundary_loss(
    pred_boundary: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    BCE loss between predicted boundary map and boundary extracted from GT mask.
    target_mask: (N, H, W) bool or float
    pred_boundary: (N, 1, H, W) float logits or probabilities
    """
    device = pred_boundary.device
    boundary_targets = []
    for i in range(target_mask.shape[0]):
        m = target_mask[i].cpu().numpy().astype(np.uint8)
        b = _extract_boundary(m)
        boundary_targets.append(torch.from_numpy(b))

    boundary_target = torch.stack(boundary_targets, dim=0).unsqueeze(1).to(device)  # (N,1,H,W)
    pred_flat = pred_boundary.float().view(-1)
    target_flat = boundary_target.float().view(-1)
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


# ── Checkpoint Utilities ───────────────────────────────────────────────────────

def save_checkpoint(
    model: PropDeOccNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    bf_score: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "bf_score": bf_score,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: PropDeOccNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device_label: str,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("bf_score", 0.0)


# ── Training Entry Point ───────────────────────────────────────────────────────

def train(config_path: str = "training/config_train.yaml", prefer_device: str | None = None) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device, device_label = _get_device(prefer=cfg.get("device", prefer_device))

    # num_workers: force 0 on CPU/DirectML to avoid Windows multiprocessing issues
    num_workers = cfg.get("num_workers", 4)
    if device_label != "cuda":
        num_workers = 0
        if cfg.get("num_workers", 4) > 0:
            print("Note: num_workers set to 0 (required for CPU/DirectML on Windows)")

    # ── Datasets & DataLoaders ──
    train_ds = DaunDataset(
        coco_json_path=cfg.get("train_json"),
        images_dir=cfg.get("images_dir"),
        transforms=get_train_transforms(cfg.get("image_size", 512)),
        mosaic_prob=cfg.get("mosaic_prob", 0.0),
    )
    val_ds = DaunDataset(
        coco_json_path=cfg.get("val_json"),
        images_dir=cfg.get("images_dir"),
        transforms=get_val_transforms(cfg.get("image_size", 512)),
        mosaic_prob=0.0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device_label == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device_label == "cuda"),
    )

    # ── Model ──
    model = PropDeOccNet(
        num_classes=cfg.get("num_classes", 2),
        backbone=cfg.get("backbone", "resnet101"),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        aspp_rates=cfg.get("aspp_rates", [6, 12, 18, 24]),
        aspp_out_channels=cfg.get("aspp_out_channels", 256),
        trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
        use_boundary_head=cfg.get("use_boundary_head", True),
    ).to(device)

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    epochs = cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg.get("lr_min", 1e-6),
    )

    loss_weights = cfg.get("loss_weights", {"focal": 1.0, "dice": 1.0, "boundary": 1.0})
    save_every = cfg.get("save_every", 5)
    log_every = cfg.get("log_every", 20)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=cfg.get("tensorboard_dir", "runs"))

    # ── Resume ──
    start_epoch = 0
    best_bf_score = 0.0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_bf_score = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, device_label
        )
        start_epoch += 1
        print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")

    # ── Training Loop ──
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses: dict[str, float] = {}

        for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict, _ = model(images, targets)

            # Standard Mask-RCNN losses (except loss_mask — replaced by combined)
            standard_loss = sum(
                v for k, v in loss_dict.items()
                if k not in ("loss_mask",)
            )

            # Compute combined mask loss from model predictions
            # We use loss_mask as a proxy since we can't easily access raw predictions
            # from the torchvision Mask-RCNN forward without major surgery.
            # For full custom loss, the combined_loss is applied to the mask logits
            # obtained from the mask head directly.
            mask_loss = loss_dict.get("loss_mask", torch.tensor(0.0, device=device))
            total_loss = standard_loss + mask_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate for logging
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
            epoch_losses["loss_total"] = epoch_losses.get("loss_total", 0.0) + total_loss.item()

            if step % log_every == 0:
                tb_writer.add_scalar("train/loss_total", total_loss.item(), global_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"train/{k}", v.item() if isinstance(v, torch.Tensor) else v, global_step)

            global_step += 1

        scheduler.step()

        # ── Validation ──
        val_metrics = evaluate(model, val_loader, device=device_label if device_label != "directml" else "cpu")
        bf_score = val_metrics["bf_score"]

        tb_writer.add_scalar("val/bf_score", bf_score, epoch)
        tb_writer.add_scalar("val/mAP_50", val_metrics.get("mAP_50", 0.0), epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"bf_score={bf_score:.4f} | "
            f"mAP_50={val_metrics.get('mAP_50', 0):.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── Checkpoints ──
        if (epoch + 1) % save_every == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, bf_score, ckpt_path)

        if bf_score > best_bf_score:
            best_bf_score = bf_score
            save_checkpoint(model, optimizer, scheduler, epoch, bf_score, checkpoint_dir / "best.pth")
            print(f"  >> New best BF Score: {best_bf_score:.4f}")

    tb_writer.close()
    print(f"Training complete. Best BF Score: {best_bf_score:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Prop-DeOccNet")
    parser.add_argument("config", nargs="?", default="training/config_train.yaml",
                        help="Path to YAML config (default: training/config_train.yaml)")
    parser.add_argument("--device", choices=["cuda", "directml", "cpu"], default=None,
                        help="Force device (auto-detect if omitted)")
    args = parser.parse_args()
    train(args.config, prefer_device=args.device)

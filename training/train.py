from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Must be set before any tensorboard/tensorflow import to prevent TF from loading.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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

from training.dataset import (
    DaunDataset,
    collate_fn,
    compute_dataset_occlusion_stats,
    get_train_transforms,
    get_val_transforms,
)
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


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the highest-numbered epoch_XXX.pth in checkpoint_dir, or None."""
    candidates = sorted(checkpoint_dir.glob("epoch_*.pth"))
    return candidates[-1] if candidates else None


def _sync_to_drive(src: Path, drive_dir: Path | None) -> None:
    """Copy a checkpoint file to Google Drive. Silent no-op if drive_dir is absent."""
    if drive_dir is None:
        return
    try:
        import shutil
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, drive_dir / src.name)
        print(f"  >> Drive sync: {src.name}")
    except Exception as exc:
        print(f"  [WARNING] Drive sync failed for {src.name}: {exc}")


# ── Training Entry Point ───────────────────────────────────────────────────────

def train(
    config_path: str = "training/config_train.yaml",
    prefer_device: str | None = None,
    drive_checkpoint_dir: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device, device_label = _get_device(prefer=cfg.get("device", prefer_device))

    # num_workers: force 0 on CPU/DirectML to avoid Windows multiprocessing issues
    num_workers = cfg.get("num_workers", 4)
    if device_label != "cuda":
        num_workers = 0
        if cfg.get("num_workers", 4) > 0:
            print("Note: num_workers set to 0 (required for CPU/DirectML on Windows)")

    # occlusion_thresholds: null di config → auto dari persentil training set
    _thresh_cfg = cfg.get("occlusion_thresholds")
    occlusion_thresholds: tuple[float, float] | None = (
        tuple(_thresh_cfg) if _thresh_cfg is not None else None
    )
    occlusion_filter = cfg.get("occlusion_filter")

    # ── Occlusion Statistics ──
    # Kalibrasi threshold dari training set, lalu pakai untuk val/test.
    _occ_stats_train: dict | None = None
    _calibrated_thresholds: tuple[float, float] | None = occlusion_thresholds

    if cfg.get("occlusion_stats", False):
        train_json = cfg.get("train_json")
        if train_json:
            print("\nOklusi [train]:")
            _occ_stats_train = compute_dataset_occlusion_stats(
                train_json, thresholds=_calibrated_thresholds, verbose=True
            )
            # Simpan threshold yang dipakai (bisa hasil auto) untuk val/test
            _calibrated_thresholds = _occ_stats_train["thresholds"]

        for split, json_key in [("val", "val_json")]:
            json_path = cfg.get(json_key)
            if json_path:
                print(f"\nOklusi [{split}]:")
                compute_dataset_occlusion_stats(
                    json_path, thresholds=_calibrated_thresholds, verbose=True
                )
        print()

    # Gunakan threshold terkalibrasi untuk DaunDataset
    if _calibrated_thresholds is not None:
        occlusion_thresholds = _calibrated_thresholds

    # ── Datasets & DataLoaders ──
    train_ds = DaunDataset(
        coco_json_path=cfg.get("train_json"),
        images_dir=cfg.get("images_dir"),
        transforms=get_train_transforms(cfg.get("image_size", 512)),
        mosaic_prob=cfg.get("mosaic_prob", 0.0),
        occlusion_filter=occlusion_filter,
        occlusion_thresholds=occlusion_thresholds,
    )
    val_ds = DaunDataset(
        coco_json_path=cfg.get("val_json"),
        images_dir=cfg.get("images_dir"),
        transforms=get_val_transforms(cfg.get("image_size", 512)),
        mosaic_prob=0.0,
        occlusion_filter=occlusion_filter,
        occlusion_thresholds=occlusion_thresholds,
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
        use_aspp=cfg.get("use_aspp", True),
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

    scaler = torch.amp.GradScaler("cuda", enabled=(device_label == "cuda"))
    loss_weights = cfg.get("loss_weights", {"focal": 1.0, "dice": 1.0, "boundary": 1.0})
    save_every = cfg.get("save_every", 5)
    log_every = cfg.get("log_every", 20)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Drive sync dir: prefer explicit arg, then config, then auto-detect Colab path.
    _drive_dir: Path | None = None
    _drive_str = drive_checkpoint_dir or cfg.get("drive_checkpoint_dir")
    if _drive_str:
        _drive_dir = Path(_drive_str)
    else:
        _colab_default = Path("/content/drive/MyDrive/labeling-daun-itoh/checkpoints")
        if _colab_default.parent.exists():
            _drive_dir = _colab_default
    if _drive_dir:
        print(f"Drive sync enabled → {_drive_dir}")

    tb_writer = SummaryWriter(log_dir=cfg.get("tensorboard_dir", "runs"))

    # Log distribusi oklusi ke TensorBoard sekali di step 0
    if _occ_stats_train is None and cfg.get("occlusion_stats", False):
        _occ_stats_train = compute_dataset_occlusion_stats(
            cfg.get("train_json"), thresholds=occlusion_thresholds, verbose=False
        )
    if _occ_stats_train is not None:
        total_imgs = max(_occ_stats_train["total"], 1)
        for level in ("rendah", "sedang", "tinggi"):
            tb_writer.add_scalar(
                f"dataset/occlusion_{level}_pct",
                _occ_stats_train[level] / total_imgs * 100,
                0,
            )
        low_t, high_t = _occ_stats_train["thresholds"]
        print(f"Threshold oklusi (dikalibrasi dari train): rendah<{low_t:.4f}, tinggi≥{high_t:.4f}")

    # ── Resume ──
    # Priority: explicit `resume` in config → auto-detect latest epoch_XXX.pth
    start_epoch = 0
    best_bf_score = 0.0

    _resume_path: Path | None = None
    _resume_cfg = cfg.get("resume")
    if _resume_cfg and Path(_resume_cfg).exists():
        _resume_path = Path(_resume_cfg)
        print(f"Resume path from config: {_resume_path}")
    else:
        _resume_path = _find_latest_checkpoint(checkpoint_dir)
        if _resume_path:
            print(f"Auto-detected checkpoint: {_resume_path}")

    if _resume_path and _resume_path.exists():
        start_epoch, _ = load_checkpoint(_resume_path, model, optimizer, scheduler, device_label)
        start_epoch += 1  # next epoch to run
        # Recover best score from best.pth (may differ from the resumed epoch).
        _best_path = checkpoint_dir / "best.pth"
        if _best_path.exists():
            _best_ckpt = torch.load(_best_path, map_location="cpu")
            best_bf_score = _best_ckpt.get("bf_score", 0.0)
        print(f"Resuming from epoch {start_epoch} | best BF Score so far: {best_bf_score:.4f}")
    else:
        print("No checkpoint found — training from scratch")

    # ── Training Loop ──
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses: dict[str, float] = {}

        for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", enabled=(device_label == "cuda")):
                    loss_dict, _ = model(images, targets)
                    standard_loss = sum(
                        v for k, v in loss_dict.items()
                        if k not in ("loss_mask",)
                    )
                    mask_loss = loss_dict.get("loss_mask", torch.tensor(0.0, device=device))
                    total_loss = standard_loss + mask_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    print(f"\n[OOM] Epoch {epoch+1} step {step} — batch skipped, cache cleared")
                    global_step += 1
                    continue
                raise

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
        _eval_device = device_label if device_label != "directml" else "cpu"
        try:
            val_metrics = evaluate(model, val_loader, device=_eval_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                val_metrics = {
                    "bf_score": 0.0, "mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
                    "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
                    "mAR_100": 0.0, "iou_mean": 0.0,
                }
                print(f"\n[OOM] Validation skipped at epoch {epoch+1} — metrics set to 0")
            else:
                raise
        bf_score = val_metrics["bf_score"]
        mAP_50   = val_metrics.get("mAP_50", 0.0)
        mAP_75   = val_metrics.get("mAP_75", 0.0)
        iou_mean = val_metrics.get("iou_mean", 0.0)
        mAR_100  = val_metrics.get("mAR_100", 0.0)

        # ── TensorBoard: metrik validasi lengkap ──
        tb_writer.add_scalar("val/bf_score",       bf_score,  epoch)
        tb_writer.add_scalar("val/mAP",            val_metrics.get("mAP", 0.0),  epoch)
        tb_writer.add_scalar("val/mAP_50",         mAP_50,    epoch)
        tb_writer.add_scalar("val/mAP_75",         mAP_75,    epoch)
        tb_writer.add_scalar("val/mAR_100",        mAR_100,   epoch)
        tb_writer.add_scalar("val/iou_mean",       iou_mean,  epoch)
        tb_writer.add_scalar("val/pixel_accuracy", val_metrics.get("pixel_accuracy", 0.0), epoch)

        # ── TensorBoard: rata-rata loss per epoch ──
        n_steps = max(len(train_loader), 1)
        for loss_key, loss_val in epoch_losses.items():
            tb_writer.add_scalar(f"train_epoch/{loss_key}", loss_val / n_steps, epoch)

        _mem_str = ""
        if device_label == "cuda":
            _alloc    = torch.cuda.memory_allocated(0) / 1e9
            _reserved = torch.cuda.memory_reserved(0) / 1e9
            _mem_str  = f" | GPU {_alloc:.1f}/{_reserved:.1f}GB"
            torch.cuda.empty_cache()

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"BF={bf_score:.4f} | mAP@50={mAP_50:.4f} | mAP@75={mAP_75:.4f} | "
            f"IoU={iou_mean:.4f} | PixAcc={val_metrics.get('pixel_accuracy', 0.0):.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            f"{_mem_str}"
        )

        # ── Checkpoints ──
        if (epoch + 1) % save_every == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            try:
                save_checkpoint(model, optimizer, scheduler, epoch, bf_score, ckpt_path)
                _sync_to_drive(ckpt_path, _drive_dir)
            except OSError as e:
                print(f"  [WARNING] Checkpoint save failed (disk full?): {e}")

        if bf_score > best_bf_score:
            best_bf_score = bf_score
            _best_save = checkpoint_dir / "best.pth"
            try:
                save_checkpoint(model, optimizer, scheduler, epoch, bf_score, _best_save)
                _sync_to_drive(_best_save, _drive_dir)
                print(f"  >> New best BF Score: {best_bf_score:.4f}")
            except OSError as e:
                print(f"  [WARNING] best.pth save failed (disk full?): {e}")

    tb_writer.close()
    print(f"Training complete. Best BF Score: {best_bf_score:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Prop-DeOccNet")
    parser.add_argument("config", nargs="?", default="training/config_train.yaml",
                        help="Path to YAML config (default: training/config_train.yaml)")
    parser.add_argument("--device", choices=["cuda", "directml", "cpu"], default=None,
                        help="Force device (auto-detect if omitted)")
    parser.add_argument("--drive-checkpoint-dir", default=None,
                        help="Google Drive path to sync checkpoints (auto-detected on Colab)")
    args = parser.parse_args()
    train(args.config, prefer_device=args.device, drive_checkpoint_dir=args.drive_checkpoint_dir)

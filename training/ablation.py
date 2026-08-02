from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from training.dataset import DaunDataset, collate_fn, get_train_transforms, get_val_transforms
from training.evaluate import evaluate
from training.model import PropDeOccNet
from training.train import load_checkpoint, save_checkpoint, train


ABLATION_CONFIGS: dict[str, dict] = {
    "M0": {
        "use_aspp": False,
        "use_boundary_head": False,
        "loss": "standard",
        "description": "Baseline Mask-RCNN ResNet-101",
    },
    "M1": {
        "use_aspp": True,
        "aspp_rates": [6, 12, 18, 24],
        "use_boundary_head": False,
        "loss": "focal+dice",
        "description": "Mask-RCNN + ASPP [6,12,18,24]",
    },
    "M2": {
        "use_aspp": False,
        "use_boundary_head": True,
        "loss": "focal+dice+boundary",
        "description": "Mask-RCNN + Boundary Attention Head",
    },
    "M3": {
        "use_aspp": True,
        "aspp_rates": [6, 12, 18, 24],
        "use_boundary_head": True,
        "description": "Prop-DeOccNet (Full)",
    },
}

BACKBONE_ABLATION_CONFIGS: dict[str, dict] = {
    "ResNet-50": {
        "backbone": "resnet50",
        "use_aspp": True,
        "aspp_rates": [6, 12, 18, 24],
        "use_boundary_head": True,
        "description": "Prop-DeOccNet (ResNet-50)",
    },
    "MobileNetV3": {
        "backbone": "mobilenet_v3_large",
        "use_aspp": True,
        "aspp_rates": [6, 12, 18, 24],
        "use_boundary_head": True,
        "description": "Prop-DeOccNet (MobileNetV3-Large)",
    },
    "ResNet-101": {
        "backbone": "resnet101",
        "use_aspp": True,
        "aspp_rates": [6, 12, 18, 24],
        "use_boundary_head": True,
        "description": "Prop-DeOccNet (ResNet-101 Default)",
    },
}


def _override_cfg(base_cfg: dict, ablation: dict) -> dict:
    """Merge base YAML config with ablation overrides."""
    cfg = copy.deepcopy(base_cfg)
    if "backbone" in ablation:
        cfg["backbone"] = ablation["backbone"]
    if "aspp_rates" in ablation:
        cfg["aspp_rates"] = ablation["aspp_rates"]
    cfg["use_boundary_head"] = ablation.get("use_boundary_head", True)
    # If use_aspp=False, set rates to empty so ASPPModule is bypassed by identity
    if not ablation.get("use_aspp", True):
        cfg["use_aspp"] = False
    else:
        cfg["use_aspp"] = True
    return cfg


def run_ablation(config_path: str = "training/config_train.yaml") -> None:
    """
    Train and evaluate all 4 ablation variants (M0–M3) sequentially.
    Saves results to ablation_results.json and checkpoints per variant.
    """
    with open(config_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    checkpoint_base = Path(base_cfg.get("checkpoint_dir", "checkpoints"))

    for variant, ablation in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation variant: {variant} — {ablation['description']}")
        print(f"{'='*60}")

        cfg = _override_cfg(base_cfg, ablation)

        # Per-variant checkpoint dir
        variant_ckpt_dir = checkpoint_base / f"ablation_{variant}"
        variant_ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg["checkpoint_dir"] = str(variant_ckpt_dir)
        cfg["tensorboard_dir"] = f"runs/ablation_{variant}"

        # ── Build model ──
        aspp_rates = cfg.get("aspp_rates", [6, 12, 18, 24]) if cfg.get("use_aspp", True) else [6]
        model = PropDeOccNet(
            num_classes=cfg.get("num_classes", 2),
            backbone=cfg.get("backbone", "resnet101"),
            pretrained_backbone=cfg.get("pretrained_backbone", True),
            aspp_rates=aspp_rates,
            aspp_out_channels=cfg.get("aspp_out_channels", 256),
            trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
            use_boundary_head=cfg.get("use_boundary_head", True),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        epochs = cfg.get("epochs", 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.get("lr_min", 1e-6)
        )

        # ── DataLoaders ──
        train_ds = DaunDataset(
            coco_json_path=cfg["train_json"],
            images_dir=cfg["images_dir"],
            transforms=get_train_transforms(cfg.get("image_size", 512)),
            mosaic_prob=cfg.get("mosaic_prob", 0.3),
        )
        test_ds = DaunDataset(
            coco_json_path=cfg["test_json"],
            images_dir=cfg["images_dir"],
            transforms=get_val_transforms(cfg.get("image_size", 512)),
            mosaic_prob=0.0,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 2),
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_fn,
        )

        # ── Training ──
        best_bf = 0.0
        for epoch in range(epochs):
            model.train()
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                loss_dict, _ = model(images, targets)
                total_loss = sum(loss_dict.values())
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        # ── Evaluate on test set ──
        test_metrics = evaluate(model, test_loader, device=device)
        save_checkpoint(model, optimizer, scheduler, epochs - 1, test_metrics["bf_score"],
                        variant_ckpt_dir / "best.pth")

        result = {
            "model": variant,
            "description": ablation["description"],
            "mAP_50": test_metrics.get("mAP_50", 0.0),
            "mAP_75": test_metrics.get("mAP_75", 0.0),
            "bf_score": test_metrics.get("bf_score", 0.0),
            "iou_mean": test_metrics.get("iou_mean", 0.0),
        }
        results.append(result)
        print(f"{variant} → BF Score: {result['bf_score']:.4f} | mAP_50: {result['mAP_50']:.4f}")

    # ── Save ablation_results.json ──
    out_path = checkpoint_base / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAblation results saved to {out_path}")

    # Summary table
    print("\n── Ablation Summary ──")
    print(f"{'Model':<5} {'BF Score':>10} {'mAP_50':>10} {'mAP_75':>10} {'IoU Mean':>10}")
    for r in results:
        print(f"{r['model']:<5} {r['bf_score']:>10.4f} {r['mAP_50']:>10.4f} {r['mAP_75']:>10.4f} {r['iou_mean']:>10.4f}")


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "training/config_train.yaml"
    run_ablation(config)

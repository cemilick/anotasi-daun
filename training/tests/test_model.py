"""Unit tests for training/model.py (tasks 2.3, 2.6, 2.9, 2.10, 2.11)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import pytest

from training.model import ASPPModule, ASPPPooling, BoundaryAttentionHead, PropDeOccNet


def make_fake_images(n: int = 2, h: int = 64, w: int = 64) -> list[torch.Tensor]:
    return [torch.rand(3, h, w) for _ in range(n)]


def make_fake_targets(n: int = 2, h: int = 64, w: int = 64, num_instances: int = 2) -> list[dict]:
    targets = []
    for _ in range(n):
        boxes = torch.tensor([[5, 5, 30, 30], [35, 35, 60, 60]], dtype=torch.float32)
        labels = torch.ones(num_instances, dtype=torch.int64)
        masks = torch.zeros(num_instances, h, w, dtype=torch.bool)
        masks[0, 5:30, 5:30] = True
        masks[1, 35:60, 35:60] = True
        targets.append({
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        })
    return targets


# --- Task 2.3: ASPPModule shape ---

def test_aspp_output_shape():
    aspp = ASPPModule(in_channels=256, out_channels=256, atrous_rates=[6, 12, 18, 24])
    x = torch.randn(2, 256, 14, 14)
    out = aspp(x)
    assert out.shape == (2, 256, 14, 14), f"Expected (2,256,14,14), got {out.shape}"


def test_aspp_preserves_small_spatial():
    aspp = ASPPModule(in_channels=128, out_channels=64, atrous_rates=[6, 12, 18, 24])
    x = torch.randn(1, 128, 7, 7)
    out = aspp(x)
    assert out.shape == (1, 64, 7, 7)


# --- Task 2.6: BoundaryAttentionHead shape and range ---

def test_boundary_head_shape():
    head = BoundaryAttentionHead(in_channels=256, hidden_channels=128)
    x = torch.randn(2, 256, 14, 14)
    out = head(x)
    assert out.shape == (2, 1, 14, 14), f"Expected (2,1,14,14), got {out.shape}"


def test_boundary_head_range():
    head = BoundaryAttentionHead(in_channels=256, hidden_channels=128)
    x = torch.randn(4, 256, 14, 14)
    out = head(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


# --- Tasks 2.9, 2.10, 2.11: PropDeOccNet ---

@pytest.fixture(scope="module")
def small_model():
    return PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                        aspp_rates=[6, 12, 18, 24], use_boundary_head=True)


def test_training_forward_returns_loss_dict(small_model):
    small_model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = small_model(images, targets)
    required_keys = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    assert required_keys.issubset(loss_dict.keys()), f"Missing keys: {required_keys - loss_dict.keys()}"


def test_inference_forward_returns_detections(small_model):
    small_model.eval()
    images = make_fake_images()
    with torch.no_grad():
        _, detections = small_model(images)
    assert len(detections) == len(images)
    for det in detections:
        assert "boxes" in det
        assert "labels" in det
        assert "scores" in det
        assert "masks" in det


def test_no_boundary_loss_when_disabled():
    model = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                         aspp_rates=[6, 12, 18, 24], use_boundary_head=False)
    model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = model(images, targets)
    assert "loss_boundary" not in loss_dict


def test_aspp_out_channels_mismatch_with_backbone_channels():
    """ASPPMaskHead.blocks must be sized from the context module's OUTPUT
    channels (aspp_out_channels), not the pre-ASPP backbone feature channels.
    Regression test: previously this only "worked" by coincidence because
    config_train.yaml always sets aspp_out_channels=256 == in_ch=256; any
    other value (e.g. during hyperparameter tuning) crashed with a Conv2d
    channel-mismatch RuntimeError."""
    model = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                         aspp_rates=[6, 12], aspp_out_channels=32, use_boundary_head=True)
    model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = model(images, targets)
    assert "loss_mask" in loss_dict


def test_use_aspp_false_replaces_aspp_with_gap_module():
    """Ablation variants M0/M2 (Tesis Bab III, Tabel 3.1) must replace ASPP with a
    single Global Average Pooling + 1x1 conv module, not shrink it to one atrous
    rate — otherwise the ablation still leaks some multi-scale context and no
    longer isolates ASPP's contribution."""
    model = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                         use_aspp=False, use_boundary_head=False)
    assert isinstance(model._model.roi_heads.mask_head.aspp, ASPPPooling)

    model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = model(images, targets)
    required_keys = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    assert required_keys.issubset(loss_dict.keys()), f"Missing keys: {required_keys - loss_dict.keys()}"


def test_combined_loss_mode_backward_works():
    """loss_mode='combined' (default) must actually run the Focal+Dice+Boundary
    path (Persamaan 3.4-3.7, Bab III) — not silently fall back — and the
    resulting loss_mask must be backprop-able end to end."""
    model = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                         aspp_rates=[6, 12], use_boundary_head=True,
                         loss_weights={"focal": 1.0, "dice": 1.0, "boundary": 1.0})
    model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = model(images, targets)
    assert not model._combined_loss_warned, "Combined loss path fell back to standard BCE"
    total_loss = sum(loss_dict.values())
    total_loss.backward()
    mask_grad = model._model.roi_heads.mask_predictor.mask_fcn_logits.weight.grad
    assert mask_grad is not None and torch.isfinite(mask_grad).all()


def test_standard_loss_mode_matches_vanilla_maskrcnn():
    """loss_mode='standard' (M0 baseline, Tabel 3.1) must bypass the custom
    combined loss entirely and use torchvision's own loss_mask untouched."""
    model = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                         use_aspp=False, use_boundary_head=False, loss_mode="standard")
    model.train()
    images = make_fake_images()
    targets = make_fake_targets()
    loss_dict, _ = model(images, targets)
    assert not model._combined_loss_warned
    required_keys = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}
    assert required_keys.issubset(loss_dict.keys())


def test_checkpoint_roundtrip(small_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pth"
        torch.save(small_model.state_dict(), path)
        loaded = PropDeOccNet(num_classes=2, backbone="resnet50", pretrained_backbone=False,
                              aspp_rates=[6, 12, 18, 24], use_boundary_head=True)
        missing, unexpected = loaded.load_state_dict(torch.load(path, map_location="cpu"))
        assert len(missing) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import MultiScaleRoIAlign, roi_align

from training.losses import boundary_loss, dice_loss, focal_loss

if TYPE_CHECKING:
    pass


def _project_masks_on_boxes(gt_masks: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """
    Crop & resize mask GT penuh-gambar (N, H, W) menjadi target seukuran
    (N, size, size) sesuai kotak (box) masing-masing instance, via RoIAlign
    pada kanal mask itu sendiri.

    Diimplementasikan ulang di sini (bukan memakai
    ``torchvision.models.detection.roi_heads.project_masks_on_boxes`` yang
    tidak didokumentasikan sebagai API publik) supaya tidak rapuh terhadap
    perubahan versi torchvision — polanya sama persis dengan yang dipakai
    torchvision secara internal.
    """
    idx = torch.arange(boxes.shape[0], device=boxes.device, dtype=boxes.dtype)
    rois = torch.cat([idx[:, None], boxes], dim=1)
    masks = gt_masks[:, None].to(rois)  # (N, 1, H, W)
    return roi_align(masks, rois, (size, size), spatial_scale=1.0)[:, 0]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # No BN after GAP: spatial size is 1×1, BN needs >1 value per channel.
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        return F.interpolate(self.gap(x), size=size, mode="bilinear", align_corners=False)


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H, W) — same spatial size as input
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        atrous_rates: list[int] = None,
    ) -> None:
        super().__init__()
        if atrous_rates is None:
            atrous_rates = [6, 12, 18, 24]

        branch_out = out_channels
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, 1, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
        )
        self.atrous_convs = nn.ModuleList(
            [ASPPConv(in_channels, branch_out, rate) for rate in atrous_rates]
        )
        self.gap = ASPPPooling(in_channels, branch_out)

        n_branches = 1 + len(atrous_rates) + 1  # 1×1 + atrous + GAP
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * branch_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.conv1x1(x)]
        for conv in self.atrous_convs:
            branches.append(conv(x))
        branches.append(self.gap(x))
        return self.project(torch.cat(branches, dim=1))


class BoundaryAttentionHead(nn.Module):
    """
    Predicts a boundary probability map from ASPP features.
    Used as a multiplicative attention gate on mask features.
    Input:  (B, in_channels, H, W)
    Output: (B, 1, H, W) with values in [0, 1]
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ASPPMaskHead(nn.Module):
    """
    Mask head that applies a multi-scale context module (ASPP, or a GAP-based
    substitute used for the "no ASPP" ablation variants) plus an optional
    boundary attention gate before the mask conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        layers: list[int],
        dilation: int,
        aspp: nn.Module,
        boundary_head: BoundaryAttentionHead | None,
    ) -> None:
        """
        Parameters
        ----------
        in_channels
            Channel count of ``aspp``'s *output* (i.e. what ``self.blocks``
            actually receives) — NOT the pre-ASPP feature channel count.
            Must equal ``aspp``'s configured ``out_channels``.
        """
        super().__init__()
        self.aspp = aspp
        self.boundary_head = boundary_head
        # Boundary map dari forward() TERAKHIR, disimpan supaya
        # PropDeOccNet._combined_mask_loss() bisa menghitung Boundary Loss
        # (Persamaan 3.6) tanpa mem-forward BoundaryAttentionHead dua kali.
        self.last_boundary_map: torch.Tensor | None = None

        # Standard mask head conv layers (same as MaskRCNNHeads)
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(nn.Conv2d(next_feature, layer_features, 3, padding=dilation, dilation=dilation))
            blocks.append(nn.ReLU(inplace=True))
            next_feature = layer_features
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aspp_out = self.aspp(x)
        if self.boundary_head is not None:
            boundary_map = self.boundary_head(aspp_out)
            self.last_boundary_map = boundary_map
            x = aspp_out * (1.0 + boundary_map)
        else:
            x = aspp_out
        return self.blocks(x)


class PropDeOccNet(nn.Module):
    """
    Prop-DeOccNet: Mask-RCNN + ResNet-101 + FPN + ASPP [6,12,18,24] + BoundaryAttentionHead.

    Training: forward(images, targets) → (loss_dict, detections)
    Inference: forward(images) → ({}, detections)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet101",
        pretrained_backbone: bool = True,
        use_aspp: bool = True,
        aspp_rates: list[int] = None,
        aspp_out_channels: int = 256,
        trainable_backbone_layers: int = 3,
        use_boundary_head: bool = True,
        loss_weights: dict | None = None,
        loss_mode: str = "combined",
    ) -> None:
        """
        loss_weights
            Bobot Combined Loss (Persamaan 3.7): ``{"focal": ..., "dice": ..., "boundary": ...}``.
            Dipakai hanya kalau ``loss_mode="combined"``.
        loss_mode
            ``"combined"`` (default): mask loss dihitung sebagai Focal + Dice
            (+ Boundary bila ``use_boundary_head=True``) — Persamaan 3.4-3.7 Bab III.
            ``"standard"``: pakai BCE loss_mask bawaan torchvision Mask R-CNN
            tanpa modifikasi (dipakai untuk varian ablasi M0 — baseline murni,
            Tabel 3.1).
        """
        super().__init__()
        if aspp_rates is None:
            aspp_rates = [6, 12, 18, 24]
        if loss_mode not in ("combined", "standard"):
            raise ValueError(f"loss_mode harus 'combined' atau 'standard', dapat: {loss_mode!r}")

        # Build FPN backbone
        weights = "DEFAULT" if pretrained_backbone else None
        if backbone in ["resnet18", "resnet34", "resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            fpn_backbone = resnet_fpn_backbone(
                backbone_name=backbone,
                weights=weights,
                trainable_layers=trainable_backbone_layers,
            )
        elif backbone in ["mobilenetv3", "mobilenet_v3_large"]:
            from torchvision.models.detection.backbone_utils import mobilenet_backbone
            fpn_backbone = mobilenet_backbone(
                backbone_name="mobilenet_v3_large",
                weights=weights,
                fpn=True,
                trainable_layers=trainable_backbone_layers,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Base Mask-RCNN without pretrained weights (we use custom backbone)
        self._model = MaskRCNN(
            backbone=fpn_backbone,
            num_classes=num_classes,
        )

        # Build ASPP and optional boundary head
        roi_out_channels = self._model.roi_heads.mask_roi_pool.output_size[0]  # 14
        in_ch = fpn_backbone.out_channels  # 256

        if use_aspp:
            context_module = ASPPModule(in_channels=in_ch, out_channels=aspp_out_channels, atrous_rates=aspp_rates)
        else:
            # Ablation variants without ASPP (M0/M2, Tesis Bab III Tabel 3.1): ASPP is
            # replaced by a single Global Average Pooling layer + 1x1 convolution,
            # not merely disabled. This keeps the feature-map dimensions identical to
            # the ASPP branch while removing all multi-scale (atrous) context, so the
            # ablation isolates ASPP's contribution rather than confounding it with a
            # loss of capacity or a leftover atrous branch.
            context_module = ASPPPooling(in_channels=in_ch, out_channels=aspp_out_channels)
        boundary_head = BoundaryAttentionHead(in_channels=aspp_out_channels) if use_boundary_head else None

        # Override mask_head with ASPP-augmented (or GAP-substituted) version.
        # in_channels here must match context_module's *output* channels
        # (aspp_out_channels), since that's what ASPPMaskHead.blocks actually
        # receives — not in_ch (the pre-ASPP feature channel count).
        mask_layers = [256, 256, 256, 256]
        self._model.roi_heads.mask_head = ASPPMaskHead(
            in_channels=aspp_out_channels,
            layers=mask_layers,
            dilation=1,
            aspp=context_module,
            boundary_head=boundary_head,
        )

        self.use_aspp = use_aspp
        self.use_boundary_head = use_boundary_head
        self.loss_weights = loss_weights or {"focal": 1.0, "dice": 1.0, "boundary": 1.0}
        self.loss_mode = loss_mode
        self._combined_loss_warned = False

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> tuple[dict, list[dict]]:
        if not self.training or targets is None or self.loss_mode == "standard":
            output = self._model(images, targets)
            if self.training:
                # MaskRCNN returns loss_dict during training
                return output, []
            return {}, output

        # ── Training, loss_mode="combined" ──
        # Mereplikasi GeneralizedRCNN.forward (transform → backbone → rpn →
        # roi_heads) secara manual supaya loss_mask bawaan torchvision (BCE
        # murni) bisa diganti dengan Focal + Dice (+ Boundary) — Persamaan
        # 3.4-3.7 Bab III — yang sebelumnya didefinisikan tapi tidak pernah
        # benar-benar dipakai saat training.
        try:
            images_t, targets_t = self._model.transform(images, targets)
            features = self._model.backbone(images_t.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            proposals, proposal_losses = self._model.rpn(images_t, features, targets_t)
            _, detector_losses = self._model.roi_heads(
                features, proposals, images_t.image_sizes, targets_t
            )

            loss_dict = {}
            loss_dict.update(proposal_losses)
            loss_dict.update(detector_losses)

            if "loss_mask" in loss_dict:
                loss_dict["loss_mask"] = self._combined_mask_loss(features, targets_t, images_t.image_sizes)

            return loss_dict, []
        except Exception as exc:
            if not self._combined_loss_warned:
                print(
                    f"[WARNING] Combined Focal+Dice+Boundary mask loss gagal dihitung "
                    f"({exc.__class__.__name__}: {exc}) — fallback ke loss_mask BCE "
                    f"bawaan torchvision untuk sisa training."
                )
                self._combined_loss_warned = True
            output = self._model(images, targets)
            return output, []

    def _combined_mask_loss(
        self,
        features: dict[str, torch.Tensor],
        targets: list[dict],
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Focal + Dice (+ Boundary bila use_boundary_head aktif) mask loss,
        dihitung langsung pada GT box sebagai ROI — bukan pada proposal RPN
        yang di-match seperti maskrcnn_loss bawaan torchvision — supaya tidak
        perlu mereplikasi logika IoU-matching proposal secara manual.
        """
        roi_heads = self._model.roi_heads
        device = next(iter(features.values())).device

        gt_boxes = [t["boxes"] for t in targets]
        gt_masks = [t["masks"] for t in targets]

        if all(b.numel() == 0 for b in gt_boxes):
            return torch.zeros((), device=device)

        pooled = roi_heads.mask_roi_pool(features, gt_boxes, image_shapes)
        mask_features = roi_heads.mask_head(pooled)
        mask_logits = roi_heads.mask_predictor(mask_features)  # (N, num_classes, M, M)

        labels = torch.cat([t["labels"] for t in targets], dim=0)
        idx = torch.arange(labels.shape[0], device=labels.device)
        pred_logits = mask_logits[idx, labels]  # (N, M, M)

        size = pred_logits.shape[-1]
        mask_targets = torch.cat(
            [_project_masks_on_boxes(m, b, size) for m, b in zip(gt_masks, gt_boxes)],
            dim=0,
        )

        weights = self.loss_weights
        l_focal = focal_loss(pred_logits, mask_targets)
        l_dice = dice_loss(pred_logits, mask_targets)

        l_boundary = torch.zeros((), device=device)
        mask_head = roi_heads.mask_head
        if getattr(mask_head, "boundary_head", None) is not None and mask_head.last_boundary_map is not None:
            pred_boundary = F.interpolate(
                mask_head.last_boundary_map, size=(size, size), mode="bilinear", align_corners=False
            )
            l_boundary = boundary_loss(pred_boundary, mask_targets)

        return (
            weights.get("focal", 1.0) * l_focal
            + weights.get("dice", 1.0) * l_dice
            + weights.get("boundary", 1.0) * l_boundary
        )

    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        self._model.train(mode)
        return super().train(mode)

    def eval(self):
        self._model.eval()
        return super().eval()

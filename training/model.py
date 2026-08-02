from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import MultiScaleRoIAlign

if TYPE_CHECKING:
    pass


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
    """Mask head that applies ASPP + optional boundary attention before conv layers."""

    def __init__(
        self,
        in_channels: int,
        layers: list[int],
        dilation: int,
        aspp: ASPPModule,
        boundary_head: BoundaryAttentionHead | None,
    ) -> None:
        super().__init__()
        self.aspp = aspp
        self.boundary_head = boundary_head

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
        aspp_rates: list[int] = None,
        aspp_out_channels: int = 256,
        trainable_backbone_layers: int = 3,
        use_boundary_head: bool = True,
    ) -> None:
        super().__init__()
        if aspp_rates is None:
            aspp_rates = [6, 12, 18, 24]

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

        aspp = ASPPModule(in_channels=in_ch, out_channels=aspp_out_channels, atrous_rates=aspp_rates)
        boundary_head = BoundaryAttentionHead(in_channels=aspp_out_channels) if use_boundary_head else None

        # Override mask_head with ASPP-augmented version
        mask_layers = [256, 256, 256, 256]
        self._model.roi_heads.mask_head = ASPPMaskHead(
            in_channels=in_ch,
            layers=mask_layers,
            dilation=1,
            aspp=aspp,
            boundary_head=boundary_head,
        )

        self.use_boundary_head = use_boundary_head

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> tuple[dict, list[dict]]:
        output = self._model(images, targets)
        if self.training:
            # MaskRCNN returns loss_dict during training
            loss_dict = output
            return loss_dict, []
        return {}, output

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

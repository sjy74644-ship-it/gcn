from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        m = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        self.stage0 = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # /4
        self.stage1 = m.layer1  # /4
        self.stage2 = m.layer2  # /8
        self.stage3 = m.layer3  # /16
        self.stage4 = m.layer4  # /32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class HeatmapHead(nn.Module):
    """Decode encoder feature map to K-joint heatmaps."""

    def __init__(self, in_channels: int, num_joints: int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_joints, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decode(feat)


class SRRLProjector(nn.Module):
    """G(Fr): map radar feature to visual feature space."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseHeatmapNet(nn.Module):
    """Common network producing feature map + heatmap."""

    def __init__(self, num_joints: int, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.head = HeatmapHead(in_channels=512, num_joints=num_joints)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        heatmap = self.head(feat)
        return {"feat": feat, "heatmap": heatmap}

    def decode_with_head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class PoseTeacher(PoseHeatmapNet):
    """Teacher takes visual PNG tensors."""


class PoseStudent(PoseHeatmapNet):
    """Student takes radar PNG tensors."""


def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """Convert heatmaps [B,K,H,W] to coordinates [B,K,2] in pixel space."""
    b, k, h, w = heatmap.shape
    prob = torch.softmax(heatmap.flatten(2), dim=-1).view(b, k, h, w)

    ys = torch.linspace(0, h - 1, h, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0, w - 1, w, device=heatmap.device, dtype=heatmap.dtype)

    exp_x = (prob.sum(dim=2) * xs.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    exp_y = (prob.sum(dim=3) * ys.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return torch.stack([exp_x, exp_y], dim=-1)

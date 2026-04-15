from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18


class PoseBackbone(nn.Module):
    """A simple ResNet18-based pose estimator returning (pose, feature)."""

    def __init__(self, pose_dim: int = 6, pretrained: bool = False) -> None:
        super().__init__()
        model = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, pose_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.stem(x)
        feat = self.pool(feat_map).flatten(1)
        pose = self.head(feat)
        return pose, feat


class PoseTeacher(PoseBackbone):
    """Teacher consumes visual PNG tensors."""


class PoseStudent(PoseBackbone):
    """Student consumes radar PNG tensors."""

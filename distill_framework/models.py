from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOv8Backbone(nn.Module):
    """Use YOLOv8 backbone graph execution to extract deep feature map."""

    def __init__(self, model_name: str = "yolov8n.yaml") -> None:
        super().__init__()
        yolo = YOLO(model_name)
        # underlying DetectionModel
        self.model = yolo.model
        self.layers = self.model.model
        self.save = self.model.save

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = []
        for m in self.layers:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = cache[m.f]
                else:
                    x = [x if j == -1 else cache[j] for j in m.f]
            x = m(x)
            cache.append(x if m.i in self.save else None)

        # DetectionModel最后输出一般是list/tuple，取最高层特征
        if isinstance(x, (list, tuple)):
            x = x[-1]
        return x


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, num_joints: int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_joints, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decode(feat)


class SRRLProjector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseHeatmapNet(nn.Module):
    """YOLOv8-backbone pose heatmap net."""

    def __init__(self, num_joints: int, yolo_model: str = "yolov8n.yaml", in_channels: int = 256) -> None:
        super().__init__()
        self.encoder = YOLOv8Backbone(model_name=yolo_model)
        self.head = HeatmapHead(in_channels=in_channels, num_joints=num_joints)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        heatmap = self.head(feat)
        return {"feat": feat, "heatmap": heatmap}

    def decode_with_head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class PoseTeacher(PoseHeatmapNet):
    pass


class PoseStudent(PoseHeatmapNet):
    pass


def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    b, k, h, w = heatmap.shape
    prob = torch.softmax(heatmap.flatten(2), dim=-1).view(b, k, h, w)

    ys = torch.linspace(0, h - 1, h, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0, w - 1, w, device=heatmap.device, dtype=heatmap.dtype)

    exp_x = (prob.sum(dim=2) * xs.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    exp_y = (prob.sum(dim=3) * ys.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return torch.stack([exp_x, exp_y], dim=-1)

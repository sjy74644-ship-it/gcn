from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class YOLOv8Backbone(nn.Module):
    """Extract deep feature map from YOLOv8 graph."""

    def __init__(self, model_name: str = "yolov8n.yaml") -> None:
        super().__init__()
        yolo = YOLO(model_name)
        self.model = yolo.model
        self.layers = self.model.model
        self.save = self.model.save

    @staticmethod
    def _pick_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            cands = [YOLOv8Backbone._pick_tensor(o) for o in obj]
            cands = [c for c in cands if isinstance(c, torch.Tensor)]
            if not cands:
                return None
            # prefer deeper/higher-channel feature
            return max(cands, key=lambda t: (t.ndim, t.shape[1] if t.ndim >= 2 else 0, t.numel()))
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = []
        for m in self.layers:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = cache[m.f]
                else:
                    x_in = [x if j == -1 else cache[j] for j in m.f]
            else:
                x_in = x

            # stop before detection/pose heads, return their input feature(s)
            if m.__class__.__name__ in {"Detect", "Pose", "Segment", "OBB", "WorldDetect", "v10Detect"}:
                feat = self._pick_tensor(x_in)
                if isinstance(feat, torch.Tensor):
                    return feat
                raise RuntimeError(f"Cannot extract tensor feature before head: {m.__class__.__name__}")

            x = m(x_in)
            cache.append(x if m.i in self.save else None)

        feat = self._pick_tensor(x)
        if not isinstance(feat, torch.Tensor):
            raise RuntimeError("YOLOv8 backbone output is not a tensor.")
        return feat


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
    """Single-frame pose heatmap net with explicit output heatmap size alignment."""

    def __init__(
        self,
        num_joints: int,
        yolo_model: str = "yolov8n.yaml",
        in_channels: int = 256,
        out_heatmap_size: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = YOLOv8Backbone(model_name=yolo_model)
        self.num_joints = num_joints
        self.head = HeatmapHead(in_channels=in_channels, num_joints=num_joints)
        self.out_heatmap_size = int(out_heatmap_size)

    def _align_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        h, w = heatmap.shape[-2:]
        if h == self.out_heatmap_size and w == self.out_heatmap_size:
            return heatmap
        return F.interpolate(
            heatmap,
            size=(self.out_heatmap_size, self.out_heatmap_size),
            mode="bilinear",
            align_corners=False,
        )

    def _ensure_head_channels(self, feat: torch.Tensor) -> None:
        head_in = self.head.decode[0].in_channels
        if feat.shape[1] != head_in:
            print(f"[Model] Adjust HeatmapHead in_channels {head_in} -> {feat.shape[1]}")
            self.head = HeatmapHead(in_channels=feat.shape[1], num_joints=self.num_joints).to(feat.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        self._ensure_head_channels(feat)
        heatmap = self._align_heatmap(self.head(feat))
        return {"feat": feat, "heatmap": heatmap}

    def decode_with_head(self, feat: torch.Tensor) -> torch.Tensor:
        self._ensure_head_channels(feat)
        return self._align_heatmap(self.head(feat))


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


def load_yolov8_pose_backbone_weights(encoder: YOLOv8Backbone, yolo_pose_ckpt: str) -> dict:
    """Load as many matching weights as possible from yolov8 pose checkpoint into encoder.model."""
    src_model = YOLO(yolo_pose_ckpt).model
    src_sd = src_model.state_dict()
    tgt_sd = encoder.model.state_dict()

    new_sd = tgt_sd.copy()
    loaded = []
    skipped = []

    for k, v in tgt_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            new_sd[k] = src_sd[k]
            loaded.append(k)
        else:
            reason = "missing" if k not in src_sd else f"shape_mismatch src={tuple(src_sd[k].shape)} tgt={tuple(v.shape)}"
            skipped.append((k, reason))

    encoder.model.load_state_dict(new_sd, strict=False)

    print(f"[TeacherInit] YOLO ckpt: {yolo_pose_ckpt}")
    print(f"[TeacherInit] backbone matched+loaded: {len(loaded)}")
    print(f"[TeacherInit] backbone skipped: {len(skipped)}")
    if skipped:
        print("[TeacherInit] first skipped keys:")
        for k, r in skipped[:20]:
            print(f"  - {k}: {r}")

    return {"loaded": loaded, "skipped": skipped}

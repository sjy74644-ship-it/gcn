from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class DatasetSpec:
    data_root: Path
    labels_csv: Path
    visual_subdir: str = "visual"
    radar_subdir: str = "radar"
    visual_dir: Optional[Path] = None
    radar_dir: Optional[Path] = None
    image_ext: str = ".png"
    num_joints: int = 17
    sigma: float = 2.5
    heatmap_size: int = 64
    input_size: int = 640


class PairedPosePngDataset(Dataset):
    """Load visual/radar png pair + generate GT heatmap from keypoint CSV.

    CSV format:
      id, x1, y1, x2, y2, ... xK, yK
    where K == num_joints
    """

    def __init__(
        self,
        spec: DatasetSpec,
        transform_visual: Optional[Callable] = None,
        transform_radar: Optional[Callable] = None,
    ) -> None:
        self.spec = spec
        self.df = pd.read_csv(spec.labels_csv)
        if "id" not in self.df.columns:
            raise ValueError("labels_csv must contain an 'id' column")

        expected_cols = 1 + spec.num_joints * 2
        if len(self.df.columns) != expected_cols:
            raise ValueError(
                f"expected {expected_cols} columns (id + 2*num_joints), got {len(self.df.columns)}"
            )

        self.kpt_columns = [c for c in self.df.columns if c != "id"]
        # YOLOv8-style input: resize to imgsz and normalize to [0,1] via ToTensor
        default_tf = T.Compose([T.Resize((spec.input_size, spec.input_size)), T.ToTensor()])

        self.transform_visual = transform_visual or default_tf
        self.transform_radar = transform_radar or default_tf

        self.visual_dir = spec.visual_dir or (spec.data_root / spec.visual_subdir)
        self.radar_dir = spec.radar_dir or (spec.data_root / spec.radar_subdir)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_png_rgb(path: Path) -> Image.Image:
        image = Image.open(path)
        return image.convert("RGB")

    def _gaussian_heatmaps(self, keypoints_xy: torch.Tensor) -> torch.Tensor:
        """Generate [K,H,W] gaussian heatmaps from [K,2] coords in input_size space."""
        k = self.spec.num_joints
        h = self.spec.heatmap_size
        w = self.spec.heatmap_size
        sigma2 = self.spec.sigma * self.spec.sigma

        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)

        # coordinates are provided in input_size space, map to heatmap space
        kp = keypoints_xy.clone()
        kp[:, 0] = kp[:, 0] * (w / float(self.spec.input_size))
        kp[:, 1] = kp[:, 1] * (h / float(self.spec.input_size))

        mu_x = kp[:, 0].view(k, 1, 1)
        mu_y = kp[:, 1].view(k, 1, 1)

        heatmaps = torch.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2.0 * sigma2))
        return heatmaps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        visual_path = self.visual_dir / f"{sample_id}{self.spec.image_ext}"
        radar_path = self.radar_dir / f"{sample_id}{self.spec.image_ext}"

        if not visual_path.exists():
            raise FileNotFoundError(f"visual png not found: {visual_path}")
        if not radar_path.exists():
            raise FileNotFoundError(f"radar png not found: {radar_path}")

        visual_img = self._load_png_rgb(visual_path)
        radar_img = self._load_png_rgb(radar_path)

        visual_tensor = self.transform_visual(visual_img)
        radar_tensor = self.transform_radar(radar_img)

        kpt = torch.tensor(row[self.kpt_columns].astype(float).to_numpy(), dtype=torch.float32)
        keypoints_xy = kpt.view(self.spec.num_joints, 2)
        gt_heatmap = self._gaussian_heatmaps(keypoints_xy)

        return {
            "id": sample_id,
            "visual": visual_tensor,
            "radar": radar_tensor,
            "keypoints": keypoints_xy,
            "heatmap_gt": gt_heatmap,
        }

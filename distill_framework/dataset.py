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
    image_ext: str = ".png"
    pose_dim: int = 6


class PairedPosePngDataset(Dataset):
    """Load paired visual/radar PNG samples and pose labels.

    Expected CSV columns:
      - id
      - pose columns (e.g., tx,ty,tz,roll,pitch,yaw) with count == pose_dim
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

        pose_columns = [c for c in self.df.columns if c != "id"]
        if len(pose_columns) != spec.pose_dim:
            raise ValueError(
                f"pose_dim={spec.pose_dim}, but found {len(pose_columns)} pose columns: {pose_columns}"
            )
        self.pose_columns = pose_columns

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        default_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

        self.transform_visual = transform_visual or default_tf
        self.transform_radar = transform_radar or default_tf

        self.visual_dir = spec.data_root / spec.visual_subdir
        self.radar_dir = spec.data_root / spec.radar_subdir

    def __len__(self) -> int:
        return len(self.df)

    def _load_png_rgb(self, path: Path) -> Image.Image:
        image = Image.open(path)
        # radar PNG can be single-channel; convert both to RGB for shared backbones
        return image.convert("RGB")

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

        pose = torch.tensor(row[self.pose_columns].astype(float).to_numpy(), dtype=torch.float32)

        return {
            "id": sample_id,
            "visual": visual_tensor,
            "radar": radar_tensor,
            "pose": pose,
        }

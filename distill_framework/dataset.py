from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class DatasetSpec:
    # 数据根目录（仅用于默认值）
    data_root: Path
    input_img_dir: Path
    input_lbl_dir: Path

    # 可选视觉目录（若不提供则 visual=radar）
    visual_img_dir: Optional[Path] = None

    # 数据划分
    split: str = "train"  # train/val/all
    val_ratio: float = 0.2
    random_seed: int = 42
    auto_create_empty_label: bool = False

    # 关键点设置
    num_joints: int = 13
    kpt_dim: int = 3

    # 输入/热图设置
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    input_size: int = 640
    heatmap_size: int = 64
    sigma: float = 2.5


def check_dir_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {path}")


def list_image_files(folder: Path, exts: Sequence[str]) -> List[Path]:
    exts_set = {e.lower() for e in exts}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts_set])


def expected_label_from_input(img_path: Path, input_img_dir: Path, input_lbl_dir: Path) -> Path:
    rel = img_path.relative_to(input_img_dir)
    return input_lbl_dir / rel.with_suffix(".txt")


def validate_label_file(label_path: Path, num_joints: int, kpt_dim: int) -> Tuple[bool, str]:
    if not label_path.exists():
        return False, f"标签不存在: {label_path}"

    lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) == 0:
        return True, "empty"

    expected_cols = 1 + 4 + num_joints * kpt_dim
    for i, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != expected_cols:
            return False, f"{label_path} 第{i}行列数错误: 期望{expected_cols}, 实际{len(parts)}"
    return True, "ok"


def collect_pairs(spec: DatasetSpec) -> List[Tuple[Path, Path]]:
    check_dir_exists(spec.input_img_dir)
    check_dir_exists(spec.input_lbl_dir)

    imgs = list_image_files(spec.input_img_dir, spec.image_exts)
    if len(imgs) == 0:
        raise RuntimeError(f"输入图片目录中没有找到图片: {spec.input_img_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing = []
    bad = []

    for img_path in imgs:
        label_path = expected_label_from_input(img_path, spec.input_img_dir, spec.input_lbl_dir)

        if not label_path.exists():
            if spec.auto_create_empty_label:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("", encoding="utf-8")
            else:
                missing.append((img_path, label_path))
                continue

        ok, msg = validate_label_file(label_path, spec.num_joints, spec.kpt_dim)
        if not ok:
            bad.append(msg)
            continue

        pairs.append((img_path, label_path))

    if missing:
        samples = "\n".join([f"图像: {i} | 标签: {l}" for i, l in missing[:10]])
        raise RuntimeError(f"存在图片无标签，示例:\n{samples}")
    if bad:
        samples = "\n".join(bad[:10])
        raise RuntimeError(f"存在格式错误标签，示例:\n{samples}")

    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], val_ratio: float, random_seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    random.seed(random_seed)
    pairs = pairs[:]
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(n_total * val_ratio)
    n_val = max(1, n_val) if n_total > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


class PairedPosePngDataset(Dataset):
    """采用 YOLOv8 目录+标签配对流程读取数据。"""

    def __init__(
        self,
        spec: DatasetSpec,
        transform_visual: Optional[Callable] = None,
        transform_radar: Optional[Callable] = None,
    ) -> None:
        self.spec = spec

        all_pairs = collect_pairs(spec)
        train_pairs, val_pairs = split_pairs(all_pairs, spec.val_ratio, spec.random_seed)
        if spec.split == "train":
            self.pairs = train_pairs
        elif spec.split == "val":
            self.pairs = val_pairs
        else:
            self.pairs = all_pairs

        default_tf = T.Compose([T.Resize((spec.input_size, spec.input_size)), T.ToTensor()])
        self.transform_visual = transform_visual or default_tf
        self.transform_radar = transform_radar or default_tf

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _parse_yolo_pose_keypoints(self, label_path: Path) -> torch.Tensor:
        """解析第一行person关键点，输出 [K,2] (input_size尺度)."""
        lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines) == 0:
            return torch.zeros((self.spec.num_joints, 2), dtype=torch.float32)

        parts = lines[0].split()
        nums = [float(x) for x in parts]
        # [cls, cx, cy, w, h, kx1, ky1, v1, ...]
        kpt_vals = nums[5:]
        keypoints = []
        for k in range(self.spec.num_joints):
            x = kpt_vals[k * self.spec.kpt_dim + 0]
            y = kpt_vals[k * self.spec.kpt_dim + 1]
            keypoints.append([x * self.spec.input_size, y * self.spec.input_size])
        return torch.tensor(keypoints, dtype=torch.float32)

    def _gaussian_heatmaps(self, keypoints_xy: torch.Tensor) -> torch.Tensor:
        k = self.spec.num_joints
        h = self.spec.heatmap_size
        w = self.spec.heatmap_size
        sigma2 = self.spec.sigma * self.spec.sigma

        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)

        kp = keypoints_xy.clone()
        kp[:, 0] = kp[:, 0] * (w / float(self.spec.input_size))
        kp[:, 1] = kp[:, 1] * (h / float(self.spec.input_size))

        mu_x = kp[:, 0].view(k, 1, 1)
        mu_y = kp[:, 1].view(k, 1, 1)
        return torch.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2.0 * sigma2))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        radar_img_path, label_path = self.pairs[idx]
        rel = radar_img_path.relative_to(self.spec.input_img_dir)

        if self.spec.visual_img_dir is not None:
            visual_img_path = self.spec.visual_img_dir / rel
            if not visual_img_path.exists():
                raise FileNotFoundError(f"视觉图像不存在: {visual_img_path}")
        else:
            visual_img_path = radar_img_path

        visual_img = self._load_rgb(visual_img_path)
        radar_img = self._load_rgb(radar_img_path)

        visual_tensor = self.transform_visual(visual_img)
        radar_tensor = self.transform_radar(radar_img)

        keypoints_xy = self._parse_yolo_pose_keypoints(label_path)
        gt_heatmap = self._gaussian_heatmaps(keypoints_xy)

        sample_id = str(rel.with_suffix(""))
        return {
            "id": sample_id,
            "visual": visual_tensor,
            "radar": radar_tensor,
            "keypoints": keypoints_xy,
            "heatmap_gt": gt_heatmap,
        }

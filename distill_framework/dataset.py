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
    data_root: Path
    input_img_dir: Path
    input_lbl_dir: Path
    visual_img_dir: Optional[Path] = None

    split: str = "train"  # train/val/all
    val_ratio: float = 0.2
    random_seed: int = 42
    auto_create_empty_label: bool = False

    num_joints: int = 13
    kpt_dim: int = 3  # expected (x,y,v)

    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    visual_input_hw: Tuple[int, int] = (320, 320)  # (H, W)
    radar_input_hw: Tuple[int, int] = (320, 320)  # (H, W)
    heatmap_size: int = 64
    sigma: float = 2.5

    require_visual: bool = True
    allow_same_modal_distill: bool = False


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


def split_pairs(
    pairs: List[Tuple[Path, Path]], val_ratio: float, random_seed: int
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
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
    """Single-frame dataset with YOLO labels and valid-point masking.

    Supports different input resolutions for visual and radar branches.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        transform_visual: Optional[Callable] = None,
        transform_radar: Optional[Callable] = None,
    ) -> None:
        self.spec = spec

        if spec.require_visual and spec.visual_img_dir is None and not spec.allow_same_modal_distill:
            raise ValueError(
                "跨模态蒸馏要求提供 --visual_img_dir。"
                "如确需同模态蒸馏，请显式设置 allow_same_modal_distill=True。"
            )

        all_pairs = collect_pairs(spec)
        train_pairs, val_pairs = split_pairs(all_pairs, spec.val_ratio, spec.random_seed)
        if spec.split == "train":
            self.pairs = train_pairs
        elif spec.split == "val":
            self.pairs = val_pairs
        else:
            self.pairs = all_pairs

        if len(self.pairs) == 0:
            raise RuntimeError(f"split={spec.split} 没有可用样本，请检查数据和划分配置")

        vh, vw = spec.visual_input_hw
        rh, rw = spec.radar_input_hw
        default_visual_tf = T.Compose([T.Resize((vh, vw)), T.ToTensor()])
        default_radar_tf = T.Compose([T.Resize((rh, rw)), T.ToTensor()])
        self.transform_visual = transform_visual or default_visual_tf
        self.transform_radar = transform_radar or default_radar_tf

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _parse_yolo_pose_keypoints(self, label_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse first label line, return keypoints [K,2] in radar pixel space, valid mask [K]."""
        lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines) == 0:
            return (
                torch.zeros((self.spec.num_joints, 2), dtype=torch.float32),
                torch.zeros((self.spec.num_joints,), dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
            )

        parts = lines[0].split()
        nums = [float(x) for x in parts]
        kpt_vals = nums[5:]

        radar_h, radar_w = self.spec.radar_input_hw
        keypoints = []
        valid = []
        for k in range(self.spec.num_joints):
            base = k * self.spec.kpt_dim
            x_norm = kpt_vals[base + 0]
            y_norm = kpt_vals[base + 1]
            v = kpt_vals[base + 2] if self.spec.kpt_dim >= 3 else 1.0

            # labels are normalized -> convert to radar pixel coordinates
            keypoints.append([x_norm * radar_w, y_norm * radar_h])
            valid.append(1.0 if v > 0 else 0.0)

        valid_t = torch.tensor(valid, dtype=torch.float32)
        person_valid = torch.tensor(1.0 if valid_t.sum() > 0 else 0.0, dtype=torch.float32)
        return torch.tensor(keypoints, dtype=torch.float32), valid_t, person_valid

    def _gaussian_heatmaps(self, keypoints_xy: torch.Tensor, kp_valid: torch.Tensor) -> torch.Tensor:
        k = self.spec.num_joints
        h = self.spec.heatmap_size
        w = self.spec.heatmap_size
        sigma2 = self.spec.sigma * self.spec.sigma

        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)

        radar_h, radar_w = self.spec.radar_input_hw

        heatmaps = torch.zeros((k, h, w), dtype=torch.float32)
        for i in range(k):
            if kp_valid[i] <= 0:
                continue
            x = keypoints_xy[i, 0] * (w / float(radar_w))
            y = keypoints_xy[i, 1] * (h / float(radar_h))
            heatmaps[i] = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma2))
        return heatmaps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        radar_img_path, label_path = self.pairs[idx]
        rel = radar_img_path.relative_to(self.spec.input_img_dir)

        if self.spec.visual_img_dir is not None:
            visual_img_path = self.spec.visual_img_dir / rel
            if not visual_img_path.exists():
                raise FileNotFoundError(f"视觉图像不存在: {visual_img_path}")
        elif self.spec.allow_same_modal_distill:
            visual_img_path = radar_img_path
        else:
            raise RuntimeError("visual_img_dir 未设置且未允许同模态蒸馏。")

        visual_img = self._load_rgb(visual_img_path)
        radar_img = self._load_rgb(radar_img_path)

        visual_tensor = self.transform_visual(visual_img)
        radar_tensor = self.transform_radar(radar_img)

        keypoints_xy, kp_valid, person_valid = self._parse_yolo_pose_keypoints(label_path)
        gt_heatmap = self._gaussian_heatmaps(keypoints_xy, kp_valid)

        sample_id = str(rel.with_suffix(""))
        return {
            "id": sample_id,
            "visual": visual_tensor,
            "radar": radar_tensor,
            "keypoints": keypoints_xy,
            "kp_valid": kp_valid,
            "person_valid": person_valid,
            "heatmap_gt": gt_heatmap,
        }


class ImageOnlyDataset(Dataset):
    """Inference-only dataset, no labels required."""

    def __init__(
        self,
        input_img_dir: Path,
        radar_input_hw: Tuple[int, int] = (320, 320),
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        check_dir_exists(input_img_dir)
        self.input_img_dir = input_img_dir
        self.images = list_image_files(input_img_dir, image_exts)
        if len(self.images) == 0:
            raise RuntimeError(f"目录无可用图片: {input_img_dir}")
        rh, rw = radar_input_hw
        self.tf = T.Compose([T.Resize((rh, rw)), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.images[idx]
        rel = p.relative_to(self.input_img_dir)
        img = Image.open(p).convert("RGB")
        return {"id": str(rel.with_suffix("")), "radar": self.tf(img)}

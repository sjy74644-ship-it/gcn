from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, ImageOnlyDataset, PairedPosePngDataset, _align_hw
from distill_framework.models import PoseStudent, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict radar 2D keypoints (labels optional)")
    parser.add_argument("--input_img_dir", type=str, required=True, help="雷达图像目录")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--input_lbl_dir", type=str, default=None, help="可选：标签目录，提供后可做简单误差评估")
    parser.add_argument("--num_joints", type=int, default=13)
    parser.add_argument("--kpt_dim", type=int, default=3)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--radar_input_h", type=int, default=640)
    parser.add_argument("--radar_input_w", type=int, default=640)
    parser.add_argument("--yolo_model", type=str, default="yolov8n.yaml")
    parser.add_argument("--feat_channels", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _build_loader(args: argparse.Namespace) -> DataLoader:
    img_dir = Path(args.input_img_dir)
    radar_model_hw = _align_hw((args.radar_input_h, args.radar_input_w), 32)
    args._radar_model_hw = radar_model_hw
    if not img_dir.exists():
        raise FileNotFoundError(f"input_img_dir 不存在: {img_dir}")

    if args.input_lbl_dir is None:
        ds = ImageOnlyDataset(input_img_dir=img_dir, radar_input_hw=radar_model_hw)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    lbl_dir = Path(args.input_lbl_dir)
    if not lbl_dir.exists():
        raise FileNotFoundError(f"input_lbl_dir 不存在: {lbl_dir}")

    spec = DatasetSpec(
        data_root=img_dir.parent,
        input_img_dir=img_dir,
        input_lbl_dir=lbl_dir,
        visual_img_dir=img_dir,
        split="all",
        num_joints=args.num_joints,
        kpt_dim=args.kpt_dim,
        heatmap_size=args.heatmap_size,
        visual_input_hw=radar_model_hw,
        radar_input_hw=radar_model_hw,
        require_visual=False,
        allow_same_modal_distill=True,
    )
    ds = PairedPosePngDataset(spec)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = _build_loader(args)

    student = PoseStudent(
        num_joints=args.num_joints,
        yolo_model=args.yolo_model,
        in_channels=args.feat_channels,
        out_heatmap_size=args.heatmap_size,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "student" in ckpt:
        student.load_state_dict(ckpt["student"])
    else:
        student.load_state_dict(ckpt)

    student.eval()
    rows = []
    l2_sum = 0.0
    l2_count = 0

    with torch.no_grad():
        for batch in dl:
            radar = batch["radar"].to(device)
            out = student(radar)
            coords = soft_argmax_2d(out["heatmap"])
            rmh, rmw = args._radar_model_hw
            coords[..., 0] = coords[..., 0] * (float(rmw) / float(args.heatmap_size))
            coords[..., 1] = coords[..., 1] * (float(rmh) / float(args.heatmap_size))

            ids = batch["id"]
            for i, sid in enumerate(ids):
                row = {"id": sid}
                for k in range(args.num_joints):
                    px = float(coords[i, k, 0].cpu().item())
                    py = float(coords[i, k, 1].cpu().item())
                    row[f"pred_x{k+1}"] = px
                    row[f"pred_y{k+1}"] = py
                rows.append(row)

            if args.input_lbl_dir is not None and "keypoints" in batch and "kp_valid" in batch:
                gt = batch["keypoints"]
                valid = batch["kp_valid"]
                diff = (coords.cpu() - gt) ** 2
                dist = torch.sqrt(diff.sum(dim=-1))
                l2_sum += float((dist * valid).sum().item())
                l2_count += int(valid.sum().item())

    out_df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved radar predicted 2D coordinates to: {args.output_csv}")

    if args.input_lbl_dir is not None:
        if l2_count > 0:
            print(f"Eval mean L2 (valid joints only): {l2_sum / l2_count:.4f}")
        else:
            print("Eval skipped: no valid joints in provided labels.")


if __name__ == "__main__":
    main()

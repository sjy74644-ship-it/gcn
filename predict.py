from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, PairedPosePngDataset
from distill_framework.models import PoseStudent, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict radar 2D keypoints from paired path inputs")
    parser.add_argument("--visual_dir", type=str, required=True, help="视觉图像路径目录")
    parser.add_argument("--radar_dir", type=str, required=True, help="雷达图像路径目录")
    parser.add_argument("--gt_csv", type=str, required=True, help="2D坐标真值CSV路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="student checkpoint路径")
    parser.add_argument("--output_csv", type=str, required=True, help="雷达预测2D坐标输出CSV")

    parser.add_argument("--num_joints", type=int, default=17)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--yolo_model", type=str, default="yolov8n.yaml")
    parser.add_argument("--feat_channels", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = DatasetSpec(
        data_root=Path("."),
        labels_csv=Path(args.gt_csv),
        visual_dir=Path(args.visual_dir),
        radar_dir=Path(args.radar_dir),
        num_joints=args.num_joints,
        heatmap_size=args.heatmap_size,
        input_size=args.input_size,
    )
    ds = PairedPosePngDataset(spec)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    student = PoseStudent(
        num_joints=args.num_joints,
        yolo_model=args.yolo_model,
        in_channels=args.feat_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "student" in ckpt:
        student.load_state_dict(ckpt["student"])
    else:
        student.load_state_dict(ckpt)

    student.eval()
    rows = []

    with torch.no_grad():
        for batch in dl:
            radar = batch["radar"].to(device)
            out = student(radar)
            coords = soft_argmax_2d(out["heatmap"])  # [B,K,2], heatmap-space
            coords = coords * (float(args.input_size) / float(args.heatmap_size))

            ids = batch["id"]
            for i, sid in enumerate(ids):
                row = {"id": sid}
                for k in range(args.num_joints):
                    row[f"pred_x{k+1}"] = float(coords[i, k, 0].cpu().item())
                    row[f"pred_y{k+1}"] = float(coords[i, k, 1].cpu().item())
                rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved radar predicted 2D coordinates to: {args.output_csv}")


if __name__ == "__main__":
    main()

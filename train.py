from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, PairedPosePngDataset
from distill_framework.models import PoseStudent, PoseTeacher, SRRLProjector
from distill_framework.trainer import DistillationConfig, DistillationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual-to-radar SRRL pose distillation (YOLOv8 backbone)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--num_joints", type=int, default=17)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=640, help="YOLOv8 imgsz")

    parser.add_argument("--yolo_model", type=str, default="yolov8n.yaml", help="e.g. yolov8n.yaml/yolov8s.yaml")
    parser.add_argument("--feat_channels", type=int, default=256, help="YOLOv8 neck deep feature channels")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_repr", type=float, default=0.5)
    parser.add_argument("--w_head", type=float, default=1.0)
    parser.add_argument("--w_rel", type=float, default=0.2)
    parser.add_argument("--w_temp", type=float, default=0.1)

    parser.add_argument("--use_pseudo_sup", action="store_true", default=False)
    parser.add_argument("--simple_repr", action="store_true", default=False)

    parser.add_argument("--freeze_teacher", action="store_true", default=True)
    parser.add_argument("--train_teacher", action="store_false", dest="freeze_teacher")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_spec = DatasetSpec(
        data_root=Path(args.data_root),
        labels_csv=Path(args.labels_csv),
        num_joints=args.num_joints,
        sigma=args.sigma,
        heatmap_size=args.heatmap_size,
        input_size=args.input_size,
    )
    train_ds = PairedPosePngDataset(ds_spec)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    teacher = PoseTeacher(num_joints=args.num_joints, yolo_model=args.yolo_model, in_channels=args.feat_channels)
    student = PoseStudent(num_joints=args.num_joints, yolo_model=args.yolo_model, in_channels=args.feat_channels)
    projector = SRRLProjector(channels=args.feat_channels)

    config = DistillationConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        freeze_teacher=args.freeze_teacher,
        w_sup=args.w_sup,
        w_repr=args.w_repr,
        w_head=args.w_head,
        w_rel=args.w_rel,
        w_temp=args.w_temp,
        use_stat_repr=not args.simple_repr,
        use_pseudo_sup=args.use_pseudo_sup,
        save_dir=args.save_dir,
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        projector=projector,
        config=config,
        device=device,
    )
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()

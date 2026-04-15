from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, PairedPosePngDataset
from distill_framework.models import PoseStudent, PoseTeacher
from distill_framework.trainer import DistillationConfig, DistillationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual-to-radar pose distillation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--pose_dim", type=int, default=6)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--alpha_pose", type=float, default=1.0)
    parser.add_argument("--alpha_kd_out", type=float, default=1.0)
    parser.add_argument("--alpha_kd_feat", type=float, default=0.5)

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
        pose_dim=args.pose_dim,
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

    teacher = PoseTeacher(pose_dim=args.pose_dim, pretrained=False)
    student = PoseStudent(pose_dim=args.pose_dim, pretrained=False)

    config = DistillationConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        freeze_teacher=args.freeze_teacher,
        alpha_pose=args.alpha_pose,
        alpha_kd_out=args.alpha_kd_out,
        alpha_kd_feat=args.alpha_kd_feat,
        save_dir=args.save_dir,
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        config=config,
        device=device,
    )
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()

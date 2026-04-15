from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, PairedPosePngDataset
from distill_framework.models import PoseStudent, PoseTeacher, SRRLProjector
from distill_framework.trainer import DistillationConfig, DistillationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-frame visual->radar SRRL distillation training")

    parser.add_argument("--input_img_dir", type=str, required=True, help="雷达图像目录 (images)")
    parser.add_argument("--input_lbl_dir", type=str, required=True, help="标签目录 (labels)")
    parser.add_argument("--visual_img_dir", type=str, required=True, help="视觉图像目录（跨模态蒸馏必填）")
    parser.add_argument("--allow_same_modal_distill", action="store_true", default=False)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--auto_create_empty_label", action="store_true", default=False)

    parser.add_argument("--num_joints", type=int, default=13)
    parser.add_argument("--kpt_dim", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--visual_input_h", type=int, default=270)
    parser.add_argument("--visual_input_w", type=int, default=480)
    parser.add_argument("--radar_input_h", type=int, default=640)
    parser.add_argument("--radar_input_w", type=int, default=640)

    parser.add_argument("--yolo_model", type=str, default="yolov8n.yaml")
    parser.add_argument("--feat_channels", type=int, default=256)

    parser.add_argument("--teacher_backbone_ckpt", type=str, default=None)
    parser.add_argument("--teacher_head_ckpt", type=str, default=None)
    parser.add_argument("--allow_random_teacher", action="store_true", default=False)
    parser.add_argument("--strict_teacher_load", action="store_true", default=True)

    parser.add_argument("--train_teacher", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_repr", type=float, default=0.5)
    parser.add_argument("--w_head", type=float, default=1.0)
    parser.add_argument("--w_rel", type=float, default=0.2)

    parser.add_argument("--use_pseudo_sup", action="store_true", default=False)
    parser.add_argument("--simple_repr", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--force_no_weights_only_load", action="store_true", default=True)
    parser.add_argument("--kmp_duplicate_lib_ok", action="store_true", default=True)
    parser.add_argument("--omp_num_threads", type=str, default="1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.force_no_weights_only_load:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if args.kmp_duplicate_lib_ok:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads

    input_img_dir = Path(args.input_img_dir)
    input_lbl_dir = Path(args.input_lbl_dir)
    visual_img_dir = Path(args.visual_img_dir)

    if not input_img_dir.exists():
        raise FileNotFoundError(f"input_img_dir 不存在: {input_img_dir}")
    if not input_lbl_dir.exists():
        raise FileNotFoundError(f"input_lbl_dir 不存在: {input_lbl_dir}")
    if not visual_img_dir.exists() and not args.allow_same_modal_distill:
        raise FileNotFoundError(f"visual_img_dir 不存在: {visual_img_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_spec = DatasetSpec(
        data_root=input_img_dir.parent,
        input_img_dir=input_img_dir,
        input_lbl_dir=input_lbl_dir,
        visual_img_dir=visual_img_dir if visual_img_dir.exists() else None,
        split="train",
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        auto_create_empty_label=args.auto_create_empty_label,
        num_joints=args.num_joints,
        kpt_dim=args.kpt_dim,
        sigma=args.sigma,
        heatmap_size=args.heatmap_size,
        visual_input_hw=(args.visual_input_h, args.visual_input_w),
        radar_input_hw=(args.radar_input_h, args.radar_input_w),
        require_visual=not args.allow_same_modal_distill,
        allow_same_modal_distill=args.allow_same_modal_distill,
    )
    train_ds = PairedPosePngDataset(ds_spec)
    val_ds = PairedPosePngDataset(DatasetSpec(**{**ds_spec.__dict__, "split": "val"}))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    teacher = PoseTeacher(
        num_joints=args.num_joints,
        yolo_model=args.yolo_model,
        in_channels=args.feat_channels,
        out_heatmap_size=args.heatmap_size,
    )
    student = PoseStudent(
        num_joints=args.num_joints,
        yolo_model=args.yolo_model,
        in_channels=args.feat_channels,
        out_heatmap_size=args.heatmap_size,
    )
    projector = SRRLProjector(channels=args.feat_channels)

    config = DistillationConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        train_teacher=args.train_teacher,
        teacher_backbone_ckpt=args.teacher_backbone_ckpt,
        teacher_head_ckpt=args.teacher_head_ckpt,
        strict_teacher_load=args.strict_teacher_load,
        allow_random_teacher=args.allow_random_teacher,
        w_sup=args.w_sup,
        w_repr=args.w_repr,
        w_head=args.w_head,
        w_rel=args.w_rel,
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
    trainer.fit(train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()

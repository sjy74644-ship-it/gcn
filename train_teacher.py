from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from distill_framework.dataset import DatasetSpec, PairedPosePngDataset
from distill_framework.models import PoseTeacher, load_yolov8_pose_backbone_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-only pretraining (single-frame)")
    parser.add_argument("--visual_img_dir", type=str, required=True, help="视觉图像目录")
    parser.add_argument("--input_lbl_dir", type=str, required=True, help="标签目录")
    parser.add_argument("--init_yolo_ckpt", type=str, required=True, help="例如 yolov8n-pose.pt")

    parser.add_argument("--num_joints", type=int, default=13)
    parser.add_argument("--kpt_dim", type=int, default=3)
    parser.add_argument("--visual_input_h", type=int, default=270)
    parser.add_argument("--visual_input_w", type=int, default=480)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.5)

    parser.add_argument("--yolo_model", type=str, default="yolov8n.yaml")
    parser.add_argument("--feat_channels", type=int, default=256)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_dir", type=str, default="./teacher_ckpts")
    return parser.parse_args()


def masked_heatmap_mse(pred: torch.Tensor, target: torch.Tensor, kp_valid: torch.Tensor) -> torch.Tensor:
    b, k, h, w = pred.shape
    mask = kp_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def build_datasets(args: argparse.Namespace) -> tuple[PairedPosePngDataset, PairedPosePngDataset]:
    visual_dir = Path(args.visual_img_dir)
    lbl_dir = Path(args.input_lbl_dir)
    if not visual_dir.exists():
        raise FileNotFoundError(f"visual_img_dir 不存在: {visual_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"input_lbl_dir 不存在: {lbl_dir}")

    common = dict(
        data_root=visual_dir.parent,
        input_img_dir=visual_dir,
        input_lbl_dir=lbl_dir,
        visual_img_dir=visual_dir,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        num_joints=args.num_joints,
        kpt_dim=args.kpt_dim,
        visual_input_hw=(args.visual_input_h, args.visual_input_w),
        radar_input_hw=(args.visual_input_h, args.visual_input_w),
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        require_visual=False,
        allow_same_modal_distill=True,
    )
    train_ds = PairedPosePngDataset(DatasetSpec(split="train", **common))
    val_ds = PairedPosePngDataset(DatasetSpec(split="val", **common))
    return train_ds, val_ds


def evaluate(model: PoseTeacher, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["visual"].to(device)
            gt = batch["heatmap_gt"].to(device)
            kp_valid = batch["kp_valid"].to(device)
            pred = model(x)["heatmap"]
            loss = masked_heatmap_mse(pred, gt, kp_valid)
            total += loss.item()
            n += 1
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ckpt = Path(args.init_yolo_ckpt)
    if not init_ckpt.exists():
        raise FileNotFoundError(f"init_yolo_ckpt 不存在: {init_ckpt}")

    train_ds, val_ds = build_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    teacher = PoseTeacher(
        num_joints=args.num_joints,
        yolo_model=args.yolo_model,
        in_channels=args.feat_channels,
        out_heatmap_size=args.heatmap_size,
    ).to(device)

    report = load_yolov8_pose_backbone_weights(teacher.encoder, str(init_ckpt))
    print(f"[TeacherInit] loaded_keys={len(report['loaded'])}, skipped_keys={len(report['skipped'])}")
    print("[TeacherInit] custom HeatmapHead is randomly initialized and will be trained from scratch.")

    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.num_epochs + 1):
        teacher.train()
        running = 0.0
        steps = 0

        for batch in train_loader:
            x = batch["visual"].to(device)
            gt = batch["heatmap_gt"].to(device)
            kp_valid = batch["kp_valid"].to(device)

            pred = teacher(x)["heatmap"]
            loss = masked_heatmap_mse(pred, gt, kp_valid)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1

        train_loss = running / max(1, steps)
        val_loss = evaluate(teacher, val_loader, device)
        print(f"[TeacherPretrain][Epoch {epoch}] train_sup={train_loss:.4f}, val_sup={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "teacher": teacher.state_dict(),
                "encoder": teacher.encoder.state_dict(),
                "head": teacher.head.state_dict(),
                "args": vars(args),
                "init_report": {
                    "loaded": len(report["loaded"]),
                    "skipped": len(report["skipped"]),
                },
            }

    if best_state is None:
        raise RuntimeError("teacher pretraining did not produce any checkpoint")

    torch.save(best_state, save_dir / "teacher_best_full.pt")
    torch.save(best_state["encoder"], save_dir / "teacher_backbone.pt")
    torch.save(best_state["head"], save_dir / "teacher_head.pt")

    print(f"[TeacherPretrain] done. best_val={best_val:.4f}")
    print(f"[TeacherPretrain] exported: {save_dir / 'teacher_backbone.pt'}")
    print(f"[TeacherPretrain] exported: {save_dir / 'teacher_head.pt'}")


if __name__ == "__main__":
    main()

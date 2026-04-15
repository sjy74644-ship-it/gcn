from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .losses import DistillLossWeights, PoseDistillLoss


@dataclass
class DistillationConfig:
    lr: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 20
    num_workers: int = 4
    freeze_teacher: bool = True
    alpha_pose: float = 1.0
    alpha_kd_out: float = 1.0
    alpha_kd_feat: float = 0.5
    save_dir: str = "./checkpoints"


class DistillationTrainer:
    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        config: DistillationConfig,
        device: torch.device,
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config
        self.device = device

        if config.freeze_teacher:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        else:
            self.teacher.train()

        self.loss_fn = PoseDistillLoss(
            DistillLossWeights(
                alpha_pose=config.alpha_pose,
                alpha_kd_out=config.alpha_kd_out,
                alpha_kd_feat=config.alpha_kd_feat,
            )
        )
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=config.lr)

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "visual": batch["visual"].to(self.device, non_blocking=True),
            "radar": batch["radar"].to(self.device, non_blocking=True),
            "pose": batch["pose"].to(self.device, non_blocking=True),
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.student.train()
        running = {"total": 0.0, "pose": 0.0, "kd_out": 0.0, "kd_feat": 0.0}

        for batch in dataloader:
            batch = self._move_batch(batch)

            with torch.no_grad() if self.config.freeze_teacher else torch.enable_grad():
                teacher_pose, teacher_feat = self.teacher(batch["visual"])

            student_pose, student_feat = self.student(batch["radar"])
            losses = self.loss_fn(
                student_pose=student_pose,
                teacher_pose=teacher_pose,
                gt_pose=batch["pose"],
                student_feat=student_feat,
                teacher_feat=teacher_feat,
            )

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()

        num_batches = max(1, len(dataloader))
        avg = {k: v / num_batches for k, v in running.items()}
        print(f"[Epoch {epoch}] " + ", ".join([f"{k}: {v:.4f}" for k, v in avg.items()]))
        return avg

    def fit(self, train_loader: DataLoader) -> None:
        best_loss = float("inf")
        for epoch in range(1, self.config.num_epochs + 1):
            metrics = self.train_one_epoch(train_loader, epoch)
            ckpt_path = self.save_dir / f"student_epoch_{epoch}.pt"
            torch.save(self.student.state_dict(), ckpt_path)

            if metrics["total"] < best_loss:
                best_loss = metrics["total"]
                torch.save(self.student.state_dict(), self.save_dir / "student_best.pt")

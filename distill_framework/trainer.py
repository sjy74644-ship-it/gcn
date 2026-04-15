from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .losses import DistillLossWeights, SRRLPoseDistillLoss
from .models import SRRLProjector, soft_argmax_2d


@dataclass
class DistillationConfig:
    lr: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 20
    num_workers: int = 4
    freeze_teacher: bool = True

    w_sup: float = 1.0
    w_repr: float = 0.5
    w_head: float = 1.0
    w_rel: float = 0.2
    w_temp: float = 0.1

    use_stat_repr: bool = True
    use_pseudo_sup: bool = False
    save_dir: str = "./checkpoints"


class DistillationTrainer:
    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        projector: SRRLProjector,
        config: DistillationConfig,
        device: torch.device,
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.projector = projector.to(device)
        self.config = config
        self.device = device

        if config.freeze_teacher:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        else:
            self.teacher.train()

        self.loss_fn = SRRLPoseDistillLoss(
            DistillLossWeights(
                w_sup=config.w_sup,
                w_repr=config.w_repr,
                w_head=config.w_head,
                w_rel=config.w_rel,
                w_temp=config.w_temp,
            ),
            use_stat_repr=config.use_stat_repr,
        )

        params = list(self.student.parameters()) + list(self.projector.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.lr)

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "visual": batch["visual"].to(self.device, non_blocking=True),
            "radar": batch["radar"].to(self.device, non_blocking=True),
            "heatmap_gt": batch["heatmap_gt"].to(self.device, non_blocking=True),
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.student.train()
        self.projector.train()
        running = {"total": 0.0, "sup": 0.0, "repr": 0.0, "head": 0.0, "rel": 0.0, "temp": 0.0}

        for batch in dataloader:
            batch = self._move_batch(batch)

            with torch.no_grad() if self.config.freeze_teacher else torch.enable_grad():
                out_v = self.teacher(batch["visual"])
            out_r = self.student(batch["radar"])

            f_r_proj = self.projector(out_r["feat"])
            # SRRL head loss: use frozen visual head D_v to decode projected radar feature
            h_rt = self.teacher.decode_with_head(f_r_proj)

            # optional temporal smooth: if your batch is sequence, reshape then pass;
            # for image batch currently disabled by None.
            losses = self.loss_fn(
                h_r=out_r["heatmap"],
                h_v=out_v["heatmap"],
                h_rt=h_rt,
                f_r_proj=f_r_proj,
                f_v=out_v["feat"],
                h_gt=None if self.config.use_pseudo_sup else batch["heatmap_gt"],
                h_pseudo=out_v["heatmap"] if self.config.use_pseudo_sup else None,
                pred_coords_seq=None,
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

    @torch.no_grad()
    def infer_keypoints(self, radar_batch: torch.Tensor) -> torch.Tensor:
        self.student.eval()
        out_r = self.student(radar_batch.to(self.device))
        return soft_argmax_2d(out_r["heatmap"])

    def fit(self, train_loader: DataLoader) -> None:
        best_loss = float("inf")
        for epoch in range(1, self.config.num_epochs + 1):
            metrics = self.train_one_epoch(train_loader, epoch)

            torch.save(
                {
                    "student": self.student.state_dict(),
                    "projector": self.projector.state_dict(),
                },
                self.save_dir / f"student_epoch_{epoch}.pt",
            )

            if metrics["total"] < best_loss:
                best_loss = metrics["total"]
                torch.save(
                    {
                        "student": self.student.state_dict(),
                        "projector": self.projector.state_dict(),
                    },
                    self.save_dir / "student_best.pt",
                )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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

    train_teacher: bool = False
    teacher_backbone_ckpt: Optional[str] = None
    teacher_head_ckpt: Optional[str] = None
    strict_teacher_load: bool = True
    allow_random_teacher: bool = False

    w_sup: float = 1.0
    w_repr: float = 0.5
    w_head: float = 1.0
    w_rel: float = 0.2

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

        self._load_teacher_weights_or_fail()
        self._setup_teacher_trainability()

        self.loss_fn = SRRLPoseDistillLoss(
            DistillLossWeights(
                w_sup=config.w_sup,
                w_repr=config.w_repr,
                w_head=config.w_head,
                w_rel=config.w_rel,
            ),
            use_stat_repr=config.use_stat_repr,
        )

        params = list(self.student.parameters()) + list(self.projector.parameters())
        if config.train_teacher:
            params += list(self.teacher.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.lr)

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[Init] teacher_trainable={config.train_teacher}, "
            f"use_pseudo_sup={config.use_pseudo_sup}, single_frame=True"
        )

    def _safe_load(self, module: torch.nn.Module, ckpt_path: str, strict: bool, module_name: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"{module_name} checkpoint 不存在: {path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = module.load_state_dict(state, strict=strict)
        print(
            f"[TeacherLoad] {module_name} <- {path} | strict={strict} "
            f"| missing={len(missing)} unexpected={len(unexpected)}"
        )

    def _load_teacher_weights_or_fail(self) -> None:
        cfg = self.config
        loaded_any = False

        if cfg.teacher_backbone_ckpt:
            self._safe_load(self.teacher.encoder, cfg.teacher_backbone_ckpt, cfg.strict_teacher_load, "teacher.backbone")
            loaded_any = True
        if cfg.teacher_head_ckpt:
            self._safe_load(self.teacher.head, cfg.teacher_head_ckpt, cfg.strict_teacher_load, "teacher.head")
            loaded_any = True

        if not loaded_any:
            if cfg.allow_random_teacher:
                print("[TeacherLoad] 未提供teacher权重，已显式允许随机teacher（不推荐）。")
            else:
                raise RuntimeError(
                    "未提供 teacher 权重。请至少设置 --teacher_backbone_ckpt 和/或 --teacher_head_ckpt，"
                    "或显式设置 --allow_random_teacher。"
                )

    def _setup_teacher_trainability(self) -> None:
        if self.config.train_teacher:
            self.teacher.train()
            for p in self.teacher.parameters():
                p.requires_grad = True
            print("[Teacher] mode=train, parameters are trainable")
        else:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print("[Teacher] mode=eval, parameters are frozen")

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "visual": batch["visual"].to(self.device, non_blocking=True),
            "radar": batch["radar"].to(self.device, non_blocking=True),
            "heatmap_gt": batch["heatmap_gt"].to(self.device, non_blocking=True),
            "kp_valid": batch["kp_valid"].to(self.device, non_blocking=True),
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.student.train()
        self.projector.train()
        if self.config.train_teacher:
            self.teacher.train()

        running = {"total": 0.0, "sup": 0.0, "repr": 0.0, "head": 0.0, "rel": 0.0}

        for batch in dataloader:
            batch = self._move_batch(batch)

            teacher_ctx = torch.enable_grad() if self.config.train_teacher else torch.no_grad()
            with teacher_ctx:
                out_v = self.teacher(batch["visual"])

            out_r = self.student(batch["radar"])

            if out_r["heatmap"].shape[-2:] != batch["heatmap_gt"].shape[-2:]:
                raise ValueError(
                    f"student heatmap与GT尺寸不一致: pred={out_r['heatmap'].shape}, gt={batch['heatmap_gt'].shape}"
                )

            f_r_proj = self.projector(out_r["feat"])
            h_rt = self.teacher.decode_with_head(f_r_proj)

            losses = self.loss_fn(
                h_r=out_r["heatmap"],
                h_v=out_v["heatmap"],
                h_rt=h_rt,
                f_r_proj=f_r_proj,
                f_v=out_v["feat"],
                h_gt=batch["heatmap_gt"],
                kp_valid=batch["kp_valid"],
                use_pseudo_sup=self.config.use_pseudo_sup,
            )

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()

        num_batches = max(1, len(dataloader))
        avg = {k: v / num_batches for k, v in running.items()}
        print(
            f"[Epoch {epoch}] "
            + ", ".join([f"{k}: {v:.4f}" for k, v in avg.items()])
            + f", teacher_trainable={self.config.train_teacher}"
        )
        return avg


    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.student.eval()
        self.projector.eval()
        self.teacher.eval()

        running = {"total": 0.0, "sup": 0.0, "repr": 0.0, "head": 0.0, "rel": 0.0}

        for batch in dataloader:
            batch = self._move_batch(batch)
            out_v = self.teacher(batch["visual"])
            out_r = self.student(batch["radar"])

            if out_r["heatmap"].shape[-2:] != batch["heatmap_gt"].shape[-2:]:
                raise ValueError(
                    f"[VAL] student heatmap与GT尺寸不一致: pred={out_r['heatmap'].shape}, gt={batch['heatmap_gt'].shape}"
                )

            f_r_proj = self.projector(out_r["feat"])
            h_rt = self.teacher.decode_with_head(f_r_proj)

            losses = self.loss_fn(
                h_r=out_r["heatmap"],
                h_v=out_v["heatmap"],
                h_rt=h_rt,
                f_r_proj=f_r_proj,
                f_v=out_v["feat"],
                h_gt=batch["heatmap_gt"],
                kp_valid=batch["kp_valid"],
                use_pseudo_sup=self.config.use_pseudo_sup,
            )

            for k in running:
                running[k] += losses[k].item()

        num_batches = max(1, len(dataloader))
        avg = {k: v / num_batches for k, v in running.items()}
        print(f"[Val {epoch}] " + ", ".join([f"{k}: {v:.4f}" for k, v in avg.items()]))
        return avg

    @torch.no_grad()
    def infer_keypoints(self, radar_batch: torch.Tensor) -> torch.Tensor:
        self.student.eval()
        out_r = self.student(radar_batch.to(self.device))
        return soft_argmax_2d(out_r["heatmap"])

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        best_loss = float("inf")
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            if val_loader is not None:
                eval_metrics = self.validate_one_epoch(val_loader, epoch)
                monitor_loss = eval_metrics["total"]
            else:
                eval_metrics = None
                monitor_loss = train_metrics["total"]

            ckpt = {
                "student": self.student.state_dict(),
                "projector": self.projector.state_dict(),
                "teacher": self.teacher.state_dict(),
                "config": self.config.__dict__,
                "train_metrics": train_metrics,
                "val_metrics": eval_metrics,
            }
            torch.save(ckpt, self.save_dir / f"student_epoch_{epoch}.pt")

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                torch.save(ckpt, self.save_dir / "student_best.pt")

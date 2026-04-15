from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillLossWeights:
    alpha_pose: float = 1.0
    alpha_kd_out: float = 1.0
    alpha_kd_feat: float = 0.5


class PoseDistillLoss(nn.Module):
    def __init__(self, weights: DistillLossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.pose_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        student_pose: torch.Tensor,
        teacher_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_pose = self.pose_criterion(student_pose, gt_pose)
        loss_kd_out = self.pose_criterion(student_pose, teacher_pose.detach())

        # cosine feature alignment in [0, 2]
        cos_sim = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=1)
        loss_kd_feat = (1.0 - cos_sim).mean()

        total = (
            self.weights.alpha_pose * loss_pose
            + self.weights.alpha_kd_out * loss_kd_out
            + self.weights.alpha_kd_feat * loss_kd_feat
        )

        return {
            "total": total,
            "pose": loss_pose,
            "kd_out": loss_kd_out,
            "kd_feat": loss_kd_feat,
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillLossWeights:
    w_sup: float = 1.0
    w_repr: float = 0.5
    w_head: float = 1.0
    w_rel: float = 0.2


class SRRLPoseDistillLoss(nn.Module):
    """Single-frame SRRL distillation loss (temporal term intentionally removed)."""

    def __init__(self, weights: DistillLossWeights, use_stat_repr: bool = True) -> None:
        super().__init__()
        self.weights = weights
        self.use_stat_repr = use_stat_repr

    @staticmethod
    def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(a, b)

    @staticmethod
    def _channel_stats(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = feat.mean(dim=(2, 3))
        var = feat.var(dim=(2, 3), unbiased=False)
        std = torch.sqrt(var + 1e-6)
        return mu, std

    @staticmethod
    def _relation_matrix(tokens: torch.Tensor) -> torch.Tensor:
        q = F.normalize(tokens, p=2, dim=-1)
        return torch.bmm(q, q.transpose(1, 2))

    @staticmethod
    def heatmap_tokens(heatmap: torch.Tensor) -> torch.Tensor:
        b, k, h, w = heatmap.shape
        return heatmap.view(b, k, h * w)

    @staticmethod
    def _expand_joint_mask(kp_valid: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # kp_valid: [B,K]
        return kp_valid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    def forward(
        self,
        h_r: torch.Tensor,
        h_v: torch.Tensor,
        h_rt: torch.Tensor,
        f_r_proj: torch.Tensor,
        f_v: torch.Tensor,
        h_gt: Optional[torch.Tensor],
        kp_valid: torch.Tensor,
        use_pseudo_sup: bool,
    ) -> Dict[str, torch.Tensor]:
        if h_r.shape != h_v.shape or h_r.shape != h_rt.shape:
            raise ValueError(f"heatmap shape mismatch: Hr={h_r.shape}, Hv={h_v.shape}, Hrt={h_rt.shape}")

        b, k, h, w = h_r.shape
        joint_mask = self._expand_joint_mask(kp_valid, h, w)
        denom = joint_mask.sum().clamp_min(1.0)

        if use_pseudo_sup:
            sup_target = h_v.detach()
        else:
            if h_gt is None:
                raise ValueError("use_pseudo_sup=False 时 h_gt 不能为空")
            sup_target = h_gt

        l_sup = (((h_r - sup_target) ** 2) * joint_mask).sum() / denom

        if self.use_stat_repr:
            mu_r, std_r = self._channel_stats(f_r_proj)
            mu_v, std_v = self._channel_stats(f_v.detach())
            l_repr = self._l2(mu_r, mu_v) + self._l2(std_r, std_v)
        else:
            l_repr = self._l2(f_r_proj, f_v.detach())

        l_head = (((h_rt - h_v.detach()) ** 2) * joint_mask).sum() / denom

        r_r = self._relation_matrix(self.heatmap_tokens(h_r))
        r_v = self._relation_matrix(self.heatmap_tokens(h_v.detach()))
        # relation mask: if either joint is invalid -> do not supervise
        rel_mask = torch.bmm(kp_valid.unsqueeze(-1), kp_valid.unsqueeze(1))
        rel_denom = rel_mask.sum().clamp_min(1.0)
        l_rel = (((r_r - r_v) ** 2) * rel_mask).sum() / rel_denom

        total = (
            self.weights.w_sup * l_sup
            + self.weights.w_repr * l_repr
            + self.weights.w_head * l_head
            + self.weights.w_rel * l_rel
        )

        return {
            "total": total,
            "sup": l_sup,
            "repr": l_repr,
            "head": l_head,
            "rel": l_rel,
        }

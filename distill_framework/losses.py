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
    w_temp: float = 0.1


class SRRLPoseDistillLoss(nn.Module):
    """L = L_sup + 0.5 L_repr + 1.0 L_head + 0.2 L_rel + 0.1 L_temp."""

    def __init__(self, weights: DistillLossWeights, use_stat_repr: bool = True) -> None:
        super().__init__()
        self.weights = weights
        self.use_stat_repr = use_stat_repr

    @staticmethod
    def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(a, b)

    @staticmethod
    def _channel_stats(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feat: [B,C,H,W]
        mu = feat.mean(dim=(2, 3))
        var = feat.var(dim=(2, 3), unbiased=False)
        std = torch.sqrt(var + 1e-6)
        return mu, std

    @staticmethod
    def _relation_matrix(tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,K,D]
        q = F.normalize(tokens, p=2, dim=-1)
        return torch.bmm(q, q.transpose(1, 2))

    @staticmethod
    def heatmap_tokens(heatmap: torch.Tensor) -> torch.Tensor:
        # [B,K,H,W] -> [B,K,D], D=H*W
        b, k, h, w = heatmap.shape
        return heatmap.view(b, k, h * w)

    @staticmethod
    def temporal_smooth_loss(coords: torch.Tensor) -> torch.Tensor:
        # coords: [B,T,K,2]
        if coords.ndim != 4 or coords.shape[1] < 2:
            return coords.new_tensor(0.0)
        diff = coords[:, 1:] - coords[:, :-1]
        return (diff * diff).mean()

    def forward(
        self,
        h_r: torch.Tensor,
        h_v: torch.Tensor,
        h_rt: torch.Tensor,
        f_r_proj: torch.Tensor,
        f_v: torch.Tensor,
        h_gt: Optional[torch.Tensor] = None,
        h_pseudo: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
        pred_coords_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1) supervision: ||Hr - Hgt||^2 or pseudo labels ||Hr - sg(Hv)||^2
        if h_gt is not None:
            if joint_weights is not None:
                w = joint_weights.view(1, -1, 1, 1)
                l_sup = ((h_r - h_gt) ** 2 * w).mean()
            else:
                l_sup = self._l2(h_r, h_gt)
        elif h_pseudo is not None:
            l_sup = self._l2(h_r, h_pseudo.detach())
        else:
            l_sup = h_r.new_tensor(0.0)

        # 2) SRRL repr loss
        if self.use_stat_repr:
            mu_r, std_r = self._channel_stats(f_r_proj)
            mu_v, std_v = self._channel_stats(f_v.detach())
            l_repr = self._l2(mu_r, mu_v) + self._l2(std_r, std_v)
        else:
            l_repr = self._l2(f_r_proj, f_v.detach())

        # 3) head distill loss: ||Dv(G(Fr)) - sg(Hv)||^2
        l_head = self._l2(h_rt, h_v.detach())

        # 4) relation distill from heatmap tokens
        r_r = self._relation_matrix(self.heatmap_tokens(h_r))
        r_v = self._relation_matrix(self.heatmap_tokens(h_v.detach()))
        l_rel = self._l2(r_r, r_v)

        # 5) temporal smooth
        if pred_coords_seq is not None:
            l_temp = self.temporal_smooth_loss(pred_coords_seq)
        else:
            l_temp = h_r.new_tensor(0.0)

        total = (
            self.weights.w_sup * l_sup
            + self.weights.w_repr * l_repr
            + self.weights.w_head * l_head
            + self.weights.w_rel * l_rel
            + self.weights.w_temp * l_temp
        )

        return {
            "total": total,
            "sup": l_sup,
            "repr": l_repr,
            "head": l_head,
            "rel": l_rel,
            "temp": l_temp,
        }

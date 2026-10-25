from __future__ import annotations

import math

import torch
import torch.nn as nn


def masked_log_mean_exp_max(
    h: torch.Tensor,
    active_mask: torch.Tensor,
    tau: torch.Tensor | float,
) -> torch.Tensor:
    """Compute stabilized masked LogMeanExp over active assets along dim=1.

    Detaching the per-row maximum shift preserves the exact forward value while
    keeping the autograd graph strictly C-infinity smooth across exact asset ties.
    """
    active_bool = active_mask > 0.5
    active_3d = active_bool.unsqueeze(-1)
    m = active_bool.to(dtype=h.dtype)
    count = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    has_active = active_bool.any(dim=1, keepdim=True)

    # Use torch.where so extreme or NaN values in inactive slots never enter arithmetic.
    h_clean = torch.where(active_3d, h, torch.zeros_like(h))

    neg_large = torch.full_like(h_clean, -1e9)
    h_for_shift = torch.where(active_3d, h_clean.detach(), neg_large)
    h_shift = h_for_shift.max(dim=1, keepdim=True).values
    h_shift = torch.where(has_active.unsqueeze(-1), h_shift, torch.zeros_like(h_shift))

    scaled = torch.where(active_3d, (h_clean - h_shift) / tau, torch.zeros_like(h_clean))
    exp_terms = torch.where(active_3d, torch.exp(scaled), torch.zeros_like(h_clean))
    mean_exp = torch.where(
        has_active,
        exp_terms.sum(dim=1) / count,
        torch.ones_like(h_clean[:, 0, :]),
    )
    lse = h_shift.squeeze(1) + tau * torch.log(mean_exp)
    return torch.where(has_active, lse, torch.zeros_like(lse))


def masked_log_mean_exp_min(
    h: torch.Tensor,
    active_mask: torch.Tensor,
    tau: torch.Tensor | float,
) -> torch.Tensor:
    """Compute stabilized masked soft-minimum via negated LogMeanExp."""
    active_bool = active_mask > 0.5
    h_neg = torch.where(active_bool.unsqueeze(-1), -h, torch.zeros_like(h))
    return -masked_log_mean_exp_max(h_neg, active_mask, tau)


class OrderStatisticPooling(nn.Module):
    """Wagstaff 5-operator permutation-invariant order-statistic readout bank.

    Computes [h_mean, h_min, h_max, h_softmin(tau_min), h_softmax(tau_max)] over
    active basket constituents and projects the 5*C concatenation to out_dim.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        init_tau: float = 0.15,
        sharp_tau: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if init_tau <= 0.0:
            raise ValueError(f"init_tau must be > 0, got {init_tau}")
        if sharp_tau <= 0.0:
            raise ValueError(f"sharp_tau must be > 0, got {sharp_tau}")

        self.hidden_dim = hidden_dim
        self.out_dim = 2 * hidden_dim if out_dim is None else out_dim
        self.init_tau = float(init_tau)
        self.sharp_tau = float(sharp_tau)

        init_log_tau = math.log(self.init_tau)
        self.log_tau_min = nn.Parameter(torch.full((1, hidden_dim), init_log_tau))
        self.log_tau_max = nn.Parameter(torch.full((1, hidden_dim), init_log_tau))

        self.proj = nn.Sequential(
            nn.Linear(5 * hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )

        self._tau_cache_key: tuple[int, int, torch.dtype, torch.device] | None = None
        self._tau_cache_val: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def tau_min(self) -> torch.Tensor:
        return torch.exp(self.log_tau_min).clamp(min=1e-3, max=10.0)

    @property
    def tau_max(self) -> torch.Tensor:
        return torch.exp(self.log_tau_max).clamp(min=1e-3, max=10.0)

    def _get_taus(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            key = (self.log_tau_min._version, self.log_tau_max._version, dtype, device)
            if self._tau_cache_key == key and self._tau_cache_val is not None:
                return self._tau_cache_val
            with torch.no_grad():
                t_min = self.tau_min.detach().to(device=device, dtype=dtype)
                t_max = self.tau_max.detach().to(device=device, dtype=dtype)
                inv_t_min = (1.0 / t_min).unsqueeze(1)
                inv_t_max = (1.0 / t_max).unsqueeze(1)
            self._tau_cache_key = key
            self._tau_cache_val = (t_min, t_max, inv_t_min, inv_t_max)
            return self._tau_cache_val
        t_min = self.tau_min.to(device=device, dtype=dtype)
        t_max = self.tau_max.to(device=device, dtype=dtype)
        return t_min, t_max, (1.0 / t_min).unsqueeze(1), (1.0 / t_max).unsqueeze(1)

    def compute_five_statistics(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        asset_weights: torch.Tensor | None = None,
        *,
        all_active: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h_mean, h_min, h_max, h_softmin, h_softmax) each of shape (B, C)."""
        inv_tau_sharp = 1.0 / self.sharp_tau
        tau_min, tau_max, inv_tau_min_3d, inv_tau_max_3d = self._get_taus(h.device, h.dtype)

        if all_active is None:
            active_bool = asset_mask > 0.5
            is_all_active = bool(h.shape[1] > 0 and active_bool.all())
        else:
            is_all_active = bool(h.shape[1] > 0 and all_active)
            active_bool = asset_mask > 0.5 if not is_all_active else asset_mask

        if is_all_active:
            if asset_weights is not None:
                w_clamped = asset_weights.clamp(min=0.0)
                w_sq = w_clamped * w_clamped
                w_gate = ((w_sq * w_clamped) / (w_sq + 1e-3)).unsqueeze(-1)
                h_clean = h * w_gate
            else:
                h_clean = h

            h_mean = h_clean.mean(dim=1)
            with torch.no_grad():
                h_det = h_clean.detach()
                shift_max = h_det.amax(dim=1, keepdim=True)
                shift_min = h_det.amin(dim=1, keepdim=True)
                s_max_sq = shift_max.squeeze(1)
                s_min_sq = shift_min.squeeze(1)

            diff_max = h_clean - shift_max
            diff_min = shift_min - h_clean
            scaled_4 = torch.cat(
                [
                    diff_max * inv_tau_sharp,
                    diff_min * inv_tau_sharp,
                    diff_max * inv_tau_max_3d,
                    diff_min * inv_tau_min_3d,
                ],
                dim=-1,
            )
            lme_4 = torch.log(torch.exp(scaled_4).mean(dim=1))
            lme_max_s, lme_min_s, lme_max_t, lme_min_t = lme_4.chunk(4, dim=-1)
            h_max = s_max_sq + self.sharp_tau * lme_max_s
            h_min = s_min_sq - self.sharp_tau * lme_min_s
            h_softmax = s_max_sq + tau_max * lme_max_t
            h_softmin = s_min_sq - tau_min * lme_min_t
            return h_mean, h_min, h_max, h_softmin, h_softmax

        active_3d = active_bool.unsqueeze(-1)
        m = active_bool.to(dtype=h.dtype)
        count = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        inv_count = 1.0 / count
        has_active = active_bool.any(dim=1, keepdim=True)
        has_active_3d = has_active.unsqueeze(-1)
        active_float = has_active.to(dtype=h.dtype)

        zeros_h = torch.zeros_like(h)
        h_clean = torch.where(active_3d, h, zeros_h)
        if asset_weights is not None:
            w_clamped = torch.where(active_bool, asset_weights.clamp(min=0.0), torch.zeros_like(asset_weights))
            w_sq = w_clamped * w_clamped
            w_gate = ((w_sq * w_clamped) / (w_sq + 1e-3)).unsqueeze(-1)
            h_clean = h_clean * w_gate

        h_mean = h_clean.sum(dim=1) * inv_count

        with torch.no_grad():
            h_det = h_clean.detach()
            neg_large = torch.full_like(h_det, -1e9)
            pos_large = torch.full_like(h_det, 1e9)
            shift_max = torch.where(
                has_active_3d,
                torch.where(active_3d, h_det, neg_large).max(dim=1, keepdim=True).values,
                torch.zeros_like(h_det[:, :1, :]),
            )
            shift_min = torch.where(
                has_active_3d,
                torch.where(active_3d, h_det, pos_large).min(dim=1, keepdim=True).values,
                torch.zeros_like(h_det[:, :1, :]),
            )

        diff_max = torch.where(active_3d, h_clean - shift_max, zeros_h)
        diff_min = torch.where(active_3d, shift_min - h_clean, zeros_h)

        exp_max_sharp = (torch.exp(diff_max * inv_tau_sharp) * m.unsqueeze(-1)).sum(dim=1) * inv_count
        exp_min_sharp = (torch.exp(diff_min * inv_tau_sharp) * m.unsqueeze(-1)).sum(dim=1) * inv_count
        exp_max_soft = (torch.exp(diff_max * inv_tau_max_3d) * m.unsqueeze(-1)).sum(dim=1) * inv_count
        exp_min_soft = (torch.exp(diff_min * inv_tau_min_3d) * m.unsqueeze(-1)).sum(dim=1) * inv_count

        s_max_sq = shift_max.squeeze(1)
        s_min_sq = shift_min.squeeze(1)

        h_max = (s_max_sq + self.sharp_tau * torch.log(exp_max_sharp.clamp(min=1e-30))) * active_float
        h_min = (s_min_sq - self.sharp_tau * torch.log(exp_min_sharp.clamp(min=1e-30))) * active_float
        h_softmax = (s_max_sq + tau_max * torch.log(exp_max_soft.clamp(min=1e-30))) * active_float
        h_softmin = (s_min_sq - tau_min * torch.log(exp_min_soft.clamp(min=1e-30))) * active_float
        return h_mean, h_min, h_max, h_softmin, h_softmax

    def forward(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        asset_weights: torch.Tensor | None = None,
        *,
        all_active: bool | None = None,
    ) -> torch.Tensor:
        h_mean, h_min, h_max, h_softmin, h_softmax = self.compute_five_statistics(
            h, asset_mask, asset_weights=asset_weights, all_active=all_active
        )
        cat_stats = torch.cat([h_mean, h_min, h_max, h_softmin, h_softmax], dim=-1)
        proj_lin = self.proj[0]
        proj_norm = self.proj[1]
        assert isinstance(proj_lin, nn.Linear) and isinstance(proj_norm, nn.LayerNorm)
        lin_out = torch.nn.functional.linear(cat_stats, proj_lin.weight, proj_lin.bias)
        norm_out = torch.nn.functional.layer_norm(
            lin_out, (self.out_dim,), proj_norm.weight, proj_norm.bias, proj_norm.eps
        )
        return torch.nn.functional.silu(norm_out)

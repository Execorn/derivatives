"""FiLM-conditioned 7-operator permutation-invariant order-statistic pooling."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    # Zero inactive slots before arithmetic so NaN/Inf padding cannot leak gradients.
    h_clean = torch.where(active_3d, h, torch.zeros_like(h))

    neg_inf = torch.full_like(h_clean, -float("inf"))
    h_for_shift = torch.where(active_3d, h_clean.detach(), neg_inf)
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


def _freeze_pool_buffers(module: nn.Module, *_args: object) -> None:
    if isinstance(module, OrderStatisticPooling):
        module.freeze_eval_buffers()


class OrderStatisticPooling(nn.Module):
    """FiLM-conditioned 7-operator permutation-invariant order-statistic readout bank.

    Computes [h_mean, h_min, h_max, h_softmin(tau_min), h_softmax(tau_max), h_std, h_wsum]
    over active basket constituents and modulates the projected summary via FiLM from
    (call_idx, cpn_idx, opt_idx) functional embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: Optional[int] = None,
        init_tau: float = 0.15,
        sharp_tau: float = 0.10,
        n_func_codes: int = 15,
        func_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if init_tau <= 0.0:
            raise ValueError(f"init_tau must be > 0, got {init_tau}")
        if sharp_tau <= 0.0:
            raise ValueError(f"sharp_tau must be > 0, got {sharp_tau}")
        if n_func_codes < 1:
            raise ValueError(f"n_func_codes must be >= 1, got {n_func_codes}")
        if func_emb_dim < 1:
            raise ValueError(f"func_emb_dim must be >= 1, got {func_emb_dim}")

        self.hidden_dim = int(hidden_dim)
        self.out_dim = 2 * int(hidden_dim) if out_dim is None else int(out_dim)
        self.init_tau = float(init_tau)
        self.sharp_tau = float(sharp_tau)
        self.n_func_codes = int(n_func_codes)
        self.func_emb_dim = int(func_emb_dim)

        init_log_tau = math.log(self.init_tau)
        self.log_tau_min = nn.Parameter(torch.full((1, self.hidden_dim), init_log_tau))
        self.log_tau_max = nn.Parameter(torch.full((1, self.hidden_dim), init_log_tau))

        self.emb_call = nn.Embedding(self.n_func_codes, self.func_emb_dim)
        self.emb_cpn = nn.Embedding(self.n_func_codes, self.func_emb_dim)
        self.emb_opt = nn.Embedding(self.n_func_codes, self.func_emb_dim)
        self.film_mlp = nn.Sequential(
            nn.Linear(3 * self.func_emb_dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, 2 * self.out_dim),
        )
        # Initialize FiLM modulation to identity (delta_gamma=0, delta_beta=0) at epoch 0.
        film_last = self.film_mlp[2]
        assert isinstance(film_last, nn.Linear)
        nn.init.zeros_(film_last.weight)
        nn.init.zeros_(film_last.bias)

        self.proj = nn.Sequential(
            nn.Linear(7 * self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )

        self.register_buffer(
            "_frozen_tau_min",
            torch.full((1, self.hidden_dim), self.init_tau),
            persistent=False,
        )
        self.register_buffer(
            "_frozen_tau_max",
            torch.full((1, self.hidden_dim), self.init_tau),
            persistent=False,
        )
        self.register_buffer(
            "_frozen_inv_tau_min_3d",
            torch.full((1, 1, self.hidden_dim), 1.0 / self.init_tau),
            persistent=False,
        )
        self.register_buffer(
            "_frozen_inv_tau_max_3d",
            torch.full((1, 1, self.hidden_dim), 1.0 / self.init_tau),
            persistent=False,
        )
        self.register_buffer(
            "_default_func_codes",
            torch.zeros((1, 3), dtype=torch.long),
            persistent=False,
        )

        self.freeze_eval_buffers()
        self.register_load_state_dict_post_hook(_freeze_pool_buffers)  # type: ignore[no-untyped-call]

    @property
    def tau_min(self) -> torch.Tensor:
        return torch.exp(self.log_tau_min).clamp(min=1e-3, max=10.0)

    @property
    def tau_max(self) -> torch.Tensor:
        return torch.exp(self.log_tau_max).clamp(min=1e-3, max=10.0)

    def freeze_eval_buffers(self) -> None:
        """Materialize static non-inference tau buffers for graph-break-free eval and CUDAGraphs."""
        with torch.inference_mode(False), torch.no_grad():
            t_min = torch.exp(self.log_tau_min).clamp(min=1e-3, max=10.0).detach()
            t_max = torch.exp(self.log_tau_max).clamp(min=1e-3, max=10.0).detach()
            frozen_t_min = self._frozen_tau_min
            frozen_t_max = self._frozen_tau_max
            frozen_inv_min = self._frozen_inv_tau_min_3d
            frozen_inv_max = self._frozen_inv_tau_max_3d
            assert isinstance(frozen_t_min, torch.Tensor)
            assert isinstance(frozen_t_max, torch.Tensor)
            assert isinstance(frozen_inv_min, torch.Tensor)
            assert isinstance(frozen_inv_max, torch.Tensor)
            frozen_t_min.copy_(t_min.to(device=frozen_t_min.device, dtype=frozen_t_min.dtype))
            frozen_t_max.copy_(t_max.to(device=frozen_t_max.device, dtype=frozen_t_max.dtype))
            frozen_inv_min.copy_(
                (1.0 / t_min).unsqueeze(1).to(device=frozen_inv_min.device, dtype=frozen_inv_min.dtype)
            )
            frozen_inv_max.copy_(
                (1.0 / t_max).unsqueeze(1).to(device=frozen_inv_max.device, dtype=frozen_inv_max.dtype)
            )

    def train(self, mode: bool = True) -> OrderStatisticPooling:
        super().train(mode)
        if not mode:
            self.freeze_eval_buffers()
        return self

    def _resolve_taus(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            frozen_t_min = self._frozen_tau_min
            frozen_t_max = self._frozen_tau_max
            frozen_inv_min = self._frozen_inv_tau_min_3d
            frozen_inv_max = self._frozen_inv_tau_max_3d
            assert isinstance(frozen_t_min, torch.Tensor)
            assert isinstance(frozen_t_max, torch.Tensor)
            assert isinstance(frozen_inv_min, torch.Tensor)
            assert isinstance(frozen_inv_max, torch.Tensor)
            return (
                frozen_t_min.to(device=device, dtype=dtype),
                frozen_t_max.to(device=device, dtype=dtype),
                frozen_inv_min.to(device=device, dtype=dtype),
                frozen_inv_max.to(device=device, dtype=dtype),
            )
        t_min = self.tau_min.to(device=device, dtype=dtype)
        t_max = self.tau_max.to(device=device, dtype=dtype)
        return t_min, t_max, (1.0 / t_min).unsqueeze(1), (1.0 / t_max).unsqueeze(1)

    def compute_seven_statistics(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        asset_weights: Optional[torch.Tensor] = None,
        *,
        all_active: Optional[bool] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return (h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum) each of shape (B, C)."""
        inv_tau_sharp = 1.0 / self.sharp_tau
        tau_min, tau_max, inv_tau_min_3d, inv_tau_max_3d = self._resolve_taus(h.device, h.dtype)

        if all_active is True and h.shape[1] > 0:
            h_mean = h.mean(dim=1)
            diff_mean = h - h_mean.unsqueeze(1)
            h_std = torch.sqrt((diff_mean * diff_mean).mean(dim=1) + 1e-6)

            if asset_weights is not None:
                w_abs_sum = asset_weights.abs().sum(dim=1, keepdim=True) + 1e-6
                w_norm = (asset_weights / w_abs_sum).unsqueeze(-1)
                h_wsum = (h * w_norm).sum(dim=1)
            else:
                h_wsum = torch.zeros_like(h_mean)

            with torch.no_grad():
                h_det = h.detach()
                shift_max = h_det.amax(dim=1, keepdim=True)
                shift_min = h_det.amin(dim=1, keepdim=True)
                s_max_sq = shift_max.squeeze(1)
                s_min_sq = shift_min.squeeze(1)

            diff_max = h - shift_max
            diff_min = shift_min - h
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
            return h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum

        active_bool = asset_mask > 0.5
        active_3d = active_bool.unsqueeze(-1)
        m = active_bool.to(dtype=h.dtype)
        count = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        inv_count = 1.0 / count
        has_active = active_bool.any(dim=1, keepdim=True)
        has_active_3d = has_active.unsqueeze(-1)
        active_float = has_active.to(dtype=h.dtype)

        zeros_h = torch.zeros_like(h)
        h_clean = torch.where(active_3d, h, zeros_h)
        h_mean = h_clean.sum(dim=1) * inv_count

        diff_mean = torch.where(active_3d, h_clean - h_mean.unsqueeze(1), zeros_h)
        h_var = (diff_mean * diff_mean).sum(dim=1) * inv_count
        h_std = torch.sqrt(h_var + 1e-6) * active_float

        if asset_weights is not None:
            w_active = torch.where(active_bool, asset_weights, torch.zeros_like(asset_weights))
            w_abs_sum = w_active.abs().sum(dim=1, keepdim=True) + 1e-6
            w_norm = (w_active / w_abs_sum).unsqueeze(-1)
            h_wsum = (h_clean * w_norm).sum(dim=1) * active_float
        else:
            h_wsum = torch.zeros_like(h_mean)

        with torch.no_grad():
            h_det = h_clean.detach()
            neg_inf = torch.full_like(h_det, -float("inf"))
            pos_inf = torch.full_like(h_det, float("inf"))
            shift_max = torch.where(
                has_active_3d,
                torch.where(active_3d, h_det, neg_inf).max(dim=1, keepdim=True).values,
                torch.zeros_like(h_det[:, :1, :]),
            )
            shift_min = torch.where(
                has_active_3d,
                torch.where(active_3d, h_det, pos_inf).min(dim=1, keepdim=True).values,
                torch.zeros_like(h_det[:, :1, :]),
            )

        diff_max = torch.where(active_3d, h_clean - shift_max, zeros_h)
        diff_min = torch.where(active_3d, shift_min - h_clean, zeros_h)

        m_3d = m.unsqueeze(-1)
        exp_max_sharp = (torch.exp(diff_max * inv_tau_sharp) * m_3d).sum(dim=1) * inv_count
        exp_min_sharp = (torch.exp(diff_min * inv_tau_sharp) * m_3d).sum(dim=1) * inv_count
        exp_max_soft = (torch.exp(diff_max * inv_tau_max_3d) * m_3d).sum(dim=1) * inv_count
        exp_min_soft = (torch.exp(diff_min * inv_tau_min_3d) * m_3d).sum(dim=1) * inv_count

        s_max_sq = shift_max.squeeze(1)
        s_min_sq = shift_min.squeeze(1)

        h_max = (s_max_sq + self.sharp_tau * torch.log(exp_max_sharp.clamp(min=1e-30))) * active_float
        h_min = (s_min_sq - self.sharp_tau * torch.log(exp_min_sharp.clamp(min=1e-30))) * active_float
        h_softmax = (s_max_sq + tau_max * torch.log(exp_max_soft.clamp(min=1e-30))) * active_float
        h_softmin = (s_min_sq - tau_min * torch.log(exp_min_soft.clamp(min=1e-30))) * active_float
        return h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum

    def compute_five_statistics(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        asset_weights: Optional[torch.Tensor] = None,
        *,
        all_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper returning the first 5 permutation-invariant statistics."""
        h_mean, h_min, h_max, h_softmin, h_softmax, _, _ = self.compute_seven_statistics(
            h, asset_mask, asset_weights=asset_weights, all_active=all_active
        )
        return h_mean, h_min, h_max, h_softmin, h_softmax

    def forward(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        asset_weights: Optional[torch.Tensor] = None,
        func_codes: Optional[torch.Tensor] = None,
        *,
        all_active: Optional[bool] = None,
    ) -> torch.Tensor:
        h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum = self.compute_seven_statistics(
            h, asset_mask, asset_weights=asset_weights, all_active=all_active
        )
        cat_stats = torch.cat(
            [h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum], dim=-1
        )
        proj_lin = self.proj[0]
        proj_norm = self.proj[1]
        assert isinstance(proj_lin, nn.Linear) and isinstance(proj_norm, nn.LayerNorm)
        lin_out = F.linear(cat_stats, proj_lin.weight, proj_lin.bias)

        if func_codes is None:
            default_fc = self._default_func_codes
            assert isinstance(default_fc, torch.Tensor)
            fc = default_fc.to(device=h.device).expand(h.shape[0], 3)
        else:
            fc = func_codes.clamp(0, self.n_func_codes - 1)

        e_call = self.emb_call(fc[:, 0])
        e_cpn = self.emb_cpn(fc[:, 1])
        e_opt = self.emb_opt(fc[:, 2])
        e_func = torch.cat([e_call, e_cpn, e_opt], dim=-1).to(dtype=lin_out.dtype)

        flin0 = self.film_mlp[0]
        flin2 = self.film_mlp[2]
        assert isinstance(flin0, nn.Linear) and isinstance(flin2, nn.Linear)
        film_raw = F.linear(F.silu(F.linear(e_func, flin0.weight, flin0.bias)), flin2.weight, flin2.bias)
        delta_gamma, delta_beta = film_raw.chunk(2, dim=-1)

        modulated = (1.0 + delta_gamma) * lin_out + delta_beta
        norm_out = F.layer_norm(
            modulated, (self.out_dim,), proj_norm.weight, proj_norm.bias, proj_norm.eps
        )
        return F.silu(norm_out)


__all__ = [
    "OrderStatisticPooling",
    "masked_log_mean_exp_max",
    "masked_log_mean_exp_min",
]

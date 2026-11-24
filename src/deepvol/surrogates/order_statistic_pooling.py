"""FiLM-conditioned 10-operator permutation-invariant order-statistic pooling."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

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
    if h.shape[1] == 0:
        return h.new_zeros((h.shape[0], h.shape[2]))
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
    """FiLM-conditioned permutation-invariant order-statistic readout bank.

    Combines 7 core basket statistics with differentiable pairwise soft-rank operators
    (2nd-worst, 3rd-worst, 2nd-best) and modulates the projection via FiLM from
    (call_idx, cpn_idx, opt_idx) functional embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: Optional[int] = None,
        init_tau: float = 0.15,
        sharp_tau: float = 0.10,
        n_func_codes: int = 18,
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
        self.soft_rank_proj = nn.Linear(3 * self.hidden_dim, self.out_dim, bias=False)

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
            for name, src in (
                ("_frozen_tau_min", t_min),
                ("_frozen_tau_max", t_max),
                ("_frozen_inv_tau_min_3d", (1.0 / t_min).unsqueeze(1)),
                ("_frozen_inv_tau_max_3d", (1.0 / t_max).unsqueeze(1)),
            ):
                buf = self._buffers.get(name)
                assert isinstance(buf, torch.Tensor)
                val = src.to(device=buf.device, dtype=buf.dtype).contiguous()
                if buf.is_inference() or val.is_inference():
                    # Rebind outside inference_mode rather than inplace copy_ on an inference tensor.
                    self._buffers[name] = val.clone()
                else:
                    buf.copy_(val)

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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Preserve strict=True compatibility with legacy 15-code checkpoints lacking soft_rank_proj.
        proj_key = prefix + "proj.0.weight"
        sr_key = prefix + "soft_rank_proj.weight"
        if proj_key in state_dict and sr_key not in state_dict:
            ref_proj = state_dict[proj_key]
            state_dict[sr_key] = torch.zeros(
                self.soft_rank_proj.weight.shape,
                dtype=ref_proj.dtype,
                device=ref_proj.device,
            )

        for emb_name in ("emb_call.weight", "emb_cpn.weight", "emb_opt.weight"):
            emb_key = prefix + emb_name
            emb_val = state_dict.get(emb_key)
            if (
                isinstance(emb_val, torch.Tensor)
                and emb_val.ndim == 2
                and emb_val.shape[0] == 15
                and self.n_func_codes > 15
                and emb_val.shape[1] == self.func_emb_dim
            ):
                padded = torch.empty(
                    (self.n_func_codes, self.func_emb_dim),
                    dtype=emb_val.dtype,
                    device=emb_val.device,
                )
                padded[:15] = emb_val
                for extra_idx in range(15, self.n_func_codes):
                    src_idx = 1 if (extra_idx == 17 and emb_val.shape[0] > 1) else 0
                    padded[extra_idx] = emb_val[src_idx]
                state_dict[emb_key] = padded

        super()._load_from_state_dict(  # type: ignore[no-untyped-call]
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        if h.shape[1] == 0:
            zeros = h.new_zeros((h.shape[0], h.shape[2]))
            return zeros, zeros, zeros, zeros, zeros, zeros, zeros

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
        eps_tiny = max(float(torch.finfo(h.dtype).tiny), 1e-30)

        h_max = (s_max_sq + self.sharp_tau * torch.log(exp_max_sharp.clamp(min=eps_tiny))) * active_float
        h_min = (s_min_sq - self.sharp_tau * torch.log(exp_min_sharp.clamp(min=eps_tiny))) * active_float
        h_softmax = (s_max_sq + tau_max * torch.log(exp_max_soft.clamp(min=eps_tiny))) * active_float
        h_softmin = (s_min_sq - tau_min * torch.log(exp_min_soft.clamp(min=eps_tiny))) * active_float
        return h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum

    def compute_soft_rank_statistics(
        self,
        h: torch.Tensor,
        asset_mask: torch.Tensor,
        *,
        all_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return differentiable order statistics (h_min2, h_min3, h_max2) each of shape (B, C)."""
        d_dim = h.shape[1]
        if d_dim == 0:
            zeros = h.new_zeros((h.shape[0], h.shape[2]))
            return zeros, zeros, zeros

        eps_tiny = max(float(torch.finfo(h.dtype).tiny), 1e-30)
        inv_2sigma2 = 0.5 / (0.55 * 0.55)
        eye_mask = (1.0 - torch.eye(d_dim, device=h.device, dtype=h.dtype)).unsqueeze(0).unsqueeze(-1)

        if all_active is True:
            h_mean = h.mean(dim=1, keepdim=True)
            diff_mean = h - h_mean
            h_std = torch.sqrt((diff_mean * diff_mean).mean(dim=1, keepdim=True) + 1e-6)
            tau_r = (0.25 * h_std).clamp(min=self.sharp_tau).unsqueeze(1)
            ranks = (torch.sigmoid((h.unsqueeze(2) - h.unsqueeze(1)) / tau_r) * eye_mask).sum(dim=2)

            t_min2_f = float(min(max(d_dim - 1, 0), 1))
            t_min3_f = float(min(max(d_dim - 1, 0), 2))
            t_max2_f = float(max(d_dim - 2, 0))
            targets_all = torch.tensor(
                [t_min2_f, t_min3_f, t_max2_f], device=h.device, dtype=h.dtype
            ).view(1, 1, 3, 1)
            log_w = -((ranks.unsqueeze(2) - targets_all) ** 2) * inv_2sigma2
            with torch.no_grad():
                shift_w = log_w.amax(dim=1, keepdim=True)
            w_raw = torch.exp((log_w - shift_w).clamp(max=0.0))
            w_sum = w_raw.sum(dim=1, keepdim=True).clamp(min=eps_tiny)
            h_soft_3 = (w_raw / w_sum * h.unsqueeze(2)).sum(dim=1)
            h_min2, h_min3, h_max2 = h_soft_3.unbind(dim=1)
            return h_min2, h_min3, h_max2

        active_bool = asset_mask > 0.5
        active_3d = active_bool.unsqueeze(-1)
        m = active_bool.to(dtype=h.dtype)
        act_count = m.sum(dim=1, keepdim=True)
        inv_count = 1.0 / act_count.clamp(min=1.0)
        has_active = active_bool.any(dim=1, keepdim=True)
        active_float = has_active.to(dtype=h.dtype)

        zeros_h = torch.zeros_like(h)
        h_clean = torch.where(active_3d, h, zeros_h)
        h_mean = h_clean.sum(dim=1) * inv_count
        diff_mean = torch.where(active_3d, h_clean - h_mean.unsqueeze(1), zeros_h)
        h_var = (diff_mean * diff_mean).sum(dim=1) * inv_count
        h_std = torch.sqrt(h_var + 1e-6) * active_float
        tau_r = (0.25 * h_std).clamp(min=self.sharp_tau).unsqueeze(1).unsqueeze(1)

        pair_mask = (m.unsqueeze(2) * m.unsqueeze(1)).unsqueeze(-1) * eye_mask
        ranks = (
            torch.sigmoid((h_clean.unsqueeze(2) - h_clean.unsqueeze(1)) / tau_r) * pair_mask
        ).sum(dim=2)

        t_min2 = (act_count - 1.0).clamp(min=0.0, max=1.0)
        t_min3 = (act_count - 1.0).clamp(min=0.0, max=2.0)
        t_max2 = (act_count - 2.0).clamp(min=0.0)
        targets = torch.cat([t_min2, t_min3, t_max2], dim=1).unsqueeze(1).unsqueeze(-1)

        log_w = -((ranks.unsqueeze(2) - targets) ** 2) * inv_2sigma2
        active_4d = active_bool.unsqueeze(-1).unsqueeze(-1)
        # Use -1e4 rather than -1e9 so float16 masking never overflows.
        log_w_masked = torch.where(active_4d, log_w, torch.full_like(log_w, -1e4))
        with torch.no_grad():
            shift_w = torch.where(
                has_active.unsqueeze(-1).unsqueeze(-1),
                log_w_masked.amax(dim=1, keepdim=True),
                torch.zeros_like(log_w_masked[:, :1]),
            )
        w_raw = torch.where(
            active_4d,
            torch.exp((log_w_masked - shift_w).clamp(max=0.0)),
            torch.zeros_like(log_w),
        )
        w_sum = w_raw.sum(dim=1, keepdim=True).clamp(min=eps_tiny)
        h_soft_3 = (w_raw / w_sum * h_clean.unsqueeze(2)).sum(dim=1) * active_float.unsqueeze(1)
        h_min2, h_min3, h_max2 = h_soft_3.unbind(dim=1)
        return h_min2, h_min3, h_max2

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
        h_min2, h_min3, h_max2 = self.compute_soft_rank_statistics(
            h, asset_mask, all_active=all_active
        )
        cat_stats = torch.cat(
            [h_mean, h_min, h_max, h_softmin, h_softmax, h_std, h_wsum], dim=-1
        )
        cat_soft_rank = torch.cat([h_min2, h_min3, h_max2], dim=-1)
        proj_lin = self.proj[0]
        proj_norm = self.proj[1]
        assert isinstance(proj_lin, nn.Linear) and isinstance(proj_norm, nn.LayerNorm)
        lin_out = F.linear(cat_stats, proj_lin.weight, proj_lin.bias) + F.linear(
            cat_soft_rank, self.soft_rank_proj.weight
        )

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

"""Tier 2 FiLM-conditioned Masked EGNO + Pre-Norm SDPA surrogate for EQAutocallable."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepvol.models.autocall.feature_layout import (
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    MonitoringIndexCode,
)
from deepvol.surrogates.order_statistic_pooling import OrderStatisticPooling


@dataclass
class AutocallSurrogateOutput:
    """Structured output of MaskedEQAutocallSurrogate."""

    stopping_probs: torch.Tensor
    survival_probs: torch.Tensor
    hazard_rates: torch.Tensor
    coupon_probs: torch.Tensor
    put_loss: torch.Tensor
    delta_hazard_logits: torch.Tensor
    delta_coupon_logits: torch.Tensor
    delta_put_loss: torch.Tensor


def _freeze_egno_layer_buffers(module: nn.Module, *_args: object) -> None:
    if isinstance(module, MaskedEGNOLayer):
        module.freeze_eval_buffers()


def _freeze_surrogate_buffers(module: nn.Module, *_args: object) -> None:
    if isinstance(module, MaskedEQAutocallSurrogate):
        module.freeze_eval_buffers()
        module._zero_res_epoch += 1
        module._zero_res_cache = None


def _normalize_batch_vec(
    val: object,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    t = torch.as_tensor(val, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(batch_size)
    flat = t.squeeze(1) if (t.ndim == 2 and t.shape[1] == 1) else t.reshape(-1)
    n_el = flat.numel()
    if n_el == 1:
        return flat.expand(batch_size)
    if n_el == batch_size:
        return flat
    if n_el > batch_size:
        return flat[:batch_size]
    raise ValueError(
        f"Expected scalar or batch-sized tensor of length {batch_size}, got shape {tuple(t.shape)}."
    )


def _resolve_is_diff_opt(global_feats: torch.Tensor) -> torch.Tensor:
    batch_size = global_feats.shape[0]
    if global_feats.shape[-1] <= int(AutocallGlobalCol.OPT_INDEX_ID):
        return torch.zeros((batch_size,), dtype=torch.bool, device=global_feats.device)
    opt_code = global_feats[:, int(AutocallGlobalCol.OPT_INDEX_ID)].detach().round().long()
    return (
        (opt_code == int(MonitoringIndexCode.DIFF_AB))
        | (opt_code == int(MonitoringIndexCode.DIFF_BA))
        | (opt_code == int(MonitoringIndexCode.DIFF_BEST_WORST))
        | (opt_code == int(MonitoringIndexCode.DIFF_BEST_AVERAGE))
        | (opt_code == int(MonitoringIndexCode.DIFF_AVERAGE_WORST))
    )


class MaskedEGNOLayer(nn.Module):
    """S_D-equivariant graph neural message passing layer over active basket assets."""

    def __init__(self, d_model: int = 128, d_edge: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_edge = int(d_edge)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model + self.d_edge, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.norm = nn.LayerNorm(self.d_model)

        self.register_buffer(
            "_eval_w_h3",
            torch.empty((3 * self.d_model, self.d_model)),
            persistent=False,
        )
        self.register_buffer(
            "_eval_b_h3",
            torch.empty((3 * self.d_model,)),
            persistent=False,
        )
        self.register_buffer(
            "_eval_w_e",
            torch.empty((self.d_model, self.d_edge)),
            persistent=False,
        )
        self.register_buffer(
            "_eval_w_agg",
            torch.empty((self.d_model, self.d_model)),
            persistent=False,
        )

        self.freeze_eval_buffers()
        self.register_load_state_dict_post_hook(_freeze_egno_layer_buffers)  # type: ignore[no-untyped-call]

    def freeze_eval_buffers(self) -> None:
        """Cache fused projection weights outside inference_mode for TorchDynamo/CUDAGraphs."""
        with torch.inference_mode(False), torch.no_grad():
            lin0 = self.edge_mlp[0]
            nlin0 = self.node_mlp[0]
            assert isinstance(lin0, nn.Linear) and isinstance(nlin0, nn.Linear)
            c = self.d_model
            w_h3 = torch.cat(
                [
                    lin0.weight[:, :c],
                    lin0.weight[:, c : 2 * c],
                    nlin0.weight[:, :c],
                ],
                dim=0,
            ).detach().contiguous()
            b_edge = (
                lin0.bias.detach()
                if lin0.bias is not None
                else torch.zeros(c, dtype=lin0.weight.dtype, device=lin0.weight.device)
            )
            b_node = (
                nlin0.bias.detach()
                if nlin0.bias is not None
                else torch.zeros(c, dtype=nlin0.weight.dtype, device=nlin0.weight.device)
            )
            b_h3 = torch.cat([b_edge, torch.zeros_like(b_edge), b_node], dim=0).contiguous()
            w_e = lin0.weight[:, 2 * c :].detach().contiguous()
            w_agg = nlin0.weight[:, c:].detach().contiguous()

            for name, src in (
                ("_eval_w_h3", w_h3),
                ("_eval_b_h3", b_h3),
                ("_eval_w_e", w_e),
                ("_eval_w_agg", w_agg),
            ):
                buf = self._buffers.get(name)
                assert isinstance(buf, torch.Tensor)
                val = src.to(device=buf.device, dtype=buf.dtype).contiguous()
                if buf.is_inference() or val.is_inference():
                    # Rebind outside inference_mode rather than inplace copy_ on an inference tensor.
                    self._buffers[name] = val.clone()
                else:
                    buf.copy_(val)

    def train(self, mode: bool = True) -> MaskedEGNOLayer:
        super().train(mode)
        if not mode:
            self.freeze_eval_buffers()
        return self

    def forward(
        self,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        asset_mask: torch.Tensor,
        pair_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        d_max = h.shape[1]
        c = self.d_model
        active_node = asset_mask > 0.5
        m_node = active_node.to(dtype=h.dtype)

        h_clean = torch.where(active_node.unsqueeze(-1), h, torch.zeros_like(h))
        eye = torch.eye(d_max, device=h.device, dtype=h.dtype).unsqueeze(0)
        pair_mask = m_node.unsqueeze(1) * m_node.unsqueeze(2) * (1.0 - eye)
        if pair_weight is not None:
            pair_mask = pair_mask * pair_weight
        active_pair = pair_mask > 0.0
        edge_clean = torch.where(
            active_pair.unsqueeze(-1), edge_attr, torch.zeros_like(edge_attr)
        )

        lin0 = self.edge_mlp[0]
        lin2 = self.edge_mlp[2]
        nlin0 = self.node_mlp[0]
        nlin2 = self.node_mlp[2]
        assert (
            isinstance(lin0, nn.Linear)
            and isinstance(lin2, nn.Linear)
            and isinstance(nlin0, nn.Linear)
            and isinstance(nlin2, nn.Linear)
        )
        proj_i = F.linear(h_clean, lin0.weight[:, :c], lin0.bias)
        proj_j = F.linear(h_clean, lin0.weight[:, c : 2 * c])
        proj_e = F.linear(edge_clean, lin0.weight[:, 2 * c :])
        hidden = F.silu(proj_i.unsqueeze(2) + proj_j.unsqueeze(1) + proj_e)
        hidden_masked = torch.where(
            active_pair.unsqueeze(-1),
            hidden * pair_mask.unsqueeze(-1),
            torch.zeros_like(hidden),
        )
        sum_p = pair_mask.sum(dim=2, keepdim=True)
        denom = sum_p.clamp(min=1.0)
        hidden_agg = hidden_masked.sum(dim=2) / denom
        agg = F.linear(hidden_agg, lin2.weight)
        if lin2.bias is not None:
            agg = agg + (sum_p / denom) * lin2.bias

        node_h = F.silu(
            F.linear(h_clean, nlin0.weight[:, :c], nlin0.bias)
            + F.linear(agg, nlin0.weight[:, c:])
        )
        node_upd = F.linear(node_h, nlin2.weight, nlin2.bias)
        out = F.layer_norm(
            h_clean + node_upd,
            (self.d_model,),
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
        )
        return torch.where(active_node.unsqueeze(-1), out, torch.zeros_like(out))

    def forward_premasked(
        self,
        h_clean: torch.Tensor,
        edge_clean: torch.Tensor,
        active_3d: torch.Tensor,
        pair_mask_4d: torch.Tensor,
        inv_denom: torch.Tensor,
        bias_scale: torch.Tensor,
    ) -> torch.Tensor:
        c = self.d_model
        lin0 = self.edge_mlp[0]
        lin2 = self.edge_mlp[2]
        nlin0 = self.node_mlp[0]
        nlin2 = self.node_mlp[2]
        assert (
            isinstance(lin0, nn.Linear)
            and isinstance(lin2, nn.Linear)
            and isinstance(nlin0, nn.Linear)
            and isinstance(nlin2, nn.Linear)
        )

        if not self.training:
            w_h3 = self._eval_w_h3
            b_h3 = self._eval_b_h3
            w_e = self._eval_w_e
            w_agg = self._eval_w_agg
            assert isinstance(w_h3, torch.Tensor)
            assert isinstance(b_h3, torch.Tensor)
            assert isinstance(w_e, torch.Tensor)
            assert isinstance(w_agg, torch.Tensor)
            proj_i, proj_j, node_self = F.linear(
                h_clean,
                w_h3.to(device=h_clean.device, dtype=h_clean.dtype),
                b_h3.to(device=h_clean.device, dtype=h_clean.dtype),
            ).chunk(3, dim=-1)
            proj_e = F.linear(edge_clean, w_e.to(device=h_clean.device, dtype=h_clean.dtype))
            w_agg_eff = w_agg.to(device=h_clean.device, dtype=h_clean.dtype)
        else:
            w_h3 = torch.cat(
                [
                    lin0.weight[:, :c],
                    lin0.weight[:, c : 2 * c],
                    nlin0.weight[:, :c],
                ],
                dim=0,
            )
            proj_i, proj_j, node_self = F.linear(h_clean, w_h3).chunk(3, dim=-1)
            if lin0.bias is not None:
                proj_i = proj_i + lin0.bias
            if nlin0.bias is not None:
                node_self = node_self + nlin0.bias
            proj_e = F.linear(edge_clean, lin0.weight[:, 2 * c :])
            w_agg_eff = nlin0.weight[:, c:]

        hidden_masked = F.silu(proj_i.unsqueeze(2) + proj_j.unsqueeze(1) + proj_e) * pair_mask_4d
        hidden_agg = hidden_masked.sum(dim=2) * inv_denom
        agg = F.linear(hidden_agg, lin2.weight)
        if lin2.bias is not None:
            agg = agg + bias_scale * lin2.bias

        node_h = F.silu(node_self + F.linear(agg, w_agg_eff))
        node_upd = F.linear(node_h, nlin2.weight, nlin2.bias)
        norm_out = F.layer_norm(
            h_clean + node_upd,
            (self.d_model,),
            self.norm.weight,
            self.norm.bias,
            self.norm.eps,
        )
        return norm_out * active_3d


class PreNormSDPALayer(nn.Module):
    """Lock-free Pre-Norm Multi-Head Self-Attention + GELU FFN layer.

    Never mutates global sdpa_kernel flags; switches cleanly to stateless manual
    attention when require_second_order_sdpa=True for jvp(grad) and Sobolev H^1.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.norm1 = nn.LayerNorm(self.d_model)
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, 2 * self.d_model)
        self.linear2 = nn.Linear(2 * self.d_model, self.d_model)

    def forward(
        self,
        h: torch.Tensor,
        attn_bias_4d: torch.Tensor,
        m_3d: torch.Tensor,
        *,
        require_second_order_sdpa: bool = False,
    ) -> torch.Tensor:
        b_sz = h.shape[0]
        m_len = h.shape[1]
        c = self.d_model

        y = F.layer_norm(h, (c,), self.norm1.weight, self.norm1.bias, self.norm1.eps)
        qkv = F.linear(y, self.qkv_proj.weight, self.qkv_proj.bias)
        qkv = qkv.view(b_sz, m_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if require_second_order_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias_4d
            probs = torch.softmax(scores, dim=-1)
            attn = torch.matmul(probs, v)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias_4d,
                dropout_p=0.0,
                is_causal=False,
            )

        attn_cat = attn.transpose(1, 2).contiguous().view(b_sz, m_len, c)
        h = h + F.linear(attn_cat, self.out_proj.weight, self.out_proj.bias)

        y2 = F.layer_norm(h, (c,), self.norm2.weight, self.norm2.bias, self.norm2.eps)
        ff = F.linear(
            F.gelu(F.linear(y2, self.linear1.weight, self.linear1.bias)),
            self.linear2.weight,
            self.linear2.bias,
        )
        return (h + ff) * m_3d


class ParallelScheduleTransformer(nn.Module):
    """Multi-layer Pre-Norm parallel schedule Transformer without host syncs or RNNs."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_sched_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_sched_layers = int(n_sched_layers)
        self.rel_time_bias = nn.Linear(4, self.n_heads)
        self.layers = nn.ModuleList(
            [PreNormSDPALayer(d_model=self.d_model, n_heads=self.n_heads) for _ in range(self.n_sched_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        sched_mask: torch.Tensor,
        *,
        tau_obs: Optional[torch.Tensor] = None,
        all_sched_active: Optional[bool] = None,
        require_second_order_sdpa: bool = False,
    ) -> torch.Tensor:
        m_dim = x.shape[1]
        if m_dim == 0:
            return torch.zeros_like(x)
        active_sched = sched_mask > 0.5
        m_3d = active_sched.to(dtype=x.dtype).unsqueeze(-1)
        h = torch.where(active_sched.unsqueeze(-1), x, torch.zeros_like(x))

        # Allow self-attention on diagonal so empty rows (M_rem == 0) never hit all -inf softmax.
        eye_m = torch.eye(m_dim, device=x.device, dtype=torch.bool).unsqueeze(0)
        allow_attn = active_sched.unsqueeze(1) | eye_m
        zero_bias = torch.zeros((), dtype=x.dtype, device=x.device)
        # Use -1e4 instead of -1e9 so float16 softmax attention biases stay finite.
        neg_bias = torch.full((), -1e4, dtype=x.dtype, device=x.device)
        if tau_obs is not None:
            tau_clean = torch.where(active_sched, tau_obs, torch.zeros_like(tau_obs))
            dt = tau_clean.unsqueeze(1) - tau_clean.unsqueeze(2)
            abs_dt = torch.sqrt(dt * dt + 1e-4)
            rel_feat = torch.stack(
                [
                    torch.tanh(dt * 0.5),
                    torch.tanh(dt * 4.0),
                    torch.exp(-abs_dt),
                    torch.sigmoid(dt * 8.0) - 0.5,
                ],
                dim=-1,
            ).to(dtype=x.dtype)
            rel_bias = F.linear(
                rel_feat,
                self.rel_time_bias.weight.to(dtype=x.dtype),
                self.rel_time_bias.bias.to(dtype=x.dtype),
            ).permute(0, 3, 1, 2)
            attn_bias_4d = torch.where(allow_attn.unsqueeze(1), rel_bias, neg_bias)
        else:
            attn_bias_4d = torch.where(allow_attn, zero_bias, neg_bias).unsqueeze(1)

        for layer in self.layers:
            assert isinstance(layer, PreNormSDPALayer)
            h = layer(
                h,
                attn_bias_4d,
                m_3d,
                require_second_order_sdpa=require_second_order_sdpa,
            )
        return h


class MaskedEQAutocallSurrogate(nn.Module):
    """Tier 2 FiLM-conditioned Masked EGNO + Parallel Schedule SDPA residual pricing surrogate."""

    def __init__(
        self,
        d_max: int = 8,
        m_max: int = 48,
        d_model: int = 128,
        n_egno_layers: int = 3,
        n_sched_layers: int = 2,
        n_heads: int = 4,
        d_node: int = 16,
        d_edge: int = 4,
        d_sched: int = 12,
        d_global: int = 16,
        init_tau: float = 0.15,
        n_func_codes: int = 18,
        func_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.d_max = int(d_max)
        self.m_max = int(m_max)
        self.d_model = int(d_model)
        self.n_egno_layers = int(n_egno_layers)
        self.n_sched_layers = int(n_sched_layers)
        self.n_heads = int(n_heads)
        self.d_node = int(d_node)
        self.d_edge = int(d_edge)
        self.d_sched = int(d_sched)
        self.d_global = int(d_global)
        self.init_tau = float(init_tau)
        self.n_func_codes = int(n_func_codes)
        self.func_emb_dim = int(func_emb_dim)

        self.d_sched_phys = 25
        self.d_put_phys = 16
        node_scales = torch.tensor(
            [1.0, 5.0, 5.0, 15.0, 3.0, 3.0, 10.0, 2.0, 1.0, 10.0, 10.0, 0.0, 2.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
        )
        edge_scales = torch.tensor([1.0, 5.0, 1.0, 1.0], dtype=torch.float32)
        sched_scales = torch.tensor(
            [0.25, 4.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.5, 2.0, 2.0, 2.0],
            dtype=torch.float32,
        )
        global_ko_cpn_mask = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        )
        global_put_scales = torch.tensor(
            [1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0, 1.0, 10.0],
            dtype=torch.float32,
        )
        self._node_scales: torch.Tensor
        self._edge_scales: torch.Tensor
        self._sched_scales: torch.Tensor
        self._global_ko_cpn_mask: torch.Tensor
        self._global_put_scales: torch.Tensor
        self.register_buffer("_node_scales", node_scales[: self.d_node], persistent=False)
        self.register_buffer("_edge_scales", edge_scales[: self.d_edge], persistent=False)
        self.register_buffer("_sched_scales", sched_scales[: self.d_sched], persistent=False)
        self.register_buffer("_global_ko_cpn_mask", global_ko_cpn_mask[: self.d_global], persistent=False)
        self.register_buffer("_global_put_scales", global_put_scales[: self.d_global], persistent=False)

        self.node_proj = nn.Linear(self.d_node, self.d_model)
        self.node_ctx_proj = nn.Linear(self.d_global + 3 * self.func_emb_dim, self.d_model)
        self.node_in_norm = nn.LayerNorm(self.d_model)
        self.egno_layers = nn.ModuleList(
            [MaskedEGNOLayer(d_model=self.d_model, d_edge=self.d_edge) for _ in range(self.n_egno_layers)]
        )
        self.pool = OrderStatisticPooling(
            hidden_dim=self.d_model,
            out_dim=2 * self.d_model,
            init_tau=self.init_tau,
            n_func_codes=self.n_func_codes,
            func_emb_dim=self.func_emb_dim,
        )

        self.sched_proj = nn.Linear(self.d_sched + 2 * self.d_model + self.d_global, self.d_model)
        self.sched_phys_proj = nn.Linear(self.d_sched_phys, self.d_model)
        self.sched_in_norm = nn.LayerNorm(self.d_model)
        self.sched_transformer = ParallelScheduleTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_sched_layers=self.n_sched_layers,
        )

        self.period_head = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 2),
        )
        # Dedicated S_D-equivariant Put-KI subgraph keeping period_head invariant to K_PUT/B_KI.
        self.put_node_proj = nn.Linear(self.d_model + 6 + self.d_global + 4, self.d_model)
        self.put_in_norm = nn.LayerNorm(self.d_model)
        self.put_egno = MaskedEGNOLayer(d_model=self.d_model, d_edge=self.d_edge)
        self.put_pool = OrderStatisticPooling(
            hidden_dim=self.d_model,
            out_dim=self.d_model,
            init_tau=self.init_tau,
            n_func_codes=self.n_func_codes,
            func_emb_dim=self.func_emb_dim,
        )
        # Put head takes [h_sched_last, h_sched_mean, z_basket, z_put_basket, global_put, func_emb_flat, h_disp, c_call_sum, put_phys].
        self.put_head = nn.Sequential(
            nn.Linear(
                5 * self.d_model + self.d_global + 3 * self.func_emb_dim + 2 + self.d_put_phys,
                2 * self.d_model,
            ),
            nn.SiLU(),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
        )
        self._zero_res_epoch: int = 0
        self._zero_res_cache: Optional[Tuple[Tuple[int, ...], bool]] = None
        self.zero_init_residual_heads(std=1e-3)
        self.freeze_eval_buffers()
        self.register_load_state_dict_post_hook(_freeze_surrogate_buffers)  # type: ignore[no-untyped-call]

    def _normalize_feature_widths(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        sched_feats: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _fit_last(t: torch.Tensor, target: int) -> torch.Tensor:
            cur = t.shape[-1]
            if cur > target:
                return t[..., :target]
            if cur < target:
                return F.pad(t, (0, target - cur))
            return t

        return (
            _fit_last(node_feats, self.d_node),
            _fit_last(edge_feats, self.d_edge),
            _fit_last(sched_feats, self.d_sched),
            _fit_last(global_feats, self.d_global),
        )

    def freeze_eval_buffers(self) -> None:
        """Freeze pooled tau and fused EGNO eval buffers safely outside inference_mode."""
        self.pool.freeze_eval_buffers()
        self.put_pool.freeze_eval_buffers()
        self.put_egno.freeze_eval_buffers()
        for layer in self.egno_layers:
            assert isinstance(layer, MaskedEGNOLayer)
            layer.freeze_eval_buffers()

    def zero_init_residual_heads(self, std: float = 0.0) -> None:
        """Initialize final residual head layers near zero to anchor at the analytical proxy."""
        last_period_linear = self.period_head[-1]
        last_put_linear = self.put_head[-1]
        if not isinstance(last_period_linear, nn.Linear) or not isinstance(last_put_linear, nn.Linear):
            raise TypeError("Expected nn.Linear as final layer of period_head and put_head.")
        nn.init.zeros_(last_period_linear.bias)
        nn.init.zeros_(last_put_linear.bias)
        if std <= 0.0:
            nn.init.zeros_(last_period_linear.weight)
            nn.init.zeros_(last_put_linear.weight)
        else:
            nn.init.normal_(last_period_linear.weight, mean=0.0, std=std)
            nn.init.normal_(last_put_linear.weight, mean=0.0, std=std)
        self._zero_res_epoch += 1
        self._zero_res_cache = None

    def is_zero_residual(self) -> bool:
        """Return True if the final residual heads are identically zero."""
        last_period_linear = self.period_head[-1]
        last_put_linear = self.put_head[-1]
        if not isinstance(last_period_linear, nn.Linear) or not isinstance(last_put_linear, nn.Linear):
            return False
        heads = (
            last_period_linear.weight,
            last_period_linear.bias,
            last_put_linear.weight,
            last_put_linear.bias,
        )
        has_inf = any(t is not None and t.is_inference() for t in heads)
        epoch = self._zero_res_epoch
        key: Optional[Tuple[int, ...]] = None
        if not has_inf:
            cached = self._zero_res_cache
            key = (
                epoch,
                id(last_period_linear.weight),
                last_period_linear.weight._version,
                id(last_period_linear.bias) if last_period_linear.bias is not None else 0,
                last_period_linear.bias._version if last_period_linear.bias is not None else -1,
                id(last_put_linear.weight),
                last_put_linear.weight._version,
                id(last_put_linear.bias) if last_put_linear.bias is not None else 0,
                last_put_linear.bias._version if last_put_linear.bias is not None else -1,
            )
            if cached is not None and cached[0] == key:
                return cached[1]
        with torch.no_grad():
            is_zero = bool(
                (last_period_linear.weight == 0).all()
                and (last_period_linear.bias is None or (last_period_linear.bias == 0).all())
                and (last_put_linear.weight == 0).all()
                and (last_put_linear.bias is None or (last_put_linear.bias == 0).all())
            )
        if key is not None and self._zero_res_epoch == epoch:
            self._zero_res_cache = (key, is_zero)
        return is_zero

    def train(self, mode: bool = True) -> MaskedEQAutocallSurrogate:
        super().train(mode)
        if not mode:
            self.freeze_eval_buffers()
        return self

    def _forward_residuals_static(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        detach_gate_weights: bool = False,
        require_second_order_sdpa: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feats, edge_feats, sched_feats, global_feats = self._normalize_feature_widths(
            node_feats, edge_feats, sched_feats, global_feats
        )
        batch_size = node_feats.shape[0]
        d_dim = node_feats.shape[1]
        m_dim = sched_feats.shape[1]

        active_asset = asset_mask > 0.5
        m_asset = active_asset.to(dtype=node_feats.dtype)
        active_3d = active_asset.unsqueeze(-1)
        m_asset_3d = m_asset.unsqueeze(-1)

        active_sched = sched_mask > 0.5
        m_sched = active_sched.to(dtype=sched_feats.dtype)

        node_clean = torch.where(active_3d, node_feats, torch.zeros_like(node_feats))
        if self.d_node > int(AutocallNodeCol.ASSET_ACTIVE_MASK):
            col_idx = int(AutocallNodeCol.ASSET_ACTIVE_MASK)
            node_clean = torch.cat(
                [
                    node_clean[:, :, :col_idx],
                    m_asset_3d,
                    node_clean[:, :, col_idx + 1 :],
                ],
                dim=-1,
            )

        eye = torch.eye(d_dim, device=node_feats.device, dtype=node_feats.dtype).unsqueeze(0)
        pair_mask = m_asset.unsqueeze(1) * m_asset.unsqueeze(2) * (1.0 - eye)
        active_pair = pair_mask > 0.5
        edge_clean = torch.where(
            active_pair.unsqueeze(-1), edge_feats, torch.zeros_like(edge_feats)
        )
        if self.d_edge > int(AutocallEdgeCol.OFF_DIAG_MASK):
            ecol = int(AutocallEdgeCol.OFF_DIAG_MASK)
            pair_col = pair_mask.unsqueeze(-1)
            edge_clean = torch.cat(
                [
                    edge_clean[:, :, :, :ecol],
                    pair_col,
                    edge_clean[:, :, :, ecol + 1 :],
                ],
                dim=-1,
            )

        sched_clean = torch.where(
            active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats)
        )
        global_clean = global_feats
        calc_dtype = (
            torch.float32
            if node_feats.dtype in (torch.float16, torch.bfloat16)
            else node_feats.dtype
        )

        if self.d_node > int(AutocallNodeCol.PROXY_DELTA_WEIGHT):
            w_col = node_clean[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)]
        else:
            w_col = torch.ones_like(m_asset)
        if detach_gate_weights:
            w_col = w_col.detach()
        w_signed = w_col * m_asset
        w_abs = w_col.abs() * m_asset
        w_sum = w_abs.sum(dim=1, keepdim=True)
        has_w = (w_sum.squeeze(1) > 1e-6).to(dtype=w_abs.dtype)
        w_norm = w_abs / w_sum.clamp(min=1e-6)
        h_disp = has_w * (1.0 - (w_norm * w_norm).sum(dim=1)).clamp(0.0, 1.0)

        has_any_asset = active_asset.any(dim=1, keepdim=True)
        if self.d_node > int(AutocallNodeCol.DIST_TO_PUT_STRIKE) and d_dim > 0:
            dput_clean = torch.where(
                active_asset,
                2.0 * torch.asinh(0.5 * node_clean[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)]),
                torch.zeros_like(m_asset),
            ).to(dtype=calc_dtype)
            m_asset_c = m_asset.to(dtype=calc_dtype)
            w_norm_c = w_norm.to(dtype=calc_dtype)
            n_act_c = m_asset_c.sum(dim=1, keepdim=True).clamp(min=1.0)
            dput_wmean = (w_norm_c * dput_clean).sum(dim=1, keepdim=True)
            dput_mean = (m_asset_c * dput_clean).sum(dim=1, keepdim=True) / n_act_c
            alpha_p = 2.0
            z_p = alpha_p * dput_clean
            z_p_max = (
                torch.where(active_asset, z_p, torch.full_like(z_p, -1e4))
                .amax(dim=1, keepdim=True)
                .detach()
            )
            exp_pos = m_asset_c * torch.exp(z_p - z_p_max)
            lse_pos = z_p_max + torch.log(
                (exp_pos.sum(dim=1, keepdim=True) / n_act_c).clamp(min=1e-12)
            )
            dput_softmax = torch.where(has_any_asset, lse_pos / alpha_p, torch.zeros_like(lse_pos))
            nz_p = -z_p
            nz_p_max = (
                torch.where(active_asset, nz_p, torch.full_like(nz_p, -1e4))
                .amax(dim=1, keepdim=True)
                .detach()
            )
            exp_neg = m_asset_c * torch.exp(nz_p - nz_p_max)
            lse_neg = nz_p_max + torch.log(
                (exp_neg.sum(dim=1, keepdim=True) / n_act_c).clamp(min=1e-12)
            )
            dput_softmin = torch.where(has_any_asset, -lse_neg / alpha_p, torch.zeros_like(lse_neg))
            dput_stats = torch.cat(
                [dput_wmean, dput_mean, dput_softmin, dput_softmax], dim=-1
            ).to(dtype=node_feats.dtype)
        else:
            dput_stats = torch.zeros(
                (batch_size, 4), device=node_feats.device, dtype=node_feats.dtype
            )

        if self.d_global > int(AutocallGlobalCol.OPT_INDEX_ID):
            fc_tensor = (
                global_clean[:, int(AutocallGlobalCol.CALL_INDEX_ID) : int(AutocallGlobalCol.OPT_INDEX_ID) + 1]
                .detach()
                .round()
                .long()
                .clamp(0, self.n_func_codes - 1)
            )
            if (
                self.d_global > int(AutocallGlobalCol.IS_LONG_BASKET)
                and self.n_func_codes > int(MonitoringIndexCode.AVERAGE_OF)
            ):
                is_long = global_clean[:, int(AutocallGlobalCol.IS_LONG_BASKET)] > 0.5
                opt_aligned = torch.where(
                    is_long & (fc_tensor[:, 2] == int(MonitoringIndexCode.WORST_OF)),
                    torch.full_like(fc_tensor[:, 2], int(MonitoringIndexCode.AVERAGE_OF)),
                    fc_tensor[:, 2],
                )
                fc_tensor = torch.stack([fc_tensor[:, 0], fc_tensor[:, 1], opt_aligned], dim=1)
            func_codes: Optional[torch.Tensor] = fc_tensor
            emb_call = self.pool.emb_call(fc_tensor[:, 0])
            emb_cpn = self.pool.emb_cpn(fc_tensor[:, 1])
            emb_opt = self.pool.emb_opt(fc_tensor[:, 2])
            func_emb_flat = torch.cat([emb_call, emb_cpn, emb_opt], dim=-1)
            func_emb_ko_cpn = torch.cat([emb_call, emb_cpn, torch.zeros_like(emb_opt)], dim=-1)
        else:
            func_codes = None
            func_emb_flat = torch.zeros(
                (batch_size, 3 * self.func_emb_dim),
                device=global_clean.device,
                dtype=global_clean.dtype,
            )
            func_emb_ko_cpn = func_emb_flat

        global_ko_cpn = global_clean * self._global_ko_cpn_mask.to(dtype=global_clean.dtype)
        global_put_scaled = global_clean * self._global_put_scales.to(dtype=global_clean.dtype)

        node_scaled = node_clean * self._node_scales.to(dtype=node_clean.dtype)
        if self.d_node > int(AutocallNodeCol.SSVI_CURV_ETA):
            col0 = 2.0 * torch.asinh(0.5 * node_clean[:, :, 0:1])
            col8 = torch.asinh(node_clean[:, :, 8:9])
            node_scaled = torch.cat(
                [col0, node_scaled[:, :, 1:8], col8, node_scaled[:, :, 9:]],
                dim=-1,
            )
        edge_scaled = edge_clean * self._edge_scales.to(dtype=edge_clean.dtype)
        if self.d_edge > int(AutocallEdgeCol.LOG_PERF_DIFF):
            ecol2 = 2.0 * torch.asinh(0.5 * edge_clean[:, :, :, 2:3])
            edge_scaled = torch.cat(
                [edge_scaled[:, :, :, :2], ecol2, edge_scaled[:, :, :, 3:]],
                dim=-1,
            )

        pair_mask_4d = pair_mask.unsqueeze(-1)
        sum_p = pair_mask.sum(dim=2, keepdim=True)
        denom = sum_p.clamp(min=1.0)
        inv_denom = 1.0 / denom
        bias_scale = sum_p * inv_denom

        if d_dim > 0:
            node_ctx = F.linear(
                torch.cat([global_ko_cpn, func_emb_ko_cpn], dim=-1),
                self.node_ctx_proj.weight.to(dtype=node_scaled.dtype),
                self.node_ctx_proj.bias.to(dtype=node_scaled.dtype),
            ).unsqueeze(1)
            h_raw = F.linear(node_scaled, self.node_proj.weight, self.node_proj.bias) + node_ctx
            h_normed = F.layer_norm(
                h_raw,
                (self.d_model,),
                self.node_in_norm.weight.to(dtype=h_raw.dtype),
                self.node_in_norm.bias.to(dtype=h_raw.dtype),
                1e-5,
            )
            h = torch.where(active_3d, h_normed, torch.zeros_like(h_normed))
        else:
            h = F.linear(node_scaled, self.node_proj.weight, self.node_proj.bias) * m_asset_3d

        for layer in self.egno_layers:
            assert isinstance(layer, MaskedEGNOLayer)
            h = layer.forward_premasked(
                h,
                edge_scaled,
                m_asset_3d,
                pair_mask_4d,
                inv_denom,
                bias_scale,
            )

        z_basket = self.pool(
            h,
            m_asset,
            asset_weights=w_signed,
            func_codes=func_codes,
            all_active=False,
        )

        k_put_c = global_clean[:, int(AutocallGlobalCol.K_PUT)].to(dtype=calc_dtype)
        k_put_pos_c = 0.5 * (k_put_c + torch.sqrt(k_put_c * k_put_c + 1e-8))
        b_ki_c = global_clean[:, int(AutocallGlobalCol.B_KI)].to(dtype=calc_dtype)
        has_ki_c = (
            global_clean[:, int(AutocallGlobalCol.HAS_KI_FLAG)]
            .detach()
            .clamp(0.0, 1.0)
            .to(dtype=calc_dtype)
        )
        l_put_px_c = global_clean[:, int(AutocallGlobalCol.L_PUT_PROXY)].to(dtype=calc_dtype)
        if self.d_global > int(AutocallGlobalCol.LOWER_STRIKE):
            k_low_c = global_clean[:, int(AutocallGlobalCol.LOWER_STRIKE)].to(dtype=calc_dtype)
        else:
            k_low_c = torch.zeros_like(k_put_c)
        k_low_pos_c = 0.5 * (k_low_c + torch.sqrt(k_low_c * k_low_c + 1e-8))
        dk_c = k_put_pos_c - k_low_pos_c
        k_eff_pos_c = 0.5 * (dk_c + torch.sqrt(dk_c * dk_c + 1e-8))
        log_k_put_s = torch.log(torch.sqrt(k_put_pos_c * k_put_pos_c + 1e-4))
        log_b_ki_s = torch.log(torch.sqrt(b_ki_c * b_ki_c + 1e-4))
        log_k_low_s = torch.log(torch.sqrt(k_low_pos_c * k_low_pos_c + 1e-4))
        has_low_c = 1.0 - torch.exp(-k_low_pos_c * k_low_pos_c * 1e4)

        if m_dim == 0:
            delta_hazard = torch.zeros(
                (batch_size, 0), device=node_feats.device, dtype=node_feats.dtype
            )
            delta_coupon = torch.zeros(
                (batch_size, 0), device=node_feats.device, dtype=node_feats.dtype
            )
            h_sched_mean = torch.zeros(
                (batch_size, self.d_model), device=node_feats.device, dtype=node_feats.dtype
            )
            h_sched_last = torch.zeros_like(h_sched_mean)
            c_call_sum = torch.zeros(
                (batch_size, 1), device=node_feats.device, dtype=node_feats.dtype
            )
            put_phys_12 = torch.zeros(
                (batch_size, 12), device=node_feats.device, dtype=node_feats.dtype
            )
            surv_term_pred_c = torch.zeros((batch_size,), device=node_feats.device, dtype=calc_dtype)
            surv_proxy_term_c = torch.zeros((batch_size,), device=node_feats.device, dtype=calc_dtype)
            tau_last_c = torch.zeros((batch_size,), device=node_feats.device, dtype=calc_dtype)
            r_proxy_c = torch.zeros((batch_size,), device=node_feats.device, dtype=calc_dtype)
            has_sched_any_c = torch.zeros((batch_size,), device=node_feats.device, dtype=calc_dtype)
        else:
            idx_grid = torch.arange(m_dim, device=sched_feats.device).unsqueeze(0)
            active_indices = torch.where(active_sched, idx_grid, torch.zeros_like(idx_grid))
            last_pos = active_indices.argmax(dim=1, keepdim=True)
            has_sched_any = active_sched.any(dim=1, keepdim=True)
            has_sched_any_c = has_sched_any.squeeze(1).to(dtype=calc_dtype)

            tau_obs_c = sched_clean[:, :, int(AutocallSchedCol.TAU_OBS)].to(dtype=calc_dtype)
            delta_tau_c = sched_clean[:, :, int(AutocallSchedCol.DELTA_TAU)].to(dtype=calc_dtype)
            log_ko_b_c = sched_clean[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)].to(dtype=calc_dtype)
            m_sched_c = m_sched.to(dtype=calc_dtype)
            c_call_c = (
                sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0).to(dtype=calc_dtype)
                * m_sched_c
            )
            log_cpn_b_c = sched_clean[:, :, int(AutocallSchedCol.LOG_CPN_BARRIER)].to(dtype=calc_dtype)
            g_cpn_c = (
                sched_clean[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)].detach().clamp(0.0, 1.0).to(dtype=calc_dtype)
                * m_sched_c
            )
            stulz_fwd_c = sched_clean[:, :, int(AutocallSchedCol.STULZ_FWD_W)].to(dtype=calc_dtype)
            stulz_vol_c = sched_clean[:, :, int(AutocallSchedCol.STULZ_VOL_W)].to(dtype=calc_dtype)
            p_proxy_c = sched_clean[:, :, int(AutocallSchedCol.PROXY_KO_PROB)].to(dtype=calc_dtype).clamp(0.0, 1.0) * c_call_c
            q_proxy_c = sched_clean[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)].to(dtype=calc_dtype).clamp(0.0, 1.0) * m_sched_c

            cum_p_proxy_c = torch.cumsum(p_proxy_c, dim=1)
            surv_prev_proxy_c = (1.0 - (cum_p_proxy_c - p_proxy_c)).clamp(min=1e-6)
            h_proxy_c = (p_proxy_c / surv_prev_proxy_c).clamp(min=1e-6, max=1.0 - 1e-6)
            logit_h_proxy_c = torch.log(h_proxy_c / (1.0 - h_proxy_c))
            q_proxy_clamped_c = q_proxy_c.clamp(min=1e-6, max=1.0 - 1e-6)
            logit_q_proxy_c = torch.log(q_proxy_clamped_c / (1.0 - q_proxy_clamped_c))

            s_tau_c = torch.sqrt(tau_obs_c * tau_obs_c + 1e-4)
            std_w_c = torch.sqrt(stulz_vol_c * stulz_vol_c * s_tau_c + 1e-4)
            z_ko_c = c_call_c * (stulz_fwd_c - log_ko_b_c) / std_w_c * 0.35
            z_cpn_c = (1.0 - g_cpn_c) * (stulz_fwd_c - log_cpn_b_c) / std_w_c * 0.35
            z_ko_gap_c = c_call_c * (1.0 - g_cpn_c) * (log_ko_b_c - log_cpn_b_c) / std_w_c * 0.35

            df_pay_c = sched_clean[:, :, int(AutocallSchedCol.DF_PAY)].to(dtype=calc_dtype)
            cpn_rate_c = sched_clean[:, :, int(AutocallSchedCol.COUPON_RATE)].to(dtype=calc_dtype)
            sched_cnt_c = m_sched_c.sum(dim=1, keepdim=True).clamp(min=1.0)
            has_gt1_sched_c = (m_sched_c.sum(dim=1, keepdim=True) > 1.5).to(dtype=calc_dtype)
            df_mean_c = (df_pay_c * m_sched_c).sum(dim=1, keepdim=True) / sched_cnt_c
            df_mean_safe_c = torch.sqrt(df_mean_c * df_mean_c + 1e-4)
            df_rel_c = (df_pay_c / df_mean_safe_c - 1.0) * m_sched_c * has_gt1_sched_c
            is_mem_c = (
                global_clean[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)]
                .detach()
                .clamp(0.0, 1.0)
                .to(dtype=calc_dtype)
                .unsqueeze(1)
                * m_sched_c
                * has_gt1_sched_c
            )
            cmem0_raw_c = global_clean[:, int(AutocallGlobalCol.C_MEM_0)].to(dtype=calc_dtype).unsqueeze(1)
            cmem0_pos_c = 0.5 * (cmem0_raw_c + torch.sqrt(cmem0_raw_c * cmem0_raw_c + 1e-8))
            cpn_pos_c = torch.sqrt(cpn_rate_c * cpn_rate_c + 1e-5)
            mem_ratio_c = is_mem_c * torch.asinh(0.25 * cmem0_pos_c / cpn_pos_c)

            tau_last_c = torch.gather(tau_obs_c, dim=1, index=last_pos).squeeze(1) * has_sched_any_c
            tau_frac_c = (tau_obs_c / torch.sqrt(tau_last_c * tau_last_c + 1e-4).unsqueeze(1)) * m_sched_c
            p_sum_all_c = p_proxy_c.sum(dim=1, keepdim=True)
            rem_p_c = ((p_sum_all_c - cum_p_proxy_c + p_proxy_c) / surv_prev_proxy_c).clamp(min=0.0, max=1.0) * m_sched_c
            rem_steps_c = ((sched_cnt_c - torch.cumsum(m_sched_c, dim=1)) / 48.0) * m_sched_c
            w1_miss_c = (1.0 - g_cpn_c) * (1.0 - q_proxy_c) * cpn_rate_c * m_sched_c
            cum_miss_c = torch.cumsum(w1_miss_c, dim=1) - w1_miss_c
            mem_traj_c = is_mem_c * torch.asinh(0.25 * (cmem0_pos_c + cum_miss_c) / cpn_pos_c)

            sched_phys = torch.stack(
                [
                    tau_obs_c * 0.25,
                    torch.sqrt(delta_tau_c * delta_tau_c + 1e-4) * 2.0,
                    torch.log(s_tau_c + 0.1) * 0.5,
                    c_call_c,
                    g_cpn_c,
                    surv_prev_proxy_c * m_sched_c,
                    (1.0 - surv_prev_proxy_c) * m_sched_c,
                    h_proxy_c * c_call_c,
                    2.0 * torch.asinh(0.5 * (logit_h_proxy_c * 0.15)) * c_call_c,
                    q_proxy_c * (1.0 - g_cpn_c) * m_sched_c,
                    2.0 * torch.asinh(0.5 * (logit_q_proxy_c * 0.15)) * (1.0 - g_cpn_c) * m_sched_c,
                    2.0 * torch.asinh(0.5 * z_ko_c),
                    2.0 * torch.asinh(0.5 * z_cpn_c) * m_sched_c,
                    c_call_c * log_ko_b_c,
                    (1.0 - g_cpn_c) * log_cpn_b_c * m_sched_c,
                    stulz_fwd_c * 0.5 * m_sched_c,
                    stulz_vol_c * 2.0 * m_sched_c,
                    2.0 * torch.asinh(0.5 * z_ko_gap_c) * m_sched_c,
                    is_mem_c,
                    mem_ratio_c,
                    df_rel_c,
                    tau_frac_c,
                    rem_p_c,
                    rem_steps_c,
                    mem_traj_c,
                ],
                dim=-1,
            ).to(dtype=sched_clean.dtype)
            sched_phys = torch.where(
                active_sched.unsqueeze(-1), sched_phys, torch.zeros_like(sched_phys)
            )

            sched_scaled = sched_clean * self._sched_scales.to(dtype=sched_clean.dtype)
            z_exp = z_basket.unsqueeze(1).expand(batch_size, m_dim, -1)
            g_exp = global_ko_cpn.unsqueeze(1).expand(batch_size, m_dim, -1)
            sched_in = torch.cat([sched_scaled, z_exp, g_exp], dim=-1)
            sched_base = F.linear(sched_in, self.sched_proj.weight, self.sched_proj.bias)
            sched_phys_emb = F.linear(
                sched_phys,
                self.sched_phys_proj.weight.to(dtype=sched_phys.dtype),
                self.sched_phys_proj.bias.to(dtype=sched_phys.dtype),
            )
            sched_tokens_raw = F.silu(sched_base + sched_phys_emb)
            sched_tokens_normed = F.layer_norm(
                sched_tokens_raw,
                (self.d_model,),
                self.sched_in_norm.weight.to(dtype=sched_tokens_raw.dtype),
                self.sched_in_norm.bias.to(dtype=sched_tokens_raw.dtype),
                1e-5,
            )
            sched_tokens = torch.where(
                active_sched.unsqueeze(-1), sched_tokens_normed, torch.zeros_like(sched_tokens_normed)
            )
            seq_out = self.sched_transformer(
                sched_tokens,
                m_sched,
                tau_obs=sched_clean[:, :, int(AutocallSchedCol.TAU_OBS)],
                all_sched_active=False,
                require_second_order_sdpa=require_second_order_sdpa,
            )

            plin0 = self.period_head[0]
            plin2 = self.period_head[2]
            assert isinstance(plin0, nn.Linear) and isinstance(plin2, nn.Linear)
            period_feat = torch.cat([seq_out, sched_tokens], dim=-1)
            period_logits = F.linear(
                F.silu(F.linear(period_feat, plin0.weight, plin0.bias)),
                plin2.weight,
                plin2.bias,
            )
            delta_hazard = torch.where(
                active_sched,
                period_logits[:, :, 0],
                torch.zeros_like(period_logits[:, :, 0]),
            )
            delta_coupon = torch.where(
                active_sched,
                period_logits[:, :, 1],
                torch.zeros_like(period_logits[:, :, 1]),
            )

            c_call_sum = c_call_c.sum(dim=1, keepdim=True).to(dtype=sched_clean.dtype) * 0.05
            sched_count = m_sched.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_sched_mean = seq_out.sum(dim=1) / sched_count
            last_idx = last_pos.unsqueeze(-1).expand(batch_size, 1, self.d_model)
            h_sched_last = torch.gather(seq_out, dim=1, index=last_idx).squeeze(1) * has_sched_any.to(
                dtype=seq_out.dtype
            )

            hazard_act_c = c_call_c * torch.sigmoid(
                logit_h_proxy_c + delta_hazard.to(dtype=calc_dtype)
            )
            survival_act_c = torch.cumprod(1.0 - hazard_act_c, dim=1) * has_sched_any_c.unsqueeze(1)
            surv_term_pred_c = survival_act_c[:, -1]
            surv_proxy_term_c = (1.0 - p_proxy_c.sum(dim=1)).clamp(min=0.0) * has_sched_any_c
            stulz_fwd_last_c = (
                torch.gather(stulz_fwd_c, dim=1, index=last_pos).squeeze(1) * has_sched_any_c
            )
            stulz_vol_last_c = (
                torch.gather(stulz_vol_c, dim=1, index=last_pos).squeeze(1) * has_sched_any_c
            )

            denom_px_c = (k_eff_pos_c * surv_proxy_term_c).clamp(min=1e-6)
            r_proxy_c = (l_put_px_c / denom_px_c).clamp(min=1e-6, max=1.0 - 1e-6) * has_sched_any_c
            r_proxy_safe_c = r_proxy_c.clamp(min=1e-6, max=1.0 - 1e-6)
            logit_r_px_c = torch.log(r_proxy_safe_c / (1.0 - r_proxy_safe_c)) * has_sched_any_c
            std_w_last_c = torch.sqrt(
                stulz_vol_last_c * stulz_vol_last_c * torch.sqrt(tau_last_c * tau_last_c + 1e-4)
                + 1e-4
            )
            z_put_last_c = (
                2.0 * torch.asinh(0.5 * (stulz_fwd_last_c - log_k_put_s) / std_w_last_c * 0.35)
                * has_sched_any_c
            )
            z_ki_last_c = (
                2.0 * torch.asinh(0.5 * (stulz_fwd_last_c - log_b_ki_s) / std_w_last_c * 0.35)
                * has_ki_c
                * has_sched_any_c
            )
            put_phys_12 = torch.stack(
                [
                    surv_term_pred_c,
                    surv_proxy_term_c,
                    surv_term_pred_c - surv_proxy_term_c,
                    r_proxy_c,
                    2.0 * torch.asinh(0.5 * (logit_r_px_c * 0.15)),
                    k_eff_pos_c,
                    has_ki_c * b_ki_c + (1.0 - has_ki_c) * k_eff_pos_c,
                    stulz_fwd_last_c * 0.5,
                    stulz_vol_last_c * 2.0,
                    z_put_last_c,
                    z_ki_last_c,
                    tau_last_c * 0.25,
                ],
                dim=-1,
            ).to(dtype=node_feats.dtype)

        if d_dim > 0:
            log_m_i = node_clean[:, :, int(AutocallNodeCol.LOG_MONEYNESS)].to(dtype=calc_dtype)
            if self.d_node > int(AutocallNodeCol.IV_TERM_ATM):
                iv_term_i = node_clean[:, :, int(AutocallNodeCol.IV_TERM_ATM)].to(dtype=calc_dtype)
            else:
                iv_term_i = node_clean[:, :, int(AutocallNodeCol.IV_SHORT_ATM)].to(dtype=calc_dtype)
            if self.d_node > int(AutocallNodeCol.DIST_TO_PUT_STRIKE):
                dist_put_i = node_clean[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)].to(dtype=calc_dtype)
            else:
                dist_put_i = log_m_i - log_k_put_s.unsqueeze(1)
            s_tau_last_c = torch.sqrt(tau_last_c * tau_last_c + 1e-4).unsqueeze(1)
            vol_tau_i = torch.sqrt(iv_term_i * iv_term_i * s_tau_last_c + 1e-4)
            d_put_node_i = 2.0 * torch.asinh(0.5 * dist_put_i)
            z_put_node_i = (
                2.0 * torch.asinh(0.5 * (dist_put_i / vol_tau_i * 0.5)) * has_sched_any_c.unsqueeze(1)
            )
            ki_diff_i = log_m_i - log_b_ki_s.unsqueeze(1)
            d_ki_node_i = has_ki_c.unsqueeze(1) * 2.0 * torch.asinh(0.5 * ki_diff_i)
            z_ki_node_i = (
                has_ki_c.unsqueeze(1)
                * 2.0
                * torch.asinh(0.5 * (ki_diff_i / vol_tau_i * 0.5))
                * has_sched_any_c.unsqueeze(1)
            )
            d_low_node_i = has_low_c.unsqueeze(1) * 2.0 * torch.asinh(0.5 * (log_m_i - log_k_low_s.unsqueeze(1)))
            vol_tau_feat_i = vol_tau_i * has_sched_any_c.unsqueeze(1)
            put_node_6 = torch.where(
                active_3d,
                torch.stack(
                    [
                        d_put_node_i,
                        z_put_node_i,
                        d_ki_node_i,
                        z_ki_node_i,
                        d_low_node_i,
                        vol_tau_feat_i,
                    ],
                    dim=-1,
                ).to(dtype=h.dtype),
                torch.zeros((batch_size, d_dim, 6), device=h.device, dtype=h.dtype),
            )
            put_ctx_s = torch.stack(
                [surv_term_pred_c, surv_proxy_term_c, tau_last_c * 0.25, r_proxy_c], dim=-1
            ).to(dtype=h.dtype)
            put_ctx_exp = torch.cat([global_put_scaled, put_ctx_s], dim=-1).unsqueeze(1).expand(batch_size, d_dim, -1)
            h_put_in = torch.cat([h, put_node_6, put_ctx_exp], dim=-1)
            h_put_raw = F.silu(
                F.linear(
                    h_put_in,
                    self.put_node_proj.weight.to(dtype=h.dtype),
                    self.put_node_proj.bias.to(dtype=h.dtype),
                )
            )
            h_put_normed = F.layer_norm(
                h_put_raw,
                (self.d_model,),
                self.put_in_norm.weight.to(dtype=h.dtype),
                self.put_in_norm.bias.to(dtype=h.dtype),
                1e-5,
            )
            h_put = torch.where(active_3d, h_put_normed, torch.zeros_like(h_put_normed))
            h_put = self.put_egno.forward_premasked(
                h_put,
                edge_scaled,
                m_asset_3d,
                pair_mask_4d,
                inv_denom,
                bias_scale,
            )
            z_put_basket = self.put_pool(
                h_put,
                m_asset,
                asset_weights=w_signed,
                func_codes=func_codes,
                all_active=False,
            )
        else:
            z_put_basket = torch.zeros(
                (batch_size, self.d_model), device=node_feats.device, dtype=node_feats.dtype
            )

        put_in = torch.cat(
            [
                h_sched_last,
                h_sched_mean,
                z_basket,
                z_put_basket,
                global_put_scaled,
                func_emb_flat,
                h_disp.unsqueeze(-1),
                c_call_sum,
                put_phys_12,
                dput_stats,
            ],
            dim=-1,
        )
        put_lin0 = self.put_head[0]
        put_lin2 = self.put_head[2]
        put_lin4 = self.put_head[4]
        assert (
            isinstance(put_lin0, nn.Linear)
            and isinstance(put_lin2, nn.Linear)
            and isinstance(put_lin4, nn.Linear)
        )
        put_h = F.silu(F.linear(put_in, put_lin0.weight, put_lin0.bias))
        put_h = F.silu(F.linear(put_h, put_lin2.weight, put_lin2.bias))
        delta_put_loss = F.linear(put_h, put_lin4.weight, put_lin4.bias).squeeze(-1)

        return delta_hazard, delta_coupon, delta_put_loss

    def forward_residuals(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        require_second_order_sdpa: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static-shape, zero-host-sync, pure torch.where-masked residual forward pass."""
        return self._forward_residuals_static(
            node_feats,
            edge_feats,
            asset_mask,
            sched_feats,
            sched_mask,
            global_feats,
            detach_gate_weights=False,
            require_second_order_sdpa=require_second_order_sdpa,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        detach_gate_weights: bool = False,
        return_residuals_only: bool = False,
        all_active: bool = False,
        require_second_order_sdpa: Optional[bool] = None,
    ) -> AutocallSurrogateOutput:
        eff_second_order_sdpa = (
            torch.is_grad_enabled()
            if require_second_order_sdpa is None
            else bool(require_second_order_sdpa)
        )
        batch_size, d_dim, _ = node_feats.shape
        _, m_dim, _ = sched_feats.shape

        node_in, edge_in, sched_in, global_in = self._normalize_feature_widths(
            node_feats, edge_feats, sched_feats, global_feats
        )

        # Skip host-syncing trailing-mask checks on CUDA tensors to preserve non-blocking/CUDAGraph capture.
        if (
            not all_active
            and not node_feats.is_cuda
            and batch_size > 0
            and d_dim > 0
            and m_dim > 0
            and not torch.compiler.is_compiling()
        ):
            active_asset_full = asset_mask > 0.5
            active_sched_full = sched_mask > 0.5
            if d_dim > 1 and not bool(active_asset_full[:, -1].all()):
                any_asset = active_asset_full.any(dim=0)
                d_act = int(any_asset.nonzero()[-1].item()) + 1 if any_asset.any() else 1
            else:
                d_act = d_dim
            if m_dim > 1 and not bool(active_sched_full[:, -1].all()):
                any_sched = active_sched_full.any(dim=0)
                m_act = int(any_sched.nonzero()[-1].item()) + 1 if any_sched.any() else 1
            else:
                m_act = m_dim
            node_sub = node_in[:, :d_act, :]
            edge_sub = edge_in[:, :d_act, :d_act, :]
            amask_sub = asset_mask[:, :d_act]
            sched_sub = sched_in[:, :m_act, :]
            smask_sub = sched_mask[:, :m_act]
        else:
            m_act = m_dim
            node_sub = node_in
            edge_sub = edge_in
            amask_sub = asset_mask
            sched_sub = sched_in
            smask_sub = sched_mask

        delta_hazard_act, delta_coupon_act, delta_put_loss = self._forward_residuals_static(
            node_sub,
            edge_sub,
            amask_sub,
            sched_sub,
            smask_sub,
            global_in,
            detach_gate_weights=detach_gate_weights,
            require_second_order_sdpa=eff_second_order_sdpa,
        )

        if return_residuals_only:
            empty_p = delta_hazard_act.detach()
            empty_s = delta_put_loss.detach()
            return AutocallSurrogateOutput(
                stopping_probs=empty_p,
                survival_probs=empty_p,
                hazard_rates=empty_p,
                coupon_probs=empty_p,
                put_loss=empty_s,
                delta_hazard_logits=delta_hazard_act,
                delta_coupon_logits=delta_coupon_act,
                delta_put_loss=delta_put_loss,
            )

        if m_act == 0:
            empty_m = torch.zeros(
                (batch_size, 0), device=sched_feats.device, dtype=sched_feats.dtype
            )
            return AutocallSurrogateOutput(
                stopping_probs=empty_m,
                survival_probs=empty_m,
                hazard_rates=empty_m,
                coupon_probs=empty_m,
                put_loss=torch.zeros_like(delta_put_loss),
                delta_hazard_logits=delta_hazard_act,
                delta_coupon_logits=delta_coupon_act,
                delta_put_loss=delta_put_loss,
            )

        # Upcast reduced-precision dtypes to float32 so 1e-10/1e-6 clamping does not underflow or round to 1.0 in float16.
        out_dtype = sched_feats.dtype
        calc_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
        active_sched = smask_sub > 0.5
        has_sched_any = active_sched.any(dim=1)
        has_sched_any_f = has_sched_any.to(dtype=calc_dtype)
        m_sched = active_sched.to(dtype=calc_dtype)
        sched_clean = torch.where(
            active_sched.unsqueeze(-1),
            sched_sub.to(dtype=calc_dtype),
            torch.zeros((), dtype=calc_dtype, device=sched_sub.device),
        )
        c_call = (
            sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0)
            * m_sched
        )
        g_cpn = (
            sched_clean[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)].detach().clamp(0.0, 1.0)
            * m_sched
        )
        p_proxy = (
            sched_clean[:, :, int(AutocallSchedCol.PROXY_KO_PROB)].clamp(0.0, 1.0)
            * c_call
        )
        q_proxy = (
            sched_clean[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)].clamp(0.0, 1.0)
            * m_sched
        )

        cum_p_proxy = torch.cumsum(p_proxy, dim=1)
        surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy)).clamp(min=1e-6)
        h_proxy = (p_proxy / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
        logit_h_proxy = torch.log(h_proxy / (1.0 - h_proxy))

        hazard_act = (
            c_call * m_sched * torch.sigmoid(logit_h_proxy + delta_hazard_act.to(dtype=calc_dtype))
        )
        survival_act = torch.cumprod(1.0 - hazard_act, dim=1) * has_sched_any_f.unsqueeze(1)
        surv_prev = torch.cat(
            [torch.ones_like(survival_act[:, :1]), survival_act[:, :-1]], dim=1
        )
        stopping_act = surv_prev * hazard_act

        q_proxy_clamped = q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
        logit_q_proxy = torch.log(q_proxy_clamped / (1.0 - q_proxy_clamped))
        # Softened 1e-14 gate avoids 5x attenuation on seasoned deep-ITM deals where q_proxy ~ 1e-6..5e-6.
        q_gate = (q_proxy * q_proxy) / (q_proxy * q_proxy + 1e-14)
        q_pred = q_gate * torch.sigmoid(logit_q_proxy + delta_coupon_act.to(dtype=calc_dtype))
        coupon_act = m_sched * (g_cpn + (1.0 - g_cpn) * q_pred)

        # C-infinity logit-space put-spread head in float64 to avoid float32 1 - eps_r rounding.
        is_diff_opt = _resolve_is_diff_opt(global_in)
        k_put_raw_64 = global_in[:, int(AutocallGlobalCol.K_PUT)].to(torch.float64)
        k_put_64 = torch.where(is_diff_opt, k_put_raw_64, k_put_raw_64.clamp(min=0.0))
        if self.d_global > int(AutocallGlobalCol.LOWER_STRIKE):
            k_low_64 = (
                global_in[:, int(AutocallGlobalCol.LOWER_STRIKE)]
                .to(torch.float64)
                .clamp(min=0.0)
            )
        else:
            k_low_64 = torch.zeros_like(k_put_64)
        k_eff_64 = torch.clamp(k_put_64 - k_low_64, min=0.0)
        l_put_proxy_64 = (
            global_in[:, int(AutocallGlobalCol.L_PUT_PROXY)].to(torch.float64).clamp(min=0.0)
        )
        surv_proxy_term_64 = (1.0 - p_proxy.to(torch.float64).sum(dim=1)).clamp(min=0.0)
        denom_64 = torch.clamp(k_eff_64 * surv_proxy_term_64, min=1e-12)
        eps_r = 1e-12
        r_proxy_64 = torch.clamp(l_put_proxy_64 / denom_64, min=eps_r, max=1.0 - eps_r)
        logit_r_64 = torch.log(r_proxy_64 / (1.0 - r_proxy_64))
        surv_term_64 = survival_act[:, -1].to(torch.float64)
        u_put_64 = k_eff_64 * surv_term_64
        std_put_loss_64 = u_put_64 * torch.sigmoid(logit_r_64 + delta_put_loss.to(torch.float64))

        # Diff payoffs allow negative strikes and losses above K*S, so parameterize in log-softplus space.
        base_diff_64 = torch.where(
            surv_proxy_term_64 > 1e-8,
            l_put_proxy_64 * surv_term_64 / surv_proxy_term_64.clamp(min=1e-8),
            l_put_proxy_64,
        )
        inv_sp_64 = torch.where(
            base_diff_64 > 20.0,
            base_diff_64,
            torch.log(torch.expm1(base_diff_64.clamp(min=1e-12, max=20.0))),
        )
        diff_put_loss_64 = F.softplus(inv_sp_64 + delta_put_loss.to(torch.float64))
        put_loss_64 = torch.where(is_diff_opt, diff_put_loss_64, std_put_loss_64)
        put_loss_64 = torch.where(has_sched_any, put_loss_64, torch.zeros_like(put_loss_64))
        put_loss = put_loss_64.to(dtype=out_dtype)

        stopping_act = stopping_act.to(dtype=out_dtype)
        survival_act = survival_act.to(dtype=out_dtype)
        hazard_act = hazard_act.to(dtype=out_dtype)
        coupon_act = coupon_act.to(dtype=out_dtype)

        if m_act < m_dim:
            pad_len = m_dim - m_act
            z_pad = torch.zeros(
                (batch_size, pad_len), device=sched_feats.device, dtype=sched_feats.dtype
            )
            stopping_probs = torch.cat([stopping_act, z_pad], dim=1)
            hazard_rates = torch.cat([hazard_act, z_pad], dim=1)
            coupon_probs = torch.cat([coupon_act, z_pad], dim=1)
            delta_hazard_logits = torch.cat([delta_hazard_act, z_pad], dim=1)
            delta_coupon_logits = torch.cat([delta_coupon_act, z_pad], dim=1)
            survival_probs = torch.cat(
                [survival_act, survival_act[:, -1:].expand(batch_size, pad_len)], dim=1
            )
        else:
            stopping_probs = stopping_act
            survival_probs = survival_act
            hazard_rates = hazard_act
            coupon_probs = coupon_act
            delta_hazard_logits = delta_hazard_act
            delta_coupon_logits = delta_coupon_act

        return AutocallSurrogateOutput(
            stopping_probs=stopping_probs,
            survival_probs=survival_probs,
            hazard_rates=hazard_rates,
            coupon_probs=coupon_probs,
            put_loss=put_loss,
            delta_hazard_logits=delta_hazard_logits,
            delta_coupon_logits=delta_coupon_logits,
            delta_put_loss=delta_put_loss,
        )

    def assemble_product_value(
        self,
        out: AutocallSurrogateOutput,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        no_notional_repayment: bool | torch.Tensor = False,
        autocall_coupon: float | torch.Tensor = 0.0,
        force_coupons_on_autocall: bool | torch.Tensor = False,
        is_long_basket: Optional[bool | torch.Tensor] = None,
        pv_in_transit: float | torch.Tensor = 0.0,
        funding_pv: float | torch.Tensor = 0.0,
        already_autocalled: bool | torch.Tensor = False,
        w_fund_per_period: Optional[torch.Tensor] = None,
        lower_strike: Optional[float | torch.Tensor] = None,
        one_star_pv: float | torch.Tensor = 0.0,
        part_call_pv: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        return assemble_surrogate_product_value(
            out=out,
            sched_feats=sched_feats,
            sched_mask=sched_mask,
            global_feats=global_feats,
            no_notional_repayment=no_notional_repayment,
            autocall_coupon=autocall_coupon,
            force_coupons_on_autocall=force_coupons_on_autocall,
            is_long_basket=is_long_basket,
            pv_in_transit=pv_in_transit,
            funding_pv=funding_pv,
            already_autocalled=already_autocalled,
            w_fund_per_period=w_fund_per_period,
            lower_strike=lower_strike,
            one_star_pv=one_star_pv,
            part_call_pv=part_call_pv,
        )


def assemble_surrogate_components(
    out: AutocallSurrogateOutput,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
    *,
    no_notional_repayment: bool | torch.Tensor = False,
    autocall_coupon: float | torch.Tensor = 0.0,
    force_coupons_on_autocall: bool | torch.Tensor = False,
    is_long_basket: Optional[bool | torch.Tensor] = None,
    pv_in_transit: float | torch.Tensor = 0.0,
    funding_pv: float | torch.Tensor = 0.0,
    already_autocalled: bool | torch.Tensor = False,
    w_fund_per_period: Optional[torch.Tensor] = None,
    lower_strike: Optional[float | torch.Tensor] = None,
    one_star_pv: float | torch.Tensor = 0.0,
    part_call_pv: float | torch.Tensor = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (ProductValue, NotionalPV, CouponPV, OptionPV) via W1 stopping-time decomposition."""
    batch_size, m_dim, _ = sched_feats.shape
    dtype = sched_feats.dtype
    device = sched_feats.device

    pv_transit_t = _normalize_batch_vec(pv_in_transit, batch_size, dtype, device)
    zeros_b = torch.zeros_like(pv_transit_t)
    if m_dim == 0:
        return pv_transit_t, zeros_b, zeros_b, zeros_b

    if is_long_basket is None:
        if global_feats.shape[-1] > int(AutocallGlobalCol.IS_LONG_BASKET):
            long_eff: object = global_feats[:, int(AutocallGlobalCol.IS_LONG_BASKET)] > 0.5
        else:
            long_eff = False
    else:
        long_eff = is_long_basket

    if lower_strike is None:
        if global_feats.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE):
            low_k_eff: Optional[torch.Tensor] = global_feats[:, int(AutocallGlobalCol.LOWER_STRIKE)]
        else:
            low_k_eff = None
    else:
        low_k_eff = _normalize_batch_vec(lower_strike, batch_size, dtype, device)

    active_sched = sched_mask > 0.5
    m_sched = active_sched.to(dtype=dtype)
    has_any = active_sched.any(dim=1)
    if (
        m_dim > 4
        and batch_size > 0
        and not sched_feats.is_cuda
        and not torch.compiler.is_compiling()
    ):
        any_sched_col = active_sched.any(dim=0)
        m_act = int(any_sched_col.nonzero()[-1].item()) + 1 if any_sched_col.any() else 1
    else:
        m_act = m_dim

    sched_clean = torch.where(
        active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats)
    )
    df_pay = sched_clean[:, :, int(AutocallSchedCol.DF_PAY)]
    cpn_rate = sched_clean[:, :, int(AutocallSchedCol.COUPON_RATE)]

    is_diff_opt = _resolve_is_diff_opt(global_feats)
    k_put_raw = global_feats[:, int(AutocallGlobalCol.K_PUT)]
    k_put = torch.where(is_diff_opt, k_put_raw, k_put_raw.clamp(min=0.0))
    g_put = global_feats[:, int(AutocallGlobalCol.G_PUT)]
    c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
    is_memory = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)

    idx_grid = torch.arange(m_dim, device=device).unsqueeze(0)
    active_indices = torch.where(active_sched, idx_grid, torch.zeros_like(idx_grid))
    last_active_idx = active_indices.argmax(dim=1, keepdim=True)
    df_term = torch.gather(df_pay, dim=1, index=last_active_idx).squeeze(1)
    df_term = torch.where(has_any, df_term, torch.zeros_like(df_term))

    surv_term = torch.gather(out.survival_probs, dim=1, index=last_active_idx).squeeze(1)
    no_notional_t = _normalize_batch_vec(no_notional_repayment, batch_size, dtype, device)
    pv_notional = (1.0 - no_notional_t) * (
        (df_pay * out.stopping_probs * m_sched).sum(dim=1) + df_term * surv_term
    )

    surv_prev = torch.cat(
        [torch.ones_like(out.survival_probs[:, :1]), out.survival_probs[:, :-1]], dim=1
    )
    mem_buf = c_mem_0 * is_memory
    pv_coupon = torch.zeros_like(pv_notional)
    ac_cpn_t = _normalize_batch_vec(autocall_coupon, batch_size, dtype, device)
    force_cpn_t = _normalize_batch_vec(force_coupons_on_autocall, batch_size, dtype, device)

    for m in range(m_act):
        c_eff_m = cpn_rate[:, m] + mem_buf
        q_m = out.coupon_probs[:, m]
        if isinstance(force_coupons_on_autocall, bool) and not force_coupons_on_autocall:
            eff_q_m = q_m
        else:
            q_forced = (
                out.stopping_probs[:, m].float() / surv_prev[:, m].float().clamp(min=1e-12)
            ).to(dtype=dtype)
            q_forced = torch.maximum(q_m, q_forced).clamp(0.0, 1.0)
            eff_q_m = q_m + force_cpn_t * (q_forced - q_m)

        cf_cpn_m = (
            df_pay[:, m]
            * (surv_prev[:, m] * eff_q_m * c_eff_m + out.stopping_probs[:, m] * ac_cpn_t)
            * m_sched[:, m]
        )
        pv_coupon = pv_coupon + cf_cpn_m
        mem_buf = torch.where(
            active_sched[:, m],
            is_memory * (1.0 - eff_q_m) * c_eff_m,
            mem_buf,
        )

    eff_put_loss = out.put_loss
    if low_k_eff is not None:
        k_eff = torch.clamp(k_put - torch.clamp(low_k_eff, min=0.0), min=0.0)
        eff_put_loss = torch.where(
            is_diff_opt,
            eff_put_loss,
            torch.minimum(eff_put_loss, k_eff * surv_term),
        )

    long_t = _normalize_batch_vec(long_eff, batch_size, dtype, device)
    # Keep unconditional put sign so long/short basket exposure stays governed by feature/payoff semantics.
    put_sign_t = -torch.ones_like(long_t, dtype=dtype, device=device)
    pv_option = put_sign_t * df_term * g_put * eff_put_loss
    pv_option = (
        pv_option
        + _normalize_batch_vec(one_star_pv, batch_size, dtype, device)
        + _normalize_batch_vec(part_call_pv, batch_size, dtype, device)
    )

    fund_t = _normalize_batch_vec(funding_pv, batch_size, dtype, device)
    if w_fund_per_period is not None:
        w_f = torch.as_tensor(w_fund_per_period, dtype=dtype, device=device)
        if w_f.ndim == 1:
            w_f = w_f.unsqueeze(0)
        w_len = min(m_act, int(w_f.shape[1]))
        period_fund_pv = (surv_prev[:, :w_len] * w_f[:, :w_len] * m_sched[:, :w_len]).sum(dim=1)
        fund_t = fund_t + period_fund_pv

    active_pv = pv_transit_t + pv_notional + pv_coupon + pv_option - fund_t
    autocalled_t = _normalize_batch_vec(already_autocalled, batch_size, torch.bool, device)
    is_live = has_any & (~autocalled_t)
    res_pv: torch.Tensor = torch.where(is_live, active_pv, pv_transit_t)
    res_not = torch.where(is_live, pv_notional, zeros_b)
    res_cpn = torch.where(is_live, pv_coupon, zeros_b)
    res_opt = torch.where(is_live, pv_option, zeros_b)
    return res_pv, res_not, res_cpn, res_opt


def assemble_surrogate_product_value(
    out: AutocallSurrogateOutput,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
    *,
    no_notional_repayment: bool | torch.Tensor = False,
    autocall_coupon: float | torch.Tensor = 0.0,
    force_coupons_on_autocall: bool | torch.Tensor = False,
    is_long_basket: Optional[bool | torch.Tensor] = None,
    pv_in_transit: float | torch.Tensor = 0.0,
    funding_pv: float | torch.Tensor = 0.0,
    already_autocalled: bool | torch.Tensor = False,
    w_fund_per_period: Optional[torch.Tensor] = None,
    lower_strike: Optional[float | torch.Tensor] = None,
    one_star_pv: float | torch.Tensor = 0.0,
    part_call_pv: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Assemble total ProductValue via the W1 affine stopping-time decomposition."""
    res_pv, _, _, _ = assemble_surrogate_components(
        out=out,
        sched_feats=sched_feats,
        sched_mask=sched_mask,
        global_feats=global_feats,
        no_notional_repayment=no_notional_repayment,
        autocall_coupon=autocall_coupon,
        force_coupons_on_autocall=force_coupons_on_autocall,
        is_long_basket=is_long_basket,
        pv_in_transit=pv_in_transit,
        funding_pv=funding_pv,
        already_autocalled=already_autocalled,
        w_fund_per_period=w_fund_per_period,
        lower_strike=lower_strike,
        one_star_pv=one_star_pv,
        part_call_pv=part_call_pv,
    )
    return res_pv


__all__ = [
    "AutocallEdgeCol",
    "AutocallGlobalCol",
    "AutocallNodeCol",
    "AutocallSchedCol",
    "AutocallSurrogateOutput",
    "MaskedEGNOLayer",
    "MaskedEQAutocallSurrogate",
    "MonitoringIndexCode",
    "ParallelScheduleTransformer",
    "PreNormSDPALayer",
    "assemble_surrogate_components",
    "assemble_surrogate_product_value",
]

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
        module._zero_res_cache_key = None


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

            buf_w_h3 = self._eval_w_h3
            buf_b_h3 = self._eval_b_h3
            buf_w_e = self._eval_w_e
            buf_w_agg = self._eval_w_agg
            assert isinstance(buf_w_h3, torch.Tensor)
            assert isinstance(buf_b_h3, torch.Tensor)
            assert isinstance(buf_w_e, torch.Tensor)
            assert isinstance(buf_w_agg, torch.Tensor)
            buf_w_h3.copy_(w_h3.to(device=buf_w_h3.device, dtype=buf_w_h3.dtype))
            buf_b_h3.copy_(b_h3.to(device=buf_b_h3.device, dtype=buf_b_h3.dtype))
            buf_w_e.copy_(w_e.to(device=buf_w_e.device, dtype=buf_w_e.dtype))
            buf_w_agg.copy_(w_agg.to(device=buf_w_agg.device, dtype=buf_w_agg.dtype))

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
        self.layers = nn.ModuleList(
            [PreNormSDPALayer(d_model=self.d_model, n_heads=self.n_heads) for _ in range(self.n_sched_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        sched_mask: torch.Tensor,
        *,
        all_sched_active: Optional[bool] = None,
        require_second_order_sdpa: bool = False,
    ) -> torch.Tensor:
        m_dim = x.shape[1]
        active_sched = sched_mask > 0.5
        m_3d = active_sched.to(dtype=x.dtype).unsqueeze(-1)
        h = torch.where(active_sched.unsqueeze(-1), x, torch.zeros_like(x))

        # Allow self-attention on diagonal so empty rows (M_rem == 0) never hit all -inf softmax.
        eye_m = torch.eye(m_dim, device=x.device, dtype=torch.bool).unsqueeze(0)
        allow_attn = active_sched.unsqueeze(1) | eye_m
        zero_bias = torch.zeros((), dtype=x.dtype, device=x.device)
        neg_bias = torch.full((), -1e9, dtype=x.dtype, device=x.device)
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
        n_func_codes: int = 15,
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

        self.node_proj = nn.Linear(self.d_node, self.d_model)
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
        self.sched_transformer = ParallelScheduleTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_sched_layers=self.n_sched_layers,
        )

        self.period_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 2),
        )
        # Put head takes [h_sched_last, h_sched_mean, z_basket, global_scaled, func_emb_flat, h_disp, sum(c_call)].
        self.put_head = nn.Sequential(
            nn.Linear(4 * self.d_model + self.d_global + 3 * self.func_emb_dim + 2, 2 * self.d_model),
            nn.SiLU(),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
        )
        self._zero_res_cache_key: Optional[Tuple[int, int, int, int, int, int, int, int]] = None
        self._zero_res_cache_val: bool = False
        self.zero_init_residual_heads(std=1e-3)
        self.freeze_eval_buffers()
        self.register_load_state_dict_post_hook(_freeze_surrogate_buffers)  # type: ignore[no-untyped-call]

    def freeze_eval_buffers(self) -> None:
        """Freeze pooled tau and fused EGNO eval buffers safely outside inference_mode."""
        self.pool.freeze_eval_buffers()
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
        self._zero_res_cache_key = None

    def is_zero_residual(self) -> bool:
        """Return True if the final residual heads are identically zero."""
        last_period_linear = self.period_head[-1]
        last_put_linear = self.put_head[-1]
        if not isinstance(last_period_linear, nn.Linear) or not isinstance(last_put_linear, nn.Linear):
            return False
        key = (
            id(last_period_linear.weight),
            last_period_linear.weight._version,
            id(last_period_linear.bias) if last_period_linear.bias is not None else 0,
            last_period_linear.bias._version if last_period_linear.bias is not None else -1,
            id(last_put_linear.weight),
            last_put_linear.weight._version,
            id(last_put_linear.bias) if last_put_linear.bias is not None else 0,
            last_put_linear.bias._version if last_put_linear.bias is not None else -1,
        )
        if self._zero_res_cache_key == key:
            return self._zero_res_cache_val
        with torch.no_grad():
            is_zero = bool(
                (last_period_linear.weight == 0).all()
                and (last_period_linear.bias is None or (last_period_linear.bias == 0).all())
                and (last_put_linear.weight == 0).all()
                and (last_put_linear.bias is None or (last_put_linear.bias == 0).all())
            )
        self._zero_res_cache_key = key
        self._zero_res_cache_val = is_zero
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

        w_col = node_clean[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)]
        if detach_gate_weights:
            w_col = w_col.detach()
        w_signed = w_col * m_asset
        w_abs = w_col.abs() * m_asset
        w_sum = w_abs.sum(dim=1, keepdim=True)
        has_w = (w_sum.squeeze(1) > 1e-6).to(dtype=w_abs.dtype)
        w_norm = w_abs / w_sum.clamp(min=1e-6)
        h_disp = has_w * (1.0 - (w_norm * w_norm).sum(dim=1)).clamp(0.0, 1.0)

        pair_mask_4d = pair_mask.unsqueeze(-1)
        sum_p = pair_mask.sum(dim=2, keepdim=True)
        denom = sum_p.clamp(min=1.0)
        inv_denom = 1.0 / denom
        bias_scale = sum_p * inv_denom

        h = F.linear(node_clean, self.node_proj.weight, self.node_proj.bias) * m_asset_3d
        for layer in self.egno_layers:
            assert isinstance(layer, MaskedEGNOLayer)
            h = layer.forward_premasked(
                h,
                edge_clean,
                m_asset_3d,
                pair_mask_4d,
                inv_denom,
                bias_scale,
            )

        if self.d_global > int(AutocallGlobalCol.OPT_INDEX_ID):
            fc_tensor = (
                global_clean[:, int(AutocallGlobalCol.CALL_INDEX_ID) : int(AutocallGlobalCol.OPT_INDEX_ID) + 1]
                .detach()
                .round()
                .long()
                .clamp(0, self.n_func_codes - 1)
            )
            func_codes: Optional[torch.Tensor] = fc_tensor
            func_emb_flat = torch.cat(
                [
                    self.pool.emb_call(fc_tensor[:, 0]),
                    self.pool.emb_cpn(fc_tensor[:, 1]),
                    self.pool.emb_opt(fc_tensor[:, 2]),
                ],
                dim=-1,
            )
            c_idx_s = int(AutocallGlobalCol.CALL_INDEX_ID)
            o_idx_e = int(AutocallGlobalCol.OPT_INDEX_ID) + 1
            global_scaled = torch.cat(
                [
                    global_clean[:, :c_idx_s],
                    torch.zeros_like(global_clean[:, c_idx_s:o_idx_e]),
                    global_clean[:, o_idx_e:],
                ],
                dim=-1,
            )
        else:
            func_codes = None
            func_emb_flat = torch.zeros(
                (batch_size, 3 * self.func_emb_dim),
                device=global_clean.device,
                dtype=global_clean.dtype,
            )
            global_scaled = global_clean

        z_basket = self.pool(
            h,
            m_asset,
            asset_weights=w_signed,
            func_codes=func_codes,
            all_active=False,
        )

        z_exp = z_basket.unsqueeze(1).expand(batch_size, m_dim, -1)
        g_exp = global_scaled.unsqueeze(1).expand(batch_size, m_dim, -1)
        sched_in = torch.cat([sched_clean, z_exp, g_exp], dim=-1)
        sched_tokens = F.silu(F.linear(sched_in, self.sched_proj.weight, self.sched_proj.bias))
        sched_tokens = torch.where(
            active_sched.unsqueeze(-1), sched_tokens, torch.zeros_like(sched_tokens)
        )
        seq_out = self.sched_transformer(
            sched_tokens,
            m_sched,
            all_sched_active=False,
            require_second_order_sdpa=require_second_order_sdpa,
        )

        plin0 = self.period_head[0]
        plin2 = self.period_head[2]
        assert isinstance(plin0, nn.Linear) and isinstance(plin2, nn.Linear)
        period_logits = F.linear(
            F.silu(F.linear(seq_out, plin0.weight, plin0.bias)),
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

        c_call = (
            sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0)
            * m_sched
        )
        sched_count = m_sched.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_sched_mean = seq_out.sum(dim=1) / sched_count
        last_idx = (sched_count.long() - 1).clamp(min=0, max=m_dim - 1).unsqueeze(-1).expand(
            batch_size, 1, self.d_model
        )
        h_sched_last = torch.gather(seq_out, dim=1, index=last_idx).squeeze(1)

        put_in = torch.cat(
            [
                h_sched_last,
                h_sched_mean,
                z_basket,
                global_scaled,
                func_emb_flat,
                h_disp.unsqueeze(-1),
                c_call.sum(dim=1, keepdim=True),
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
        batch_size, d_dim, n_feat = node_feats.shape
        _, m_dim, _ = sched_feats.shape

        node_in = (
            F.pad(node_feats, (0, self.d_node - n_feat))
            if n_feat < self.d_node
            else node_feats
        )
        g_feat = global_feats.shape[-1]
        global_in = (
            F.pad(global_feats, (0, self.d_global - g_feat))
            if g_feat < self.d_global
            else global_feats
        )

        if not all_active and not torch.compiler.is_compiling():
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
            edge_sub = edge_feats[:, :d_act, :d_act, :]
            amask_sub = asset_mask[:, :d_act]
            sched_sub = sched_feats[:, :m_act, :]
            smask_sub = sched_mask[:, :m_act]
        else:
            m_act = m_dim
            node_sub = node_in
            edge_sub = edge_feats
            amask_sub = asset_mask
            sched_sub = sched_feats
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

        active_sched = smask_sub > 0.5
        m_sched = active_sched.to(dtype=sched_sub.dtype)
        sched_clean = torch.where(
            active_sched.unsqueeze(-1), sched_sub, torch.zeros_like(sched_sub)
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
            c_call * m_sched * torch.sigmoid(logit_h_proxy + delta_hazard_act)
        )
        survival_act = torch.cumprod(1.0 - hazard_act, dim=1)
        surv_prev = torch.cat(
            [torch.ones_like(survival_act[:, :1]), survival_act[:, :-1]], dim=1
        )
        stopping_act = surv_prev * hazard_act

        q_proxy_clamped = q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
        logit_q_proxy = torch.log(q_proxy_clamped / (1.0 - q_proxy_clamped))
        q_gate = (q_proxy * q_proxy) / (q_proxy * q_proxy + 1e-10)
        q_pred = q_gate * torch.sigmoid(logit_q_proxy + delta_coupon_act)
        coupon_act = m_sched * (g_cpn + (1.0 - g_cpn) * q_pred)

        # C-infinity logit-space put-spread head in float64 to avoid float32 1 - eps_r rounding.
        k_put_64 = global_in[:, int(AutocallGlobalCol.K_PUT)].to(torch.float64).clamp(min=0.0)
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
        surv_proxy_term_64 = (1.0 - p_proxy.sum(dim=1)).to(torch.float64).clamp(min=0.0)
        denom_64 = torch.clamp(k_eff_64 * surv_proxy_term_64, min=1e-12)
        eps_r = 1e-12
        r_proxy_64 = torch.clamp(l_put_proxy_64 / denom_64, min=eps_r, max=1.0 - eps_r)
        logit_r_64 = torch.log(r_proxy_64 / (1.0 - r_proxy_64))
        u_put_64 = k_eff_64 * survival_act[:, -1].to(torch.float64)
        put_loss_64 = u_put_64 * torch.sigmoid(logit_r_64 + delta_put_loss.to(torch.float64))
        put_loss = put_loss_64.to(dtype=sched_feats.dtype)

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
    batch_size, m_dim, _ = sched_feats.shape
    dtype = sched_feats.dtype
    device = sched_feats.device

    if m_dim == 0:
        pv_transit_t = torch.as_tensor(pv_in_transit, dtype=dtype, device=device)
        return pv_transit_t.expand(batch_size) if pv_transit_t.ndim == 0 else pv_transit_t

    if is_long_basket is None:
        if global_feats.shape[-1] > int(AutocallGlobalCol.IS_LONG_BASKET):
            long_eff: bool | torch.Tensor = global_feats[:, int(AutocallGlobalCol.IS_LONG_BASKET)] > 0.5
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
        low_k_eff = torch.as_tensor(lower_strike, dtype=dtype, device=device)

    active_sched = sched_mask > 0.5
    m_sched = active_sched.to(dtype=dtype)
    has_any = active_sched.any(dim=1)
    if m_dim > 4 and not torch.compiler.is_compiling():
        any_sched_col = active_sched.any(dim=0)
        m_act = int(any_sched_col.nonzero()[-1].item()) + 1 if any_sched_col.any() else 1
    else:
        m_act = m_dim

    pv_transit_t = torch.as_tensor(pv_in_transit, dtype=dtype, device=device)
    if pv_transit_t.ndim == 0:
        pv_transit_t = pv_transit_t.expand(batch_size)

    sched_clean = torch.where(
        active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats)
    )
    df_pay = sched_clean[:, :, int(AutocallSchedCol.DF_PAY)]
    cpn_rate = sched_clean[:, :, int(AutocallSchedCol.COUPON_RATE)]

    k_put = global_feats[:, int(AutocallGlobalCol.K_PUT)].clamp(min=0.0)
    g_put = global_feats[:, int(AutocallGlobalCol.G_PUT)]
    c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
    is_memory = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)

    idx_grid = torch.arange(m_dim, device=device).unsqueeze(0)
    active_indices = torch.where(active_sched, idx_grid, torch.zeros_like(idx_grid))
    last_active_idx = active_indices.argmax(dim=1, keepdim=True)
    df_term = torch.gather(df_pay, dim=1, index=last_active_idx).squeeze(1)
    df_term = torch.where(has_any, df_term, torch.zeros_like(df_term))

    surv_term = torch.gather(out.survival_probs, dim=1, index=last_active_idx).squeeze(1)
    no_notional_t = torch.as_tensor(no_notional_repayment, dtype=dtype, device=device)
    pv_notional = (1.0 - no_notional_t) * (
        (df_pay * out.stopping_probs * m_sched).sum(dim=1) + df_term * surv_term
    )

    surv_prev = torch.cat(
        [torch.ones_like(out.survival_probs[:, :1]), out.survival_probs[:, :-1]], dim=1
    )
    mem_buf = c_mem_0 * is_memory
    pv_coupon = torch.zeros_like(pv_notional)
    ac_cpn_t = torch.as_tensor(autocall_coupon, dtype=dtype, device=device)
    force_cpn_t = torch.as_tensor(force_coupons_on_autocall, dtype=dtype, device=device)

    for m in range(m_act):
        c_eff_m = cpn_rate[:, m] + mem_buf
        q_m = out.coupon_probs[:, m]
        if isinstance(force_coupons_on_autocall, bool) and not force_coupons_on_autocall:
            eff_q_m = q_m
        else:
            q_forced = torch.maximum(
                q_m, out.stopping_probs[:, m] / surv_prev[:, m].clamp(min=1e-12)
            )
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
        eff_put_loss = torch.minimum(eff_put_loss, k_eff * surv_term)

    long_t = torch.as_tensor(long_eff, dtype=dtype, device=device)
    put_sign_t = -torch.ones_like(long_t, dtype=dtype, device=device)
    pv_option = put_sign_t * df_term * g_put * eff_put_loss
    pv_option = (
        pv_option
        + torch.as_tensor(one_star_pv, dtype=dtype, device=device)
        + torch.as_tensor(part_call_pv, dtype=dtype, device=device)
    )

    fund_t = torch.as_tensor(funding_pv, dtype=dtype, device=device)
    if w_fund_per_period is not None:
        w_f = torch.as_tensor(w_fund_per_period, dtype=dtype, device=device)
        if w_f.ndim == 1:
            w_f = w_f.unsqueeze(0)
        w_len = min(m_act, int(w_f.shape[1]))
        period_fund_pv = (surv_prev[:, :w_len] * w_f[:, :w_len] * m_sched[:, :w_len]).sum(dim=1)
        fund_t = fund_t + period_fund_pv

    active_pv = pv_transit_t + pv_notional + pv_coupon + pv_option - fund_t
    autocalled_t = torch.as_tensor(already_autocalled, dtype=torch.bool, device=device)
    is_live = has_any & (~autocalled_t)
    res_pv: torch.Tensor = torch.where(is_live, active_pv, pv_transit_t)
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
    "assemble_surrogate_product_value",
]

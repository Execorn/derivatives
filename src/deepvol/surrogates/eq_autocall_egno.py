from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from deepvol.surrogates.order_statistic_pooling import OrderStatisticPooling


class AutocallNodeCol(IntEnum):
    """Per-asset node feature column indices (D_max x 14)."""

    LOG_MONEYNESS = 0
    LOG_FWD_CARRY_1Y = 1
    CASH_DIV_LEVERAGE = 2
    REPO_SPREAD_AVG = 3
    IV_SHORT_ATM = 4
    IV_TERM_ATM = 5
    IV_TERM_SLOPE = 6
    SSVI_SKEW_RHO = 7
    SSVI_CURV_ETA = 8
    SMILE_RESIDUAL_PC1 = 9
    SMILE_RESIDUAL_PC2 = 10
    DIST_TO_PUT_STRIKE = 11
    PROXY_DELTA_WEIGHT = 12
    ASSET_ACTIVE_MASK = 13


class AutocallEdgeCol(IntEnum):
    """Pairwise asset edge feature column indices (D_max x D_max x 4)."""

    CORR_RHO = 0
    COVARIANCE_TERM = 1
    LOG_PERF_DIFF = 2
    OFF_DIAG_MASK = 3


class AutocallSchedCol(IntEnum):
    """Per-observation-period schedule feature column indices (M_max x 12)."""

    TAU_OBS = 0
    DELTA_TAU = 1
    DF_PAY = 2
    LOG_KO_BARRIER = 3
    IS_CALLABLE_FLAG = 4
    LOG_CPN_BARRIER = 5
    IS_GUARANTEED_CPN = 6
    COUPON_RATE = 7
    STULZ_FWD_W = 8
    STULZ_VOL_W = 9
    PROXY_KO_PROB = 10
    PROXY_CPN_PROB = 11


class AutocallGlobalCol(IntEnum):
    """Contract-level global context feature column indices (10,)."""

    K_PUT = 0
    PUT_STRIKE = 0
    B_KI = 1
    KI_BARRIER = 1
    HAS_KI_FLAG = 2
    I_KI_PAST = 3
    PAST_KI_BREACH = 3
    G_PUT = 4
    PUT_GEARING = 4
    C_MEM_0 = 5
    MEMORY_COUPON_BUFFER = 5
    IS_MEMORY_FLAG = 6
    ABS_KO_SPREAD = 7
    ABS_CPN_SPREAD = 8
    L_PUT_PROXY = 9
    PROXY_PUT_LOSS = 9


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


class MaskedEGNOLayer(nn.Module):
    """S_D-equivariant graph neural message passing layer over active basket assets."""

    def __init__(self, d_model: int = 128, d_edge: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_edge = d_edge
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + d_edge, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self._eval_cache_key: tuple[int, int, torch.dtype, torch.device] | None = None
        self._eval_cache_val: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        asset_mask: torch.Tensor,
        pair_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, d_max, channels = h.shape
        active_node = asset_mask > 0.5
        m_node = active_node.to(dtype=h.dtype)

        # Replace inactive node and edge slots via torch.where before MLP evaluation.
        h_clean = torch.where(active_node.unsqueeze(-1), h, torch.zeros_like(h))
        eye = torch.eye(d_max, device=h.device, dtype=h.dtype).unsqueeze(0)
        pair_mask = m_node.unsqueeze(1) * m_node.unsqueeze(2) * (1.0 - eye)
        if pair_weight is not None:
            pair_mask = pair_mask * pair_weight
        active_pair = pair_mask > 0.0
        edge_clean = torch.where(
            active_pair.unsqueeze(-1), edge_attr, torch.zeros_like(edge_attr)
        )

        # Algebraic linear fusion: split edge_mlp[0] (2*C + d_edge -> C) into node-i, node-j, and edge projections
        lin0 = self.edge_mlp[0]
        assert isinstance(lin0, nn.Linear)
        lin2 = self.edge_mlp[2]
        assert isinstance(lin2, nn.Linear)
        proj_i = F.linear(h_clean, lin0.weight[:, :channels], lin0.bias)
        proj_j = F.linear(h_clean, lin0.weight[:, channels : 2 * channels])
        proj_e = F.linear(edge_clean, lin0.weight[:, 2 * channels :])
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

        nlin0 = self.node_mlp[0]
        assert isinstance(nlin0, nn.Linear)
        nlin2 = self.node_mlp[2]
        assert isinstance(nlin2, nn.Linear)
        node_h = F.silu(
            F.linear(h_clean, nlin0.weight[:, :channels], nlin0.bias)
            + F.linear(agg, nlin0.weight[:, channels:])
        )
        node_upd = F.linear(node_h, nlin2.weight, nlin2.bias)
        out = self.norm(h_clean + node_upd)
        return torch.where(active_node.unsqueeze(-1), out, torch.zeros_like(out))

    def _get_eval_weights(
        self,
        lin0: nn.Linear,
        nlin0: nn.Linear,
        channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        key = (
            lin0.weight._version,
            nlin0.weight._version,
            lin0.weight.dtype,
            lin0.weight.device,
        )
        if self._eval_cache_key == key and self._eval_cache_val is not None:
            return self._eval_cache_val
        with torch.no_grad():
            w_h3 = torch.cat(
                [
                    lin0.weight[:, :channels],
                    lin0.weight[:, channels : 2 * channels],
                    nlin0.weight[:, :channels],
                ],
                dim=0,
            ).detach().contiguous()
            b_h3: torch.Tensor | None = None
            if lin0.bias is not None and nlin0.bias is not None:
                b_h3 = torch.cat(
                    [
                        lin0.bias.detach(),
                        torch.zeros_like(lin0.bias.detach()),
                        nlin0.bias.detach(),
                    ],
                    dim=0,
                ).contiguous()
            w_e = lin0.weight[:, 2 * channels :].detach().contiguous()
            w_agg = nlin0.weight[:, channels:].detach().contiguous()
        self._eval_cache_key = key
        self._eval_cache_val = (w_h3, b_h3, w_e, w_agg)
        return self._eval_cache_val

    def forward_premasked(
        self,
        h_clean: torch.Tensor,
        edge_clean: torch.Tensor,
        g_asset_3d: torch.Tensor,
        pair_mask_4d: torch.Tensor,
        inv_denom: torch.Tensor,
        bias_scale: torch.Tensor,
    ) -> torch.Tensor:
        channels = h_clean.shape[-1]
        lin0 = self.edge_mlp[0]
        assert isinstance(lin0, nn.Linear)
        lin2 = self.edge_mlp[2]
        assert isinstance(lin2, nn.Linear)
        nlin0 = self.node_mlp[0]
        assert isinstance(nlin0, nn.Linear)
        nlin2 = self.node_mlp[2]
        assert isinstance(nlin2, nn.Linear)

        if not self.training:
            w_h3, b_h3, w_e, w_agg = self._get_eval_weights(lin0, nlin0, channels)
            proj_i, proj_j, node_self = F.linear(h_clean, w_h3, b_h3).chunk(3, dim=-1)
            if b_h3 is None:
                if lin0.bias is not None:
                    proj_i = proj_i + lin0.bias
                if nlin0.bias is not None:
                    node_self = node_self + nlin0.bias
            proj_e = F.linear(edge_clean, w_e)
        else:
            w_h3 = torch.cat(
                [
                    lin0.weight[:, :channels],
                    lin0.weight[:, channels : 2 * channels],
                    nlin0.weight[:, :channels],
                ],
                dim=0,
            )
            proj_i, proj_j, node_self = F.linear(h_clean, w_h3).chunk(3, dim=-1)
            if lin0.bias is not None:
                proj_i = proj_i + lin0.bias
            if nlin0.bias is not None:
                node_self = node_self + nlin0.bias
            proj_e = F.linear(edge_clean, lin0.weight[:, 2 * channels :])
            w_agg = nlin0.weight[:, channels:]

        hidden_masked = F.silu(proj_i.unsqueeze(2) + proj_j.unsqueeze(1) + proj_e) * pair_mask_4d
        hidden_agg = hidden_masked.sum(dim=2) * inv_denom
        agg = F.linear(hidden_agg, lin2.weight)
        if lin2.bias is not None:
            agg = agg + bias_scale * lin2.bias

        node_h = F.silu(node_self + F.linear(agg, w_agg))
        node_upd = F.linear(node_h, nlin2.weight, nlin2.bias)
        norm_out = F.layer_norm(
            h_clean + node_upd, (channels,), self.norm.weight, self.norm.bias, self.norm.eps
        )
        return norm_out * g_asset_3d


class ParallelScheduleTransformer(nn.Module):
    """2-layer Pre-Norm parallel schedule Transformer encoder without CuDNN RNNs."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_sched_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_sched_layers = n_sched_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_sched_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        sched_mask: torch.Tensor,
        *,
        all_sched_active: bool | None = None,
    ) -> torch.Tensor:
        if all_sched_active is None:
            active_sched = sched_mask > 0.5
            is_all_active = bool(active_sched.all())
        else:
            is_all_active = all_sched_active
            active_sched = sched_mask > 0.5 if not is_all_active else sched_mask

        active_3d = None if is_all_active else active_sched.unsqueeze(-1)
        h = x if is_all_active else torch.where(active_3d, x, torch.zeros_like(x))  # type: ignore[arg-type]

        # Fast-path when sequence length is 1: self-attention on a single token has softmax([s]) == 1.0
        # identically, so MultiheadAttention reduces exactly to out_proj(v_proj(norm1(h))).
        if x.shape[1] == 1:
            c = self.d_model
            m_3d = None if is_all_active else active_3d.to(dtype=x.dtype)  # type: ignore[union-attr]
            for layer in self.encoder.layers:
                y = F.layer_norm(
                    h, (c,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps
                )
                v_w = layer.self_attn.in_proj_weight[2 * c :]
                v_b = (
                    layer.self_attn.in_proj_bias[2 * c :]
                    if layer.self_attn.in_proj_bias is not None
                    else None
                )
                v_val = F.linear(y, v_w, v_b)
                attn_out = F.linear(
                    v_val, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias
                )
                h = h + attn_out
                y2 = F.layer_norm(
                    h, (c,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps
                )
                ff_out = F.linear(
                    F.gelu(F.linear(y2, layer.linear1.weight, layer.linear1.bias)),
                    layer.linear2.weight,
                    layer.linear2.bias,
                )
                h = (h + ff_out) if m_3d is None else (h + ff_out) * m_3d
            return h

        # Avoid all-True padding rows that would produce softmax(-inf) NaNs when M_rem == 0.
        if is_all_active:
            padding_mask = None
        else:
            has_any = active_sched.any(dim=1, keepdim=True)
            padding_mask = torch.where(has_any, ~active_sched, torch.zeros_like(active_sched))

        # Force SDPBackend.MATH so second-order autograd.grad(create_graph=True) succeeds in eval().
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.encoder.layers:
                h = layer(h, src_key_padding_mask=padding_mask)
                if not is_all_active:
                    h = torch.where(active_3d, h, torch.zeros_like(h))  # type: ignore[arg-type]
        return h


class MaskedEQAutocallSurrogate(nn.Module):
    """Tier 2 Masked EGNO + Parallel Schedule Transformer residual pricing surrogate."""

    def __init__(
        self,
        d_max: int = 8,
        m_max: int = 48,
        d_model: int = 128,
        n_egno_layers: int = 3,
        n_sched_layers: int = 2,
        n_heads: int = 4,
        d_node: int = 14,
        d_edge: int = 4,
        d_sched: int = 12,
        d_global: int = 10,
        init_tau: float = 0.15,
    ) -> None:
        super().__init__()
        self.d_max = d_max
        self.m_max = m_max
        self.d_model = d_model
        self.n_egno_layers = n_egno_layers
        self.n_sched_layers = n_sched_layers
        self.n_heads = n_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_sched = d_sched
        self.d_global = d_global

        self.node_proj = nn.Linear(d_node, d_model)
        self.egno_layers = nn.ModuleList(
            [MaskedEGNOLayer(d_model=d_model, d_edge=d_edge) for _ in range(n_egno_layers)]
        )
        self.pool = OrderStatisticPooling(
            hidden_dim=d_model,
            out_dim=2 * d_model,
            init_tau=init_tau,
        )

        self.sched_proj = nn.Linear(d_sched + 2 * d_model + d_global, d_model)
        self.sched_transformer = ParallelScheduleTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_sched_layers=n_sched_layers,
        )

        self.period_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )
        self.put_head = nn.Sequential(
            nn.Linear(3 * d_model + d_global, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self._zero_res_cache_key: tuple[int, int, int, int] | None = None
        self._zero_res_cache_val: bool = False
        self.zero_init_residual_heads(std=1e-3)

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
            last_period_linear.weight._version,
            last_period_linear.bias._version if last_period_linear.bias is not None else -1,
            last_put_linear.weight._version,
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
        for p in self.parameters():
            p.requires_grad_(mode)
        return self

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
    ) -> AutocallSurrogateOutput:
        batch_size, d_dim, _ = node_feats.shape
        _, m_dim, _ = sched_feats.shape

        if all_active:
            d_act = d_dim
            m_act = m_dim
            active_asset = asset_mask
            m_asset = asset_mask
            active_sched = sched_mask
            m_sched = sched_mask
            node_sub = node_feats
            edge_sub = edge_feats
            sched_sub = sched_feats
            all_assets_active = True
            all_sched_active = True
        else:
            active_asset_full = asset_mask > 0.5
            active_sched_full = sched_mask > 0.5

            # Slice to active (d_act, m_act) extent across the batch while preserving exact Gate G1.7 invariance
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

            active_asset = active_asset_full[:, :d_act]
            m_asset = active_asset.to(dtype=node_feats.dtype)
            active_sched = active_sched_full[:, :m_act]
            m_sched = active_sched.to(dtype=sched_feats.dtype)

            node_sub = node_feats[:, :d_act, :]
            edge_sub = edge_feats[:, :d_act, :d_act, :]
            sched_sub = sched_feats[:, :m_act, :]

            all_assets_active = bool(active_asset.all())
            all_sched_active = bool(active_sched.all())

        if all_assets_active:
            node_clean = node_sub
        else:
            node_clean = torch.where(
                active_asset.unsqueeze(-1), node_sub, torch.zeros_like(node_sub)
            )
            if node_clean.shape[-1] > int(AutocallNodeCol.ASSET_ACTIVE_MASK):
                mask_col = m_asset.unsqueeze(-1)
                node_clean = torch.cat(
                    [
                        node_clean[:, :, : int(AutocallNodeCol.ASSET_ACTIVE_MASK)],
                        mask_col,
                        node_clean[:, :, int(AutocallNodeCol.ASSET_ACTIVE_MASK) + 1 :],
                    ],
                    dim=-1,
                )

        # Smooth worst-of relevance gate g_i = w_i^2 / (w_i^2 + 1e-6) so uninvolved assets (w_i < 1e-6)
        # cannot inject spurious Delta/Vega/Cega noise while preserving S_D equivariance and ties.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not detach_gate_weights):
            w_col = node_clean[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)]
            if detach_gate_weights:
                w_col = w_col.detach()
            w_proxy = w_col.clamp(min=0.0) if all_assets_active else w_col.clamp(min=0.0) * m_asset
            w_sq = w_proxy * w_proxy
            g_asset = (w_sq / (w_sq + 1e-3)) if all_assets_active else (w_sq / (w_sq + 1e-3)) * m_asset
            g_asset_3d = g_asset.unsqueeze(-1)
            w_sum = w_proxy.sum(dim=1, keepdim=True).clamp(min=1e-6)
            w_norm = w_proxy / w_sum
            h_disp = (1.0 - (w_norm * w_norm).sum(dim=1)).clamp(0.0, 1.0)

            eye = torch.eye(d_act, device=node_feats.device, dtype=node_feats.dtype).unsqueeze(0)
            pair_mask = (1.0 - eye) if all_assets_active else m_asset.unsqueeze(1) * m_asset.unsqueeze(2) * (1.0 - eye)
            pair_weight = g_asset.unsqueeze(1) * g_asset.unsqueeze(2)
            pair_mask_w = pair_mask * pair_weight
            pair_mask_4d = pair_mask_w.unsqueeze(-1)
            sum_p = pair_mask_w.sum(dim=2, keepdim=True)
            denom = sum_p.clamp(min=1.0)
            inv_denom = 1.0 / denom
            bias_scale = sum_p * inv_denom
            active_pair = None if all_assets_active else (pair_mask > 0.5)
        if all_assets_active and (all_active or not torch.isnan(edge_sub).any()):
            edge_clean = edge_sub * pair_mask.unsqueeze(-1)
        else:
            assert active_pair is not None
            edge_clean = torch.where(
                active_pair.unsqueeze(-1), edge_sub, torch.zeros_like(edge_sub)
            )
            if edge_clean.shape[-1] > int(AutocallEdgeCol.OFF_DIAG_MASK):
                pair_col = pair_mask.unsqueeze(-1)
                edge_clean = torch.cat(
                    [
                        edge_clean[:, :, :, : int(AutocallEdgeCol.OFF_DIAG_MASK)],
                        pair_col,
                        edge_clean[:, :, :, int(AutocallEdgeCol.OFF_DIAG_MASK) + 1 :],
                    ],
                    dim=-1,
                )

        if all_sched_active:
            sched_clean = sched_sub
        else:
            sched_clean = torch.where(
                active_sched.unsqueeze(-1), sched_sub, torch.zeros_like(sched_sub)
            )
        global_clean = global_feats

        h = F.linear(node_clean, self.node_proj.weight, self.node_proj.bias) * g_asset_3d
        for layer in self.egno_layers:
            h = layer.forward_premasked(  # type: ignore[operator]
                h,
                edge_clean,
                g_asset_3d,
                pair_mask_4d,
                inv_denom,
                bias_scale,
            )

        z_basket = self.pool(h, m_asset, asset_weights=w_proxy, all_active=all_assets_active)

        z_exp = z_basket.unsqueeze(1).expand(batch_size, m_act, -1)
        g_exp = global_clean.unsqueeze(1).expand(batch_size, m_act, -1)
        sched_in = torch.cat([sched_clean, z_exp, g_exp], dim=-1)
        sched_tokens = F.silu(F.linear(sched_in, self.sched_proj.weight, self.sched_proj.bias))
        if not all_sched_active:
            sched_tokens = torch.where(
                active_sched.unsqueeze(-1), sched_tokens, torch.zeros_like(sched_tokens)
            )
        seq_out = self.sched_transformer(sched_tokens, m_sched, all_sched_active=all_sched_active)

        plin0 = self.period_head[0]
        plin2 = self.period_head[2]
        assert isinstance(plin0, nn.Linear) and isinstance(plin2, nn.Linear)
        period_logits = F.linear(
            F.silu(F.linear(seq_out, plin0.weight, plin0.bias)),
            plin2.weight,
            plin2.bias,
        )
        if all_sched_active:
            delta_hazard_act = period_logits[:, :, 0]
            delta_coupon_act = period_logits[:, :, 1]
        else:
            delta_hazard_act = torch.where(
                active_sched,
                period_logits[:, :, 0],
                torch.zeros_like(period_logits[:, :, 0]),
            )
            delta_coupon_act = torch.where(
                active_sched,
                period_logits[:, :, 1],
                torch.zeros_like(period_logits[:, :, 1]),
            )

        c_call = (
            sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0)
            * m_sched
        )
        if all_sched_active:
            h_sched_mean = seq_out.mean(dim=1)
        else:
            sched_count = m_sched.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_sched_mean = seq_out.sum(dim=1) / sched_count
        put_in = torch.cat([h_sched_mean, z_basket, global_clean], dim=-1)
        path_or_disp_gate = torch.clamp(c_call.sum(dim=1) + h_disp, 0.0, 1.0)
        put_lin0 = self.put_head[0]
        put_lin2 = self.put_head[2]
        assert isinstance(put_lin0, nn.Linear) and isinstance(put_lin2, nn.Linear)
        delta_put_loss = (
            F.linear(
                F.silu(F.linear(put_in, put_lin0.weight, put_lin0.bias)),
                put_lin2.weight,
                put_lin2.bias,
            ).squeeze(-1)
            * path_or_disp_gate
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
        q_gate = (q_proxy / 1e-5).clamp(0.0, 1.0)
        q_pred = q_gate * torch.sigmoid(logit_q_proxy + delta_coupon_act)
        coupon_act = m_sched * (g_cpn + (1.0 - g_cpn) * q_pred)

        surv_term = survival_act[:, -1]
        k_put = global_clean[:, int(AutocallGlobalCol.K_PUT)].clamp(min=0.0)
        l_put_proxy = global_clean[:, int(AutocallGlobalCol.L_PUT_PROXY)].clamp(min=0.0)
        max_put_loss = k_put * surv_term

        raw_put_loss = (
            l_put_proxy + surv_term * k_put * 0.25 * torch.tanh(delta_put_loss)
        )
        put_loss = torch.minimum(
            torch.maximum(raw_put_loss, torch.zeros_like(raw_put_loss)),
            max_put_loss,
        )

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
        is_long_basket: bool | torch.Tensor = False,
        pv_in_transit: float | torch.Tensor = 0.0,
        funding_pv: float | torch.Tensor = 0.0,
        already_autocalled: bool | torch.Tensor = False,
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
    is_long_basket: bool | torch.Tensor = False,
    pv_in_transit: float | torch.Tensor = 0.0,
    funding_pv: float | torch.Tensor = 0.0,
    already_autocalled: bool | torch.Tensor = False,
) -> torch.Tensor:
    """Assemble total ProductValue via the W1 affine stopping-time decomposition."""
    batch_size, m_dim, _ = sched_feats.shape
    dtype = sched_feats.dtype
    device = sched_feats.device

    if m_dim == 0:
        pv_transit_t = torch.as_tensor(pv_in_transit, dtype=dtype, device=device)
        return pv_transit_t.expand(batch_size) if pv_transit_t.ndim == 0 else pv_transit_t

    active_sched = sched_mask > 0.5
    all_sched_active = bool(m_dim <= 4 and active_sched.all())

    if (
        all_sched_active
        and isinstance(no_notional_repayment, bool)
        and isinstance(autocall_coupon, (int, float))
        and isinstance(force_coupons_on_autocall, bool)
        and isinstance(is_long_basket, bool)
        and isinstance(pv_in_transit, (int, float))
        and isinstance(funding_pv, (int, float))
        and isinstance(already_autocalled, bool)
        and not already_autocalled
    ):
        df_pay = sched_feats[:, :, int(AutocallSchedCol.DF_PAY)]
        cpn_rate = sched_feats[:, :, int(AutocallSchedCol.COUPON_RATE)]
        g_put = global_feats[:, int(AutocallGlobalCol.G_PUT)]
        df_term = df_pay[:, -1]
        surv_term = out.survival_probs[:, -1]

        if no_notional_repayment:
            pv_notional = torch.zeros_like(df_term)
        else:
            pv_notional = (df_pay * out.stopping_probs).sum(dim=1) + df_term * surv_term

        if m_dim == 1 and not force_coupons_on_autocall:
            c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
            is_memory = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)
            c_eff_0 = cpn_rate[:, 0] + c_mem_0 * is_memory
            q_0 = out.coupon_probs[:, 0]
            if autocall_coupon == 0.0:
                pv_coupon = df_pay[:, 0] * q_0 * c_eff_0
            else:
                pv_coupon = df_pay[:, 0] * (q_0 * c_eff_0 + out.stopping_probs[:, 0] * float(autocall_coupon))
        else:
            c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
            is_memory = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)
            surv_prev = torch.cat(
                [torch.ones_like(out.survival_probs[:, :1]), out.survival_probs[:, :-1]], dim=1
            )
            mem_buf = c_mem_0 * is_memory
            pv_coupon = torch.zeros_like(df_term)
            ac_f = float(autocall_coupon)
            for m in range(m_dim):
                c_eff_m = cpn_rate[:, m] + mem_buf
                q_m = out.coupon_probs[:, m]
                eff_q_m = (
                    torch.maximum(q_m, out.stopping_probs[:, m] / surv_prev[:, m].clamp(min=1e-12))
                    if force_coupons_on_autocall
                    else q_m
                )
                pv_coupon = pv_coupon + df_pay[:, m] * (
                    surv_prev[:, m] * eff_q_m * c_eff_m + out.stopping_probs[:, m] * ac_f
                )
                mem_buf = is_memory * (1.0 - eff_q_m) * c_eff_m

        put_sign = 1.0 if is_long_basket else -1.0
        pv_option = put_sign * df_term * g_put * out.put_loss
        net_const = float(pv_in_transit) - float(funding_pv)
        if net_const == 0.0:
            return pv_notional + pv_coupon + pv_option
        return (pv_notional + pv_coupon + pv_option) + net_const

    pv_transit_t = torch.as_tensor(pv_in_transit, dtype=dtype, device=device)
    if pv_transit_t.ndim == 0:
        pv_transit_t = pv_transit_t.expand(batch_size)

    m_sched = active_sched.to(dtype=dtype)
    has_any = active_sched.any(dim=1)
    if m_dim > 4:
        any_sched_col = active_sched.any(dim=0)
        m_act = int(any_sched_col.nonzero()[-1].item()) + 1 if any_sched_col.any() else 1
    else:
        m_act = m_dim

    sched_clean = torch.where(
        active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats)
    )
    df_pay = sched_clean[:, :, int(AutocallSchedCol.DF_PAY)]
    cpn_rate = sched_clean[:, :, int(AutocallSchedCol.COUPON_RATE)]

    g_put = global_feats[:, int(AutocallGlobalCol.G_PUT)]
    c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
    is_memory = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)

    idx_grid = torch.arange(m_dim, device=device).unsqueeze(0)
    active_indices = torch.where(active_sched, idx_grid, torch.zeros_like(idx_grid))
    last_active_idx = active_indices.argmax(dim=1, keepdim=True)
    df_term = torch.gather(df_pay, dim=1, index=last_active_idx).squeeze(1)
    df_term = torch.where(has_any, df_term, torch.zeros_like(df_term))

    surv_term = out.survival_probs[:, -1]
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

    long_t = torch.as_tensor(is_long_basket, dtype=dtype, device=device)
    put_sign_t = 2.0 * long_t - 1.0
    pv_option = put_sign_t * df_term * g_put * out.put_loss
    fund_t = torch.as_tensor(funding_pv, dtype=dtype, device=device)
    active_pv = pv_transit_t + pv_notional + pv_coupon + pv_option - fund_t

    autocalled_t = torch.as_tensor(already_autocalled, dtype=torch.bool, device=device)
    is_live = has_any & (~autocalled_t)
    res_pv: torch.Tensor = torch.where(is_live, active_pv, pv_transit_t)
    return res_pv

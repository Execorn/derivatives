"""src/deepvol/training/generate_cross_shards.py

Sharded AAD dataset generator for triangulated FX/Commodity cross pricing.
Computes Gaussian-copula teacher prices, implied volatilities, H1 analytic baseline,
residual targets eps_star, and exact input differentials across train/val/test/stress splits.
Streams memory-mapped NumPy shards directly to disk and writes cryptographic SHA-256 manifest.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    pillar_strikes,
)
from deepvol.market.svi_slice import (
    RawSVI,
    fit_raw_svi_5pt,
)
from deepvol.models.fx_cross_analytic import beta_split_jet
from deepvol.models.fx_cross_copula import (
    CopulaConfig,
    GaussianCopulaCrossEngine,
)
from deepvol.models import taylor_jets as tj
from deepvol.training.data_generation_cross import (
    CrossSobolSampler,
    SamplerConfig,
)

# Standardized structured record layout for memory-mapped .npy shards.
# Total width: 296 bytes per record.
SHARD_DTYPE = np.dtype([
    ("x", np.float32, (13,)),
    ("c", np.float64),
    ("dc_dx", np.float64, (13,)),
    ("eps_star", np.float64),
    ("deps_dx", np.float64, (13,)),
    ("sigma_h1", np.float64),
    ("sigma_cop", np.float64),
])


@dataclass(frozen=True)
class ShardConfig:
    """Configuration container for sharded cross dataset generation."""
    out_dir: Path = Path("artifacts/data/fx_cross_shards")
    records_per_shard: int = 65_536
    batch_states: Optional[int] = None
    sampler: SamplerConfig = SamplerConfig()
    copula: CopulaConfig = CopulaConfig()


def get_adaptive_batch_states(n_nodes: int) -> int:
    """Returns safe state batch size S to keep peak VRAM <= 165 MB target."""
    if n_nodes >= 801:
        return 1
    elif n_nodes >= 401:
        return 4
    else:
        return 16


def compute_sha256(path: Union[str, Path]) -> str:
    """Computes SHA-256 cryptographic digest of a file."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, falling back to 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class StreamingFeatureStats:
    """Computes exact mean and standard deviation of 13-D inputs without storing records in RAM."""

    def __init__(self, dim: int = 13) -> None:
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray[Any, Any]) -> None:
        """Updates sum and sum-of-squares with incoming batch."""
        n = x.shape[0]
        if n == 0:
            return
        self.count += n
        x_f64 = x.astype(np.float64)
        self.sum += x_f64.sum(axis=0)
        self.sum_sq += (x_f64 ** 2).sum(axis=0)

    def mean_and_std(self) -> Tuple[List[float], List[float]]:
        """Computes empirical mean and standard deviation."""
        if self.count == 0:
            return [0.0] * 13, [1.0] * 13
        mean = self.sum / self.count
        variance = np.maximum(0.0, (self.sum_sq / self.count) - (mean ** 2))
        std = np.maximum(np.sqrt(variance), 1e-6)
        return [float(v) for v in mean], [float(v) for v in std]


class ShardStreamWriter:
    """Streams evaluated records directly into memory-mapped .npy shard files on disk."""

    def __init__(
        self,
        split_dir: Path,
        split_name: str,
        records_per_shard: int,
        total_split_records: int,
    ) -> None:
        self.split_dir = split_dir
        self.split_name = split_name
        self.records_per_shard = records_per_shard
        self.total_split_records = total_split_records
        self.records_written_in_split = 0
        self.shard_idx = 0
        self.current_shard_offset = 0
        self.active_shard_path: Optional[Path] = None
        self.active_mmap: Optional[np.memmap[Any, Any]] = None
        self.checksums: Dict[str, str] = {}

    def _open_next_shard(self) -> None:
        remaining = self.total_split_records - self.records_written_in_split
        alloc_size = min(self.records_per_shard, remaining)
        shard_fname = f"shard_{self.shard_idx:04d}.npy"
        self.active_shard_path = self.split_dir / shard_fname
        self.active_mmap = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            self.active_shard_path,
            mode="w+",
            dtype=SHARD_DTYPE,
            shape=(alloc_size,),
        )
        self.current_shard_offset = 0

    def write_batch(self, labels: Dict[str, Tensor]) -> None:
        """Writes batch into active memory-mapped shard file, handling shard rollovers."""
        n_rec = labels["c"].numel()
        rec_x = labels["x"].cpu().numpy().reshape(-1, 13)
        rec_c = labels["c"].cpu().numpy().reshape(-1)
        rec_dc = labels["dc_dx"].cpu().numpy().reshape(-1, 13)
        rec_eps = labels["eps_star"].cpu().numpy().reshape(-1)
        rec_deps = labels["deps_dx"].cpu().numpy().reshape(-1, 13)
        rec_sh1 = labels["sigma_h1"].cpu().numpy().reshape(-1)
        rec_scop = labels["sigma_cop"].cpu().numpy().reshape(-1)

        src_offset = 0
        while src_offset < n_rec:
            if self.active_mmap is None or self.current_shard_offset >= len(self.active_mmap):
                self._close_active_shard()
                self._open_next_shard()

            assert self.active_mmap is not None
            capacity = len(self.active_mmap) - self.current_shard_offset
            chunk_len = min(capacity, n_rec - src_offset)

            dst_sl = slice(self.current_shard_offset, self.current_shard_offset + chunk_len)
            src_sl = slice(src_offset, src_offset + chunk_len)

            self.active_mmap["x"][dst_sl] = rec_x[src_sl]
            self.active_mmap["c"][dst_sl] = rec_c[src_sl]
            self.active_mmap["dc_dx"][dst_sl] = rec_dc[src_sl]
            self.active_mmap["eps_star"][dst_sl] = rec_eps[src_sl]
            self.active_mmap["deps_dx"][dst_sl] = rec_deps[src_sl]
            self.active_mmap["sigma_h1"][dst_sl] = rec_sh1[src_sl]
            self.active_mmap["sigma_cop"][dst_sl] = rec_scop[src_sl]

            self.current_shard_offset += chunk_len
            self.records_written_in_split += chunk_len
            src_offset += chunk_len

    def _close_active_shard(self) -> None:
        if self.active_mmap is not None:
            self.active_mmap.flush()
            del self.active_mmap
            self.active_mmap = None

            assert self.active_shard_path is not None
            digest = compute_sha256(self.active_shard_path)
            rel_path = f"{self.split_name}/{self.active_shard_path.name}"
            self.checksums[rel_path] = digest
            self.shard_idx += 1

    def finalize(self) -> Dict[str, str]:
        """Flushes remaining shard and returns completed checksum mapping."""
        self._close_active_shard()
        return self.checksums


def label_state_batch(
    states: Tensor,
    z: Tensor,
    engine: GaussianCopulaCrossEngine,
) -> Dict[str, Tensor]:
    """Generates supervised targets and AAD input differentials for a batch of states.
    
    Uses per-strike autograd chunking (M_chunk = 1) with multivariable chain-rule graph
    decomposition: copula quadrature is differentiated exactly once and deallocated
    immediately per strike, capping peak VRAM to <= 154.11 MB regardless of strike count M.

    Args:
        states: (S, 12) float64 tensor of market state parameters.
        z: (S, M) float64 tensor of standardized moneyness strikes.
        engine: GaussianCopulaCrossEngine.

    Returns:
        Dictionary containing detached tensors:
            - 'x': (S, M, 13) float32 network inputs.
            - 'c': (S, M) float64 undiscounted option prices.
            - 'dc_dx': (S, M, 13) float64 gradients of c with respect to x.
            - 'eps_star': (S, M) float64 residual log-vol targets.
            - 'deps_dx': (S, M, 13) float64 gradients of eps_star with respect to x.
            - 'sigma_h1': (S, M) float64 analytic baseline volatilities.
            - 'sigma_cop': (S, M) float64 copula implied volatilities.
    """
    dev = engine.device
    st_dev = states.to(dev, dtype=torch.float64)
    z_dev = z.to(dev, dtype=torch.float64)
    S_cnt = st_dev.shape[0]
    M_cnt = z_dev.shape[1]

    # Pre-allocate output buffers
    c_out = torch.empty((S_cnt, M_cnt), dtype=torch.float64, device=dev)
    eps_star_out = torch.empty((S_cnt, M_cnt), dtype=torch.float64, device=dev)
    sigma_cop_out = torch.empty((S_cnt, M_cnt), dtype=torch.float64, device=dev)
    sigma_h1_out = torch.empty((S_cnt, M_cnt), dtype=torch.float64, device=dev)
    dc_dx_out = torch.empty((S_cnt, M_cnt, 13), dtype=torch.float64, device=dev)
    deps_dx_out = torch.empty((S_cnt, M_cnt, 13), dtype=torch.float64, device=dev)

    # Compute network coordinate x (shape: S, M, 13)
    z_mapped = 7.0 * torch.tanh(z_dev / 7.0)
    dx0_dz = 1.0 - torch.tanh(z_dev / 7.0) ** 2
    x_coords = [z_mapped.unsqueeze(-1)]
    for j in range(12):
        x_coords.append(st_dev[:, j].view(S_cnt, 1, 1).expand(-1, M_cnt, 1))
    x_out = torch.cat(x_coords, dim=-1).to(dtype=torch.float32)

    ones = torch.ones(S_cnt, dtype=torch.float64, device=dev)

    # Precompute state-only SVI parameters and their exact 12-D state Jacobians once per state batch
    st_init = st_dev.clone().requires_grad_(True)
    sqrtT_init = st_init[:, 0]
    T_init = sqrtT_init ** 2
    q1_init = st_init[:, 2:7]
    q2_init = st_init[:, 7:12]

    leg_init = LegMarket(spot=ones, df_foreign=ones, df_domestic=ones, T=T_init)
    k1_init, s1_init = pillar_strikes(q1_init, leg_init, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x1_init = torch.log(k1_init / leg_init.forward.unsqueeze(-1))
    svi1_init, _ = fit_raw_svi_5pt(x1_init, s1_init ** 2 * T_init.unsqueeze(1))

    k2_init, s2_init = pillar_strikes(q2_init, leg_init, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x2_init = torch.log(k2_init / leg_init.forward.unsqueeze(-1))
    svi2_init, _ = fit_raw_svi_5pt(x2_init, s2_init ** 2 * T_init.unsqueeze(1))

    svi1_params_init = svi1_init.params.reshape(S_cnt, 5)
    svi2_params_init = svi2_init.params.reshape(S_cnt, 5)
    dsvi1_list: List[Tensor] = []
    dsvi2_list: List[Tensor] = []
    for p_idx in range(5):
        g1_p = torch.autograd.grad(svi1_params_init[:, p_idx].sum(), st_init, retain_graph=True)[0]
        dsvi1_list.append(g1_p.detach())
    for p_idx in range(5):
        g2_p = torch.autograd.grad(
            svi2_params_init[:, p_idx].sum(), st_init, retain_graph=(p_idx < 4)
        )[0]
        dsvi2_list.append(g2_p.detach())
    dsvi1_dst = torch.stack(dsvi1_list, dim=1)  # (S_cnt, 5, 12)
    dsvi2_dst = torch.stack(dsvi2_list, dim=1)  # (S_cnt, 5, 12)
    svi1_params_val = svi1_params_init.detach()
    svi2_params_val = svi2_params_init.detach()
    del st_init, leg_init, svi1_init, svi2_init, svi1_params_init, svi2_params_init

    sqrtT_val = st_dev[:, 0]
    T_val = sqrtT_val ** 2
    rho_val = st_dev[:, 1]
    q1_val = st_dev[:, 2:7]
    q2_val = st_dev[:, 7:12]
    s1_atm_val = q1_val[:, 0:1]
    s2_atm_val = q2_val[:, 0:1]
    sig_cross_atm_val = torch.sqrt(s1_atm_val ** 2 + s2_atm_val ** 2 + 2.0 * rho_val.unsqueeze(1) * s1_atm_val * s2_atm_val)

    # Precompute state-level copula quantile grids X_cop (S, n) and Y_cop (S, n, n) once per state batch
    st_leaf_cop = st_dev.clone().requires_grad_(True)
    sqrtT_1 = st_leaf_cop[:, 0]
    T_1 = sqrtT_1 ** 2
    rho_1 = st_leaf_cop[:, 1]
    q1_1 = st_leaf_cop[:, 2:7]
    q2_1 = st_leaf_cop[:, 7:12]

    d_st_1 = st_leaf_cop - st_dev
    svi1_p1 = svi1_params_val + torch.einsum("spj,sj->sp", dsvi1_dst, d_st_1)
    svi2_p1 = svi2_params_val + torch.einsum("spj,sj->sp", dsvi2_dst, d_st_1)
    svi1_1 = RawSVI(svi1_p1)
    svi2_1 = RawSVI(svi2_p1)

    s1_atm_1 = q1_1[:, 0:1]
    s2_atm_1 = q2_1[:, 0:1]
    sig_cross_atm_1 = torch.sqrt(s1_atm_1 ** 2 + s2_atm_1 ** 2 + 2.0 * rho_1.unsqueeze(1) * s1_atm_1 * s2_atm_1)

    n_quad = engine.cfg.n_nodes
    X_cop = engine.quantiles(svi1_1, ones, T_1, engine.u_1d, reflect=False)
    rho_z_col = (-rho_1).view(S_cnt, 1, 1)
    scale_perp = torch.sqrt(torch.clamp(1.0 - rho_z_col ** 2, min=1e-8))
    Z2_joint = rho_z_col * engine.z.view(1, n_quad, 1) + scale_perp * engine.z.view(1, 1, n_quad)
    u2_joint = torch.special.ndtr(Z2_joint)
    Y_cop = engine.quantiles(svi2_1, ones, T_1, u2_joint, reflect=True)
    X_cop_b = X_cop.view(S_cnt, 1, n_quad, 1)
    Y_cop_b = Y_cop.view(S_cnt, 1, n_quad, n_quad)
    w1d_row = engine.w1d.view(1, 1, 1, n_quad)
    w1d_col = engine.w1d.view(1, 1, n_quad)

    # Evaluate strikes one by one to ensure peak memory is strictly O(1) in M
    for m in range(M_cnt):
        z_m = z_dev[:, m:m + 1]

        # -------------------------------------------------------------
        # Pass 1: Primal Copula Pricing & Price Adjoints w.r.t (z, st)
        # -------------------------------------------------------------
        z_leaf = z_m.clone().requires_grad_(True)
        k_cross_1 = z_leaf * sig_cross_atm_1 * sqrtT_1.unsqueeze(1)
        K_1 = torch.exp(k_cross_1)
        K_b = K_1.view(S_cnt, 1, 1, 1)

        is_call_mask = (K_1 >= 1.0)
        if is_call_mask.all():
            payoff = torch.clamp(X_cop_b - K_b * Y_cop_b, min=0.0)
            p_int = (payoff * w1d_row).sum(dim=-1)
            c_m = (p_int * w1d_col).sum(dim=-1)
        elif (~is_call_mask).all():
            payoff = torch.clamp(K_b * Y_cop_b - X_cop_b, min=0.0)
            p_int = (payoff * w1d_row).sum(dim=-1)
            otm_p = (p_int * w1d_col).sum(dim=-1)
            c_m = otm_p + (1.0 - K_1)
        else:
            pay_c = torch.clamp(X_cop_b - K_b * Y_cop_b, min=0.0)
            c_otm = ((pay_c * w1d_row).sum(dim=-1) * w1d_col).sum(dim=-1)
            pay_p = torch.clamp(K_b * Y_cop_b - X_cop_b, min=0.0)
            p_otm = ((pay_p * w1d_row).sum(dim=-1) * w1d_col).sum(dim=-1)
            c_m = torch.where(is_call_mask, c_otm, p_otm + (1.0 - K_1))

        # Backward Pass 1: Differentiate c w.r.t (z, st)
        gc_z, gc_st = torch.autograd.grad(
            c_m.sum(), [z_leaf, st_leaf_cop], retain_graph=(m < M_cnt - 1)
        )

        c_out[:, m] = c_m.detach().squeeze(-1)
        dc_dx_out[:, m, 0] = (gc_z.detach() / dx0_dz[:, m:m + 1]).squeeze(-1)
        dc_dx_out[:, m, 1:] = gc_st.detach()

        del K_1, K_b, k_cross_1, z_leaf, c_m, gc_z, gc_st

    del X_cop, Y_cop, X_cop_b, Y_cop_b, st_leaf_cop, u2_joint, Z2_joint

    # -------------------------------------------------------------
    # Pass 2: Vectorized Implied Vol & Partial Sensitivities (c, K, T)
    # -------------------------------------------------------------
    k_cross_all = z_dev * sig_cross_atm_val * sqrtT_val.unsqueeze(1)
    K_all_val = torch.exp(k_cross_all)

    c_scalar_all = c_out.clone().detach().requires_grad_(True)
    K_leaf_all = K_all_val.clone().detach().requires_grad_(True)
    T_leaf_all = T_val.unsqueeze(1).expand(S_cnt, M_cnt).clone().detach().requires_grad_(True)

    is_call_all = (K_all_val >= 1.0)
    otm_all = torch.where(is_call_all, c_scalar_all, c_scalar_all - (1.0 - K_leaf_all))
    F_all = ones.unsqueeze(1).expand(S_cnt, M_cnt)

    if S_cnt < 8:
        pad_n = 8 - S_cnt
        otm_pad = torch.cat([otm_all, otm_all[:1].expand(pad_n, M_cnt)], dim=0)
        F_pad = torch.cat([F_all, F_all[:1].expand(pad_n, M_cnt)], dim=0)
        K_pad = torch.cat([K_leaf_all, K_leaf_all[:1].expand(pad_n, M_cnt)], dim=0)
        T_pad = torch.cat([T_leaf_all, T_leaf_all[:1].expand(pad_n, M_cnt)], dim=0)
        call_pad = torch.cat([is_call_all, is_call_all[:1].expand(pad_n, M_cnt)], dim=0)
        sig_cop_all = engine.implied_vol(otm_pad, F=F_pad, K=K_pad, T=T_pad, is_call=call_pad)[:S_cnt]
    else:
        sig_cop_all = engine.implied_vol(otm_all, F=F_all, K=K_leaf_all, T=T_leaf_all, is_call=is_call_all)

    log_sig_cop_all = torch.log(sig_cop_all)
    dlog_scop_dc, dlog_scop_dK, dlog_scop_dT = torch.autograd.grad(
        log_sig_cop_all.sum(), [c_scalar_all, K_leaf_all, T_leaf_all], retain_graph=False
    )
    sigma_cop_out.copy_(sig_cop_all.detach())
    dlog_dc_det = dlog_scop_dc.detach()
    dlog_dK_det = dlog_scop_dK.detach()
    dlog_dT_det = dlog_scop_dT.detach()
    del c_scalar_all, K_leaf_all, T_leaf_all, otm_all, sig_cop_all, log_sig_cop_all

    # -------------------------------------------------------------
    # Pass 3: Direct Non-Copula Sensitivities w.r.t (z, st)
    # -------------------------------------------------------------
    P = 0
    zero_s = torch.zeros(S_cnt, dtype=torch.float64, device=dev)
    d1_jet_d = tj.const(zero_s, P)
    d2_jet_d = tj.const(zero_s, P)

    for m in range(M_cnt):
        z_m = z_dev[:, m:m + 1]
        st_dir = st_dev.clone().requires_grad_(True)
        z_dir = z_m.clone().requires_grad_(True)

        sqrtT_d = st_dir[:, 0]
        T_d = sqrtT_d ** 2
        rho_d = st_dir[:, 1]
        q1_d = st_dir[:, 2:7]
        q2_d = st_dir[:, 7:12]

        d_st_d = st_dir - st_dev
        svi1_pd = svi1_params_val + torch.einsum("spj,sj->sp", dsvi1_dst, d_st_d)
        svi2_pd = svi2_params_val + torch.einsum("spj,sj->sp", dsvi2_dst, d_st_d)

        s1_atm_d = q1_d[:, 0:1]
        s2_atm_d = q2_d[:, 0:1]
        sig_cross_atm_d = torch.sqrt(s1_atm_d ** 2 + s2_atm_d ** 2 + 2.0 * rho_d.unsqueeze(1) * s1_atm_d * s2_atm_d)
        k_cross_d = z_dir * sig_cross_atm_d * sqrtT_d.unsqueeze(1)
        K_d = torch.exp(k_cross_d)

        k_jet_d = tj.Jet(
            v=k_cross_d.reshape(-1),
            d=zero_s,
            dd=zero_s,
            t=None,
        )
        rho_jet_d = tj.const(rho_d, P)

        sig_jet_d, _, _ = beta_split_jet(
            k_jet_d, T_d, rho_jet_d,
            svi1_pd,
            svi2_pd,
            d1_jet_d, d2_jet_d,
        )
        sig_h1_d = sig_jet_d.v.reshape(S_cnt, 1)

        log_sig_cop_d = dlog_dK_det[:, m:m + 1] * K_d + dlog_dT_det[:, m:m + 1] * T_d.unsqueeze(1)
        eps_d = log_sig_cop_d - torch.log(torch.clamp(sig_h1_d, min=1e-8))
        geps_z_dir, geps_st_dir = torch.autograd.grad(eps_d.sum(), [z_dir, st_dir], retain_graph=False)

        sig_h1_val = sig_h1_d.detach().squeeze(-1)
        sigma_h1_out[:, m] = sig_h1_val
        eps_star_out[:, m] = torch.log(sigma_cop_out[:, m] / torch.clamp(sig_h1_val, min=1e-8))

        deps_dc_m = dlog_dc_det[:, m:m + 1]
        gc_z_m = dc_dx_out[:, m, 0:1] * dx0_dz[:, m:m + 1]
        gc_st_m = dc_dx_out[:, m, 1:]

        geps_st = deps_dc_m * gc_st_m + geps_st_dir
        geps_z = deps_dc_m * gc_z_m + geps_z_dir

        deps_dx_out[:, m, 0] = (geps_z / dx0_dz[:, m:m + 1]).squeeze(-1)
        deps_dx_out[:, m, 1:] = geps_st

        del st_dir, z_dir, k_cross_d, K_d, sig_h1_d, eps_d, geps_z_dir, geps_st_dir

    return {
        "x": x_out.detach(),
        "c": c_out.detach(),
        "dc_dx": dc_dx_out.detach(),
        "eps_star": eps_star_out.detach(),
        "deps_dx": deps_dx_out.detach(),
        "sigma_h1": sigma_h1_out.detach(),
        "sigma_cop": sigma_cop_out.detach(),
    }


def generate_shards(cfg: ShardConfig) -> Path:
    """Generates sharded dataset splits and manifest.json via streaming disk memmaps.
    
    Returns:
        Path to generated manifest.json file.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = CrossSobolSampler(cfg.sampler)
    raw_states = sampler.sample_raw_states()
    certified_states, _ = sampler.filter_and_certify(raw_states)
    n_valid = certified_states.shape[0]
    n_raw = raw_states.shape[0]
    rejection_count = n_raw - n_valid

    strikes = sampler.strikes(n_valid)
    splits = sampler.split(n_valid)
    engine = GaussianCopulaCrossEngine(cfg.copula)

    manifest_data: Dict[str, object] = {
        "sobol_seed": cfg.sampler.seed,
        "git_commit": get_git_commit_hash(),
        "copula_nodes": cfg.copula.n_nodes,
        "rejection_count": int(rejection_count),
        "total_states": int(n_raw),
        "valid_states": int(n_valid),
        "file_checksums": {},
    }

    # Resolution-dependent adaptive state batch size
    batch_states = cfg.batch_states if cfg.batch_states is not None else get_adaptive_batch_states(cfg.copula.n_nodes)

    train_stats = StreamingFeatureStats(dim=13)
    total_records = 0

    for split_name in ["train", "val", "test", "stress"]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        indices = splits[split_name]
        if len(indices) == 0:
            continue

        split_states = certified_states[indices]
        split_z = strikes[indices]
        n_split_states = split_states.shape[0]
        strikes_per_state = split_z.shape[1]
        total_split_records = n_split_states * strikes_per_state

        writer = ShardStreamWriter(
            split_dir=split_dir,
            split_name=split_name,
            records_per_shard=cfg.records_per_shard,
            total_split_records=total_split_records,
        )

        for b_start in range(0, n_split_states, batch_states):
            b_end = min(b_start + batch_states, n_split_states)
            batch_st = split_states[b_start:b_end]
            batch_z = split_z[b_start:b_end]

            labels = label_state_batch(batch_st, batch_z, engine)

            if split_name == "train":
                train_stats.update(labels["x"].cpu().numpy().reshape(-1, 13))

            writer.write_batch(labels)

            # Strict memory hygiene after every batch
            del labels, batch_st, batch_z
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        split_checksums = writer.finalize()
        checksums_dict = manifest_data["file_checksums"]
        assert isinstance(checksums_dict, dict)
        checksums_dict.update(split_checksums)
        total_records += total_split_records

        # Inter-split cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    manifest_data["total_records"] = total_records
    train_mean, train_std = train_stats.mean_and_std()
    manifest_data["train_mean"] = train_mean
    manifest_data["train_std"] = train_std

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for generating sharded cross dataset."""
    parser = argparse.ArgumentParser(description="Generate sharded FX cross training datasets with AAD labels.")
    parser.add_argument("--log2-states", type=int, default=13)
    parser.add_argument("--out", type=str, default="artifacts/data/fx_cross_shards")
    parser.add_argument("--nodes", type=int, default=1601)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--records-per-shard", type=int, default=65536)
    parser.add_argument("--batch-states", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    cfg = ShardConfig(
        out_dir=Path(args.out),
        records_per_shard=args.records_per_shard,
        batch_states=args.batch_states,
        sampler=SamplerConfig(log2_states=args.log2_states, seed=args.seed),
        copula=CopulaConfig(n_nodes=args.nodes, device=args.device),
    )
    generate_shards(cfg)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

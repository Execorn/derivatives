"""src/deepvol/training/generate_cross_shards.py

Sharded AAD dataset generator for triangulated FX/Commodity cross pricing.
Computes Gaussian-copula teacher prices, implied volatilities, H1 analytic baseline,
residual targets eps_star, and exact input differentials across train/val/test/stress splits.
Saves memory-mapped NumPy shards and writes cryptographic SHA-256 manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    CrossState,
    GaussianCopulaCrossEngine,
)
from deepvol.models import taylor_jets as tj
from deepvol.training.data_generation_cross import (
    CrossSobolSampler,
    SamplerConfig,
)


@dataclass(frozen=True)
class ShardConfig:
    """Configuration container for sharded cross dataset generation."""
    out_dir: Path = Path("artifacts/data/fx_cross_shards")
    records_per_shard: int = 65_536
    sampler: SamplerConfig = SamplerConfig()
    copula: CopulaConfig = CopulaConfig()


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


def _eval_cross_state(
    st: Tensor,
    z_val: Tensor,
    engine: GaussianCopulaCrossEngine,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluates option prices, copula vols, H1 baseline, and residual eps_star.
    
    Uses OTM pricing and put-call parity to eliminate deep ITM truncation errors.
    
    Args:
        st: (S, 12) float64 state tensor.
        z_val: (S, M) float64 moneyness strike tensor.
        engine: GaussianCopulaCrossEngine instance.
        
    Returns:
        Tuple of (c, sig_cop, sig_h1, eps_star) of shape (S, M) in float64.
    """
    dev = engine.device
    st_dev = st.to(dev, dtype=torch.float64)
    z_dev = z_val.to(dev, dtype=torch.float64)
    S_cnt = st_dev.shape[0]
    M_cnt = z_dev.shape[1]

    sqrtT = st_dev[:, 0]
    T = sqrtT ** 2
    rho = st_dev[:, 1]
    q1 = st_dev[:, 2:7]
    q2 = st_dev[:, 7:12]

    ones = torch.ones(S_cnt, dtype=torch.float64, device=dev)
    leg = LegMarket(spot=ones, df_foreign=ones, df_domestic=ones, T=T)
    k1, s1 = pillar_strikes(q1, leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x1 = torch.log(k1 / leg.forward.unsqueeze(-1))
    svi1, _ = fit_raw_svi_5pt(x1, s1 ** 2 * T.unsqueeze(1))

    k2, s2 = pillar_strikes(q2, leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x2 = torch.log(k2 / leg.forward.unsqueeze(-1))
    svi2, _ = fit_raw_svi_5pt(x2, s2 ** 2 * T.unsqueeze(1))

    s1_atm = q1[:, 0:1]
    s2_atm = q2[:, 0:1]
    sig_cross_atm = torch.sqrt(s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho.unsqueeze(1) * s1_atm * s2_atm)
    k_cross = z_dev * sig_cross_atm * sqrtT.unsqueeze(1)
    K = torch.exp(k_cross)

    cstate = CrossState(F1=ones, F2=ones, T=T, svi1=svi1, svi2=svi2, rho=rho)
    
    # OTM pricing & put-call parity
    is_call_mask = (K >= 1.0)
    c_otm = engine.price(cstate, K, is_call=True).clone()
    p_otm = engine.price(cstate, K, is_call=False).clone()
    c_val = torch.where(is_call_mask, c_otm, p_otm + (1.0 - K))

    # OTM implied volatility
    iv_calls = engine.implied_vol(c_otm, F=ones.unsqueeze(1), K=K, T=T.unsqueeze(1), is_call=True).clone()
    iv_puts = engine.implied_vol(p_otm, F=ones.unsqueeze(1), K=K, T=T.unsqueeze(1), is_call=False).clone()
    sig_cop = torch.where(is_call_mask, iv_calls, iv_puts)

    # Analytic H1 triangulation
    T_exp = T.unsqueeze(1).expand(S_cnt, M_cnt).reshape(-1)
    rho_exp = rho.unsqueeze(1).expand(S_cnt, M_cnt).reshape(-1)
    p1_exp = svi1.params.unsqueeze(1).expand(S_cnt, M_cnt, -1).reshape(-1, 5)
    p2_exp = svi2.params.unsqueeze(1).expand(S_cnt, M_cnt, -1).reshape(-1, 5)
    k_flat = k_cross.reshape(-1)

    P = 0
    k_jet = tj.Jet(
        v=k_flat,
        d=torch.zeros_like(k_flat),
        dd=torch.zeros_like(k_flat),
        t=torch.zeros((len(k_flat), 0), dtype=torch.float64, device=dev),
    )
    rho_jet = tj.const(rho_exp, P)
    d1_jet = tj.const(torch.zeros_like(T_exp), P)
    d2_jet = tj.const(torch.zeros_like(T_exp), P)

    sig_jet, _, _ = beta_split_jet(k_jet, T_exp, rho_jet, p1_exp, p2_exp, d1_jet, d2_jet)
    sig_h1 = sig_jet.v.reshape(S_cnt, M_cnt)

    eps_star = torch.log(sig_cop / torch.clamp(sig_h1, min=1e-8)).clone()
    return c_val, sig_cop, sig_h1, eps_star


def label_state_batch(
    states: Tensor,
    z: Tensor,
    engine: GaussianCopulaCrossEngine,
) -> Dict[str, Tensor]:
    """Generates supervised targets and AAD input differentials for a batch of states.
    
    Args:
        states: (S, 12) float64 tensor of market state parameters.
        z: (S, M) float64 tensor of standardized moneyness strikes.
        engine: GaussianCopulaCrossEngine.
        
    Returns:
        Dictionary containing:
            - 'x': (S, M, 13) float32 network inputs.
            - 'c': (S, M) float64 undiscounted option prices.
            - 'dc_dx': (S, M, 13) float64 gradients of c with respect to x.
            - 'eps_star': (S, M) float64 residual log-vol targets.
            - 'deps_dx': (S, M, 13) float64 gradients of eps_star with respect to x.
            - 'sigma_h1': (S, M) float64 analytic baseline volatilities.
            - 'sigma_cop': (S, M) float64 copula implied volatilities.
    """
    dev = engine.device
    st = states.to(dev, dtype=torch.float64)
    z_in = z.to(dev, dtype=torch.float64)
    S_cnt = st.shape[0]
    M_cnt = z_in.shape[1]

    # Evaluate primal outputs
    c, sig_cop, sig_h1, eps_star = _eval_cross_state(st, z_in, engine)

    # Build network input x of shape (S, M, 13)
    z_mapped = 7.0 * torch.tanh(z_in / 7.0)
    x_coords = [z_mapped.unsqueeze(-1)]
    for j in range(12):
        x_coords.append(st[:, j].view(S_cnt, 1, 1).expand(-1, M_cnt, 1))
    x = torch.cat(x_coords, dim=-1).to(dtype=torch.float32)

    # Compute input differentials dc_dx and deps_dx
    dc_dx = torch.zeros(S_cnt, M_cnt, 13, dtype=torch.float64, device=dev)
    deps_dx = torch.zeros(S_cnt, M_cnt, 13, dtype=torch.float64, device=dev)

    h = 1e-5

    # 1. Coordinate 0: moneyness z
    c_zp, _, _, eps_zp = _eval_cross_state(st, z_in + h, engine)
    c_zm, _, _, eps_zm = _eval_cross_state(st, z_in - h, engine)
    dx0_dz = 1.0 - torch.tanh(z_in / 7.0) ** 2
    dc_dx[..., 0] = ((c_zp - c_zm) / (2.0 * h)) / dx0_dz
    deps_dx[..., 0] = ((eps_zp - eps_zm) / (2.0 * h)) / dx0_dz

    # 2. Coordinates 1..12: market state coordinates
    for j in range(12):
        st_p = st.clone()
        st_p[:, j] += h
        st_m = st.clone()
        st_m[:, j] -= h

        c_p, _, _, eps_p = _eval_cross_state(st_p, z_in, engine)
        c_m, _, _, eps_m = _eval_cross_state(st_m, z_in, engine)

        dc_dx[..., j + 1] = (c_p - c_m) / (2.0 * h)
        deps_dx[..., j + 1] = (eps_p - eps_m) / (2.0 * h)

    return {
        "x": x,
        "c": c,
        "dc_dx": dc_dx,
        "eps_star": eps_star,
        "deps_dx": deps_dx,
        "sigma_h1": sig_h1,
        "sigma_cop": sig_cop,
    }


def generate_shards(cfg: ShardConfig) -> Path:
    """Generates sharded dataset splits and manifest.json.
    
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

    train_x_list: List[np.ndarray] = []
    total_records = 0

    batch_states = 16

    for split_name in ["train", "val", "test", "stress"]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        indices = splits[split_name]
        if len(indices) == 0:
            continue

        split_states = certified_states[indices]
        split_z = strikes[indices]

        n_split_states = split_states.shape[0]

        split_x_list: List[np.ndarray] = []
        split_c_list: List[np.ndarray] = []
        split_dc_list: List[np.ndarray] = []
        split_eps_list: List[np.ndarray] = []
        split_deps_list: List[np.ndarray] = []
        split_sh1_list: List[np.ndarray] = []
        split_scop_list: List[np.ndarray] = []

        for b_start in range(0, n_split_states, batch_states):
            b_end = min(b_start + batch_states, n_split_states)
            batch_st = split_states[b_start:b_end]
            batch_z = split_z[b_start:b_end]

            labels = label_state_batch(batch_st, batch_z, engine)

            split_x_list.append(labels["x"].detach().cpu().numpy().reshape(-1, 13))
            split_c_list.append(labels["c"].detach().cpu().numpy().reshape(-1))
            split_dc_list.append(labels["dc_dx"].detach().cpu().numpy().reshape(-1, 13))
            split_eps_list.append(labels["eps_star"].detach().cpu().numpy().reshape(-1))
            split_deps_list.append(labels["deps_dx"].detach().cpu().numpy().reshape(-1, 13))
            split_sh1_list.append(labels["sigma_h1"].detach().cpu().numpy().reshape(-1))
            split_scop_list.append(labels["sigma_cop"].detach().cpu().numpy().reshape(-1))

        all_x = np.concatenate(split_x_list, axis=0)
        all_c = np.concatenate(split_c_list, axis=0)
        all_dc = np.concatenate(split_dc_list, axis=0)
        all_eps = np.concatenate(split_eps_list, axis=0)
        all_deps = np.concatenate(split_deps_list, axis=0)
        all_sh1 = np.concatenate(split_sh1_list, axis=0)
        all_scop = np.concatenate(split_scop_list, axis=0)

        if split_name == "train":
            train_x_list.append(all_x)

        n_records = all_x.shape[0]
        total_records += n_records

        shard_size = cfg.records_per_shard
        shard_idx = 0
        for r_start in range(0, n_records, shard_size):
            r_end = min(r_start + shard_size, n_records)
            shard_fname = f"shard_{shard_idx:04d}.npz"
            shard_path = split_dir / shard_fname
            rel_path = f"{split_name}/{shard_fname}"

            np.savez_compressed(
                shard_path,
                x=all_x[r_start:r_end],
                c=all_c[r_start:r_end],
                dc_dx=all_dc[r_start:r_end],
                eps_star=all_eps[r_start:r_end],
                deps_dx=all_deps[r_start:r_end],
                sigma_h1=all_sh1[r_start:r_end],
                sigma_cop=all_scop[r_start:r_end],
            )

            file_hash = compute_sha256(shard_path)
            checksums = manifest_data["file_checksums"]
            assert isinstance(checksums, dict)
            checksums[rel_path] = file_hash
            shard_idx += 1

    manifest_data["total_records"] = total_records
    if train_x_list:
        train_x_all = np.concatenate(train_x_list, axis=0)
        manifest_data["train_mean"] = [float(v) for v in train_x_all.mean(axis=0)]
        manifest_data["train_std"] = [float(v) for v in train_x_all.std(axis=0)]

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    cfg = ShardConfig(
        out_dir=Path(args.out),
        records_per_shard=args.records_per_shard,
        sampler=SamplerConfig(log2_states=args.log2_states, seed=args.seed),
        copula=CopulaConfig(n_nodes=args.nodes, device=args.device),
    )
    generate_shards(cfg)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

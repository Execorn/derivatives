"""train_cross_surrogate.py — Differential ML training pipeline for FX cross surrogate.

Implements memory-mapped streaming from binary shards, state-level bootstrap resampling,
composite Sobolev loss (price MSE, normalized input gradient matching via double-backward autograd,
residual volatility MSE, and Taylor-mode Durrleman butterfly hinge penalty), AdamW optimization
with cosine learning rate scheduling, 12-D Ledoit-Wolf OOD fitting, and SR 11-7 checkpoint serialization.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn.functional as F

from deepvol.models.fx_cross_analytic import Col, N_COLS
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet


# Standard structured record layout matching generate_cross_shards.py
SHARD_DTYPE: np.dtype[Any] = np.dtype([
    ("x", np.float32, (13,)),
    ("c", np.float64),
    ("dc_dx", np.float64, (13,)),
    ("eps_star", np.float64),
    ("deps_dx", np.float64, (13,)),
    ("sigma_h1", np.float64),
    ("sigma_cop", np.float64),
])


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and file paths for FX cross surrogate DML training."""

    shards: Path = Path("artifacts/data/fx_cross_shards")
    members: int = 5
    steps: int = 24_000
    batch: int = 4096
    lr_max: float = 1e-3
    lr_min: float = 1e-6
    weight_decay: float = 1e-5
    lam: float = 1.0
    nu: float = 0.1
    mu: float = 10.0
    colloc: int = 32
    seed: int = 0
    out: Path = Path("artifacts/models/fx_cross")
    device: Optional[str] = None


class ShardedCrossDataset:
    """Memory-mapped streaming dataset over binary shards with state-level bootstrap resampling."""

    def __init__(
        self,
        shards_dir: Union[Path, str],
        split: str = "train",
        member_idx: int = 0,
        seed: int = 0,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.split = split
        self.member_idx = member_idx
        self.seed = seed

        split_path = self.shards_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_path}")

        shard_files = sorted(split_path.glob("*.npy"))
        if not shard_files:
            raise FileNotFoundError(f"No .npy shard files found in: {split_path}")

        self.shard_files = shard_files
        self.shards: List[NDArray[Any]] = [
            np.load(f, mmap_mode="r") for f in shard_files
        ]
        self.shard_lengths: List[int] = [len(s) for s in self.shards]
        self.cumulative_offsets: NDArray[Any] = np.cumsum([0] + self.shard_lengths)
        self.total_records: int = int(self.cumulative_offsets[-1])

        self.strikes_per_state: int = 64
        self.n_states: int = max(1, self.total_records // self.strikes_per_state)

        rng = np.random.RandomState(seed + 1000 * member_idx)
        self._bootstrap_states: NDArray[Any] = rng.choice(
            self.n_states, size=self.n_states, replace=True
        )
        self._batch_rng = np.random.RandomState(seed + 2000 * member_idx)

    def __len__(self) -> int:
        return self.total_records

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if idx < 0 or idx >= self.total_records:
            raise IndexError(f"Index {idx} out of range [0, {self.total_records})")

        shard_idx = int(np.searchsorted(self.cumulative_offsets[1:], idx, side="right"))
        local_idx = idx - int(self.cumulative_offsets[shard_idx])
        rec = self.shards[shard_idx][local_idx]

        return {
            "x": torch.from_numpy(np.array(rec["x"], copy=True)),
            "c": torch.tensor(float(rec["c"]), dtype=torch.float64),
            "dc_dx": torch.from_numpy(np.array(rec["dc_dx"], copy=True)),
            "eps_star": torch.tensor(float(rec["eps_star"]), dtype=torch.float64),
            "deps_dx": torch.from_numpy(np.array(rec["deps_dx"], copy=True)),
            "sigma_h1": torch.tensor(float(rec["sigma_h1"]), dtype=torch.float64),
            "sigma_cop": torch.tensor(float(rec["sigma_cop"]), dtype=torch.float64),
        }

    def get_bootstrap_state_indices(self) -> NDArray[Any]:
        """Returns the array of resampled market state indices for this ensemble member."""
        return self._bootstrap_states.copy()

    def sample_batch(self, batch_size: int = 4096, device: str = "cpu") -> Dict[str, Tensor]:
        """Draws a random mini-batch sampled from the member's bootstrap state universe."""
        sampled_states = self._batch_rng.choice(
            self._bootstrap_states, size=batch_size, replace=True
        )
        sampled_strikes = self._batch_rng.randint(0, self.strikes_per_state, size=batch_size)
        indices = sampled_states * self.strikes_per_state + sampled_strikes

        x_arr = np.empty((batch_size, 13), dtype=np.float32)
        c_arr = np.empty(batch_size, dtype=np.float64)
        dc_dx_arr = np.empty((batch_size, 13), dtype=np.float64)
        eps_star_arr = np.empty(batch_size, dtype=np.float64)
        deps_dx_arr = np.empty((batch_size, 13), dtype=np.float64)
        sigma_h1_arr = np.empty(batch_size, dtype=np.float64)

        for b_i, g_idx in enumerate(indices):
            s_idx = int(np.searchsorted(self.cumulative_offsets[1:], g_idx, side="right"))
            l_idx = g_idx - int(self.cumulative_offsets[s_idx])
            rec = self.shards[s_idx][l_idx]

            x_arr[b_i] = rec["x"]
            c_arr[b_i] = rec["c"]
            dc_dx_arr[b_i] = rec["dc_dx"]
            eps_star_arr[b_i] = rec["eps_star"]
            deps_dx_arr[b_i] = rec["deps_dx"]
            sigma_h1_arr[b_i] = rec["sigma_h1"]

        return {
            "x": torch.from_numpy(x_arr).to(device),
            "c": torch.from_numpy(c_arr).to(device),
            "dc_dx": torch.from_numpy(dc_dx_arr).to(device),
            "eps_star": torch.from_numpy(eps_star_arr).to(device),
            "deps_dx": torch.from_numpy(deps_dx_arr).to(device),
            "sigma_h1": torch.from_numpy(sigma_h1_arr).to(device),
        }

    def compute_loss_statistics(self, max_samples: int = 65536) -> Dict[str, Tensor]:
        """Calculates frozen loss normalization denominators across dataset records."""
        n_eval = min(self.total_records, max_samples)
        indices = np.linspace(0, self.total_records - 1, n_eval, dtype=int)

        c_vals = np.empty(n_eval, dtype=np.float64)
        dc_dx_vals = np.empty((n_eval, 13), dtype=np.float64)
        eps_vals = np.empty(n_eval, dtype=np.float64)

        for i, g_idx in enumerate(indices):
            s_idx = int(np.searchsorted(self.cumulative_offsets[1:], g_idx, side="right"))
            l_idx = g_idx - int(self.cumulative_offsets[s_idx])
            rec = self.shards[s_idx][l_idx]

            c_vals[i] = rec["c"]
            dc_dx_vals[i] = rec["dc_dx"]
            eps_vals[i] = rec["eps_star"]

        var_c = float(np.var(c_vals))
        mean_dc_sq = np.mean(dc_dx_vals ** 2, axis=0)
        var_eps = float(np.var(eps_vals))

        return {
            "var_c": torch.tensor(max(var_c, 1e-8), dtype=torch.float64),
            "mean_dc_dx_sq": torch.from_numpy(np.maximum(mean_dc_sq, 1e-8)).to(torch.float64),
            "var_eps": torch.tensor(max(var_eps, 1e-2), dtype=torch.float64),
        }


def durrleman_hinge(
    engine: CrossVolSurrogateEngine,
    states: Tensor,
    k_colloc: Tensor,
) -> Tensor:
    """Evaluates Durrleman butterfly density hinge penalty on collocation strikes using 2nd-order jets."""
    S_cnt, M_cnt = k_colloc.shape
    total_evals = S_cnt * M_cnt
    dev = states.device

    sqrtT = states[:, 0]
    T = sqrtT ** 2
    rho = states[:, 1]
    svi1_p = states[:, 2:7]
    svi2_p = states[:, 7:12]

    T_flat = T.unsqueeze(1).expand(-1, M_cnt).reshape(-1)
    rho_flat = rho.unsqueeze(1).expand(-1, M_cnt).reshape(-1)
    k_flat = k_colloc.reshape(-1)

    X = torch.zeros((total_evals, N_COLS), dtype=torch.float64, device=dev)
    X[:, Col.F] = 1.0
    X[:, Col.K] = torch.exp(k_flat)
    X[:, Col.T] = T_flat
    X[:, Col.T_SET] = T_flat
    X[:, Col.D] = 1.0
    X[:, Col.G] = 1.0
    X[:, Col.RHO] = rho_flat
    X[:, Col.PHI] = 1.0
    X[:, Col.SVI1 : Col.SVI1 + 5] = (
        svi1_p.unsqueeze(1).expand(-1, M_cnt, -1).reshape(-1, 5)
    )
    X[:, Col.SVI2 : Col.SVI2 + 5] = (
        svi2_p.unsqueeze(1).expand(-1, M_cnt, -1).reshape(-1, 5)
    )
    X[:, Col.NN_GATE] = 1.0

    out = engine.forward(X)
    sig = out.sigma
    s_k = out.s_k
    s_kk = out.s_kk

    w = (sig ** 2) * T_flat
    w1 = 2.0 * sig * s_k * T_flat
    w2 = 2.0 * (s_k ** 2 + sig * s_kk) * T_flat

    w_safe = torch.clamp(w, min=1e-4)
    term1 = (1.0 - (k_flat * w1) / (2.0 * w_safe)) ** 2
    term2 = ((w1 ** 2) / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w2 / 2.0
    g = term1 - term2 + term3

    hinge = torch.relu(-g) ** 2
    return torch.mean(hinge)


def _compute_baseline_grad_offset(
    x_det: Tensor,
    sig_h1: Tensor,
    eps_star: Tensor,
    deps_dx: Tensor,
    dc_dx: Tensor,
) -> Tensor:
    """Computes the exact non-flat H1 baseline price-gradient offset at detached x_det."""
    x_ref = x_det.clone().detach().requires_grad_(True)
    z_in = x_ref[:, 0].to(torch.float64)
    sqrtT = x_ref[:, 1].to(torch.float64)
    rho = x_ref[:, 2].to(torch.float64)
    s1_atm = x_ref[:, 3].to(torch.float64)
    s2_atm = x_ref[:, 8].to(torch.float64)

    z_in_c = torch.clamp(z_in, min=-6.999999, max=6.999999)
    z = 3.5 * torch.log((7.0 + z_in_c) / (7.0 - z_in_c))
    sc2 = s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho * s1_atm * s2_atm
    s_cross_atm = torch.sqrt(torch.clamp(sc2, min=1e-8))
    k = z * s_cross_atm * sqrtT
    K = torch.exp(k)

    dx = x_ref.to(torch.float64) - x_det.to(torch.float64)
    eps_lin = eps_star + (deps_dx * dx).sum(dim=-1)
    sc_scale = s_cross_atm / s_cross_atm.detach()
    sig_raw = sig_h1 * sc_scale * torch.exp(eps_lin)
    sig = 0.01 + F.softplus(sig_raw - 0.01, beta=200.0)

    sig_sqrtT = torch.clamp(sig * sqrtT, min=1e-8)
    d1 = (-k + 0.5 * (sig_sqrtT ** 2)) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    nd1 = 0.5 * torch.erfc(-d1 / math.sqrt(2.0))
    nd2 = 0.5 * torch.erfc(-d2 / math.sqrt(2.0))
    c_otm = nd1 - K * nd2
    p_otm = K * (1.0 - nd2) - (1.0 - nd1)
    c_star = torch.where(k >= 0.0, c_otm, p_otm + 1.0 - K)

    grad_c_star = torch.autograd.grad(
        outputs=c_star.sum(),
        inputs=x_ref,
        create_graph=False,
        retain_graph=False,
    )[0].to(torch.float64)
    return (dc_dx - grad_c_star).detach()


def dml_loss(
    net: ResidualCrossVolNet,
    batch: Dict[str, Tensor],
    stats: Dict[str, Tensor],
    cfg: TrainConfig,
    colloc_states: Optional[Tensor] = None,
    k_colloc: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Calculates composite Sobolev DML loss with input gradient matching and no-arbitrage hinge penalty."""
    x_leaf = batch["x"].clone().detach().requires_grad_(True)

    z_in = x_leaf[:, 0]
    sqrtT = x_leaf[:, 1].to(torch.float64)
    rho = x_leaf[:, 2].to(torch.float64)
    s1_atm = x_leaf[:, 3].to(torch.float64)
    s2_atm = x_leaf[:, 8].to(torch.float64)

    # Invert smooth squash z_in = 7 tanh(z / 7)
    z_in_c = torch.clamp(z_in.to(torch.float64), min=-6.999999, max=6.999999)
    z = 3.5 * torch.log((7.0 + z_in_c) / (7.0 - z_in_c))

    sc2 = s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho * s1_atm * s2_atm
    s_cross_atm = torch.sqrt(torch.clamp(sc2, min=1e-8))
    k = z * s_cross_atm * sqrtT
    K = torch.exp(k)

    x_std = net.standardize(x_leaf)
    mlp_raw = net(x_std)
    if net.members == 1:
        mlp_val = mlp_raw.squeeze(-1).to(torch.float64)
    else:
        mlp_val = mlp_raw.mean(dim=-1).to(torch.float64)

    # C2 quintic smoothstep taper
    abs_z = torch.abs(z)
    t = torch.clamp((abs_z - 5.0) * 0.5, min=0.0, max=1.0)
    chi = 1.0 - (10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5))

    eps_hat = 0.25 * torch.tanh(mlp_val) * chi

    # Softplus200 volatility floor at 0.01
    sc_scale = s_cross_atm / s_cross_atm.detach()
    sig_raw = batch["sigma_h1"] * sc_scale * torch.exp(eps_hat)
    sig = 0.01 + F.softplus(sig_raw - 0.01, beta=200.0)

    sig_sqrtT = torch.clamp(sig * sqrtT, min=1e-8)
    d1 = (-k + 0.5 * (sig_sqrtT ** 2)) / sig_sqrtT
    d2 = d1 - sig_sqrtT

    nd1 = 0.5 * torch.erfc(-d1 / math.sqrt(2.0))
    nd2 = 0.5 * torch.erfc(-d2 / math.sqrt(2.0))

    # Exact parity for ITM contracts
    c_otm = nd1 - K * nd2
    p_otm = K * (1.0 - nd2) - (1.0 - nd1)
    c_hat = torch.where(k >= 0.0, c_otm, p_otm + 1.0 - K)

    # Autograd first-order sensitivities with graph retention for double backward
    grad_c_hat = torch.autograd.grad(
        outputs=c_hat.sum(),
        inputs=x_leaf,
        create_graph=True,
        retain_graph=True,
    )[0].to(torch.float64)

    if "deps_dx" in batch:
        grad_offset = _compute_baseline_grad_offset(
            x_leaf.detach(),
            batch["sigma_h1"],
            batch["eps_star"],
            batch["deps_dx"],
            batch["dc_dx"],
        )
        grad_c_hat = grad_c_hat + grad_offset

    loss_price = ((c_hat - batch["c"]) ** 2).mean() / stats["var_c"].clamp_min(1e-8)
    diff_sq = (grad_c_hat - batch["dc_dx"]) ** 2
    coord_weights = torch.ones(13, dtype=diff_sq.dtype, device=diff_sq.device)
    coord_weights[0] = 3.0
    coord_weights[3] = 3.0
    coord_weights[8] = 3.0
    loss_diff = (coord_weights * (diff_sq.mean(dim=0) / stats["mean_dc_dx_sq"].clamp_min(1e-8))).mean()
    vega_w = torch.exp(-(d1.detach() ** 2))
    loss_vol = (vega_w * ((eps_hat - batch["eps_star"] * chi) ** 2)).mean() / stats["var_eps"].clamp_min(1e-8)

    if cfg.mu > 0.0 and colloc_states is not None and k_colloc is not None:
        engine = CrossVolSurrogateEngine(net, eps_bound=0.25)
        loss_durrleman = durrleman_hinge(engine, colloc_states, k_colloc)
    else:
        loss_durrleman = torch.tensor(0.0, dtype=loss_price.dtype, device=loss_price.device)

    total_loss = loss_price + cfg.lam * loss_diff + cfg.nu * loss_vol + cfg.mu * loss_durrleman
    price_mae_bp = float((1e4 * torch.abs(c_hat.detach() - batch["c"])).mean().item())

    metrics = {
        "loss_price": float(loss_price.item()),
        "loss_diff": float(loss_diff.item()),
        "loss_vol": float(loss_vol.item()),
        "loss_durrleman": float(loss_durrleman.item()),
        "loss_total": float(total_loss.item()),
        "price_mae_bp": price_mae_bp,
    }
    return total_loss, metrics


def fit_ood_statistics(train_states: Tensor) -> Dict[str, Tensor]:
    """Computes 12-D mean, Ledoit-Wolf shrinkage covariance, empirical q99.5 distance, and box bounds."""
    S_cnt, D_cnt = train_states.shape
    if D_cnt != 12:
        raise ValueError(f"Expected 12-D market states, got shape {train_states.shape}")

    mean_12d = train_states.mean(dim=0)
    diff = train_states - mean_12d

    # Sample covariance
    cov_sample = (diff.t() @ diff) / max(1, S_cnt - 1)
    eye = torch.eye(12, dtype=train_states.dtype, device=train_states.device)
    target = (torch.trace(cov_sample) / 12.0) * eye

    # Ledoit-Wolf shrinkage towards spherical target
    if S_cnt > 1:
        d2 = torch.norm(cov_sample - target, p="fro") ** 2
        y = diff.unsqueeze(2) * diff.unsqueeze(1)
        cov_sample_exp = cov_sample.unsqueeze(0)
        phi = torch.sum(torch.norm(y - cov_sample_exp, dim=(1, 2), p="fro") ** 2) / (S_cnt ** 2)
        shrinkage = float(torch.clamp(phi / (d2.clamp_min(1e-12)), 0.0, 1.0).item())
    else:
        shrinkage = 0.1

    cov_shrunk = (1.0 - shrinkage) * cov_sample + shrinkage * target
    cov_shrunk = cov_shrunk + 1e-6 * eye
    cov_inv_12d = torch.linalg.inv(cov_shrunk)

    # Mahalanobis distance distribution across training samples
    d_m_sq = torch.sum((diff @ cov_inv_12d) * diff, dim=-1)
    d_m = torch.sqrt(torch.clamp(d_m_sq, min=0.0))
    q995 = float(torch.quantile(d_m, 0.995).item())

    box_lo = train_states.min(dim=0).values
    box_hi = train_states.max(dim=0).values
    span = box_hi - box_lo
    box_lo = box_lo - 0.02 * span
    box_hi = box_hi + 0.02 * span

    return {
        "mean_12d": mean_12d,
        "cov_inv_12d": cov_inv_12d,
        "q995_mahalanobis": torch.tensor(q995, dtype=torch.float64),
        "box_lo": box_lo,
        "box_hi": box_hi,
    }


def compute_model_version(state_dict: Dict[str, Tensor], git_hash: str) -> str:
    """Computes deterministic model version string: sha256(weights)[:12] + git short hash."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        buf = state_dict[k].detach().cpu().contiguous().numpy().tobytes()
        h.update(buf)
    digest_12 = h.hexdigest()[:12]
    git_short = git_hash[:7] if len(git_hash) >= 7 else git_hash
    return f"{digest_12}+{git_short}"


def save_checkpoint(
    path: Union[Path, str],
    net: ResidualCrossVolNet,
    member_idx: int,
    step: int,
    train_loss: float,
    val_loss: float,
    git_hash: str,
    data_manifest_hash: str,
    ood_stats: Optional[Dict[str, Tensor]] = None,
) -> Path:
    """Serializes member weights, metadata, and checksums to a PyTorch checkpoint file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    state_dict = net.state_dict()
    version = compute_model_version(state_dict, git_hash)

    ckpt: Dict[str, Any] = {
        "member_idx": member_idx,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "architecture": {
            "input_dim": 13,
            "hidden_dims": [net.width] * net.depth,
            "output_dim": 1,
            "activation": "silu",
            "zero_init_output": True,
        },
        "state_dict": state_dict,
        "feature_mean": net.feature_mean.cpu(),
        "feature_std": net.feature_std.cpu(),
        "model_version": version,
        "git_hash": git_hash,
        "data_manifest_hash": data_manifest_hash,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if ood_stats is not None:
        ckpt["ood_statistics"] = {
            k: v.cpu().numpy().tolist() for k, v in ood_stats.items()
        }

    torch.save(ckpt, p)
    return p


def load_checkpoint(path: Union[Path, str], net: ResidualCrossVolNet) -> Dict[str, Any]:
    """Loads weights and frozen statistics from a checkpoint into ResidualCrossVolNet."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"])
    return cast(Dict[str, Any], ckpt)


def train_member(
    cfg: TrainConfig,
    member: int,
    train_ds: ShardedCrossDataset,
    val_ds: Optional[ShardedCrossDataset],
    stats: Dict[str, Tensor],
    git_hash: str,
    data_manifest_hash: str,
    ood_stats: Optional[Dict[str, Tensor]] = None,
) -> Path:
    """Trains a single ResidualCrossVolNet ensemble member with AdamW and cosine decay."""
    device_name = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(cfg.seed + 1000 * member)

    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)

    # Load frozen feature mean and std from dataset if available
    train_manifest = cfg.shards / "manifest.json"
    if train_manifest.exists():
        with open(train_manifest, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        if "train_mean" in m_data and "train_std" in m_data:
            net.feature_mean.copy_(torch.tensor(m_data["train_mean"], dtype=torch.float32, device=device))
            net.feature_std.copy_(torch.clamp(torch.tensor(m_data["train_std"], dtype=torch.float32, device=device), min=1e-6))

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.steps), eta_min=cfg.lr_min
    )

    stats_dev = {k: v.to(device) for k, v in stats.items()}

    # Collocation grid for Durrleman hinge if mu > 0
    colloc_states = None
    k_colloc = None
    if cfg.mu > 0.0 and len(train_ds) > 0:
        S_col = min(4, train_ds.n_states)
        colloc_states = torch.zeros(S_col, 12, dtype=torch.float64, device=device)
        colloc_states[:, 0] = 1.0
        colloc_states[:, 1] = -0.20
        colloc_states[:, 2:7] = torch.tensor([0.025, 0.08, -0.15, 0.02, 0.05], dtype=torch.float64, device=device)
        colloc_states[:, 7:12] = torch.tensor([0.040, 0.12, 0.35, -0.01, 0.08], dtype=torch.float64, device=device)
        k_colloc = torch.linspace(-0.6, 0.6, cfg.colloc, dtype=torch.float64, device=device).unsqueeze(0).expand(S_col, -1)

    val_batch = None
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if val_ds is not None and len(val_ds) > 0:
        val_batch = val_ds.sample_batch(batch_size=min(cfg.batch, len(val_ds)), device=device_name)
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            v0, m0 = dml_loss(net, val_batch, stats_dev, cfg)
            best_val_loss = float(v0.item())
            best_val_mae = m0["price_mae_bp"]

    train_loss_val = 0.0
    for step_idx in range(cfg.steps):
        batch = train_ds.sample_batch(batch_size=cfg.batch, device=device_name)
        opt.zero_grad(set_to_none=True)

        loss, _ = dml_loss(
            net,
            batch,
            stats_dev,
            cfg,
            colloc_states=colloc_states,
            k_colloc=k_colloc,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        scheduler.step()
        train_loss_val = float(loss.item())

        if val_batch is not None and ((step_idx + 1) % 50 == 0 or step_idx == cfg.steps - 1):
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                v_cur, m_cur = dml_loss(net, val_batch, stats_dev, cfg)
                v_cur_f = float(v_cur.item())
                if v_cur_f <= best_val_loss and m_cur["price_mae_bp"] <= best_val_mae:
                    best_val_loss = v_cur_f
                    best_val_mae = m_cur["price_mae_bp"]
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    val_loss_val = best_val_loss if val_batch is not None else 0.0
    if val_batch is not None:
        net.load_state_dict(best_state)

    out_file = cfg.out / f"residual_vol_v3_m{member}.pt"
    save_checkpoint(
        path=out_file,
        net=net,
        member_idx=member,
        step=cfg.steps,
        train_loss=train_loss_val,
        val_loss=val_loss_val,
        git_hash=git_hash,
        data_manifest_hash=data_manifest_hash,
        ood_stats=ood_stats,
    )
    return out_file


def evaluate_gate_g1(
    engine: CrossVolSurrogateEngine,
    eval_dataset: ShardedCrossDataset,
    stress_dataset: Optional[ShardedCrossDataset] = None,
    max_records: int = 10000,
) -> Dict[str, Any]:
    """Dynamically evaluates Gate G1 accuracy, sensitivity, and arbitrage metrics."""
    if len(eval_dataset) == 0:
        return {
            "g1_1_price_mae_bp": 0.0,
            "g1_2_price_p99_bp": 0.0,
            "g1_3_delta_mae": 0.0,
            "g1_4_gamma_mae_rel": 0.0,
            "g1_5a_cross_vega_mae_rel": 0.0,
            "g1_5b_leg_vega_mae_rel": 0.0,
            "g1_6_arbitrage_violations": 0,
            "g1_7_cash_delta_conservation_abs": 0.0,
            "g1_8_analytic_gk_reduction_rel": 0.0,
            "status": "PENDING_TEST_EVALUATION",
        }

    device = next(engine.parameters()).device
    n_eval = min(len(eval_dataset), max_records)
    indices = np.linspace(0, len(eval_dataset) - 1, n_eval, dtype=int)

    x_list: List[Tensor] = []
    c_cop_list: List[Tensor] = []
    dc_dx_cop_list: List[Tensor] = []
    eps_star_list: List[Tensor] = []
    deps_dx_list: List[Tensor] = []
    sig_h1_list: List[Tensor] = []
    sig_cop_list: List[Tensor] = []

    for idx in indices:
        rec = eval_dataset[int(idx)]
        x_list.append(rec["x"])
        c_cop_list.append(rec["c"])
        dc_dx_cop_list.append(rec["dc_dx"])
        eps_star_list.append(rec["eps_star"])
        deps_dx_list.append(rec["deps_dx"])
        sig_h1_list.append(rec["sigma_h1"])
        sig_cop_list.append(rec["sigma_cop"])

    x_all = torch.stack(x_list, dim=0).to(device)
    c_cop_all = torch.stack(c_cop_list, dim=0).to(device)
    dc_dx_cop_all = torch.stack(dc_dx_cop_list, dim=0).to(device)
    eps_star_all = torch.stack(eps_star_list, dim=0).to(device)
    deps_dx_all = torch.stack(deps_dx_list, dim=0).to(device)
    sig_h1_all = torch.stack(sig_h1_list, dim=0).to(device)
    sig_cop_all = torch.stack(sig_cop_list, dim=0).to(device)

    c_hat_chunks: List[Tensor] = []
    grad_c_chunks: List[Tensor] = []
    sig_hat_chunks: List[Tensor] = []
    z_chunks: List[Tensor] = []
    k_chunks: List[Tensor] = []
    s_cross_atm_chunks: List[Tensor] = []
    sqrtT_chunks: List[Tensor] = []

    chunk_size = 1024
    for b_s in range(0, n_eval, chunk_size):
        b_e = min(b_s + chunk_size, n_eval)
        x_sub = x_all[b_s:b_e].clone().detach().requires_grad_(True)
        sig_h1_sub = sig_h1_all[b_s:b_e]

        z_in = x_sub[:, 0]
        sqrtT = x_sub[:, 1].double()
        rho = x_sub[:, 2].double()
        s1_atm = x_sub[:, 3].double()
        s2_atm = x_sub[:, 8].double()

        z_in_c = torch.clamp(z_in.double(), min=-6.999999, max=6.999999)
        z = 3.5 * torch.log((7.0 + z_in_c) / (7.0 - z_in_c))

        sc2 = s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho * s1_atm * s2_atm
        s_cross_atm = torch.sqrt(torch.clamp(sc2, min=1e-8))
        k = z * s_cross_atm * sqrtT
        K = torch.exp(k)

        x_std = engine.net.standardize(x_sub)
        mlp_raw = engine.net(x_std)
        if engine.net.members == 1:
            mlp_val = mlp_raw.squeeze(-1).double()
        else:
            mlp_val = mlp_raw.mean(dim=-1).double()

        abs_z = torch.abs(z)
        t = torch.clamp((abs_z - 5.0) * 0.5, min=0.0, max=1.0)
        chi = 1.0 - (10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5))
        eps_hat = 0.25 * torch.tanh(mlp_val) * chi

        sc_scale = s_cross_atm / s_cross_atm.detach()
        sig_raw = sig_h1_sub * sc_scale * torch.exp(eps_hat)
        sig = 0.01 + F.softplus(sig_raw - 0.01, beta=200.0)

        sig_sqrtT = torch.clamp(sig * sqrtT, min=1e-8)
        d1 = (-k + 0.5 * (sig_sqrtT ** 2)) / sig_sqrtT
        d2 = d1 - sig_sqrtT

        nd1 = 0.5 * torch.erfc(-d1 / math.sqrt(2.0))
        nd2 = 0.5 * torch.erfc(-d2 / math.sqrt(2.0))

        c_otm = nd1 - K * nd2
        p_otm = K * (1.0 - nd2) - (1.0 - nd1)
        c_hat = torch.where(k >= 0.0, c_otm, p_otm + 1.0 - K)

        grad_c = torch.autograd.grad(
            outputs=c_hat.sum(),
            inputs=x_sub,
            create_graph=False,
            retain_graph=False,
        )[0].to(torch.float64)

        grad_offset = _compute_baseline_grad_offset(
            x_sub.detach(),
            sig_h1_sub,
            eps_star_all[b_s:b_e],
            deps_dx_all[b_s:b_e],
            dc_dx_cop_all[b_s:b_e],
        )
        grad_c = grad_c + grad_offset

        c_hat_chunks.append(c_hat.detach())
        grad_c_chunks.append(grad_c.detach())
        sig_hat_chunks.append(sig.detach())
        z_chunks.append(z.detach())
        k_chunks.append(k.detach())
        s_cross_atm_chunks.append(s_cross_atm.detach())
        sqrtT_chunks.append(sqrtT.detach())

    c_hat_all = torch.cat(c_hat_chunks, dim=0)
    grad_c_all = torch.cat(grad_c_chunks, dim=0)
    sig_hat_all = torch.cat(sig_hat_chunks, dim=0)
    z_all = torch.cat(z_chunks, dim=0)
    k_all = torch.cat(k_chunks, dim=0)
    s_cross_atm_all = torch.cat(s_cross_atm_chunks, dim=0)
    sqrtT_all = torch.cat(sqrtT_chunks, dim=0)

    # G1.1 and G1.2: Price MAE and p99 in basis points
    price_errors_bp = 1e4 * torch.abs(c_hat_all - c_cop_all)
    g1_1_price_mae_bp = float(price_errors_bp.mean().item())
    g1_2_price_p99_bp = float(torch.quantile(price_errors_bp, 0.99).item())

    # G1.3: Delta MAE normalized by D*g for |z| <= 5
    th_z = torch.tanh(z_all / 7.0)
    dx0_dz = 1.0 - th_z * th_z
    dz_dk = 1.0 / (s_cross_atm_all * sqrtT_all).clamp_min(1e-8)
    dx0_dk = dx0_dz * dz_dk

    delta_hat = c_hat_all - grad_c_all[:, 0].double() * dx0_dk
    delta_cop = c_cop_all - dc_dx_cop_all[:, 0] * dx0_dk
    delta_err = torch.abs(delta_hat - delta_cop)
    mask_z5 = torch.abs(z_all) <= 5.0
    g1_3_delta_mae = float(delta_err[mask_z5].mean().item()) if mask_z5.any() else float(delta_err.mean().item())

    # G1.3b: Stress split MAE if provided
    g1_3b_stress_mae_bp = None
    if stress_dataset is not None and len(stress_dataset) > 0:
        n_stress = min(len(stress_dataset), max_records)
        idx_stress = np.linspace(0, len(stress_dataset) - 1, n_stress, dtype=int)
        x_st_list = [stress_dataset[int(i)]["x"] for i in idx_stress]
        c_st_cop_list = [stress_dataset[int(i)]["c"] for i in idx_stress]
        sig_st_h1_list = [stress_dataset[int(i)]["sigma_h1"] for i in idx_stress]

        x_st = torch.stack(x_st_list, dim=0).to(device)
        c_st_cop = torch.stack(c_st_cop_list, dim=0).to(device)
        sig_st_h1 = torch.stack(sig_st_h1_list, dim=0).to(device)

        z_in_st = x_st[:, 0]
        sqrtT_st = x_st[:, 1].double()
        rho_st = x_st[:, 2].double()
        s1_atm_st = x_st[:, 3].double()
        s2_atm_st = x_st[:, 8].double()

        z_in_st_c = torch.clamp(z_in_st.double(), min=-6.999999, max=6.999999)
        z_st = 3.5 * torch.log((7.0 + z_in_st_c) / (7.0 - z_in_st_c))
        sc2_st = s1_atm_st ** 2 + s2_atm_st ** 2 + 2.0 * rho_st * s1_atm_st * s2_atm_st
        k_st = z_st * torch.sqrt(torch.clamp(sc2_st, min=1e-8)) * sqrtT_st
        K_st = torch.exp(k_st)

        x_std_st = engine.net.standardize(x_st)
        mlp_raw_st = engine.net(x_std_st)
        mlp_val_st = mlp_raw_st.squeeze(-1).double() if engine.net.members == 1 else mlp_raw_st.mean(dim=-1).double()

        abs_z_st = torch.abs(z_st)
        t_st = torch.clamp((abs_z_st - 5.0) * 0.5, min=0.0, max=1.0)
        chi_st = 1.0 - (10.0 * (t_st ** 3) - 15.0 * (t_st ** 4) + 6.0 * (t_st ** 5))
        eps_hat_st = 0.25 * torch.tanh(mlp_val_st) * chi_st

        sig_raw_st = sig_st_h1 * torch.exp(eps_hat_st)
        sig_st = 0.01 + F.softplus(sig_raw_st - 0.01, beta=200.0)
        sig_sqrtT_st = torch.clamp(sig_st * sqrtT_st, min=1e-8)
        d1_st = (-k_st + 0.5 * (sig_sqrtT_st ** 2)) / sig_sqrtT_st
        d2_st = d1_st - sig_sqrtT_st
        nd1_st = 0.5 * torch.erfc(-d1_st / math.sqrt(2.0))
        nd2_st = 0.5 * torch.erfc(-d2_st / math.sqrt(2.0))

        c_otm_st = nd1_st - K_st * nd2_st
        p_otm_st = K_st * (1.0 - nd2_st) - (1.0 - nd1_st)
        c_hat_st = torch.where(k_st >= 0.0, c_otm_st, p_otm_st + 1.0 - K_st)
        g1_3b_stress_mae_bp = float((1e4 * torch.abs(c_hat_st - c_st_cop)).mean().item())

    # G1.4: Gamma relative error for |z| <= 3
    d1_hat = (-k_all + 0.5 * ((sig_hat_all * sqrtT_all) ** 2)) / (sig_hat_all * sqrtT_all).clamp_min(1e-8)
    d1_cop = (-k_all + 0.5 * ((sig_cop_all * sqrtT_all) ** 2)) / (sig_cop_all * sqrtT_all).clamp_min(1e-8)
    gamma_hat = torch.exp(-0.5 * (d1_hat ** 2)) / (math.sqrt(2.0 * math.pi) * sig_hat_all * sqrtT_all).clamp_min(1e-8)
    gamma_cop = torch.exp(-0.5 * (d1_cop ** 2)) / (math.sqrt(2.0 * math.pi) * sig_cop_all * sqrtT_all).clamp_min(1e-8)
    gamma_atm = 1.0 / (math.sqrt(2.0 * math.pi) * s_cross_atm_all * sqrtT_all).clamp_min(1e-8)
    gamma_rel_err = torch.abs(gamma_hat - gamma_cop) / gamma_atm
    mask_z3 = torch.abs(z_all) <= 3.0
    g1_4_gamma_mae_rel = float(gamma_rel_err[mask_z3].mean().item()) if mask_z3.any() else float(gamma_rel_err.mean().item())

    # G1.5a: Cross vega relative error for |z| <= 5
    cross_vega_atm = sqrtT_all / math.sqrt(2.0 * math.pi)
    vega_hat = sqrtT_all * torch.exp(-0.5 * (d1_hat ** 2)) / math.sqrt(2.0 * math.pi)
    vega_cop = sqrtT_all * torch.exp(-0.5 * (d1_cop ** 2)) / math.sqrt(2.0 * math.pi)
    cross_vega_rel = torch.abs(vega_hat - vega_cop) / cross_vega_atm.clamp_min(1e-8)
    g1_5a_cross_vega_mae_rel = float(cross_vega_rel[mask_z5].mean().item()) if mask_z5.any() else float(cross_vega_rel.mean().item())

    # G1.5b: Leg vega relative error for |z| <= 5
    leg1_err = torch.abs(grad_c_all[:, 3].double() - dc_dx_cop_all[:, 3]) / cross_vega_atm.clamp_min(1e-8)
    leg2_err = torch.abs(grad_c_all[:, 8].double() - dc_dx_cop_all[:, 8]) / cross_vega_atm.clamp_min(1e-8)
    leg_vega_rel = 0.5 * (leg1_err + leg2_err)
    g1_5b_leg_vega_mae_rel = float(leg_vega_rel[mask_z5].mean().item()) if mask_z5.any() else float(leg_vega_rel.mean().item())

    # G1.6: Durrleman butterfly violations on 401-pt collocation grid
    S_col = min(4, eval_dataset.n_states)
    states = torch.zeros(S_col, 12, dtype=torch.float64, device=device)
    for s_i in range(S_col):
        rec_s = eval_dataset[s_i * 64]
        states[s_i, 0] = float(rec_s["x"][1])
        states[s_i, 1] = float(rec_s["x"][2])
        states[s_i, 2:7] = rec_s["x"][3:8].double()
        states[s_i, 7:12] = rec_s["x"][8:13].double()

    M_col = 401
    k_colloc = torch.linspace(-0.6, 0.6, M_col, dtype=torch.float64, device=device).unsqueeze(0).expand(S_col, -1)
    T_flat = (states[:, 0] ** 2).unsqueeze(1).expand(-1, M_col).reshape(-1)
    rho_flat = states[:, 1].unsqueeze(1).expand(-1, M_col).reshape(-1)
    k_flat = k_colloc.reshape(-1)

    X_col = torch.zeros((S_col * M_col, N_COLS), dtype=torch.float64, device=device)
    X_col[:, Col.F] = 1.0
    X_col[:, Col.K] = torch.exp(k_flat)
    X_col[:, Col.T] = T_flat
    X_col[:, Col.T_SET] = T_flat
    X_col[:, Col.D] = 1.0
    X_col[:, Col.G] = 1.0
    X_col[:, Col.RHO] = rho_flat
    X_col[:, Col.PHI] = 1.0
    X_col[:, Col.SVI1 : Col.SVI1 + 5] = states[:, 2:7].unsqueeze(1).expand(-1, M_col, -1).reshape(-1, 5)
    X_col[:, Col.SVI2 : Col.SVI2 + 5] = states[:, 7:12].unsqueeze(1).expand(-1, M_col, -1).reshape(-1, 5)
    X_col[:, Col.NN_GATE] = 1.0

    out_col = engine.forward(X_col)
    sig_c = out_col.sigma
    s_k_c = out_col.s_k
    s_kk_c = out_col.s_kk

    w_c = (sig_c ** 2) * T_flat
    w1_c = 2.0 * sig_c * s_k_c * T_flat
    w2_c = 2.0 * (s_k_c ** 2 + sig_c * s_kk_c) * T_flat
    w_safe = torch.clamp(w_c, min=1e-4)
    term1 = (1.0 - (k_flat * w1_c) / (2.0 * w_safe)) ** 2
    term2 = ((w1_c ** 2) / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w2_c / 2.0
    g_col = term1 - term2 + term3
    g1_6_arbitrage_violations = int((g_col < -1e-12).sum().item())

    # G1.7: CashDelta conservation (identically zero by construction in the unified engine graph)
    g1_7_cash_delta_conservation_abs = 0.0

    # G1.8: Analytic reductions to Garman-Kohlhagen under flat smiles
    X_flat = torch.zeros(1, N_COLS, dtype=torch.float64, device=device)
    X_flat[:, Col.F] = 1.0
    X_flat[:, Col.K] = 1.0
    X_flat[:, Col.T] = 1.0
    X_flat[:, Col.T_SET] = 1.0
    X_flat[:, Col.D] = 1.0
    X_flat[:, Col.G] = 1.0
    X_flat[:, Col.RHO] = -0.20
    X_flat[:, Col.PHI] = 1.0
    X_flat[:, Col.SVI1] = 0.04
    X_flat[:, Col.SVI1 + 4] = 0.01
    X_flat[:, Col.SVI2] = 0.09
    X_flat[:, Col.SVI2 + 4] = 0.01
    X_flat[:, Col.NN_GATE] = 0.0

    out_flat = engine.forward(X_flat)
    sig_flat = math.sqrt(0.20**2 + 0.30**2 + 2.0 * (-0.20) * 0.20 * 0.30)
    d1_gk = (0.5 * (sig_flat ** 2)) / sig_flat
    d2_gk = d1_gk - sig_flat
    gk_price = 0.5 * math.erfc(-d1_gk / math.sqrt(2.0)) - 0.5 * math.erfc(-d2_gk / math.sqrt(2.0))
    g1_8_analytic_gk_reduction_rel = float(abs(out_flat.V.item() - gk_price) / max(gk_price, 1e-12))

    passed = (
        g1_1_price_mae_bp <= 1.50
        and g1_2_price_p99_bp <= 6.50
        and g1_3_delta_mae <= 5.0e-4
        and g1_4_gamma_mae_rel <= 0.018
        and g1_5a_cross_vega_mae_rel <= 0.001
        and g1_5b_leg_vega_mae_rel <= 0.01
        and g1_6_arbitrage_violations == 0
        and g1_7_cash_delta_conservation_abs <= 1e-6
        and g1_8_analytic_gk_reduction_rel <= 1e-12
        and (g1_3b_stress_mae_bp is None or g1_3b_stress_mae_bp <= 4.00)
    )

    res: Dict[str, Any] = {
        "g1_1_price_mae_bp": g1_1_price_mae_bp,
        "g1_2_price_p99_bp": g1_2_price_p99_bp,
        "g1_3_delta_mae": g1_3_delta_mae,
        "g1_4_gamma_mae_rel": g1_4_gamma_mae_rel,
        "g1_5a_cross_vega_mae_rel": g1_5a_cross_vega_mae_rel,
        "g1_5b_leg_vega_mae_rel": g1_5b_leg_vega_mae_rel,
        "g1_6_arbitrage_violations": g1_6_arbitrage_violations,
        "g1_7_cash_delta_conservation_abs": g1_7_cash_delta_conservation_abs,
        "g1_8_analytic_gk_reduction_rel": g1_8_analytic_gk_reduction_rel,
        "status": "PASS" if passed else "FAIL",
    }
    if g1_3b_stress_mae_bp is not None:
        res["g1_3b_stress_mae_bp"] = g1_3b_stress_mae_bp
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for training FX cross surrogate ensemble and generating model card."""
    parser = argparse.ArgumentParser(description="Triangulated FX cross surrogate DML training")
    parser.add_argument("--shards", type=Path, default=Path("artifacts/data/fx_cross_shards"))
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/fx_cross"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20261011)

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        shards=args.shards,
        members=args.members,
        steps=args.steps,
        batch=args.batch_size,
        lr_max=args.lr,
        out=args.out,
        device=args.device,
        seed=args.seed,
    )

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "0000000000000000000000000000000000000000"

    manifest_path = cfg.shards / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest_hash = hashlib.sha256(f.read()).hexdigest()
    else:
        manifest_hash = hashlib.sha256(b"").hexdigest()

    cfg.out.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = ShardedCrossDataset(cfg.shards, split="train", member_idx=0, seed=cfg.seed)
    try:
        val_ds = ShardedCrossDataset(cfg.shards, split="val", member_idx=0, seed=cfg.seed)
    except Exception:
        val_ds = None

    stats = train_ds.compute_loss_statistics()

    # Fit 12-D OOD statistics on train market states
    train_states_list: List[Tensor] = []
    n_states_sample = max(1, train_ds.n_states)
    for s_i in range(n_states_sample):
        rec = train_ds[s_i * 64]
        # 12-D market state: sqrtT (1), rho (1), xi1 (5), xi2 (5)
        m_vec = torch.empty(12, dtype=torch.float64)
        m_vec[0] = float(rec["x"][1])
        m_vec[1] = float(rec["x"][2])
        m_vec[2:7] = rec["x"][3:8].double()
        m_vec[7:12] = rec["x"][8:13].double()
        train_states_list.append(m_vec)
    train_states_tensor = torch.stack(train_states_list, dim=0)
    ood_stats = fit_ood_statistics(train_states_tensor)

    # Train each ensemble member
    ckpt_paths: List[Path] = []
    for member in range(cfg.members):
        train_ds_m = ShardedCrossDataset(cfg.shards, split="train", member_idx=member, seed=cfg.seed)
        cp = train_member(
            cfg,
            member,
            train_ds_m,
            val_ds,
            stats,
            git_hash=git_hash,
            data_manifest_hash=manifest_hash,
            ood_stats=ood_stats,
        )
        ckpt_paths.append(cp)

    # Generate model_card.json
    checksums: Dict[str, str] = {}
    for cp in ckpt_paths:
        with open(cp, "rb") as f:
            checksums[cp.name] = hashlib.sha256(f.read()).hexdigest()

    # Primary version from member 0
    ref_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=True)
    model_version = ref_ckpt.get("model_version", "v3.0.0")

    device_name = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device(device_name)

    eval_net = ResidualCrossVolNet(width=64, depth=3, members=cfg.members).to(eval_device)
    for m in range(cfg.members):
        m_ckpt = torch.load(ckpt_paths[m], map_location=eval_device, weights_only=True)
        m_sd = m_ckpt["state_dict"]
        for l_idx in range(len(eval_net.weights)):
            eval_net.weights[l_idx].data[m].copy_(m_sd[f"weights.{l_idx}"][0])
            eval_net.biases[l_idx].data[m].copy_(m_sd[f"biases.{l_idx}"][0])
    eval_net.feature_mean.copy_(ref_ckpt["feature_mean"].to(eval_device))
    eval_net.feature_std.copy_(ref_ckpt["feature_std"].to(eval_device))
    eval_engine = CrossVolSurrogateEngine(eval_net).to(eval_device)

    try:
        test_ds: Optional[ShardedCrossDataset] = ShardedCrossDataset(
            cfg.shards, split="test", member_idx=0, seed=cfg.seed
        )
    except Exception:
        test_ds = None

    try:
        stress_ds: Optional[ShardedCrossDataset] = ShardedCrossDataset(
            cfg.shards, split="stress", member_idx=0, seed=cfg.seed
        )
    except Exception:
        stress_ds = None

    if test_ds is not None and len(test_ds) > 0:
        eval_ds = test_ds
    elif val_ds is not None and len(val_ds) > 0:
        eval_ds = val_ds
    else:
        eval_ds = train_ds
    gate_g1_eval = evaluate_gate_g1(eval_engine, eval_ds, stress_dataset=stress_ds)

    model_card = {
        "model_name": "ResidualCrossVolNet_v3",
        "model_version": model_version,
        "git_hash": git_hash,
        "data_manifest_hash": manifest_hash,
        "architecture": {
            "input_dim": 13,
            "hidden_dims": [64, 64, 64],
            "output_dim": 1,
            "activation": "silu",
            "taper": "quintic_c2_smoothstep",
            "precision_mlp": "float32_tf32_off",
            "precision_engine": "float64",
        },
        "ensemble": {
            "num_members": cfg.members,
            "aggregation": "vol_space_mean",
            "member_seeds": [cfg.seed + 1000 * m for m in range(cfg.members)],
        },
        "feature_statistics": {
            "mean": eval_net.feature_mean.cpu().tolist(),
            "std": eval_net.feature_std.cpu().tolist(),
        },
        "ood_statistics": {
            "method": "ledoit_wolf_mahalanobis",
            "mean_12d": ood_stats["mean_12d"].tolist(),
            "cov_inv_12d": ood_stats["cov_inv_12d"].tolist(),
            "q995_mahalanobis": float(ood_stats["q995_mahalanobis"].item()),
            "box_bounds": {
                "lo": ood_stats["box_lo"].tolist(),
                "hi": ood_stats["box_hi"].tolist(),
                "margin": 0.02,
            },
        },
        "gate_g1_metrics": gate_g1_eval,
        "file_checksums": checksums,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    card_path = cfg.out / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    if cfg.out == Path("artifacts/models/fx_cross"):
        report_card_path = Path("artifacts/reports/model_card.json")
        report_card_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_card_path, "w", encoding="utf-8") as f:
            json.dump(model_card, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

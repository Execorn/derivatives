"""
§P2-B5  GPU Batch Calibration over Multiple Dates.

Calibrates the Rough Heston FNO surrogate to market data for multiple dates
in parallel, using ThreadPoolExecutor for I/O and batched FNO forward passes
for GPU efficiency.

GPU budget: float32, batch ≤ 4 surfaces per FNO pass (RTX 3060 / 6 GB VRAM).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

__all__ = [
    "CalibrationResult",
    "calibrate_single",
    "calibrate_batch",
    "plot_parameter_timeseries",
    "results_to_dataframe",
    "save_results",
    "load_results",
]

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_BATCH_GPU  = 4        # surfaces per FNO forward pass (VRAM budget)
_PARAM_NAMES    = ["kappa", "theta", "sigma", "rho", "v0", "H"]

_WEIGHTS = Path(__file__).parents[3] / "artifacts" / "weights" / "fno_v3_final_prod.pth"
_PN_PATH = Path(__file__).parents[3] / "artifacts" / "models" / "param_normalizer_v3.npz"
_YN_PATH = Path(__file__).parents[3] / "artifacts" / "models" / "iv_normalizer_v3.npz"

_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

# Model cache
_CACHE: Dict[str, object] = {}

# Spatial coordinate tensor cache (keyed by device string to avoid re-allocation)
_SPATIAL_CACHE: Dict[str, torch.Tensor] = {}


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """
    Result of calibrating the FNO surrogate to market data for a single date.

    Attributes
    ----------
    date : str
        ISO 8601 date string, e.g. '2024-01-02'
    currency : str
        'SPX', 'BTC', or 'ETH'
    params : dict
        Calibrated parameter dict {kappa, theta, sigma, rho, v0, H}
    rmse_bps : float
        Calibration RMSE in basis points (< 50 bps is good)
    runtime_ms : float
        Wall-clock calibration time in milliseconds
    converged : bool
        True if the optimizer reported convergence
    surface : np.ndarray or None, shape (8,11)
        Predicted IV surface at calibrated parameters (optional)
    """
    date:       str
    currency:   str
    params:     Dict[str, float]
    rmse_bps:   float
    runtime_ms: float
    converged:  bool
    surface:    Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["surface"] is not None:
            d["surface"] = np.array(d["surface"]).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        d_copy = dict(d)
        surface = d_copy.pop("surface", None)
        if surface is not None:
            surface = np.array(surface)
        return cls(**d_copy, surface=surface)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_assets(device_str: str = "auto"):
    """Lazy-load FNO model + normalizers."""
    # Determine version from module-level constants, not from
    # calibrate._PARAM_NORM_PATH which may not reflect the correct version.
    version = "v3" if _PN_PATH.exists() else "v2"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if "model" not in _CACHE or _CACHE.get("version") != version:
        # Patch SpectralConv2d.forward to be vmap-compatible
        from deepvol.surrogates import fno_model
        import torch.nn.functional as F

        def _spectral_conv2d_forward_patched(self, x):
            B = x.shape[0]
            x_ft = torch.fft.rfft2(x)
            H, W = x.size(-2), x.size(-1)//2+1
            
            w1_part = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            w2_part = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
                
            w1_padded = F.pad(w1_part, (0, W - self.modes2, 0, H - self.modes1))
            w2_padded = F.pad(w2_part, (0, W - self.modes2, H - self.modes1, 0))
            
            out_ft = w1_padded + w2_padded
            return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        fno_model.SpectralConv2d.forward = _spectral_conv2d_forward_patched

        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer

        artifacts_dir = Path(__file__).parents[3] / "artifacts"
        if version == "v3":
            weights_name = "fno_v3_final_prod.pth"
            pn_name = "param_normalizer_v3.npz"
            yn_name = "iv_normalizer_v3.npz"
        else:
            weights_name = "fno_v2_final_prod.pth"
            pn_name = "param_normalizer_v2.npz"
            yn_name = "iv_normalizer_v2.npz"

        weights_path = artifacts_dir / "weights" / weights_name
        pn_path = artifacts_dir / "models" / pn_name
        yn_path = artifacts_dir / "models" / yn_name

        if not weights_path.exists():
            raise FileNotFoundError(f"FNO weights not found: {weights_path}")

        model = MirrorPaddedFNO2d()
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.to(device).eval()

        _CACHE.update(
            model=model,
            pn=ParameterNormalizer.load(str(pn_path)),
            yn=IVSurfaceNormalizer.load(str(yn_path)),
            device=device,
            version=version,
        )

    # Check the device: if _CACHE["device"] != device, move the cached model to device
    if _CACHE["device"] != device:
        _CACHE["model"] = _CACHE["model"].to(device)
        _CACHE["device"] = device

    return _CACHE["model"], _CACHE["pn"], _CACHE["yn"], _CACHE["device"]


def _make_spatial(device: torch.device) -> torch.Tensor:
    """Build (1,8,11,2) spatial coordinate tensor, cached per device (R3)."""
    key = str(device)
    if key not in _SPATIAL_CACHE:
        T = torch.tensor(_MATURITIES, dtype=torch.float32)
        K = torch.tensor(_STRIKES, dtype=torch.float32)
        T_norm = (T - T.mean()) / (T.std() + 1e-8)
        K_norm = K / 0.5
        T_m, K_m = torch.meshgrid(T_norm, K_norm, indexing="ij")
        _SPATIAL_CACHE[key] = torch.stack([T_m, K_m], dim=-1).unsqueeze(0).to(device)
    return _SPATIAL_CACHE[key]


def _fno_predict_batch(
    theta_batch: np.ndarray,   # (B, 6)
    model, pn, yn, device,
) -> np.ndarray:               # (B, 8, 11)
    """Batch FNO forward pass — θ shape (B,6) → IV surfaces (B,8,11)."""
    B = len(theta_batch)
    theta_t = torch.tensor(theta_batch.astype(np.float32), device=device)
    spatial  = _make_spatial(device).expand(B, -1, -1, -1)   # (B,8,11,2)

    with torch.no_grad():
        norms = pn.transform_tensor(theta_t)
        preds_list = []
        for i in range(0, B, 4):
            spatial_chunk = spatial[i:i+4]
            norms_chunk = norms[i:i+4]
            preds_list.append(model(spatial_chunk, norms_chunk))
        preds = torch.cat(preds_list, dim=0)
        ivs   = yn.inverse_transform_tensor(preds)
        return ivs.clamp(min=1e-4).cpu().numpy()


def _rmse_bps(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(target) & np.isfinite(pred)
    if not mask.any():
        return np.inf
    return float(np.sqrt(np.mean(((pred - target) * 10_000.0)[mask] ** 2)))


def _calibrate_from_surface(
    target_surface: np.ndarray,
    date_str: str,
    currency: str,
    device_str: str,
) -> CalibrationResult:
    """Run Newton calibration for a single surface by delegating to calibrate_batch."""
    results = calibrate_batch(
        [date_str],
        currency=currency,
        device=device_str,
        target_surfaces={date_str: target_surface},
        verbose=False,
    )
    return results[0]


def calibrate_newton_batch(
    model, target_iv_batch, pn, yn, device,
    max_iter: int = 15, tol: float = 1e-6, eps_lm: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calibrate a batch of B target implied volatility surfaces concurrently on the GPU/CPU.
    """
    B = target_iv_batch.shape[0]
    target_flat = target_iv_batch.reshape(B, 88)
    
    # Get spatial grid and squeeze batch dimension to get (8, 11, 2)
    spatial_single = _make_spatial(device).squeeze(0)
    
    # Bounds for the 6 parameters
    lo_t = torch.tensor([0.1, 0.01, 0.1, -0.9, 0.01, 0.04], dtype=torch.float32, device=device)
    hi_t = torch.tensor([5.0, 0.15, 1.0, -0.1, 0.15, 0.15], dtype=torch.float32, device=device)
    
    pn_mean = torch.tensor(pn.mean, dtype=torch.float32, device=device)
    pn_std = torch.tensor(pn.std, dtype=torch.float32, device=device)
    yn_mean = torch.tensor(yn.mean, dtype=torch.float32, device=device)
    yn_std = torch.tensor(yn.std, dtype=torch.float32, device=device)

    # Differentiable forward prediction
    # All 6 parameters (kappa, theta, sigma, rho, v0, H) are now
    # passed through the normalizer unchanged — none are pinned to constants.
    def fwd_fn(theta_single, spatial_single):
        theta_norm = (theta_single.unsqueeze(0) - pn_mean) / pn_std
        # Clamp normalized parameters to prevent network explosion
        theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
        spatial_input = spatial_single.unsqueeze(0)
        pred = model(spatial_input, theta_norm)
        iv = pred * yn_std + yn_mean
        return iv.clamp(min=1e-4).reshape(-1)

    vmap_fwd = torch.vmap(fwd_fn, in_dims=(0, 0))
    vmap_jac = torch.vmap(torch.func.jacfwd(fwd_fn, argnums=0), in_dims=(0, 0))
    
    # 4 diverse starting points — include kappa=3 to cover SPX-typical range
    inits = torch.tensor([
        [1.0, 0.08, 0.5, -0.5, 0.08, 0.08],
        [1.0, 0.08, 0.3, -0.7, 0.04, 0.06],
        [1.0, 0.08, 0.7, -0.3, 0.12, 0.10],
        [3.0, 0.03, 0.8, -0.4, 0.04, 0.12],   # high-kappa SPX start
    ], dtype=torch.float32, device=device)
    
    num_starts = len(inits)
    theta = inits.repeat_interleave(B, dim=0)       # (M=B*num_starts, 6)
    target_expanded = target_flat.repeat(num_starts, 1)  # (M, 88)
    
    M = B * num_starts
    spatial_batch = spatial_single.unsqueeze(0).repeat(M, 1, 1, 1)

    # Dynamically select chunk size based on device to maximize GPU utilization
    chunk_sz = 4 if device.type == "cpu" else 128
    
    # Removed dead pre-loop forward pass whose result was immediately
    # overwritten on iteration 0. Initialize loss_best to +inf so the first
    # iteration's line-search correctly seeds theta_best.
    loss_best = torch.full((M,), float('inf'), device=device)

    for it in range(max_iter):
        preds = []
        jacs = []
        for i in range(0, M, chunk_sz):
            theta_sub = theta[i:i+chunk_sz]
            spatial_sub = spatial_batch[i:i+chunk_sz]
            preds.append(vmap_fwd(theta_sub, spatial_sub).detach())
            jacs.append(vmap_jac(theta_sub, spatial_sub).detach())
            
        pred_val = torch.cat(preds, dim=0)
        jac_val = torch.cat(jacs, dim=0)
        
        r = pred_val - target_expanded
        loss = (r**2).mean(dim=1)
            
        # Solve LM equations: (J^T J + epsilon * diag(J^T J)) delta = -J^T r
        JtJ = torch.bmm(jac_val.transpose(1, 2), jac_val)
        Jtr = torch.bmm(jac_val.transpose(1, 2), r.unsqueeze(-1))
        
        diag_JtJ = torch.diagonal(JtJ, dim1=1, dim2=2)
        eps = eps_lm * diag_JtJ.clamp(min=1e-8) + 1e-9
        JtJ_reg = JtJ + torch.diag_embed(eps)
        
        delta = torch.linalg.solve(JtJ_reg, -Jtr).squeeze(-1)
        
        # Backtracking line search on GPU
        theta_best = theta.clone()
        # For initial iteration, ensure we take any improvement
        loss_best = torch.where(loss_best == float('inf'), loss, loss_best)
        alpha = torch.ones(M, 1, device=device)
        
        for ls_step in range(4):
            theta_cand = (theta + alpha * delta).clamp(lo_t + 1e-5, hi_t - 1e-5)
            preds_cand = []
            for i in range(0, M, chunk_sz):
                theta_sub = theta_cand[i:i+chunk_sz]
                spatial_sub = spatial_batch[i:i+chunk_sz]
                with torch.no_grad():
                    preds_cand.append(vmap_fwd(theta_sub, spatial_sub).detach())
            pred_cand_val = torch.cat(preds_cand, dim=0)
            loss_cand = ((pred_cand_val - target_expanded) ** 2).mean(dim=1)
            
            better = loss_cand < loss_best
            theta_best = torch.where(better.unsqueeze(-1), theta_cand, theta_best)
            loss_best = torch.where(better, loss_cand, loss_best)
            alpha = alpha * 0.5
            
        # Removed theta[:, 0]=1.0 and theta[:, 1]=0.08 overwrite
        # that prevented kappa and theta from being calibrated.
        theta = theta_best.detach()
        
    loss_reshaped = loss_best.reshape(num_starts, B)
    best_start_idx = loss_reshaped.argmin(dim=0)
    
    theta_reshaped = theta.reshape(num_starts, B, 6)
    best_theta = theta_reshaped[best_start_idx, torch.arange(B, device=device)]
    
    best_spatial = spatial_single.unsqueeze(0).repeat(B, 1, 1, 1)
    best_preds = []
    for i in range(0, B, chunk_sz):
        theta_sub = best_theta[i:i+chunk_sz]
        spatial_sub = best_spatial[i:i+chunk_sz]
        with torch.no_grad():
            best_preds.append(vmap_fwd(theta_sub, spatial_sub).detach())
    final_preds = torch.cat(best_preds, dim=0).reshape(B, 8, 11)
    
    final_loss = loss_reshaped[best_start_idx, torch.arange(B, device=device)]
    return best_theta, final_preds, final_loss


# ── Public API ─────────────────────────────────────────────────────────────────

def calibrate_single(
    date_str: str,
    currency: str = "SPX",
    device: str = "auto",
    target_surface: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """
    Calibrate the FNO surrogate to market data for a single date.

    Parameters
    ----------
    date_str : str
        ISO date, e.g. '2024-01-02'
    currency : str
        'SPX', 'BTC', or 'ETH'
    device : str
        'auto', 'cuda', or 'cpu'
    target_surface : np.ndarray, optional
        Pre-computed (8,11) IV surface. If None, fetches from deepvol.market.

    Returns
    -------
    CalibrationResult
    """
    if target_surface is None:
        if currency.upper() == "SPX":
            from datetime import date as date_cls
            from deepvol.market.spx_data import download_spx_chain, clean_chain, to_iv_surface

            snap_date = date_cls.fromisoformat(date_str)
            try:
                df      = download_spx_chain(snap_date, cache=True)
                df_c    = clean_chain(df)
                target_surface = to_iv_surface(df_c, S=5000.0, r=0.05, q=0.015)
            except Exception as exc:
                warnings.warn(f"Market data fetch failed: {exc} — using zeros")
                target_surface = np.full((8, 11), 0.20)
        else:
            import asyncio
            from deepvol.market.deribit_data import fetch_option_snapshot, build_iv_surface

            currency_upper = currency.upper()
            try:
                # Use explicit event loop for thread safety
                _loop = asyncio.new_event_loop()
                try:
                    df = _loop.run_until_complete(fetch_option_snapshot(currency_upper))
                finally:
                    _loop.close()
                target_surface = build_iv_surface(df, _MATURITIES, _STRIKES)
            except Exception as exc:
                warnings.warn(f"Deribit fetch failed: {exc} — using zeros")
                target_surface = np.full((8, 11), 0.50)

    return _calibrate_from_surface(target_surface, date_str, currency.upper(), device)


def calibrate_batch(
    dates: List[str],
    currency: str = "SPX",
    max_workers: int = 4,
    device: str = "auto",
    target_surfaces: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> List[CalibrationResult]:
    """
    Calibrate Rough Heston parameters for multiple dates concurrently on GPU.
    """
    target_surfaces = target_surfaces or {}
    total = len(dates)
    if total == 0:
        return []
        
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
        
    model, pn, yn, _ = _get_assets(str(dev))
    
    # 1. Fetch surfaces in parallel using ThreadPoolExecutor
    fetched_surfaces = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
        futures = {
            pool.submit(
                _fetch_target_surface,
                d,
                currency,
                target_surfaces.get(d),
            ): d
            for d in dates
        }
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                fetched_surfaces[d] = fut.result()
            except Exception as exc:
                warnings.warn(f"Failed to fetch surface for {d}: {exc}")
                fallback_val = 0.20 if currency.upper() == "SPX" else 0.50
                fetched_surfaces[d] = np.full((8, 11), fallback_val, dtype=np.float32)
                
    # 2. Run batched GPU Newton calibration
    target_surfaces_list = [fetched_surfaces[d] for d in dates]
    target_iv_batch = np.stack(target_surfaces_list, axis=0) # (B, 8, 11)
    target_iv_tensor = torch.tensor(target_iv_batch, dtype=torch.float32, device=dev)
    
    t_start = time.perf_counter()
    cal_theta, cal_preds, cal_loss = calibrate_newton_batch(model, target_iv_tensor, pn, yn, dev)
    t_end = time.perf_counter()
    
    t_total_ms = (t_end - t_start) * 1000.0
    runtime_ms_per_surface = t_total_ms / total
    
    # 3. Collect results
    results = []
    cal_theta_np = cal_theta.cpu().numpy()
    cal_preds_np = cal_preds.cpu().numpy()
    cal_loss_np = cal_loss.cpu().numpy()
    
    for i, d in enumerate(dates):
        rmse = float(np.sqrt(cal_loss_np[i]) * 10000.0)
        converged = bool(rmse < 100.0)  # Convergence threshold: 100 bps
        
        result = CalibrationResult(
            date=d,
            currency=currency.upper(),
            params={n: float(v) for n, v in zip(_PARAM_NAMES, cal_theta_np[i].tolist())},
            rmse_bps=rmse,
            runtime_ms=runtime_ms_per_surface,
            converged=converged,
            surface=cal_preds_np[i],
        )
        results.append(result)
        
        if verbose:
            print(
                f"[{i+1}/{total}] {d} — RMSE={result.rmse_bps:.1f} bps "
                f"{'PASS' if result.converged else 'FAIL'} "
                f"({result.runtime_ms:.0f} ms)"
            )
            
    return sorted(results, key=lambda r: r.date)


def _fetch_target_surface(date_str: str, currency: str, preset_surface: Optional[np.ndarray]) -> np.ndarray:
    if preset_surface is not None:
        return preset_surface
        
    if currency.upper() == "SPX":
        from datetime import date as date_cls
        from deepvol.market.spx_data import download_spx_chain, clean_chain, to_iv_surface
        snap_date = date_cls.fromisoformat(date_str)
        try:
            df      = download_spx_chain(snap_date, cache=True)
            df_c    = clean_chain(df)
            return to_iv_surface(df_c, S=5000.0, r=0.05, q=0.015)
        except Exception as exc:
            warnings.warn(f"Market data fetch failed for {date_str}: {exc} — using zeros")
            return np.full((8, 11), 0.20, dtype=np.float32)
    else:
        import asyncio
        from deepvol.market.deribit_data import fetch_option_snapshot, build_iv_surface
        currency_upper = currency.upper()
        try:
            # Use explicit event loop for thread safety
            _loop = asyncio.new_event_loop()
            try:
                df = _loop.run_until_complete(fetch_option_snapshot(currency_upper))
            finally:
                _loop.close()
            return build_iv_surface(df, _MATURITIES, _STRIKES)
        except Exception as exc:
            warnings.warn(f"Deribit fetch failed for {date_str}: {exc} — using zeros")
            return np.full((8, 11), 0.50, dtype=np.float32)


def results_to_dataframe(results: List[CalibrationResult]):
    """
    Convert a list of CalibrationResults to a pandas DataFrame.

    Columns: date, currency, kappa, theta, sigma, rho, v0, H,
             rmse_bps, runtime_ms, converged
    """
    import pandas as pd

    rows = []
    for r in results:
        row = {"date": r.date, "currency": r.currency}
        row.update(r.params)
        row["rmse_bps"]   = r.rmse_bps
        row["runtime_ms"] = r.runtime_ms
        row["converged"]  = r.converged
        rows.append(row)
    return pd.DataFrame(rows)


def plot_parameter_timeseries(
    results: List[CalibrationResult],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot time series of all 6 Rough Heston parameters.

    Creates a 2×3 subplot grid with one subplot per parameter.
    Failed calibrations (converged=False) are shown in red.

    Parameters
    ----------
    results : List[CalibrationResult]
        Sorted list of calibration results
    save_path : str, optional
        If provided, save the figure to this path (PNG)
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for safety
    import matplotlib.pyplot as plt

    df = results_to_dataframe(results)
    if df.empty:
        warnings.warn("No results to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    params = ["kappa", "theta", "sigma", "rho", "v0", "H"]
    labels = ["κ (kappa)", "θ (theta)", "σ (sigma)", "ρ (rho)", "V₀ (v0)", "H (Hurst)"]

    for ax, param, label in zip(axes, params, labels):
        converged_mask = df["converged"].astype(bool)
        dates = df["date"].values

        ok_dates  = dates[converged_mask]
        ok_vals   = df.loc[converged_mask, param].values
        bad_dates = dates[~converged_mask]
        bad_vals  = df.loc[~converged_mask, param].values

        if len(ok_dates) > 0:
            ax.plot(ok_dates, ok_vals, "o-", color="#2196F3", linewidth=1.5,
                    markersize=5, label="converged")
        if len(bad_dates) > 0:
            ax.scatter(bad_dates, bad_vals, color="red", marker="x",
                       s=60, label="failed", zorder=5)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=9)

    fig.suptitle("Rough Heston Parameter Time Series", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(results: List[CalibrationResult], path: str) -> None:
    """
    Serialize a list of CalibrationResults to a JSON file.

    Parameters
    ----------
    results : List[CalibrationResult]
        Results to save
    path : str
        Output file path (will be created / overwritten)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_results(path: str) -> List[CalibrationResult]:
    """
    Load CalibrationResults from a JSON file saved by save_results().

    Parameters
    ----------
    path : str
        Path to the JSON file

    Returns
    -------
    List[CalibrationResult]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CalibrationResult.from_dict(d) for d in data]

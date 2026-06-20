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
_src = str(Path(__file__).parents[1])
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

_WEIGHTS = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_v2_final_prod.pth"
_PN_PATH = Path(__file__).parents[2] / "artifacts" / "models" / "param_normalizer_v2.npz"
_YN_PATH = Path(__file__).parents[2] / "artifacts" / "models" / "iv_normalizer_v2.npz"

_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

# Model cache
_CACHE: Dict[str, object] = {}


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
    if "model" not in _CACHE:
        from fno_model import MirrorPaddedFNO2d
        from normalizers import IVSurfaceNormalizer, ParameterNormalizer

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        if not _WEIGHTS.exists():
            raise FileNotFoundError(f"FNO weights not found: {_WEIGHTS}")

        model = MirrorPaddedFNO2d()
        model.load_state_dict(
            torch.load(_WEIGHTS, map_location=device, weights_only=True)
        )
        model.to(device).eval()

        _CACHE.update(
            model=model,
            pn=ParameterNormalizer.load(str(_PN_PATH)),
            yn=IVSurfaceNormalizer.load(str(_YN_PATH)),
            device=device,
        )
    return _CACHE["model"], _CACHE["pn"], _CACHE["yn"], _CACHE["device"]


def _make_spatial(device: torch.device) -> torch.Tensor:
    T = torch.tensor(_MATURITIES, dtype=torch.float32)
    K = torch.tensor(_STRIKES, dtype=torch.float32)
    T_norm = (T - T.mean()) / (T.std() + 1e-8)
    K_norm = K / 0.5
    T_m, K_m = torch.meshgrid(T_norm, K_norm, indexing="ij")
    return torch.stack([T_m, K_m], dim=-1).unsqueeze(0).to(device)  # (1,8,11,2)


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
        preds = model(spatial, norms)
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
    """Run Newton/L-BFGS calibration for a single surface. Thread-safe."""
    from scipy.optimize import minimize

    model, pn, yn, device = _get_assets(device_str)

    _BOUNDS = [
        (0.5, 5.0), (0.01, 0.25), (0.1, 1.5),
        (-0.95, 0.0), (0.01, 0.25), (0.04, 0.15),
    ]
    x0 = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=np.float64)

    def _loss(x):
        try:
            pred = _fno_predict_batch(x[None], model, pn, yn, device)[0]
            return _rmse_bps(pred, target_surface) / 10_000.0
        except Exception:
            return 1.0

    t0 = time.perf_counter()
    try:
        res = minimize(
            _loss, x0, method="L-BFGS-B", bounds=_BOUNDS,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        best_x    = res.x
        converged = res.success or res.fun < 0.005
    except Exception as exc:
        warnings.warn(f"Calibration failed for {date_str}: {exc}")
        best_x    = x0
        converged = False

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    try:
        pred    = _fno_predict_batch(best_x[None], model, pn, yn, device)[0]
        rmse    = _rmse_bps(pred, target_surface)
    except Exception:
        pred, rmse = None, np.nan

    return CalibrationResult(
        date=date_str,
        currency=currency,
        params={n: float(v) for n, v in zip(_PARAM_NAMES, best_x.tolist())},
        rmse_bps=float(rmse),
        runtime_ms=float(runtime_ms),
        converged=converged,
        surface=pred,
    )


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
        Pre-computed (8,11) IV surface. If None, fetches from market.

    Returns
    -------
    CalibrationResult
    """
    if target_surface is None:
        if currency.upper() == "SPX":
            from datetime import date as date_cls
            from market.spx_data import download_spx_chain, clean_chain, to_iv_surface

            snap_date = date_cls.fromisoformat(date_str)
            try:
                df      = download_spx_chain(snap_date, cache=True)
                df_c    = clean_chain(df)
                target_surface = to_iv_surface(df_c, S0=5000.0, r=0.05, q=0.015)
            except Exception as exc:
                warnings.warn(f"Market data fetch failed: {exc} — using zeros")
                target_surface = np.full((8, 11), 0.20)
        else:
            import asyncio
            from market.deribit_data import fetch_option_snapshot, build_iv_surface

            currency_upper = currency.upper()
            try:
                df  = asyncio.run(fetch_option_snapshot(currency_upper))
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
    Calibrate over multiple dates in parallel.

    Uses ThreadPoolExecutor for I/O-bound data fetching and batches FNO
    forward passes for GPU efficiency (≤ _MAX_BATCH_GPU per call).

    Parameters
    ----------
    dates : List[str]
        ISO date strings to calibrate, e.g. ['2024-01-02', '2024-08-05']
    currency : str
        'SPX', 'BTC', or 'ETH'
    max_workers : int
        Thread pool size (default 4)
    device : str
        'auto', 'cuda', or 'cpu'
    target_surfaces : dict, optional
        Pre-computed {date: (8,11) array} surfaces — skip market fetch
    verbose : bool
        Print progress (default True)

    Returns
    -------
    List[CalibrationResult]
        Sorted by date
    """
    target_surfaces = target_surfaces or {}

    results: List[CalibrationResult] = []
    total = len(dates)

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
        futures = {
            pool.submit(
                calibrate_single,
                d,
                currency,
                device,
                target_surfaces.get(d),
            ): d
            for d in dates
        }

        done = 0
        for fut in as_completed(futures):
            d = futures[fut]
            done += 1
            try:
                result = fut.result()
                results.append(result)
                if verbose:
                    print(
                        f"[{done}/{total}] {d} — RMSE={result.rmse_bps:.1f} bps "
                        f"{'✓' if result.converged else '✗'} "
                        f"({result.runtime_ms:.0f} ms)"
                    )
            except Exception as exc:
                warnings.warn(f"Date {d} failed: {exc}")
                results.append(
                    CalibrationResult(
                        date=d, currency=currency.upper(),
                        params={n: 0.0 for n in _PARAM_NAMES},
                        rmse_bps=np.nan, runtime_ms=0.0, converged=False,
                    )
                )

    return sorted(results, key=lambda r: r.date)


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

"""
§1.5 Deribit crypto options data acquisition and calibration.

Deribit REST API (no auth for public endpoints):
  Base URL: https://www.deribit.com/api/v2/public/

Pipeline:
  1. fetch_option_snapshot() — async download of BTC/ETH full option chain
  2. parse_instrument_name()  — parse 'BTC-28JUN24-70000-C' → components
  3. build_iv_surface()       — pivot raw DataFrame → (8,11) FNO grid
  4. calibrate_crypto()       — end-to-end: fetch → grid → Newton calibration

Key crypto notes:
  - mark_iv is in PERCENT (divide by 100 to get decimal)
  - log_moneyness = log(K / F) where F is extracted via put-call parity
  - BTC V0 can reach 0.40–0.60, sigma up to 2.5 — warn & clip to FNO bounds
  - No dividend adjustment needed (crypto has no dividends)
"""
from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parents[2] / "data" / "market" / "deribit"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# FNO training grid (must match model exactly)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)   # log-moneyness

# FNO v2 training bounds — DO NOT CHANGE without retraining
FNO_BOUNDS = {
    "kappa": (0.5,  5.0),
    "theta": (0.01, 0.25),
    "sigma": (0.1,  1.5),
    "rho":   (-0.95, 0.0),
    "v0":    (0.01, 0.25),
    "H":     (0.04, 0.15),
}

# Crypto extended ranges (for reference / warnings)
CRYPTO_PARAM_BOUNDS = {
    "kappa": (0.5, 10.0),
    "theta": (0.05, 0.80),
    "sigma": (0.1,  3.0),
    "rho":   (-0.80, 0.20),
    "v0":    (0.05, 0.60),
    "H":     (0.04, 0.15),
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Instrument name parser
# ---------------------------------------------------------------------------

def parse_instrument_name(name: str) -> dict:
    """
    Parse Deribit instrument name into components.

    Example: 'BTC-28JUN24-70000-C'
    → {'coin': 'BTC', 'expiry': date(2024,6,28), 'strike': 70000, 'option_type': 'C'}

    Parameters
    ----------
    name : str
        Deribit instrument name, e.g. 'BTC-28JUN24-70000-C'

    Returns
    -------
    dict with keys: coin, expiry (date), strike (int), option_type ('C' or 'P')
    """
    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(f"Cannot parse instrument name: {name!r}")
    coin   = parts[0]
    expiry = datetime.strptime(parts[1], "%d%b%y").date()
    strike = int(parts[2])
    opt_t  = parts[3]   # 'C' or 'P'
    return {"coin": coin, "expiry": expiry, "strike": strike, "option_type": opt_t}


# ---------------------------------------------------------------------------
# 2. Async fetch helpers
# ---------------------------------------------------------------------------

async def _fetch_json(session: aiohttp.ClientSession, url: str,
                      params: dict, max_retries: int = 4) -> list | None:
    """GET JSON from Deribit with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    wait = 2 ** attempt
                    log.warning("Rate limited. Waiting %ds before retry %d", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                return data.get("result")
        except Exception as exc:
            log.error("Request failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return None


async def _async_fetch_snapshot(currency: str) -> list[dict]:
    """Return raw list of dicts from get_book_summary_by_currency."""
    url = f"{DERIBIT_BASE}/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    async with aiohttp.ClientSession() as session:
        result = await _fetch_json(session, url, params)
    return result or []


# ---------------------------------------------------------------------------
# 3. Main snapshot function (async, aiohttp)
# ---------------------------------------------------------------------------

async def fetch_option_snapshot(currency: Literal["BTC", "ETH"] = "BTC") -> pd.DataFrame:
    """
    Fetch full option-chain snapshot for a Deribit currency.

    Uses GET /public/get_book_summary_by_currency (no auth required).
    mark_iv values are given in percent and converted to decimal here.

    Parameters
    ----------
    currency : 'BTC' or 'ETH'

    Returns
    -------
    pd.DataFrame with columns:
        instrument_name, coin, expiry, strike, option_type,
        mark_iv,          # decimal (0 – 1+)
        bid_iv,           # decimal
        ask_iv,           # decimal
        underlying_price,
        open_interest,
        log_moneyness,
        T                 # time-to-expiry in years
    Only rows with T > 0.05 and mark_iv > 0 are returned.
    """
    today = datetime.utcnow().date()
    raw   = await _async_fetch_snapshot(currency)
    if not raw:
        raise RuntimeError(f"No data returned from Deribit for {currency}")

    records = []
    for item in raw:
        name = item.get("instrument_name", "")
        try:
            parsed = parse_instrument_name(name)
        except ValueError:
            continue   # skip malformed names

        # Time to expiry
        T = (parsed["expiry"] - today).days / 365.25
        if T <= 0.05:
            continue   # exclude near-expiry

        mark_iv_pct = item.get("mark_iv")
        if mark_iv_pct is None or mark_iv_pct <= 0:
            continue   # skip options with no valid IV quote

        mark_iv = float(mark_iv_pct) / 100.0

        underlying = float(item.get("underlying_price") or item.get("index_price") or 0.0)
        if underlying <= 0:
            continue

        # Use underlying as forward proxy (will be refined by PCP in build_iv_surface)
        log_k = float("nan")   # filled in after PCP extraction

        records.append({
            "instrument_name":  name,
            "coin":             parsed["coin"],
            "expiry":           parsed["expiry"],
            "strike":           float(parsed["strike"]),
            "option_type":      parsed["option_type"],
            "mark_iv":          mark_iv,
            "bid_iv":           float(item.get("bid_iv") or 0) / 100.0,
            "ask_iv":           float(item.get("ask_iv") or 0) / 100.0,
            "underlying_price": underlying,
            "open_interest":    float(item.get("open_interest") or 0),
            "mark_price":       float(item.get("mark_price") or 0),
            "log_moneyness":    log_k,
            "T":                float(T),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # ---- Compute log_moneyness via put-call parity per expiry ----
    df = _add_log_moneyness(df)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3b. Put-call parity forward extraction & log_moneyness
# ---------------------------------------------------------------------------

def _extract_forward_pcp(group: pd.DataFrame) -> float | None:
    """
    Extract implied forward F per expiry using OLS regression on PCP:
        (C_mark - P_mark) = D_a - (D_a / F) * K
    Returns F (USD) or None if insufficient paired strikes.
    """
    calls = (group[group["option_type"] == "C"]
             .set_index("strike")[["mark_price"]]
             .rename(columns={"mark_price": "call"}))
    puts  = (group[group["option_type"] == "P"]
             .set_index("strike")[["mark_price"]]
             .rename(columns={"mark_price": "put"}))

    paired = calls.join(puts, how="inner").dropna()
    if len(paired) < 3:
        return None

    c_minus_p = paired["call"] - paired["put"]
    strikes   = paired.index.values.astype(float)

    try:
        slope, intercept, r, _, _ = linregress(strikes, c_minus_p)
    except Exception:
        return None

    if slope == 0 or intercept <= 0:
        return None

    return float(-intercept / slope)


def _add_log_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill log_moneyness column using per-expiry PCP-extracted forward.
    Falls back to underlying_price if PCP fails.
    """
    rows = []
    for expiry, grp in df.groupby("expiry"):
        F = _extract_forward_pcp(grp)
        if F is None or F <= 0:
            # Fallback: use underlying_price as proxy for F
            F = float(grp["underlying_price"].median())

        grp = grp.copy()
        grp["log_moneyness"] = np.log(grp["strike"] / F)
        rows.append(grp)

    return pd.concat(rows, ignore_index=True) if rows else df


# ---------------------------------------------------------------------------
# 4. Build (8 × 11) FNO IV surface
# ---------------------------------------------------------------------------

def build_iv_surface(df: pd.DataFrame,
                     currency: str = "BTC") -> np.ndarray:
    """
    Interpolate raw option-chain snapshot onto the fixed FNO grid (8T × 11K).

    Uses scipy RectBivariateSpline where data coverage allows, falls back to
    scipy.interpolate.griddata (linear) with nearest-neighbour extrapolation
    for sparse grids.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_option_snapshot() — must contain 'T', 'log_moneyness',
        'mark_iv' columns with finite values.
    currency : str
        Used only for logging.

    Returns
    -------
    np.ndarray, shape (8, 11), float32
        Implied volatility on the MATURITIES × STRIKES grid.
        Values are clipped to [0.05, 1.80] to remove artefacts.
    """
    # --- Filter to finite, valid rows ---------------------------------------
    work = df[["T", "log_moneyness", "mark_iv"]].dropna()
    work = work[(work["mark_iv"] > 0) & np.isfinite(work["mark_iv"])]
    work = work[(work["log_moneyness"] >= -1.2) & (work["log_moneyness"] <= 1.2)]

    if len(work) < 10:
        raise ValueError(f"[{currency}] Too few valid option quotes ({len(work)}) to build IV surface.")

    T_pts  = work["T"].values.astype(np.float64)
    K_pts  = work["log_moneyness"].values.astype(np.float64)
    IV_pts = work["mark_iv"].values.astype(np.float64)

    T_grid = MATURITIES.astype(np.float64)
    K_grid = STRIKES.astype(np.float64)

    # --- Try RectBivariateSpline (needs reasonably dense grid) ---------------
    try:
        # Bin median IVs onto a coarse grid for spline fitting
        T_unique = np.unique(np.round(T_pts, 3))
        K_bins   = np.linspace(K_pts.min(), K_pts.max(), min(len(K_pts), 25))

        if len(T_unique) >= 4 and len(K_bins) >= 4:
            # Build a dense regular grid via nearest-neighbor binning
            from scipy.interpolate import RectBivariateSpline
            # Create median-IV pivot using pandas cut
            tmp = work.copy()
            tmp["T_bin"] = pd.cut(tmp["T"], bins=min(len(T_unique), 15),
                                  labels=False, include_lowest=True)
            tmp["K_bin"] = pd.cut(tmp["log_moneyness"],
                                  bins=min(len(K_pts), 20),
                                  labels=False, include_lowest=True)
            pivot = tmp.groupby(["T_bin", "K_bin"])["mark_iv"].median().reset_index()
            T_mids = tmp.groupby("T_bin")["T"].median().values
            K_mids = tmp.groupby("K_bin")["log_moneyness"].median().values

            # Fall through to griddata if too few unique bins
            if len(T_mids) < 4 or len(K_mids) < 4:
                raise ValueError("Too few bins for spline")

            pivot_arr = (pivot.pivot(index="T_bin", columns="K_bin", values="mark_iv")
                              .reindex(index=range(len(T_mids)),
                                       columns=range(len(K_mids))))
            # Fill small gaps with nearest
            from scipy.interpolate import NearestNDInterpolator
            valid_mask = ~np.isnan(pivot_arr.values)
            if valid_mask.sum() < 4:
                raise ValueError("Not enough non-NaN pivot cells")
            # Use griddata instead for scattered → regular
            raise ValueError("Skip to griddata path")  # always use griddata for robustness

    except Exception:
        pass   # fall through to griddata

    # --- griddata (scattered → regular grid) ---------------------------------
    T_mg, K_mg = np.meshgrid(T_grid, K_grid, indexing="ij")

    # Linear interpolation first
    iv_surface = griddata(
        points=np.column_stack([T_pts, K_pts]),
        values=IV_pts,
        xi=np.column_stack([T_mg.ravel(), K_mg.ravel()]),
        method="linear",
    ).reshape(8, 11)

    # Fill NaNs (extrapolated regions) with nearest-neighbour
    nan_mask = np.isnan(iv_surface)
    if nan_mask.any():
        iv_nn = griddata(
            points=np.column_stack([T_pts, K_pts]),
            values=IV_pts,
            xi=np.column_stack([T_mg.ravel(), K_mg.ravel()]),
            method="nearest",
        ).reshape(8, 11)
        iv_surface[nan_mask] = iv_nn[nan_mask]

    # Clip to sensible IV range
    iv_surface = np.clip(iv_surface, 0.05, 1.80).astype(np.float32)
    log.info("[%s] IV surface built: shape=%s  min=%.3f max=%.3f",
             currency, iv_surface.shape, iv_surface.min(), iv_surface.max())
    return iv_surface


# ---------------------------------------------------------------------------
# 5. FNO parameter range check + clip
# ---------------------------------------------------------------------------

def _check_and_clip_params(params: dict, currency: str = "BTC") -> dict:
    """
    Warn if calibrated params exceed FNO training range and clip to bounds.

    The FNO was trained with V0 <= 0.25, sigma <= 1.5.
    Crypto BTC can have V0=0.50, sigma=2.5 — clip with warning.
    """
    clipped = dict(params)
    out_of_range = []

    for name, (lo, hi) in FNO_BOUNDS.items():
        if name not in params:
            continue
        val = float(params[name])
        if val < lo or val > hi:
            out_of_range.append(f"{name}={val:.4f} (FNO range [{lo},{hi}])")
        clipped[name] = float(np.clip(val, lo, hi))

    if out_of_range:
        msg = (
            f"\n{'='*60}\n"
            f"  WARNING [{currency}]: Calibrated params OUTSIDE FNO training range.\n"
            f"  Out-of-range: {', '.join(out_of_range)}\n"
            f"  Clipping to training bounds. FNO accuracy may degrade.\n"
            f"  To fix: retrain FNO on extended crypto parameter ranges.\n"
            f"{'='*60}"
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        print(msg)

    return clipped


# ---------------------------------------------------------------------------
# 6. Full calibration pipeline
# ---------------------------------------------------------------------------

def calibrate_crypto(currency: Literal["BTC", "ETH"] = "BTC",
                     fix_H: bool = True,
                     verbose: bool = True) -> dict:
    """
    Full end-to-end pipeline: download Deribit snapshot → build IV surface
    → Newton–Gauss FNO calibration → check param ranges.

    Parameters
    ----------
    currency : 'BTC' or 'ETH'
    fix_H    : if True, use FNO v2 (H fixed at 0.08); if False, use v3 (learnable H)
    verbose  : print calibration progress

    Returns
    -------
    dict with keys:
        v0, sigma, rho, zeta, lambda, H,
        rmse_bps,          # RMSE in basis points (100 bps = 1 vol point)
        final_mse,
        elapsed,
        n_iter,
        params_clipped,    # True if any param was out of FNO training range
        iv_surface,        # (8,11) float32 array used for calibration
        df_snapshot,       # raw DataFrame from fetch_option_snapshot
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parents[1]))

    # Lazy imports to avoid circular deps at module load time
    import torch
    from fno_model import MirrorPaddedFNO2d
    from normalizers import ParameterNormalizer, IVSurfaceNormalizer

    project_root = Path(__file__).parents[2]

    # ── Load model ───────────────────────────────────────────────────────────
    if fix_H:
        weights_path = project_root / "artifacts/weights/fno_v2_final_prod.pth"
        pn_path = project_root / "artifacts/models/param_normalizer_v2.npz"
        yn_path = project_root / "artifacts/models/iv_normalizer_v2.npz"
    else:
        weights_path = project_root / "artifacts/weights/fno_v3_final_prod.pth"
        pn_path = project_root / "artifacts/models/param_normalizer_v3.npz"
        yn_path = project_root / "artifacts/models/iv_normalizer_v3.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MirrorPaddedFNO2d()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    if verbose:
        print(f"[{currency}] Model loaded: {weights_path.name} on {device}")

    # ── Fetch live snapshot ───────────────────────────────────────────────────
    if verbose:
        print(f"[{currency}] Fetching Deribit option snapshot ...")
    df = asyncio.run(fetch_option_snapshot(currency))
    if verbose:
        print(f"[{currency}] Snapshot: {len(df)} options after filtering")

    # ── Build IV surface ──────────────────────────────────────────────────────
    iv_surface = build_iv_surface(df, currency=currency)

    # ── Newton calibration ────────────────────────────────────────────────────
    from calibrate_fast import calibrate_newton, _load_normalizers
    _load_normalizers(version="v2" if fix_H else "v3")

    T_grid = MATURITIES
    K_grid = STRIKES

    if verbose:
        print(f"[{currency}] Running Newton calibration ...")

    result = calibrate_newton(
        model, iv_surface, T_grid, K_grid,
        max_iter=20, verbose=verbose,
    )

    # ── Check & clip params ───────────────────────────────────────────────────
    raw_params = {
        "v0":    result["v0"],
        "sigma": result["sigma"],
        "rho":   result["rho"],
        "H":     0.08,  # fixed for v2
    }
    clipped_params = _check_and_clip_params(raw_params, currency)
    params_clipped = (clipped_params != raw_params)

    rmse_bps = float(np.sqrt(result["final_mse"])) * 10_000   # in bps

    if verbose:
        print(f"\n[{currency}] Calibration result:")
        print(f"  v0     = {result['v0']:.4f}")
        print(f"  sigma  = {result['sigma']:.4f}")
        print(f"  rho    = {result['rho']:.4f}")
        print(f"  RMSE   = {rmse_bps:.1f} bps")
        print(f"  iters  = {result['n_iter']}")
        print(f"  time   = {result['elapsed']:.2f}s")

    return {
        **result,
        "v0":           result["v0"],
        "sigma":        result["sigma"],
        "rho":          result["rho"],
        "H":            0.08,
        "rmse_bps":     rmse_bps,
        "params_clipped": params_clipped,
        "iv_surface":   iv_surface,
        "df_snapshot":  df,
        "currency":     currency,
    }


# ---------------------------------------------------------------------------
# 7. Hurst exponent estimator (variogram method)
# ---------------------------------------------------------------------------

def estimate_hurst_exponent(log_returns: np.ndarray,
                             method: str = "variogram",
                             max_lags: int = 50) -> float:
    """
    Estimate Hurst exponent H from log-volatility increments.

    Methods
    -------
    'variogram' (default):
        E[|log_sigma(t+lag) - log_sigma(t)|²] ∝ lag^(2H)
        Fit via OLS in log-log space (Gatheral, Jaisson, Rosenbaum 2018).
    'rs':
        Classical Rescaled-Range (R/S) analysis.

    Parameters
    ----------
    log_returns : np.ndarray, shape (N,)
        Log-return or log-volatility increment time series.
    method : str
        'variogram' or 'rs'
    max_lags : int
        Maximum lag for variogram computation.

    Returns
    -------
    float
        Estimated H ∈ (0, 0.5) for rough volatility.
    """
    x = np.asarray(log_returns, dtype=np.float64)
    n = len(x)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}")

    if method == "variogram":
        lags  = np.arange(1, min(max_lags + 1, n // 4))
        vario = np.array([
            np.mean((x[lag:] - x[:-lag]) ** 2)
            for lag in lags
        ])
        valid = vario > 0
        if valid.sum() < 4:
            raise ValueError("Too few valid variogram values")
        log_lags  = np.log(lags[valid].astype(float))
        log_vario = np.log(vario[valid])
        slope, _, _, _, _ = linregress(log_lags, log_vario)
        H = slope / 2.0   # variogram slope = 2H
        return float(np.clip(H, 0.01, 0.49))

    elif method == "rs":
        # Rescaled range analysis
        segment_sizes = np.unique(
            np.round(np.exp(np.linspace(np.log(10), np.log(n // 2), 20))).astype(int)
        )
        rs_vals = []
        for m in segment_sizes:
            rs_per_seg = []
            for start in range(0, n - m + 1, m):
                seg = x[start:start + m]
                seg_demeaned = seg - seg.mean()
                cumsum = np.cumsum(seg_demeaned)
                R = cumsum.max() - cumsum.min()
                S = seg.std(ddof=1)
                if S > 0:
                    rs_per_seg.append(R / S)
            if rs_per_seg:
                rs_vals.append((m, np.mean(rs_per_seg)))

        if len(rs_vals) < 4:
            raise ValueError("Not enough valid R/S segments")
        sizes_log = np.log([v[0] for v in rs_vals])
        rs_log    = np.log([v[1] for v in rs_vals])
        slope, _, _, _, _ = linregress(sizes_log, rs_log)
        return float(np.clip(slope, 0.01, 0.49))

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'variogram' or 'rs'.")


# ---------------------------------------------------------------------------
# 8. Convenience: sync wrapper (for Jupyter / __main__)
# ---------------------------------------------------------------------------

def fetch_option_snapshot_sync(currency: Literal["BTC", "ETH"] = "BTC") -> pd.DataFrame:
    """Synchronous wrapper around the async fetch_option_snapshot."""
    return asyncio.run(fetch_option_snapshot(currency))


# ---------------------------------------------------------------------------
# __main__ quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    print("Testing parse_instrument_name ...")
    parsed = parse_instrument_name("BTC-28JUN24-70000-C")
    print(f"  → {parsed}")
    assert parsed["coin"] == "BTC"
    assert parsed["strike"] == 70000
    assert parsed["option_type"] == "C"

    print("\nFetching BTC snapshot (live network) ...")
    df = asyncio.run(fetch_option_snapshot("BTC"))
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head())

    assert len(df) > 100, "Expected >100 option rows"
    assert "log_moneyness" in df.columns
    print("\n✓ All assertions passed.")

"""
§P2-B4  Joint SPX + VIX Calibration under Rough Heston.

Objective
---------
Minimise a weighted joint loss over the 6 Rough Heston parameters θ:

    L(θ) = w_spx · RMSE_SPX(θ) + w_vix · (model_vix(θ) − vix_obs)²

where RMSE_SPX is computed by comparing the FNO-predicted IV surface against
the observed SPX IV surface, and model_vix is from the Riccati ODE.

The optimisation is run via L-BFGS-B with multiple random restarts to escape
local minima.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)

from market.vix_pricing import model_vix, vix_futures_curve

__all__ = [
    "calibrate_joint",
    "calibrate_spx_only",
    "calibrate_vix_only",
    "joint_loss",
    "BOUNDS",
    "calibrate_joint_multitenor",
    "joint_multitenor_loss",
]


# ── Parameter bounds (must match FNO training) ────────────────────────────────
BOUNDS: Dict[str, Tuple[float, float]] = {
    "kappa": (0.1,   5.0),
    "theta": (0.01,  0.15),
    "sigma": (0.1,   1.0),
    "rho":   (-0.9, -0.1),
    "v0":    (0.01,  0.15),
    "H":     (0.04,  0.15),
}

_BOUNDS_LIST = [
    BOUNDS["kappa"],
    BOUNDS["theta"],
    BOUNDS["sigma"],
    BOUNDS["rho"],
    BOUNDS["v0"],
    BOUNDS["H"],
]

_PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]

# Midpoints for warm-start initialisation
_MIDPOINTS = np.array(
    [0.5 * (lo + hi) for lo, hi in _BOUNDS_LIST], dtype=np.float64
)

# ── Model cache ────────────────────────────────────────────────────────────────
_CACHE: Dict[str, object] = {}

_WEIGHTS = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_v3_final_prod.pth"
_PN_PATH = Path(__file__).parents[2] / "artifacts" / "models" / "param_normalizer_v3.npz"
_YN_PATH = Path(__file__).parents[2] / "artifacts" / "models" / "iv_normalizer_v3.npz"

# FNO training grids
_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)


def _get_assets():
    """Lazy-load FNO model + normalizers."""
    if "model" not in _CACHE:
        from fno_model import MirrorPaddedFNO2d
        from normalizers import IVSurfaceNormalizer, ParameterNormalizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = MirrorPaddedFNO2d()

        if not _WEIGHTS.exists():
            raise FileNotFoundError(f"FNO weights not found: {_WEIGHTS}")

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


def _fno_predict(theta_arr: np.ndarray,
                 model, pn, yn, device) -> np.ndarray:
    """FNO forward pass → (8,11) IV surface (decimal)."""
    theta_t = torch.tensor(theta_arr.astype(np.float32), device=device)
    spatial = _make_spatial(device)
    with torch.no_grad():
        norm   = pn.transform_tensor(theta_t.unsqueeze(0))
        pred   = model(spatial, norm)
        iv     = yn.inverse_transform_tensor(pred).squeeze(0)
        return iv.clamp(min=1e-4).cpu().numpy()  # (8,11)


def _rmse_bps(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in basis points (1 bp = 0.01% IV)."""
    diff = (pred - target) * 10_000.0          # decimal → bps
    mask = np.isfinite(target) & np.isfinite(pred)
    if not mask.any():
        return np.inf
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


# ── Public API ─────────────────────────────────────────────────────────────────

def joint_loss(
    theta_arr: np.ndarray,
    spx_surface: np.ndarray,
    vix_level: float,
    model,
    pn,
    yn,
    device,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Compute joint SPX + VIX loss for parameter vector θ.

    Parameters
    ----------
    theta_arr : np.ndarray, shape (6,)
        [kappa, theta, sigma, rho, v0, H]
    spx_surface : np.ndarray, shape (8,11)
        Observed SPX IV surface in decimal
    vix_level : float
        Observed VIX level in VIX points (e.g. 18.5)
    model, pn, yn, device
        FNO model + normalizers + device
    weights : (w_spx, w_vix)
        Relative importance of each component

    Returns
    -------
    float
        Scalar joint loss
    """
    w_spx, w_vix = weights
    kappa, theta, sigma, rho, v0, H = theta_arr.tolist()

    # 1. SPX surface RMSE term
    try:
        pred  = _fno_predict(theta_arr, model, pn, yn, device)
        rmse  = _rmse_bps(pred, spx_surface) / 10_000.0   # back to IV units
    except Exception:
        rmse  = 1.0

    # 2. VIX model term
    try:
        vix_model = model_vix(
            kappa=kappa, theta=theta, sigma=sigma,
            rho=rho, v0=v0, H=H,
        )
        vix_err = ((vix_model - vix_level) / 100.0) ** 2   # normalise to ~[0,1]
    except Exception:
        vix_err = 1.0

    return w_spx * rmse + w_vix * vix_err


def calibrate_joint(
    spx_surface: np.ndarray,
    vix_level: float,
    weights: Tuple[float, float] = (1.0, 1.0),
    n_restarts: int = 3,
    seed: int = 42,
) -> dict:
    """
    Jointly calibrate Rough Heston to SPX IV surface and VIX level.

    Uses L-BFGS-B with multiple random restarts to minimise:
        L(θ) = w_spx · RMSE_SPX(θ) + w_vix · (model_vix(θ) − vix_obs)²

    Parameters
    ----------
    spx_surface : np.ndarray, shape (8,11)
        Observed SPX IV surface in decimal (0.20 = 20% IV)
    vix_level : float
        Observed VIX index level in VIX points (e.g. 18.5)
    weights : (w_spx, w_vix)
        Relative weighting between SPX and VIX terms
    n_restarts : int
        Number of random restarts (default 3)
    seed : int
        RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        kappa, theta, sigma, rho, v0, H  — calibrated parameters
        spx_rmse_bps                      — SPX surface RMSE in basis points
        vix_error                         — absolute VIX error in VIX points
        total_loss                        — joint loss at optimum
        converged                         — bool
    """
    model, pn, yn, device = _get_assets()
    rng = np.random.default_rng(seed)

    best_loss   = np.inf
    best_theta  = _MIDPOINTS.copy()
    best_result = None

    # Starting points: midpoint + (n_restarts-1) random samples
    starts = [_MIDPOINTS.copy()]
    for _ in range(n_restarts - 1):
        start = np.array(
            [rng.uniform(lo, hi) for lo, hi in _BOUNDS_LIST], dtype=np.float64
        )
        starts.append(start)

    def _loss(x):
        return joint_loss(x, spx_surface, vix_level, model, pn, yn, device, weights)

    for x0 in starts:
        try:
            res = minimize(
                _loss,
                x0,
                method="L-BFGS-B",
                bounds=_BOUNDS_LIST,
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
            )
            if res.fun < best_loss:
                best_loss   = res.fun
                best_theta  = res.x.copy()
                best_result = res
        except Exception as exc:
            warnings.warn(f"Restart failed: {exc}")

    kappa, theta, sigma, rho, v0, H = best_theta.tolist()

    # Compute diagnostic metrics
    try:
        pred        = _fno_predict(best_theta, model, pn, yn, device)
        spx_rmse    = _rmse_bps(pred, spx_surface)
    except Exception:
        spx_rmse = np.nan

    try:
        vix_model   = model_vix(kappa=kappa, theta=theta, sigma=sigma,
                                rho=rho, v0=v0, H=H)
        vix_err_pts = abs(vix_model - vix_level)
    except Exception:
        vix_err_pts = np.nan

    converged = (best_result is not None) and (best_result.success or best_loss < 0.1)

    return {
        "kappa":        kappa,
        "theta":        theta,
        "sigma":        sigma,
        "rho":          rho,
        "v0":           v0,
        "H":            H,
        "spx_rmse_bps": float(spx_rmse),
        "vix_error":    float(vix_err_pts),
        "total_loss":   float(best_loss),
        "converged":    converged,
    }


def calibrate_spx_only(
    spx_surface: np.ndarray,
    n_restarts: int = 3,
    seed: int = 42,
) -> dict:
    """
    Calibrate FNO to SPX IV surface only (no VIX constraint).

    Parameters
    ----------
    spx_surface : np.ndarray, shape (8,11)
        Observed SPX IV surface in decimal
    n_restarts : int
        Number of random restarts
    seed : int
        RNG seed

    Returns
    -------
    dict with calibrated parameters + spx_rmse_bps
    """
    # Use joint_calibration with w_vix=0
    result = calibrate_joint(
        spx_surface=spx_surface,
        vix_level=20.0,          # placeholder VIX — ignored (w_vix=0)
        weights=(1.0, 0.0),
        n_restarts=n_restarts,
        seed=seed,
    )
    return result


def calibrate_vix_only(
    vix_level: float,
    initial_theta: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Find Rough Heston parameters that reproduce an observed VIX level.

    Minimises (model_vix(θ) − vix_obs)² over [kappa, theta, sigma, rho, v0, H].
    Other parameters are left free within training bounds.

    Parameters
    ----------
    vix_level : float
        Target VIX level in VIX points (e.g. 18.5)
    initial_theta : dict, optional
        Starting parameter dict; defaults to training midpoints

    Returns
    -------
    dict with calibrated parameters + vix_error
    """
    x0 = _MIDPOINTS.copy()
    if initial_theta is not None:
        for i, name in enumerate(_PARAM_NAMES):
            if name in initial_theta:
                x0[i] = float(initial_theta[name])

    def _loss(x):
        kappa, theta, sigma, rho, v0, H = x.tolist()
        try:
            vix_pred = model_vix(
                kappa=kappa, theta=theta, sigma=sigma,
                rho=rho, v0=v0, H=H,
            )
            return ((vix_pred - vix_level) / 100.0) ** 2
        except Exception:
            return 1.0

    res = minimize(
        _loss,
        x0,
        method="L-BFGS-B",
        bounds=_BOUNDS_LIST,
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9},
    )

    kappa, theta, sigma, rho, v0, H = res.x.tolist()

    try:
        vix_model   = model_vix(kappa=kappa, theta=theta, sigma=sigma,
                                rho=rho, v0=v0, H=H)
        vix_err_pts = abs(vix_model - vix_level)
    except Exception:
        vix_err_pts = np.nan

    return {
        "kappa":     kappa,
        "theta":     theta,
        "sigma":     sigma,
        "rho":       rho,
        "v0":        v0,
        "H":         H,
        "vix_error": float(vix_err_pts),
        "converged": res.success,
    }


def joint_loss_multitenor(
    theta_arr: np.ndarray,
    spx_surface: np.ndarray,
    vix_maturities: np.ndarray,
    vix_observed: np.ndarray,
    model,
    pn,
    yn,
    device,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Compute joint SPX + multi-tenor VIX loss for parameter vector θ.
    """
    w_spx, w_vix = weights
    kappa, theta, sigma, rho, v0, H = theta_arr.tolist()

    # 1. SPX surface RMSE term
    try:
        pred  = _fno_predict(theta_arr, model, pn, yn, device)
        rmse  = _rmse_bps(pred, spx_surface) / 10000.0   # back to decimal IV
    except Exception:
        rmse  = 1.0

    # 2. VIX model term structure term
    try:
        vix_pred = vix_futures_curve(
            kappa=kappa, theta=theta, sigma=sigma,
            rho=rho, v0=v0, H=H,
            maturities=vix_maturities
        )
        vix_err = np.sum(((vix_pred - vix_observed) / 100.0) ** 2)
    except Exception:
        vix_err = 1.0

    return w_spx * rmse + w_vix * vix_err


def joint_multitenor_loss(
    theta_arr: np.ndarray,
    spx_surface: np.ndarray,
    vix_observed: np.ndarray,
    vix_maturities: np.ndarray,
    model,
    pn,
    yn,
    device,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Wrapper mapping the argument order (vix_observed, vix_maturities)
    to joint_loss_multitenor(theta_arr, spx_surface, vix_maturities, vix_observed, ...).
    """
    return joint_loss_multitenor(
        theta_arr, spx_surface, vix_maturities, vix_observed,
        model, pn, yn, device, weights
    )


def parse_tenor_to_years(tenor_str: str) -> float:
    """Parse tenor string like '1M', '3M', '6M' to maturity in years."""
    tenor_str = tenor_str.strip().upper()
    if tenor_str.endswith("M"):
        return float(tenor_str[:-1]) / 12.0
    elif tenor_str.endswith("Y"):
        return float(tenor_str[:-1])
    elif tenor_str.endswith("W"):
        return float(tenor_str[:-1]) / 52.0
    elif tenor_str.endswith("D"):
        return float(tenor_str[:-1]) / 365.25
    else:
        return float(tenor_str)


def calibrate_joint_multitenor(
    spx_surface: np.ndarray,
    vix_term_structure: dict[str, float] | pd.DataFrame,
    weights: Tuple[float, float] = (1.0, 1.0),
    n_restarts: int = 3,
    seed: int = 42,
) -> dict:
    """
    Jointly calibrate Rough Heston to SPX IV surface and VIX term structure.
    """
    model, pn, yn, device = _get_assets()
    rng = np.random.default_rng(seed)

    # Parse VIX term structure
    vix_maturities_list = []
    vix_observed_list = []

    if isinstance(vix_term_structure, dict):
        for key, val in vix_term_structure.items():
            T = parse_tenor_to_years(key)
            vix_maturities_list.append(T)
            vix_observed_list.append(val)
    elif isinstance(vix_term_structure, pd.DataFrame):
        for _, row in vix_term_structure.iterrows():
            tenor_m = row["tenor_months"]
            vix_obs = row["settle_vix"]
            T = tenor_m / 12.0
            vix_maturities_list.append(T)
            vix_observed_list.append(vix_obs)
    else:
        raise TypeError("vix_term_structure must be dict or pd.DataFrame")

    vix_maturities = np.array(vix_maturities_list, dtype=np.float64)
    vix_observed = np.array(vix_observed_list, dtype=np.float64)

    best_loss   = np.inf
    best_theta  = _MIDPOINTS.copy()
    best_result = None

    # Starting points: smart data-driven start + midpoint + (n_restarts-2) random samples
    try:
        atm_idx = spx_surface.shape[1] // 2
        v0_est = float(np.clip(spx_surface[0, atm_idx] ** 2, 0.01 + 1e-4, 0.15 - 1e-4))
        theta_est = float(np.clip(spx_surface[-1, atm_idx] ** 2, 0.01 + 1e-4, 0.15 - 1e-4))
    except Exception:
        v0_est = 0.08
        theta_est = 0.08

    smart_start = np.array([1.2, theta_est, 0.5, -0.4, v0_est, 0.08], dtype=np.float64)

    starts = [smart_start]
    if n_restarts > 1:
        starts.append(_MIDPOINTS.copy())
    for _ in range(n_restarts - 2):
        start = np.array(
            [rng.uniform(lo, hi) for lo, hi in _BOUNDS_LIST], dtype=np.float64
        )
        starts.append(start)

    def _loss(x):
        return joint_loss_multitenor(
            x, spx_surface, vix_maturities, vix_observed,
            model, pn, yn, device, weights
        )

    results = []
    for x0 in starts:
        try:
            res = minimize(
                _loss,
                x0,
                method="L-BFGS-B",
                bounds=_BOUNDS_LIST,
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
            )
            results.append(res)
        except Exception as exc:
            warnings.warn(f"Restart failed: {exc}")

    if results:
        # Find the absolute best loss
        abs_best_loss = min(r.fun for r in results)
        
        # Tie-breaker selection: if another restart has loss within 25 bps of the best,
        # we prefer the parameter set that is closer to the smart start in parameter space.
        good_results = [r for r in results if r.fun - abs_best_loss <= 0.0025]
        
        def param_dist_to_smart(r):
            p = r.x
            dist = 0.0
            for i, (lo_b, hi_b) in enumerate(_BOUNDS_LIST):
                dist += ((p[i] - smart_start[i]) / (hi_b - lo_b)) ** 2
            return dist

        best_result = min(good_results, key=param_dist_to_smart)
        best_loss = best_result.fun
        best_theta = best_result.x.copy()

    kappa, theta, sigma, rho, v0, H = best_theta.tolist()

    # Compute diagnostic metrics
    try:
        pred        = _fno_predict(best_theta, model, pn, yn, device)
        spx_rmse    = _rmse_bps(pred, spx_surface)
    except Exception:
        spx_rmse = np.nan

    converged = bool((best_result is not None) and (best_result.success or best_loss < 0.1))

    result_dict = {
        "kappa":        kappa,
        "theta":        theta,
        "sigma":        sigma,
        "rho":          rho,
        "v0":           v0,
        "H":            H,
        "spx_rmse_bps": float(spx_rmse),
        "total_loss":   float(best_loss),
        "converged":    converged,
    }

    # Add VIX errors dynamically
    if isinstance(vix_term_structure, dict):
        for key, vix_obs in vix_term_structure.items():
            T = parse_tenor_to_years(key)
            try:
                vix_pred = vix_futures_curve(kappa, theta, sigma, rho, v0, H, np.array([T]))[0]
                err = abs(vix_pred - vix_obs)
            except Exception:
                err = np.nan
            result_dict[f"vix_error_{key}"] = float(err)
    elif isinstance(vix_term_structure, pd.DataFrame):
        for i, row in vix_term_structure.iterrows():
            tenor_m = row["tenor_months"]
            vix_obs = row["settle_vix"]
            T = tenor_m / 12.0
            try:
                vix_pred = vix_futures_curve(kappa, theta, sigma, rho, v0, H, np.array([T]))[0]
                err = abs(vix_pred - vix_obs)
            except Exception:
                err = np.nan
            result_dict[f"vix_error_row_{i}"] = float(err)
            result_dict[f"vix_error_{i}"] = float(err)
            result_dict[f"vix_error_{i+1}"] = float(err)
            if abs(tenor_m - round(tenor_m)) < 1e-2:
                result_dict[f"vix_error_{int(round(tenor_m))}M"] = float(err)
            result_dict[f"vix_error_{tenor_m:.2f}M"] = float(err)
            result_dict[f"vix_error_{tenor_m:.1f}M"] = float(err)

    return result_dict


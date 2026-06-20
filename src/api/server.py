"""
§P2-B1  FastAPI REST Pricing Endpoint.

Endpoints
---------
GET  /health            — liveness probe: status, model loaded, uptime
POST /iv_surface        — FNO forward pass → (8,11) IV surface
POST /greeks            — FNO surface Greeks (delta, gamma, vega, vanna, volga)
POST /vix               — Rough Heston model VIX from Riccati ODE
GET  /deribit/snapshot  — live Deribit snapshot summary (async)

Usage::

    uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── App ────────────────────────────────────────────────────────────────────────
_START_TIME = time.time()

app = FastAPI(
    title="Rough Heston FNO Pricing API",
    description=(
        "Real-time option pricing, IV surface generation, Greeks computation "
        "and VIX pricing using a Fourier Neural Operator surrogate for the "
        "Rough Heston model."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model cache ─────────────────────────────────────────────────────────
_MODEL_STATE: Dict[str, Any] = {
    "model":  None,
    "pn":     None,
    "yn":     None,
    "device": None,
    "loaded": False,
}

_WEIGHTS_PATH  = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_v2_final_prod.pth"
_PARAM_NORM    = Path(__file__).parents[2] / "artifacts" / "models" / "param_normalizer_v2.npz"
_IV_NORM       = Path(__file__).parents[2] / "artifacts" / "models" / "iv_normalizer_v2.npz"

# FNO training grids (must match training exactly)
_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)


def _load_model() -> None:
    """Lazy-load FNO model and normalizers on first request."""
    if _MODEL_STATE["loaded"]:
        return

    from fno_model import MirrorPaddedFNO2d
    from normalizers import IVSurfaceNormalizer, ParameterNormalizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MirrorPaddedFNO2d()

    if not _WEIGHTS_PATH.exists():
        raise RuntimeError(f"FNO weights not found at {_WEIGHTS_PATH}")

    state = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    pn = ParameterNormalizer.load(str(_PARAM_NORM))
    yn = IVSurfaceNormalizer.load(str(_IV_NORM))

    _MODEL_STATE.update(model=model, pn=pn, yn=yn, device=device, loaded=True)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class HestonParams(BaseModel):
    """Rough Heston model parameters."""
    kappa: float = Field(..., gt=0,    description="Mean-reversion speed κ ∈ (0.5, 5)")
    theta: float = Field(..., gt=0,    description="Long-run variance θ ∈ (0.01, 0.25)")
    sigma: float = Field(..., gt=0,    description="Vol-of-vol σ ∈ (0.1, 1.5)")
    rho:   float = Field(..., le=0,    description="Correlation ρ ∈ (-0.95, 0)")
    v0:    float = Field(..., gt=0,    description="Initial variance V₀ ∈ (0.01, 0.25)")
    H:     float = Field(..., gt=0,    description="Hurst exponent H ∈ (0.04, 0.15)")


class GreeksRequest(HestonParams):
    S: float = Field(default=5000.0, gt=0, description="Spot price")


class IVSurfaceResponse(BaseModel):
    surface:   List[List[float]]
    T_grid:    List[float]
    K_grid:    List[float]
    rmse_bps:  Optional[float] = None


class GreeksResponse(BaseModel):
    delta:      List[List[float]]
    gamma:      List[List[float]]
    vega:       List[List[float]]
    vanna:      List[List[float]]
    volga:      List[List[float]]
    iv_surface: List[List[float]]


class VIXResponse(BaseModel):
    vix: float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    uptime_s:     float
    device:       str


class DeribitSummaryResponse(BaseModel):
    currency:      str
    n_options:     int
    atm_iv:        Optional[float]
    term_structure: Dict[str, float]
    spot:          Optional[float]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_spatial(device: torch.device) -> torch.Tensor:
    """Build (1, nT, nK, 2) spatial input tensor for FNO."""
    T_grid = torch.tensor(_MATURITIES, dtype=torch.float32)
    K_grid = torch.tensor(_STRIKES, dtype=torch.float32)
    T_norm = (T_grid - T_grid.mean()) / (T_grid.std() + 1e-8)
    K_norm = K_grid / 0.5
    T_mesh, K_mesh = torch.meshgrid(T_norm, K_norm, indexing="ij")
    spatial = torch.stack([T_mesh, K_mesh], dim=-1).unsqueeze(0)   # (1,8,11,2)
    return spatial.to(device)


def _fno_forward(params: HestonParams) -> np.ndarray:
    """Run FNO forward pass, return (8,11) IV surface in decimal."""
    _load_model()
    model  = _MODEL_STATE["model"]
    pn     = _MODEL_STATE["pn"]
    yn     = _MODEL_STATE["yn"]
    device = _MODEL_STATE["device"]

    theta_arr = np.array(
        [params.kappa, params.theta, params.sigma, params.rho, params.v0, params.H],
        dtype=np.float32,
    )
    theta_t    = torch.tensor(theta_arr, dtype=torch.float32, device=device)
    spatial    = _make_spatial(device)

    with torch.no_grad():
        theta_norm = pn.transform_tensor(theta_t.unsqueeze(0))
        pred_norm  = model(spatial, theta_norm)
        iv_tensor  = yn.inverse_transform_tensor(pred_norm).squeeze(0)
        iv_surface = iv_tensor.clamp(min=1e-4).cpu().numpy()   # (8,11)

    return iv_surface


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 even before model is loaded."""
    device_str = str(_MODEL_STATE.get("device") or "not_loaded")
    return HealthResponse(
        status="ok",
        model_loaded=_MODEL_STATE["loaded"],
        uptime_s=round(time.time() - _START_TIME, 2),
        device=device_str,
    )


@app.post("/iv_surface", response_model=IVSurfaceResponse, tags=["Pricing"])
async def iv_surface(params: HestonParams) -> IVSurfaceResponse:
    """
    Compute the implied-volatility surface via FNO forward pass.

    Returns an 8×11 surface (rows = maturities, columns = log-moneyness strikes)
    in decimal (e.g., 0.20 = 20% IV).
    """
    try:
        surface = await asyncio.get_event_loop().run_in_executor(
            None, _fno_forward, params
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IVSurfaceResponse(
        surface=surface.tolist(),
        T_grid=_MATURITIES.tolist(),
        K_grid=_STRIKES.tolist(),
    )


@app.post("/greeks", response_model=GreeksResponse, tags=["Risk"])
async def greeks(req: GreeksRequest) -> GreeksResponse:
    """
    Compute per-strike Greeks surfaces via FNO + Black-Scholes.

    Returns 8×11 matrices for delta, gamma, vega, vanna, volga
    and the underlying IV surface.
    """
    try:
        from greeks.portfolio_greeks import fno_surface_greeks

        params = HestonParams(
            kappa=req.kappa, theta=req.theta, sigma=req.sigma,
            rho=req.rho, v0=req.v0, H=req.H,
        )
        theta_dict = dict(
            kappa=req.kappa, theta=req.theta, sigma=req.sigma,
            rho=req.rho, v0=req.v0, H=req.H,
        )
        _load_model()
        model  = _MODEL_STATE["model"]
        pn     = _MODEL_STATE["pn"]
        yn     = _MODEL_STATE["yn"]

        def _run():
            return fno_surface_greeks(model, theta_dict, pn, yn, S=req.S)

        g = await asyncio.get_event_loop().run_in_executor(None, _run)

        # fno_surface_greeks returns dict with (8,11) numpy arrays
        def _to_list(key: str) -> List[List[float]]:
            v = g.get(key)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        return GreeksResponse(
            delta=_to_list("delta"),
            gamma=_to_list("gamma"),
            vega=_to_list("vega"),
            vanna=_to_list("vanna"),
            volga=_to_list("volga"),
            iv_surface=_to_list("iv_surface"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/vix", response_model=VIXResponse, tags=["Pricing"])
async def vix(params: HestonParams) -> VIXResponse:
    """
    Compute model VIX under Rough Heston via Riccati ODE.

    Returns the VIX level in VIX points (e.g., 18.5).
    """
    try:
        from market.vix_pricing import model_vix

        vix_val = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model_vix(
                kappa=params.kappa,
                theta=params.theta,
                sigma=params.sigma,
                rho=params.rho,
                v0=params.v0,
                H=params.H,
            ),
        )
        return VIXResponse(vix=float(vix_val))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/deribit/snapshot", response_model=DeribitSummaryResponse, tags=["Market Data"])
async def deribit_snapshot(
    currency: str = Query(default="BTC", description="BTC or ETH"),
) -> DeribitSummaryResponse:
    """
    Fetch live Deribit option snapshot and return a summary.

    Returns ATM IV, term structure of ATM IVs, and option count.
    """
    currency = currency.upper()
    if currency not in ("BTC", "ETH"):
        raise HTTPException(status_code=422, detail="currency must be BTC or ETH")

    try:
        from market.deribit_data import fetch_option_snapshot

        df = await fetch_option_snapshot(currency)

        spot: Optional[float] = None
        if "underlying_price" in df.columns and len(df) > 0:
            spot = float(df["underlying_price"].median())

        # ATM IV: options near log_moneyness ≈ 0, T closest to 1/12
        atm_iv: Optional[float] = None
        if len(df) > 0:
            df_atm = df[df["T"].between(0.05, 0.15) & (df["log_moneyness"].abs() < 0.1)]
            if len(df_atm) > 0:
                atm_iv = round(float(df_atm["mark_iv"].median()), 4)

        # Term structure: median ATM IV per maturity bucket
        term_structure: Dict[str, float] = {}
        if len(df) > 0:
            df_near_atm = df[df["log_moneyness"].abs() < 0.1].copy()
            df_near_atm["T_bucket"] = (df_near_atm["T"] * 12).round().astype(int)
            for bucket, group in df_near_atm.groupby("T_bucket"):
                label = f"{int(bucket)}M"
                term_structure[label] = round(float(group["mark_iv"].median()), 4)

        return DeribitSummaryResponse(
            currency=currency,
            n_options=len(df),
            atm_iv=atm_iv,
            term_structure=term_structure,
            spot=spot,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

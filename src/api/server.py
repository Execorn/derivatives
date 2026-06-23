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


# ── Calibration Schemas & Route ───────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    market_iv: List[List[float]] = Field(..., description="8x11 implied volatility surface in decimal")
    n_starts: int = Field(default=2, ge=1, le=5)


class CalibrateResponse(BaseModel):
    params: Dict[str, float]
    final_mse: float
    rmse_bps: float
    elapsed_ms: float
    converged: bool


_MODEL_CACHE: Dict[str, Any] = {}

def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from fno_model import MirrorPaddedFNO2d
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "heston":
            model = MirrorPaddedFNO2d(param_dim=5)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_heston_final_prod.pth"
        elif model_name == "sabr":
            model = MirrorPaddedFNO2d(param_dim=3)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_sabr_final_prod.pth"
        elif model_name == "ssvi":
            model = MirrorPaddedFNO2d(param_dim=11)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_ssvi_final_prod.pth"
        elif model_name == "rbergomi":
            model = MirrorPaddedFNO2d(param_dim=4)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "fno_rbergomi_final_prod.pth"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
            
        if not path.exists():
            raise HTTPException(status_code=500, detail=f"Weights not found for {model_name} at {path}")
            
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _MODEL_CACHE[model_name] = model
        
    return _MODEL_CACHE[model_name]


@app.post("/calibrate/{model_name}", response_model=CalibrateResponse, tags=["Calibration"])
async def calibrate_route(model_name: str, req: CalibrateRequest) -> CalibrateResponse:
    """
    Calibrate model parameters to an 8x11 market implied volatility surface.

    Supported model names: heston, sabr, ssvi, rbergomi
    """
    model_name = model_name.lower()
    if model_name not in ("heston", "sabr", "ssvi", "rbergomi"):
        raise HTTPException(status_code=400, detail=f"Unsupported model name: {model_name}")

    try:
        model = _get_model(model_name)
        iv_target = np.array(req.market_iv, dtype=np.float32)
        if iv_target.shape != (8, 11):
            raise HTTPException(status_code=422, detail="market_iv must have shape (8, 11)")

        from calibrate_fast import (calibrate_heston, calibrate_sabr,
                                    calibrate_ssvi, calibrate_rbergomi)

        def _run():
            if model_name == "heston":
                return calibrate_heston(model, iv_target, _MATURITIES, _STRIKES, max_iter=25, n_starts=req.n_starts)
            elif model_name == "sabr":
                return calibrate_sabr(model, iv_target, _MATURITIES, _STRIKES, max_iter=25, n_starts=req.n_starts)
            elif model_name == "ssvi":
                return calibrate_ssvi(model, iv_target, _MATURITIES, _STRIKES, max_iter=25, n_starts=req.n_starts)
            elif model_name == "rbergomi":
                return calibrate_rbergomi(model, iv_target, _MATURITIES, _STRIKES, max_iter=25, n_starts=req.n_starts)
            
        res = await asyncio.get_event_loop().run_in_executor(None, _run)

        # Build response parameters dictionary
        if model_name == "heston":
            p_dict = res["params"]
        elif model_name == "sabr":
            p_dict = {"alpha": res["alpha"], "rho": res["rho"], "nu": res["nu"]}
        elif model_name == "ssvi":
            p_dict = {"rho": res["rho"], "eta": res["eta"], "gamma": res["gamma"]}
            for i, val in enumerate(res["theta_atm"]):
                p_dict[f"theta_atm_{i+1}"] = float(val)
        elif model_name == "rbergomi":
            p_dict = {"v0": res["v0"], "H": res["H"], "eta": res["eta"], "rho": res["rho"]}

        return CalibrateResponse(
            params=p_dict,
            final_mse=float(res["final_mse"]),
            rmse_bps=float(res["rmse_bps"]),
            elapsed_ms=float(res["elapsed_ms"]),
            converged=bool(res["converged"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Phase 5-6 REST Endpoints ──────────────────────────────────────────────────

class CalibrateNeuralSDERequest(BaseModel):
    market_iv: List[List[float]] = Field(..., description="8x11 market implied volatility surface in decimal")
    S0: float = Field(default=100.0, gt=0, description="Initial stock price")
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.015, ge=0, description="Dividend yield")
    epochs: int = Field(default=30, ge=1, le=100, description="Number of training epochs")
    N_paths: int = Field(default=1024, ge=128, le=10000, description="Number of simulated paths")


class CalibrateNeuralSDEResponse(BaseModel):
    v0: float
    rho: float
    final_rmse: float
    loss_history: List[float]
    elapsed_ms: float
    fitted_iv: Optional[List[List[float]]] = None


class PredictSignatureVolRequest(BaseModel):
    v0: float = Field(default=0.04, gt=0, description="Initial variance")
    ell: List[float] = Field(..., description="Signature volatility coefficients (30 elements)")
    rho: float = Field(default=-0.5, le=0, ge=-1, description="Correlation parameter")
    T: float = Field(default=0.25, gt=0, description="Option maturity in years")
    S0: float = Field(default=100.0, gt=0, description="Initial stock price")
    r: float = Field(default=0.0, ge=0, description="Risk-free rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")
    N_paths: int = Field(default=4096, ge=100, le=50000, description="Path simulation count")
    strikes: List[float] = Field(default_factory=lambda: [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])


class PredictSignatureVolResponse(BaseModel):
    strikes: List[float]
    implied_vols: List[float]
    option_prices: List[float]
    paths_S: Optional[List[List[float]]] = None
    paths_vol: Optional[List[List[float]]] = None


class HedgeSimulateRequest(BaseModel):
    option_type: str = Field(..., description="Option type: 'european', 'barrier', or 'minimax'")
    S0: float = Field(default=100.0, gt=0, description="Initial spot price")
    strike: float = Field(default=100.0, gt=0, description="Option strike price")
    barrier: float = Field(default=85.0, gt=0, description="Option knock-out barrier (for barrier style)")
    expiry: float = Field(default=0.1, gt=0, description="Maturity in years")
    mu: float = Field(default=0.0, description="Underlying drift")
    sigma: float = Field(default=0.2, gt=0, description="Asset volatility")
    steps: int = Field(default=30, ge=5, le=100, description="Rebalancing step frequency")
    N_paths: int = Field(default=100, ge=1, le=1000, description="Path simulation count")
    cost_stock: float = Field(default=0.0001, description="Stock transaction cost coeff")
    cost_vol: float = Field(default=0.0005, description="Volatility option transaction cost coeff")


class HedgeSimulateResponse(BaseModel):
    paths_S: List[List[float]]
    paths_vol: List[List[float]]
    deltas_stock: List[List[float]]
    deltas_vol: List[List[float]]
    costs: List[float]
    wealth: List[float]
    payoff: List[float]
    pnl: List[float]
    std_pnl: float
    final_loss: float


def _pytorch_black_scholes_call(S0, K, T, sigma, r, q):
    d1 = (torch.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + 1e-8)
    d2 = d1 - sigma * torch.sqrt(T)
    normal = torch.distributions.Normal(0.0, 1.0)
    return S0 * torch.exp(-q * T) * normal.cdf(d1) - K * torch.exp(-r * T) * normal.cdf(d2)


def _black_scholes_call_price_numpy(S0, K, T, sigma, r, q=0.0):
    from scipy.stats import norm
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _black_scholes_vega_numpy(S0, K, T, sigma, r, q=0.0):
    from scipy.stats import norm
    if T <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def _implied_volatility_newton(price, S0, K, T, r, q=0.0, max_iter=100, tol=1e-6):
    intrinsic = max(S0 - K, 0.0)
    if price <= intrinsic + 1e-6:
        return 0.0
    sigma = 0.3
    for _ in range(max_iter):
        p = _black_scholes_call_price_numpy(S0, K, T, sigma, r, q)
        diff = p - price
        if abs(diff) < tol:
            return float(sigma)
        vega = _black_scholes_vega_numpy(S0, K, T, sigma, r, q)
        if vega < 1e-6:
            sigma = sigma - 0.5 * diff / S0
        else:
            sigma = sigma - diff / vega
        sigma = np.clip(sigma, 1e-4, 5.0)
    return float(sigma)


@app.post("/calibrate_neural_sde", response_model=CalibrateNeuralSDEResponse, tags=["Calibration"])
async def calibrate_neural_sde(req: CalibrateNeuralSDERequest) -> CalibrateNeuralSDEResponse:
    """
    Calibrate a non-parametric Neural SDE model to an 8x11 market implied volatility surface.
    """
    try:
        if len(req.market_iv) != 8 or any(len(row) != 11 for row in req.market_iv):
            raise HTTPException(status_code=400, detail="market_iv must be an 8x11 surface")

        from src.pricing.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build option price target grid
        S0 = req.S0
        r = req.r
        q = req.q
        
        T_mkt_list = []
        K_mkt_list = []
        prices_mkt_list = []
        
        for i, t in enumerate(_MATURITIES):
            for j, k in enumerate(_STRIKES):
                strike_val = S0 * np.exp(k)
                iv_val = req.market_iv[i][j]
                
                T_mkt_list.append(t)
                K_mkt_list.append(strike_val)
                prices_mkt_list.append(
                    _pytorch_black_scholes_call(
                        torch.tensor(S0, dtype=torch.float32, device=device),
                        torch.tensor(strike_val, dtype=torch.float32, device=device),
                        torch.tensor(t, dtype=torch.float32, device=device),
                        torch.tensor(iv_val, dtype=torch.float32, device=device),
                        r, q
                    ).item()
                )
                
        strikes_mkt = torch.tensor(K_mkt_list, dtype=torch.float32, device=device)
        prices_mkt = torch.tensor(prices_mkt_list, dtype=torch.float32, device=device)
        maturities_mkt = torch.tensor(T_mkt_list, dtype=torch.float32, device=device)
        
        # Initialize NeuralSDE and pricer
        epsilon = 1e-4
        sde = NeuralSDE(r=r, q=q, rho_init=-0.7, hidden_dim=16, epsilon=epsilon)
        pricer = NeuralSDEPricer(sde, v0_init=0.04)
        pricer.to(device)
        
        optimizer = torch.optim.Adam(pricer.parameters(), lr=0.01)
        loss_history = []
        
        t0 = time.time()
        
        # Optimization Loop running asynchronously in threadpool to prevent blocking FastAPI main thread loop
        def _run_optimization():
            for epoch in range(req.epochs):
                optimizer.zero_grad()
                pred, ys = pricer.price_options(
                    S0=S0, strikes=strikes_mkt, maturities=maturities_mkt,
                    N_paths=req.N_paths, dt=0.01, method="euler"
                )
                loss_dict = compute_calibration_loss(
                    model_prices=pred, market_prices=prices_mkt,
                    vegas=torch.ones_like(prices_mkt), ys=ys,
                    lambda_bound=0.01, epsilon=epsilon
                )
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
            
            # Predict final prices for final RMSE
            with torch.no_grad():
                pred_final, _ = pricer.price_options(
                    S0=S0, strikes=strikes_mkt, maturities=maturities_mkt,
                    N_paths=req.N_paths, dt=0.01, method="euler"
                )
                final_rmse = torch.sqrt(torch.mean((pred_final - prices_mkt) ** 2)).item()
                
                # Invert final predicted option prices to implied volatility surface (8x11)
                pred_final_np = pred_final.detach().cpu().numpy()
                fitted_iv = []
                idx = 0
                for i, t in enumerate(_MATURITIES):
                    row_iv = []
                    for j, k in enumerate(_STRIKES):
                        strike_val = S0 * np.exp(k)
                        price_val = float(pred_final_np[idx])
                        idx += 1
                        iv_val = _implied_volatility_newton(price_val, S0, strike_val, t, r, q)
                        row_iv.append(iv_val)
                    fitted_iv.append(row_iv)
                
            return float(pricer.v0.item()), float(sde.rho.item()), final_rmse, fitted_iv

        v0_fitted, rho_fitted, rmse_fitted, fitted_iv = await asyncio.get_event_loop().run_in_executor(None, _run_optimization)
        elapsed_ms = (time.time() - t0) * 1000.0
        
        return CalibrateNeuralSDEResponse(
            v0=v0_fitted,
            rho=rho_fitted,
            final_rmse=rmse_fitted,
            loss_history=loss_history,
            elapsed_ms=elapsed_ms,
            fitted_iv=fitted_iv
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/signature_vol", response_model=PredictSignatureVolResponse, tags=["Pricing"])
async def predict_signature_vol(req: PredictSignatureVolRequest) -> PredictSignatureVolResponse:
    """
    Forecasting options smile using the Signature Volatility model.
    """
    try:
        if len(req.ell) != 30:
            raise HTTPException(status_code=400, detail="ell coefficients must have exactly 30 elements")

        from src.pricing.signature_vol import simulate_signature_vol_paths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ell_tensor = torch.tensor(req.ell, dtype=torch.float32, device=device)
        
        t0 = time.time()
        
        def _run_pricing():
            S, V, _, _ = simulate_signature_vol_paths(
                v0=torch.tensor(req.v0, dtype=torch.float32, device=device),
                ell=ell_tensor,
                rho=torch.tensor(req.rho, dtype=torch.float32, device=device),
                T=req.T,
                steps_per_unit=252,
                N_paths=req.N_paths,
                S0=req.S0,
                r=req.r,
                q=req.q,
                antithetic=True,
                device=device
            )
            
            S_T = S[:, -1].detach().cpu().numpy()
            
            prices = []
            for strike in req.strikes:
                payoff = np.maximum(S_T - strike, 0.0)
                prices.append(float(payoff.mean() * np.exp(-req.r * req.T)))
                
            # Invert call prices to implied volatility using the robust Newton solver
            implied_vols = []
            for strike, price in zip(req.strikes, prices):
                iv = _implied_volatility_newton(price, req.S0, strike, req.T, req.r, req.q)
                implied_vols.append(iv)
                
            # Extract first 5 sample paths for visualization
            paths_S = S[:5].detach().cpu().numpy().tolist()
            paths_vol = V[:5].detach().cpu().numpy().tolist()
            
            return implied_vols, prices, paths_S, paths_vol

        implied_vols, prices, paths_S, paths_vol = await asyncio.get_event_loop().run_in_executor(None, _run_pricing)
        
        return PredictSignatureVolResponse(
            strikes=req.strikes,
            implied_vols=implied_vols,
            option_prices=prices,
            paths_S=paths_S,
            paths_vol=paths_vol
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/hedge/simulate", response_model=HedgeSimulateResponse, tags=["Hedging"])
async def hedge_simulate(req: HedgeSimulateRequest) -> HedgeSimulateResponse:
    """
    Evaluate optimal deep hedging policy rebalancing simulation and output metrics.
    """
    try:
        from src.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv
        from src.hedging.barrier_hedging import BarrierHedgingEnv
        from scripts.train_deep_hedging import simulate_gbm_paths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option_type = req.option_type.lower()
        
        # 1. Initialize policy and load weights
        if option_type == "european":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "deep_hedger_european_prod.pth"
        elif option_type == "barrier":
            policy = HedgingPolicy(input_dim=6, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "deep_hedger_barrier_prod.pth"
        elif option_type == "minimax":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[2] / "artifacts" / "weights" / "minimax_policy_prod.pth"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported option type: {option_type}")
            
        if not path.exists():
            raise HTTPException(status_code=500, detail=f"Policy weights not found at {path}")
            
        policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        policy.eval()
        
        # 2. Simulate asset paths
        H, t_grid = simulate_gbm_paths(
            S0=req.S0, mu=req.mu, sigma=req.sigma,
            T=req.expiry, steps=req.steps, N_paths=req.N_paths,
            d=2, device=device
        )
        
        cost_coeffs = torch.tensor([req.cost_stock, req.cost_vol], device=device)
        
        # 3. Initialize Env
        if option_type in ("european", "minimax"):
            payoff = torch.clamp(H[:, -1, 0] - req.strike, min=0.0)
            env = DeepHedgingEnv(
                H=H, payoff=payoff, cost_coeffs=cost_coeffs,
                strike=req.strike, expiry=req.expiry, risk_aversion=1.0,
                risk_measure="quad", t_grid=t_grid
            )
        else:
            env = BarrierHedgingEnv(
                H=H, cost_coeffs=cost_coeffs, strike=req.strike,
                barrier=req.barrier, expiry=req.expiry, risk_aversion=1.0,
                risk_measure="quad", t_grid=t_grid
            )
            
        # 4. Simulate optimal hedging
        with torch.no_grad():
            wealth, costs, deltas = env.simulate_hedging_episode(policy)
            loss = env.compute_loss(wealth).item()
            
        pnl = (wealth - env.payoff).cpu().numpy()
        std_pnl = float(np.std(pnl))
        
        return HedgeSimulateResponse(
            paths_S=H[:, :, 0].cpu().numpy().tolist(),
            paths_vol=H[:, :, 1].cpu().numpy().tolist(),
            deltas_stock=deltas[:, :, 0].cpu().numpy().tolist(),
            deltas_vol=deltas[:, :, 1].cpu().numpy().tolist(),
            costs=costs.cpu().numpy().tolist(),
            wealth=wealth.cpu().numpy().tolist(),
            payoff=env.payoff.cpu().numpy().tolist(),
            pnl=pnl.tolist(),
            std_pnl=std_pnl,
            final_loss=loss
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


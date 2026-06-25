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
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from cachetools import TTLCache
import orjson

import numpy as np
import torch

# Limit intra-op threads to prevent CPU core thrashing when executing in thread pools
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[2])
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

# ── Global model cache & Lock Management ───────────────────────────────────────

class AsyncReadWriteLock:
    """Non-blocking, asyncio-compatible Read-Write Lock."""
    def __init__(self):
        self._readers = 0
        self._write_lock = asyncio.Lock()
        self._read_ready = asyncio.Condition()

    async def acquire_read(self):
        async with self._read_ready:
            while self._write_lock.locked():
                await self._read_ready.wait()
            self._readers += 1

    async def release_read(self):
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    async def acquire_write(self):
        await self._write_lock.acquire()
        t0 = asyncio.get_event_loop().time()
        _WRITE_STARVATION_WARN_S = 30.0
        async with self._read_ready:
            while self._readers > 0:
                await self._read_ready.wait()
                elapsed = asyncio.get_event_loop().time() - t0
                if elapsed > _WRITE_STARVATION_WARN_S:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "AsyncReadWriteLock: writer waiting >%.0fs for readers to drain "
                        "(%d active reader(s)). Consider reducing concurrent read load.",
                        elapsed, self._readers,
                    )
                    t0 = asyncio.get_event_loop().time()  # reset to avoid log spam

    async def release_write(self):
        if self._write_lock.locked():
            self._write_lock.release()
        async with self._read_ready:
            self._read_ready.notify_all()

    class ReaderContext:
        def __init__(self, rwlock):
            self.rwlock = rwlock
        async def __aenter__(self):
            await self.rwlock.acquire_read()
        async def __aexit__(self, exc_type, exc, tb):
            await self.rwlock.release_read()

    class WriterContext:
        def __init__(self, rwlock):
            self.rwlock = rwlock
        async def __aenter__(self):
            await self.rwlock.acquire_write()
        async def __aexit__(self, exc_type, exc, tb):
            await self.rwlock.release_write()

    def reader(self):
        return self.ReaderContext(self)
    def writer(self):
        return self.WriterContext(self)


class CachedModel:
    def __init__(self, model_name: str, model: torch.nn.Module, pn: Any, yn: Any, path: Path):
        self.name = model_name
        self.model = model
        self.pn = pn
        self.yn = yn
        self.path = path
        self.rwlock = AsyncReadWriteLock()
        self.last_loaded = time.time()


_MODEL_CACHE: Dict[str, CachedModel] = {}
_MODEL_STATE: Dict[str, Any] = {
    "model":  None,
    "pn":     None,
    "yn":     None,
    "device": None,
    "loaded": False,
}

_WEIGHTS_PATH  = Path(__file__).parents[3] / "artifacts" / "weights" / "fno_v2_final_prod.pth"
_PARAM_NORM    = Path(__file__).parents[3] / "artifacts" / "models" / "param_normalizer_v2.npz"
_IV_NORM       = Path(__file__).parents[3] / "artifacts" / "models" / "iv_normalizer_v2.npz"

# FNO training grids (must match training exactly)
_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)


def _get_cached_container(model_name: str) -> CachedModel:
    model_name = model_name.lower()
    if model_name not in _MODEL_CACHE:
        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == "heston":
            param_dim = 5
            weights_file = "fno_heston_final_prod.pth"
            pn_file = "param_normalizer_heston.npz"
            yn_file = "iv_normalizer_heston.npz"
        elif model_name == "sabr":
            param_dim = 3
            weights_file = "fno_sabr_final_prod.pth"
            pn_file = "param_normalizer_sabr.npz"
            yn_file = "iv_normalizer_sabr.npz"
        elif model_name == "ssvi":
            param_dim = 11
            weights_file = "fno_ssvi_final_prod.pth"
            pn_file = "param_normalizer_ssvi.npz"
            yn_file = "iv_normalizer_ssvi.npz"
        elif model_name == "rbergomi":
            param_dim = 4
            weights_file = "fno_rbergomi_final_prod.pth"
            pn_file = "param_normalizer_rbergomi.npz"
            yn_file = "iv_normalizer_rbergomi.npz"
        elif model_name in ("rough_heston", "v2", "default"):
            param_dim = 6
            weights_file = "fno_v2_final_prod.pth"
            pn_file = "param_normalizer_v2.npz"
            yn_file = "iv_normalizer_v2.npz"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

        from deepvol.utils.path_helpers import get_project_root
        artifacts_dir = get_project_root() / "artifacts"
        path = artifacts_dir / "weights" / weights_file
        pn_path = artifacts_dir / "models" / pn_file
        yn_path = artifacts_dir / "models" / yn_file

        if not path.exists():
            raise HTTPException(status_code=500, detail=f"Weights not found for {model_name} at {path}")
        if not pn_path.exists():
            raise HTTPException(status_code=500, detail=f"Parameter normalizer not found for {model_name} at {pn_path}")
        if not yn_path.exists():
            raise HTTPException(status_code=500, detail=f"IV normalizer not found for {model_name} at {yn_path}")

        model = MirrorPaddedFNO2d(param_dim=param_dim)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        pn = ParameterNormalizer.load(str(pn_path))
        yn = IVSurfaceNormalizer.load(str(yn_path))

        from deepvol.arbitrage.projection_layer import DifferentiableArbitrageFreeProjection, ArbitrageFreeFNO
        proj = DifferentiableArbitrageFreeProjection(
            T_grid=torch.tensor(_MATURITIES, dtype=torch.float64, device=device),
            K_grid=torch.tensor(_STRIKES, dtype=torch.float64, device=device),
            S0=1.0,
            is_log_moneyness=True
        )
        wrapped_model = ArbitrageFreeFNO(base_fno=model, projection_layer=proj, normalizer=yn).to(device)

        container = CachedModel(model_name, wrapped_model, pn, yn, path)
        _MODEL_CACHE[model_name] = container
        
        # Populate _MODEL_STATE for backward compatibility if default model is loaded
        if model_name in ("rough_heston", "default"):
            _MODEL_STATE.update(model=wrapped_model, pn=pn, yn=yn, device=device, loaded=True)

    return _MODEL_CACHE[model_name]


async def _hot_reload_model_weights(model_name: str):
    """
    Acquires exclusive write lock and reloads the model parameters from disk.
    This runs with ZERO server downtime.
    """
    model_name = model_name.lower()
    if model_name not in _MODEL_CACHE:
        return
        
    cached_container = _MODEL_CACHE[model_name]
    
    async with cached_container.rwlock.writer():
        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state = torch.load(cached_container.path, map_location=device, weights_only=True)
        if hasattr(cached_container.model, "base_fno"):
            cached_container.model.base_fno.load_state_dict(state)
        else:
            cached_container.model.load_state_dict(state)
        cached_container.model.to(device)
        cached_container.model.eval()
        
        if model_name == "heston":
            pn_file = "param_normalizer_heston.npz"
            yn_file = "iv_normalizer_heston.npz"
        elif model_name == "sabr":
            pn_file = "param_normalizer_sabr.npz"
            yn_file = "iv_normalizer_sabr.npz"
        elif model_name == "ssvi":
            pn_file = "param_normalizer_ssvi.npz"
            yn_file = "iv_normalizer_ssvi.npz"
        elif model_name == "rbergomi":
            pn_file = "param_normalizer_rbergomi.npz"
            yn_file = "iv_normalizer_rbergomi.npz"
        else:
            pn_file = "param_normalizer_v2.npz"
            yn_file = "iv_normalizer_v2.npz"

        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.utils.path_helpers import get_project_root

        artifacts_dir = get_project_root() / "artifacts"
        pn_path = artifacts_dir / "models" / pn_file
        yn_path = artifacts_dir / "models" / yn_file

        cached_container.pn = ParameterNormalizer.load(str(pn_path))
        cached_container.yn = IVSurfaceNormalizer.load(str(yn_path))
        cached_container.last_loaded = time.time()
        
        if model_name in ("rough_heston", "default"):
            _MODEL_STATE.update(model=cached_container.model, pn=cached_container.pn, yn=cached_container.yn, device=device, loaded=True)


def _load_model() -> None:
    """Lazy-load FNO model for backward compatibility."""
    _get_cached_container("rough_heston")


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
    container = _get_cached_container("rough_heston")
    model  = container.model
    pn     = container.pn
    yn     = container.yn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    loaded = "rough_heston" in _MODEL_CACHE
    device_str = "not_loaded"
    if loaded:
        container = _MODEL_CACHE["rough_heston"]
        device_str = str(next(container.model.parameters()).device)
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
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
        container = _get_cached_container("rough_heston")
        async with container.rwlock.reader():
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


# ── Polymorphic Greeks Request Schemas ──────────────────────────────────────────

class BaseGreeksRequest(BaseModel):
    S: float = Field(default=100.0, gt=0, description="Spot price of the underlying asset")
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")

class HestonGreeksRequest(BaseGreeksRequest):
    kappa: float = Field(..., gt=0.1, le=10.0, description="Mean-reversion speed")
    theta: float = Field(..., gt=0.01, le=0.5, description="Long-run variance")
    sigma: float = Field(..., gt=0.05, le=2.0, description="Vol-of-vol")
    rho: float = Field(..., ge=-0.99, le=0.0, description="Asset-vol correlation")
    v0: float = Field(..., gt=0.01, le=0.5, description="Initial variance")
    H: Optional[float] = Field(default=None, description="Hurst exponent (optional for Rough Heston)")

class SABRGreeksRequest(BaseGreeksRequest):
    alpha: float = Field(..., gt=0.001, description="Initial volatility parameter")
    rho: float = Field(..., ge=-0.99, le=0.99, description="Correlation parameter")
    nu: float = Field(..., gt=0.001, description="Volatility of volatility")

class SSVIGreeksRequest(BaseGreeksRequest):
    rho: float = Field(..., ge=-0.99, le=0.99, description="Correlation")
    eta: float = Field(..., gt=0, description="SSVI slope parameters")
    gamma: float = Field(..., gt=0, description="Power exponent")
    theta_atm: List[float] = Field(..., description="8 ATM variance values")

    @field_validator("theta_atm")
    @classmethod
    def validate_theta_atm(cls, v: List[float]) -> List[float]:
        if len(v) != 8:
            raise ValueError("theta_atm must have exactly 8 values")
        return v

class RBergomiGreeksRequest(BaseGreeksRequest):
    v0: float = Field(..., gt=0.0, description="Initial variance")
    H: float = Field(..., gt=0.0, lt=0.5, description="Hurst index")
    eta: float = Field(..., gt=0.0, description="Vol-of-vol")
    rho: float = Field(..., ge=-0.99, le=0.0, description="Correlation")


def _compute_greeks_from_surface(iv_surface: np.ndarray, S: float, r: float, q: float) -> dict:
    from deepvol.greeks.portfolio_greeks import bs_greeks
    T_grid = _MATURITIES
    k_grid = _STRIKES
    nT, nK = len(T_grid), len(k_grid)

    delta_surf  = np.zeros((nT, nK), dtype=np.float32)
    gamma_surf  = np.zeros((nT, nK), dtype=np.float32)
    vega_surf   = np.zeros((nT, nK), dtype=np.float32)
    vanna_surf  = np.zeros((nT, nK), dtype=np.float32)
    volga_surf  = np.zeros((nT, nK), dtype=np.float32)

    for i in range(nT):
        for j in range(nK):
            T_val   = float(T_grid[i])
            kk_val   = float(k_grid[j])
            K_val   = float(S) * float(np.exp(kk_val))
            sig_val = float(iv_surface[i, j])

            g = bs_greeks(float(S), K_val, T_val, r, sig_val, q=q)

            delta_surf[i, j] = g["delta"]
            gamma_surf[i, j] = g["gamma"]
            vega_surf[i, j]  = g["vega"]
            vanna_surf[i, j] = g["vanna"]
            volga_surf[i, j] = g["volga"]

    gamma_max = np.percentile(np.abs(gamma_surf[np.isfinite(gamma_surf)]), 99.5)
    if gamma_max > 1e6:
        gamma_surf = np.clip(gamma_surf, -1e6, 1e6)

    return {
        "delta": delta_surf.tolist(),
        "gamma": gamma_surf.tolist(),
        "vega": vega_surf.tolist(),
        "vanna": vanna_surf.tolist(),
        "volga": volga_surf.tolist(),
        "iv_surface": iv_surface.tolist(),
    }


@app.post("/greeks/{model_name}", response_model=GreeksResponse, tags=["Risk"])
async def compute_model_greeks(
    model_name: str,
    req: Dict[str, Any]
) -> GreeksResponse:
    """
    Compute per-strike Greeks surfaces for a specific model name.
    Supported models: heston, sabr, ssvi, rbergomi, rough_heston
    """
    model_name = model_name.lower()
    try:
        if model_name == "heston":
            parsed_req = HestonGreeksRequest(**req)
        elif model_name == "sabr":
            parsed_req = SABRGreeksRequest(**req)
        elif model_name == "ssvi":
            parsed_req = SSVIGreeksRequest(**req)
        elif model_name == "rbergomi":
            parsed_req = RBergomiGreeksRequest(**req)
        elif model_name in ("rough_heston", "default", "v2"):
            parsed_req = HestonGreeksRequest(**req)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model name: {model_name}")
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))

    try:
        cached_container = _get_cached_container(model_name)
        
        async with cached_container.rwlock.reader():
            # Build the model parameter array/tensor
            if model_name in ("heston", "rough_heston", "default", "v2"):
                actual_model = cached_container.model.base_fno if hasattr(cached_container.model, "base_fno") else cached_container.model
                if actual_model.film.mlp[0].in_features == 5:
                    # Classic Heston (5 params)
                    theta_arr = np.array([parsed_req.kappa, parsed_req.theta, parsed_req.sigma, parsed_req.rho, parsed_req.v0], dtype=np.float32)
                else:
                    # Rough Heston (6 params)
                    H_val = parsed_req.H if parsed_req.H is not None else 0.08
                    theta_arr = np.array([parsed_req.kappa, parsed_req.theta, parsed_req.sigma, parsed_req.rho, parsed_req.v0, H_val], dtype=np.float32)
            elif model_name == "sabr":
                theta_arr = np.array([parsed_req.alpha, parsed_req.rho, parsed_req.nu], dtype=np.float32)
            elif model_name == "ssvi":
                theta_arr = np.array(parsed_req.theta_atm + [parsed_req.rho, parsed_req.eta, parsed_req.gamma], dtype=np.float32)
            elif model_name == "rbergomi":
                theta_arr = np.array([parsed_req.v0, parsed_req.H, parsed_req.eta, parsed_req.rho], dtype=np.float32)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            theta_t = torch.tensor(theta_arr, dtype=torch.float32, device=device)
            spatial = _make_spatial(device)

            def _run_forward():
                with torch.no_grad():
                    theta_norm = cached_container.pn.transform_tensor(theta_t.unsqueeze(0))
                    pred_norm  = cached_container.model(spatial, theta_norm)
                    iv_tensor  = cached_container.yn.inverse_transform_tensor(pred_norm).squeeze(0)
                    return iv_tensor.clamp(min=1e-4).cpu().numpy()

            iv_surface = await asyncio.get_event_loop().run_in_executor(None, _run_forward)

            g_results = await asyncio.get_event_loop().run_in_executor(
                None, _compute_greeks_from_surface, iv_surface, parsed_req.S, parsed_req.r, parsed_req.q
            )
            return GreeksResponse(**g_results)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/greeks", response_model=GreeksResponse, tags=["Risk"])
async def greeks(req: GreeksRequest) -> GreeksResponse:
    """
    Compute per-strike Greeks surfaces via FNO + Black-Scholes (default Rough Heston model).
    """
    req_dict = req.model_dump()  # Pydantic V2: model_dump() replaces deprecated .dict()
    return await compute_model_greeks("rough_heston", req_dict)


@app.post("/vix", response_model=VIXResponse, tags=["Pricing"])
async def vix(params: HestonParams) -> VIXResponse:
    """
    Compute model VIX under Rough Heston via Riccati ODE.

    Returns the VIX level in VIX points (e.g., 18.5).
    """
    try:
        from deepvol.market.vix_pricing import model_vix

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
        from deepvol.market.deribit_data import fetch_option_snapshot

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

# ── Calibration Schemas & Route ───────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    market_iv: List[List[float]] = Field(
        ..., 
        description="8x11 implied volatility surface in decimal (e.g., 0.20 = 20% IV)"
    )
    n_starts: int = Field(default=2, ge=1, le=5, description="Number of random restarts for solver")
    max_iter: int = Field(default=25, ge=5, le=100, description="Max iterations per optimization run")
    tol: float = Field(default=1e-5, gt=0, description="MSE convergence tolerance")

    @field_validator("market_iv")
    @classmethod
    def validate_surface(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 8 or any(len(row) != 11 for row in v):
            raise ValueError("implied volatility surface must have dimensions (8, 11)")
        return v


class CalibrateResponse(BaseModel):
    params: Dict[str, float] = Field(..., description="Calibrated model-specific parameters")
    final_mse: float = Field(..., description="Mean squared error of calibrated surface vs target")
    rmse_bps: float = Field(..., description="Root mean squared error in basis points")
    elapsed_ms: float = Field(..., description="Optimization execution time in milliseconds")
    converged: bool = Field(..., description="True if optimization converged within tolerance")


_GPU_CALIBRATION_SEMAPHORE = asyncio.Semaphore(1)


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
        cached_container = _get_cached_container(model_name)
        iv_target = np.array(req.market_iv, dtype=np.float32)
        if iv_target.shape != (8, 11):
            raise HTTPException(status_code=422, detail="market_iv must have shape (8, 11)")

        from deepvol.calibration.calibrate_newton import (calibrate_heston, calibrate_sabr,
                                    calibrate_ssvi, calibrate_rbergomi)

        # Acquire GPU semaphore to serialize calibration and avoid VRAM exhaustion
        async with _GPU_CALIBRATION_SEMAPHORE:
            # Acquire Read Lock on model weights to prevent reload crashes
            async with cached_container.rwlock.reader():
                def _run():
                    if model_name == "heston":
                        return calibrate_heston(cached_container.model, iv_target, _MATURITIES, _STRIKES, max_iter=req.max_iter, n_starts=req.n_starts)
                    elif model_name == "sabr":
                        return calibrate_sabr(cached_container.model, iv_target, _MATURITIES, _STRIKES, max_iter=req.max_iter, n_starts=req.n_starts)
                    elif model_name == "ssvi":
                        return calibrate_ssvi(cached_container.model, iv_target, _MATURITIES, _STRIKES, max_iter=req.max_iter, n_starts=req.n_starts)
                    elif model_name == "rbergomi":
                        return calibrate_rbergomi(cached_container.model, iv_target, _MATURITIES, _STRIKES, max_iter=req.max_iter, n_starts=req.n_starts)
                    
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

        from deepvol.models.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss

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

        from deepvol.models.signature_vol import simulate_signature_vol_paths

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
        from deepvol.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv, simulate_gbm_paths
        from deepvol.hedging.barrier_hedging import BarrierHedgingEnv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option_type = req.option_type.lower()
        
        # 1. Initialize policy and load weights
        if option_type == "european":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[3] / "artifacts" / "weights" / "deep_hedger_european_prod.pth"
        elif option_type == "barrier":
            policy = HedgingPolicy(input_dim=6, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[3] / "artifacts" / "weights" / "deep_hedger_barrier_prod.pth"
        elif option_type == "minimax":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = Path(__file__).parents[3] / "artifacts" / "weights" / "minimax_policy_prod.pth"
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


# ── Process-Isolated Training Background Task Manager ─────────────────────────

_TRAINING_TASKS: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    epochs: int = Field(default=150, ge=1, le=1000)
    batch_size: int = Field(default=4096, ge=32)
    lr: float = Field(default=8e-4, gt=0)
    use_swa: bool = Field(default=True)
    dataset_path: Optional[str] = Field(default=None)

class TrainResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TrainStatusResponse(BaseModel):
    task_id: str
    model_name: str
    status: str  # "queued", "running", "completed", "failed"
    started_at: float
    completed_at: Optional[float] = None
    elapsed_seconds: float
    error_message: Optional[str] = None
    recent_logs: List[str]


async def _run_training_subprocess(task_id: str, model_name: str, req: TrainRequest):
    """Spawns scripts/train_fno_{model_name}.py in an isolated OS process."""
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/train_{task_id}.log"

    # Determine command (running with active virtual environment python if possible)
    python_exe = sys.executable
    script_path = str(Path(__file__).parents[2] / "scripts" / f"train_fno_{model_name}.py")

    cmd = [python_exe, script_path]
    
    # Check for smoke test
    if req.epochs <= 3:
        cmd.append("--smoke")

    # Set up environment variables for training configuration
    env = os.environ.copy()
    env["EPOCHS"] = str(req.epochs)
    env["BATCH_SIZE"] = str(req.batch_size)
    env["LR"] = str(req.lr)
    # Compute reasonable SWA start
    swa_start = int(0.8 * req.epochs) if req.epochs >= 10 else 2
    env["SWA_START"] = str(swa_start)
    if req.dataset_path:
        env["DATASET_PATH"] = req.dataset_path

    try:
        with open(log_path, "w") as log_file:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file,
                stderr=log_file,
                env=env,
                cwd=str(Path(__file__).parents[2])
            )
            
            exit_code = await process.wait()

        if exit_code == 0:
            _TRAINING_TASKS[task_id]["status"] = "completed"
            _TRAINING_TASKS[task_id]["completed_at"] = time.time()
            # Hot-reload hook: dynamically load new weights into API server
            await _hot_reload_model_weights(model_name)
        else:
            _TRAINING_TASKS[task_id]["status"] = "failed"
            _TRAINING_TASKS[task_id]["completed_at"] = time.time()
            _TRAINING_TASKS[task_id]["error_message"] = f"Subprocess exited with code {exit_code}"

    except Exception as exc:
        _TRAINING_TASKS[task_id]["status"] = "failed"
        _TRAINING_TASKS[task_id]["completed_at"] = time.time()
        _TRAINING_TASKS[task_id]["error_message"] = str(exc)


@app.post("/train/{model_name}", response_model=TrainResponse, tags=["Training"])
async def trigger_training(model_name: str, req: TrainRequest) -> TrainResponse:
    model_name = model_name.lower()
    if model_name not in ("heston", "sabr", "ssvi", "rbergomi"):
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    # Check for active training task of this model
    for tid, tstate in _TRAINING_TASKS.items():
        if tstate["model_name"] == model_name and tstate["status"] == "running":
            raise HTTPException(
                status_code=409, 
                detail=f"Training for {model_name} is already in progress (Task ID: {tid})"
            )

    task_id = str(uuid.uuid4())
    _TRAINING_TASKS[task_id] = {
        "model_name": model_name,
        "status": "running",
        "started_at": time.time(),
        "completed_at": None,
        "error_message": None,
    }

    # Spawn training as a non-blocking background task
    asyncio.create_task(_run_training_subprocess(task_id, model_name, req))

    return TrainResponse(
        task_id=task_id,
        status="running",
        message=f"Training started for {model_name}. Check progress at /train/status/{task_id}"
    )


@app.get("/train/status/{task_id}", response_model=TrainStatusResponse, tags=["Training"])
async def get_train_status(task_id: str) -> TrainStatusResponse:
    if task_id not in _TRAINING_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    state = _TRAINING_TASKS[task_id]
    log_path = f"logs/train_{task_id}.log"
    
    recent_logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            recent_logs = f.readlines()[-20:]

    elapsed = (state["completed_at"] or time.time()) - state["started_at"]

    return TrainStatusResponse(
        task_id=task_id,
        model_name=state["model_name"],
        status=state["status"],
        started_at=state["started_at"],
        completed_at=state["completed_at"],
        elapsed_seconds=round(elapsed, 2),
        error_message=state["error_message"],
        recent_logs=[line.strip() for line in recent_logs]
    )


# ── Session-Based Calibration Cache ───────────────────────────────────────────

# Thread-safe in-memory session cache with 1-hour expiration
_SESSION_CACHE = TTLCache(maxsize=10000, ttl=3600)

class SessionCalibrateResponse(BaseModel):
    session_id: str
    model_name: str
    params: Dict[str, float]
    rmse_bps: float
    expires_in_seconds: int = 3600

class SessionGreeksRequest(BaseModel):
    S: float = Field(default=100.0, gt=0, description="Spot price of the underlying asset")
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")


@app.post("/session/calibrate/{model_name}", response_model=SessionCalibrateResponse, tags=["Session"])
async def session_calibrate(model_name: str, req: CalibrateRequest) -> SessionCalibrateResponse:
    """
    Run standard calibration for a model name and cache the resulting parameters in a session.
    """
    calib_res = await calibrate_route(model_name, req)
    
    session_id = str(uuid.uuid4())
    _SESSION_CACHE[session_id] = {
        "model_name": model_name,
        "params": calib_res.params,
        "last_accessed": time.time()
    }
    
    return SessionCalibrateResponse(
        session_id=session_id,
        model_name=model_name,
        params=calib_res.params,
        rmse_bps=calib_res.rmse_bps
    )


@app.post("/session/{session_id}/greeks", response_model=GreeksResponse, tags=["Session"])
async def session_greeks(session_id: str, req: SessionGreeksRequest) -> GreeksResponse:
    """
    Compute Greeks surface using calibrated parameters saved under session_id.
    """
    if session_id not in _SESSION_CACHE:
        raise HTTPException(status_code=404, detail="Session expired or not found")
        
    session_data = _SESSION_CACHE[session_id]
    session_data["last_accessed"] = time.time()  # Update access time
    
    model_name = session_data["model_name"]
    params = session_data["params"]
    
    greeks_req = {
        "S": req.S,
        "r": req.r,
        "q": req.q
    }
    
    if model_name == "ssvi":
        # Extract theta_atm_1...8 from params
        theta_atm_list = []
        for i in range(1, 9):
            key = f"theta_atm_{i}"
            if key in params:
                theta_atm_list.append(params[key])
            else:
                theta_atm_list.append(0.1)
        greeks_req["theta_atm"] = theta_atm_list
        greeks_req["rho"] = params["rho"]
        greeks_req["eta"] = params["eta"]
        greeks_req["gamma"] = params["gamma"]
    else:
        for k, v in params.items():
            greeks_req[k] = v

    return await compute_model_greeks(model_name, greeks_req)


# ── WebSocket Risk Streaming Endpoint ──────────────────────────────────────────
from fastapi import WebSocket, WebSocketDisconnect
from deepvol.api.websocket_server import ConnectionManager, JSONRouter, TaskGroup

manager = ConnectionManager()
router = JSONRouter(manager)

@app.websocket("/ws/risk")
async def websocket_risk_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        async with TaskGroup() as tg:
            conflated_queue = manager.get_queue(websocket)
            
            async def socket_writer_consumer():
                try:
                    while manager.is_active(websocket):
                        batch = await conflated_queue.get()
                        for item in batch.values():
                            await manager.send_binary(websocket, item)
                except WebSocketDisconnect:
                    pass
                except asyncio.CancelledError:
                    pass

            tg.create_task(socket_writer_consumer())

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                
                if "text" in message:
                    data = orjson.loads(message["text"])
                elif "bytes" in message:
                    data = orjson.loads(message["bytes"])
                else:
                    continue
                await router.handle_message(websocket, data, tg)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("WebSocket endpoint error: %s", exc)
        await manager.disconnect(websocket)



def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="DeepVol FastAPI Server CLI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    uvicorn.run("deepvol.api.server:app", host=args.host, port=args.port, reload=False)


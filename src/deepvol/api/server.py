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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from cachetools import TTLCache
import orjson

import numpy as np
from numpy.typing import NDArray
import torch

# Limit intra-op threads to prevent CPU core thrashing when executing in thread pools
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import math
from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator

from deepvol.api.metrics import (
    setup_metrics,
    fno_pricing_latency_seconds,
    fno_greeks_calculation_seconds,
)
from deepvol.api.compliance import compliance_monitor

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── App ────────────────────────────────────────────────────────────────────────
_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle: preloads models, warms up CUDA graphs, and stops executor."""
    try:
        _load_model()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("FNO model preload skipped: %s", exc)
    try:
        _get_autocall_state()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Autocall ensemble preload skipped: %s", exc
        )
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _init_fx_cross_state()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("FX Cross executor startup failed: %s", exc)

    yield

    try:
        _shutdown_fx_cross_state()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "FX Cross executor shutdown failed: %s", exc
        )


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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)

# ── Global model cache & Lock Management ───────────────────────────────────────


class AsyncReadWriteLock:
    """Non-blocking, asyncio-compatible Read-Write Lock."""

    def __init__(self):
        self._readers = 0
        self._write_lock = None
        self._read_ready = None
        self._loop = None

    def _get_locks(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        if self._loop is not loop or self._write_lock is None or self._read_ready is None:
            self._loop = loop
            self._write_lock = asyncio.Lock()
            self._read_ready = asyncio.Condition()
            self._readers = 0
        return self._write_lock, self._read_ready

    async def acquire_read(self):
        write_lock, read_ready = self._get_locks()
        async with read_ready:
            while write_lock.locked():
                await read_ready.wait()
            self._readers += 1

    async def release_read(self):
        write_lock, read_ready = self._get_locks()
        async with read_ready:
            self._readers -= 1
            if self._readers == 0:
                read_ready.notify_all()

    async def acquire_write(self):
        write_lock, read_ready = self._get_locks()
        await write_lock.acquire()
        t0 = asyncio.get_event_loop().time()
        _WRITE_STARVATION_WARN_S = 30.0
        async with read_ready:
            while self._readers > 0:
                await read_ready.wait()
                elapsed = asyncio.get_event_loop().time() - t0
                if elapsed > _WRITE_STARVATION_WARN_S:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "AsyncReadWriteLock: writer waiting >%.0fs for readers to drain "
                        "(%d active reader(s)). Consider reducing concurrent read load.",
                        elapsed,
                        self._readers,
                    )
                    t0 = asyncio.get_event_loop().time()  # reset to avoid log spam

    async def release_write(self):
        write_lock, read_ready = self._get_locks()
        if write_lock.locked():
            write_lock.release()
        async with read_ready:
            read_ready.notify_all()

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
    def __init__(
        self, model_name: str, model: torch.nn.Module, pn: Any, yn: Any, path: Path
    ):
        self.name = model_name
        self.model = model
        self.pn = pn
        self.yn = yn
        self.path = path
        self.rwlock = AsyncReadWriteLock()
        self.last_loaded = time.time()


_MODEL_CACHE: Dict[str, CachedModel] = {}
_MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "pn": None,
    "yn": None,
    "device": None,
    "loaded": False,
}

_WEIGHTS_PATH = (
    Path(__file__).parents[3] / "artifacts" / "weights" / "fno_v2_final_prod.pth"
)
_PARAM_NORM = (
    Path(__file__).parents[3] / "artifacts" / "models" / "param_normalizer_v2.npz"
)
_IV_NORM = Path(__file__).parents[3] / "artifacts" / "models" / "iv_normalizer_v2.npz"

# FNO training grids (must match training exactly)
_MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STRIKES = np.linspace(-0.5, 0.5, 11, dtype=np.float32)


def _get_cached_container(model_name: str) -> CachedModel:
    model_name = model_name.lower()
    if model_name not in _MODEL_CACHE:
        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.surrogates.normalizers import (
            IVSurfaceNormalizer,
            ParameterNormalizer,
        )

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
            raise HTTPException(
                status_code=400, detail=f"Unsupported model: {model_name}"
            )

        from deepvol.utils.path_helpers import get_project_root

        artifacts_dir = get_project_root() / "artifacts"
        path = artifacts_dir / "weights" / weights_file
        pn_path = artifacts_dir / "models" / pn_file
        yn_path = artifacts_dir / "models" / yn_file

        if not path.exists():
            raise HTTPException(
                status_code=500, detail=f"Weights not found for {model_name} at {path}"
            )
        if not pn_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Parameter normalizer not found for {model_name} at {pn_path}",
            )
        if not yn_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"IV normalizer not found for {model_name} at {yn_path}",
            )

        model = MirrorPaddedFNO2d(param_dim=param_dim)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        pn = ParameterNormalizer.load(str(pn_path))
        yn = IVSurfaceNormalizer.load(str(yn_path))

        from deepvol.arbitrage.projection_layer import (
            DifferentiableArbitrageFreeProjection,
            ArbitrageFreeFNO,
        )

        proj = DifferentiableArbitrageFreeProjection(
            T_grid=torch.tensor(_MATURITIES, dtype=torch.float64, device=device),
            K_grid=torch.tensor(_STRIKES, dtype=torch.float64, device=device),
            S0=1.0,
            is_log_moneyness=True,
        )
        wrapped_model = ArbitrageFreeFNO(
            base_fno=model, projection_layer=proj, normalizer=yn
        ).to(device)

        container = CachedModel(model_name, wrapped_model, pn, yn, path)
        _MODEL_CACHE[model_name] = container

        # Populate _MODEL_STATE for backward compatibility if default model is loaded
        if model_name in ("rough_heston", "default"):
            _MODEL_STATE.update(
                model=wrapped_model, pn=pn, yn=yn, device=device, loaded=True
            )

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
        from deepvol.surrogates.normalizers import (
            IVSurfaceNormalizer,
            ParameterNormalizer,
        )

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
            _MODEL_STATE.update(
                model=cached_container.model,
                pn=cached_container.pn,
                yn=cached_container.yn,
                device=device,
                loaded=True,
            )


def _load_model() -> None:
    """Lazy-load FNO model for backward compatibility."""
    _get_cached_container("rough_heston")


# ── Autocall Model State & Cache ───────────────────────────────────────────────

_AUTOCALL_CACHE: Dict[str, Any] = {
    "ensemble": None,
    "norm_in": None,
    "norm_out": None,
    "guardian": None,
    "tau_ood": 2.17,
    "calib": {},
    "device": None,
    "loaded": False,
}


def _get_autocall_state() -> Dict[str, Any]:
    """Retrieve or lazy-load the Phase D Autocall CorrectionEnsemble and Guardian."""
    if not _AUTOCALL_CACHE["loaded"]:
        import json
        from deepvol.utils.path_helpers import get_project_root
        from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
        from deepvol.training.train_correction import (
            CorrectionInputNormalizer,
            CorrectionOutputNormalizer,
        )
        from deepvol.mrm.autocall_guardian import AutocallModelGuardian

        root = get_project_root()
        weights_dir = root / "artifacts" / "weights"
        scalers_dir = root / "artifacts" / "scalers"

        member_paths = [
            str(weights_dir / f"autocall_correction_mlp_member_{k}.pth")
            for k in range(5)
        ]
        norm_in_path = str(scalers_dir / "correction_input_normalizer.npz")
        norm_out_path = str(scalers_dir / "correction_output_normalizer.npz")
        calib_path = str(weights_dir / "ensemble_calibration.json")

        for p in member_paths:
            if not os.path.exists(p):
                raise HTTPException(
                    status_code=500, detail=f"Member weight not found: {p}"
                )
        if not os.path.exists(norm_in_path) or not os.path.exists(norm_out_path):
            raise HTTPException(
                status_code=500, detail="Correction normalizers not found"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        norm_in = CorrectionInputNormalizer.load(norm_in_path)
        norm_out = CorrectionOutputNormalizer.load(norm_out_path)

        tau_ood = 2.17
        calib = {}
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            tau_ood = float(calib.get("tau_ood", calib.get("tau_ood_p99_bps", 2.17)))

        ensemble = CorrectionEnsemble(
            K=5, in_dim=19, hidden=256, n_layers=4, dropout=0.0
        )
        ensemble.load_members(member_paths, device=device)
        ensemble.eval()

        guardian = AutocallModelGuardian(
            model=ensemble,
            norm_in=norm_in,
            norm_out=norm_out,
            tau_ood=tau_ood,
            device=str(device),
        )

        _AUTOCALL_CACHE.update(
            {
                "ensemble": ensemble,
                "norm_in": norm_in,
                "norm_out": norm_out,
                "guardian": guardian,
                "tau_ood": tau_ood,
                "calib": calib,
                "device": device,
                "loaded": True,
            }
        )
    return _AUTOCALL_CACHE


# ── Pydantic schemas ───────────────────────────────────────────────────────────


class HestonParams(BaseModel):
    """Rough Heston model parameters."""

    kappa: float = Field(..., gt=0, description="Mean-reversion speed κ ∈ (0.5, 5)")
    theta: float = Field(..., gt=0, description="Long-run variance θ ∈ (0.01, 0.25)")
    sigma: float = Field(..., gt=0, description="Vol-of-vol σ ∈ (0.1, 1.5)")
    rho: float = Field(..., le=0, description="Correlation ρ ∈ (-0.95, 0)")
    v0: float = Field(..., gt=0, description="Initial variance V₀ ∈ (0.01, 0.25)")
    H: float = Field(..., gt=0, description="Hurst exponent H ∈ (0.04, 0.15)")

    # Optional fields for compliance testing
    T_grid: Optional[List[float]] = Field(
        default=None, description="Optional maturities grid"
    )
    K_grid: Optional[List[float]] = Field(
        default=None, description="Optional strike grid"
    )
    S0: Optional[float] = Field(default=None, description="Optional spot price")


class GreeksRequest(HestonParams):
    S: float = Field(default=5000.0, gt=0, description="Spot price")


class IVSurfaceResponse(BaseModel):
    surface: List[List[float]]
    T_grid: List[float]
    K_grid: List[float]
    rmse_bps: Optional[float] = None


class GreeksResponse(BaseModel):
    delta: List[List[float]]
    gamma: List[List[float]]
    vega: List[List[float]]
    vanna: List[List[float]]
    volga: List[List[float]]
    iv_surface: List[List[float]]


class VIXResponse(BaseModel):
    vix: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_s: float
    device: str


class DeribitSummaryResponse(BaseModel):
    currency: str
    n_options: int
    atm_iv: Optional[float]
    term_structure: Dict[str, float]
    spot: Optional[float]


class AutocallPriceRequest(BaseModel):
    kappa: float = Field(2.0, description="Heston mean reversion speed")
    theta: float = Field(0.04, description="Heston long-term variance")
    sigma: float = Field(0.3, description="Heston vol of vol")
    rho: float = Field(-0.7, description="Heston spot-vol correlation")
    v0: float = Field(0.04, description="Heston initial variance")
    B: float = Field(1.0, description="Autocall barrier level as fraction of S0")
    coupon: float = Field(0.10, description="Annual coupon rate")
    T: float = Field(1.5, description="Maturity in years")
    n_obs_per_year: float = Field(4.0, description="Observation frequency per year")
    r: float = Field(0.03, description="Risk-free rate")
    use_guardian: bool = Field(
        True, description="Route through SR 26-2 model risk guardian"
    )


class AutocallPriceResponse(BaseModel):
    npv: float
    pde_npv: float
    correction_bps: float
    uncertainty_bps: float
    is_ood: bool
    is_fallback: bool
    fallback_trigger: Optional[str] = None
    fallback_reasons: List[str] = []
    df_dB: float
    latency_ms: float
    tau_ood: float


class AutocallPriceBatchRequest(BaseModel):
    items: List[AutocallPriceRequest] = Field(..., max_length=100)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _make_spatial(device: torch.device) -> torch.Tensor:
    """Build (1, nT, nK, 2) spatial input tensor for FNO."""
    T_grid = torch.tensor(_MATURITIES, dtype=torch.float32)
    K_grid = torch.tensor(_STRIKES, dtype=torch.float32)
    T_norm = (T_grid - T_grid.mean()) / (T_grid.std() + 1e-8)
    K_norm = K_grid / 0.5
    T_mesh, K_mesh = torch.meshgrid(T_norm, K_norm, indexing="ij")
    spatial = torch.stack([T_mesh, K_mesh], dim=-1).unsqueeze(0)  # (1,8,11,2)
    return spatial.to(device)


def _fno_forward(params: HestonParams) -> np.ndarray:
    """Run FNO forward pass, return (8,11) IV surface in decimal."""
    t0 = time.perf_counter()
    container = _get_cached_container("rough_heston")
    model = container.model
    pn = container.pn
    yn = container.yn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H_val = (
        float(pn.mean[5])
        if hasattr(pn, "mean") and pn.mean is not None and len(pn.mean) > 5
        else (params.H if params.H is not None else 0.08)
    )
    theta_arr = np.array(
        [params.kappa, params.theta, params.sigma, params.rho, params.v0, H_val],
        dtype=np.float32,
    )
    theta_t = torch.tensor(theta_arr, dtype=torch.float32, device=device)
    spatial = _make_spatial(device)

    with torch.no_grad():
        theta_norm = pn.transform_tensor(theta_t.unsqueeze(0))
        pred_norm = model(spatial, theta_norm)
        iv_tensor = yn.inverse_transform_tensor(pred_norm).squeeze(0)
        iv_surface = iv_tensor.clamp(min=0.01).cpu().numpy()  # (8,11)

    elapsed = time.perf_counter() - t0
    fno_pricing_latency_seconds.observe(elapsed)

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

        if params.T_grid is not None and params.K_grid is not None:
            S0 = params.S0 if params.S0 is not None else 100.0

            clamped_T_grid = []
            for T in params.T_grid:
                T_clamped, _, _ = compliance_monitor.audit_ood_and_clamp(
                    T, S0, S0, "rough_heston"
                )
                clamped_T_grid.append(T_clamped)

            clamped_K_grid = []
            for K in params.K_grid:
                _, K_clamped, _ = compliance_monitor.audit_ood_and_clamp(
                    1.0, K, S0, "rough_heston"
                )
                clamped_K_grid.append(K_clamped)

            from deepvol.greeks.portfolio_greeks import _bilinear_interp
            import math

            interpolated_surface = np.zeros(
                (len(clamped_T_grid), len(clamped_K_grid)), dtype=np.float32
            )
            for i, T_val in enumerate(clamped_T_grid):
                for j, K_val in enumerate(clamped_K_grid):
                    k_val = math.log(K_val / S0)
                    interpolated_surface[i, j] = _bilinear_interp(
                        _MATURITIES, _STRIKES, surface, T_val, k_val
                    )

            surface_out = interpolated_surface.tolist()
            T_grid_out = clamped_T_grid
            K_grid_out = clamped_K_grid
        else:
            surface_out = surface.tolist()
            T_grid_out = _MATURITIES.tolist()
            K_grid_out = _STRIKES.tolist()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IVSurfaceResponse(
        surface=surface_out,
        T_grid=T_grid_out,
        K_grid=K_grid_out,
    )


# ── Polymorphic Greeks Request Schemas ──────────────────────────────────────────


class BaseGreeksRequest(BaseModel):
    S: float = Field(
        default=100.0, gt=0, description="Spot price of the underlying asset"
    )
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")


class HestonGreeksRequest(BaseGreeksRequest):
    kappa: float = Field(..., gt=0.1, le=10.0, description="Mean-reversion speed")
    theta: float = Field(..., gt=0.01, le=0.5, description="Long-run variance")
    sigma: float = Field(..., gt=0.05, le=2.0, description="Vol-of-vol")
    rho: float = Field(..., ge=-0.99, le=0.0, description="Asset-vol correlation")
    v0: float = Field(..., gt=0.01, le=0.5, description="Initial variance")
    H: Optional[float] = Field(
        default=None, description="Hurst exponent (optional for Rough Heston)"
    )


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


def _compute_greeks_from_surface(
    iv_surface: np.ndarray, S: float, r: float, q: float
) -> dict:
    from deepvol.greeks.portfolio_greeks import bs_greeks

    T_grid = _MATURITIES
    k_grid = _STRIKES
    nT, nK = len(T_grid), len(k_grid)

    delta_surf = np.zeros((nT, nK), dtype=np.float32)
    gamma_surf = np.zeros((nT, nK), dtype=np.float32)
    vega_surf = np.zeros((nT, nK), dtype=np.float32)
    vanna_surf = np.zeros((nT, nK), dtype=np.float32)
    volga_surf = np.zeros((nT, nK), dtype=np.float32)

    for i in range(nT):
        for j in range(nK):
            T_val = float(T_grid[i])
            kk_val = float(k_grid[j])
            K_val = float(S) * float(np.exp(kk_val))
            sig_val = float(iv_surface[i, j])

            g = bs_greeks(float(S), K_val, T_val, r, sig_val, q=q)

            delta_surf[i, j] = g["delta"]
            gamma_surf[i, j] = g["gamma"]
            vega_surf[i, j] = g["vega"]
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
async def compute_model_greeks(model_name: str, req: Dict[str, Any]) -> GreeksResponse:
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
            raise HTTPException(
                status_code=400, detail=f"Unsupported model name: {model_name}"
            )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))

    try:
        cached_container = _get_cached_container(model_name)

        async with cached_container.rwlock.reader():
            # Build the model parameter array/tensor
            if model_name in ("heston", "rough_heston", "default", "v2"):
                actual_model = (
                    cached_container.model.base_fno
                    if hasattr(cached_container.model, "base_fno")
                    else cached_container.model
                )
                if actual_model.film.mlp[0].in_features == 5:
                    # Classic Heston (5 params)
                    theta_arr = np.array(
                        [
                            parsed_req.kappa,
                            parsed_req.theta,
                            parsed_req.sigma,
                            parsed_req.rho,
                            parsed_req.v0,
                        ],
                        dtype=np.float32,
                    )
                else:
                    # Rough Heston (6 params)
                    # Note: FNO v2 was trained with fixed H=0.08001582; pin H to normalizer mean to avoid catastrophic OOD z-score divergence
                    H_val = (
                        float(cached_container.pn.mean[5])
                        if hasattr(cached_container.pn, "mean")
                        and cached_container.pn.mean is not None
                        and len(cached_container.pn.mean) > 5
                        else 0.08
                    )
                    theta_arr = np.array(
                        [
                            parsed_req.kappa,
                            parsed_req.theta,
                            parsed_req.sigma,
                            parsed_req.rho,
                            parsed_req.v0,
                            H_val,
                        ],
                        dtype=np.float32,
                    )
            elif model_name == "sabr":
                theta_arr = np.array(
                    [parsed_req.alpha, parsed_req.rho, parsed_req.nu], dtype=np.float32
                )
            elif model_name == "ssvi":
                theta_arr = np.array(
                    parsed_req.theta_atm
                    + [parsed_req.rho, parsed_req.eta, parsed_req.gamma],
                    dtype=np.float32,
                )
            elif model_name == "rbergomi":
                theta_arr = np.array(
                    [parsed_req.v0, parsed_req.H, parsed_req.eta, parsed_req.rho],
                    dtype=np.float32,
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            theta_t = torch.tensor(theta_arr, dtype=torch.float32, device=device)
            spatial = _make_spatial(device)

            def _run_forward():
                with torch.no_grad():
                    theta_norm = cached_container.pn.transform_tensor(
                        theta_t.unsqueeze(0)
                    )
                    pred_norm = cached_container.model(spatial, theta_norm)
                    iv_tensor = cached_container.yn.inverse_transform_tensor(
                        pred_norm
                    ).squeeze(0)
                    return iv_tensor.clamp(min=0.01).cpu().numpy()

            iv_surface = await asyncio.get_event_loop().run_in_executor(
                None, _run_forward
            )

            t_greeks_start = time.perf_counter()
            g_results = await asyncio.get_event_loop().run_in_executor(
                None,
                _compute_greeks_from_surface,
                iv_surface,
                parsed_req.S,
                parsed_req.r,
                parsed_req.q,
            )
            elapsed_greeks = time.perf_counter() - t_greeks_start
            fno_greeks_calculation_seconds.observe(elapsed_greeks)

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


@app.get(
    "/deribit/snapshot", response_model=DeribitSummaryResponse, tags=["Market Data"]
)
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
        description="8x11 implied volatility surface in decimal (e.g., 0.20 = 20% IV)",
    )
    n_starts: int = Field(
        default=2, ge=1, le=5, description="Number of random restarts for solver"
    )
    max_iter: int = Field(
        default=25, ge=5, le=100, description="Max iterations per optimization run"
    )
    tol: float = Field(default=1e-5, gt=0, description="MSE convergence tolerance")

    @field_validator("market_iv")
    @classmethod
    def validate_surface(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 8 or any(len(row) != 11 for row in v):
            raise ValueError("implied volatility surface must have dimensions (8, 11)")
        return v


class CalibrateResponse(BaseModel):
    params: Dict[str, float] = Field(
        ..., description="Calibrated model-specific parameters"
    )
    final_mse: float = Field(
        ..., description="Mean squared error of calibrated surface vs target"
    )
    rmse_bps: float = Field(..., description="Root mean squared error in basis points")
    elapsed_ms: float = Field(
        ..., description="Optimization execution time in milliseconds"
    )
    converged: bool = Field(
        ..., description="True if optimization converged within tolerance"
    )


_GPU_CALIBRATION_SEMAPHORE = asyncio.Semaphore(1)


@app.post(
    "/calibrate/{model_name}", response_model=CalibrateResponse, tags=["Calibration"]
)
async def calibrate_route(model_name: str, req: CalibrateRequest) -> CalibrateResponse:
    """
    Calibrate model parameters to an 8x11 market implied volatility surface.

    Supported model names: heston, sabr, ssvi, rbergomi
    """
    model_name = model_name.lower()
    if model_name not in ("heston", "sabr", "ssvi", "rbergomi"):
        raise HTTPException(
            status_code=400, detail=f"Unsupported model name: {model_name}"
        )

    try:
        cached_container = _get_cached_container(model_name)
        iv_target = np.array(req.market_iv, dtype=np.float32)
        if iv_target.shape != (8, 11):
            raise HTTPException(
                status_code=422, detail="market_iv must have shape (8, 11)"
            )

        from deepvol.calibration.calibrate_newton import (
            calibrate_heston,
            calibrate_sabr,
            calibrate_ssvi,
            calibrate_rbergomi,
        )

        # Acquire GPU semaphore to serialize calibration and avoid VRAM exhaustion
        async with _GPU_CALIBRATION_SEMAPHORE:
            # Acquire Read Lock on model weights to prevent reload crashes
            async with cached_container.rwlock.reader():

                def _run():
                    if model_name == "heston":
                        return calibrate_heston(
                            cached_container.model,
                            iv_target,
                            _MATURITIES,
                            _STRIKES,
                            max_iter=req.max_iter,
                            n_starts=req.n_starts,
                        )
                    elif model_name == "sabr":
                        return calibrate_sabr(
                            cached_container.model,
                            iv_target,
                            _MATURITIES,
                            _STRIKES,
                            max_iter=req.max_iter,
                            n_starts=req.n_starts,
                        )
                    elif model_name == "ssvi":
                        return calibrate_ssvi(
                            cached_container.model,
                            iv_target,
                            _MATURITIES,
                            _STRIKES,
                            max_iter=req.max_iter,
                            n_starts=req.n_starts,
                        )
                    elif model_name == "rbergomi":
                        return calibrate_rbergomi(
                            cached_container.model,
                            iv_target,
                            _MATURITIES,
                            _STRIKES,
                            max_iter=req.max_iter,
                            n_starts=req.n_starts,
                        )

                res = await asyncio.get_event_loop().run_in_executor(None, _run)

        # Build response parameters dictionary
        if model_name == "heston":
            p_dict = res["params"]
        elif model_name == "sabr":
            p_dict = {"alpha": res["alpha"], "rho": res["rho"], "nu": res["nu"]}
        elif model_name == "ssvi":
            p_dict = {"rho": res["rho"], "eta": res["eta"], "gamma": res["gamma"]}
            for i, val in enumerate(res["theta_atm"]):
                p_dict[f"theta_atm_{i + 1}"] = float(val)
        elif model_name == "rbergomi":
            p_dict = {
                "v0": res["v0"],
                "H": res["H"],
                "eta": res["eta"],
                "rho": res["rho"],
            }

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
    market_iv: List[List[float]] = Field(
        ..., description="8x11 market implied volatility surface in decimal"
    )
    S0: float = Field(default=100.0, gt=0, description="Initial stock price")
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.015, ge=0, description="Dividend yield")
    epochs: int = Field(
        default=30, ge=1, le=100, description="Number of training epochs"
    )
    N_paths: int = Field(
        default=1024, ge=128, le=10000, description="Number of simulated paths"
    )


class CalibrateNeuralSDEResponse(BaseModel):
    v0: float
    rho: float
    final_rmse: float
    loss_history: List[float]
    elapsed_ms: float
    fitted_iv: Optional[List[List[float]]] = None


class PredictSignatureVolRequest(BaseModel):
    v0: float = Field(default=0.04, gt=0, description="Initial variance")
    ell: List[float] = Field(
        ..., description="Signature volatility coefficients (30 elements)"
    )
    rho: float = Field(default=-0.5, le=0, ge=-1, description="Correlation parameter")
    T: float = Field(default=0.25, gt=0, description="Option maturity in years")
    S0: float = Field(default=100.0, gt=0, description="Initial stock price")
    r: float = Field(default=0.0, ge=0, description="Risk-free rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")
    N_paths: int = Field(
        default=4096, ge=100, le=50000, description="Path simulation count"
    )
    strikes: List[float] = Field(
        default_factory=lambda: [
            80.0,
            85.0,
            90.0,
            95.0,
            100.0,
            105.0,
            110.0,
            115.0,
            120.0,
        ]
    )


class PredictSignatureVolResponse(BaseModel):
    strikes: List[float]
    implied_vols: List[float]
    option_prices: List[float]
    paths_S: Optional[List[List[float]]] = None
    paths_vol: Optional[List[List[float]]] = None


class HedgeSimulateRequest(BaseModel):
    option_type: str = Field(
        ..., description="Option type: 'european', 'barrier', or 'minimax'"
    )
    S0: float = Field(default=100.0, gt=0, description="Initial spot price")
    strike: float = Field(default=100.0, gt=0, description="Option strike price")
    barrier: float = Field(
        default=85.0, gt=0, description="Option knock-out barrier (for barrier style)"
    )
    expiry: float = Field(default=0.1, gt=0, description="Maturity in years")
    mu: float = Field(default=0.0, description="Underlying drift")
    sigma: float = Field(default=0.2, gt=0, description="Asset volatility")
    steps: int = Field(
        default=30, ge=5, le=100, description="Rebalancing step frequency"
    )
    N_paths: int = Field(
        default=100, ge=1, le=1000, description="Path simulation count"
    )
    cost_stock: float = Field(
        default=0.0001, description="Stock transaction cost coeff"
    )
    cost_vol: float = Field(
        default=0.0005, description="Volatility option transaction cost coeff"
    )


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
    d1 = (torch.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (
        sigma * torch.sqrt(T) + 1e-8
    )
    d2 = d1 - sigma * torch.sqrt(T)
    normal = torch.distributions.Normal(0.0, 1.0)
    return S0 * torch.exp(-q * T) * normal.cdf(d1) - K * torch.exp(-r * T) * normal.cdf(
        d2
    )


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


@app.post(
    "/calibrate_neural_sde",
    response_model=CalibrateNeuralSDEResponse,
    tags=["Calibration"],
)
async def calibrate_neural_sde(
    req: CalibrateNeuralSDERequest,
) -> CalibrateNeuralSDEResponse:
    """
    Calibrate a non-parametric Neural SDE model to an 8x11 market implied volatility surface.
    """
    try:
        if len(req.market_iv) != 8 or any(len(row) != 11 for row in req.market_iv):
            raise HTTPException(
                status_code=400, detail="market_iv must be an 8x11 surface"
            )

        from deepvol.models.neural_sde import (
            NeuralSDE,
            NeuralSDEPricer,
            compute_calibration_loss,
        )

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
                        r,
                        q,
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
                    S0=S0,
                    strikes=strikes_mkt,
                    maturities=maturities_mkt,
                    N_paths=req.N_paths,
                    dt=0.01,
                    method="euler",
                )
                loss_dict = compute_calibration_loss(
                    model_prices=pred,
                    market_prices=prices_mkt,
                    vegas=torch.ones_like(prices_mkt),
                    ys=ys,
                    lambda_bound=0.01,
                    epsilon=epsilon,
                )
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

            # Predict final prices for final RMSE
            with torch.no_grad():
                pred_final, _ = pricer.price_options(
                    S0=S0,
                    strikes=strikes_mkt,
                    maturities=maturities_mkt,
                    N_paths=req.N_paths,
                    dt=0.01,
                    method="euler",
                )
                final_rmse = torch.sqrt(
                    torch.mean((pred_final - prices_mkt) ** 2)
                ).item()

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
                        iv_val = _implied_volatility_newton(
                            price_val, S0, strike_val, t, r, q
                        )
                        row_iv.append(iv_val)
                    fitted_iv.append(row_iv)

            return float(pricer.v0.item()), float(sde.rho.item()), final_rmse, fitted_iv

        (
            v0_fitted,
            rho_fitted,
            rmse_fitted,
            fitted_iv,
        ) = await asyncio.get_event_loop().run_in_executor(None, _run_optimization)
        elapsed_ms = (time.time() - t0) * 1000.0

        return CalibrateNeuralSDEResponse(
            v0=v0_fitted,
            rho=rho_fitted,
            final_rmse=rmse_fitted,
            loss_history=loss_history,
            elapsed_ms=elapsed_ms,
            fitted_iv=fitted_iv,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict/signature_vol",
    response_model=PredictSignatureVolResponse,
    tags=["Pricing"],
)
async def predict_signature_vol(
    req: PredictSignatureVolRequest,
) -> PredictSignatureVolResponse:
    """
    Forecasting options smile using the Signature Volatility model.
    """
    try:
        if len(req.ell) != 30:
            raise HTTPException(
                status_code=400, detail="ell coefficients must have exactly 30 elements"
            )

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
                device=device,
            )

            S_T = S[:, -1].detach().cpu().numpy()

            prices = []
            for strike in req.strikes:
                payoff = np.maximum(S_T - strike, 0.0)
                prices.append(float(payoff.mean() * np.exp(-req.r * req.T)))

            # Invert call prices to implied volatility using the robust Newton solver
            implied_vols = []
            for strike, price in zip(req.strikes, prices):
                iv = _implied_volatility_newton(
                    price, req.S0, strike, req.T, req.r, req.q
                )
                implied_vols.append(iv)

            # Extract first 5 sample paths for visualization
            paths_S = S[:5].detach().cpu().numpy().tolist()
            paths_vol = V[:5].detach().cpu().numpy().tolist()

            return implied_vols, prices, paths_S, paths_vol

        (
            implied_vols,
            prices,
            paths_S,
            paths_vol,
        ) = await asyncio.get_event_loop().run_in_executor(None, _run_pricing)

        return PredictSignatureVolResponse(
            strikes=req.strikes,
            implied_vols=implied_vols,
            option_prices=prices,
            paths_S=paths_S,
            paths_vol=paths_vol,
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
        from deepvol.hedging.deep_hedging import (
            HedgingPolicy,
            DeepHedgingEnv,
            simulate_gbm_paths,
        )
        from deepvol.hedging.barrier_hedging import BarrierHedgingEnv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option_type = req.option_type.lower()

        # 1. Initialize policy and load weights
        if option_type == "european":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = (
                Path(__file__).parents[3]
                / "artifacts"
                / "weights"
                / "deep_hedger_european_prod.pth"
            )
        elif option_type == "barrier":
            policy = HedgingPolicy(input_dim=6, hidden_dim=64, output_dim=2).to(device)
            path = (
                Path(__file__).parents[3]
                / "artifacts"
                / "weights"
                / "deep_hedger_barrier_prod.pth"
            )
        elif option_type == "minimax":
            policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
            path = (
                Path(__file__).parents[3]
                / "artifacts"
                / "weights"
                / "minimax_policy_prod.pth"
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported option type: {option_type}"
            )

        if not path.exists():
            raise HTTPException(
                status_code=500, detail=f"Policy weights not found at {path}"
            )

        policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        policy.eval()

        # 2. Simulate asset paths
        H, t_grid = simulate_gbm_paths(
            S0=req.S0,
            mu=req.mu,
            sigma=req.sigma,
            T=req.expiry,
            steps=req.steps,
            N_paths=req.N_paths,
            d=2,
            device=device,
        )

        cost_coeffs = torch.tensor([req.cost_stock, req.cost_vol], device=device)

        # 3. Initialize Env
        if option_type in ("european", "minimax"):
            payoff = torch.clamp(H[:, -1, 0] - req.strike, min=0.0)
            env = DeepHedgingEnv(
                H=H,
                payoff=payoff,
                cost_coeffs=cost_coeffs,
                strike=req.strike,
                expiry=req.expiry,
                risk_aversion=1.0,
                risk_measure="quad",
                t_grid=t_grid,
            )
        else:
            env = BarrierHedgingEnv(
                H=H,
                cost_coeffs=cost_coeffs,
                strike=req.strike,
                barrier=req.barrier,
                expiry=req.expiry,
                risk_aversion=1.0,
                risk_measure="quad",
                t_grid=t_grid,
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
            final_loss=loss,
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
    script_path = str(
        Path(__file__).parents[2] / "scripts" / f"train_fno_{model_name}.py"
    )

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
                cwd=str(Path(__file__).parents[2]),
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
            _TRAINING_TASKS[task_id]["error_message"] = (
                f"Subprocess exited with code {exit_code}"
            )

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
                detail=f"Training for {model_name} is already in progress (Task ID: {tid})",
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
        message=f"Training started for {model_name}. Check progress at /train/status/{task_id}",
    )


@app.get(
    "/train/status/{task_id}", response_model=TrainStatusResponse, tags=["Training"]
)
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
        recent_logs=[line.strip() for line in recent_logs],
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
    S: float = Field(
        default=100.0, gt=0, description="Spot price of the underlying asset"
    )
    r: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    q: float = Field(default=0.0, ge=0, description="Dividend yield")


@app.post(
    "/session/calibrate/{model_name}",
    response_model=SessionCalibrateResponse,
    tags=["Session"],
)
async def session_calibrate(
    model_name: str, req: CalibrateRequest
) -> SessionCalibrateResponse:
    """
    Run standard calibration for a model name and cache the resulting parameters in a session.
    """
    calib_res = await calibrate_route(model_name, req)

    session_id = str(uuid.uuid4())
    _SESSION_CACHE[session_id] = {
        "model_name": model_name,
        "params": calib_res.params,
        "last_accessed": time.time(),
    }

    return SessionCalibrateResponse(
        session_id=session_id,
        model_name=model_name,
        params=calib_res.params,
        rmse_bps=calib_res.rmse_bps,
    )


@app.post(
    "/session/{session_id}/greeks", response_model=GreeksResponse, tags=["Session"]
)
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

    greeks_req = {"S": req.S, "r": req.r, "q": req.q}

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


# ── Autocall Endpoints ────────────────────────────────────────────────────────


@app.post("/autocall/price", response_model=AutocallPriceResponse, tags=["Autocall"])
async def autocall_price(req: AutocallPriceRequest) -> AutocallPriceResponse:
    """
    Price a capital-protected autocallable note using the Phase D CorrectionEnsemble.
    Computes Gatheral second-order effective vol Crank-Nicolson PDE base price,
    applies the 5-member residual MLP correction with epistemic uncertainty,
    and screens through SR 26-2 AutocallModelGuardian.
    """
    state = _get_autocall_state()
    ensemble = state["ensemble"]
    norm_in = state["norm_in"]
    norm_out = state["norm_out"]
    guardian = state["guardian"]
    tau_ood = state["tau_ood"]
    device = state["device"]

    t0 = time.perf_counter()

    from deepvol.calibration.generate_pde_labels import compute_pde_label
    from deepvol.training.train_correction import compute_total_barrier_derivative

    # 1. Crank-Nicolson PDE base price with Gatheral second-order effective vol
    pde_npv, pde_delta, pde_gamma = compute_pde_label(
        v0=req.v0,
        B=req.B,
        coupon=req.coupon,
        T=req.T,
        n_obs_per_year=req.n_obs_per_year,
        r=req.r,
        kappa=req.kappa,
        theta=req.theta,
        sigma=req.sigma,
    )

    raw_params = req.model_dump()
    raw_params["pde_npv"] = float(pde_npv)
    raw_params["pde_delta"] = float(pde_delta)
    raw_params["pde_gamma"] = float(pde_gamma)

    # 2. 19-dim feature vector and ensemble forward pass with routing
    X = norm_in.build_feature_matrix({k: [v] for k, v in raw_params.items()})
    X_t = norm_in.to_tensor(X, device=device)

    with torch.no_grad():
        mean_pred, std_bps_tensor, ood_mask_tensor = ensemble.predict_with_routing(
            X_t, norm_out, tau_ood=tau_ood
        )

    uncertainty_bps = float(std_bps_tensor.item())
    is_ood = bool(ood_mask_tensor.item())

    # 3. Route through Guardian or evaluate surrogate directly
    if req.use_guardian:
        guardian_res = guardian.predict_with_guardian(raw_params)
        total_npv = float(guardian_res["npv"])
        delta_v = float(guardian_res["delta_v"])
        is_fallback = bool(guardian_res["is_fallback"])
        fallback_trigger = guardian_res.get("trigger")
        fallback_reasons = guardian_res.get("reasons", [])
        df_dB = float(guardian_res.get("df_dB") or 0.0)
        if is_fallback:
            is_ood = True
    else:
        delta_v = float(norm_out.inverse_transform_tensor(mean_pred).item())
        total_npv = float(pde_npv) + delta_v
        is_fallback = False
        fallback_trigger = None
        fallback_reasons = []

        try:
            X_t_grad = X_t.clone().detach().requires_grad_(True)
            pred_grad = ensemble._forward_uncompiled(X_t_grad)[0]
            jac = torch.autograd.grad(pred_grad.sum(), X_t_grad, create_graph=False)[0]
            df_dB_tensor = compute_total_barrier_derivative(
                jac,
                {
                    k: torch.tensor([float(v)], device=device)
                    for k, v in raw_params.items()
                },
                norm_in,
                norm_out,
            )
            df_dB = float(df_dB_tensor.item())
        except Exception:
            df_dB = 0.0

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return AutocallPriceResponse(
        npv=total_npv,
        pde_npv=float(pde_npv),
        correction_bps=delta_v * 10000.0,
        uncertainty_bps=uncertainty_bps,
        is_ood=is_ood,
        is_fallback=is_fallback,
        fallback_trigger=fallback_trigger,
        fallback_reasons=fallback_reasons,
        df_dB=df_dB,
        latency_ms=latency_ms,
        tau_ood=tau_ood,
    )


@app.post(
    "/autocall/price_batch",
    response_model=List[AutocallPriceResponse],
    tags=["Autocall"],
)
async def autocall_price_batch(
    req: Union[List[AutocallPriceRequest], AutocallPriceBatchRequest],
) -> List[AutocallPriceResponse]:
    items = req.items if isinstance(req, AutocallPriceBatchRequest) else req
    if len(items) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 100")

    responses: List[AutocallPriceResponse] = []
    for item in items:
        resp = await autocall_price(item)
        responses.append(resp)
    return responses


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


# ── Triangulated FX Cross ML Engine State, Router & Endpoints ─────────────────

from deepvol.api.fx_cross_arrow import (  # noqa: E402
    ARROW_MIME,
    decode_arrow_batch,
    decode_json_batch,
    encode_arrow_response,
    encode_json_response,
)
from deepvol.api.fx_cross_runtime import (  # noqa: E402
    BATCH_BUCKETS,
    QUOTE_BUCKET_GPU,
    FXCrossGPUExecutor,
    FXCrossRuntime,
)
from deepvol.api.schemas_fx_cross import (  # noqa: E402
    CashDeltaLegs,
    DeltaLegs,
    FallbackReasonEnum,
    ModelRiskTelemetry,
    ParityEnum,
    PricingEngineEnum,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
    VegaLegs,
    WarningCodeEnum,
)
from deepvol.market.fx_conventions import (  # noqa: E402
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
)
from deepvol.market.smile_canonicalizer import (  # noqa: E402
    CanonicalSmile,
    SmileArbitrageError,
    SmileCanonicalizer,
)
from deepvol.models.fx_cross_analytic import Col, N_COLS  # noqa: E402
from deepvol.models.fx_cross_surrogate import (  # noqa: E402
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)
from deepvol.mrm.fx_cross_guardian import FXCrossGuardian  # noqa: E402

_FX_CROSS_STATE: Dict[str, Any] = {
    "engine": None,
    "runtime": None,
    "executor": None,
    "guardian": None,
    "canonicalizer": None,
    "model_version": "deepvol-fx-cross-v3.0-b24977e7",
}
_FX_CROSS_RW_LOCK = AsyncReadWriteLock()


def _init_fx_cross_state() -> None:
    import sys

    sys.setswitchinterval(0.0002)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    executor = FXCrossGPUExecutor(runtime=runtime, window_us=200)
    executor.start()
    guardian = FXCrossGuardian()
    canonicalizer = SmileCanonicalizer()

    _FX_CROSS_STATE.update(
        {
            "engine": engine,
            "runtime": runtime,
            "executor": executor,
            "guardian": guardian,
            "canonicalizer": canonicalizer,
            "device": device,
        }
    )


def _shutdown_fx_cross_state() -> None:
    executor = _FX_CROSS_STATE.get("executor")
    if executor is not None:
        executor.stop()
    runtime = _FX_CROSS_STATE.get("runtime")
    if runtime is not None:
        runtime.quote_bucket = None
        runtime.batch_buckets.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _FX_CROSS_STATE.update(
        {
            "engine": None,
            "runtime": None,
            "executor": None,
            "guardian": None,
            "canonicalizer": None,
        }
    )


def _get_fx_cross_state() -> Dict[str, Any]:
    if _FX_CROSS_STATE.get("executor") is None:
        _init_fx_cross_state()
    return _FX_CROSS_STATE


async def reload_fx_cross_weights(weights_path: Union[str, Path]) -> None:
    """Safe model weight reload under AsyncReadWriteLock writer."""
    async with _FX_CROSS_RW_LOCK.writer():
        state = _get_fx_cross_state()
        new_state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state["engine"].net.load_state_dict(new_state)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, state["runtime"].warmup_and_capture)


fx_cross_router = APIRouter(prefix="/fx_cross", tags=["FX Cross"])


@fx_cross_router.get("/health")
async def fx_cross_health() -> Dict[str, Any]:
    """Exposes serving device, model version, and CUDA graph capture state."""
    state = _get_fx_cross_state()
    runtime: FXCrossRuntime = state["runtime"]
    executor: FXCrossGPUExecutor = state["executor"]
    return {
        "status": "ok",
        "device": str(state.get("device", "cpu")),
        "model_version": state["model_version"],
        "graphs_captured": runtime.graphs_captured,
        "quote_bucket_gpu": QUOTE_BUCKET_GPU,
        "batch_buckets": list(BATCH_BUCKETS),
        "queue_depth": executor.queue_depth,
        "executor_alive": executor.is_alive,
    }


async def _fx_cross_price_impl(
    req: TriangulatedFXOptionRequest,
) -> TriangulatedFXOptionResponse:
    state = _get_fx_cross_state()
    guardian: FXCrossGuardian = state["guardian"]
    canonicalizer: SmileCanonicalizer = state["canonicalizer"]
    executor: FXCrossGPUExecutor = state["executor"]

    as_of = req.contract.as_of_date
    exp_dt = req.contract.expiration_date
    set_dt = req.contract.settlement_date
    sec_year = 365.0 * 86400.0
    T_val = (exp_dt - as_of).total_seconds() / sec_year
    T_set_val = (set_dt - as_of).total_seconds() / sec_year

    leg1 = req.market_data.leg1
    leg2 = req.market_data.leg2
    s1, s2 = leg1.spot, leg2.spot
    d1_f, d1_d = leg1.foreign_discount_factor, leg1.domestic_discount_factor
    d2_f, d2_d = leg2.foreign_discount_factor, leg2.domestic_discount_factor
    D_A, _, D_C = d1_f, d1_d, d2_d
    D_set = req.market_data.settlement_discount_factor
    rho = req.market_data.correlation
    phi = 1.0 if req.contract.parity == ParityEnum.CALL else -1.0
    g = D_A / D_C
    F = s1 * s2 * g
    K = req.contract.strike

    ood_from_smile = False
    try:
        q1 = torch.tensor(
            [
                [
                    leg1.smile.atm_vol,
                    leg1.smile.rr25,
                    leg1.smile.bf25,
                    leg1.smile.rr10,
                    leg1.smile.bf10,
                ]
            ],
            dtype=torch.float64,
        )
        conv1 = [
            (
                DeltaConvention(leg1.smile.delta_convention.value),
                AtmConvention(leg1.smile.atm_convention.value),
                StrangleConvention(leg1.smile.strangle_convention.value),
            )
        ]
        lm1 = LegMarket(
            spot=torch.tensor([s1], dtype=torch.float64),
            df_foreign=torch.tensor([d1_f], dtype=torch.float64),
            df_domestic=torch.tensor([d1_d], dtype=torch.float64),
            T=torch.tensor([max(T_val, 1e-4)], dtype=torch.float64),
        )
        cs1 = canonicalizer.canonicalize(q1, conv1, lm1, compute_jacobian=False)
    except (ValueError, SmileArbitrageError) as e:
        raise HTTPException(
            status_code=422, detail=f"SMILE_ARBITRAGE: leg1 smile calibration failed ({e})"
        )

    try:
        q2 = torch.tensor(
            [
                [
                    leg2.smile.atm_vol,
                    leg2.smile.rr25,
                    leg2.smile.bf25,
                    leg2.smile.rr10,
                    leg2.smile.bf10,
                ]
            ],
            dtype=torch.float64,
        )
        conv2 = [
            (
                DeltaConvention(leg2.smile.delta_convention.value),
                AtmConvention(leg2.smile.atm_convention.value),
                StrangleConvention(leg2.smile.strangle_convention.value),
            )
        ]
        lm2 = LegMarket(
            spot=torch.tensor([s2], dtype=torch.float64),
            df_foreign=torch.tensor([d2_f], dtype=torch.float64),
            df_domestic=torch.tensor([d2_d], dtype=torch.float64),
            T=torch.tensor([max(T_val, 1e-4)], dtype=torch.float64),
        )
        cs2 = canonicalizer.canonicalize(q2, conv2, lm2, compute_jacobian=False)
    except (ValueError, SmileArbitrageError) as e:
        raise HTTPException(
            status_code=422, detail=f"SMILE_ARBITRAGE: leg2 smile calibration failed ({e})"
        )

    r_C = -math.log(D_C) / T_val if T_val > 0 else 0.0
    r_A = -math.log(D_A) / T_val if T_val > 0 else 0.0
    r_set = -math.log(D_set) / T_set_val if T_set_val > 0 else 0.0

    X = np.zeros((1, N_COLS), dtype=np.float64)
    X[0, Col.F] = F
    X[0, Col.K] = K
    X[0, Col.T] = T_val
    X[0, Col.T_SET] = T_set_val
    X[0, Col.D] = D_set
    X[0, Col.G] = g
    X[0, Col.RHO] = rho
    X[0, Col.PHI] = phi
    X[0, Col.SVI1 : Col.SVI1 + 5] = cs1.slice_params[0].cpu().numpy()
    X[0, Col.SVI2 : Col.SVI2 + 5] = cs2.slice_params[0].cpu().numpy()
    X[0, Col.XI1 : Col.XI1 + 5] = cs1.xi[0].cpu().numpy()
    X[0, Col.XI2 : Col.XI2 + 5] = cs2.xi[0].cpu().numpy()
    X[0, Col.DELTA1] = 0.0
    X[0, Col.DELTA2] = 0.0
    X[0, Col.R_C] = r_C
    X[0, Col.R_A] = r_A
    X[0, Col.R_SET] = r_set
    X[0, Col.NN_GATE] = 1.0

    X_t = torch.from_numpy(X).to(dtype=torch.float64)
    decision = guardian.route(X_t, engine=req.execution_options.engine)

    sig_ref = math.sqrt(
        leg1.smile.atm_vol**2
        + leg2.smile.atm_vol**2
        + 2.0 * rho * leg1.smile.atm_vol * leg2.smile.atm_vol
    )
    z = math.log(K / F) / (sig_ref * math.sqrt(max(T_val, 1e-4)))

    fallback_activated = False
    fallback_reason: Optional[FallbackReasonEnum] = None
    engine_used = req.execution_options.engine

    if ood_from_smile:
        fallback_activated = True
        fallback_reason = FallbackReasonEnum.OOD_MARKET_REGIME
        engine_used = PricingEngineEnum.ANALYTIC_TRIANGULATED
        X[0, Col.NN_GATE] = 0.0
    elif abs(z) >= 7.0:
        fallback_activated = True
        fallback_reason = FallbackReasonEnum.EXTREME_MONEYNESS
        engine_used = PricingEngineEnum.ANALYTIC_TRIANGULATED
        X[0, Col.NN_GATE] = 0.0
    elif bool(decision.fallback_activated[0].item()):
        fallback_activated = True
        reason_str = decision.fallback_reason[0] or "OOD_MARKET_REGIME"
        fallback_reason = (
            FallbackReasonEnum(reason_str)
            if reason_str in FallbackReasonEnum._value2member_map_
            else FallbackReasonEnum.OOD_MARKET_REGIME
        )
        engine_used = PricingEngineEnum.ANALYTIC_TRIANGULATED
        X[0, Col.NN_GATE] = 0.0
    elif (
        req.execution_options.engine == PricingEngineEnum.ANALYTIC_TRIANGULATED
        or req.execution_options.engine == PricingEngineEnum.ANALYTIC_GK
    ):
        fallback_activated = True
        fallback_reason = FallbackReasonEnum.ENGINE_REQUESTED
        engine_used = PricingEngineEnum.ANALYTIC_TRIANGULATED
        X[0, Col.NN_GATE] = 0.0

    t0 = time.perf_counter()
    fut = executor.submit(X, single=True)
    res_rows = await fut
    latency_ms = (time.perf_counter() - t0) * 1e3
    res = res_rows[0]

    product_val = float(res[0])
    implied_vol = float(res[1])
    delta_cross = float(res[2])
    cash_delta_val = float(res[3])
    gamma_val = float(res[4])
    vega_cross = float(res[5])
    vega_l1 = float(res[6])
    vega_l2 = float(res[7])
    theta_val = float(res[8])
    vanna_val = float(res[9])
    vomma_val = float(res[10])
    corr_sens = float(res[11])

    delta_l1 = cash_delta_val / s1
    delta_l2 = cash_delta_val / s2

    warnings: List[WarningCodeEnum] = []
    if 5.0 < abs(z) < 7.0:
        warnings.append(WarningCodeEnum.NN_TAPERED)
    if bool(cs1.used_ssvi.any().item()) or bool(cs2.used_ssvi.any().item()):
        warnings.append(WarningCodeEnum.SMILE_SSVI_FALLBACK)

    telemetry = ModelRiskTelemetry(
        engine_used=engine_used,
        standardized_moneyness=float(z),
        mahalanobis_distance=float(decision.D_M[0].item()),
        ood_threshold=float(guardian.ood_detector.threshold)
        if guardian.ood_detector is not None
        else 5.35,
        is_out_of_distribution=bool(decision.is_ood[0].item()) or ood_from_smile,
        ensemble_epistemic_std_bps=0.0,
        fallback_activated=fallback_activated,
        fallback_reason=fallback_reason,
        durrleman_min_g=float(min(cs1.min_g[0].item(), cs2.min_g[0].item())),
        smile_dynamics=req.execution_options.smile_dynamics,
        model_version=state["model_version"],
    )

    return TriangulatedFXOptionResponse(
        status="done",
        product_value=product_val,
        implied_vol_cross=implied_vol,
        forward_cross=float(F),
        delta=DeltaLegs(cross=delta_cross, leg1=delta_l1, leg2=delta_l2),
        cash_delta=CashDeltaLegs(
            cross=cash_delta_val, leg1=cash_delta_val, leg2=cash_delta_val
        ),
        gamma=gamma_val,
        vega=VegaLegs(cross=vega_cross, leg1=vega_l1, leg2=vega_l2),
        theta=theta_val,
        vanna=vanna_val,
        vomma=vomma_val,
        correlation_sensitivity=corr_sens,
        warnings=warnings,
        latency_ms=round(latency_ms, 3),
        telemetry=telemetry,
    )


@fx_cross_router.post("/price", response_model=TriangulatedFXOptionResponse)
async def fx_cross_price(
    req: TriangulatedFXOptionRequest,
) -> TriangulatedFXOptionResponse:
    """Prices a single triangulated cross option with smile-inclusive Greeks."""
    async with _FX_CROSS_RW_LOCK.reader():
        return await _fx_cross_price_impl(req)


async def _fx_cross_price_batch_impl(request: Request) -> Response:
    """Columnar batch evaluation supporting JSON and Apache Arrow IPC content negotiation."""
    state = _get_fx_cross_state()
    guardian: FXCrossGuardian = state["guardian"]
    canonicalizer: SmileCanonicalizer = state["canonicalizer"]
    executor: FXCrossGPUExecutor = state["executor"]

    content_type = request.headers.get("content-type", "application/json")
    body = await request.body()

    def _prepare_inputs() -> Tuple[
        NDArray[np.float64],
        NDArray[np.bool_],
        List[Optional[str]],
        List[float],
        int,
        str,
    ]:
        if ARROW_MIME in content_type:
            batch = decode_arrow_batch(body)
        else:
            batch = decode_json_batch(body)

        sec_year = 365.0 * 86400.0
        S_cnt = len(batch.states)
        state_table = np.zeros((S_cnt, N_COLS), dtype=np.float64)

        q1_list, conv1_list, s1_list, d1f_list, d1d_list = [], [], [], [], []
        q2_list, conv2_list, s2_list, d2f_list, d2d_list = [], [], [], [], []
        T_list, T_set_list, D_set_list, rho_list = [], [], [], []

        for st in batch.states:
            as_of = st.as_of_date
            exp_dt = st.expiration_date
            set_dt = st.settlement_date
            T_val = (exp_dt - as_of).total_seconds() / sec_year
            T_set_val = (set_dt - as_of).total_seconds() / sec_year
            T_list.append(max(T_val, 1e-4))
            T_set_list.append(max(T_set_val, 1e-4))

            leg1 = st.market_data.leg1
            leg2 = st.market_data.leg2
            q1_list.append(
                [
                    leg1.smile.atm_vol,
                    leg1.smile.rr25,
                    leg1.smile.bf25,
                    leg1.smile.rr10,
                    leg1.smile.bf10,
                ]
            )
            conv1_list.append(
                (
                    DeltaConvention(leg1.smile.delta_convention.value),
                    AtmConvention(leg1.smile.atm_convention.value),
                    StrangleConvention(leg1.smile.strangle_convention.value),
                )
            )
            s1_list.append(leg1.spot)
            d1f_list.append(leg1.foreign_discount_factor)
            d1d_list.append(leg1.domestic_discount_factor)

            q2_list.append(
                [
                    leg2.smile.atm_vol,
                    leg2.smile.rr25,
                    leg2.smile.bf25,
                    leg2.smile.rr10,
                    leg2.smile.bf10,
                ]
            )
            conv2_list.append(
                (
                    DeltaConvention(leg2.smile.delta_convention.value),
                    AtmConvention(leg2.smile.atm_convention.value),
                    StrangleConvention(leg2.smile.strangle_convention.value),
                )
            )
            s2_list.append(leg2.spot)
            d2f_list.append(leg2.foreign_discount_factor)
            d2d_list.append(leg2.domestic_discount_factor)

            D_set_list.append(st.market_data.settlement_discount_factor)
            rho_list.append(st.market_data.correlation)

        q1 = torch.tensor(q1_list, dtype=torch.float64)
        lm1 = LegMarket(
            spot=torch.tensor(s1_list, dtype=torch.float64),
            df_foreign=torch.tensor(d1f_list, dtype=torch.float64),
            df_domestic=torch.tensor(d1d_list, dtype=torch.float64),
            T=torch.tensor(T_list, dtype=torch.float64),
        )
        cs1 = canonicalizer.canonicalize(q1, conv1_list, lm1, compute_jacobian=False)

        q2 = torch.tensor(q2_list, dtype=torch.float64)
        lm2 = LegMarket(
            spot=torch.tensor(s2_list, dtype=torch.float64),
            df_foreign=torch.tensor(d2f_list, dtype=torch.float64),
            df_domestic=torch.tensor(d2d_list, dtype=torch.float64),
            T=torch.tensor(T_list, dtype=torch.float64),
        )
        cs2 = canonicalizer.canonicalize(q2, conv2_list, lm2, compute_jacobian=False)

        s1_arr = np.array(s1_list, dtype=np.float64)
        s2_arr = np.array(s2_list, dtype=np.float64)
        d1f_arr = np.array(d1f_list, dtype=np.float64)
        d2d_arr = np.array(d2d_list, dtype=np.float64)
        T_arr = np.array(T_list, dtype=np.float64)
        T_set_arr = np.array(T_set_list, dtype=np.float64)
        D_set_arr = np.array(D_set_list, dtype=np.float64)
        rho_arr = np.array(rho_list, dtype=np.float64)

        g_arr = d1f_arr / d2d_arr
        F_arr = s1_arr * s2_arr * g_arr

        state_table[:, Col.F] = F_arr
        state_table[:, Col.T] = T_arr
        state_table[:, Col.T_SET] = T_set_arr
        state_table[:, Col.D] = D_set_arr
        state_table[:, Col.G] = g_arr
        state_table[:, Col.RHO] = rho_arr
        state_table[:, Col.SVI1 : Col.SVI1 + 5] = cs1.slice_params.cpu().numpy()
        state_table[:, Col.SVI2 : Col.SVI2 + 5] = cs2.slice_params.cpu().numpy()
        state_table[:, Col.XI1 : Col.XI1 + 5] = cs1.xi.cpu().numpy()
        state_table[:, Col.XI2 : Col.XI2 + 5] = cs2.xi.cpu().numpy()
        state_table[:, Col.R_C] = np.where(T_arr > 0, -np.log(d2d_arr) / T_arr, 0.0)
        state_table[:, Col.R_A] = np.where(T_arr > 0, -np.log(d1f_arr) / T_arr, 0.0)
        state_table[:, Col.R_SET] = np.where(T_set_arr > 0, -np.log(D_set_arr) / T_set_arr, 0.0)
        state_table[:, Col.NN_GATE] = 1.0

        atm1 = q1[:, 0].cpu().numpy()
        atm2 = q2[:, 0].cpu().numpy()
        sig_ref = np.sqrt(atm1**2 + atm2**2 + 2.0 * rho_arr * atm1 * atm2)
        state_z_ref = sig_ref * np.sqrt(np.maximum(T_arr, 1e-4))
        state_min_g = np.minimum(cs1.min_g.cpu().numpy(), cs2.min_g.cpu().numpy()).tolist()

        states_t = torch.from_numpy(state_table).to(dtype=torch.float64)
        state_decision = guardian.route(states_t, engine=batch.options.engine)

        N_cnt = len(batch.strike)
        X = np.zeros((N_cnt, N_COLS), dtype=np.float64)
        X[:] = state_table[batch.state_index]
        X[:, Col.K] = batch.strike
        X[:, Col.PHI] = batch.parity

        fallback_mask = np.zeros(N_cnt, dtype=bool)
        fallback_reasons_arr = np.full(N_cnt, "", dtype=object)

        F_arr = X[:, Col.F]
        denom = state_z_ref[batch.state_index]
        denom = np.where(denom <= 0, 1e-4, denom)
        z_arr = np.log(batch.strike / F_arr) / denom

        for i in range(N_cnt):
            s_i = batch.state_index[i]
            if abs(z_arr[i]) >= 7.0:
                fallback_mask[i] = True
                fallback_reasons_arr[i] = FallbackReasonEnum.EXTREME_MONEYNESS.value
                X[i, Col.NN_GATE] = 0.0
            elif bool(state_decision.fallback_activated[s_i].item()):
                fallback_mask[i] = True
                fallback_reasons_arr[i] = (
                    state_decision.fallback_reason[s_i]
                    or FallbackReasonEnum.OOD_MARKET_REGIME.value
                )
                X[i, Col.NN_GATE] = 0.0
            elif batch.options.engine == PricingEngineEnum.ANALYTIC_TRIANGULATED:
                fallback_mask[i] = True
                fallback_reasons_arr[i] = FallbackReasonEnum.ENGINE_REQUESTED.value
                X[i, Col.NN_GATE] = 0.0

        return (
            X,
            fallback_mask,
            fallback_reasons_arr,
            state_min_g,
            S_cnt,
            batch.options.engine.value,
        )

    try:
        (
            X,
            fallback_mask,
            fallback_reasons_arr,
            state_min_g,
            S_cnt,
            engine_used_str,
        ) = await asyncio.to_thread(_prepare_inputs)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    t0 = time.perf_counter()
    fut = executor.submit(X, single=False)
    res = await fut
    latency_ms = (time.perf_counter() - t0) * 1e3

    accept = request.headers.get("accept", "application/json")

    def _assemble_and_encode() -> Tuple[bytes, str]:
        cols = {
            "product_value": res[:, 0],
            "implied_vol_cross": res[:, 1],
            "delta_cross": res[:, 2],
            "cash_delta": res[:, 3],
            "gamma_cross": res[:, 4],
            "vega_cross": res[:, 5],
            "vega_leg1": res[:, 6],
            "vega_leg2": res[:, 7],
            "theta": res[:, 8],
            "vanna_cross": res[:, 9],
            "vomma_cross": res[:, 10],
            "correlation_sensitivity": res[:, 11],
            "fallback_mask": fallback_mask,
            "fallback_reason": fallback_reasons_arr,
        }

        scalars = {
            "status": "done",
            "state_model_reserve_bps": [None] * S_cnt,
            "state_durrleman_min_g": state_min_g,
            "cold_states": S_cnt,
            "engine_used": engine_used_str,
            "model_version": state["model_version"],
            "latency_ms": round(latency_ms, 3),
        }

        if ARROW_MIME in accept:
            meta = {
                "status": "done",
                "state_durrleman_min_g": orjson.dumps(state_min_g).decode("utf-8"),
                "cold_states": str(S_cnt),
                "engine_used": engine_used_str,
                "model_version": state["model_version"],
                "latency_ms": str(round(latency_ms, 3)),
            }
            return encode_arrow_response(cols, meta), ARROW_MIME
        else:
            return encode_json_response(cols, scalars), "application/json"

    resp_bytes, media_type = await asyncio.to_thread(_assemble_and_encode)
    return Response(content=resp_bytes, media_type=media_type)


@fx_cross_router.post("/price_batch")
async def fx_cross_price_batch(request: Request) -> Response:
    """Columnar batch evaluation supporting JSON and Apache Arrow IPC content negotiation."""
    async with _FX_CROSS_RW_LOCK.reader():
        return await _fx_cross_price_batch_impl(request)


app.include_router(fx_cross_router)


def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="DeepVol FastAPI Server CLI")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    uvicorn.run("deepvol.api.server:app", host=args.host, port=args.port, reload=False)

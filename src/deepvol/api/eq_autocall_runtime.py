from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import orjson
from pydantic import ConfigDict, Field
import torch

from deepvol.api.eq_autocall_arrow import (
    MAX_BATCH_TRADES as ARROW_MAX_BATCH_TRADES,
    DecodedEQAutocallArrowBatch,
    decode_eq_autocall_arrow_request,
    encode_eq_autocall_arrow_response,
)
from deepvol.api.schemas_eq_autocall import (
    AutocallQueryMetric,
    AutoCallableProductSpec,
    CashFlowRow,
    DupireModelSpec,
    EQAutocallableTradeRequest,
    EQAutocallableTradeResponse,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall.analytical_baseline import (
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.feature_layout import (
    AutocallGlobalCol,
    AutocallSchedCol,
)
import deepvol.models.autocall.greeks_engine as _greeks_mod
from deepvol.models.autocall.greeks_engine import (
    AutocallGreeksResult,
    BatchedSurrogateGreeksOutput,
    compute_batched_surrogate_greeks,
    compute_eq_autocall_greeks,
)
from deepvol.models.autocall.pre_post_processor import (
    D_MAX_DEFAULT,
    M_MAX_DEFAULT,
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.mrm.eq_autocall_guardian import (
    DEFAULT_GUARDIAN_STATS_PATH,
    AutocallFallbackReason,
    EQAutocallModelGuardian,
    GuardianPricingResult,
    SR262AuditRecord,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)

DEFAULT_CHECKPOINT_PATH: Path = (
    Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"
)


def _load_surrogate_from_ckpt(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaskedEQAutocallSurrogate, dict[str, Any]]:
    """Loads `MaskedEQAutocallSurrogate` using M4 `load_eq_autocall_surrogate` or direct `weights_only=True`."""
    try:
        train_mod = importlib.import_module("deepvol.training.train_eq_autocall")
        loader = getattr(train_mod, "load_eq_autocall_surrogate")
        return loader(checkpoint_path, device=device, dtype=dtype)  # type: ignore[no-any-return]
    except ImportError:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_kwargs: dict[str, Any] = {
            "d_max": int(cfg.get("d_max", D_MAX_DEFAULT)),
            "m_max": int(cfg.get("m_max", M_MAX_DEFAULT)),
            "d_model": int(cfg.get("d_model", 128)),
            "n_egno_layers": int(cfg.get("n_egno_layers", 3)),
            "n_sched_layers": int(cfg.get("n_sched_layers", 2)),
            "n_heads": int(cfg.get("n_heads", 4)),
        }
        for extra_k in ("d_node", "d_edge", "d_sched", "d_global", "n_func_codes", "func_emb_dim"):
            if extra_k in cfg and extra_k in _SURR_INIT_PARAMS:
                model_kwargs[extra_k] = int(cfg[extra_k])
        model = MaskedEQAutocallSurrogate(**model_kwargs).to(device=device, dtype=dtype)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd, strict=True)
        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        return model.eval(), dict(metrics)

try:
    _compiled_mod = importlib.import_module("deepvol.surrogates.compiled_runner")
    _CompiledSurrogateRunner: Any = getattr(_compiled_mod, "CompiledSurrogateRunner", None)
except ImportError:  # Pre-M4 merge fallback
    _CompiledSurrogateRunner = None

logger = logging.getLogger("deepvol.api.eq_autocall_runtime")

EngineMode = Literal["auto", "surrogate", "proxy", "qmc"]
_VALID_ENGINES: frozenset[str] = frozenset({"auto", "surrogate", "proxy", "qmc"})
MAX_BATCH_TRADES: int = int(ARROW_MAX_BATCH_TRADES)
CUDA_MICROBATCH_LIMIT_M48: int = 1024
_TINY_CUDA_BATCH_THRESHOLD: int = 128

_GREEKS_BATCH_PARAMS = frozenset(inspect.signature(compute_batched_surrogate_greeks).parameters.keys())
_ASSEMBLE_SURR_PARAMS = frozenset(inspect.signature(assemble_surrogate_product_value).parameters.keys())
_SURR_INIT_PARAMS = tuple(inspect.signature(MaskedEQAutocallSurrogate.__init__).parameters.keys())

ALL_8_GREEK_QUERIES: tuple[str, ...] = (
    "Delta",
    "Gamma",
    "MultivariateAggregatedCashGamma",
    "Vega",
    "Rho",
    "DividendsSensitivities",
    "Theta",
    "Cega",
)

_MUREX_DAG_MARKER_RE = re.compile(r"(?m)(?:^@[0-9a-fA-F]{40}\s*$|class\s*:\s*ConstantMessageMap)")


class EQAutocallRuntimeResponse(EQAutocallableTradeResponse):
    """Canonical runtime trade response extending `EQAutocallableTradeResponse` with full Greek and MRM fields."""

    model_config = ConfigDict(extra="allow")

    Gamma: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None  # type: ignore[assignment]
    Cega: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None  # type: ignore[assignment]
    fallback_reasons: List[str] = Field(default_factory=list)
    stripped_legacy_fields: List[str] = Field(default_factory=list)
    Option: Optional[float] = None
    Notional: Optional[float] = None
    Coupon: Optional[float] = None
    FundingLeg: Optional[float] = None
    audit_record: Optional[Dict[str, Any]] = None


def _validate_surfaces_for_arbitrage(req: EQAutocallableTradeRequest) -> None:
    """Screens input `VolatilitySurfaces` for corrupt or uncertifiable quotes (REM-S0-02).

    Raises:
        ValueError: With prefix `SMILE_ARBITRAGE:` when volatility quotes are non-finite,
            non-positive (<= 0.0), or exceed 5.00, or when strikes/expiration dates are invalid.
    """
    as_of_dt = req.Model.MarketDataSet.AsOfDate
    for ticker, surf in req.Model.MarketDataSet.VolatilitySurfaces.items():
        quotes = np.asarray(surf.VolatilityQuotes, dtype=np.float64)
        if not np.all(np.isfinite(quotes)):
            raise ValueError(f"SMILE_ARBITRAGE: Non-finite volatility surface quotes for {ticker}.")
        if np.any(quotes <= 0.0) or np.any(quotes > 5.00):
            raise ValueError(
                f"SMILE_ARBITRAGE: Volatility quotes for {ticker} out of certifiable bounds (0.0, 5.00]."
            )
        strikes_attr = getattr(surf, "Strikes", None)
        exp_dates_attr = getattr(surf, "ExpirationDates", None)
        if strikes_attr is None or exp_dates_attr is None:
            continue
        strikes = np.asarray(strikes_attr, dtype=np.float64)
        if not np.all(np.isfinite(strikes)):
            raise ValueError(f"SMILE_ARBITRAGE: Non-finite volatility surface strikes for {ticker}.")
        if np.any(strikes <= 0.0):
            raise ValueError(f"SMILE_ARBITRAGE: Non-positive strikes in surface for {ticker}.")
        if len(np.unique(np.round(strikes, 9))) < 2:
            raise ValueError(f"SMILE_ARBITRAGE: Fewer than 2 distinct strikes in surface for {ticker}.")
        prev_exp = as_of_dt
        for idx, exp_dt in enumerate(exp_dates_attr):
            if idx == 0 and exp_dt <= as_of_dt:
                raise ValueError(
                    f"SMILE_ARBITRAGE: ExpirationDate {exp_dt.isoformat()} <= AsOfDate for {ticker}."
                )
            if idx > 0 and exp_dt <= prev_exp:
                raise ValueError(
                    f"SMILE_ARBITRAGE: Non-increasing ExpirationDates in surface for {ticker}."
                )
            prev_exp = exp_dt


def _validate_engine(engine: str) -> str:
    """Strictly validates and normalizes `engine` against `EngineMode` (`auto`, `surrogate`, `proxy`, `qmc`)."""
    eng_norm = str(engine).strip().lower()
    if eng_norm not in _VALID_ENGINES:
        raise ValueError(
            f"Unsupported engine {engine!r}; expected one of {sorted(_VALID_ENGINES)}."
        )
    return eng_norm


def _build_cpu_surrogate_mirror(surrogate: MaskedEQAutocallSurrogate) -> MaskedEQAutocallSurrogate:
    """Constructs an exact CPU float32 mirror matching `surrogate` constructor attributes."""
    candidate_attrs = (
        "d_max",
        "m_max",
        "d_model",
        "n_egno_layers",
        "n_sched_layers",
        "n_heads",
        "d_node",
        "d_edge",
        "d_sched",
        "d_global",
        "init_tau",
        "n_func_codes",
        "func_emb_dim",
    )
    kwargs: dict[str, Any] = {}
    allowed = set(_SURR_INIT_PARAMS)
    for attr in candidate_attrs:
        if attr in allowed and hasattr(surrogate, attr):
            kwargs[attr] = int(getattr(surrogate, attr)) if attr != "init_tau" else float(getattr(surrogate, attr))
    cpu_surr = MaskedEQAutocallSurrogate(**kwargs).to(device="cpu", dtype=torch.float32)
    cpu_surr.load_state_dict(
        {k: v.detach().cpu() for k, v in surrogate.state_dict().items()},
        strict=True,
    )
    cpu_surr.eval()
    return cpu_surr


def _decode_residuals_to_surrogate_output(
    delta_hazard: torch.Tensor,
    delta_coupon: torch.Tensor,
    delta_put_loss: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
) -> AutocallSurrogateOutput:
    """Reconstructs `AutocallSurrogateOutput` probabilities from raw residual predictions in float64 put space."""
    active_sched = sched_mask > 0.5
    m_sched = active_sched.to(dtype=sched_feats.dtype)
    sched_clean = torch.where(active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats))

    c_call = sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0) * m_sched
    g_cpn = sched_clean[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)].detach().clamp(0.0, 1.0) * m_sched
    p_proxy = sched_clean[:, :, int(AutocallSchedCol.PROXY_KO_PROB)].clamp(0.0, 1.0) * c_call
    q_proxy = sched_clean[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)].clamp(0.0, 1.0) * m_sched

    cum_p_proxy = torch.cumsum(p_proxy, dim=1)
    surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy)).clamp(min=1e-6)
    h_proxy = (p_proxy / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
    logit_h_proxy = torch.log(h_proxy / (1.0 - h_proxy))

    hazard = c_call * m_sched * torch.sigmoid(logit_h_proxy + delta_hazard)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    surv_prev = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    stopping = surv_prev * hazard

    q_proxy_clamped = q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
    logit_q_proxy = torch.log(q_proxy_clamped / (1.0 - q_proxy_clamped))
    q_gate = (q_proxy * q_proxy) / (q_proxy * q_proxy + 1e-10)
    q_pred = q_gate * torch.sigmoid(logit_q_proxy + delta_coupon)
    coupon = m_sched * (g_cpn + (1.0 - g_cpn) * q_pred)

    k_put_64 = global_feats[:, int(AutocallGlobalCol.K_PUT)].to(torch.float64).clamp(min=0.0)
    if global_feats.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE):
        k_low_64 = global_feats[:, int(AutocallGlobalCol.LOWER_STRIKE)].to(torch.float64).clamp(min=0.0)
    else:
        k_low_64 = torch.zeros_like(k_put_64)
    k_eff_64 = torch.clamp(k_put_64 - k_low_64, min=0.0)
    l_put_proxy_64 = global_feats[:, int(AutocallGlobalCol.L_PUT_PROXY)].to(torch.float64).clamp(min=0.0)
    surv_proxy_term_64 = (1.0 - p_proxy.sum(dim=1)).to(torch.float64).clamp(min=0.0)
    denom_64 = torch.clamp(k_eff_64 * surv_proxy_term_64, min=1e-12)
    eps_r = 1e-12
    r_proxy_64 = torch.clamp(l_put_proxy_64 / denom_64, min=eps_r, max=1.0 - eps_r)
    logit_r_64 = torch.log(r_proxy_64 / (1.0 - r_proxy_64))
    u_put_64 = k_eff_64 * survival[:, -1].to(torch.float64)
    put_loss = (u_put_64 * torch.sigmoid(logit_r_64 + delta_put_loss.to(torch.float64))).to(
        dtype=sched_feats.dtype
    )

    return AutocallSurrogateOutput(
        stopping_probs=stopping,
        survival_probs=survival,
        hazard_rates=hazard,
        coupon_probs=coupon,
        put_loss=put_loss,
        delta_hazard_logits=delta_hazard,
        delta_coupon_logits=delta_coupon,
        delta_put_loss=delta_put_loss,
    )


def _call_with_supported_kwargs(
    fn: Callable[..., Any],
    allowed_params: frozenset[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Forwards only keyword arguments accepted by `fn` (preserves pre-/post-merge shard compatibility)."""
    filtered = {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}
    return fn(*args, **filtered)


def _trim_active_dimensions(
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    asset_mask: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
) -> tuple[int, int]:
    """Computes minimal active `(d_act, m_act)` across a batch using masks."""
    amask_b = asset_mask > 0.5
    smask_b = sched_mask > 0.5
    any_a = amask_b.any(dim=0)
    any_s = smask_b.any(dim=0)
    d_act = int(any_a.nonzero()[-1].item()) + 1 if bool(any_a.any()) else 1
    m_act = int(any_s.nonzero()[-1].item()) + 1 if bool(any_s.any()) else 1
    return min(d_act, int(node_feats.shape[1])), min(m_act, int(sched_feats.shape[1]))


def _concat_batched_greeks(
    chunks: list[BatchedSurrogateGreeksOutput],
) -> BatchedSurrogateGreeksOutput:
    """Concatenates chunked `BatchedSurrogateGreeksOutput` slices along dimension 0."""
    if len(chunks) == 1:
        return chunks[0]
    first = chunks[0]
    kwargs: dict[str, Any] = {}
    for field_name in (
        "product_value",
        "stopping_probs",
        "survival_probs",
        "coupon_probs",
        "put_loss",
        "delta",
        "vega_atm",
        "vega_surface_feats",
        "rho_df",
        "cega",
        "cash_gamma_euler",
        "div_sensitivities",
        "theta",
    ):
        if not hasattr(first, field_name):
            continue
        vals = [getattr(c, field_name) for c in chunks]
        if vals[0] is None:
            kwargs[field_name] = None
        else:
            kwargs[field_name] = torch.cat(vals, dim=0)
    return BatchedSurrogateGreeksOutput(**kwargs)


def _compute_qmc_crn_if_available(
    state: AutocallPreprocState,
    *,
    queries: Sequence[str],
    n_paths: int,
) -> AutocallGreeksResult:
    """Uses M5 `compute_qmc_crn_greeks` when merged, else falls back cleanly to `compute_eq_autocall_greeks`."""
    crn_fn = getattr(_greeks_mod, "compute_qmc_crn_greeks", None)
    if callable(crn_fn):
        try:
            qmc_mod = importlib.import_module("deepvol.models.autocall.dupire_qmc_solver")
            cfg_cls = getattr(qmc_mod, "DupireQMCConfig", None)
            if cfg_cls is not None:
                cfg = cfg_cls(n_paths=int(n_paths))
                return crn_fn(state, queries=queries, config=cfg)  # type: ignore[no-any-return]
            return crn_fn(state, queries=queries)  # type: ignore[no-any-return]
        except Exception:
            pass
    return compute_eq_autocall_greeks(state, queries=queries, surrogate=None)


def _market_dedup_key(req: EQAutocallableTradeRequest) -> str:
    """Computes a stable digest for `Model.MarketDataSet` to deduplicate surface validation in batches."""
    mds = req.Model.MarketDataSet
    raw = orjson.dumps(mds.model_dump(mode="json", by_alias=True))
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


class EQAutocallRuntime:
    """Unified 4-Tier pricing, Greek sensitivity, Murex DAG, and Apache Arrow IPC runtime for Equity Autocallables."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        stats_path: str | Path | None = DEFAULT_GUARDIAN_STATS_PATH,
        device: str | torch.device | None = None,
        d_max: int = D_MAX_DEFAULT,
        m_max: int = M_MAX_DEFAULT,
        guardian: EQAutocallModelGuardian | None = None,
        surrogate: MaskedEQAutocallSurrogate | None = None,
        n_fallback_paths: int = 32768,
        strict_checkpoint: bool = False,
        enable_compiled_runner: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.d_max = int(d_max)
        self.m_max = int(m_max)
        self.n_fallback_paths = int(n_fallback_paths)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.checkpoint_metrics: dict[str, Any] = {}
        self.checkpoint_loaded: bool = False
        self.checkpoint_error: str | None = None
        self.guardian_stats_loaded: bool = False
        self.guardian_stats_error: str | None = None
        self.model_version: str = "eq-autocall-egno-zero-init"

        if surrogate is not None:
            self.surrogate = surrogate.to(device=self.device, dtype=torch.float32).eval()
            self.model_version = "eq-autocall-egno-custom"
            self.checkpoint_loaded = True
        elif self.checkpoint_path is not None and self.checkpoint_path.is_file():
            try:
                loaded_model, loaded_metrics = _load_surrogate_from_ckpt(
                    self.checkpoint_path,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.surrogate = loaded_model
                self.checkpoint_metrics = loaded_metrics
                self.model_version = f"eq-autocall-egno-{self.checkpoint_path.stem}"
                self.checkpoint_loaded = True
            except Exception as exc:
                if strict_checkpoint:
                    raise RuntimeError(
                        f"Failed to load strict checkpoint {self.checkpoint_path}: {exc}"
                    ) from exc
                logger.warning(
                    "Failed to load checkpoint %s (%s); falling back to zero_init_residual_heads.",
                    self.checkpoint_path,
                    exc,
                )
                self.checkpoint_error = str(exc)
                self.surrogate = MaskedEQAutocallSurrogate(
                    d_max=self.d_max,
                    m_max=self.m_max,
                    d_node=16,
                    d_global=16,
                ).to(device=self.device, dtype=torch.float32)
                self.surrogate.zero_init_residual_heads(std=0.0)
                self.surrogate.eval()
        else:
            missing_msg = f"Checkpoint file not found: {self.checkpoint_path}"
            if strict_checkpoint:
                raise FileNotFoundError(missing_msg)
            self.checkpoint_error = missing_msg
            self.surrogate = MaskedEQAutocallSurrogate(
                d_max=self.d_max,
                m_max=self.m_max,
                d_node=16,
                d_global=16,
            ).to(device=self.device, dtype=torch.float32)
            self.surrogate.zero_init_residual_heads(std=0.0)
            self.surrogate.eval()

        # Maintain a CPU mirror of the surrogate for zero-transfer single-trade CPU preprocessing & Greeks
        if self.device.type == "cuda":
            self._cpu_surrogate = _build_cpu_surrogate_mirror(self.surrogate)
        else:
            self._cpu_surrogate = self.surrogate

        self._compiled_runner: Any = None
        if (
            enable_compiled_runner
            and self.device.type == "cuda"
            and _CompiledSurrogateRunner is not None
        ):
            try:
                self._compiled_runner = _CompiledSurrogateRunner(
                    self.surrogate,
                    use_tf32=True,
                    enable_cudagraphs=True,
                    enable_inductor=True,
                )
            except Exception as exc:
                logger.warning("CompiledSurrogateRunner disabled due to init error: %s", exc)
                self._compiled_runner = None

        self.guardian = (
            guardian
            if guardian is not None
            else EQAutocallModelGuardian(
                d_max=self.d_max,
                m_max=self.m_max,
                record_drift_by_default=True,
            )
        )
        if self.stats_path is not None and self.stats_path.is_file():
            try:
                self.guardian.load_calibrated_stats(self.stats_path)
                self.guardian_stats_loaded = True
            except Exception as exc:
                self.guardian_stats_error = str(exc)
                logger.warning(
                    "Failed to load Guardian calibrated stats %s: %s",
                    self.stats_path,
                    exc,
                )
        elif self.stats_path is not None:
            self.guardian_stats_error = f"Guardian stats file not found: {self.stats_path}"

    def _parse_request(
        self,
        req: EQAutocallableTradeRequest | str | Path | Mapping[str, Any],
    ) -> tuple[EQAutocallableTradeRequest, bool]:
        """Normalizes `req` into a validated `EQAutocallableTradeRequest` and returns `(req_model, is_murex_dag)`."""
        if isinstance(req, EQAutocallableTradeRequest):
            return req, False

        if isinstance(req, Path):
            parsed_req, _ = parse_murex_autocall_dag(req)
            return parsed_req, True

        if isinstance(req, str):
            s = req.strip()
            if not _MUREX_DAG_MARKER_RE.search(s) and s.startswith("{") and "\nclass:" not in s:
                try:
                    obj = orjson.loads(s)
                    if isinstance(obj, dict) and "Product" in obj and "Model" in obj:
                        return EQAutocallableTradeRequest.model_validate(obj), False
                except Exception:
                    pass
            parsed_req, _ = parse_murex_autocall_dag(s)
            return parsed_req, True

        if isinstance(req, Mapping):
            raw_dict = dict(req)
            if "dag_text" in raw_dict and isinstance(raw_dict["dag_text"], str):
                parsed_req, _ = parse_murex_autocall_dag(raw_dict["dag_text"])
                return parsed_req, True
            if "Product" in raw_dict and "Model" in raw_dict:
                raw_copy = dict(raw_dict)
                raw_copy.pop("Result", None)
                return EQAutocallableTradeRequest.model_validate(raw_copy), False
            parsed_req, _ = parse_murex_autocall_dag(raw_dict)
            return parsed_req, True

        raise TypeError(f"Unsupported request type: {type(req).__name__}")

    def price_trade(
        self,
        req: EQAutocallableTradeRequest | str | Path | dict[str, Any],
        engine: str = "auto",
        *,
        include_all_greeks: bool | None = None,
    ) -> EQAutocallRuntimeResponse:
        """Prices a single Multi-Asset Equity Autocallable trade and computes requested Greeks in `float64`.

        Accepts:
          - A validated `EQAutocallableTradeRequest`,
          - A raw Python `dict`, or
          - A raw Murex `ConstantMessageMap` string or `Path` (`parse_murex_autocall_dag`).

        Executes:
          - Tier 0 legacy field stripping & Tier 1 `preprocess_eq_autocall` + `compute_analytical_autocall_baseline`
          - Tier 3 `EQAutocallModelGuardian` pre-/post-inference gates
          - Tier 2 `MaskedEQAutocallSurrogate` (or Dupire QMC fallback if guardian trips or `engine == "qmc"`)
          - `compute_eq_autocall_greeks` / `compute_qmc_crn_greeks` for requested `Queries` (`pi_lex^{-1}`).
        """
        t0 = time.perf_counter()
        eng_norm = _validate_engine(engine)
        req_model, is_dag = self._parse_request(req)
        _validate_surfaces_for_arbitrage(req_model)

        stripped_fields = list(req_model.stripped_legacy_fields)
        state: AutocallPreprocState = preprocess_eq_autocall(
            req_model,
            d_max=self.d_max,
            m_max=self.m_max,
        )
        if state.baseline is None:
            state.baseline = compute_analytical_autocall_baseline(state)

        force_qmc = eng_norm == "qmc"
        use_proxy_only = eng_norm == "proxy"

        requested_queries = [str(getattr(q, "value", q)) for q in req_model.Queries]
        compute_all_8 = is_dag if include_all_greeks is None else bool(include_all_greeks)
        if compute_all_8:
            for gq in ALL_8_GREEK_QUERIES:
                if gq not in requested_queries:
                    requested_queries.append(gq)

        greek_queries_present = [
            q for q in requested_queries if q in ALL_8_GREEK_QUERIES or q == "ConvexCega"
        ]
        greeks_out: AutocallGreeksResult | None = None
        greeks_engine_used = "None"

        if use_proxy_only:
            post_proxy = assemble_autocall_cashflows_and_pv(state)
            pv_val = float(post_proxy.product_value.detach().cpu().item())
            se_val = 0.0
            notional_val = float(post_proxy.notional_pv.detach().cpu().item())
            coupon_val = float(post_proxy.coupon_pv.detach().cpu().item())
            option_val = float(post_proxy.option_pv.detach().cpu().item())
            funding_val = float(-post_proxy.funding_pv.detach().cpu().item())
            ccy_val = post_proxy.reporting_currency
            cash_today_val = float(post_proxy.cash_to_settle_today)
            cf_items = post_proxy.cashflows
            engine_used_str = "Tier1_AnalyticalProxy"
            fb_triggered = False
            fb_reasons_str: list[str] = []
            audit_dict: dict[str, Any] | None = None
            active_state = state
            surrogate_for_greeks: MaskedEQAutocallSurrogate | None = None
            if len(greek_queries_present) > 0:
                greeks_engine_used = "Tier1_AnalyticalProxy"
        else:
            greeks_fn = (
                (
                    lambda st, surr: compute_eq_autocall_greeks(
                        st,
                        queries=requested_queries,
                        surrogate=surr,
                    )
                )
                if (len(greek_queries_present) > 0 and not force_qmc)
                else None
            )
            guardian_res: GuardianPricingResult = self.guardian.price_with_guardian(
                state,
                surrogate=self._cpu_surrogate,
                n_fallback_paths=self.n_fallback_paths,
                force_qmc=force_qmc,
                greeks_fn=greeks_fn,
            )
            pv_val = float(guardian_res.product_value)
            se_val = float(guardian_res.standard_error)
            notional_val = float(guardian_res.notional_pv)
            coupon_val = float(guardian_res.coupon_pv)
            option_val = float(guardian_res.option_pv)
            funding_val = float(-guardian_res.funding_pv)
            ccy_val = guardian_res.reporting_currency
            cash_today_val = float(guardian_res.cash_to_settle_today)
            cf_items = guardian_res.cashflows
            engine_used_str = guardian_res.engine_selected
            fb_triggered = bool(guardian_res.fallback_triggered)
            fb_reasons_str = [r.value for r in guardian_res.fallback_reasons]
            active_state = guardian_res.preproc_state
            surrogate_for_greeks = (
                self._cpu_surrogate
                if (not fb_triggered and not force_qmc)
                else None
            )
            if guardian_res.greeks_out is not None:
                greeks_out = guardian_res.greeks_out
                greeks_engine_used = "Tier2_SingleVJP_AD"
            elif len(greek_queries_present) > 0:
                if fb_triggered or force_qmc or engine_used_str == "Tier3_DupireQMC":
                    greeks_engine_used = "Tier4_DupireQMC_CRN"
                else:
                    greeks_engine_used = "Tier1_AnalyticalProxy"
            audit_dict = guardian_res.audit_record.to_dict()
            audit_dict["greeks_engine_used"] = greeks_engine_used

        if len(greek_queries_present) > 0 and greeks_out is None:
            if not use_proxy_only and (fb_triggered or force_qmc or engine_used_str == "Tier3_DupireQMC"):
                greeks_out = _compute_qmc_crn_if_available(
                    active_state,
                    queries=requested_queries,
                    n_paths=self.n_fallback_paths,
                )
            else:
                greeks_out = compute_eq_autocall_greeks(
                    active_state,
                    queries=requested_queries,
                    surrogate=surrogate_for_greeks,
                )

        cf_rows = [
            CashFlowRow.model_construct(
                Description=str(cf.Description),
                Date=cf.Date,
                Amount=float(cf.Amount),
                StandardError=float(cf.StandardError),
                Currency=str(cf.Currency),
                Leg=int(cf.Leg),
                StatusText=str(cf.StatusText),
            )
            for cf in cf_items
        ]

        latency_ms = (time.perf_counter() - t0) * 1e3
        primary_fb_reason = fb_reasons_str[0] if len(fb_reasons_str) > 0 else None

        return EQAutocallRuntimeResponse(
            ProductValue=pv_val,
            ReportingCurrency=ccy_val,
            StandardError=se_val,
            CashToSettleToday=cash_today_val,
            CashFlows=cf_rows,
            Delta=greeks_out.delta if greeks_out is not None else None,
            Gamma=greeks_out.gamma if greeks_out is not None else None,
            MultivariateAggregatedCashGamma=(
                greeks_out.multivariate_aggregated_cash_gamma
                if greeks_out is not None
                else None
            ),
            Vega=greeks_out.vega if greeks_out is not None else None,
            Rho=greeks_out.rho if greeks_out is not None else None,
            DividendsSensitivities=(
                greeks_out.dividends_sensitivities if greeks_out is not None else None
            ),
            Theta=greeks_out.theta if greeks_out is not None else None,
            Cega=greeks_out.cega if greeks_out is not None else None,
            TradeInfo=req_model.TradeInfo,
            engine_used=engine_used_str,
            fallback_triggered=fb_triggered,
            fallback_reason=primary_fb_reason,
            fallback_reasons=fb_reasons_str,
            stripped_legacy_fields=stripped_fields,
            Option=option_val,
            Notional=notional_val,
            Coupon=coupon_val,
            FundingLeg=funding_val,
            audit_record=audit_dict,
            latency_ms=round(latency_ms, 4),
        )

    def price_murex_dag(
        self,
        raw_dag_text: str | Path | dict[str, Any],
        engine: str = "auto",
        *,
        include_all_greeks: bool = True,
    ) -> EQAutocallRuntimeResponse:
        """Parses and prices a raw Murex `ConstantMessageMap` DAG payload (such as `eq_autocallable_request.txt`)."""
        req_model, _ = parse_murex_autocall_dag(raw_dag_text)
        return self.price_trade(
            req_model,
            engine=engine,
            include_all_greeks=include_all_greeks,
        )

    def _run_tensor_microbatch(
        self,
        node_d: torch.Tensor,
        edge_d: torch.Tensor,
        amask_d: torch.Tensor,
        sched_d: torch.Tensor,
        smask_d: torch.Tensor,
        glob_d: torch.Tensor,
        *,
        spots_t0: torch.Tensor | None,
        ref_spots: torch.Tensor | None,
        pv_in_transit: float | torch.Tensor,
        compute_greeks: bool,
        compute_cash_gamma: bool,
        no_notional_repayment: bool,
        autocall_coupon: float,
        force_coupons_on_autocall: bool,
        is_long_basket: torch.Tensor | None,
        w_fund_per_period: torch.Tensor | None,
        lower_strike: torch.Tensor | None,
        funding_pv: float | torch.Tensor,
    ) -> BatchedSurrogateGreeksOutput:
        """Executes one validated tensor microbatch on `self.device`."""
        dev = node_d.device
        batch_size, d_max, _ = node_d.shape
        _, m_max, _ = sched_d.shape

        if compute_greeks:
            return _call_with_supported_kwargs(  # type: ignore[no-any-return]
                compute_batched_surrogate_greeks,
                _GREEKS_BATCH_PARAMS,
                self.surrogate,
                node_d,
                edge_d,
                amask_d,
                sched_d,
                smask_d,
                glob_d,
                spots_t0=spots_t0,
                ref_spots=ref_spots,
                pv_in_transit=pv_in_transit,
                compute_cash_gamma=compute_cash_gamma,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                is_long_basket=is_long_basket,
                w_fund_per_period=w_fund_per_period,
                lower_strike=lower_strike,
            )

        with torch.no_grad():
            node_f32 = node_d.to(dtype=torch.float32)
            edge_f32 = edge_d.to(dtype=torch.float32)
            amask_f32 = amask_d.to(dtype=torch.float32)
            sched_f32 = sched_d.to(dtype=torch.float32)
            smask_f32 = smask_d.to(dtype=torch.float32)
            glob_f32 = glob_d.to(dtype=torch.float32)

            if self._compiled_runner is not None and dev.type == "cuda":
                dhaz, dcpn, dput = self._compiled_runner.predict_residuals(
                    node_f32, edge_f32, amask_f32, sched_f32, smask_f32, glob_f32
                )
                out = _decode_residuals_to_surrogate_output(
                    dhaz, dcpn, dput, sched_f32, smask_f32, glob_f32
                )
            else:
                out = self.surrogate(
                    node_f32, edge_f32, amask_f32, sched_f32, smask_f32, glob_f32
                )
            pv_f64 = _call_with_supported_kwargs(
                assemble_surrogate_product_value,
                _ASSEMBLE_SURR_PARAMS,
                out,
                sched_d,
                smask_d,
                glob_d,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                is_long_basket=is_long_basket,
                pv_in_transit=pv_in_transit,
                funding_pv=funding_pv,
                w_fund_per_period=w_fund_per_period,
                lower_strike=lower_strike,
            )
        return BatchedSurrogateGreeksOutput(
            product_value=pv_f64.detach().to(dtype=torch.float64),
            stopping_probs=out.stopping_probs.detach().to(dtype=torch.float64),
            survival_probs=out.survival_probs.detach().to(dtype=torch.float64),
            coupon_probs=out.coupon_probs.detach().to(dtype=torch.float64),
            put_loss=out.put_loss.detach().to(dtype=torch.float64),
            delta=torch.zeros((batch_size, d_max), dtype=torch.float64, device=dev),
            vega_atm=torch.zeros((batch_size, d_max, 2), dtype=torch.float64, device=dev),
            vega_surface_feats=torch.zeros((batch_size, d_max, 7), dtype=torch.float64, device=dev),
            rho_df=torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev),
            cega=torch.zeros((batch_size, d_max, d_max), dtype=torch.float64, device=dev),
            cash_gamma_euler=torch.zeros(batch_size, dtype=torch.float64, device=dev),
        )

    def price_tensor_batch(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        spots_t0: torch.Tensor | None = None,
        ref_spots: torch.Tensor | None = None,
        pv_in_transit: float | torch.Tensor = 0.0,
        compute_greeks: bool = True,
        compute_cash_gamma: bool = True,
        no_notional_repayment: bool = False,
        autocall_coupon: float = 0.0,
        force_coupons_on_autocall: bool = False,
        is_long_basket: torch.Tensor | None = None,
        w_fund_per_period: torch.Tensor | None = None,
        lower_strike: torch.Tensor | None = None,
        funding_pv: float | torch.Tensor = 0.0,
        run_guardian: bool = True,
    ) -> BatchedSurrogateGreeksOutput:
        """Vectorized GPU/CPU tensor batch execution with microbatch chunking and Guardian telemetry."""
        if node_feats.ndim != 3 or edge_feats.ndim != 4 or sched_feats.ndim != 3 or global_feats.ndim != 2:
            raise ValueError("Invalid tensor rank in price_tensor_batch.")
        batch_size, d_dim, n_col = node_feats.shape
        _, m_dim, s_col = sched_feats.shape
        g_col = int(global_feats.shape[-1])

        if batch_size <= 0 or batch_size > MAX_BATCH_TRADES:
            raise ValueError(f"batch_size={batch_size} out of bounds [1, {MAX_BATCH_TRADES}].")
        if d_dim > self.surrogate.d_max or m_dim > self.surrogate.m_max:
            raise ValueError(
                f"Tensor batch dimensions (D={d_dim}, M={m_dim}) exceed surrogate capacity "
                f"({self.surrogate.d_max}, {self.surrogate.m_max})."
            )
        expected_d_node = int(getattr(self.surrogate, "d_node", 16))
        expected_d_global = int(getattr(self.surrogate, "d_global", 16))
        if n_col != expected_d_node or g_col != expected_d_global:
            raise ValueError(
                f"Feature schema width mismatch: got (d_node={n_col}, d_global={g_col}), "
                f"expected ({expected_d_node}, {expected_d_global})."
            )
        if edge_feats.shape[:3] != (batch_size, d_dim, d_dim) or asset_mask.shape != (batch_size, d_dim):
            raise ValueError("edge_feats or asset_mask shape mismatch with node_feats.")
        if sched_mask.shape != (batch_size, m_dim) or global_feats.shape[0] != batch_size:
            raise ValueError("sched_mask or global_feats batch dimension mismatch.")

        d_act, m_act = _trim_active_dimensions(node_feats, edge_feats, asset_mask, sched_feats, sched_mask)

        pre_diag: dict[str, Any] | None = None
        if run_guardian:
            pre_diag = self.guardian.evaluate_tensor_batch_pre(
                node_feats, edge_feats, asset_mask, sched_feats, sched_mask, global_feats
            )

        dev = self.device
        node_d = node_feats.to(device=dev, dtype=torch.float64)
        edge_d = edge_feats.to(device=dev, dtype=torch.float64)
        amask_d = asset_mask.to(device=dev, dtype=torch.float64)
        sched_d = sched_feats.to(device=dev, dtype=torch.float64)
        smask_d = sched_mask.to(device=dev, dtype=torch.float64)
        glob_d = global_feats.to(device=dev, dtype=torch.float64)
        sp_d = spots_t0.to(device=dev, dtype=torch.float64) if spots_t0 is not None else None
        ref_d = ref_spots.to(device=dev, dtype=torch.float64) if ref_spots is not None else None
        pvit_t = pv_in_transit.to(device=dev, dtype=torch.float64) if isinstance(pv_in_transit, torch.Tensor) else pv_in_transit
        fund_t = funding_pv.to(device=dev, dtype=torch.float64) if isinstance(funding_pv, torch.Tensor) else funding_pv
        ilb_d = is_long_basket.to(device=dev, dtype=torch.float64) if is_long_basket is not None else None
        wf_d = w_fund_per_period.to(device=dev, dtype=torch.float64) if w_fund_per_period is not None else None
        lk_d = lower_strike.to(device=dev, dtype=torch.float64) if lower_strike is not None else None

        need_chunking = (
            dev.type == "cuda"
            and (
                (m_act > 24 and batch_size > CUDA_MICROBATCH_LIMIT_M48)
                or batch_size > CUDA_MICROBATCH_LIMIT_M48
            )
        )
        if not need_chunking:
            out = self._run_tensor_microbatch(
                node_d,
                edge_d,
                amask_d,
                sched_d,
                smask_d,
                glob_d,
                spots_t0=sp_d,
                ref_spots=ref_d,
                pv_in_transit=pvit_t,
                compute_greeks=compute_greeks,
                compute_cash_gamma=compute_cash_gamma,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                is_long_basket=ilb_d,
                w_fund_per_period=wf_d,
                lower_strike=lk_d,
                funding_pv=fund_t,
            )
        else:
            chunk_size = CUDA_MICROBATCH_LIMIT_M48
            chunks: list[BatchedSurrogateGreeksOutput] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                sl = slice(start, end)
                pvit_sl = pvit_t[sl] if isinstance(pvit_t, torch.Tensor) and pvit_t.ndim > 0 else pvit_t
                fund_sl = fund_t[sl] if isinstance(fund_t, torch.Tensor) and fund_t.ndim > 0 else fund_t
                chunks.append(
                    self._run_tensor_microbatch(
                        node_d[sl],
                        edge_d[sl],
                        amask_d[sl],
                        sched_d[sl],
                        smask_d[sl],
                        glob_d[sl],
                        spots_t0=sp_d[sl] if sp_d is not None else None,
                        ref_spots=ref_d[sl] if ref_d is not None else None,
                        pv_in_transit=pvit_sl,
                        compute_greeks=compute_greeks,
                        compute_cash_gamma=compute_cash_gamma,
                        no_notional_repayment=no_notional_repayment,
                        autocall_coupon=autocall_coupon,
                        force_coupons_on_autocall=force_coupons_on_autocall,
                        is_long_basket=ilb_d[sl] if ilb_d is not None else None,
                        w_fund_per_period=wf_d[sl] if wf_d is not None else None,
                        lower_strike=lk_d[sl] if lk_d is not None else None,
                        funding_pv=fund_sl,
                    )
                )
            out = _concat_batched_greeks(chunks)

        if run_guardian and pre_diag is not None:
            post_diag = self.guardian.evaluate_tensor_batch_post(
                out.stopping_probs,
                out.coupon_probs,
                out.put_loss,
                out.product_value,
                smask_d,
                glob_d,
            )
            self.guardian.record_batch_audit(
                trade_id=f"TENSOR_BATCH_{batch_size}",
                pre_diag=pre_diag,
                post_diag=post_diag,
                engine_selected="MaskedEGNO_Surrogate",
                greeks_engine_used="Tier2_BatchedAD" if compute_greeks else "None",
            )
        return out

    def _build_qmc_fallback_response(
        self,
        req_m: EQAutocallableTradeRequest,
        state_k: AutocallPreprocState,
        stripped_f: list[str],
        requested_queries: Sequence[str],
        compute_greeks: bool,
        t0: float,
    ) -> EQAutocallRuntimeResponse:
        """Executes post-gate Tier-3 Dupire QMC fallback directly on an already preprocessed state."""
        greek_queries_present = [
            q for q in requested_queries if q in ALL_8_GREEK_QUERIES or q == "ConvexCega"
        ]
        guardian_res = self.guardian.price_with_guardian(
            state_k,
            surrogate=None,
            n_fallback_paths=self.n_fallback_paths,
            force_qmc=True,
            greeks_fn=None,
        )
        greeks_out: AutocallGreeksResult | None = None
        greeks_engine_used = "None"
        if compute_greeks and len(greek_queries_present) > 0:
            greeks_engine_used = "Tier4_DupireQMC_CRN"
            greeks_out = _compute_qmc_crn_if_available(
                guardian_res.preproc_state,
                queries=requested_queries,
                n_paths=self.n_fallback_paths,
            )

        audit_dict = guardian_res.audit_record.to_dict()
        audit_dict["greeks_engine_used"] = greeks_engine_used
        cf_rows = [
            CashFlowRow.model_construct(
                Description=str(cf.Description),
                Date=cf.Date,
                Amount=float(cf.Amount),
                StandardError=float(cf.StandardError),
                Currency=str(cf.Currency),
                Leg=int(cf.Leg),
                StatusText=str(cf.StatusText),
            )
            for cf in guardian_res.cashflows
        ]
        fb_reasons = [r.value for r in guardian_res.fallback_reasons]
        return EQAutocallRuntimeResponse(
            ProductValue=float(guardian_res.product_value),
            ReportingCurrency=guardian_res.reporting_currency,
            StandardError=float(guardian_res.standard_error),
            CashToSettleToday=float(guardian_res.cash_to_settle_today),
            CashFlows=cf_rows,
            Delta=greeks_out.delta if greeks_out is not None else None,
            Gamma=greeks_out.gamma if greeks_out is not None else None,
            MultivariateAggregatedCashGamma=(
                greeks_out.multivariate_aggregated_cash_gamma
                if greeks_out is not None
                else None
            ),
            Vega=greeks_out.vega if greeks_out is not None else None,
            Rho=greeks_out.rho if greeks_out is not None else None,
            DividendsSensitivities=(
                greeks_out.dividends_sensitivities if greeks_out is not None else None
            ),
            Theta=greeks_out.theta if greeks_out is not None else None,
            Cega=greeks_out.cega if greeks_out is not None else None,
            TradeInfo=req_m.TradeInfo,
            engine_used=guardian_res.engine_selected,
            fallback_triggered=bool(guardian_res.fallback_triggered),
            fallback_reason=fb_reasons[0] if len(fb_reasons) > 0 else None,
            fallback_reasons=fb_reasons,
            stripped_legacy_fields=stripped_f,
            Option=float(guardian_res.option_pv),
            Notional=float(guardian_res.notional_pv),
            Coupon=float(guardian_res.coupon_pv),
            FundingLeg=float(-guardian_res.funding_pv),
            audit_record=audit_dict,
            latency_ms=round((time.perf_counter() - t0) * 1e3, 4),
        )

    def price_batch_json(
        self,
        requests: Sequence[EQAutocallableTradeRequest | dict[str, Any] | str],
        compute_greeks: bool = True,
        engine: str = "auto",
    ) -> list[EQAutocallRuntimeResponse]:
        """Prices a sequence of JSON/Pydantic requests via collated batching with market dedup & SR 26-2 audit."""
        if len(requests) == 0 or len(requests) > MAX_BATCH_TRADES:
            raise ValueError(
                f"Batch requests length {len(requests)} out of bounds [1, {MAX_BATCH_TRADES}]."
            )

        engine_norm = _validate_engine(engine)
        full_jacobian_queries = {"Gamma", "Vega", "Rho", "DividendsSensitivities", "Theta"}

        model_cache: dict[str, DupireModelSpec] = {}
        parsed_items: list[tuple[EQAutocallableTradeRequest, list[str], list[str], set[str], str]] = []
        _standard_keys = {
            "Product",
            "Model",
            "Queries",
            "TradeInfo",
            "Result",
            "stripped_legacy_fields",
        }
        for item in requests:
            req_m: EQAutocallableTradeRequest
            mkt_key: str
            if isinstance(item, Mapping) and "Product" in item and "Model" in item and set(item.keys()) <= _standard_keys:
                raw_model_obj = item["Model"]
                if isinstance(raw_model_obj, Mapping):
                    m_bytes = orjson.dumps(
                        dict(raw_model_obj),
                        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
                    )
                    mkt_key = hashlib.blake2b(m_bytes, digest_size=16).hexdigest()
                    cached_model = model_cache.get(mkt_key)
                    if cached_model is None:
                        cached_model = DupireModelSpec.model_validate(raw_model_obj)
                        temp_req = EQAutocallableTradeRequest.model_construct(
                            Product=AutoCallableProductSpec.model_validate(item["Product"]),
                            Model=cached_model,
                            Queries=[
                                AutocallQueryMetric(q) for q in item.get("Queries", ("ProductValue", "CashFlows"))
                            ],
                            TradeInfo=item.get("TradeInfo"),
                            stripped_legacy_fields=list(item.get("stripped_legacy_fields", [])),
                        )
                        _validate_surfaces_for_arbitrage(temp_req)
                        model_cache[mkt_key] = cached_model
                        req_m = temp_req
                    else:
                        req_m = EQAutocallableTradeRequest.model_construct(
                            Product=AutoCallableProductSpec.model_validate(item["Product"]),
                            Model=cached_model,
                            Queries=[
                                AutocallQueryMetric(q) for q in item.get("Queries", ("ProductValue", "CashFlows"))
                            ],
                            TradeInfo=item.get("TradeInfo"),
                            stripped_legacy_fields=list(item.get("stripped_legacy_fields", [])),
                        )
                else:
                    req_m, _is_dag = self._parse_request(item)
                    mkt_key = _market_dedup_key(req_m)
                    if mkt_key not in model_cache:
                        _validate_surfaces_for_arbitrage(req_m)
                        model_cache[mkt_key] = req_m.Model
                    else:
                        req_m = req_m.model_copy(update={"Model": model_cache[mkt_key]})
            else:
                req_m, _is_dag = self._parse_request(item)
                mkt_key = _market_dedup_key(req_m)
                if mkt_key not in model_cache:
                    _validate_surfaces_for_arbitrage(req_m)
                    model_cache[mkt_key] = req_m.Model
                else:
                    req_m = req_m.model_copy(update={"Model": model_cache[mkt_key]})

            stripped_f = list(req_m.stripped_legacy_fields)
            q_list = [str(getattr(q, "value", q)) for q in req_m.Queries]
            if not compute_greeks:
                q_list = [q for q in q_list if q in ("ProductValue", "CashFlows")]
            parsed_items.append((req_m, stripped_f, q_list, set(q_list), mkt_key))

        results: list[EQAutocallRuntimeResponse | None] = [None] * len(requests)
        buckets: dict[tuple[bool, float, bool], list[tuple[int, AutocallPreprocState, list[AutocallFallbackReason], dict[str, Any]]]] = {}
        pre_state_cache: dict[str, tuple[AutocallPreprocState, bool, list[AutocallFallbackReason], Any, dict[str, Any]]] = {}

        if engine_norm in ("auto", "surrogate"):
            for idx, (req_m, _stripped, _q_list, q_set, mkt_key) in enumerate(parsed_items):
                if compute_greeks and bool(q_set & full_jacobian_queries):
                    continue
                raw_item = requests[idx]
                if isinstance(raw_item, Mapping) and "Product" in raw_item and isinstance(raw_item["Product"], Mapping):
                    prod_bytes = orjson.dumps(
                        dict(raw_item["Product"]),
                        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
                    )
                else:
                    prod_bytes = orjson.dumps(
                        req_m.Product.model_dump(mode="json", by_alias=True),
                        option=orjson.OPT_SORT_KEYS,
                    )
                req_cache_key = f"{mkt_key}:{hashlib.blake2b(prod_bytes, digest_size=16).hexdigest()}"
                cached_pre = pre_state_cache.get(req_cache_key)
                if cached_pre is None:
                    state_i = preprocess_eq_autocall(
                        req_m,
                        d_max=self.d_max,
                        m_max=self.m_max,
                    )
                    if state_i.baseline is None:
                        state_i.baseline = compute_analytical_autocall_baseline(state_i)
                    if state_i.m_rem == 0 or state_i.seasoned_state.already_autocalled:
                        continue
                    pre_passed, pre_reasons, r_psd, diag = self.guardian.evaluate_pre_inference(state_i)
                    if bool(diag.get("correlation_shrunk", False)):
                        _ = self.guardian._apply_shrunk_correlation_to_state(state_i, r_psd)
                    pre_state_cache[req_cache_key] = (state_i, pre_passed, pre_reasons, r_psd, diag)
                else:
                    state_i, pre_passed, pre_reasons, r_psd, diag = cached_pre
                    if state_i.m_rem == 0 or state_i.seasoned_state.already_autocalled:
                        continue
                if (not pre_passed) and engine_norm != "surrogate":
                    continue
                mod_key = (
                    bool(state_i.no_notional_repayment),
                    round(float(state_i.autocall_coupon), 12),
                    bool(state_i.force_coupons_on_autocall),
                )
                buckets.setdefault(mod_key, []).append(
                    (idx, state_i, pre_reasons, diag)
                )

        for (no_notional_flag, ac_cpn, force_cpn_flag), entries in buckets.items():
            t0_batch = time.perf_counter()
            b_indices = [e[0] for e in entries]
            b_states = [e[1] for e in entries]
            b_pre_reasons = [e[2] for e in entries]
            b_pre_diags = [e[3] for e in entries]

            need_batched_greeks = bool(
                compute_greeks
                and any(
                    bool(parsed_items[i][3] & {"Delta", "MultivariateAggregatedCashGamma", "Cega"})
                    for i in b_indices
                )
            )
            need_cg = bool(
                need_batched_greeks
                and any("MultivariateAggregatedCashGamma" in parsed_items[i][3] for i in b_indices)
            )
            b_surr = len(b_states)
            d_max = self.d_max
            node_b = torch.cat([s.node_feats for s in b_states], dim=0)
            edge_b = torch.cat([s.edge_feats for s in b_states], dim=0)
            amask_b = torch.cat([s.asset_mask for s in b_states], dim=0)
            sched_b = torch.cat([s.sched_feats for s in b_states], dim=0)
            smask_b = torch.cat([s.sched_mask for s in b_states], dim=0)
            glob_b = torch.cat([s.global_feats for s in b_states], dim=0)

            spots_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            refs_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            pvit_b = torch.zeros(b_surr, dtype=torch.float64)
            ilb_b = torch.zeros(b_surr, dtype=torch.float64)
            lstrike_b = torch.zeros(b_surr, dtype=torch.float64)
            for k, s in enumerate(b_states):
                d_act = int(s.d_assets)
                spots_b[k, :d_act] = s.spots_t0.detach().cpu().to(dtype=torch.float64)
                refs_b[k, :d_act] = s.ref_spots.detach().cpu().to(dtype=torch.float64)
                pvit_b[k] = float(s.seasoned_state.pv_in_transit)
                ilb_b[k] = 1.0 if bool(s.is_long_basket) else 0.0
                lstrike_b[k] = float(s.lower_strike) if s.lower_strike is not None else 0.0

            batch_out = self.price_tensor_batch(
                node_b,
                edge_b,
                amask_b,
                sched_b,
                smask_b,
                glob_b,
                spots_t0=spots_b,
                ref_spots=refs_b,
                pv_in_transit=pvit_b,
                compute_greeks=need_batched_greeks,
                compute_cash_gamma=need_cg,
                no_notional_repayment=no_notional_flag,
                autocall_coupon=ac_cpn,
                force_coupons_on_autocall=force_cpn_flag,
                is_long_basket=ilb_b,
                lower_strike=lstrike_b,
                run_guardian=False,
            )
            per_trade_ms = ((time.perf_counter() - t0_batch) * 1e3) / max(b_surr, 1)

            stop_np_b = batch_out.stopping_probs.detach().cpu().numpy()
            cpn_np_b = batch_out.coupon_probs.detach().cpu().numpy()
            put_np_b = batch_out.put_loss.detach().cpu().numpy()
            stop_t = batch_out.stopping_probs.detach().cpu()
            cpn_t = batch_out.coupon_probs.detach().cpu()
            put_t = batch_out.put_loss.detach().cpu()
            delta_np = batch_out.delta.detach().cpu().numpy()
            cega_np = batch_out.cega.detach().cpu().numpy()
            cg_np = (
                batch_out.cash_gamma_euler.detach().cpu().numpy()
                if batch_out.cash_gamma_euler is not None
                else np.zeros(b_surr, dtype=np.float64)
            )
            g_engine_label = "Tier2_BatchedAD" if need_batched_greeks else "None"
            post_cache: dict[
                tuple[int, bytes, bytes, float, bool],
                tuple[Any, bool, list[AutocallFallbackReason], list[CashFlowRow]],
            ] = {}
            ts_now = datetime.now(timezone.utc).isoformat()
            audit_batch_append: list[SR262AuditRecord] = []

            for k, orig_idx in enumerate(b_indices):
                req_m, stripped_f, q_list, q_set, _mkt_k = parsed_items[orig_idx]
                state_k = b_states[k]
                pre_reasons_k = b_pre_reasons[k]
                pre_diag_k = b_pre_diags[k]
                m_rem = int(state_k.m_rem)
                d_act = int(state_k.d_assets)
                orig_ids = state_k.original_identifiers
                inv_perm = state_k.inv_lex_perm
                want_cfs = "CashFlows" in q_set

                post_key = (
                    id(state_k),
                    stop_np_b[k, :m_rem].tobytes(),
                    cpn_np_b[k, :m_rem].tobytes(),
                    float(put_np_b[k]),
                    want_cfs,
                )
                cached_post = post_cache.get(post_key)
                if cached_post is None:
                    post_res = assemble_autocall_cashflows_and_pv(
                        state_k,
                        p_stop=stop_t[k, :m_rem],
                        q_cpn=cpn_t[k, :m_rem],
                        l_put=put_t[k],
                        build_cashflow_items=want_cfs,
                    )
                    post_passed, post_reasons_k = self.guardian.evaluate_post_inference(
                        {
                            "stopping_probs": stop_t[k, :m_rem],
                            "coupon_probs": cpn_t[k, :m_rem],
                            "put_loss": put_t[k],
                        },
                        post_res,
                        state_k,
                    )
                    cf_rows_k = (
                        [
                            CashFlowRow.model_construct(
                                Description=str(cf.Description),
                                Date=cf.Date,
                                Amount=float(cf.Amount),
                                StandardError=float(cf.StandardError),
                                Currency=str(cf.Currency),
                                Leg=int(cf.Leg),
                                StatusText=str(cf.StatusText),
                            )
                            for cf in post_res.cashflows
                        ]
                        if want_cfs
                        else []
                    )
                    post_cache[post_key] = (post_res, post_passed, post_reasons_k, cf_rows_k)
                else:
                    post_res, post_passed, post_reasons_k, cf_rows_k = cached_post

                if (not post_passed) and engine_norm != "surrogate":
                    results[orig_idx] = self._build_qmc_fallback_response(
                        req_m,
                        state_k,
                        stripped_f,
                        q_list,
                        compute_greeks=compute_greeks,
                        t0=t0_batch,
                    )
                    continue

                fb_reasons_enum = [*pre_reasons_k, *post_reasons_k]
                fb_reasons = [r.value for r in fb_reasons_enum]
                min_pin_z = float(pre_diag_k.get("min_pin_risk_z", 100.0))
                conf_hw = float(pre_diag_k.get("conformal_half_width_bps", 0.0))
                audit_rec = SR262AuditRecord(
                    trade_id=state_k.trade_info if state_k.trade_info is not None else f"BATCH_{orig_idx}",
                    timestamp_utc=ts_now,
                    pre_gate_passed=(len(pre_reasons_k) == 0),
                    post_gate_passed=bool(post_passed),
                    fallback_triggered=bool(len(fb_reasons) > 0),
                    fallback_reasons=fb_reasons,
                    warnings=list(pre_diag_k.get("warnings", [])),
                    min_corr_eigenvalue_raw=float(pre_diag_k.get("min_corr_eigenvalue_raw", 1.0)),
                    min_corr_eigenvalue_shrunk=float(pre_diag_k.get("min_corr_eigenvalue_shrunk", 1.0)),
                    correlation_shrunk_applied=bool(pre_diag_k.get("correlation_shrunk", False)),
                    mahalanobis_d2=float(pre_diag_k.get("mahalanobis_d2", 0.0)),
                    max_psi_score=float(pre_diag_k.get("max_psi_score", 0.0)),
                    min_pin_risk_z=min_pin_z,
                    engine_selected="MaskedEGNO_Surrogate",
                    conformal_half_width_bps=conf_hw,
                    greeks_engine_used=g_engine_label,
                )
                audit_batch_append.append(audit_rec)

                primary_fb_reason = fb_reasons[0] if len(fb_reasons) > 0 else None
                resp_kwargs: dict[str, Any] = {
                    "ProductValue": float(post_res.product_value.detach().cpu().item()),
                    "ReportingCurrency": post_res.reporting_currency,
                    "StandardError": 0.0,
                    "CashToSettleToday": float(post_res.cash_to_settle_today),
                    "CashFlows": list(cf_rows_k),
                    "TradeInfo": req_m.TradeInfo,
                    "engine_used": "MaskedEGNO_Surrogate",
                    "fallback_triggered": bool(len(fb_reasons) > 0),
                    "fallback_reason": primary_fb_reason,
                    "fallback_reasons": fb_reasons,
                    "stripped_legacy_fields": stripped_f,
                    "Option": float(post_res.option_pv.detach().cpu().item()),
                    "Notional": float(post_res.notional_pv.detach().cpu().item()),
                    "Coupon": float(post_res.coupon_pv.detach().cpu().item()),
                    "FundingLeg": float(-post_res.funding_pv.detach().cpu().item()),
                    "audit_record": audit_rec.to_dict(),
                    "latency_ms": round(float(per_trade_ms), 4),
                }
                if "Delta" in q_set:
                    delta_orig = delta_np[k, :d_act][inv_perm]
                    resp_kwargs["Delta"] = {
                        orig_ids[i]: float(delta_orig[i]) for i in range(d_act)
                    }
                if "MultivariateAggregatedCashGamma" in q_set:
                    resp_kwargs["MultivariateAggregatedCashGamma"] = float(cg_np[k])
                if "Cega" in q_set:
                    cega_orig = cega_np[k, :d_act, :d_act][np.ix_(inv_perm, inv_perm)]
                    resp_kwargs["Cega"] = {
                        orig_ids[i]: {
                            orig_ids[j]: float(cega_orig[i, j]) for j in range(d_act)
                        }
                        for i in range(d_act)
                    }
                results[orig_idx] = EQAutocallRuntimeResponse.model_construct(**resp_kwargs)

            if audit_batch_append:
                with self.guardian._lock:
                    self.guardian.audit_log.extend(audit_batch_append)

        for idx, (req_m, stripped_f, _q_list, _q_set, _mkt_k) in enumerate(parsed_items):
            if results[idx] is None:
                res_i = self.price_trade(
                    req_m,
                    engine=engine_norm,
                    include_all_greeks=False,
                )
                if stripped_f and not res_i.stripped_legacy_fields:
                    res_i = res_i.model_copy(update={"stripped_legacy_fields": stripped_f})
                results[idx] = res_i

        return [r for r in results if r is not None]

    def price_arrow_ipc(
        self,
        body: bytes,
        compute_greeks: bool = True,
    ) -> bytes:
        """Decodes an Apache Arrow IPC request, enforces schema bounds, runs `price_tensor_batch`, and encodes Arrow IPC."""
        t0 = time.perf_counter()
        decoded: DecodedEQAutocallArrowBatch = decode_eq_autocall_arrow_request(body)
        if decoded.d_max > self.d_max or decoded.m_max > self.m_max:
            raise ValueError(
                f"Arrow batch dimensions (d_max={decoded.d_max}, m_max={decoded.m_max}) "
                f"exceed runtime capacity ({self.d_max}, {self.m_max})."
            )
        pvit: float | torch.Tensor = (
            decoded.pv_in_transit
            if decoded.pv_in_transit is not None
            else float(decoded.metadata.get("pv_in_transit", 0.0))
        )
        batch_out = self.price_tensor_batch(
            decoded.node_feats,
            decoded.edge_feats,
            decoded.asset_mask,
            decoded.sched_feats,
            decoded.sched_mask,
            decoded.global_feats,
            spots_t0=decoded.spots_t0,
            ref_spots=decoded.ref_spots,
            pv_in_transit=pvit,
            compute_greeks=compute_greeks,
            compute_cash_gamma=compute_greeks,
            run_guardian=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1e3
        return encode_eq_autocall_arrow_response(
            batch_out,
            latency_ms=latency_ms,
            engine_used="MaskedEGNO_Surrogate",
            metadata_dict=decoded.metadata,
        )


__all__ = [
    "ALL_8_GREEK_QUERIES",
    "CUDA_MICROBATCH_LIMIT_M48",
    "EngineMode",
    "EQAutocallRuntimeResponse",
    "EQAutocallRuntime",
    "MAX_BATCH_TRADES",
]

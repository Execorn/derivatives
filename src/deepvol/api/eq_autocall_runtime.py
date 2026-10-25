from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import orjson
from pydantic import ConfigDict, Field
import torch

from deepvol.api.eq_autocall_arrow import (
    DecodedEQAutocallArrowBatch,
    decode_eq_autocall_arrow_request,
    encode_eq_autocall_arrow_response,
)
from deepvol.api.schemas_eq_autocall import (
    CashFlowRow,
    EQAutocallableTradeRequest,
    EQAutocallableTradeResponse,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall.analytical_baseline import (
    compute_analytical_autocall_baseline,
)
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
    EQAutocallModelGuardian,
    GuardianPricingResult,
)
from deepvol.surrogates.eq_autocall_egno import (
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)
from deepvol.training.train_eq_autocall import (
    DEFAULT_CHECKPOINT_PATH,
    load_eq_autocall_surrogate,
)

logger = logging.getLogger("deepvol.api.eq_autocall_runtime")

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


class EQAutocallRuntime:
    """Unified 4-Tier pricing, Greek sensitivity, Murex DAG, and Apache Arrow IPC runtime for Equity Autocallables."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        device: str | torch.device | None = None,
        d_max: int = D_MAX_DEFAULT,
        m_max: int = M_MAX_DEFAULT,
        guardian: EQAutocallModelGuardian | None = None,
        surrogate: MaskedEQAutocallSurrogate | None = None,
        n_fallback_paths: int = 32768,
    ) -> None:
        self.d_max = int(d_max)
        self.m_max = int(m_max)
        self.n_fallback_paths = int(n_fallback_paths)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_metrics: dict[str, Any] = {}
        self.model_version: str = "eq-autocall-egno-zero-init"

        if surrogate is not None:
            self.surrogate = surrogate.to(device=self.device, dtype=torch.float32).eval()
            self.model_version = "eq-autocall-egno-custom"
        elif self.checkpoint_path is not None and self.checkpoint_path.is_file():
            try:
                loaded_model, loaded_metrics = load_eq_autocall_surrogate(
                    self.checkpoint_path,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.surrogate = loaded_model
                self.checkpoint_metrics = loaded_metrics
                self.model_version = f"eq-autocall-egno-{self.checkpoint_path.stem}"
            except Exception as exc:
                logger.warning(
                    "Failed to load checkpoint %s (%s); falling back to zero_init_residual_heads.",
                    self.checkpoint_path,
                    exc,
                )
                self.surrogate = MaskedEQAutocallSurrogate(
                    d_max=self.d_max,
                    m_max=self.m_max,
                ).to(device=self.device, dtype=torch.float32)
                self.surrogate.zero_init_residual_heads(std=0.0)
                self.surrogate.eval()
        else:
            self.surrogate = MaskedEQAutocallSurrogate(
                d_max=self.d_max,
                m_max=self.m_max,
            ).to(device=self.device, dtype=torch.float32)
            self.surrogate.zero_init_residual_heads(std=0.0)
            self.surrogate.eval()

        # Maintain a CPU mirror of the surrogate for zero-transfer single-trade CPU preprocessing & Greeks
        if self.device.type == "cuda":
            self._cpu_surrogate = MaskedEQAutocallSurrogate(
                d_max=self.surrogate.d_max,
                m_max=self.surrogate.m_max,
                d_model=self.surrogate.d_model,
                n_egno_layers=self.surrogate.n_egno_layers,
                n_sched_layers=self.surrogate.n_sched_layers,
                n_heads=self.surrogate.n_heads,
            ).to(device="cpu", dtype=torch.float32)
            self._cpu_surrogate.load_state_dict(
                {k: v.detach().cpu() for k, v in self.surrogate.state_dict().items()}
            )
            self._cpu_surrogate.eval()
        else:
            self._cpu_surrogate = self.surrogate

        self.guardian = (
            guardian
            if guardian is not None
            else EQAutocallModelGuardian(
                d_max=self.d_max,
                m_max=self.m_max,
                record_drift_by_default=False,
            )
        )

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
          - `compute_eq_autocall_greeks` for all requested `Queries` in original caller asset order (`pi_lex^{-1}`).
        """
        t0 = time.perf_counter()
        req_model, is_dag = self._parse_request(req)
        _validate_surfaces_for_arbitrage(req_model)

        stripped_fields = list(req_model.stripped_legacy_fields)
        state: AutocallPreprocState = preprocess_eq_autocall(
            req_model,
            d_max=self.d_max,
            m_max=self.m_max,
        )
        # Ensure analytical baseline is freshly evaluated in float64
        if state.baseline is None:
            state.baseline = compute_analytical_autocall_baseline(state)

        eng_norm = str(engine).strip().lower()
        force_qmc = eng_norm in ("qmc", "dupire_qmc", "tier3_dupireqmc", "mc")
        use_proxy_only = eng_norm in ("proxy", "analytic", "stulz", "tier1_proxy")

        # Determine requested Queries before pricing so Tier 2 surrogate runs once when Greeks are requested
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

        if use_proxy_only:
            post_proxy = assemble_autocall_cashflows_and_pv(state)
            pv_val = float(post_proxy.product_value.detach().cpu().item())
            se_val = 0.0
            notional_val = float(post_proxy.notional_pv.detach().cpu().item())
            coupon_val = float(post_proxy.coupon_pv.detach().cpu().item())
            option_val = float(post_proxy.option_pv.detach().cpu().item())
            funding_val = float(-post_proxy.funding_pv.detach().cpu().item())
            ccy_val = post_proxy.reporting_currency
            cash_today_val = post_proxy.cash_to_settle_today
            cf_items = post_proxy.cashflows
            engine_used_str = "Tier1_AnalyticalProxy"
            fb_triggered = False
            fb_reasons_str: list[str] = []
            audit_dict: dict[str, Any] | None = None
            active_state = state
            surrogate_for_greeks: MaskedEQAutocallSurrogate | None = None
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
            audit_dict = guardian_res.audit_record.to_dict()
            active_state = guardian_res.preproc_state
            surrogate_for_greeks = (
                self._cpu_surrogate
                if (not fb_triggered and not force_qmc)
                else None
            )
            if guardian_res.greeks_out is not None:
                greeks_out = guardian_res.greeks_out

        if len(greek_queries_present) > 0 and greeks_out is None:
            greeks_out = compute_eq_autocall_greeks(
                active_state,
                queries=requested_queries,
                surrogate=surrogate_for_greeks,
            )

        cf_rows = [
            CashFlowRow(
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
    ) -> BatchedSurrogateGreeksOutput:
        """Vectorized GPU/CPU tensor batch execution with zero Python loops across B (REM-S0-03).

        Outer inputs and returned tensors are strictly `torch.float64` (REM-S0-01), while internal
        `MaskedEQAutocallSurrogate` execution runs in `torch.float32`.
        """
        dev = self.device
        node_d = node_feats.to(device=dev, dtype=torch.float64)
        edge_d = edge_feats.to(device=dev, dtype=torch.float64)
        amask_d = asset_mask.to(device=dev, dtype=torch.float64)
        sched_d = sched_feats.to(device=dev, dtype=torch.float64)
        smask_d = sched_mask.to(device=dev, dtype=torch.float64)
        glob_d = global_feats.to(device=dev, dtype=torch.float64)
        sp_d = spots_t0.to(device=dev, dtype=torch.float64) if spots_t0 is not None else None
        ref_d = ref_spots.to(device=dev, dtype=torch.float64) if ref_spots is not None else None

        if compute_greeks:
            return compute_batched_surrogate_greeks(
                self.surrogate,
                node_d,
                edge_d,
                amask_d,
                sched_d,
                smask_d,
                glob_d,
                spots_t0=sp_d,
                ref_spots=ref_d,
                pv_in_transit=pv_in_transit,
                compute_cash_gamma=compute_cash_gamma,
            )

        # Forward-only vectorized path (compute_greeks=False) with float64 Tier-1 assembly (REM-S0-01)
        batch_size, d_max, _ = node_d.shape
        _, m_max, _ = sched_d.shape
        with torch.no_grad():
            out = self.surrogate(
                node_d.to(dtype=torch.float32),
                edge_d.to(dtype=torch.float32),
                amask_d.to(dtype=torch.float32),
                sched_d.to(dtype=torch.float32),
                smask_d.to(dtype=torch.float32),
                glob_d.to(dtype=torch.float32),
            )
            pv_f64 = assemble_surrogate_product_value(
                out,
                sched_d,
                smask_d,
                glob_d,
                pv_in_transit=pv_in_transit,
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

    def price_batch_json(
        self,
        requests: Sequence[EQAutocallableTradeRequest | dict[str, Any] | str],
        compute_greeks: bool = True,
        engine: str = "auto",
    ) -> list[EQAutocallRuntimeResponse]:
        """Prices a sequence of JSON/Pydantic `EQAutocallableTradeRequest` objects or dicts via collated batching."""
        if len(requests) == 0:
            raise ValueError("Batch requests list must not be empty.")

        engine_norm = str(engine).strip().lower()
        full_jacobian_queries = {"Gamma", "Vega", "Rho", "DividendsSensitivities", "Theta"}

        parsed_items: list[tuple[EQAutocallableTradeRequest, list[str], set[str]]] = []
        for item in requests:
            req_m, _is_dag = self._parse_request(item)
            _validate_surfaces_for_arbitrage(req_m)
            stripped_f = list(req_m.stripped_legacy_fields)
            if not compute_greeks:
                req_m = req_m.model_copy(
                    update={
                        "Queries": [
                            q
                            for q in req_m.Queries
                            if str(getattr(q, "value", q)) in ("ProductValue", "CashFlows")
                        ]
                    }
                )
            q_set = {str(getattr(q, "value", q)) for q in req_m.Queries}
            parsed_items.append((req_m, stripped_f, q_set))

        results: list[EQAutocallRuntimeResponse | None] = [None] * len(requests)
        batchable_indices: list[int] = []
        batchable_states: list[AutocallPreprocState] = []
        batchable_pre_reasons: list[list[Any]] = []

        if engine_norm in ("auto", "surrogate"):
            for idx, (req_m, _stripped, q_set) in enumerate(parsed_items):
                if compute_greeks and bool(q_set & full_jacobian_queries):
                    continue
                state_i = preprocess_eq_autocall(
                    req_m,
                    d_max=self.d_max,
                    m_max=self.m_max,
                )
                if state_i.baseline is None:
                    state_i.baseline = compute_analytical_autocall_baseline(state_i)
                if state_i.m_rem == 0 or state_i.seasoned_state.already_autocalled:
                    continue
                pre_passed, pre_reasons, _r_psd, _diag = self.guardian.evaluate_pre_inference(state_i)
                if (not pre_passed) and engine_norm != "surrogate":
                    continue
                batchable_indices.append(idx)
                batchable_states.append(state_i)
                batchable_pre_reasons.append(pre_reasons)

        if len(batchable_indices) > 0:
            t0_batch = time.perf_counter()
            need_batched_greeks = bool(
                compute_greeks
                and any(
                    bool(parsed_items[i][2] & {"Delta", "MultivariateAggregatedCashGamma", "Cega"})
                    for i in batchable_indices
                )
            )
            need_cg = bool(
                need_batched_greeks
                and any(
                    "MultivariateAggregatedCashGamma" in parsed_items[i][2]
                    for i in batchable_indices
                )
            )
            b_surr = len(batchable_states)
            d_max = self.d_max
            node_b = torch.cat([s.node_feats for s in batchable_states], dim=0)
            edge_b = torch.cat([s.edge_feats for s in batchable_states], dim=0)
            amask_b = torch.cat([s.asset_mask for s in batchable_states], dim=0)
            sched_b = torch.cat([s.sched_feats for s in batchable_states], dim=0)
            smask_b = torch.cat([s.sched_mask for s in batchable_states], dim=0)
            glob_b = torch.cat([s.global_feats for s in batchable_states], dim=0)

            spots_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            refs_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            pvit_b = torch.zeros(b_surr, dtype=torch.float64)
            for k, s in enumerate(batchable_states):
                d_act = int(s.d_assets)
                spots_b[k, :d_act] = s.spots_t0.detach().cpu().to(dtype=torch.float64)
                refs_b[k, :d_act] = s.ref_spots.detach().cpu().to(dtype=torch.float64)
                pvit_b[k] = float(s.seasoned_state.pv_in_transit)

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
            )
            per_trade_ms = ((time.perf_counter() - t0_batch) * 1e3) / max(b_surr, 1)

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

            for k, orig_idx in enumerate(batchable_indices):
                req_m, stripped_f, q_set = parsed_items[orig_idx]
                state_k = batchable_states[k]
                pre_reasons_k = batchable_pre_reasons[k]
                m_rem = int(state_k.m_rem)
                d_act = int(state_k.d_assets)
                orig_ids = state_k.original_identifiers
                inv_perm = state_k.inv_lex_perm

                post_res = assemble_autocall_cashflows_and_pv(
                    state_k,
                    p_stop=stop_t[k, :m_rem],
                    q_cpn=cpn_t[k, :m_rem],
                    l_put=put_t[k],
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
                if (not post_passed) and engine_norm != "surrogate":
                    continue

                cf_rows_k = [
                    CashFlowRow(
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
                fb_reasons = [r.value for r in (*pre_reasons_k, *post_reasons_k)]
                primary_fb_reason = fb_reasons[0] if len(fb_reasons) > 0 else None
                resp_kwargs: dict[str, Any] = {
                    "ProductValue": float(post_res.product_value.detach().cpu().item()),
                    "ReportingCurrency": post_res.reporting_currency,
                    "StandardError": 0.0,
                    "CashToSettleToday": float(post_res.cash_to_settle_today),
                    "CashFlows": cf_rows_k,
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
                results[orig_idx] = EQAutocallRuntimeResponse(**resp_kwargs)

        for idx, (req_m, stripped_f, _q_set) in enumerate(parsed_items):
            if results[idx] is None:
                res_i = self.price_trade(
                    req_m,
                    engine=engine,
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
        """Decodes an Apache Arrow IPC request, runs vectorized `price_tensor_batch`, and encodes an Arrow IPC response."""
        t0 = time.perf_counter()
        decoded: DecodedEQAutocallArrowBatch = decode_eq_autocall_arrow_request(body)
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
    "EQAutocallRuntimeResponse",
    "EQAutocallRuntime",
]

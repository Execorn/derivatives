"""test_eq_autocall_guardian.py — Verification suite for Task 4.1 (Tier 3 SR 26-2 Guardian & Failover Router).

Covers:
  - Gate G3.1 (`test_G3_1_cartesian_stress_matrix_500_states`):
      500-state Cartesian Stress Matrix sweeping extreme spot moneyness M_{0,i} in [0.05, 3.0],
      extreme implied volatilities sigma in [0.02, 2.50], near-singular and broken non-PSD
      correlation matrices (rho in [-0.95, 0.999] + rank-1 perturbations), large discrete cash
      dividends D_pv / S_0 in [0.0, 0.85], imminent barrier pin-risk (tau_1 = 0.01, F_W ~ B_KO),
      and exotic monitoring indices (Rainbow, Outperformance, Lookback, TwinWin, Spread).
      Asserts 0 NaN/Inf, 100% finite ProductValue & CashFlows, and valid SR262AuditRecord on all 500 states.
  - Gate G3.2 (`test_G3_2_real_payload_quirks_missing_repo_duplicate_strikes_sentinels`):
      100% graceful handling of missing RepoSpreadCurves (`POSIRX_RUB`), duplicate strikes &
      zero calendar variance slices (`VKCO_RUB`), and sentinel barriers (`500.0`, `1e-05`) on
      `eq_autocallable_request.txt`.
  - Gate G3.4 (`test_G3_4_guardian_tripwires_and_audit_log`):
      100% tripwire detection accuracy and audit log completeness across all 13 `AutocallFallbackReason`
      codes (pre-inference & post-inference gates, Ledoit-Wolf PSD shrinkage, 16D Mahalanobis OOD,
      and 10-decile PSI drift monitoring).
"""

from __future__ import annotations

import copy
import datetime
import inspect
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from deepvol.api.schemas_eq_autocall import (
    EQAutocallableTradeRequest,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall import (
    AutocallPreprocState,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.pre_post_processor import assemble_autocall_cashflows_and_pv
from deepvol.mrm.eq_autocall_guardian import (
    CHI2_16_0999,
    FEATURE_NAMES_16D,
    AutocallFallbackReason,
    EQAutocallModelGuardian,
    SR262AuditRecord,
    extract_16d_basket_summary,
    project_correlation_to_psd,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


@pytest.fixture(scope="module")
def parsed_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


@pytest.fixture(scope="module")
def reference_state(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> AutocallPreprocState:
    req_model, _ = parsed_dag
    return preprocess_eq_autocall(req_model)


@pytest.fixture(scope="module")
def zero_head_surrogate() -> MaskedEQAutocallSurrogate:
    torch.manual_seed(42)
    model = MaskedEQAutocallSurrogate()
    model.zero_init_residual_heads(std=0.0)
    model.eval()
    return model


def _build_synthetic_autocall_dict(
    *,
    d_assets: int = 3,
    m_periods: int = 4,
    moneyness: float | list[float] = 1.0,
    vol_level: float = 0.30,
    rho: float = 0.50,
    corr_matrix_override: npt.NDArray[np.float64] | None = None,
    div_ratio: float = 0.0,
    tau_first: float = 0.25,
    ko_barrier: float = 1.00,
    cpn_barrier: float = 0.75,
    callability_index: str = "WorstOf",
    coupon_index: str | None = None,
    option_index: str | None = None,
    as_of_str: str = "2026-10-23T00:00:00+00:00",
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Constructs a self-contained synthetic Autocallable request dict for rapid stress sweeps."""
    as_of_dt = datetime.datetime.fromisoformat(as_of_str)
    start_dt = as_of_dt - datetime.timedelta(days=30)

    tickers = [f"ASSET_{i:02d}_RUB" for i in range(d_assets)]
    ref_spots = [100.0 for _ in range(d_assets)]
    if isinstance(moneyness, (int, float)):
        spots_map = {t: 100.0 * float(moneyness) for t in tickers}
    else:
        spots_map = {t: 100.0 * float(moneyness[i]) for i, t in enumerate(tickers)}

    obs_dates: list[str] = []
    pay_dates: list[str] = []
    ko_levels: list[float] = []
    cpn_levels: list[float] = []

    for m in range(m_periods):
        if m == 0:
            days_m = max(1, int(round(tau_first * 365.0)))
        else:
            days_m = max(1, int(round((tau_first + 0.25 * m) * 365.0)))
        obs_dt = as_of_dt + datetime.timedelta(days=days_m)
        pay_dt = obs_dt + datetime.timedelta(days=5)
        obs_dates.append(obs_dt.isoformat())
        pay_dates.append(pay_dt.isoformat())
        ko_levels.append(float(ko_barrier))
        cpn_levels.append(float(cpn_barrier))

    corrs: dict[str, float] = {}
    for i in range(d_assets):
        for j in range(i + 1, d_assets):
            corrs[f"{tickers[i]}&{tickers[j]}"] = float(rho)

    exp_dates = [
        (as_of_dt + datetime.timedelta(days=90)).isoformat(),
        (as_of_dt + datetime.timedelta(days=365)).isoformat(),
        (as_of_dt + datetime.timedelta(days=730)).isoformat(),
    ]
    vol_surfaces: dict[str, Any] = {}
    dividends: dict[str, Any] = {}
    for t in tickers:
        s0 = spots_map[t]
        strikes = [0.5 * s0, 0.8 * s0, 1.0 * s0, 1.2 * s0, 1.5 * s0]
        v = float(vol_level)
        quotes = [
            [max(v + 0.02, 0.01), max(v + 0.01, 0.01), v, max(v - 0.005, 0.01), v],
            [max(v + 0.02, 0.01), max(v + 0.01, 0.01), v, max(v - 0.005, 0.01), v],
            [max(v + 0.02, 0.01), max(v + 0.01, 0.01), v, max(v - 0.005, 0.01), v],
        ]
        vol_surfaces[t] = {
            "Identifier": t,
            "DayCountConvention": "Act365",
            "ExpirationDates": exp_dates,
            "Strikes": strikes,
            "VolatilityQuotes": quotes,
        }
        if div_ratio > 0.0:
            ex_dt = as_of_dt + datetime.timedelta(days=max(1, int(round(0.5 * tau_first * 365.0))))
            dividends[t] = {
                "ExDate": [ex_dt.isoformat()],
                "PaymentDate": [ex_dt.isoformat()],
                "Amount": [float(div_ratio) * s0],
            }

    disc_curve = {
        "Identifier": "RUB_DISC",
        "Currency": "RUB",
        "AsOfDate": as_of_str,
        "InterpolationDayCountConvention": "Act365",
        "InterpolationMethod": "Linear",
        "InterpolationSpaceCoordinates": "Rates",
        "ExtrapolationMethod": "Flat",
        "RateQuoteExpression": "Standard",
        "RatesSchedule": {
            "Dates": [
                (as_of_dt + datetime.timedelta(days=30)).isoformat(),
                (as_of_dt + datetime.timedelta(days=365)).isoformat(),
                (as_of_dt + datetime.timedelta(days=1095)).isoformat(),
            ],
            "Rates": [0.10, 0.10, 0.10],
        },
    }

    req_dict: dict[str, Any] = {
        "TradeInfo": "SYNTH_STRESS_TRADE",
        "Queries": ["ProductValue", "CashFlows"],
        "Product": {
            "TypeName": "AutoCallableProduct",
            "PaymentCurrency": "RUB",
            "StartDate": start_dt.isoformat(),
            "ObservationDates": obs_dates,
            "PaymentDates": pay_dates,
            "DayCountConvention": "Act365",
            "UnderlyingIdentifiers": tickers,
            "UnderlyingReferenceSpots": ref_spots,
            "AutoCallTriggerLevels": ko_levels,
            "CouponBarrierLevels": cpn_levels,
            "FixedCouponAmount": 0.04,
            "IsMemory": True,
            "Strike": 0.90,
            "OptionGearing": 1.11111,
            "OptionDownAndInBarrierLevel": 0.65,
            "AutoCallTriggerAbsoluteDigitalSpread": -0.10,
            "CouponBarrierAbsoluteDigitalSpread": -0.10,
            "OptionDownAndInBarrierAbsoluteDigitalSpread": -0.10,
            "AllowAutocallOnExpiry": False,
            "CallabilityMonitoringIndex": callability_index,
            "CouponBarrierMonitoringIndex": coupon_index,
            "OptionMonitoringIndex": option_index,
        },
        "Model": {
            "FinancingCurveIdentifiers": ["RUB_DISC"],
            "MarketDataSet": {
                "AsOfDate": as_of_str,
                "Spots": spots_map,
                "Correlations": corrs,
                "RatesCurvesBundles": {"RUB_DISC": disc_curve},
                "RepoSpreadCurves": {},
                "Dividends": dividends,
                "VolatilitySurfaces": vol_surfaces,
                "Fixings": {},
            },
        },
    }
    corr_t = (
        torch.as_tensor(corr_matrix_override, dtype=torch.float64)
        if corr_matrix_override is not None
        else None
    )
    return req_dict, corr_t


def _assert_valid_audit_and_result(res: Any) -> None:
    """Asserts 0 NaN/Inf across ProductValue, CashFlows, and SR262AuditRecord."""
    assert np.isfinite(res.product_value), f"Non-finite ProductValue: {res.product_value}"
    assert np.isfinite(res.standard_error) and res.standard_error >= 0.0
    assert np.isfinite(res.notional_pv)
    assert np.isfinite(res.coupon_pv)
    assert np.isfinite(res.option_pv)
    assert np.isfinite(res.funding_pv)
    assert np.isfinite(res.pv_in_transit)
    assert np.isfinite(res.l_put)
    assert np.isfinite(res.l_call)
    assert np.all(np.isfinite(res.p_stop))
    assert np.all(np.isfinite(res.q_cpn))
    assert np.all(np.isfinite(res.shrunk_corr_matrix))
    assert len(res.cashflows) > 0
    for cf in res.cashflows:
        assert np.isfinite(cf.Amount), f"Non-finite CashFlow Amount: {cf}"
        assert np.isfinite(cf.StandardError) and cf.StandardError >= 0.0

    rec: SR262AuditRecord = res.audit_record
    assert isinstance(rec, SR262AuditRecord)
    assert bool(rec.trade_id)
    assert bool(rec.timestamp_utc)
    assert isinstance(rec.pre_gate_passed, bool)
    assert isinstance(rec.post_gate_passed, bool)
    assert isinstance(rec.fallback_triggered, bool)
    assert isinstance(rec.fallback_reasons, list)
    assert isinstance(rec.warnings, list)
    assert np.isfinite(rec.min_corr_eigenvalue_raw)
    assert np.isfinite(rec.min_corr_eigenvalue_shrunk)
    assert rec.min_corr_eigenvalue_shrunk >= 1e-4 - 1e-12
    assert isinstance(rec.correlation_shrunk_applied, bool)
    assert np.isfinite(rec.mahalanobis_d2) and rec.mahalanobis_d2 >= 0.0
    assert np.isfinite(rec.max_psi_score) and rec.max_psi_score >= 0.0
    assert np.isfinite(rec.min_pin_risk_z) and rec.min_pin_risk_z >= 0.0
    assert rec.engine_selected in (
        "Tier1_Deterministic",
        "Tier2_MaskedEGNO",
        "Tier3_DupireQMC",
    )


def test_G3_1_cartesian_stress_matrix_500_states(
    reference_state: AutocallPreprocState,
    zero_head_surrogate: MaskedEQAutocallSurrogate,
) -> None:
    """Gate G3.1: Execute 500-state Cartesian Stress Matrix with 0 NaN/Inf and 100% finite outputs."""
    guardian = EQAutocallModelGuardian(record_drift_by_default=False)

    # 1. Core Cartesian product grid: 5 moneyness x 5 vols x 5 correlations x 4 dividend ratios = 500 states!
    moneyness_grid = [0.05, 0.25, 0.85, 1.25, 3.00]
    vol_grid = [0.02, 0.15, 0.45, 1.20, 2.50]
    rho_specs = ["rho_-0.95", "rho_-0.25", "rho_0.50", "rho_0.999", "broken_non_psd"]
    div_grid = [0.0, 0.15, 0.45, 0.85]
    exotic_indices = [
        "WorstOf",
        "BestOf",
        "Average",
        "Rainbow",
        "Outperformance",
        "Lookback",
        "TwinWin",
        "Spread",
    ]

    broken_3x3 = np.array(
        [
            [1.0, 0.95, -0.90],
            [0.95, 1.0, 0.95],
            [-0.90, 0.95, 1.0],
        ],
        dtype=np.float64,
    )
    assert float(np.min(np.linalg.eigvalsh(broken_3x3))) < -0.50

    cartesian_combos = list(
        itertools.product(moneyness_grid, vol_grid, rho_specs, div_grid)
    )
    assert len(cartesian_combos) == 500

    # Pre-build base states per (vol, div_ratio) to accelerate the 500-state Cartesian sweep
    base_states: dict[tuple[float, float], AutocallPreprocState] = {}
    for v_val in vol_grid:
        for d_val in div_grid:
            req_d, _ = _build_synthetic_autocall_dict(
                d_assets=3,
                m_periods=2,
                moneyness=1.0,
                vol_level=v_val,
                rho=0.50,
                div_ratio=d_val,
                tau_first=0.20,
                ko_barrier=1.00,
                cpn_barrier=0.75,
            )
            base_states[(v_val, d_val)] = preprocess_eq_autocall(req_d)

    total_evaluated = 0
    fallback_count = 0
    tier2_count = 0

    for idx, (m_val, v_val, r_spec, d_val) in enumerate(cartesian_combos):
        base_st = base_states[(v_val, d_val)]
        st = copy.copy(base_st)
        st.trade_info = f"STRESS_STATE_{idx:03d}"

        # Apply moneyness scaling
        m_vec = torch.tensor(
            [m_val, min(m_val * 1.05, 3.0), max(m_val * 0.95, 0.05)],
            dtype=torch.float64,
        )
        st.moneyness_t0 = m_vec
        st.spots_t0 = m_vec * st.ref_spots
        st.forward_perf = m_vec.unsqueeze(1) * st.forward_rel

        # Update node_feats LOG_MONEYNESS
        node_c = st.node_feats.clone()
        node_c[0, :3, 0] = torch.log(m_vec.clamp(min=1e-12))
        st.node_feats = node_c

        # Apply correlation specification (including extreme -0.95, 0.999, and broken non-PSD)
        if r_spec == "broken_non_psd":
            corr_np = broken_3x3.copy()
        else:
            rho_num = float(r_spec.split("_")[1])
            corr_np = np.full((3, 3), rho_num, dtype=np.float64)
            np.fill_diagonal(corr_np, 1.0)
        st.corr_matrix = torch.as_tensor(corr_np, dtype=torch.float64)

        # Inject imminent pin-risk on every 10th state (tau_1 = 0.01, B_KO = F_W)
        if idx % 10 == 0:
            st.tau_obs = torch.tensor([0.01, 0.26], dtype=torch.float64)
            st.is_callable = torch.tensor([1.0, 1.0], dtype=torch.float64)
            fw_est = float(torch.min(st.forward_perf[:, 0]).item())
            st.ko_barriers = torch.tensor([fw_est, 1.00], dtype=torch.float64)
        else:
            st.tau_obs = base_st.tau_obs
            st.is_callable = base_st.is_callable
            st.ko_barriers = base_st.ko_barriers

        # Inject exotic monitoring indices across the sweep
        st.callability_index = exotic_indices[idx % len(exotic_indices)]
        st.coupon_index = exotic_indices[(idx + 1) % len(exotic_indices)] if idx % 7 == 0 else None
        st.option_index = exotic_indices[(idx + 2) % len(exotic_indices)] if idx % 11 == 0 else None

        res = guardian.price_with_guardian(
            st,
            surrogate=zero_head_surrogate,
            n_fallback_paths=128,
            qmc_mode="dupire_lv" if idx % 5 == 0 else "terminal_iv",
            n_substeps_per_period=2,
        )
        _assert_valid_audit_and_result(res)
        total_evaluated += 1
        if res.fallback_triggered:
            fallback_count += 1
        else:
            tier2_count += 1

    assert total_evaluated == 500
    assert len(guardian.audit_log) == 500
    assert fallback_count > 0
    assert tier2_count > 0


def test_G3_2_real_payload_quirks_missing_repo_duplicate_strikes_sentinels(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    zero_head_surrogate: MaskedEQAutocallSurrogate,
) -> None:
    """Gate G3.2: Verify graceful handling of missing RepoSpreadCurves, duplicate strikes, and sentinels."""
    req_model, expected_res = parsed_dag
    ref_pv = float(expected_res["ProductValue"])
    ref_se = float(expected_res["StandardError"])

    # 1. Verify raw payload quirks in `eq_autocallable_request.txt`
    assert "POSIRX_RUB" in req_model.Product.UnderlyingIdentifiers
    assert "POSIRX_RUB" not in req_model.Model.MarketDataSet.RepoSpreadCurves

    vkco_raw_strikes = req_model.Model.MarketDataSet.VolatilitySurfaces["VKCO_RUB"].Strikes
    assert len(vkco_raw_strikes) == 31
    assert vkco_raw_strikes[-2:] == [845.0, 870.0]

    # Also verify duplicate strike deduplication & zero calendar variance projection on VKCO_RUB
    req_dup = copy.deepcopy(req_model)
    vkco_spec_dup = req_dup.Model.MarketDataSet.VolatilitySurfaces["VKCO_RUB"]
    vkco_spec_dup.Strikes.append(870.0)
    for row in vkco_spec_dup.VolatilityQuotes:
        row.append(row[-1])
    # Set last slice implied vols so calendar total variance delta is 0.0 (w_last == w_prev)
    vkco_spec_dup.VolatilityQuotes[-1] = [0.05 for _ in vkco_spec_dup.Strikes]
    state_dup = preprocess_eq_autocall(req_dup)
    assert state_dup.vol_surfaces["VKCO_RUB"].was_projected is True
    assert state_dup.vol_surfaces["VKCO_RUB"].n_unique_strikes == 31
    assert state_dup.vol_surfaces["VKCO_RUB"].n_raw_strikes == 32
    w_vkco_dup = (
        state_dup.vol_surfaces["VKCO_RUB"].quotes ** 2
    ) * state_dup.vol_surfaces["VKCO_RUB"].tenors[:, None]
    assert np.all(np.diff(w_vkco_dup, axis=0) > 0.0)

    ko_raw = req_model.Product.AutoCallTriggerLevels
    cpn_raw = req_model.Product.CouponBarrierLevels
    assert ko_raw[0] == 500.0 and ko_raw[-1] == 500.0
    assert cpn_raw is not None and cpn_raw[0] == pytest.approx(1e-5, abs=1e-10)

    # 2. Run preprocess_eq_autocall and verify deterministic quirk resolution
    state = preprocess_eq_autocall(req_model)
    posirx_idx = state.sorted_identifiers.index("POSIRX_RUB")
    assert float(state.dividend_schedule.repo_spread_avg[posirx_idx].item()) == 0.0

    vkco_surf = state.vol_surfaces["VKCO_RUB"]
    np.testing.assert_allclose(vkco_surf.quotes[:, -2], vkco_surf.quotes[:, -1], rtol=1e-12, atol=1e-12)
    w_vkco = (vkco_surf.quotes**2) * vkco_surf.tenors[:, None]
    assert np.all(np.diff(w_vkco, axis=0) > 0.0)

    # Period 12 has B_KO = 500.0 -> is_callable == 0.0 and log_ko_barrier_feat == ln(3.0)
    assert float(state.is_callable[0].item()) == 0.0
    assert float(state.sched_feats[0, 0, 3].item()) == pytest.approx(np.log(3.0), abs=1e-12)

    # 3. Verify EQAutocallModelGuardian passes `eq_autocallable_request.txt` on Pre-Inference Gate
    guardian = EQAutocallModelGuardian()
    pre_passed, pre_reasons, r_psd, diag = guardian.evaluate_pre_inference(state)
    assert pre_passed is True, f"Reference trade failed pre-gate with reasons: {pre_reasons}"
    assert len(pre_reasons) == 0
    assert diag["mahalanobis_d2"] <= CHI2_16_0999
    assert diag["min_corr_eigenvalue_raw"] >= 1e-4
    assert diag["correlation_shrunk_applied"] is False
    np.testing.assert_allclose(r_psd, state.corr_matrix.numpy(), rtol=1e-14, atol=1e-14)

    # 4. Guarded Tier 2 pricing with zero-head surrogate (matches Stulz-Margrabe proxy and within 1 SE of ref_pv)
    res_t2 = guardian.price_with_guardian(req_model, surrogate=zero_head_surrogate)
    _assert_valid_audit_and_result(res_t2)
    assert res_t2.engine_selected == "Tier2_MaskedEGNO"
    assert res_t2.fallback_triggered is False
    assert len(res_t2.cashflows) == 37
    assert abs(res_t2.product_value - float(state.baseline.product_value_proxy.item())) <= 1e-5
    assert abs(res_t2.product_value - ref_pv) <= ref_se

    # 5. Guarded Tier 3 Dupire QMC fallback pricing on `eq_autocallable_request.txt`
    res_qmc = guardian.price_with_guardian(
        req_model,
        surrogate=zero_head_surrogate,
        force_qmc=True,
        n_fallback_paths=32768,
    )
    _assert_valid_audit_and_result(res_qmc)
    assert res_qmc.engine_selected == "Tier3_DupireQMC"
    assert res_qmc.fallback_triggered is True
    assert AutocallFallbackReason.ENGINE_REQUESTED_QMC in res_qmc.fallback_reasons
    assert len(res_qmc.cashflows) == 37
    assert abs(res_qmc.product_value - ref_pv) <= 1.0 * ref_se


def test_G3_4_guardian_tripwires_and_audit_log(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    zero_head_surrogate: MaskedEQAutocallSurrogate,
) -> None:
    """Gate G3.4: Test every single Pre-Inference & Post-Inference tripwire and audit log record."""
    req_model, _ = parsed_dag
    guardian = EQAutocallModelGuardian()

    # ── 1. BASKET_CARDINALITY_EXCEEDED (D = 9 > D_max = 8) ──────────────────
    req_d9, _ = _build_synthetic_autocall_dict(d_assets=9, m_periods=3)
    res_d9 = guardian.price_with_guardian(req_d9, surrogate=zero_head_surrogate, n_fallback_paths=256)
    _assert_valid_audit_and_result(res_d9)
    assert res_d9.fallback_triggered is True
    assert AutocallFallbackReason.BASKET_CARDINALITY_EXCEEDED in res_d9.fallback_reasons
    assert res_d9.engine_selected == "Tier3_DupireQMC"

    # ── 2. SCHEDULE_HORIZON_EXCEEDED (M_rem = 50 > M_max = 48) ──────────────
    req_m50, _ = _build_synthetic_autocall_dict(d_assets=2, m_periods=50, tau_first=0.05)
    res_m50 = guardian.price_with_guardian(
        req_m50,
        surrogate=zero_head_surrogate,
        n_fallback_paths=128,
        qmc_mode="terminal_iv",
    )
    _assert_valid_audit_and_result(res_m50)
    assert res_m50.fallback_triggered is True
    assert AutocallFallbackReason.SCHEDULE_HORIZON_EXCEEDED in res_m50.fallback_reasons
    assert res_m50.engine_selected == "Tier3_DupireQMC"

    # ── 3. ZERO_FUTURE_PERIODS (M_rem == 0, deterministic Tier 1 in 0 ms) ───
    state_auto = preprocess_eq_autocall(req_model)
    state_auto.m_rem = 0
    state_auto.already_autocalled = True
    res_m0 = guardian.price_with_guardian(state_auto, surrogate=zero_head_surrogate)
    assert res_m0.fallback_triggered is True
    assert AutocallFallbackReason.ZERO_FUTURE_PERIODS in res_m0.fallback_reasons
    assert res_m0.engine_selected == "Tier1_Deterministic"
    assert res_m0.standard_error == 0.0
    assert np.isfinite(res_m0.product_value)

    # ── 4. EXOTIC_MONITORING_INDEX (Rainbow, Outperformance, Lookback, TwinWin, Spread) ──
    for exotic_idx in ("Rainbow", "Outperformance", "Lookback", "TwinWin", "Spread"):
        st_exotic = preprocess_eq_autocall(req_model)
        st_exotic.callability_index = exotic_idx
        res_ex = guardian.price_with_guardian(
            st_exotic,
            surrogate=zero_head_surrogate,
            n_fallback_paths=256,
        )
        _assert_valid_audit_and_result(res_ex)
        assert res_ex.fallback_triggered is True
        assert AutocallFallbackReason.EXOTIC_MONITORING_INDEX in res_ex.fallback_reasons
        assert res_ex.engine_selected == "Tier3_DupireQMC"

    # ── 5. CORRELATION_NOT_PSD & Ledoit-Wolf PSD Shrinkage ──────────────────
    # 5a. Mild near-singular correlation (lambda_min in [-1e-2, 1e-4)): shrinks with warning, no fallback
    r_mild = np.array(
        [
            [1.0, 0.99998, 0.99998],
            [0.99998, 1.0, 0.99998],
            [0.99998, 0.99998, 1.0],
        ],
        dtype=np.float64,
    )
    r_psd_mild, eig_raw_mild, eig_shrunk_mild, shrunk_mild = project_correlation_to_psd(r_mild)
    assert eig_raw_mild < 1e-4
    assert eig_shrunk_mild >= 1e-4 - 1e-12
    assert shrunk_mild is True
    np.testing.assert_allclose(np.diag(r_psd_mild), np.ones(3), atol=1e-14)
    # Verify Cholesky succeeds
    np.linalg.cholesky(r_psd_mild)

    # 5b. Severe non-PSD correlation (lambda_min < -1e-2): flags CORRELATION_NOT_PSD + shrinks to PSD
    r_severe = np.array(
        [
            [1.0, 0.95, -0.90],
            [0.95, 1.0, 0.95],
            [-0.90, 0.95, 1.0],
        ],
        dtype=np.float64,
    )
    req_psd, corr_t_severe = _build_synthetic_autocall_dict(
        d_assets=3,
        m_periods=2,
        corr_matrix_override=r_severe,
    )
    st_severe = preprocess_eq_autocall(req_psd, corr_override=corr_t_severe)
    res_severe = guardian.price_with_guardian(
        st_severe,
        surrogate=zero_head_surrogate,
        n_fallback_paths=256,
    )
    _assert_valid_audit_and_result(res_severe)
    assert res_severe.fallback_triggered is True
    assert AutocallFallbackReason.CORRELATION_NOT_PSD in res_severe.fallback_reasons
    assert res_severe.audit_record.min_corr_eigenvalue_raw < -0.50
    assert res_severe.audit_record.min_corr_eigenvalue_shrunk >= 1e-4 - 1e-12
    assert res_severe.audit_record.correlation_shrunk_applied is True
    np.linalg.cholesky(res_severe.shrunk_corr_matrix)

    # ── 6. IMMINENT_BARRIER_PIN_RISK (tau_1 = 0.01 <= 0.08, |F_W - B_KO|/v_W < 0.15) ──
    req_pin, _ = _build_synthetic_autocall_dict(
        d_assets=3,
        m_periods=3,
        moneyness=1.0,
        vol_level=0.30,
        tau_first=0.01,
        ko_barrier=0.95,
    )
    st_pin = preprocess_eq_autocall(req_pin)
    # Set B_KO(t_1) equal to Stulz forward F_{W,1} so pin-risk z-score is ~0.0 < 0.15
    fw_0 = float(st_pin.baseline.f_w[0].item())
    st_pin.ko_barriers = st_pin.ko_barriers.clone()
    st_pin.ko_barriers[0] = fw_0
    st_pin.is_callable = torch.ones_like(st_pin.is_callable)

    res_pin = guardian.price_with_guardian(
        st_pin,
        surrogate=zero_head_surrogate,
        n_fallback_paths=256,
    )
    _assert_valid_audit_and_result(res_pin)
    assert res_pin.fallback_triggered is True
    assert AutocallFallbackReason.IMMINENT_BARRIER_PIN_RISK in res_pin.fallback_reasons
    assert res_pin.audit_record.min_pin_risk_z < 0.15
    assert res_pin.engine_selected == "Tier3_DupireQMC"

    # ── 7. MAHALANOBIS_OOD (extreme moneyness M_{0,i} = 10.0 and extreme vol sigma = 5.0) ──
    req_ood_spot, _ = _build_synthetic_autocall_dict(
        d_assets=3,
        m_periods=2,
        moneyness=10.0,
        vol_level=0.30,
    )
    res_ood_spot = guardian.price_with_guardian(
        req_ood_spot,
        surrogate=zero_head_surrogate,
        n_fallback_paths=256,
    )
    _assert_valid_audit_and_result(res_ood_spot)
    assert res_ood_spot.fallback_triggered is True
    assert AutocallFallbackReason.MAHALANOBIS_OOD in res_ood_spot.fallback_reasons
    assert res_ood_spot.audit_record.mahalanobis_d2 > CHI2_16_0999

    req_ood_vol, _ = _build_synthetic_autocall_dict(
        d_assets=3,
        m_periods=2,
        moneyness=1.0,
        vol_level=4.50,
    )
    res_ood_vol = guardian.price_with_guardian(
        req_ood_vol,
        surrogate=zero_head_surrogate,
        n_fallback_paths=256,
    )
    _assert_valid_audit_and_result(res_ood_vol)
    assert res_ood_vol.fallback_triggered is True
    assert AutocallFallbackReason.MAHALANOBIS_OOD in res_ood_vol.fallback_reasons
    assert res_ood_vol.audit_record.mahalanobis_d2 > CHI2_16_0999

    # ── 8. PSI_DRIFT_BREACH (10-decile PSI >= 0.25) ─────────────────────────
    guardian_psi = EQAutocallModelGuardian(psi_min_samples=20, record_drift_by_default=True)
    st_ref = preprocess_eq_autocall(req_model)
    feat_ref = extract_16d_basket_summary(st_ref)
    assert feat_ref.shape == (16,)
    assert len(FEATURE_NAMES_16D) == 16

    # Populate rolling history with 25 shifted feature vectors in the top decile bin
    shifted_feat = feat_ref.copy()
    shifted_feat[0] = 1.50  # Shifted mean_log_moneyness into top decile
    for _ in range(25):
        guardian_psi.record_feature_vector(shifted_feat)

    max_psi, psi_map = guardian_psi.compute_psi()
    assert max_psi >= 0.25
    assert psi_map["mean_log_moneyness"] >= 0.25

    res_psi = guardian_psi.price_with_guardian(
        st_ref,
        surrogate=zero_head_surrogate,
        n_fallback_paths=256,
    )
    _assert_valid_audit_and_result(res_psi)
    assert res_psi.fallback_triggered is True
    assert AutocallFallbackReason.PSI_DRIFT_BREACH in res_psi.fallback_reasons
    assert res_psi.audit_record.max_psi_score >= 0.25

    # ── 9. POST_INFERENCE_SIMPLEX_VIOLATION ─────────────────────────────────
    st_post = preprocess_eq_autocall(req_model)
    bad_simplex_out = AutocallSurrogateOutput(
        stopping_probs=torch.tensor([[1.25]], dtype=torch.float64),
        survival_probs=torch.tensor([[0.0]], dtype=torch.float64),
        hazard_rates=torch.tensor([[1.25]], dtype=torch.float64),
        coupon_probs=torch.tensor([[0.5]], dtype=torch.float64),
        put_loss=torch.tensor([0.10], dtype=torch.float64),
        delta_hazard_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_coupon_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_put_loss=torch.zeros(1, dtype=torch.float64),
    )
    post_ok, post_reasons = guardian.evaluate_post_inference(
        bad_simplex_out,
        None,
        st_post,
        st_post.baseline,
    )
    assert post_ok is False
    assert AutocallFallbackReason.POST_INFERENCE_SIMPLEX_VIOLATION in post_reasons

    # ── 10. POST_INFERENCE_PUT_BOUND_VIOLATION ──────────────────────────────
    bad_put_out = AutocallSurrogateOutput(
        stopping_probs=torch.tensor([[0.0]], dtype=torch.float64),
        survival_probs=torch.tensor([[1.0]], dtype=torch.float64),
        hazard_rates=torch.tensor([[0.0]], dtype=torch.float64),
        coupon_probs=torch.tensor([[0.0]], dtype=torch.float64),
        put_loss=torch.tensor([1.50], dtype=torch.float64),  # > K_put * S_1 = 0.90
        delta_hazard_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_coupon_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_put_loss=torch.zeros(1, dtype=torch.float64),
    )
    post_ok_put, post_reasons_put = guardian.evaluate_post_inference(
        bad_put_out,
        None,
        st_post,
        st_post.baseline,
    )
    assert post_ok_put is False
    assert AutocallFallbackReason.POST_INFERENCE_PUT_BOUND_VIOLATION in post_reasons_put

    # ── 11. POST_INFERENCE_RESIDUAL_OUTLIER (|PV_surrogate - PV_proxy| > 0.15) ──
    outlier_surrogate = copy.deepcopy(zero_head_surrogate)
    # Bias the put_head so put_loss shifts to 0.0 (reducing option loss by ~0.74 > 0.15)
    with torch.no_grad():
        last_put = outlier_surrogate.put_head[-1]
        assert isinstance(last_put, torch.nn.Linear)
        last_put.bias.fill_(-100.0)

    res_outlier = guardian.price_with_guardian(
        st_post,
        surrogate=outlier_surrogate,
        n_fallback_paths=512,
    )
    _assert_valid_audit_and_result(res_outlier)
    assert res_outlier.fallback_triggered is True
    assert AutocallFallbackReason.POST_INFERENCE_RESIDUAL_OUTLIER in res_outlier.fallback_reasons
    assert res_outlier.audit_record.pre_gate_passed is True
    assert res_outlier.audit_record.post_gate_passed is False
    assert res_outlier.engine_selected == "Tier3_DupireQMC"

    # ── 12. NON_FINITE_OUTPUT (NaN in surrogate output) ─────────────────────
    nan_out = AutocallSurrogateOutput(
        stopping_probs=torch.tensor([[float("nan")]], dtype=torch.float64),
        survival_probs=torch.tensor([[1.0]], dtype=torch.float64),
        hazard_rates=torch.tensor([[0.0]], dtype=torch.float64),
        coupon_probs=torch.tensor([[0.0]], dtype=torch.float64),
        put_loss=torch.tensor([0.5], dtype=torch.float64),
        delta_hazard_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_coupon_logits=torch.zeros((1, 1), dtype=torch.float64),
        delta_put_loss=torch.zeros(1, dtype=torch.float64),
    )
    post_ok_nan, post_reasons_nan = guardian.evaluate_post_inference(
        nan_out,
        None,
        st_post,
        st_post.baseline,
    )
    assert post_ok_nan is False
    assert AutocallFallbackReason.NON_FINITE_OUTPUT in post_reasons_nan

    # Verify audit log has recorded all guarded calls and fit_reference_distribution / to_dict work
    rng = np.random.default_rng(123)
    ref_samples = guardian.ref_mean_16d[None, :] + 0.1 * rng.standard_normal((100, 16))
    guardian.fit_reference_distribution(ref_samples)
    resp_dict = res_outlier.to_dict()
    assert resp_dict["EngineSelected"] == "Tier3_DupireQMC"
    assert resp_dict["FallbackTriggered"] is True
    assert len(guardian.audit_log) >= 10
    for entry in guardian.audit_log:
        d_rec = entry.to_dict()
        assert set(d_rec.keys()) == {
            "trade_id",
            "timestamp_utc",
            "pre_gate_passed",
            "post_gate_passed",
            "fallback_triggered",
            "fallback_reasons",
            "warnings",
            "min_corr_eigenvalue_raw",
            "min_corr_eigenvalue_shrunk",
            "correlation_shrunk_applied",
            "mahalanobis_d2",
            "max_psi_score",
            "min_pin_risk_z",
            "engine_selected",
            "conformal_half_width_bps",
            "greeks_engine_used",
        }


def test_guardian_v2_calibrated_5d_conformal_exotic_routing_and_batch_audit(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    zero_head_surrogate: MaskedEQAutocallSurrogate,
) -> None:
    """Covers v2.0 5D conformal calibration, tau_desk_bps=60.0, 45476647 acceptance, exotic modifiers, smile OOD, and batch audit."""
    from deepvol.mrm.eq_autocall_guardian import (
        DEFAULT_GUARDIAN_STATS_PATH,
        DEFAULT_TAU_DESK_BPS,
    )
    from deepvol.training.train_eq_autocall import (
        DEFAULT_CHECKPOINT_PATH,
        load_eq_autocall_surrogate,
    )

    assert DEFAULT_TAU_DESK_BPS == pytest.approx(15.0)
    assert DEFAULT_CHECKPOINT_PATH.is_file() and DEFAULT_GUARDIAN_STATS_PATH.is_file()

    req_model, _ = parsed_dag
    surr, _ = load_eq_autocall_surrogate(DEFAULT_CHECKPOINT_PATH, device="cpu")
    guardian = EQAutocallModelGuardian(
        stats_path=DEFAULT_GUARDIAN_STATS_PATH,
        tau_desk_bps=DEFAULT_TAU_DESK_BPS,
        record_drift_by_default=False,
    )
    assert guardian.tau_desk_bps == pytest.approx(15.0)

    # 1. Production reference 45476647/EMTN_CIB_SEC enforces tau_desk_bps=15.0 cleanly
    res_ref = guardian.price_with_guardian(req_model, surrogate=surr)
    if guardian.conformal_mode == "signed_eb_mondrian_8d":
        assert res_ref.fallback_triggered is False
        assert res_ref.engine_selected == "Tier2_MaskedEGNO"
        assert res_ref.audit_record.conformal_half_width_bps <= 15.0
    else:
        assert math.isfinite(res_ref.product_value)
        assert len(res_ref.cashflows) > 0
        assert res_ref.audit_record.conformal_half_width_bps <= 15.0 or res_ref.fallback_triggered is True
    assert res_ref.ProductValue == pytest.approx(res_ref.product_value)
    assert res_ref.StandardError == pytest.approx(res_ref.standard_error)
    assert res_ref.Option == pytest.approx(res_ref.option_pv)
    assert res_ref.Notional == pytest.approx(res_ref.notional_pv)
    assert res_ref.Coupon == pytest.approx(res_ref.coupon_pv)
    assert res_ref.CashFlows is res_ref.cashflows
    assert res_ref.cash_flows is res_ref.cashflows

    # 2. Legacy conformal stats without scale_weights_5d falls back cleanly
    guardian_legacy = EQAutocallModelGuardian(record_drift_by_default=False)
    stats_no_5d = copy.deepcopy(dict(torch.load(DEFAULT_GUARDIAN_STATS_PATH, map_location="cpu", weights_only=True)))
    stats_no_5d.pop("scale_weights_5d", None)
    assert guardian_legacy.compute_conformal_half_width_bps(preprocess_eq_autocall(req_model)) == 0.0
    guardian_legacy.q_err_conformal = float(stats_no_5d["q_err_conformal"])
    guardian_legacy.q_ood_conformal = float(stats_no_5d["q_ood_conformal"])
    guardian_legacy._conformal_calibrated = True
    assert guardian_legacy.compute_conformal_half_width_bps(preprocess_eq_autocall(req_model)) >= 0.0

    # 3. Exotic terminal modifiers -> EXOTIC_TERMINAL_MODIFIER_ROUTED
    for mod_field, mod_val in (
        ("AllowAutocallOnExpiry", True),
        ("LowerStrike", 0.50),
        ("OneStarStrike", 1.15),
        ("ParticipationCallRate", 0.25),
        ("CouponBarrierType", "Out"),
        ("IsLongBasket", True),
    ):
        req_m, _ = _build_synthetic_autocall_dict(d_assets=2, m_periods=3)
        req_m["Product"][mod_field] = mod_val
        res_m = guardian.price_with_guardian(req_m, surrogate=zero_head_surrogate, n_fallback_paths=128)
        assert res_m.fallback_triggered is True
        assert AutocallFallbackReason.EXOTIC_TERMINAL_MODIFIER_ROUTED in res_m.fallback_reasons

    # 4. Smile box OOD branches (SSVI skew/curv/PC1, repo spread) & pin-risk on LowerStrike / KI
    st_smile = preprocess_eq_autocall(req_model)
    st_smile.node_feats = st_smile.node_feats.clone()
    st_smile.node_feats[0, :, 10] = 0.90  # SSVI_SKEW_RHO > 0.75 across active assets
    pre_ok_s, pre_reasons_s, _, _ = guardian.evaluate_pre_inference(st_smile)
    assert pre_ok_s is False and AutocallFallbackReason.MAHALANOBIS_OOD in pre_reasons_s

    req_low_pin, _ = _build_synthetic_autocall_dict(d_assets=2, m_periods=2, tau_first=0.01)
    req_low_pin["Product"]["LowerStrike"] = 0.99
    req_low_pin["Product"]["OptionDownAndInBarrierLevel"] = 0.99
    st_low_pin = preprocess_eq_autocall(req_low_pin)
    st_low_pin.tau_obs = torch.tensor([0.01, 0.02], dtype=torch.float64)
    z_pin, breached = guardian.compute_min_pin_risk_z(st_low_pin)
    assert z_pin >= 0.0 and isinstance(breached, bool)

    # 5. Dict-typed evaluate_post_inference and vector batch pre/post/audit gates
    st_ref = preprocess_eq_autocall(req_model)
    ok_dict, reasons_dict = guardian.evaluate_post_inference(
        {"p_stop": np.zeros(st_ref.m_rem), "q_cpn": np.zeros(st_ref.m_rem), "l_put": 0.0},
        {"ProductValue": float(st_ref.baseline.product_value_proxy.item())},
        st_ref,
    )
    assert ok_dict is True and reasons_dict == []

    t2 = st_ref.to_tier2_float32_tensors()
    pre_batch_diag = guardian.evaluate_tensor_batch_pre(
        t2["node_feats"], t2["edge_feats"], t2["asset_mask"], t2["sched_feats"], t2["sched_mask"], t2["global_feats"]
    )
    with torch.no_grad():
        s_out = surr(
            t2["node_feats"], t2["edge_feats"], t2["asset_mask"], t2["sched_feats"], t2["sched_mask"], t2["global_feats"]
        )
    post_batch_diag = guardian.evaluate_tensor_batch_post(
        s_out.stopping_probs,
        s_out.coupon_probs,
        s_out.put_loss,
        torch.tensor([float(st_ref.baseline.product_value_proxy.item())], dtype=torch.float64),
        t2["sched_mask"],
        t2["global_feats"],
    )
    batch_rec = guardian.record_batch_audit(
        trade_id="BATCH_GUARD_TEST",
        pre_diag=pre_batch_diag,
        post_diag=post_batch_diag,
        engine_selected="Tier2_MaskedEGNO",
        greeks_engine_used="Tier2_BatchedAD",
    )
    assert batch_rec.trade_id == "BATCH_GUARD_TEST"

    bad_post = guardian.evaluate_tensor_batch_post(
        torch.full((1, st_ref.sched_mask.shape[1]), 0.9, dtype=torch.float64),
        torch.zeros_like(t2["sched_mask"], dtype=torch.float64),
        torch.tensor([5.0], dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float64),
        t2["sched_mask"],
        t2["global_feats"],
    )
    assert bad_post["all_passed"] is False and bad_post["non_finite_count"] == 1
    bad_rec = guardian.record_batch_audit(
        trade_id="BATCH_BAD_TEST",
        pre_diag={**pre_batch_diag, "ood_count": 1, "conformal_breach_count": 1, "psi_breached": True},
        post_diag=bad_post,
        engine_selected="Tier2_MaskedEGNO",
        greeks_engine_used="None",
    )
    assert bad_rec.fallback_triggered is True and len(bad_rec.warnings) >= 2

    # Additional guardian coverage: reset_drift_history, residual_threshold, conformal breach, non-finite inputs, stopping_probs dict
    guardian.reset_drift_history()
    guardian_res_th = EQAutocallModelGuardian(residual_threshold=0.05, record_drift_by_default=False)
    ok_th, _ = guardian_res_th.evaluate_post_inference(
        {"stopping_probs": np.zeros(st_ref.m_rem), "coupon_probs": np.zeros(st_ref.m_rem), "put_loss": 0.0},
        {"product_value": float(st_ref.baseline.product_value_proxy.item())},
        st_ref,
    )
    assert ok_th is True
    g_conf = EQAutocallModelGuardian(
        stats_path=DEFAULT_GUARDIAN_STATS_PATH,
        tau_desk_bps=0.01,
        record_drift_by_default=False,
    )
    ok_cf, reasons_cf, _, _ = g_conf.evaluate_pre_inference(st_ref)
    assert ok_cf is False and AutocallFallbackReason.CONFORMAL_UNCERTAINTY_BREACH in reasons_cf

    st_nan_in = copy.copy(st_ref)
    st_nan_in.spots_t0 = torch.full_like(st_ref.spots_t0, float("nan"))
    ok_ni, reasons_ni, _, _ = guardian.evaluate_pre_inference(st_nan_in)
    assert ok_ni is False and AutocallFallbackReason.NON_FINITE_OUTPUT in reasons_ni

    # Edge-case PSD shapes & domain_box_ood triggers (zero rate, quanto drift, repo spread) + funded post gate
    r0, _, _, _ = project_correlation_to_psd(np.zeros((0, 0), dtype=np.float64))
    assert r0.shape == (0, 0)
    r1, _, _, s1 = project_correlation_to_psd(np.array([[1.2]], dtype=np.float64))
    assert r1.shape == (1, 1) and s1 is True
    r_nf, _, _, s_nf = project_correlation_to_psd(np.array([[1.0, float("nan")], [0.2, 1.0]], dtype=np.float64))
    assert np.all(np.isfinite(r_nf)) and s_nf is True
    with pytest.raises(ValueError, match="square"):
        project_correlation_to_psd(np.ones((2, 3), dtype=np.float64))

    st_zr = copy.copy(st_ref)
    st_zr.df_pay = torch.full_like(st_ref.df_pay, 0.20)
    ok_zr, r_zr, _, _ = guardian.evaluate_pre_inference(st_zr)
    assert ok_zr is False and AutocallFallbackReason.MAHALANOBIS_OOD in r_zr

    st_qnt = copy.copy(st_ref)
    st_qnt.quanto_meta = {st_ref.sorted_identifiers[0]: {"rho_s_fx": 0.9, "sig_fx_obs": [0.8]}}
    ok_qnt, r_qnt, _, _ = guardian.evaluate_pre_inference(st_qnt)
    assert ok_qnt is False and AutocallFallbackReason.MAHALANOBIS_OOD in r_qnt

    st_repo = copy.copy(st_ref)
    from deepvol.models.autocall.feature_layout import AutocallNodeCol

    st_repo.node_feats = st_ref.node_feats.clone()
    st_repo.node_feats[0, 0, int(AutocallNodeCol.REPO_SPREAD_AVG)] = 0.25
    ok_rp, r_rp, _, _ = guardian.evaluate_pre_inference(st_repo)
    assert ok_rp is False and AutocallFallbackReason.MAHALANOBIS_OOD in r_rp

    req_f = copy.deepcopy(req_model)
    req_f.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_f.Product.FundingSpread = 0.015
    req_f.Product.FundingLegPaymentDates = list(req_f.Product.PaymentDates)
    st_f = preprocess_eq_autocall(req_f)
    ok_f, _ = guardian.evaluate_post_inference(None, None, st_f)
    assert isinstance(ok_f, bool)

    # Wide vol smile (ratio > 5.0 -> OOD), dict postproc_out format, pin risk (non-guar cpn + KI), and batch helpers
    from types import SimpleNamespace

    st_wide = copy.copy(st_ref)
    wide_map = dict(st_ref.vol_surfaces)
    for idx_i, tid_i in enumerate(st_ref.sorted_identifiers):
        f_ref_i = float(st_ref.dividend_schedule.forward_price[idx_i, -1])
        wide_map[tid_i] = SimpleNamespace(
            strikes=np.array([0.5 * f_ref_i, 1.0 * f_ref_i, 1.5 * f_ref_i], dtype=np.float64),
            quotes=np.array([[0.05, 0.20, 0.35]], dtype=np.float64),
        )
    st_wide.vol_surfaces = wide_map
    ok_w, r_w, _, _ = guardian.evaluate_pre_inference(st_wide)
    assert ok_w is False and AutocallFallbackReason.MAHALANOBIS_OOD in r_w

    st_pin = copy.copy(st_ref)
    st_pin.sched_feats = st_ref.sched_feats.clone()
    st_pin.tau_obs = torch.full_like(st_ref.tau_obs, 0.01)
    st_pin.sched_feats[0, 0, 1] = 0.01  # t_rem <= 14/365
    st_pin.sched_feats[0, 0, 5] = 0.0   # non-callable
    st_pin.sched_feats[0, 0, 7] = 0.0   # non-guaranteed coupon barrier
    st_pin.sched_feats[0, 0, 4] = 0.0   # log_cpn_barrier ~= 0
    z_pin, _ = guardian.compute_min_pin_risk_z(st_pin)
    assert z_pin >= 0.0

    m_s = st_ref.sched_feats.shape[1]
    ok_dp, r_dp = guardian.evaluate_post_inference(
        {
            "stopping_probs": torch.full((1, m_s), 1.0 / max(m_s, 1)),
            "coupon_probs": torch.full((1, m_s), 0.2),
            "put_loss": torch.tensor([0.05]),
        },
        None,
        st_ref,
    )
    assert isinstance(ok_dp, bool) and isinstance(r_dp, list)

    sum16 = guardian._extract_batch_16d_summary(
        st_ref.node_feats,
        st_ref.edge_feats,
        st_ref.asset_mask,
        st_ref.sched_feats,
        st_ref.sched_mask,
        st_ref.global_feats,
    )
    assert sum16.shape == (1, 16)
    g_nocal = EQAutocallModelGuardian(d_max=8, m_max=48, stats_path=None)
    res_nc = g_nocal.evaluate_tensor_batch_pre(
        st_ref.node_feats,
        st_ref.edge_feats,
        st_ref.asset_mask,
        st_ref.sched_feats,
        st_ref.sched_mask,
        st_ref.global_feats,
    )
    assert isinstance(res_nc, dict) and res_nc["conformal_breach_count"] == 0


def test_guardian_audit_blind_spots(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    from deepvol.api.eq_autocall_runtime import EQAutocallRuntime

    req_model, _ = parsed_dag
    st_ref = preprocess_eq_autocall(req_model)
    guardian = EQAutocallModelGuardian(
        d_max=8,
        m_max=48,
        tau_desk_bps=80.0,
        psi_window_size=4,
    )

    # Non-finite post_inference / NaN put_loss / empty put_loss / None optional bounds.
    ok_nan_out, reasons_nan = guardian.evaluate_post_inference(
        {
            "stopping_probs": torch.full((1, st_ref.m_rem), float("nan")),
            "coupon_probs": torch.full((1, st_ref.m_rem), 0.1),
            "put_loss": torch.tensor([0.05]),
        },
        None,
        st_ref,
    )
    assert ok_nan_out is False and AutocallFallbackReason.NON_FINITE_OUTPUT in reasons_nan

    st_inf_hw = copy.copy(st_ref)
    guardian_hw = EQAutocallModelGuardian(d_max=8, m_max=48, tau_desk_bps=80.0)
    guardian_hw.q_err_conformal = float("nan")
    guardian_hw._conformal_calibrated = True
    ok_nan_hw, reasons_hw, _, _ = guardian_hw.evaluate_pre_inference(st_inf_hw)
    assert ok_nan_hw is False and AutocallFallbackReason.NON_FINITE_OUTPUT in reasons_hw

    ok_empty_put, reasons_empty_put = guardian.evaluate_post_inference(
        {
            "stopping_probs": torch.zeros((1, st_ref.m_rem)),
            "coupon_probs": torch.zeros((1, st_ref.m_rem)),
            "put_loss": torch.zeros((0,)),
        },
        None,
        st_ref,
    )
    assert ok_empty_put is False and AutocallFallbackReason.NON_FINITE_OUTPUT in reasons_empty_put

    # baseline_out=None and sig_fx_obs=None must not crash pre-inference screening.
    st_null_opt = copy.copy(st_ref)
    st_null_opt.quanto_meta = {st_ref.sorted_identifiers[0]: {"rho_s_fx": 0.0, "sig_fx_obs": None}}
    ok_null, reasons_null, _, _ = guardian.evaluate_pre_inference(st_null_opt, baseline_out=None)
    assert ok_null is True and reasons_null == []

    # NaN in pv_proxy / v_w must fail closed rather than leaking through min(nan, 0.075).
    b_nan = copy.copy(st_ref.baseline)
    assert b_nan is not None
    b_nan.product_value_proxy = torch.tensor(float("nan"), dtype=torch.float64)
    ok_b_nan, _, _, _ = guardian.evaluate_pre_inference(st_ref, baseline_out=b_nan)
    assert ok_b_nan is False

    # PSI tracking must stay clean when trade_info is None/"" or psi_window_size=0.
    g_psi0 = EQAutocallModelGuardian(d_max=8, m_max=48, psi_window_size=0)
    st_no_tid = copy.copy(st_ref)
    st_no_tid.trade_info = None
    for _ in range(3):
        ok_p0, reasons_p0, _, _ = g_psi0.evaluate_pre_inference(st_no_tid)
        assert ok_p0 is True and reasons_p0 == []

    # evaluate_tensor_batch_post enforces simplex / put bounds checks cleanly.
    m_b = st_ref.m_rem
    post_diag = guardian.evaluate_tensor_batch_post(
        torch.full((2, m_b), 0.1),
        torch.full((2, m_b), 0.1),
        torch.tensor([9.9, 9.9]),
        torch.tensor([0.9, 0.9]),
        torch.ones((2, m_b)),
        torch.zeros((2, 16)),
    )
    assert post_diag["all_passed"] is False

    # Runtime engine="surrogate" must trigger failover/reject on corrupted surrogate NaN output
    # and persist audit_log[-1] fields cleanly.
    runtime = EQAutocallRuntime(device="cpu", enable_compiled_runner=False)
    orig_model = runtime.surrogate
    if orig_model is not None:
        class _NanSurrogate(torch.nn.Module):
            d_max = getattr(orig_model, "d_max", 8)
            m_max = getattr(orig_model, "m_max", 48)

            def is_zero_residual(self) -> bool:
                return False

            def forward(self, *args: Any, **kwargs: Any) -> AutocallSurrogateOutput:
                b = args[0].shape[0]
                m = args[3].shape[1]
                nan_m = torch.full((b, m), float("nan"))
                return AutocallSurrogateOutput(
                    stopping_probs=nan_m,
                    survival_probs=torch.ones((b, m)),
                    hazard_rates=nan_m,
                    coupon_probs=torch.zeros((b, m)),
                    put_loss=torch.full((b,), float("nan")),
                    delta_hazard_logits=torch.zeros((b, m)),
                    delta_coupon_logits=torch.zeros((b, m)),
                    delta_put_loss=torch.zeros((b,)),
                )

        runtime.surrogate = _NanSurrogate()  # type: ignore[assignment]
        resp_surr_fail = runtime.price_trade(req_model, engine="surrogate", include_all_greeks=False)
        assert np.isfinite(resp_surr_fail.ProductValue)
        assert len(runtime.guardian.audit_log) >= 1
        assert runtime.guardian.audit_log[-1].greeks_engine_used is not None

    # m_rem == 0 request stays deterministic and finite without Tier-2 fallback failure.
    req_m0 = copy.deepcopy(req_model)
    as_of_0 = req_m0.Model.MarketDataSet.AsOfDate
    req_m0.Product.ObservationDates[-1] = as_of_0 - datetime.timedelta(days=1)
    for tkr, ref_s in zip(req_m0.Product.UnderlyingIdentifiers, req_m0.Product.UnderlyingReferenceSpots):
        req_m0.Model.MarketDataSet.Fixings[tkr].Dates.append(req_m0.Product.ObservationDates[-1])
        req_m0.Model.MarketDataSet.Fixings[tkr].Values.append(float(ref_s) * 0.85)
    resp_m0 = EQAutocallRuntime(device="cpu", enable_compiled_runner=False).price_trade(
        req_m0, engine="auto", include_all_greeks=True
    )
    assert np.isfinite(resp_m0.ProductValue)


def test_guardian_8d_signed_eb_mondrian_and_tau_res_tightening(
    tmp_path: Path,
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    zero_head_surrogate: MaskedEQAutocallSurrogate,
) -> None:
    """Verifies 8D signed EB-Mondrian conformal gating, w8[1]>=5e-4 floor, tau_desk=15 bps, and tau_res<=25 bps."""
    from deepvol.mrm.eq_autocall_guardian import extract_16d_basket_summary

    req_model, _ = parsed_dag
    st_ref = preprocess_eq_autocall(req_model)

    guardian_ref = EQAutocallModelGuardian(record_drift_by_default=False)
    ref_vecs = np.tile(extract_16d_basket_summary(st_ref)[None, :], (32, 1))
    guardian_ref.fit_reference_distribution(ref_vecs)

    # Save a signed 8D Empirical-Bayes Mondrian checkpoint with tau_desk_bps=60.0 and sub-floor w8[1]=0.0
    stats_8d_path = tmp_path / "guardian_8d_stats.pt"
    w8_init = np.array([1e-4, 0.0, 2e-4, 1e-4, 2e-4, 1e-4, 1e-4, 0.05], dtype=np.float64)
    stats_payload = {
        "schema_version": "3.0",
        "conformal_mode": "signed_eb_mondrian_8d",
        "ref_mean_16d": torch.tensor(guardian_ref.ref_mean_16d),
        "ref_cov_inv_16d": torch.tensor(guardian_ref.ref_cov_inv_16d),
        "ref_cov_oas_16d": torch.tensor(np.linalg.pinv(guardian_ref.ref_cov_inv_16d)),
        "ref_marginal_std_16d": torch.tensor(guardian_ref.ref_marginal_std_16d),
        "psi_bin_edges": torch.tensor(guardian_ref.psi_bin_edges),
        "psi_ref_probs": torch.tensor(guardian_ref.psi_ref_probs),
        "q_ood_conformal": 39.25,
        "q_err_conformal": 2.0,
        "b_global_conformal": -2e-4,
        "q_lo_conformal": 2.0,
        "q_hi_conformal": 1.8,
        "scale_weights_8d": torch.tensor(w8_init),
        "mondrian_bias_table": {"ki1_m1_d0_mem0": -1e-4},
        "mondrian_q_lo_table": {"ki1_m1_d0_mem0": 2.2},
        "mondrian_q_hi_table": {"ki1_m1_d0_mem0": 1.6},
        "tau_desk_bps": 60.0,
        "conformal_alpha": 0.01,
        "n_cal": 4096,
    }
    torch.save(stats_payload, stats_8d_path)

    g8 = EQAutocallModelGuardian(stats_path=stats_8d_path, record_drift_by_default=False)
    assert g8.conformal_mode == "signed_eb_mondrian_8d"
    assert g8.scale_weights_8d[1] >= 5e-4 - 1e-12
    assert g8.tau_desk_bps == pytest.approx(15.0)

    hw_ref = g8.compute_conformal_half_width_bps(st_ref)
    assert math.isfinite(hw_ref) and 0.0 < hw_ref <= 15.0

    pre_diag_8d = g8.evaluate_tensor_batch_pre(
        st_ref.node_feats,
        st_ref.edge_feats,
        st_ref.asset_mask,
        st_ref.sched_feats,
        st_ref.sched_mask,
        st_ref.global_feats,
        record_drift=False,
    )
    assert pre_diag_8d["max_conformal_half_width_bps"] == pytest.approx(hw_ref, rel=1e-6, abs=1e-6)

    # Tighten residual gate: 300 bps (0.030) gap between surrogate PV and proxy must reject (tau_res <= 25 bps).
    m_r = st_ref.m_rem
    b_out = st_ref.baseline
    assert b_out is not None
    post_surr = assemble_autocall_cashflows_and_pv(
        st_ref,
        p_stop=b_out.p_proxy,
        q_cpn=b_out.q_proxy,
        l_put=b_out.l_put_proxy,
        build_cashflow_items=False,
    )
    post_surr.product_value = post_surr.product_value + 0.030
    s_out_bad = zero_head_surrogate(
        node_feats=st_ref.to_tier2_float32_tensors()["node_feats"],
        edge_feats=st_ref.to_tier2_float32_tensors()["edge_feats"],
        asset_mask=st_ref.to_tier2_float32_tensors()["asset_mask"],
        sched_feats=st_ref.to_tier2_float32_tensors()["sched_feats"],
        sched_mask=st_ref.to_tier2_float32_tensors()["sched_mask"],
        global_feats=st_ref.to_tier2_float32_tensors()["global_feats"],
    )
    ok_post, post_reasons = g8.evaluate_post_inference(s_out_bad, post_surr, st_ref, b_out)
    assert ok_post is False
    assert AutocallFallbackReason.POST_INFERENCE_RESIDUAL_OUTLIER in post_reasons

    # Batched post check also enforces max 25.0 bps (0.0250) residual ceiling
    pv_px_val = float(b_out.product_value_proxy.item())
    batch_post_bad = g8.evaluate_tensor_batch_post(
        b_out.p_proxy.unsqueeze(0),
        b_out.q_proxy.unsqueeze(0),
        b_out.l_put_proxy.unsqueeze(0),
        torch.tensor([pv_px_val + 0.026], dtype=torch.float64),
        st_ref.sched_mask,
        st_ref.global_feats,
        node_feats=st_ref.node_feats,
        asset_mask=st_ref.asset_mask,
        sched_feats=st_ref.sched_feats,
        pv_proxy=torch.tensor([pv_px_val], dtype=torch.float64),
    )
    assert batch_post_bad["residual_outlier_count"] == 1 and m_r > 0

    # High-uncertainty 8D cell triggers CONFORMAL_UNCERTAINTY_BREACH above 15.0 bps
    g8_high = EQAutocallModelGuardian(stats_path=stats_8d_path, record_drift_by_default=False)
    g8_high.mondrian_q_hi_table["ki1_m1_d0_mem0"] = 200.0
    g8_high.q_hi_conformal = 200.0
    g8_high._config_epoch += 1
    pre_ok_hi, pre_reasons_hi, _, _ = g8_high.evaluate_pre_inference(st_ref, b_out)
    assert pre_ok_hi is False
    assert AutocallFallbackReason.CONFORMAL_UNCERTAINTY_BREACH in pre_reasons_hi


def test_guardian_canonical_tier3_preserves_qmc_greeks_fn_pv_without_overwrite(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies price_with_guardian wires CanonicalTier3Config and never overwrites qmc_greeks_fn PV/CFs."""
    from deepvol.models.autocall.greeks_engine import AutocallGreeksResult

    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)
    guardian = EQAutocallModelGuardian(record_drift_by_default=False)
    captured: dict[str, Any] = {}

    def _mock_qmc_fn(st: AutocallPreprocState, *, config: Any = None, device: str = "cpu") -> AutocallGreeksResult:
        captured["config"] = config
        captured["device"] = device
        return AutocallGreeksResult(
            product_value=0.912345,
            cash_flows=[],
            standard_error=0.00042,
            greeks_engine_used="MockCanonicalQMC",
        )

    res = guardian.price_with_guardian(
        state,
        surrogate=None,
        force_qmc=True,
        qmc_greeks_fn=_mock_qmc_fn,
    )
    assert captured["config"] is not None
    assert int(captured["config"].n_paths) == 4096
    assert captured["config"].cv_mode == "multifidelity_gbm"
    assert res.product_value == pytest.approx(0.912345)
    assert res.standard_error == pytest.approx(0.00042)
    assert res.audit_record.greeks_engine_used == "MockCanonicalQMC"


def test_greeks_engine_v3_onestar_islongbasket_quanto_and_theta_alignment() -> None:
    """Covers DETACHED_NODE_COLS, V3 memory, OneStar left-spread, IsLongBasket I_ramp, quanto FX keys, and theta default."""
    from deepvol.models.autocall.greeks_engine import (
        DETACHED_NODE_COLS,
        _collect_quanto_fx_ids,
        _eval_autocall_payoff_from_3d_indices_np,
        _eval_autocall_payoff_pv_batch_np,
        compute_eq_autocall_greeks,
        compute_qmc_crn_greeks,
    )

    assert DETACHED_NODE_COLS == ()
    assert inspect.signature(compute_eq_autocall_greeks).parameters["theta_convention"].default == "murex_bizday_forward_fixed"
    assert inspect.signature(compute_qmc_crn_greeks).parameters["theta_convention"].default == "murex_bizday_forward_fixed"

    # 1. IsLongBasket preserves WorstOf KI monitoring and evaluates Payoff_IsLB(T) = 1 - (1 - Avg_T) * ki_wof_K
    req_lb, _ = _build_synthetic_autocall_dict(d_assets=2, m_periods=2)
    req_lb["Product"]["IsLongBasket"] = True
    req_lb["Product"]["Strike"] = 0.90
    req_lb["Product"]["PutSpreadStrike"] = 0.90
    req_lb["Product"]["OptionDownAndInBarrierLevel"] = 0.90
    req_lb["Product"]["OptionGearing"] = 1.0 / 0.90
    req_lb["Product"]["OptionGearingRatio"] = 1.0 / 0.90
    req_lb["Product"]["CallabilityBarrierLevels"] = [2.0, 2.0]
    req_lb["Product"]["CouponLevels"] = [0.0, 0.0]
    req_lb["Product"]["IsGuaranteedCoupon"] = [True, True]
    req_lb["Product"]["CouponBarrierLevels"] = [0.0, 0.0]
    st_lb = preprocess_eq_autocall(req_lb)
    st_lb.option_index = "WorstOf"
    st_lb.coupon_amounts = torch.zeros_like(st_lb.coupon_amounts)

    # Asset 0 = 0.80 (< K=0.90 -> triggers WorstOf KI!), Asset 1 = 1.20 -> Avg_T = 1.00 -> Payoff = 1.0 - (1.0 - 1.0) = 1.0
    perf_lb = np.ones((1, 2, 2, 1), dtype=np.float64)
    perf_lb[0, 0, :, 0] = 0.80
    perf_lb[0, 1, :, 0] = 1.20
    pv_lb = float(
        _eval_autocall_payoff_pv_batch_np(
            perf_lb,
            st_lb,
            df_pay_b=np.ones((1, 2), dtype=np.float64),
            w_fund_b=np.zeros((1, 2), dtype=np.float64),
            pv_in_transit_b=np.zeros(1, dtype=np.float64),
        )[0]
    )
    assert pv_lb == pytest.approx(1.0, abs=1e-10)

    # Asset 0 = 0.70, Asset 1 = 0.90 -> Avg_T = 0.80, WorstOf = 0.70 < 0.90 -> Payoff = 0.80 (no gearing 1.1111 factor!)
    perf_lb[0, 0, :, 0] = 0.70
    perf_lb[0, 1, :, 0] = 0.90
    pv_lb2 = float(
        _eval_autocall_payoff_pv_batch_np(
            perf_lb,
            st_lb,
            df_pay_b=np.ones((1, 2), dtype=np.float64),
            w_fund_b=np.zeros((1, 2), dtype=np.float64),
            pv_in_transit_b=np.zeros(1, dtype=np.float64),
        )[0]
    )
    assert pv_lb2 == pytest.approx(0.80, abs=1e-10)

    # 2. V3 memory carry over partial KO ramp (surv_prev < 1) & OneStar left-shifted spread
    req_os, _ = _build_synthetic_autocall_dict(d_assets=2, m_periods=3)
    req_os["Product"]["IsMemory"] = True
    req_os["Product"]["IsGuaranteedCoupon"] = [False, False, False]
    req_os["Product"]["CouponBarrierLevels"] = [0.95, 0.95, 0.95]
    req_os["Product"]["CallabilityBarrierLevels"] = [1.00, 1.00, 1.00]
    req_os["Product"]["AutoCallSpread"] = -0.10
    req_os["Product"]["Strike"] = 1.0
    req_os["Product"]["PutSpreadStrike"] = 1.0
    req_os["Product"]["OptionGearing"] = 1.0
    req_os["Product"]["OptionGearingRatio"] = 1.0
    req_os["Product"]["OptionDownAndInBarrierLevel"] = 0.85
    req_os["Product"]["OneStarStrike"] = 1.10
    req_os["Product"]["OneStarCoupon"] = 0.0
    req_os["Product"]["NoNotionalRepayment"] = False
    st_os = preprocess_eq_autocall(req_os)
    st_os.coupon_amounts = torch.full_like(st_os.coupon_amounts, 0.05)

    # m=0: g_ko=0.5, g_cpn=0 -> p_0=0.5, surv_1=0.5, U_0 = 1.0 * 0.05 = 0.05
    # m=1: g_ko=0.0, g_cpn=0 -> p_1=0.0, surv_2=0.5, U_1 = surv_1 * (0.05 + 0.05) = 0.05 (V3 literal, vs 0.10 without surv!)
    # m=2: g_ko=1.0, g_cpn=1 -> p_2=0.5, cpn_2 = surv_2 * (0.05 + U_1) = 0.5 * 0.10 = 0.05 -> PV = 1.05
    w_call_v3 = np.array([[[0.95], [0.50], [1.20]]], dtype=np.float64)
    w_cpn_v3 = np.array([[[0.50], [0.50], [1.20]]], dtype=np.float64)
    w_opt_v3 = np.array([[[0.95], [0.50], [1.20]]], dtype=np.float64)
    w_best_v3 = np.array([[1.05]], dtype=np.float64)  # Inside left-spread [1.00, 1.10] -> star_ind = 0.5
    pv_v3_os = float(
        _eval_autocall_payoff_from_3d_indices_np(
            w_call_v3,
            w_cpn_v3,
            w_opt_v3,
            w_best_v3,
            st_os,
            df_pay_b=np.ones((1, 3), dtype=np.float64),
            w_fund_b=np.zeros((1, 3), dtype=np.float64),
            pv_in_transit_b=np.zeros(1, dtype=np.float64),
        )[0]
    )
    assert pv_v3_os == pytest.approx(1.05, abs=1e-10)

    # OneStar left-shifted check with KI put active (w_opt=0.60, strike=1.0): put_intr=0.40, star_ind=0.5 -> l_put=0.20 (vs 0.40 under centered)
    w_call_os = np.array([[[0.50], [0.50], [0.50]]], dtype=np.float64)
    w_cpn_os = np.array([[[0.50], [0.50], [0.50]]], dtype=np.float64)
    w_opt_os = np.array([[[0.60], [0.60], [0.60]]], dtype=np.float64)
    pv_os = float(
        _eval_autocall_payoff_from_3d_indices_np(
            w_call_os,
            w_cpn_os,
            w_opt_os,
            w_best_v3,
            st_os,
            df_pay_b=np.ones((1, 3), dtype=np.float64),
            w_fund_b=np.zeros((1, 3), dtype=np.float64),
            pv_in_transit_b=np.zeros(1, dtype=np.float64),
        )[0]
    )
    # notional = 1.0, cpn = 0.0, l_put = 0.40 * (1 - 0.5) = 0.20 -> PV = 0.80
    assert pv_os == pytest.approx(0.80, abs=1e-10)

    # 3. Quanto FX zero Delta/Gamma key extraction helper
    st_lb.quanto_fx_specs = {"USDRUB": None, "CNHRUB": None}
    fx_keys = _collect_quanto_fx_ids(st_lb)
    assert fx_keys == ["CNHRUB", "USDRUB"]







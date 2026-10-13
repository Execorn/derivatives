import pytest
import numpy as np
import torch
from deepvol.mrm.arbitrage import check_calendar_arbitrage, check_butterfly_arbitrage_durrleman, check_butterfly_arbitrage_price, check_arbitrage
from deepvol.mrm.guardian import ModelRiskGuardian

def test_calendar_arbitrage_detection():
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Safe surface: total variance w = iv^2 * T increases with T
    # T=0.1: w = 0.2^2 * 0.1 = 0.004
    # T=0.5: w = 0.2^2 * 0.5 = 0.02
    # T=1.0: w = 0.2^2 * 1.0 = 0.04
    safe_iv = np.full((3, 3), 0.2)
    res_safe = check_calendar_arbitrage(safe_iv, T_grid)
    assert not res_safe["has_arbitrage"]
    
    # Anomalous surface: total variance decreases with T
    # T=0.1: w = 0.4^2 * 0.1 = 0.016
    # T=0.5: w = 0.1^2 * 0.5 = 0.005 (decreases!)
    bad_iv = np.array([
        [0.4, 0.4, 0.4],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2]
    ])
    res_bad = check_calendar_arbitrage(bad_iv, T_grid)
    assert res_bad["has_arbitrage"]
    assert res_bad["violations"][0, 0]  # First transition decreases

def test_butterfly_arbitrage_detection():
    T_grid = np.array([0.5, 1.0])
    K_grid = np.array([-0.5, 0.0, 0.5])
    
    # Flat smile (no arbitrage)
    safe_iv = np.full((2, 3), 0.2)
    res_safe_dur = check_butterfly_arbitrage_durrleman(safe_iv, K_grid, T_grid)
    res_safe_prc = check_butterfly_arbitrage_price(safe_iv, K_grid, T_grid)
    assert not res_safe_dur["has_arbitrage"]
    assert not res_safe_prc["has_arbitrage"]
    
    # Severe butterfly arbitrage: put a massive hump in the middle strike IV
    # This creates a non-convex option price surface / negative probability density
    bad_iv = np.array([
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1]
    ])
    res_bad_dur = check_butterfly_arbitrage_durrleman(bad_iv, K_grid, T_grid)
    res_bad_prc = check_butterfly_arbitrage_price(bad_iv, K_grid, T_grid)
    assert res_bad_dur["has_arbitrage"]
    assert res_bad_prc["has_arbitrage"]

def test_guardian_parameter_checks():
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99, hurst_limit=0.015, residual_limit=0.0150)
    
    # 1. Clean Heston
    res_clean_heston = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04}
    )
    assert not res_clean_heston["anomaly_detected"]
    
    # 2. Pinned Heston vol-of-vol
    res_pinned_heston = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.995, "rho": -0.6, "v0": 0.04}
    )
    assert res_pinned_heston["anomaly_detected"]
    assert any("vol-of-vol pinned" in a for a in res_pinned_heston["anomalies"])
    
    # 3. Clean Rough Bergomi
    res_clean_rb = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.1, "eta": 0.5, "rho": -0.7}
    )
    assert not res_clean_rb["anomaly_detected"]
    
    # 4. Pinned Hurst exponent
    res_pinned_rb = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.01, "eta": 0.5, "rho": -0.7}
    )
    assert res_pinned_rb["anomaly_detected"]
    assert any("Hurst exponent pinned" in a for a in res_pinned_rb["anomalies"])
    
    # 5. Pinned vol-of-vol rough Bergomi
    res_pinned_rb_vol = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.1, "eta": 1.0, "rho": -0.7}
    )
    assert res_pinned_rb_vol["anomaly_detected"]
    assert any("vol-of-vol pinned" in a for a in res_pinned_rb_vol["anomalies"])
    
    # 6. High residuals (> 150 bps)
    res_high_res = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04},
        rmse=0.0180
    )
    assert res_high_res["anomaly_detected"]
    assert any("residual exceeds" in a for a in res_high_res["anomalies"])

def test_guardian_price_or_fallback_clean():
    guardian = ModelRiskGuardian()
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="fourier"
    )
    assert not res["fallback_triggered"]
    assert res["guardian_status"] == "passed"
    assert res["prices"].shape == (3, 3)
    assert res["ivs"].shape == (3, 3)
    assert not np.isnan(res["prices"]).any()

def test_guardian_price_or_fallback_anomalous_fourier():
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99)
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Pinned parameters to trigger fallback
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 1.0, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="fourier"
    )
    assert res["fallback_triggered"]
    assert res["guardian_status"] == "fallback_fourier_safe_params"
    assert len(res["anomalies"]) > 0
    assert res["prices"].shape == (3, 3)
    assert res["ivs"].shape == (3, 3)
    assert not np.isnan(res["prices"]).any()
    
def test_guardian_price_or_fallback_anomalous_particle():
    # Only run MLSV particle solver if CPU/GPU supports it
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99)
    T_grid = np.array([0.1, 0.5])
    K_grid = np.array([-0.1, 0.0, 0.1])
    
    # Pinned parameters to trigger fallback
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 1.0, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="particle"
    )
    assert res["fallback_triggered"]
    assert res["guardian_status"] == "fallback_particle_safe_params"
    assert len(res["anomalies"]) > 0
    assert res["prices"].shape == (2, 3)
    assert res["ivs"].shape == (2, 3)


# ── F-07: Dedicated price_to_iv edge-case tests ──────────────────────────────

from deepvol.mrm.guardian import price_to_iv


def test_price_to_iv_round_trip():
    """Verify that price_to_iv inverts a known BS price back to the correct IV."""
    import scipy.stats as stats
    S, K, T, sigma_true = 100.0, 100.0, 1.0, 0.25
    d1 = (np.log(S / K) + 0.5 * sigma_true**2 * T) / (sigma_true * np.sqrt(T))
    d2 = d1 - sigma_true * np.sqrt(T)
    price = S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)

    iv_recovered = price_to_iv(price, S, K, T)
    assert abs(iv_recovered - sigma_true) < 1e-6, f"Expected {sigma_true}, got {iv_recovered}"


def test_price_to_iv_below_intrinsic():
    """Price ≤ intrinsic should return the floor IV (1e-4)."""
    S, K, T = 110.0, 100.0, 0.5
    intrinsic = S - K  # = 10.0
    iv = price_to_iv(intrinsic - 1.0, S, K, T)
    assert iv == 1e-4, f"Expected floor 1e-4 for sub-intrinsic price, got {iv}"

    iv_exact = price_to_iv(intrinsic, S, K, T)
    assert iv_exact == 1e-4, f"Expected floor 1e-4 for exact intrinsic price, got {iv_exact}"


def test_price_to_iv_at_spot():
    """Price ≥ spot should return the ceiling IV (5.0)."""
    S, K, T = 100.0, 100.0, 1.0
    iv = price_to_iv(S, S, K, T)
    assert iv == 5.0, f"Expected ceiling 5.0 for price=spot, got {iv}"

    iv_above = price_to_iv(S + 10.0, S, K, T)
    assert iv_above == 5.0, f"Expected ceiling 5.0 for price>spot, got {iv_above}"


def test_price_to_iv_deep_otm():
    """Deep OTM option (very small price) should return a small but valid IV."""
    S, K, T = 100.0, 200.0, 0.5  # Very deep OTM
    price = 0.001  # Tiny premium
    iv = price_to_iv(price, S, K, T)
    assert 1e-4 <= iv <= 5.0, f"IV {iv} outside valid range"
    assert np.isfinite(iv), f"IV is not finite: {iv}"


def test_price_to_iv_zero_and_negative():
    """Zero or negative prices should return floor IV without error."""
    S, K, T = 100.0, 100.0, 1.0
    iv_zero = price_to_iv(0.0, S, K, T)
    assert iv_zero == 1e-4

    iv_neg = price_to_iv(-5.0, S, K, T)
    assert iv_neg == 1e-4


# ===========================================================================
# Milestone 3 — Tasks 3.3 & 3.4: FX Cross Guardian, OOD Detector & Reserve Tracker
# ===========================================================================

import asyncio
from datetime import datetime, timedelta, timezone
import time
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
    MarketStateDriftMonitor,
    MarketStateOODDetector,
    ModelReserveTracker,
    PricingEngineEnum,
    RoutingDecision,
)
from deepvol.market.svi_slice import RawSVI
from deepvol.models.fx_cross_analytic import AnalyticTriangulated, Col
from deepvol.models.fx_cross_copula import CopulaConfig, CrossState, GaussianCopulaCrossEngine
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet


def _make_sample_12d_market_states(n: int, seed: int = 42) -> torch.Tensor:
    """Generates synthetic 12-D market states within typical ranges."""
    gen = torch.Generator().manual_seed(seed)
    # 0: sqrt(T) in [0.05, 2.2]
    sqrt_T = torch.empty(n, 1, dtype=torch.float64).uniform_(0.05, 1.5, generator=gen)
    # 1: rho in [-0.8, 0.8]
    rho = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.8, 0.8, generator=gen)
    # 2..6: xi1 (atm, rr25, bf25, rr10, bf10)
    atm1 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.10, 0.40, generator=gen)
    rr25_1 = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.15, 0.15, generator=gen)
    bf25_1 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.01, 0.08, generator=gen)
    rr10_1 = rr25_1 * 1.8
    bf10_1 = bf25_1 * 3.0
    # 7..11: xi2
    atm2 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.10, 0.50, generator=gen)
    rr25_2 = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.20, 0.20, generator=gen)
    bf25_2 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.01, 0.10, generator=gen)
    rr10_2 = rr25_2 * 1.8
    bf10_2 = bf25_2 * 3.0

    return torch.cat([sqrt_T, rho, atm1, rr25_1, bf25_1, rr10_1, bf10_1, atm2, rr25_2, bf25_2, rr10_2, bf10_2], dim=1)


def _make_34col_state_from_12d(m_12d: torch.Tensor) -> torch.Tensor:
    """Expands 12-D market state into a 34-column engine tensor."""
    S_cnt = m_12d.shape[0]
    X = torch.zeros(S_cnt, 34, dtype=torch.float64)
    sq_T = m_12d[:, 0]
    X[:, Col.T] = sq_T ** 2
    X[:, Col.T_SET] = sq_T ** 2
    X[:, Col.RHO] = m_12d[:, 1]
    X[:, Col.XI1 : Col.XI1 + 5] = m_12d[:, 2:7]
    X[:, Col.XI2 : Col.XI2 + 5] = m_12d[:, 7:12]

    # Standard defaults for pricing
    X[:, Col.F] = 100.0
    X[:, Col.K] = 100.0
    X[:, Col.D] = 0.95
    X[:, Col.G] = 1.0
    X[:, Col.PHI] = 1.0
    X[:, Col.NN_GATE] = 1.0

    # Basic SVI params matching ATM vols
    atm1 = m_12d[:, 2]
    atm2 = m_12d[:, 7]
    T_val = sq_T ** 2
    X[:, Col.SVI1] = (atm1 ** 2) * T_val
    X[:, Col.SVI1 + 1] = 0.05
    X[:, Col.SVI1 + 4] = 0.10
    X[:, Col.SVI2] = (atm2 ** 2) * T_val
    X[:, Col.SVI2 + 1] = 0.05
    X[:, Col.SVI2 + 4] = 0.10

    return X


def test_market_state_ood_detector_basic() -> None:
    """Verifies MarketStateOODDetector fits on training states and evaluates scores correctly."""
    train_states = _make_sample_12d_market_states(1000, seed=123)
    detector = MarketStateOODDetector.fit(train_states, margin=0.02)

    assert detector.mean.shape == (12,)
    assert detector.cov_inv.shape == (12, 12)
    assert detector.q995 > 0.0
    assert detector.box_lo.shape == (12,)
    assert detector.box_hi.shape == (12,)

    # Mean state should have near-zero Mahalanobis distance and not be OOD
    mean_state = detector.mean.unsqueeze(0)
    d_m, box_viol, is_ood = detector.score(mean_state)
    assert float(d_m[0].item()) < 1e-6
    assert box_viol[0].item() is False
    assert is_ood[0].item() is False


def test_in_distribution_false_positive_rate_g33() -> None:
    """Gate G3.3: In-distribution test-split false-positive rate is 0.5% +/- 0.3%."""
    train_states = _make_sample_12d_market_states(5000, seed=100)
    detector = MarketStateOODDetector.fit(train_states, margin=0.02)

    # 2,000 holdout in-distribution test states from same generator
    test_states = _make_sample_12d_market_states(2000, seed=200)
    _, _, is_ood = detector.score(test_states)

    fpr = float(is_ood.double().mean().item())
    # 0.5% +/- 0.3% -> [0.002, 0.008]
    assert 0.002 <= fpr <= 0.008, f"False positive rate {fpr:.4f} outside Gate G3.3 range [0.002, 0.008]"


def test_g32_crisis_shocks_100_percent_fallback() -> None:
    """Gate G3.2: 1,000 synthetic crisis shocks achieve 100% fallback and bitwise equality to Analytic_Triangulated."""
    train_states = _make_sample_12d_market_states(2000, seed=300)
    detector = MarketStateOODDetector.fit(train_states, margin=0.02)
    guardian = FXCrossGuardian(ood_detector=detector)

    # Build 1,000 synthetic crisis shocks:
    # 500 Mahalanobis distance outliers beyond q995
    gen = torch.Generator().manual_seed(400)
    cov = torch.linalg.inv(detector.cov_inv)
    L = torch.linalg.cholesky(cov)

    unit_dirs = torch.randn(500, 12, dtype=torch.float64, generator=gen)
    unit_dirs = unit_dirs / torch.norm(unit_dirs, dim=-1, keepdim=True)
    radii = torch.empty(500, 1, dtype=torch.float64).uniform_(1.2 * detector.q995, 2.5 * detector.q995, generator=gen)
    mahalanobis_crisis = detector.mean + (unit_dirs @ L.t()) * radii

    # 500 box violations (e.g. RUB ATM vol 150%, extreme correlation, extreme RR)
    box_crisis = train_states[:500].clone()
    for i in range(500):
        mode = i % 4
        if mode == 0:
            box_crisis[i, 7] = 1.50  # Leg 2 ATM vol = 150% (box_hi typically ~ 0.50)
        elif mode == 1:
            box_crisis[i, 1] = 0.998  # extreme correlation
        elif mode == 2:
            box_crisis[i, 3] = 0.40   # RR25 = 40% (box_hi ~ 0.15)
        else:
            box_crisis[i, 0] = 0.01   # short expiry < 1 day (box_lo ~ 0.05)

    crisis_1000 = torch.cat([mahalanobis_crisis, box_crisis], dim=0)
    assert crisis_1000.shape == (1000, 12)

    # Convert to 34-column engine tensor
    X_crisis = _make_34col_state_from_12d(crisis_1000)

    # Route through guardian
    decision = guardian.route(X_crisis, engine=PricingEngineEnum.DIFFERENTIAL_ML)

    # 1. 100% fallback activated
    assert decision.fallback_activated.all().item() is True, "Expected 100% fallback activation"
    assert (decision.nn_gate == 0.0).all().item() is True, "Expected nn_gate == 0.0 for all crisis states"
    assert all(r == FallbackReasonEnum.OOD_MARKET_REGIME.value for r in decision.fallback_reason)

    # 2. Bitwise equality to Analytic_Triangulated
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    surrogate_engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)
    analytic_engine = AnalyticTriangulated()

    # Pass gated tensor to surrogate (nn_gate = 0.0)
    X_gated = X_crisis.clone()
    X_gated[:, Col.NN_GATE] = decision.nn_gate

    out_surrogate = surrogate_engine(X_gated)
    out_analytic = analytic_engine(X_crisis)

    assert torch.equal(out_surrogate.V, out_analytic.V), "Surrogate with nn_gate=0 not bitwise equal to Analytic_Triangulated"


def test_epistemic_uncertainty_post_check() -> None:
    """Task 3.3(c): Epistemic uncertainty post-check trigger (std > max(1.5 bp, 0.15 * reserve))."""
    guardian = FXCrossGuardian(epistemic_abs_floor_bp=1.5, epistemic_reserve_frac=0.15)

    # Case 1: Low std <= 1.5 bp, reserve = 0.0 -> no trigger
    std1 = torch.tensor([1.2, 1.49, 1.50], dtype=torch.float64)
    res1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    mask1 = guardian.post_check(std1, res1)
    assert not mask1.any().item()

    # Case 2: High std > 1.5 bp, reserve = 0.0 -> trigger
    std2 = torch.tensor([1.51, 2.5], dtype=torch.float64)
    res2 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    mask2 = guardian.post_check(std2, res2)
    assert mask2.all().item()

    # Case 3: High reserve = 20 bp -> threshold = max(1.5, 0.15 * 20) = 3.0 bp
    # std = 2.5 bp < 3.0 bp -> no trigger; std = 3.2 bp > 3.0 bp -> trigger
    std3 = torch.tensor([2.5, 3.2], dtype=torch.float64)
    res3 = torch.tensor([20.0, 20.0], dtype=torch.float64)
    mask3 = guardian.post_check(std3, res3)
    assert mask3[0].item() is False
    assert mask3[1].item() is True


def test_market_state_drift_monitor_psi() -> None:
    """Task 3.3(d): Rolling PSI drift alert (> 0.25) forcing Hybrid_Ensemble gating."""
    baseline = _make_sample_12d_market_states(500, seed=500)
    monitor = MarketStateDriftMonitor(baseline=baseline, window_size=200, psi_alert_threshold=0.25)

    # Record 200 in-distribution states
    normal_states = _make_sample_12d_market_states(200, seed=501)
    monitor.record(normal_states)
    assert monitor.is_drift_detected() is False

    # Inject severe parameter drift: shift ATM vol by +4 standard deviations
    drifted_states = normal_states.clone()
    drifted_states[:, 2] = drifted_states[:, 2] + 0.50  # Massive shift in leg 1 ATM vol
    monitor.record(drifted_states)

    assert monitor.is_drift_detected() is True

    # Test that guardian route() forces HYBRID_ENSEMBLE when drift is active
    guardian = FXCrossGuardian(drift_monitor=monitor)
    X = _make_34col_state_from_12d(normal_states[:5])
    decision = guardian.route(X, engine=PricingEngineEnum.DIFFERENTIAL_ML)

    assert all(eng == PricingEngineEnum.HYBRID_ENSEMBLE for eng in decision.engine_used)


# ---------------------------------------------------------------------------
# Task 3.4 Tests: Model Reserve Tracker
# ---------------------------------------------------------------------------


class _MockCopulaEngine:
    """Mock Gaussian copula engine for model reserve testing."""

    def __init__(self, delay_sec: float = 0.0) -> None:
        self.delay_sec = delay_sec
        self.device = "cpu"
        self.call_count = 0

    def price(self, state: CrossState, strikes: torch.Tensor, is_call: bool = True) -> torch.Tensor:
        self.call_count += 1
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        # Returns deterministic synthetic price: max(F - K, 0) + 2.0
        F_val = float(state.F.squeeze()[0].item() if state.F.dim() > 0 else state.F.item())
        K_f64 = strikes.to(dtype=torch.float64)
        c = torch.clamp(F_val - K_f64, min=0.0) + 2.0
        if c.dim() == 1:
            c = c.unsqueeze(0)
        return c


def _make_test_cross_state() -> CrossState:
    """Builds a simple CrossState for testing."""
    F1 = torch.tensor([2650.0], dtype=torch.float64)
    F2 = torch.tensor([95.0], dtype=torch.float64)
    T = torch.tensor([1.0], dtype=torch.float64)
    svi1 = RawSVI(torch.tensor([[0.04, 0.10, -0.20, 0.0, 0.10]], dtype=torch.float64))
    svi2 = RawSVI(torch.tensor([[0.09, 0.15, 0.25, 0.0, 0.12]], dtype=torch.float64))
    rho = torch.tensor([-0.20], dtype=torch.float64)
    return CrossState(F1=F1, F2=F2, T=T, svi1=svi1, svi2=svi2, rho=rho)


def test_model_reserve_tracker_direct_equivalence() -> None:
    """Task 3.4(a): Model reserve equals direct computation to 10^-10 relative error."""
    mock_copula = _MockCopulaEngine()
    tracker = ModelReserveTracker(copula=mock_copula)  # type: ignore[arg-type]
    state = _make_test_cross_state()

    # Served pricing function returning max(F-K, 0) + 1.8 (diff is 0.2)
    def served_fn(K: torch.Tensor) -> torch.Tensor:
        F_val = float(state.F.item())
        return torch.clamp(F_val - K, min=0.0) + 1.8

    # Tracker computation
    reserve_tracker = tracker.compute_reserve(state, served_fn)

    # Independent direct computation
    strikes = tracker.generate_strikes(state)
    copula_prices = mock_copula.price(state, strikes, is_call=True).squeeze(0)
    served_prices = served_fn(strikes).squeeze(0)
    denom = float(state.F.item())
    max_err = float(torch.max(torch.abs(copula_prices - served_prices)).item())
    direct_reserve = (max_err / denom) * 10000.0

    rel_err = abs(reserve_tracker - direct_reserve) / direct_reserve
    assert rel_err <= 1e-10, f"Relative error {rel_err} > 1e-10"


def test_model_reserve_tracker_staleness_lifecycle() -> None:
    """Task 3.4(b): Staleness flags asserted on aging and market-data updates."""
    mock_copula = _MockCopulaEngine()
    tracker = ModelReserveTracker(copula=mock_copula, max_age=timedelta(minutes=15))  # type: ignore[arg-type]
    state = _make_test_cross_state()

    def served_fn(K: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(K)

    # 1. Cache miss returns (None, True)
    val, stale = tracker.get("missing_key")
    assert val is None
    assert stale is True

    # 2. Fresh computation
    t0 = datetime(2026, 10, 13, 10, 0, 0, tzinfo=timezone.utc)
    asyncio.run(tracker.refresh("state_1", state, served_fn, state_timestamp=t0))

    val, stale = tracker.get("state_1", state_timestamp=t0)
    assert val is not None
    assert stale is False

    # 3. Market data timestamp update (newer market data invalidates old entry)
    t1 = datetime(2026, 10, 13, 10, 5, 0, tzinfo=timezone.utc)
    val, stale = tracker.get("state_1", state_timestamp=t1)
    assert val is not None
    assert stale is True  # Outdated compared to t1


def test_model_reserve_tracker_non_blocking_latency() -> None:
    """Task 3.4(c): Non-blocking execution verified with 0.5s mocked copula (< 10 us quote lookup)."""
    # Copula with 0.5s delay
    slow_copula = _MockCopulaEngine(delay_sec=0.5)
    tracker = ModelReserveTracker(copula=slow_copula)  # type: ignore[arg-type]
    state = _make_test_cross_state()

    def served_fn(K: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(K)

    async def run_test() -> None:
        # Start background refresh
        task = asyncio.create_task(tracker.refresh("test_state", state, served_fn))

        # While refresh is running in worker thread, quote lookups must be immediate (< 10 us)
        t_start = time.perf_counter()
        for _ in range(100):
            res, stale = tracker.get("test_state")
            assert stale is True
        elapsed_us = (time.perf_counter() - t_start) * 1e6 / 100
        assert elapsed_us < 10.0, f"Quote lookup took {elapsed_us:.2f} us > 10 us"

        # Await refresh
        await task

        # Subsequent lookup is fresh
        res, stale = tracker.get("test_state")
        assert res is not None
        assert stale is False

    asyncio.run(run_test())


def test_model_reserve_tracker_single_flight_dedup() -> None:
    """Task 3.4: 10 concurrent refresh tasks execute exactly 1 copula calculation."""
    slow_copula = _MockCopulaEngine(delay_sec=0.1)
    tracker = ModelReserveTracker(copula=slow_copula)  # type: ignore[arg-type]
    state = _make_test_cross_state()

    def served_fn(K: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(K)

    async def run_concurrent() -> None:
        tasks = [tracker.refresh("same_key", state, served_fn) for _ in range(10)]
        await asyncio.gather(*tasks)

    asyncio.run(run_concurrent())
    assert slow_copula.call_count == 1, f"Expected 1 copula call, got {slow_copula.call_count}"

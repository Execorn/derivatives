"""
tests/test_stress.py — Unit and stress testing for OptionPortfolioStressTester.
"""
import numpy as np
import torch
from deepvol.risk.stress_tester import (
    apply_surface_shifts,
    stress_portfolio,
    generate_stress_grid,
    OptionPortfolioStressTester,
    HISTORICAL_SCENARIOS
)

# Test grids matching typical FNO surfaces
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
K_GRID = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
S0 = 100.0
r = 0.05

# Setup a flat baseline vol surface (e.g. 20% flat vol)
BASE_VOL = 0.20
IV_SURFACE = np.full((len(T_GRID), len(K_GRID)), BASE_VOL, dtype=np.float32)


def test_apply_surface_shifts_numpy():
    """Verify that surface shifts are applied correctly on NumPy arrays."""
    flat_shift = 0.10
    skew_shift = -0.05
    term_shift = 0.15
    term_decay = 1.0

    # K_GRID contains log-moneyness values in [-0.5, 0.5]
    shifted = apply_surface_shifts(
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE,
        flat_shift=flat_shift,
        skew_shift=skew_shift,
        term_shift=term_shift,
        term_decay=term_decay,
        min_vol=1e-4
    )

    assert shifted.shape == IV_SURFACE.shape
    assert isinstance(shifted, np.ndarray)

    # Let's verify values at specific points mathematically
    for i, T in enumerate(T_GRID):
        for j, k in enumerate(K_GRID):
            expected = BASE_VOL + flat_shift + skew_shift * k + term_shift * np.exp(-term_decay * T)
            expected = max(expected, 1e-4)
            np.testing.assert_allclose(shifted[i, j], expected, rtol=1e-5)


def test_apply_surface_shifts_torch():
    """Verify that surface shifts work correctly on PyTorch tensors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iv_t = torch.tensor(IV_SURFACE, device=device, dtype=torch.float32)

    flat_shift = 0.15
    skew_shift = 0.05
    term_shift = -0.10
    term_decay = 2.0

    shifted = apply_surface_shifts(
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=iv_t,
        flat_shift=flat_shift,
        skew_shift=skew_shift,
        term_shift=term_shift,
        term_decay=term_decay,
        min_vol=1e-4
    )

    assert isinstance(shifted, torch.Tensor)
    assert shifted.device == iv_t.device
    assert shifted.shape == iv_t.shape

    # Compare with expected numpy calculations
    shifted_np = shifted.cpu().numpy()
    for i, T in enumerate(T_GRID):
        for j, k in enumerate(K_GRID):
            expected = BASE_VOL + flat_shift + skew_shift * k + term_shift * np.exp(-term_decay * T)
            expected = max(expected, 1e-4)
            np.testing.assert_allclose(shifted_np[i, j], expected, atol=1e-5)


def test_stress_portfolio_basic():
    """Test stress_portfolio on a simple portfolio with a Call and a Put option."""
    portfolio = [
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": 1.0, "notional": 100.0},
        {"K": 100.0, "T": 0.5, "type": "put", "quantity": -1.0, "notional": 100.0}
    ]

    # No shift baseline
    res_base = stress_portfolio(
        positions=portfolio, S0=S0, r=r, T_grid=T_GRID, K_grid=K_GRID, iv_surface=IV_SURFACE
    )

    assert res_base["baseline_price"] == res_base["stressed_price"]
    assert res_base["portfolio_pnl"] == 0.0

    # Call-Put parity check:
    # Portfolio of 1 Long Call and 1 Short Put with same strike/maturity has value:
    # V = C - P = S - K * exp(-r * T)
    expected_value = S0 - 100.0 * np.exp(-r * 0.5)
    # Scaled by quantity (1.0) and notional (100.0)
    expected_portfolio_value = expected_value * 1.0 * 100.0

    np.testing.assert_allclose(res_base["baseline_price"], expected_portfolio_value, rtol=1e-5)

    # Let's apply a spot shift only (+10%)
    res_spot_shift = stress_portfolio(
        positions=portfolio, S0=S0, r=r, T_grid=T_GRID, K_grid=K_GRID, iv_surface=IV_SURFACE,
        spot_shift=0.10
    )

    # Stressed spot = 110.0
    # Stressed value = C - P = 110.0 - 100.0 * exp(-r * T)
    expected_stressed_value = (110.0 - 100.0 * np.exp(-r * 0.5)) * 1.0 * 100.0
    np.testing.assert_allclose(res_spot_shift["stressed_price"], expected_stressed_value, rtol=1e-5)
    np.testing.assert_allclose(res_spot_shift["portfolio_pnl"], 10.0 * 1.0 * 100.0, rtol=1e-5)


def test_stress_portfolio_out_of_bounds():
    """Verify that pathological options are filtered out or handle cleanly."""
    portfolio = [
        {"K": -10.0, "T": 0.5, "type": "call", "quantity": 1.0},  # Negative strike (invalid)
        {"K": 100.0, "T": -0.1, "type": "call", "quantity": 1.0}, # Negative maturity (invalid)
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": float('nan')}, # NaN quantity (invalid)
    ]

    res = stress_portfolio(
        positions=portfolio, S0=S0, r=r, T_grid=T_GRID, K_grid=K_GRID, iv_surface=IV_SURFACE
    )

    assert res["baseline_price"] == 0.0
    assert res["stressed_price"] == 0.0
    assert res["portfolio_pnl"] == 0.0


def test_generate_stress_grid_vs_loop():
    """Verify that vectorized generate_stress_grid matches iterative loop pricing."""
    portfolio = [
        {"K": 90.0, "T": 0.2, "type": "call", "quantity": 2.5, "notional": 100.0},
        {"K": 105.0, "T": 0.8, "type": "put", "quantity": -1.5, "notional": 100.0},
        {"K": 100.0, "T": 1.5, "type": "call", "quantity": 0.8, "notional": 50.0}
    ]

    spot_shifts = [-0.30, -0.15, 0.0, 0.15, 0.30]
    vol_shifts = [-0.10, 0.0, 0.20, 0.50]

    # Generate 2D grid using the vectorized method
    grid_pnl, baseline_price = generate_stress_grid(
        positions=portfolio,
        S0=S0,
        r=r,
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE,
        spot_shifts=spot_shifts,
        vol_shifts=vol_shifts
    )

    assert grid_pnl.shape == (len(spot_shifts), len(vol_shifts))

    # Center of the grid (where spot_shift=0.0 and vol_shift=0.0) should be 0.0
    np.testing.assert_allclose(grid_pnl[2, 1], 0.0, atol=1e-5)

    # Compute grid elements manually using stress_portfolio loop
    for i, s_shift in enumerate(spot_shifts):
        for j, v_shift in enumerate(vol_shifts):
            loop_res = stress_portfolio(
                positions=portfolio,
                S0=S0,
                r=r,
                T_grid=T_GRID,
                K_grid=K_GRID,
                iv_surface=IV_SURFACE,
                spot_shift=s_shift,
                flat_shift=v_shift
            )
            # Baseline price must match
            np.testing.assert_allclose(baseline_price, loop_res["baseline_price"], rtol=1e-5)
            # P&L must match
            np.testing.assert_allclose(grid_pnl[i, j], loop_res["portfolio_pnl"], rtol=1e-4)


def test_tester_class_historical_replay():
    """Verify that OptionPortfolioStressTester wrapper works as expected with historical replays."""
    portfolio = [
        {"K": 95.0, "T": 0.5, "type": "call", "quantity": 1.0},
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": -2.0},
        {"K": 105.0, "T": 0.5, "type": "call", "quantity": 1.0}
    ]

    tester = OptionPortfolioStressTester(
        positions=portfolio,
        S0=S0,
        r=r,
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE
    )

    for name in HISTORICAL_SCENARIOS.keys():
        res = tester.historical_replay(name)
        assert res["scenario_name"] == name
        assert isinstance(res["description"], str)
        assert np.isfinite(res["baseline_price"])
        assert np.isfinite(res["stressed_price"])
        assert np.isfinite(res["portfolio_pnl"])


def test_extreme_stress_testing_stability():
    """Ensure stability and bounded values under extreme stress (up to 50% spot crash, 100% vol spike)."""
    portfolio = [
        {"K": 50.0, "T": 0.01, "type": "call", "quantity": 1e6},  # Deep ITM short maturity
        {"K": 150.0, "T": 2.0, "type": "put", "quantity": -1e6},  # Deep OTM long maturity
        {"K": 1e-3, "T": 1e-4, "type": "call", "quantity": 1.0},  # Pathological tiny strike/maturity
        {"K": 1e6, "T": 10.0, "type": "put", "quantity": 1.0}     # Pathological huge strike
    ]

    tester = OptionPortfolioStressTester(
        positions=portfolio,
        S0=S0,
        r=r,
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE
    )

    # Extreme Shifts
    res = tester.stress_scenario(
        spot_shift=-0.50,   # -50% spot crash
        flat_shift=1.00,    # +100% flat vol spike
        skew_shift=-0.20,
        term_shift=0.50
    )

    assert np.isfinite(res["baseline_price"])
    assert np.isfinite(res["stressed_price"])
    assert np.isfinite(res["portfolio_pnl"])

    # Verify that nothing returns NaN or Inf
    assert not np.isnan(res["portfolio_pnl"])
    assert not np.isinf(res["portfolio_pnl"])


def test_apply_surface_shifts_sota():
    """Verify that SOTA surface shifts (Eq. 18) are applied correctly on NumPy arrays."""
    flat_shift = 0.20
    skew_shift = 0.05
    term_decay = 1.5
    skew_steepness = 2.5
    skew_decay = 0.8

    shifted = apply_surface_shifts(
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE,
        flat_shift=flat_shift,
        skew_shift=skew_shift,
        term_decay=term_decay,
        sota_mode=True,
        skew_steepness=skew_steepness,
        skew_decay=skew_decay,
        min_vol=1e-4
    )

    assert shifted.shape == IV_SURFACE.shape
    assert isinstance(shifted, np.ndarray)

    # Verify mathematics for Eq 18
    for i, T in enumerate(T_GRID):
        for j, k in enumerate(K_GRID):
            d_flat = flat_shift * np.exp(-term_decay * T)
            d_skew = skew_shift * np.tanh(-skew_steepness * k) * (1.0 - np.sign(k)) * np.exp(-skew_decay * T)
            expected = max(BASE_VOL + d_flat + d_skew, 1e-4)
            np.testing.assert_allclose(shifted[i, j], expected, atol=1e-5)


def test_stress_portfolio_sota():
    """Test stress_portfolio under SOTA shift mode."""
    portfolio = [
        {"K": 95.0, "T": 0.5, "type": "call", "quantity": 1.0, "notional": 100.0},
        {"K": 105.0, "T": 0.5, "type": "put", "quantity": -1.0, "notional": 100.0}
    ]

    res = stress_portfolio(
        positions=portfolio,
        S0=S0,
        r=r,
        T_grid=T_GRID,
        K_grid=K_GRID,
        iv_surface=IV_SURFACE,
        spot_shift=-0.20,
        flat_shift=0.30,
        skew_shift=0.10,
        sota_mode=True
    )

    assert np.isfinite(res["baseline_price"])
    assert np.isfinite(res["stressed_price"])
    assert np.isfinite(res["portfolio_pnl"])


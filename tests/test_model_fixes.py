import pytest
import numpy as np
import torch
import math
from deepvol.models.rbergomi_gpu import rBergomiEngine
from deepvol.models.bachelier import (
    bachelier_price,
    black_price,
    shifted_black_price,
    bachelier_implied_vol,
    black_implied_vol
)
from deepvol.models.heston import HestonEngine

def test_rbergomi_engine_simulate_paths():
    """Verify that the fixed rBergomiEngine.simulate_paths correctly aligns with simulate_rbergomi_paths."""
    engine = rBergomiEngine()
    
    # [v0, H, eta, rho]
    params = torch.tensor([[0.04, 0.1, 1.5, -0.7]], dtype=torch.float32)
    T = 0.5
    steps_per_unit = 100
    N_paths = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    S, V, t_grid = engine.simulate_paths(
        params=params,
        T=T,
        steps_per_unit=steps_per_unit,
        N_paths=N_paths,
        antithetic=True,
        device=device,
        dtype=torch.float32
    )
    
    # 0.5 * 100 = 50 steps, grid size is 51
    expected_steps = 51
    assert S.shape == (1, N_paths, expected_steps)
    assert V.shape == (1, N_paths, expected_steps)
    assert t_grid.shape == (expected_steps,)
    
    # Initial S_0 must be 1.0 (log stock starts at 0, S = exp(x))
    np.testing.assert_allclose(S[0, :, 0].cpu().numpy(), 1.0, atol=1e-6)
    # Initial V_0 must be v0 = 0.04
    np.testing.assert_allclose(V[0, :, 0].cpu().numpy(), 0.04, atol=1e-6)


def test_bachelier_pricing_boundaries():
    """Verify boundary cases for CPU Bachelier pricing."""
    # 1. Zero volatility should return intrinsic value
    assert bachelier_price(100.0, 100.0, 0.5, 0.0, option_type='call') == 0.0
    assert bachelier_price(100.0, 95.0, 0.5, 0.0, option_type='call') == 5.0
    assert bachelier_price(100.0, 105.0, 0.5, 0.0, option_type='put') == 5.0
    
    # 2. Zero time-to-maturity should return intrinsic value
    assert bachelier_price(100.0, 100.0, 0.0, 15.0, option_type='call') == 0.0
    assert bachelier_price(100.0, 95.0, 0.0, 15.0, option_type='call') == 5.0
    
    # 3. Negative maturity or negative vol should return NaN
    assert np.isnan(bachelier_price(100.0, 100.0, -0.5, 10.0))
    assert np.isnan(bachelier_price(100.0, 100.0, 0.5, -10.0))
    
    # 4. ATM call option formula: sigma * sqrt(T) / sqrt(2*pi)
    price_atm = bachelier_price(100.0, 100.0, 1.0, 10.0, option_type='call')
    expected_atm = 10.0 * 1.0 / np.sqrt(2.0 * np.pi)
    assert np.isclose(price_atm, expected_atm, atol=1e-6)


def test_black_pricing_boundaries():
    """Verify boundary cases for CPU Black pricing."""
    # 1. Zero volatility should return intrinsic value
    assert black_price(100.0, 100.0, 0.5, 0.0, option_type='call') == 0.0
    assert black_price(100.0, 95.0, 0.5, 0.0, option_type='call') == 5.0
    assert black_price(100.0, 105.0, 0.5, 0.0, option_type='put') == 5.0
    
    # 2. Zero time-to-maturity should return intrinsic value
    assert black_price(100.0, 100.0, 0.0, 0.2, option_type='call') == 0.0
    assert black_price(100.0, 95.0, 0.0, 0.2, option_type='call') == 5.0
    
    # 3. Negative parameters should return NaN
    assert np.isnan(black_price(100.0, 100.0, -0.5, 0.2))
    assert np.isnan(black_price(100.0, 100.0, 0.5, -0.2))
    assert np.isnan(black_price(-100.0, 100.0, 0.5, 0.2))
    
    # 4. Shifted Black pricing
    price_shifted = shifted_black_price(0.02, 0.02, 1.0, 0.20, shift=0.03, option_type='call')
    # This is equivalent to Black price with forward = 0.05, strike = 0.05
    price_ref = black_price(0.05, 0.05, 1.0, 0.20, option_type='call')
    assert np.isclose(price_shifted, price_ref, atol=1e-6)


def test_bachelier_implied_vol_boundaries():
    """Verify boundary cases for Bachelier implied volatility solver."""
    # 1. Below intrinsic price should return NaN (call intrinsic is 5.0, price is 4.0)
    assert np.isnan(bachelier_implied_vol(4.0, 100.0, 95.0, 0.5, option_type='call'))
    
    # 2. At or near intrinsic should return 0.0
    assert bachelier_implied_vol(5.0, 100.0, 95.0, 0.5, option_type='call') == 0.0
    
    # 3. ATM option should recover the volatility exactly
    vol_target = 15.0
    price = bachelier_price(100.0, 100.0, 0.5, vol_target, option_type='call')
    vol_solved = bachelier_implied_vol(price, 100.0, 100.0, 0.5, option_type='call')
    assert np.isclose(vol_solved, vol_target, atol=1e-6)


def test_black_implied_vol_boundaries():
    """Verify boundary cases for Black implied volatility solver."""
    # 1. Below intrinsic price should return NaN (call intrinsic is 5.0, price is 4.0)
    assert np.isnan(black_implied_vol(4.0, 100.0, 95.0, 0.5, option_type='call'))
    
    # 2. At or near intrinsic should return 0.0
    assert black_implied_vol(5.0, 100.0, 95.0, 0.5, option_type='call') == 0.0
    
    # 3. ATM option should recover the volatility exactly
    vol_target = 0.25
    price = black_price(100.0, 100.0, 0.5, vol_target, option_type='call')
    vol_solved = black_implied_vol(price, 100.0, 100.0, 0.5, option_type='call')
    assert np.isclose(vol_solved, vol_target, atol=1e-6)

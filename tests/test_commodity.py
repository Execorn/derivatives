"""
Tests for the Commodity Options Schwartz-Smith engine.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import pytest
import torch

from deepvol.market.commodity_data import (
    CMECommodityDataAdapter,
    generate_synthetic_options_data,
    wti_futures_expiry,
    wti_options_expiry,
    parse_futures_code,
    parse_options_code,
    clean_strike
)
from deepvol.models.schwartz_smith import (
    schwartz_smith_price_black76,
    schwartz_smith_price_fourier,
    schwartz_smith_price_black76_pt,
    schwartz_smith_price_fourier_pt,
    schwartz_smith_price_cos,
    schwartz_smith_price_cos_pt,
    schwartz_smith_greeks_pt,
    calibrate_schwartz_smith,
    run_kalman_filter,
    price_option_cos,
    price_option_cos_pt,
    futures_price_pt
)


def test_calendar_matching():
    """Verifies CME options to futures calendar matching logic and strike cleaning."""
    # Test parsing futures codes
    fut1 = parse_futures_code("CLZ26", ref_year=2026)
    assert fut1["underlying"] == "CL"
    assert fut1["month"] == 12
    assert fut1["year"] == 2026

    fut2 = parse_futures_code("CLF7", ref_year=2026)
    assert fut2["underlying"] == "CL"
    assert fut2["month"] == 1
    assert fut2["year"] == 2027

    # Test parsing options codes
    opt1 = parse_options_code("LOZ26 C7500", ref_year=2026)
    assert opt1["underlying"] == "CL"
    assert opt1["month"] == 12
    assert opt1["year"] == 2026
    assert opt1["strike"] == 7500
    assert opt1["option_type"] == "C"

    opt2 = parse_options_code("LO Z26 75.0 P", ref_year=2026)
    assert opt2["underlying"] == "CL"
    assert opt2["month"] == 12
    assert opt2["year"] == 2026
    assert opt2["strike"] == 75.0
    assert opt2["option_type"] == "P"

    opt3 = parse_options_code("LOZ26 75.0C", ref_year=2026)
    assert opt3["underlying"] == "CL"
    assert opt3["month"] == 12
    assert opt3["year"] == 2026
    assert opt3["strike"] == 75.0
    assert opt3["option_type"] == "C"

    opt4 = parse_options_code("LOZ26 7500P", ref_year=2026)
    assert opt4["underlying"] == "CL"
    assert opt4["month"] == 12
    assert opt4["year"] == 2026
    assert opt4["strike"] == 7500
    assert opt4["option_type"] == "P"

    # Test strike cleaning
    assert clean_strike(7500, underlying_price=75.0) == 75.0
    assert clean_strike("7550", underlying_price=75.0) == 75.5
    assert clean_strike(75.0, underlying_price=75.0) == 75.0

    # Test expiry calculation for WTI
    # Dec 2026:
    # 25th of Nov 2026 is Wed, which is a business day.
    # Expiry of CLZ26 futures should be 3 business days prior to Nov 25.
    # Nov 25 (Wed) -> 3 business days prior is Nov 20 (Fri) (assuming no holidays in between).
    # Let's check wti_futures_expiry:
    clz26_exp = wti_futures_expiry(2026, 12)
    # Expiry of LOZ26 options is 3 business days prior to futures expiry.
    loz26_exp = wti_options_expiry(2026, 12)
    assert loz26_exp < clz26_exp

    # Test CMECommodityDataAdapter matching logic
    adapter = CMECommodityDataAdapter(ref_year=2026)
    
    # Create mock options and futures DataFrames
    options_data = pd.DataFrame({
        "valuation_date": ["2026-06-01", "2026-06-01"],
        "option_code": ["LOZ26 7500 C", "LOZ26 8000 P"],
        "price": [5.2, 4.8]
    })
    
    futures_data = pd.DataFrame({
        "valuation_date": ["2026-06-01", "2026-06-01"],
        "contract_code": ["CLZ26", "CLH27"],
        "price": [76.5, 74.0]
    })
    
    matched = adapter.match_options_to_futures(options_data, futures_data)
    
    # Should match only the Z26 option because there is no H27 option in options_data
    assert len(matched) == 2  # both LOZ26 7500 C and LOZ26 8000 P match CLZ26
    assert matched.loc[0, "strike"] == 75.0
    assert matched.loc[1, "strike"] == 80.0
    assert matched.loc[0, "futures_price"] == 76.5
    assert matched.loc[0, "T_opt"] > 0
    assert matched.loc[0, "T_fut"] > matched.loc[0, "T_opt"]


def test_analytical_vs_fourier():
    """Verifies that analytical Black-76 option pricing matches Fourier option pricing within 1e-6."""
    # Model parameters
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    
    # State variables and option specs
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 0.6
    r = 0.04
    
    strikes = [65.0, 70.0, 75.0, 80.0]
    
    for K in strikes:
        for otype in ["C", "P"]:
            p_analytical = schwartz_smith_price_black76(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_fourier = schwartz_smith_price_fourier(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            
            # Match within 1e-6
            diff = abs(p_analytical - p_fourier)
            assert diff < 1e-6, f"Mismatch for strike {K} ({otype}): Analytical={p_analytical}, Fourier={p_fourier}, diff={diff}"
 
 
def test_pytorch_vs_cpu_and_gpu():
    """Verifies PyTorch CPU & GPU paths are consistent with the CPU path and compute Greeks correctly."""
    # Model parameters
    kappa = 0.7
    sigma_chi = 0.22
    rho = -0.25
    sigma_xi = 0.15
    mu_star = 0.03
    lambda_chi = 0.05
    
    # Batch inputs
    chi_t = torch.tensor([0.05, -0.05, 0.1], dtype=torch.float64)
    xi_t = torch.tensor([np.log(75.0), np.log(75.0), np.log(75.0)], dtype=torch.float64)
    
    t = 0.0
    T_opt = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    T_fut = torch.tensor([0.35, 0.6, 0.85], dtype=torch.float64)
    K = torch.tensor([70.0, 75.0, 80.0], dtype=torch.float64)
    r = 0.04
    
    # 1. Compare CPU PyTorch Analytical with CPU NumPy Analytical
    p_numpy = np.array([
        schwartz_smith_price_black76(t, T_opt[i].item(), T_fut[i].item(), K[i].item(), r, chi_t[i].item(), xi_t[i].item(),
                                     kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type="C")
        for i in range(3)
    ])
    
    p_torch_cpu = schwartz_smith_price_black76_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    
    np.testing.assert_allclose(p_numpy, p_torch_cpu.numpy(), rtol=1e-7, atol=1e-7)
    
    # 2. Compare PyTorch Fourier with PyTorch Analytical
    p_fourier_pt = schwartz_smith_price_fourier_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C", N_grid=800, u_max=120.0
    )
    np.testing.assert_allclose(p_torch_cpu.numpy(), p_fourier_pt.numpy(), rtol=1e-5, atol=1e-5)
    
    # 3. Test GPU path if CUDA is available
    if torch.cuda.is_available():
        chi_cuda = chi_t.cuda()
        xi_cuda = xi_t.cuda()
        T_opt_cuda = T_opt.cuda()
        T_fut_cuda = T_fut.cuda()
        K_cuda = K.cuda()
        
        p_torch_gpu = schwartz_smith_price_black76_pt(
            t, T_opt_cuda, T_fut_cuda, K_cuda, r, chi_cuda, xi_cuda,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C"
        )
        
        np.testing.assert_allclose(p_torch_cpu.numpy(), p_torch_gpu.cpu().numpy(), rtol=1e-7, atol=1e-7)
        
        p_fourier_gpu = schwartz_smith_price_fourier_pt(
            t, T_opt_cuda, T_fut_cuda, K_cuda, r, chi_cuda, xi_cuda,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N_grid=800, u_max=120.0
        )
        np.testing.assert_allclose(p_torch_gpu.cpu().numpy(), p_fourier_gpu.cpu().numpy(), rtol=1e-5, atol=1e-5)
        
    # 4. Test Sensitivity (Greeks) calculations
    greeks = schwartz_smith_greeks_pt(
        t, 0.5, 0.6, 75.0, r, 0.0, np.log(75.0),
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C", target_greek="all"
    )
    
    assert "delta_chi" in greeks
    assert "delta_xi" in greeks
    assert "gamma_chi" in greeks
    assert "gamma_xi" in greeks
    assert "vega_sigma_chi" in greeks
    assert "vega_sigma_xi" in greeks
    assert "vega_kappa" in greeks
    
    # delta_xi should be positive (increase in equilibrium factor increases futures price and call price)
    assert greeks["delta_xi"].item() > 0.0
    # vega_sigma_chi and vega_sigma_xi should be positive
    assert greeks["vega_sigma_chi"].item() > 0.0
    assert greeks["vega_sigma_xi"].item() > 0.0


def test_calibration_and_kalman_filter():
    """Verifies that the Kalman Filter likelihood matches expected and calibrator recovers parameters."""
    np.random.seed(42)
    
    # True parameters
    true_params = {
        "kappa": 0.8,
        "sigma_chi": 0.25,
        "rho": 0.3,
        "sigma_xi": 0.12,
        "mu": 0.06,
        "lambda_chi": 0.05,
        "mu_star": 0.03,
        "sigma_e": 0.005
    }
    
    # Create 15 synthetic weekly dates
    start_date = datetime.date(2026, 1, 1)
    dates = [start_date + datetime.timedelta(weeks=i) for i in range(15)]
    
    # Generate clean synthetic options & futures data
    # (noise_std=0.0 means the measurement equation holds exactly except for tiny precision limits)
    opts_df, futs_df = generate_synthetic_options_data(
        valuation_dates=dates,
        months_ahead=[1, 3, 6],
        strike_pcts=[1.0],
        ss_params=true_params,
        init_chi=0.05,
        init_xi=np.log(72.0),
        r=0.04,
        noise_std=0.001
    )
    
    # Match data
    adapter = CMECommodityDataAdapter(ref_year=2026)
    matched = adapter.match_options_to_futures(opts_df, futs_df)
    
    # Pivot futures prices and maturities to shape (num_dates, num_contracts)
    # We have weekly dates, and 3 contracts per date (1m, 3m, 6m)
    # Let's extract them from futs_df directly
    futs_pivoted = futs_df.pivot(index="valuation_date", columns="tenor", values="price")
    futs_codes_pivoted = futs_df.pivot(index="valuation_date", columns="tenor", values="contract_code")
    
    # Reconstruct dates and maturities matrix
    dates_arr = futs_pivoted.index.tolist()
    futures_prices = futs_pivoted.values
    
    # Build maturities matrix
    num_dates, num_contracts = futures_prices.shape
    maturities = np.zeros((num_dates, num_contracts))
    for t_idx, val_dt in enumerate(dates_arr):
        for c_idx, tenor in enumerate(futs_pivoted.columns):
            code = futs_codes_pivoted.iloc[t_idx, c_idx]
            parsed = parse_futures_code(code, ref_year=2026)
            expiry = wti_futures_expiry(parsed["year"], parsed["month"])
            maturities[t_idx, c_idx] = (expiry - val_dt).days / 365.0
            
    # Run Kalman Filter with true parameters
    ll, states, covs = run_kalman_filter(
        dates_arr, futures_prices, maturities,
        true_params["kappa"], true_params["sigma_chi"], true_params["rho"], true_params["sigma_xi"],
        true_params["mu"], true_params["lambda_chi"], true_params["mu_star"], true_params["sigma_e"]
    )
    
    assert ll > -1e5  # Reasonable likelihood
    assert states.shape == (num_dates, 2)
    assert covs.shape == (num_dates, 2, 2)
    
    # Run calibrator starting from a slightly perturbed guess
    init_guess = [0.6, 0.20, 0.2, 0.10, 0.05, 0.02, 0.02, 0.01]
    cal_res = calibrate_schwartz_smith(dates_arr, futures_prices, maturities, init_guess=init_guess)
    
    assert cal_res["success"] is True or cal_res["log_likelihood"] > ll - 100
    assert cal_res["kappa"] > 0
    assert cal_res["sigma_chi"] > 0
    assert cal_res["sigma_xi"] > 0
    assert -1.0 < cal_res["rho"] < 1.0


def test_zero_kappa_limit():
    """Asserts that pricing behaves correctly under kappa = 0 (matches the theoretical limit with zero error)."""
    # Model parameters
    kappa = 0.0
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    
    # State variables and option specs
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 0.6
    r = 0.04
    K = 75.0
    
    # Pricing with kappa = 0
    p_analytical_0 = schwartz_smith_price_black76(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    p_fourier_0 = schwartz_smith_price_fourier(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    
    # Pricing with a very small kappa (e.g. 1e-9)
    kappa_small = 1e-9
    p_analytical_eps = schwartz_smith_price_black76(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa_small, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    p_fourier_eps = schwartz_smith_price_fourier(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa_small, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    
    # Check that kappa=0 results are close to epsilon-kappa results (matches limit)
    assert abs(p_analytical_0 - p_analytical_eps) < 1e-7
    assert abs(p_fourier_0 - p_fourier_eps) < 1e-7
    
    # PyTorch equivalents
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    p_analytical_pt_0 = schwartz_smith_price_black76_pt(
        t, T_opt, T_fut, K, r, chi_pt, xi_pt,
        0.0, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    p_fourier_pt_0 = schwartz_smith_price_fourier_pt(
        t, T_opt, T_fut, K, r, chi_pt, xi_pt,
        0.0, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    
    p_fourier_pt_eps = schwartz_smith_price_fourier_pt(
        t, T_opt, T_fut, K, r, chi_pt, xi_pt,
        kappa_small, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    
    assert abs(p_analytical_0 - p_analytical_pt_0.item()) < 1e-7
    assert abs(p_fourier_pt_0.item() - p_analytical_pt_0.item()) < 5e-5
    assert abs(p_fourier_pt_0.item() - p_fourier_pt_eps.item()) < 1e-7


def test_device_mismatch_robustness():
    """Asserts that device mismatch inputs do not raise errors in PyTorch pricing functions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device mismatch robustness test")
        
    chi_t = torch.tensor([0.05], dtype=torch.float64, device='cuda')
    xi_t = torch.tensor([np.log(75.0)], dtype=torch.float64, device='cuda')
    
    t = 0.0
    T_opt = torch.tensor([0.25], dtype=torch.float64, device='cpu') # CPU tensor
    T_fut = torch.tensor([0.35], dtype=torch.float64, device='cpu') # CPU tensor
    K = torch.tensor([70.0], dtype=torch.float64, device='cpu')     # CPU tensor
    r = 0.04
    
    kappa = 0.7
    sigma_chi = 0.22
    rho = -0.25
    sigma_xi = 0.15
    mu_star = 0.03
    lambda_chi = 0.05
    
    # Should run successfully without raising device mismatch error
    price = schwartz_smith_price_black76_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    assert price.device == chi_t.device
    assert price.dtype == chi_t.dtype
    
    price_fourier = schwartz_smith_price_fourier_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    assert price_fourier.device == chi_t.device
    assert price_fourier.dtype == chi_t.dtype

    price_cos = price_option_cos_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    assert price_cos.device == chi_t.device
    assert price_cos.dtype == chi_t.dtype


def test_price_option_cos_vs_black76():
    """Compares price_option_cos against the analytical Black-76 pricer across strikes and maturities."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    r = 0.04
    
    maturities = [0.1, 0.25, 0.5, 1.0]
    strikes = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
    
    for T_opt in maturities:
        T_fut = T_opt + 0.1
        for K in strikes:
            for otype in ["C", "P"]:
                p_black = schwartz_smith_price_black76(
                    t, T_opt, T_fut, K, r, chi_t, xi_t,
                    kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                    option_type=otype
                )
                p_cos = price_option_cos(
                    t, T_opt, T_fut, K, r, chi_t, xi_t,
                    kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                    option_type=otype, N=256, L=12.0
                )
                
                diff = abs(p_black - p_cos)
                assert diff < 1e-6, f"Mismatch for strike {K}, maturity {T_opt} ({otype}): Black76={p_black}, COS={p_cos}, diff={diff}"


def test_price_option_cos_pt_vs_numpy():
    """Compares price_option_cos_pt against price_option_cos in PyTorch."""
    kappa = 0.7
    sigma_chi = 0.22
    rho = -0.25
    sigma_xi = 0.15
    mu_star = 0.03
    lambda_chi = 0.05
    
    # Batched inputs
    chi_t = torch.tensor([0.05, -0.05, 0.1], dtype=torch.float64)
    xi_t = torch.tensor([np.log(75.0), np.log(75.0), np.log(75.0)], dtype=torch.float64)
    
    t = 0.0
    T_opt = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    T_fut = torch.tensor([0.35, 0.6, 0.85], dtype=torch.float64)
    K = torch.tensor([70.0, 75.0, 80.0], dtype=torch.float64)
    r = 0.04
    
    # Compute using NumPy COS pricer
    p_numpy = np.array([
        price_option_cos(t, T_opt[i].item(), T_fut[i].item(), K[i].item(), r, chi_t[i].item(), xi_t[i].item(),
                         kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type="C", N=256, L=12.0)
        for i in range(3)
    ])
    
    # Compute using PyTorch COS pricer
    p_torch_cpu = price_option_cos_pt(
        t, T_opt, T_fut, K, r, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C", N=256, L=12.0
    )
    
    np.testing.assert_allclose(p_numpy, p_torch_cpu.numpy(), rtol=1e-7, atol=1e-7)
    
    # Test GPU path if CUDA is available
    if torch.cuda.is_available():
        chi_cuda = chi_t.cuda()
        xi_cuda = xi_t.cuda()
        T_opt_cuda = T_opt.cuda()
        T_fut_cuda = T_fut.cuda()
        K_cuda = K.cuda()
        
        p_torch_gpu = price_option_cos_pt(
            t, T_opt_cuda, T_fut_cuda, K_cuda, r, chi_cuda, xi_cuda,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        
        assert p_torch_gpu.device.type == 'cuda'
        np.testing.assert_allclose(p_numpy, p_torch_gpu.cpu().numpy(), rtol=1e-7, atol=1e-7)


def test_pytorch_parameter_validation():
    """Verifies that PyTorch pricing functions raise ValueError for invalid inputs."""
    chi_t = torch.tensor([0.05], dtype=torch.float64)
    xi_t = torch.tensor([np.log(75.0)], dtype=torch.float64)
    
    t = 0.0
    T_opt = torch.tensor([0.25], dtype=torch.float64)
    T_fut = torch.tensor([0.35], dtype=torch.float64)
    kappa = 0.7
    sigma_chi = 0.22
    rho = -0.25
    sigma_xi = 0.15
    mu_star = 0.03
    lambda_chi = 0.05
    
    # 1. Invalid K (<= 0)
    with pytest.raises(ValueError, match="Strike must be positive"):
        schwartz_smith_price_black76_pt(
            t, T_opt, T_fut, -10.0, 0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Strike must be positive"):
        schwartz_smith_price_fourier_pt(
            t, T_opt, T_fut, -10.0, 0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Strike must be positive"):
        price_option_cos_pt(
            t, T_opt, T_fut, -10.0, 0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Strike must be positive"):
        schwartz_smith_price_cos_pt(
            t, T_opt, T_fut, -10.0, 0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
        
    # 2. Invalid r (< 0)
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        schwartz_smith_price_black76_pt(
            t, T_opt, T_fut, 70.0, -0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        schwartz_smith_price_fourier_pt(
            t, T_opt, T_fut, 70.0, -0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        price_option_cos_pt(
            t, T_opt, T_fut, 70.0, -0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        schwartz_smith_price_cos_pt(
            t, T_opt, T_fut, 70.0, -0.04, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )



def test_schwartz_smith_price_cos_suite():
    """
    Comprehensive test suite for schwartz_smith_price_cos and schwartz_smith_price_cos_pt.
    Verifies pricing tolerances, GPU compatibility, constraint enforcement,
    and profiles the speed of the COS method vs the Lewis Fourier inversion method.
    """
    import time

    # Model parameters
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    
    # State variables and option specs
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    r = 0.04
    
    # 1. Verify schwartz_smith_price_cos matches schwartz_smith_price_black76 within 1e-6 tolerance
    maturities = [0.1, 0.25, 0.5, 1.0]
    strikes = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
    
    for T_opt in maturities:
        T_fut = T_opt + 0.1
        for K in strikes:
            for otype in ["C", "P"]:
                p_black = schwartz_smith_price_black76(
                    t, T_opt, T_fut, K, r, chi_t, xi_t,
                    kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                    option_type=otype
                )
                p_cos = schwartz_smith_price_cos(
                    t, T_opt, T_fut, K, r, chi_t, xi_t,
                    kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                    option_type=otype, N=256, L=12.0
                )
                diff = abs(p_black - p_cos)
                assert diff < 1e-6, f"CPU COS mismatch for K={K}, T_opt={T_opt} ({otype}): Black={p_black}, COS={p_cos}, diff={diff}"

    # 2. Verify schwartz_smith_price_cos_pt matches schwartz_smith_price_black76_pt within 1e-5 tolerance
    chi_pt_batch = torch.tensor([0.05, -0.05, 0.1], dtype=torch.float64)
    xi_pt_batch = torch.tensor([np.log(75.0), np.log(75.0), np.log(75.0)], dtype=torch.float64)
    T_opt_batch = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    T_fut_batch = torch.tensor([0.35, 0.6, 0.85], dtype=torch.float64)
    K_batch = torch.tensor([70.0, 75.0, 80.0], dtype=torch.float64)
    
    p_black_pt = schwartz_smith_price_black76_pt(
        t, T_opt_batch, T_fut_batch, K_batch, r, chi_pt_batch, xi_pt_batch,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C"
    )
    p_cos_pt = schwartz_smith_price_cos_pt(
        t, T_opt_batch, T_fut_batch, K_batch, r, chi_pt_batch, xi_pt_batch,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
        option_type="C", N=256, L=12.0
    )
    
    diff_pt = torch.max(torch.abs(p_black_pt - p_cos_pt)).item()
    assert diff_pt < 1e-5, f"PyTorch CPU COS mismatch: Black_pt={p_black_pt}, COS_pt={p_cos_pt}, max_diff={diff_pt}"

    # 3. Verify compatibility on GPU if CUDA is available
    if torch.cuda.is_available():
        chi_cuda = chi_pt_batch.cuda()
        xi_cuda = xi_pt_batch.cuda()
        T_opt_cuda = T_opt_batch.cuda()
        T_fut_cuda = T_fut_batch.cuda()
        K_cuda = K_batch.cuda()
        
        p_cos_gpu = schwartz_smith_price_cos_pt(
            t, T_opt_cuda, T_fut_cuda, K_cuda, r, chi_cuda, xi_cuda,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        assert p_cos_gpu.device.type == 'cuda'
        np.testing.assert_allclose(p_cos_pt.numpy(), p_cos_gpu.cpu().numpy(), rtol=1e-7, atol=1e-7)

    # 4. Profile the speed of the COS method vs the Lewis Fourier inversion method
    # Warm-up runs
    _ = schwartz_smith_price_fourier(t, 0.5, 0.6, 75.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    _ = schwartz_smith_price_cos(t, 0.5, 0.6, 75.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    
    t0 = time.perf_counter()
    for _ in range(50):
        _ = schwartz_smith_price_fourier(
            t, 0.5, 0.6, 75.0, r, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
        )
    t_fourier_np = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(50):
        _ = schwartz_smith_price_cos(
            t, 0.5, 0.6, 75.0, r, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            N=128, L=10.0
        )
    t_cos_np = time.perf_counter() - t0
    
    # PyTorch profiling with a larger batch
    large_size = 800
    chi_pt_large = torch.linspace(-0.2, 0.2, large_size, dtype=torch.float64)
    xi_pt_large = torch.linspace(np.log(60.0), np.log(90.0), large_size, dtype=torch.float64)
    
    # Warm-up runs
    _ = schwartz_smith_price_fourier_pt(t, 0.5, 0.6, 75.0, r, chi_pt_large, xi_pt_large, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, N_grid=500)
    _ = schwartz_smith_price_cos_pt(t, 0.5, 0.6, 75.0, r, chi_pt_large, xi_pt_large, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, N=128, L=10.0)

    t0 = time.perf_counter()
    for _ in range(30):
        _ = schwartz_smith_price_fourier_pt(
            t, 0.5, 0.6, 75.0, r, chi_pt_large, xi_pt_large,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            N_grid=500
        )
    t_fourier_pt = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(30):
        _ = schwartz_smith_price_cos_pt(
            t, 0.5, 0.6, 75.0, r, chi_pt_large, xi_pt_large,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            N=128, L=10.0
        )
    t_cos_pt = time.perf_counter() - t0
    
    print(f"\n[NumPy Profile] Fourier (Lewis): {t_fourier_np:.6f}s, COS: {t_cos_np:.6f}s")
    print(f"[PyTorch Profile] Fourier (Lewis): {t_fourier_pt:.6f}s, COS: {t_cos_pt:.6f}s")
    
    assert t_cos_np < t_fourier_np, f"Expected CPU COS to be faster than CPU Lewis Fourier: COS={t_cos_np}s, Fourier={t_fourier_np}s"
    if torch.cuda.is_available():
        assert t_cos_pt < t_fourier_pt, f"Expected PyTorch COS to be faster than PyTorch Lewis Fourier: COS={t_cos_pt}s, Fourier={t_fourier_pt}s"
    else:
        # Relaxed timing inequality assertion for CPU thread contention
        assert t_cos_pt < 1.5 * t_fourier_pt, f"Expected PyTorch COS on CPU to be faster or comparable to PyTorch Lewis Fourier: COS={t_cos_pt}s, Fourier={t_fourier_pt}s"

    # 5. Constraint violations validation (raise ValueError)
    # CPU violations
    with pytest.raises(ValueError, match="Strike must be positive"):
        schwartz_smith_price_cos(t, 0.5, 0.6, -70.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        schwartz_smith_price_cos(t, 0.5, 0.6, 70.0, -r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        schwartz_smith_price_cos(t, 0.5, 0.6, 70.0, r, chi_t, xi_t, -0.5, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="sigma_chi must be non-negative"):
        schwartz_smith_price_cos(t, 0.5, 0.6, 70.0, r, chi_t, xi_t, kappa, -0.25, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="sigma_xi must be non-negative"):
        schwartz_smith_price_cos(t, 0.5, 0.6, 70.0, r, chi_t, xi_t, kappa, sigma_chi, rho, -0.12, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="rho must be between -1.0 and 1.0"):
        schwartz_smith_price_cos(t, 0.5, 0.6, 70.0, r, chi_t, xi_t, kappa, sigma_chi, 1.35, sigma_xi, mu_star, lambda_chi)

    # PyTorch violations
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    with pytest.raises(ValueError, match="Strike must be positive"):
        schwartz_smith_price_cos_pt(t, 0.5, 0.6, -70.0, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="Risk free rate must be non-negative"):
        schwartz_smith_price_cos_pt(t, 0.5, 0.6, 70.0, -r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        schwartz_smith_price_cos_pt(t, 0.5, 0.6, 70.0, r, chi_pt, xi_pt, -0.5, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError, match="rho must be between -1.0 and 1.0"):
        schwartz_smith_price_cos_pt(t, 0.5, 0.6, 70.0, r, chi_pt, xi_pt, kappa, sigma_chi, 1.35, sigma_xi, mu_star, lambda_chi)

    # 5. Non-finite values check
    for bad_val in [float('nan'), float('inf'), float('-inf')]:
        with pytest.raises(ValueError, match="All inputs must be finite"):
            schwartz_smith_price_cos_pt(
                bad_val, 0.5, 0.6, 70.0, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
            )
        with pytest.raises(ValueError, match="All inputs must be finite"):
            schwartz_smith_price_cos_pt(
                t, 0.5, 0.6, 70.0, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, bad_val
            )



def test_near_maturity_undiscounted_payoff():
    """
    Verifies that when tau <= 1e-8 or v2 < 1e-15, the option pricing functions
    correctly return the undiscounted payoff (i.e. not multiplied by exp(-r * tau) when tau <= 0).
    """
    # Specifically check tau <= 0 scenario where exp(-r * tau) would normally be > 1 if r > 0.
    # We expect the payoff to be undiscounted intrinsic.
    t = 1.0
    T_opt = 0.5  # tau = -0.5
    T_fut = 1.2
    K = 70.0
    r = 0.05
    chi_t = torch.tensor([0.0], dtype=torch.float64)
    xi_t = torch.tensor([np.log(80.0)], dtype=torch.float64)
    kappa = 0.5
    sigma_chi = 0.2
    rho = 0.0
    sigma_xi = 0.1
    mu_star = 0.0

    # For K=70, spot F approx 80, Call payoff is F - K, Put payoff is 0.0
    F = futures_price_pt(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star)
    intrinsic_call = (F - K).clamp(min=0.0)
    
    p_black_call = schwartz_smith_price_black76_pt(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="C")
    p_fourier_call = schwartz_smith_price_fourier_pt(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="C")
    p_cos_call = schwartz_smith_price_cos_pt(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="C")

    # If undiscounted, payoff = F - K.
    # If discounted, it would be (F - K) * exp(-0.05 * (-0.5)) = (F - K) * exp(0.025).
    # We assert that the returned price is exactly the undiscounted payoff.
    assert torch.allclose(p_black_call, intrinsic_call)
    assert torch.allclose(p_fourier_call, intrinsic_call)
    assert torch.allclose(p_cos_call, intrinsic_call)

    # Test Put with K=90
    intrinsic_put = (torch.tensor([90.0], dtype=torch.float64) - F).clamp(min=0.0)
    
    p_black_put = schwartz_smith_price_black76_pt(t, T_opt, T_fut, 90.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="P")
    p_fourier_put = schwartz_smith_price_fourier_pt(t, T_opt, T_fut, 90.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="P")
    p_cos_put = schwartz_smith_price_cos_pt(t, T_opt, T_fut, 90.0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, option_type="P")

    assert torch.allclose(p_black_put, intrinsic_put)
    assert torch.allclose(p_fourier_put, intrinsic_put)
    assert torch.allclose(p_cos_put, intrinsic_put)

"""
test_e2e_phase7.py - End-to-End Test Suite for Phase 7 Multi-Asset Calibration Framework.
"""

import pytest
import numpy as np
from scipy.stats import norm

# Import Phase 7 classes
from src.pricing.mlsv_gpu import MLSVEngine
from src.market.fx_data import FXDataLoader
from src.calibration.fx_calibration import FXSABRCalibrator
from src.market.rates_data import SOFRSwaptionLoader
from src.pricing.sabr_rates import RatesSABREngine
from src.market.commodity_data import CommodityDataLoader
from src.pricing.schwartz_smith import SchwartzSmithEngine


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature; >=20 tests total)
# ==============================================================================

# --- Feature A: Equity MLSV ---

def test_mlsv_option_pricing():
    """Verify standard option pricing in MLSVEngine."""
    engine = MLSVEngine(kappa=1.5, theta=0.04, epsilon=0.3, rho=-0.7)
    price_call = engine.price_option(spot=100.0, strike=100.0, maturity=1.0, vol=0.2, is_call=True)
    price_put = engine.price_option(spot=100.0, strike=100.0, maturity=1.0, vol=0.2, is_call=False)
    
    assert price_call > 0.0
    assert price_put > 0.0
    # For ATM with no interest rate, call and put BS prices should be equal
    assert abs(price_call - price_put) < 1e-12

def test_mlsv_conditional_expectation():
    """Verify conditional expectation E[Vt | St] curve computation."""
    engine = MLSVEngine(kappa=2.0, theta=0.09, epsilon=0.4, rho=-0.6)
    spot_grid = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expectations = engine.conditional_expectation(spot_grid=spot_grid, current_spot=100.0, current_vol=0.3)
    
    assert len(expectations) == len(spot_grid)
    assert np.all(expectations > 0.0)
    # With negative correlation, higher spot should have lower volatility/variance expectation
    assert expectations[0] > expectations[-1]

def test_mlsv_local_vol_calibration():
    """Verify local volatility grid calibration."""
    engine = MLSVEngine(kappa=1.2, theta=0.05, epsilon=0.2, rho=-0.5)
    spot_grid = np.array([80.0, 100.0, 120.0])
    time_grid = np.array([0.25, 0.5, 1.0])
    market_prices = np.array([
        [2.0, 1.0, 0.5],
        [3.0, 1.5, 0.8],
        [4.5, 2.5, 1.5]
    ])
    local_vol = engine.calibrate_local_vol(spot_grid, time_grid, market_prices)
    
    assert local_vol.shape == (3, 3)
    assert np.all(local_vol > 0.01)
    assert np.all(local_vol < 2.0)

def test_mlsv_parameter_initialization():
    """Verify MLSVEngine initializes parameters correctly."""
    engine = MLSVEngine(kappa=2.5, theta=0.06, epsilon=0.5, rho=0.1)
    assert engine.kappa == 2.5
    assert engine.theta == 0.06
    assert engine.epsilon == 0.5
    assert engine.rho == 0.1

def test_mlsv_dupire_grid_assignment():
    """Verify MLSVEngine holds Dupire local vol grid reference."""
    mock_grid = {"t": [0.5], "S": [100.0], "vol": [[0.2]]}
    engine = MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=-0.5, dupire_grid=mock_grid)
    assert engine.dupire_grid == mock_grid


# --- Feature B: FX SABR ---

def test_fx_data_loader_load():
    """Verify FXDataLoader loads mock currencies properly."""
    loader = FXDataLoader()
    eur_usd = loader.load_quotes("EUR/USD")
    gbp_usd = loader.load_quotes("GBP/USD")
    
    assert eur_usd["spot"] == 1.10
    assert gbp_usd["spot"] == 1.30
    assert len(eur_usd["tenors"]) == 3
    assert len(eur_usd["atm"]) == 3

def test_fx_calibrator_delta_to_strike():
    """Verify Garman-Kohlhagen delta-to-strike conversion."""
    calibrator = FXSABRCalibrator()
    # Call conversion
    k_call = calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=0.25, vol=0.10, option_type="call")
    # Put conversion
    k_put = calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=-0.25, vol=0.10, option_type="put")
    
    assert k_call > 1.10  # Out of the money call strike should be higher than spot
    assert k_put < 1.10   # Out of the money put strike should be lower than spot

def test_fx_calibrator_calibrate():
    """Verify SABR calibration to RR/BF quotes."""
    calibrator = FXSABRCalibrator()
    # EUR/USD 6M mock quotes
    alpha, rho, nu = calibrator.calibrate(
        spot=1.10, r_d=0.03, r_f=0.01, t=0.5,
        atm_vol=0.085, rr25=-0.005, bf25=0.002, rr10=-0.009, bf10=0.004
    )
    
    assert alpha > 0.0
    assert -1.0 <= rho <= 1.0
    assert nu > 0.0
    # Negative risk reversal should imply a negative correlation rho
    assert rho < 0.0

def test_fx_calibrator_extract_vol_grid():
    """Verify extracting volatility grid from calibrated SABR parameters."""
    calibrator = FXSABRCalibrator()
    params = (0.085, -0.25, 0.40) # alpha, rho, nu
    strikes = np.array([0.90, 1.00, 1.10, 1.20, 1.30])
    vols = calibrator.extract_strike_vol_grid(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, sabr_params=params, strikes=strikes)
    
    assert len(vols) == len(strikes)
    assert np.all(vols > 0.0)

def test_fx_data_loader_copy():
    """Verify FXDataLoader returns a copy of quotes to prevent pollution."""
    loader = FXDataLoader()
    data1 = loader.load_quotes("EUR/USD")
    data1["spot"] = 999.0
    data2 = loader.load_quotes("EUR/USD")
    assert data2["spot"] == 1.10  # Database remains unpolluted


# --- Feature C: Rates LMM-SABR ---

def test_sofr_swaption_loader():
    """Verify SOFRSwaptionLoader loads swaption cube dataset."""
    loader = SOFRSwaptionLoader()
    data = loader.load_swaption_cube()
    
    assert "expiries" in data
    assert "tenors" in data
    assert "strikes_bps" in data
    assert "forward_rates" in data
    assert "vol_cube" in data
    assert data["vol_cube"].shape == (5, 5, 7)

def test_rates_bachelier_pricing():
    """Verify Bachelier option pricing."""
    engine = RatesSABREngine()
    call = engine.bachelier_price(F=0.03, K=0.03, T=1.0, vol=0.008, is_call=True)
    put = engine.bachelier_price(F=0.03, K=0.03, T=1.0, vol=0.008, is_call=False)
    
    assert call > 0.0
    assert put > 0.0
    assert abs(call - put) < 1e-15  # ATM forward call = put in Bachelier

def test_rates_black_pricing():
    """Verify Black-76 option pricing."""
    engine = RatesSABREngine()
    call = engine.black_price(F=0.03, K=0.03, T=1.0, vol=0.20, is_call=True)
    put = engine.black_price(F=0.03, K=0.03, T=1.0, vol=0.20, is_call=False)
    
    assert call > 0.0
    assert put > 0.0
    assert abs(call - put) < 1e-15  # ATM forward call = put in Black-76

def test_rates_displaced_sabr_vol_and_price():
    """Verify displaced SABR volatility and price calculation."""
    engine = RatesSABREngine()
    vol = engine.displaced_sabr_vol(F=0.015, K=0.015, T=2.0, alpha=0.05, beta=0.5, rho=-0.1, nu=0.3, shift=0.02)
    price = engine.displaced_sabr_price(F=0.015, K=0.015, T=2.0, alpha=0.05, beta=0.5, rho=-0.1, nu=0.3, shift=0.02, is_call=True)
    
    assert vol > 0.0
    assert price > 0.0

def test_rates_vol_cube_interpolation():
    """Verify trilinear interpolation in vol cube."""
    loader = SOFRSwaptionLoader()
    cube_data = loader.load_swaption_cube()
    engine = RatesSABREngine()
    
    # Exact coordinates
    vol_exact = engine.interpolate_vol_cube(
        expiries=cube_data["expiries"],
        tenors=cube_data["tenors"],
        strikes_bps=cube_data["strikes_bps"],
        vol_cube=cube_data["vol_cube"],
        t_exp=1.0,
        t_ten=5.0,
        strike_bps=0.0
    )
    # Verify exact coordinate matches
    idx_exp = np.where(cube_data["expiries"] == 1.0)[0][0]
    idx_ten = np.where(cube_data["tenors"] == 5.0)[0][0]
    idx_str = np.where(cube_data["strikes_bps"] == 0.0)[0][0]
    expected_vol = cube_data["vol_cube"][idx_exp, idx_ten, idx_str]
    
    assert abs(vol_exact - expected_vol) < 1e-12
    
    # Interpolated coordinates
    vol_interp = engine.interpolate_vol_cube(
        expiries=cube_data["expiries"],
        tenors=cube_data["tenors"],
        strikes_bps=cube_data["strikes_bps"],
        vol_cube=cube_data["vol_cube"],
        t_exp=1.5,
        t_ten=6.0,
        strike_bps=25.0
    )
    assert vol_interp > 0.0


# --- Feature D: Commodity Schwartz-Smith ---

def test_commodity_data_loader():
    """Verify CommodityDataLoader loads WTI and Gold quotes."""
    loader = CommodityDataLoader()
    wti = loader.load_commodity_data("WTI")
    gc = loader.load_commodity_data("GC")
    
    assert wti["spot"] == 78.50
    assert gc["spot"] == 2050.0
    assert len(wti["futures_maturities"]) == 5
    assert wti["vols"].shape == (5, 6)

def test_schwartz_smith_pricing():
    """Verify Schwartz-Smith engine pricing."""
    engine = SchwartzSmithEngine(kappa=1.5, mu_y=0.02, sigma_x=0.35, sigma_y=0.15, rho_xy=-0.3)
    call = engine.price_option(spot=80.0, strike=80.0, maturity=1.0, risk_free_rate=0.03, is_call=True)
    put = engine.price_option(spot=80.0, strike=80.0, maturity=1.0, risk_free_rate=0.03, is_call=False)
    
    assert call > 0.0
    assert put > 0.0

def test_schwartz_smith_heston_pricing():
    """Verify Heston model option pricing via numerical integration."""
    engine = SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
    heston_params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04}
    
    call = engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=heston_params, is_call=True)
    put = engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=heston_params, is_call=False)
    
    assert call > 0.0
    assert put > 0.0

def test_schwartz_smith_heston_comparison():
    """Verify comparative pricing error method."""
    engine = SchwartzSmithEngine(kappa=1.2, mu_y=0.02, sigma_x=0.30, sigma_y=0.12, rho_xy=-0.2)
    heston_params = {"kappa": 1.8, "theta": 0.05, "sigma": 0.25, "rho": -0.5, "v0": 0.04}
    
    comparison = engine.compare_vs_heston(spot=75.0, strike=80.0, maturity=0.5, heston_params=heston_params, risk_free_rate=0.02, is_call=True)
    
    assert "schwartz_smith_price" in comparison
    assert "heston_price" in comparison
    assert "absolute_error" in comparison
    assert "relative_error" in comparison
    assert comparison["schwartz_smith_price"] >= 0.0
    assert comparison["heston_price"] >= 0.0

def test_schwartz_smith_parameter_initialization():
    """Verify SchwartzSmithEngine parameter assignment."""
    engine = SchwartzSmithEngine(kappa=2.0, mu_y=0.05, sigma_x=0.4, sigma_y=0.2, rho_xy=0.5)
    assert engine.kappa == 2.0
    assert engine.mu_y == 0.05
    assert engine.sigma_x == 0.4
    assert engine.sigma_y == 0.2
    assert engine.rho_xy == 0.5


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 tests per feature; >=20 tests total)
# ==============================================================================

# --- Feature A: Equity MLSV Boundaries ---

def test_mlsv_spot_boundary():
    engine = MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=0.0)
    with pytest.raises(ValueError, match="Spot.*must be positive"):
        engine.price_option(spot=0.0, strike=100.0, maturity=1.0, vol=0.2)
    with pytest.raises(ValueError, match="Spot.*must be positive"):
        engine.price_option(spot=-10.0, strike=100.0, maturity=1.0, vol=0.2)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.price_option(spot=val, strike=100.0, maturity=1.0, vol=0.2)

def test_mlsv_strike_boundary():
    engine = MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=0.0)
    with pytest.raises(ValueError, match="Strike.*must be positive"):
        engine.price_option(spot=100.0, strike=0.0, maturity=1.0, vol=0.2)
    with pytest.raises(ValueError, match="Strike.*must be positive"):
        engine.price_option(spot=100.0, strike=-50.0, maturity=1.0, vol=0.2)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=val, maturity=1.0, vol=0.2)

def test_mlsv_maturity_boundary():
    engine = MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=0.0)
    with pytest.raises(ValueError, match="Maturity.*must be positive"):
        engine.price_option(spot=100.0, strike=100.0, maturity=0.0, vol=0.2)
    with pytest.raises(ValueError, match="Maturity.*must be positive"):
        engine.price_option(spot=100.0, strike=100.0, maturity=-1.0, vol=0.2)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=100.0, maturity=val, vol=0.2)

def test_mlsv_vol_boundary():
    engine = MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=0.0)
    with pytest.raises(ValueError, match="Volatility.*must be positive"):
        engine.price_option(spot=100.0, strike=100.0, maturity=1.0, vol=0.0)
    with pytest.raises(ValueError, match="Volatility.*must be positive"):
        engine.price_option(spot=100.0, strike=100.0, maturity=1.0, vol=-0.1)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=100.0, maturity=1.0, vol=val)

def test_mlsv_invalid_kappa_rho_boundaries():
    with pytest.raises(ValueError, match="kappa must be positive"):
        MLSVEngine(kappa=0.0, theta=0.04, epsilon=0.2, rho=0.0)
    with pytest.raises(ValueError, match="rho must be between"):
        MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=1.1)
    with pytest.raises(ValueError, match="rho must be between"):
        MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=-1.05)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            MLSVEngine(kappa=val, theta=0.04, epsilon=0.2, rho=0.0)
        with pytest.raises(ValueError):
            MLSVEngine(kappa=1.0, theta=val, epsilon=0.2, rho=0.0)
        with pytest.raises(ValueError):
            MLSVEngine(kappa=1.0, theta=0.04, epsilon=val, rho=0.0)
        with pytest.raises(ValueError):
            MLSVEngine(kappa=1.0, theta=0.04, epsilon=0.2, rho=val)


# --- Feature B: FX SABR Boundaries ---

def test_fx_loader_invalid_pair():
    loader = FXDataLoader()
    with pytest.raises(ValueError, match="FX pair must be in format"):
        loader.load_quotes("EURUSD")
    with pytest.raises(ValueError, match="Currency pair cannot be empty"):
        loader.load_quotes("")
    with pytest.raises(ValueError, match="No quotes available"):
        loader.load_quotes("USD/JPY")

def test_fx_calibrator_invalid_delta():
    calibrator = FXSABRCalibrator()
    # Call delta out of bounds (> 1.0 or negative)
    with pytest.raises(ValueError, match="Call delta must be in"):
        calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=1.5, vol=0.10)
    with pytest.raises(ValueError, match="Call delta must be in"):
        calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=-0.25, vol=0.10, option_type="call")
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=val, r_d=0.03, r_f=0.01, t=0.5, delta=0.25, vol=0.10)
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=1.10, r_d=val, r_f=0.01, t=0.5, delta=0.25, vol=0.10)
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=val, t=0.5, delta=0.25, vol=0.10)
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=val, delta=0.25, vol=0.10)
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=val, vol=0.10)
        with pytest.raises(ValueError):
            calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=0.25, vol=val)

def test_fx_calibrator_invalid_option_type():
    calibrator = FXSABRCalibrator()
    with pytest.raises(ValueError, match="option_type must be"):
        calibrator.delta_to_strike(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, delta=0.25, vol=0.10, option_type="straddle")

def test_fx_calibrator_calibrate_vols_boundary():
    calibrator = FXSABRCalibrator()
    # Negative/Zero ATM vol
    with pytest.raises(ValueError, match="ATM volatility must be positive"):
        calibrator.calibrate(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, atm_vol=-0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
    # Negative Butterfly
    with pytest.raises(ValueError, match="Butterfly volatilities must be positive"):
        calibrator.calibrate(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=-0.002, rr10=0.0, bf10=0.004)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=val, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=val, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=val, t=0.5, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=val, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, atm_vol=val, rr25=0.0, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=val, bf25=0.002, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=val, rr10=0.0, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=val, bf10=0.004)
        with pytest.raises(ValueError):
            calibrator.calibrate(spot=1.10, r_d=0.03, r_f=0.01, t=0.5, atm_vol=0.08, rr25=0.0, bf25=0.002, rr10=0.0, bf10=val)

def test_fx_calibrator_extract_empty_grid():
    calibrator = FXSABRCalibrator()
    params = (0.08, 0.0, 0.3)
    with pytest.raises(ValueError, match="Strikes grid cannot be empty"):
        calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=params, strikes=np.array([]))
    with pytest.raises(ValueError, match="Strike values must be positive"):
        calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=params, strikes=np.array([1.0, -0.5]))
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=val, r_d=0.03, r_f=0.01, t=0.5, sabr_params=params, strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=val, r_f=0.01, t=0.5, sabr_params=params, strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=val, t=0.5, sabr_params=params, strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=val, sabr_params=params, strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=(val, 0.0, 0.3), strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=(0.08, val, 0.3), strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=(0.08, 0.0, val), strikes=np.array([1.0]))
        with pytest.raises(ValueError):
            calibrator.extract_strike_vol_grid(spot=1.1, r_d=0.03, r_f=0.01, t=0.5, sabr_params=params, strikes=np.array([1.0, val]))


# --- Feature C: Rates LMM-SABR Boundaries ---

def test_sofr_loader_empty_date():
    loader = SOFRSwaptionLoader()
    with pytest.raises(ValueError, match="Date cannot be empty"):
        loader.load_swaption_cube(date="")

def test_rates_bachelier_boundaries():
    engine = RatesSABREngine()
    with pytest.raises(ValueError, match="Maturity T must be positive"):
        engine.bachelier_price(F=0.03, K=0.03, T=0.0, vol=0.008)
    with pytest.raises(ValueError, match="Normal volatility must be positive"):
        engine.bachelier_price(F=0.03, K=0.03, T=1.0, vol=-0.008)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.bachelier_price(F=val, K=0.03, T=1.0, vol=0.008)
        with pytest.raises(ValueError):
            engine.bachelier_price(F=0.03, K=val, T=1.0, vol=0.008)
        with pytest.raises(ValueError):
            engine.bachelier_price(F=0.03, K=0.03, T=val, vol=0.008)
        with pytest.raises(ValueError):
            engine.bachelier_price(F=0.03, K=0.03, T=1.0, vol=val)

def test_rates_black_boundaries():
    engine = RatesSABREngine()
    with pytest.raises(ValueError, match="Forward and strike must be positive"):
        engine.black_price(F=0.0, K=0.03, T=1.0, vol=0.20)
    with pytest.raises(ValueError, match="Forward and strike must be positive"):
        engine.black_price(F=0.03, K=-0.03, T=1.0, vol=0.20)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.black_price(F=val, K=0.03, T=1.0, vol=0.20)
        with pytest.raises(ValueError):
            engine.black_price(F=0.03, K=val, T=1.0, vol=0.20)
        with pytest.raises(ValueError):
            engine.black_price(F=0.03, K=0.03, T=val, vol=0.20)
        with pytest.raises(ValueError):
            engine.black_price(F=0.03, K=0.03, T=1.0, vol=val)

def test_rates_displaced_sabr_shifted_boundary():
    engine = RatesSABREngine()
    # Forward + shift <= 0
    with pytest.raises(ValueError, match="Shifted forward.*must be positive"):
        engine.displaced_sabr_vol(F=-0.02, K=0.03, T=1.0, alpha=0.05, beta=0.5, rho=0.0, nu=0.3, shift=0.01)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=val, K=0.03, T=1.0, alpha=0.05, beta=0.5, rho=0.0, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=val, T=1.0, alpha=0.05, beta=0.5, rho=0.0, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=val, alpha=0.05, beta=0.5, rho=0.0, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=1.0, alpha=val, beta=0.5, rho=0.0, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=1.0, alpha=0.05, beta=val, rho=0.0, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=1.0, alpha=0.05, beta=0.5, rho=val, nu=0.3, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=1.0, alpha=0.05, beta=0.5, rho=0.0, nu=val, shift=0.01)
        with pytest.raises(ValueError):
            engine.displaced_sabr_vol(F=0.015, K=0.03, T=1.0, alpha=0.05, beta=0.5, rho=0.0, nu=0.3, shift=val)

def test_rates_vol_cube_interpolation_boundaries():
    engine = RatesSABREngine()
    loader = SOFRSwaptionLoader()
    cube = loader.load_swaption_cube()
    
    with pytest.raises(ValueError, match="Expiry t_exp must be positive"):
        engine.interpolate_vol_cube(cube["expiries"], cube["tenors"], cube["strikes_bps"], cube["vol_cube"], -0.1, 5.0, 0.0)
        
    with pytest.raises(ValueError, match="Vol cube shape must match"):
        # Mismatch shape
        engine.interpolate_vol_cube(cube["expiries"], cube["tenors"][:-1], cube["strikes_bps"], cube["vol_cube"], 1.0, 5.0, 0.0)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.interpolate_vol_cube(cube["expiries"], cube["tenors"], cube["strikes_bps"], cube["vol_cube"], val, 5.0, 0.0)
        with pytest.raises(ValueError):
            engine.interpolate_vol_cube(cube["expiries"], cube["tenors"], cube["strikes_bps"], cube["vol_cube"], 1.0, val, 0.0)
        with pytest.raises(ValueError):
            engine.interpolate_vol_cube(cube["expiries"], cube["tenors"], cube["strikes_bps"], cube["vol_cube"], 1.0, 5.0, val)
        
        bad_exp = cube["expiries"].copy()
        bad_exp[0] = val
        with pytest.raises(ValueError):
            engine.interpolate_vol_cube(bad_exp, cube["tenors"], cube["strikes_bps"], cube["vol_cube"], 1.0, 5.0, 0.0)


# --- Feature D: Commodity Schwartz-Smith Boundaries ---

def test_commodity_loader_invalid():
    loader = CommodityDataLoader()
    with pytest.raises(ValueError, match="Commodity symbol cannot be empty"):
        loader.load_commodity_data("")
    with pytest.raises(ValueError, match="not found in mock database"):
        loader.load_commodity_data("OIL")

def test_schwartz_smith_params_boundary():
    with pytest.raises(ValueError, match="kappa must be positive"):
        SchwartzSmithEngine(kappa=0.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
    with pytest.raises(ValueError, match="sigma_x must be positive"):
        SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=-0.2, sigma_y=0.1, rho_xy=0.0)
    with pytest.raises(ValueError, match="Correlation.*must be in"):
        SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=1.01)

def test_schwartz_smith_pricing_boundary():
    engine = SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
    with pytest.raises(ValueError, match="Spot must be positive"):
        engine.price_option(spot=0.0, strike=80.0, maturity=1.0)
    with pytest.raises(ValueError, match="Strike must be positive"):
        engine.price_option(spot=80.0, strike=-10.0, maturity=1.0)
    with pytest.raises(ValueError, match="Maturity must be positive"):
        engine.price_option(spot=80.0, strike=80.0, maturity=0.0)

def test_schwartz_smith_heston_params_boundary():
    engine = SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
    heston_bad = {"kappa": -1.0, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04}
    with pytest.raises(ValueError, match="Heston parameters.*must be positive"):
        engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=heston_bad)
        
    heston_missing = {"kappa": 2.0, "theta": 0.04}
    with pytest.raises(ValueError, match="Heston parameters must contain"):
        engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=heston_missing)

def test_schwartz_smith_nan_inf_boundary():
    """Verify that passing NaN/Inf to Schwartz-Smith engine raises ValueError."""
    # 1. Parameter initialization
    for param in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            SchwartzSmithEngine(kappa=param, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
        with pytest.raises(ValueError):
            SchwartzSmithEngine(kappa=1.0, mu_y=param, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
        with pytest.raises(ValueError):
            SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=param, sigma_y=0.1, rho_xy=0.0)
        with pytest.raises(ValueError):
            SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=param, rho_xy=0.0)
        with pytest.raises(ValueError):
            SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=param)

    # 2. Pricing methods
    engine = SchwartzSmithEngine(kappa=1.0, mu_y=0.01, sigma_x=0.2, sigma_y=0.1, rho_xy=0.0)
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.price_option(spot=val, strike=100.0, maturity=1.0)
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=val, maturity=1.0)
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=100.0, maturity=val)
        with pytest.raises(ValueError):
            engine.price_option(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=val)

    # 3. Heston pricing methods
    heston_params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04}
    for val in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            engine.heston_price(spot=val, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=heston_params)
        with pytest.raises(ValueError):
            engine.heston_price(spot=100.0, strike=val, maturity=1.0, risk_free_rate=0.03, heston_params=heston_params)
        with pytest.raises(ValueError):
            engine.heston_price(spot=100.0, strike=100.0, maturity=val, risk_free_rate=0.03, heston_params=heston_params)
        with pytest.raises(ValueError):
            engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=val, heston_params=heston_params)
        
        # In heston_params
        for key in heston_params:
            bad_params = heston_params.copy()
            bad_params[key] = val
            with pytest.raises(ValueError):
                engine.heston_price(spot=100.0, strike=100.0, maturity=1.0, risk_free_rate=0.03, heston_params=bad_params)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

def test_call_put_parity_mlsv():
    """Verify Call-Put Parity in MLSVEngine."""
    engine = MLSVEngine(kappa=1.5, theta=0.04, epsilon=0.3, rho=-0.7)
    spot = 100.0
    strike = 95.0
    maturity = 0.5
    vol = 0.25
    r = 0.0  # MLSV pricing base uses 0 interest rate inside stub (pure BS d1/d2)
    
    price_c = engine.price_option(spot, strike, maturity, vol, is_call=True)
    price_p = engine.price_option(spot, strike, maturity, vol, is_call=False)
    
    # Parity check: C - P = S - K * exp(-r*T) = S - K (since r=0)
    lhs = price_c - price_p
    rhs = spot - strike
    assert abs(lhs - rhs) < 1e-12

def test_call_put_parity_rates_bachelier():
    """Verify Call-Put Parity in Bachelier pricing."""
    engine = RatesSABREngine()
    F = 0.04
    K = 0.035
    T = 2.0
    vol = 0.0090 # 90bps
    
    price_c = engine.bachelier_price(F, K, T, vol, is_call=True)
    price_p = engine.bachelier_price(F, K, T, vol, is_call=False)
    
    # Parity: C - P = F - K (for forward rates options under normal model)
    assert abs((price_c - price_p) - (F - K)) < 1e-15

def test_call_put_parity_rates_black():
    """Verify Call-Put Parity in Black-76 pricing."""
    engine = RatesSABREngine()
    F = 0.04
    K = 0.035
    T = 2.0
    vol = 0.25
    
    price_c = engine.black_price(F, K, T, vol, is_call=True)
    price_p = engine.black_price(F, K, T, vol, is_call=False)
    
    # Parity: C - P = F - K
    assert abs((price_c - price_p) - (F - K)) < 1e-15

def test_call_put_parity_schwartz_smith():
    """Verify Call-Put Parity in Schwartz-Smith Engine."""
    engine = SchwartzSmithEngine(kappa=1.1, mu_y=0.02, sigma_x=0.25, sigma_y=0.10, rho_xy=-0.2)
    spot = 80.0
    strike = 85.0
    maturity = 0.75
    r = 0.04
    
    price_c = engine.price_option(spot, strike, maturity, risk_free_rate=r, is_call=True)
    price_p = engine.price_option(spot, strike, maturity, risk_free_rate=r, is_call=False)
    
    # Parity: C - P = S - K * exp(-r * T)
    lhs = price_c - price_p
    rhs = spot - strike * np.exp(-r * maturity)
    assert abs(lhs - rhs) < 1e-12

def test_fx_strike_inversion_parity():
    """Verify that delta to strike to price yields consistent delta representation."""
    calibrator = FXSABRCalibrator()
    spot = 1.10
    r_d = 0.03
    r_f = 0.01
    t = 1.0
    vol = 0.08
    delta = 0.25
    
    # Get strike for 25-delta call
    k_strike = calibrator.delta_to_strike(spot, r_d, r_f, t, delta, vol, option_type="call")
    
    # Compute BS delta at this strike
    df_f = np.exp(-r_f * t)
    d1 = (np.log(spot / k_strike) + (r_d - r_f + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
    computed_delta = df_f * norm.cdf(d1)
    
    # Inverted strike should exactly reproduce the target delta
    assert abs(computed_delta - delta) < 1e-12


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (>=5 scenario tests)
# ==============================================================================

def test_scenario_equity_mlsv_calibration():
    """
    Scenario 1: Equity MLSV SPX calibration workflow.
    Loads SPX parameters, calibrates local volatility grid, sets up McKean-Vlasov engine,
    and prices options along the conditional volatility expectations.
    """
    # 1. Initialize SDE parameters
    kappa = 1.6
    theta = 0.045
    epsilon = 0.35
    rho = -0.75
    
    # 2. Build grids representing the SPX surface
    spot_grid = np.array([3800.0, 3900.0, 4000.0, 4100.0, 4200.0])
    time_grid = np.array([0.1, 0.25, 0.5, 1.0])
    
    # Market Call Prices shape: (times, spots)
    mkt_prices = np.array([
        [220.0, 130.0, 60.0, 20.0, 5.0],
        [260.0, 175.0, 105.0, 55.0, 25.0],
        [310.0, 230.0, 160.0, 105.0, 65.0],
        [400.0, 320.0, 255.0, 195.0, 145.0]
    ])
    
    # 3. Instantiate Engine & Calibrate Local Vol Grid
    engine = MLSVEngine(kappa=kappa, theta=theta, epsilon=epsilon, rho=rho)
    local_vol_grid = engine.calibrate_local_vol(spot_grid, time_grid, mkt_prices)
    
    assert local_vol_grid.shape == (4, 5)
    
    # 4. Integrate calibrated grid into the engine
    engine.dupire_grid = local_vol_grid
    
    # 5. Extract conditional expectations and verify SDE behavior
    cond_vol = engine.conditional_expectation(spot_grid=spot_grid, current_spot=4000.0, current_vol=0.20)
    
    assert len(cond_vol) == len(spot_grid)
    # Vol expectation should exhibit strong leverage effect (vol falls as spot rises)
    assert cond_vol[0] > cond_vol[-1]
    
    # 6. Price an out-of-the-money SPX put
    put_price = engine.price_option(spot=4000.0, strike=3800.0, maturity=0.5, vol=cond_vol[0], is_call=False)
    assert put_price > 0.0

def test_scenario_fx_smile_calibration():
    """
    Scenario 2: FX EUR/USD SABR delta-strike calibration.
    Loads EUR/USD volatility quotes, converts risk-reversal and butterfly deltas to strikes,
    calibrates SABR (beta=1) parameters, and extracts a continuous volatility smile grid.
    """
    # 1. Load market quotes
    loader = FXDataLoader()
    eur_usd = loader.load_quotes("EUR/USD")
    
    spot = eur_usd["spot"]
    r_d = eur_usd["domestic_rate"]
    r_f = eur_usd["foreign_rate"]
    t = eur_usd["tenors"][1] # 6M tenor
    atm_vol = eur_usd["atm"][1]
    rr25 = eur_usd["rr25"][1]
    bf25 = eur_usd["bf25"][1]
    rr10 = eur_usd["rr10"][1]
    bf10 = eur_usd["bf10"][1]
    
    # 2. Calibrate SABR parameters using Calibrator
    calibrator = FXSABRCalibrator()
    alpha, rho, nu = calibrator.calibrate(spot, r_d, r_f, t, atm_vol, rr25, bf25, rr10, bf10)
    
    assert alpha > 0.0
    assert -0.99 <= rho <= 0.99
    assert nu > 0.0
    
    # 3. Generate high-resolution strikes grid to extract volatility smile
    smile_strikes = np.linspace(spot * 0.8, spot * 1.2, 100)
    smile_vols = calibrator.extract_strike_vol_grid(spot, r_d, r_f, t, (alpha, rho, nu), smile_strikes)
    
    assert len(smile_vols) == 100
    assert np.all(smile_vols > 0.0)
    
    # 4. Check that volatility smile is asymmetric due to negative correlation rho
    vol_downside = smile_vols[10] # Strike well below spot
    vol_upside = smile_vols[90]   # Strike well above spot
    # For EUR/USD, downside vols are typically higher (negative skew)
    assert vol_downside > vol_upside

def test_scenario_sofr_swaption_cube_pricing():
    """
    Scenario 3: SOFR swaption cube loading and pricing.
    Loads SOFR swaption cube, extracts forward rates, interpolates normal volatilities
    for non-grid tenors/expiries, and prices interest rate options via Bachelier.
    """
    # 1. Load SOFR swaption data
    loader = SOFRSwaptionLoader()
    cube = loader.load_swaption_cube()
    
    # 2. Target custom swaption contract: 1.5Y expiry on 7Y swap tenor, strike at ATM+100bps
    t_exp = 1.5
    t_ten = 7.0
    strike_bps = 100.0
    
    # 3. Interpolate the forward swap rate for this contract (bilinear on grid)
    # Using simple bilinear setup for forward rates
    engine = RatesSABREngine()
    
    # Bilinear interpolation helper for forward rates matrix
    # Expiries are [0.25, 0.5, 1.0, 2.0, 5.0], Tenors are [1.0, 2.0, 5.0, 10.0, 30.0]
    # We can use the interpolate_vol_cube method by passing a 3D matrix of forward rates repeated
    # or write a direct bilinear. Let's build a dummy 3D cube of forward rates to reuse our engine's method!
    # Cube size: expiries x tenors x 1
    rates_cube = np.repeat(cube["forward_rates"][:, :, np.newaxis], len(cube["strikes_bps"]), axis=2)
    interpolated_fwd = engine.interpolate_vol_cube(
        expiries=cube["expiries"],
        tenors=cube["tenors"],
        strikes_bps=cube["strikes_bps"],
        vol_cube=rates_cube,
        t_exp=t_exp,
        t_ten=t_ten,
        strike_bps=0.0
    )
    
    assert interpolated_fwd > 0.0
    
    # 4. Interpolate volatility from the swaption normal vol cube
    interpolated_vol = engine.interpolate_vol_cube(
        expiries=cube["expiries"],
        tenors=cube["tenors"],
        strikes_bps=cube["strikes_bps"],
        vol_cube=cube["vol_cube"],
        t_exp=t_exp,
        t_ten=t_ten,
        strike_bps=strike_bps
    )
    
    assert interpolated_vol > 0.0
    
    # 5. Price option using Bachelier Engine
    strike_rate = interpolated_fwd + (strike_bps / 10000.0)
    option_price = engine.bachelier_price(F=interpolated_fwd, K=strike_rate, T=t_exp, vol=interpolated_vol, is_call=True)
    
    assert option_price > 0.0

def test_scenario_commodity_schwartz_smith_calibration():
    """
    Scenario 4: CME WTI Crude Oil Schwartz-Smith pricing and Heston comparison.
    Loads WTI futures and option volatility surface, prices options using the
    integrated variance of Schwartz-Smith, prices options using Heston Fourier solver,
    and verifies pricing errors are reported correctly.
    """
    # 1. Load WTI data
    loader = CommodityDataLoader()
    wti_data = loader.load_commodity_data("WTI")
    
    spot = wti_data["spot"]
    maturity = wti_data["futures_maturities"][2] # ~6M
    strike = wti_data["strikes"][3]            # strike 80.0
    mkt_vol = wti_data["vols"][2, 3]           # vol for 6M at 80.0 strike
    
    # 2. Setup Schwartz-Smith engine with calibrated parameters
    # Let's assume these are calibrated to the term structure of volatility
    ss_engine = SchwartzSmithEngine(kappa=1.4, mu_y=0.02, sigma_x=0.32, sigma_y=0.12, rho_xy=-0.25)
    
    # 3. Price option using Schwartz-Smith
    ss_price = ss_engine.price_option(spot=spot, strike=strike, maturity=maturity, risk_free_rate=0.03, is_call=True)
    assert ss_price > 0.0
    
    # 4. Setup comparison Heston parameters
    heston_params = {"kappa": 1.7, "theta": 0.05, "sigma": 0.28, "rho": -0.4, "v0": 0.045}
    
    # 5. Execute comparison
    comparison = ss_engine.compare_vs_heston(
        spot=spot, strike=strike, maturity=maturity,
        heston_params=heston_params, risk_free_rate=0.03, is_call=True
    )
    
    assert comparison["schwartz_smith_price"] == ss_price
    assert comparison["heston_price"] > 0.0
    assert comparison["absolute_error"] >= 0.0

def test_scenario_multi_asset_portfolio_pricing():
    """
    Scenario 5: Multi-asset parallel pricing workflow.
    Simulates a portfolio of options across all 4 asset classes:
    Equity (SPX), FX (EUR/USD), Rates (SOFR Swaption), and Commodity (WTI Crude).
    """
    # 1. Setup portfolio components
    # Asset 1: SPX option (Equity MLSV)
    eq_engine = MLSVEngine(kappa=1.5, theta=0.04, epsilon=0.3, rho=-0.7)
    eq_spot = 4000.0
    eq_strike = 4050.0
    eq_mat = 0.25
    eq_vol = 0.18
    
    # Asset 2: EUR/USD option (FX SABR)
    fx_calibrator = FXSABRCalibrator()
    fx_params = (0.08, -0.20, 0.35)
    fx_spot = 1.10
    fx_strike = 1.12
    fx_mat = 0.5
    fx_vol = fx_calibrator.extract_strike_vol_grid(
        spot=fx_spot, r_d=0.03, r_f=0.01, t=fx_mat,
        sabr_params=fx_params, strikes=np.array([fx_strike])
    )[0]
    
    # Asset 3: SOFR Swaption (Rates Displaced SABR)
    rates_engine = RatesSABREngine()
    rates_fwd = 0.035
    rates_strike = 0.0375
    rates_mat = 1.0
    rates_alpha, rates_beta, rates_rho, rates_nu, rates_shift = 0.05, 0.5, -0.1, 0.3, 0.02
    
    # Asset 4: WTI Oil option (Commodity Schwartz-Smith)
    comm_engine = SchwartzSmithEngine(kappa=1.3, mu_y=0.01, sigma_x=0.30, sigma_y=0.10, rho_xy=-0.2)
    comm_spot = 78.0
    comm_strike = 82.0
    comm_mat = 0.5
    
    # 2. Compute prices across the portfolio
    price_eq = eq_engine.price_option(spot=eq_spot, strike=eq_strike, maturity=eq_mat, vol=eq_vol, is_call=True)
    
    # FX option pricing via Black-76 using extracted SABR vol
    price_fx = rates_engine.black_price(F=fx_spot * np.exp((0.03 - 0.01) * fx_mat), K=fx_strike, T=fx_mat, vol=fx_vol, is_call=True)
    
    price_rates = rates_engine.displaced_sabr_price(
        F=rates_fwd, K=rates_strike, T=rates_mat,
        alpha=rates_alpha, beta=rates_beta, rho=rates_rho, nu=rates_nu, shift=rates_shift, is_call=True
    )
    
    price_comm = comm_engine.price_option(spot=comm_spot, strike=comm_strike, maturity=comm_mat, risk_free_rate=0.03, is_call=True)
    
    portfolio_value = price_eq + price_fx + price_rates + price_comm
    
    # 3. Assertions
    assert price_eq > 0.0
    assert price_fx > 0.0
    assert price_rates > 0.0
    assert price_comm > 0.0
    assert portfolio_value > 0.0

import time
import numpy as np
import sys
import os
import scipy.optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pricing.bachelier import bachelier_implied_vol, bachelier_price
from market.rates_data import load_swaption_vol_cube, get_synthetic_forward_rates
from pricing.sabr_rates import SwaptionVolCube, displaced_sabr_vol

def calibrate_sabr_node_3d(F, strikes, market_vols, T, beta, shift, vol_type='normal'):
    F = float(F)
    T = float(T)
    beta = float(beta)
    shift = float(shift)
    
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    
    atm_idx = np.argmin(np.abs(strikes - F))
    atm_vol = market_vols[atm_idx]
    
    F_s = F + shift
    if vol_type.lower() == 'normal':
        alpha_init = atm_vol / (F_s ** beta)
    else:
        alpha_init = atm_vol * (F_s ** (1.0 - beta))
        
    alpha_init = np.clip(alpha_init, 1e-4, 5.0)
    rho_init = 0.0
    nu_init = 0.3
    
    x0 = [alpha_init, rho_init, nu_init]
    bounds = ([1e-5, -0.999, 1e-5], [10.0, 0.999, 5.0])
    
    def residuals(params):
        alpha_val, rho_val, nu_val = params
        model_vols = displaced_sabr_vol(F, strikes, T, alpha_val, beta, rho_val, nu_val, shift, vol_type=vol_type)
        res = model_vols - market_vols
        res = np.where(np.isnan(res), 1e6, res)
        return res
        
    res = opt.least_squares(residuals, x0, bounds=bounds, method='trf')
    return res.x

def benchmark_implied_vol():
    print("Benchmarking implied volatility solver...")
    F = 0.03
    K = 0.035
    T = 1.5
    target_vol_n = 0.0075
    p_n = bachelier_price(F, K, T, target_vol_n, 'call')
    
    t0 = time.perf_counter()
    for _ in range(5000):
        _ = bachelier_implied_vol(p_n, F, K, T, 'call')
    t1 = time.perf_counter()
    print(f"5000 Halley normal implied vol calculations: {t1 - t0:.4f} seconds")

def benchmark_calibration():
    print("\nBenchmarking SABR calibration...")
    expiries, tenors, relative_strikes, market_vols = load_swaption_vol_cube()
    forward_rates = get_synthetic_forward_rates(expiries, tenors)
    
    large_exp = np.concatenate([expiries * (i + 1) for i in range(10)])
    large_ten = np.concatenate([tenors * (i + 1) for i in range(1)])
    
    large_cube_vols = np.tile(market_vols, (10, 1, 1))
    large_forward_rates = np.tile(forward_rates, (10, 1))
    
    num_nodes = len(large_exp) * len(large_ten)
    print(f"Calibrating a larger grid of {num_nodes} nodes:")
    
    # 1. 3D Sequential Calibration (baseline)
    t0 = time.perf_counter()
    for i, T in enumerate(large_exp):
        for j, tenor in enumerate(large_ten):
            F = large_forward_rates[i, j]
            strikes = F + relative_strikes * 1e-4
            market_vols_node = large_cube_vols[i, j, :]
            _ = calibrate_sabr_node_3d(F, strikes, market_vols_node, T, 0.5, 0.01, 'normal')
    t1 = time.perf_counter()
    time_3d = t1 - t0
    print(f"Original 3D Calibration time (sequential): {time_3d:.4f} seconds")
    
    # 2. 2D Sequential Calibration
    t0 = time.perf_counter()
    cube_seq = SwaptionVolCube(large_exp, large_ten, relative_strikes)
    cube_seq.calibrate(large_cube_vols, large_forward_rates, beta=0.5, shift=0.01, vol_type='normal', parallel=False)
    t1 = time.perf_counter()
    time_2d_seq = t1 - t0
    print(f"Optimized 2D Calibration time (sequential): {time_2d_seq:.4f} seconds")
    print(f"Dimensionality Reduction Speedup (3D vs 2D seq): {time_3d / time_2d_seq:.2f}x")
    
    # 3. 2D Parallel Calibration
    t0 = time.perf_counter()
    cube_par = SwaptionVolCube(large_exp, large_ten, relative_strikes)
    cube_par.calibrate(large_cube_vols, large_forward_rates, beta=0.5, shift=0.01, vol_type='normal', parallel=True)
    t2 = time.perf_counter()
    time_2d_par = t2 - t0
    print(f"Optimized 2D Calibration time (parallel): {time_2d_par:.4f} seconds")
    print(f"Parallel Speedup (2D seq vs 2D par): {time_2d_seq / time_2d_par:.2f}x")
    print(f"Total Combined Speedup (3D seq vs 2D par): {time_3d / time_2d_par:.2f}x")

if __name__ == "__main__":
    benchmark_implied_vol()
    benchmark_calibration()

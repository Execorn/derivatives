#!/usr/bin/env python3
"""
validate_pricing_math.py — Programmatic validation script to check pricing surfaces
from all models for arbitrage violations, positivity, monotonicity, and mathematical correctness.
"""

import os
import sys
import time
import numpy as np
import torch
import scipy.stats as stats

# Ensure src/ is in the python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepvol.mrm.arbitrage import check_arbitrage
from deepvol.mrm.guardian import price_to_iv
from deepvol.models.heston import HestonEngine
from deepvol.models.lifted_heston_gpu import price_iv_surface_gpu
from deepvol.models.rbergomi_gpu import rBergomiEngine
from deepvol.models.mlsv_gpu import MLSVSolverGPU
from deepvol.models.sabr import sabr_iv_surface, ssvi_iv_surface
from deepvol.models.sabr_rates import RatesSABREngine
from deepvol.models.schwartz_smith import SchwartzSmithEngine

# ---------------------------------------------------------------------------
# Black-Scholes Call/Put Pricing Helpers
# ---------------------------------------------------------------------------

def bs_call_np(S, K, T, iv):
    vol_std = iv * np.sqrt(T)
    if vol_std <= 1e-12:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * vol_std**2) / vol_std
    d2 = d1 - vol_std
    return S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)

def bs_put_np(S, K, T, iv):
    vol_std = iv * np.sqrt(T)
    if vol_std <= 1e-12:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + 0.5 * vol_std**2) / vol_std
    d2 = d1 - vol_std
    return K * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# ---------------------------------------------------------------------------
# Main Validation Execution
# ---------------------------------------------------------------------------

def run_math_validation():
    print("=" * 80)
    print(" DEEPVOL MATHEMATICAL CORRECTNESS VALIDATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        torch.cuda.synchronize()
    print("-" * 80)
    
    # Define validation grids
    S0 = 100.0
    T_grid = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    K_grid = np.array([-0.3, -0.15, 0.0, 0.15, 0.3]) # log-moneyness
    strikes_abs = S0 * np.exp(K_grid)
    
    nT = len(T_grid)
    nK = len(K_grid)
    
    results = {}
    
    # -----------------------------------------------------------------------
    # 1. Classic Heston Model
    # -----------------------------------------------------------------------
    print("\n[1/8] Validating Classic Heston (Gatheral Fourier-COS)...")
    t0 = time.time()
    try:
        heston_engine = HestonEngine()
        heston_params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}
        heston_ivs = heston_engine.price_surface(heston_params, T_grid, K_grid, S0=S0)
        t_elapsed = time.time() - t0
        results["Classic Heston"] = {
            "iv_surface": heston_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": heston_params
        }
    except Exception as e:
        results["Classic Heston"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 2. Rough Heston Model (Lifted Heston ODE Solver)
    # -----------------------------------------------------------------------
    print("[2/8] Validating Rough Heston (Lifted Heston ODE RK4 GPU/CPU)...")
    t0 = time.time()
    try:
        lifted_params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04, "H": 0.08}
        if device == "cuda":
            torch.cuda.synchronize()
        lifted_ivs = price_iv_surface_gpu(lifted_params, T_grid, K_grid, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_elapsed = time.time() - t0
        results["Rough Heston"] = {
            "iv_surface": lifted_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": lifted_params
        }
    except Exception as e:
        results["Rough Heston"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 3. Rough Bergomi Model (Hybrid Monte Carlo GPU/CPU)
    # -----------------------------------------------------------------------
    print("[3/8] Validating Rough Bergomi (Hybrid MC Bennedsen)...")
    t0 = time.time()
    try:
        rb_engine = rBergomiEngine()
        # [v0, H, eta, rho]
        if device == "cuda":
            torch.cuda.synchronize()
        rb_ivs = rb_engine.price_surface(v0=0.04, H=0.1, eta=0.5, rho=-0.7, T_grid=T_grid, K_grid=K_grid, N_paths=10000, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_elapsed = time.time() - t0
        results["Rough Bergomi"] = {
            "iv_surface": rb_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": {"v0": 0.04, "H": 0.1, "eta": 0.5, "rho": -0.7}
        }
    except Exception as e:
        results["Rough Bergomi"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 4. MLSV Model (McKean-Vlasov SDE Particle Solver)
    # -----------------------------------------------------------------------
    print("[4/8] Validating MLSV / McKean-Vlasov (Nadaraya-Watson KDR)...")
    t0 = time.time()
    try:
        def dup_vol_fn(t, s):
            return torch.full_like(s, 0.2)
        if device == "cuda":
            torch.cuda.synchronize()
        mlsv_solver = MLSVSolverGPU(
            S0=S0,
            r=0.0,
            q=0.0,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            T=max(T_grid),
            steps_per_unit=50,
            N_paths=2000,
            dupire_vol_fn=dup_vol_fn,
            device=device
        )
        mlsv_solver.simulate(method="nadaraya_watson")
        
        # Price surface via vectorized evaluation
        mlsv_prices_torch = mlsv_solver.price_european_option(strike=strikes_abs, maturity=T_grid)
        mlsv_prices = mlsv_prices_torch.cpu().numpy() # shape (nT, nK)
        
        # Invert prices to implied vols
        mlsv_ivs = np.zeros((nT, nK))
        for i, T in enumerate(T_grid):
            for j, K in enumerate(strikes_abs):
                mlsv_ivs[i, j] = price_to_iv(mlsv_prices[i, j], S0, K, T)
                
        if device == "cuda":
            torch.cuda.synchronize()
        t_elapsed = time.time() - t0
        results["MLSV"] = {
            "iv_surface": mlsv_ivs,
            "prices": mlsv_prices,
            "status": "Success",
            "time": t_elapsed,
            "params": {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}
        }
    except Exception as e:
        results["MLSV"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 5. SABR Model (Hagan Lognormal)
    # -----------------------------------------------------------------------
    print("[5/8] Validating SABR Lognormal (Hagan 2002 Lognormal)...")
    t0 = time.time()
    try:
        sabr_log_ivs = sabr_iv_surface(F=S0, T_grid=T_grid, k_grid=K_grid, alpha=0.2, beta=0.5, rho=-0.5, nu=0.4, iv_type="lognormal")
        t_elapsed = time.time() - t0
        results["SABR Lognormal"] = {
            "iv_surface": sabr_log_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": {"F": S0, "alpha": 0.2, "beta": 0.5, "rho": -0.5, "nu": 0.4}
        }
    except Exception as e:
        results["SABR Lognormal"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 6. Displaced SABR (Rates normal/lognormal shifted)
    # -----------------------------------------------------------------------
    print("[6/8] Validating Displaced SABR (Shifted Lognormal Swaption)...")
    t0 = time.time()
    try:
        rates_engine = RatesSABREngine()
        # parameters
        F_rate = 0.02
        shift = 0.01
        alpha_rate = 0.05
        beta_rate = 0.5
        rho_rate = -0.3
        nu_rate = 0.3
        
        # Displaced strikes relative to rates
        strikes_rate = F_rate * np.exp(K_grid)
        
        # Compute displaced SABR implied vol surface (lognormal shifted representation)
        disp_sabr_ivs = np.zeros((nT, nK))
        for i, T in enumerate(T_grid):
            for j, K in enumerate(strikes_rate):
                disp_sabr_ivs[i, j] = rates_engine.displaced_sabr_vol(
                    F=F_rate, K=K, T=T, alpha=alpha_rate, beta=beta_rate, rho=rho_rate, nu=nu_rate, shift=shift, vol_type='lognormal'
                )
                
        t_elapsed = time.time() - t0
        # For arbitrage testing of rates surface, S0 is equivalent to F_rate
        results["Displaced SABR"] = {
            "iv_surface": disp_sabr_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": {"F": F_rate, "shift": shift, "alpha": alpha_rate, "beta": beta_rate, "rho": rho_rate, "nu": nu_rate},
            "S0": F_rate + shift,  # Shifted spot representation
            "K_grid": np.log((strikes_rate + shift) / (F_rate + shift)) # Shifted log-moneyness
        }
    except Exception as e:
        results["Displaced SABR"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 7. SSVI Model (Gatheral & Jacquier Power-Law)
    # -----------------------------------------------------------------------
    print("[7/8] Validating SSVI (Analytical Power-Law Variance)...")
    t0 = time.time()
    try:
        theta_grid = 0.04 * T_grid # ATM variance linear in T
        ssvi_ivs = ssvi_iv_surface(T_grid=T_grid, k_grid=K_grid, theta_grid=theta_grid, rho=-0.5, eta=1.5, gamma=0.5)
        t_elapsed = time.time() - t0
        results["SSVI"] = {
            "iv_surface": ssvi_ivs,
            "status": "Success",
            "time": t_elapsed,
            "params": {"rho": -0.5, "eta": 1.5, "gamma": 0.5}
        }
    except Exception as e:
        results["SSVI"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    # -----------------------------------------------------------------------
    # 8. Schwartz-Smith Model (Two-Factor Commodity)
    # -----------------------------------------------------------------------
    print("[8/8] Validating Schwartz-Smith (Black-76 Spot Option mapping)...")
    t0 = time.time()
    try:
        ss_engine = SchwartzSmithEngine(kappa=1.2, mu_y=0.05, sigma_x=0.3, sigma_y=0.15, rho_xy=-0.3)
        ss_prices = np.zeros((nT, nK))
        ss_ivs = np.zeros((nT, nK))
        
        for i, T in enumerate(T_grid):
            for j, k in enumerate(K_grid):
                K = S0 * np.exp(k)
                price = ss_engine.price_option(spot=S0, strike=K, maturity=T, risk_free_rate=0.0, is_call=True)
                ss_prices[i, j] = price
                ss_ivs[i, j] = price_to_iv(price, S0, K, T)
                
        t_elapsed = time.time() - t0
        results["Schwartz-Smith"] = {
            "iv_surface": ss_ivs,
            "prices": ss_prices,
            "status": "Success",
            "time": t_elapsed,
            "params": {"kappa": 1.2, "mu_y": 0.05, "sigma_x": 0.3, "sigma_y": 0.15, "rho_xy": -0.3}
        }
    except Exception as e:
        results["Schwartz-Smith"] = {"status": f"Failed: {str(e)}", "time": time.time() - t0}
        print(f"Error: {e}")

    print("\n" + "=" * 80)
    print(" PROGRAMMATIC NO-ARBITRAGE AND MATHEMATICAL CONSISTENCY TESTS")
    print("=" * 80)
    
    # Now run programmatic validation on generated surfaces
    for model_name, res in results.items():
        print(f"\nModel: {model_name}")
        if res["status"] != "Success":
            print(f"  Status: {res['status']}")
            continue
            
        iv_surface = res["iv_surface"]
        
        # 1. Positivity and Finite Checks
        nan_count = np.isnan(iv_surface).sum()
        inf_count = np.isinf(iv_surface).sum()
        min_vol = np.nanmin(iv_surface)
        max_vol = np.nanmax(iv_surface)
        
        print(f"  IV Grid Range: [{min_vol*100:.2f}%, {max_vol*100:.2f}%]")
        print(f"  NaN Count: {nan_count} | Inf Count: {inf_count}")
        
        # We need custom moneyness/spot if displaced
        k_eval = res.get("K_grid", K_grid)
        s_eval = res.get("S0", S0)
        
        # 2. Arbitrage Checks via deepvol.mrm.arbitrage
        arb_res = check_arbitrage(iv_surface, k_eval, T_grid, S=s_eval)
        
        calendar_arb = arb_res["calendar"]["has_arbitrage"]
        butterfly_durr = arb_res["butterfly_durrleman"]["has_arbitrage"]
        butterfly_price = arb_res["butterfly_price"]["has_arbitrage"]
        
        print(f"  Calendar Spread Arbitrage: {'[VIOLATION DETECTED]' if calendar_arb else '[CLEAN]'}")
        print(f"  Durrleman Butterfly Arbitrage: {'[VIOLATION DETECTED]' if butterfly_durr else '[CLEAN]'}")
        print(f"  Price Convexity Butterfly Arbitrage: {'[VIOLATION DETECTED]' if butterfly_price else '[CLEAN]'}")
        
        # 3. Call-Put Parity and Payoff Consistency Checks
        # Price call and put options from the IV surface
        calls = np.zeros((nT, nK))
        puts = np.zeros((nT, nK))
        for i, T in enumerate(T_grid):
            for j, k in enumerate(k_eval):
                K = s_eval * np.exp(k)
                calls[i, j] = bs_call_np(s_eval, K, T, iv_surface[i, j])
                puts[i, j] = bs_put_np(s_eval, K, T, iv_surface[i, j])
                
        # Put-Call Parity: C - P = S - K (since r=0, q=0)
        parity_violations = 0
        max_parity_diff = 0.0
        for i, T in enumerate(T_grid):
            for j, k in enumerate(k_eval):
                K = s_eval * np.exp(k)
                lhs = calls[i, j] - puts[i, j]
                rhs = s_eval - K
                diff = abs(lhs - rhs)
                max_parity_diff = max(max_parity_diff, diff)
                if diff > 1e-10:
                    parity_violations += 1
                    
        # Price Positivity and Monotonicity checks
        price_neg_count = np.sum(calls < 0.0) + np.sum(puts < 0.0)
        intrinsic_neg_count = 0
        for i, T in enumerate(T_grid):
            for j, k in enumerate(k_eval):
                K = s_eval * np.exp(k)
                if calls[i, j] < np.maximum(s_eval - K, 0.0) - 1e-10:
                    intrinsic_neg_count += 1
                    
        # Monotonicity in Strikes: Call price must be strictly decreasing in K
        monotonicity_violations = 0
        for i in range(nT):
            diffs = np.diff(calls[i])
            # call prices should decrease as K increases, so diffs should be negative
            violations = diffs > 1e-8
            monotonicity_violations += np.sum(violations)
            
        print(f"  Price Positivity: {'[VIOLATION]' if price_neg_count > 0 else '[PASS]'}")
        print(f"  Call Intrinsic Value Bound: {'[VIOLATION]' if intrinsic_neg_count > 0 else '[PASS]'}")
        print(f"  Strike Monotonicity: {'[VIOLATION]' if monotonicity_violations > 0 else '[PASS]'}")
        print(f"  Put-Call Parity Max Residual: {max_parity_diff:.2e} ({'[PASS]' if max_parity_diff < 1e-10 else '[FAIL]'})")
        
        # Save validation results back to dictionary
        res["metrics"] = {
            "calendar_arb": calendar_arb,
            "butterfly_dur": butterfly_durr,
            "butterfly_price": butterfly_price,
            "price_neg_count": price_neg_count,
            "intrinsic_neg": intrinsic_neg_count,
            "monotonicity_violations": monotonicity_violations,
            "max_parity_diff": max_parity_diff
        }
        
    print("\n" + "=" * 80)
    print(" SUMMARY OF MATH CORRECTNESS METRICS")
    print("=" * 80)
    print(f"{'Model Name':<20} | {'Status':<8} | {'Time (s)':<8} | {'Cal Arb':<7} | {'But Arb':<7} | {'Parity':<7} | {'Monotonic':<9}")
    print("-" * 80)
    for model_name, res in results.items():
        if res["status"] != "Success":
            print(f"{model_name:<20} | {'FAILED':<8} | {res['time']:<8.3f} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<9}")
        else:
            m = res["metrics"]
            cal_str = "[FAIL]" if m["calendar_arb"] else "[OK]"
            but_str = "[FAIL]" if (m["butterfly_dur"] or m["butterfly_price"]) else "[OK]"
            par_str = "[FAIL]" if m["max_parity_diff"] >= 1e-10 else "[OK]"
            mon_str = "[FAIL]" if m["monotonicity_violations"] > 0 else "[OK]"
            print(f"{model_name:<20} | {'SUCCESS':<8} | {res['time']:<8.3f} | {cal_str:<7} | {but_str:<7} | {par_str:<7} | {mon_str:<9}")
    print("=" * 80)

if __name__ == "__main__":
    run_math_validation()

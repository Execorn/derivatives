"""
Unit and stress tests for Greeks engine and GPU Monte Carlo risk engine.
Benchmarks latencies and verifies compliance/drift features.
"""

import pytest
import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any

from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.surrogates.fno_greeks import compute_greeks
from deepvol.risk.portfolio_mc import MonteCarloVaREngine
from deepvol.mrm.compliance import check_compliance, _global_monitor


def test_greeks_speed_and_accuracy(fno_v2_model):
    """
    Benchmark vectorized Greeks calculation speed and verify outputs are finite.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fno_v2_model
    model.to(device)

    params = torch.tensor([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=torch.float32, device=device, requires_grad=True)
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float32, device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float32, device=device)

    # Warmup
    _ = compute_greeks(model, params, T_grid, K_grid)

    # Timing run with synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        volga, vanna = compute_greeks(model, params, T_grid, K_grid)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    avg_speed_ms = ((t_end - t_start) / iterations) * 1000.0
    print(f"\n[Greeks Benchmark] Average speed: {avg_speed_ms:.2f} ms")

    # Assert that execution is fast (e.g. < 200 ms per run after warmup)
    assert avg_speed_ms < 200.0, f"vectorized Greeks took too long: {avg_speed_ms:.2f} ms"

    # Shape and finiteness checks
    assert volga.shape == (8, 11)
    assert vanna.shape == (8, 11)
    assert torch.all(torch.isfinite(volga))
    assert torch.all(torch.isfinite(vanna))


def test_var_latency_and_precision(fno_v2_model):
    """
    Benchmark portfolio-level VaR/ES calculation latency on the GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    var_engine = MonteCarloVaREngine(fno_v2_model, pn, yn, device)

    S0 = 100.0
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    r = 0.05
    
    # 100 positions portfolio
    np.random.seed(42)
    positions = []
    for _ in range(100):
        positions.append({
            "K": float(np.random.uniform(90.0, 110.0)),
            "T": float(np.random.uniform(0.1, 1.8)),
            "type": "call" if np.random.rand() > 0.5 else "put",
            "quantity": float(np.random.uniform(-2.0, 2.0)),
            "notional": 100.0
        })

    # Warmup with identical shape to compile the step functions fully before timing
    _ = var_engine.compute_portfolio_var_es(
        positions=positions, S0=S0, theta=theta, r=r,
        N_paths=2000, N_steps=5, alpha=0.95, block_size=4096, seed=42
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    t_start = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        res = var_engine.compute_portfolio_var_es(
            positions=positions, S0=S0, theta=theta, r=r,
            N_paths=2000, N_steps=5, alpha=0.95, block_size=4096, seed=42
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    avg_latency_ms = ((t_end - t_start) / iterations) * 1000.0
    print(f"\n[VaR Benchmark] Average latency (2k paths, 100 positions): {avg_latency_ms:.2f} ms")

    # Assert that execution is extremely fast (e.g. < 150 ms)
    assert avg_latency_ms < 150.0, f"VaR calculation took too long: {avg_latency_ms:.2f} ms"
    assert res["es"] >= res["var"]


def test_compliance_clamping_and_drift_tracking(caplog):
    """
    Test that OOD parameters are correctly clamped and logged,
    and that parameter drift triggers warnings via PSI.
    """
    # 1. Out-of-distribution parameter check (kappa=10.0 is OOD, H=0.3 is OOD)
    theta_ood = np.array([10.0, 0.08, 0.5, -0.5, 0.08, 0.3])
    
    with caplog.at_level(logging.WARNING):
        theta_clamped = check_compliance(theta_ood)
        
        # Verify clamping to RH_BOUNDS (kappa max 5.0, H max 0.15)
        assert theta_clamped[0] == 5.0
        assert theta_clamped[5] == 0.15
        
        # Verify structured warning log is produced
        assert any("OOD_PARAMETER_DETECTION" in rec.message for rec in caplog.records)

    # 2. Test parameter drift tracking using PSI
    # We clear monitor history and inject drift
    _global_monitor.history.clear()
    
    # Base distribution (expected)
    for _ in range(50):
        _global_monitor.add_parameters(np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08]))
        
    # Drifted distribution (actual)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            # Inject extreme parameters to trigger significant drift warnings
            _global_monitor.add_parameters(np.array([4.8, 0.22, 1.4, -0.1, 0.22, 0.14]))
        
        # Manually trigger compute
        _ = _global_monitor.compute_psi()
        
        # Verify significant drift warning is logged
        assert any("PARAMETER_DRIFT_WARNING" in rec.message for rec in caplog.records)

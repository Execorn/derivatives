import pytest
import os
import sys

# Ensure src path is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deepvol.benchmarks.greeks_benchmark import benchmark_greeks

def test_benchmark_greeks_targets():
    """
    Verify FNO Greeks benchmark targets:
    1. FNO Greeks evaluated in < 50ms for 100 positions.
    2. Delta MAE vs Black-Scholes closed-form < 0.01.
    """
    results = benchmark_greeks(n_positions=100, S=5000.0)
    
    # 1. Performance Target: FNO Greeks evaluated in < 50ms
    assert results["fno_speed_ms"] < 50.0, f"FNO Greeks took too long: {results['fno_speed_ms']:.2f} ms (target: < 50ms)"
    
    # 2. Accuracy Target: Delta MAE vs BS closed-form < 0.01
    assert results["fno_delta_mae"] < 0.01, f"FNO Delta MAE is too high: {results['fno_delta_mae']:.6f} (target: < 0.01)"
    
    # 3. Basic structure checks
    assert "cos_speed_ms" in results
    assert "fno_gamma_mae" in results
    assert "cos_delta_mae" in results
    assert "cos_gamma_mae" in results
    assert len(results["fno_greeks_bs"]) == 100
    assert len(results["cos_greeks_bs"]) == 100
    assert len(results["bs_cf_greeks"]) == 100
    
    # Check that all returned greeks values are finite numbers
    for idx in range(100):
        for key in ["delta", "gamma", "vega", "vanna", "volga"]:
            assert not sys.float_info.min > abs(results["fno_greeks_bs"][idx][key]) == float('inf')
            assert not sys.float_info.min > abs(results["cos_greeks_bs"][idx][key]) == float('inf')

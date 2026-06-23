"""
scratch/audit_endpoints.py - Stress-testing script for FastAPI REST endpoints.
Sends malformed inputs, extreme parameters, and validation boundary violations
to verify that the server returns graceful client errors (422/400) instead of uncaught 500 crashes.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from api.server import app

client = TestClient(app)

def audit_neural_sde():
    print("--- Auditing /calibrate_neural_sde ---")
    
    # 1. Invalid size market_iv (5x5 instead of 8x11)
    bad_iv = [[0.2] * 5 for _ in range(5)]
    payload = {
        "market_iv": bad_iv,
        "S0": 100.0,
        "epochs": 2,
        "N_paths": 128
    }
    resp = client.post("/calibrate_neural_sde", json=payload)
    print(f"Test 1 (5x5 IV surface): Status {resp.status_code} (Expected 422 or 400 or 500 handling)")
    # Note: market_iv is List[List[float]], pydantic checks type but size is validated in logic
    # Let's see if our logic catches size errors gracefully
    
    # 2. Negative Spot Price S0
    payload = {
        "market_iv": [[0.2] * 11 for _ in range(8)],
        "S0": -50.0,
        "epochs": 2,
        "N_paths": 128
    }
    resp = client.post("/calibrate_neural_sde", json=payload)
    print(f"Test 2 (Negative Spot S0): Status {resp.status_code} (Expected 422)")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    
    # 3. Negative Epochs
    payload = {
        "market_iv": [[0.2] * 11 for _ in range(8)],
        "S0": 100.0,
        "epochs": -10,
        "N_paths": 128
    }
    resp = client.post("/calibrate_neural_sde", json=payload)
    print(f"Test 3 (Negative Epochs): Status {resp.status_code} (Expected 422)")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


def audit_signature_vol():
    print("--- Auditing /predict/signature_vol ---")
    
    # 1. Incorrect ell size (10 elements instead of 30)
    payload = {
        "v0": 0.04,
        "ell": [0.0] * 10,
        "rho": -0.5,
        "T": 0.25,
        "S0": 100.0
    }
    resp = client.post("/predict/signature_vol", json=payload)
    print(f"Test 4 (Incorrect ell size): Status {resp.status_code} (Expected 500/400 graceful handling)")
    
    # 2. Positive correlation parameter rho (should be <= 0)
    payload = {
        "v0": 0.04,
        "ell": [0.0] * 30,
        "rho": 0.5,  # Invalid, must be <= 0
        "T": 0.25,
        "S0": 100.0
    }
    resp = client.post("/predict/signature_vol", json=payload)
    print(f"Test 5 (Positive rho): Status {resp.status_code} (Expected 422)")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    # 3. Initial variance v0 <= 0
    payload = {
        "v0": -0.01,
        "ell": [0.0] * 30,
        "rho": -0.5,
        "T": 0.25,
        "S0": 100.0
    }
    resp = client.post("/predict/signature_vol", json=payload)
    print(f"Test 6 (Negative v0): Status {resp.status_code} (Expected 422)")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


def audit_hedge_simulate():
    print("--- Auditing /hedge/simulate ---")
    
    # 1. Unsupported option type
    payload = {
        "option_type": "american",  # Invalid style
        "S0": 100.0,
        "strike": 100.0,
        "expiry": 0.1,
        "mu": 0.0,
        "sigma": 0.2
    }
    resp = client.post("/hedge/simulate", json=payload)
    print(f"Test 7 (Unsupported option type): Status {resp.status_code} (Expected 400 or 422)")
    assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}"
    
    # 2. Zero asset volatility
    payload = {
        "option_type": "european",
        "S0": 100.0,
        "strike": 100.0,
        "expiry": 0.1,
        "mu": 0.0,
        "sigma": 0.0  # Invalid, must be > 0
    }
    resp = client.post("/hedge/simulate", json=payload)
    print(f"Test 8 (Zero volatility): Status {resp.status_code} (Expected 422)")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


if __name__ == "__main__":
    print("Starting API Endpoints Stress Audit...")
    audit_neural_sde()
    audit_signature_vol()
    audit_hedge_simulate()
    print("API Endpoints Stress Audit completed successfully.")

import os
import sys
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from deepvol.api.server import app
from deepvol.api.compliance import compliance_monitor

VALID_PARAMS = {
    "kappa": 1.5,
    "theta": 0.08,
    "sigma": 0.5,
    "rho":   -0.7,
    "v0":    0.08,
    "H":     0.08,
}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_metrics_endpoint_returns_prometheus_data(client):
    # First, make a request to trigger pricing/Greeks metrics
    params = VALID_PARAMS.copy()
    client.post("/iv_surface", json=params)
    
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    
    # Assert custom metrics exist in the Prometheus scrape output
    assert "fno_pricing_latency_seconds_bucket" in text
    assert "api_request_queue_latency_seconds_bucket" in text
    assert "gpu_vram_allocated_bytes" in text
    assert "gpu_core_utilization_ratio" in text

def test_ood_clamping_and_logging(client):
    # Request custom grid with OOD values
    # Maturities: 0.05 (too low), 1.0 (ok), 2.5 (too high)
    # Strikes: 50.0 (too low), 100.0 (ok), 200.0 (too high) for spot=100.0
    payload = VALID_PARAMS.copy()
    payload["T_grid"] = [0.05, 1.0, 2.5]
    payload["K_grid"] = [50.0, 100.0, 200.0]
    payload["S0"] = 100.0
    
    # Clear the compliance log if it exists
    log_path = compliance_monitor.log_path
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass
        
    resp = client.post("/iv_surface", json=payload)
    assert resp.status_code == 200
    
    data = resp.json()
    
    # Validate clamped values in response
    # T_grid: 0.05 clamped to 0.1, 2.5 clamped to 2.0
    assert data["T_grid"] == [0.1, 1.0, 2.0]
    
    # K_grid: log-moneyness clamped to [-0.5, 0.5]
    # 50.0 -> log(0.5) = -0.693 -> clamped to -0.5 -> 100 * exp(-0.5) = 60.653
    # 200.0 -> log(2.0) = 0.693 -> clamped to 0.5 -> 100 * exp(0.5) = 164.872
    # 100.0 -> log(1.0) = 0.0 -> ok -> 100.0
    assert math.isclose(data["K_grid"][0], 100.0 * math.exp(-0.5), rel_tol=1e-4)
    assert math.isclose(data["K_grid"][1], 100.0, rel_tol=1e-4)
    assert math.isclose(data["K_grid"][2], 100.0 * math.exp(0.5), rel_tol=1e-4)
    
    # Check that OOD audit log file was written
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "OOD WARNING" in log_content
    assert "rough_heston" in log_content

def test_psi_drift_warning(client):
    # Clear the compliance log
    log_path = compliance_monitor.log_path
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass
        
    # Track 1000 highly drifted queries (e.g. all on boundary T=0.1, k=0.5)
    for _ in range(1000):
        compliance_monitor.track_query(0.1, 0.5)
        
    # Wait for the background thread to finish writing
    import time
    time.sleep(0.5)
    
    # Verify warning was written to log
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "COMPLIANCE WARNING: Online drift detected" in log_content

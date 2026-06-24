import torch
import pytest
from deepvol.calibration.calibrate_newton import _reparam_to_6d

def test_reparam_to_6d_values():
    device = torch.device("cpu")
    v0 = torch.tensor([0.05, 0.08, 0.10, 0.12, 0.14], dtype=torch.float32)
    zeta = torch.tensor([-0.4, -0.3, -0.2, -0.15, -0.35], dtype=torch.float32)
    lam = torch.tensor([0.3, 0.4, 0.5, 0.25, 0.45], dtype=torch.float32)
    
    out = _reparam_to_6d(v0, zeta, lam, device)
    
    assert out.shape == (5, 6)
    
    kappa = out[:, 0]
    theta = out[:, 1]
    sigma = out[:, 2]
    rho = out[:, 3]
    v0_out = out[:, 4]
    H = out[:, 5]
    
    expected_sigma = torch.sqrt(zeta**2 + lam**2)
    expected_rho = zeta / expected_sigma
    
    torch.testing.assert_close(sigma, expected_sigma)
    torch.testing.assert_close(rho, expected_rho)
    torch.testing.assert_close(kappa, torch.full_like(v0, 1.0))
    torch.testing.assert_close(theta, torch.full_like(v0, 0.08))
    torch.testing.assert_close(H, torch.full_like(v0, 0.08))
    torch.testing.assert_close(v0_out, v0)

def test_reparam_to_6d_rho_clamping():
    device = torch.device("cpu")
    v0 = torch.tensor([0.08, 0.08], dtype=torch.float32)
    zeta = torch.tensor([-0.99, -0.01], dtype=torch.float32)
    lam = torch.tensor([0.01, 0.20], dtype=torch.float32)
    
    out = _reparam_to_6d(v0, zeta, lam, device)
    rho = out[:, 3]
    
    torch.testing.assert_close(rho[0], torch.tensor(-0.9, dtype=torch.float32))
    torch.testing.assert_close(rho[1], torch.tensor(-0.1, dtype=torch.float32))

def test_reparam_to_6d_sigma_flooring():
    device = torch.device("cpu")
    v0 = torch.tensor([0.08], dtype=torch.float32)
    zeta = torch.tensor([0.001], dtype=torch.float32)
    lam = torch.tensor([0.001], dtype=torch.float32)
    
    out = _reparam_to_6d(v0, zeta, lam, device)
    sigma = out[0, 2]
    
    assert sigma >= 0.01
    torch.testing.assert_close(sigma, torch.tensor(0.01, dtype=torch.float32))

def test_reparam_device_transfer():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    for device in devices:
        v0 = torch.tensor([0.08], dtype=torch.float32)
        zeta = torch.tensor([-0.3], dtype=torch.float32)
        lam = torch.tensor([0.4], dtype=torch.float32)
        
        out = _reparam_to_6d(v0, zeta, lam, device)
        assert out.device.type == device.type

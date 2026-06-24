import numpy as np
import pytest
import torch
from deepvol.models.local_vol import svi_slice, svi_to_lv_surface, check_arbitrage_free
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

def test_flat_surface_identity():
    # If the implied volatility surface is completely flat, local volatility = implied volatility
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # Let flat IV be 0.20 (total variance w = 0.20^2 * T)
    w0_val = 0.20 ** 2
    svi_params = np.zeros((len(T_grid), 5))
    for i, T in enumerate(T_grid):
        w0 = w0_val * T
        # Set a = w0, b = 0, rho = 0, m = 0, sigma = 0.1
        svi_params[i] = [w0, 0.0, 0.0, 0.0, 0.1]
        
    # Check that flat surface passes arbitrage check
    assert check_arbitrage_free(T_grid, K_grid, svi_params) is True
    
    # Compute local volatility surface
    lv_surf = svi_to_lv_surface(T_grid, K_grid, svi_params)
    
    # Since IV is flat at 0.20, local vol should be flat at 0.20
    # Allow a small numerical tolerance due to finite differences and bilinear interpolation
    np.testing.assert_allclose(lv_surf, 0.20, rtol=1e-2, atol=1e-2)

def test_arbitrage_checks():
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # Case 1: Valid SVI surface (flat 0.20)
    svi_valid = np.zeros((len(T_grid), 5))
    for i, T in enumerate(T_grid):
        svi_valid[i] = [0.04 * T, 0.0, 0.0, 0.0, 0.1]
    assert check_arbitrage_free(T_grid, K_grid, svi_valid) is True
    
    # Case 2: Calendar spread arbitrage (w decreases in T)
    svi_cal_arb = svi_valid.copy()
    # Swap slice 2 and slice 3 variance
    svi_cal_arb[2, 0] = svi_valid[1, 0]
    svi_cal_arb[1, 0] = svi_valid[2, 0]
    assert check_arbitrage_free(T_grid, K_grid, svi_cal_arb) is False
    
    # Case 3: Butterfly arbitrage (negative density)
    svi_butt_arb = svi_valid.copy()
    # Large b and negative rho at slice 0 (causes negative risk-neutral density in wing)
    svi_butt_arb[0] = [0.004, 1.5, -0.9, 0.0, 0.01]
    assert check_arbitrage_free(T_grid, K_grid, svi_butt_arb) is False

def test_fno_forward_pass():
    # Verify FNO model forward pass with param_dim=40
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MirrorPaddedFNO2d(param_dim=40).to(device)
    
    B = 4
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing='ij')
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, 8, 11, 2)
    spatial_tensor = torch.tensor(spatial, dtype=torch.float32, device=device).expand(B, -1, -1, -1)
    
    params_tensor = torch.randn(B, 40, device=device, requires_grad=True)
    
    pred = model(spatial_tensor, params_tensor)
    assert pred.shape == (B, 8, 11)
    
    # Check that gradient flows back to parameters
    loss = pred.sum()
    loss.backward()
    assert params_tensor.grad is not None
    assert not torch.isnan(params_tensor.grad).any()


def test_compute_local_vol_surface():
    from deepvol.calibration.calibrate_newton import compute_local_vol_surface
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # 40 SVI parameters representing flat 0.20
    svi_params = np.zeros((len(T_grid), 5))
    for i, T in enumerate(T_grid):
        svi_params[i] = [0.20**2 * T, 0.0, 0.0, 0.0, 0.1]
        
    # Test analytic computation
    lv_surf_analytic = compute_local_vol_surface(svi_params, T_grid, K_grid, use_fno=False)
    np.testing.assert_allclose(lv_surf_analytic, 0.20, rtol=1e-2, atol=1e-2)
    
    # Test FNO surrogate computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MirrorPaddedFNO2d(param_dim=40).to(device)
    model.load_state_dict(torch.load("artifacts/weights/fno_localvol_final_prod.pth", map_location=device, weights_only=True))
    model.eval()
    
    lv_surf_fno = compute_local_vol_surface(svi_params, T_grid, K_grid, use_fno=True, model=model)
    assert lv_surf_fno.shape == (len(T_grid), len(K_grid))


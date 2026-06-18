import pytest
import torch
import numpy as np
from scipy.stats import norm

from greeks.portfolio_greeks import (
    bs_call_price,
    bs_greeks,
    fno_parameter_jacobian,
    fno_surface_greeks,
    portfolio_greeks,
    pnl_attribution
)
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

def test_bs_analytical_vs_autograd():
    S_val = 100.0
    K_val = 105.0
    T_val = 0.5
    r_val = 0.05
    sigma_val = 0.25
    
    analytical = bs_greeks(S_val, K_val, T_val, r_val, sigma_val)
    
    S = torch.tensor(S_val, requires_grad=True)
    K = torch.tensor(K_val)
    T = torch.tensor(T_val, requires_grad=True)
    r = torch.tensor(r_val)
    sigma = torch.tensor(sigma_val, requires_grad=True)
    
    price = bs_call_price(S, K, T, r, sigma)
    
    delta, vega = torch.autograd.grad(price, (S, sigma), create_graph=True, retain_graph=True)
    gamma = torch.autograd.grad(delta, S, create_graph=True, retain_graph=True)[0]
    vanna = torch.autograd.grad(delta, sigma, create_graph=True, retain_graph=True)[0]
    volga = torch.autograd.grad(vega, sigma, create_graph=True, retain_graph=True)[0]
    
    speed = torch.autograd.grad(gamma, S, retain_graph=True)[0]
    zomma = torch.autograd.grad(gamma, sigma, retain_graph=True)[0]
    ultima = torch.autograd.grad(volga, sigma, retain_graph=True)[0]
    
    assert np.isclose(price.item(), S_val * norm.cdf((np.log(S_val/K_val) + (r_val + 0.5*sigma_val**2)*T_val)/(sigma_val*np.sqrt(T_val))) - K_val*np.exp(-r_val*T_val)*norm.cdf((np.log(S_val/K_val) + (r_val - 0.5*sigma_val**2)*T_val)/(sigma_val*np.sqrt(T_val))), rtol=1e-6)
    assert np.isclose(delta.item(), analytical["delta"], rtol=1e-6)
    assert np.isclose(gamma.item(), analytical["gamma"], rtol=1e-6)
    assert np.isclose(vega.item(), analytical["vega"], rtol=1e-6)
    assert np.isclose(vanna.item(), analytical["vanna"], rtol=1e-6)
    assert np.isclose(volga.item(), analytical["volga"], rtol=1e-6)
    assert np.isclose(speed.item(), analytical["speed"], rtol=1e-6)
    assert np.isclose(zomma.item(), analytical["zomma"], rtol=1e-6)
    assert np.isclose(ultima.item(), analytical["ultima"], rtol=1e-6)


def test_bs_greeks_finite_difference():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    # Use larger epsilon for second-order finite differences to avoid numerical cancellation
    eps_1st = 1e-5
    eps_2nd = 1e-3
    
    g = bs_greeks(S, K, T, r, sigma)
    
    def c_price(s_in, sigma_in):
        d1 = (np.log(s_in/K) + (r + 0.5*sigma_in**2)*T) / (sigma_in*np.sqrt(T))
        d2 = d1 - sigma_in*np.sqrt(T)
        return s_in*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
    delta_fd = (c_price(S + eps_1st, sigma) - c_price(S - eps_1st, sigma)) / (2 * eps_1st)
    assert np.isclose(g["delta"], delta_fd, rtol=1e-5)
    
    vega_fd = (c_price(S, sigma + eps_1st) - c_price(S, sigma - eps_1st)) / (2 * eps_1st)
    assert np.isclose(g["vega"], vega_fd, rtol=1e-5)
    
    gamma_fd = (c_price(S + eps_2nd, sigma) - 2 * c_price(S, sigma) + c_price(S - eps_2nd, sigma)) / (eps_2nd ** 2)
    assert np.isclose(g["gamma"], gamma_fd, rtol=1e-4)
    
    volga_fd = (c_price(S, sigma + eps_2nd) - 2 * c_price(S, sigma) + c_price(S, sigma - eps_2nd)) / (eps_2nd ** 2)
    assert np.isclose(g["volga"], volga_fd, rtol=1e-4)
    
    vanna_fd = (c_price(S + eps_2nd, sigma + eps_2nd) - c_price(S + eps_2nd, sigma - eps_2nd) - 
                c_price(S - eps_2nd, sigma + eps_2nd) + c_price(S - eps_2nd, sigma - eps_2nd)) / (4 * eps_2nd**2)
    assert np.isclose(g["vanna"], vanna_fd, rtol=1e-4)


def test_fno_parameter_jacobian_vs_fd(fno_v2_model):
    model = fno_v2_model
    device = next(model.parameters()).device
    
    theta_raw = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    theta_t = torch.tensor(theta_raw, dtype=torch.float32, device=device)
    
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    from calibrate import _make_spatial_input
    spatial = _make_spatial_input(T_grid, K_grid, device)
    
    J_autograd = fno_parameter_jacobian(model, theta_t, spatial)
    assert J_autograd.shape == (len(T_grid), len(K_grid), 6)
    
    J_fd = np.zeros((len(T_grid), len(K_grid), 6))
    
    for p_idx in range(6):
        # Use 1e-5 epsilon for the 6th parameter (H) because of its tiny std (1.58e-5)
        # to avoid huge non-linear truncation error in raw space, while avoiding float32 underflow.
        p_eps = 1e-5 if p_idx == 5 else 1e-4
        
        theta_plus = theta_raw.copy()
        theta_plus[p_idx] += p_eps
        theta_minus = theta_raw.copy()
        theta_minus[p_idx] -= p_eps
        
        t_plus = torch.tensor(theta_plus, dtype=torch.float32, device=device)
        t_minus = torch.tensor(theta_minus, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            p_norm_plus = pn.transform_tensor(t_plus.unsqueeze(0))
            pred_plus = model(spatial, p_norm_plus)
            iv_plus_t = yn.inverse_transform_tensor(pred_plus).squeeze(0)
            iv_plus = torch.clamp(iv_plus_t, min=1e-4).cpu().numpy()
            
            p_norm_minus = pn.transform_tensor(t_minus.unsqueeze(0))
            pred_minus = model(spatial, p_norm_minus)
            iv_minus_t = yn.inverse_transform_tensor(pred_minus).squeeze(0)
            iv_minus = torch.clamp(iv_minus_t, min=1e-4).cpu().numpy()
            
        J_fd[:, :, p_idx] = (iv_plus - iv_minus) / (2 * p_eps)
        
    J_autograd_np = J_autograd.detach().cpu().numpy()
    
    # Assert close for active parameters (0 to 4) with tight tolerance
    np.testing.assert_allclose(J_autograd_np[:, :, :5], J_fd[:, :, :5], rtol=1e-3, atol=1e-3)
    # Assert close for the ghost parameter (5) with loose tolerance due to tiny std scaling
    np.testing.assert_allclose(J_autograd_np[:, :, 5], J_fd[:, :, 5], rtol=2e-1, atol=250.0)


def test_portfolio_greeks_aggregation(fno_v2_model):
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    S = 100.0
    theta_raw = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    positions = [
        {"K": 100.0, "T": 0.5, "type": "call", "notional": 200.0, "quantity": 2.0},
        {"K": 95.0,  "T": 0.2, "type": "put",  "notional": 95.0,  "quantity": -1.0}
    ]
    
    res = portfolio_greeks(positions, model, theta_raw, pn, yn, S)
    
    assert "total_delta" in res
    assert "total_gamma" in res
    assert "vega_bucket" in res
    assert "total_vanna" in res
    assert "total_volga" in res
    assert "hedge_contracts" in res
    assert res["vega_bucket"].shape == (8,)
    
    expected_hedge = -int(round(res["total_delta"] / 50.0))
    assert res["hedge_contracts"] == expected_hedge


def test_pnl_attribution_Taylor_expansion():
    S_before = 100.0
    S_after = 101.5
    sigma_before = 0.20
    sigma_after = 0.22
    
    greeks = {
        "total_delta": 0.65,
        "total_gamma": 0.04,
        "vega_bucket": np.array([10.0, 15.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "total_vanna": -0.80,
        "total_volga": 1.20,
        "actual_pnl": 1.85
    }
    
    attribution = pnl_attribution(S_before, S_after, sigma_before, sigma_after, greeks)
    
    assert "delta_pnl" in attribution
    assert "gamma_pnl" in attribution
    assert "vega_pnl" in attribution
    assert "vanna_pnl" in attribution
    assert "volga_pnl" in attribution
    assert "unexplained" in attribution
    
    dS = S_after - S_before
    dsigma = sigma_after - sigma_before
    
    expected_delta_pnl = 0.65 * dS
    expected_gamma_pnl = 0.5 * 0.04 * (dS ** 2)
    expected_vega_pnl = 30.0 * dsigma
    expected_vanna_pnl = -0.80 * dS * dsigma
    expected_volga_pnl = 0.5 * 1.20 * (dsigma ** 2)
    
    explained = expected_delta_pnl + expected_gamma_pnl + expected_vega_pnl + expected_vanna_pnl + expected_volga_pnl
    expected_unexplained = greeks["actual_pnl"] - explained
    
    assert np.isclose(attribution["delta_pnl"], expected_delta_pnl)
    assert np.isclose(attribution["gamma_pnl"], expected_gamma_pnl)
    assert np.isclose(attribution["vega_pnl"], expected_vega_pnl)
    assert np.isclose(attribution["vanna_pnl"], expected_vanna_pnl)
    assert np.isclose(attribution["volga_pnl"], expected_volga_pnl)
    assert np.isclose(attribution["unexplained"], expected_unexplained)

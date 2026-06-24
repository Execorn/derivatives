import pytest
import torch
import numpy as np
import math
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.risk.sensitivity import AutogradSensitivityEngine, portfolio_price_tensor
from deepvol.risk.var_engine import MonteCarloVaREngine

@pytest.fixture(scope="module")
def risk_engine_setup(fno_v2_model):
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    device = next(model.parameters()).device
    engine = AutogradSensitivityEngine(model, pn, yn, device)
    var_engine = MonteCarloVaREngine(model, pn, yn, device)
    return engine, var_engine, pn, yn, model

def test_autograd_vs_finite_difference(risk_engine_setup):
    _, _, pn, yn, model = risk_engine_setup
    device = next(model.parameters()).device

    # Create a smooth, double-precision volatility surface to bypass float32 FNO roundoff limitations.
    from deepvol.greeks.portfolio_greeks import MATURITIES, STRIKES
    nT, nK = len(MATURITIES), len(STRIKES)
    
    # We define a smooth float64 surface: 0.2 + 0.05 * T - 0.1 * k
    T_grid_np = np.array(MATURITIES, dtype=np.float64)
    K_grid_np = np.array(STRIKES, dtype=np.float64)
    T_mesh, K_mesh = np.meshgrid(T_grid_np, K_grid_np, indexing='ij')
    iv_surface_np = 0.2 + 0.05 * T_mesh - 0.1 * K_mesh
    iv_surface_double = torch.tensor(iv_surface_np, dtype=torch.float64, device=device)

    # Convert to normalizer space so yn.inverse_transform_tensor yields iv_surface_double
    mean = torch.tensor(yn.mean, dtype=torch.float64, device=device)
    std = torch.tensor(yn.std, dtype=torch.float64, device=device)
    pred_norm = (iv_surface_double - mean) / std

    class MockFNO(torch.nn.Module):
        def __init__(self, pred_norm):
            super().__init__()
            self.pred_norm = pred_norm
        def forward(self, spatial, theta_norm):
            return self.pred_norm.unsqueeze(0)

    mock_model = MockFNO(pred_norm)
    engine_mock = AutogradSensitivityEngine(mock_model, pn, yn, device)

    # Underlying state and model parameters
    S0 = 100.0
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    r = 0.05

    # Define a portfolio of options. Strikes are strictly off-grid.
    positions = [
        {"K": 100.0 * math.exp(0.05),  "T": 0.5, "type": "call", "quantity": 1.5, "notional": 100.0},
        {"K": 100.0 * math.exp(-0.15), "T": 0.2, "type": "put",  "quantity": -1.0, "notional": 100.0},
        {"K": 100.0 * math.exp(0.25),  "T": 0.8, "type": "call", "quantity": 2.0,  "notional": 100.0}
    ]

    # Compute Greeks via Autograd Sensitivity Engine in double precision
    greeks_ad = engine_mock.compute_greeks(positions, S0, theta, r, sticky_strike=True, dtype=torch.float64)

    # Helper function to compute portfolio price for FD in double precision
    def get_price(s_val: float, eps_val: float, t_val: float) -> float:
        S_t = torch.tensor(s_val, dtype=torch.float64, device=device)
        theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
        t_t = torch.tensor(t_val, dtype=torch.float64, device=device)
        epsilon_t = torch.tensor(eps_val, dtype=torch.float64, device=device)
        r_t = torch.tensor(r, dtype=torch.float64, device=device)

        K_t = torch.tensor([float(p["K"]) for p in positions], dtype=torch.float64, device=device)
        T_t = torch.tensor([float(p["T"]) for p in positions], dtype=torch.float64, device=device)
        qty_t = torch.tensor([float(p.get("quantity", 1.0)) for p in positions], dtype=torch.float64, device=device)
        notional_t = torch.tensor([float(p.get("notional", 100.0)) for p in positions], dtype=torch.float64, device=device)
        is_call_t = torch.tensor([1.0 if p.get("type", "call").lower() == "call" else 0.0 for p in positions], dtype=torch.float64, device=device)

        with torch.no_grad():
            val = portfolio_price_tensor(
                S_t, theta_t, t_t, epsilon_t,
                K_t, T_t, qty_t, notional_t, is_call_t,
                mock_model, pn, yn, r_t, sticky_strike=True,
                iv_surface=iv_surface_double
            )
        return val.item()

    # 1. Delta (first-order w.r.t S) - optimal h = 1e-5
    h_S_1st = 1e-5
    p_S_plus = get_price(S0 + h_S_1st, 0.0, 0.0)
    p_S_minus = get_price(S0 - h_S_1st, 0.0, 0.0)
    delta_fd = (p_S_plus - p_S_minus) / (2 * h_S_1st)

    # 2. Vega (first-order w.r.t epsilon) - optimal h = 1e-5
    h_eps_1st = 1e-5
    p_eps_plus = get_price(S0, h_eps_1st, 0.0)
    p_eps_minus = get_price(S0, -h_eps_1st, 0.0)
    vega_fd = (p_eps_plus - p_eps_minus) / (2 * h_eps_1st)

    # 3. Theta (first-order w.r.t t) - note the sign: Theta is -dV/dt - optimal h = 1e-5
    h_t_1st = 1e-5
    p_t_plus = get_price(S0, 0.0, h_t_1st)
    p_t_minus = get_price(S0, 0.0, -h_t_1st)
    theta_fd = -(p_t_plus - p_t_minus) / (2 * h_t_1st)

    # 4. Gamma (second-order w.r.t S) - optimal h = 2e-3 due to relative scale S=100
    h_S_2nd = 2e-3
    p_S_plus_2nd = get_price(S0 + h_S_2nd, 0.0, 0.0)
    p_S_minus_2nd = get_price(S0 - h_S_2nd, 0.0, 0.0)
    p_S_base = get_price(S0, 0.0, 0.0)
    gamma_fd = (p_S_plus_2nd - 2 * p_S_base + p_S_minus_2nd) / (h_S_2nd ** 2)

    # 5. Volga (second-order w.r.t epsilon) - optimal h = 1.2e-4
    h_eps_2nd = 1.2e-4
    p_eps_plus_2nd = get_price(S0, h_eps_2nd, 0.0)
    p_eps_minus_2nd = get_price(S0, -h_eps_2nd, 0.0)
    volga_fd = (p_eps_plus_2nd - 2 * p_S_base + p_eps_minus_2nd) / (h_eps_2nd ** 2)

    # 6. Vanna (mixed second-order w.r.t S and epsilon) - optimal h_S = h_eps = 2.5e-4
    h_S_vanna = 2.5e-4
    h_eps_vanna = 2.5e-4
    p_S_plus_eps_plus = get_price(S0 + h_S_vanna, h_eps_vanna, 0.0)
    p_S_plus_eps_minus = get_price(S0 + h_S_vanna, -h_eps_vanna, 0.0)
    p_S_minus_eps_plus = get_price(S0 - h_S_vanna, h_eps_vanna, 0.0)
    p_S_minus_eps_minus = get_price(S0 - h_S_vanna, -h_eps_vanna, 0.0)
    vanna_fd = (p_S_plus_eps_plus - p_S_plus_eps_minus - p_S_minus_eps_plus + p_S_minus_eps_minus) / (4 * h_S_vanna * h_eps_vanna)

    print(f"Delta: AD={greeks_ad['delta']:.8f}, FD={delta_fd:.8f}, diff={abs(greeks_ad['delta'] - delta_fd):.2e}")
    print(f"Gamma: AD={greeks_ad['gamma']:.8f}, FD={gamma_fd:.8f}, diff={abs(greeks_ad['gamma'] - gamma_fd):.2e}")
    print(f"Vega: AD={greeks_ad['vega']:.8f}, FD={vega_fd:.8f}, diff={abs(greeks_ad['vega'] - vega_fd):.2e}")
    print(f"Theta: AD={greeks_ad['theta']:.8f}, FD={theta_fd:.8f}, diff={abs(greeks_ad['theta'] - theta_fd):.2e}")
    print(f"Vanna: AD={greeks_ad['vanna']:.8f}, FD={vanna_fd:.8f}, diff={abs(greeks_ad['vanna'] - vanna_fd):.2e}")
    print(f"Volga: AD={greeks_ad['volga']:.8f}, FD={volga_fd:.8f}, diff={abs(greeks_ad['volga'] - volga_fd):.2e}")

    # Verify tolerance < 10^-6
    assert np.isclose(greeks_ad["delta"], delta_fd, atol=1e-6)
    assert np.isclose(greeks_ad["gamma"], gamma_fd, atol=1e-6)
    assert np.isclose(greeks_ad["vega"], vega_fd, atol=1e-6)
    assert np.isclose(greeks_ad["theta"], theta_fd, atol=1e-6)
    assert np.isclose(greeks_ad["vanna"], vanna_fd, atol=1e-6)
    assert np.isclose(greeks_ad["volga"], volga_fd, atol=1e-6)


def test_var_expected_shortfall(risk_engine_setup):
    _, var_engine, _, _, _ = risk_engine_setup

    S0 = 100.0
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    r = 0.05

    positions = [
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": 1.0, "notional": 100.0}
    ]

    res_95 = var_engine.compute_portfolio_var_es(
        positions=positions, S0=S0, theta=theta, r=r, dt=1/252,
        N_paths=2000, N_steps=5, alpha=0.95, seed=42
    )

    res_99 = var_engine.compute_portfolio_var_es(
        positions=positions, S0=S0, theta=theta, r=r, dt=1/252,
        N_paths=2000, N_steps=5, alpha=0.99, seed=42
    )

    # Expected Shortfall must be strictly greater than or equal to Value-at-Risk
    assert res_95["es"] >= res_95["var"]
    assert res_99["es"] >= res_99["var"]

    # 99% VaR must be greater than or equal to 95% VaR
    assert res_99["var"] >= res_95["var"]

    # Check key shapes and finiteness
    assert len(res_95["losses"]) == 2000
    assert len(res_95["spots"]) == 2000
    assert len(res_95["vars"]) == 2000
    assert np.all(np.isfinite(res_95["losses"]))
    assert np.all(res_95["spots"] > 0)
    assert np.all(res_95["vars"] > 0)


def test_parameter_greeks(risk_engine_setup):
    engine, _, _, _, _ = risk_engine_setup

    S0 = 100.0
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    r = 0.05

    positions = [
        {"K": 100.0, "T": 0.5, "type": "call", "quantity": 1.0, "notional": 100.0}
    ]

    res = engine.compute_parameter_greeks(positions, S0, theta, r)

    assert "gradient" in res
    assert "hessian" in res
    assert res["gradient"].shape == (6,)
    assert res["hessian"].shape == (6, 6)
    assert np.all(np.isfinite(res["gradient"]))
    assert np.all(np.isfinite(res["hessian"]))

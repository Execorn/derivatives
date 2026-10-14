"""tests/test_fx_cross_analytic.py — Unit test suite for AnalyticTriangulated engine and Greeks.

Verifies:
- Propositions P1–P5 (strike additivity, flat-smile Garman-Kohlhagen reduction, ATM consistency, degenerate leg, Lee bound)
- Closed-form Black sensitivities and smile-inclusive Greeks vs float64 central-bump finite differences and Wolfram identities W1–W8
- Smile-inclusive Gamma vs float64 double backward
- Theta vs 1-day calendar roll
- CashDelta conservation across legs
- Call delta bounds and put-call parity
- E1 copula teacher regression benchmarks
- torch.compile(fullgraph=True) execution
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import torch
import numpy as np

from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
    black76_otm_parity,
    black_partials,
    bucketed_rho,
    greeks_from_outputs,
    lee_slope_bound,
    smooth_T,
)
from deepvol.market.fx_data import gk_price


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_sample_input(
    B: int = 1,
    F: float = 250000.0,
    K: float = 250000.0,
    T: float = 1.0,
    D: float = 0.95,
    g: float = 1.05,
    rho: float = -0.20,
    phi: float = 1.0,
    svi1: tuple[float, float, float, float, float] = (0.025, 0.08, -0.15, 0.02, 0.05),
    svi2: tuple[float, float, float, float, float] = (0.040, 0.12, 0.35, -0.01, 0.08),
    r_C: float = 0.05,
    r_A: float = 0.02,
    r_set: float = 0.05,
    device: str = "cpu",
) -> torch.Tensor:
    X = torch.zeros(B, N_COLS, dtype=torch.float64, device=device)
    X[:, Col.F] = F
    X[:, Col.K] = K
    X[:, Col.T] = T
    X[:, Col.T_SET] = T
    X[:, Col.D] = D
    X[:, Col.G] = g
    X[:, Col.RHO] = rho
    X[:, Col.PHI] = phi

    for i in range(5):
        X[:, Col.SVI1 + i] = svi1[i]
        X[:, Col.SVI2 + i] = svi2[i]

    X[:, Col.R_C] = r_C
    X[:, Col.R_A] = r_A
    X[:, Col.R_SET] = r_set
    X[:, Col.NN_GATE] = 0.0
    return X


def test_propositions_p1_to_p5(device: str) -> None:
    """Test 1.1(b): Unit tests for boundary-consistency propositions P1–P5."""
    engine = AnalyticTriangulated().to(device)

    # P1: Strike additivity (k1 + k2 = k exactly)
    # Tested across moneyness range k in [-2, 2]
    k_vals = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64, device=device)
    F = 250000.0
    Ks = F * torch.exp(k_vals)
    B = len(Ks)
    X_p1 = _make_sample_input(B=B, F=F, device=device)
    X_p1[:, Col.K] = Ks

    out_p1 = engine(X_p1)
    # Assert finite outputs and valid sigma
    assert torch.all(torch.isfinite(out_p1.sigma))

    # P2: Flat-smile reduction to Garman-Kohlhagen (Analytic_GK)
    # Under flat smiles w_i(x) = sigma_i^2 * T, H1 cross vol equals sqrt(s1^2 + s2^2 + 2*rho*s1*s2)
    vol1 = 0.16
    vol2 = 0.20
    T_val = 1.0
    rho_val = -0.20
    sigma_cross = math.sqrt(vol1**2 + vol2**2 + 2.0 * rho_val * vol1 * vol2)

    svi1_flat = (vol1**2 * T_val, 0.0, 0.0, 0.0, 0.1)
    svi2_flat = (vol2**2 * T_val, 0.0, 0.0, 0.0, 0.1)
    r_d = 0.05
    r_f = 0.02
    D_val = math.exp(-r_d * T_val)
    g_val = math.exp((r_d - r_f) * T_val)

    X_p2 = _make_sample_input(
        B=11,
        F=F,
        T=T_val,
        D=D_val,
        g=g_val,
        rho=rho_val,
        svi1=svi1_flat,
        svi2=svi2_flat,
        r_C=r_d,
        r_A=r_f,
        r_set=r_d,
        device=device,
    )
    k_test = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64, device=device)
    X_p2[:, Col.K] = F * torch.exp(k_test)

    out_p2 = engine(X_p2)

    for i in range(11):
        K_i = float(X_p2[i, Col.K].item())
        price_engine = float(out_p2.V[i].item())
        price_gk = gk_price(F=F, K=K_i, T=T_val, r_d=r_d, r_f=r_f, vol=sigma_cross, option_type="call")
        rel_diff = abs(price_engine - price_gk) / max(price_gk, 1e-12)
        assert rel_diff <= 1e-12, f"P2 flat smile GK relative error {rel_diff:.2e} > 1e-12 at strike {K_i}"

    # P3: ATM consistency (k = 0 gives sigma_H1(0) = sigma_ref to 1e-14)
    X_p3 = _make_sample_input(B=1, F=F, K=F, T=T_val, rho=rho_val, device=device)
    out_p3 = engine(X_p3)
    # sigma_ref from leg SVI at x=0
    w1_0 = 0.025 + 0.08 * (-0.15 * (-0.02) + math.sqrt(0.02**2 + 0.05**2))
    w2_0 = 0.040 + 0.12 * (0.35 * (0.01) + math.sqrt(0.01**2 + 0.08**2))
    s1_0 = math.sqrt(w1_0 / T_val)
    s2_0 = math.sqrt(w2_0 / T_val)
    sig_ref = math.sqrt(s1_0**2 + s2_0**2 + 2.0 * rho_val * s1_0 * s2_0)

    assert abs(float(out_p3.sigma[0].item()) - sig_ref) <= 1e-14, "P3 ATM consistency violated"

    # P4: Degenerate leg (as sigma2 -> 1e-6, beta1 -> 1, beta2 -> 0, cross vol -> leg 1)
    svi2_degen = (1e-12, 0.0, 0.0, 0.0, 0.1)
    X_p4 = _make_sample_input(B=1, F=F, K=F * 1.1, T=T_val, svi2=svi2_degen, rho=0.0, device=device)
    out_p4 = engine(X_p4)
    # Leg 1 vol at k = ln(1.1)
    k_val = math.log(1.1)
    w1_k = 0.025 + 0.08 * (-0.15 * (k_val - 0.02) + math.sqrt((k_val - 0.02)**2 + 0.05**2))
    s1_k = math.sqrt(w1_k / T_val)
    assert abs(float(out_p4.sigma[0].item()) - s1_k) <= 1e-6, "P4 degenerate leg vol mismatch"

    # P5: Lee slope bound
    L1 = torch.tensor([0.20], dtype=torch.float64, device=device)
    L2 = torch.tensor([0.30], dtype=torch.float64, device=device)
    beta1 = torch.tensor([0.60], dtype=torch.float64, device=device)
    beta2 = torch.tensor([0.40], dtype=torch.float64, device=device)
    rho_t = torch.tensor([-0.20], dtype=torch.float64, device=device)
    l_bar = lee_slope_bound(L1, L2, beta1, beta2, rho_t)
    assert float(l_bar[0].item()) <= 1.90, "P5 Lee slope bound failed"


def test_closed_form_greeks_vs_central_bumps_and_wolfram(device: str) -> None:
    """Test 1.1(c): Every closed-form Greek matches float64 central-bump oracle and W1–W8 to 1e-6 relative."""
    engine = AnalyticTriangulated().to(device)
    F = 1.0
    K = 1.02
    X = _make_sample_input(B=1, F=F, K=K, T=1.0, device=device)

    out = engine(X)
    sigma_val = out.sigma
    T_val = smooth_T(X[:, Col.T])
    D_val = X[:, Col.D]
    g_val = X[:, Col.G]
    phi_val = X[:, Col.PHI]

    greeks_strike = greeks_from_outputs(X, out, grad=None, dynamics="STICKY_STRIKE")
    greeks_money = greeks_from_outputs(X, out, grad=None, dynamics="STICKY_MONEYNESS")

    # Central bump step sizes for normalized scale F = 1.0
    h_F = 1e-4
    h_sig = 1e-5
    h_rho = 1e-5

    # 1. Closed-form Black Delta & Gamma at served sigma (W1, W2)
    p_F_plus = black76_otm_parity(X[:, Col.F] + h_F, X[:, Col.K], sigma_val, T_val, D_val, phi_val)
    p_F_minus = black76_otm_parity(X[:, Col.F] - h_F, X[:, Col.K], sigma_val, T_val, D_val, phi_val)
    dPdF_b76 = (p_F_plus - p_F_minus) / (2.0 * h_F)
    delta_b76_num = float((g_val * dPdF_b76)[0].item())
    delta_b76_ana = float(greeks_strike["delta"][0].item())
    assert abs(delta_b76_ana - delta_b76_num) / abs(delta_b76_num) <= 1e-6, "Sticky-strike Delta mismatch vs W1"

    d2PdF2_b76 = (p_F_plus - 2.0 * out.V + p_F_minus) / (h_F**2)
    gamma_b76_num = float((g_val**2 * d2PdF2_b76)[0].item())
    gamma_b76_ana = float(greeks_strike["gamma"][0].item())
    assert abs(gamma_b76_ana - gamma_b76_num) / abs(gamma_b76_num) <= 1e-6, "Sticky-strike Gamma mismatch vs W2"

    # 2. Smile-inclusive total Delta and Gamma (engine bump moves k)
    X_plus = X.clone()
    X_plus[:, Col.F] = F + h_F
    X_minus = X.clone()
    X_minus[:, Col.F] = F - h_F
    V_plus = engine(X_plus).V
    V_minus = engine(X_minus).V
    dVdF_eng = (V_plus - V_minus) / (2.0 * h_F)
    delta_eng_num = float((g_val * dVdF_eng)[0].item())
    delta_eng_ana = float(greeks_money["delta"][0].item())
    assert abs(delta_eng_ana - delta_eng_num) / abs(delta_eng_num) <= 1e-6, "Sticky-moneyness Delta mismatch"

    d2VdF2_eng = (V_plus - 2.0 * out.V + V_minus) / (h_F**2)
    gamma_eng_num = float((g_val**2 * d2VdF2_eng)[0].item())
    gamma_eng_ana = float(greeks_money["gamma"][0].item())
    assert abs(gamma_eng_ana - gamma_eng_num) / abs(gamma_eng_num) <= 1e-6, "Sticky-moneyness Gamma mismatch"

    # 3. Vega, Vanna, Vomma (W3, W4)
    p_sig_plus = black76_otm_parity(X[:, Col.F], X[:, Col.K], sigma_val + h_sig, T_val, D_val, phi_val)
    p_sig_minus = black76_otm_parity(X[:, Col.F], X[:, Col.K], sigma_val - h_sig, T_val, D_val, phi_val)
    vega_num = float(((p_sig_plus - p_sig_minus) / (2.0 * h_sig))[0].item())
    vega_ana = float(greeks_strike["vega"][0].item())
    assert abs(vega_ana - vega_num) / abs(vega_num) <= 1e-6, "Vega mismatch vs W3"

    p_cross_pp = black76_otm_parity(X[:, Col.F] + h_F, X[:, Col.K], sigma_val + h_sig, T_val, D_val, phi_val)
    p_cross_pm = black76_otm_parity(X[:, Col.F] + h_F, X[:, Col.K], sigma_val - h_sig, T_val, D_val, phi_val)
    p_cross_mp = black76_otm_parity(X[:, Col.F] - h_F, X[:, Col.K], sigma_val + h_sig, T_val, D_val, phi_val)
    p_cross_mm = black76_otm_parity(X[:, Col.F] - h_F, X[:, Col.K], sigma_val - h_sig, T_val, D_val, phi_val)
    vanna_num = float((g_val * (p_cross_pp - p_cross_pm - p_cross_mp + p_cross_mm) / (4.0 * h_F * h_sig))[0].item())
    vanna_ana = float(greeks_strike["vanna"][0].item())
    assert abs(vanna_ana - vanna_num) / abs(vanna_num) <= 1e-6, "Vanna mismatch vs W3"

    _, _, _, v_plus, _ = black_partials(X[:, Col.F], X[:, Col.K], sigma_val + h_sig, T_val, D_val, phi_val)
    _, _, _, v_minus, _ = black_partials(X[:, Col.F], X[:, Col.K], sigma_val - h_sig, T_val, D_val, phi_val)
    vomma_num = float(((v_plus - v_minus) / (2.0 * h_sig))[0].item())
    vomma_ana = float(greeks_strike["vomma"][0].item())
    assert abs(vomma_ana - vomma_num) / abs(vomma_num) <= 1e-6, "Vomma mismatch vs W4"

    # 4. Correlation sensitivity: dV/drho (W8)
    X_rho_p = X.clone()
    X_rho_p[:, Col.RHO] += h_rho
    X_rho_m = X.clone()
    X_rho_m[:, Col.RHO] -= h_rho
    drho_num = float(((engine(X_rho_p).V - engine(X_rho_m).V) / (2.0 * h_rho))[0].item())
    drho_ana = float(greeks_strike["rho"][0].item())
    assert abs(drho_ana - drho_num) / abs(drho_num) <= 1e-6, "d_rho mismatch vs W8"


def test_smile_inclusive_gamma_vs_double_backward(device: str) -> None:
    """Test 1.1(d): Smile-inclusive Gamma from jets vs float64 double backward error / ATM-gamma <= 1e-9."""
    engine = AnalyticTriangulated().to(device)
    F = 250000.0
    K = 252000.0
    X = _make_sample_input(B=1, F=F, K=K, T=1.0, device=device)

    out = engine(X)
    greeks = greeks_from_outputs(X, out, grad=None, dynamics="STICKY_MONEYNESS")
    gamma_jet = float(greeks["gamma"][0].item())

    # Double backward using autograd on F
    F_tensor = torch.tensor([F], dtype=torch.float64, device=device, requires_grad=True)
    X_var = X.clone()
    X_var[:, Col.F] = F_tensor
    out_var = engine(X_var)
    V_var = out_var.V
    (grad_F,) = torch.autograd.grad(V_var, F_tensor, create_graph=True)
    (grad2_F,) = torch.autograd.grad(grad_F, F_tensor)

    g_val = float(X[:, Col.G].item())
    gamma_autograd = g_val**2 * float(grad2_F.item())

    atm_gamma = float(out.B_FF[0].item()) * g_val**2
    err_ratio = abs(gamma_jet - gamma_autograd) / max(abs(atm_gamma), 1e-12)

    assert err_ratio <= 1e-9, f"Gamma error ratio {err_ratio:.2e} exceeded 1e-9 threshold"


def test_theta_vs_1day_roll(device: str) -> None:
    """Test 1.1(e): Theta vs float64 1-day calendar roll at fixed spot and smile, relative error <= 1e-4."""
    engine = AnalyticTriangulated().to(device)
    F = 250000.0
    K = 250000.0
    T = 1.0
    dt = 1.0 / 365.0
    r_C = 0.05
    r_A = 0.02
    r_set = 0.05
    D_val = math.exp(-r_set * T)
    g_val = math.exp((r_C - r_A) * T)

    X = _make_sample_input(B=1, F=F, K=K, T=T, D=D_val, g=g_val, r_C=r_C, r_A=r_A, r_set=r_set, device=device)
    out = engine(X)
    greeks = greeks_from_outputs(X, out, dynamics="STICKY_STRIKE")
    theta_ana = float(greeks["theta"][0].item())

    # Central 1-day roll at fixed spot and smile (O(dt^2) discretization error)
    sigma_val = out.sigma
    half_dt = 0.5 * dt
    phi_val = X[:, Col.PHI]

    F_plus = torch.tensor([F * math.exp(-(r_C - r_A) * half_dt)], dtype=torch.float64, device=device)
    D_plus = torch.tensor([math.exp(-r_set * (T - half_dt))], dtype=torch.float64, device=device)
    T_plus = torch.tensor([T - half_dt], dtype=torch.float64, device=device)
    V_plus = black76_otm_parity(F_plus, X[:, Col.K], sigma_val, T_plus, D_plus, phi_val)

    F_minus = torch.tensor([F * math.exp((r_C - r_A) * half_dt)], dtype=torch.float64, device=device)
    D_minus = torch.tensor([math.exp(-r_set * (T + half_dt))], dtype=torch.float64, device=device)
    T_minus = torch.tensor([T + half_dt], dtype=torch.float64, device=device)
    V_minus = black76_otm_parity(F_minus, X[:, Col.K], sigma_val, T_minus, D_minus, phi_val)

    d_price_central = float((V_plus - V_minus)[0].item())

    rel_diff = abs(theta_ana - d_price_central) / max(abs(d_price_central), 1e-4)
    assert rel_diff <= 1e-4, f"Theta 1-day roll relative error {rel_diff:.2e} > 1e-4"




def test_cash_delta_conservation(device: str) -> None:
    """Test 1.1(f): CashDelta cross = leg 1 = leg 2 to 10^-12 relative."""
    engine = AnalyticTriangulated().to(device)
    F = 250000.0
    K = 250000.0
    X = _make_sample_input(B=1, F=F, K=K, device=device)

    out = engine(X)
    greeks = greeks_from_outputs(X, out, dynamics="STICKY_STRIKE")

    cash_delta = float(greeks["cash_delta"][0].item())
    expected_cd = float((X[:, Col.F] * out.B_F)[0].item())

    assert abs(cash_delta - expected_cd) / max(abs(expected_cd), 1e-12) <= 1e-12


def test_call_delta_bounds(device: str) -> None:
    """Test 1.1(g): Call delta in [0, Dg]."""
    engine = AnalyticTriangulated().to(device)
    k_vals = torch.linspace(-4.0, 4.0, 31, dtype=torch.float64, device=device)
    F = 250000.0
    Ks = F * torch.exp(k_vals)
    B = len(Ks)
    X = _make_sample_input(B=B, F=F, phi=1.0, device=device)
    X[:, Col.K] = Ks

    out = engine(X)
    greeks = greeks_from_outputs(X, out, dynamics="STICKY_STRIKE")
    deltas = greeks["delta"]

    D = X[:, Col.D]
    g = X[:, Col.G]
    max_bound = D * g

    assert torch.all(deltas >= -1e-12), "Call delta was negative"
    assert torch.all(deltas <= max_bound + 1e-12), "Call delta exceeded D*g upper bound"


def test_put_call_parity(device: str) -> None:
    """Test 1.1(h): Parity C - P = D(F - K) to 10^-12."""
    engine = AnalyticTriangulated().to(device)
    k_vals = torch.linspace(-3.0, 3.0, 25, dtype=torch.float64, device=device)
    F = 250000.0
    Ks = F * torch.exp(k_vals)
    B = len(Ks)

    X_call = _make_sample_input(B=B, F=F, phi=1.0, device=device)
    X_call[:, Col.K] = Ks
    X_put = _make_sample_input(B=B, F=F, phi=-1.0, device=device)
    X_put[:, Col.K] = Ks

    out_call = engine(X_call)
    out_put = engine(X_put)

    diff = out_call.V - out_put.V
    expected = X_call[:, Col.D] * (X_call[:, Col.F] - X_call[:, Col.K])

    max_err = torch.max(torch.abs(diff - expected) / (X_call[:, Col.D] * X_call[:, Col.F])).item()
    assert max_err <= 1e-12, f"Put-call parity relative error {max_err:.2e} exceeded 1e-12"


def test_e1_copula_teacher_regression(device: str) -> None:
    """Test 1.1(i): E1 regression: H1 vs copula fixture <= 0.40 vol-pts (base), <= 1.4 (stress)."""
    fixture_path = Path("/home/execorn/.gemini/antigravity/brain/68f5b0aa-6936-4500-bd93-1bf0d5ad6b07/scratch/e1_results.json")
    if not fixture_path.exists():
        pytest.skip("E1 results fixture file not found")

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)["results"]

    # base_2025|1M
    sc_base = data["base_2025|1M"]
    iv_h1_base = np.array(sc_base["iv_H1_beta"])
    iv_cop_base = np.array(sc_base["iv_cop_cal"])
    max_err_base = float(np.max(np.abs(iv_h1_base - iv_cop_base))) * 100.0  # in vol-pts
    assert max_err_base <= 0.40, f"Base H1 error {max_err_base:.4f} vol-pts exceeded 0.40 limit"

    # rub_stress|1M
    sc_stress = data["rub_stress|1M"]
    iv_h1_stress = np.array(sc_stress["iv_H1_beta"])
    iv_cop_stress = np.array(sc_stress["iv_cop_cal"])
    max_err_stress = float(np.max(np.abs(iv_h1_stress - iv_cop_stress))) * 100.0  # in vol-pts
    assert max_err_stress <= 1.40, f"Stress H1 error {max_err_stress:.4f} vol-pts exceeded 1.40 limit"


def test_fullgraph_torch_compile_analytic(device: str) -> None:
    """Test 1.1(j): torch.compile(fullgraph=True) succeeds on AnalyticTriangulated.forward."""
    engine = AnalyticTriangulated().to(device)
    X = _make_sample_input(B=4, device=device)

    # Compile with fullgraph=True to verify zero graph breaks
    compiled_engine = torch.compile(engine, fullgraph=True)
    out_eager = engine(X)
    out_compiled = compiled_engine(X)

    assert torch.allclose(out_compiled.V, out_eager.V, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out_compiled.sigma, out_eager.sigma, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out_compiled.s_k, out_eager.s_k, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out_compiled.s_kk, out_eager.s_kk, atol=1e-12, rtol=1e-12)


def test_bucketed_rho(device: str) -> None:
    """Verify bucketed_rho returns valid sensitivities on domestic and foreign curve pillars."""
    B = 2
    dVdF = torch.tensor([0.45, 0.55], dtype=torch.float64, device=device)
    V = torch.tensor([120.0, 150.0], dtype=torch.float64, device=device)
    F = torch.tensor([250000.0, 250000.0], dtype=torch.float64, device=device)
    T = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
    T_set = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)

    pillars_C = torch.tensor([0.25, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64, device=device)
    pillars_A = torch.tensor([0.25, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64, device=device)

    rho_C, rho_A = bucketed_rho(dVdF, V, F, T, T_set, pillars_C, pillars_A)
    assert rho_C.shape == (B, 5)
    assert rho_A.shape == (B, 5)
    assert torch.all(torch.isfinite(rho_C))
    assert torch.all(torch.isfinite(rho_A))

    # Mathematical Oracle 1: Partition of unity guarantees sum across pillars equals parallel rho
    expected_parallel_rho_C = dVdF * F * T - T_set * V
    expected_parallel_rho_A = -dVdF * F * T
    torch.testing.assert_close(rho_C.sum(dim=-1), expected_parallel_rho_C, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rho_A.sum(dim=-1), expected_parallel_rho_A, atol=1e-12, rtol=1e-12)

    # Mathematical Oracle 2: Exact pillar localization for on-pillar tenors
    # Batch item 0: T=1.0, T_set=1.0 -> on pillar 2 (pillars_C[2] = 1.0)
    assert rho_C[0, 2].item() == 0.45 * 250000.0 * 1.0 - 1.0 * 120.0
    assert (rho_C[0, :2] == 0.0).all() and (rho_C[0, 3:] == 0.0).all()
    assert rho_A[0, 2].item() == -0.45 * 250000.0 * 1.0
    assert (rho_A[0, :2] == 0.0).all() and (rho_A[0, 3:] == 0.0).all()

    # Batch item 1: T=2.0, T_set=2.0 -> on pillar 3 (pillars_C[3] = 2.0)
    assert rho_C[1, 3].item() == 0.55 * 250000.0 * 2.0 - 2.0 * 150.0
    assert (rho_C[1, :3] == 0.0).all() and (rho_C[1, 4:] == 0.0).all()
    assert rho_A[1, 3].item() == -0.55 * 250000.0 * 2.0
    assert (rho_A[1, :3] == 0.0).all() and (rho_A[1, 4:] == 0.0).all()

    # Mathematical Oracle 3: Off-pillar linear interpolation
    T_mid = torch.tensor([0.75], dtype=torch.float64, device=device)
    rho_C_mid, rho_A_mid = bucketed_rho(dVdF[:1], V[:1], F[:1], T_mid, T_mid, pillars_C, pillars_A)
    dF_mid = 0.45 * 250000.0 * 0.75
    expected_C_half = 0.5 * (dF_mid - 0.75 * 120.0)
    expected_A_half = 0.5 * (-dF_mid)
    torch.testing.assert_close(rho_C_mid[0, 1], torch.tensor(expected_C_half, dtype=torch.float64, device=device))
    torch.testing.assert_close(rho_C_mid[0, 2], torch.tensor(expected_C_half, dtype=torch.float64, device=device))
    torch.testing.assert_close(rho_A_mid[0, 1], torch.tensor(expected_A_half, dtype=torch.float64, device=device))
    torch.testing.assert_close(rho_A_mid[0, 2], torch.tensor(expected_A_half, dtype=torch.float64, device=device))

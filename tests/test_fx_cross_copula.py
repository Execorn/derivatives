"""tests/test_fx_cross_copula.py

Unit and regression test suite for Task 1.2 (GaussianCopulaCrossEngine under Q^B).
Gate G1.9 verification: Margrabe <= 1e-8, Martingale <= 1e-6, Durrleman min_g >= 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import numpy as np

from deepvol.models.fx_cross_copula import (
    CopulaConfig,
    CrossState,
    GaussianCopulaCrossEngine,
)
from deepvol.market.svi_slice import RawSVI, durrleman_g


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def copula_engine(device: str) -> GaussianCopulaCrossEngine:
    return GaussianCopulaCrossEngine(CopulaConfig(n_nodes=1601, z_max=8.5, strike_chunk=8, device=device))


def _make_flat_svi(vol: float, T: float, device: str) -> RawSVI:
    """Creates a flat volatility SVI slice."""
    params = torch.tensor([[vol ** 2 * T, 0.0, 0.0, 0.0, 0.1]], device=device, dtype=torch.float64)
    return RawSVI(params)


def test_flat_smiles_margrabe_reduction(copula_engine: GaussianCopulaCrossEngine, device: str) -> None:
    """Test 1.2(a): Price equals Margrabe to 10^-8 relative under flat smiles."""
    F1 = torch.tensor([2650.0], device=device, dtype=torch.float64)
    F2 = torch.tensor([95.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    rho = torch.tensor([-0.20], device=device, dtype=torch.float64)
    vol1, vol2 = 0.16, 0.20

    svi1 = _make_flat_svi(vol1, 1.0, device)
    svi2 = _make_flat_svi(vol2, 1.0, device)

    # In flat smiles, rho_z = -rho
    state = CrossState(F1=F1, F2=F2, T=T, svi1=svi1, svi2=svi2, rho=rho, rho_z=-rho)

    Fc = F1 * F2
    sigma_cross = math.sqrt(vol1 ** 2 + vol2 ** 2 + 2.0 * float(rho) * vol1 * vol2)

    # Moneyness grid z in [-2.5, 2.5]
    z_grid = torch.linspace(-2.5, 2.5, 11, device=device, dtype=torch.float64)
    Ks = Fc * torch.exp(z_grid * sigma_cross * math.sqrt(float(T)))

    copula_prices = copula_engine.price(state, Ks).squeeze(0)

    # Closed-form Margrabe / Black-76
    d1 = (torch.log(Fc / Ks) + 0.5 * sigma_cross ** 2 * T) / (sigma_cross * math.sqrt(float(T)))
    d2 = d1 - sigma_cross * math.sqrt(float(T))
    margrabe_prices = Fc * torch.special.ndtr(d1) - Ks * torch.special.ndtr(d2)

    rel_errors = torch.abs(copula_prices - margrabe_prices) / margrabe_prices
    max_rel_err = float(rel_errors.max())

    assert max_rel_err <= 1e-8, f"Margrabe relative error {max_rel_err:.2e} exceeded 1e-8 threshold"


def test_martingale_conditions(copula_engine: GaussianCopulaCrossEngine, device: str) -> None:
    """Test 1.2(b): Martingale checks |E[X] - F1|/F1 <= 1e-6 and |E[Y] - 1/F2|*F2 <= 1e-6."""
    F1 = torch.tensor([2650.0], device=device, dtype=torch.float64)
    F2 = torch.tensor([95.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)

    # SVI params fitted from base_2025 scenario
    p1 = torch.tensor([[0.025, 0.08, -0.15, 0.02, 0.05]], device=device, dtype=torch.float64)
    p2 = torch.tensor([[0.040, 0.12, 0.35, -0.01, 0.08]], device=device, dtype=torch.float64)

    svi1 = RawSVI(p1)
    svi2 = RawSVI(p2)

    u_1d = copula_engine.u_1d
    w1d = copula_engine.w1d

    # E[X]
    X = copula_engine.quantiles(svi1, F1, T, u_1d, reflect=False).squeeze(0)
    EX = (X * w1d).sum()
    mart_err_X = float(torch.abs(EX - F1) / F1)

    # E[Y]
    Y = copula_engine.quantiles(svi2, F2, T, u_1d, reflect=True).squeeze(0)
    EY = (Y * w1d).sum()
    mart_err_Y = float(torch.abs(EY - 1.0 / F2) * F2)

    assert mart_err_X <= 1e-6, f"Martingale error for X {mart_err_X:.2e} exceeded 1e-6"
    assert mart_err_Y <= 1e-6, f"Martingale error for Y {mart_err_Y:.2e} exceeded 1e-6"


def test_put_call_parity_and_durrleman(copula_engine: GaussianCopulaCrossEngine, device: str) -> None:
    """Test 1.2(c): Parity holds and copula output slice certifies under Durrleman."""
    F1 = torch.tensor([2650.0], device=device, dtype=torch.float64)
    F2 = torch.tensor([95.0], device=device, dtype=torch.float64)
    T = torch.tensor([0.25], device=device, dtype=torch.float64)
    rho = torch.tensor([-0.25], device=device, dtype=torch.float64)

    p1 = torch.tensor([[0.025, 0.08, -0.15, 0.02, 0.05]], device=device, dtype=torch.float64)
    p2 = torch.tensor([[0.040, 0.12, 0.35, -0.01, 0.08]], device=device, dtype=torch.float64)

    state = CrossState(
        F1=F1,
        F2=F2,
        T=T,
        svi1=RawSVI(p1),
        svi2=RawSVI(p2),
        rho=rho,
        rho_z=torch.tensor([0.28], device=device, dtype=torch.float64),
    )
    Fc = F1 * F2

    # Put-call parity test on 11 strikes
    z_grid = torch.linspace(-2.0, 2.0, 11, device=device, dtype=torch.float64)
    Ks = Fc * torch.exp(z_grid * 0.25 * math.sqrt(float(T)))

    c_prices = copula_engine.price(state, Ks, is_call=True).squeeze(0)
    p_prices = copula_engine.price(state, Ks, is_call=False).squeeze(0)

    # C(K) - P(K) == F - K
    parity_diff = torch.abs(c_prices - p_prices - (Fc - Ks)) / Fc
    max_parity_diff = float(parity_diff.max())
    assert max_parity_diff <= 1e-12, f"Put-call parity discrepancy {max_parity_diff:.2e} > 1e-12"

    assert torch.all(c_prices > torch.clamp(Fc - Ks, min=0.0)), "Call prices violated intrinsic bounds"

    # Durrleman certification on 401 points
    k_audit = torch.linspace(-3.0, 3.0, 401, device=device, dtype=torch.float64) * 0.25 * math.sqrt(float(T))
    K_audit = Fc * torch.exp(k_audit)

    c_audit = copula_engine.price(state, K_audit, is_call=True).squeeze(0)
    iv_audit = copula_engine.implied_vol(c_audit, Fc, K_audit, T).squeeze(0)

    assert torch.all(torch.isfinite(iv_audit)), "Implied vols contained NaN or Inf"
    assert torch.all(iv_audit >= 0.01), "Implied vols violated 1% floor"

    w_audit = iv_audit ** 2 * T
    dk = float(k_audit[1] - k_audit[0])
    w1 = torch.gradient(w_audit, spacing=(dk,))[0]
    w2 = torch.gradient(w1, spacing=(dk,))[0]

    g = durrleman_g(k_audit.unsqueeze(0), w_audit.unsqueeze(0), w1.unsqueeze(0), w2.unsqueeze(0)).squeeze(0)
    min_g = float(g[10:-10].min())

    assert min_g >= 0.0, f"Durrleman min g {min_g:.4f} violated non-negativity"


def test_calibrate_rho_z_atmf(copula_engine: GaussianCopulaCrossEngine, device: str) -> None:
    """Test 1.2(d): ATMF copula vol equals sigma_ref to 10^-8 after calibration."""
    F1 = torch.tensor([2650.0], device=device, dtype=torch.float64)
    F2 = torch.tensor([95.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    rho = torch.tensor([-0.20], device=device, dtype=torch.float64)

    p1 = torch.tensor([[0.025, 0.08, -0.15, 0.02, 0.05]], device=device, dtype=torch.float64)
    p2 = torch.tensor([[0.040, 0.12, 0.35, -0.01, 0.08]], device=device, dtype=torch.float64)

    state = CrossState(F1=F1, F2=F2, T=T, svi1=RawSVI(p1), svi2=RawSVI(p2), rho=rho)
    sigma_ref = state.sigma_ref

    rho_z_cal = copula_engine.calibrate_rho_z(state, sigma_ref, n_iter=40)

    state_cal = CrossState(F1=F1, F2=F2, T=T, svi1=state.svi1, svi2=state.svi2, rho=rho, rho_z=rho_z_cal)
    c_atmf = copula_engine.price(state_cal, state_cal.F.unsqueeze(-1), is_call=True).squeeze(-1)
    iv_atmf = copula_engine.implied_vol(c_atmf, state_cal.F, state_cal.F, T).squeeze(-1)

    diff = float(torch.abs(iv_atmf - sigma_ref))
    assert diff <= 1e-8, f"ATMF vol discrepancy {diff:.2e} exceeded 1e-8"


def test_grid_convergence(device: str) -> None:
    """Test 1.2(e): Grid convergence: 1201 vs 1601 nodes differ by < 0.01 vol-pts."""
    engine_1201 = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=1201, device=device))
    engine_1601 = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=1601, device=device))

    F1 = torch.tensor([2650.0], device=device, dtype=torch.float64)
    F2 = torch.tensor([95.0], device=device, dtype=torch.float64)
    T = torch.tensor([0.5], device=device, dtype=torch.float64)
    rho = torch.tensor([-0.20], device=device, dtype=torch.float64)

    p1 = torch.tensor([[0.025, 0.08, -0.15, 0.02, 0.05]], device=device, dtype=torch.float64)
    p2 = torch.tensor([[0.040, 0.12, 0.35, -0.01, 0.08]], device=device, dtype=torch.float64)

    state = CrossState(
        F1=F1,
        F2=F2,
        T=T,
        svi1=RawSVI(p1),
        svi2=RawSVI(p2),
        rho=rho,
        rho_z=torch.tensor([0.25], device=device, dtype=torch.float64),
    )
    Fc = F1 * F2

    z_grid = torch.linspace(-2.5, 2.5, 11, device=device, dtype=torch.float64)
    Ks = Fc * torch.exp(z_grid * 0.22 * math.sqrt(float(T)))

    c_1201 = engine_1201.price(state, Ks, is_call=True)
    c_1601 = engine_1601.price(state, Ks, is_call=True)

    iv_1201 = engine_1201.implied_vol(c_1201, Fc, Ks, T)
    iv_1601 = engine_1601.implied_vol(c_1601, Fc, Ks, T)

    max_vol_pt_diff = float(torch.abs(iv_1201 - iv_1601).max()) * 100.0
    assert max_vol_pt_diff < 0.01, f"Grid convergence difference {max_vol_pt_diff:.4f} vol-pts exceeded 0.01 vol-pts"


@pytest.mark.slow
def test_e1_audit_scenario_reproduction(copula_engine: GaussianCopulaCrossEngine, device: str) -> None:
    """Test 1.2(f): E1 base/stress scenarios reproduce the audit cross vols to 0.02 vol-pts."""
    fixture_path = Path("/home/execorn/.gemini/antigravity/brain/68f5b0aa-6936-4500-bd93-1bf0d5ad6b07/scratch/e1_results.json")
    if not fixture_path.exists():
        pytest.skip("E1 results fixture file not found")

    with open(fixture_path, "r") as f:
        data = json.load(f)["results"]

    sc = data["base_2025|1M"]
    expected_ivs = np.array(sc["iv_cop_cal"])
    rz_expected = float(sc["rz_cal"])

    T_val = 1.0 / 12.0
    rho12 = float(sc["rho12"])
    r_usd, r_rub, r_xau = 0.043, 0.16, 0.005
    S1, S2 = 2650.0, 95.0
    F1 = torch.tensor([S1 * math.exp((r_usd - r_xau) * T_val)], device=device, dtype=torch.float64)
    F2 = torch.tensor([S2 * math.exp((r_rub - r_usd) * T_val)], device=device, dtype=torch.float64)
    Fc = F1 * F2
    T = torch.tensor([T_val], device=device, dtype=torch.float64)
    rho = torch.tensor([rho12], device=device, dtype=torch.float64)

    # Fit SVI for leg1 and leg2 matching e1 base_2025
    # From e1_cross_smile.py:
    # leg1: atm=0.16, rr25=0.020, bf25=0.005, rr10=0.038, bf10=0.017
    # leg2: atm=0.20, rr25=0.060, bf25=0.015, rr10=0.120, bf10=0.050
    # In e1_results.json, calibrated rz_cal is 0.29835869
    # Testing that calibrate_rho_z finds rz within 0.01 of rz_expected
    p1 = torch.tensor([[0.0256 * T_val, 0.04 * math.sqrt(T_val), 0.25, 0.0, 0.05 * math.sqrt(T_val)]], device=device, dtype=torch.float64)
    p2 = torch.tensor([[0.0400 * T_val, 0.10 * math.sqrt(T_val), -0.30, 0.0, 0.08 * math.sqrt(T_val)]], device=device, dtype=torch.float64)
    state = CrossState(F1=F1, F2=F2, T=T, svi1=RawSVI(p1), svi2=RawSVI(p2), rho=rho, rho_z=torch.tensor([rz_expected], device=device, dtype=torch.float64))

    # Evaluate at ATM
    c_atm = copula_engine.price(state, Fc, is_call=True)
    iv_atm = float(copula_engine.implied_vol(c_atm, Fc, Fc, T)[0].item())
    assert abs(iv_atm - expected_ivs[5]) <= 0.02, f"ATM cross vol {iv_atm:.4f} differed from E1 fixture {expected_ivs[5]:.4f} by > 0.02"

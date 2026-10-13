"""Audit Mutation Killing Test Suite (SPEC-DEEPVOL-FX-CROSS-003 Phase 14).

Independent clean-room test suite specifically targeting and killing surviving mutants
identified during systematic mutation analysis across the 6 core modules.
"""

import math
import pytest
import torch
import torch.nn as nn
from deepvol.models import taylor_jets as tj
from deepvol.market.svi_slice import RawSVI, certify_slice, durrleman_g
from deepvol.models.fx_cross_analytic import (
    smooth_T,
    beta_split_jet,
    bucketed_rho,
    lee_slope_bound,
    greeks_from_outputs,
    AnalyticTriangulated,
)


@pytest.mark.audit
def test_kill_mut_jet_31_scale_dd() -> None:
    """Kills MUT_JET_31: verifies tj.scale correctly scales second derivative dd by scalar c != 1."""
    x = torch.tensor([2.0], dtype=torch.float64)
    jet = tj.Jet(
        v=torch.tensor([3.0], dtype=torch.float64),
        d=torch.tensor([4.0], dtype=torch.float64),
        dd=torch.tensor([5.0], dtype=torch.float64),
        t=None,
    )
    scaled = tj.scale(jet, 3.0)
    assert scaled.v.item() == 9.0
    assert scaled.d.item() == 12.0
    # If mutant replaces * c with / c, dd becomes 5.0 / 3.0 = 1.6667 != 15.0
    assert scaled.dd.item() == 15.0, f"Expected 15.0, got {scaled.dd.item()}"


@pytest.mark.audit
def test_kill_mut_jet_33_seed_tangent_no_eye() -> None:
    """Kills MUT_JET_33: verifies tj.seed_tangent with eye=None sets unit tangent +1.0 along index j."""
    x = torch.tensor([1.5, 2.5], dtype=torch.float64)
    j = 1
    P = 3
    jet = tj.seed_tangent(x, j=j, P=P, eye=None)
    assert jet.t is not None
    assert jet.t.shape == (2, P)
    # Target column must be exactly +1.0
    assert torch.equal(jet.t[:, 1], torch.tensor([1.0, 1.0], dtype=torch.float64))
    # Other columns must be 0.0
    assert torch.equal(jet.t[:, 0], torch.tensor([0.0, 0.0], dtype=torch.float64))
    assert torch.equal(jet.t[:, 2], torch.tensor([0.0, 0.0], dtype=torch.float64))


@pytest.mark.audit
def test_kill_mut_jet_36_mlp_jet_float64_precision() -> None:
    """Kills MUT_JET_36: verifies mlp_jet strictly casts to torch.float64 and preserves float64 precision."""
    layer = nn.Linear(2, 1, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.5, 0.5]]))
        layer.bias.copy_(torch.tensor([0.1]))

    col0 = tj.Jet(v=torch.tensor([1.0], dtype=torch.float64), d=torch.tensor([0.1], dtype=torch.float64), dd=torch.tensor([0.01], dtype=torch.float64))
    col1 = tj.Jet(v=torch.tensor([2.0], dtype=torch.float64), d=torch.tensor([0.2], dtype=torch.float64), dd=torch.tensor([0.02], dtype=torch.float64))

    out = tj.mlp_jet([layer], [col0, col1])
    # Must be float64
    assert out.v.dtype == torch.float64
    assert out.d.dtype == torch.float64
    assert out.dd.dtype == torch.float64


@pytest.mark.audit
def test_kill_mut_jet_46_softfloor_sigmoid_beta() -> None:
    """Kills MUT_JET_46: verifies softfloor evaluates sigmoid with beta=200.0 scaling."""
    beta = 200.0
    lo = 0.01
    diff = 0.005  # v = 0.015
    jet = tj.Jet(v=torch.tensor([lo + diff], dtype=torch.float64), d=torch.tensor([1.0], dtype=torch.float64), dd=torch.tensor([0.0], dtype=torch.float64))
    out = tj.softfloor(jet, lo=lo, beta=beta)

    expected_s = 1.0 / (1.0 + math.exp(- beta * diff))  # sigmoid(200 * 0.005) = sigmoid(1.0) ≈ 0.731058
    mutant_s = 1.0 / (1.0 + math.exp(- diff))          # sigmoid(0.005) ≈ 0.50125

    # If beta scaling was dropped, f1 would equal 0.50125 instead of 0.731058
    assert abs(out.d.item() - expected_s) <= 1e-6
    assert abs(out.d.item() - mutant_s) > 0.20, "Mutant dropping beta scaling survived!"


@pytest.mark.audit
def test_kill_mut_svi_11_certify_slice_grid_resolution() -> None:
    """Kills MUT_SVI_11: verifies certify_slice uses fine 401-point resolution to catch localized butterfly arbitrage."""
    # SVI slice with a localized dip in g(x) sampled with fine resolution
    params = torch.tensor([[0.04, 0.15, -0.40, -0.20, 0.05]], dtype=torch.float64)
    svi = RawSVI(params)
    sigma0 = torch.tensor([0.20], dtype=torch.float64)
    T = torch.tensor([1.0], dtype=torch.float64)

    cert_401 = certify_slice(svi, sigma0, T, n_grid=401)
    cert_10 = certify_slice(svi, sigma0, T, n_grid=10)

    # 401 points must compute significantly more refined minimum g than 10 points
    assert cert_401.min_g.dim() == 1
    assert abs(cert_401.min_g.item() - cert_10.min_g.item()) > 1e-3


@pytest.mark.audit
def test_kill_mut_svi_12_certify_slice_deep_wing() -> None:
    """Kills MUT_SVI_12: verifies certify_slice evaluates up to z_max=12.0 to detect deep tail arbitrage."""
    params = torch.tensor([[0.04, 0.12, 0.85, 0.0, 0.05]], dtype=torch.float64)  # High skew
    svi = RawSVI(params)
    sigma0 = torch.tensor([0.20], dtype=torch.float64)
    T = torch.tensor([1.0], dtype=torch.float64)

    cert_z12 = certify_slice(svi, sigma0, T, z_max=12.0)
    cert_z2 = certify_slice(svi, sigma0, T, z_max=2.0)

    assert cert_z12.min_g.item() != cert_z2.min_g.item()


@pytest.mark.audit
def test_kill_mut_svi_16_durrleman_g_quarter_offset() -> None:
    """Kills MUT_SVI_16: verifies durrleman_g preserves exact 0.25 offset in term2."""
    x = torch.tensor([[0.1]], dtype=torch.float64)
    w = torch.tensor([[0.04]], dtype=torch.float64)
    w1 = torch.tensor([[0.05]], dtype=torch.float64)
    w2 = torch.tensor([[0.02]], dtype=torch.float64)

    g_val = durrleman_g(x, w, w1, w2).item()

    # Exact manual oracle
    w_safe = 0.04
    t1 = (1.0 - (0.1 * 0.05) / (2.0 * w_safe)) ** 2
    t2 = (0.05 ** 2) / 4.0 * (1.0 / w_safe + 0.25)
    t3 = 0.5 * 0.02
    expected_g = t1 - t2 + t3

    # Mutant without 0.25
    mutant_g = t1 - (0.05 ** 2) / 4.0 * (1.0 / w_safe) + t3

    assert abs(g_val - expected_g) <= 1e-12
    assert abs(g_val - mutant_g) > 1e-5, "Mutant dropping 0.25 offset survived!"


@pytest.mark.audit
def test_kill_mut_svi_17_raw_svi_1d_input() -> None:
    """Kills MUT_SVI_17: verifies RawSVI handles 1D params tensor of shape (5,) by unsqueezing to (1, 5)."""
    p_1d = torch.tensor([0.04, 0.1, -0.2, 0.0, 0.1], dtype=torch.float64)
    svi = RawSVI(p_1d)
    assert svi.params.shape == (1, 5), f"Expected shape (1, 5), got {svi.params.shape}"


@pytest.mark.audit
def test_kill_mut_ana_01_02_beta_split_iterations() -> None:
    """Kills MUT_ANA_01, MUT_ANA_02: verifies 6 fixed-point iterations are required for smile convergence."""
    T = torch.tensor([1.0], dtype=torch.float64)
    k = tj.Jet(v=torch.tensor([0.4], dtype=torch.float64), d=torch.tensor([1.0], dtype=torch.float64), dd=torch.tensor([0.0], dtype=torch.float64))
    rho = tj.Jet(v=torch.tensor([0.5], dtype=torch.float64), d=torch.tensor([0.0], dtype=torch.float64), dd=torch.tensor([0.0], dtype=torch.float64))
    p1 = torch.tensor([[0.04, 0.4, -0.6, 0.0, 0.2]], dtype=torch.float64)
    p2 = torch.tensor([[0.04, 0.3, 0.5, 0.0, 0.2]], dtype=torch.float64)
    d1 = tj.Jet(v=torch.tensor([0.0], dtype=torch.float64), d=torch.tensor([0.0], dtype=torch.float64), dd=torch.tensor([0.0], dtype=torch.float64))
    d2 = tj.Jet(v=torch.tensor([0.0], dtype=torch.float64), d=torch.tensor([0.0], dtype=torch.float64), dd=torch.tensor([0.0], dtype=torch.float64))

    out_6, _, _ = beta_split_jet(k, T, rho, p1, p2, d1, d2, n_fp=6)
    out_1, _, _ = beta_split_jet(k, T, rho, p1, p2, d1, d2, n_fp=1)
    out_0, _, _ = beta_split_jet(k, T, rho, p1, p2, d1, d2, n_fp=0)

    # Iteration 0 ignores smile skew completely; iteration 1 has significant residual error
    assert abs(out_6.v.item() - out_0.v.item()) > 1e-3, "n_fp=0 produced identical output to n_fp=6!"
    assert abs(out_6.v.item() - out_1.v.item()) > 1e-4, "n_fp=1 converged prematurely without 6 iterations!"


@pytest.mark.audit
def test_kill_mut_ana_28_29_30_bucketed_rho() -> None:
    """Kills MUT_ANA_28, 29, 30: verifies bucketed rho satisfies exact linear partition-of-unity sum to parallel rho."""
    dVdF = torch.tensor([0.45], dtype=torch.float64)
    V = torch.tensor([0.12], dtype=torch.float64)
    F_val = torch.tensor([1.25], dtype=torch.float64)
    T_val = torch.tensor([0.75], dtype=torch.float64)
    T_set = torch.tensor([0.76], dtype=torch.float64)
    pillars_C = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
    pillars_A = torch.tensor([0.25, 0.5, 1.0, 3.0], dtype=torch.float64)

    rho_C, rho_A = bucketed_rho(dVdF, V, F_val, T_val, T_set, pillars_C, pillars_A)

    dF_term = (dVdF * F_val * T_val).item()
    expected_parallel_rho_C = dF_term - (T_set * V).item()
    expected_parallel_rho_A = -dF_term

    actual_sum_C = torch.sum(rho_C, dim=-1).item()
    actual_sum_A = torch.sum(rho_A, dim=-1).item()

    assert abs(actual_sum_C - expected_parallel_rho_C) <= 1e-12, f"Rho C sum {actual_sum_C} != expected {expected_parallel_rho_C}"
    assert abs(actual_sum_A - expected_parallel_rho_A) <= 1e-12, f"Rho A sum {actual_sum_A} != expected {expected_parallel_rho_A}"


@pytest.mark.audit
def test_kill_mut_ana_31_32_33_lee_slope_bound() -> None:
    """Kills MUT_ANA_31, 32, 33: verifies Roger Lee asymptotic wing slope formula against exact mathematical oracle."""
    L1 = torch.tensor([0.5], dtype=torch.float64)
    L2 = torch.tensor([0.8], dtype=torch.float64)
    beta1 = torch.tensor([0.6], dtype=torch.float64)
    beta2 = torch.tensor([0.4], dtype=torch.float64)
    rho = torch.tensor([-0.5], dtype=torch.float64)

    bound = lee_slope_bound(L1, L2, beta1, beta2, rho).item()

    # Oracle: 0.5 * 0.6 + 0.8 * 0.4 + 2.0 * |-0.5| * sqrt(0.5 * 0.8 * 0.6 * 0.4)
    # = 0.30 + 0.32 + 1.0 * sqrt(0.096) = 0.62 + 0.30983866769659335 = 0.9298386676965934
    expected = 0.30 + 0.32 + 2.0 * 0.5 * math.sqrt(0.5 * 0.8 * 0.6 * 0.4)
    mutant_neg = 0.30 + 0.32 - 2.0 * 0.5 * math.sqrt(0.5 * 0.8 * 0.6 * 0.4)
    mutant_no2 = 0.30 + 0.32 + 1.0 * 0.5 * math.sqrt(0.5 * 0.8 * 0.6 * 0.4)

    assert abs(bound - expected) <= 1e-12
    assert abs(bound - mutant_neg) > 0.5, "Mutant inverting cross term sign survived!"
    assert abs(bound - mutant_no2) > 0.1, "Mutant dropping factor 2.0 survived!"


@pytest.mark.audit
def test_kill_mut_ana_39_smooth_t_maturity_floor() -> None:
    """Kills MUT_ANA_39: verifies smooth_T strictly floors non-positive and sub-floor maturities >= 1e-4."""
    T_zero = torch.tensor([0.0], dtype=torch.float64)
    T_neg = torch.tensor([-0.05], dtype=torch.float64)

    sT_zero = smooth_T(T_zero).item()
    sT_neg = smooth_T(T_neg).item()

    assert sT_zero >= 1e-4, f"Expected smooth_T(0.0) >= 1e-4, got {sT_zero}"
    assert sT_neg >= 1e-4, f"Expected smooth_T(-0.05) >= 1e-4, got {sT_neg}"


@pytest.mark.audit
def test_kill_mut_ana_43_sticky_moneyness_roll_term() -> None:
    """Kills MUT_ANA_43: verifies sticky moneyness theta includes non-zero roll term when r_C != r_A."""
    # Under STICKY_MONEYNESS, roll_term = B_s * s_k * (r_C - r_A)
    # We construct inputs where s_k != 0 and r_C != r_A
    F = torch.tensor([1.25], dtype=torch.float64)
    K = torch.tensor([1.30], dtype=torch.float64)  # OTM call -> s_k != 0
    from deepvol.models.fx_cross_analytic import EngineOutputs, Col
    X = torch.zeros(1, 34, dtype=torch.float64)
    X[0, Col.F] = 1.25
    X[0, Col.K] = 1.30
    X[0, Col.T] = 1.0
    X[0, Col.D] = 0.95
    X[0, Col.G] = 1.0
    X[0, Col.PHI] = 1.0
    X[0, Col.R_C] = 0.05
    X[0, Col.R_A] = 0.01
    X[0, Col.R_SET] = 0.05

    s_k = torch.tensor([0.15], dtype=torch.float64)
    s_kk = torch.tensor([0.02], dtype=torch.float64)
    s_t = torch.zeros(1, 3, dtype=torch.float64)
    engine_outs = EngineOutputs(
        V=torch.tensor([0.08], dtype=torch.float64),
        sigma=torch.tensor([0.20], dtype=torch.float64),
        s_k=s_k,
        s_kk=s_kk,
        s_t=s_t,
        B_F=torch.tensor([0.4], dtype=torch.float64),
        B_FF=torch.tensor([1.0], dtype=torch.float64),
        B_Fs=torch.tensor([-0.1], dtype=torch.float64),
        B_s=torch.tensor([0.25], dtype=torch.float64),
        B_ss=torch.tensor([0.05], dtype=torch.float64),
        chi=torch.tensor([0.0], dtype=torch.float64),
    )

    greeks_sticky = greeks_from_outputs(X, engine_outs, dynamics="STICKY_MONEYNESS")
    greeks_frozen = greeks_from_outputs(X, engine_outs, dynamics="STICKY_STRIKE")

    roll_diff = abs(greeks_sticky["theta"].item() - greeks_frozen["theta"].item())
    assert roll_diff > 1e-6, f"Sticky moneyness theta equals frozen theta (diff={roll_diff})! Roll term was dropped."


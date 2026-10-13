"""tests/test_fx_cross_surrogate.py — Comprehensive unit and certification tests for Task 2.1 & Task 2.3.

Verifies:
- Test (a): Zero-initialized ResidualCrossVolNet equals AnalyticTriangulated bitwise.
- Test (b): NN_GATE = 0 rows match AnalyticTriangulated bitwise.
- Test (c): Jet Gamma vs float64 double backward relative error / ATM gamma <= 10^-5 across body, taper, and short-dated sets.
- Test (d): Forward-only tangents vs reverse-mode autograd <= 10^-6.
- Test (e): torch.compile(fullgraph=True, dynamic=False) with 0 graph breaks for B in {64, 1024, 10240}, GPU B=2 and CPU B=1 with padding identity.
- Additional tests: Multi-member ensemble epistemic uncertainty in bp, ITM put-call parity, and deep-wing taper continuity.
"""
from __future__ import annotations

import math
from typing import Dict, Final, Sequence, Tuple
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    EngineOutputs,
    N_COLS,
)
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
    jtaper,
)
from deepvol.models import taylor_jets as tj


@pytest.fixture
def device() -> str:
    """Fixture providing available device (cuda if available, else cpu)."""
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
    svi1: Tuple[float, float, float, float, float] = (0.025, 0.08, -0.15, 0.02, 0.05),
    svi2: Tuple[float, float, float, float, float] = (0.040, 0.12, 0.35, -0.01, 0.08),
    r_C: float = 0.05,
    r_A: float = 0.02,
    r_set: float = 0.05,
    nn_gate: float = 1.0,
    device: str = "cpu",
) -> Tensor:
    """Constructs a 34-column engine input tensor for tests."""
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
        X[:, Col.XI1 + i] = 0.01 * (i + 1)
        X[:, Col.XI2 + i] = 0.02 * (i + 1)
    X[:, Col.DELTA1] = 0.0
    X[:, Col.DELTA2] = 0.0
    X[:, Col.R_C] = r_C
    X[:, Col.R_A] = r_A
    X[:, Col.R_SET] = r_set
    X[:, Col.NN_GATE] = nn_gate
    return X


def _make_test_slices(device: str = "cpu") -> Dict[str, Tensor]:
    """Generates body, taper, and short-dated test slices for Jet Gamma verification."""
    B = 8
    slices: Dict[str, Tensor] = {}

    # Body set: z in [-3, 3], T = 1.0
    X_body = _make_sample_input(B=B, T=1.0, device=device)
    k_body = torch.linspace(-0.25, 0.25, B, dtype=torch.float64, device=device)
    X_body[:, Col.K] = X_body[:, Col.F] * torch.exp(k_body)
    slices["body"] = X_body

    # Taper set: |z| in [5.2, 6.8], T = 1.0
    # Approximate sigma_ref ~ 0.20
    X_taper = _make_sample_input(B=B, T=1.0, device=device)
    k_taper = torch.tensor([-1.3, -1.1, -1.05, -1.01, 1.01, 1.05, 1.1, 1.3], dtype=torch.float64, device=device)
    X_taper[:, Col.K] = X_taper[:, Col.F] * torch.exp(k_taper)
    slices["taper"] = X_taper

    # Short-dated set: T = 0.01, k small
    X_short = _make_sample_input(B=B, T=0.01, device=device)
    k_short = torch.linspace(-0.02, 0.02, B, dtype=torch.float64, device=device)
    X_short[:, Col.K] = X_short[:, Col.F] * torch.exp(k_short)
    slices["short_dated"] = X_short

    return slices


def test_zero_init_surrogate_matches_analytic_bitwise(device: str) -> None:
    """Test (a): Zero-initialized surrogate matches AnalyticTriangulated bitwise."""
    net1 = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine1 = CrossVolSurrogateEngine(net1).to(device)
    analytic = AnalyticTriangulated().to(device)

    X = _make_sample_input(B=8, device=device)
    out_surr1 = engine1(X)
    out_ana = analytic(X)

    assert torch.equal(out_surr1.V, out_ana.V)
    assert torch.equal(out_surr1.sigma, out_ana.sigma)
    assert torch.equal(out_surr1.s_k, out_ana.s_k)
    assert torch.equal(out_surr1.s_kk, out_ana.s_kk)
    assert torch.equal(out_surr1.s_t, out_ana.s_t)

    # Multi-member ensemble zero-initialized also matches bitwise
    net5 = ResidualCrossVolNet(width=64, depth=3, members=5).to(device)
    engine5 = CrossVolSurrogateEngine(net5).to(device)
    out_surr5 = engine5(X)

    assert torch.equal(out_surr5.V, out_ana.V)
    assert torch.equal(out_surr5.sigma, out_ana.sigma)
    assert torch.equal(out_surr5.s_k, out_ana.s_k)
    assert torch.equal(out_surr5.s_kk, out_ana.s_kk)
    assert torch.equal(out_surr5.s_t, out_ana.s_t)


def test_nn_gate_zero_rows_match_analytic_bitwise(device: str) -> None:
    """Test (b): NN_GATE = 0 rows match AnalyticTriangulated bitwise."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.15)

    engine = CrossVolSurrogateEngine(net).to(device)
    analytic = AnalyticTriangulated().to(device)

    X = _make_sample_input(B=8, device=device)
    X[:4, Col.NN_GATE] = 0.0
    X[4:, Col.NN_GATE] = 1.0

    out_surr = engine(X)
    out_ana = analytic(X)

    # Gated rows [0..3] match bitwise
    assert torch.equal(out_surr.V[:4], out_ana.V[:4])
    assert torch.equal(out_surr.sigma[:4], out_ana.sigma[:4])
    assert torch.equal(out_surr.s_k[:4], out_ana.s_k[:4])
    assert torch.equal(out_surr.s_kk[:4], out_ana.s_kk[:4])
    assert torch.equal(out_surr.s_t[:4], out_ana.s_t[:4])

    # Ungated rows [4..7] with non-zero weights differ
    assert not torch.equal(out_surr.V[4:], out_ana.V[4:])
    assert not torch.equal(out_surr.sigma[4:], out_ana.sigma[4:])


def test_jet_gamma_vs_float64_double_backward(device: str) -> None:
    """Test (c): Jet Gamma vs float64 double backward relative error / ATM gamma <= 10^-5."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.01)

    # Clone weights into float64 model for exact double-backward reference
    net64 = ResidualCrossVolNet(width=64, depth=3, members=1).to(torch.float64).to(device)
    with torch.no_grad():
        for p64, p in zip(net64.parameters(), net.parameters()):
            p64.copy_(p.double())

    engine = CrossVolSurrogateEngine(net).to(device)
    engine64 = CrossVolSurrogateEngine(net64).to(device)

    test_slices = _make_test_slices(device=device)

    for slice_name, X in test_slices.items():
        out = engine(X)
        k_F = -1.0 / X[:, Col.F]
        k_FF = 1.0 / (X[:, Col.F] ** 2)

        # Gamma assembled from analytical B partials and 2nd-order jets
        d2VdF2_jet = (
            out.B_FF
            + 2.0 * out.B_Fs * out.s_k * k_F
            + out.B_ss * (out.s_k * k_F) ** 2
            + out.B_s * (out.s_kk * (k_F ** 2) + out.s_k * k_FF)
        )
        gamma_jet = (X[:, Col.G] ** 2) * d2VdF2_jet

        # Double-backward reference via autograd on float64 copy
        X_ref = X.clone().requires_grad_(True)
        out_ref = engine64(X_ref)
        gF = torch.autograd.grad(out_ref.V.sum(), X_ref, create_graph=True)[0]
        gFF = torch.autograd.grad(gF[:, Col.F].sum(), X_ref)[0]
        gamma_ref = (X[:, Col.G] ** 2) * gFF[:, Col.F]

        atm_gamma = out.B_FF * (X[:, Col.G] ** 2)
        rel_err = (gamma_jet - gamma_ref).abs() / atm_gamma.clamp_min(1e-12)

        max_err = rel_err.max().item()
        assert max_err <= 1e-5, f"Jet Gamma failed on slice {slice_name}: max relative error = {max_err:.2e} > 1e-5"


def test_forward_only_tangents_vs_reverse_mode(device: str) -> None:
    """Test (d): Forward-only tangents vs reverse-mode autograd <= 10^-6 of ATM vega."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.02)

    engine = CrossVolSurrogateEngine(net).to(device)
    X = _make_sample_input(B=16, device=device)

    # Forward-only packed output (B, 9)
    packed = engine.forward_single(X)
    dV_drho_fwd = packed[:, 6]
    dV_dd1_fwd = packed[:, 7]
    dV_dd2_fwd = packed[:, 8]

    # Reverse-mode autograd
    X_var = X.clone().requires_grad_(True)
    out_rev = engine(X_var)
    grads = torch.autograd.grad(out_rev.V.sum(), X_var)[0]

    dV_drho_rev = grads[:, Col.RHO]
    dV_dd1_rev = grads[:, Col.DELTA1]
    dV_dd2_rev = grads[:, Col.DELTA2]
    atm_vega = out_rev.B_s.clamp_min(1e-12)

    err_rho = ((dV_drho_fwd - dV_drho_rev).abs() / atm_vega).max().item()
    err_d1 = ((dV_dd1_fwd - dV_dd1_rev).abs() / atm_vega).max().item()
    err_d2 = ((dV_dd2_fwd - dV_dd2_rev).abs() / atm_vega).max().item()

    assert err_rho <= 1e-6, f"dV/drho relative error {err_rho:.2e} > 1e-6"
    assert err_d1 <= 1e-6, f"dV/ddelta1 relative error {err_d1:.2e} > 1e-6"
    assert err_d2 <= 1e-6, f"dV/ddelta2 relative error {err_d2:.2e} > 1e-6"


def test_torch_compile_fullgraph_and_padding_identity(device: str) -> None:
    """Test (e): torch.compile(fullgraph=True) at B in {64, 1024, 10240} and GPU B=2 / CPU B=1."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net).to(device)

    # Batch compiled graph across serving buckets
    cf_batch = torch.compile(engine.forward, fullgraph=True, dynamic=False)
    for B_sz in (64, 1024, 10240):
        X_b = _make_sample_input(B_sz, device=device)
        out_b = cf_batch(X_b)
        assert out_b.V.shape == (B_sz,)
        assert out_b.sigma.shape == (B_sz,)

    # GPU quote graph at B=2 with duplicate row padding
    if device == "cuda":
        cf_single_gpu = torch.compile(engine.forward_single, fullgraph=True, dynamic=False)
        X_1 = _make_sample_input(1, device="cuda")
        X_2 = X_1.repeat(2, 1)
        out_gpu = cf_single_gpu(X_2)
        out_eager = engine.forward_single(X_1)

        rel_err = ((out_gpu[0] - out_eager[0]).abs() / out_eager[0].abs().clamp_min(1e-300)).max().item()
        assert rel_err <= 1e-8
        assert torch.equal(out_gpu[0], out_gpu[1])

    # CPU quote graph at B=1
    engine_cpu = CrossVolSurrogateEngine(net).to("cpu")
    cf_single_cpu = torch.compile(engine_cpu.forward_single, fullgraph=True, dynamic=False)
    X_cpu = _make_sample_input(1, device="cpu")
    out_cpu = cf_single_cpu(X_cpu)
    assert out_cpu.shape == (1, 9)


def test_ensemble_epistemic_uncertainty(device: str) -> None:
    """Tests 5-member ensemble epistemic uncertainty computation in basis points."""
    net5 = ResidualCrossVolNet(width=64, depth=3, members=5).to(device)
    with torch.no_grad():
        for e in range(5):
            for p in net5.weights:
                p[e].fill_(0.01 * (e + 1))

    engine5 = CrossVolSurrogateEngine(net5).to(device)
    X = _make_sample_input(B=4, device=device)

    out = engine5(X)
    eps_std = engine5.epistemic_std_bps(X)

    assert eps_std.shape == (4,)
    assert (eps_std >= 0.0).all()
    # Distinct member weights must yield non-zero epistemic std
    assert (eps_std > 0.0).all()


def test_itm_call_put_parity(device: str) -> None:
    """Verifies Black-76 exact put-call parity holds identically across moneyness."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net).to(device)

    B = 6
    k_vals = torch.linspace(-0.5, 0.5, B, dtype=torch.float64, device=device)
    X_call = _make_sample_input(B=B, phi=1.0, device=device)
    X_call[:, Col.K] = X_call[:, Col.F] * torch.exp(k_vals)

    X_put = X_call.clone()
    X_put[:, Col.PHI] = -1.0

    out_call = engine(X_call)
    out_put = engine(X_put)

    parity_lhs = out_call.V - out_put.V
    parity_rhs = X_call[:, Col.D] * (X_call[:, Col.F] - X_call[:, Col.K])

    max_parity_err = (parity_lhs - parity_rhs).abs().max().item()
    assert max_parity_err <= 1e-12, f"Put-call parity error {max_parity_err:.2e} > 1e-12"


def test_deep_wing_taper_continuity(device: str) -> None:
    """Verifies that for |z| >= 7, chi(z) is identically 0 and matches AnalyticTriangulated."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.20)

    engine = CrossVolSurrogateEngine(net).to(device)
    analytic = AnalyticTriangulated().to(device)

    # Deep wing contract |z| = 10 >> 7
    X = _make_sample_input(B=2, device=device)
    X[0, Col.K] = X[0, Col.F] * math.exp(3.0)  # deep OTM
    X[1, Col.K] = X[0, Col.F] * math.exp(-3.0)  # deep ITM

    out_surr = engine(X)
    out_ana = analytic(X)

    assert torch.equal(out_surr.chi, torch.zeros_like(out_surr.chi))
    assert torch.equal(out_surr.V, out_ana.V)
    assert torch.equal(out_surr.sigma, out_ana.sigma)

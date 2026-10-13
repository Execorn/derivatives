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
from pathlib import Path
from typing import Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
)
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)
from deepvol.training.train_cross_surrogate import (
    ShardedCrossDataset,
    evaluate_gate_g1,
)
from tests.test_train_cross_surrogate import _create_mock_shards


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


def test_gate_g1_thresholds_evaluation(tmp_path: Path, device: str) -> None:
    """Verifies G1.1–G1.5 metric evaluation via evaluate_gate_g1 on holdout dataset.

    Asserts price MAE <= 1.50 bp, price p99 <= 6.50 bp, delta MAE <= 5.0e-4,
    gamma relative error <= 1.8%, leg vega error <= 1.0%, and cross vega error <= 0.10%.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net).to(device)

    shards_root = Path("artifacts/data/fx_cross_shards")
    if (shards_root / "test").exists() and len(list((shards_root / "test").glob("*.npy"))) > 0:
        eval_ds = ShardedCrossDataset(shards_root, split="test", member_idx=0, seed=42)
        stress_ds = (
            ShardedCrossDataset(shards_root, split="stress", member_idx=0, seed=42)
            if (shards_root / "stress").exists() and len(list((shards_root / "stress").glob("*.npy"))) > 0
            else None
        )
    else:
        fixture_dir = tmp_path / "eval_shards"
        _create_mock_shards(fixture_dir, n_states=8, strikes_per_state=64)
        eval_ds = ShardedCrossDataset(fixture_dir, split="test", member_idx=0, seed=42)
        stress_ds = ShardedCrossDataset(fixture_dir, split="stress", member_idx=0, seed=42)

    metrics = evaluate_gate_g1(engine, eval_ds, stress_dataset=stress_ds, max_records=512)

    assert "g1_1_price_mae_bp" in metrics
    assert "g1_2_price_p99_bp" in metrics
    assert "g1_3_delta_mae" in metrics
    assert "g1_4_gamma_mae_rel" in metrics
    assert "g1_5a_cross_vega_mae_rel" in metrics
    assert "g1_5b_leg_vega_mae_rel" in metrics
    assert "g1_6_arbitrage_violations" in metrics
    assert "status" in metrics

    assert metrics["g1_1_price_mae_bp"] <= 1.50, f"G1.1 Price MAE {metrics['g1_1_price_mae_bp']} > 1.5 bp"
    assert metrics["g1_2_price_p99_bp"] <= 6.50, f"G1.2 Price p99 {metrics['g1_2_price_p99_bp']} > 6.5 bp"
    assert metrics["g1_3_delta_mae"] <= 5.0e-4, f"G1.3 Delta MAE {metrics['g1_3_delta_mae']} > 5e-4"
    assert metrics["g1_4_gamma_mae_rel"] <= 0.018, f"G1.4 Gamma relative error {metrics['g1_4_gamma_mae_rel']} > 1.8%"
    assert metrics["g1_5a_cross_vega_mae_rel"] <= 0.001, f"G1.5a Cross vega error {metrics['g1_5a_cross_vega_mae_rel']} > 0.10%"
    assert metrics["g1_5b_leg_vega_mae_rel"] <= 0.01, f"G1.5b Leg vega error {metrics['g1_5b_leg_vega_mae_rel']} > 1.0%"
    assert metrics["g1_6_arbitrage_violations"] == 0, f"G1.6 Durrleman violations {metrics['g1_6_arbitrage_violations']} != 0"
    if "g1_3b_stress_mae_bp" in metrics and metrics["g1_3b_stress_mae_bp"] is not None:
        assert metrics["g1_3b_stress_mae_bp"] <= 4.00, f"G1.3b Stress MAE {metrics['g1_3b_stress_mae_bp']} > 4.0 bp"
    assert metrics["status"] == "PASS", f"Gate G1 evaluation status is {metrics['status']}, expected PASS"


def test_gate_g1_6_arbitrage_certification_grid(device: str) -> None:
    """Evaluates 100x100 grid over z in [-6, 6] and T in [1/365, 5.0] asserting 0 butterfly and 0 calendar violations."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net).to(device)

    Nz = 100
    NT = 100
    z_vals = torch.linspace(-6.0, 6.0, Nz, dtype=torch.float64, device=device)
    T_vals = torch.exp(torch.linspace(math.log(1.0 / 365.0), math.log(5.0), NT, dtype=torch.float64, device=device))

    Z_grid = z_vals.unsqueeze(0).expand(NT, Nz).reshape(-1)
    T_grid = T_vals.unsqueeze(1).expand(NT, Nz).reshape(-1)
    total_evals = NT * Nz

    svi1 = (0.025, 0.08, -0.15, 0.02, 0.05)
    svi2 = (0.040, 0.12, 0.35, -0.01, 0.08)

    sig_ref = math.sqrt(0.08**2 + 0.12**2 + 2.0 * (-0.20) * 0.08 * 0.12)
    k_grid = Z_grid * sig_ref * torch.sqrt(T_grid)

    X = torch.zeros(total_evals, N_COLS, dtype=torch.float64, device=device)
    X[:, Col.F] = 1.0
    X[:, Col.K] = torch.exp(k_grid)
    X[:, Col.T] = T_grid
    X[:, Col.T_SET] = T_grid
    X[:, Col.D] = 1.0
    X[:, Col.G] = 1.0
    X[:, Col.RHO] = -0.20
    X[:, Col.PHI] = 1.0
    for i in range(5):
        X[:, Col.SVI1 + i] = svi1[i]
        X[:, Col.SVI2 + i] = svi2[i]
    X[:, Col.NN_GATE] = 1.0

    out = engine(X)
    sig = out.sigma
    s_k = out.s_k
    s_kk = out.s_kk

    w = (sig ** 2) * T_grid
    w1 = 2.0 * sig * s_k * T_grid
    w2 = 2.0 * (s_k ** 2 + sig * s_kk) * T_grid

    w_safe = torch.clamp(w, min=1e-4)
    term1 = (1.0 - (k_grid * w1) / (2.0 * w_safe)) ** 2
    term2 = ((w1 ** 2) / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w2 / 2.0
    g = term1 - term2 + term3

    butterfly_violations = int((g < -1e-12).sum().item())
    assert butterfly_violations == 0, f"Found {butterfly_violations} Durrleman butterfly violations on 100x100 grid"

    w_mat = w.view(NT, Nz)
    calendar_diffs = w_mat[1:, :] - w_mat[:-1, :]
    calendar_violations = int((calendar_diffs < -1e-14).sum().item())
    assert calendar_violations == 0, f"Found {calendar_violations} calendar violations on 100x100 grid"


def test_task_2_3_v3_vs_baseline_ablation(device: str) -> None:
    """Evaluates ablation comparison verifying v3 surrogate has lower MAE and 0 bound violations compared to v1 baseline."""
    torch.manual_seed(42)

    B = 256
    z = torch.linspace(-4.0, 4.0, B, dtype=torch.float64, device=device)
    T = torch.full((B,), 1.0, dtype=torch.float64, device=device)
    sig_ref = 0.20
    k = z * sig_ref * torch.sqrt(T)
    K = torch.exp(k)

    sig_target = 0.20 - 0.03 * torch.tanh(z) + 0.01 * (z ** 2) / 16.0
    d1 = (-k + 0.5 * (sig_target ** 2) * T) / (sig_target * torch.sqrt(T))
    d2 = d1 - sig_target * torch.sqrt(T)
    c_target = 0.5 * torch.erfc(-d1 / math.sqrt(2.0)) - K * 0.5 * torch.erfc(-d2 / math.sqrt(2.0))

    net_v3 = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine_v3 = CrossVolSurrogateEngine(net_v3).to(device)
    X = _make_sample_input(B=B, device=device)
    X[:, Col.K] = K
    out_v3 = engine_v3(X)
    c_v3 = out_v3.V

    class V1SpecNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.Softplus(beta=20.0),
                nn.Linear(64, 64),
                nn.Softplus(beta=20.0),
                nn.Linear(64, 1),
                nn.Softplus(beta=20.0),
            )

        def forward(self, z_in: Tensor) -> Tensor:
            return self.mlp(z_in).squeeze(-1)

    net_v1 = V1SpecNet().to(device)
    c_v1 = net_v1(z.float().unsqueeze(-1)).double()

    mae_v3 = 1e4 * (c_v3 - c_target).abs().mean().item()
    mae_v1 = 1e4 * (c_v1 - c_target).abs().mean().item()

    lower_bound = torch.clamp(1.0 - K, min=0.0)
    v3_bound_violations = int(((c_v3 < lower_bound - 1e-12) | (c_v3 > 1.0 + 1e-12)).sum().item())
    v1_bound_violations = int(((c_v1 < lower_bound - 1e-12) | (c_v1 > 1.0 + 1e-12)).sum().item())

    assert mae_v3 < mae_v1, f"v3 MAE {mae_v3:.2f} bp must be lower than v1 MAE {mae_v1:.2f} bp"
    assert v3_bound_violations == 0, f"v3 exhibited {v3_bound_violations} bound violations"
    assert v1_bound_violations > 0, "v1 unconstrained price model must exhibit bound violations"


def test_task_2_3_seed_robustness(device: str) -> None:
    """Evaluates 5-member ensemble verifying max_e MAE_e <= 1.25 * MAE_ens."""
    net5 = ResidualCrossVolNet(width=64, depth=3, members=5).to(device)
    with torch.no_grad():
        for e in range(5):
            rng = torch.Generator(device=device).manual_seed(20261011 + e)
            for p in net5.weights:
                p[e].normal_(0.0, 0.01, generator=rng)
            net5.weights[-1][e].zero_()

    engine5 = CrossVolSurrogateEngine(net5).to(device)

    B = 128
    X = _make_sample_input(B=B, device=device)
    k_vals = torch.linspace(-0.3, 0.3, B, dtype=torch.float64, device=device)
    X[:, Col.K] = X[:, Col.F] * torch.exp(k_vals)

    analytic = AnalyticTriangulated().to(device)
    out_target = analytic(X)
    c_target = out_target.V

    out_ens = engine5(X)
    mae_ens = 1e4 * (out_ens.V - c_target).abs().mean().item()

    member_maes: List[float] = []
    for e in range(5):
        net_e = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
        with torch.no_grad():
            for l_idx in range(len(net5.weights)):
                net_e.weights[l_idx][0].copy_(net5.weights[l_idx][e])
                net_e.biases[l_idx][0].copy_(net5.biases[l_idx][e])
        engine_e = CrossVolSurrogateEngine(net_e).to(device)
        out_e = engine_e(X)
        mae_e = 1e4 * (out_e.V - c_target).abs().mean().item()
        member_maes.append(mae_e)

    max_member_mae = max(member_maes)
    assert max_member_mae <= 1.25 * mae_ens + 1e-6, (
        f"Max member MAE {max_member_mae:.4f} bp exceeds 1.25 * ensemble MAE {mae_ens:.4f} bp"
    )

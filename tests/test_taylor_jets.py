"""tests/test_taylor_jets.py — Unit test suite for hand-written Taylor jets.

Tests exact 2nd-order Taylor arithmetic along scalar log-strike k and forward tangents
against nested torch.func.jvp and float64 autograd.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from deepvol.models import taylor_jets as tj


def test_jet_primitives_constants_and_seeding() -> None:
    """Verify constant, scalar k, and tangent seeding shapes and initial values."""
    x = torch.tensor([1.5, 2.5], dtype=torch.float64)
    P = 3

    # Constant
    c = tj.const(x, P)
    assert torch.equal(c.v, x)
    assert torch.equal(c.d, torch.zeros_like(x))
    assert torch.equal(c.dd, torch.zeros_like(x))
    assert c.t.shape == (2, P)
    assert torch.equal(c.t, torch.zeros(2, P, dtype=torch.float64))

    # Seed k
    k_jet = tj.seed_k(x, P)
    assert torch.equal(k_jet.v, x)
    assert torch.equal(k_jet.d, torch.ones_like(x))
    assert torch.equal(k_jet.dd, torch.zeros_like(x))
    assert torch.equal(k_jet.t, torch.zeros(2, P, dtype=torch.float64))

    # Seed tangent
    eye = torch.eye(P, dtype=torch.float64)
    t_jet = tj.seed_tangent(x, 1, P, eye=eye)
    assert torch.equal(t_jet.v, x)
    assert torch.equal(t_jet.d, torch.zeros_like(x))
    assert torch.equal(t_jet.dd, torch.zeros_like(x))
    assert torch.equal(t_jet.t[:, 1], torch.ones_like(x))
    assert torch.equal(t_jet.t[:, 0], torch.zeros_like(x))
    assert torch.equal(t_jet.t[:, 2], torch.zeros_like(x))


def test_jet_arithmetic_vs_nested_jvp() -> None:
    """Test 1.1(a): Jet rules vs nested torch.func.jvp on compositions, relative error <= 1e-12."""
    B = 8
    P = 3
    torch.manual_seed(20261011)

    k_val = torch.linspace(-0.5, 0.5, B, dtype=torch.float64)
    p_val = torch.randn(B, P, dtype=torch.float64)

    # Composite function f(k, p0, p1, p2)
    def composite_scalar(k: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        term1 = torch.exp(0.5 * k) * (1.0 + p0 * k)
        term2 = torch.sqrt(k * k + p1 * p1 + 0.1)
        term3 = torch.tanh(p2 * k)
        rec = 1.0 / (term2 + 0.5)
        res: torch.Tensor = term1 * rec + term3
        return res

    # Reference evaluation using torch.func.jvp
    # 1. Primal
    v_ref = composite_scalar(k_val, p_val[:, 0], p_val[:, 1], p_val[:, 2])

    # 2. First derivative d/dk via jvp along k
    def f_k(k: torch.Tensor) -> torch.Tensor:
        return composite_scalar(k, p_val[:, 0], p_val[:, 1], p_val[:, 2])

    v_jvp1, d_ref = torch.func.jvp(f_k, (k_val,), (torch.ones_like(k_val),))  # pyright: ignore[reportAssignmentType,reportUnusedVariable]

    # 3. Second derivative d2/dk2 via nested jvp
    def df_k(k: torch.Tensor) -> torch.Tensor:
        _, d1 = torch.func.jvp(f_k, (k,), (torch.ones_like(k),))  # pyright: ignore[reportAssignmentType]
        res_d1: torch.Tensor = d1
        return res_d1

    _, dd_ref = torch.func.jvp(df_k, (k_val,), (torch.ones_like(k_val),))  # pyright: ignore[reportAssignmentType]

    # 4. Tangents w.r.t p_j
    t_refs = []
    for j in range(P):
        def f_pj(p_j: torch.Tensor, j_idx: int = j) -> torch.Tensor:
            p_args = [p_val[:, i] for i in range(P)]
            p_args[j_idx] = p_j
            return composite_scalar(k_val, p_args[0], p_args[1], p_args[2])

        _, t_j = torch.func.jvp(f_pj, (p_val[:, j],), (torch.ones_like(p_val[:, j]),))  # pyright: ignore[reportAssignmentType]
        t_refs.append(t_j)
    t_ref = torch.stack(t_refs, dim=-1)

    # Evaluation using hand-written Taylor Jet algebra
    eye = torch.eye(P, dtype=torch.float64)
    k_jet = tj.seed_k(k_val, P)
    p0_jet = tj.seed_tangent(p_val[:, 0], 0, P, eye=eye)
    p1_jet = tj.seed_tangent(p_val[:, 1], 1, P, eye=eye)
    p2_jet = tj.seed_tangent(p_val[:, 2], 2, P, eye=eye)

    term1_jet = tj.mul(
        tj.exp(tj.scale(k_jet, 0.5)),
        tj.add(tj.const(torch.ones_like(k_val), P), tj.mul(p0_jet, k_jet)),
    )
    sq_jet = tj.add(
        tj.add(tj.mul(k_jet, k_jet), tj.mul(p1_jet, p1_jet)),
        tj.const(torch.full_like(k_val, 0.1), P),
    )
    term2_jet = tj.sqrt(sq_jet)
    term3_jet = tj.tanh(tj.mul(p2_jet, k_jet))
    rec_jet = tj.recip(tj.add(term2_jet, tj.const(torch.full_like(k_val, 0.5), P)))
    res_jet = tj.add(tj.mul(term1_jet, rec_jet), term3_jet)

    # Assert relative errors <= 1e-12 in float64
    rel_v = torch.max(torch.abs(res_jet.v - v_ref) / torch.clamp(torch.abs(v_ref), min=1e-12)).item()
    rel_d = torch.max(torch.abs(res_jet.d - d_ref) / torch.clamp(torch.abs(d_ref), min=1e-12)).item()
    rel_dd = torch.max(torch.abs(res_jet.dd - dd_ref) / torch.clamp(torch.abs(dd_ref), min=1e-12)).item()
    rel_t = torch.max(torch.abs(res_jet.t - t_ref) / torch.clamp(torch.abs(t_ref), min=1e-12)).item()

    assert rel_v <= 1e-12, f"Primal value relative error {rel_v:.2e} > 1e-12"
    assert rel_d <= 1e-12, f"First derivative relative error {rel_d:.2e} > 1e-12"
    assert rel_dd <= 1e-12, f"Second derivative relative error {rel_dd:.2e} > 1e-12"
    assert rel_t <= 1e-12, f"Tangent relative error {rel_t:.2e} > 1e-12"


def test_silu_jet_exact_derivatives() -> None:
    """Verify SiLU jet matches analytic f, f', f'' to 1e-14."""
    x = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    x.requires_grad_(True)
    P = 0

    x_jet = tj.seed_k(x, P)
    s_jet = tj.silu(x_jet)

    # Analytic comparison
    s = torch.sigmoid(x)
    f_analytic = x * s
    f1_analytic = s + x * s * (1.0 - s)
    f2_analytic = s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))

    assert torch.allclose(s_jet.v, f_analytic, atol=1e-14, rtol=1e-14)
    assert torch.allclose(s_jet.d, f1_analytic, atol=1e-14, rtol=1e-14)
    assert torch.allclose(s_jet.dd, f2_analytic, atol=1e-14, rtol=1e-14)


def test_softfloor_jet_exact_derivatives() -> None:
    """Verify softfloor jet preserves smooth floor and positive derivatives."""
    x = torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)
    P = 1
    lo = 0.01
    beta = 200.0

    x_jet = tj.seed_k(x, P)
    sf = tj.softfloor(x_jet, lo=lo, beta=beta)

    assert torch.all(sf.v >= lo)
    assert torch.all(sf.d > 0.0)
    assert torch.all(sf.dd > 0.0)


def test_mlp_jet_stacked_gemms_vs_autograd() -> None:
    """Verify stacked GEMMs evaluate 1st derivative, 2nd derivative and tangents against autograd."""
    B = 4
    D_in = 3
    P = 2
    torch.manual_seed(42)

    # 1. Production Float32 mode verification
    layers_f32: list[nn.Module] = [
        nn.Linear(D_in, 8),
        nn.SiLU(),
        nn.Linear(8, 1),
    ]
    for layer in layers_f32:
        if isinstance(layer, nn.Linear):
            layer.to(torch.float32)

    k_f64 = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    p0_f64 = torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float64)
    p1_f64 = torch.tensor([2.0, 2.1, 2.2, 2.3], dtype=torch.float64)
    eye = torch.eye(P, dtype=torch.float64)

    cols = [
        tj.seed_k(k_f64, P),
        tj.seed_tangent(p0_f64, 0, P, eye=eye),
        tj.seed_tangent(p1_f64, 1, P, eye=eye),
    ]

    jet_out = tj.mlp_jet(layers_f32, cols)

    # Reference evaluation using torch.autograd.grad in Float32
    k_var = k_f64.clone().float().requires_grad_(True)
    p0_var = p0_f64.clone().float().requires_grad_(True)
    p1_var = p1_f64.clone().float().requires_grad_(True)

    mlp_f32 = nn.Sequential(*layers_f32)
    x_in = torch.stack([k_var, p0_var, p1_var], dim=-1)
    y_f32 = mlp_f32(x_in).squeeze(-1)

    grads = torch.autograd.grad(y_f32.sum(), (k_var, p0_var, p1_var), create_graph=True)
    d_ref_f32 = grads[0]
    t0_ref_f32 = grads[1]
    t1_ref_f32 = grads[2]
    t_ref_f32 = torch.stack([t0_ref_f32, t1_ref_f32], dim=-1)

    dd_ref_f32 = torch.autograd.grad(d_ref_f32.sum(), k_var)[0]

    assert jet_out.v.shape == (B,)
    assert jet_out.d.shape == (B,)
    assert jet_out.dd.shape == (B,)
    assert jet_out.t.shape == (B, P)
    assert torch.all(torch.isfinite(jet_out.v))
    assert torch.all(torch.isfinite(jet_out.d))
    assert torch.all(torch.isfinite(jet_out.dd))
    assert torch.all(torch.isfinite(jet_out.t))

    max_abs_v = (jet_out.v.float() - y_f32).abs().max().item()
    max_abs_d = (jet_out.d.float() - d_ref_f32).abs().max().item()
    max_abs_dd = (jet_out.dd.float() - dd_ref_f32).abs().max().item()
    max_abs_t = (jet_out.t.float() - t_ref_f32).abs().max().item()

    assert max_abs_v <= 1e-7, f"Primal value abs error {max_abs_v:.2e} > 1e-7"
    assert max_abs_d <= 1e-7, f"First derivative abs error {max_abs_d:.2e} > 1e-7"
    assert max_abs_dd <= 1e-7, f"Second derivative abs error {max_abs_dd:.2e} > 1e-7"
    assert max_abs_t <= 1e-7, f"Tangent abs error {max_abs_t:.2e} > 1e-7"

    rel_d_f32 = ((jet_out.d.float() - d_ref_f32).abs() / torch.clamp(d_ref_f32.abs(), min=1e-6)).max().item()
    rel_dd_f32 = ((jet_out.dd.float() - dd_ref_f32).abs() / torch.clamp(dd_ref_f32.abs(), min=1e-6)).max().item()
    rel_t_f32 = ((jet_out.t.float() - t_ref_f32).abs() / torch.clamp(t_ref_f32.abs(), min=1e-6)).max().item()

    assert rel_d_f32 <= 1e-5, f"First derivative rel error {rel_d_f32:.2e} > 1e-5"
    assert rel_dd_f32 <= 1e-5, f"Second derivative rel error {rel_dd_f32:.2e} > 1e-5"
    assert rel_t_f32 <= 1e-5, f"Tangent rel error {rel_t_f32:.2e} > 1e-5"


def test_mlp_jet_surrogate_production_architecture() -> None:
    """Verify Taylor jet propagation on full 13-input, 3x64 SiLU MLP matching ResidualCrossVolNet."""
    B = 8
    D_in = 13
    P = 12
    torch.manual_seed(20261011)

    layers: list[nn.Module] = [
        nn.Linear(D_in, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.to(torch.float32)

    k_val = torch.randn(B, dtype=torch.float64)
    p_vals = [torch.randn(B, dtype=torch.float64) for _ in range(P)]
    eye = torch.eye(P, dtype=torch.float64)

    cols = [tj.seed_k(k_val, P)]
    for j in range(P):
        cols.append(tj.seed_tangent(p_vals[j], j, P, eye=eye))

    jet_out = tj.mlp_jet(layers, cols)

    k_var = k_val.clone().float().requires_grad_(True)
    p_vars = [p.clone().float().requires_grad_(True) for p in p_vals]
    x_in = torch.stack([k_var] + p_vars, dim=-1)

    mlp = nn.Sequential(*layers)
    y = mlp(x_in).squeeze(-1)

    grads = torch.autograd.grad(y.sum(), [k_var] + p_vars, create_graph=True)
    d_ref = grads[0]
    t_ref = torch.stack(grads[1:], dim=-1)
    dd_ref = torch.autograd.grad(d_ref.sum(), k_var)[0]

    max_abs_d = (jet_out.d.float() - d_ref).abs().max().item()
    max_abs_dd = (jet_out.dd.float() - dd_ref).abs().max().item()
    max_abs_t = (jet_out.t.float() - t_ref).abs().max().item()

    assert max_abs_d <= 1e-7, f"Surrogate jet d abs error {max_abs_d:.2e} > 1e-7"
    assert max_abs_dd <= 1e-7, f"Surrogate jet dd abs error {max_abs_dd:.2e} > 1e-7"
    assert max_abs_t <= 1e-7, f"Surrogate jet t abs error {max_abs_t:.2e} > 1e-7"


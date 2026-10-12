from __future__ import annotations

import math
import numpy as np
import torch

from deepvol.market.svi_slice import (
    RawSVI,
    certify_slice,
    durrleman_g,
    fit_raw_svi_5pt,
)
from deepvol.mrm.arbitrage import check_butterfly_arbitrage_durrleman


# ── Test (a): Exact 5-Point Recovery from Known SVI Parameters ──

def test_exact_5pt_recovery() -> None:
    # Known admissible raw SVI parameters [a, b, rho_s, m, s]
    true_params = torch.tensor([
        [0.025, 0.08, -0.25, 0.05, 0.15],
        [0.040, 0.12, 0.35, -0.08, 0.20],
    ], dtype=torch.float64)
    true_svi = RawSVI(true_params)

    # 5 standard forward log-moneyness pillar points x
    x = torch.tensor([
        [-0.30, -0.15, 0.00, 0.15, 0.30],
        [-0.40, -0.20, 0.00, 0.20, 0.40],
    ], dtype=torch.float64)

    w = true_svi.w(x)
    fitted_svi, vol_rmse = fit_raw_svi_5pt(x, w)

    # Vol RMSE should be <= 1e-8 for points generated exactly from raw SVI
    assert torch.all(vol_rmse <= 1e-8), f"Vol RMSE {vol_rmse} exceeds 1e-8"

    # Total variance evaluation should match to machine precision
    w_fitted = fitted_svi.w(x)
    diff = torch.abs(w_fitted - w)
    assert torch.all(diff <= 1e-8), f"Variance difference {diff} exceeds 1e-8"


# ── Test (b): Analytic g vs Float64 Finite-Difference g ──

def test_analytic_vs_fd_durrleman_g() -> None:
    params = torch.tensor([[0.03, 0.10, -0.20, 0.02, 0.18]], dtype=torch.float64)
    svi = RawSVI(params)

    x = torch.linspace(-0.6, 0.6, 121, dtype=torch.float64).unsqueeze(0)
    w, w1, w2 = svi.derivs(x)
    g_analytic = durrleman_g(x, w, w1, w2)

    # Central finite differences for w' and w''
    h = 2e-4
    x_plus = x + h
    x_minus = x - h
    w_plus = svi.w(x_plus)
    w_minus = svi.w(x_minus)

    w1_fd = (w_plus - w_minus) / (2.0 * h)
    w2_fd = (w_plus - 2.0 * w + w_minus) / (h ** 2)

    g_fd = durrleman_g(x, w, w1_fd, w2_fd)

    rel_diff = torch.abs(g_analytic - g_fd) / torch.clamp(torch.abs(g_analytic), min=1e-3)
    max_rel = rel_diff.max().item()
    assert max_rel <= 1e-7, f"Analytic vs FD Durrleman g relative error {max_rel} > 1e-7"


# ── Test (c): Axel Vogt Arbitrageable Slice Rejection ──

def test_axel_vogt_slice_rejection() -> None:
    # Axel Vogt classic arbitrageable slice parameters
    a = -0.0410
    b = 0.1331
    rho_s = 0.3060
    m = 0.3586
    s = 0.4153
    T = 1.0

    vogt_params = torch.tensor([[a, b, rho_s, m, s]], dtype=torch.float64)
    svi = RawSVI(vogt_params)
    sigma0 = torch.tensor([0.20], dtype=torch.float64)
    T_tensor = torch.tensor([T], dtype=torch.float64)

    cert = certify_slice(svi, sigma0, T_tensor, n_grid=401, z_max=12.0)

    # Vogt slice has min g < 0 (approx -0.0328) despite w > 0 everywhere
    assert cert.min_g.item() < 0.0, f"Expected min_g < 0, got {cert.min_g.item()}"
    assert not cert.ok.item(), "Axel Vogt slice must fail certification (ok == False)"
    # Positivity and Lee condition pass for Vogt slice, only butterfly fails
    assert cert.positive_ok.item()
    assert cert.lee_ok.item()


# ── Test (d): Cross-check with check_butterfly_arbitrage_durrleman on 1000 Random Slices ──

def test_cross_check_1000_slices() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    n_slices = 1000

    # Sample random SVI parameters across admissible and butterfly-violating domains
    a = torch.empty(n_slices, dtype=torch.float64).uniform_(0.01, 0.08)
    b = torch.empty(n_slices, dtype=torch.float64).uniform_(0.02, 0.35)
    rho_s = torch.empty(n_slices, dtype=torch.float64).uniform_(-0.85, 0.85)
    m = torch.empty(n_slices, dtype=torch.float64).uniform_(-0.25, 0.25)
    s = torch.empty(n_slices, dtype=torch.float64).uniform_(0.05, 0.40)

    params = torch.stack([a, b, rho_s, m, s], dim=1)
    svi = RawSVI(params)
    T_val = 1.0
    T_tensor = torch.full((n_slices,), T_val, dtype=torch.float64)
    sigma0 = torch.sqrt(torch.clamp(a + b * s * torch.sqrt(1.0 - rho_s ** 2), min=1e-4) / T_val)

    # Analytic certification
    cert = certify_slice(svi, sigma0, T_tensor, n_grid=401, z_max=12.0)

    # Compare against check_butterfly_arbitrage_durrleman on dense grid
    k_grid_np = np.linspace(-1.5, 1.5, 401)
    k_grid_torch = torch.from_numpy(k_grid_np).unsqueeze(0).expand(n_slices, -1)
    w_grid = svi.w(k_grid_torch).numpy()

    mismatches = 0
    for idx in range(n_slices):
        iv_slice = np.sqrt(np.maximum(w_grid[idx:idx+1, :] / T_val, 1e-12))
        res = check_butterfly_arbitrage_durrleman(iv_slice, k_grid_np, np.array([T_val]))
        has_arbitrage_fd = res["has_arbitrage"]
        has_arbitrage_analytic = bool(cert.min_g[idx].item() < -1e-6)

        if has_arbitrage_fd != has_arbitrage_analytic:
            mismatches += 1

    # Over 1000 slices, discrepancy rate between analytic and discrete FD must be negligible (< 1%)
    assert mismatches <= 10, f"Discrepancies {mismatches}/1000 between analytic and FD certifier"


# ── Test (e): Audit E1 Scenarios Certify with min g >= 0.28 ──

def test_audit_e1_scenarios_certify() -> None:
    # Audit E1 quotes for Base 2025 and RUB stress across tenors 1M, 3M, 1Y, 2Y
    e1_quotes = [
        # (name, tenor, leg, atm, rr25, bf25, rr10, bf10)
        ("base_leg1_1Y", 1.0, 0.16, 0.020, 0.005, 0.038, 0.017),
        ("base_leg2_1Y", 1.0, 0.20, 0.060, 0.015, 0.120, 0.050),
        ("stress_leg1_1Y", 1.0, 0.22, 0.030, 0.008, 0.060, 0.025),
        ("stress_leg2_1Y", 1.0, 0.45, 0.150, 0.040, 0.300, 0.120),
        ("base_leg1_1M", 1.0 / 12.0, 0.16, 0.020, 0.005, 0.038, 0.017),
        ("base_leg2_1M", 1.0 / 12.0, 0.20, 0.060, 0.015, 0.120, 0.050),
        ("stress_leg1_1M", 1.0 / 12.0, 0.22, 0.030, 0.008, 0.060, 0.025),
        ("stress_leg2_1M", 1.0 / 12.0, 0.45, 0.150, 0.040, 0.300, 0.120),
    ]

    for name, T, atm, rr25, bf25, rr10, bf10 in e1_quotes:
        vols = [
            atm + bf10 - 0.5 * rr10,
            atm + bf25 - 0.5 * rr25,
            atm,
            atm + bf25 + 0.5 * rr25,
            atm + bf10 + 0.5 * rr10,
        ]
        deltas = [-0.10, -0.25, 0.0, 0.25, 0.10]
        ks = []
        for d, v in zip(deltas, vols):
            if d == 0.0:
                k = 0.5 * (v ** 2) * T
            else:
                phi = 1.0 if d > 0 else -1.0
                d1 = phi * float(torch.special.ndtri(torch.tensor(phi * d)).item())
                k = -d1 * v * math.sqrt(T) + 0.5 * (v ** 2) * T
            ks.append(k)

        x_tensor = torch.tensor([ks], dtype=torch.float64)
        w_tensor = torch.tensor([[v ** 2 * T for v in vols]], dtype=torch.float64)
        sigma0 = torch.tensor([atm], dtype=torch.float64)
        T_tensor = torch.tensor([T], dtype=torch.float64)

        svi, rmse = fit_raw_svi_5pt(x_tensor, w_tensor)
        cert = certify_slice(svi, sigma0, T_tensor, n_grid=401, z_max=6.0)

        assert cert.ok.item(), f"E1 scenario {name} failed certification"
        assert cert.min_g.item() >= 0.28, f"E1 scenario {name} min_g {cert.min_g.item():.4f} < 0.28"

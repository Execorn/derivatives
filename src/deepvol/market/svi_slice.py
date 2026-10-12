from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import torch
from torch import Tensor

__all__ = [
    "Certification",
    "RawSVI",
    "_fit_single_slice",
    "certify_slice",
    "durrleman_g",
    "fit_raw_svi_5pt",
]


@dataclass(frozen=True)
class RawSVI:
    """Raw SVI parametric total variance slice container.

    Attributes:
        params: Float64 tensor of shape (S, 5) storing [a, b, rho_s, m, s] per market state.
    """
    params: Tensor

    def __post_init__(self) -> None:
        if self.params.dim() == 1:
            object.__setattr__(self, "params", self.params.unsqueeze(0))

    def w(self, x: Tensor) -> Tensor:
        """Evaluates total variance w(x) = a + b * (rho_s * (x - m) + sqrt((x - m)^2 + s^2)).

        Args:
            x: Log-moneyness tensor of shape (S, N) or (N,), float64.

        Returns:
            Total variance tensor of shape (S, N), float64.
        """
        x_in = x.unsqueeze(0) if x.dim() == 1 and self.params.shape[0] == 1 else x
        a = self.params[:, 0:1]
        b = self.params[:, 1:2]
        rho_s = self.params[:, 2:3]
        m = self.params[:, 3:4]
        s = self.params[:, 4:5]

        delta = x_in - m
        hypot = torch.sqrt(delta ** 2 + s ** 2)
        return a + b * (rho_s * delta + hypot)

    def derivs(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluates analytic total variance w, w', and w''.

        Args:
            x: Log-moneyness tensor of shape (S, N) or (N,), float64.

        Returns:
            Tuple (w, w', w'') each of shape (S, N), float64.
        """
        x_in = x.unsqueeze(0) if x.dim() == 1 and self.params.shape[0] == 1 else x
        a = self.params[:, 0:1]
        b = self.params[:, 1:2]
        rho_s = self.params[:, 2:3]
        m = self.params[:, 3:4]
        s = self.params[:, 4:5]

        delta = x_in - m
        hypot = torch.sqrt(delta ** 2 + s ** 2)

        w_val = a + b * (rho_s * delta + hypot)
        w1_val = b * (rho_s + delta / hypot)
        w2_val = b * (s ** 2) / (hypot ** 3)
        return w_val, w1_val, w2_val


@dataclass(frozen=True)
class Certification:
    """Certification result container for butterfly and asymptotic no-arbitrage bounds."""
    min_g: Tensor
    lee_ok: Tensor
    positive_ok: Tensor
    ok: Tensor


def durrleman_g(x: Tensor, w: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """Evaluates Durrleman butterfly function g(x) representing risk-neutral probability density.

    Args:
        x: Log-moneyness tensor of shape (S, N), float64.
        w: Total variance tensor of shape (S, N), float64.
        w1: First derivative w'(x) of shape (S, N), float64.
        w2: Second derivative w''(x) of shape (S, N), float64.

    Returns:
        Durrleman function g(x) of shape (S, N), float64.
    """
    w_safe = torch.clamp(w, min=1e-12)
    term1 = (1.0 - (x * w1) / (2.0 * w_safe)) ** 2
    term2 = (w1 ** 2) / 4.0 * (1.0 / w_safe + 0.25)
    term3 = w2 / 2.0
    return cast(Tensor, term1 - term2 + term3)


def certify_slice(
    svi: RawSVI,
    sigma0: Tensor,
    T: Tensor,
    *,
    n_grid: int = 401,
    z_max: float = 12.0,
) -> Certification:
    """Certifies a raw-SVI slice against butterfly arbitrage, Lee bounds, and strict positivity.

    Args:
        svi: RawSVI slice container.
        sigma0: Reference ATM volatility of shape (S,), float64.
        T: Maturity of shape (S,), float64.
        n_grid: Number of grid evaluation points.
        z_max: Moneyness grid bound multiplier.

    Returns:
        Certification container.
    """
    dev = svi.params.device

    u = torch.linspace(-1.0, 1.0, n_grid, dtype=torch.float64, device=dev)
    x_span = z_max * sigma0 * torch.sqrt(T)
    x_grid = u.unsqueeze(0) * x_span.unsqueeze(1)

    w, w1, w2 = svi.derivs(x_grid)
    g = durrleman_g(x_grid, w, w1, w2)

    min_g = torch.min(g, dim=-1).values
    min_w = torch.min(w, dim=-1).values

    a = svi.params[:, 0]
    b = svi.params[:, 1]
    rho_s = svi.params[:, 2]
    s = svi.params[:, 4]

    # Asymptotic total-variance wing slope condition: b * (1 + |rho_s|) <= 2
    lee_ok = (b * (1.0 + torch.abs(rho_s))) <= (2.0 + 1e-10)

    # Minimum total variance positivity: a + b * s * sqrt(1 - rho_s^2) > 0
    radicand = torch.clamp(1.0 - (rho_s ** 2), min=0.0)
    w_min_analytic = a + b * s * torch.sqrt(radicand)
    positive_ok = (w_min_analytic > 0.0) & (min_w > 0.0)

    g_ok = min_g >= -1e-12
    admissible_b = b >= 0.0
    admissible_rho = torch.abs(rho_s) < 1.0
    admissible_s = s > 0.0

    ok = lee_ok & positive_ok & g_ok & admissible_b & admissible_rho & admissible_s
    return Certification(min_g=min_g, lee_ok=lee_ok, positive_ok=positive_ok, ok=ok)


def fit_raw_svi_5pt(
    x: Tensor,
    w: Tensor,
    *,
    lm_iters: int = 25,
    init_grid: int = 9,
) -> Tuple[RawSVI, Tensor]:
    """Fits 5-point raw-SVI total variance slice via batched quasi-explicit lstsq and outer Levenberg-Marquardt.

    Fully vectorized across the batch dimension S without sequential Python loops.

    Args:
        x: Log-moneyness tensor of shape (S, 5) or (5,), float64.
        w: Total variance tensor of shape (S, 5) or (5,), float64.
        lm_iters: Levenberg-Marquardt iteration count (default 25).
        init_grid: Initial grid search density per dimension (default 9, creating 81 candidates).

    Returns:
        Tuple of (RawSVI, fit_rmse_vol) where fit_rmse_vol has shape (S,) or scalar.
    """
    dev = x.device
    is_1d = (x.dim() == 1)
    x_in = x.unsqueeze(0) if is_1d else x
    w_in = w.unsqueeze(0) if is_1d else w
    S = x_in.shape[0]

    x_min = torch.min(x_in, dim=-1, keepdim=True).values
    x_max = torch.max(x_in, dim=-1, keepdim=True).values
    x_span = x_max - x_min
    x_span = torch.where(x_span < 1e-4, torch.full_like(x_span, 0.2), x_span)
    x_mid = 0.5 * (x_max + x_min)

    # 1. Deterministic grid search over (m, s) across all S states simultaneously: shape (S, G)
    u_m = torch.linspace(-0.5, 0.5, init_grid, dtype=torch.float64, device=dev)
    u_s = torch.linspace(0.02, 0.8, init_grid, dtype=torch.float64, device=dev)
    U_m, U_s = torch.meshgrid(u_m, u_s, indexing="ij")
    u_m_flat = U_m.reshape(1, -1)
    u_s_flat = U_s.reshape(1, -1)

    m_grid = x_mid + u_m_flat * x_span  # (S, G)
    s_grid = u_s_flat * x_span          # (S, G)

    y_grid = (x_in.unsqueeze(1) - m_grid.unsqueeze(-1)) / s_grid.unsqueeze(-1)  # (S, G, 5)
    col0 = torch.ones_like(y_grid)
    col1 = y_grid
    col2 = torch.sqrt(y_grid ** 2 + 1.0)
    A_grid = torch.stack([col0, col1, col2], dim=-1)  # (S, G, 5, 3)

    At_g = A_grid.transpose(-2, -1)  # (S, G, 3, 5)
    AtA_g = At_g @ A_grid + 1e-12 * torch.eye(3, dtype=torch.float64, device=dev)  # (S, G, 3, 3)
    w_exp = w_in.unsqueeze(1).unsqueeze(-1)  # (S, 1, 5, 1)
    Atw_g = At_g @ w_exp  # (S, G, 3, 1)

    sol_g = torch.linalg.solve(AtA_g, Atw_g)  # (S, G, 3, 1)
    a_g = sol_g[..., 0, 0]
    d_g = sol_g[..., 1, 0]
    c_g = sol_g[..., 2, 0]

    rho_g = d_g / torch.clamp(c_g, min=1e-12)
    b_g = c_g / s_grid
    radicand_g = torch.clamp(c_g ** 2 - d_g ** 2, min=0.0)
    w_min_g = a_g + torch.sqrt(radicand_g)
    lee_g = b_g * (1.0 + torch.abs(rho_g))

    valid_g = (c_g > 1e-8) & (torch.abs(rho_g) < 0.999) & (w_min_g > 0.0) & (lee_g <= 2.01)
    pred_g = (A_grid @ sol_g).squeeze(-1)  # (S, G, 5)
    loss_g = torch.sum((pred_g - w_in.unsqueeze(1)) ** 2, dim=-1)  # (S, G)
    loss_g = torch.where(valid_g, loss_g, torch.full_like(loss_g, float("inf")))

    best_idx = torch.argmin(loss_g, dim=-1, keepdim=True)  # (S, 1)
    best_loss = torch.gather(loss_g, 1, best_idx).squeeze(1)  # (S,)
    best_m = torch.gather(m_grid, 1, best_idx).squeeze(1)     # (S,)
    best_s = torch.gather(s_grid, 1, best_idx).squeeze(1)     # (S,)
    best_a = torch.gather(a_g, 1, best_idx).squeeze(1)        # (S,)
    best_b = torch.gather(b_g, 1, best_idx).squeeze(1)        # (S,)
    best_rho = torch.gather(rho_g, 1, best_idx).squeeze(1)    # (S,)

    # Fallback for any slice where the initial 9x9 grid was completely inadmissible
    inadmissible_mask = torch.isinf(best_loss)
    if torch.any(inadmissible_mask):
        m_w = torch.linspace(-1.5, 1.5, 21, dtype=torch.float64, device=dev)
        s_w = torch.linspace(0.005, 2.0, 21, dtype=torch.float64, device=dev)
        Mw, Sw = torch.meshgrid(m_w, s_w, indexing="ij")
        u_mw = Mw.reshape(1, -1)
        u_sw = Sw.reshape(1, -1)

        bad_idx = torch.where(inadmissible_mask)[0]
        x_bad = x_in[bad_idx]
        w_bad = w_in[bad_idx]
        x_span_bad = x_span[bad_idx]
        x_mid_bad = x_mid[bad_idx]

        m_gw = x_mid_bad + u_mw * x_span_bad
        s_gw = u_sw * x_span_bad
        y_gw = (x_bad.unsqueeze(1) - m_gw.unsqueeze(-1)) / s_gw.unsqueeze(-1)
        A_gw = torch.stack([torch.ones_like(y_gw), y_gw, torch.sqrt(y_gw ** 2 + 1.0)], dim=-1)
        At_gw = A_gw.transpose(-2, -1)
        AtA_gw = At_gw @ A_gw + 1e-12 * torch.eye(3, dtype=torch.float64, device=dev)
        w_exp_bad = w_bad.unsqueeze(1).unsqueeze(-1)
        sol_gw = torch.linalg.solve(AtA_gw, At_gw @ w_exp_bad)

        a_gw = sol_gw[..., 0, 0]
        d_gw = sol_gw[..., 1, 0]
        c_gw = sol_gw[..., 2, 0]
        rho_gw = d_gw / torch.clamp(c_gw, min=1e-12)
        b_gw = c_gw / s_gw
        rad_gw = torch.clamp(c_gw ** 2 - d_gw ** 2, min=0.0)
        w_min_gw = a_gw + torch.sqrt(rad_gw)
        lee_gw = b_gw * (1.0 + torch.abs(rho_gw))
        val_gw = (c_gw > 1e-8) & (torch.abs(rho_gw) < 0.999) & (w_min_gw > 0.0) & (lee_gw <= 2.01)
        pred_gw = (A_gw @ sol_gw).squeeze(-1)
        loss_gw = torch.sum((pred_gw - w_bad.unsqueeze(1)) ** 2, dim=-1)
        loss_gw = torch.where(val_gw, loss_gw, torch.full_like(loss_gw, float("inf")))

        b_idx_w = torch.argmin(loss_gw, dim=-1, keepdim=True)
        best_m[bad_idx] = torch.gather(m_gw, 1, b_idx_w).squeeze(1)
        best_s[bad_idx] = torch.gather(s_gw, 1, b_idx_w).squeeze(1)
        best_a[bad_idx] = torch.gather(a_gw, 1, b_idx_w).squeeze(1)
        best_b[bad_idx] = torch.gather(b_gw, 1, b_idx_w).squeeze(1)
        best_rho[bad_idx] = torch.gather(rho_gw, 1, b_idx_w).squeeze(1)
        best_loss[bad_idx] = torch.gather(loss_gw, 1, b_idx_w).squeeze(1)

    # 2. Vectorized Levenberg-Marquardt polish in (m, ln s)
    m_curr = best_m.clone()
    eta_curr = torch.log(torch.clamp(best_s, min=1e-6))
    lam = torch.full((S,), 1e-3, dtype=torch.float64, device=dev)
    curr_loss = best_loss.clone()

    x_span_1d = x_span.squeeze(-1)
    hm = 1e-6 * x_span_1d
    heta = 1e-6

    def eval_residuals_and_params(m_v: Tensor, eta_v: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s_v = torch.exp(eta_v)
        y_v = (x_in.unsqueeze(1) - m_v.unsqueeze(-1)) / s_v.unsqueeze(-1)
        A_v = torch.stack([torch.ones_like(y_v), y_v, torch.sqrt(y_v ** 2 + 1.0)], dim=-1)
        At_v = A_v.transpose(-2, -1)
        AtA_v = At_v @ A_v + 1e-12 * torch.eye(3, dtype=torch.float64, device=dev)
        w_v = w_in.unsqueeze(1).unsqueeze(-1)
        sol_v = torch.linalg.solve(AtA_v, At_v @ w_v)
        r_v = (A_v @ sol_v - w_v).squeeze(-1)

        a_v = sol_v[..., 0, 0]
        d_v = sol_v[..., 1, 0]
        c_v = sol_v[..., 2, 0]
        rho_val = d_v / torch.clamp(c_v, min=1e-12)
        b_val = c_v / s_v
        w_min_val = a_v + torch.sqrt(torch.clamp(c_v ** 2 - d_v ** 2, min=0.0))
        lee_val = b_val * (1.0 + torch.abs(rho_val))

        valid_v = (c_v > 1e-8) & (torch.abs(rho_val) < 0.999) & (w_min_val > 0.0) & (lee_val <= 2.01)
        loss_v = torch.sum(r_v ** 2, dim=-1)
        loss_v = torch.where(valid_v, loss_v, torch.full_like(loss_v, float("inf")))

        return r_v, a_v, b_val, rho_val, s_v, loss_v

    best_params = torch.stack([best_a, best_b, best_rho, best_m, best_s], dim=-1)

    for _ in range(lm_iters):
        if torch.all(curr_loss < 1e-22):
            break

        # 5-point central stencil: [base, +hm, -hm, +heta, -heta]
        m_pert = torch.stack([m_curr, m_curr + hm, m_curr - hm, m_curr, m_curr], dim=1)
        eta_pert = torch.stack([eta_curr, eta_curr, eta_curr, eta_curr + heta, eta_curr - heta], dim=1)

        r_all, _, _, _, _, _ = eval_residuals_and_params(m_pert, eta_pert)

        r_base = r_all[:, 0, :]
        r_mp   = r_all[:, 1, :]
        r_mm   = r_all[:, 2, :]
        r_ep   = r_all[:, 3, :]
        r_em   = r_all[:, 4, :]

        hm_exp = hm.unsqueeze(-1)
        J0 = (r_mp - r_mm) / (2.0 * hm_exp)
        J1 = (r_ep - r_em) / (2.0 * heta)

        J = torch.stack([J0, J1], dim=-1)  # (S, 5, 2)
        Jt = J.transpose(-2, -1)           # (S, 2, 5)

        H = Jt @ J + lam.view(S, 1, 1) * torch.eye(2, dtype=torch.float64, device=dev)  # (S, 2, 2)
        grad = (Jt @ r_base.unsqueeze(-1)).squeeze(-1)  # (S, 2)

        if torch.all(curr_loss < 1e-22):
            break

        # 2x2 batched closed-form solve
        det = H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] * H[:, 1, 0]
        delta0 = (H[:, 1, 1] * grad[:, 0] - H[:, 0, 1] * grad[:, 1]) / det
        delta1 = (-H[:, 1, 0] * grad[:, 0] + H[:, 0, 0] * grad[:, 1]) / det

        m_cand = m_curr - delta0
        eta_cand = eta_curr - delta1

        _, a_c, b_c, rho_c, s_c, loss_c = eval_residuals_and_params(m_cand.unsqueeze(1), eta_cand.unsqueeze(1))
        loss_cand = loss_c[:, 0]

        accept_mask = (loss_cand < curr_loss)

        m_curr = torch.where(accept_mask, m_cand, m_curr)
        eta_curr = torch.where(accept_mask, eta_cand, eta_curr)
        curr_loss = torch.where(accept_mask, loss_cand, curr_loss)
        lam = torch.where(accept_mask, torch.clamp(lam / 3.0, min=1e-12), torch.clamp(lam * 4.0, max=1e8))

        new_params = torch.stack([a_c[:, 0], b_c[:, 0], rho_c[:, 0], m_cand, s_c[:, 0]], dim=-1)
        best_params = torch.where(accept_mask.unsqueeze(-1), new_params, best_params)

    a_f = best_params[:, 0:1]
    b_f = best_params[:, 1:2]
    rho_f = best_params[:, 2:3]
    m_f = best_params[:, 3:4]
    s_f = best_params[:, 4:5]

    delta_f = x_in - m_f
    hypot_f = torch.sqrt(delta_f ** 2 + s_f ** 2)
    w_pred = a_f + b_f * (rho_f * delta_f + hypot_f)
    vol_rmse = torch.sqrt(torch.mean((torch.sqrt(torch.clamp(w_pred, min=1e-12)) - torch.sqrt(torch.clamp(w_in, min=1e-12))) ** 2, dim=-1))

    if is_1d:
        return RawSVI(best_params[0]), vol_rmse[0]
    return RawSVI(best_params), vol_rmse


def _fit_single_slice(
    x_5: Tensor,
    w_5: Tensor,
    lm_iters: int = 25,
    init_grid: int = 9,
) -> Tuple[Tuple[float, float, float, float, float], float]:
    """Backward compatibility wrapper delegating single slice fitting to vectorized solver."""
    svi, rmse = fit_raw_svi_5pt(x_5.unsqueeze(0), w_5.unsqueeze(0), lm_iters=lm_iters, init_grid=init_grid)
    p = svi.params[0].tolist()
    return (float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])), float(rmse[0].item())


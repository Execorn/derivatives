from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, cast

import torch
from torch import Tensor


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


def _fit_single_slice(
    x_5: Tensor,
    w_5: Tensor,
    lm_iters: int,
    init_grid: int,
) -> Tuple[Tuple[float, float, float, float, float], float]:
    x_span = (x_5.max() - x_5.min()).item()
    if x_span < 1e-4:
        x_span = 0.2
    x_mid = 0.5 * (x_5.max() + x_5.min()).item()

    m_grid = torch.linspace(x_mid - 0.5 * x_span, x_mid + 0.5 * x_span, init_grid, dtype=torch.float64, device=x_5.device)
    s_grid = torch.linspace(0.02 * x_span, 0.8 * x_span, init_grid, dtype=torch.float64, device=x_5.device)
    M, S_mesh = torch.meshgrid(m_grid, s_grid, indexing="ij")
    m_flat = M.reshape(-1)
    s_flat = S_mesh.reshape(-1)

    # Vectorized initial grid solve across all (m, s) pairs
    y_grid = (x_5.unsqueeze(0) - m_flat.unsqueeze(1)) / s_flat.unsqueeze(1)
    A_grid = torch.stack([torch.ones_like(y_grid), y_grid, torch.sqrt(y_grid ** 2 + 1.0)], dim=-1)
    w_exp = w_5.unsqueeze(0).expand(m_flat.shape[0], 5).unsqueeze(-1)

    At_g = A_grid.transpose(-2, -1)
    AtA_g = At_g @ A_grid + 1e-12 * torch.eye(3, dtype=torch.float64, device=x_5.device)
    sol_g = torch.linalg.solve(AtA_g, At_g @ w_exp).squeeze(-1)

    a_g = sol_g[:, 0]
    d_g = sol_g[:, 1]
    c_g = sol_g[:, 2]

    rho_g = d_g / torch.clamp(c_g, min=1e-12)
    b_g = c_g / s_flat
    radicand_g = torch.clamp(c_g ** 2 - d_g ** 2, min=0.0)
    w_min_g = a_g + torch.sqrt(radicand_g)
    lee_g = b_g * (1.0 + torch.abs(rho_g))

    valid_g = (c_g > 1e-8) & (torch.abs(rho_g) < 0.999) & (w_min_g > 0.0) & (lee_g <= 2.01)
    pred_g = (A_grid @ sol_g.unsqueeze(-1)).squeeze(-1)
    loss_g = torch.sum((pred_g - w_5.unsqueeze(0)) ** 2, dim=-1)
    loss_g = torch.where(valid_g, loss_g, torch.tensor(float("inf"), dtype=torch.float64, device=x_5.device))

    best_idx = int(torch.argmin(loss_g).item())
    best_loss = float(loss_g[best_idx].item())
    best_p: Tuple[float, float, float, float, float] = (
        float(a_g[best_idx].item()),
        float(b_g[best_idx].item()),
        float(rho_g[best_idx].item()),
        float(m_flat[best_idx].item()),
        float(s_flat[best_idx].item()),
    )
    best_m = float(m_flat[best_idx].item())
    best_s = float(s_flat[best_idx].item())

    def solve_linear(m_val: float, s_val: float) -> Tuple[Tuple[float, float, float, float, float] | None, float]:
        y = (x_5 - m_val) / s_val
        col0 = torch.ones_like(y)
        col1 = y
        col2 = torch.sqrt(y ** 2 + 1.0)
        A = torch.stack([col0, col1, col2], dim=1)
        AtA = A.T @ A + 1e-12 * torch.eye(3, dtype=torch.float64, device=x_5.device)
        sol = torch.linalg.solve(AtA, A.T @ w_5)
        a_val, d_val, c_val = sol[0].item(), sol[1].item(), sol[2].item()
        if c_val <= 1e-8:
            return None, float("inf")
        b_val = c_val / s_val
        rho_val = d_val / c_val
        if abs(rho_val) >= 0.999:
            return None, float("inf")
        if a_val + math.sqrt(max(c_val ** 2 - d_val ** 2, 0.0)) <= 0.0:
            return None, float("inf")
        if b_val * (1.0 + abs(rho_val)) > 2.01:
            return None, float("inf")
        pred = A @ sol
        loss_val = torch.sum((pred - w_5) ** 2).item()
        return (a_val, b_val, rho_val, m_val, s_val), loss_val

    # Wider search if default grid failed to produce an admissible candidate
    if best_loss == float("inf"):
        for m_val in torch.linspace(x_mid - 1.5 * x_span, x_mid + 1.5 * x_span, 21, dtype=torch.float64, device=x_5.device).tolist():
            for s_val in torch.linspace(0.005 * x_span, 2.0 * x_span, 21, dtype=torch.float64, device=x_5.device).tolist():
                p_cand, loss = solve_linear(m_val, s_val)
                if p_cand is not None and loss < best_loss:
                    best_loss = loss
                    best_p = p_cand
                    best_m = m_val
                    best_s = s_val

    # Levenberg-Marquardt polish in (m, ln s)
    m_curr = best_m
    eta_curr = math.log(max(best_s, 1e-6))
    lam = 1e-3
    curr_loss = best_loss

    for _ in range(lm_iters):
        if curr_loss < 1e-22:
            break
        s_curr = math.exp(eta_curr)
        hm = 1e-6 * x_span
        heta = 1e-6

        def get_r(m_v: float, s_v: float) -> Tensor:
            y_v = (x_5 - m_v) / s_v
            A_v = torch.stack([torch.ones_like(y_v), y_v, torch.sqrt(y_v ** 2 + 1.0)], dim=1)
            AtA_v = A_v.T @ A_v + 1e-12 * torch.eye(3, dtype=torch.float64, device=x_5.device)
            sol_v = torch.linalg.solve(AtA_v, A_v.T @ w_5)
            return cast(Tensor, A_v @ sol_v - w_5)

        r = get_r(m_curr, s_curr)
        r_mp = get_r(m_curr + hm, s_curr)
        r_mm = get_r(m_curr - hm, s_curr)
        J0 = (r_mp - r_mm) / (2.0 * hm)

        r_ep = get_r(m_curr, math.exp(eta_curr + heta))
        r_em = get_r(m_curr, math.exp(eta_curr - heta))
        J1 = (r_ep - r_em) / (2.0 * heta)

        J = torch.stack([J0, J1], dim=1)
        H = J.T @ J + lam * torch.eye(2, dtype=torch.float64, device=x_5.device)
        grad = J.T @ r
        step = torch.linalg.solve(H, grad)

        m_new = m_curr - step[0].item()
        eta_new = eta_curr - step[1].item()
        s_new = math.exp(eta_new)

        p_new, new_loss = solve_linear(m_new, s_new)
        if p_new is not None and new_loss < curr_loss:
            m_curr = m_new
            eta_curr = eta_new
            best_p = p_new
            curr_loss = new_loss
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 4.0, 1e8)

    a_f, b_f, r_f, m_f, s_f = best_p
    delta_f = x_5 - m_f
    hypot_f = torch.sqrt(delta_f ** 2 + s_f ** 2)
    w_pred = a_f + b_f * (r_f * delta_f + hypot_f)
    vol_rmse = torch.sqrt(torch.mean((torch.sqrt(torch.clamp(w_pred, min=1e-12)) - torch.sqrt(torch.clamp(w_5, min=1e-12))) ** 2)).item()

    return best_p, vol_rmse


def fit_raw_svi_5pt(
    x: Tensor,
    w: Tensor,
    *,
    lm_iters: int = 60,
    init_grid: int = 9,
) -> Tuple[RawSVI, Tensor]:
    """Fits 5-point raw-SVI total variance slice via quasi-explicit lstsq and outer Levenberg-Marquardt.

    Args:
        x: Log-moneyness tensor of shape (S, 5), float64.
        w: Total variance tensor of shape (S, 5), float64.
        lm_iters: Levenberg-Marquardt iteration count.
        init_grid: Initial grid search density per dimension.

    Returns:
        Tuple of (RawSVI, fit_rmse_vol) where fit_rmse_vol has shape (S,).
    """
    S_len = x.shape[0]
    dev = x.device

    if S_len > 1 and torch.all(x == x[0]) and torch.all(w == w[0]):
        p_fit, rmse_val = _fit_single_slice(x[0], w[0], lm_iters, init_grid)
        params_tensor = torch.tensor([list(p_fit)] * S_len, dtype=torch.float64, device=dev)
        rmse_tensor = torch.full((S_len,), rmse_val, dtype=torch.float64, device=dev)
        return RawSVI(params_tensor), rmse_tensor

    params_list = []
    rmse_list = []

    for i in range(S_len):
        p_fit, rmse_val = _fit_single_slice(x[i], w[i], lm_iters, init_grid)
        params_list.append(list(p_fit))
        rmse_list.append(rmse_val)

    params_tensor = torch.tensor(params_list, dtype=torch.float64, device=dev)
    rmse_tensor = torch.tensor(rmse_list, dtype=torch.float64, device=dev)
    return RawSVI(params_tensor), rmse_tensor

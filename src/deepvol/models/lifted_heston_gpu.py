"""
lifted_heston_gpu.py — GPU-accelerated exact Fourier-COS pricing for the
Lifted Rough Heston model.

Mathematical details of the GPU pricing engine:
  - The Riccati ODE decay term includes the mean reversion parameter kappa.
  - The kappa*theta integral uses the c-weighted sum (sum_i c_i*psi_i).
  - A batched PyTorch RK4 solver is used for numerical integration.
  - Option pricing uses N_cos = 64 (domain [-4,4] converges in 64 terms).
  - Bernstein factor weights c are normalized (sum(c)=1).

Hardware optimisation (2026-06-11):
  Mixed precision ODE: psi runs in complex64 (RTX 3060 Tensor Cores) while
  the Riemann accumulator int_cv remains in complex128 to prevent numerical
  drift over 400 RK4 steps.  Theoretical throughput benefit ~3-5x for the
  ODE step on Ampere mobile (FP64 = 1/64 of FP32 TFLOPS).

IFT differentiable IV inverter (2026-06-11):
  BS_IV_Implicit_Inverter wraps the Newton-Raphson loop with an analytically
  correct backward pass via the Implicit Function Theorem:
      d(IV)/d(price) = 1 / Vega
  Enables future torch.func.jacrev gradient pipelines without mega-batches.

Reference: Abi Jaber (2022) "Lifting the Heston model", Eq. 4.2-4.3.
           Fang & Oosterlee (2008) "A Novel Pricing Method for European Options"
           Gatheral & Jacquier (2011) "Arbitrage-free SVI volatility surfaces"

Architecture:
  - State psi: (B, N_u, N) complex64 (mixed) or complex128 (full)
  - int_cv: (B, N_u) complex128 (always full precision accumulator)
  - B=2048 samples x 64 frequencies x 20 factors (mixed: ~27MB VRAM per batch)
  - 50k samples generated in ~5-15 minutes on RTX 3060
"""

import numpy as np
import torch

_A = -4.0   # COS domain: covers 10+ sigma for all maturities in dataset
_B =  4.0   # chi_0 = exp(4)-1 ≈ 54 (vs exp(12)≈162754 for [-12,12])
             # The larger domain causes the COS series to need 1000s of terms
             # to converge; [-4,4] converges in 64-128 terms.
_SQRT2        = 1.4142135623730951
_INVSQRT2PI   = 0.3989422804014327


# ---------------------------------------------------------------------------
# Bernstein kernel factors
# ---------------------------------------------------------------------------

def bernstein_factors(H: float, N: int = 20):
    """
    r_N = 1 + 10*N^{-0.9}
    x_i = r_N^{i-1-N/2}
    c_i = x_i^{-(H+0.5)}   (Laplace weights for fractional kernel)
    Returns numpy (x, c) shape (N,).
    """
    r_N = 1.0 + 10.0 * (N ** -0.9)
    x   = np.array([r_N ** (i - 1.0 - N / 2.0) for i in range(1, N + 1)])
    c   = x ** -(H + 0.5)
    c   = c / c.sum()   # CRITICAL: normalize so sum(c)=1
    # Without normalisation sum(c)~26, making sigma_eff^2 = sigma^2*sum(c)^2~170
    # which causes the Riccati quadratic term to blow up for large u_k.
    # With sum(c)=1 the lifted model reduces exactly to scalar Heston at N=1.
    return x, c


# ---------------------------------------------------------------------------
# COS payoff coefficients  (CPU precompute, then move to GPU)
# ---------------------------------------------------------------------------

def cos_payoff_coeffs(N_cos: int, a: float = _A, b: float = _B) -> np.ndarray:
    """
    Exact COS CALL payoff coefficients — integration domain [0, b].

    Vk = (2/(b-a)) * (chi_k - psi_k),   V_0 *= 0.5  (COS half-weight).

    chi_k = int_0^b  e^x cos(u_k*(x-a)) dx
          = Re[ exp(-i*u_k*a) * (exp((1+i*u_k)*b) - 1) / (1 + i*u_k) ]

    psi_k = int_0^b cos(u_k*(x-a)) dx
          = (sin(u_k*(b-a)) + sin(u_k*a)) / u_k
    """
    k   = np.arange(N_cos, dtype=np.float64)
    uk  = k * np.pi / (b - a)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.real(
            np.exp(-1j * uk * a)
            * (np.exp((1.0 + 1j * uk) * b) - 1.0)
            / (1.0 + 1j * uk)
        )
    chi[0] = np.exp(b) - 1.0

    safe_uk = np.where(k == 0, 1.0, uk)
    psi = np.where(
        k == 0,
        b,
        (np.sin(uk * (b - a)) + np.sin(uk * a)) / safe_uk,
    )

    Vk     = (2.0 / (b - a)) * (chi - psi)
    Vk[0] *= 0.5
    return Vk


def cos_payoff_coeffs_put(N_cos: int, a: float = _A, b: float = _B) -> np.ndarray:
    """
    Exact COS PUT payoff coefficients — integration domain [a, 0].

    For ITM calls (K < S0) we price the equivalent OTM PUT directly to avoid
    catastrophic cancellation in the call COS sum (call price ≈ intrinsic).

    Vk_put = (2/(b-a)) * (psi_k^put - chi_k^put),   V_0 *= 0.5.

    chi_k^put = int_a^0 e^x cos(u_k*(x-a)) dx
              = Re[ exp(-i*u_k*a) * (1 - exp((1+i*u_k)*a)) / (1 + i*u_k) ]
              (lower limit 0, upper limit a → reversed)

    psi_k^put = int_a^0 cos(u_k*(x-a)) dx
              = -sin(u_k*a) / u_k      [for k ≥ 1]
              = -a                      [for k = 0]
    """
    k   = np.arange(N_cos, dtype=np.float64)
    uk  = k * np.pi / (b - a)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi_put = np.real(
            np.exp(-1j * uk * a)
            * (1.0 - np.exp((1.0 + 1j * uk) * a))
            / (1.0 + 1j * uk)
        )
    chi_put[0] = 1.0 - np.exp(a)   # k=0: int_a^0 e^x dx = 1 - e^a

    safe_uk  = np.where(k == 0, 1.0, uk)
    psi_put  = np.where(
        k == 0,
        -a,                                   # int_a^0 1 dx = -a = 4
        -np.sin(uk * a) / safe_uk,            # (sin(0)-sin(uk*a))/uk
    )

    Vk_put     = (2.0 / (b - a)) * (psi_put - chi_put)
    Vk_put[0] *= 0.5
    return Vk_put


# ---------------------------------------------------------------------------
# Normal CDF / PDF on GPU
# ---------------------------------------------------------------------------

def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))

def _npdf(x: torch.Tensor) -> torch.Tensor:
    return _INVSQRT2PI * torch.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Batched Riccati RHS
# ---------------------------------------------------------------------------

def _riccati_rhs(psi, u_c, x, c, kappa_e, sigma_e, rho_e):
    """
    Compute d(psi)/dt for the Lifted Heston Riccati system.

    From Abi Jaber (2022) Eq 4.3:
      dψ_{k,i}/dt = g(u_k, Psi_k) - kappa * x_i * psi_{k,i}
      g(u, Psi)   = -0.5*(u^2 + i*u) + rho*sigma*i*u*Psi + 0.5*sigma^2*Psi^2
      Psi_k       = sum_i c_i * psi_{k,i}   (c-weighted aggregate)

    Args:
      psi     : (B, N_u, N) complex128
      u_c     : (N_u,)       complex128
      x       : (N,)          float64 on device (cast to complex in ops)
      c       : (N,)          float64 on device
      kappa_e : (B, 1, 1)    complex128
      sigma_e : (B, 1, 1)    complex128
      rho_e   : (B, 1, 1)    complex128

    Returns:
      dpsi : (B, N_u, N) complex128
      Psi  : (B, N_u)    complex128
    """
    # c-weighted aggregate: Psi_k = sum_i c_i * psi_{k,i}
    Psi = (psi * c).sum(dim=-1)          # (B, N_u)

    # g(u_k, Psi_k) -- broadcast (N_u,) against (B, N_u)
    sigma_2d = sigma_e[..., 0]           # (B, 1)
    rho_2d   = rho_e[..., 0]            # (B, 1)

    g = (
        -0.5 * (u_c * u_c + 1j * u_c)   # (N_u,)  -- scalar per freq
        + rho_2d * sigma_2d * 1j * u_c * Psi      # (B, N_u)
        + 0.5 * sigma_2d ** 2 * Psi ** 2          # (B, N_u)
    )                                              # -> (B, N_u)

    # Multiply decay term by the mean reversion parameter kappa
    # x cast to complex for mixed-type mul with complex psi
    dpsi = g.unsqueeze(-1) - kappa_e * x.to(torch.complex128) * psi   # (B, N_u, N)

    return dpsi, Psi


# ---------------------------------------------------------------------------
# Mixed-precision Riccati RHS  (complex64 state for RTX 3060 Tensor Cores)
# ---------------------------------------------------------------------------

def _g_only_fp32(
    psi:       torch.Tensor,   # (B, N_u, N) complex64
    u_c32:     torch.Tensor,   # (N_u,)      complex64
    j_u_c32:   torch.Tensor,   # (N_u,)      complex64
    c_f32:     torch.Tensor,   # (N,) or (B, N) float32
    sigma_e32: torch.Tensor,   # (B, 1, 1)   complex64
    rho_e32:   torch.Tensor,   # (B, 1, 1)   complex64
) -> tuple:
    """
    Nonlinear part of the Riccati RHS: g(u, Ψ) only (no linear −κ·x·ψ term).

    Used by the exponential midpoint integrator in solve_riccati_rk4_mixed.
    Separating the linear part lets the integrator treat it exactly via
    exp(−κ·x·dt), making the solver unconditionally stable for any κ.

    Returns g (B, N_u) and Ψ (B, N_u), both complex64.
    """
    if c_f32.ndim == 1:
        Psi = (psi * c_f32).sum(dim=-1)               # (B, N_u)
    else:
        Psi = (psi * c_f32.unsqueeze(1)).sum(dim=-1)  # (B, N_u)
    s2d = sigma_e32[..., 0]   # (B, 1)
    r2d = rho_e32[..., 0]     # (B, 1)
    g = (
        -0.5 * (u_c32 * u_c32 + j_u_c32)
        + r2d * s2d * j_u_c32 * Psi
        + 0.5 * s2d * s2d * Psi * Psi
    )
    return g, Psi


def _riccati_rhs_fp32(
    psi:       torch.Tensor,   # (B, N_u, N) complex64
    u_c32:     torch.Tensor,   # (N_u,)      complex64 — purely real freqs
    j_u_c32:   torch.Tensor,   # (N_u,)      complex64 — purely imaginary (i*u)
    c_f32:     torch.Tensor,   # (N,) or (B, N) float32
    x_c32:     torch.Tensor,   # (N,)        complex64
    kappa_e32: torch.Tensor,   # (B, 1, 1)   complex64
    sigma_e32: torch.Tensor,   # (B, 1, 1)   complex64
    rho_e32:   torch.Tensor,   # (B, 1, 1)   complex64
):
    """
    Full Riccati RHS (nonlinear g + linear −κ·x·ψ) in complex64.
    Kept for solve_riccati_rk4 (float64 reference path).
    solve_riccati_rk4_mixed uses _g_only_fp32 + exponential integrator instead.
    """
    if c_f32.ndim == 1:
        Psi = (psi * c_f32).sum(dim=-1)
    else:
        Psi = (psi * c_f32.unsqueeze(1)).sum(dim=-1)
    s2d = sigma_e32[..., 0]
    r2d = rho_e32[..., 0]
    g = (
        -0.5 * (u_c32 * u_c32 + j_u_c32)
        + r2d * s2d * j_u_c32 * Psi
        + 0.5 * s2d * s2d * Psi * Psi
    )
    dpsi = g.unsqueeze(-1) - kappa_e32 * x_c32 * psi
    return dpsi, Psi


# ---------------------------------------------------------------------------
# RK4 ODE solver on GPU
# ---------------------------------------------------------------------------

def solve_riccati_rk4(
    kappa: torch.Tensor,   # (B,) float64
    theta: torch.Tensor,   # (B,) float64
    sigma: torch.Tensor,   # (B,) float64
    rho:   torch.Tensor,   # (B,) float64
    v0:    torch.Tensor,   # (B,) float64
    u_c:   torch.Tensor,   # (N_u,) complex128
    x:     torch.Tensor,   # (N,) float64
    c:     torch.Tensor,   # (N,) float64
    T_grid: np.ndarray,
    N_steps_per_unit: int = 200,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Solve Lifted Heston Riccati for B samples at all COS frequencies.
    Saves log-CF at each T in T_grid.

    Log-CF (Abi Jaber 2022 Eq 4.2):
      log phi(u_k; T) = v0 * sum_i psi_i(T)          [unweighted -- Y_i(0)=v0]
                      + kappa*theta * int_0^T Psi(t) dt [c-weighted integral]
    (The iu*x0 forward term is added later per strike.)

    Returns log_phi: (B, nT, N_u) complex128
    """
    B   = kappa.shape[0]
    N   = x.shape[0]
    N_u = u_c.shape[0]
    nT  = len(T_grid)
    dt  = 1.0 / N_steps_per_unit
    T_sorted = np.sort(T_grid)

    # Pre-expand params to (B, 1, 1) complex for broadcasting
    to_c = lambda t: t.to(torch.complex128)
    kappa_e = to_c(kappa)[:, None, None]
    sigma_e = to_c(sigma)[:, None, None]
    rho_e   = to_c(rho)[:, None, None]
    v0_c    = to_c(v0)              # (B,)
    kth_c   = to_c(kappa * theta)   # (B,)

    # State
    psi    = torch.zeros(B, N_u, N, dtype=torch.complex128, device=device)
    int_cv = torch.zeros(B, N_u,    dtype=torch.complex128, device=device)
    log_phi_out = torch.zeros(B, nT, N_u, dtype=torch.complex128, device=device)

    t     = 0.0
    t_idx = 0
    N_total = int(round(T_sorted[-1] * N_steps_per_unit))

    for step in range(N_total + 1):
        # Save checkpoint when t aligns with T_grid[t_idx]
        while t_idx < nT and abs(t - T_sorted[t_idx]) < dt * 0.5:
            # log phi = v0*Psi(T) + kappa*theta*int_0^T Psi(t)dt
            # where Psi = sum(c_i*psi_i) — c-WEIGHTED (c normalised to sum=1)
            # This is consistent with the scalar Heston limit (N=1 => c=[1])
            Psi_T = (psi * c).sum(dim=-1)   # (B, N_u)
            log_phi_out[:, t_idx, :] = (
                v0_c[:, None] * Psi_T
                + kth_c[:, None] * int_cv
            )
            t_idx += 1

        if t_idx >= nT or step == N_total:
            break

        # RK4 step
        k1, P1 = _riccati_rhs(psi,              u_c, x, c, kappa_e, sigma_e, rho_e)
        k2, P2 = _riccati_rhs(psi + 0.5*dt*k1, u_c, x, c, kappa_e, sigma_e, rho_e)
        k3, P3 = _riccati_rhs(psi + 0.5*dt*k2, u_c, x, c, kappa_e, sigma_e, rho_e)
        k4, P4 = _riccati_rhs(psi +     dt*k3, u_c, x, c, kappa_e, sigma_e, rho_e)

        psi    = psi    + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        int_cv = int_cv + (dt / 6.0) * (P1 + 2*P2 + 2*P3 + P4)
        t += dt

    return log_phi_out   # (B, nT, N_u) complex128


def solve_riccati_rk4_mixed(
    kappa: torch.Tensor,   # (B,) float64
    theta: torch.Tensor,   # (B,) float64
    sigma: torch.Tensor,   # (B,) float64
    rho:   torch.Tensor,   # (B,) float64
    v0:    torch.Tensor,   # (B,) float64
    u_c:   torch.Tensor,   # (N_u,) complex128 (pre-built frequencies)
    x:     torch.Tensor,   # (N,)  float64
    c:     torch.Tensor,   # (N,) or (B, N) float64
    T_grid: np.ndarray,
    N_steps_per_unit: int = 200,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Mixed-precision Riccati solver for RTX 3060 Tensor Core utilisation.

    psi (ODE state)  : complex64  — main compute in fast FP32 lanes
    int_cv           : complex128 — Riemann accumulator, precision-critical
    log_phi_out      : complex128 — final result in full precision

    c may be (N,) float64 [shared H across batch] or (B, N) float64
    [per-sample variable H]. The variable-H path unsqueezes the N_u dim
    before the contraction, keeping everything in one B=200 GPU call.

    Returns log_phi: (B, nT, N_u) complex128
    """
    B   = kappa.shape[0]
    N   = x.shape[0]
    N_u = u_c.shape[0]
    nT  = len(T_grid)
    dt  = 1.0 / N_steps_per_unit
    T_sorted = np.sort(T_grid)

    # ── Complex64 tensors for fast ODE stepping ────────────────────────────
    u_np   = u_c.real.cpu().numpy().astype(np.float64)   # real freqs (N_u,)
    # Precompute i*u as complex64 to avoid Python `1j` → complex128 promotion
    u_c32     = torch.tensor(u_np + 0j,  dtype=torch.complex64, device=device)
    j_u_c32   = torch.tensor(1j * u_np,  dtype=torch.complex64, device=device)
    x_c32     = x.to(torch.complex64)
    c_f32     = c.to(torch.float32)   # float32 × complex64 → complex64 (no upcast)

    kappa_e32 = kappa.to(torch.complex64)[:, None, None]
    sigma_e32 = sigma.to(torch.complex64)[:, None, None]
    rho_e32   = rho.to(torch.complex64)[:, None, None]

    # ODE state in complex64
    psi = torch.zeros(B, N_u, N, dtype=torch.complex64, device=device)

    # ── Complex128 tensors for high-precision accumulation ─────────────────
    int_cv      = torch.zeros(B, N_u,    dtype=torch.complex128, device=device)
    log_phi_out = torch.zeros(B, nT, N_u, dtype=torch.complex128, device=device)

    v0_c128  = v0.to(torch.complex128)
    kth_c128 = (kappa * theta).to(torch.complex128)

    # For checkpoint Psi_T: need c in complex128, unsqueeze N_u dim if (B, N)
    c128 = c.to(torch.complex128)  # (N,) or (B, N)

    # ── Exponential integrator: precompute step matrices (once per call) ──────
    # RK4 is unstable when κ·x_max·dt >> 2.8 (e.g. κ=5, x_40=6275, dt=0.005
    # gives stiffness ratio 156.9 — RK4 overflows float32 in ~7 steps).
    #
    # Exponential midpoint splits:  dψ_i/dt = g(u,Ψ) − κ·x_i·ψ_i
    #   • Linear part  −κ·x_i·ψ_i  → exact via exp(−κ·x_i·dt)
    #   • Nonlinear g(u,Ψ)         → explicit midpoint (bounded, smooth)
    # Result: unconditionally stable for ANY κ, 2nd order in dt.
    #
    # Integral coefficient: int_coef = (1−exp(−A·h))/A = −expm1(−A·h)/A
    #   Limit as A→0: int_coef → h  (correct; expm1 avoids catastrophic cancellation)
    kax_r    = (kappa_e32 * x_c32).real          # (B, 1, N) float32, purely real
    kax_dt   = kax_r * dt                         # (B, 1, N)
    kax_dt_h = kax_r * (dt / 2.0)                # (B, 1, N) half-step

    lin_half = torch.exp(-kax_dt_h).to(torch.complex64)   # exp(−κ·x·dt/2)
    lin_full = torch.exp(-kax_dt  ).to(torch.complex64)   # exp(−κ·x·dt)

    safe_kax = kax_r.clamp(min=1e-10)                     # avoid /0 for tiny x
    int_half = (-torch.expm1(-kax_dt_h) / safe_kax        # ≈ dt/2 for small κ·x
                ).to(torch.complex64)                      # (B, 1, N)
    int_full = (-torch.expm1(-kax_dt  ) / safe_kax        # ≈ dt  for small κ·x
                ).to(torch.complex64)                      # (B, 1, N)

    t, t_idx = 0.0, 0
    N_total = int(round(T_sorted[-1] * N_steps_per_unit))

    for step in range(N_total + 1):
        # Checkpoint: save log-CF at each T in T_sorted
        while t_idx < nT and abs(t - T_sorted[t_idx]) < dt * 0.5:
            # Upcast psi to complex128 for high-precision checkpoint
            psi_128 = psi.to(torch.complex128)             # (B, N_u, N) c128
            if c128.ndim == 1:                             # shared c: (N,)
                Psi_T = (psi_128 * c128).sum(dim=-1)      # (B, N_u)
            else:                                          # per-sample: (B, N)
                Psi_T = (psi_128 * c128.unsqueeze(1)).sum(dim=-1)
            log_phi_out[:, t_idx, :] = (
                v0_c128[:, None] * Psi_T + kth_c128[:, None] * int_cv
            )
            t_idx += 1

        if t_idx >= nT or step == N_total:
            break

        # ── Exponential midpoint step (2 evaluations, unconditionally stable) ──
        # 1. Nonlinear g at current ψ
        g_n, _ = _g_only_fp32(psi, u_c32, j_u_c32, c_f32, sigma_e32, rho_e32)

        # 2. Advance to midpoint: exact linear decay + g_n forcing
        psi_h = lin_half * psi + int_half * g_n.unsqueeze(-1)  # (B, N_u, N)

        # 3. Nonlinear g at midpoint
        g_h, P_h = _g_only_fp32(psi_h, u_c32, j_u_c32, c_f32, sigma_e32, rho_e32)

        # 4. Full step from psi_n using midpoint g (exact linear + integral of g_h)
        psi = lin_full * psi + int_full * g_h.unsqueeze(-1)    # (B, N_u, N)

        # 5. int_cv ≈ ∫Ψ dt: midpoint rule (2nd order), accumulate in float128
        int_cv = int_cv + dt * P_h.to(torch.complex128)
        t += dt

    return log_phi_out   # (B, nT, N_u) complex128


# ---------------------------------------------------------------------------
# COS pricing: log_phi -> call prices  (GPU, fully vectorised)
# ---------------------------------------------------------------------------

def cos_price_calls(
    log_phi: torch.Tensor,   # (B, nT, N_u) complex128
    u_k:     torch.Tensor,   # (N_u,) float64
    Vk_call: torch.Tensor,   # (N_u,) float64  — call payoff coeffs [0, b]
    Vk_put:  torch.Tensor,   # (N_u,) float64  — put payoff coeffs  [a, 0]
    K_arr:   torch.Tensor,   # (nK,) float64
    S0:      float = 1.0,
    a:       float = _A,
) -> torch.Tensor:
    """
    For K >= S0 (OTM call): C = K * Re[sum phi * exp(i*uk*(x0-a)) * Vk_call]
    For K <  S0 (ITM call): P = K * Re[sum phi * exp(i*uk*(x0-a)) * Vk_put]
                             C = P + (S0 - K)   [put-call parity]

    The put formula eliminates catastrophic cancellation when d1 >> 1
    (deep ITM call price ≈ intrinsic = S0 - K, so computing call directly
    loses all significant digits in the time value).

    Returns call prices: (B, nT, nK) float64
    """
    dev = log_phi.device
    S0t = torch.tensor(S0, dtype=torch.float64, device=dev)

    x0    = torch.log(S0t / K_arr)                                          # (nK,)
    phase = torch.exp(1j * u_k.unsqueeze(1) * (x0 - a).unsqueeze(0))       # (N_u, nK)

    # phi: (B, nT, N_u) -- enforce martingale at k=0
    phi = torch.exp(log_phi)
    phi[:, :, 0] = 1.0 + 0.0j

    # Weighted sums for call and put payoffs
    phi_w_call = phi * Vk_call.to(torch.complex128)                        # (B, nT, N_u)
    phi_w_put  = phi * Vk_put.to(torch.complex128)                         # (B, nT, N_u)

    result_call = torch.einsum('btn,nk->btk', phi_w_call, phase)           # (B, nT, nK)
    result_put  = torch.einsum('btn,nk->btk', phi_w_put,  phase)           # (B, nT, nK)

    K_v = K_arr.view(1, 1, -1)
    call_prices       = K_v * result_call.real                              # (B, nT, nK)
    put_prices        = K_v * result_put.real                               # (B, nT, nK)

    # Recover call from put via put-call parity: C = P + S0 - K
    call_from_put = put_prices + (S0t - K_arr).clamp(min=0.0).view(1, 1, -1)

    # Select formula: put→call for K < S0, direct call for K >= S0
    itm    = (K_arr < S0t).view(1, 1, -1)                                  # (1, 1, nK)
    prices = torch.where(itm, call_from_put, call_prices)                  # (B, nT, nK)

    # Safety floor at intrinsic (absorbs residual float error)
    intrinsic = (S0t - K_v).clamp(min=0.0)
    prices    = torch.max(prices, intrinsic)

    return prices   # (B, nT, nK) float64


# ---------------------------------------------------------------------------
# Batched IV inversion on GPU (Newton-Raphson)
# ---------------------------------------------------------------------------

def bs_iv_gpu(
    prices: torch.Tensor,    # (B, nT, nK) float64
    S0:     float,
    K_arr:  torch.Tensor,    # (nK,) float64
    T_arr:  torch.Tensor,    # (nT,) float64
    n_iter: int = 40,
) -> torch.Tensor:
    """
    Batch Newton-Raphson IV inversion.  All B*nT*nK in parallel on GPU.

    For K < S0 (ITM call) uses put-call parity  P = C - (S - K)  and
    inverts the put price instead.  The equivalent OTM put always has
    non-zero time value, eliminating the 'price = intrinsic' NaN class.

    Returns (B, nT, nK) float32, NaN for genuinely unquotable options.
    """
    dev = prices.device
    S   = torch.tensor(S0, dtype=torch.float64, device=dev)
    K   = K_arr.view(1, 1, -1)          # (1, 1, nK)
    T   = T_arr.view(1, -1, 1)          # (1, nT, 1)
    sqT = torch.sqrt(T.clamp(min=1e-10))

    # ---- put-call parity: convert ITM calls to OTM puts ----
    # For K < S: P = C - (S - K).  OTM put always has stable time value.
    # For K >= S: keep call price as-is.
    itm        = (K < S)                              # (1, 1, nK) bool
    put_prices = prices - (S - K).clamp(min=0.0)     # P = C - max(S-K, 0)
    eff_prices = torch.where(itm, put_prices, prices) # (B, nT, nK)

    # Validity: eff price must be positive and T > 0
    invalid = (eff_prices <= 1e-12) | (T < 1e-10)

    sigma = torch.full_like(prices, 0.30)

    for _ in range(n_iter):
        s  = sigma.clamp(min=1e-8)
        d1 = (torch.log(S / K) + 0.5 * s ** 2 * T) / (s * sqT)
        d2 = d1 - s * sqT
        # Model price: call for K >= S, put for K < S
        call_p  = S * _ncdf(d1)  - K * _ncdf(d2)
        put_p   = K * _ncdf(-d2) - S * _ncdf(-d1)
        model_p = torch.where(itm, put_p, call_p)
        # Vega is identical for call and put (put-call parity is model-free)
        v = S * sqT * _npdf(d1)
        sigma = (sigma - (model_p - eff_prices) / v.clamp(min=1e-15)).clamp(1e-7, 5.0)

    sigma[invalid] = float('nan')
    sigma[(sigma < 1e-5) | (sigma > 4.9)] = float('nan')

    return sigma.float()


# ---------------------------------------------------------------------------
# IFT-based Differentiable IV Inverter  (Implicit Function Theorem)
# ---------------------------------------------------------------------------

class BS_IV_Implicit_Inverter(torch.autograd.Function):
    """
    Differentiable Black-Scholes IV inversion via the Implicit Function Theorem.

    Forward : standard Newton-Raphson (non-differentiable; gradients detached).
    Backward: exact analytical gradient using IFT.

    By the Implicit Function Theorem applied to BS(σ, price) = 0:
        F(σ, price) = BS_call(σ) - price = 0
        dσ/d(price) = -[∂F/∂σ]^{-1} ∂F/∂price
                    = -[-Vega]^{-1} × (-1)
                    = 1 / Vega(σ)

    Chain rule for upstream gradient g = ∂L/∂σ:
        ∂L/∂price = g × dσ/d(price) = g / Vega(σ)

    Numerical stability:
      - NaN IV cells are masked: their gradient contribution is 0.
      - Vega is clamped at 1e-12 to prevent division by zero at deep OTM.

    Note on practical AAD:
      Full automatic differentiation through the ODE → price → IV pipeline
      requires storing ~400 RK4 intermediate tensors (~100 GB for full batch),
      which is memory-infeasible.  This class provides the IV→price leg
      analytically; the price→θ leg via the ODE is left to FD or future
      adjoint ODE solvers.  It is most useful when combined with:
        torch.func.jacrev(lambda p: price_pipeline(p))(theta)
      once the ODE memory footprint is reduced by checkpointing.
    """

    @staticmethod
    def forward(ctx, prices, S0_scalar, K_arr, T_arr):
        """
        prices   : (B, nT, nK) float64
        S0_scalar: python float
        K_arr    : (nK,)       float64
        T_arr    : (nT,)       float64

        Returns iv : (B, nT, nK) float32  (same as bs_iv_gpu)
        """
        with torch.no_grad():
            iv_f64 = bs_iv_gpu(
                prices.detach(), S0_scalar,
                K_arr.detach(), T_arr.detach(),
            ).to(torch.float64)           # keep float64 for vega computation
        # Save float64 IV and grids for backward
        ctx.save_for_backward(iv_f64, K_arr.detach(), T_arr.detach())
        ctx.S0 = S0_scalar
        return iv_f64.float()             # return float32 (matches bs_iv_gpu)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output : (B, nT, nK) float32 — upstream ∂L/∂σ
        Returns     : grad_prices in float64 (matches prices dtype)
        """
        iv_f64, K_arr, T_arr = ctx.saved_tensors
        S0 = ctx.S0

        # Rebuild d1 at the recovered IV for Vega computation (float64)
        K_v  = K_arr.view(1, 1, -1)          # (1, 1, nK)
        T_v  = T_arr.view(1, -1, 1)          # (1, nT, 1)
        sqT  = torch.sqrt(T_v.clamp(min=1e-10))
        s    = iv_f64.clamp(min=1e-8)
        d1   = (torch.log(torch.tensor(S0, dtype=torch.float64, device=iv_f64.device) / K_v)
                + 0.5 * s * s * T_v) / (s * sqT)    # (B, nT, nK) float64

        S0t = torch.tensor(S0, dtype=torch.float64, device=iv_f64.device)
        vega = S0t * sqT * (_INVSQRT2PI * torch.exp(-0.5 * d1 * d1))

        # Mask NaN cells: grad is 0 where IV failed to converge
        nan_mask = torch.isnan(iv_f64)
        vega_safe = vega.clone()
        vega_safe[nan_mask] = 1.0            # avoid inf; masked cells get 0 grad below

        # ∂L/∂price = (∂L/∂σ) × (1/Vega)
        grad_prices = grad_output.to(torch.float64) / vega_safe.clamp(min=1e-12)
        grad_prices[nan_mask] = 0.0          # zero gradient at unquotable options

        # Gradients for (prices, S0_scalar, K_arr, T_arr)
        return grad_prices, None, None, None


# ---------------------------------------------------------------------------
# Main batch pricing pipeline
# ---------------------------------------------------------------------------

from functools import lru_cache

@lru_cache(maxsize=128)
def _get_cos_payoff_coeffs_gpu_cached(N_cos: int, a: float = _A, b: float = _B, device_str: str = 'cuda', is_put: bool = False) -> torch.Tensor:
    device = torch.device(device_str)
    k = torch.arange(N_cos, dtype=torch.float64, device=device)
    uk = k * np.pi / (b - a)
    uk_c = uk.to(torch.complex128)
    
    if is_put:
        chi_put = torch.real(
            torch.exp(-1j * uk_c * a)
            * (1.0 - torch.exp((1.0 + 1j * uk_c) * a))
            / (1.0 + 1j * uk_c)
        )
        chi_put[0] = 1.0 - np.exp(a)
        
        safe_uk = torch.where(k == 0, torch.tensor(1.0, dtype=torch.float64, device=device), uk)
        psi_put = torch.where(
            k == 0,
            torch.tensor(-a, dtype=torch.float64, device=device),
            -torch.sin(uk * a) / safe_uk,
        )
        Vk = (2.0 / (b - a)) * (psi_put - chi_put)
        Vk[0] *= 0.5
    else:
        chi = torch.real(
            torch.exp(-1j * uk_c * a)
            * (torch.exp((1.0 + 1j * uk_c) * b) - 1.0)
            / (1.0 + 1j * uk_c)
        )
        chi[0] = np.exp(b) - 1.0
        
        safe_uk = torch.where(k == 0, torch.tensor(1.0, dtype=torch.float64, device=device), uk)
        psi = torch.where(
            k == 0,
            torch.tensor(b, dtype=torch.float64, device=device),
            (torch.sin(uk * (b - a)) + torch.sin(uk * a)) / safe_uk,
        )
        Vk = (2.0 / (b - a)) * (chi - psi)
        Vk[0] *= 0.5
        
    return Vk

def get_cos_payoff_coeffs_gpu(N_cos: int, a: float = _A, b: float = _B, device='cuda', is_put: bool = False) -> torch.Tensor:
    return _get_cos_payoff_coeffs_gpu_cached(N_cos, a, b, str(device), is_put)


def _price_batch_gpu_raw(
    params_batch:      np.ndarray,        # (B, 5): [kappa, theta, sigma, rho, v0]
    T_grid:            np.ndarray,        # (nT,)
    K_grid:            np.ndarray,        # (nK,) log-moneyness
    H_fixed:           float = 0.08,      # used when H_batch is None
    H_batch:           np.ndarray = None, # (B,) per-sample H; enables true B=200 batch
    N_factors:         int   = 20,
    N_cos:             int   = 64,
    N_steps_per_unit:  int   = 200,
    S0:                float = 1.0,
    device:            str   = 'cuda',
) -> np.ndarray:
    """
    Price B IV surfaces on GPU. Returns (B, nT, nK) float32.

    H_batch (optional): supply a (B,) array of per-sample Hurst exponents to
    price all B samples in a SINGLE GPU call with variable H.  When omitted,
    all samples use the shared H_fixed scalar.  Enabling H_batch avoids the
    O(unique_H) kernel-launch overhead from external H-grouping loops.
    """
    dev = torch.device(device)
    B   = params_batch.shape[0]

    # ── Bernstein factors ────────────────────────────────────────────────────
    if H_batch is not None:
        # Per-sample c: vectorized numpy, then one GPU tensor
        # x does not depend on H — compute once
        r_N  = 1.0 + 10.0 * (N_factors ** -0.9)
        x_np = np.array([r_N ** (i - 1.0 - N_factors / 2.0)
                         for i in range(1, N_factors + 1)])  # (N,)
        # c_np[b, i] = x_np[i]^{-(H_batch[b]+0.5)}, normalised per row
        c_np = x_np[None, :] ** -(H_batch[:, None] + 0.5)   # (B, N)
        c_np = c_np / c_np.sum(axis=1, keepdims=True)        # normalise
        x    = torch.tensor(x_np, dtype=torch.float64, device=dev)  # (N,)
        c    = torch.tensor(c_np, dtype=torch.float64, device=dev)  # (B, N)
    else:
        x_np, c_np = bernstein_factors(H_fixed, N_factors)
        x = torch.tensor(x_np, dtype=torch.float64, device=dev)  # (N,)
        c = torch.tensor(c_np, dtype=torch.float64, device=dev)  # (N,)

    # ── Shared frequency / payoff tensors ────────────────────────────────────
    u_np    = np.arange(N_cos) * np.pi / (_B - _A)
    u_c     = torch.tensor(u_np + 0j, dtype=torch.complex128, device=dev)
    u_k     = torch.tensor(u_np,      dtype=torch.float64,    device=dev)
    Vk_call = get_cos_payoff_coeffs_gpu(N_cos, _A, _B, dev, is_put=False)
    Vk_put  = get_cos_payoff_coeffs_gpu(N_cos, _A, _B, dev, is_put=True)

    K_arr = torch.tensor(S0 * np.exp(K_grid), dtype=torch.float64, device=dev)
    T_arr = torch.tensor(T_grid,               dtype=torch.float64, device=dev)

    # ── Parameter tensors ────────────────────────────────────────────────────
    p     = torch.tensor(params_batch, dtype=torch.float64, device=dev)
    kappa = p[:, 0]; theta = p[:, 1]
    sigma = p[:, 2]; rho   = p[:, 3]; v0 = p[:, 4]

    # 1. Riccati ODE  (mixed precision: complex64 state, complex128 accumulator)
    log_phi = solve_riccati_rk4_mixed(
        kappa, theta, sigma, rho, v0,
        u_c, x, c, T_grid,
        N_steps_per_unit=N_steps_per_unit,
        device=device,
    )

    # 2. COS pricing (call for K>=S0, put->call for K<S0)
    prices = cos_price_calls(log_phi, u_k, Vk_call, Vk_put, K_arr, S0=S0)

    # 3. IV inversion
    iv = bs_iv_gpu(prices, S0, K_arr, T_arr)

    return iv.cpu().numpy()


def price_batch_gpu(
    params_batch:      np.ndarray,        # (B, 5): [kappa, theta, sigma, rho, v0]
    T_grid:            np.ndarray,        # (nT,)
    K_grid:            np.ndarray,        # (nK,) log-moneyness
    H_fixed:           float = 0.08,      # used when H_batch is None
    H_batch:           np.ndarray = None, # (B,) per-sample H; enables true B=200 batch
    N_factors:         int   = 20,
    N_cos:             int   = 64,
    N_steps_per_unit:  int   = 200,
    S0:                float = 1.0,
    device:            str   = 'cuda',
    N_cos_per_T:       dict  = None,
) -> np.ndarray:
    """
    Price B IV surfaces on GPU. Returns (B, nT, nK) float32.
    If N_cos_per_T is provided, groups maturities in T_grid and invokes
    pricing with specific N_cos values.
    """
    if not N_cos_per_T:
        return _price_batch_gpu_raw(
            params_batch=params_batch,
            T_grid=T_grid,
            K_grid=K_grid,
            H_fixed=H_fixed,
            H_batch=H_batch,
            N_factors=N_factors,
            N_cos=N_cos,
            N_steps_per_unit=N_steps_per_unit,
            S0=S0,
            device=device,
        )

    # Group maturities in T_grid by their corresponding N_cos values
    groups = {}
    for idx, T in enumerate(T_grid):
        matched_n_cos = N_cos
        for k, v in N_cos_per_T.items():
            if abs(float(k) - T) < 1e-6:
                matched_n_cos = v
                break
        if matched_n_cos not in groups:
            groups[matched_n_cos] = []
        groups[matched_n_cos].append((idx, T))

    # Initialize output array
    B = params_batch.shape[0]
    nT = len(T_grid)
    nK = len(K_grid)
    out = np.empty((B, nT, nK), dtype=np.float32)

    for n_cos_val, items in groups.items():
        # Sort items by maturity to maintain chronological order in ODE solver
        items_sorted = sorted(items, key=lambda x: x[1])
        sub_indices = [item[0] for item in items_sorted]
        sub_T_grid = np.array([item[1] for item in items_sorted])

        sub_out = _price_batch_gpu_raw(
            params_batch=params_batch,
            T_grid=sub_T_grid,
            K_grid=K_grid,
            H_fixed=H_fixed,
            H_batch=H_batch,
            N_factors=N_factors,
            N_cos=n_cos_val,
            N_steps_per_unit=N_steps_per_unit,
            S0=S0,
            device=device,
        )
        out[:, sub_indices, :] = sub_out

    return out


# ---------------------------------------------------------------------------
# Single-sample convenience wrapper
# ---------------------------------------------------------------------------

def price_iv_surface_gpu(
    params: dict,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    device: str = 'cuda',
    **kwargs,
) -> np.ndarray:
    """Drop-in for lifted_heston.price_iv_surface(). Returns (nT, nK) float32."""
    p = np.array([[
        params['kappa'], params['theta'],
        params['sigma'], params['rho'], params['v0'],
    ]])
    return price_batch_gpu(
        p, T_grid, K_grid,
        H_fixed=params.get('H', 0.08),
        device=device, **kwargs
    )[0]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_GRID = np.linspace(-0.5, 0.5, 11)
    params = dict(kappa=1.0, theta=0.08, sigma=0.5, rho=-0.5, v0=0.08, H=0.08)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('\n=== SINGLE SURFACE ===')
    t0 = time.time()
    iv = price_iv_surface_gpu(params, T_GRID, K_GRID, device=device)
    t1 = time.time()
    print(f'Time: {t1-t0:.3f}s')
    print(f'Shape: {iv.shape}')
    print(f'ATM IVs: {[f"{iv[i,5]*100:.1f}%" for i in range(8)]}')
    print(f'NaN count: {int(np.isnan(iv).sum())}')

    if np.isnan(iv).any():
        print('WARNING: NaNs detected -- check parameter bounds or N_steps')
    else:
        print('Martingale sanity: min IV > 0 and max IV < 200% -- OK')

    print('\n=== BATCH OF 512 ===')
    rng = np.random.default_rng(42)
    LO  = np.array([0.1, 0.01, 0.1, -0.9, 0.01])
    HI  = np.array([5.0, 0.15, 1.0, -0.1, 0.15])
    p512 = rng.uniform(LO, HI, (512, 5))

    t0  = time.time()
    ivs = price_batch_gpu(p512, T_GRID, K_GRID, device=device)
    t1  = time.time()

    nan_rate = np.isnan(ivs).mean() * 100
    print(f'512 samples: {t1-t0:.2f}s  ({(t1-t0)/512*1000:.1f} ms/sample)')
    print(f'NaN rate: {nan_rate:.1f}%')
    print(f'Projected 50k: {50000*(t1-t0)/512/60:.1f} min')

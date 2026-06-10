import math
import torch


def _bs_call(F, K, T, v):
    """
    Vectorized Black-Scholes call price.
    All inputs must be broadcastable tensors.
    """
    sqrt_T = torch.sqrt(T)
    # Guard against v*sqrt_T -> 0 (deep OTM + tiny vol)
    denom = torch.clamp(v * sqrt_T, min=1e-10)
    d1 = (torch.log(F / K) + 0.5 * v ** 2 * T) / denom
    d2 = d1 - sqrt_T * v
    N_d1 = 0.5 * (1.0 + torch.erf(d1 * 0.7071067811865476))  # 1/sqrt(2)
    N_d2 = 0.5 * (1.0 + torch.erf(d2 * 0.7071067811865476))
    return F * N_d1 - K * N_d2


def _bs_vega(F, K, T, v):
    """Vectorized Black-Scholes vega: ∂C/∂σ = F·n(d1)·√T."""
    sqrt_T = torch.sqrt(T)
    denom  = torch.clamp(v * sqrt_T, min=1e-10)
    d1     = (torch.log(F / K) + 0.5 * v ** 2 * T) / denom
    n_d1   = torch.exp(-0.5 * d1 ** 2) * 0.3989422804014327  # 1/sqrt(2π)
    return F * n_d1 * sqrt_T


def jaeckel_iv(prices, F, K, T, max_iter=20, tol=1e-6):
    """
    Vectorized Implied Volatility via Newton-Raphson with a
    Brenner-Subrahmanyam (1988) initial guess.

    Initial guess: σ₀ ≈ √(2π/T) · C/F  (ATM straddle approximation).
    This is far superior to a flat 0.3 for OTM strikes and short maturities;
    it halves the NR iterations needed to converge.

    Convergence is tracked per-element: once a strike has |step| < tol its
    volatility is frozen and not updated further, preventing drift from
    continued steps on already-converged points.

    Args:
        prices:   Call option prices (Tensor, shape K).
        F:        Forward price (scalar or Tensor).
        K:        Strikes (Tensor, shape K).
        T:        Time to maturity (scalar float or 0-dim Tensor).
        max_iter: Maximum Newton-Raphson iterations (default: 20).
        tol:      Convergence tolerance on step size (default: 1e-6).

    Returns:
        iv: Implied volatilities (Tensor, shape K). NaN-free; clamped to [1e-4, 5.0].
    """
    prices = torch.as_tensor(prices, dtype=torch.float32)
    F      = torch.as_tensor(F,      dtype=torch.float32)
    K      = torch.as_tensor(K,      dtype=torch.float32)
    T      = torch.as_tensor(T,      dtype=torch.float32)

    T_safe = T.clamp(min=1e-6)

    # ── Brenner-Subrahmanyam initial guess ──────────────────────────────────
    # For ATM options: C ≈ F·σ·√(T/2π)  =>  σ₀ = C/F · √(2π/T)
    # For OTM options this overestimates σ, but it's still better than 0.3
    # because it scales correctly with maturity.
    v = torch.sqrt(2.0 * math.pi / T_safe) * prices / F.clamp(min=1e-8)
    v = v.clamp(min=0.05, max=5.0)

    converged = torch.zeros_like(v, dtype=torch.bool)

    for _ in range(max_iter):
        if converged.all():
            break

        p_curr = _bs_call(F, K, T_safe, v)
        vega   = _bs_vega(F, K, T_safe, v)

        # Clamp vega to prevent division by zero for deep OTM / tiny T
        vega_safe = vega.clamp(min=1e-8)
        step      = (p_curr - prices) / vega_safe

        # Freeze already-converged elements
        step = torch.where(converged, torch.zeros_like(step), step)

        v = (v - step).clamp(min=1e-4, max=5.0)

        converged = converged | (step.abs() < tol)

    return v


if __name__ == "__main__":
    import numpy as np

    F_val = torch.tensor(100.0)
    K_val = torch.tensor([90.0, 95.0, 100.0, 105.0, 110.0])
    T_val = 0.5
    true_iv = torch.tensor([0.25, 0.22, 0.20, 0.21, 0.23])

    # Compute BS prices at true IV, then invert
    prices_val = _bs_call(F_val, K_val, torch.tensor(T_val), true_iv)
    iv_recovered = jaeckel_iv(prices_val, F_val, K_val, T_val)

    print("IV Inverter self-test:")
    print(f"  True IV:      {true_iv.numpy()}")
    print(f"  Recovered IV: {iv_recovered.numpy()}")
    print(f"  Max error:    {(iv_recovered - true_iv).abs().max().item():.2e}")
    assert (iv_recovered - true_iv).abs().max().item() < 1e-4, "IV inversion failed!"
    print("  ✓ PASSED")

"""
Phoenix Autocallable Option Pricing Kernel under Stochastic Volatility.

Implements a two-barrier structured note Monte Carlo pricing kernel:
  - B_call: Early redemption (autocall) barrier. If S_{t_i} >= B_call * S_0,
    the note is called, paying par + accrued coupon, and terminates.
  - B_cpn: Coupon corridor barrier (B_cpn < B_call). If B_cpn * S_0 <= S_{t_i} < B_call * S_0,
    the note pays the coupon for the period and continues.
  - Memory variant (memory=True): Missed coupons accumulate and are paid
    at the next observation date where S_{t_j} >= B_cpn * S_0.
  - Capital protection at maturity: If never called, pays par (1.0) discounted.

Mathematical References:
  - Overhaus, M. et al. (2007). Equity Hybrid Derivatives. John Wiley & Sons.
  - Brigo, D., Mercurio, F. (2006). Interest Rate Models - Theory and Practice. Springer.
"""

import sys
sys.path.insert(0, "src")

from typing import Dict, List, Tuple
import torch
from torch import Tensor
from deepvol.models.autocall import price_autocall_mc


@torch.compile(mode="default", dynamic=True)
def _phoenix_mc_kernel(
    S: Tensor,            # (B, N_paths, N_steps+1) float64
    obs_steps: Tensor,    # (N_obs,) int64
    B_call: Tensor,       # (B,) float64
    B_cpn: Tensor,        # (B,) float64
    coupon: Tensor,       # (B,) float64
    r: Tensor,            # (B,) float64
    T: float,
    dt: float,
    memory: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    B_batch, N_paths, _ = S.shape
    device = S.device
    S0 = S[:, :, 0:1]  # (B, N_paths, 1)

    called = torch.zeros(B_batch, N_paths, dtype=torch.bool, device=device)
    had_coupon = torch.zeros(B_batch, N_paths, dtype=torch.bool, device=device)
    npv = torch.zeros(B_batch, N_paths, dtype=torch.float64, device=device)
    life = torch.zeros(B_batch, N_paths, dtype=torch.float64, device=device)
    missed_cpns = torch.zeros(B_batch, N_paths, dtype=torch.float64, device=device)

    N_obs = obs_steps.shape[0]
    for i in range(N_obs):
        step_idx = obs_steps[i]
        t_i = step_idx.to(torch.float64) * dt
        S_obs = S[:, :, step_idx]  # (B, N_paths)
        S_rel = S_obs / S0.squeeze(-1)

        active = ~called
        call_trig = active & (S_rel >= B_call.unsqueeze(1))
        cpn_trig = active & (~call_trig) & (S_rel >= B_cpn.unsqueeze(1))

        disc = torch.exp(-r.unsqueeze(1) * t_i)

        # Call payoff: par + accrued coupon up to date t_i
        call_payoff = disc * (1.0 + coupon.unsqueeze(1) * (float(i + 1)))
        npv = npv + call_trig.to(torch.float64) * call_payoff
        life = life + call_trig.to(torch.float64) * t_i
        had_coupon = had_coupon | call_trig

        # Coupon corridor payoff
        if memory:
            cpn_multiplier = 1.0 + missed_cpns
            cpn_payoff = disc * coupon.unsqueeze(1) * cpn_multiplier
            npv = npv + cpn_trig.to(torch.float64) * cpn_payoff
            had_coupon = had_coupon | cpn_trig
            missed_cpns = torch.where(cpn_trig, torch.zeros_like(missed_cpns), missed_cpns + 1.0)
        else:
            cpn_payoff = disc * coupon.unsqueeze(1)
            npv = npv + cpn_trig.to(torch.float64) * cpn_payoff
            had_coupon = had_coupon | cpn_trig

        called = called | call_trig

    # Capital protection at maturity for uncalled paths
    uncalled = ~called
    disc_T = torch.exp(-r.unsqueeze(1) * T)
    npv = npv + uncalled.to(torch.float64) * disc_T * 1.0
    life = life + uncalled.to(torch.float64) * T

    mean_npv = npv.mean(dim=1).clone()
    mean_call_prob = called.to(torch.float64).mean(dim=1).clone()
    mean_cpn_prob = had_coupon.to(torch.float64).mean(dim=1).clone()
    mean_life = life.mean(dim=1).clone()

    return mean_npv, mean_call_prob, mean_cpn_prob, mean_life


def price_phoenix_mc(
    S: Tensor,
    obs_indices: List[int],
    B_call: Tensor,
    B_cpn: Tensor,
    coupon: Tensor,
    r: Tensor,
    T: float,
    dt: float,
    memory: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Price a Phoenix 2-barrier autocallable note via GPU Monte Carlo.

    Returns:
        (npv, call_prob, cpn_prob, exp_life)
    """
    if (B_cpn >= B_call).any():
        raise ValueError("B_cpn must be strictly less than B_call for all contracts.")

    obs_steps = torch.tensor(obs_indices, dtype=torch.int64, device=S.device)
    return _phoenix_mc_kernel(S, obs_steps, B_call, B_cpn, coupon, r, T, dt, memory)


def phoenix_decomposition(
    npv_phoenix: Tensor,
    npv_vanilla: Tensor,
    call_prob: Tensor,
    cpn_prob: Tensor,
) -> Dict[str, Tensor]:
    """Decompose Phoenix NPV into autocall leg and coupon corridor leg."""
    corridor = (npv_phoenix - npv_vanilla).clamp(min=0.0)
    return {
        "autocall_leg": npv_vanilla,
        "coupon_corridor_leg": corridor,
        "call_prob": call_prob,
        "cpn_prob": cpn_prob,
        "corridor_fraction": corridor / npv_phoenix.clamp(min=1e-8),
    }

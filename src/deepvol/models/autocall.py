"""
autocall.py — GPU-Accelerated Monte Carlo Pricing Engine for Vanilla Autocallable Notes.

Autocall payoff reference:
  Brigo, D., Mercurio, F. (2006). Interest Rate Models - Theory and Practice. Springer.
  Overhaus, M. et al. (2007). Equity Hybrid Derivatives. Wiley.

Autocall Payoff Dynamics:
  At observation dates t_i in {t_1, ..., t_N}:
    If S(t_i) >= B * S0 and not previously called:
      Trigger autocall event:
        Payoff = exp(-r * t_i) * (1 + coupon * t_i)
        Life = t_i
  If not called by maturity T:
    Capital protected redemption:
      Payoff = exp(-r * T) * 1.0
      Life = T
"""

import math
from typing import List, Tuple, Union
import torch


def make_obs_indices(n_obs: int, T: float, N_steps: int) -> List[int]:
    """
    Returns a sorted list of step indices (ints) where observation dates fall,
    uniformly spaced over (0, T].

    Parameters:
        n_obs: Number of observation dates.
        T: Maturity in years.
        N_steps: Total simulation steps across (0, T].

    Returns:
        Sorted list of integer step indices in [1, N_steps].
    """
    dt = T / N_steps
    indices: List[int] = []
    for i in range(1, n_obs + 1):
        t_obs_i = i * T / n_obs
        idx_i = int(round(t_obs_i / dt))
        idx_i = max(1, min(N_steps, idx_i))
        indices.append(idx_i)
    return sorted(indices)


def autocall_upper_bound(n_obs: int, coupon: float, T: float, r: float) -> float:
    """
    Analytical upper bound price assuming spot always above barrier (called at first observation date):
        upper_bound = exp(-r * t_1) * (1 + coupon * t_1)
    where t_1 = T / n_obs.
    """
    t_1 = T / n_obs
    return math.exp(-r * t_1) * (1.0 + coupon * t_1)


@torch.compile(mode="reduce-overhead")
def _price_autocall_kernel(
    S: torch.Tensor,
    obs_indices: torch.Tensor,
    B: torch.Tensor,
    coupon: torch.Tensor,
    r: torch.Tensor,
    T: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiled GPU kernel for autocall Monte Carlo payoff accumulation.
    Returns cloned outputs to prevent CUDAGraphs static buffer corruption.
    """
    B_batch = S.shape[0]
    N_paths = S.shape[1]
    S0 = S[:, :, 0]

    called = torch.zeros((B_batch, N_paths), dtype=torch.bool, device=S.device)
    npv = torch.zeros((B_batch, N_paths), dtype=torch.float64, device=S.device)
    life = torch.zeros((B_batch, N_paths), dtype=torch.float64, device=S.device)

    B_mat = B.view(-1, 1)
    coupon_mat = coupon.view(-1, 1)
    r_mat = r.view(-1, 1)

    for idx in obs_indices:
        t_i = idx.to(torch.float64) * dt
        trigger = (S[:, :, idx] >= (B_mat * S0)) & (~called)
        discount = torch.exp(-r_mat * t_i)
        npv = npv + trigger.to(torch.float64) * discount * (1.0 + coupon_mat * t_i)
        life = life + trigger.to(torch.float64) * t_i
        called = called | trigger

    # Capital protection for uncalled paths at maturity
    uncalled = ~called
    discount_T = torch.exp(-r_mat * T)
    npv = npv + uncalled.to(torch.float64) * discount_T
    life = life + uncalled.to(torch.float64) * T

    mean_npv = npv.mean(dim=1)
    call_prob = called.to(torch.float64).mean(dim=1)
    exp_life = life.mean(dim=1)

    return mean_npv.clone(), call_prob.clone(), exp_life.clone()


def price_autocall_mc(
    S: torch.Tensor,
    obs_indices: Union[List[int], torch.Tensor],
    B: torch.Tensor,
    coupon: torch.Tensor,
    r: torch.Tensor,
    T: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Monte Carlo pricing for 1-leg vanilla autocallable notes.

    Parameters:
        S: Spot paths tensor of shape (B, N_paths, N_steps + 1) in float64.
        obs_indices: Step indices of observation dates.
        B: Barrier tensor of shape (B,) as fraction of S0.
        coupon: Annual coupon rate tensor of shape (B,).
        r: Risk-free rate tensor of shape (B,).
        T: Total maturity.
        dt: Time step size.

    Returns:
        npv: Expected discounted payoff of shape (B,).
        call_prob: Autocall early redemption probability of shape (B,).
        exp_life: Expected time to redemption in years of shape (B,).
    """
    if isinstance(obs_indices, list):
        obs_tensor = torch.tensor(obs_indices, dtype=torch.long, device=S.device)
    else:
        obs_tensor = obs_indices.to(device=S.device, dtype=torch.long)

    S_f64 = S.to(torch.float64)
    B_f64 = B.to(device=S.device, dtype=torch.float64).view(-1)
    coupon_f64 = coupon.to(device=S.device, dtype=torch.float64).view(-1)
    r_f64 = r.to(device=S.device, dtype=torch.float64).view(-1)

    return _price_autocall_kernel(
        S_f64, obs_tensor, B_f64, coupon_f64, r_f64, float(T), float(dt)
    )

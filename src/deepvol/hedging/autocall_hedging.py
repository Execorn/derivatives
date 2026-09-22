"""
Deep Hedging Environment and LSTM Policy for Autocallable Notes.

Implements:
  - AutocallHedgingEnv: Observation-date aware hedging environment tracking
    gap risk, early redemption status, and non-linear transaction costs.
  - AutocallHedgePolicy: 2-layer LSTM policy network outputting delta positions.
  - train_autocall_hedger: Training pipeline optimizing Entropic / CVaR risk measures.

Mathematical Formulation:
  - Buehler, H. et al. (2019). Deep Hedging. Quantitative Finance, 19(8), 1271-1291.
  - Sharma, A. et al. (2024). Hedging and Pricing Structured Products Featuring Multiple Underlying Assets. ACM ICAIF.
"""

import sys
sys.path.insert(0, "src")

import os
import time
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from deepvol.hedging.barrier_hedging import BarrierHedgingEnv


class AutocallHedgingEnv(BarrierHedgingEnv):
    """
    Hedging environment tailored for autocallable structured products.
    State representation captures proximity to observation dates, distance to barrier,
    called status, and delta history to control transaction costs.
    """
    def __init__(
        self,
        H: Tensor,               # (N_paths, N_t+1, d) hedging instrument paths
        cost_coeffs: Tensor,     # (d,) proportional transaction cost
        S0: float = 100.0,
        strike: float = 100.0,
        B_call: float = 1.0,     # relative barrier (e.g. 1.0 = 100%)
        coupon: float = 0.02,    # per-period coupon
        r: float = 0.05,
        T: float = 1.0,
        obs_indices: Optional[List[int]] = None,
        risk_aversion: float = 1.0,
        risk_measure: Literal["entropic", "cvar"] = "entropic",
        cvar_alpha: float = 0.05,
    ) -> None:
        super().__init__(
            H=H,
            cost_coeffs=cost_coeffs,
            strike=strike,
            barrier=B_call * S0,
            expiry=T,
            risk_aversion=risk_aversion,
            risk_measure=risk_measure,
        )
        self.S0 = S0
        self.B_call = B_call
        self.coupon = coupon
        self.r = r
        self.T = T
        self.cvar_alpha = cvar_alpha
        self.obs_indices = sorted(obs_indices) if obs_indices is not None else [63, 126, 189, 252]
        self.obs_set = set(self.obs_indices)
        self.obs_map = {step: i + 1 for i, step in enumerate(self.obs_indices)}

        self.called_mask = torch.zeros(self.N_paths, dtype=torch.bool, device=H.device)
        self.prev_delta = torch.zeros(self.N_paths, self.d, dtype=H.dtype, device=H.device)

    def get_state(
        self,
        k: int,
        called_mask: Optional[Tensor] = None,
        prev_delta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Construct 7-dimensional state tensor at step k.
        Features:
          0: log(S_k / S0)
          1: 0.0 (vol proxy or variance level)
          2: tau_k / T = (T - t_k) / T
          3: steps to next obs date / (T * 252)
          4: called_flag (1.0 if called, 0.0 if active)
          5: steps since last obs date / (T * 252)
          6: prev_delta (underlying delta)
        """
        device = self.H.device
        dtype = self.H.dtype
        S_k = self.H[:, k, 0]
        log_m = torch.log(torch.clamp(S_k / self.S0, min=1e-6))
        tau = (self.T - (k * self.dt)) / self.T

        # Find steps to next obs date and steps since last obs date
        next_obs_dist = float(self.N_t)
        for obs_step in self.obs_indices:
            if obs_step >= k:
                next_obs_dist = float(obs_step - k)
                break
        norm_next_obs = next_obs_dist / max(1.0, float(self.N_t))

        last_obs_dist = float(k)
        for obs_step in reversed(self.obs_indices):
            if obs_step <= k:
                last_obs_dist = float(k - obs_step)
                break
        norm_last_obs = last_obs_dist / max(1.0, float(self.N_t))

        c_mask = self.called_mask if called_mask is None else called_mask
        p_delta = self.prev_delta if prev_delta is None else prev_delta

        f0 = log_m
        f1 = torch.zeros_like(log_m)
        f2 = torch.full_like(log_m, tau)
        f3 = torch.full_like(log_m, norm_next_obs)
        f4 = c_mask.to(dtype)
        f5 = torch.full_like(log_m, norm_last_obs)
        f6 = p_delta[:, 0]

        return torch.stack([f0, f1, f2, f3, f4, f5, f6], dim=1)

    def simulate_episode(
        self,
        policy: "AutocallHedgePolicy",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run one full hedging trajectory episode.
        Returns:
            (wealth, payoff, pnl_per_step)
        """
        device = self.H.device
        dtype = self.H.dtype

        called = torch.zeros(self.N_paths, dtype=torch.bool, device=device)
        wealth = torch.zeros(self.N_paths, dtype=dtype, device=device)
        payoff = torch.zeros(self.N_paths, dtype=dtype, device=device)
        pnl_per_step = torch.zeros(self.N_paths, self.N_t, dtype=dtype, device=device)

        prev_delta = torch.zeros(self.N_paths, self.d, dtype=dtype, device=device)
        h, c = None, None

        for k in range(self.N_t):
            t_k = k * self.dt
            # Observation check
            if k in self.obs_set:
                obs_num = self.obs_map[k]
                S_k = self.H[:, k, 0]
                trigger = (~called) & (S_k >= self.B_call * self.S0)
                disc = torch.exp(torch.tensor(-self.r * t_k, dtype=dtype, device=device))
                call_payoff = disc * (1.0 + self.coupon * float(obs_num))
                payoff = payoff + trigger.to(dtype) * call_payoff
                called = called | trigger

            state = self.get_state(k, called_mask=called, prev_delta=prev_delta)
            delta, h, c = policy(state, h, c)

            # If called, zero out delta positions to unwind
            delta = torch.where(called.unsqueeze(1), torch.zeros_like(delta), delta)

            # Transaction cost
            delta_diff = (delta - prev_delta).abs()
            tc = (delta_diff * self.cost_coeffs).sum(dim=1)

            # Price change of hedging instruments
            dH = self.H[:, k + 1, :] - self.H[:, k, :]
            gain = (delta * dH).sum(dim=1)

            step_pnl = gain - tc
            wealth = wealth + step_pnl
            pnl_per_step[:, k] = step_pnl
            prev_delta = delta

        # Terminal maturity check for uncalled paths
        uncalled = ~called
        disc_T = torch.exp(torch.tensor(-self.r * self.T, dtype=dtype, device=device))
        payoff = payoff + uncalled.to(dtype) * disc_T * 1.0

        return wealth, payoff, pnl_per_step

    def compute_loss(self, wealth: Tensor, payoff: Tensor) -> Tensor:
        """Compute hedging loss under Entropic risk measure or CVaR at alpha=5%."""
        shortfall = payoff - wealth
        if self.risk_measure == "cvar":
            var_idx = int(self.N_paths * (1.0 - self.cvar_alpha))
            sorted_shortfall, _ = torch.sort(shortfall)
            var_alpha = sorted_shortfall[min(var_idx, self.N_paths - 1)]
            tail_loss = torch.clamp(shortfall - var_alpha, min=0.0)
            return var_alpha + (1.0 / self.cvar_alpha) * tail_loss.mean()
        else:
            # Entropic loss: E[exp(lambda * (payoff - wealth))]
            lam = self.risk_aversion
            return torch.mean(torch.exp(torch.clamp(lam * shortfall, -20.0, 20.0)))


class AutocallHedgePolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_instruments: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_instruments = n_instruments

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_instruments),
            nn.Tanh(),
        )

    def forward(
        self,
        state: Tensor,
        h: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # state: (N_paths, state_dim) -> unsqueeze to (N_paths, 1, state_dim)
        x = state.unsqueeze(1)
        if h is not None and c is not None:
            out, (h_new, c_new) = self.lstm(x, (h, c))
        else:
            out, (h_new, c_new) = self.lstm(x)

        # Clamp output leverage to [-2.0, 2.0]
        delta = self.head(out.squeeze(1)) * 2.0
        return delta, h_new, c_new


def train_autocall_hedger(
    env_factory: Callable[[], AutocallHedgingEnv],
    policy: AutocallHedgePolicy,
    n_epochs: int = 200,
    lr: float = 1e-3,
    n_paths_per_epoch: int = 500,
    device: str = "cuda",
) -> AutocallHedgePolicy:
    """Train AutocallHedgePolicy across episodes."""
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_std_pnl = float("inf")
    best_state_dict = None

    for epoch in range(1, n_epochs + 1):
        policy.train()
        env = env_factory()

        optimizer.zero_grad()
        wealth, payoff, _ = env.simulate_episode(policy)
        loss = env.compute_loss(wealth, payoff)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pnl = wealth - payoff
            mean_pnl = float(pnl.mean().item())
            std_pnl = float(pnl.std().item())

        if std_pnl < best_std_pnl:
            best_std_pnl = std_pnl
            best_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"Hedging Epoch {epoch:3d}/{n_epochs:3d} | Loss: {loss.item():.4f} | "
                  f"Mean P&L: {mean_pnl:.4f} | Std P&L: {std_pnl:.4f}")

    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)
        weights_path = "artifacts/weights/autocall_hedge_policy.pth"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(best_state_dict, weights_path)
        print(f"Saved best hedging policy weights to {weights_path}.")

    return policy


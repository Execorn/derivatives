"""
barrier_hedging.py — Custom Deep Hedging environment for Down-and-Out Barrier Call options.
Implements dynamic pathwise knockout conditions and boundary-aware state representations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BarrierHedgingEnv:
    """
    Vectorized deep hedging environment for Down-and-Out Barrier Call (DOBC) options.
    Manages barrier touch events, active/knocked-out states, and transaction costs.
    """
    def __init__(
        self,
        H: torch.Tensor,
        cost_coeffs: torch.Tensor,
        strike: float = 100.0,
        barrier: float = 85.0,
        expiry: float = 1.0,
        risk_aversion: float = 1.0,
        risk_measure: str = "entropic",
        t_grid: Optional[torch.Tensor] = None
    ):
        """
        Parameters:
            H: Price path of hedging instruments of shape (N_paths, N_t + 1, d).
               H[:, :, 0] is the underlying stock spot S_t.
            cost_coeffs: Proportional cost coefficients of shape (d,).
            strike: Strike price K.
            barrier: Lower knock-out barrier B (must be < S_0).
            expiry: Maturity T.
            risk_aversion: Risk aversion lambda.
            risk_measure: "entropic" or "quad".
            t_grid: Optional time steps grid of shape (N_t + 1,).
        """
        self.H = H
        self.cost_coeffs = cost_coeffs.to(device=H.device, dtype=H.dtype)
        self.strike = strike
        self.barrier = barrier
        self.expiry = expiry
        self.risk_aversion = risk_aversion
        self.risk_measure = risk_measure.lower()
        
        self.N_paths, self.N_t_plus_1, self.d = H.shape
        self.N_t = self.N_t_plus_1 - 1
        self.dt = expiry / self.N_t
        
        if t_grid is None:
            self.t_grid = torch.arange(self.N_t_plus_1, device=H.device, dtype=H.dtype) * self.dt
        else:
            self.t_grid = t_grid.to(device=H.device, dtype=H.dtype)
            
        # Payoff is computed dynamically during the hedging episode
        self.payoff = None
            
    def get_state(self, k: int, prev_delta: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        Constructs boundary-aware state features for step k.
        Features: [log(S_k / K), log(S_k / B), T - t_k, active_mask, prev_delta]
        """
        if hasattr(self, "_precomputed_log_moneyness") and self._precomputed_log_moneyness is not None:
            log_moneyness = self._precomputed_log_moneyness[:, k]
            log_barrier_dist = self._precomputed_log_barrier_dist[:, k]
            time_to_expiry_tensor = self._precomputed_time_to_expiry[:, k]
        else:
            S_k = self.H[:, k, 0:1]  # (N_paths, 1)
            log_moneyness = torch.log(torch.clamp(S_k / self.strike, min=1e-5))
            
            # log-distance to barrier. Clamp to prevent log(negative) if spot breaches barrier.
            log_barrier_dist = torch.log(torch.clamp(S_k / self.barrier, min=1e-5))
            
            time_to_expiry = self.expiry - self.t_grid[k]
            time_to_expiry_tensor = torch.full_like(S_k, time_to_expiry)
        
        # Combine log_moneyness, log_barrier_dist, time_to_expiry, active_mask, and all dimensions of prev_delta
        # shape: (N_paths, 4 + d)
        state = torch.cat([log_moneyness, log_barrier_dist, time_to_expiry_tensor, active_mask, prev_delta], dim=-1)
        return state

    def simulate_hedging_episode(self, policy: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates the barrier option hedging episode, tracking the knockout condition dynamically.
        
        Returns:
            wealth: Final portfolio wealth of shape (N_paths,)
            total_costs: Total transaction costs of shape (N_paths,)
            all_deltas: Hedge paths of shape (N_paths, N_t, d)
        """
        device = self.H.device
        dtype = self.H.dtype
        
        # Precompute state features if enabled
        if getattr(self, "precompute", True):
            S = self.H[:, :, 0:1]
            self._precomputed_log_moneyness = torch.log(torch.clamp(S / self.strike, min=1e-5))
            self._precomputed_log_barrier_dist = torch.log(torch.clamp(S / self.barrier, min=1e-5))
            
            time_to_expiry_all = self.expiry - self.t_grid
            self._precomputed_time_to_expiry = time_to_expiry_all.view(1, -1, 1).expand(self.N_paths, -1, -1)
        else:
            self._precomputed_log_moneyness = None
            self._precomputed_log_barrier_dist = None
            self._precomputed_time_to_expiry = None
        
        wealth = torch.zeros(self.N_paths, device=device, dtype=dtype)
        total_costs = torch.zeros(self.N_paths, device=device, dtype=dtype)
        
        prev_delta = torch.zeros(self.N_paths, self.d, device=device, dtype=dtype)
        lstm_state = None
        
        # Track the running minimum of the spot price to check for knockout events
        running_min = self.H[:, 0, 0].clone()  # (N_paths,)
        
        deltas = []
        
        for k in range(self.N_t):
            # Update running minimum with current spot S_k
            S_k = self.H[:, k, 0]
            running_min = torch.minimum(running_min, S_k)
            
            # Option is active if running minimum is strictly above the barrier
            active_mask = (running_min > self.barrier).float().unsqueeze(-1)  # (N_paths, 1)
            
            # 1. Get environment state
            state = self.get_state(k, prev_delta, active_mask)
            
            # 2. Get hedge action
            delta, lstm_state = policy(state, lstm_state)  # delta shape: (N_paths, d)
            
            # Force delta to zero if option has knocked out
            delta = delta * active_mask
            
            deltas.append(delta)
            
            # 3. Calculate cost
            delta_diff = torch.abs(delta - prev_delta)
            step_costs = torch.sum(self.cost_coeffs.unsqueeze(0) * self.H[:, k, :] * delta_diff, dim=-1)
            total_costs = total_costs + step_costs
            
            # 4. Update wealth
            price_change = self.H[:, k+1, :] - self.H[:, k, :]
            trading_gain = torch.sum(delta * price_change, dim=-1)
            wealth = wealth + trading_gain - step_costs
            
            prev_delta = delta
            
        # 5. Unwind portfolio to 0 at maturity (T)
        terminal_unwind_cost = torch.sum(self.cost_coeffs.unsqueeze(0) * self.H[:, -1, :] * torch.abs(prev_delta), dim=-1)
        total_costs = total_costs + terminal_unwind_cost
        wealth = wealth - terminal_unwind_cost
        
        # 6. Evaluate final payoff under knockout condition
        S_N = self.H[:, -1, 0]
        running_min = torch.minimum(running_min, S_N)
        final_active_mask = (running_min > self.barrier).float()
        
        # Terminal payoff payoff = max(S_T - K, 0) * I_T
        payoff = torch.clamp(S_N - self.strike, min=0.0) * final_active_mask
        self.payoff = payoff  # Store the dynamically computed payoff
        
        all_deltas = torch.stack(deltas, dim=1)
        
        # Clean up precomputed features
        self._precomputed_log_moneyness = None
        self._precomputed_log_barrier_dist = None
        self._precomputed_time_to_expiry = None
        
        return wealth, total_costs, all_deltas

    def compute_loss(self, wealth: torch.Tensor) -> torch.Tensor:
        """
        Computes the risk measure loss comparing wealth against the dynamic barrier payoff.
        """
        hedging_error = wealth - self.payoff
        
        if self.risk_measure == "entropic":
            loss = torch.mean(torch.exp(-self.risk_aversion * hedging_error))
        elif self.risk_measure == "quad":
            loss = torch.mean(hedging_error ** 2)
        else:
            raise ValueError(f"Unknown risk measure: {self.risk_measure}")
            
        return loss

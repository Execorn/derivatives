"""
frictional_env.py — Vectorized environment for deep hedging under market friction.
Incorporates proportional transaction costs and power-law temporary market impact (slippage).
Calculations are performed in torch.float64 to ensure numerical stability.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


@torch.compile(mode="reduce-overhead")
def _step_wealth_and_cost_compiled(
    wealth: torch.Tensor,
    total_costs: torch.Tensor,
    H_k: torch.Tensor,
    H_k_next: torch.Tensor,
    delta: torch.Tensor,
    prev_delta: torch.Tensor,
    gamma_0: torch.Tensor,
    gamma_1: torch.Tensor,
    alpha: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compiled step calculation to update portfolio wealth and transaction costs.
    """
    delta_diff = torch.abs(delta - prev_delta)
    # Power-law transaction cost: S_k * |delta_diff| * (gamma_0 + gamma_1 * |delta_diff|^alpha)
    step_costs = torch.sum(H_k * delta_diff * (gamma_0 + gamma_1 * torch.pow(delta_diff, alpha)), dim=-1)
    
    price_change = H_k_next - H_k
    trading_gain = torch.sum(delta * price_change, dim=-1)
    
    new_wealth = wealth + trading_gain - step_costs
    new_costs = total_costs + step_costs
    return new_wealth.clone(), new_costs.clone()


@torch.compile(mode="reduce-overhead")
def _terminal_unwind_compiled(
    wealth: torch.Tensor,
    total_costs: torch.Tensor,
    H_T: torch.Tensor,
    prev_delta: torch.Tensor,
    gamma_0: torch.Tensor,
    gamma_1: torch.Tensor,
    alpha: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compiled step calculation to unwind the hedging portfolio at maturity.
    """
    delta_diff = torch.abs(prev_delta)
    # Power-law transaction cost for final unwind to zero position
    step_costs = torch.sum(H_T * delta_diff * (gamma_0 + gamma_1 * torch.pow(delta_diff, alpha)), dim=-1)
    
    new_wealth = wealth - step_costs
    new_costs = total_costs + step_costs
    return new_wealth.clone(), new_costs.clone()


class FrictionalHedgingEnv:
    """
    Vectorized deep hedging environment managing pathwise wealth, transaction costs,
    and power-law slippage / market impact under friction.
    """
    def __init__(
        self,
        H: torch.Tensor,
        payoff: Optional[torch.Tensor] = None,
        gamma_0: Union[float, torch.Tensor] = 0.0,
        gamma_1: Union[float, torch.Tensor] = 0.0,
        alpha: Union[float, torch.Tensor] = 1.0,
        risk_aversion: float = 1.0,
        risk_measure: str = "entropic",
        strike: float = 100.0,
        expiry: float = 1.0,
        t_grid: Optional[torch.Tensor] = None
    ):
        """
        Parameters:
            H: Tensor of shape (N_paths, N_t + 1, d) - prices of hedging instruments.
               H[:, :, 0] is assumed to be the underlying stock spot S_t.
            payoff: Tensor of shape (N_paths,) - terminal payoff of the derivative.
            gamma_0: Proportional cost coefficient (linear half-spread).
            gamma_1: Market impact cost coefficient.
            alpha: Power-law exponent.
            risk_aversion: Risk aversion parameter lambda.
            risk_measure: "entropic" or "quad".
            strike: Strike price K of the option.
            expiry: Maturity T of the option.
            t_grid: Optional tensor of shape (N_t + 1,) representing the simulation time steps.
        """
        # Force double precision internally for all pricing and wealth paths
        self.H = H.to(dtype=torch.float64)
        
        self.N_paths, self.N_t_plus_1, self.d = H.shape
        self.N_t = self.N_t_plus_1 - 1
        self.dt = expiry / self.N_t
        self.risk_aversion = risk_aversion
        self.risk_measure = risk_measure.lower()
        self.strike = strike
        self.expiry = expiry
        
        if payoff is None:
            self.payoff = torch.zeros(self.N_paths, device=H.device, dtype=torch.float64)
        else:
            self.payoff = payoff.to(device=H.device, dtype=torch.float64)
            
        if t_grid is None:
            self.t_grid = torch.arange(self.N_t_plus_1, device=H.device, dtype=torch.float64) * self.dt
        else:
            self.t_grid = t_grid.to(device=H.device, dtype=torch.float64)
            
        # Parse transaction cost coefficients
        self.gamma_0 = self._parse_coeff(gamma_0, "gamma_0")
        self.gamma_1 = self._parse_coeff(gamma_1, "gamma_1")
        self.alpha = self._parse_coeff(alpha, "alpha")
        
        self.precompute = True
        self._precomputed_log_moneyness = None
        self._precomputed_time_to_expiry = None
        self._precomputed_vol_proxy = None
        
    def _parse_coeff(self, val: Union[float, torch.Tensor], name: str) -> torch.Tensor:
        if isinstance(val, (int, float)):
            return torch.full((self.d,), float(val), device=self.H.device, dtype=torch.float64)
        elif torch.is_tensor(val):
            if val.dim() == 0:
                return val.expand(self.d).to(device=self.H.device, dtype=torch.float64)
            elif val.dim() == 1 and val.shape[0] == self.d:
                return val.to(device=self.H.device, dtype=torch.float64)
            else:
                raise ValueError(f"{name} must be a scalar or a 1D tensor of shape ({self.d},)")
        else:
            raise TypeError(f"Invalid type for {name}: {type(val)}")

    def get_state(self, k: int, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Constructs the state feature vector for time step k.
        Features are output in float32 for neural network policy evaluation.
        """
        if self._precomputed_log_moneyness is not None:
            log_moneyness = self._precomputed_log_moneyness[:, k]
            time_to_expiry_tensor = self._precomputed_time_to_expiry[:, k]
            vol_proxy = self._precomputed_vol_proxy[:, k]
        else:
            S_k = self.H[:, k, 0:1]
            log_moneyness = torch.log(torch.clamp(S_k / self.strike, min=1e-5))
            time_to_expiry = self.expiry - self.t_grid[k]
            time_to_expiry_tensor = torch.full_like(S_k, time_to_expiry)
            
            if k < 5:
                vol_proxy = torch.full_like(S_k, 0.2)
            else:
                past_S = self.H[:, k-5:k+1, 0]
                log_returns = torch.log(torch.clamp(past_S[:, 1:] / torch.clamp(past_S[:, :-1], min=1e-5), min=1e-5))
                annualization = np.sqrt(1.0 / self.dt)
                vol_proxy = torch.std(log_returns, dim=-1, keepdim=True) * annualization
                
        # Ensure prev_delta is float64 internally and state is returned in float32
        state = torch.cat([log_moneyness, time_to_expiry_tensor, vol_proxy, prev_delta], dim=-1)
        return state.to(dtype=torch.float32)

    def simulate_hedging_episode(self, policy: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates the hedging episode across all time steps.
        """
        device = self.H.device
        dtype = torch.float64
        
        if self.precompute:
            S = self.H[:, :, 0:1]
            self._precomputed_log_moneyness = torch.log(torch.clamp(S / self.strike, min=1e-5))
            time_to_expiry_all = self.expiry - self.t_grid
            self._precomputed_time_to_expiry = time_to_expiry_all.view(1, -1, 1).expand(self.N_paths, -1, -1)
            
            self._precomputed_vol_proxy = torch.full_like(S, 0.2)
            if self.N_t >= 5:
                S_0 = self.H[:, :, 0]
                log_returns = torch.log(torch.clamp(S_0[:, 1:] / torch.clamp(S_0[:, :-1], min=1e-5), min=1e-5))
                windows = log_returns.unfold(dimension=-1, size=5, step=1)
                vol_proxy_windows = torch.std(windows, dim=-1, keepdim=True) * np.sqrt(1.0 / self.dt)
                self._precomputed_vol_proxy[:, 5:self.N_t + 1] = vol_proxy_windows
        else:
            self._precomputed_log_moneyness = None
            self._precomputed_time_to_expiry = None
            self._precomputed_vol_proxy = None
            
        wealth = torch.zeros(self.N_paths, device=device, dtype=dtype)
        total_costs = torch.zeros(self.N_paths, device=device, dtype=dtype)
        prev_delta = torch.zeros(self.N_paths, self.d, device=device, dtype=dtype)
        lstm_state = None
        deltas = []
        
        for k in range(self.N_t):
            state = self.get_state(k, prev_delta)
            
            # Policy is evaluated in float32, output is cast to float64 for wealth dynamics
            delta, lstm_state = policy(state, lstm_state)
            delta = delta.to(dtype=dtype)
            deltas.append(delta)
            
            wealth, total_costs = _step_wealth_and_cost_compiled(
                wealth=wealth,
                total_costs=total_costs,
                H_k=self.H[:, k, :],
                H_k_next=self.H[:, k+1, :],
                delta=delta,
                prev_delta=prev_delta,
                gamma_0=self.gamma_0,
                gamma_1=self.gamma_1,
                alpha=self.alpha
            )
            wealth = wealth.clone()
            total_costs = total_costs.clone()
            prev_delta = delta
            
        # Unwind portfolio to zero position at maturity
        wealth, total_costs = _terminal_unwind_compiled(
            wealth=wealth,
            total_costs=total_costs,
            H_T=self.H[:, -1, :],
            prev_delta=prev_delta,
            gamma_0=self.gamma_0,
            gamma_1=self.gamma_1,
            alpha=self.alpha
        )
        wealth = wealth.clone()
        total_costs = total_costs.clone()
        
        all_deltas = torch.stack(deltas, dim=1)
        
        # Clean up precomputed features
        self._precomputed_log_moneyness = None
        self._precomputed_time_to_expiry = None
        self._precomputed_vol_proxy = None
        
        return wealth, total_costs, all_deltas

    def compute_loss(self, wealth: torch.Tensor) -> torch.Tensor:
        """
        Computes the hedging risk measure loss in float64.
        """
        hedging_error = wealth - self.payoff
        
        if self.risk_measure == "entropic":
            loss = torch.mean(torch.exp(torch.clamp(-self.risk_aversion * hedging_error, max=20.0)))
        elif self.risk_measure == "quad":
            loss = torch.mean(hedging_error ** 2)
        else:
            raise ValueError(f"Unknown risk measure: {self.risk_measure}")
            
        return loss

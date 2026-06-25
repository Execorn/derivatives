"""
indifference_pricing.py — Utility indifference pricing engine under market friction.
Trains policy networks to compute option bid/ask pricing surfaces and spreads.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List, Union

from deepvol.hedging.frictional_env import FrictionalHedgingEnv
from deepvol.hedging.deep_hedging import HedgingPolicy


def bs_call_price(S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the Black-Scholes Call price in torch.float64.
    """
    # S, K, T, sigma must be double precision
    S = S.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)
    sigma = sigma.to(dtype=torch.float64)
    
    d1 = (torch.log(S / K) + 0.5 * (sigma ** 2) * T) / (sigma * torch.sqrt(T) + 1e-15)
    d2 = d1 - sigma * torch.sqrt(T)
    
    # Vectorized normal CDF approximation
    cdf1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))
    cdf2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2.0)))
    
    price = S * cdf1 - K * cdf2
    return price


def bs_call_vega(S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the Black-Scholes Call vega in torch.float64.
    """
    S = S.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)
    sigma = sigma.to(dtype=torch.float64)
    
    d1 = (torch.log(S / K) + 0.5 * (sigma ** 2) * T) / (sigma * torch.sqrt(T) + 1e-15)
    pdf1 = torch.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    
    vega = S * torch.sqrt(T) * pdf1
    return vega


def invert_implied_volatility_hybrid(
    price: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Vectorized and differentiable implied volatility solver using hybrid bisection + Newton-Raphson.
    Clamps the minimum volatility parameter to 0.01 (100 bps) to prevent Durrleman singularities.
    Operates strictly in torch.float64.
    """
    price = price.to(dtype=torch.float64)
    S = S.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)
    
    # Intrinsic value
    intrinsic = torch.clamp(S - K, min=0.0)
    
    # Check valid option prices (must be strictly between intrinsic value and spot price)
    valid_mask = (price > intrinsic) & (price < S)
    
    # Set bounds for hybrid search
    low = torch.full_like(price, 0.01)  # Minimum volatility clamped to 0.01
    high = torch.full_like(price, 5.0)
    
    # Initial guess is the midpoint
    sigma = 0.5 * (low + high)
    
    for _ in range(max_iter):
        p = bs_call_price(S, K, T, sigma)
        diff = p - price
        
        # Check convergence
        max_diff = torch.max(torch.abs(diff)[valid_mask]) if torch.any(valid_mask) else 0.0
        if max_diff < tol:
            break
            
        vega = bs_call_vega(S, K, T, sigma)
        
        # Update bounds
        low = torch.where(diff < 0, sigma, low)
        high = torch.where(diff >= 0, sigma, high)
        
        # Newton-Raphson step
        vega_guarded = torch.where(vega > 1e-8, vega, torch.tensor(1e-8, device=vega.device, dtype=torch.float64))
        newton_step = sigma - diff / vega_guarded
        
        # Fallback to bisection if Newton step goes outside active interval
        use_newton = (newton_step > low) & (newton_step < high)
        sigma = torch.where(use_newton, newton_step, 0.5 * (low + high))
        
        # Clamp to mathematical boundaries
        sigma = torch.clamp(sigma, min=0.01, max=5.0)
        
    result = torch.where(valid_mask, sigma, torch.tensor(float('nan'), device=price.device, dtype=torch.float64))
    return result


def train_frictional_hedger(
    env: FrictionalHedgingEnv,
    policy: nn.Module,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    device: str = "cuda"
) -> List[float]:
    """
    Trains the hedging policy on a frictional environment.
    """
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    losses = []
    
    num_paths = env.N_paths
    num_batches = (num_paths + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        indices = torch.randperm(num_paths, device=device)
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            if len(batch_idx) == 0:
                continue
                
            optimizer.zero_grad()
            
            # Slice batch
            batch_H = env.H[batch_idx]
            batch_payoff = env.payoff[batch_idx]
            
            sub_env = FrictionalHedgingEnv(
                H=batch_H,
                payoff=batch_payoff,
                gamma_0=env.gamma_0,
                gamma_1=env.gamma_1,
                alpha=env.alpha,
                risk_aversion=env.risk_aversion,
                risk_measure=env.risk_measure,
                strike=env.strike,
                expiry=env.expiry,
                t_grid=env.t_grid
            )
            sub_env.precompute = env.precompute
            
            wealth, _, _ = sub_env.simulate_hedging_episode(policy)
            loss = sub_env.compute_loss(wealth)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_idx)
            
        avg_loss = epoch_loss / num_paths
        losses.append(avg_loss)
        
    return losses


@torch.no_grad()
def evaluate_loss(env: FrictionalHedgingEnv, policy: nn.Module, batch_size: int = 1024) -> float:
    """
    Evaluates the expected utility loss on the environment in float64.
    """
    policy.eval()
    num_paths = env.N_paths
    num_batches = (num_paths + batch_size - 1) // batch_size
    
    total_loss = 0.0
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, num_paths)
        if start_idx >= end_idx:
            continue
            
        batch_H = env.H[start_idx:end_idx]
        batch_payoff = env.payoff[start_idx:end_idx]
        
        sub_env = FrictionalHedgingEnv(
            H=batch_H,
            payoff=batch_payoff,
            gamma_0=env.gamma_0,
            gamma_1=env.gamma_1,
            alpha=env.alpha,
            risk_aversion=env.risk_aversion,
            risk_measure=env.risk_measure,
            strike=env.strike,
            expiry=env.expiry,
            t_grid=env.t_grid
        )
        sub_env.precompute = env.precompute
        
        wealth, _, _ = sub_env.simulate_hedging_episode(policy)
        loss = sub_env.compute_loss(wealth)
        total_loss += loss.item() * (end_idx - start_idx)
        
    return total_loss / num_paths


class IndifferencePricingEngine:
    """
    Utility indifference pricing engine under market friction.
    Trains policy networks for pure asset, short option, and long option portfolios
    to compute utility indifference bid/ask prices and spreads.
    """
    def __init__(
        self,
        H: torch.Tensor,
        payoff: torch.Tensor,
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
            H: Price paths of shape (N_paths, N_t + 1, d)
            payoff: Option payoff at maturity T of shape (N_paths,)
            gamma_0: Proportional cost coefficient
            gamma_1: Market impact cost coefficient
            alpha: Power-law exponent
            risk_aversion: Risk aversion parameter lambda
            risk_measure: Risk measure type
            strike: Strike price K
            expiry: Expiry T
            t_grid: Optional time steps grid
        """
        self.H = H
        self.payoff = payoff
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.alpha = alpha
        self.risk_aversion = risk_aversion
        self.risk_measure = risk_measure
        self.strike = strike
        self.expiry = expiry
        self.t_grid = t_grid
        
        self.N_paths, _, self.d = H.shape
        
        # Policies
        self.policy_pure = HedgingPolicy(input_dim=3 + self.d, hidden_dim=64, output_dim=self.d)
        self.policy_short = HedgingPolicy(input_dim=3 + self.d, hidden_dim=64, output_dim=self.d)
        self.policy_long = HedgingPolicy(input_dim=3 + self.d, hidden_dim=64, output_dim=self.d)
        
        # Environment configurations
        # Pure asset env (payoff = 0)
        self.env_pure = FrictionalHedgingEnv(
            H=H, payoff=None, gamma_0=gamma_0, gamma_1=gamma_1, alpha=alpha,
            risk_aversion=risk_aversion, risk_measure=risk_measure, strike=strike, expiry=expiry, t_grid=t_grid
        )
        # Short option env (payoff = Y)
        self.env_short = FrictionalHedgingEnv(
            H=H, payoff=payoff, gamma_0=gamma_0, gamma_1=gamma_1, alpha=alpha,
            risk_aversion=risk_aversion, risk_measure=risk_measure, strike=strike, expiry=expiry, t_grid=t_grid
        )
        # Long option env (payoff = -Y)
        self.env_long = FrictionalHedgingEnv(
            H=H, payoff=-payoff, gamma_0=gamma_0, gamma_1=gamma_1, alpha=alpha,
            risk_aversion=risk_aversion, risk_measure=risk_measure, strike=strike, expiry=expiry, t_grid=t_grid
        )

    def train_policies(
        self,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-3,
        device: str = "cuda"
    ) -> dict:
        """
        Trains the pure, short, and long hedging policies.
        """
        losses_pure = train_frictional_hedger(self.env_pure, self.policy_pure, lr=lr, epochs=epochs, batch_size=batch_size, device=device)
        losses_short = train_frictional_hedger(self.env_short, self.policy_short, lr=lr, epochs=epochs, batch_size=batch_size, device=device)
        losses_long = train_frictional_hedger(self.env_long, self.policy_long, lr=lr, epochs=epochs, batch_size=batch_size, device=device)
        
        return {
            "losses_pure": losses_pure,
            "losses_short": losses_short,
            "losses_long": losses_long
        }

    def compute_prices(self, batch_size: int = 1024) -> Tuple[float, float, float]:
        """
        Computes indifference bid and ask prices and the bid-ask spread.
        Returns:
            bid: Indifference bid price (float)
            ask: Indifference ask price (float)
            spread: Bid-ask spread (float)
        """
        if self.risk_measure.lower() != "entropic":
            raise NotImplementedError("Indifference pricing is only defined for entropic risk measure.")
            
        L_0 = evaluate_loss(self.env_pure, self.policy_pure, batch_size=batch_size)
        L_Y = evaluate_loss(self.env_short, self.policy_short, batch_size=batch_size)
        L_minus_Y = evaluate_loss(self.env_long, self.policy_long, batch_size=batch_size)
        
        # Safeguards to prevent potential numerical/training noise issues
        # Mathematically, L_Y >= L_0 and L_minus_Y <= L_0 (since Y >= 0)
        # But we compute them as-is, adding a small clamp for log stability
        L_0 = max(L_0, 1e-15)
        L_Y = max(L_Y, 1e-15)
        L_minus_Y = max(L_minus_Y, 1e-15)
        
        ask = (1.0 / self.risk_aversion) * np.log(L_Y / L_0)
        bid = - (1.0 / self.risk_aversion) * np.log(L_minus_Y / L_0)
        
        # Enforce spread non-negativity
        spread = ask - bid
        if spread < 0:
            # Recompute spread using a floor or issue a warning
            spread = max(spread, 0.0)
            
        return float(bid), float(ask), float(spread)

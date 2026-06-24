"""
GPU Monte Carlo Value-at-Risk (VaR) and Expected Shortfall (ES) Engine.
"""

import math
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from deepvol.greeks.portfolio_greeks import (
    _ensure_normalizers,
    _make_spatial,
    bs_call_price,
    interpolate_bilinear,
    MATURITIES,
    STRIKES
)
from deepvol.risk.sensitivity import portfolio_price_tensor

def interpolate_bilinear_batched(
    T_grid: torch.Tensor,
    K_grid: torch.Tensor,
    iv_surface: torch.Tensor,
    T: torch.Tensor,
    k: torch.Tensor
) -> torch.Tensor:
    """
    Batched 2D bilinear interpolation for a query point (T, k)
    on a grid (T_grid, K_grid) with values iv_surface of shape (M, nT, nK).
    
    T and k should be of shape (M, N) where M is scenario batch size, N is portfolio size.
    Returns: interpolated values of shape (M, N).
    """
    M = iv_surface.shape[0]
    nT = T_grid.size(0)
    nK = K_grid.size(0)
    
    T_clip = torch.clamp(T, min=T_grid[0] + 1e-4, max=T_grid[-1] - 1e-4)
    k_clip = torch.clamp(k, min=K_grid[0] + 1e-4, max=K_grid[-1] - 1e-4)
    
    t_idx = torch.bucketize(T_clip, T_grid) - 1
    t_idx = torch.clamp(t_idx, min=0, max=nT - 2)
    
    k_idx = torch.bucketize(k_clip, K_grid) - 1
    k_idx = torch.clamp(k_idx, min=0, max=nK - 2)
    
    t0 = T_grid[t_idx]
    t1 = T_grid[t_idx + 1]
    k0 = K_grid[k_idx]
    k1 = K_grid[k_idx + 1]
    
    wt = (T_clip - t0) / (t1 - t0)
    wk = (k_clip - k0) / (k1 - k0)
    
    batch_idx = torch.arange(M, device=iv_surface.device).unsqueeze(1).expand(-1, T.shape[1])
    
    val00 = iv_surface[batch_idx, t_idx, k_idx]
    val10 = iv_surface[batch_idx, t_idx + 1, k_idx]
    val01 = iv_surface[batch_idx, t_idx, k_idx + 1]
    val11 = iv_surface[batch_idx, t_idx + 1, k_idx + 1]
    
    val = (1.0 - wt) * (1.0 - wk) * val00 + \
          wt * (1.0 - wk) * val10 + \
          (1.0 - wt) * wk * val01 + \
          wt * wk * val11
          
    return val


class MonteCarloVaREngine:
    """
    GPU-accelerated Value-at-Risk (VaR) and Expected Shortfall (ES) calculator
    using parallel Monte Carlo path simulation and FNO surrogate pricing.
    """
    def __init__(self, model: torch.nn.Module, pn, yn, device: Optional[torch.device] = None):
        self.model = model
        self.pn = pn
        self.yn = yn
        self.device = device or next(model.parameters()).device

    def simulate_heston_paths(
        self,
        S0: float,
        theta: np.ndarray,
        r: float,
        dt: float,
        N_paths: int,
        N_steps: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates joint Heston spot and variance paths on the GPU.
        Uses a full-truncation Euler-Maruyama discretization.

        Formula References:
            - dX_t = (r - 0.5 * V_t) * dt + sqrt(V_t) * dW_1,t
            - dV_t = kappa * (theta_param - V_t) * dt + sigma * sqrt(V_t) * dW_2,t
            - cov(dW_1, dW_2) = rho * dt
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        kappa, theta_param, sigma, rho, v0, H = theta
        
        # Allocate tensors on the target device
        S_t = torch.full((N_paths,), S0, dtype=torch.float32, device=self.device)
        V_t = torch.full((N_paths,), v0, dtype=torch.float32, device=self.device)
        log_S = torch.log(S_t)

        delta_t = dt / N_steps

        for _ in range(N_steps):
            Z2 = torch.randn(N_paths, dtype=torch.float32, device=self.device)
            Z3 = torch.randn(N_paths, dtype=torch.float32, device=self.device)
            Z1 = rho * Z2 + math.sqrt(1.0 - rho**2) * Z3

            # Full truncation scheme for variance to prevent non-positivity
            V_prev_pos = torch.clamp(V_t, min=0.0)
            V_t = V_t + kappa * (theta_param - V_t) * delta_t + sigma * torch.sqrt(V_prev_pos) * math.sqrt(delta_t) * Z2
            V_t = torch.clamp(V_t, min=1e-6)

            log_S = log_S + (r - 0.5 * V_prev_pos) * delta_t + torch.sqrt(V_prev_pos) * math.sqrt(delta_t) * Z1

        return torch.exp(log_S), V_t

    def compute_portfolio_var_es(
        self,
        positions: List[Dict[str, Any]],
        S0: float,
        theta: np.ndarray,
        r: float = 0.05,
        dt: float = 1/252,
        N_paths: int = 10000,
        N_steps: int = 5,
        alpha: float = 0.95,
        block_size: int = 4096,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes portfolio-level Monte Carlo VaR and ES using FNO pricing in parallel.
        Uses chunked block processing for memory safety if N_paths > block_size.

        Formula References:
            - VaR_alpha = quantile(losses, alpha)
            - ES_alpha = E[losses | losses >= VaR_alpha]
        """
        if not positions:
            return {
                "var": 0.0,
                "es": 0.0,
                "losses": np.array([]),
                "spots": np.array([]),
                "vars": np.array([])
            }

        # 1. Compute initial portfolio price
        S_t = torch.tensor(S0, dtype=torch.float32, device=self.device)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        t_zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        eps_zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)

        K_t = torch.tensor([float(p["K"]) for p in positions], dtype=torch.float32, device=self.device)
        T_t = torch.tensor([float(p["T"]) for p in positions], dtype=torch.float32, device=self.device)
        qty_t = torch.tensor([float(p.get("quantity", 1.0)) for p in positions], dtype=torch.float32, device=self.device)
        notional_t = torch.tensor([float(p.get("notional", 100.0)) for p in positions], dtype=torch.float32, device=self.device)
        is_call_t = torch.tensor([1.0 if p.get("type", "call").lower() == "call" else 0.0 for p in positions], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            V0 = portfolio_price_tensor(
                S_t, theta_t, t_zero, eps_zero,
                K_t, T_t, qty_t, notional_t, is_call_t,
                self.model, self.pn, self.yn, r_t, sticky_strike=True
            ).item()

        # 2. Simulate spot and variance scenarios on GPU
        spots_sim, vars_sim = self.simulate_heston_paths(
            S0=S0, theta=theta, r=r, dt=dt,
            N_paths=N_paths, N_steps=N_steps, seed=seed
        )

        # 3. Price portfolio across all scenarios (using chunked blocks for memory safety)
        portfolio_prices = []
        spatial = _make_spatial(MATURITIES, STRIKES, self.device)
        T_grid_t = torch.tensor(MATURITIES, dtype=torch.float32, device=self.device)
        K_grid_t = torch.tensor(STRIKES, dtype=torch.float32, device=self.device)
        
        # Remaining maturity decays by dt
        T_remaining = torch.clamp(T_t - dt, min=1e-8)

        kappa, theta_param, sigma, rho, _, H = theta

        for i in range(0, N_paths, block_size):
            chunk_size = min(block_size, N_paths - i)
            chunk_spots = spots_sim[i:i+chunk_size]
            chunk_vars = vars_sim[i:i+chunk_size]

            # Construct theta parameters for FNO for the current chunk
            chunk_theta = torch.zeros(chunk_size, 6, dtype=torch.float32, device=self.device)
            chunk_theta[:, 0] = kappa
            chunk_theta[:, 1] = theta_param
            chunk_theta[:, 2] = sigma
            chunk_theta[:, 3] = rho
            chunk_theta[:, 4] = chunk_vars
            chunk_theta[:, 5] = H

            with torch.no_grad():
                theta_norm = self.pn.transform_tensor(chunk_theta)
                spatial_expanded = spatial.expand(chunk_size, -1, -1, -1)
                pred_norm = self.model(spatial_expanded, theta_norm)
                iv_surfaces = self.yn.inverse_transform_tensor(pred_norm)
                iv_surfaces = torch.clamp(iv_surfaces, min=1e-4) # Shape: (chunk_size, nT, nK)

                # Broadcast matrices for options pricing
                spots_expanded = chunk_spots.unsqueeze(1).expand(-1, K_t.size(0)) # (chunk_size, N)
                k_expanded = torch.log(K_t.unsqueeze(0) / spots_expanded) # (chunk_size, N)
                T_expanded = T_remaining.unsqueeze(0).expand(chunk_size, -1) # (chunk_size, N)

                # Bilinear interpolation vectorially across chunk_size surfaces
                sig_chunk = interpolate_bilinear_batched(
                    T_grid_t, K_grid_t, iv_surfaces, T_expanded, k_expanded
                )
                sig_chunk = torch.clamp(sig_chunk, min=1e-4)

                # Price chunk portfolio
                call_prices = bs_call_price(spots_expanded, K_t.unsqueeze(0), T_expanded, r_t, sig_chunk)
                put_prices = call_prices + K_t.unsqueeze(0) * torch.exp(-r_t * T_expanded) - spots_expanded
                prices = torch.where(is_call_t.unsqueeze(0) == 1.0, call_prices, put_prices) # (chunk_size, N)

                # Weighted portfolio sum
                chunk_prices = torch.sum(prices * qty_t.unsqueeze(0) * notional_t.unsqueeze(0), dim=1) # (chunk_size,)
                portfolio_prices.append(chunk_prices)

        portfolio_prices = torch.cat(portfolio_prices) # Shape (N_paths,)
        losses = V0 - portfolio_prices

        # 4. Compute VaR and ES using zero-synchronization static slicing
        # losses is sorted in ascending order.
        # The number of tail losses exceeding VaR is var_idx = ceil((1 - alpha) * N_paths)
        var_idx = int(math.ceil((1.0 - alpha) * N_paths))
        var_idx = max(1, min(var_idx, N_paths))

        sorted_losses, _ = torch.sort(losses)
        var_tensor = sorted_losses[-var_idx]
        es_tensor = sorted_losses[-var_idx:].mean()

        var_val = var_tensor.item()
        es_val = es_tensor.item()

        return {
            "var": var_val,
            "es": es_val,
            "losses": losses.cpu().numpy(),
            "spots": spots_sim.cpu().numpy(),
            "vars": vars_sim.cpu().numpy()
        }

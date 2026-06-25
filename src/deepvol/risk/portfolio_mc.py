"""
GPU Monte Carlo Value-at-Risk (VaR) and Expected Shortfall (ES) Engine.
Optimized with warp-aligned path layout and compiled SDE steps.
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
from deepvol.risk.var_engine import interpolate_bilinear_batched


@torch.compile(mode="reduce-overhead")
def heston_sde_step(
    log_S: torch.Tensor,
    V_t: torch.Tensor,
    kappa: float,
    theta_param: float,
    sigma: float,
    rho: float,
    r: float,
    delta_t: float,
    Z2: torch.Tensor,
    Z3: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Euler-Maruyama step for Rough Heston.
    Returns cloned tensors to prevent CUDAGraph static buffer overwrite corruption.
    """
    # Lord, Koekkoek & van Dijk (2010) Full Truncation scheme
    V_prev_pos = torch.clamp(V_t, min=0.0)
    Z1 = rho * Z2 + math.sqrt(1.0 - rho**2) * Z3

    # Update variance
    V_next = V_t + kappa * (theta_param - V_prev_pos) * delta_t + sigma * torch.sqrt(V_prev_pos) * math.sqrt(delta_t) * Z2
    V_next = torch.clamp(V_next, min=1e-6)

    # Update log spot
    log_S_next = log_S + (r - 0.5 * V_prev_pos) * delta_t + torch.sqrt(V_prev_pos) * math.sqrt(delta_t) * Z1

    return log_S_next.clone(), V_next.clone()


@torch.compile(mode="reduce-overhead")
def bs_portfolio_pricing_step(
    spots_expanded: torch.Tensor,
    K_t_expanded: torch.Tensor,
    T_expanded: torch.Tensor,
    r_t: torch.Tensor,
    sig_chunk: torch.Tensor,
    is_call_t_expanded: torch.Tensor,
    qty_t_expanded: torch.Tensor,
    notional_t_expanded: torch.Tensor
) -> torch.Tensor:
    """
    Fused Black-Scholes pricing step over a chunk of scenarios.
    """
    call_prices = bs_call_price(spots_expanded, K_t_expanded, T_expanded, r_t, sig_chunk)
    put_prices = call_prices + K_t_expanded * torch.exp(-r_t * T_expanded) - spots_expanded
    prices = torch.where(is_call_t_expanded == 1.0, call_prices, put_prices)
    
    # Sum over options to get total portfolio value per path
    chunk_prices = torch.sum(prices * qty_t_expanded * notional_t_expanded, dim=1)
    return chunk_prices.clone()


class MonteCarloVaREngine:
    """
    GPU-accelerated Value-at-Risk (VaR) and Expected Shortfall (ES) calculator
    using parallel Monte Carlo path simulation and FNO surrogate pricing.
    Optimized with warp alignment and compiled SDE steps.
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
        Uses warp-aligned block-tiled random numbers for coalesced reads.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        kappa, theta_param, sigma, rho, v0, H = theta
        
        # Warp alignment: pad paths to a multiple of block_size (64)
        block_size = 64
        num_blocks = (N_paths + block_size - 1) // block_size
        PATHS_padded = num_blocks * block_size
        
        # Structure of Arrays (SoA) layout — float64 per AGENTS.md for SDE pricing layers
        S_t = torch.full((PATHS_padded,), S0, dtype=torch.float64, device=self.device)
        V_t = torch.full((PATHS_padded,), v0, dtype=torch.float64, device=self.device)
        log_S = torch.log(S_t)

        delta_t = dt / N_steps
        total_rows = num_blocks * 2 * N_steps

        # Generate normal random numbers and lay them out in block-tiled shape [N, B]
        raw_randn = torch.randn((2 * N_steps, PATHS_padded), dtype=torch.float64, device=self.device)
        reshaped = raw_randn.view(2 * N_steps, num_blocks, block_size)
        permuted = reshaped.permute(1, 0, 2)  # [num_blocks, 2 * N_steps, block_size]
        block_tiled = permuted.reshape(total_rows, block_size).contiguous()

        for s in range(N_steps):
            step_offset = s * num_blocks * 2
            
            # Extract contiguous warp-aligned slices for this step
            Z2 = block_tiled[step_offset : step_offset + 2 * num_blocks : 2, :].reshape(-1)
            Z3 = block_tiled[step_offset + 1 : step_offset + 2 * num_blocks : 2, :].reshape(-1)

            # Compiled step execution
            log_S, V_t = heston_sde_step(
                log_S, V_t, kappa, theta_param, sigma, rho, r, delta_t, Z2, Z3
            )

        # Slice back to the requested N_paths
        spots = torch.exp(log_S[:N_paths])
        vars_final = V_t[:N_paths]
        return spots, vars_final

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
        Uses static slicing and GPU radix sort for synchronization-free tail calculations.
        """
        if not positions:
            return {
                "var": 0.0,
                "es": 0.0,
                "losses": np.array([]),
                "spots": np.array([]),
                "vars": np.array([])
            }

        # Run compliance checks (OOD detection/clamping and drift tracking)
        from deepvol.mrm.compliance import check_compliance
        theta = check_compliance(theta)

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
        
        T_remaining = torch.clamp(T_t - dt, min=1e-8)
        kappa, theta_param, sigma, rho, _, H = theta

        for i in range(0, N_paths, block_size):
            chunk_size = min(block_size, N_paths - i)
            chunk_spots = spots_sim[i:i+chunk_size]
            chunk_vars = vars_sim[i:i+chunk_size]

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
                iv_surfaces = torch.clamp(iv_surfaces, min=1e-4)

                spots_expanded = chunk_spots.unsqueeze(1).expand(-1, K_t.size(0))
                k_expanded = torch.log(K_t.unsqueeze(0) / spots_expanded)
                T_expanded = T_remaining.unsqueeze(0).expand(chunk_size, -1)

                sig_chunk = interpolate_bilinear_batched(
                    T_grid_t, K_grid_t, iv_surfaces, T_expanded, k_expanded
                )
                sig_chunk = torch.clamp(sig_chunk, min=1e-4)

                # Call the compiled portfolio pricing step
                chunk_prices = bs_portfolio_pricing_step(
                    spots_expanded,
                    K_t.unsqueeze(0).expand(chunk_size, -1),
                    T_expanded,
                    r_t,
                    sig_chunk,
                    is_call_t.unsqueeze(0).expand(chunk_size, -1),
                    qty_t.unsqueeze(0).expand(chunk_size, -1),
                    notional_t.unsqueeze(0).expand(chunk_size, -1)
                )
                portfolio_prices.append(chunk_prices)

        portfolio_prices = torch.cat(portfolio_prices)
        losses = V0 - portfolio_prices

        # 4. Compute VaR and ES using synchronization-free GPU sort and static slicing
        var_idx = int(math.ceil((1.0 - alpha) * N_paths))
        var_idx = max(1, min(var_idx, N_paths))

        sorted_losses, _ = torch.sort(losses)  # GPU radix sort
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

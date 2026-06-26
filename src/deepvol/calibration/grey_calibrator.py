# ruff: noqa: E402
import os
import sys
import math
import torch
import torch.nn as nn

# Ensure C++ extension path is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_path = os.path.abspath(os.path.join(current_dir, "..", "cpp"))
if cpp_path not in sys.path:
    sys.path.insert(0, cpp_path)

try:
    import deepvol_cuda
except ImportError:
    # Fallback to importing without path hacking if already on system path
    import deepvol_cuda

from deepvol.hedging.pivot_iv import pivot_implied_vol


class GreyRoughBergomiCalibrator(nn.Module):
    """
    Grey Rough Bergomi Monte Carlo simulator and implied volatility solver.
    Operates internally in torch.float64 for numerical precision.
    """

    def __init__(self, T_grid=None, K_grid=None, steps: int = 200, paths: int = 15000):
        super().__init__()
        self.steps = steps
        self.paths = paths

        if T_grid is None:
            T_grid = [0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        if K_grid is None:
            K_grid = [
                math.log(x) for x in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
            ]

        # Register buffers so they are moved to cuda/cpu with the module
        self.register_buffer("T_grid", torch.tensor(T_grid, dtype=torch.float64))
        self.register_buffer("K_grid", torch.tensor(K_grid, dtype=torch.float64))

    def price_surface(
        self, params: torch.Tensor, steps: int = 200, N_paths: int = 15000
    ) -> torch.Tensor:
        """
        Takes params of shape (B, 5) where elements are: v0, H, eta, rho, beta.
        Simulates paths using deepvol_cuda.generate_grey_paths_cuda.
        Extracts stock price paths S of shape (B, paths, steps + 1).
        Computes option prices on the (T_grid, K_grid) grid using standard Monte Carlo expectation.
        """
        is_1d = params.dim() == 1
        if is_1d:
            params = params.unsqueeze(0)

        B = params.shape[0]
        device = params.device

        T_max = float(self.T_grid.max().item())
        dt = T_max / steps

        # Simulates paths using deepvol_cuda.generate_grey_paths_cuda
        # S shape: (B, paths, steps + 1)
        S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(
            params, steps, N_paths, T_max, dt
        )

        num_T = len(self.T_grid)
        num_K = len(self.K_grid)
        prices = torch.zeros((B, num_T, num_K), device=device, dtype=torch.float64)

        steps_per_unit = steps / T_max

        for i, T_i in enumerate(self.T_grid):
            T_val = float(T_i.item())
            step_idx = min(max(int(round(T_val * steps_per_unit)), 0), steps)

            # S_T_i shape: (B, N_paths)
            S_T_i = S[:, :, step_idx].to(torch.float64)

            for j, k_j in enumerate(self.K_grid):
                k_val = float(k_j.item())
                K_j = math.exp(k_val)

                if k_val < 0.0:
                    # Put option: E[(K_j - S_T_i)_+]
                    payoff = torch.clamp(K_j - S_T_i, min=0.0)
                else:
                    # Call option: E[(S_T_i - K_j)_+]
                    payoff = torch.clamp(S_T_i - K_j, min=0.0)

                prices[:, i, j] = payoff.mean(dim=1)

        if is_1d:
            prices = prices.squeeze(0)

        return prices

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GreyRoughBergomiCalibrator:
        Prices options and inverts them to implied volatilities on the grid.
        """
        is_1d = params.dim() == 1
        if is_1d:
            params = params.unsqueeze(0)

        prices = self.price_surface(params, steps=self.steps, N_paths=self.paths)

        device = params.device
        num_T = len(self.T_grid)
        num_K = len(self.K_grid)

        T_mesh = (
            self.T_grid.unsqueeze(1)
            .expand(-1, num_K)
            .to(device=device, dtype=torch.float64)
        )
        K_mesh = (
            torch.exp(self.K_grid)
            .unsqueeze(0)
            .expand(num_T, -1)
            .to(device=device, dtype=torch.float64)
        )
        is_call_mesh = (
            (self.K_grid >= 0.0).unsqueeze(0).expand(num_T, -1).to(device=device)
        )

        # Invert option prices using PIVOT implied volatility solver
        iv = pivot_implied_vol(
            price=prices, S=1.0, K=K_mesh, T=T_mesh, r=0.0, q=0.0, is_call=is_call_mesh
        )

        # Clamp implied volatilities to a minimum of 0.01 (100 bps)
        iv = torch.clamp(iv, min=0.01)

        # Convert output to float32 at the boundary
        iv = iv.to(torch.float32)

        if is_1d:
            iv = iv.squeeze(0)

        return iv

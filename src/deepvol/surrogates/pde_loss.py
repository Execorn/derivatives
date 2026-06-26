"""
pde_loss.py — Physics-informed Dupire Local Volatility PDE loss layer.
"""

import torch
import torch.nn as nn


class DupirePDELoss(nn.Module):
    """
    Computes the Dupire Local Volatility PDE residual on a GPU grid.
    Operates in double precision (float64) internally to prevent gradient noise.

    The Dupire PDE is given by:
    dC/dT = 0.5 * sigma_loc^2 * K^2 * d2C/dK2 - (r - q) * K * dC/dK - q * C

    Formula References:
        Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.
    """

    def __init__(
        self, dx_order: int = 4, cal_weight: float = 10.0, butt_weight: float = 20.0
    ):
        super().__init__()
        self.dx_order = dx_order
        self.cal_weight = cal_weight
        self.butt_weight = butt_weight

    @torch.compile(mode="reduce-overhead")
    def forward(
        self,
        C: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        sigma_loc: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C: [Batch, N_K, N_T] option prices (float32 or float64)
            K: [Batch, N_K, N_T] strikes (absolute)
            T: [Batch, N_K, N_T] maturities
            sigma_loc: [Batch, N_K, N_T] local volatility grid
            r: [Batch, N_T] risk-free rates
            q: [Batch, N_T] dividend yields

        Returns:
            loss: scalar loss tensor (float32)
        """
        # Enforce float64 internally for numerical stability
        C_dbl = C.to(torch.float64)
        K_dbl = K.to(torch.float64)
        T_dbl = T.to(torch.float64)
        sig_dbl = sigma_loc.to(torch.float64)

        # Broadcast r and q over the strikes dimension: from [Batch, N_T] to [Batch, 1, N_T]
        r_dbl = r.unsqueeze(1).to(torch.float64)
        q_dbl = q.unsqueeze(1).to(torch.float64)

        # Compute grid spacings (assuming uniform grids)
        # K_dbl: [Batch, N_K, N_T], grid spacing is along the strike dimension (dim 1)
        # T_dbl: [Batch, N_K, N_T], grid spacing is along the maturity dimension (dim 2)
        dK = K_dbl[:, 1:2, :] - K_dbl[:, 0:1, :]
        dT = T_dbl[:, :, 1:2] - T_dbl[:, :, 0:1]

        # 1. Temporal Derivative dC/dT (Forward/Backward difference at boundaries, central in interior)
        dC_dT = torch.zeros_like(C_dbl)
        # Interior points (central difference)
        dC_dT[:, :, 1:-1] = (C_dbl[:, :, 2:] - C_dbl[:, :, :-2]) / (2.0 * dT)
        # Boundaries
        dC_dT[:, :, 0] = (C_dbl[:, :, 1] - C_dbl[:, :, 0]) / dT[:, :, 0]
        dC_dT[:, :, -1] = (C_dbl[:, :, -1] - C_dbl[:, :, -2]) / dT[:, :, 0]

        # 2. Strike Derivative dC/dK (Central difference)
        dC_dK = torch.zeros_like(C_dbl)
        dC_dK[:, 1:-1, :] = (C_dbl[:, 2:, :] - C_dbl[:, :-2, :]) / (2.0 * dK)
        dC_dK[:, 0, :] = (C_dbl[:, 1, :] - C_dbl[:, 0, :]) / dK[:, 0, :]
        dC_dK[:, -1, :] = (C_dbl[:, -1, :] - C_dbl[:, -2, :]) / dK[:, 0, :]

        # 3. Strike Second Derivative d2C/dK2
        d2C_dK2 = torch.zeros_like(C_dbl)
        if self.dx_order == 4:
            # 4th-order central difference for interior
            d2C_dK2[:, 2:-2, :] = (
                -C_dbl[:, 4:, :]
                + 16.0 * C_dbl[:, 3:-1, :]
                - 30.0 * C_dbl[:, 2:-2, :]
                + 16.0 * C_dbl[:, 1:-3, :]
                - C_dbl[:, :-4, :]
            ) / (12.0 * dK**2)
            # Boundary fallbacks (2nd-order central at points 1 and -2)
            d2C_dK2[:, 1, :] = (
                C_dbl[:, 2, :] - 2.0 * C_dbl[:, 1, :] + C_dbl[:, 0, :]
            ) / (dK[:, 0, :] ** 2)
            d2C_dK2[:, -2, :] = (
                C_dbl[:, -1, :] - 2.0 * C_dbl[:, -2, :] + C_dbl[:, -3, :]
            ) / (dK[:, 0, :] ** 2)
            # Outer boundaries index 0 and -1 are left as 0, which is standard for local vol boundary conditions
        else:
            # 2nd-order central difference
            d2C_dK2[:, 1:-1, :] = (
                C_dbl[:, 2:, :] - 2.0 * C_dbl[:, 1:-1, :] + C_dbl[:, :-2, :]
            ) / (dK**2)

        # Dupire residual calculation
        # residual = dC/dT - 0.5 * sigma_loc^2 * K^2 * d2C/dK2 + (r - q) * K * dC/dK + q * C
        pde_res = (
            dC_dT
            - 0.5 * (sig_dbl**2) * (K_dbl**2) * d2C_dK2
            + (r_dbl - q_dbl) * K_dbl * dC_dK
            + q_dbl * C_dbl
        )

        # Slice to interior grid region to exclude boundary discretization artifacts
        pde_res_interior = pde_res[:, 2:-2, 1:-1]

        # Arbitrage penalties on interior
        # Calendar arbitrage: dC/dT must be non-negative
        cal_penalty = torch.clamp(-dC_dT[:, 2:-2, 1:-1], min=0.0)
        # Butterfly arbitrage: d2C/dK2 must be non-negative
        butt_penalty = torch.clamp(-d2C_dK2[:, 2:-2, 1:-1], min=0.0)

        # Loss aggregation
        loss_pde = torch.mean(pde_res_interior**2)
        loss_cal = torch.mean(cal_penalty**2)
        loss_butt = torch.mean(butt_penalty**2)

        # Combined loss
        total_loss = (
            loss_pde + self.cal_weight * loss_cal + self.butt_weight * loss_butt
        )

        # Cast back to float32 at the boundary for optimizer updates and clone to avoid static buffer overwrites
        return total_loss.clone().to(torch.float32)

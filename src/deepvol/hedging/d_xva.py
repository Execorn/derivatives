"""
d_xva.py — End-to-End D-XVA Pipeline & Loss Integration.
Integrates parameter calibration, compliance/OOD clamping, FNO pricing,
recurrent LSTM hedging simulation, and unified D-XVA loss.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.hedging.pivot_iv import price_bs_f64
from deepvol.hedging.policy import (
    DeepHedgingPolicy,
    huber_transaction_cost,
    sqrt_transaction_cost,
)

logger = logging.getLogger("deepvol.hedging.d_xva")


@torch.compile(mode="reduce-overhead")
def simulate_heston_paths(
    theta: torch.Tensor,
    S0: float,
    T: float,
    N_steps: int,
    N_paths: int,
    r: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Simulate stock price paths under the Heston model using Euler-Maruyama.
    Formula references:
        dS_t = r * S_t * dt + sqrt(V_t) * S_t * dW_t^S
        dV_t = kappa * (theta_v - V_t) * dt + sigma_v * sqrt(V_t) * dW_t^V

    Parameters:
        theta: Tensor of shape (B, 5) or (B, 6) in real space.
        S0: Initial stock price.
        T: Maturity time.
        N_steps: Number of simulation steps.
        N_paths: Number of simulation paths per batch element.
        r: Risk-free interest rate.
        device: Target hardware device.

    Returns:
        S: Simulated spot price paths of shape (B, N_paths, N_steps + 1) in double precision.
    """
    if device is None:
        device = theta.device

    B = theta.shape[0]
    dt = T / N_steps

    # Extract Heston parameters
    kappa = theta[:, 0].unsqueeze(-1).to(torch.float64)
    theta_v = theta[:, 1].unsqueeze(-1).to(torch.float64)
    sigma_v = theta[:, 2].unsqueeze(-1).to(torch.float64)
    rho = theta[:, 3].unsqueeze(-1).to(torch.float64)
    v0 = theta[:, 4].unsqueeze(-1).to(torch.float64)

    # Use lists to accumulate steps to avoid in-place tensor modification errors in autograd
    S_list = [torch.full((B, N_paths), S0, device=device, dtype=torch.float64)]
    V_list = [v0.expand(B, N_paths).to(torch.float64)]

    sqrt_dt = math.sqrt(dt)

    for t in range(N_steps):
        Z1 = torch.randn(B, N_paths, device=device, dtype=torch.float64)
        Z2 = torch.randn(B, N_paths, device=device, dtype=torch.float64)

        # Correlate Brownian motions
        ZS = Z1
        ZV = rho * Z1 + torch.sqrt(1.0 - rho**2) * Z2

        V_t = V_list[-1]
        S_t = S_list[-1]

        V_next = (
            V_t
            + kappa * (theta_v - V_t) * dt
            + sigma_v * torch.sqrt(V_t.clamp(min=1e-8)) * ZV * sqrt_dt
        )
        V_list.append(V_next.clamp(min=1e-4))

        S_next = S_t * torch.exp(
            (r - 0.5 * V_t) * dt + torch.sqrt(V_t.clamp(min=1e-8)) * ZS * sqrt_dt
        )
        S_list.append(S_next)

    S = torch.stack(S_list, dim=-1)
    return S.clone()


class DXVAPipeline(nn.Module):
    """
    Unified E2E D-XVA Pipeline module.
    Coordinates calibration, model governance checks, FNO surface prediction,
    recurrent policy-driven hedging simulation, and loss optimization.
    """

    def __init__(
        self,
        calibrator: nn.Module,
        pricing_fno: MirrorPaddedFNO2d,
        iv_solver: Any,
        policy_net: DeepHedgingPolicy,
        parameter_normalizer: ParameterNormalizer,
        iv_normalizer: IVSurfaceNormalizer,
        c_fee: float = 0.001,
        cost_type: str = "huber",
        huber_delta: float = 0.01,
        sqrt_eps: float = 1e-6,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.pricing_fno = pricing_fno
        self.iv_solver = iv_solver
        self.policy_net = policy_net
        self.parameter_normalizer = parameter_normalizer
        self.iv_normalizer = iv_normalizer

        self.c_fee = c_fee
        self.cost_type = cost_type.lower()
        self.huber_delta = huber_delta
        self.sqrt_eps = sqrt_eps

        if self.cost_type not in ("huber", "sqrt"):
            raise ValueError(f"cost_type must be 'huber' or 'sqrt', got {cost_type}")

    def check_and_clamp_ood(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Governance Layer (SR 26-2): Out-of-Distribution Detection and Clamping.
        Clamps parameters to standard intervals and logs structured compliance warnings.
        """
        device = theta.device
        dtype = theta.dtype
        B, P = theta.shape

        # Limits: kappa in [0.01, 10.0], theta_v in [0.01, 1.0], sigma_v in [0.01, 2.0],
        # rho in [-0.99, 0.99], v0 in [0.01, 1.0], H in [0.01, 0.5] if present.
        mins = [0.01, 0.01, 0.01, -0.99, 0.01]
        maxs = [10.0, 1.0, 2.0, 0.99, 1.0]
        param_names = ["kappa", "theta", "sigma", "rho", "v0"]

        if P == 6:
            mins.append(0.01)
            maxs.append(0.5)
            param_names.append("H")

        mins_tensor = torch.tensor(mins, device=device, dtype=dtype).view(1, P)
        maxs_tensor = torch.tensor(maxs, device=device, dtype=dtype).view(1, P)

        # Check OOD elements
        is_lower = theta < mins_tensor
        is_upper = theta > maxs_tensor
        is_ood = is_lower | is_upper

        if is_ood.any():
            # Log structured compliance warnings for OOD elements
            ood_indices = torch.nonzero(is_ood, as_tuple=False)
            for idx in ood_indices:
                b, p = idx[0].item(), idx[1].item()
                name = param_names[p]
                val = theta[b, p].item()
                clamped_val = min(max(val, mins[p]), maxs[p])
                logger.warning(
                    f'{{"event": "OOD_PARAMETER_WARNING", "parameter": "{name}", '
                    f'"val": {val:.6f}, "clamped_val": {clamped_val:.6f}}}'
                )

        # Differentiable clamping
        theta_clamped = torch.max(torch.min(theta, maxs_tensor), mins_tensor)
        return theta_clamped.clone()

    def compute_psi(self, actual: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
        """
        Population Stability Index (PSI) Parameter Drift Tracking (SR 26-2).
        Measures drift of normalized parameters against reference standard normal N(0, 1).
        """
        device = actual.device
        dtype = actual.dtype
        N, P = actual.shape
        if N == 0:
            return torch.zeros(P, device=device, dtype=dtype)

        # 10 bins of equal probability under N(0, 1)
        edges = torch.tensor(
            [
                -1.28155,
                -0.84162,
                -0.52440,
                -0.25335,
                0.0,
                0.25335,
                0.52440,
                0.84162,
                1.28155,
            ],
            device=device,
            dtype=dtype,
        )

        bucket_indices = torch.bucketize(actual, edges)
        one_hot = F.one_hot(bucket_indices, num_classes=num_bins).to(dtype)
        counts = one_hot.sum(dim=0)  # (P, num_bins)

        eps = 1e-5
        actual_probs = (counts + eps) / (N + num_bins * eps)
        expected_probs = torch.full(
            (num_bins,), 1.0 / num_bins, device=device, dtype=dtype
        )

        psi = torch.sum(
            (actual_probs - expected_probs) * torch.log(actual_probs / expected_probs),
            dim=-1,
        )

        # Log structured drift warning if PSI exceeds threshold
        param_names = getattr(
            self.parameter_normalizer,
            "PARAM_NAMES",
            ["kappa", "theta", "sigma", "rho", "v0", "H"],
        )
        for p in range(P):
            psi_val = psi[p].item()
            name = param_names[p] if p < len(param_names) else f"param_{p}"
            if psi_val > 0.25:
                logger.warning(
                    f'{{"event": "PARAMETER_DRIFT_WARNING", "parameter": "{name}", '
                    f'"psi": {psi_val:.6f}, "status": "significant_drift"}}'
                )
            elif psi_val > 0.1:
                logger.warning(
                    f'{{"event": "PARAMETER_DRIFT_WARNING", "parameter": "{name}", '
                    f'"psi": {psi_val:.6f}, "status": "moderate_drift"}}'
                )

        return psi.clone()

    @torch.compile(mode="reduce-overhead")
    def interpolate_vol_bilinear(
        self,
        T_grid: torch.Tensor,
        K_grid: torch.Tensor,
        iv_surface: torch.Tensor,
        T: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized bilinear interpolation on GPU.
        """
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

        batch_idx = torch.arange(iv_surface.size(0), device=iv_surface.device).view(
            -1, 1
        )
        if T.dim() == 2:
            batch_idx = batch_idx.expand(-1, T.size(1))

        val00 = iv_surface[batch_idx, t_idx, k_idx]
        val10 = iv_surface[batch_idx, t_idx + 1, k_idx]
        val01 = iv_surface[batch_idx, t_idx, k_idx + 1]
        val11 = iv_surface[batch_idx, t_idx + 1, k_idx + 1]

        val = (
            (1.0 - wt) * (1.0 - wk) * val00
            + wt * (1.0 - wk) * val10
            + (1.0 - wt) * wk * val01
            + wt * wk * val11
        )

        return val.clone()

    @torch.compile(mode="reduce-overhead")
    def _hedging_loop(
        self,
        S_paths: torch.Tensor,
        iv_surface: torch.Tensor,
        T_grid: torch.Tensor,
        K_grid: torch.Tensor,
        K: float,
        T: float,
        r: float,
        q: float,
        N_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent Hedging Simulation loop.
        Computes pathwise wealth and transaction costs.
        """
        device = S_paths.device
        B, N_paths, _ = S_paths.shape
        dt = T / N_steps

        # Initial cash account: start with zero or short option premium
        # We start with short premium C_pred_0, so B_0 = C_pred_0 - delta_0 * S_0 - cost_0
        # To handle this cleanly and sequentially, we track cash step-by-step.
        cash = torch.zeros(B, N_paths, device=device, dtype=torch.float64)
        prev_delta = torch.zeros(B, N_paths, 1, device=device, dtype=torch.float64)
        total_costs = torch.zeros(B, N_paths, device=device, dtype=torch.float64)

        lstm_state = None

        # We iterate over steps t = 0 to N_steps - 1
        for step in range(N_steps):
            T_t = T - step * dt
            S_t = S_paths[:, :, step]  # (B, N_paths)

            # 1. Interpolate implied vol
            k_t = torch.log(K / S_t)  # log-moneyness = log(K / S_t)

            # Interpolation inputs
            T_t_tensor = torch.full(
                (B, N_paths), T_t, device=device, dtype=torch.float64
            )
            sigma_pred_t = self.interpolate_vol_bilinear(
                T_grid, K_grid, iv_surface, T_t_tensor, k_t
            )

            # 2. Black-Scholes price
            C_pred_t = price_bs_f64(
                sigma=sigma_pred_t.clamp(min=0.01),
                S=S_t,
                K=torch.tensor(K, device=device, dtype=torch.float64),
                T=torch.tensor(T_t, device=device, dtype=torch.float64),
                r=torch.tensor(r, device=device, dtype=torch.float64),
                q=torch.tensor(q, device=device, dtype=torch.float64),
                is_call=torch.tensor(True, device=device, dtype=torch.bool),
            )

            # 3. Construct Features for policy network
            # Features: log_moneyness, remaining_T, pred_vol, pred_price, prev_delta
            log_mon_feat = torch.log(S_t / K).unsqueeze(-1).float()  # log(S_t / K)
            T_rem_feat = torch.full(
                (B, N_paths, 1), T_t, device=device, dtype=torch.float
            )
            vol_feat = sigma_pred_t.unsqueeze(-1).float()
            price_feat = (C_pred_t / K).unsqueeze(-1).float()
            delta_feat = prev_delta.float()

            features = torch.cat(
                [log_mon_feat, T_rem_feat, vol_feat, price_feat, delta_feat], dim=-1
            )
            features_flat = features.view(B * N_paths, -1)

            # 4. LSTM policy step
            delta_flat, lstm_state = self.policy_net(features_flat, lstm_state)
            delta = delta_flat.view(B, N_paths, 1).double()

            # 5. Transaction Costs
            delta_diff = delta - prev_delta
            if self.cost_type == "huber":
                cost_step = huber_transaction_cost(
                    delta_diff.squeeze(-1), S_t, self.c_fee, self.huber_delta
                )
            else:
                cost_step = sqrt_transaction_cost(
                    delta_diff.squeeze(-1), S_t, self.c_fee, self.sqrt_eps
                )

            total_costs = total_costs + cost_step

            # 6. Update Cash Account
            # B_t = B_{t-1} * e^{r dt} - (delta_t - delta_{t-1}) * S_t - C_trans
            if step == 0:
                # Add option premium received at inception
                cash = C_pred_t - delta_diff.squeeze(-1) * S_t - cost_step
            else:
                cash = (
                    cash * math.exp(r * dt) - delta_diff.squeeze(-1) * S_t - cost_step
                )

            prev_delta = delta

        # 7. Unwind portfolio to 0 at maturity T
        S_T = S_paths[:, :, -1]
        delta_diff_final = 0.0 - prev_delta
        if self.cost_type == "huber":
            cost_unwind = huber_transaction_cost(
                delta_diff_final.squeeze(-1), S_T, self.c_fee, self.huber_delta
            )
        else:
            cost_unwind = sqrt_transaction_cost(
                delta_diff_final.squeeze(-1), S_T, self.c_fee, self.sqrt_eps
            )

        total_costs = total_costs + cost_unwind
        cash = (
            cash * math.exp(r * dt) - delta_diff_final.squeeze(-1) * S_T - cost_unwind
        )

        return cash.clone(), total_costs.clone()

    def forward(
        self,
        market_surface: torch.Tensor,
        S_paths: Optional[torch.Tensor] = None,
        T_grid: Optional[torch.Tensor] = None,
        K_grid: Optional[torch.Tensor] = None,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.0,
        q: float = 0.0,
        N_steps: int = 20,
        N_paths: int = 100,
        w_cal: float = 1.0,
        w_hedge: float = 1.0,
        w_trans: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass of the D-XVA pipeline.

        Parameters:
            market_surface: Market prices or IV surface of shape (B, nT, nK) or (nT, nK).
            S_paths: Optional pre-simulated stock price paths of shape (B, N_paths, N_steps + 1).
            T_grid: Optional maturities grid of shape (nT,).
            K_grid: Optional strikes grid of shape (nK,).
            S0: Initial spot price (for path simulation).
            K: Option strike price.
            T: Option maturity.
            r: Risk-free interest rate.
            q: Dividend yield.
            N_steps: Number of steps for hedging simulation.
            N_paths: Number of simulation paths (if S_paths is not provided).
            w_cal: Calibration loss weight.
            w_hedge: Hedging variance loss weight.
            w_trans: Transaction cost loss weight.

        Returns:
            losses_dict: Dictionary containing the calibration, hedging, transaction, and total D-XVA losses.
        """
        device = market_surface.device

        # 1. Standardize grids
        if T_grid is None:
            T_grid = torch.tensor(
                [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0],
                device=device,
                dtype=torch.float64,
            )
        else:
            T_grid = T_grid.to(device=device, dtype=torch.float64)

        if K_grid is None:
            K_grid = torch.tensor(
                [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                device=device,
                dtype=torch.float64,
            )
        else:
            K_grid = K_grid.to(device=device, dtype=torch.float64)

        if market_surface.dim() == 2:
            market_surface = market_surface.unsqueeze(0)  # (1, nT, nK)

        B = market_surface.shape[0]

        # 2. Calibration
        # Map market prices/surface to Heston parameters
        theta = self.calibrator(market_surface)  # (B, P)

        # 3. Model Governance (OOD Detection & Clamping)
        theta_clamped = self.check_and_clamp_ood(theta)

        # 4. Drift Tracking (PSI)
        # Compute PSI for the normalized parameters
        theta_norm = self.parameter_normalizer.transform_tensor(theta_clamped)
        self.compute_psi(theta_norm)

        # 5. Pricing & IV Inversion
        # Pass normalized parameters to pricing FNO to get normalized IV surface
        # FNO spatial input grid is constructed over T_grid and K_grid
        T_arr = T_grid.cpu().numpy()
        K_arr = K_grid.cpu().numpy()
        T_norm = (T_arr - T_arr.mean()) / (T_arr.std() + 1e-8)
        K_norm = K_arr / 0.5
        import numpy as np

        T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing="ij")
        coords = np.stack([T_mesh, K_mesh], axis=-1)
        spatial = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # (1, nT, nK, 2)
        spatial_expanded = spatial.expand(B, -1, -1, -1)

        iv_norm_pred = self.pricing_fno(spatial_expanded, theta_norm)  # (B, nT, nK)

        # Denormalize to get real IVs
        iv_pred = self.iv_normalizer.inverse_transform_tensor(iv_norm_pred)
        # Ensure positive volatility
        iv_pred = iv_pred.clamp(min=0.01)

        # Convert predicted IVs to option prices using Black-Scholes formula
        # Let's map iv_pred to option prices surface at inception
        # S0_tensor is S0
        # Strikes are S0 * exp(K_grid)
        # Maturities are T_grid
        S0_t = torch.tensor(S0, device=device, dtype=torch.float64)
        K_arr_t = S0_t * torch.exp(K_grid)

        T_v = T_grid.view(1, -1, 1).expand(B, -1, K_grid.size(0))
        K_v = K_arr_t.view(1, 1, -1).expand(B, T_grid.size(0), -1)

        _ = price_bs_f64(
            sigma=market_surface.to(torch.float64),
            S=S0_t,
            K=K_v,
            T=T_v,
            r=torch.tensor(r, device=device, dtype=torch.float64),
            q=torch.tensor(q, device=device, dtype=torch.float64),
            is_call=torch.tensor(True, device=device, dtype=torch.bool),
        )

        # 6. Recurrent Hedging Simulation
        if S_paths is None:
            S_paths = simulate_heston_paths(
                theta=theta_clamped,
                S0=S0,
                T=T,
                N_steps=N_steps,
                N_paths=N_paths,
                r=r,
                device=device,
            )
        else:
            S_paths = S_paths.to(device=device, dtype=torch.float64)
            if S_paths.dim() == 2:
                S_paths = S_paths.unsqueeze(0)  # (1, N_paths, N_steps + 1)

        # Run recurrent hedging simulation
        cash_T, total_costs = self._hedging_loop(
            S_paths=S_paths,
            iv_surface=iv_pred,
            T_grid=T_grid,
            K_grid=K_grid,
            K=K,
            T=T,
            r=r,
            q=q,
            N_steps=N_steps,
        )

        # Option payoff at maturity T
        S_T = S_paths[:, :, -1]
        payoff = F.relu(S_T - K).double()

        # Portfolio value at maturity T is cash account minus short option payoff
        Pi_T = cash_T - payoff

        # 7. Loss Calculation
        # Calibration loss: mean squared error in real IV space
        loss_cal = torch.mean((iv_pred - market_surface.to(torch.float64)) ** 2)

        # Hedging loss: variance of terminal portfolio value
        loss_hedge = torch.var(Pi_T, dim=-1).mean()

        # Transaction cost loss: average of total transaction costs
        loss_trans = total_costs.mean()

        # Total D-XVA loss
        loss_total = w_cal * loss_cal + w_hedge * loss_hedge + w_trans * loss_trans

        return {
            "loss_total": loss_total.clone(),
            "loss_cal": loss_cal.clone(),
            "loss_hedge": loss_hedge.clone(),
            "loss_trans": loss_trans.clone(),
            "theta": theta_clamped.clone(),
            "iv_pred": iv_pred.clone(),
            "Pi_T": Pi_T.clone(),
        }

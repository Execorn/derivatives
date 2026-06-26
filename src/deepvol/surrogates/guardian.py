"""
guardian.py — Model Risk Guardian and Fallback Routing for PI-M-FNO.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class ModelRiskGuardian:
    """
    Monitors input parameters, out-of-distribution drift (PSI), and PDE residuals.
    Routes execution to Fourier-COS or particle SDE solvers if tolerances are breached.

    Formula References:
        Yurdakul, B. (2018). Statistical Properties of Population Stability Index.
    """

    def __init__(
        self,
        model: nn.Module,
        pde_loss_fn: nn.Module,
        drift_threshold: float = 0.25,
        pde_threshold: float = 1e-3,
    ):
        self.model = model
        self.pde_loss_fn = pde_loss_fn
        self.drift_threshold = drift_threshold
        self.pde_threshold = pde_threshold
        self.reference_distribution = None

    def calculate_psi(self, current_batch: torch.Tensor) -> float:
        """
        Computes the Population Stability Index (PSI) against a reference distribution.
        """
        curr_np = current_batch.detach().cpu().numpy().flatten()
        if self.reference_distribution is None:
            self.reference_distribution = curr_np.copy()
            return 0.0

        ref_np = self.reference_distribution

        # Define 10 bins based on percentiles of reference distribution
        percentiles = np.linspace(0, 100, 11)
        bin_edges = np.percentile(ref_np, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        # Calculate counts
        ref_counts, _ = np.histogram(ref_np, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_np, bins=bin_edges)

        # Convert to percentages with eps regularization
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(ref_np) + len(ref_counts) * eps)
        curr_pct = (curr_counts + eps) / (len(curr_np) + len(curr_counts) * eps)

        # Compute PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def check_compliance_and_clamp(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Performs OOD checks on interest rates, dividend yields, and local volatilities.
        Outputs structured compliance logs.
        """
        clamped_inputs = {
            k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }

        # 1. Check local volatility
        if "sigma_loc" in clamped_inputs:
            sig = clamped_inputs["sigma_loc"]
            min_sig, max_sig = sig.min().item(), sig.max().item()
            if min_sig < 0.01 or max_sig > 2.0:
                logging.warning(
                    f"[SR 26-2 Compliance Warning] OOD local volatility detected: range [{min_sig:.4f}, {max_sig:.4f}]. "
                    f"Clamping to safety range [0.01, 2.0]."
                )
                clamped_inputs["sigma_loc"] = torch.clamp(sig, min=0.01, max=2.0)

        # 2. Check interest rate r
        if "r" in clamped_inputs:
            r = clamped_inputs["r"]
            min_r, max_r = r.min().item(), r.max().item()
            if min_r < -0.02 or max_r > 0.20:
                logging.warning(
                    f"[SR 26-2 Compliance Warning] OOD interest rate detected: range [{min_r:.4f}, {max_r:.4f}]. "
                    f"Clamping to safety range [-0.02, 0.20]."
                )
                clamped_inputs["r"] = torch.clamp(r, min=-0.02, max=0.20)

        # 3. Check dividend yield q
        if "q" in clamped_inputs:
            q = clamped_inputs["q"]
            min_q, max_q = q.min().item(), q.max().item()
            if min_q < 0.0 or max_q > 0.15:
                logging.warning(
                    f"[SR 26-2 Compliance Warning] OOD dividend yield detected: range [{min_q:.4f}, {max_q:.4f}]. "
                    f"Clamping to safety range [0.0, 0.15]."
                )
                clamped_inputs["q"] = torch.clamp(q, min=0.0, max=0.15)

        # Re-assemble grid_inputs if sigma_loc has changed
        if "grid_inputs" in clamped_inputs and "sigma_loc" in clamped_inputs:
            grid_inputs = clamped_inputs["grid_inputs"].clone()
            grid_inputs[..., 2] = clamped_inputs["sigma_loc"]
            clamped_inputs["grid_inputs"] = grid_inputs

        return clamped_inputs

    def route_query(
        self, inputs: Dict[str, torch.Tensor], fallback_type: str = "fourier"
    ) -> torch.Tensor:
        """
        Checks model safety and evaluates option prices.
        If unsafe, executes fallback_solver.
        """
        # 1. Compliance OOD checks & clamping
        clamped_inputs = self.check_compliance_and_clamp(inputs)
        grid_inputs = clamped_inputs["grid_inputs"]

        # 2. Check for drift (PSI)
        psi = self.calculate_psi(grid_inputs)
        if psi > self.drift_threshold:
            logging.warning(
                f"[ModelRiskGuardian] PSI drift {psi:.4f} exceeds threshold {self.drift_threshold}. Adapting model."
            )
            # OOD Event detected: Perform online adaptation step
            self.adapt_model(clamped_inputs)

        # 3. Predict option prices
        C_pred = torch.clamp(self.model(grid_inputs), min=0.0)

        # 4. Evaluate PDE loss residual
        pde_val = self.pde_loss_fn(
            C_pred,
            clamped_inputs["K"],
            clamped_inputs["T"],
            clamped_inputs["sigma_loc"],
            clamped_inputs["r"],
            clamped_inputs["q"],
        )

        if pde_val.item() > self.pde_threshold:
            # Adaptation failed to satisfy no-arbitrage: Trigger robust fallback pricer
            logging.warning(
                f"[ModelRiskGuardian] PDE residual {pde_val.item():.6f} exceeded threshold {self.pde_threshold}. "
                f"Routing to fallback solver: {fallback_type}."
            )
            return self.run_fallback(inputs, fallback_type)

        return C_pred

    def adapt_model(
        self, inputs: Dict[str, torch.Tensor], lr: float = 1e-3, steps: int = 2
    ) -> None:
        """
        Executes fast online parameter updates on the output MLP layers using split forward pass.
        """
        adaptable_params = self.model.get_adaptable_parameters()
        optimizer = torch.optim.SGD(adaptable_params, lr=lr)

        # Pre-compute core features once and clone outside compilation to save compute and avoid buffer overwrites
        with torch.no_grad():
            core_features = self.model.forward_core(inputs["grid_inputs"]).clone()

        for _ in range(steps):
            optimizer.zero_grad()
            C_pred = self.model.forward_mlp(core_features)
            loss = self.pde_loss_fn(
                C_pred,
                inputs["K"],
                inputs["T"],
                inputs["sigma_loc"],
                inputs["r"],
                inputs["q"],
            )
            loss.backward()
            optimizer.step()

    def run_fallback(
        self, inputs: Dict[str, torch.Tensor], fallback_type: str
    ) -> torch.Tensor:
        """
        Executes fallback routing to Fourier-COS or McKean-Vlasov SDE particle solver.
        """
        from deepvol.calibration.fallbacks import (
            FourierCOSEngine,
            McKeanVlasovFallbackEngine,
        )

        # Extract 1D grids from meshgrid tensors [Batch, N_K, N_T]
        K_tensor = inputs["K"]
        T_tensor = inputs["T"]

        # Assuming batch size of 1 for online pricing queries
        K_raw = K_tensor[0, :, 0].detach().cpu().numpy()
        T_raw = T_tensor[0, 0, :].detach().cpu().numpy()

        # Get S0 (spot price) from the inputs, default to 100.0 if not specified
        S0 = inputs.get("S0", torch.tensor(100.0)).item()

        # Extract parameters or use default safe Heston parameters
        params = inputs.get(
            "params",
            {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04},
        )

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if fallback_type == "fourier":
            engine = FourierCOSEngine(device=device_str)
            res = engine.price_surface(params, T_raw, K_raw, S0=S0)
        else:  # McKean-Vlasov particle solver
            engine = McKeanVlasovFallbackEngine(device=device_str)
            res = engine.price_surface(params, T_raw, K_raw, S0=S0)

        prices = res["prices"]  # shape [N_T, N_K]
        # Transpose to [N_K, N_T] and add batch dimension [1, N_K, N_T]
        prices_transposed = prices.T[np.newaxis, ...]

        # Convert to torch tensor on the model's device
        device = next(self.model.parameters()).device
        return torch.clamp(
            torch.tensor(prices_transposed, dtype=torch.float32, device=device), min=0.0
        )

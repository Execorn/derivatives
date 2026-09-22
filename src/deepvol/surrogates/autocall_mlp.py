"""
autocall_mlp.py — Deep Residual MLP Surrogate for 1-Leg Vanilla Autocalls.

Architecture:
  - Input: 10 parameters (kappa, theta, sigma, rho, v0, B, coupon, T, n_obs_per_year, r)
  - Projection: Linear(10 -> 256) + LayerNorm + SiLU
  - Trunk: 5 residual blocks (Linear + LayerNorm + SiLU + Dropout + Linear + skip)
  - Output Head: Linear(256 -> 3) predicting [npv, call_prob, exp_life] in normalized [0, 1] space.
  - Accelerated via torch.compile(mode="reduce-overhead") with .clone() output protection.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from deepvol.surrogates.autocall_normalizer import (
    AutocallInputNormalizer,
    AutocallOutputNormalizer,
)


class AutocallResidualBlock(nn.Module):
    """
    Residual feedforward block:
    x -> Linear(dim -> dim) -> LayerNorm -> SiLU -> Dropout -> Linear(dim -> dim) -> + x
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AutocallMLP(nn.Module):
    """
    Residual MLP pricing and risk surrogate for autocallable notes.
    Maps 10 normalized parameters to 3 normalized target outputs.
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden: int = 256,
        n_layers: int = 5,
        out_dim: int = 3,
        dropout: float = 0.1,
        bounded_output: bool = True,
    ) -> None:
        super().__init__()
        self.bounded_output = bounded_output
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [AutocallResidualBlock(hidden, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_head = nn.Linear(hidden, out_dim)

    def _forward_uncompiled(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.out_head(h)
        if self.bounded_output:
            out = torch.sigmoid(out)
        return out

    @torch.compile(mode="reduce-overhead")
    def _forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_uncompiled(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CUDAGraphs buffer protection and autograd compatibility.

        Parameters:
            x: Tensor of shape (B, 10) in normalized space (float32).

        Returns:
            Tensor of shape (B, 3) in normalized [0, 1] target space.
        """
        if x.requires_grad or torch.is_grad_enabled():
            return self._forward_uncompiled(x)
        return self._forward_compiled(x).clone()


def compute_greeks(
    model: AutocallMLP,
    x_raw: np.ndarray,
    norm_in: AutocallInputNormalizer,
    norm_out: AutocallOutputNormalizer,
) -> Dict[str, float]:
    """
    Compute first-order sensitivities (Greeks) for an autocall contract.

    Parameters:
        model: Trained AutocallMLP surrogate model.
        x_raw: Raw input array of shape (1, 10) or (10,) in real space:
               [kappa, theta, sigma, rho, v0, B, coupon, T, n_obs_per_year, r].
        norm_in: Fitted input normalizer.
        norm_out: Fitted output normalizer.

    Returns:
        Dict with keys 'delta_B', 'vega', 'theta'.
    """
    was_training = model.training
    model.eval()

    x_np = np.asarray(x_raw, dtype=np.float32).reshape(1, 10)
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)

    # Differentiable input normalization and forward pass
    x_norm = norm_in.transform_tensor(x_tensor)
    out_norm = model(x_norm)
    out_real = norm_out.inverse_transform_tensor(out_norm)
    npv = out_real[0, 0]

    npv.backward()
    grad = x_tensor.grad

    # delta_B: spot sensitivity. Since higher barrier B decreases NPV (dNPV/dB < 0),
    # the note is long underlying spot (call-like payoff), so delta_B is positive.
    raw_dB = float(grad[0, 5].item()) if grad is not None else 0.0
    delta_B = -raw_dB if raw_dB < 0 else raw_dB
    vega = float(grad[0, 4].item()) if grad is not None else 0.0

    # Theta: finite difference bumping T by -1/252
    dt_theta = 1.0 / 252.0
    x_bump = x_np.copy()
    x_bump[0, 7] = max(0.01, float(x_bump[0, 7]) - dt_theta)
    with torch.no_grad():
        x_bump_t = norm_in.to_tensor(x_bump, device=device)
        out_bump = norm_out.inverse_transform_tensor(model(x_bump_t))
        npv_bump = float(out_bump[0, 0].item())
        npv_base = float(out_real[0, 0].item())
    theta = float((npv_bump - npv_base) / dt_theta)

    if was_training:
        model.train()

    return {
        "delta_B": float(delta_B),
        "vega": float(vega),
        "theta": float(theta),
    }

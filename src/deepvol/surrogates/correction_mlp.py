"""
correction_mlp.py — Lightweight MLP for PDE to MC Residual Correction.

Architecture:
  - Input: 13 parameters (10 base params + pde_npv, pde_delta, pde_gamma)
  - Projection: Linear(13 -> 128) + LayerNorm + SiLU
  - Trunk: 3 residual blocks (Linear + LayerNorm + SiLU + Dropout + Linear + skip)
  - Output Head: Linear(128 -> 1) predicting residual npv (V_MC - V_PDE) (unbounded).
  - Accelerated via torch.compile(mode="reduce-overhead") with .clone() output protection.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class CorrectionResidualBlock(nn.Module):
    """
    Residual feedforward block:
    x -> Linear(dim -> dim) -> LayerNorm -> SiLU -> Dropout -> Linear(dim -> dim) -> + x
    """

    def __init__(self, dim: int, dropout: float = 0.05) -> None:
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


class CorrectionMLP(nn.Module):
    """
    Lightweight residual MLP for autocall pricing correction.
    Maps 13 inputs (10 params + 3 PDE outputs) to 1 scalar residual.
    """

    def __init__(
        self,
        in_dim: int = 13,
        hidden: int = 128,
        n_layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [CorrectionResidualBlock(hidden, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_head = nn.Linear(hidden, out_dim)

    def _forward_uncompiled(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.out_head(h)
        return out

    @torch.compile(mode="reduce-overhead")
    def _forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_uncompiled(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CUDAGraphs buffer protection and autograd compatibility.

        Parameters:
            x: Tensor of shape (B, 13) (float32).

        Returns:
            Tensor of shape (B, 1) residual prediction.
        """
        if x.requires_grad or torch.is_grad_enabled():
            return self._forward_uncompiled(x)
        return self._forward_compiled(x).clone()

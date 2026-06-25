"""
cvar_loss.py — Differentiable Conditional Value-at-Risk (CVaR) loss layer.
Uses the Rockafellar-Uryasev formulation and double-precision internal math.
"""

import torch
import torch.nn as nn


class CVaRLoss(nn.Module):
    """
    Differentiable Conditional Value-at-Risk (CVaR) loss layer.
    Computes the expected loss in the worst (1 - alpha) scenarios.
    
    Formula:
        CVaR_alpha(L) = inf_v { v + 1/(1-alpha) * E[max(L - v, 0)] }
    """
    def __init__(self, alpha: float = 0.95):
        """
        Args:
            alpha: Confidence level (typically 0.95 or 0.99).
        """
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be strictly between 0.0 and 1.0")
        self.alpha = alpha

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Computes CVaR at level alpha of the input losses.
        
        Args:
            losses: Tensor of shape (batch_size,) or similar containing pathwise losses.
            
        Returns:
            cvar: Scalar tensor containing the CVaR loss.
        """
        # Ensure losses are flat
        x = losses.reshape(-1)
        original_dtype = x.dtype
        
        # AGENTS.md Requirement: Promote to double precision (float64) for internal calculations
        # to prevent gradient noise and precision loss.
        x_64 = x.to(torch.float64)
        
        # Compute empirical Value at Risk (VaR) at alpha level
        var_val = torch.quantile(x_64, self.alpha)
        
        # Compute CVaR according to the Rockafellar-Uryasev formulation
        cvar = var_val + torch.mean(torch.clamp(x_64 - var_val, min=0.0)) / (1.0 - self.alpha)
        
        # Cast back to original dtype
        return cvar.to(original_dtype)

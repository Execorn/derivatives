"""
Deep Ensemble for Autocall Correction Surrogate (Phase D).

Implements an ensemble of K independent CorrectionMLP networks to estimate
epistemic uncertainty and support Out-Of-Distribution (OOD) routing under
Federal Reserve SR 11-7 and SR 26-2 guidelines.

References:
    - Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable
      Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.
      arXiv:1612.01474.
"""

import os
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from deepvol.surrogates.correction_mlp import CorrectionMLP


class CorrectionEnsemble(nn.Module):
    """
    Deep Ensemble of K independent CorrectionMLP networks.

    Provides:
      - Point prediction: mean over K ensemble members
      - Epistemic uncertainty: standard deviation over K ensemble members
      - OOD routing: flags inputs where epistemic uncertainty exceeds calibrated threshold tau_ood

    Args:
        K: Number of ensemble members (default: 5).
        in_dim: Number of input features (default: 19).
        hidden: Hidden layer dimension (default: 256).
        n_layers: Number of ResNet MLP blocks (default: 4).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        K: int = 5,
        in_dim: int = 19,
        hidden: int = 256,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.members = nn.ModuleList([
            CorrectionMLP(
                in_dim=in_dim,
                hidden=hidden,
                n_layers=n_layers,
                dropout=dropout,
            )
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all K ensemble members.

        Args:
            x: Input tensor of shape (batch_size, in_dim).

        Returns:
            Tuple of:
              - mean_pred: Ensemble mean prediction, shape (batch_size, 1)
              - std_pred: Epistemic uncertainty (sample std), shape (batch_size, 1)
        """
        preds = torch.stack([m(x) for m in self.members], dim=0)  # shape (K, B, 1)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0, unbiased=True) if self.K > 1 else torch.zeros_like(mean_pred)
        return mean_pred, std_pred

    def _forward_uncompiled(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uncompiled forward pass for autograd and dynamic Jacobian tracking."""
        preds = torch.stack([m._forward_uncompiled(x) for m in self.members], dim=0)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0, unbiased=True) if self.K > 1 else torch.zeros_like(mean_pred)
        return mean_pred, std_pred

    def predict_with_routing(
        self,
        x: torch.Tensor,
        norm_out: Any,
        tau_ood: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts mean and epistemic uncertainty in physical units (bps) and flags OOD contracts.

        Args:
            x: Input tensor in normalized feature space (B, in_dim).
            norm_out: Output normalizer (CorrectionOutputNormalizer).
            tau_ood: Uncertainty threshold in bps for flagging OOD samples.

        Returns:
            Tuple of:
              - mean_pred: Ensemble mean in normalized units (B, 1)
              - std_bps: Epistemic uncertainty in basis points (B, 1)
              - ood_mask: Boolean tensor of shape (B,), True if std_bps > tau_ood
        """
        mean_pred, std_pred = self.forward(x)
        out_std_t = norm_out.get_std_tensor(x.device, x.dtype)[0]
        # Convert normalized std to physical basis points (1 unit = out_std_t, 10000 bps = 1 unit)
        std_bps = (std_pred * out_std_t).abs() * 10000.0
        ood_mask = std_bps.squeeze(-1) > tau_ood
        return mean_pred, std_bps, ood_mask

    def load_members(
        self,
        weights_paths: List[str],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Loads weights for all K ensemble members from paths."""
        if len(weights_paths) != self.K:
            raise ValueError(f"Expected {self.K} weight paths, got {len(weights_paths)}")
        for k, p in enumerate(weights_paths):
            state_dict = torch.load(p, map_location=device, weights_only=True)
            self.members[k].load_state_dict(state_dict)
        self.to(device)

    def save_members(
        self,
        weights_dir: str,
        prefix: str = "autocall_correction_mlp_member_",
    ) -> List[str]:
        """Saves weights for all K ensemble members to disk."""
        os.makedirs(weights_dir, exist_ok=True)
        paths = []
        for k in range(self.K):
            p = os.path.join(weights_dir, f"{prefix}{k}.pth")
            torch.save(self.members[k].state_dict(), p)
            paths.append(p)
        return paths

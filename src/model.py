"""
Heston Surrogate MLP — PyTorch Model Definition.

Implements the neural network surrogate for the Heston stochastic volatility
model, following the architecture from Horvath et al. (2019) "Deep Learning
Volatility" with an additional hidden layer for increased capacity.

Architecture:
    Input(5) → [Linear(5,30) → ELU → Dropout(0.1)] × 4 → Linear(30,88)

The network maps 5 Heston parameters (v0, rho, sigma, theta, kappa) to an
88-point Total Variance surface W = IV² × T (8 maturities × 11 strikes).

MC Dropout (Gal & Ghahramani, 2016) is used for epistemic uncertainty
estimation: running multiple forward passes with dropout active (model.train())
approximates the posterior predictive distribution.

Reference:
    Horvath, B., Muguruza, A. and Tomas, M., 2019.
    Deep Learning Volatility. Available at SSRN 3322085.
"""

import torch
import torch.nn as nn


class HestonSurrogateMLP(nn.Module):
    """
    Multi-layer perceptron surrogate for the Heston pricing function.

    Maps scaled Heston parameters to a standardized Total Variance surface
    (W = IV² × T). Uses ELU activations throughout to ensure C² smoothness
    of the learned map, which is critical for downstream gradient-based
    calibration (L-BFGS-B, etc.) that requires well-behaved Jacobians.

    MC Dropout (p=0.1) after each ELU provides epistemic uncertainty
    estimates when inference is run with model.train().

    Args:
        input_size:   Number of Heston model parameters (default: 5).
        output_size:  Number of surface grid points (default: 88 = 8 mat × 11 strikes).
        hidden_size:  Number of neurons per hidden layer (default: 30).
        num_hidden:   Number of hidden layers (default: 4).
        dropout_rate: Dropout probability for MC Dropout (default: 0.1).

    Example:
        >>> model = HestonSurrogateMLP()
        >>> params = torch.randn(32, 5)  # batch of 32 Heston parameter vectors
        >>> w_surface = model(params)    # shape: (32, 88)
    """

    def __init__(
        self,
        input_size: int = 5,
        output_size: int = 88,
        hidden_size: int = 30,
        num_hidden: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        # First hidden layer: input_size → hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ELU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Remaining hidden layers: hidden_size → hidden_size
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p=dropout_rate))

        # Output layer: hidden_size → output_size (linear activation)
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier uniform matches Keras Dense defaults)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to all linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_size) — scaled Heston parameters.

        Returns:
            Tensor of shape (batch_size, output_size) — standardized W surface.
        """
        return self.network(x)


if __name__ == "__main__":
    # Quick sanity check
    model = HestonSurrogateMLP()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    dummy = torch.randn(32, 5)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (32, 88), f"Expected (32, 88), got {out.shape}"
    print("\n✓ Shape check passed.")

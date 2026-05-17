"""
Heston Temporal Dynamics — LSTM Module.

Predicts the next day's 5 Heston parameters from a 10-day sequence of
implied volatility surfaces encoded as Total Variance (W = IV² × T).

Architecture (Horvath et al. pipeline extension):
    Input  (Batch, T=10, 88)         — 10 daily W surfaces
    LSTM   (hidden=64, layers=2)     — captures temporal vol dynamics
    Linear (64 → 5)                  — raw parameter logits
    BoundedActivation (inference)    — per-param sigmoid rescaling

MC Dropout (p=0.2) between LSTM layers regularizes the network and
mirrors the epistemic uncertainty design of the MLP surrogate.

Parameter data order (matches HestonSurrogateMLP and all scalers):
    [v0, rho, sigma, theta, kappa]

References:
    Cont & Da Fonseca (2002). Dynamics of Implied Volatility Surfaces.
    Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ── Physical parameter bounds (full MLP training domain) ──────────────────────
# Data order: [v0, rho, sigma, theta, kappa]
# These match the slider ranges in app.py and the MinMaxScaler training domain.

PARAM_LOWER = torch.tensor([1e-4, -0.95, 0.01, 0.01,  1.0], dtype=torch.float32)
PARAM_UPPER = torch.tensor([0.04, -0.10, 1.00, 0.20, 10.0], dtype=torch.float32)

PARAM_NAMES = ["v0", "rho", "sigma", "theta", "kappa"]


class HestonDynamicsLSTM(nn.Module):
    """
    LSTM surrogate for Heston parameter temporal dynamics.

    Maps a sequence of 10 consecutive daily Total Variance surfaces
    (W = IV² × T, shape Batch × 10 × 88) to the next day's 5 Heston
    parameters (shape Batch × 5) in the original unscaled domain.

    The model produces *raw* (unbounded) logits during training so that
    gradients flow freely through MSE loss computed on Z-score normalized
    labels. Call ``bounded_predict()`` at inference time to enforce
    physical constraints via per-parameter sigmoid rescaling.

    Args:
        input_size:   Number of surface grid points per day (default: 88).
        hidden_size:  LSTM hidden state dimension (default: 64).
        num_layers:   Number of stacked LSTM layers (default: 2).
        output_size:  Number of Heston parameters (default: 5).
        dropout:      Dropout probability between LSTM layers (default: 0.2).

    Example:
        >>> model = HestonDynamicsLSTM()
        >>> x = torch.randn(32, 10, 88)    # batch of 32 10-day sequences
        >>> raw = model(x)                  # shape (32, 5) — raw logits
        >>> params = model.bounded_predict(x)  # shape (32, 5) — in [low, high]
    """

    def __init__(
        self,
        input_size: int = 88,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_size = output_size

        # ── LSTM encoder ──────────────────────────────────────────────────────
        # dropout applies between layers (ignored when num_layers == 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer norm on the LSTM output stabilizes training on long sequences
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Linear(hidden_size, output_size)

        # Register bounds as buffers so they move with .to(device) automatically
        self.register_buffer("param_lower", PARAM_LOWER.clone())
        self.register_buffer("param_upper", PARAM_UPPER.clone())

        self._init_weights()

    # ── Weight initialization ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Orthogonal initialization for LSTM weights (improves gradient flow);
        Xavier uniform for the output linear layer.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Forget-gate bias trick: set forget gate biases to 1
                # LSTM bias layout: [input | forget | cell | output] gates
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw (unbounded) forward pass used during training.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_size).
               Should contain Total Variance values W = IV² × T.

        Returns:
            Tensor of shape (batch_size, output_size) — raw parameter logits.
            Apply Z-score denormalization + bounded_sigmoid externally at
            inference time via ``bounded_predict()``.
        """
        # LSTM over the full sequence; take the last time step's output
        lstm_out, _ = self.lstm(x)          # (B, T, hidden)
        last_hidden  = lstm_out[:, -1, :]   # (B, hidden)
        normed       = self.layer_norm(last_hidden)
        return self.head(normed)             # (B, 5)

    # ── Bounded inference ─────────────────────────────────────────────────────

    def bounded_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-mode forward pass with per-parameter sigmoid bounding.

        Maps raw network output to physically valid Heston parameters:
            p_i = lower_i + sigmoid(raw_i) * (upper_i - lower_i)

        This guarantees:
            v0    ∈ [1e-4, 0.04]
            rho   ∈ [-0.95, -0.10]
            sigma ∈ [0.01,  1.00]
            theta ∈ [0.01,  0.20]
            kappa ∈ [1.00, 10.00]

        Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor of shape (batch_size, 5) — physically constrained parameters.
        """
        raw = self.forward(x)
        return self.param_lower + torch.sigmoid(raw) * (self.param_upper - self.param_lower)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 50,
        label_mean: Optional[torch.Tensor] = None,
        label_std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation for parameter forecasts.

        Runs ``num_samples`` stochastic forward passes with the LSTM dropout
        layers active (``model.train()``) to approximate the posterior
        predictive distribution of the predicted Heston parameters.

        Two inference modes:

        - **With label statistics** (recommended): pass ``label_mean`` and
          ``label_std`` computed from the training set. The raw logits are
          Z-score denormalized and clamped to physical bounds. This is the
          correct mode because the model is trained with MSE on Z-scored labels.
        - **Without label statistics** (legacy): falls back to
          ``bounded_predict()``, which interprets raw logits as sigmoid inputs.
          Produces physically valid but potentially biased parameter estimates.

        Args:
            x:           Tensor of shape (batch_size, seq_len, input_size).
            num_samples: Number of MC forward passes (default: 50).
            label_mean:  Training-set label means, shape (5,). Required for
                         correct Z-score denormalization (default: None).
            label_std:   Training-set label stds,  shape (5,). Required for
                         correct Z-score denormalization (default: None).

        Returns:
            mean_params: Mean predicted parameters, shape (batch_size, 5).
            std_params:  Std-dev across MC samples, shape (batch_size, 5).
                         Represents epistemic uncertainty in the forecast.
        """
        self.train()   # activate dropout for stochastic sampling

        use_denorm = (label_mean is not None) and (label_std is not None)

        samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(num_samples):
                if use_denorm:
                    # Correct path: Z-score denormalization + physical clamp
                    raw = self.forward(x)                            # (B, 5)
                    params = raw * label_std + label_mean            # denormalize
                    params = torch.clamp(params,                     # enforce bounds
                                         self.param_lower,
                                         self.param_upper)
                    samples.append(params)
                else:
                    # Legacy path: sigmoid bounding (no label stats needed)
                    samples.append(self.bounded_predict(x))          # (B, 5)

        self.eval()    # restore deterministic mode

        stack = torch.stack(samples, dim=0)   # (num_samples, B, 5)
        return stack.mean(dim=0), stack.std(dim=0)

    # ── Feller condition check ─────────────────────────────────────────────────

    @staticmethod
    def feller_satisfied(params: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask indicating which rows satisfy 2κθ > σ².

        Data order: [v0, rho, sigma, theta, kappa]

        Args:
            params: Tensor of shape (batch_size, 5).

        Returns:
            Boolean tensor of shape (batch_size,).
        """
        sigma = params[:, 2]
        theta = params[:, 3]
        kappa = params[:, 4]
        return (2.0 * kappa * theta) > (sigma ** 2)

    # ── String representation ─────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, output_size={self.output_size}"
        )


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model = HestonDynamicsLSTM()
    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {n_params:,}")

    # Shape check
    batch = torch.randn(32, 10, 88)
    raw   = model(batch)
    bound = model.bounded_predict(batch)

    print(f"\nInput shape  : {batch.shape}")
    print(f"Raw output   : {raw.shape}")
    print(f"Bounded out  : {bound.shape}")
    assert raw.shape   == (32, 5), f"Expected (32,5), got {raw.shape}"
    assert bound.shape == (32, 5), f"Expected (32,5), got {bound.shape}"

    # Bound enforcement
    lo = PARAM_LOWER.numpy()
    hi = PARAM_UPPER.numpy()
    b  = bound.detach().numpy()
    for i, name in enumerate(PARAM_NAMES):
        assert np.all(b[:, i] >= lo[i] - 1e-6), f"{name} below lower bound"
        assert np.all(b[:, i] <= hi[i] + 1e-6), f"{name} above upper bound"
        print(f"  {name:6s}: [{b[:, i].min():.5f}, {b[:, i].max():.5f}]  "
              f"bounds [{lo[i]:.4f}, {hi[i]:.4f}] ✓")

    # Feller check (probabilistic — not guaranteed with random weights)
    feller = HestonDynamicsLSTM.feller_satisfied(bound)
    print(f"\nFeller satisfied: {feller.sum().item()}/32 rows")
    print("\n✓ All shape and bound checks passed.")

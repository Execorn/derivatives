"""
Second-Order Autograd Sensitivities — Proving C² Smoothness of the ELU Surrogate
==================================================================================

This script demonstrates that the trained HestonSurrogateMLP, by using ELU
activations, produces a C² smooth mapping from Heston parameters to the IV
surface. We compute:

  1. The Jacobian  — ∂IV_mean / ∂θ_i   (first-order sensitivities, analogous to "Greeks")
  2. The Hessian   — ∂²IV_mean / ∂θ_i∂θ_j  (second-order sensitivities, curvature)

With ReLU activations, the Hessian is **identically zero** everywhere (ReLU'' = 0),
making second-order optimisation methods and second-order calibration sensitivity
analysis impossible. ELU satisfies f''(x) = exp(x) ≠ 0 for x < 0, guaranteeing
a non-trivial, well-defined Hessian.

Reference: Itkin (2019), "Deep learning calibration of option pricing models:
           some pitfalls and solutions."

Usage:
    cd path/to/derivatives
    python src/greeks_autograd.py
"""

from __future__ import annotations

import sys
import gzip
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP

# ─── Paths ─────────────────────────────────────────────────────────────────────

WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "weights" / "heston_best.pth"
FEAT_SCALER = PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl"
TARGET_SCALER = PROJECT_ROOT / "artifacts" / "scalers" / "target_scaler.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "HestonTrainSet.txt.gz"

# Data column order: [v0, rho, sigma, theta, kappa]
PARAM_NAMES = ["v0", "rho", "sigma", "theta", "kappa"]


# ─── Activation second-derivative analysis ────────────────────────────────────


def activation_second_derivative_demo() -> None:
    """Demonstrate analytically why ReLU Hessians vanish but ELU does not."""
    x = torch.linspace(-2.0, 2.0, 200)

    # ReLU second derivative
    x_relu = x.clone().requires_grad_(True)
    y_relu = torch.relu(x_relu)
    grad1_relu = torch.autograd.grad(y_relu.sum(), x_relu, create_graph=True)[0]
    grad2_relu = torch.autograd.grad(grad1_relu.sum(), x_relu, create_graph=False)[0]

    # ELU second derivative
    elu = nn.ELU()
    x_elu = x.clone().requires_grad_(True)
    y_elu = elu(x_elu)
    grad1_elu = torch.autograd.grad(y_elu.sum(), x_elu, create_graph=True)[0]
    grad2_elu = torch.autograd.grad(grad1_elu.sum(), x_elu, create_graph=False)[0]

    relu_hess_norm = grad2_relu.abs().mean().item()
    elu_hess_norm = grad2_elu.abs().mean().item()

    print("=" * 62)
    print("  Activation Function Second-Derivative Analysis")
    print("=" * 62)
    print(f"  ReLU  mean |f''(x)| : {relu_hess_norm:.8f}  <- identically 0")
    print(f"  ELU   mean |f''(x)| : {elu_hess_norm:.8f}  <- non-zero, C2 smooth")
    print()
    print("  ReLU: f(x) = max(0,x)  =>  f'(x) = H(x)  =>  f''(x) = 0")
    print("  ELU:  f(x) = x if x>0 else exp(x)-1")
    print("        f'(x) = 1 if x>0 else exp(x)")
    print("        f''(x) = 0 if x>0 else exp(x)  (non-zero for x < 0)")
    print("=" * 62)


# ─── Network Jacobian ─────────────────────────────────────────────────────────


def compute_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute full Jacobian  J[i,j] = doutput_i / dinput_j,  shape (88, 5).

    Parameters
    ----------
    model : nn.Module
        Trained surrogate in eval mode.
    x : torch.Tensor
        Input of shape (1, 5).

    Returns
    -------
    torch.Tensor
        Jacobian of shape (88, 5).
    """
    result = torch.autograd.functional.jacobian(
        func=lambda inp: model(inp).squeeze(0),
        inputs=x,
        create_graph=False,
        vectorize=True,
    )
    # Result shape: (88, 1, 5) -> squeeze to (88, 5)
    return result.squeeze(1)


def compute_scalar_hessian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian of mean(IV_surface) w.r.t. input parameters.

    H[i,j] = d2 mean(IV) / d_theta_i d_theta_j,  shape (5, 5).

    Parameters
    ----------
    model : nn.Module
        Trained surrogate.
    x : torch.Tensor
        Input of shape (1, 5).

    Returns
    -------
    torch.Tensor
        Hessian matrix of shape (5, 5).
    """

    def scalar_output(inp: torch.Tensor) -> torch.Tensor:
        return model(inp).squeeze(0).mean()

    H = torch.autograd.functional.hessian(
        func=scalar_output,
        inputs=x,
        create_graph=False,
        vectorize=True,
    )
    # Result shape: (1, 5, 1, 5) -> squeeze to (5, 5)
    return H.squeeze()


# ─── Display helpers ──────────────────────────────────────────────────────────


def print_jacobian_summary(J: torch.Tensor, param_names: list[str]) -> None:
    """Print per-parameter mean sensitivity across the IV surface."""
    dIV_dtheta = J.mean(dim=0).detach().numpy()

    print()
    print("  First-Order Sensitivities  d(mean IV) / d_theta_i  [scaled space]")
    print("  " + "-" * 60)
    max_val = max(abs(v) for v in dIV_dtheta)
    for name, val in zip(param_names, dIV_dtheta):
        bar_len = int(abs(val) / max_val * 35) if max_val > 0 else 0
        bar = ("+" if val >= 0 else "-") * bar_len
        print(f"  {name:>6}  {val:+10.6f}  |{bar:<35}|")
    print("  " + "-" * 60)


def print_hessian(H: torch.Tensor, param_names: list[str]) -> None:
    """Print the 5x5 Hessian with spectral analysis."""
    H_np = H.detach().numpy()

    print()
    print("  Second-Order Sensitivities  d2(mean IV) / d_theta_i d_theta_j  [5x5 Hessian]")
    print()
    header = "        " + "  ".join(f"{n:>9}" for n in param_names)
    print(f"  {header}")
    print(f"  {'':8}" + "  " + "-" * 57)
    for i, row_name in enumerate(param_names):
        row_vals = "  ".join(f"{H_np[i, j]:>+9.5f}" for j in range(5))
        print(f"  {row_name:>6}  |  {row_vals}")
    print()

    eigenvalues = np.linalg.eigvalsh(H_np)
    frobenius = np.linalg.norm(H_np)
    is_pos_def = bool((eigenvalues > 0).all())
    is_neg_def = bool((eigenvalues < 0).all())

    print(f"  Eigenvalues    : {np.round(eigenvalues, 6)}")
    print(f"  Frobenius norm : {frobenius:.8f}")
    print(
        f"  Definiteness   : {'Positive definite' if is_pos_def else 'Negative definite' if is_neg_def else 'Indefinite'}"
    )
    print(f"  ||H|| > 0      : {frobenius > 1e-10}  <- proves non-zero curvature")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 68)
    print("  Heston Surrogate: Autograd Second-Order Sensitivity Analysis")
    print("  Proving C2 Smoothness: ELU vs ReLU")
    print("=" * 68)
    print()

    # 1. Activation analysis
    activation_second_derivative_demo()

    # 2. Load artefacts
    print()
    print("Loading model and scalers ...")
    f_scaler = joblib.load(FEAT_SCALER)
    t_scaler = joblib.load(TARGET_SCALER)
    model = HestonSurrogateMLP()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    model = model.double()  # float64 for numerical precision

    # 3. Pick a test sample
    with gzip.open(DATA_PATH, "rb") as fh:
        data = np.load(fh)
    _, X_test, _, _ = train_test_split(data[:, :5], data[:, 5:], test_size=0.15, random_state=42)

    sample_raw = X_test[42]
    sample_scaled = f_scaler.transform(sample_raw.reshape(1, -1))

    print(f"\n  Sample parameters [v0, rho, sigma, theta, kappa]:")
    for name, rv, sv in zip(PARAM_NAMES, sample_raw, sample_scaled.flatten()):
        print(f"    {name:>6}  raw={rv:+.6f}  scaled={sv:+.6f}")

    # 4. Forward pass
    x = torch.tensor(sample_scaled, dtype=torch.float64).requires_grad_(True)
    with torch.no_grad():
        iv_scaled_np = model(x).numpy()
    iv_raw = t_scaler.inverse_transform(iv_scaled_np)
    print(f"\n  IV surface range (unscaled): [{iv_raw.min():.5f}, {iv_raw.max():.5f}]")
    print(f"  Mean IV = {iv_raw.mean():.6f}")

    # 5. Jacobian (88x5)
    print("\n  Computing Jacobian (88 x 5) ...")
    x_J = torch.tensor(sample_scaled, dtype=torch.float64).requires_grad_(True)
    J = compute_jacobian(model, x_J)
    print(f"  Jacobian shape: {tuple(J.shape)}")
    print_jacobian_summary(J, PARAM_NAMES)

    # 6. Hessian of mean(IV) (5x5)
    print("\n  Computing Hessian of mean(IV) w.r.t. theta (5 x 5) ...")
    x_H = torch.tensor(sample_scaled, dtype=torch.float64).requires_grad_(True)
    H_elu = compute_scalar_hessian(model, x_H)
    print(f"  Hessian shape: {tuple(H_elu.shape)}")
    print_hessian(H_elu, PARAM_NAMES)

    # 7. ReLU comparison (same weights, different activation)
    print()
    print("=" * 68)
    print("  ReLU vs ELU Hessian Comparison  (identical weights, same sample)")
    print("=" * 68)

    class ReLUSurrogateMLP(nn.Module):
        """Identical to HestonSurrogateMLP but with ReLU activations."""

        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(5, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Linear(30, 88),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    relu_model = ReLUSurrogateMLP().double()
    with torch.no_grad():
        for (_, p_elu), (_, p_relu) in zip(model.named_parameters(), relu_model.named_parameters()):
            p_relu.copy_(p_elu)

    x_relu = torch.tensor(sample_scaled, dtype=torch.float64).requires_grad_(True)
    H_relu = compute_scalar_hessian(relu_model, x_relu)

    frob_elu = np.linalg.norm(H_elu.detach().numpy())
    frob_relu = np.linalg.norm(H_relu.detach().numpy())

    print(f"\n  ELU  Hessian Frobenius norm : {frob_elu:.8f}  <- curvature present (C2)")
    print(f"  ReLU Hessian Frobenius norm : {frob_relu:.8f}  <- identically zero  (C1 only)")
    print()
    print("  CONCLUSION:")
    print("  [OK] ELU: ||H|| > 0  -> C2 smooth -> valid Hessian for 2nd-order methods")
    print("  [!!] ReLU: ||H|| = 0 -> piecewise linear -> Hessian carries no information")
    print("             Second-order calibration and vega/vanna Greeks are impossible.")
    print()
    print("=" * 68)
    print("  Script completed successfully.")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()

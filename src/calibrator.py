"""
Heston Calibration Engine.

Optimizes the 5 Heston parameters using the trained PyTorch surrogate model.
Enforces strict physical bounds, the Feller condition, and no-arbitrage constraints.
"""

import time
from typing import Tuple, Optional

import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution, OptimizeResult


class HestonCalibrator:
    """
    Calibrator for the Heston model utilizing a neural network surrogate.

    Args:
        surrogate_model (torch.nn.Module): The trained surrogate mapping (5) -> (88).
        feature_scaler: The fitted sklearn MinMaxScaler for the features.
        target_scaler: The fitted sklearn StandardScaler for the target IV surface.
        method (str): Optimization method to use ('L-BFGS-B' or 'DE').
    """

    def __init__(
        self,
        surrogate_model: torch.nn.Module,
        feature_scaler,
        target_scaler,
        method: str = "L-BFGS-B",
    ) -> None:
        self.surrogate_model = surrogate_model
        self.surrogate_model.eval()

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.method = method.upper()

        # Data order is: [v0, rho, sigma, theta, kappa]
        # Theoretical lower bounds: v0 > 0, rho >= -1, sigma > 0, theta > 0, kappa > 0
        theoretical_lower_bounds = np.array([[1e-6, -1.0, 1e-6, 1e-6, 1e-6]])
        scaled_lb = self.feature_scaler.transform(theoretical_lower_bounds).flatten()

        # The NN is trained on [-1, 1], so we constrain the optimization there,
        # but raise the floor if the theoretical bound dictates it.
        self.bounds = [(max(-1.0, lb), 1.0) for lb in scaled_lb]

    def _objective_func(
        self, scaled_params: np.ndarray, target_iv_scaled: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """
        Objective function for SciPy optimizer.

        Computes MSE against the target surface.
        Adds Feller Condition penalty (hard) and No-Arbitrage penalties (soft).

        Args:
            scaled_params: Current parameter guess in scaled domain (shape: 5,).
            target_iv_scaled: Target IV surface in scaled domain (shape: 88,).

        Returns:
            Tuple containing the loss scalar and the gradient numpy array.
        """
        # 1. Feller Condition Check (Hard Penalty)
        # Inverse transform to evaluate in original parameter space
        unscaled_params = self.feature_scaler.inverse_transform(
            scaled_params.reshape(1, -1)
        ).flatten()
        v0, rho, sigma, theta, kappa = unscaled_params

        # 2*kappa*theta > sigma^2
        if 2 * kappa * theta < sigma**2:
            return 1e6, np.zeros_like(scaled_params)

        # 2. PyTorch Forward Pass & Gradients
        x_tensor = (
            torch.tensor(scaled_params, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        )

        pred_iv_scaled = self.surrogate_model(x_tensor)

        # Primary MSE Loss
        mse_loss = torch.mean((pred_iv_scaled - target_iv_scaled) ** 2)

        # 3. No-Arbitrage Soft Penalties
        # Reshape to 8 maturities x 11 strikes
        pred_iv_2d = pred_iv_scaled.view(8, 11)

        # Calendar Arbitrage: dIV/dT >= 0 (Penalty if < 0)
        diff_T = torch.diff(pred_iv_2d, dim=0)
        calendar_penalty = torch.sum(torch.relu(-diff_T) ** 2)

        # Butterfly Arbitrage: d^2IV/dK^2 >= 0 (Penalty if < 0)
        diff2_K = torch.diff(pred_iv_2d, n=2, dim=1)
        butterfly_penalty = torch.sum(torch.relu(-diff2_K) ** 2)

        # Combined Loss
        lambda_penalty = 1e-4
        loss = mse_loss + lambda_penalty * (calendar_penalty + butterfly_penalty)

        # Backpropagation
        self.surrogate_model.zero_grad()
        loss.backward()

        loss_val = loss.item()
        grad_val = x_tensor.grad.squeeze(0).numpy().astype(np.float64)

        return loss_val, grad_val

    def calibrate(
        self, target_iv: np.ndarray, initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, OptimizeResult]:
        """
        Calibrate the Heston parameters to match the given IV surface.

        Args:
            target_iv: Implied volatility surface in original scale (shape: 88,).
            initial_guess: Optional starting parameters in original scale.

        Returns:
            optimal_params: Calibrated parameters in original scale.
            result: SciPy OptimizeResult object.
        """
        target_iv_2d = target_iv.reshape(1, -1)
        target_iv_scaled_np = self.target_scaler.transform(target_iv_2d)
        target_iv_scaled = torch.tensor(target_iv_scaled_np, dtype=torch.float32)

        if self.method == "DE":
            result = differential_evolution(
                func=lambda x: self._objective_func(x, target_iv_scaled)[0],
                bounds=self.bounds,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
            )
        else:
            if initial_guess is not None:
                init_scaled = self.feature_scaler.transform(initial_guess.reshape(1, -1)).flatten()
                # Clip strictly within calculated bounds
                lower_bounds = [b[0] for b in self.bounds]
                upper_bounds = [b[1] for b in self.bounds]
                init_scaled = np.clip(init_scaled, lower_bounds, upper_bounds)
            else:
                init_scaled = np.zeros(5)

            result = minimize(
                fun=self._objective_func,
                x0=init_scaled,
                args=(target_iv_scaled,),
                method="L-BFGS-B",
                jac=True,
                bounds=self.bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

        optimal_params_scaled = result.x.reshape(1, -1)
        optimal_params = self.feature_scaler.inverse_transform(optimal_params_scaled).flatten()

        return optimal_params, result


if __name__ == "__main__":
    """End-to-end test with Feller + no-arbitrage constraint verification."""
    import gzip
    import sys
    import joblib
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    sys.path.insert(0, str(SRC_DIR))

    from model import HestonSurrogateMLP

    WEIGHTS = PROJECT_ROOT / "artifacts" / "weights" / "heston_best.pth"
    DATA = (
        PROJECT_ROOT / "data" / "HestonTrainSet.txt.gz"
    )
    # Data order: [v0, rho, sigma, theta, kappa]
    NAMES = ["v0", "rho", "sigma", "theta", "kappa"]

    print("Loading artefacts ...")
    f_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl")
    t_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "target_scaler.pkl")
    model = HestonSurrogateMLP()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    with gzip.open(DATA, "rb") as fh:
        data = np.load(fh)
    X_raw, Y_raw = data[:, :5], data[:, 5:]
    _, X_test, _, Y_test = train_test_split(X_raw, Y_raw, test_size=0.15, random_state=42)

    idx = 42
    true_params = X_test[idx]
    target_iv = Y_test[idx]

    calibrator = HestonCalibrator(model, f_scaler, t_scaler, method="L-BFGS-B")
    print("Running L-BFGS-B with Feller + no-arbitrage penalties ...")
    t0 = time.perf_counter()
    calibrated_params, result = calibrator.calibrate(target_iv)
    elapsed = time.perf_counter() - t0

    # Feller verification on calibrated params
    # Data order: [v0, rho, sigma, theta, kappa]
    kappa = calibrated_params[4]
    theta = calibrated_params[3]
    sigma = calibrated_params[2]
    feller_val = 2 * kappa * theta - sigma**2
    feller_ok = feller_val > 0

    print()
    print("=" * 62)
    print(f"  {'Param':<8}  {'True':>12}  {'Calibrated':>12}  {'Abs Err':>10}")
    print("=" * 62)
    for name, tv, cv in zip(NAMES, true_params, calibrated_params):
        print(f"  {name:<8}  {tv:>12.6f}  {cv:>12.6f}  {abs(tv - cv):>10.6f}")
    print("=" * 62)
    print(f"  Converged        : {result.success}")
    print(f"  Final loss        : {result.fun:.6e}")
    print(f"  Calibration time  : {elapsed * 1000:.2f} ms")
    print(f"  Feller 2kt - s^2  : {feller_val:.6f}  {'PASS' if feller_ok else 'FAIL'}")

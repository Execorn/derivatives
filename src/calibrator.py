"""
Heston Calibration Engine.

Optimizes the 5 Heston parameters using the trained PyTorch surrogate model.
Enforces strict physical bounds, the Feller condition, and no-arbitrage constraints.

The surrogate predicts Total Variance W = IV² × T in standardized space.
Inverse math (IV = √(W / T)) is applied differentiably before computing
no-arbitrage penalties, preserving the autograd graph for exact Jacobians.
"""

import time
from typing import Tuple, Optional

import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution, OptimizeResult

# Grid constants (must match training data layout)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])


class HestonCalibrator:
    """
    Calibrator for the Heston model utilizing a neural network surrogate.

    The surrogate network predicts a standardized Total Variance surface
    (W = IV² × T). Calibration is performed in the scaled W domain for
    MSE, while no-arbitrage penalties are computed on the recovered IV
    surface via a differentiable inverse transform.

    Args:
        surrogate_model (torch.nn.Module): The trained surrogate mapping (5) -> (88).
        feature_scaler: The fitted sklearn MinMaxScaler for the features.
        target_scaler: The fitted sklearn StandardScaler for the target W surface.
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

        # ── Total Variance grid ────────────────────────────────────────────
        self.T_vector = np.repeat(MATURITIES, 11)  # shape (88,)
        self.T_tensor = torch.tensor(self.T_vector, dtype=torch.float32)

        # Cache StandardScaler parameters as tensors for differentiable
        # inverse transform inside the autograd graph.
        self.target_scale = torch.tensor(
            target_scaler.scale_, dtype=torch.float32
        )
        self.target_mean = torch.tensor(
            target_scaler.mean_, dtype=torch.float32
        )

        # Data order is: [v0, rho, sigma, theta, kappa]
        # Theoretical lower bounds: v0 > 0, rho >= -1, sigma > 0, theta > 0, kappa > 0
        theoretical_lower_bounds = np.array([[1e-6, -1.0, 1e-6, 1e-6, 1e-6]])
        scaled_lb = self.feature_scaler.transform(theoretical_lower_bounds).flatten()

        # The NN is trained on [-1, 1], so we constrain the optimization there,
        # but raise the floor if the theoretical bound dictates it.
        self.bounds = [(max(-1.0, lb), 1.0) for lb in scaled_lb]

    def _objective_func(
        self, scaled_params: np.ndarray, target_w_scaled: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """
        Objective function for SciPy optimizer.

        Computes MSE against the target surface in scaled Total Variance space.
        Adds Feller Condition penalty (hard) and No-Arbitrage penalties (soft)
        computed on the recovered IV surface.

        Args:
            scaled_params: Current parameter guess in scaled domain (shape: 5,).
            target_w_scaled: Target W surface in scaled domain (shape: 1, 88).

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

        pred_w_scaled = self.surrogate_model(x_tensor)

        # Primary MSE Loss (in scaled Total Variance domain)
        mse_loss = torch.mean((pred_w_scaled - target_w_scaled) ** 2)

        # 3. No-Arbitrage Soft Penalties (computed on recovered IV surface)
        # Differentiable inverse StandardScaler: W = pred_scaled * scale + mean
        pred_W = pred_w_scaled * self.target_scale + self.target_mean

        # IV = sqrt(W / T), clamped for numerical stability
        pred_iv = torch.sqrt(torch.clamp(pred_W / self.T_tensor, min=1e-8))
        pred_iv_2d = pred_iv.view(8, 11)

        # Calendar Arbitrage: dW/dT >= 0 (total variance non-decreasing in T)
        # Correct Carr-Madan condition: W = IV²×T must be non-decreasing in T.
        # Checking dIV/dT >= 0 is wrong: IV can decrease (when v0 > theta) while
        # W is still non-decreasing, giving a spurious penalty.
        pred_W_2d = pred_W.view(8, 11)
        diff_W = torch.diff(pred_W_2d, dim=0)  # (7, 11)
        calendar_penalty = torch.sum(torch.relu(-diff_W) ** 2)

        # Butterfly Arbitrage: d²IV/dK² >= 0 (positive gamma — convexity in strike)
        diff2_K = torch.diff(pred_iv_2d, n=2, dim=1)  # (8, 9)
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

        Internally converts the target IV to Total Variance (W = IV² × T)
        before scaling, since the surrogate predicts in the W domain.

        Args:
            target_iv: Implied volatility surface in original scale (shape: 88,).
            initial_guess: Optional starting parameters in original scale.

        Returns:
            optimal_params: Calibrated parameters in original scale.
            result: SciPy OptimizeResult object.
        """
        # Convert target IV → Total Variance: W = IV² × T
        target_w = (target_iv ** 2) * self.T_vector
        target_w_2d = target_w.reshape(1, -1)
        target_w_scaled_np = self.target_scaler.transform(target_w_2d)
        target_w_scaled = torch.tensor(target_w_scaled_np, dtype=torch.float32)

        if self.method == "DE":
            result = differential_evolution(
                func=lambda x: self._objective_func(x, target_w_scaled)[0],
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
                args=(target_w_scaled,),
                method="L-BFGS-B",
                jac=True,
                bounds=self.bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

        optimal_params_scaled = result.x.reshape(1, -1)
        optimal_params = self.feature_scaler.inverse_transform(optimal_params_scaled).flatten()

        return optimal_params, result

    def predict_with_uncertainty(
        self,
        params: np.ndarray,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs ``num_samples`` stochastic forward passes with dropout active
        (model.train()) to approximate the posterior predictive distribution
        of the IV surface.  Based on Gal & Ghahramani (2016).

        IMPORTANT: This method requires the surrogate model to contain at least
        one nn.Dropout layer.  HestonSurrogateMLP satisfies this (p=0.1 after
        each hidden layer).  MirrorPaddedFNO2d does NOT have dropout — calling
        this method with FNO will produce std_iv ≈ 0 (all forward passes
        identical in eval-like behaviour).  Use the FNO confidence scores from
        calibrate.compute_confidence_scores() instead.

        Args:
            params: Heston parameters in original scale (shape: 5,).
            num_samples: Number of MC forward passes (default: 100).

        Returns:
            mean_iv: Mean IV surface across MC samples (shape: 88,).
            std_iv:  Std-dev IV surface across MC samples (shape: 88,).
        """
        # Guard: verify the model has at least one Dropout layer
        has_dropout = any(
            isinstance(m, torch.nn.Dropout) for m in self.surrogate_model.modules()
        )
        if not has_dropout:
            raise RuntimeError(
                "predict_with_uncertainty requires a model with nn.Dropout layers "
                "(MC Dropout).  MirrorPaddedFNO2d has no dropout — use "
                "calibrate.compute_confidence_scores() for FNO uncertainty instead."
            )
        # Scale input parameters
        params_scaled = self.feature_scaler.transform(params.reshape(1, -1))
        x_tensor = torch.tensor(params_scaled, dtype=torch.float32)

        # Enable dropout for MC sampling
        self.surrogate_model.train()

        iv_samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(num_samples):
                w_scaled = self.surrogate_model(x_tensor).numpy()
                w = self.target_scaler.inverse_transform(w_scaled).flatten()
                # W → IV with numerical clamping
                iv = np.sqrt(np.maximum(w / self.T_vector, 1e-8))
                iv = np.maximum(iv, 1e-6)
                iv_samples.append(iv)

        # Restore deterministic mode
        self.surrogate_model.eval()

        iv_stack = np.stack(iv_samples, axis=0)  # (num_samples, 88)
        mean_iv = iv_stack.mean(axis=0)
        std_iv = iv_stack.std(axis=0)

        return mean_iv, std_iv


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
    DATA = PROJECT_ROOT / "data" / "HestonTrainSet.txt.gz"
    # Data order: [v0, rho, sigma, theta, kappa]
    NAMES = ["v0", "rho", "sigma", "theta", "kappa"]

    print("Loading artefacts ...")
    f_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl")
    t_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "target_scaler.pkl")
    model = HestonSurrogateMLP()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    model.eval()

    with gzip.open(DATA, "rb") as fh:
        data = np.load(fh)
    X_raw, Y_raw = data[:, :5], data[:, 5:]
    _, X_test, _, Y_test = train_test_split(X_raw, Y_raw, test_size=0.15, random_state=42)

    idx = 42
    true_params = X_test[idx]
    target_iv = Y_test[idx]  # original IV — calibrate() handles W conversion

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

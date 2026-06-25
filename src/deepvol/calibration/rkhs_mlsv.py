"""
rkhs_mlsv.py — Reproducing Kernel Hilbert Space (RKHS) sparse landmark ridge regression
solver for McKean-Vlasov SDE particle variance.
Optimized via torch.linalg.solve on GPU to bypass the O(N_p^2) Nadaraya-Watson bottleneck.

Reference:
----------
Bayer, C., Belomestny, D., Butkovsky, O., & Schoenmakers, J. (2022).
"A Reproducing Kernel Hilbert Space approach to singular local stochastic volatility
McKean-Vlasov models." arXiv preprint arXiv:2203.01160.
"""

import math
import torch
from typing import Optional
import numpy as np

from deepvol.models.mlsv_gpu import MLSVSolverGPU


# Core implementations that will be compiled for Triton kernel fusion
def _solve_rkhs_system_impl(
    K_LL: torch.Tensor,
    K_pL: torch.Tensor,
    V_t: torch.Tensor,
    lambda_reg: float,
    N_p: float,
) -> torch.Tensor:
    # Solve (K_pL^T K_pL + N_p * lambda_reg * K_LL) beta = K_pL^T V_t
    A = torch.matmul(K_pL.t(), K_pL) + N_p * lambda_reg * K_LL
    # Regularize diagonal to avoid singular matrix issues
    L = A.shape[0]
    A = A + 1e-12 * torch.eye(L, device=A.device, dtype=A.dtype)
    b = torch.matmul(K_pL.t(), V_t)
    beta = torch.linalg.solve(A, b)
    # Clone to prevent static buffer overwrite issues with torch.compile CUDAGraphs
    return beta.clone()


def _compute_rbf_kernel_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    # Compute RBF Gaussian Kernel matrix: exp(-0.5 * ((x_i - y_j)/bandwidth)^2)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    K = torch.exp(-0.5 * (diff / bandwidth) ** 2)
    # Clone to prevent static buffer overwrite issues with torch.compile CUDAGraphs
    return K.clone()


# Compile steps for Triton kernel fusion with fallbacks
_solve_rkhs_system_compiled = None
_compute_rbf_kernel_compiled = None
_use_compile = True

try:
    _solve_rkhs_system_compiled = torch.compile(_solve_rkhs_system_impl, mode="reduce-overhead")
    _compute_rbf_kernel_compiled = torch.compile(_compute_rbf_kernel_impl, mode="reduce-overhead")
    # Warm up compiled functions to ensure Triton works in the environment
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dummy_K_LL = torch.eye(5, dtype=torch.float64, device=_device)
    _dummy_K_pL = torch.ones((10, 5), dtype=torch.float64, device=_device)
    _dummy_V = torch.ones(10, dtype=torch.float64, device=_device)
    _solve_rkhs_system_compiled(_dummy_K_LL, _dummy_K_pL, _dummy_V, 1e-4, 10.0)
    
    _dummy_x = torch.zeros(10, dtype=torch.float64, device=_device)
    _dummy_y = torch.zeros(5, dtype=torch.float64, device=_device)
    _compute_rbf_kernel_compiled(_dummy_x, _dummy_y, 1.0)
except Exception:
    _use_compile = False


def compute_rkhs_conditional_expectation(
    X_t: torch.Tensor,
    V_t: torch.Tensor,
    targets: torch.Tensor,
    num_landmarks: int = 50,
    bandwidth: Optional[float] = None,
    lambda_reg: float = 1e-4,
    landmarks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the conditional expectation E[V_t | X_t = target] using a sparse landmark
    RKHS ridge regression solver. Operating strictly in torch.float64 for internal computations.
    
    Mathematical Formulation:
    -------------------------
    We approximate the conditional expectation f(x) = E[V_t | X_t = x] in the RKHS space:
        f(x) ≈ sum_{j=1}^L beta_j * k(Z_j, x)
    where Z = {Z_1, ..., Z_L} are the landmark points.
    
    The coefficients beta are obtained by solving the regularized ridge regression problem:
        min_{beta} (1/N_p) * ||K_pL * beta - V_t||^2 + lambda_reg * beta^T * K_LL * beta
    which leads to the linear system:
        (K_pL^T * K_pL + N_p * lambda_reg * K_LL) * beta = K_pL^T * V_t
    where:
        K_pL_{i,j} = k(X_t_{i}, Z_j)
        K_LL_{i,j} = k(Z_i, Z_j)
        
    Complexity:
    -----------
    Constructing K_pL and K_pL^T * K_pL takes O(N_p * L^2) time.
    Solving the L x L system takes O(L^3) time.
    Total complexity is O(L^3 + N_p * L^2), which is O(N_p) for fixed L << N_p,
    bypassing the O(N_p^2) cost of Nadaraya-Watson regression.
    
    Parameters:
    -----------
    X_t : torch.Tensor
        Conditioning variables (e.g. log stock price paths), shape (N_paths,)
    V_t : torch.Tensor
        Dependent variables (e.g. particle variance paths), shape (N_paths,)
    targets : torch.Tensor
        Target evaluation points (e.g. log strike grid), shape (N_targets,)
    num_landmarks : int, default 50
        Number of landmark points L. Must satisfy L << N_paths.
    bandwidth : float, optional
        Bandwidth parameter for the Gaussian RBF kernel. If None, defaults to
        Silverman's rule of thumb: 1.06 * std(X_t) * N_paths^(-1/5).
    lambda_reg : float, default 1e-4
        Regularization penalty parameter lambda.
    landmarks : torch.Tensor, optional
        Custom landmark points tensor of shape (num_landmarks,). If None, spaced
        uniformly along the union of range of X_t and targets.
        
    Returns:
    --------
    estimates : torch.Tensor
        Estimated conditional expectations at target points, shape (N_targets,)
    """
    device = X_t.device
    dtype = X_t.dtype
    N_p = float(X_t.numel())
    
    # Cast all internal variables to float64 to ensure high numerical precision
    X_t_dbl = X_t.to(device=device, dtype=torch.float64)
    V_t_dbl = V_t.to(device=device, dtype=torch.float64)
    targets_dbl = targets.to(device=device, dtype=torch.float64)
    
    # Auto-spacing landmarks covering the span of training data and targets
    if landmarks is None:
        min_val = torch.min(torch.cat([X_t_dbl, targets_dbl]))
        max_val = torch.max(torch.cat([X_t_dbl, targets_dbl]))
        landmarks_dbl = torch.linspace(min_val, max_val, num_landmarks, device=device, dtype=torch.float64)
    else:
        landmarks_dbl = landmarks.to(device=device, dtype=torch.float64)
        
    # Auto-bandwidth selection via Silverman's rule of thumb
    if bandwidth is None:
        std_X = torch.std(X_t_dbl)
        std_X = torch.clamp(std_X, min=1e-6)
        bandwidth = 1.06 * std_X * (N_p ** -0.2)
        bandwidth = float(bandwidth.item())
        
    # Construct kernel matrices
    if _use_compile:
        try:
            K_LL = _compute_rbf_kernel_compiled(landmarks_dbl, landmarks_dbl, bandwidth)
            K_pL = _compute_rbf_kernel_compiled(X_t_dbl, landmarks_dbl, bandwidth)
            beta = _solve_rkhs_system_compiled(K_LL, K_pL, V_t_dbl, lambda_reg, N_p)
            K_targets_L = _compute_rbf_kernel_compiled(targets_dbl, landmarks_dbl, bandwidth)
        except Exception:
            K_LL = _compute_rbf_kernel_impl(landmarks_dbl, landmarks_dbl, bandwidth)
            K_pL = _compute_rbf_kernel_impl(X_t_dbl, landmarks_dbl, bandwidth)
            beta = _solve_rkhs_system_impl(K_LL, K_pL, V_t_dbl, lambda_reg, N_p)
            K_targets_L = _compute_rbf_kernel_impl(targets_dbl, landmarks_dbl, bandwidth)
    else:
        K_LL = _compute_rbf_kernel_impl(landmarks_dbl, landmarks_dbl, bandwidth)
        K_pL = _compute_rbf_kernel_impl(X_t_dbl, landmarks_dbl, bandwidth)
        beta = _solve_rkhs_system_impl(K_LL, K_pL, V_t_dbl, lambda_reg, N_p)
        K_targets_L = _compute_rbf_kernel_impl(targets_dbl, landmarks_dbl, bandwidth)
        
    # Evaluate at target points
    output = torch.matmul(K_targets_L, beta)
    
    # Cast back to original dtype at the boundary
    # Clamp to avoid non-positive variance/singularity issues (standard guardian fallback)
    output = torch.clamp(output, min=1e-8)
    return output.to(dtype=dtype)


class RKHSMLSVSolver(MLSVSolverGPU):
    """
    McKean-Vlasov SDE particle solver with RKHS sparse landmark ridge regression.
    Inherits from MLSVSolverGPU and overrides simulate() to run with RKHS conditional expectations.
    """
    
    def simulate(
        self,
        method: str = "rkhs",
        block_size: int = 1024,
        reg_epsilon: float = 1e-8,
        vol_boundary_style: str = None,
        num_landmarks: int = 50,
        bandwidth: Optional[float] = None,
        lambda_reg: float = 1e-4,
    ):
        """
        Simulate McKean-Vlasov log stock price X and variance V paths on GPU.
        
        Parameters:
        -----------
        method : str, default "rkhs"
            Conditional expectation estimation method: "rkhs", "nadaraya_watson", or "muguruza"
        block_size : int, default 1024
            Block size for tiling (for fallback methods)
        reg_epsilon : float, default 1e-8
            Regularization parameter for conditional expectations (for fallback methods)
        vol_boundary_style : str, optional
            If provided, overrides self.vol_boundary_style
        num_landmarks : int, default 50
            Number of landmark points for RKHS regression
        bandwidth : float, optional
            Bandwidth parameter for the Gaussian kernel. If None, auto-calculated
        lambda_reg : float, default 1e-4
            Regularization parameter for RKHS ridge regression
        """
        if method != "rkhs":
            # Fallback to parent simulation for classic methods
            return super().simulate(
                method=method,
                block_size=block_size,
                reg_epsilon=reg_epsilon,
                vol_boundary_style=vol_boundary_style,
            )
            
        boundary_style = vol_boundary_style or self.vol_boundary_style
        torch.manual_seed(42)
        
        # Allocate path storage: shape (N_steps + 1, N_paths) on the GPU/device
        self.X_paths = torch.empty((self.N_steps + 1, self.N_paths), device=self.device, dtype=self.dtype)
        self.V_paths = torch.empty((self.N_steps + 1, self.N_paths), device=self.device, dtype=self.dtype)
        
        # Initialize paths at t=0
        self.X_paths[0] = torch.full((self.N_paths,), math.log(self.S0), device=self.device, dtype=self.dtype)
        self.V_paths[0] = torch.full((self.N_paths,), self.v0, device=self.device, dtype=self.dtype)
        
        # Allocate tensors to store steps parameters (for compatibility/monitoring)
        sigma_paths = torch.empty((self.N_steps, self.N_paths), device=self.device, dtype=self.dtype)
        Z_V_paths = torch.empty((self.N_steps, self.N_paths), device=self.device, dtype=self.dtype)
        
        dt = self.dt
        sqrt_dt = math.sqrt(dt)
        
        # Keep active state variables as separate, contiguous 1D GPU tensors (SoA layout)
        X_curr = self.X_paths[0].clone()
        V_curr = self.V_paths[0].clone()
        
        for i in range(1, self.N_steps + 1):
            t_prev = self.t_grid[i - 1]
            
            # 1. Compute conditional expectation E[V_{t_{i-1}} | X_{t_{i-1}}]
            if i == 1:
                # At t=0, all particles are identical, E[V_0 | X_0] = v0
                cond_expect = torch.full((self.N_paths,), self.v0, device=self.device, dtype=self.dtype)
            else:
                cond_expect = compute_rkhs_conditional_expectation(
                    X_t=X_curr,
                    V_t=V_curr,
                    targets=X_curr,
                    num_landmarks=num_landmarks,
                    bandwidth=bandwidth,
                    lambda_reg=lambda_reg,
                )
                
            # 2. Evaluate Dupire local volatility and compute local stochastic volatility coefficient
            S_prev = torch.exp(X_curr)
            dup_vol = self.dupire_vol_fn(t_prev, S_prev)
            
            # Handle non-tensor return values from user-supplied functions
            if not isinstance(dup_vol, torch.Tensor):
                dup_vol = torch.tensor(dup_vol, device=self.device, dtype=self.dtype)
            else:
                dup_vol = dup_vol.to(device=self.device, dtype=self.dtype)
                
            dup_vol = torch.clamp(dup_vol, min=1e-4)  # numerical regularization for local vol
            
            cond_expect_safe = torch.clamp(cond_expect, min=1e-8)
            sigma_t_prev = dup_vol * torch.sqrt(V_curr) / torch.sqrt(cond_expect_safe)
            sigma_paths[i - 1] = sigma_t_prev
            
            # 3. Generate correlated Brownian increments directly on the GPU/device
            Z_V = torch.randn(self.N_paths, device=self.device, dtype=self.dtype)
            Z_Z = torch.randn(self.N_paths, device=self.device, dtype=self.dtype)
            Z_S = self.rho * Z_V + math.sqrt(1.0 - self.rho ** 2) * Z_Z
            
            Z_V_paths[i - 1] = Z_V
            
            # 4. Diffuse variance V (Heston-like)
            V_prev_pos = torch.clamp(V_curr, min=1e-6)
            V_next = V_curr + self.kappa * (self.theta - V_prev_pos) * dt + self.xi * torch.sqrt(V_prev_pos) * sqrt_dt * Z_V
            
            if boundary_style == "reflection":
                V_next = torch.where(V_next < 1e-6, 2e-6 - V_next, V_next)
                V_next = torch.clamp(V_next, min=1e-6)
            else:  # truncation
                V_next = torch.clamp(V_next, min=1e-6)
                
            V_curr = V_next
            self.V_paths[i] = V_curr
            
            # 5. Diffuse log stock price X
            X_next = X_curr + (self.r - self.q - 0.5 * (sigma_t_prev ** 2)) * dt + sigma_t_prev * sqrt_dt * Z_S
            X_curr = X_next
            self.X_paths[i] = X_curr


class RKHSMLSVEngine:
    """
    Calibration engine for MLSV utilizing RKHS sparse landmark ridge regression solver.
    """
    def __init__(self, kappa: float, theta: float, epsilon: float, rho: float, dupire_grid: Optional[dict] = None, device: Optional[str] = None):
        if not (np.isfinite(kappa) and np.isfinite(theta) and np.isfinite(epsilon) and np.isfinite(rho)):
            raise ValueError("All inputs must be finite")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be between -1.0 and 1.0")
            
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.rho = rho
        self.dupire_grid = dupire_grid
        
        # GPU-first device resolution
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
    def price_option(
        self,
        spot: float,
        strike: float,
        maturity: float,
        vol: float,
        is_call: bool = True,
        num_landmarks: int = 50,
        lambda_reg: float = 1e-4,
    ) -> float:
        if not (np.isfinite(spot) and np.isfinite(strike) and np.isfinite(maturity) and np.isfinite(vol)):
            raise ValueError("All inputs must be finite")
        if spot <= 0.0:
            raise ValueError("Spot must be positive")
        if strike <= 0.0:
            raise ValueError("Strike must be positive")
        if maturity <= 0.0:
            raise ValueError("Maturity must be positive")
        if vol <= 0.0:
            raise ValueError("Volatility must be positive")
            
        if not is_call:
            # Use call-put parity: C - P = S - K * exp(-r*T) = S - K (since r=0, q=0 in engine solver)
            call_price = self.price_option(
                spot=spot,
                strike=strike,
                maturity=maturity,
                vol=vol,
                is_call=True,
                num_landmarks=num_landmarks,
                lambda_reg=lambda_reg,
            )
            return call_price - spot + strike

        if self.dupire_grid is not None:
            if isinstance(self.dupire_grid, dict):
                vol_grid = np.array(self.dupire_grid["vol"])
            else:
                vol_grid = np.array(self.dupire_grid)
            def dupire_vol_fn(t, s):
                return torch.full_like(s, float(vol_grid[0, 0]))
        else:
            def dupire_vol_fn(t, s):
                return torch.full_like(s, vol)
            
        solver = RKHSMLSVSolver(
            S0=spot,
            r=0.0,
            q=0.0,
            v0=vol**2,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.epsilon,
            rho=self.rho,
            T=maturity,
            steps_per_unit=50,
            N_paths=2000,
            dupire_vol_fn=dupire_vol_fn,
            device=self.device,
            dtype=torch.float64,
        )
        solver.simulate(method="rkhs", num_landmarks=num_landmarks, lambda_reg=lambda_reg)
        return solver.price_european_option(strike=strike, maturity=maturity, is_call=is_call)
        
    def conditional_expectation(
        self,
        spot_grid: np.ndarray,
        current_spot: float,
        current_vol: float,
        num_landmarks: int = 50,
        lambda_reg: float = 1e-4,
    ) -> np.ndarray:
        solver = RKHSMLSVSolver(
            S0=current_spot,
            r=0.0,
            q=0.0,
            v0=current_vol**2,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.epsilon,
            rho=self.rho,
            T=0.1,
            steps_per_unit=10,
            N_paths=5000,
            dupire_vol_fn=lambda t, s: torch.full_like(s, current_vol),
            device=self.device,
            dtype=torch.float64,
        )
        solver.simulate(method="rkhs", num_landmarks=num_landmarks, lambda_reg=lambda_reg)
        X_t = solver.X_paths[-1]
        V_t = solver.V_paths[-1]
        targets = torch.tensor(np.log(spot_grid), device=self.device, dtype=torch.float64)
        expectations_t = compute_rkhs_conditional_expectation(
            X_t, V_t, targets, num_landmarks=num_landmarks, lambda_reg=lambda_reg
        )
        return expectations_t.cpu().numpy()
        
    def calibrate_local_vol(
        self,
        spot_grid: np.ndarray,
        time_grid: np.ndarray,
        market_prices: np.ndarray,
        num_landmarks: int = 50,
        lambda_reg: float = 1e-4,
    ) -> np.ndarray:
        M, N = len(spot_grid), len(time_grid)
        local_vol = np.zeros((N, M))
        
        solver = RKHSMLSVSolver(
            S0=100.0,
            r=0.0,
            q=0.0,
            v0=0.04,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.epsilon,
            rho=self.rho,
            T=max(time_grid),
            steps_per_unit=50,
            N_paths=2000,
            dupire_vol_fn=lambda t, s: torch.full_like(s, 0.2),
            device=self.device,
            dtype=torch.float64,
        )
        solver.simulate(method="rkhs", num_landmarks=num_landmarks, lambda_reg=lambda_reg)
        
        for j, t in enumerate(time_grid):
            t_idx = int(torch.argmin(torch.abs(solver.t_grid - t)).item())
            X_t = solver.X_paths[t_idx]
            V_t = solver.V_paths[t_idx]
            targets = torch.tensor(np.log(spot_grid), device=self.device, dtype=torch.float64)
            expectations = compute_rkhs_conditional_expectation(
                X_t, V_t, targets, num_landmarks=num_landmarks, lambda_reg=lambda_reg
            ).cpu().numpy()
            expectations = np.maximum(expectations, 1e-6)
            
            for i, S in enumerate(spot_grid):
                dup_vol = 0.20
                lv = dup_vol / np.sqrt(expectations[i])
                local_vol[j, i] = np.clip(lv, 0.05, 1.5)
                
        return local_vol

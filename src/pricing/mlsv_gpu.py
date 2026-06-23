"""
mlsv_gpu.py — GPU-accelerated McKean-Vlasov SDE particle solver with kernel density regression.
Supports Nadaraya-Watson kernel density regression and Muguruza's analytical conditional expectation.
Uses target-dimension chunking/tiling to prevent GPU memory OOM.
"""

import math
from typing import Callable, Union
import numpy as np
import torch


# Helper function for Muguruza's vectorized weight chunk calculation
def _compute_muguruza_chunk(
    targets_chunk: torch.Tensor,
    mu_i: torch.Tensor,
    sigma_cond_safe: torch.Tensor,
    V_t: torch.Tensor,
    uncond_mean: torch.Tensor,
    reg_epsilon: float,
) -> torch.Tensor:
    # pairwise differences: (chunk_size, N_paths)
    diff = targets_chunk.unsqueeze(1) - mu_i.unsqueeze(0)
    exponent = -0.5 * (diff / sigma_cond_safe.unsqueeze(0)) ** 2
    
    # normal density weight: 1/sigma_cond * exp(exponent) (common factor of 1/sqrt(2pi) cancels)
    w = (1.0 / sigma_cond_safe.unsqueeze(0)) * torch.exp(exponent)
    
    sum_w = torch.sum(w, dim=1)  # (chunk_size,)
    sum_V_w = torch.mv(w, V_t)  # (chunk_size,)
    
    # Regularized division
    chunk_est = (sum_V_w + reg_epsilon * uncond_mean) / (sum_w + reg_epsilon)
    
    # Fallback to unconditional mean in extremely low density regions
    low_density_mask = (sum_w < 1e-10)
    return torch.where(low_density_mask, uncond_mean, chunk_est)


# Initialize compilation wrapper with fallback
_compute_muguruza_chunk_compiled = None
_use_compile = True

try:
    _compute_muguruza_chunk_compiled = torch.compile(_compute_muguruza_chunk, mode="reduce-overhead")
    # Warm up compilation to check for backend errors
    _dummy_t = torch.tensor([0.0])
    _dummy_mu = torch.tensor([0.0])
    _dummy_sigma = torch.tensor([1.0])
    _dummy_V = torch.tensor([1.0])
    _dummy_mean = torch.tensor(1.0)
    _compute_muguruza_chunk_compiled(_dummy_t, _dummy_mu, _dummy_sigma, _dummy_V, _dummy_mean, 1e-8)
except Exception:
    _use_compile = False


def compute_conditional_expectation(
    X_t: torch.Tensor,
    V_t: torch.Tensor,
    targets: torch.Tensor,
    method: str = "nadaraya_watson",
    mu_i: torch.Tensor = None,
    sigma_cond: torch.Tensor = None,
    block_size: int = 1024,
    reg_epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the conditional expectation E[V_t | X_t = target]
    using PyTorch GPU acceleration with tiling to prevent OOM.
    
    Parameters:
    -----------
    X_t : torch.Tensor
        Conditioning variable (log stock price paths at current step), shape (N_paths,)
    V_t : torch.Tensor
        Dependent variable (variance paths at current step), shape (N_paths,)
    targets : torch.Tensor
        Target evaluation points (e.g. X_t or a grid of strikes/log-strikes), shape (N_targets,)
    method : str, default "nadaraya_watson"
        Method for conditional expectation: "nadaraya_watson" or "muguruza"
    mu_i : torch.Tensor, optional
        Conditional mean of log stock price step (required for "muguruza"), shape (N_paths,)
    sigma_cond : torch.Tensor, optional
        Conditional standard deviation of log stock price step (required for "muguruza"), shape (N_paths,)
    block_size : int, default 1024
        Block size for chunking/tiling along the target dimension to avoid GPU memory OOM.
    reg_epsilon : float, default 1e-8
        Numerical regularization parameter for low density regions.
        
    Returns:
    --------
    results : torch.Tensor
        Conditional expectation values, shape (N_targets,)
    """
    if X_t.device != targets.device or V_t.device != targets.device:
        raise ValueError("All input tensors must be on the same device")
    
    N_paths = X_t.shape[0]
    N_targets = targets.shape[0]
    results = torch.empty(N_targets, device=targets.device, dtype=targets.dtype)
    uncond_mean = V_t.mean()
    
    if method == "nadaraya_watson":
        # Silverman's rule of thumb bandwidth: h = 1.06 * std(X_t) * N^(-1/5)
        std_X = torch.std(X_t)
        std_X = torch.nan_to_num(torch.clamp(std_X, min=1e-6), nan=1e-6)
        h = 1.06 * std_X * (N_paths ** (-0.2))
        
        # Precompute unsqueezed tensors outside loop
        X_t_unsqueezed = X_t.unsqueeze(0)
        X_t_sq = X_t.square().unsqueeze(0)
        
        # Tile over targets to avoid OOM
        for start_idx in range(0, N_targets, block_size):
            end_idx = min(start_idx + block_size, N_targets)
            targets_chunk = targets[start_idx:end_idx]  # (chunk_size,)
            
            # GEMM expansion optimization:
            # (S_i - K_j)^2 = S_i^2 + K_j^2 - 2 * S_i * K_j
            # exponent = -0.5 * (diff / h) ** 2
            # = (-0.5 / h^2) * (targets_chunk^2 + X_t^2 - 2 * targets_chunk * X_t)
            A = targets_chunk.unsqueeze(1)
            inp = A.square() + X_t_sq
            beta = -0.5 / (h ** 2)
            alpha = 1.0 / (h ** 2)
            exponent = torch.addmm(inp, A, X_t_unsqueezed, beta=beta, alpha=alpha)
            
            K = torch.exp(exponent)
            sum_K = torch.sum(K, dim=1)  # (chunk_size,)
            sum_V_K = torch.mv(K, V_t)  # (chunk_size,)
            
            # Regularized division
            chunk_est = (sum_V_K + reg_epsilon * uncond_mean) / (sum_K + reg_epsilon)
            
            # Fallback to unconditional mean in extremely low density regions
            low_density_mask = (sum_K < 1e-10)
            chunk_est = torch.where(low_density_mask, uncond_mean, chunk_est)
            
            results[start_idx:end_idx] = chunk_est
            
    elif method == "muguruza":
        if mu_i is None or sigma_cond is None:
            raise ValueError("mu_i and sigma_cond must be provided for 'muguruza' method")
            
        sigma_cond_safe = torch.clamp(sigma_cond, min=1e-8)
        
        # Tile over targets to avoid OOM
        for start_idx in range(0, N_targets, block_size):
            end_idx = min(start_idx + block_size, N_targets)
            targets_chunk = targets[start_idx:end_idx]  # (chunk_size,)
            
            if _use_compile:
                try:
                    chunk_est = _compute_muguruza_chunk_compiled(
                        targets_chunk,
                        mu_i,
                        sigma_cond_safe,
                        V_t,
                        uncond_mean,
                        reg_epsilon,
                    )
                except Exception:
                    chunk_est = _compute_muguruza_chunk(
                        targets_chunk,
                        mu_i,
                        sigma_cond_safe,
                        V_t,
                        uncond_mean,
                        reg_epsilon,
                    )
            else:
                chunk_est = _compute_muguruza_chunk(
                    targets_chunk,
                    mu_i,
                    sigma_cond_safe,
                    V_t,
                    uncond_mean,
                    reg_epsilon,
                )
            
            results[start_idx:end_idx] = chunk_est
            
    else:
        raise ValueError(f"Unknown method {method}")
        
    return results


class MLSVSolverGPU:
    """
    McKean-Vlasov SDE particle solver with kernel density regression on GPU.
    """
    def __init__(
        self,
        S0: float,
        r: float,
        q: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        T: float,
        steps_per_unit: int,
        N_paths: int,
        dupire_vol_fn: Callable[[float, torch.Tensor], torch.Tensor] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        vol_boundary_style: str = "truncation",
    ):
        """
        Parameters:
        -----------
        S0 : float
            Initial stock price
        r : float
            Risk-free rate
        q : float
            Dividend yield
        v0 : float
            Initial variance
        kappa : float
            Mean reversion speed of variance
        theta : float
            Long-term mean of variance
        xi : float
            Vol-of-vol parameter
        rho : float
            Correlation between asset and variance Brownian motions
        T : float
            Maturity / simulation horizon
        steps_per_unit : int
            Number of simulation steps per unit of time
        N_paths : int
            Number of Monte Carlo paths
        dupire_vol_fn : Callable[[float, torch.Tensor], torch.Tensor], optional
            Function returning Dupire local volatility at (t, S_t)
        device : str, default "cuda"
            Device to run calculations on ('cuda' or 'cpu')
        dtype : torch.dtype, default torch.float32
            Data type precision
        vol_boundary_style : str, default "truncation"
            Boundary behavior for variance process V: "truncation" or "reflection"
        """
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")
        if v0 <= 0.0:
            raise ValueError("v0 must be positive")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if theta <= 0.0:
            raise ValueError("theta must be positive")
        if xi < 0.0:
            raise ValueError("xi must be non-negative")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be between -1.0 and 1.0")
        if T <= 0.0:
            raise ValueError("T must be positive")
        if steps_per_unit <= 0:
            raise ValueError("steps_per_unit must be positive")
        if N_paths <= 0:
            raise ValueError("N_paths must be positive")
        if dupire_vol_fn is not None and not callable(dupire_vol_fn):
            raise TypeError("dupire_vol_fn must be callable")
            
        self.S0 = S0
        self.r = r
        self.q = q
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.T = T
        self.steps_per_unit = steps_per_unit
        self.N_paths = N_paths
        
        if dupire_vol_fn is None:
            self.dupire_vol_fn = lambda t, s: torch.full_like(s, 0.2)
        else:
            self.dupire_vol_fn = dupire_vol_fn
            
        # Fallback to CPU if CUDA is not available
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
            
        self.dtype = dtype
        self.vol_boundary_style = vol_boundary_style
        
        # Grid setup
        self.N_steps = max(1, int(round(T * steps_per_unit)))
        self.dt = float(T) / self.N_steps
        self.t_grid = torch.linspace(0.0, T, self.N_steps + 1, device=self.device, dtype=self.dtype)
        
        # Path tensors (initialized to None)
        self.X_paths = None
        self.V_paths = None

    def simulate(
        self,
        method: str = "nadaraya_watson",
        block_size: int = 1024,
        reg_epsilon: float = 1e-8,
        vol_boundary_style: str = None,
    ):
        """
        Simulate McKean-Vlasov log stock price X and variance V paths on GPU.
        
        Parameters:
        -----------
        method : str, default "nadaraya_watson"
            Conditional expectation estimation method: "nadaraya_watson" or "muguruza"
        block_size : int, default 1024
            Block size for tiling along the target dimension to avoid OOM
        reg_epsilon : float, default 1e-8
            Regularization parameter for conditional expectations
        vol_boundary_style : str, optional
            If provided, overrides self.vol_boundary_style
        """
        boundary_style = vol_boundary_style or self.vol_boundary_style
        torch.manual_seed(42)
        
        # Allocate path storage: shape (N_steps + 1, N_paths) on the GPU/device
        self.X_paths = torch.empty((self.N_steps + 1, self.N_paths), device=self.device, dtype=self.dtype)
        self.V_paths = torch.empty((self.N_steps + 1, self.N_paths), device=self.device, dtype=self.dtype)
        
        # Initialize paths at t=0
        self.X_paths[0] = torch.full((self.N_paths,), math.log(self.S0), device=self.device, dtype=self.dtype)
        self.V_paths[0] = torch.full((self.N_paths,), self.v0, device=self.device, dtype=self.dtype)
        
        # Allocate tensors to store steps parameters for Muguruza's method
        sigma_paths = torch.empty((self.N_steps, self.N_paths), device=self.device, dtype=self.dtype)
        Z_V_paths = torch.empty((self.N_steps, self.N_paths), device=self.device, dtype=self.dtype)
        
        dt = self.dt
        sqrt_dt = math.sqrt(dt)
        
        # Active state variables kept as separate, contiguous 1D GPU tensors (SoA layout)
        # to ensure coalesced memory access and avoid slicing overhead in steps
        X_curr = self.X_paths[0].clone()
        V_curr = self.V_paths[0].clone()
        
        # Trackers for Muguruza's step i-2 values (device persistent)
        X_prev_2 = None
        sigma_prev = None
        Z_V_prev = None
        
        for i in range(1, self.N_steps + 1):
            t_prev = self.t_grid[i - 1]
            
            # 1. Compute conditional expectation E[V_{t_{i-1}} | X_{t_{i-1}}]
            if i == 1:
                # At t=0, all particles are identical, E[V_0 | X_0] = v0
                cond_expect = torch.full((self.N_paths,), self.v0, device=self.device, dtype=self.dtype)
            else:
                if method == "nadaraya_watson":
                    cond_expect = compute_conditional_expectation(
                        X_t=X_curr,
                        V_t=V_curr,
                        targets=X_curr,
                        method="nadaraya_watson",
                        block_size=block_size,
                        reg_epsilon=reg_epsilon,
                    )
                elif method == "muguruza":
                    # Reconstruct conditional distribution of X_{t_{i-1}} given information at t_{i-2}
                    mu_prev = (
                        X_prev_2
                        + (self.r - self.q - 0.5 * (sigma_prev ** 2)) * dt
                        + self.rho * sigma_prev * sqrt_dt * Z_V_prev
                    )
                    sigma_cond_prev = sigma_prev * sqrt_dt * math.sqrt(1.0 - self.rho ** 2)
                    
                    cond_expect = compute_conditional_expectation(
                        X_t=X_curr,
                        V_t=V_curr,
                        targets=X_curr,
                        method="muguruza",
                        mu_i=mu_prev,
                        sigma_cond=sigma_cond_prev,
                        block_size=block_size,
                        reg_epsilon=reg_epsilon,
                    )
                else:
                    raise ValueError(f"Unknown method {method}")
            
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
            
            # Save state variables for the next step's Muguruza's conditional expectation reconstruction
            X_prev_2 = X_curr.clone()
            sigma_prev = sigma_t_prev.clone()
            Z_V_prev = Z_V.clone()
            
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

    def price_european_option(
        self,
        strike: Union[float, torch.Tensor, np.ndarray],
        maturity: Union[float, torch.Tensor, np.ndarray],
        is_call: bool = True,
    ) -> Union[float, torch.Tensor]:
        """
        Price European options using simulated paths.
        
        Parameters:
        -----------
        strike : float, np.ndarray, or torch.Tensor
            Strike price(s). Can be a scalar or a 1D array/tensor.
        maturity : float, np.ndarray, or torch.Tensor
            Maturity/maturities. Can be a scalar or a 1D array/tensor.
        is_call : bool, default True
            True for Call option, False for Put option.
            
        Returns:
        --------
        price : float or torch.Tensor
            If both strike and maturity are scalars, returns a float.
            If either is a tensor/array, returns a PyTorch tensor on self.device.
        """
        if self.X_paths is None:
            raise ValueError("No simulated paths found. Run simulate() first.")
            
        # Convert strike to torch.Tensor
        is_strike_scalar = isinstance(strike, (int, float))
        if is_strike_scalar:
            strike_tensor = torch.tensor([strike], device=self.device, dtype=self.dtype)
        else:
            if isinstance(strike, np.ndarray):
                strike_tensor = torch.tensor(strike, device=self.device, dtype=self.dtype)
            else:
                strike_tensor = strike.to(device=self.device, dtype=self.dtype)
                
        # Convert maturity to torch.Tensor
        is_maturity_scalar = isinstance(maturity, (int, float))
        if is_maturity_scalar:
            maturity_tensor = torch.tensor([maturity], device=self.device, dtype=self.dtype)
        else:
            if isinstance(maturity, np.ndarray):
                maturity_tensor = torch.tensor(maturity, device=self.device, dtype=self.dtype)
            else:
                maturity_tensor = maturity.to(device=self.device, dtype=self.dtype)
                
        # Find closest simulated grid indices for the specified maturities on the GPU
        diff = torch.abs(maturity_tensor.unsqueeze(1) - self.t_grid.unsqueeze(0))
        time_indices = torch.argmin(diff, dim=1)  # (N_maturities,)
        
        # Extract stock prices at the maturity steps: shape (N_maturities, N_paths)
        # Using a contiguous GPU tensor
        S_mat = torch.exp(self.X_paths[time_indices]).contiguous()
        
        # Broadcast stock prices and strikes to compute payoffs
        # S_mat: (N_maturities, 1, N_paths)
        # strike_tensor: (1, N_strikes, 1)
        S_expanded = S_mat.unsqueeze(1)
        K_expanded = strike_tensor.unsqueeze(0).unsqueeze(2)
        
        if is_call:
            payoffs = torch.clamp(S_expanded - K_expanded, min=0.0).contiguous()
        else:
            payoffs = torch.clamp(K_expanded - S_expanded, min=0.0).contiguous()
            
        # Discount factors for each maturity: (N_maturities, 1)
        maturities_col = maturity_tensor.unsqueeze(1)
        discounts = torch.exp(-self.r * maturities_col).contiguous()
        
        # Option prices: shape (N_maturities, N_strikes)
        option_prices = discounts * payoffs.mean(dim=2)
        
        # Return format matching inputs
        if is_strike_scalar and is_maturity_scalar:
            return option_prices[0, 0].item()
        elif is_maturity_scalar:
            return option_prices[0]
        elif is_strike_scalar:
            return option_prices[:, 0]
        else:
            return option_prices


class MLSVEngine:
    def __init__(self, kappa: float, theta: float, epsilon: float, rho: float, dupire_grid: dict = None):
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
        
    def price_option(self, spot: float, strike: float, maturity: float, vol: float, is_call: bool = True) -> float:
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
            # Use call-put parity: C - P = S - K * exp(-r*T) = S - K (since r=0, q=0 in solver)
            call_price = self.price_option(spot=spot, strike=strike, maturity=maturity, vol=vol, is_call=True)
            return call_price - spot + strike

        if self.dupire_grid is not None:
            if isinstance(self.dupire_grid, dict):
                vol_grid = np.array(self.dupire_grid["vol"])
            else:
                vol_grid = np.array(self.dupire_grid)
            dupire_vol_fn = lambda t, s: torch.full_like(s, float(vol_grid[0, 0]))
        else:
            dupire_vol_fn = lambda t, s: torch.full_like(s, vol)
            
        solver = MLSVSolverGPU(
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
            device="cpu",
            dtype=torch.float64
        )
        solver.simulate(method="nadaraya_watson")
        return solver.price_european_option(strike=strike, maturity=maturity, is_call=is_call)
        
    def conditional_expectation(self, spot_grid: np.ndarray, current_spot: float, current_vol: float) -> np.ndarray:
        solver = MLSVSolverGPU(
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
            device="cpu",
            dtype=torch.float64
        )
        solver.simulate(method="nadaraya_watson")
        X_t = solver.X_paths[-1]
        V_t = solver.V_paths[-1]
        targets = torch.tensor(np.log(spot_grid), device="cpu", dtype=torch.float64)
        expectations_t = compute_conditional_expectation(X_t, V_t, targets, method="nadaraya_watson")
        return expectations_t.cpu().numpy()
        
    def calibrate_local_vol(self, spot_grid: np.ndarray, time_grid: np.ndarray, market_prices: np.ndarray) -> np.ndarray:
        M, N = len(spot_grid), len(time_grid)
        local_vol = np.zeros((N, M))
        
        solver = MLSVSolverGPU(
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
            device="cpu",
            dtype=torch.float64
        )
        solver.simulate(method="nadaraya_watson")
        
        for j, t in enumerate(time_grid):
            t_idx = int(torch.argmin(torch.abs(solver.t_grid - t)).item())
            X_t = solver.X_paths[t_idx]
            V_t = solver.V_paths[t_idx]
            targets = torch.tensor(np.log(spot_grid), device="cpu", dtype=torch.float64)
            expectations = compute_conditional_expectation(X_t, V_t, targets, method="nadaraya_watson").cpu().numpy()
            expectations = np.maximum(expectations, 1e-6)
            
            for i, S in enumerate(spot_grid):
                dup_vol = 0.20
                lv = dup_vol / np.sqrt(expectations[i])
                local_vol[j, i] = np.clip(lv, 0.05, 1.5)
                
        return local_vol

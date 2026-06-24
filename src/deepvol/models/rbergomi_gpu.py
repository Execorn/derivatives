"""
rbergomi_gpu.py — GPU-accelerated Monte Carlo pricing for the Rough Bergomi model.
Uses the Bennedsen, Lunde & Pakkanen (2017) hybrid scheme with F.conv1d for fast fBm simulation.
"""

import sys
if sys.version_info >= (3, 14):
    import os
    os.environ["NUMBA_DISABLE_JIT"] = "1"
import numpy as np
import torch
import torch.nn.functional as F
import py_vollib_vectorized


def simulate_rbergomi_paths(
    params: torch.Tensor,
    T: float,
    steps_per_unit: int = 200,
    N_paths: int = 10000,
    antithetic: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """
    Simulates Rough Bergomi stock and variance paths on GPU using Bennedsen hybrid scheme
    and Euler step on log-price.

    Parameters:
      params : torch.Tensor of shape (B, 4) containing [v0, H, eta, rho]
      T : float, maximum maturity to simulate
      steps_per_unit : int, number of steps per unit of time
      N_paths : int, total number of paths (must be even if antithetic=True)
      antithetic : bool, whether to use antithetic variables
      device : str, device to run on
      dtype : torch.dtype, data precision

    Returns:
      S : torch.Tensor of shape (B, N_paths, N_t + 1), stock price paths
      V : torch.Tensor of shape (B, N_paths, N_t + 1), variance paths
      t_grid : torch.Tensor of shape (N_t + 1,), time grid
    """
    assert (params[:, 0] > 0).all(), "v0 must be > 0"
    assert ((params[:, 1] > 0) & (params[:, 1] < 0.5)).all(), "H must be in (0, 0.5)"
    assert (params[:, 2] > 0).all(), "eta must be > 0"
    assert ((params[:, 3] >= -1) & (params[:, 3] <= 0)).all(), "rho must be in [-1, 0]"
    B = params.shape[0]
    v0 = params[:, 0:1].unsqueeze(-1).to(device=device, dtype=dtype)   # (B, 1, 1)
    H = params[:, 1:2].unsqueeze(-1).to(device=device, dtype=dtype)    # (B, 1, 1)
    eta = params[:, 2:3].unsqueeze(-1).to(device=device, dtype=dtype)  # (B, 1, 1)
    rho = params[:, 3:4].unsqueeze(-1).to(device=device, dtype=dtype)  # (B, 1, 1)

    dt = 1.0 / steps_per_unit
    N_t = int(round(T * steps_per_unit))

    # 1. Generate normal random variables
    if antithetic:
        half_paths = N_paths // 2
        Z1_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)
        Z2_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)
        Z3_half = torch.randn(B, half_paths, N_t, device=device, dtype=dtype)

        Z1 = torch.cat([Z1_half, -Z1_half], dim=1)   # (B, N_paths, N_t)
        Z2 = torch.cat([Z2_half, -Z2_half], dim=1)
        Z3 = torch.cat([Z3_half, -Z3_half], dim=1)
    else:
        Z1 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)
        Z2 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)
        Z3 = torch.randn(B, N_paths, N_t, device=device, dtype=dtype)

    # 2. Hybrid scheme kernel for convolution (regular part)
    # w_k = dt^H * ((k+1)^(H+0.5) - k^(H+0.5)) / (H+0.5) for k = 1, ..., N_t-1
    k_vec = torch.arange(1, N_t, device=device, dtype=dtype).unsqueeze(0)   # (1, N_t - 1)
    H_2d = H.squeeze(-1)   # (B, 1)
    w = (dt ** H_2d) * ((k_vec + 1) ** (H_2d + 0.5) - k_vec ** (H_2d + 0.5)) / (H_2d + 0.5)  # (B, N_t - 1)

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    w_full = torch.cat([zeros, w], dim=1)   # (B, N_t)
    w_rev = torch.flip(w_full, dims=[1]).unsqueeze(1)   # (B, 1, N_t)

    # 3. Perform FFT-based causal convolution
    # Z1 shape: (B, N_paths, N_t)
    # w_full shape: (B, N_t)
    fft_len = 2 * N_t
    Z1_fft = torch.fft.rfft(Z1, n=fft_len, dim=-1)
    w_fft = torch.fft.rfft(w_full, n=fft_len, dim=-1)
    conv_out = torch.fft.irfft(Z1_fft * w_fft.unsqueeze(1), n=fft_len, dim=-1)[..., :N_t]

    # 4. Singular components
    # c1 = 1.0 / (H + 0.5)
    # c2 = sqrt(1.0 / (2H) - 1.0 / (H + 0.5)^2)
    c1 = 1.0 / (H + 0.5)   # (B, 1, 1)
    c2 = torch.sqrt(1.0 / (2.0 * H) - 1.0 / ((H + 0.5) ** 2))   # (B, 1, 1)

    Y = torch.sqrt(2.0 * H) * (conv_out + (dt ** H) * (c1 * Z1 + c2 * Z2))   # (B, N_paths, N_t)

    # Prepend Y_0 = 0
    zeros_Y = torch.zeros(B, N_paths, 1, device=device, dtype=dtype)
    Y_full = torch.cat([zeros_Y, Y], dim=2)   # (B, N_paths, N_t + 1)

    # 5. Compute variance paths
    t_grid = torch.arange(0, N_t + 1, device=device, dtype=dtype) * dt   # (N_t + 1,)
    t_grid_expanded = t_grid.view(1, 1, N_t + 1)   # (1, 1, N_t + 1)

    # V_t = v0 * exp(eta * Y_t - 0.5 * eta^2 * t^(2H))
    V = v0 * torch.exp(eta * Y_full - 0.5 * (eta ** 2) * (t_grid_expanded ** (2.0 * H)))

    # 6. Log-price paths
    dB = torch.sqrt(torch.tensor(dt, device=device, dtype=dtype)) * (
        rho * Z1 + torch.sqrt(1.0 - rho ** 2) * Z3
    )   # (B, N_paths, N_t)

    # dx = -0.5 * V_{t_{j-1}} * dt + sqrt(V_{t_{j-1}}) * dB_j
    dx = -0.5 * V[:, :, :-1] * dt + torch.sqrt(V[:, :, :-1]) * dB
    x = torch.cat(
        [
            torch.zeros(B, N_paths, 1, device=device, dtype=dtype),
            torch.cumsum(dx, dim=2),
        ],
        dim=2,
    )

    S = torch.exp(x)

    return S, V, t_grid


def fill_nans(ivs: np.ndarray, default_val: float = 0.3) -> np.ndarray:
    """
    Interpolates or fills NaN values in IV surfaces.
    """
    B, M, L = ivs.shape
    for b in range(B):
        for m in range(M):
            slice_ml = ivs[b, m, :]
            nans = np.isnan(slice_ml)
            if nans.any():
                if not nans.all():
                    # Linear interpolation along strikes
                    xp = np.where(~nans)[0]
                    fp = slice_ml[~nans]
                    x_nan = np.where(nans)[0]
                    slice_ml[nans] = np.interp(x_nan, xp, fp)
                else:
                    # Look at other maturities of the same sample
                    other_slices = ivs[b, :, :]
                    valid_mask = ~np.isnan(other_slices).any(axis=1)
                    if valid_mask.any():
                        slice_ml[:] = other_slices[valid_mask].mean()
                    else:
                        slice_ml[:] = default_val

    # Global fallback if any NaNs remain
    ivs[np.isnan(ivs)] = default_val
    return ivs


@torch.no_grad()
def batch_rbergomi_iv_surface(
    params,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    N_paths: int = 10000,
    antithetic: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    Prices options for a batch of Rough Bergomi parameters and inverts them to get Black-Scholes IVs.

    Parameters:
      params : np.ndarray or torch.Tensor of shape (B, 4) containing [v0, H, eta, rho]
      T_grid : np.ndarray, maturities
      K_grid : np.ndarray, log-strikes
      N_paths : int, number of Monte Carlo paths
      antithetic : bool, whether to use antithetic variables
      device : str, device to use for PyTorch
      dtype : torch.dtype, data precision

    Returns:
      ivs : np.ndarray of shape (B, len(T_grid), len(K_grid))
    """
    if isinstance(params, np.ndarray):
        params_t = torch.tensor(params, device=device, dtype=dtype)
    else:
        params_t = params.to(device=device, dtype=dtype)

    B = params_t.shape[0]
    M = len(T_grid)
    L = len(K_grid)
    T_max = float(max(T_grid))

    # Adaptive step count: 500 for H < 0.07, 200 for H >= 0.07
    H_vals = params_t[:, 1].cpu().numpy()
    idx_500 = np.where(H_vals < 0.07)[0]
    idx_200 = np.where(H_vals >= 0.07)[0]

    prices = torch.zeros((B, M, L), device=device, dtype=dtype)
    def price_subbatch(sub_indices, steps_per_unit):
        if len(sub_indices) == 0:
            return
        chunk_size = max(1, 40000 // N_paths)
        for i in range(0, len(sub_indices), chunk_size):
            chunk_idx = sub_indices[i:i+chunk_size]
            sub_params = params_t[chunk_idx]
            S, _, _ = simulate_rbergomi_paths(
                sub_params,
                T_max,
                steps_per_unit=steps_per_unit,
                N_paths=N_paths,
                antithetic=antithetic,
                device=device,
                dtype=dtype,
            )

            step_indices = [int(round(T * steps_per_unit)) for T in T_grid]
            S_maturities = S[:, :, step_indices].permute(0, 2, 1)  # (chunk_B, M, N_paths)

            K_tensor = torch.exp(torch.tensor(K_grid, device=device, dtype=dtype))  # (L,)
            is_call = (K_tensor >= 1.0).view(1, 1, L, 1)  # (1, 1, L, 1)

            payoffs = torch.where(
                is_call,
                torch.clamp(S_maturities.unsqueeze(2) - K_tensor.view(1, 1, L, 1), min=0.0),
                torch.clamp(K_tensor.view(1, 1, L, 1) - S_maturities.unsqueeze(2), min=0.0),
            )  # (chunk_B, M, L, N_paths)
            prices[chunk_idx] = payoffs.mean(dim=-1)

    price_subbatch(idx_500, 500)
    price_subbatch(idx_200, 200)
    # Black-Scholes inversion using py_vollib_vectorized
    S0 = 1.0
    from deepvol.models.heston import bs_iv_gpu
    K_tensor = torch.exp(torch.tensor(K_grid, device=device, dtype=torch.float64))
    T_tensor = torch.tensor(T_grid, device=device, dtype=torch.float64)

    ivs_gpu = bs_iv_gpu(prices.double(), float(S0), K_tensor, T_tensor)
    ivs = ivs_gpu.cpu().numpy()
    ivs = fill_nans(ivs)
    return ivs


def rbergomi_iv_surface(
    v0: float,
    H: float,
    eta: float,
    rho: float,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    N_paths: int = 10000,
    antithetic: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    Computes a single Rough Bergomi implied volatility surface.
    """
    params = np.array([[v0, H, eta, rho]])
    ivs = batch_rbergomi_iv_surface(
        params,
        T_grid,
        K_grid,
        N_paths=N_paths,
        antithetic=antithetic,
        device=device,
        dtype=dtype,
    )
    return ivs[0]


class rBergomiEngine:
    def simulate_paths(self, H, eta, T, N_steps, N_paths, antithetic=True, device="cpu"):
        return simulate_rbergomi_paths(H, eta, T, N_steps, N_paths, antithetic, device)
        
    def price_surface(self, v0, H, eta, rho, T_grid, K_grid, N_paths=10000, antithetic=True, device="cpu") -> np.ndarray:
        return rbergomi_iv_surface(v0, H, eta, rho, T_grid, K_grid, N_paths, antithetic, device)
        
    def batch_price_surface(self, params, T_grid, K_grid, N_paths=10000, antithetic=True, device="cpu") -> np.ndarray:
        return batch_rbergomi_iv_surface(params, T_grid, K_grid, N_paths, antithetic, device)


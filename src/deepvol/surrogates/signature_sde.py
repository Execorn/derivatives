"""
src/deepvol/surrogates/signature_sde.py — Path-dependent Signature Neural SDE model and Dual-Leg FNO surrogate.

Mathematical Formulations:
-------------------------
1. Signature Neural SDE (Ito SDE driven by path signatures):
   dX_t = (r - q - 0.5 * V_t) dt + sqrt(V_t) * (rho * dW_t^1 + sqrt(1 - rho^2) * dW_t^2)
   dV_t = (W_drift * S(Y)_t + b_drift) dt + Softplus(W_diff * S(Y)_t + b_diff) dW_t^1
   where Y_t = (X_t, V_t)^T is the log-spot and variance process, and S(Y)_t is the truncated level-3 path signature.

   Ref: Cuchiero, C., Larsson, M., & Teichmann, J. (2020). "SDEs driven by path signatures." arXiv preprint arXiv:2006.00222.
   Ref: Lyons, T., & Qian, Z. (2002). "System Control and Rough Paths." Oxford University Press.

2. Population Stability Index (PSI):
   PSI = sum_{bin} (Actual% - Expected%) * ln(Actual% / Expected%)

   Ref: SR 26-2 Model Risk Governance guidelines.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from deepvol.surrogates.fno_model import SpectralConv2d, FiLMGenerator

# Configure logging
logger = logging.getLogger("deepvol.surrogates.signature_sde")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ─── Path Signature Utilities ──────────────────────────────────────────────────

def compute_path_signature(path: torch.Tensor, depth: int = 3) -> torch.Tensor:
    """
    Computes the path signature of a batch of D-dimensional paths using Chen's relation.
    Ensures full differentiability and GPU acceleration.
    
    Args:
        path: Tensor of shape (B, L, D) where B is batch size, L is path length, D is dimension.
        depth: Signature depth (supports up to 3).
        
    Returns:
        Tensor of shape (B, N_features) containing concatenated signature levels.
    """
    B, L, D = path.shape
    device = path.device
    dtype = path.dtype
    
    if L < 2:
        num_features = sum(D ** i for i in range(1, depth + 1))
        return torch.zeros(B, num_features, device=device, dtype=dtype)
        
    deltas = path[:, 1:, :] - path[:, :-1, :]  # (B, L-1, D)
    
    S1 = torch.zeros(B, D, device=device, dtype=dtype)
    S2 = torch.zeros(B, D, D, device=device, dtype=dtype)
    S3 = torch.zeros(B, D, D, D, device=device, dtype=dtype)
    
    for step in range(L - 1):
        delta = deltas[:, step, :]  # (B, D)
        A1 = delta
        
        if depth >= 2:
            A2 = 0.5 * torch.einsum('bi,bj->bij', delta, delta)
        if depth >= 3:
            A3 = (1.0 / 6.0) * torch.einsum('bi,bj,bk->bijk', delta, delta, delta)
            
        if depth >= 3:
            S3 = (S3 + 
                  torch.einsum('bij,bk->bijk', S2, A1) + 
                  torch.einsum('bi,bjk->bijk', S1, A2) + 
                  A3)
        if depth >= 2:
            S2 = S2 + torch.einsum('bi,bj->bij', S1, A1) + A2
            
        S1 = S1 + A1
        
    features = []
    if depth >= 1:
        features.append(S1.reshape(B, -1))
    if depth >= 2:
        features.append(S2.reshape(B, -1))
    if depth >= 3:
        features.append(S3.reshape(B, -1))
        
    return torch.cat(features, dim=-1)


# ─── Compiled Euler step helper ────────────────────────────────────────────────

@torch.compile(mode="reduce-overhead")
def sim_step_compiled(
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    X: torch.Tensor,
    V: torch.Tensor,
    dW1: torch.Tensor,
    dW2: torch.Tensor,
    W_drift: torch.Tensor,
    b_drift: torch.Tensor,
    W_diff: torch.Tensor,
    b_diff: torch.Tensor,
    rho: torch.Tensor,
    r: float,
    q: float,
    dt: float,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes a single step of the Euler-Maruyama simulation.
    Uses Structure of Arrays (SoA) layout.
    """
    B = X.shape[0]
    
    # 1. Concatenate signature levels
    sig = torch.cat([S1, S2.reshape(B, 4), S3.reshape(B, 8)], dim=-1)
    
    # 2. Compute drift & diffusion drivers (linear in signature)
    drift_v = torch.matmul(sig, W_drift) + b_drift
    drift_v = torch.clamp(drift_v, -50.0, 50.0)  # Bound drift growth for stability
    
    diff_v = F.softplus(torch.matmul(sig, W_diff) + b_diff) + epsilon
    
    # 3. Update variance
    V_next = V + drift_v * dt + diff_v * dW1
    V_next = torch.clamp(V_next, min=epsilon)
    
    # 4. Update log stock returns
    drift_x = (r - q - 0.5 * V) * dt
    diffusion_x = torch.sqrt(V) * (rho * dW1 + torch.sqrt(1.0 - rho**2) * dW2)
    X_next = X + drift_x + diffusion_x
    
    # 5. Compute path increments
    dX = X_next - X
    dV = V_next - V
    delta = torch.stack([dX, dV], dim=1)  # (B, 2)
    
    # 6. Update signature recursively via Chen's relation
    A1 = delta
    A2 = 0.5 * torch.einsum('bi,bj->bij', delta, delta)
    A3 = (1.0 / 6.0) * torch.einsum('bi,bj,bk->bijk', delta, delta, delta)
    
    S3_next = S3 + torch.einsum('bij,bk->bijk', S2, A1) + torch.einsum('bi,bjk->bijk', S1, A2) + A3
    S2_next = S2 + torch.einsum('bi,bj->bij', S1, A1) + A2
    S1_next = S1 + A1
    
    # Clone outputs to prevent static buffer overwrites under CUDAGraphs
    return S1_next.clone(), S2_next.clone(), S3_next.clone(), X_next.clone(), V_next.clone()


# ─── Signature Neural SDE Solver ───────────────────────────────────────────────

class SignatureNeuralSDE(nn.Module):
    """
    Path-dependent Signature Neural SDE model.
    Internal computations are done in double precision (float64) for pricing layers.
    """
    def __init__(self, r: float = 0.0, q: float = 0.0, epsilon: float = 1e-4):
        super().__init__()
        self.r = r
        self.q = q
        self.epsilon = epsilon
        
        # Signature linear coefficients (dim = 14 for 2D path level 3 signature)
        self.W_drift = nn.Parameter(torch.zeros(14, dtype=torch.float64))
        self.b_drift = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        self.W_diff = nn.Parameter(torch.zeros(14, dtype=torch.float64))
        self.b_diff = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        # Log-v0 and logit-rho parameters
        self.raw_v0 = nn.Parameter(torch.tensor(np.log(0.04), dtype=torch.float64))
        self.raw_rho = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
    @property
    def v0(self) -> torch.Tensor:
        return torch.exp(self.raw_v0)
        
    @property
    def rho(self) -> torch.Tensor:
        # Enforce range [-0.95, -0.05]
        return -0.05 - 0.90 * torch.sigmoid(self.raw_rho)
        
    def simulate_paths(
        self,
        T: float,
        steps_per_unit: int,
        N_paths: int,
        S0: float = 1.0,
        antithetic: bool = True,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate joint stock price and variance paths using Euler-Maruyama.
        """
        N_steps = int(round(T * steps_per_unit))
        dt = 1.0 / steps_per_unit
        sqrt_dt = np.sqrt(dt)
        
        # 1. Initialize states in double precision
        S1 = torch.zeros(N_paths, 2, device=device, dtype=torch.float64)
        S2 = torch.zeros(N_paths, 2, 2, device=device, dtype=torch.float64)
        S3 = torch.zeros(N_paths, 2, 2, 2, device=device, dtype=torch.float64)
        
        X = torch.zeros(N_paths, device=device, dtype=torch.float64)
        V = self.v0.expand(N_paths).to(device)
        
        # 2. Setup Brownian increments
        if antithetic:
            half = N_paths // 2
            Z1_half = torch.randn(half, N_steps, device=device, dtype=torch.float64)
            Z2_half = torch.randn(half, N_steps, device=device, dtype=torch.float64)
            Z1 = torch.cat([Z1_half, -Z1_half], dim=0)
            Z2 = torch.cat([Z2_half, -Z2_half], dim=0)
        else:
            Z1 = torch.randn(N_paths, N_steps, device=device, dtype=torch.float64)
            Z2 = torch.randn(N_paths, N_steps, device=device, dtype=torch.float64)
            
        dW1 = Z1 * sqrt_dt
        dW2 = Z2 * sqrt_dt
        
        X_list = [X.clone()]
        V_list = [V.clone()]
        
        # 3. Simulate step-by-step
        for i in range(N_steps):
            S1, S2, S3, X, V = sim_step_compiled(
                S1.clone(), S2.clone(), S3.clone(), X.clone(), V.clone(),
                dW1[:, i], dW2[:, i],
                self.W_drift, self.b_drift,
                self.W_diff, self.b_diff,
                self.rho, self.r, self.q, dt, self.epsilon
            )
            X_list.append(X.clone())
            V_list.append(V.clone())
            
        X_stacked = torch.stack(X_list, dim=1)
        V_stacked = torch.stack(V_list, dim=1)
        t_grid = torch.linspace(0.0, T, N_steps + 1, device=device, dtype=torch.float64)
        
        S_stacked = S0 * torch.exp(X_stacked)
        return S_stacked, V_stacked, t_grid


# ─── Option Pricing & Implied Volatility Inversion ─────────────────────────────

def black_scholes_call(
    S0: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """Standard Black-Scholes European call formula."""
    S0 = S0.to(torch.float64)
    K = K.to(torch.float64)
    T = T.to(torch.float64)
    r = r.to(torch.float64)
    sigma = sigma.to(torch.float64)
    
    d1 = (torch.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + 1e-12)
    d2 = d1 - sigma * torch.sqrt(T)
    
    normal_cdf = lambda x: 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    return S0 * normal_cdf(d1) - K * torch.exp(-r * T) * normal_cdf(d2)


def black_scholes_vega(
    S0: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """Vega (derivative w.r.t volatility) of European call."""
    d1 = (torch.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + 1e-12)
    normal_pdf = lambda x: torch.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    return S0 * torch.sqrt(T) * normal_pdf(d1)


def implied_volatility(
    S0: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    market_price: torch.Tensor,
    max_iter: int = 15,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Vectorized and differentiable implied volatility solver.
    Operates strictly in double precision (float64) and clamps minimum volatility to 0.01.
    """
    S0 = S0.to(torch.float64)
    K = K.to(torch.float64)
    T = T.to(torch.float64)
    r = r.to(torch.float64)
    market_price = market_price.to(torch.float64)
    
    low = torch.full_like(market_price, 1e-4, dtype=torch.float64)
    high = torch.full_like(market_price, 5.0, dtype=torch.float64)
    
    # 1. Bisection search to initialize
    for _ in range(8):
        mid = 0.5 * (low + high)
        p = black_scholes_call(S0, K, T, r, mid)
        low = torch.where(p < market_price, mid, low)
        high = torch.where(p >= market_price, mid, high)
        
    sigma = 0.5 * (low + high)
    
    # 2. Newton-Raphson iterations
    for _ in range(8):
        p = black_scholes_call(S0, K, T, r, sigma)
        v = black_scholes_vega(S0, K, T, r, sigma)
        diff = p - market_price
        
        step = diff / (v + 1e-8)
        step = torch.clamp(step, -0.2, 0.2)
        sigma = sigma - step
        sigma = torch.clamp(sigma, min=1e-4, max=5.0)
        
    # Enforce minimum volatility clamping to prevent Durrleman singularities
    sigma = torch.clamp(sigma, min=0.01)
    return sigma


def vix_implied_volatility(
    F_vix: torch.Tensor,
    K_vix: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    option_price: torch.Tensor,
    max_iter: int = 15,
    tol: float = 1e-6
) -> torch.Tensor:
    """Inverts Black formula for VIX options to retrieve implied volatility."""
    discounted_price = option_price * torch.exp(r * T)
    return implied_volatility(F_vix, K_vix, T, torch.zeros_like(r), discounted_price, max_iter, tol)


class SignatureSDEPricer(nn.Module):
    """
    Pricer that simulates paths under SignatureNeuralSDE and prices SPX and VIX options.
    """
    def __init__(self, sde: SignatureNeuralSDE):
        super().__init__()
        self.sde = sde
        
    def price_spx_options(
        self,
        S0: float,
        strikes: torch.Tensor,
        maturities: torch.Tensor,
        N_paths: int = 2048,
        steps_per_unit: int = 100,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Prices European calls and returns implied volatilities in double precision."""
        # Find maximum maturity to simulate
        T_max = float(torch.max(maturities).item())
        
        # Simulate paths
        S_paths, _, t_grid = self.sde.simulate_paths(
            T=T_max, steps_per_unit=steps_per_unit, N_paths=N_paths, S0=S0, device=device
        )
        
        prices = []
        for i in range(len(maturities)):
            T_i = maturities[i].item()
            K_i = strikes[i].item()
            
            # Find closest index on time grid
            t_idx = int(torch.argmin(torch.abs(t_grid - T_i)).item())
            S_T = S_paths[:, t_idx]
            
            payoff = torch.clamp(S_T - K_i, min=0.0)
            price = payoff.mean() * np.exp(-self.sde.r * T_i)
            prices.append(price)
            
        prices_t = torch.stack(prices)
        
        # Invert to implied volatilities
        S0_t = torch.full_like(prices_t, S0)
        r_t = torch.full_like(prices_t, self.sde.r)
        
        ivs = implied_volatility(S0_t, strikes, maturities, r_t, prices_t)
        return ivs
        
    def price_vix_options(
        self,
        strikes: torch.Tensor,
        maturities: torch.Tensor,
        N_paths: int = 2048,
        steps_per_unit: int = 100,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prices VIX call options and returns VIX futures and implied volatilities."""
        T_max = float(torch.max(maturities).item())
        
        # Simulate paths
        _, V_paths, t_grid = self.sde.simulate_paths(
            T=T_max, steps_per_unit=steps_per_unit, N_paths=N_paths, device=device
        )
        
        prices = []
        futures = []
        for i in range(len(maturities)):
            T_i = maturities[i].item()
            K_i = strikes[i].item()
            
            t_idx = int(torch.argmin(torch.abs(t_grid - T_i)).item())
            V_T = V_paths[:, t_idx]
            
            # VIX index pathwise approximation: VIX = sqrt(V) * 100
            VIX_T = torch.sqrt(torch.clamp(V_T, min=self.sde.epsilon)) * 100.0
            
            F_vix = VIX_T.mean()
            payoff = torch.clamp(VIX_T - K_i, min=0.0)
            price = payoff.mean() * np.exp(-self.sde.r * T_i)
            
            prices.append(price)
            futures.append(F_vix)
            
        prices_t = torch.stack(prices)
        futures_t = torch.stack(futures)
        r_t = torch.full_like(prices_t, self.sde.r)
        
        ivs = vix_implied_volatility(futures_t, strikes, maturities, r_t, prices_t)
        return futures_t, ivs


# ─── Dual-Leg Fourier Neural Operator (FNO) Surrogate ────────────────────────

class DualLegSignatureFNO2d(nn.Module):
    """
    Dual-Leg FNO model mapping Signature SDE parameters (dim 32)
    to both SPX implied volatility surface and VIX option smile simultaneously.
    """
    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 6,
        width: int = 40,
        spatial_in_channels: int = 2,
        param_dim: int = 32
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        N_LAYERS = 4
        
        # FiLM parameter routing generator: mapping signature parameters to layer modulations
        self.film = FiLMGenerator(
            param_dim=param_dim, hidden=128, width=width, n_layers=N_LAYERS
        )
        
        # Coordinate projection
        self.p = nn.Linear(spatial_in_channels, width)
        
        # Spectral layers
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        
        # Residual pointwise layers
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        # Leg 1: SPX option smile projection
        self.q_spx = nn.Linear(width, 1)
        
        # Leg 2: VIX option smile projection
        self.q_vix = nn.Linear(width, 1)
        
    @staticmethod
    def _film_modulate(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return F.elu(g * h + b)
        
    def forward(
        self,
        spatial_spx: torch.Tensor,
        spatial_vix: torch.Tensor,
        theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_spx: (B, T_spx, K_spx, 2) normalized coordinate grids for SPX
            spatial_vix: (B, T_vix, K_vix, 2) normalized coordinate grids for VIX
            theta: (B, 32) parameter vector of Signature Neural SDE
            
        Returns:
            Tuple of:
                spx_iv_surface: (B, T_spx, K_spx)
                vix_iv_smile: (B, T_vix, K_vix)
        """
        # Generate FiLM modulation vectors
        gamma, beta = self.film(theta)
        
        # ── 1. SPX surface processing ──
        original_T_spx = spatial_spx.size(1)
        x_spx_mirrored = torch.flip(spatial_spx, dims=[1])
        x_spx_ext = torch.cat([spatial_spx, x_spx_mirrored], dim=1)  # (B, 2T, K, 2)
        
        x_spx_ext = self.p(x_spx_ext).permute(0, 3, 1, 2)  # (B, width, 2T, K)
        x_spx_ext = self._film_modulate(self.conv0(x_spx_ext) + self.w0(x_spx_ext), gamma[:, 0], beta[:, 0])
        x_spx_ext = self._film_modulate(self.conv1(x_spx_ext) + self.w1(x_spx_ext), gamma[:, 1], beta[:, 1])
        x_spx_ext = self._film_modulate(self.conv2(x_spx_ext) + self.w2(x_spx_ext), gamma[:, 2], beta[:, 2])
        x_spx_ext = self._film_modulate(self.conv3(x_spx_ext) + self.w3(x_spx_ext), gamma[:, 3], beta[:, 3])
        
        x_spx_ext = x_spx_ext.permute(0, 2, 3, 1)
        out_spx = self.q_spx(x_spx_ext)
        out_spx = out_spx[:, :original_T_spx, :, :].squeeze(-1)
        
        # ── 2. VIX smile processing ──
        original_T_vix = spatial_vix.size(1)
        x_vix_mirrored = torch.flip(spatial_vix, dims=[1])
        x_vix_ext = torch.cat([spatial_vix, x_vix_mirrored], dim=1)
        
        x_vix_ext = self.p(x_vix_ext).permute(0, 3, 1, 2)
        x_vix_ext = self._film_modulate(self.conv0(x_vix_ext) + self.w0(x_vix_ext), gamma[:, 0], beta[:, 0])
        x_vix_ext = self._film_modulate(self.conv1(x_vix_ext) + self.w1(x_vix_ext), gamma[:, 1], beta[:, 1])
        x_vix_ext = self._film_modulate(self.conv2(x_vix_ext) + self.w2(x_vix_ext), gamma[:, 2], beta[:, 2])
        x_vix_ext = self._film_modulate(self.conv3(x_vix_ext) + self.w3(x_vix_ext), gamma[:, 3], beta[:, 3])
        
        x_vix_ext = x_vix_ext.permute(0, 2, 3, 1)
        out_vix = self.q_vix(x_vix_ext)
        out_vix = out_vix[:, :original_T_vix, :, :].squeeze(-1)
        
        return out_spx, out_vix


# ─── SR 26-2 Model Risk Guardian ───────────────────────────────────────────────

class ModelRiskGuardian:
    """
    SR 26-2 Model Risk Guardian.
    Tracks parameter drift using Population Stability Index (PSI),
    detects out-of-distribution (OOD) parameter inputs, and routes to robust fallbacks.
    """
    def __init__(self, expected_prior: np.ndarray, param_names: Optional[list] = None):
        self.expected_prior = expected_prior
        if param_names is None:
            self.param_names = [f"param_{i}" for i in range(len(expected_prior))]
        else:
            self.param_names = param_names
            
    def compute_psi(self, actual: np.ndarray, expected: np.ndarray, num_bins: int = 10) -> float:
        """Computes the Population Stability Index (PSI) between two samples."""
        combined = np.concatenate([actual, expected])
        bins = np.percentile(combined, np.linspace(0, 100, num_bins + 1))
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        
        act_counts, _ = np.histogram(actual, bins=bins)
        exp_counts, _ = np.histogram(expected, bins=bins)
        
        act_pcts = act_counts / max(len(actual), 1)
        exp_pcts = exp_counts / max(len(expected), 1)
        
        # Smoothen counts to prevent infinite log values
        act_pcts = np.where(act_pcts == 0, 1e-4, act_pcts)
        exp_pcts = np.where(exp_pcts == 0, 1e-4, exp_pcts)
        
        act_pcts /= act_pcts.sum()
        exp_pcts /= exp_pcts.sum()
        
        psi = np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts))
        return float(psi)
        
    def log_and_verify_drift(self, calibrated_history: np.ndarray) -> Dict[str, float]:
        """
        Verifies drift for all linear drivers and parameter coefficients.
        Logs violations under SR 26-2.
        """
        results = {}
        for idx, name in enumerate(self.param_names):
            actual = calibrated_history[:, idx]
            expected = np.random.normal(self.expected_prior[idx], 0.05, size=1000)
            psi = self.compute_psi(actual, expected)
            results[name] = psi
            
            if psi > 0.25:
                logger.warning(
                    f"[SR 26-2 COMPLIANCE VIOLATION] Parameter '{name}' drift detected: PSI = {psi:.4f} > 0.25 threshold."
                )
            elif psi > 0.1:
                logger.info(
                    f"[SR 26-2 DRIFT ALERT] Parameter '{name}' moderate drift: PSI = {psi:.4f}."
                )
        return results
        
    def detect_ood_and_clamp(self, params: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Detects out-of-distribution parameters and clamps them to stable bounds.
        """
        is_ood = False
        clamped_params = params.clone()
        
        # Check initial variance bounds
        v0 = params[0]
        if v0 < 0.001 or v0 > 0.40:
            is_ood = True
            clamped_params[0] = torch.clamp(v0, 0.005, 0.35)
            logger.warning(f"[SR 26-2 OOD DETECTED] v0={v0.item():.4f} is OOD. Clamping to {clamped_params[0].item():.4f}.")
            
        # Check correlation bounds
        rho = params[1]
        if rho < -0.99 or rho > 0.0:
            is_ood = True
            clamped_params[1] = torch.clamp(rho, -0.95, -0.05)
            logger.warning(f"[SR 26-2 OOD DETECTED] rho={rho.item():.4f} is OOD. Clamping to {clamped_params[1].item():.4f}.")
            
        return clamped_params, is_ood
        
    def route_to_fallback(
        self,
        S0: float,
        strikes: torch.Tensor,
        maturities: torch.Tensor,
        r: float
    ) -> torch.Tensor:
        """
        Routes the option pricing request to a fallback analytical solver (Black-Scholes with flat vol).
        Prevents SDE system crashes under extreme stress.
        """
        logger.error("[SR 26-2 GUARDIAN ACTION] Anomaly triggered! Routing request to analytical fallback solver.")
        # Fallback to flat Black-Scholes pricing with vol=0.20
        flat_vol = torch.full_like(strikes, 0.20, dtype=torch.float64)
        S0_t = torch.full_like(strikes, S0, dtype=torch.float64)
        r_t = torch.full_like(strikes, r, dtype=torch.float64)
        return black_scholes_call(S0_t, strikes, maturities, r_t, flat_vol)

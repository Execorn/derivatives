"""
neural_sde.py — Non-parametric Neural SDE model prior for option pricing.
Parameterizes variance drift and diffusion as MLPs, and solves paths using torchsde.
"""

import torch
import torch.nn as nn
import torchsde


class DriftMLP(nn.Module):
    """
    MLP representing variance drift f_theta(t, V_t).
    Output is in R.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Ensure t matches the batch size of v
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=v.dtype, device=v.device)
        if t.dim() == 0:
            t = t.expand(v.shape[0], 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(v.shape[0], 1)
            else:
                t = t.unsqueeze(-1)
            
        x = torch.cat([t, v], dim=-1)
        return self.net(x)


class DiffusionMLP(nn.Module):
    """
    MLP representing variance diffusion g_theta(t, V_t).
    Output is constrained to R+ using Softplus to guarantee volatility positivity.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive diffusion coefficient
        )
        
    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Ensure t matches the batch size of v
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=v.dtype, device=v.device)
        if t.dim() == 0:
            t = t.expand(v.shape[0], 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t = t.expand(v.shape[0], 1)
            else:
                t = t.unsqueeze(-1)
            
        x = torch.cat([t, v], dim=-1)
        return self.net(x)


class NeuralSDE(torchsde.SDEIto):
    """
    Itô SDE system:
      dX_t = (r - q - 0.5 * V_t) dt + sqrt(V_t) * (rho * dW_t^1 + sqrt(1 - rho^2) * dW_t^2)
      dV_t = f_theta(t, V_t) dt + g_theta(t, V_t) * dW_t^1
    """
    def __init__(self, r: float = 0.0, q: float = 0.0, rho_init: float = -0.7, 
                 hidden_dim: int = 64, epsilon: float = 1e-4):
        # We use general noise as our SDE is driven by 2-dim Brownian motion (W^1, W^2)
        # with cross-diffusion terms.
        super().__init__(noise_type="general")
        self.r = r
        self.q = q
        self.epsilon = epsilon
        
        # Raw parameter for leverage correlation rho, constrained to [-0.95, 0.0]
        self.raw_rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
        
        # Drift and Diffusion MLP sub-modules
        self.drift_mlp = DriftMLP(hidden_dim=hidden_dim)
        self.diff_mlp = DiffusionMLP(hidden_dim=hidden_dim)

    @property
    def rho(self) -> torch.Tensor:
        # Constrain correlation to [-0.95, 0.0]
        return -0.95 * torch.sigmoid(self.raw_rho)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (N_paths, 2)
        # y[:, 0:1] is log-return X_t
        # y[:, 1:2] is variance V_t
        v = y[:, 1:2]
        
        # Apply positivity floor during evaluation
        v_floor = torch.clamp(v, min=self.epsilon)
        
        # dX_t drift: r - q - 0.5 * V_t
        drift_x = self.r - self.q - 0.5 * v_floor
        
        # dV_t drift: f_theta(t, V_t)
        drift_v = self.drift_mlp(t, v_floor)
        
        return torch.cat([drift_x, drift_v], dim=-1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (N_paths, 2)
        v = y[:, 1:2]
        
        # Apply positivity floor during evaluation
        v_floor = torch.clamp(v, min=self.epsilon)
        sqrt_v = torch.sqrt(v_floor)
        
        # dV_t diffusion: g_theta(t, V_t)
        diff_v = self.diff_mlp(t, v_floor)
        
        # Construct diffusion matrix: shape (N_paths, state_dim, noise_dim)
        g_mat = torch.zeros(y.shape[0], 2, 2, device=y.device, dtype=y.dtype)
        
        current_rho = self.rho
        # Row 0: dX_t noise coefficients (dW_1, dW_2)
        g_mat[:, 0, 0] = (sqrt_v * current_rho).squeeze(-1)
        g_mat[:, 0, 1] = (sqrt_v * torch.sqrt(1.0 - current_rho**2)).squeeze(-1)
        
        # Row 1: dV_t noise coefficients (dW_1, dW_2)
        g_mat[:, 1, 0] = diff_v.squeeze(-1)
        # g_mat[:, 1, 1] is 0
        
        return g_mat


class NeuralSDEPricer(nn.Module):
    """
    Wrapper model to manage pricing, initial variance V_0, and simulation.
    """
    def __init__(self, sde: NeuralSDE, v0_init: float = 0.04):
        super().__init__()
        self.sde = sde
        self.raw_v0 = nn.Parameter(torch.tensor(v0_init, dtype=torch.float32))

    @property
    def v0(self) -> torch.Tensor:
        # Ensure initial variance is positive and strictly greater than the floor
        return torch.nn.functional.softplus(self.raw_v0) + self.sde.epsilon

    def price_options(
        self,
        S0: float,
        strikes: torch.Tensor,
        maturities: torch.Tensor,
        N_paths: int = 2048,
        dt: float = 0.01,
        method: str = "euler",
        bm: torchsde.BaseBrownian = None,
    ):
        """
        Price a batch of European call options using SDE adjoint simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price.
        strikes : torch.Tensor
            Vector of strikes, shape (N_options,).
        maturities : torch.Tensor
            Vector of maturities, shape (N_options,).
        N_paths : int
            Number of Monte Carlo paths.
        dt : float
            Simulation time step.
        method : str
            Solver method name (e.g. 'euler', 'reversible_heun').
        bm : torchsde.BaseBrownian
            Optional fixed noise path for the reparameterization trick.
            
        Returns:
        --------
        prices : torch.Tensor
            Vector of option prices, shape (N_options,).
        ys : torch.Tensor
            The full simulated states (clamped to floor), shape (N_ts, N_paths, 2).
        """
        device = strikes.device
        dtype = strikes.dtype
        
        # 1. Identify unique sorted maturities on CPU to prevent CPU-GPU synchronization barrier
        maturities_cpu = maturities.to("cpu")
        unique_maturities, inverse_indices = torch.unique(maturities_cpu, return_inverse=True)
        unique_maturities_sorted, sort_indices = torch.sort(unique_maturities)
        
        # 2. Build simulation evaluation time grid starting at 0.0
        # Since unique_maturities_sorted is on CPU, indexing and comparison are on CPU
        t0_val = unique_maturities_sorted[0].item()
        if t0_val > 0.0:
            ts_cpu = torch.cat([torch.tensor([0.0], dtype=dtype), unique_maturities_sorted])
            shift = 1
        else:
            ts_cpu = unique_maturities_sorted
            shift = 0
            
        # 3. Setup initial state
        y0 = torch.zeros(N_paths, 2, device=device, dtype=dtype)
        y0[:, 1] = self.v0
        
        # 4. Handle Brownian Interval
        if bm is None:
            bm = torchsde.BrownianInterval(
                t0=ts_cpu[0].item(),
                t1=ts_cpu[-1].item(),
                size=(N_paths, 2),
                device=device,
                dtype=dtype
            )
            
        ts = ts_cpu.to(device)
            
        # 5. Run SDE integration
        ys = torchsde.sdeint_adjoint(
            self.sde,
            y0,
            ts,
            bm=bm,
            method=method,
            dt=dt
        )
        
        # Clamp variance component of ys to guarantee positivity floor
        ys_clamped = torch.cat([ys[..., 0:1], torch.clamp(ys[..., 1:2], min=self.sde.epsilon)], dim=-1)
        
        # 6. Map options to output states
        mapped_indices = sort_indices[inverse_indices] + shift
        ys_options = ys_clamped[mapped_indices.to(device)]  # Shape: (N_options, N_paths, 2)
        
        X_T = ys_options[:, :, 0]  # (N_options, N_paths)
        S_T = S0 * torch.exp(X_T)
        
        # 7. Compute European Call payoff and discounted prices
        payoff = torch.clamp(S_T - strikes.unsqueeze(-1), min=0.0)
        discount = torch.exp(-self.sde.r * maturities).unsqueeze(-1)
        prices = (payoff * discount).mean(dim=-1)
        
        return prices, ys_clamped


def compute_calibration_loss(
    model_prices: torch.Tensor,
    market_prices: torch.Tensor,
    vegas: torch.Tensor,
    ys: torch.Tensor,
    lambda_bound: float = 0.01,
    epsilon: float = 1e-4
) -> dict:
    """
    Computes calibration loss combining Vega-weighted MSE and Feller-like boundary penalty.
    """
    # 1. Base loss: Vega-weighted MSE
    if vegas is None:
        weights = torch.ones_like(model_prices)
    else:
        weights = 1.0 / torch.clamp(vegas, min=1e-4)
    weighted_diff = weights * (model_prices - market_prices)
    loss_base = torch.mean(weighted_diff ** 2)
    
    # 2. Boundary regularization loss
    v_t = ys[:, :, 1]
    
    # Use continuous differentiable approximation for 1/v to avoid division by zero or negative values.
    # f(v) = 1/v for v >= epsilon, and 2/epsilon - v/epsilon^2 for v < epsilon
    inv_v = torch.where(v_t >= epsilon, 1.0 / v_t, 2.0 / epsilon - v_t / (epsilon ** 2))
    v_sq = v_t ** 2
    loss_reg = lambda_bound * torch.mean(inv_v + v_sq)
    
    loss_total = loss_base + loss_reg
    
    return {
        "loss": loss_total,
        "loss_base": loss_base,
        "loss_reg": loss_reg
    }

"""
mp_diffusion.py — Martingale-Preserving Denoising Diffusion Probabilistic Model (MP-DDPM)
for risk-neutral spot and variance path simulation.
"""

import math
import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("deepvol.hedging.mp_diffusion")


class MartingaleViolationError(Exception):
    """
    Exception raised when path martingale audit bounds are breached (SR 26-2).
    """
    pass


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion time steps.
    Reference: Vaswani et al. (2017) / DDPM (Ho et al. 2020).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed diffusion steps into a sinusoidal space.
        
        Parameters:
            t: Tensor of shape (B,) containing step indices.
            
        Returns:
            embeddings: Tensor of shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with time step embedding projection.
    Uses 1D temporal convolutions to process paths.
    """
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(emb_dim, channels)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
            x: Tensor of shape (B, C, T)
            emb: Tensor of shape (B, emb_dim)
            
        Returns:
            out: Tensor of shape (B, C, T)
        """
        h = self.conv1(x)
        emb_proj = self.time_emb_proj(emb).unsqueeze(-1)
        h = h + emb_proj
        h = self.activation(h)
        h = self.conv2(h)
        return x + h


class PathDenoisingNet(nn.Module):
    """
    Temporal 1D ResNet for predicting noise in joint spot and variance paths.
    """
    def __init__(self, in_channels: int = 2, hidden_dim: int = 64, num_blocks: int = 3, emb_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, emb_dim)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise.
        
        Parameters:
            x: Tensor of shape (B, in_channels, T)
            t: Tensor of shape (B,)
            
        Returns:
            out: Tensor of shape (B, in_channels, T)
        """
        emb = self.time_emb(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.output_proj(h)


def project_spot_martingale(S: torch.Tensor, S_0: float) -> torch.Tensor:
    """
    Differentiably project spot paths S onto the martingale subspace.
    Enforces the condition: E[S_t] = S_0, approximated via the sample mean:
    1/N_p * sum_{i=1}^{N_p} S_{i, t} = S_0 at every time step t.
    
    The projection is performed strictly in double precision (float64) to
    prevent numerical drift and gradient noise in cumulative operations.
    
    Mathematical Formulation:
        Let S_t in R^{N_p} be the spot price vector across paths at time t.
        The orthogonal projection of S_t onto the martingale hyperplane
        sum(S_{i,t}) = N_p * S_0 is given by:
            S_{i,t}^{projected} = S_{i,t} - (1/N_p * sum(S_{j,t}) - S_0).
        
    Parameters:
        S: Spot paths tensor of shape (N_p, T) or (N_p, 2, T) containing spot prices in channel 0.
        S_0: Initial spot price.
        
    Returns:
        Projected spot paths of the same shape as S.
    """
    is_3d = (S.dim() == 3)
    if is_3d:
        S_spot = S[:, 0, :]
    elif S.dim() == 2:
        S_spot = S
    else:
        raise ValueError(f"Expected S to be 2D or 3D, got dimension {S.dim()}")
        
    # Cast to float64 for internal projection
    S_double = S_spot.to(torch.float64)
    S_0_double = torch.tensor(S_0, dtype=torch.float64, device=S.device)
    
    # Calculate sample mean at each time step t over the path dimension (dim=0)
    mean_S = S_double.mean(dim=0, keepdim=True)  # (1, T)
    
    # Orthogonal projection onto the martingale subspace
    S_proj_double = S_double - mean_S + S_0_double
    
    # Cast back to original dtype
    S_projected = S_proj_double.to(S.dtype)
    
    if is_3d:
        S_out = S.clone()
        S_out[:, 0, :] = S_projected
        return S_out
    else:
        return S_projected


def audit_martingale_paths(S: torch.Tensor, S_0: float, raise_on_failure: bool = True) -> torch.Tensor:
    """
    Audit path martingale residuals. Enforces SR 26-2 compliance.
    Logs the residual checks and raises alerts/halts if bounds are breached.
    
    Residual definition:
        residual_t = | 1/N_p * sum(S_t) - S_0 |
    
    Boundary: max_t residual_t < 1e-5.
    
    Parameters:
        S: Spot paths tensor of shape (N_p, T) or (N_p, 2, T)
        S_0: Initial spot price.
        raise_on_failure: If True, raises MartingaleViolationError if bounds are breached.
        
    Returns:
        residuals: Tensor of shape (T,) containing residuals for each time step.
    """
    if S.dim() == 3:
        S_spot = S[:, 0, :]
    else:
        S_spot = S
        
    # Strictly double precision to avoid false alarms
    S_double = S_spot.to(torch.float64)
    S_0_double = torch.tensor(S_0, dtype=torch.float64, device=S.device)
    
    mean_S = S_double.mean(dim=0)  # (T,)
    residuals = torch.abs(mean_S - S_0_double)
    
    max_residual = residuals.max().item()
    logger.info(f"Martingale Audit: max residual = {max_residual:.16e} (tolerance threshold = 1.0000000000000000e-05)")
    
    if max_residual >= 1e-5:
        err_msg = f"Martingale violation detected! Max residual {max_residual:.6e} breached tolerance limit of 1e-5."
        logger.error(err_msg)
        if raise_on_failure:
            raise MartingaleViolationError(err_msg)
            
    return residuals


@torch.compile(mode="reduce-overhead")
def _reverse_diffusion_step_compiled(
    x: torch.Tensor,
    eps_pred: torch.Tensor,
    k_idx: int,
    sqrt_recip_alphas_cumprod_k: torch.Tensor,
    sqrt_recipm1_alphas_cumprod_k: torch.Tensor,
    coef1: torch.Tensor,
    coef2: torch.Tensor,
    posterior_variance_k: torch.Tensor,
    noise: torch.Tensor,
    S_0: float,
    V_0: float,
    project_at_each_step: bool,
    variance_positivity_constraint: int
) -> torch.Tensor:
    """
    Perform a single reverse diffusion step.
    Compiled with torch.compile to enable Triton kernel fusion and minimize VRAM IO latency.
    
    Parameters:
        x: Current path states tensor of shape (N_p, 2, T)
        eps_pred: Predicted noise tensor of shape (N_p, 2, T)
        k_idx: Current diffusion step index.
        sqrt_recip_alphas_cumprod_k: Scale factor for x_k.
        sqrt_recipm1_alphas_cumprod_k: Scale factor for eps_pred.
        coef1: Mean coefficient 1.
        coef2: Mean coefficient 2.
        posterior_variance_k: Step posterior variance.
        noise: Pre-sampled Gaussian noise tensor of shape (N_p, 2, T)
        S_0: Initial spot price.
        V_0: Initial variance.
        project_at_each_step: Whether to project spot onto martingale subspace.
        variance_positivity_constraint: 0 for softplus, 1 for clamp.
        
    Returns:
        new_x: Updated path states tensor of shape (N_p, 2, T) (cloned to guard static buffers).
    """
    # 1. Reconstruct denoised x0_hat
    x0_hat = sqrt_recip_alphas_cumprod_k * x - sqrt_recipm1_alphas_cumprod_k * eps_pred
    
    x0_hat_spot = x0_hat[:, 0, :]
    x0_hat_var = x0_hat[:, 1, :]
    
    # 2. Differentiably project spot onto martingale subspace
    if project_at_each_step:
        # Cast to double precision internally
        S_double = x0_hat_spot.to(torch.float64)
        mean_S = S_double.mean(dim=0, keepdim=True)
        x0_hat_spot_proj = (S_double - mean_S + S_0).to(x.dtype)
    else:
        x0_hat_spot_proj = x0_hat_spot
        
    # 3. Enforce variance positivity
    if variance_positivity_constraint == 0:
        x0_hat_var_proj = torch.nn.functional.softplus(x0_hat_var)
    else:
        x0_hat_var_proj = torch.clamp(x0_hat_var, min=1e-4)
        
    # Reassemble channels
    x0_hat_proj = torch.stack([x0_hat_spot_proj, x0_hat_var_proj], dim=1)
    
    # Force exact boundary at t=0
    x0_hat_proj = x0_hat_proj.clone()
    x0_hat_proj[:, 0, 0] = S_0
    x0_hat_proj[:, 1, 0] = V_0
    
    # 4. Compute posterior mean
    posterior_mean = coef1 * x0_hat_proj + coef2 * x
    
    # 5. Compute new state x_{k-1}
    if k_idx == 0:
        new_x = x0_hat_proj
    else:
        new_x = posterior_mean + torch.sqrt(posterior_variance_k) * noise

        
    # Re-enforce boundaries at t=0
    new_x = new_x.clone()
    new_x[:, 0, 0] = S_0
    new_x[:, 1, 0] = V_0
    
    # Return clone to avoid memory overwrite issues with CUDA Graphs static buffers
    return new_x.clone()


class MPDDPM(nn.Module):
    """
    Martingale-Preserving Denoising Diffusion Probabilistic Model (MP-DDPM).
    Simulates joint risk-neutral spot and variance paths with exact martingale constraints.
    """
    def __init__(
        self,
        denoising_net: nn.Module,
        T_d: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_schedule: str = "linear"
    ):
        super().__init__()
        self.denoising_net = denoising_net
        self.T_d = T_d
        
        # Build noise schedule
        if noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T_d)
        elif noise_schedule == "cosine":
            steps = T_d + 1
            x = torch.linspace(0, T_d, steps)
            alphas_cumprod = torch.cos(((x / T_d) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer("posterior_mean_coef1", torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
    def forward_loss(self, S_0_paths: torch.Tensor, V_0_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute the denoising loss (MSE) for a batch of joint spot and variance paths.
        
        Parameters:
            S_0_paths: Real spot paths of shape (B, T)
            V_0_paths: Real variance paths of shape (B, T)
            
        Returns:
            loss: Mean squared error of the noise prediction.
        """
        # Formulate joint paths in Structure of Arrays style but stacked for Conv1D processing
        x_0 = torch.stack([S_0_paths, V_0_paths], dim=1)
        B, C, T = x_0.shape
        
        # Sample step index k
        k = torch.randint(0, self.T_d, (B,), device=x_0.device)
        
        # Sample target noise
        eps = torch.randn_like(x_0)
        
        # Compute noisy state x_k
        sqrt_alphas_cumprod_k = self.sqrt_alphas_cumprod[k].view(B, 1, 1)
        sqrt_one_minus_alphas_cumprod_k = self.sqrt_one_minus_alphas_cumprod[k].view(B, 1, 1)
        x_k = sqrt_alphas_cumprod_k * x_0 + sqrt_one_minus_alphas_cumprod_k * eps
        
        # Predict noise
        eps_pred = self.denoising_net(x_k, k)
        
        return nn.functional.mse_loss(eps_pred, eps)
        
    def sample(
        self,
        num_paths: int,
        T: int,
        S_0: float,
        V_0: float,
        device: torch.device,
        project_at_each_step: bool = True,
        variance_positivity_constraint: str = "softplus",
        bypass_martingale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate spot and variance paths using the reverse diffusion process,
        applying the differentiable martingale projection at each step.
        
        Parameters:
            num_paths: Number of paths (N_p) to generate.
            T: Length of paths.
            S_0: Initial spot price.
            V_0: Initial variance.
            device: Executing device.
            project_at_each_step: Whether to project spot onto martingale subspace at each step.
            variance_positivity_constraint: "softplus" or "clamp".
            bypass_martingale: If True, disables all martingale projections and audit failures.
            
        Returns:
            S: Projected spot paths of shape (num_paths, T)
            V: Variance paths of shape (num_paths, T)
        """
        # Start from Gaussian noise
        x = torch.randn(num_paths, 2, T, device=device)
        x[:, 0, 0] = S_0
        x[:, 1, 0] = V_0
        
        var_const = 0 if variance_positivity_constraint == "softplus" else 1
        
        # Determine whether to project during reverse diffusion steps
        do_project = project_at_each_step and not bypass_martingale
        
        # Reverse diffusion loop
        for k_idx in reversed(range(self.T_d)):
            k = torch.full((num_paths,), k_idx, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                eps_pred = self.denoising_net(x, k)
                
            # Sample step noise (exclude t=0)
            noise = torch.randn_like(x)
            noise[:, :, 0] = 0.0
            
            # Coefficients for the step
            sqrt_recip_alphas_cumprod_k = self.sqrt_recip_alphas_cumprod[k_idx]
            sqrt_recipm1_alphas_cumprod_k = self.sqrt_recipm1_alphas_cumprod[k_idx]
            coef1 = self.posterior_mean_coef1[k_idx]
            coef2 = self.posterior_mean_coef2[k_idx]
            posterior_variance_k = self.posterior_variance[k_idx]
            
            # For the very last step (k=0), if not bypass_martingale, we always project
            step_project = do_project or (k_idx == 0 and not bypass_martingale)
            
            # Execute step (compiled Triton kernel if on CUDA)
            x = _reverse_diffusion_step_compiled(
                x, eps_pred, k_idx,
                sqrt_recip_alphas_cumprod_k, sqrt_recipm1_alphas_cumprod_k,
                coef1, coef2, posterior_variance_k, noise,
                S_0, V_0, step_project, var_const
            )
            x = x.clone()

            
        # Extract channels (SoA layout)
        S = x[:, 0, :]
        V = x[:, 1, :]
        
        # Enforce compliance check (only raise if not bypassed)
        audit_martingale_paths(S, S_0, raise_on_failure=not bypass_martingale)
        
        return S, V


"""
src/pricing/signature_vol.py - Signature-Based Volatility Model.
Implements dependency-free, GPU-accelerated signature path simulations,
martingale property enforcement, and positivity constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_path_signature(path: torch.Tensor, depth: int = 4) -> torch.Tensor:
    """
    Computes the path signature of a batch of D-dimensional paths using Chen's relation.
    Ensures full differentiability and GPU acceleration.
    
    Args:
        path: Tensor of shape (B, L, D) where B is batch size, L is path length, D is dimension.
        depth: Signature depth (supports up to 4).
        
    Returns:
        Tensor of shape (B, N_features) containing concatenated signature levels.
    """
    B, L, D = path.shape
    device = path.device
    dtype = path.dtype
    
    if L < 2:
        num_features = sum(D ** i for i in range(1, depth + 1))
        return torch.zeros(B, num_features, device=device, dtype=dtype)
        
    # Compute path increments
    deltas = path[:, 1:, :] - path[:, :-1, :]  # (B, L-1, D)
    
    # Initialize signature tensors for levels 1..depth
    S1 = torch.zeros(B, D, device=device, dtype=dtype)
    S2 = torch.zeros(B, D, D, device=device, dtype=dtype)
    S3 = torch.zeros(B, D, D, D, device=device, dtype=dtype)
    S4 = torch.zeros(B, D, D, D, D, device=device, dtype=dtype)
    
    for step in range(L - 1):
        delta = deltas[:, step, :]  # (B, D)
        
        # Segment signature components: A^k = (1/k!) * delta^{\otimes k}
        A1 = delta
        
        if depth >= 2:
            A2 = 0.5 * torch.einsum('bi,bj->bij', delta, delta)
        if depth >= 3:
            A3 = (1.0 / 6.0) * torch.einsum('bi,bj,bk->bijk', delta, delta, delta)
        if depth >= 4:
            A4 = (1.0 / 24.0) * torch.einsum('bi,bj,bk,bl->bijkl', delta, delta, delta, delta)
        
        # Chen's relation update (order is crucial to use old states)
        if depth >= 4:
            S4 = (S4 + 
                  torch.einsum('bijk,bl->bijkl', S3, A1) + 
                  torch.einsum('bij,bkl->bijkl', S2, A2) + 
                  torch.einsum('bi,bjkl->bijkl', S1, A3) + 
                  A4)
                  
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
    if depth >= 4:
        features.append(S4.reshape(B, -1))
        
    return torch.cat(features, dim=-1)


def simulate_signature_vol_paths(
    v0: torch.Tensor,
    ell: torch.Tensor,
    rho: torch.Tensor,
    T: float,
    steps_per_unit: int,
    N_paths: int,
    S0: float = 1.0,
    r: float = 0.0,
    q: float = 0.0,
    antithetic: bool = True,
    device: str = "cpu",
    positivity_func: str = "relu",
    variance_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates pathwise stock and variance dynamics under the Signature Volatility model.
    
    Returns:
        S: Stock price paths (N_paths, N_steps + 1)
        V: Variance paths (N_paths, N_steps + 1)
        V_raw: Raw unthresholded variance paths (N_paths, N_steps + 1)
        t_grid: Time grid tensor (N_steps + 1,)
    """
    N_steps = int(round(T * steps_per_unit))
    dt = 1.0 / steps_per_unit
    sqrt_dt = np.sqrt(dt)
    
    dtype = v0.dtype
    t_grid = torch.linspace(0.0, T, N_steps + 1, device=device, dtype=dtype)
    
    # Path dimension is 2: (time, W^1)
    S1 = torch.zeros(N_paths, 2, device=device, dtype=dtype)
    S2 = torch.zeros(N_paths, 2, 2, device=device, dtype=dtype)
    S3 = torch.zeros(N_paths, 2, 2, 2, device=device, dtype=dtype)
    S4 = torch.zeros(N_paths, 2, 2, 2, 2, device=device, dtype=dtype)
    
    # Extract linear coefficients for each level
    ell_t = torch.as_tensor(ell, device=device, dtype=dtype)
    ell1 = ell_t[0:2]
    ell2 = ell_t[2:6].view(2, 2)
    ell3 = ell_t[6:14].view(2, 2, 2)
    ell4 = ell_t[14:30].view(2, 2, 2, 2)
    
    # Setup Brownian increment generators
    if antithetic:
        half_paths = N_paths // 2
        Z1_half = torch.randn(half_paths, N_steps, device=device, dtype=dtype)
        Z2_half = torch.randn(half_paths, N_steps, device=device, dtype=dtype)
        Z1 = torch.cat([Z1_half, -Z1_half], dim=0)
        Z2 = torch.cat([Z2_half, -Z2_half], dim=0)
    else:
        Z1 = torch.randn(N_paths, N_steps, device=device, dtype=dtype)
        Z2 = torch.randn(N_paths, N_steps, device=device, dtype=dtype)
        
    dW1 = Z1 * sqrt_dt
    dW2 = Z2 * sqrt_dt
    
    # Log asset return initialization
    X_list = [torch.full((N_paths,), np.log(S0), device=device, dtype=dtype)]
    
    v0_t = torch.as_tensor(v0, device=device, dtype=dtype)
    if v0_t.dim() == 0:
        v0_t = v0_t.unsqueeze(0)
    v0_path = v0_t.expand(N_paths)
    V_list = [v0_path]
    V_raw_list = [v0_path]
    
    # Select thresholding activation
    if positivity_func == "relu":
        pos_fn = lambda val: torch.clamp(val, min=variance_floor)
    elif positivity_func == "softplus":
        pos_fn = lambda val: F.softplus(val) + variance_floor
    else:
        raise ValueError(f"Unknown positivity function: {positivity_func}")
        
    def sim_step(S1_c, S2_c, S3_c, S4_c, X_c, V_c, dW1_i, dW2_i, v0_val, ell1_val, ell2_val, ell3_val, ell4_val, rho_val):
        delta_step = torch.stack([torch.full((N_paths,), dt, device=device, dtype=dtype), dW1_i], dim=1)
        
        # Segment signature using elementwise, broadcasting, and sums to avoid einsum launcher overhead
        A2 = 0.5 * (delta_step.unsqueeze(2) * delta_step.unsqueeze(1))
        A3 = (1.0 / 6.0) * (delta_step.view(N_paths, 2, 1, 1) * delta_step.view(N_paths, 1, 2, 1) * delta_step.view(N_paths, 1, 1, 2))
        A4 = (1.0 / 24.0) * (delta_step.view(N_paths, 2, 1, 1, 1) * delta_step.view(N_paths, 1, 2, 1, 1) * delta_step.view(N_paths, 1, 1, 2, 1) * delta_step.view(N_paths, 1, 1, 1, 2))
        
        # Recursive signature updates
        S4_n = (S4_c + 
                S3_c.unsqueeze(4) * delta_step.view(N_paths, 1, 1, 1, 2) + 
                S2_c.view(N_paths, 2, 2, 1, 1) * A2.view(N_paths, 1, 1, 2, 2) + 
                S1_c.view(N_paths, 2, 1, 1, 1) * A3.view(N_paths, 1, 2, 2, 2) + 
                A4)
                
        S3_n = (S3_c + 
                S2_c.unsqueeze(3) * delta_step.view(N_paths, 1, 1, 2) + 
                S1_c.view(N_paths, 2, 1, 1) * A2.view(N_paths, 1, 2, 2) + 
                A3)
                
        S2_n = S2_c + S1_c.unsqueeze(2) * delta_step.unsqueeze(1) + A2
        
        S1_n = S1_c + delta_step
        
        # Compute raw and thresholded variance using sum along non-batch dimensions
        term1 = (S1_n * ell1_val).sum(dim=1)
        term2 = (S2_n * ell2_val).sum(dim=(1, 2))
        term3 = (S3_n * ell3_val).sum(dim=(1, 2, 3))
        term4 = (S4_n * ell4_val).sum(dim=(1, 2, 3, 4))
        
        v_raw_val = v0_val + term1 + term2 + term3 + term4
        V_n = pos_fn(v_raw_val)
        
        # Log stock price step update
        drift = (r - q - 0.5 * V_c) * dt
        diffusion = torch.sqrt(V_c) * (rho_val * dW1_i + torch.sqrt(1.0 - rho_val**2) * dW2_i)
        X_n = X_c + drift + diffusion
        
        return S1_n, S2_n, S3_n, S4_n, X_n, V_n, v_raw_val

    rho_t = torch.as_tensor(rho, device=device, dtype=dtype)

    for i in range(N_steps):
        if torch.is_grad_enabled():
            S1, S2, S3, S4, X_next, V_next, v_raw = torch.utils.checkpoint.checkpoint(
                sim_step,
                S1, S2, S3, S4, X_list[-1], V_list[-1],
                dW1[:, i], dW2[:, i],
                v0_t, ell1, ell2, ell3, ell4, rho_t,
                use_reentrant=False
            )
        else:
            S1, S2, S3, S4, X_next, V_next, v_raw = sim_step(
                S1, S2, S3, S4, X_list[-1], V_list[-1],
                dW1[:, i], dW2[:, i],
                v0_t, ell1, ell2, ell3, ell4, rho_t
            )
        X_list.append(X_next)
        V_list.append(V_next)
        V_raw_list.append(v_raw)
        
    X_stacked = torch.stack(X_list, dim=1)
    V_stacked = torch.stack(V_list, dim=1)
    V_raw_stacked = torch.stack(V_raw_list, dim=1)
    
    return torch.exp(X_stacked), V_stacked, V_raw_stacked, t_grid


class SignatureVolatilityModel(nn.Module):
    """
    PyTorch module for the Signature-Based Volatility Model.
    Enforces the martingale property by masking even-order signature coefficients,
    and guarantees negative leverage correlation.
    """
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        
        # Parameterize v0 in log-space for positivity: initial v0 = 0.04
        self.v0_raw = nn.Parameter(torch.tensor(np.log(0.04), device=device, dtype=dtype))
        
        # Parameterize rho in logit-space to enforce range [-0.95, -0.05]
        # Initial value of 0.0 maps to -0.5
        self.rho_raw = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        
        # Signature coefficients up to depth 4 (30 elements)
        self.ell_raw = nn.Parameter(torch.zeros(30, device=device, dtype=dtype))
        
        # Indices of odd-order (Levels 1 & 3) and even-order (Levels 2 & 4) signature terms
        self.odd_indices = [0, 1] + list(range(6, 14))
        self.even_indices = [2, 3, 4, 5] + list(range(14, 30))
        
        # Setup binary mask as a non-trainable buffer
        mask = torch.zeros(30, device=device, dtype=dtype)
        mask[self.odd_indices] = 1.0
        self.register_buffer("mask", mask)
        
    @property
    def device(self) -> torch.device:
        return self.mask.device
        
    @property
    def dtype(self) -> torch.dtype:
        return self.mask.dtype
        
    @property
    def v0(self) -> torch.Tensor:
        return torch.exp(self.v0_raw)
        
    @property
    def rho(self) -> torch.Tensor:
        return -0.05 - 0.90 * torch.sigmoid(self.rho_raw)
        
    def get_constrained_ell(self) -> torch.Tensor:
        """Applies the odd-order mask to the coefficients."""
        return self.ell_raw * self.mask
        
    def project_parameters(self):
        """Zeroes out the even-order coefficients in parameter memory."""
        with torch.no_grad():
            self.ell_raw.data[self.even_indices] = 0.0
            
    def forward(
        self,
        T: float,
        steps_per_unit: int,
        N_paths: int,
        S0: float = 1.0,
        r: float = 0.0,
        q: float = 0.0,
        antithetic: bool = True,
        positivity_func: str = "relu",
        variance_floor: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ell_constrained = self.get_constrained_ell()
        return simulate_signature_vol_paths(
            v0=self.v0,
            ell=ell_constrained,
            rho=self.rho,
            T=T,
            steps_per_unit=steps_per_unit,
            N_paths=N_paths,
            S0=S0,
            r=r,
            q=q,
            antithetic=antithetic,
            device=self.device,
            positivity_func=positivity_func,
            variance_floor=variance_floor,
        )
        
    def compute_loss(
        self,
        S_target: torch.Tensor,
        T: float,
        steps_per_unit: int,
        N_paths: int,
        S0: float = 1.0,
        r: float = 0.0,
        q: float = 0.0,
        mu_pen: float = 1e4,
    ) -> torch.Tensor:
        """
        Computes standard MSE pricing loss + negative variance penalty.
        """
        S, V, V_raw, _ = self.forward(
            T=T, steps_per_unit=steps_per_unit, N_paths=N_paths,
            S0=S0, r=r, q=q, antithetic=True
        )
        
        # Example pricing loss: match terminal distribution MSE
        pricing_loss = F.mse_loss(S.mean(dim=0), S_target.mean(dim=0))
        
        # Volatility positivity penalty
        penalty = mu_pen * torch.mean(torch.clamp(-V_raw, min=0.0) ** 2)
        
        return pricing_loss + penalty

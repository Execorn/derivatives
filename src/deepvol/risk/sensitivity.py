"""
Differentiable Greeks Sensitivity Engine.
Computes portfolio-level option Greeks (Delta, Vega, Theta, Gamma, Vanna, Volga)
using PyTorch autograd through FNO surrogates.
"""

import math
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from deepvol.greeks.portfolio_greeks import (
    _ensure_normalizers,
    _make_spatial,
    bs_call_price,
    interpolate_bilinear,
    MATURITIES,
    STRIKES
)

def portfolio_price_tensor(
    S: torch.Tensor,
    theta: torch.Tensor,
    t: torch.Tensor,
    epsilon: torch.Tensor,
    K_t: torch.Tensor,
    T_t: torch.Tensor,
    qty_t: torch.Tensor,
    notional_t: torch.Tensor,
    is_call_t: torch.Tensor,
    model: torch.nn.Module,
    pn,
    yn,
    r_t: torch.Tensor,
    sticky_strike: bool = True,
    iv_surface: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes portfolio price as a fully differentiable PyTorch tensor.
    Supports autograd tracking w.r.t Spot (S), Volatility shift (epsilon), and Time (t).
    
    Formula References:
        V(S, theta, t, epsilon) = sum_i qty_i * notional_i * BS_Price(S, K_i, T_i - t, r, sigma_i + epsilon)
        where sigma_i is interpolated from FNO(theta) surface.
    """
    device = S.device
    dtype = S.dtype
    
    if iv_surface is None:
        # Run FNO model
        # If theta requires grad, we must run FNO with gradients tracked.
        # Otherwise, we can run FNO with torch.no_grad() to speed up and save memory.
        if theta.requires_grad:
            theta_norm = pn.transform_tensor(theta.unsqueeze(0))
            spatial = _make_spatial(MATURITIES, STRIKES, device).to(theta.dtype)
            pred_norm = model(spatial, theta_norm)
            iv_surface = yn.inverse_transform_tensor(pred_norm).squeeze(0)
        else:
            theta_norm = pn.transform_tensor(theta.unsqueeze(0)).float()
            spatial = _make_spatial(MATURITIES, STRIKES, device).float()
            with torch.no_grad():
                pred_norm = model(spatial, theta_norm)
                iv_surface = yn.inverse_transform_tensor(pred_norm).squeeze(0)
                
    iv_surface = torch.clamp(iv_surface, min=1e-4).to(dtype)
    
    T_grid_t = torch.tensor(MATURITIES, dtype=dtype, device=device)
    K_grid_t = torch.tensor(STRIKES, dtype=dtype, device=device)
    
    T_pos = T_t.to(dtype) - t
    if sticky_strike:
        k_pos = torch.log(K_t.to(dtype) / S)
    else:
        k_pos = torch.log(K_t.to(dtype) / S.detach())
        
    sig = interpolate_bilinear(T_grid_t, K_grid_t, iv_surface, T_pos, k_pos)
    sig = torch.clamp(sig + epsilon, min=1e-4)
    
    call_prices = bs_call_price(S, K_t.to(dtype), T_pos, r_t.to(dtype), sig)
    # Put-call parity: Put = Call + K * exp(-r * T) - S
    put_prices = call_prices + K_t.to(dtype) * torch.exp(-r_t.to(dtype) * T_pos) - S
    
    prices = torch.where(is_call_t.to(dtype) == 1.0, call_prices, put_prices)
    
    return torch.sum(prices * qty_t.to(dtype) * notional_t.to(dtype))


class AutogradSensitivityEngine:
    """
    Vectorized PyTorch autograd engine for first-order (Delta, Vega, Theta)
    and second-order (Gamma, Vanna, Volga) portfolio-level Greeks.
    """
    def __init__(self, model: torch.nn.Module, pn, yn, device: Optional[torch.device] = None):
        self.model = model
        self.pn = pn
        self.yn = yn
        self.device = device or next(model.parameters()).device

    def compute_greeks(
        self,
        positions: List[Dict[str, Any]],
        S: float,
        theta: np.ndarray,
        r: float = 0.05,
        sticky_strike: bool = True,
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, float]:
        """
        Computes portfolio-level Greeks: Delta, Vega, Theta, Gamma, Vanna, Volga.
        All derivatives are computed analytically using PyTorch autograd.

        Formula References:
            - Delta: dV / dS
            - Vega: dV / d_epsilon (shift in IV surface)
            - Theta: -dV / dt (time decay)
            - Gamma: d^2V / dS^2
            - Vanna: d^2V / (dS * d_epsilon)
            - Volga: d^2V / d_epsilon^2
        """
        if not positions:
            return {
                "price": 0.0,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "vanna": 0.0,
                "volga": 0.0
            }

        # Setup leaf variables for AD
        S_t = torch.tensor(S, dtype=dtype, device=self.device, requires_grad=True)
        theta_t = torch.tensor(theta, dtype=dtype, device=self.device)
        t_t = torch.tensor(0.0, dtype=dtype, device=self.device, requires_grad=True)
        epsilon_t = torch.tensor(0.0, dtype=dtype, device=self.device, requires_grad=True)
        r_t = torch.tensor(r, dtype=dtype, device=self.device)

        # Precompute IV surface once under no_grad to decouple FNO forward pass from AD graph
        theta_norm = self.pn.transform_tensor(theta_t.unsqueeze(0)).float()
        spatial = _make_spatial(MATURITIES, STRIKES, self.device).float()
        with torch.no_grad():
            pred_norm = self.model(spatial, theta_norm)
            iv_surface = self.yn.inverse_transform_tensor(pred_norm).squeeze(0)
            iv_surface = torch.clamp(iv_surface, min=1e-4).to(dtype)

        # Parse position details
        K_t = torch.tensor([float(p["K"]) for p in positions], dtype=dtype, device=self.device)
        T_t = torch.tensor([float(p["T"]) for p in positions], dtype=dtype, device=self.device)
        qty_t = torch.tensor([float(p.get("quantity", 1.0)) for p in positions], dtype=dtype, device=self.device)
        notional_t = torch.tensor([float(p.get("notional", 100.0)) for p in positions], dtype=dtype, device=self.device)
        is_call_t = torch.tensor([1.0 if p.get("type", "call").lower() == "call" else 0.0 for p in positions], dtype=dtype, device=self.device)

        # Forward pass inside graph using precomputed iv_surface
        val = portfolio_price_tensor(
            S_t, theta_t, t_t, epsilon_t,
            K_t, T_t, qty_t, notional_t, is_call_t,
            self.model, self.pn, self.yn, r_t, sticky_strike,
            iv_surface=iv_surface
        )

        # First-order AD pass
        grad_1st = torch.autograd.grad(val, (S_t, epsilon_t, t_t), create_graph=True, retain_graph=True)
        delta_t = grad_1st[0]
        vega_t = grad_1st[1]
        theta_t_decay = -grad_1st[2]

        # Second-order AD pass
        gamma_t = torch.autograd.grad(delta_t, S_t, create_graph=False, retain_graph=True)[0]
        vanna_t = torch.autograd.grad(vega_t, S_t, create_graph=False, retain_graph=True)[0]
        volga_t = torch.autograd.grad(vega_t, epsilon_t, create_graph=False, retain_graph=False)[0]

        # Convert back to standard scalars
        return {
            "price": val.detach().item(),
            "delta": delta_t.detach().item(),
            "gamma": gamma_t.detach().item(),
            "vega": vega_t.detach().item(),
            "theta": theta_t_decay.detach().item(),
            "vanna": vanna_t.detach().item(),
            "volga": volga_t.detach().item()
        }

    def compute_parameter_greeks(
        self,
        positions: List[Dict[str, Any]],
        S: float,
        theta: np.ndarray,
        r: float = 0.05,
        sticky_strike: bool = True,
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, Any]:
        """
        Computes sensitivities w.r.t the 6 Rough Heston parameters (theta).
        Returns first-order parameter sensitivities (gradient) and the Hessian.
        """
        if not positions:
            return {
                "price": 0.0,
                "gradient": np.zeros(6, dtype=np.float32),
                "hessian": np.zeros((6, 6), dtype=np.float32)
            }

        # Setup leaf parameter tensor
        theta_leaf = torch.tensor(theta, dtype=dtype, device=self.device, requires_grad=True)
        S_t = torch.tensor(S, dtype=dtype, device=self.device)
        t_t = torch.tensor(0.0, dtype=dtype, device=self.device)
        epsilon_t = torch.tensor(0.0, dtype=dtype, device=self.device)
        r_t = torch.tensor(r, dtype=dtype, device=self.device)

        K_t = torch.tensor([float(p["K"]) for p in positions], dtype=dtype, device=self.device)
        T_t = torch.tensor([float(p["T"]) for p in positions], dtype=dtype, device=self.device)
        qty_t = torch.tensor([float(p.get("quantity", 1.0)) for p in positions], dtype=dtype, device=self.device)
        notional_t = torch.tensor([float(p.get("notional", 100.0)) for p in positions], dtype=dtype, device=self.device)
        is_call_t = torch.tensor([1.0 if p.get("type", "call").lower() == "call" else 0.0 for p in positions], dtype=dtype, device=self.device)

        def price_fn(theta_var: torch.Tensor) -> torch.Tensor:
            return portfolio_price_tensor(
                S_t, theta_var, t_t, epsilon_t,
                K_t, T_t, qty_t, notional_t, is_call_t,
                self.model, self.pn, self.yn, r_t, sticky_strike
            )

        val = price_fn(theta_leaf)
        grad = torch.autograd.grad(val, theta_leaf, create_graph=True, retain_graph=True)[0]
        
        # Compute Hessian
        hess = torch.zeros(6, 6, dtype=dtype, device=self.device)
        for i in range(6):
            hess[i] = torch.autograd.grad(grad[i], theta_leaf, retain_graph=True)[0]

        return {
            "price": val.detach().item(),
            "gradient": grad.detach().cpu().numpy(),
            "hessian": hess.detach().cpu().numpy()
        }

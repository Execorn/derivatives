import torch
from typing import Union

# BS Pricing Formula and Vega in double precision
def price_bs_f64(
    sigma: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Black-Scholes option price in float64.
    Formula references:
        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        Call = S * e^(-q * T) * N(d1) - K * e^(-r * T) * N(d2)
        Put = K * e^(-r * T) * N(-d2) - S * e^(-q * T) * N(-d1)
    """
    sqrt_T = torch.sqrt(T)
    denom = sigma * sqrt_T
    
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    
    SQRT_2 = 1.4142135623730951
    phi_d1 = 0.5 * (1.0 + torch.erf(d1 / SQRT_2))
    phi_d2 = 0.5 * (1.0 + torch.erf(d2 / SQRT_2))
    
    exp_q = torch.exp(-q * T)
    exp_r = torch.exp(-r * T)
    
    call_price = S * exp_q * phi_d1 - K * exp_r * phi_d2
    put_price = K * exp_r * (1.0 - phi_d2) - S * exp_q * (1.0 - phi_d1)
    
    return torch.where(is_call, call_price, put_price)


def vega_bs_f64(
    sigma: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Black-Scholes option Vega in float64.
    Formula references:
        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        Vega = S * e^(-q * T) * sqrt(T) * N'(d1)
    """
    sqrt_T = torch.sqrt(T)
    denom = sigma * sqrt_T
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
    
    SQRT_2PI = 2.5066282746310005
    phi_d1 = torch.exp(-0.5 * d1**2) / SQRT_2PI
    
    vega = S * torch.exp(-q * T) * sqrt_T * phi_d1
    return vega


@torch.compile(mode="reduce-overhead")
def _solve_iv_compiled(
    price: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    is_call: torch.Tensor,
    max_iter: int = 30,
) -> torch.Tensor:
    """
    Triton-compilable hybrid bisection + Newton solver for implied volatility.
    Always returns a clone of the final tensor to prevent static buffer overwriting.
    """
    # Clamp inputs for numerical safety
    S_safe = torch.clamp(S, min=1e-8)
    K_safe = torch.clamp(K, min=1e-8)
    T_safe = torch.clamp(T, min=1e-8)
    
    # Boundary checks for clamping final values
    c_min = price_bs_f64(torch.tensor(0.01, device=price.device, dtype=torch.float64), S_safe, K_safe, T_safe, r, q, is_call)
    c_max = price_bs_f64(torch.tensor(5.0, device=price.device, dtype=torch.float64), S_safe, K_safe, T_safe, r, q, is_call)
    
    a = torch.full_like(price, 0.01)
    b = torch.full_like(price, 5.0)
    
    x = 0.5 * (a + b)
    
    for _ in range(max_iter):
        p_x = price_bs_f64(x, S_safe, K_safe, T_safe, r, q, is_call)
        f_val = p_x - price
        
        # Update search bracket [a, b]
        a = torch.where(f_val < 0.0, x, a)
        b = torch.where(f_val > 0.0, x, b)
        
        # Compute vega and newton step
        v = vega_bs_f64(x, S_safe, K_safe, T_safe, r, q)
        v_clamped = torch.clamp(v, min=1e-12)
        dx = - f_val / v_clamped
        x_newton = x + dx
        
        # Accept Newton step if it falls within the interval
        accept = (x_newton > a) & (x_newton < b)
        x_bisect = 0.5 * (a + b)
        x = torch.where(accept, x_newton, x_bisect)
        
        # Hard clamp volatility guess
        x = torch.clamp(x, min=0.01, max=5.0)
        
    # Set final results based on price boundary checks
    x = torch.where(price <= c_min, torch.tensor(0.01, device=price.device, dtype=torch.float64), x)
    x = torch.where(price >= c_max, torch.tensor(5.0, device=price.device, dtype=torch.float64), x)
    
    return x.clone()


def reduce_grad(grad: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Reduces the gradient tensor to the original shape by summing over broadcasted dimensions.
    """
    if grad.shape == original_shape:
        return grad
        
    if len(original_shape) == 0:
        return grad.sum()
        
    grad_ndim = grad.ndim
    orig_ndim = len(original_shape)
    
    sum_dims = []
    for i in range(grad_ndim):
        orig_dim_idx = orig_ndim - 1 - i
        if orig_dim_idx < 0:
            sum_dims.append(grad_ndim - 1 - i)
        else:
            orig_size = original_shape[orig_dim_idx]
            if orig_size == 1:
                sum_dims.append(grad_ndim - 1 - i)
                
    if sum_dims:
        grad = grad.sum(dim=sum_dims, keepdim=True)
        
    if orig_ndim < grad_ndim:
        grad = grad.reshape(original_shape)
    else:
        grad = grad.reshape(original_shape)
        
    return grad


class PIVOTImpliedVolFunction(torch.autograd.Function):
    """
    Custom autograd function for Jäckel-style implied volatility conversion (PIVOT).
    Includes low-vega gradient gating to prevent division-by-zero singularities during backpropagation.
    """
    @staticmethod
    def forward(ctx, price, S, K, T, r, q, is_call, vega_epsilon):
        device = price.device
        dtype = price.dtype
        
        ctx.price_shape = price.shape
        ctx.price_dtype = dtype
        ctx.vega_epsilon = vega_epsilon
        
        def to_f64(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float64)
            return torch.tensor(x, device=device, dtype=torch.float64)
            
        price_f64 = to_f64(price)
        S_f64 = to_f64(S)
        K_f64 = to_f64(K)
        T_f64 = to_f64(T)
        r_f64 = to_f64(r)
        q_f64 = to_f64(q)
        
        if isinstance(is_call, torch.Tensor):
            is_call_tensor = is_call.to(device=device, dtype=torch.bool)
        else:
            is_call_tensor = torch.tensor(is_call, device=device, dtype=torch.bool)
            
        price_f64, S_f64, K_f64, T_f64, r_f64, q_f64 = torch.broadcast_tensors(
            price_f64, S_f64, K_f64, T_f64, r_f64, q_f64
        )
        is_call_tensor = torch.broadcast_to(is_call_tensor, price_f64.shape)
        
        sigma_f64 = _solve_iv_compiled(price_f64, S_f64, K_f64, T_f64, r_f64, q_f64, is_call_tensor)
        
        ctx.save_for_backward(sigma_f64, S_f64, K_f64, T_f64, r_f64, q_f64)
        
        return sigma_f64.to(dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sigma, S, K, T, r, q = ctx.saved_tensors
        vega_epsilon = ctx.vega_epsilon
        
        S_safe = torch.clamp(S, min=1e-8)
        K_safe = torch.clamp(K, min=1e-8)
        T_safe = torch.clamp(T, min=1e-8)
        
        sqrt_T = torch.sqrt(T_safe)
        denom = sigma * sqrt_T
        d1 = (torch.log(S_safe / K_safe) + (r - q + 0.5 * sigma**2) * T_safe) / denom
        
        SQRT_2PI = 2.5066282746310005
        phi_d1 = torch.exp(-0.5 * d1**2) / SQRT_2PI
        
        vega = S_safe * torch.exp(-q * T_safe) * sqrt_T * phi_d1
        gated_vega = torch.clamp(vega, min=vega_epsilon)
        
        grad_price = grad_output.to(dtype=torch.float64) / gated_vega
        grad_price = reduce_grad(grad_price, ctx.price_shape)
        
        return grad_price.to(dtype=ctx.price_dtype), None, None, None, None, None, None, None


def pivot_implied_vol(
    price: torch.Tensor,
    S: Union[torch.Tensor, float],
    K: torch.Tensor,
    T: torch.Tensor,
    r: Union[torch.Tensor, float] = 0.0,
    q: Union[torch.Tensor, float] = 0.0,
    is_call: Union[torch.Tensor, bool] = True,
    vega_epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    Differentiable Implied Volatility Solver (PIVOT) using hybrid bisection + Newton-Raphson.
    Applies PIVOT Low-Vega Gradient Gating in the backward pass.
    
    Args:
        price: Predicted option price tensor.
        S: Spot price tensor or scalar.
        K: Strike tensor.
        T: Maturity tensor.
        r: Risk-free rate tensor or scalar.
        q: Dividend yield tensor or scalar.
        is_call: Boolean or boolean tensor indicating if call option.
        vega_epsilon: Threshold for gating minimum vega in backward pass.
        
    Returns:
        Implied volatility tensor of the same shape as price and original dtype.
    """
    return PIVOTImpliedVolFunction.apply(price, S, K, T, r, q, is_call, vega_epsilon)

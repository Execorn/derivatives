import torch
import time
import lifted_heston_cuda

def compute_weights_and_speeds(N, H):
    """
    Computes Bernstein approximation weights (c_i) and mean-reversion speeds (x_i)
    for the Lifted Heston model based on the Hurst exponent H.
    This uses a simplified geometric progression as a proxy.
    """
    alpha = H + 0.5
    # Simplified geometric grid for factors
    r_c = 2.5
    x_speeds = torch.tensor([1.5 * (r_c ** i) for i in range(N)], dtype=torch.float32)
    # Weights decay based on alpha
    c_weights = torch.tensor([ (r_c ** (i * (alpha - 1))) for i in range(N)], dtype=torch.float32)
    c_weights = c_weights / torch.sum(c_weights) # Normalize for testing
    return c_weights, x_speeds

def simulate_lifted_heston_python(num_paths, num_steps, dt, S0, V0, rho, kappa, theta, sigma, c_weights, x_speeds, seed):
    torch.manual_seed(seed)
    
    S_t = torch.full((num_paths,), S0, dtype=torch.float32)
    V_t = torch.full((num_paths,), V0, dtype=torch.float32)
    U_t = torch.zeros((num_paths, 20), dtype=torch.float32)
    
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    sqrt_1_minus_rho2 = torch.sqrt(torch.tensor(1.0 - rho**2))
    
    for _ in range(num_steps):
        Z1 = torch.randn(num_paths, dtype=torch.float32)
        Z2 = torch.randn(num_paths, dtype=torch.float32)
        
        Z_S = rho * Z1 + sqrt_1_minus_rho2 * Z2
        
        V_clipped = torch.clamp(V_t, min=0.0)
        sqrt_V = torch.sqrt(V_clipped)
        
        # Advance S_t
        S_t = S_t * torch.exp(-0.5 * V_clipped * dt + sqrt_V * sqrt_dt * Z_S)
        
        # Advance U_t
        V_clipped_expanded = V_clipped.unsqueeze(1) # (num_paths, 1)
        Z1_expanded = Z1.unsqueeze(1)               # (num_paths, 1)
        
        drift_vol_part = -kappa * V_clipped_expanded * dt + sigma * torch.sqrt(V_clipped_expanded) * sqrt_dt * Z1_expanded
        U_t = (U_t + drift_vol_part) / (1.0 + x_speeds.unsqueeze(0) * dt)
        
        # Update V_t
        V_t = V0 + torch.sum(c_weights.unsqueeze(0) * U_t, dim=1)
        
    return S_t

if __name__ == "__main__":
    num_paths = 100_000
    num_steps = 252
    dt = 1.0 / 252.0
    S0 = 100.0
    V0 = 0.04
    rho = -0.7
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    H = 0.1
    seed = 42
    
    c_weights, x_speeds = compute_weights_and_speeds(20, H)
    
    # 1. Pure Python Simulation
    print("Running Pure Python Simulation...")
    t0 = time.time()
    prices_py = simulate_lifted_heston_python(num_paths, num_steps, dt, S0, V0, rho, kappa, theta, sigma, c_weights, x_speeds, seed)
    t1 = time.time()
    time_py = t1 - t0
    print(f"Pure Python Time: {time_py:.4f} seconds")
    print(f"Mean Price (Python): {prices_py.mean().item():.4f}")
    
    # 2. CUDA Simulation
    print("\nRunning CUDA Simulation...")
    c_weights_cuda = c_weights.cuda().contiguous()
    x_speeds_cuda = x_speeds.cuda().contiguous()
    
    # Warmup
    _ = lifted_heston_cuda.simulate_lifted_heston(1024, num_steps, dt, S0, V0, rho, kappa, theta, sigma, c_weights_cuda, x_speeds_cuda, seed)
    torch.cuda.synchronize()
    
    t0 = time.time()
    prices_cu = lifted_heston_cuda.simulate_lifted_heston(num_paths, num_steps, dt, S0, V0, rho, kappa, theta, sigma, c_weights_cuda, x_speeds_cuda, seed)
    t1 = time.time()
    time_cu = t1 - t0
    print(f"CUDA Time: {time_cu:.4f} seconds")
    print(f"Mean Price (CUDA): {prices_cu.mean().item():.4f}")
    
    print(f"\nSpeedup: {time_py / time_cu:.2f}x")

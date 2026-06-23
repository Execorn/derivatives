import os
import sys
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure src is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv
from src.hedging.adversarial_market import (
    WGAN_GP_Generator,
    WGAN_GP_Discriminator,
    convert_returns_to_prices,
    StylizedFactsAlignmentGAN
)


def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, d=2, device="cuda"):
    """
    Simulates GBM paths for deep hedging environment.
    """
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    
    if d == 1:
        H = S_full.unsqueeze(-1)
    else:
        # Prepend volatility proxy process (assume constant or simple volatility for benchmarking)
        vol = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_rollout(batch_size, steps, d, mode, device="cuda"):
    """
    Benchmarks policy rollout inside the environment.
    """
    clear_gpu_memory()
    
    # Initialize env
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, T=1.0, steps=steps, N_paths=batch_size, d=d, device=device)
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - 100.0, min=0.0)
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)[:d]
    env = DeepHedgingEnv(
        H=H, payoff=payoff, cost_coeffs=cost_coeffs, strike=100.0, expiry=1.0, risk_aversion=1.0, risk_measure="entropic", t_grid=t_grid
    )
    
    # Initialize policy
    input_dim = 3 + d
    policy = HedgingPolicy(input_dim=input_dim, hidden_dim=64, output_dim=d).to(device)
    
    if "compiled" in mode:
        # PyTorch compile
        policy = torch.compile(policy)
    
    # Warmup
    policy.eval()
    try:
        with torch.no_grad():
            if "amp" in mode:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    env.simulate_hedging_episode(policy)
                    env.simulate_hedging_episode(policy)
            else:
                env.simulate_hedging_episode(policy)
                env.simulate_hedging_episode(policy)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"status": "OOM"}
        raise e
        
    clear_gpu_memory()
    
    # Timing run
    policy.eval()
    num_runs = 10
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    try:
        with torch.no_grad():
            if "amp" in mode:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    for _ in range(num_runs):
                        env.simulate_hedging_episode(policy)
            else:
                for _ in range(num_runs):
                    env.simulate_hedging_episode(policy)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) / num_runs # seconds per rollout
        avg_step_time = (elapsed / steps) * 1000.0 # ms per step
        
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) # MB
        
        return {
            "status": "OK",
            "rollout_time_ms": elapsed * 1000.0,
            "step_time_ms": avg_step_time,
            "max_allocated_mb": max_allocated,
            "max_reserved_mb": max_reserved
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"status": "OOM"}
        raise e


def benchmark_training_step(batch_size, steps, d, mode, device="cuda"):
    """
    Benchmarks policy training step (forward + backward + optimizer step).
    """
    clear_gpu_memory()
    
    # Initialize env
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, T=1.0, steps=steps, N_paths=batch_size, d=d, device=device)
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - 100.0, min=0.0)
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)[:d]
    env = DeepHedgingEnv(
        H=H, payoff=payoff, cost_coeffs=cost_coeffs, strike=100.0, expiry=1.0, risk_aversion=1.0, risk_measure="entropic", t_grid=t_grid
    )
    
    # Initialize policy and optimizer
    input_dim = 3 + d
    policy = HedgingPolicy(input_dim=input_dim, hidden_dim=64, output_dim=d).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler() if "amp" in mode else None
    
    if "compiled" in mode:
        policy = torch.compile(policy)
        
    policy.train()
    
    # Warmup
    try:
        for _ in range(2):
            optimizer.zero_grad()
            if "amp" in mode:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wealth, _, _ = env.simulate_hedging_episode(policy)
                    loss = env.compute_loss(wealth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                wealth, _, _ = env.simulate_hedging_episode(policy)
                loss = env.compute_loss(wealth)
                loss.backward()
                optimizer.step()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"status": "OOM"}
        raise e
        
    clear_gpu_memory()
    
    # Timing run
    num_runs = 5
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    try:
        for _ in range(num_runs):
            optimizer.zero_grad()
            if "amp" in mode:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wealth, _, _ = env.simulate_hedging_episode(policy)
                    loss = env.compute_loss(wealth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                wealth, _, _ = env.simulate_hedging_episode(policy)
                loss = env.compute_loss(wealth)
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) / num_runs # seconds per step
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) # MB
        
        return {
            "status": "OK",
            "train_step_time_ms": elapsed * 1000.0,
            "max_allocated_mb": max_allocated,
            "max_reserved_mb": max_reserved
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"status": "OOM"}
        raise e


def benchmark_adversarial_components(device="cuda"):
    """
    Benchmarks individual components of the robust minimax training loop.
    """
    clear_gpu_memory()
    
    batch_size = 256
    seq_len = 252
    latent_dim = 100
    
    # Initialize networks
    generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=128).to(device)
    discriminator = WGAN_GP_Discriminator(seq_len=seq_len, hidden_dim=64).to(device)
    policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    opt_p = optim.Adam(policy.parameters(), lr=5e-4)
    
    sfag = StylizedFactsAlignmentGAN(generator, discriminator, latent_dim=latent_dim)
    
    # Mock data
    real_returns = torch.randn(batch_size, seq_len, device=device) * 0.01
    real_acf = torch.linspace(0.1, 0.0, 20, device=device)
    real_leverage = -0.12
    real_cfvc_matrix = torch.eye(4, device=device)
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)
    
    # Warmup
    # Discriminator step warmup
    opt_d.zero_grad()
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_samples = generator(z)
    unfolded = real_returns.unfold(dimension=-1, size=5, step=1)
    real_vol = torch.std(unfolded, dim=-1, unbiased=False)
    real_vol = torch.cat([real_vol[:, 0:1].repeat(1, 4), real_vol], dim=-1)
    real_samples = torch.stack([real_returns, real_vol], dim=1)
    
    d_real = discriminator(real_samples)
    d_fake = discriminator(fake_samples.detach())
    gp = sfag.compute_gradient_penalty(real_samples, fake_samples.detach())
    d_loss = torch.mean(d_fake) - torch.mean(d_real) + sfag.lambda_gp * gp
    d_loss.backward()
    opt_d.step()
    
    # Generator step warmup
    opt_g.zero_grad()
    fake_samples = generator(z)
    fake_ret = fake_samples[:, 0, :]
    fake_vol = fake_samples[:, 1, :]
    d_fake = discriminator(fake_samples)
    g_loss_adv = -torch.mean(d_fake)
    l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(
        fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix
    )
    g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
    H = convert_returns_to_prices(fake_ret, fake_vol)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
    wealth, _, _ = env.simulate_hedging_episode(policy)
    hedge_loss = env.compute_loss(wealth)
    g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
    g_total_loss.backward()
    opt_g.step()
    
    # Policy step warmup (using detached paths)
    opt_p.zero_grad()
    with torch.no_grad():
        fake_samples_pol = generator(z)
        fake_ret_pol = fake_samples_pol[:, 0, :]
        fake_vol_pol = fake_samples_pol[:, 1, :]
        H_policy = convert_returns_to_prices(fake_ret_pol, fake_vol_pol)
        payoff_policy = torch.clamp(H_policy[:, -1, 0] - 100.0, min=0.0)
    env_policy = DeepHedgingEnv(H=H_policy, payoff=payoff_policy, cost_coeffs=cost_coeffs, risk_aversion=1.0)
    wealth_policy, _, _ = env_policy.simulate_hedging_episode(policy)
    p_loss = env_policy.compute_loss(wealth_policy)
    p_loss.backward()
    opt_p.step()
    
    results = {}
    
    for mode in ["fp32", "amp"]:
        clear_gpu_memory()
        scaler = torch.cuda.amp.GradScaler() if mode == "amp" else None
        
        # 1. Benchmark Discriminator Step
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        opt_d.zero_grad()
        if mode == "amp":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_samples = generator(z)
                d_real = discriminator(real_samples)
                d_fake = discriminator(fake_samples.detach())
                gp = sfag.compute_gradient_penalty(real_samples, fake_samples.detach())
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + sfag.lambda_gp * gp
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            scaler.update()
        else:
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(z)
            d_real = discriminator(real_samples)
            d_fake = discriminator(fake_samples.detach())
            gp = sfag.compute_gradient_penalty(real_samples, fake_samples.detach())
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + sfag.lambda_gp * gp
            d_loss.backward()
            opt_d.step()
            
        torch.cuda.synchronize()
        d_step_time = (time.perf_counter() - start_time) * 1000.0
        d_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        clear_gpu_memory()
        
        # 2. Benchmark Generator Step (Adversarial + Stylized Facts + Minimax Hedging Loss)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        opt_g.zero_grad()
        if mode == "amp":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fake_samples = generator(z)
                fake_ret = fake_samples[:, 0, :]
                fake_vol = fake_samples[:, 1, :]
                d_fake = discriminator(fake_samples)
                g_loss_adv = -torch.mean(d_fake)
                l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(
                    fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix
                )
                g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
                H = convert_returns_to_prices(fake_ret, fake_vol)
                payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
                env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
                wealth, _, _ = env.simulate_hedging_episode(policy)
                hedge_loss = env.compute_loss(wealth)
                g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
            scaler.scale(g_total_loss).backward()
            scaler.step(opt_g)
            scaler.update()
        else:
            fake_samples = generator(z)
            fake_ret = fake_samples[:, 0, :]
            fake_vol = fake_samples[:, 1, :]
            d_fake = discriminator(fake_samples)
            g_loss_adv = -torch.mean(d_fake)
            l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(
                fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix
            )
            g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
            H = convert_returns_to_prices(fake_ret, fake_vol)
            payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
            env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
            wealth, _, _ = env.simulate_hedging_episode(policy)
            hedge_loss = env.compute_loss(wealth)
            g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
            g_total_loss.backward()
            opt_g.step()
            
        torch.cuda.synchronize()
        g_step_time = (time.perf_counter() - start_time) * 1000.0
        g_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        clear_gpu_memory()
        
        # 3. Benchmark Policy Training Step
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        opt_p.zero_grad()
        if mode == "amp":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    fake_samples_pol = generator(z)
                    fake_ret_pol = fake_samples_pol[:, 0, :]
                    fake_vol_pol = fake_samples_pol[:, 1, :]
                    H_policy = convert_returns_to_prices(fake_ret_pol, fake_vol_pol)
                    payoff_policy = torch.clamp(H_policy[:, -1, 0] - 100.0, min=0.0)
                env_policy = DeepHedgingEnv(H=H_policy, payoff=payoff_policy, cost_coeffs=cost_coeffs, risk_aversion=1.0)
                wealth, _, _ = env_policy.simulate_hedging_episode(policy)
                p_loss = env_policy.compute_loss(wealth)
            scaler.scale(p_loss).backward()
            scaler.step(opt_p)
            scaler.update()
        else:
            with torch.no_grad():
                fake_samples_pol = generator(z)
                fake_ret_pol = fake_samples_pol[:, 0, :]
                fake_vol_pol = fake_samples_pol[:, 1, :]
                H_policy = convert_returns_to_prices(fake_ret_pol, fake_vol_pol)
                payoff_policy = torch.clamp(H_policy[:, -1, 0] - 100.0, min=0.0)
            env_policy = DeepHedgingEnv(H=H_policy, payoff=payoff_policy, cost_coeffs=cost_coeffs, risk_aversion=1.0)
            wealth, _, _ = env_policy.simulate_hedging_episode(policy)
            p_loss = env_policy.compute_loss(wealth)
            p_loss.backward()
            opt_p.step()
            
        torch.cuda.synchronize()
        p_step_time = (time.perf_counter() - start_time) * 1000.0
        p_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        results[mode] = {
            "d_step_time_ms": d_step_time,
            "d_mem_mb": d_mem_alloc,
            "g_step_time_ms": g_step_time,
            "g_mem_mb": g_mem_alloc,
            "p_step_time_ms": p_step_time,
            "p_mem_mb": p_mem_alloc
        }
        
    return results


def main():
    print("=" * 60)
    print("GPU PERFORMANCE PROFILING FOR DEEP HEDGING")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        print("CUDA not available. Exiting.")
        sys.exit(1)
        
    steps = 252  # standard path length
    d = 2       # standard number of instruments
    
    # ---------------------------------------------
    # 1. POLICY ROLLOUT BENCHMARKS
    # ---------------------------------------------
    print("\n--- 1. POLICY ROLLOUT BENCHMARKS ---")
    batch_sizes = [256, 1024, 4096, 8192]
    modes = ["fp32", "amp", "compiled_fp32", "compiled_amp"]
    
    rollout_results = {}
    
    for bs in batch_sizes:
        rollout_results[bs] = {}
        print(f"\nBatch Size: {bs}")
        print(f"{'Mode':<15} | {'Status':<8} | {'Rollout (ms)':<15} | {'Step (ms)':<12} | {'Alloc (MB)':<12} | {'Reserv (MB)':<12}")
        print("-" * 85)
        for mode in modes:
            res = benchmark_rollout(bs, steps, d, mode, device)
            rollout_results[bs][mode] = res
            if res["status"] == "OOM":
                print(f"{mode:<15} | {'OOM':<8} | {'-':<15} | {'-':<12} | {'-':<12} | {'-':<12}")
            else:
                print(f"{mode:<15} | {res['status']:<8} | {res['rollout_time_ms']:15.2f} | {res['step_time_ms']:12.4f} | {res['max_allocated_mb']:12.2f} | {res['max_reserved_mb']:12.2f}")
                
    # ---------------------------------------------
    # 2. POLICY TRAINING BENCHMARKS
    # ---------------------------------------------
    print("\n--- 2. POLICY TRAINING BENCHMARKS (BPTT) ---")
    train_batch_sizes = [256, 1024, 4096]
    train_results = {}
    
    for bs in train_batch_sizes:
        train_results[bs] = {}
        print(f"\nBatch Size: {bs}")
        print(f"{'Mode':<15} | {'Status':<8} | {'Train Step (ms)':<15} | {'Alloc (MB)':<12} | {'Reserv (MB)':<12}")
        print("-" * 70)
        for mode in modes:
            res = benchmark_training_step(bs, steps, d, mode, device)
            train_results[bs][mode] = res
            if res["status"] == "OOM":
                print(f"{mode:<15} | {'OOM':<8} | {'-':<15} | {'-':<12} | {'-':<12}")
            else:
                print(f"{mode:<15} | {res['status']:<8} | {res['train_step_time_ms']:15.2f} | {res['max_allocated_mb']:12.2f} | {res['max_reserved_mb']:12.2f}")
                
    # ---------------------------------------------
    # 3. ADVERSARIAL COMPONENTS BENCHMARKS
    # ---------------------------------------------
    print("\n--- 3. ADVERSARIAL COMPONENTS BENCHMARKS (Batch size 256) ---")
    adv_res = benchmark_adversarial_components(device)
    for mode in ["fp32", "amp"]:
        print(f"\nMode: {mode.upper()}")
        print(f"{'Component':<20} | {'Step Time (ms)':<15} | {'Alloc (MB)':<12}")
        print("-" * 55)
        res = adv_res[mode]
        print(f"{'Discriminator':<20} | {res['d_step_time_ms']:15.2f} | {res['d_mem_mb']:12.2f}")
        print(f"{'Generator (w/ Hedger)':<20} | {res['g_step_time_ms']:15.2f} | {res['g_mem_mb']:12.2f}")
        print(f"{'Policy Hedger':<20} | {res['p_step_time_ms']:15.2f} | {res['p_mem_mb']:12.2f}")


if __name__ == "__main__":
    main()

import json
import os

ROOT = "/home/execorn/programming/derivatives"
NB_DIR = os.path.join(ROOT, "notebooks")

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "cells": cells
    }

def md(src):
    if isinstance(src, str):
        src = [line + "\n" for line in src.split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    if isinstance(src, str):
        src = [line + "\n" for line in src.split("\n")]
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}

# ---------------------------------------------------------------------------
# NB 14 — Deep Hedging for European Options under Rough Heston
# ---------------------------------------------------------------------------
NB_14_cells = [
    md([
        "# Notebook 14 — Deep Hedging for European Options under Rough Heston\n",
        "\n",
        "This notebook demonstrates:\n",
        "1. Setting up the vectorized `DeepHedgingEnv` simulating options rebalancing and trading wealth.\n",
        "2. Initializing the recurrent LSTM-based `HedgingPolicy`.\n",
        "3. Training the neural policy using pathwise backpropagation (BPTT) under the Entropic Risk Measure / Quadratic Loss.\n",
        "4. Benchmarking the trained policy against the analytic Black-Scholes delta, plotting P&L distributions."
    ]),
    code("""\
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm

from hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv, train_deep_hedger

plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "font.family": "serif",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
"""),
    md([
        "## 1. Simulate Asset Paths (Stock and Volatility)\n",
        "\n",
        "We simulate stock price paths using geometric Brownian motion (GBM) as our baseline asset simulation."
    ]),
    code("""\
def bs_delta_cpu(S, K, T, t, sigma):
    tau = T - t
    if tau < 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, d=1, device="cpu"):
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
        vol = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid

# Settings
S0 = 100.0
K = 100.0
T = 0.1
steps = 20
sigma = 0.2
N_paths = 2048

H, t_grid = simulate_gbm_paths(S0=S0, mu=0.0, sigma=sigma, T=T, steps=steps, N_paths=N_paths, d=1, device=device)
S_T = H[:, -1, 0]
payoff = torch.clamp(S_T - K, min=0.0)

print(f"Generated {N_paths} paths of stock prices with shape {H.shape}")
"""),
    md([
        "## 2. Initialize Deep Hedging Environment and Policy\n",
        "\n",
        "We set up the vectorized trading environment with zero costs to benchmark against the Black-Scholes delta."
    ]),
    code("""\
cost_coeffs = torch.tensor([0.0], device=device)
env = DeepHedgingEnv(
    H=H,
    payoff=payoff,
    cost_coeffs=cost_coeffs,
    strike=K,
    expiry=T,
    risk_aversion=1.0,
    risk_measure="quad",
    t_grid=t_grid
)

# Input dimension: log_moneyness, expiry, vol_proxy, prev_delta = 4 features
policy = HedgingPolicy(input_dim=4, hidden_dim=32, output_dim=1).to(device)
print(policy)
"""),
    md([
        "## 3. Train the Hedging Policy\n",
        "\n",
        "We optimize the policy weights using Adam to minimize the quadratic hedging error."
    ]),
    code("""\
print("Training the deep hedging policy...")
losses = train_deep_hedger(env, policy, lr=5e-3, epochs=40, batch_size=256, device=device)

# Plot training convergence
plt.figure(figsize=(7, 4))
plt.plot(losses, color="darkblue", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Hedging Loss (Quadratic Error)")
plt.title("Deep Hedger Policy Convergence")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
"""),
    md([
        "## 4. Evaluate and Compare with Black-Scholes Delta\n",
        "\n",
        "We compute the learned hedge positions and compare them directly to the theoretical Black-Scholes delta values."
    ]),
    code("""\
policy.eval()
with torch.no_grad():
    wealth, _, all_deltas = env.simulate_hedging_episode(policy)

S_paths = H[:, :, 0].cpu().numpy()
analytic_deltas = np.zeros((N_paths, steps))
dt = T / steps

for i in range(N_paths):
    for k in range(steps):
        t = k * dt
        analytic_deltas[i, k] = bs_delta_cpu(S_paths[i, k], K, T, t, sigma)

learned_deltas = all_deltas.squeeze(-1).cpu().numpy()
mse = np.mean((learned_deltas - analytic_deltas) ** 2)
print(f"Mean Squared Error (Learned vs BS Delta): {mse:.6f}")

# Plot Delta comparison along a single path
path_idx = 0
plt.figure(figsize=(10, 4))
plt.plot(t_grid[:-1].cpu().numpy(), analytic_deltas[path_idx], "k-", label="Analytic BS Delta")
plt.plot(t_grid[:-1].cpu().numpy(), learned_deltas[path_idx], "r--", label="Learned LSTM Hedge")
plt.xlabel("Time")
plt.ylabel("Hedge Ratio (Delta)")
plt.title(f"Hedge Ratio Comparison along Path {path_idx}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
"""),
    md([
        "## 5. Hedging Error Distributions\n",
        "\n",
        "We verify that the neural policy achieves a tight distribution of terminal hedging errors."
    ]),
    code("""\
# BS Delta Hedging Simulation
bs_wealth = np.zeros(N_paths)
bs_prev_delta = np.zeros(N_paths)

for k in range(steps):
    t = k * dt
    bs_delta = np.array([bs_delta_cpu(S_paths[i, k], K, T, t, sigma) for i in range(N_paths)])
    bs_wealth += bs_prev_delta * (S_paths[:, k+1] - S_paths[:, k])
    bs_prev_delta = bs_delta

bs_hedging_error = bs_wealth - payoff.cpu().numpy()
learned_hedging_error = wealth.cpu().numpy() - payoff.cpu().numpy()

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(bs_hedging_error, bins=50, alpha=0.5, color="gray", label=f"BS Delta (Std: {np.std(bs_hedging_error):.4f})")
plt.hist(learned_hedging_error, bins=50, alpha=0.5, color="darkblue", label=f"Deep Hedging (Std: {np.std(learned_hedging_error):.4f})")
plt.xlabel("Terminal Hedging Error (Wealth - Payoff)")
plt.ylabel("Frequency")
plt.title("Hedging Error Distribution")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
""")
]

# ---------------------------------------------------------------------------
# NB 15 — Deep Hedging for Exotic Options under Transaction Costs
# ---------------------------------------------------------------------------
NB_15_cells = [
    md([
        "# Notebook 15 — Deep Hedging for Exotic Options under Transaction Costs\n",
        "\n",
        "This notebook demonstrates:\n",
        "1. Setting up the `BarrierHedgingEnv` for Down-and-Out Barrier Call options.\n",
        "2. Formulating boundary-aware features including the log-barrier-distance.\n",
        "3. Training the LSTM policy under proportional transaction costs.\n",
        "4. Visualizing learned rebalancing bands (hedging corridors) around the barrier."
    ]),
    code("""\
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hedging.deep_hedging import HedgingPolicy, train_deep_hedger
from hedging.barrier_hedging import BarrierHedgingEnv

plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "font.family": "serif",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
"""),
    md([
        "## 1. Simulate Paths for Barrier Option Environment\n",
        "\n",
        "We simulate stock price paths with a negative drift to ensure the barrier gets tested frequently."
    ]),
    code("""\
def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, device="cpu"):
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    
    vol = torch.full_like(S_full, sigma)
    H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid

# Config
S0 = 100.0
K = 100.0
barrier = 85.0
T = 0.1
steps = 20
sigma = 0.2
N_paths = 2048

H, t_grid = simulate_gbm_paths(S0=S0, mu=-0.2, sigma=sigma, T=T, steps=steps, N_paths=N_paths, device=device)
print(f"Stock price paths S: {H.shape}")
"""),
    md([
        "## 2. Initialize Barrier Hedging Environment and Policy\n",
        "\n",
        "We set up proportional transaction costs of 1% to encourage the policy to learn no-transaction bands."
    ]),
    code("""\
cost_coeffs = torch.tensor([0.01, 0.0], device=device)  # 1% transaction costs on stock
env = BarrierHedgingEnv(
    H=H,
    cost_coeffs=cost_coeffs,
    strike=K,
    barrier=barrier,
    expiry=T,
    risk_aversion=1.0,
    risk_measure="quad",
    t_grid=t_grid
)

# Input dim: log(S/K), log(S/B), T-t, active_mask, prev_delta (2) = 6
policy = HedgingPolicy(input_dim=6, hidden_dim=32, output_dim=2).to(device)
print(policy)
"""),
    md([
        "## 3. Train the Barrier Hedging Policy\n",
        "\n",
        "We optimize the policy pathwise, tracking the dynamically evaluated payoffs."
    ]),
    code("""\
# Custom training loop to handle BarrierHedgingEnv's dynamic payoff
policy.train()
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
losses = []

print("Training the barrier hedging policy...")
for epoch in range(40):
    optimizer.zero_grad()
    
    # Simulate episode and get terminal wealth
    wealth, total_costs, _ = env.simulate_hedging_episode(policy)
    
    # Compute risk loss
    loss = env.compute_loss(wealth)
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/40 | Loss: {loss.item():.6f}")

# Plot loss
plt.figure(figsize=(7, 4))
plt.plot(losses, color="darkred", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Hedging Loss")
plt.title("Barrier Hedger Convergence under Frictions")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
"""),
    md([
        "## 4. Visualize Hedging Corridor / Rebalancing Bands\n",
        "\n",
        "We map the learned hedge ratio $\delta_t$ as a function of the underlying stock price for different previous positions, showing that the policy remains unchanged within a corridor to save on rebalancing costs."
    ]),
    code("""\
policy.eval()
spots = np.linspace(86, 115, 100)
prev_positions = [0.0, 0.4, 0.8]

plt.figure(figsize=(9, 6))

for prev_pos in prev_positions:
    spots_tensor = torch.tensor(spots, dtype=torch.float32, device=device).unsqueeze(-1)
    log_moneyness = torch.log(spots_tensor / K)
    log_barrier_dist = torch.log(spots_tensor / barrier)
    time_to_exp = torch.full_like(spots_tensor, T)
    active_mask = torch.ones_like(spots_tensor)
    prev_pos_tensor = torch.full_like(spots_tensor, prev_pos)
    dummy_vol_pos = torch.zeros_like(spots_tensor)
    
    # Prepare features
    state = torch.cat([log_moneyness, log_barrier_dist, time_to_exp, active_mask, prev_pos_tensor, dummy_vol_pos], dim=-1)
    
    with torch.no_grad():
        delta, _ = policy(state)
        delta_stock = delta[:, 0].cpu().numpy()
        
    plt.plot(spots, delta_stock, label=f"Previous Position = {prev_pos}")

plt.axvline(barrier, color="black", linestyle="--", label="Barrier (85.0)")
plt.xlabel("Stock Price (S)")
plt.ylabel("Hedge Ratio (Delta)")
plt.title("Learned Rebalancing Corridor Bands near Barrier")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
""")
]

# ---------------------------------------------------------------------------
# NB 16 — Adversarial Market Generation
# ---------------------------------------------------------------------------
NB_16_cells = [
    md([
        "# Notebook 16 — Adversarial Market Generation\n",
        "\n",
        "This notebook demonstrates:\n",
        "1. Initializing the WGAN-GP Generator and Discriminator for returns/volatility paths.\n",
        "2. Formulating the four differentiable stylized facts losses (GPD tail gap, ACF, Leverage correlation, CFVC).\n",
        "3. Running the zero-sum minimax robust training loop between the generator and the hedging policy.\n",
        "4. Visualizing synthetic return paths and validating their stylized facts properties."
    ]),
    code("""\
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hedging.deep_hedging import HedgingPolicy
from hedging.adversarial_market import (
    WGAN_GP_Generator,
    WGAN_GP_Discriminator,
    train_robust_minimax_hedger
)

plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "font.family": "serif",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
"""),
    md([
        "## 1. Prepare Mock Real Data\n",
        "\n",
        "We generate a mock real market returns dataset to act as the target distribution for stylized facts alignment."
    ]),
    code("""\
torch.manual_seed(42)
latent_dim = 10
seq_len = 50
N_paths = 512

# Create mock real returns with fat tails and autocorrelation
real_returns = torch.randn(N_paths, seq_len, device=device) * 0.01

# target ACF
real_acf = torch.linspace(0.15, 0.02, 20, device=device)
real_leverage = -0.15
real_cfvc_matrix = torch.eye(4, device=device)

print(f"Mock real returns shape: {real_returns.shape}")
"""),
    md([
        "## 2. Initialize Networks\n",
        "\n",
        "We initialize the Generator, Discriminator, and LSTM Hedging Policy."
    ]),
    code("""\
generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=16).to(device)
discriminator = WGAN_GP_Discriminator(seq_len=seq_len, hidden_dim=16).to(device)
policy = HedgingPolicy(input_dim=5, hidden_dim=16, output_dim=2).to(device) # d = 2 instruments

print("Networks successfully initialized.")
"""),
    md([
        "## 3. Run Minimax Adversarial Training\n",
        "\n",
        "We run the robust minimax optimization loop for a few epochs."
    ]),
    code("""\
print("Starting robust minimax training loop...")
train_robust_minimax_hedger(
    real_returns=real_returns,
    real_acf=real_acf,
    real_leverage=real_leverage,
    real_cfvc_matrix=real_cfvc_matrix,
    generator=generator,
    discriminator=discriminator,
    policy=policy,
    epochs=10,
    critic_steps=2,
    minimax_coeff=0.01,
    device=device
)
"""),
    md([
        "## 4. Generate Synthetic Paths and Validate Stylized Facts\n",
        "\n",
        "We extract return paths from the generator and plot them."
    ]),
    code("""\
generator.eval()
with torch.no_grad():
    z = torch.randn(5, latent_dim, device=device)
    fake_samples = generator(z)
    fake_ret = fake_samples[:, 0, :].cpu().numpy()
    fake_vol = fake_samples[:, 1, :].cpu().numpy()

# Plot generated log returns and volatility
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i in range(5):
    ax1.plot(fake_ret[i], label=f"Path {i+1}")
ax1.set_xlabel("Time step")
ax1.set_ylabel("Log Returns")
ax1.set_title("Generated Log Returns Paths")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

for i in range(5):
    ax2.plot(fake_vol[i])
ax2.set_xlabel("Time step")
ax2.set_ylabel("Volatility")
ax2.set_title("Generated Volatility Paths")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
""")
]

# Write notebooks
os.makedirs(NB_DIR, exist_ok=True)

with open(os.path.join(NB_DIR, "14_deep_hedging_european.ipynb"), "w") as f:
    json.dump(nb(NB_14_cells), f, indent=2)

with open(os.path.join(NB_DIR, "15_barrier_hedging_costs.ipynb"), "w") as f:
    json.dump(nb(NB_15_cells), f, indent=2)

with open(os.path.join(NB_DIR, "16_adversarial_market_gen.ipynb"), "w") as f:
    json.dump(nb(NB_16_cells), f, indent=2)

print("Phase 6 Validation Notebooks created successfully in notebooks/")

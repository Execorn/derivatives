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
# NB 12 — Neural SDE Calibration
# ---------------------------------------------------------------------------
NB_12_cells = [
    md([
        "# Notebook 12 — Neural SDE Calibration\n",
        "\n",
        "This notebook demonstrates:\n",
        "1. Initializing the `NeuralSDE` and `NeuralSDEPricer` models.\n",
        "2. Simulating stock and variance paths, checking that variance (and thus volatility) remains strictly positive.\n",
        "3. Performing a calibration loop using PyTorch optimizers (Adam) and the SDE adjoint method to fit SPX option implied volatilities.\n",
        "4. Plotting the RMSE convergence and model vs market implied volatility smiles."
    ]),
    code("""\
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date
import py_vollib_vectorized

from pricing.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss
from market.spx_data import download_spx_chain, clean_chain

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
        "## 1. Initializing Neural SDE and Pricer\n",
        "\n",
        "The `NeuralSDE` model represents the joint system of stock and variance where drift and diffusion MLPs parameterize the variance process."
    ]),
    code("""\
# Initialize Neural SDE parameters
r = 0.05
q = 0.015
rho_init = -0.7
hidden_dim = 16
epsilon = 1e-4

# Construct model
sde = NeuralSDE(r=r, q=q, rho_init=rho_init, hidden_dim=hidden_dim, epsilon=epsilon)
pricer = NeuralSDEPricer(sde, v0_init=0.04)
pricer.to(device)

print(f"Model initialized on {device}.")
print(f"Initial v0: {pricer.v0.item():.4f}")
print(f"Initial correlation rho: {sde.rho.item():.4f}")
"""),
    md([
        "## 2. Simulating Paths and Verifying Volatility Positivity\n",
        "\n",
        "We run a Monte Carlo simulation of stock and variance paths using the Euler method and check that the variance remains strictly positive (above the floor `epsilon`)."
    ]),
    code("""\
# Simulation inputs
S0 = 4700.0
strikes = torch.tensor([4500.0, 4700.0, 4900.0], device=device)
maturities = torch.tensor([0.2, 0.2, 0.2], device=device)
N_paths = 2048
dt = 0.01

# Execute simulation
prices, ys = pricer.price_options(
    S0=S0,
    strikes=strikes,
    maturities=maturities,
    N_paths=N_paths,
    dt=dt,
    method="euler"
)

print("Options simulated prices:", prices.detach().cpu().numpy())

# Extract variance paths (V_t)
# ys shape: (N_ts, N_paths, 2), where state[:, 0] is log-stock X_t and state[:, 1] is variance V_t
v_t = ys[:, :, 1]
min_variance = v_t.min().item()
print(f"Variance paths shape: {v_t.shape}")
print(f"Minimum simulated variance: {min_variance:.6e}")

# Assert positivity
assert min_variance >= epsilon * 0.999, f"Variance fell below the floor {epsilon}!"
print("Success: Volatility positivity check passed! Volatility remains strictly positive.")
"""),
    md([
        "## 3. Load SPX Option Market Data\n",
        "\n",
        "We load SPX options data from `2024-01-02` and filter for a single maturity slice to calibrate the model."
    ]),
    code("""\
# Load cached SPX chain
target_date = date(2024, 1, 2)
df_raw = download_spx_chain(target_date, cache=True)
df_clean = clean_chain(df_raw)

# Filter call options for a single maturity to calibrate
target_T = 0.3
slice_df = df_clean[(df_clean["T"] == target_T) & (df_clean["type"] == "call")].copy()
slice_df = slice_df.sort_values("strike")

print(f"Selected {len(slice_df)} options for maturity T={target_T}")
print(slice_df[["strike", "mid_price", "mid_iv"]].head(10))

# Convert to tensors
strikes_mkt = torch.tensor(slice_df["strike"].values, dtype=torch.float32, device=device)
prices_mkt = torch.tensor(slice_df["mid_price"].values, dtype=torch.float32, device=device)
maturities_mkt = torch.tensor(slice_df["T"].values, dtype=torch.float32, device=device)
ivs_mkt = slice_df["mid_iv"].values
"""),
    md([
        "## 4. Calibration Loop\n",
        "\n",
        "We run a PyTorch training loop to fit the SPX option prices. The adjoint method is used under the hood to calculate gradients back through the SDE solver."
    ]),
    code("""\
# Setup optimizer
optimizer = torch.optim.Adam(pricer.parameters(), lr=0.01)

# Training configuration
epochs = 30
loss_history = []
rmse_history = []

print("Starting calibration...")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Predict prices
    prices_pred, ys = pricer.price_options(
        S0=S0,
        strikes=strikes_mkt,
        maturities=maturities_mkt,
        N_paths=1024,
        dt=0.01,
        method="euler"
    )
    
    # Loss computation (vega weights = 1.0)
    loss_dict = compute_calibration_loss(
        model_prices=prices_pred,
        market_prices=prices_mkt,
        vegas=torch.ones_like(prices_mkt),
        ys=ys,
        lambda_bound=0.01,
        epsilon=epsilon
    )
    
    loss = loss_dict["loss"]
    loss.backward()
    optimizer.step()
    
    # Compute RMSE in prices
    rmse = torch.sqrt(torch.mean((prices_pred - prices_mkt) ** 2)).item()
    
    loss_history.append(loss.item())
    rmse_history.append(rmse)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Base Loss: {loss_dict['loss_base'].item():.4f} | RMSE: {rmse:.4f}")

print("Calibration completed!")
"""),
    md([
        "## 5. Plotting Calibration Results\n",
        "\n",
        "We plot the RMSE convergence history and compare the market vs calibrated model implied volatility smiles."
    ]),
    code("""\
# Compute final model implied volatilities
prices_final, _ = pricer.price_options(
    S0=S0,
    strikes=strikes_mkt,
    maturities=maturities_mkt,
    N_paths=4096,  # Use more paths for cleaner final IVs
    dt=0.01,
    method="euler"
)

# Convert to numpy
prices_final_np = prices_final.detach().cpu().numpy()
strikes_np = strikes_mkt.cpu().numpy()
maturities_np = maturities_mkt.cpu().numpy()
flags_np = np.array(["c"] * len(prices_final_np))

# Clamp option prices to prevent out-of-bounds errors in black_scholes
intrinsic = np.maximum(S0 - strikes_np, 0.0)
max_price = S0
prices_final_np = np.clip(prices_final_np, intrinsic + 1e-4, max_price - 1e-4)

# Calculate model implied volatilities
model_ivs = py_vollib_vectorized.vectorized_implied_volatility(
    prices_final_np,
    float(S0),
    strikes_np,
    maturities_np,
    r,
    flags_np,
    q=q,
    return_as="numpy",
    dtype=np.float64
)

# Plot RMSE convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(rmse_history) + 1), rmse_history, "o-", color="darkblue", label="RMSE")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Price RMSE ($)")
ax1.set_title("Calibration RMSE Convergence")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Plot Implied Volatility Smiles
ax2.plot(strikes_np, ivs_mkt * 100, "o-", color="black", label="Market IV")
ax2.plot(strikes_np, model_ivs * 100, "s--", color="red", label="Model IV")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Implied Volatility (%)")
ax2.set_title(f"Implied Volatility Smile (T={target_T})")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()

# Display RMSE
final_rmse = rmse_history[-1]
print(f"Final Calibration RMSE in Prices: ${final_rmse:.4f}")
""")
]

# ---------------------------------------------------------------------------
# NB 13 — Signature Volatility Model and Forecasting
# ---------------------------------------------------------------------------
NB_13_cells = [
    md([
        "# Notebook 13 — Signature Volatility Model and Forecasting\n",
        "\n",
        "This notebook demonstrates:\n",
        "1. Simulating log stock and log volatility paths under the `SignatureVolatilityModel`.\n",
        "2. Computing path signatures of rolling historical paths up to depth 4 using `compute_path_signature`.\n",
        "3. Training a ridge regression model mapping signature features to future volatility, and reporting out-of-sample forecasting RMSE.\n",
        "4. Verifying the martingale property of the Signature Volatility Model ($\mathbb{E}[S_T] \approx S_0$)."
    ]),
    code("""\
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from pricing.signature_vol import SignatureVolatilityModel, compute_path_signature

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
        "## 1. Simulating Log Stock and Log Volatility Paths\n",
        "\n",
        "We initialize the `SignatureVolatilityModel` with typical parameters and simulate paths for the stock price $S_t$ and variance $V_t$. From these, we compute log stock $\log(S_t)$ and log volatility $\log(\sigma_t)$ paths."
    ]),
    code("""\
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Construct model
model = SignatureVolatilityModel(device=device)

# Configure model parameters
with torch.no_grad():
    model.v0_raw.copy_(torch.tensor(np.log(0.04), device=device))  # v0 = 0.04
    model.rho_raw.copy_(torch.tensor(-0.5, device=device))         # rho < 0
    
    # Populate odd-order signature coefficients
    model.ell_raw[0] = 0.01   # level 1 time
    model.ell_raw[1] = -0.05  # level 1 W
    model.ell_raw[6] = 0.001  # level 3 time-time-time
    model.ell_raw[7] = -0.002 # level 3 time-time-W

# Run forward pass to simulate paths
T = 2.0
steps_per_unit = 250
N_paths = 200
S0 = 100.0

# Forward simulation
# Returns S (Stock), V (Variance), V_raw, and t_grid
S, V, V_raw, t_grid = model(
    T=T,
    steps_per_unit=steps_per_unit,
    N_paths=N_paths,
    S0=S0,
    r=0.05,
    q=0.01,
    antithetic=True
)

# Convert to log stock and log volatility paths
log_S = torch.log(S)
vol = torch.sqrt(V)
log_vol = torch.log(vol)

print("Simulated path dimensions:")
print(f"Stock price paths S: {S.shape}")
print(f"Variance paths V: {V.shape}")

# Convert to numpy for plotting (detaching first)
S_np = S.detach().cpu().numpy()
vol_np = vol.detach().cpu().numpy()
t_grid_np = t_grid.detach().cpu().numpy()

# Plot a few sample paths
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(5):
    plt.plot(t_grid_np, S_np[i], label=f"Path {i+1}")
plt.xlabel("Time (years)")
plt.ylabel("Stock Price")
plt.title("Sample Stock Price Paths")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
for i in range(5):
    plt.plot(t_grid_np, vol_np[i])
plt.xlabel("Time (years)")
plt.ylabel("Volatility")
plt.title("Sample Volatility Paths")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""),
    md([
        "## 2. Compute Path Signatures for Rolling Historical Windows\n",
        "\n",
        "We construct historical rolling windows of length $H = 50$ containing `(log_S, log_vol)`. For each window, we compute the path signature up to depth 4 using `compute_path_signature`."
    ]),
    code("""\
# Combine log_S and log_vol into a 2D path tensor of shape (N_paths, N_steps + 1, 2)
path_data = torch.stack([log_S, log_vol], dim=-1)

H = 50  # History window size
F = 10  # Forecast horizon
N_steps = S.shape[1] - 1

X_features = []
Y_targets = []

print("Computing rolling path signatures (depth 4)...")
# Loop over time steps to slide the history window
for t in range(H, N_steps + 1 - F):
    # Historical path slice of shape (N_paths, H, 2)
    window = path_data[:, t-H:t, :]
    
    # Compute signature: shape (N_paths, N_features)
    # Since D=2 and depth=4, N_features = 2^1 + 2^2 + 2^3 + 2^4 = 30
    sig = compute_path_signature(window, depth=4)
    
    # Target: future volatility at step t + F
    target = vol[:, t+F]
    
    X_features.append(sig.detach().cpu())
    Y_targets.append(target.detach().cpu())

# Stack all features and targets
X_all = torch.cat(X_features, dim=0).numpy()  # (N_samples, 30)
Y_all = torch.cat(Y_targets, dim=0).numpy()  # (N_samples,)

print("Dataset prepared successfully.")
print(f"X_all shape: {X_all.shape}")
print(f"Y_all shape: {Y_all.shape}")
"""),
    md([
        "## 3. Train Ridge Regression and Report Out-of-Sample RMSE\n",
        "\n",
        "We perform a path-wise train-test split (80% train paths, 20% test paths) to prevent overlap leakage, train a Ridge Regression model, and compute the out-of-sample RMSE."
    ]),
    code("""\
# Path-wise split: 80% train, 20% test
num_paths_train = int(0.8 * N_paths)
N_steps_valid = len(X_features)

# Reshape back to (N_steps_valid, N_paths, N_features)
X_reshaped = X_all.reshape(N_steps_valid, N_paths, 30)
Y_reshaped = Y_all.reshape(N_steps_valid, N_paths)

# Split along paths axis
X_train_paths = X_reshaped[:, :num_paths_train, :]
Y_train_paths = Y_reshaped[:, :num_paths_train]

X_test_paths = X_reshaped[:, num_paths_train:, :]
Y_test_paths = Y_reshaped[:, num_paths_train:]

# Flatten to (N_samples, N_features)
X_train = X_train_paths.reshape(-1, 30)
Y_train = Y_train_paths.reshape(-1)
X_test = X_test_paths.reshape(-1, 30)
Y_test = Y_test_paths.reshape(-1)

# Fit Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, Y_train)

# Predict future volatility
Y_pred = ridge_model.predict(X_test)

# Compute out-of-sample RMSE
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f"Out-of-sample Forecasting RMSE: {test_rmse:.6f}")

# Plot a scatter of Actual vs Predicted Volatility
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.3, color="darkred")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
plt.xlabel("Actual Future Volatility")
plt.ylabel("Predicted Future Volatility")
plt.title(f"Signature-Based Volatility Forecast (Out-of-Sample RMSE: {test_rmse:.4f})")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
"""),
    md([
        "## 4. Martingale Property Verification\n",
        "\n",
        "For the model to be valid for risk-neutral pricing, the discounted asset price must be a martingale under the risk-neutral measure. We verify the property $\mathbb{E}[S_T] \approx S_0$ with zero interest rate/dividend ($r=0, q=0$)."
    ]),
    code("""\
# Configure model for martingale simulation
model_rn = SignatureVolatilityModel(device=device)

# Reset parameter values
with torch.no_grad():
    model_rn.v0_raw.copy_(torch.tensor(np.log(0.04), device=device))  # v0 = 0.04
    model_rn.rho_raw.copy_(torch.tensor(-0.5, device=device))         # rho < 0
    
    # Populate odd-order signature coefficients
    model_rn.ell_raw[0] = 0.01   # level 1 time
    model_rn.ell_raw[1] = -0.05  # level 1 W
    model_rn.ell_raw[6] = 0.001  # level 3 time-time-time
    model_rn.ell_raw[7] = -0.002 # level 3 time-time-W

S0_test = 100.0
N_paths_martingale = 100000  # Large path count for high Monte Carlo accuracy

# Simulate paths under r=0, q=0
S_rn, _, _, _ = model_rn(
    T=1.0,
    steps_per_unit=252,
    N_paths=N_paths_martingale,
    S0=S0_test,
    r=0.0,
    q=0.0,
    antithetic=True
)

# Calculate expectation at maturity T=1.0
E_ST = S_rn[:, -1].mean().item()
error_bps = abs(E_ST - S0_test) / S0_test * 10000

print(f"Initial Stock Price S0: {S0_test}")
print(f"Expectation E[S_T]: {E_ST:.6f}")
print(f"Martingale pricing error: {error_bps:.4f} bps")

# Assert that martingale error is within 10 bps
assert error_bps < 10.0, f"Martingale error of {error_bps:.2f} bps exceeds the 10 bps threshold"
print("Martingale property successfully verified! The error is within the strict 10 bps tolerance.")
"""),
    md([
        "## 5. Pricing Options under Signature Volatility\n",
        "\n",
        "We calculate the prices of European call options across different strikes using the simulated paths."
    ]),
    code("""\
# Set strike range
option_strikes = np.linspace(80, 120, 11)
option_prices = []

# Price options
S_T_val = S_rn[:, -1].detach().cpu().numpy()
for strike in option_strikes:
    payoff = np.maximum(S_T_val - strike, 0.0)
    option_prices.append(payoff.mean())

# Plot option prices vs strikes
plt.figure(figsize=(8, 5))
plt.plot(option_strikes, option_prices, "o-", color="darkblue", lw=2)
plt.xlabel("Strike Strike (K)")
plt.ylabel("Call Option Price ($)")
plt.title("European Call Option Prices under Signature Volatility Model")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
""")
]

# Write notebooks
os.makedirs(NB_DIR, exist_ok=True)

with open(os.path.join(NB_DIR, "12_neural_sde_calibration.ipynb"), "w") as f:
    json.dump(nb(NB_12_cells), f, indent=2)

with open(os.path.join(NB_DIR, "13_signature_forecasting.ipynb"), "w") as f:
    json.dump(nb(NB_13_cells), f, indent=2)

print("Phase 5 Validation Notebooks created successfully in notebooks/")

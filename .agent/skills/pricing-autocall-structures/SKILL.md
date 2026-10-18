---
name: pricing-autocall-structures
description: >-
  Comprehensive pricing, calibration, and risk management of complex structured products:
  Worst-Of Multi-Asset Autocallables, Phoenix Autocallables, and Bermudan Cancellable Swaptions.
  Features fuzzy barrier regularization, Brownian bridge hitting probability, EGNO graph surrogates,
  Monte Carlo/PDE reference solvers, and SR 26-2 model guardians.
---

# Complex Structured Products: Autocallables & Cancellables (`autocall-engine`)

This skill provides an authoritative, production-grade guide for pricing, hedging, calibrating, and auditing complex structured products in `deepvol`—specifically **Worst-Of Multi-Asset Autocallables**, **Phoenix Notes**, and **Bermudan Cancellable Swaptions / Caps / Floors**.

---

## 1. Mathematical Payoff Functional of Worst-Of Autocallables

Consider a structured note written on a basket of $d$ underlying assets $\mathbf{S}_t = (S_t^{(1)}, \dots, S_t^{(d)})$ with initial spots $\mathbf{S}_0$:

### Worst-Of Performance Metric
At each discrete observation schedule date $t_m \in \{t_1, \dots, t_M\}$:
$$W_{t_m} = \min_{i \in \{1, \dots, d\}} \left(\frac{S_{t_m}^{(i)}}{S_0^{(i)}}\right)$$

### Early Autocall Redemption
Early redemption occurs at the stopping index $\tau$:
$$\tau = \min\left\{m \in \{1, \dots, M\} : W_{t_m} \ge B_{\text{KO}}(t_m)\right\}$$
where $B_{\text{KO}}(t_m)$ is the autocall knock-out barrier (typically $100\%$ of initial fixing). If $\tau \le M$, the note terminates early, returning $100\%$ notional plus the coupon.

### Snowballing Memory Coupon
At each observation date $t_m \le t_\tau$, if $W_{t_m} \ge B_{\text{coupon}}$:
$$\text{CouponPayment}_m = c \cdot N_{\text{notional}} \cdot (m - m_{\text{last}})$$
compensating all unpaid coupon periods since the previous coupon event $m_{\text{last}}$.

### Downside Knock-In Put Barrier (Capital at Risk)
If no early redemption occurs ($\tau = \infty$):
* **No Knock-In Breach** ($\min_{t \in \mathcal{T}_{\text{KI}}} W_t \ge B_{\text{KI}}$): Return $100\%$ of principal $N_{\text{notional}}$.
* **Knock-In Breached** ($\min_{t \in \mathcal{T}_{\text{KI}}} W_t < B_{\text{KI}}$): The investor absorbs the full downside:
  $$\text{TerminalPayoff} = N_{\text{notional}} \cdot \min\left(1, \; \frac{W_T}{K_{\text{put}}}\right)$$

---

## 2. Differentiable Smoothing & Pin Risk Regularization

Standard automatic differentiation fails across autocall and barrier barriers because the indicator function $\mathbb{I}_{\{W_{t_m} \ge B_{\text{KO}}\}}$ has zero derivative almost everywhere and an infinite Dirac delta $\delta(W - B)$ at the barrier (Pin Risk).

### A. Logistic Fuzzy Barrier
During Greek sensitivity passes, replace hard indicator steps with a smooth logistic sigmoid:
$$\sigma_\varepsilon(W_{t_m} - B_{\text{KO}}) = \frac{1}{1 + \exp\left(-\frac{W_{t_m} - B_{\text{KO}}}{\varepsilon_m}\right)}$$
with adaptive width scaling with local volatility:
$$\varepsilon_m = \kappa \cdot \sigma_{\text{ATM}} \sqrt{\Delta t}, \qquad \kappa \in [0.05, 0.20]$$

### B. Brownian Bridge Continuous Knock-In Survival
For continuous knock-in barriers monitored between discrete simulation steps $t$ and $t+\Delta t$, evaluate the exact conditional minimum probability:
$$P_{\text{breach}}(W_t, W_{t+\Delta t}) = \exp\left(-\frac{2 \ln(W_t / B_{\text{KI}}) \cdot \ln(W_{t+\Delta t} / B_{\text{KI}})}{\sigma_W^2 \Delta t}\right)$$
$$P_{\text{surv}} = 1 - P_{\text{breach}}$$

---

## 3. Production Engines in `deepvol`

The `deepvol` repository contains three verified pricing engines for structured products:

### Engine 1: GPU Monte Carlo Engine (`src/deepvol/models/wof_autocall.py`)
```python
from deepvol.models.wof_autocall import simulate_correlated_heston_paths, price_wof_autocall_mc

# Simulate correlated multi-asset Heston trajectories
paths = simulate_correlated_heston_paths(
    S0=torch.tensor([100.0, 100.0, 100.0]),
    corr_matrix=corr_matrix,  # (3, 3) PSD
    heston_params=heston_params,
    n_paths=131072,
    n_steps=252,
    T=1.0,
    device="cuda",
)

# Price Worst-Of Autocallable
pv, se, autocall_prob = price_wof_autocall_mc(
    paths=paths,
    obs_times=[0.25, 0.50, 0.75, 1.0],
    ko_barrier=1.0,
    ki_barrier=0.70,
    coupon_rate=0.08,
    memory_coupon=True,
)
```

### Engine 2: 1D Scalar PDE Solver (`src/deepvol/models/autocall_pde.py`)
Used for single-asset Autocallables and ground-truth validation. Implements a finite-difference Crank-Nicolson / implicit scheme with a tridiagonal Thomas banded solver:
```python
from deepvol.models.autocall_pde import price_autocall_pde_scalar

# Deterministic fp64 ground-truth price
price_pde = price_autocall_pde_scalar(
    S0=100.0,
    T=1.0,
    r=0.05,
    q=0.01,
    vol_fn=dupire_local_vol,
    ko_barrier=100.0,
    ki_barrier=70.0,
    coupon=0.08,
    n_S=500,
    n_T=1000,
)
```

### Engine 3: Permutation-Equivariant Graph Surrogate (`src/deepvol/surrogates/wof_autocall_egno.py`)
Provides sub-millisecond ($< 0.5\text{ ms}$) pricing and pathwise Greek sensitivities:
```python
from deepvol.surrogates.wof_autocall_egno import WoFAutocallEGNO, WoFInputNormalizer

model = WoFAutocallEGNO(n_assets=3, hidden_dim=64, n_layers=3).cuda()
# Evaluates Worst-Of price and exact dPV/dS vector satisfying S_3 symmetry
pv_pred, deltas = model.forward_with_greeks(asset_features, corr_matrix)
```

---

## 4. Model Risk Management & Guardian Fallback (`src/deepvol/mrm/autocall_guardian.py`)

Under SR 26-2 governance, every autocall valuation passes through `AutocallModelGuardian`:
1. **Ledoit-Wolf Shrinkage**: Guarantees that the empirical correlation matrix $\boldsymbol{\Sigma}_{\text{LW}} = (1-\alpha^*)\hat{\boldsymbol{\Sigma}} + \alpha^* \mathbf{I}$ remains strictly positive definite.
2. **Mahalanobis Out-of-Distribution Detector**:
   $$D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_{\text{train}})^\top \boldsymbol{\Sigma}_{\text{train}}^{-1} (\mathbf{x} - \boldsymbol{\mu}_{\text{train}})$$
   If $D_M > \tau_{\text{OOD}}$ or spot is within $\pm 0.5\%$ of barrier, switch instantly to GPU Monte Carlo.
3. **Delta/Gamma Consistency**: Asserts that $\sum_i \Delta_i \le 1.0$ and pin-risk Gamma does not exceed regulatory risk budget.

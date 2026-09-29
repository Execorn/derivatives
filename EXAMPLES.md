# Mathematical Specifications and Usage Examples for the DeepVol Model Zoo

This document provides the mathematical formulations, discretization schemes, and functional Python API examples for the sixteen volatility and asset pricing models implemented in the `DeepVol` framework.

---

## 1. Classic Heston Model

### Mathematical Formulation
The Heston model (1993) represents the asset price $S_t$ and its variance $v_t$ as a system of stochastic differential equations (SDEs):

$$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^1$$

$$dv_t = \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^2$$

$$d\langle W^1, W^2\rangle_t = \rho dt$$

where:
* $\kappa > 0$ is the mean-reversion speed.
* $\theta > 0$ is the long-term variance.
* $\sigma > 0$ is the volatility of volatility.
* $\rho \in [-1, 1]$ is the correlation between the asset and variance shocks (the leverage effect).
* $v_0 > 0$ is the initial variance.

### Option Pricing via Fourier-COS Method
Option prices are computed by integrating the model's characteristic function $\phi(u, T)$ using the Fourier-COS expansion method (Fang & Oosterlee 2008). To prevent branch-cut discontinuities during complex logarithm evaluation, the stable characteristic function representation (Gatheral 2006) is used:

$$\phi(u, T) = \exp\left( D(u, T) + \frac{\kappa \theta}{\sigma^2} G(u, T) \right)$$

where:

$$D(u, T) = \frac{\kappa - d(u) - (\kappa + d(u))e^{-d(u)T}}{\sigma^2 (1 - g(u)e^{-d(u)T})} v_0$$

$$G(u, T) = (\kappa - d(u))T - 2 \log\left( \frac{1 - g(u)e^{-d(u)T}}{1 - g(u)} \right)$$

$$d(u) = \sqrt{(\kappa - i \rho \sigma u)^2 + \sigma^2 (u^2 + i u)}$$

$$g(u) = \frac{\kappa - i \rho \sigma u - d(u)}{\kappa - i \rho \sigma u + d(u)}$$

### Python Example
```python
import numpy as np
from deepvol.models.heston import HestonEngine

# Instantiate the engine
engine = HestonEngine()

# Model parameters
params = {
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7,
    "v0": 0.04
}

# Grid definitions
T_grid = np.array([0.5, 1.0])
K_grid = np.array([-0.1, 0.0, 0.1]) # Log-moneyness

# Compute implied volatility surface
iv_surface = engine.price_surface(params, T_grid, K_grid, S0=100.0)
print("IV Surface:\n", iv_surface)
```

---

## 2. Rough Heston Model

### Mathematical Formulation
The Rough Heston model (El Euch & Rosenbaum 2019) replaces the standard mean-reversion drift with a fractional integral of Hurst parameter $H \in (0, \tfrac{1}{2})$:

$$v_t = v_0 + \frac{1}{\Gamma(H + \tfrac{1}{2})} \int_0^t (t-s)^{H-\tfrac{1}{2}} \kappa(\theta - v_s) ds + \frac{\sigma}{\Gamma(H + \tfrac{1}{2})} \int_0^t (t-s)^{H-\tfrac{1}{2}}\sqrt{v_s} dW_s$$

### Markovian Approximation (Lifted Heston)
To make path simulations computationally tractable, the fractional kernel is approximated by a sum of $N$ Markovian factors (Abi Jaber 2019) using Bernstein weights $c_i^N$ and mean-reversion rates $x_i^N$:

$$v_t^N = \sum_{i=1}^N c_i^N Z_t^{(i, N)}$$

$$dZ_t^{(i, N)} = -\left(x_i^N Z_t^{(i, N)} + \kappa(v_t^N - \theta)\right) dt + \sigma \sqrt{v_t^N} dW_t$$

### Python Example (Using FNO Surrogate)
```python
import torch
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_fast import _make_spatial_input

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device))
model.eval()

# Normalized parameters: [kappa, theta, sigma, rho, v0, H]
theta = torch.tensor([[2.0, 0.04, 0.5, -0.7, 0.04, 0.08]], dtype=torch.float32, device=device)
spatial = _make_spatial_input(T_grid=np.linspace(0.1, 2.0, 8), K_grid=np.linspace(-0.5, 0.5, 11), device=device)

with torch.no_grad():
    normalized_output = model(spatial, theta)
    print("Output Shape:", normalized_output.shape)
```

---

## 3. Rough Bergomi Model

### Mathematical Formulation
The Rough Bergomi model (Bayer, Friz & Gatheral 2016) is a lognormal rough volatility model where the variance process is defined as:

$$v_t = v_0 \exp\left( W_t^H - \frac{1}{2} t^{2H} \right)$$

where $W_t^H$ is a fractional Brownian motion represented via Riemann-Liouville integration:

$$W_t^H = \eta \sqrt{2H} \int_0^t (t-s)^{H-\tfrac{1}{2}} dW_s$$

### Bennedsen-Lunde-Pakkanen Hybrid Scheme
Paths are simulated on a discrete time grid $t_i = i \Delta t$ by partitioning the stochastic integral into a local singular component and a non-singular convolution resolved via 1D FFT:

$$\int_0^{t_i} (t_i - s)^{H - \tfrac{1}{2}} dW_s \approx \sum_{j=1}^{i-1} b_j^* \Delta W_{i-j} + \int_{t_{i-1}}^{t_i} (t_i - s)^{H - \tfrac{1}{2}} dW_s$$

### Python Example
```python
import numpy as np
from deepvol.models.rbergomi_gpu import rBergomiEngine

engine = rBergomiEngine()
T_grid = np.array([0.5, 1.0])
K_grid = np.array([-0.1, 0.0, 0.1])

# Run path simulations and compute implied volatilities on GPU/CPU
ivs = engine.price_surface(
    v0=0.04, H=0.1, eta=1.5, rho=-0.7,
    T_grid=T_grid, K_grid=K_grid, N_paths=10000
)
print("Rough Bergomi IV Surface:\n", ivs)
```

---

## 4. McKean-Vlasov SDE (MLSV)

### Mathematical Formulation
The Local Stochastic Volatility (LSV) model formulated as a McKean-Vlasov SDE adjusts the stochastic volatility process with a leverage function $\lambda(S_t, t)$ to match market option prices exactly:

$$dS_t = (r - q) S_t dt + \lambda(S_t, t) \sqrt{V_t} S_t dW_t^1$$

$$dV_t = \kappa(\theta - V_t) dt + \xi \sqrt{V_t} dW_t^2$$

According to Dupire's equation, the Leverage function $\lambda(K, t)$ is defined by the conditional expectation:

$$\lambda^2(K, t) = \frac{\sigma_{\text{Dup}}^2(K, t)}{\mathbb{E}[V_t \mid S_t = K]}$$

### Particle Calibration Scheme
The conditional expectation is evaluated using a particle system of size $N$ with Nadaraya-Watson kernel density estimation:

$$\mathbb{E}[V_t \mid S_t = K] \approx \frac{\sum_{i=1}^N V_t^i K_h(S_t^i - K)}{\sum_{i=1}^N K_h(S_t^i - K)}$$

where $K_h(x) = \frac{1}{h} \exp(-\frac{x^2}{2h^2})$ is a Gaussian kernel with bandwidth $h$.

### Python Example
```python
import torch
from deepvol.models.mlsv_gpu import MLSVSolverGPU

# Define Dupire local volatility function
def dupire_vol(t, s):
    return torch.full_like(s, 0.2)

solver = MLSVSolverGPU(
    S0=100.0, r=0.0, q=0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    T=1.0, steps_per_unit=50, N_paths=2000, dupire_vol_fn=dupire_vol
)

# Simulate McKean-Vlasov particle system on GPU
solver.simulate(method="nadaraya_watson")
option_prices = solver.price_european_option(strike=torch.tensor([90.0, 100.0, 110.0]), maturity=np.array([0.5, 1.0]))
print("Simulated LSV Option Prices:\n", option_prices)
```

---

## 5. SABR Model (Hagan/Displaced)

### Mathematical Formulation
The SABR model (Hagan et al. 2002) is a two-factor stochastic volatility model:

$$dF_t = \alpha_t F_t^\beta dW_t^1$$

$$d\alpha_t = \nu \alpha_t dW_t^2$$

$$d\langle W^1, W^2\rangle_t = \rho dt$$

where $F_t$ is the forward rate, $\alpha_t$ is the volatility parameter, $\beta \in [0, 1]$ is the elasticity parameter, and $\nu$ is the volatility of volatility. 

The Displaced SABR extension replaces $F_t$ with $F_t + s$, where $s$ is a constant displacement shift parameter, allowing for negative interest rates:

$$d(F_t + s) = \alpha_t (F_t + s)^\beta dW_t^1$$

### Python Example
```python
import numpy as np
from deepvol.models.sabr import sabr_iv_surface

# Generate IV surface under SABR model
sabr_surface = sabr_iv_surface(
    F=100.0,
    T_grid=np.array([0.5, 1.0]),
    k_grid=np.array([-0.1, 0.0, 0.1]),
    alpha=0.2, beta=0.5, rho=-0.5, nu=0.4,
    iv_type="lognormal"
)
print("SABR IV Surface:\n", sabr_surface)
```

---

## 6. SSVI Model

### Mathematical Formulation
The Surface SVI (SSVI) model (Gatheral & Jacquier 2011) parameterizes the implied volatility surface using total variance slices $w(k, \theta_t)$ linked to the At-The-Money (ATM) variance $\theta_t$:

$$w(k, \theta_t) = \frac{\theta_t}{2} \left[ 1 + \rho \varphi(\theta_t) k + \sqrt{(\varphi(\theta_t) k + \rho)^2 + (1-\rho^2)} \right]$$

where $\varphi(\theta_t)$ is a power-law function of the ATM variance:

$$\varphi(\theta) = \frac{\eta}{\theta^\gamma (1+\theta)^{1-\gamma}}$$

No-arbitrage conditions require $\theta_t$ to be strictly increasing, $\rho \in (-1, 1)$, and:

$$\theta \varphi(\theta) (1 + |\rho|) \leq 4$$

### Python Example
```python
import numpy as np
from deepvol.models.sabr import ssvi_iv_surface

T_grid = np.array([0.25, 0.5, 1.0])
k_grid = np.array([-0.2, 0.0, 0.2])
theta_grid = 0.04 * T_grid # ATM variance linear in maturity

ssvi_surface = ssvi_iv_surface(T_grid, k_grid, theta_grid, rho=-0.4, eta=1.2, gamma=0.5)
print("SSVI IV Surface:\n", ssvi_surface)
```

---

## 7. Local Volatility Model

### Mathematical Formulation
Dupire's local volatility (1994) represents volatility as a deterministic function of time $t$ and asset price level $K$:

$$\sigma_{\text{loc}}^2(t, K) = \frac{\frac{\partial C}{\partial t} + (r-q) K \frac{\partial C}{\partial K} + q C}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}$$

Using an implied volatility surface $w(t, k) = \sigma_{\text{IV}}^2(t, k) \cdot t$ where $k = \log(K/S_0)$, the local variance is calculated as:

$$\sigma_{\text{loc}}^2(t, k) = \frac{\frac{\partial w}{\partial t}}{\left( 1 - \frac{k}{w}\frac{\partial w}{\partial k} + \frac{1}{4}\left(-\frac{1}{8} - \frac{1}{w} + \frac{k^2}{w^2}\right)\left(\frac{\partial w}{\partial k}\right)^2 + \frac{1}{2}\frac{\partial^2 w}{\partial k^2} \right)}$$

### Python Example
```python
import numpy as np
from deepvol.models.local_vol import DupireLocalVolSolver

# Generate dummy input IV surface
T_grid = np.array([0.5, 1.0])
K_grid = np.array([-0.2, 0.0, 0.2])
iv_surface = np.full((len(T_grid), len(K_grid)), 0.20)

solver = DupireLocalVolSolver(T_grid, K_grid, S0=100.0, r=0.05, q=0.0)
lv_surface = solver.solve(iv_surface)
print("Local Volatility Surface:\n", lv_surface)
```

---

## 8. Neural SDE

### Mathematical Formulation
A Neural Stochastic Differential Equation (Neural SDE) parameterizes the drift $\mu_\theta$ and diffusion $\sigma_\theta$ coefficients using neural networks:

$$dX_t = \mu_\theta(X_t, t) dt + \sigma_\theta(X_t, t) dW_t$$

### Adjoint Calibration
Training is completed by defining a loss function $L(X_T)$ based on option pricing errors, and computing gradients with respect to weights $\theta$ using the adjoint sensitivity method (Pontryagin's maximum principle) implemented via `torchsde`:

$$\frac{dL}{d\theta} = -\int_0^T a_t \frac{\partial \mu_\theta}{\partial \theta}(X_t, t) dt - \int_0^T b_t \frac{\partial \sigma_\theta}{\partial \theta}(X_t, t) dW_t$$

### Python Example
```python
import torch
import torch.nn as nn

class DiffusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Softplus(), nn.Linear(32, 1), nn.Softplus())
        
    def forward(self, t, x):
        # Returns state-dependent diffusion coefficient
        inputs = torch.cat([t.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)
        return self.net(inputs).squeeze(-1)

net = DiffusionMLP()
t = torch.tensor(0.5)
x = torch.tensor(100.0)
print("Neural Diffusion at (t=0.5, S=100):", net(t, x).item())
```

---

## 9. Signature Volatility Model

### Mathematical Formulation
The Signature Volatility model expresses the asset's variance $V_t$ as a linear functional of the path signature $\mathbb{S}(Y)_{0,t}$ of a driving window process $Y_t$:

$$V_t = \langle \ell, \mathbb{S}(Y)_{0, t} \rangle$$

where $\mathbb{S}(Y)_{0, t}$ consists of iterated integrals of $Y$ along time:

$$\mathbb{S}(Y)_{0,t}^{i_1, \dots, i_k} = \int_{0 < u_1 < \dots < u_k < t} dY_{u_1}^{i_1} \dots dY_{u_k}^{i_k}$$

### Python Example
```python
import torch
from deepvol.models.signature_vol import compute_signature_paths

# Generate paths of shape (batch, time, features)
paths = torch.randn(10, 100, 2)
signatures = compute_signature_paths(paths, depth=3)
print("Signature Features Shape:", signatures.shape)
```

---

## 10. Schwartz-Smith Model

### Mathematical Formulation
The Schwartz-Smith (2000) model represents the log commodity spot price $\ln S_t$ as the sum of a short-term mean-reverting deviation $\chi_t$ and a long-term equilibrium price path $\xi_t$:

$$\ln S_t = \chi_t + \xi_t$$

$$d\chi_t = -\kappa \chi_t dt + \sigma_\chi dW_t^1$$

$$d\xi_t = \mu_\xi dt + \sigma_\xi dW_t^2$$

$$d\langle W^1, W^2\rangle_t = \rho_{\chi\xi} dt$$

### State-Space Representation & Kalman Filter
The state-space model transition is defined as:

$$x_t = F x_{t-1} + C + \epsilon_t, \qquad y_t = H x_t + D + v_t$$

where $y_t$ is a vector of log futures prices across different contract maturities, and $x_t = [\chi_t, \xi_t]^T$. The parameters are estimated by maximizing the log-likelihood function using the Kalman Filter.

### Python Example
```python
import numpy as np
from deepvol.models.schwartz_smith import SchwartzSmithEngine

engine = SchwartzSmithEngine(
    kappa=1.2, mu_y=0.05, sigma_x=0.3, sigma_y=0.15, rho_xy=-0.3
)

# Price commodity futures options using analytical Black-76 mapping
option_price = engine.price_option(
    spot=100.0, strike=105.0, maturity=0.5, risk_free_rate=0.05, is_call=True
)
print("Schwartz-Smith Futures Option Price:", option_price)
```

---

## 11. Physics-Informed Meta-Learning FNO (PI-M-FNO)

### Mathematical Formulation
The PI-M-FNO surrogate integrates the Dupire local volatility partial differential equation directly into operator training:

$$\mathcal{L}_{\text{PDE}}(u) = \left\| \frac{\partial C}{\partial T} - \frac{1}{2} \sigma_{\text{loc}}^2(K, T) K^2 \frac{\partial^2 C}{\partial K^2} + (r-q) K \frac{\partial C}{\partial K} + q C \right\|_{L^2}^2$$

During extreme market stress or regime shifts, the base operator weights $\theta$ are adapted via an inner-loop gradient step on the PDE residual without requiring new labeled market data:

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{PDE}}(u_\theta)$$

### Python Example
```python
import torch
from deepvol.surrogates.meta_fno import MetaLearnerFNO
from deepvol.surrogates.pde_loss import compute_pde_loss_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
meta_fno = MetaLearnerFNO(param_dim=6, inner_lr=1e-3).to(device)

# Parameter vector and spatial evaluation grid
theta = torch.tensor([[2.5, 0.05, 0.6, -0.75, 0.05, 0.09]], device=device)
spatial_grid = torch.randn(1, 8, 11, 2, device=device)

# Perform online inner adaptation step (< 10 ms)
adapted_surface = meta_fno.adapt_online(theta, spatial_grid, num_inner_steps=2)
print("Adapted Volatility Surface Shape:", adapted_surface.shape)
```

---

## 12. End-to-End Differentiable Calibration & Hedging (D-XVA)

### Mathematical Formulation
D-XVA unifies model calibration, implied volatility inversion via PIVOT, and recurrent deep hedging into a single differentiable computational graph:

$$\min_{\theta, \psi} \mathcal{L}(\theta, \psi) = \text{Var}\left( \Pi_T(\psi) \right) + \lambda_{\text{cost}} \mathbb{E}\left[ \sum_{t=0}^{T-1} c |\Delta_{t+1} - \Delta_t| S_t \right] + \lambda_{\text{cal}} \|\sigma_{\text{IV}}(\theta) - \sigma_{\text{market}}\|_2^2$$

where $\psi$ parametrizes an LSTM/GRU hedging policy network and $\theta$ parametrizes the rough volatility dynamics. Gradients backpropagate directly from terminal portfolio variance into calibration parameters.

### Python Example
```python
import torch
from deepvol.hedging.deep_hedging import DeepHedgingEngine
from deepvol.hedging.d_xva import compute_dxva_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
hedger = DeepHedgingEngine(input_dim=5, hidden_dim=64, num_layers=2).to(device)

# Batch of simulated asset and implied vol trajectories (batch, time, features)
trajectories = torch.randn(128, 50, 5, device=device)
strike = 100.0

# Compute hedge ratios and differentiable trading loss
deltas = hedger(trajectories)
loss = compute_dxva_loss(deltas, trajectories[:, :, 0], strike=strike, cost_bps=5.0)
print("D-XVA Optimization Loss:", loss.item())
```

---

## 13. Grey Rough Bergomi (gRB) Model

### Mathematical Formulation
The Grey Rough Bergomi model generalizes rough Bergomi by driving the variance process with generalized grey Brownian motion (ggBm) $B_t^{\beta, \alpha}$:

$$v_t = v_0 \mathcal{E}\left( \eta \sqrt{2H} Y_t \right), \qquad Y_t = \int_0^t (t-s)^{H-\tfrac{1}{2}} dB_s^{\beta, \alpha}$$

The characteristic function of ggBm is governed by the two-parameter Mittag-Leffler function $E_{\alpha, \beta}(z) = \sum_{k=0}^\infty \frac{z^k}{\Gamma(\alpha k + \beta)}$. Path generation is performed using Wood-Chan circulant embedding on GPU.

### Python Example
```python
import torch
from deepvol.models.rbergomi_gpu import rBergomiEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = rBergomiEngine()

# Simulate paths on GPU using Wood-Chan circulant embedding
paths = engine.simulate_paths(
    v0=0.04, H=0.10, eta=1.4, rho=-0.75,
    N_paths=10000, N_steps=100, T=1.0, device=device
)
print("gRB Simulated Asset Paths Shape:", paths.shape)
```

---

## 14. Autocallable Structured Notes (Multi-Fidelity PDE + Ensemble MLP)

### Mathematical Formulation
An autocallable structured note defines early redemption stopping times $\tau = \min \{ t_m \in \{T_1, \dots, T_M\} \mid S_{t_m} \geq B_{\text{autocall}} \} \wedge T_M$. At redemption, the contract pays:

$$V_\tau = \begin{cases} 1 + m \cdot c & \text{if early redemption occurs at } t_m \\ 1 + M \cdot c & \text{if } \tau = T_M \text{ and } S_{T_M} \geq B_{\text{autocall}} \\ 1 & \text{if } \tau = T_M, \, S_{T_M} < B_{\text{autocall}}, \text{ and } \min_{t \leq T} S_t > B_{\text{dip}} \\ \frac{S_{T_M}}{S_0} & \text{if capital protection barrier } B_{\text{dip}} \text{ was breached} \end{cases}$$

The multi-fidelity valuation decomposes pricing into an analytical Gatheral boundary PDE base $V_{\text{PDE}}(x)$ and a 5-member residual deep ensemble $\{f_{\psi_k}\}_{k=1}^5$:

$$\hat{V}(x) = V_{\text{PDE}}(x) + \frac{1}{K} \sum_{k=1}^K f_{\psi_k}(x), \qquad \sigma_{\text{bps}}(x) = \sqrt{\frac{1}{K-1} \sum_{k=1}^K \left( f_{\psi_k}(x) - \bar{f}(x) \right)^2} \times 10^4$$

### Python Example
```python
import torch
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.mrm.autocall_guardian import AutocallGuardian

device = "cuda" if torch.cuda.is_available() else "cpu"
ensemble = CorrectionEnsemble.load_default(device=device)
guardian = AutocallGuardian(ensemble)

contract = {
    "spot": 100.0, "strike": 100.0, "autocall_barrier": 105.0, "dip_barrier": 70.0,
    "coupon_rate": 0.08, "volatility": 0.22, "rate": 0.03, "dividend": 0.01,
    "maturity": 2.0, "obs_freq": 0.5
}

result = guardian.price_contract(contract)
print(f"Autocall NPV: {result['price']:.4f} ± {result['uncertainty_bps']:.2f} bps")
print(f"Guardian Routing: {result['routing']}, Fallback Active: {result['fallback_active']}")
```

---

## 15. Phoenix Memory Coupon Autocallables

### Mathematical Formulation
Phoenix notes pay conditional memory coupons at observation dates $t_m$. If the underlying asset price exceeds the coupon barrier $B_{\text{coupon}}$, the investor receives the current coupon plus all accrued unpaid coupons:

$$\text{Coupon Payout at } t_m = \begin{cases} (m - \text{last\_paid}) \times c & \text{if } S_{t_m} \geq B_{\text{coupon}} \\ 0 & \text{if } S_{t_m} < B_{\text{coupon}} \end{cases}$$

Early redemption occurs if $S_{t_m} \geq B_{\text{autocall}}$. If the note survives until maturity $T$, capital protection applies provided $S_T \geq B_{\text{dip}}$.

### Python Example
```python
import torch
from deepvol.models.phoenix import price_phoenix_mc

device = "cuda" if torch.cuda.is_available() else "cpu"

# Vector of contract specs: [S0, K, B_autocall, B_coupon, B_dip, coupon, vol, r, q, T, n_obs]
specs = torch.tensor([[100.0, 100.0, 105.0, 80.0, 70.0, 0.07, 0.20, 0.02, 0.01, 1.5, 6]], device=device)
price = price_phoenix_mc(specs, N_paths=50000, device=device)
print(f"Phoenix Note Price: {price.item():.4f}")
```

---

## 16. Worst-Of (WoF) Basket Autocallables (EGNO)

### Mathematical Formulation
Worst-of basket autocallables depend on the joint evolution of $d$ correlated assets $S_t = (S_t^{(1)}, \dots, S_t^{(d)})$. The contract payoff is dictated by the worst-performing asset relative to its initial value:

$$I_t = \min_{j=1, \dots, d} \frac{S_t^{(j)}}{S_0^{(j)}}$$

The asset index permutation group $\mathcal{S}_d$ defines an exact symmetry: for any permutation $\pi \in \mathcal{S}_d$, the contract price satisfies $f(\pi \circ X) = f(X)$. The Equivariant Graph Neural Operator (EGNO) architecture preserves this permutation symmetry through invariant graph pooling and projects correlation matrices onto the positive semi-definite cone $\mathcal{S}_+^d$.

### Python Example
```python
import torch
from deepvol.models.wof_autocall import price_wof_autocall_mc
from deepvol.surrogates.wof_autocall_egno import WorstOfAutocallEGNO

device = "cuda" if torch.cuda.is_available() else "cpu"
egno_model = WorstOfAutocallEGNO(n_assets=3, hidden_dim=64).to(device)

# Correlated basket specs: 3 asset spots, volatilities, and correlation matrix
spots = torch.tensor([100.0, 100.0, 100.0], device=device)
vols = torch.tensor([0.25, 0.22, 0.28], device=device)
corr = torch.tensor([[1.0, 0.6, 0.5], [0.6, 1.0, 0.55], [0.5, 0.55, 1.0]], device=device)

# Evaluate via EGNO surrogate
features = torch.cat([spots, vols, corr.flatten()]).unsqueeze(0)
basket_price = egno_model(features)
print(f"WoF Basket Autocall Price: {basket_price.item():.4f}")
```

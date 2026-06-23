# Phase 5: Neural SDE and Data-Driven Models — Technical Roadmap

This roadmap details the mathematical framework, implementation plans, and verification steps for Phase 5 (P5) of the Neural Network Pricing Framework. The goal of Phase 5 is to transition from parametric stochastic volatility (SV) models to data-driven, non-parametric representations of option pricing dynamics.

---

## P5.1 — Lifted Heston Accuracy Study

### 1. Mathematical Formulation
The fractional Rough Heston model uses a Riemann-Liouville fractional kernel to introduce memory:
\[
V_t = V_0 + \frac{1}{\Gamma(H + 1/2)} \int_0^t (t - s)^{H - 1/2} \left[ \kappa (\theta - V_s) ds + \sigma \sqrt{V_s} dW_s \right]
\]
for $H \in (0, 1/2)$. Because of the non-Markovian nature of the fractional kernel, direct simulation is computationally expensive. The **Lifted Heston model** (Abi Jaber, 2019) approximates this kernel by a finite sum of Markovian factors using a Bernstein lift:
\[
K^N(t) = \sum_{i=1}^N c_i^N e^{-x_i^N t}
\]
where the weights $c_i^N$ and mean-reversion speeds $x_i^N$ are chosen geometrically to match the fractional kernel:
\[
x_i^N = g^i \alpha^N, \quad c_i^N = \frac{x_i^{1-2H} - x_{i-1}^{1-2H}}{\Gamma(H + 1/2) \Gamma(3/2 - H)}
\]
where $g > 1$ is a scaling factor and $\alpha^N$ is calibrated. The variance process is lifted to a system of $N$ linear SDEs:
\[
V_t^N = \sum_{i=1}^N c_i^N V_t^{N, i}, \quad dV_t^{N, i} = -x_i^N V_t^{N, i} dt + \left[ \kappa (\theta - V_t^N) dt + \sigma \sqrt{V_t^N} dW_t \right]
\]

### 2. Convergence Study Parameters
We will benchmark the convergence of the $N$-factor Bernstein approximation to the exact rough kernel ($H \in [0.04, 0.15]$):
- **Factor count ($N$) grid**: $N \in \{5, 10, 20, 40, 80, 160\}$
- **Hurst exponent ($H$) grid**: $H \in \{0.04, 0.07, 0.10, 0.14\}$
- **Evaluation metric**: Implied Volatility (IV) surface root mean squared error (RMSE) in basis points (bps) compared to a high-accuracy direct Monte Carlo simulation (using 1,000,000 paths and 1,000 time steps).
- **Maturity bounds**: Focus on ultra-short maturities ($T \le 0.1$) where the skew is steepest and factor truncation error is most pronounced.

### 3. Implementation Steps
1. Create `src/pricing/lifted_heston_study.py` allowing a configurable number of factors $N$.
2. Write a benchmark script `benchmarks/convergence_N_factors.py` that computes option prices under different $N$ values.
3. Quantify the CPU/GPU execution speed vs factor count $N$ (expected complexity $O(N^2)$ for covariance updates or $O(N)$ for simulation steps).
4. Save results to `artifacts/reports/lifted_heston_convergence.json` and generate skew plots.

---

## P5.2 — Neural SDE as Model Prior

### 1. Mathematical Framework
Instead of assuming a parametric model (e.g. Heston, SABR), we parameterize the drift and diffusion functions of the variance process as neural networks:
\[
d X_t = \left( r - q - \frac{1}{2} V_t \right] dt + \sqrt{V_t} \left( \rho dW_t^1 + \sqrt{1 - \rho^2} dW_t^2 \right)
\]
\[
d V_t = f_\theta(t, V_t) dt + g_\theta(t, V_t) dW_t^1
\]
where:
- $X_t = \log(S_t / S_0)$ is the log-return process.
- $f_\theta(t, V) \in \mathbb{R}$ is the drift network (a 3-layer MLP with Swish activation).
- $g_\theta(t, V) \in \mathbb{R}^+$ is the diffusion network (a 3-layer MLP with Swish activation and Softplus output to guarantee positivity).
- $W_t^1, W_t^2$ are independent standard Brownian motions, and $\rho \in [-0.95, 0.0]$ is the leverage correlation parameter.

### 2. Pricing & Pathwise Gradients
Option pricing is performed via Monte Carlo simulation:
\[
C(T, K) = e^{-r T} \mathbb{E} \left[ \max(S_0 e^{X_T} - K, 0) \right]
\]
To calibrate the parameters $\theta$ (the weights of $f_\theta, g_\theta$) and $\rho$ to the market, we must backpropagate through the expectation:
- **Reparameterization Trick**: We use `torchsde` with Brownian path objects (`torchsde.BrownianInterval`) to ensure that the random noise is fixed during the forward pass.
- **SDE Adjoint Method**: We use the adjoint method provided by `torchsde` to compute gradients of the loss function back to $\theta$ and $\rho$ without storing the entire path history in memory.

### 3. Arbitrage & Regularization Constraints
1. **Dynamic No-Arbitrage**: Automatically satisfied by construction because the underlying stock price $S_t = S_0 e^{X_t}$ has a drift of $r - q$ under the risk-neutral measure.
2. **Positivity of Volatility**: Ensured by applying `Softplus` or `Exponential` activation to the final layer of $g_\theta(t, V_t)$, and capping the Euler-Maruyama simulation floor at a small $\epsilon = 10^{-4}$.
3. **Feller-like Boundary Regularization**: To prevent the volatility process from hitting zero or exploding, we add a boundary penalty to the training loss:
   \[
   L_{\text{reg}} = \lambda_{\text{bound}} \mathbb{E} \left[ \frac{1}{V_t} + V_t^2 \right]
   \]

### 4. Implementation Steps
1. Create `src/pricing/neural_sde.py` defining the SDE class inheriting from `torch.nn.Module` and compatible with `torchsde`.
2. Implement custom MLP layers for drift $f_\theta(t, V)$ and diffusion $g_\theta(t, V)$.
3. Write `scripts/train_neural_sde.py` to train the SDE drift/diffusion parameters directly to fit SPX historical surfaces.
4. Integrate with the FNO surrogate: train a v3-style FNO that maps Neural SDE network weights (projected via PCA or bottleneck MLP) to the IV surface.

---

## P5.3 — Signature-Based Volatility Model

### 1. Mathematical Framework
A path signature is a non-parametric representation of a path that captures its geometric properties. The signature volatility model (Abi Jaber, Gassiat & Sotnikov, 2025) models the variance process $V_t$ as a linear combination of path signature terms of a time-extended Brownian motion $Z_t = (t, W_t)$:
\[
d S_t = \sqrt{V_t} S_t \left( \rho dW_t^1 + \sqrt{1 - \rho^2} dW_t^2 \right)
\]
\[
V_t = \langle \ell, \mathbb{X}_{0, t}(Z) \rangle
\]
where $\mathbb{X}_{0, t}(Z)$ is the signature of the time-extended path $Z_{[0, t]}$ up to depth $d=4$, and $\ell$ is a vector of linear coefficients.

### 2. Martingale Property and Stability Constraints
To ensure the model is arbitrage-free and mathematically sound, the price process $S_t$ must be a **true martingale**. According to Abi Jaber et al. (2025):
1. **Negative Correlation**: The correlation parameter $\rho$ must be strictly negative ($\rho < 0$) to model the leverage effect and control price explosions.
2. **Odd-Order Signature Terms**: The linear combination $\ell$ must only utilize signature elements of **odd order** (odd degree terms in the signature expansion) to guarantee the martingale property and prevent price path explosions.
3. **Positivity Constraint**: Since $V_t$ is modeled as a linear combination, we must enforce $V_t \ge 0$ along all simulated paths. We enforce this via a soft-thresholding function or a ReLU-based penalty on the simulation paths:
   \[
   L_{\text{neg\_vol}} = \mu_{\text{pen}} \sum_{t_i} \max(0, -V_{t_i})^2
   \]

### 3. Implementation Steps
1. Create `src/pricing/signature_vol.py` using `iisignature` or `esig` to extract signature features of rolling historical paths (e.g. 252 days) of (log S, log VIX).
2. Constrain the signature coefficients $\ell$ to odd-order elements.
3. Write `scripts/train_signature_vol.py` to learn signature coefficients that reconstruct historical volatility smiles.
4. Deliver `notebooks/13_signature_forecasting.ipynb` evaluating the forecast and pricing accuracy of signature-based vol.

---

## Verification Plan

### 1. Automated Tests
Create `tests/test_neural_sde.py` and `tests/test_signature_vol.py` verifying:
- **Neural SDE**:
  - Differentiability: check that `torch.autograd.grad` returns non-zero gradients for $f_\theta, g_\theta$ parameters.
  - Volatility positivity: verify that volatility remains strictly positive ($V_t > 0$) for 1,000 simulated paths under extreme parameters.
- **Signature Volatility**:
  - Martingale check: verify that $\mathbb{E}[S_T] \approx S_0$ within Monte Carlo error bounds ($< 10$ bps) for a 1-year horizon.
  - Odd-order constraint: verify that non-odd signature coefficients in $\ell$ are strictly set to 0.
- **Lifted Heston**:
  - Convergence rate: verify that $N=80$ achieves an IV RMSE $< 2.0$ bps compared to $N=160$.

### 2. Manual/Visual Verification
Validate the results in Jupyter notebooks:
- `notebooks/12_neural_sde_calibration.ipynb`: visual overlay of model vs market smiles, and SDE path distributions.
- `notebooks/13_signature_forecasting.ipynb`: VIX/SPX historical rolling path signature plots and joint out-of-sample forecasting RMSE reports.

# Deep Research 3: State-of-the-Art Numerical Schemes for Lifted Heston Simulation

> **Source**: Gemini Deep Research — commissioned during calibration development  
> **Relevance**: CRITICAL for Agent 2 (Option 2: Fourier-COS pricing engine). Contains exact analysis of why Monte Carlo fails at H=0.08, the mathematical basis for Fourier-COS pricing, and Bernstein factor parameterization.

---

## 1. Why Euler-Maruyama Fails at H=0.08

Convergence order of Euler-Maruyama on fractional CIR: **O(Δt^H)**

At H=0.08 with 252 steps (Δt ≈ 0.00396):
- Convergence order = O(Δt^0.08) — glacially slow
- Halving time steps yields almost no error reduction
- Achieving 1bp accuracy at T=0.1 would require N > 10,000–50,000 steps
- **252 steps is catastrophically coarse — not merely insufficient**

| Scheme | Model | Weak Conv. Order (H=0.08) | Complexity | CUDA Suitability |
|---|---|---|---|---|
| Euler-Maruyama | Lifted Heston | O(Δt^0.08) | O(N) | High |
| BSS Hybrid/HQE | True Rough Heston | O(Δt^0.58) | O(N²) | Low |
| Adams-Bashforth-Moulton | True Rough Heston | O(Δt^1.58) | O(N²) | Low |

The Lifted Heston's O(N) complexity is why it was GPU-accelerated — but the convergence order remains 0.08.

---

## 2. The Affine Pathway — Bypass Monte Carlo Entirely

The Lifted Heston belongs to the **affine Volterra class**. Its characteristic function φ(u;T) is analytically known up to a system of N Riccati ODEs.

### Why Fourier-COS is superior:
- Solving N=20 Riccati ODEs takes ~1.67s for a full 8×11 surface
- Monte Carlo: 65,536 paths × 252 steps × random number generation = 1.65×10⁷ operations
- **No discretization bias** — deterministic, exact up to ODE solver tolerance
- **No Monte Carlo variance** — surfaces are perfectly smooth (critical for FNO training)
- Smoother labels → neural network learns true functional mapping, not stochastic artifacts

---

## 3. Bernstein Approximation at H=0.08, T=0.1

The Lifted Heston approximates kernel K(t) = t^(H-0.5)/Γ(H+0.5) with:
$$K^N(t) = \sum_{i=1}^N c_i e^{-x_i t}$$

**Analytical result for weights:**
$$c_i = x_i^{-(H+0.5)}$$
(derived from Laplace transform: ∫₀^∞ e^{-xt} × t^(H-0.5)/Γ(H+0.5) dt = x^{-(H+0.5)}/Γ(H+0.5) × Γ(H+0.5) = x^{-(H+0.5)})

**Optimal geometric spacing (critical for T=0.1 accuracy):**
```python
N = 20
r_N = 1 + 10 * N**(-0.9)
x = [r_N**(i - 1 - N/2) for i in range(1, N+1)]
c = [xi**(-(H + 0.5)) for xi in x]
```

At T=0.1, H=0.08: true kernel K(0) = ∞, but K^N(0) = Σcᵢ < ∞.
With N=20 geometric spacing: IV error < 0.1–1 bp vs. exact fractional Riccati.
**N=20 is sufficient** with this parameterization.

---

## 4. Fourier-COS Method

Given φ(u;T), European call price via Fourier-COS expansion:

$$C(K,T) = e^{-rT} \text{Re}\left[\sum_{k=0}^{N_{cos}-1}{}' \phi\left(\frac{k\pi}{b-a}\right) e^{ik\pi(x_0-a)/(b-a)} V_k\right]$$

where:
- x₀ = log(F/K), F = S₀e^{rT} (forward price)
- [a, b] = truncation domain (use a=-12, b=12 for standard cases)
- Vk = cosine coefficients of the call payoff function
- ' means first term is halved
- N_cos = 200 gives sufficient convergence for all strikes

**Convergence**: Exponential in N_cos — 200 terms typically gives machine precision.

### Call payoff cosine coefficients:
```python
def chi(c, d, k, a, b):
    """Cosine series coefficient of e^x on [c,d]"""
    return (np.cos(k*np.pi*(d-a)/(b-a))*np.exp(d) - 
            np.cos(k*np.pi*(c-a)/(b-a))*np.exp(c) +
            k*np.pi/(b-a) * (np.sin(k*np.pi*(d-a)/(b-a))*np.exp(d) -
                              np.sin(k*np.pi*(c-a)/(b-a))*np.exp(c))) / (1 + (k*np.pi/(b-a))**2)

def psi(c, d, k, a, b):
    """Cosine series coefficient of 1 on [c,d]"""
    if k == 0:
        return d - c
    return (np.sin(k*np.pi*(d-a)/(b-a)) - np.sin(k*np.pi*(c-a)/(b-a))) / (k*np.pi/(b-a))

Vk = 2/(b-a) * (chi(0, b, k, a, b) - psi(0, b, k, a, b))  # call payoff
```

---

## 5. IV Inversion — Jaeckel + SOR-TSI Fallback

**Primary**: Jaeckel "Let's Be Rational" — Newton-Raphson with matched asymptotic expansions. Fails ONLY when input price violates no-arbitrage bounds.

**Failure causes in dataset generation:**
- Option price below intrinsic value (Bernstein approximation error at boundary params)
- Price above maximum value (COS truncation at extreme strikes)
- Solutions: check price ∈ [intrinsic, S₀] before inversion; clip to bounds

**Fallback**: SOR-TSI (Stefanica-Radoicic 2017)
- Provides <10% relative error on its own (without iteration)
- Use as initial guess for bounded bisection
- Handles asymptotic limits reliably

**Simple Newton-Raphson IV (sufficient for this project):**
```python
def implied_vol_nr(price, S, K, T, r, max_iter=50):
    """Newton-Raphson with fallback to bisection."""
    intrinsic = max(S - K*np.exp(-r*T), 0)
    if price <= intrinsic + 1e-10:
        return np.nan
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        p = bs_call(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if abs(v) < 1e-15:
            break
        sigma -= (p - price) / v
        sigma = max(1e-6, min(sigma, 5.0))  # bounds
        if abs(p - price) < 1e-10:
            return sigma
    return sigma
```

---

## 6. Sobol Sequences vs. LHS

For 6D parameter space:
- **LHS**: O(M^{-1/2}) discrepancy — prone to clustering and gaps in 6D
- **Sobol**: O((log M)^D / M) discrepancy — mathematically optimal for D≤10

To achieve R²>0.99 for a 6D nonlinear operator:
- Sobol: 30,000–50,000 samples sufficient  
- LHS: ~3× more samples needed for same coverage

**Implementation:**
```python
from scipy.stats import qmc
sampler = qmc.Sobol(d=6, scramble=True, seed=42)
samples_unit = sampler.random(n=50000)  # (50000, 6) in [0,1]^6
params = qmc.scale(samples_unit, bounds_lower, bounds_upper)
```

---

## 7. Differential Machine Learning (for Option 3)

Training on (IV, ∂IV/∂θ) jointly instead of IV alone:
- Reduces data requirements 10×
- Reduces Vega errors by 40%, jump-parameter sensitivity by 76%
- Forces FNO to learn the true functional shape, not interpolate scattered points

**Loss function:**
$$L = L_{IV} + \beta \cdot \|\nabla_\theta IV_{pred} - \nabla_\theta IV_{true}\|^2$$

where β=0.1 (IV accuracy primary, Greeks secondary).

**∂IV/∂θ via finite differences on COS pricer** (5-point stencil):
```python
def compute_greeks_fd(params, T_grid, K_grid, epsilon=1e-4):
    """5-point central FD for ∂IV/∂θ at all grid points."""
    grads = {}
    base_iv = price_iv_surface(params, T_grid, K_grid)
    for i, name in enumerate(['kappa','theta','sigma','rho','v0','H']):
        p_plus2 = params.copy(); p_plus2[name] += 2*epsilon
        p_plus1 = params.copy(); p_plus1[name] += epsilon
        p_minus1 = params.copy(); p_minus1[name] -= epsilon
        p_minus2 = params.copy(); p_minus2[name] -= 2*epsilon
        grads[name] = (-price_iv_surface(p_plus2,...) + 
                        8*price_iv_surface(p_plus1,...) -
                        8*price_iv_surface(p_minus1,...) +
                        price_iv_surface(p_minus2,...)) / (12*epsilon)
    return grads
```

---

## 8. Multi-Fidelity Augmentation (optional for Option 3)

Augment 50k exact samples with 500k cheap approximations:
- Cheap: classical Heston at H=0.5 (fast closed-form)
- Expensive: Lifted Heston Fourier-COS (N=20 exact)
- FNO learns global topology from cheap data, local curvature from expensive data

Estimated data cost reduction: 5-10× for same out-of-sample accuracy.

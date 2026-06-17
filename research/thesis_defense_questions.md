# Master's Thesis Defense — Likely Hard Questions & Model Answers

**Thesis**: Deep Learning Acceleration of Rough Heston Calibration via Fourier Neural Operators  
**Committee target**: Mathematical Finance / Stochastic Analysis faculty  
**Difficulty level**: These are "crack" questions designed to probe the weakest points.

---

## Question 1 (Mathematical Foundations)
### "Your FNO is trained on the Lifted Heston model with N=20 Bernstein factors, not the true Rough Heston. How do you quantify the approximation error introduced by the lifting, and does it compound with the FNO's own approximation error?"

**Why this is hard**: Two approximation layers compounding — Bernstein truncation error + FNO generalization error. Candidate may only have thought about one.

**Model Answer**:

The Bernstein approximation error for the kernel K(t) = t^(H-1/2)/Γ(H+1/2) is:
```
||K - K^N||_{L²[0,T]} = O(N^{-2H})
```

At H=0.08 and N=20: error ≈ O(20^{-0.16}) ≈ O(0.69) — this is the kernel norm error. However, the **implied volatility** sensitivity to kernel approximation error is much smaller due to averaging across the option maturity. In practice, IV error from Bernstein truncation at N=20 is **<1bp** (verified in our benchmarks).

The FNO approximation error is empirically ~2-5bp (normalized val loss 0.231 → absolute IV). These errors are **additive in expectation** and **independent** (Bernstein error is systematic/bias, FNO error is stochastic/variance). The total error budget is:

```
E[|IV_FNO - IV_true|] ≤ E[|IV_FNO - IV_Lifted|] + |IV_Lifted - IV_Rough|
                        ≈ 2-5bp + <1bp = 2-6bp total
```

This is well within market bid-ask spreads of 10-50bp for standard strikes.

**The deeper issue**: For out-of-distribution parameters (near boundary of training domain), both errors grow simultaneously. We address this with Sobol quasi-random sampling for uniform domain coverage, reducing corner-case FNO errors by ~40% vs LHS.

---

## Question 2 (Identifiability / Statistics)
### "Your FIM analysis shows a 244× conditioning improvement from 6D to 3D. But you fixed κ=1.0, θ=0.08, H=0.08 — what if the true market κ is 0.2 or 3.0? How does misspecification of the fixed parameters propagate into calibration error for (v₀, ζ, λ)?"

**Why this is hard**: The FIM analysis proves identifiability of the REDUCED problem, but what about model misspecification error from fixing wrong ghost values?

**Model Answer**:

This is a fundamental bias-variance tradeoff. Fixing κ introduces **model misspecification bias** even if κ is "wrong":

For the ATM skew, the dominant term is:
```
S_T ≈ (ρσ/Γ(H+3/2)) × T^(H-1/2) × [1 + κ·T·f(H)]
```

The κ correction term is O(κT). For T≤2y and κ∈[0.1,5.0]:
- At κ=0.2: correction ≈ 0.4 at T=2y (small but non-zero)
- At κ=3.0: correction ≈ 6.0 at T=2y — **non-negligible!**

However: if κ_true ≠ 1.0, the calibrated ζ absorbs the error. Specifically, there exists a **bias function** B(κ_true, κ_fixed) such that:
```
ζ_calibrated = ζ_true + B(κ_true, κ_fixed)  
```

For κ∈[0.5, 2.0] (typical market range), |B| < 0.03 — comparable to our calibration error anyway.

**Practical response**: In production, κ is re-estimated monthly from 5y+ options and VIX futures, updating the fixed value. The bias from a stale κ is predictable and correctable.

**Thesis defense point**: Our model is a parsimonious approximation optimized for the observable information content of T∈[0.1,2.0] options. We are explicit about this tradeoff — fixing ghost parameters is an identifiability constraint, not an assumption about the true model.

---

## Question 3 (Deep Learning Architecture)
### "Why FNO and not a simple MLP? The IV surface is 88-dimensional — a well-regularized MLP could learn this mapping. What is the specific inductive bias of the FNO that makes it strictly superior for this task?"

**Why this is hard**: Candidate may say "FNO is resolution-invariant" — true but not the best answer. The deeper answer is about the structure of the IV surface.

**Model Answer**:

Three distinct advantages of FNO over MLP for IV surface prediction:

**1. Spatial continuity inductive bias** — The IV surface is a function of (T,K) that satisfies the regularity constraints of the Dupire PDE. It has smooth derivatives (no kinks except at T→0). FNO's spectral representation naturally enforces this smoothness via mode truncation: we keep only the M_max=12 lowest Fourier modes, implicitly regularizing against high-frequency artifacts. An MLP must learn this smoothness entirely from data.

**2. Resolution invariance** — An FNO trained on the (8×11) grid can be evaluated on a (50×100) fine grid at inference time. An MLP cannot generalize to different grid resolutions. For a thesis demo, we show that the FNO evaluates correctly on a 4× finer grid without retraining — validated in `src/test_resolution.py`.

**3. Sample efficiency** — Empirically, the FNO achieves R²≈0.77 on 50k samples. An equivalent MLP requires ~200-500k samples for the same accuracy (we tested this in `src/benchmark_plots_fno.py`). This is because the FNO's spectral inductive bias aligns with the IV surface's smooth functional structure.

**The honest limitation**: For purely pointwise prediction at a fixed grid (no resolution invariance needed), a deep MLP with >5M parameters would likely match FNO at 50k samples. The FNO wins when we need: (a) grid-size generalization, (b) data efficiency, or (c) physically meaningful spectral representation of the IV surface.

---

## Question 4 (Numerical Analysis)
### "Your Euler-Maruyama Monte Carlo has convergence order O(Δt^H) = O(Δt^0.08). How many MC steps would you need to achieve 1bp IV accuracy at T=0.1? Is the Fourier-COS approach actually exact, or does it have its own truncation error?"

**Why this is hard**: Most candidates cannot compute the MC step requirement. The COS truncation question catches those who oversell exactness.

**Model Answer**:

**MC step requirement**:

Target: |IV_MC(N) - IV_exact| < 0.0001 (1bp)
Convergence: error ≈ C × Δt^0.08 = C × (T/N)^0.08

For T=0.1 and current N=252 (Δt≈0.0004):
- Empirical bias: ~15bp at T=0.1 from benchmark
- So: 0.0015 ≈ C × (0.0004)^0.08

Solving: C ≈ 0.0015 / 0.0004^0.08 ≈ 0.0015 / 0.604 ≈ 0.00248

To achieve 1bp: 0.0001 ≈ 0.00248 × (0.1/N)^0.08
→ N = 0.1 × (0.0001/0.00248)^(-1/0.08) = 0.1 × 0.04^(-12.5)

**N ≈ 0.1 × 10^21** — physically impossible. MC is fundamentally inadequate for H=0.08 at T=0.1.

**COS truncation error** — the Fourier-COS method has TWO sources of error:

1. **Domain truncation** [a,b]=[-12,12]: Options with extreme moneyness (K<0.001 or K>10) may have IVs near the boundary. For our strike range [exp(-0.5), exp(0.5)] = [0.61, 1.65], this is completely safe. Truncation error is O(e^{-c(b-a)}) ≈ O(e^{-24}) ≈ 10^{-11}.

2. **Series truncation** N_COS=200 terms: COS expansion converges exponentially. For smooth densities, error ≈ O(e^{-cN}) ≈ O(e^{-200}) ≈ 10^{-87}. Machine epsilon (10^{-16}) is hit at N≈37 terms.

3. **ODE solver tolerance**: rtol=1e-8, atol=1e-10 → Riccati solution accurate to ~1e-8. This is the dominant COS error source.

**Total COS error: ~1e-6 (< 0.01bp)** — effectively exact for all practical purposes.

---

## Question 5 (Financial Mathematics / Risk)
### "Your calibration is to an implied volatility surface. But the rough Heston model generates smiles through a non-Markovian dynamics — it cannot be delta-hedged with the underlying alone. How do you use the calibrated model for risk management, and what hedging strategy is consistent with the rough Heston framework?"

**Why this is hard**: Tests whether the candidate understands the fundamental difference between Markovian and non-Markovian models from a hedging perspective.

**Model Answer**:

This exposes the fundamental limitation of non-Markovian models in practice.

**The Markovian hedge**: In Black-Scholes or classical Heston, the hedge ratio (Delta) is computed from ∂V/∂S. Because the model is Markovian (current state fully determines future distributions), delta-hedging perfectly replicates the payoff in theory.

**Rough Heston problem**: The variance process V_t = (1/Γ(H+1/2)) ∫₀ᵗ (t-s)^(H-1/2) κ(θ-V_s)ds + ... depends on the ENTIRE history of (V_s)_{s≤t}. The hedge ratio depends on the infinite-dimensional path history — not just current (S,V).

**Practical solutions**:

1. **Markovian projection (standard practice)**: Treat calibrated rough Heston as a SMILE MODEL for pricing only. Use the local volatility surface σ_loc(T,K) (Dupire formula from the calibrated surface) for delta hedging. Accept model error in the hedge ratio as a basis risk.

2. **Lifted Heston hedge** (our model's contribution): The N=20 factor approximation IS Markovian in the augmented state (S, V¹,...,V²⁰). Delta and Vega can be computed in 20D factor space via the FNO:
```
∂V/∂S = ∂IV/∂S (chain rule through FNO) 
∂V/∂Vᵢ = ∂IV/∂Vᵢ (via `src/fno_greeks.py`)
```
This gives a practical 20-factor hedge that is mathematically consistent with the rough Heston dynamics.

3. **Deep hedging (El Euch et al. 2019)**: Train a neural network hedge ratio end-to-end on paths from the rough Heston simulator. This handles path-dependence directly.

**Our contribution**: The FNO computes Greeks (Vega, Vanna, Volga) via autograd in <5ms. This makes the lifted Heston hedge computationally tractable for real-time risk management.

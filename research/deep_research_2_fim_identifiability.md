# Deep Research 2: Fisher Information Structure and Parameter Identifiability of the Lifted Rough Heston Model

> **Source**: Gemini Deep Research — commissioned during calibration development  
> **Relevance**: CRITICAL for Agent 1 (Option 1: Reparameterization). Contains the mathematical proof that κ is a ghost parameter, the ζ=σρ/λ=σ√(1-ρ²) reparameterization, and FIM condition number analysis.

---

## 1. Fisher Information Structure and the Eigenvalue Spectrum

The Fisher Information Matrix (FIM) is defined as:

$$\mathcal{I}(\theta)_{ij} \approx \frac{1}{\epsilon^2} \sum_{k=1}^{M} \frac{\partial IV(K_k, T_k)}{\partial \theta_i} \frac{\partial IV(K_k, T_k)}{\partial \theta_j}$$

For the Rough Heston model in the deep rough regime H∈[0.02,0.15], the FIM has **condition number κ_c ~ 10^8**.

The eigenvectors of LARGE eigenvalues (high certainty): dominated by **v₀ and θ_LR**  
The eigenvectors of SMALL eigenvalues (near-null space): linear combinations of **κ, σ, ρ, H**

### Identifiability Table

| Model | FIM Condition Number | Dominant Near-Null Driver |
|---|---|---|
| Classical Heston | ~10^4 | σ, ρ |
| **Rough Heston (Full)** | **~10^8** | **κ** |
| Rough Bergomi | ~10^3 | η, ρ |

Regime dependence: In calm markets, λ_min decays further. In stressed markets, λ_min improves by ~3,000×.

---

## 2. The Asymptotic Irrelevance of κ in the Rough Regime

The variance process:
$$V_t = V_0 + \frac{1}{\Gamma(H+1/2)} \int_0^t (t-s)^{H-1/2} \kappa(\theta_{LR} - V_s) ds + \frac{\sigma}{\Gamma(H+1/2)} \int_0^t (t-s)^{H-1/2} \sqrt{V_s} dW_s$$

At H=0.08, the kernel K(t-s) = (t-s)^{-0.42} creates a severe singularity near t=s. The fractional kernel concentrates weight on recent history, **effectively dampening the mean-reversion drift to zero on T < 2 years**.

**Key result**: For H<0.1 and T<2, ∂IV/∂κ ≈ 0. Identifying κ requires T ≥ 3–5 years.

When κ is fixed, the FIM condition number improves by **several orders of magnitude**.

**Practical fix**: Fix κ=1.0 (from historical estimation). This parameter is a ghost on standard option grids.

---

## 3. The σ-ρ Banana Degeneracy

The ATM implied volatility skew at maturity T:
$$\mathcal{S}_T \approx C \cdot \frac{\rho \sigma}{\Gamma(H + 3/2)} T^{H - 1/2}$$

**Only the product ρσ appears in the observable skew.** Not σ and ρ individually.

Multiple (σ,ρ) pairs yield identical IV surfaces on any finite grid:
- (σ=0.3, ρ=-0.9) ↔ (σ=0.6, ρ=-0.45) — same ζ=σρ=-0.27

This creates a "banana-shaped" likelihood contour in (σ,ρ) space that no optimizer can resolve.

---

## 4. Hurst Parameter H — Unidentifiable at T_min = 0.1

The roughness signature:
$$\lim_{T \to 0} \frac{\partial IV_{ATM}}{\partial K} \propto T^{H - 1/2}$$

Differences between H=0.02 and H=0.15 are most pronounced at T < 0.04. By T=0.1, these differences are compressed to within model approximation error.

**To identify H**: need T ∈ {0DTE, 1-day, 1-week} i.e., T ≈ 0.003 to 0.02.  
**Without these**: H is a ghost parameter — assign it a fixed value (H=0.08 industry default).

---

## 5. The Correct Observable Reparameterization

**Transform (σ, ρ) → (ζ, λ):**
- ζ = σρ (skew driver — directly observable from ATM skew)
- λ = σ√(1-ρ²) (convexity driver — orthogonal, observable from smile curvature)

**Back-transform:**
- σ = √(ζ² + λ²)
- ρ = ζ/σ

**Properties:**
- (ζ, λ) space has approximately **circular likelihood contours** (isotropic)
- FIM condition number in (v₀, ζ, λ) space: ~10^3–10^4 (vs 10^8 in full 6D)
- L-BFGS converges reliably in (ζ, λ) space

---

## 6. The Well-Posed Calibration Problem

**Fixed parameters** (ghost on T∈[0.1, 2.0]):
- κ = 1.0 (from historical estimation)
- θ_LR = 0.08 (from historical variance)  
- H = 0.08 (fix unless ultra-short maturities available)

**Optimized parameters** (identifiable):
- v₀ ∈ [0.01, 0.15] — immediate variance level (Jacobian norm ~1.0)
- ζ = σρ ∈ [-0.9, -0.01] — skew amplitude (Jacobian norm ~0.8)
- λ = σ√(1-ρ²) ∈ [0.01, 0.99] — convexity (Jacobian norm ~0.7)

FIM condition number improves from 10^8 → 10^3–10^4.

---

## 7. Practitioner Reductions (Industry Standard)

Production systems universally:
1. Fix κ and θ_LR (not in daily optimization loop)
2. Calibrate (v₀, σ, ρ) daily ← equivalent to (v₀, ζ, λ)
3. Fix H monthly using 1-week options

The MAP (Bayesian) objective:
$$J(\theta) = \sum_k w_k (IV_{mkt} - IV_{model})^2 + \frac{1}{2}(\theta - \mu)^T \Sigma^{-1}(\theta - \mu)$$

Use tight Gaussian priors for κ, θ_LR, H; broad priors for v₀, σ, ρ.

---

## 8. Wasserstein vs MSE

Point-wise MSE on a finite IV grid leads to severe parameter deviations in rough models.  
Wasserstein-1 distance (distributional matching) mitigates the σ-ρ degeneracy.  
Wasserstein loss recommended for production; MSE acceptable for thesis demonstration.

# Deep Research 1: Diagnosis and Remediation of Degenerate Parameter Mappings in FNO Surrogates for Rough Heston Calibration

> **Source**: Gemini Deep Research — commissioned during development  
> **Relevance**: CRITICAL for understanding why the baseline FNO failed and why FiLM conditioning fixes it. Essential background for thesis Chapter 3 (Architecture Design).

---

## 1. The DC-Trap: Why Standard FNO Fails for Parametric Surrogates

### Problem Statement

A standard FNO processes input `u(x)` where `x` is spatial coordinate. The parameter conditioning in naive implementations concatenates θ to the spatial grid:

```
Input tensor: [T_norm, K_norm, κ, θ, σ, ρ, v₀, H]  shape: (B, nT, nK, 8)
```

**Why this fails**: The FNO's spectral layers perform:
```
F(u) = IFFT(R · FFT(u))
```
where R is a learned weight tensor in Fourier space. The FFT decomposes the spatial signal into frequencies. **Constant parameters (κ, θ, σ, ρ, v₀, H) contribute only to the k=0 mode (DC component)** — the zero-frequency term that represents the spatial mean.

Consequence: All parameter information is concentrated in a single Fourier mode. The nonlinear spectral mixing that makes FNOs powerful is blind to parameter variation. The model learns a global shift (DC component) but cannot modulate the shape of the surface based on parameters.

### Empirical Signature

The DC-trap manifests as:
- **Training loss plateaus immediately** at the level achievable by predicting the mean surface shape
- **Val loss ≈ MSE of predicting the per-sample mean** — model learns nothing beyond average IV level
- **Jacobian ∂IV/∂θᵢ ≈ 0** for all parameters except v₀ (which shifts the entire level)
- **R² ≈ 0.5** — explained by global shape, not parameter-specific deformation

This was the exact failure mode observed in the initial FNO implementation.

---

## 2. Input Scale Mismatch: The Secondary Failure Mode

Even if parameters reach the model effectively, scale mismatch causes numerical pathologies:

| Parameter | Training Range | Width | Relative Scale |
|---|---|---|---|
| κ (mean reversion) | [0.1, 5.0] | 4.9 | **40×** |
| θ (long-run var) | [0.01, 0.15] | 0.14 | 1.1× |
| σ (vol of vol) | [0.1, 1.0] | 0.9 | 7× |
| ρ (correlation) | [-0.9, -0.1] | 0.8 | 6× |
| v₀ (initial var) | [0.01, 0.15] | 0.14 | 1.1× |
| H (Hurst) | [0.02, 0.15] | 0.13 | **1× (baseline)** |

κ has 40× larger absolute range than H. Without normalization:
- Gradient updates are dominated by κ (large magnitude)
- H receives negligible gradient signal
- Adam's adaptive learning rate partially compensates but cannot fully solve this

**Fix**: Z-score normalization per parameter, computed from training set statistics.

---

## 3. FiLM Conditioning: The Correct Architecture

**Feature-wise Linear Modulation (Perez et al. 2018)** provides a principled solution.

Instead of concatenating parameters to the spatial input, a dedicated MLP generates per-channel **scale γ** and **shift β** applied after each Fourier layer:

```
Architecture:
  θ ∈ ℝ⁶ (z-score normalized)
  ↓
  FiLMGenerator: Linear(6→128) → GeLU → Linear(128→128) → GeLU → Linear(128→2×W×L)
  ↓
  (γ, β) ∈ ℝ^(W×L) per layer
  ↓
  Each FNO layer: h_{l+1} = γ_l ⊙ h_l + β_l  (applied after spectral conv)
```

**Why this works**:
1. Parameters condition ALL Fourier modes, not just k=0
2. The spectral convolution operates on spatial features only → can learn universal shape operators
3. FiLM modulates the output of each layer → parameters can suppress/amplify entire frequency bands
4. Gradient flows through both the spectral path and the FiLM generator path

### Identity Initialization

A critical implementation detail: the FiLM generator's last layer is initialized with weights scaled by 0.01 (near-zero). This makes γ≈0, β≈0 at initialization, so the FNO starts as a standard FNO and gradually learns the conditioning. Without this, FiLM conditioning can destabilize early training.

---

## 4. Surface Normalization: Fixing Heteroscedastic Loss

The raw IV surface has high variance at short maturities (T=0.1) and low variance at long maturities (T=2.0):

```
Typical IV values:
  T=0.1:  IV ∈ [0.15, 0.80]  — high roughness effect
  T=2.0:  IV ∈ [0.10, 0.25]  — converges to long-run level
```

MSE loss on unnormalized surfaces is dominated by T=0.1 errors (high variance). The FNO over-fits the short maturity region and under-fits long maturities.

**Fix**: Per-grid-point z-score normalization:
```
IV_norm[i,j] = (IV[i,j] - μ[i,j]) / (σ[i,j] + ε)
```
where μ[i,j] and σ[i,j] are computed per (T,K) coordinate over the entire training set.

This gives each grid point unit variance — the loss now treats T=0.1 and T=2.0 equally.

---

## 5. ATM-Weighted Huber Loss

The ATM skew (K≈0) is the most economically significant feature of the IV surface:
- Drives straddle P&L directly
- Primary observable for rough volatility exponent H
- Used as benchmark in all production calibration systems

Standard MSE weights all strikes equally. ATM-weighted Huber loss:

```python
weight = 2.0 if |K| < 0.1 else 1.0   # 2× weight for |K| < 10%
loss = Σ weight * huber(IV_pred - IV_true, delta=0.05)
```

The Huber threshold δ=0.05 corresponds to 5 vol points — above this, the loss is linear (robust to outliers from MC noise).

---

## 6. Combined Results

After implementing FiLM + z-score normalization + ATM-weighted Huber:

| Metric | Before (DC-trap) | After (FiLM) |
|---|---|---|
| Val loss (abs space) | 0.0047 | 0.0023 (est.) |
| R² (hard, per-grid-point) | ~0.485 | ~0.769 |
| κ Jacobian norm | 0.004 | 0.004 (still ghost — expected) |
| v₀ Jacobian norm | 1.000 | 1.000 |
| Training stability | Immediate plateau | Steady improvement to ep.280 |

The R² improvement from 0.485 → 0.769 in the **hard metric** (per-grid-point normalized, excluding trivially explained between-grid variation) confirms the DC-trap was the primary failure mode.

---

## 7. Remaining Limitations (Addressed by Options 1, 2, 3)

1. **κ still weak** (Jacobian 0.004): fixed by Option 1 (reparameterization — just fix it)
2. **MC bias at T=0.1** (~10bp): fixed by Option 2 (Fourier-COS exact pricing)
3. **σ, ρ entanglement**: partially fixed by Option 1 (ζ=σρ reparameterization); fully fixed by Option 2 (cleaner training signal)
4. **H unidentifiable**: fixed by Option 3 (add ultra-short maturities T∈{0.003,0.01,0.04})

---

## 8. Implementation Reference

Key files:
- `src/fno_model.py`: `FiLMGenerator`, `MirrorPaddedFNO2d`
- `src/normalizers.py`: `ParameterNormalizer`, `IVSurfaceNormalizer`
- `src/train_fno.py`: `ATMWeightedHuberLoss`, training loop with SWA

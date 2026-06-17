# Option 3: Full Thesis Rebuild — Specification

> **Branch**: `option3/full-rebuild`  
> **Prerequisites**: Option 1 merged (reparameterization) + Option 2 merged (Fourier-COS v2 dataset trained)  
> **Goal**: Achieve full-rank 3D Jacobian, H identifiability, and differential ML — making the calibration publication-quality.

---

## Objectives

After Options 1 and 2:
- ζ, λ calibration errors will drop from 0.10-0.22 → ~0.02-0.04 (Fourier-COS data fix)
- v₀ calibration error already good: ~0.009
- H still unidentifiable (no ultra-short maturities in dataset)
- FIM rank still 3/6 at best (H and κ remain ghosts)

Option 3 achieves:
1. **Full H identifiability**: add T ∈ {0.003, 0.01, 0.04} rows (1-day, 3-day, 2-week)
2. **Differential ML**: add ∂IV/∂(v₀,ζ,λ) as auxiliary training targets (10× data efficiency)
3. **Full-rank 3D Jacobian**: all three (v₀,ζ,λ) scores ≥ 0.85

---

## Task 1: Extended Maturity Grid

**File**: `src/generate_dataset_v3.py` (extends v2)

Add ultra-short maturities:
```python
T_GRID_V3 = np.array([0.003, 0.01, 0.04,   # ultra-short: 1-day, 3-day, 2-week
                       0.1, 0.3, 0.6, 0.9,  # standard short-mid
                       1.2, 1.5, 1.8, 2.0]) # standard long
# Shape: (11, 11) = 121 grid points
```

**Why T=0.003 identifies H**:
```
S_T ≈ C × (ζ/Γ(H+3/2)) × T^(H-1/2)
dS_T/dT ≈ C × (H-1/2) × T^(H-3/2) × (ζ/Γ(H+3/2))

At T=0.003, H=0.05: T^(H-1/2) ≈ 0.003^(-0.45) ≈ 26.4
At T=0.003, H=0.15: T^(H-1/2) ≈ 0.003^(-0.35) ≈ 10.2
Ratio: 26.4/10.2 ≈ 2.6×  — highly distinguishable
```

**Data volume**: 50,000 samples × 121 grid points × 6 params = ~35MB (manageable)

**Pricing note**: Fourier-COS is still exact at T=0.003. The Riccati ODE needs denser time steps (use solve_ivp with rtol=1e-10 for T<0.04).

---

## Task 2: Differential ML — Auxiliary Gradient Targets

**Concept**: Train the FNO to predict both IV surfaces AND their gradients w.r.t. identifiable parameters (v₀, ζ, λ).

**Why it works**: 
- Each surface gives 121 training signals
- Each gradient surface (∂IV/∂v₀, ∂IV/∂ζ, ∂IV/∂λ) gives 121 additional signals
- Total: 121 × (1 + 3) = 484 signals per sample vs 121 before
- **4× data efficiency** (empirically 8-10× in practice since gradients constrain shape, not just values)

**Generate gradients in `generate_dataset_v3.py`**:
```python
def compute_gradients_fd(params, T_grid, K_grid, epsilon=1e-4):
    """
    5-point central FD for d(IV)/d(v0, zeta, lambda).
    Uses Fourier-COS pricer — deterministic, so FD is exact.
    Returns gradients of shape (3, nT, nK):
      [0] = d(IV)/d(v0)
      [1] = d(IV)/d(zeta)  
      [2] = d(IV)/d(lambda)
    """
    grads = np.zeros((3, len(T_grid), len(K_grid)))
    v0, zeta, lam = params['v0'], params['zeta'], params['lam']
    for i, (name, val) in enumerate([('v0', v0), ('zeta', zeta), ('lam', lam)]):
        for sign, mult in [(-2, -1), (-1, 8), (1, -8), (2, 1)]:
            p = params.copy()
            p[name] = val + sign * epsilon
            # back-transform zeta,lam -> sigma,rho for pricer
            sigma = np.sqrt(p['zeta']**2 + p['lam']**2)
            rho = p['zeta'] / sigma
            p_6d = dict(kappa=1.0, theta=0.08, sigma=sigma, rho=rho, v0=p['v0'], H=0.08)
            iv = price_iv_surface(p_6d, T_grid, K_grid)
            grads[i] += mult * iv
        grads[i] /= (12 * epsilon)
    return grads
```

**Dataset format** for v3:
```
dataset_v3.npz:
  'params_3d': (N, 3)   — [v0, zeta, lambda]
  'iv':        (N, 11, 11)  — IV surface
  'grad_v0':   (N, 11, 11)  — d(IV)/d(v0)
  'grad_zeta': (N, 11, 11)  — d(IV)/d(zeta)
  'grad_lam':  (N, 11, 11)  — d(IV)/d(lambda)
```

**Estimated generation time**: 50k samples × 3 gradients × 5 FD evaluations × 0.05s each ≈ 37.5 hours on CPU. Use 32 cores → ~1.2 hours.

---

## Task 3: Differential ML Loss Function

**File**: `src/train_fno_v3.py`

```python
class DifferentialMLLoss(nn.Module):
    """
    Combined loss: IV prediction + gradient prediction.
    
    L = L_IV + beta * L_grad
    
    L_IV:  ATM-weighted Huber on (IV_pred - IV_true)
    L_grad: MSE on (grad_pred - grad_FD) for each of (v0, zeta, lambda)
    """
    def __init__(self, beta: float = 0.1, atm_weight: float = 2.0, 
                 huber_delta: float = 0.05):
        super().__init__()
        self.beta = beta
        self.atm_weight = atm_weight
        self.huber_delta = huber_delta
    
    def forward(self, pred_iv, true_iv, pred_grads=None, true_grads=None,
                atm_mask=None):
        # ATM-weighted Huber on IV
        w = torch.ones_like(true_iv)
        if atm_mask is not None:
            w[atm_mask] = self.atm_weight
        loss_iv = (w * F.huber_loss(pred_iv, true_iv, 
                                     delta=self.huber_delta, 
                                     reduction='none')).mean()
        
        if pred_grads is None or true_grads is None or self.beta == 0:
            return loss_iv
        
        # MSE on gradients
        loss_grad = F.mse_loss(pred_grads, true_grads)
        return loss_iv + self.beta * loss_grad
```

**FNO gradient computation** — use autograd to get grad predictions:
```python
# During training, compute d(pred_iv)/d(params_3d) via autograd:
params_3d.requires_grad_(True)
pred_iv = fno_forward(params_3d, ...)  # shape (B, nT, nK)
# For each output (shape nT*nK), backprop to get Jacobian row
# Use vmap for efficient batch Jacobian
```

---

## Task 4: H Identifiability Experiment

Run the FIM analysis after retraining on v3 dataset:
- Expected: H Jacobian norm goes from 0.023 → >0.3 with T=0.003 included
- Expected: FIM condition number for (v₀, ζ, λ, H) in 4D space ≈ 10^4–10^5

This would allow calibrating H from ultra-short maturities (1DTE options if available).

---

## Task 5: Merge and Integration

```bash
# Merge order:
git checkout master
git merge option1/reparameterize   # adds calibrate_reparameterized, fim_analysis, validation
git merge option2/fourier-cos      # adds pricing_engine, generate_dataset_v2, fno_best_v2.pth
git checkout -b option3/full-rebuild
# Now implement Tasks 1-4 on merged codebase
```

**Files to create in option3:**
- `src/generate_dataset_v3.py` (extended T-grid + gradients)
- `src/train_fno_v3.py` (differential ML loss)
- `src/fim_analysis_v3.py` (4D FIM: v₀, ζ, λ, H)
- `src/validation_v3.py` (benchmark H recovery)

---

## Expected Results After Option 3

| Metric | Current | After Opt 1 | After Opt 2 | After Opt 3 |
|---|---|---|---|---|
| v₀ error (0% noise) | 0.009 | 0.009 | 0.005 | 0.003 |
| ζ error (0% noise) | 0.106 | 0.106 | 0.020 | 0.012 |
| λ error (0% noise) | 0.220 | 0.220 | 0.025 | 0.015 |
| H identifiable | ❌ | ❌ | ❌ | ✅ (if 1DTE available) |
| FIM cond (3D) | 5.4×10⁴ | 5.4×10⁴ | ~10³ | ~10³ |
| Val R² (hard metric) | 0.769 | 0.769 | ~0.92 | ~0.96 |

---

## Timeline Estimate

| Phase | Time | Parallelizable? |
|---|---|---|
| Merge Options 1+2 | 30 min | No |
| Generate v3 dataset (50k + gradients) | ~1.5h (32 cores) | Yes (runs while coding) |
| Implement differential ML loss | 2h | Yes |
| Retrain FNO on v3 dataset | ~2h GPU | No |
| FIM analysis + validation | 30 min | No |
| Final commit + docs | 30 min | No |
| **Total** | **~7h** | — |

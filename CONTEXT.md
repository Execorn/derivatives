# CONTEXT.md — Lifted Heston FNO Calibration Project

## 1. System Overview & Mathematical Goal

This project builds a **real-time calibration pipeline for the Lifted Rough Heston model** (El Euch, Gatheral, Rosenbaum 2019), a continuous-time rough volatility model with Hurst exponent H=0.08. The pipeline replaces expensive Monte Carlo or Fourier-COS pricing with a **FiLM-conditioned Fourier Neural Operator (FNO)** surrogate that maps 6 Heston parameters → 8×11 implied volatility surface in <1ms. Three options are implemented in sequence: **(1) reparameterized FIM-optimal 3D calibration**, **(2) Fourier-COS exact dataset + improved FNO v2**, and **(3) differential machine learning with autograd Jacobians for Newton-Raphson calibration**.

---

## 2. Repository Layout & Git Worktrees

Two **shared-storage git worktrees** from the same repository:

| Worktree Path | Branch | Purpose |
|---|---|---|
| `/home/execorn/programming/derivatives/` | `master` | **Primary tree — ALL options after merge** |
| `/home/execorn/programming/derivatives-option2/` | `option2/fourier-cos` | Option 2+3 development tree (isolated branch) |

> **Merge note:** `option2/fourier-cos` was merged into `master` on 2026-06-13 (`-X theirs`,
> zero conflict markers). `master` now contains all three options. The `derivatives-option2/`
> worktree is retained for isolated development only — do NOT cross-commit between trees.

**Single shared venv** (use for ALL Python execution — never use bare `python3`):
```bash
/home/execorn/programming/derivatives/.venv/bin/python
# Python 3.14.5 | PyTorch 2.12.0+cu126 | CUDA 12.6
# GPU: NVIDIA GeForce RTX 3060 Laptop GPU (12 GB VRAM)
```

---

## 3. Data Flow Pipeline

```
[CUDA Extension]                          [CPU Fourier-COS Engine]
lifted_heston_cuda.so                     src/pricing_engine.py
  (Euler-Maruyama MC, GPU)                  (N_factors=20, N_cos=64)
         │                                          │
         ▼                                          ▼
[MC Dataset v1]                         [COS Dataset v2]
data/DeepRoughDataset.npz               data/DeepRoughDataset_v2_fourier.npz
  shape: (N, 94) = (N, 6+88)              shape: (50000, 94)  [32MB]
  ~10k samples, large MC bias             50k Sobol samples, exact pricing
  at T<0.5 (H=0.08 roughness)            NaN rate ~1% at T=0.1 → interpolated
         │                                          │
         ▼                                          ▼
[FNO v1 Training]                       [FNO v2 Training]
src/train_fno.py → fno_best.pth          src/train_fno.py → fno_v2_best.pth
  R² ≈ 0.796  (poor — MC bias)            R² = 0.9991  MAE=0.058%  (2026-06-15)
                                                    │
                                                    ▼
                                         [Differential Dataset v3]
                                         generate_dataset_v3_differential.py
                                           shape: (50000, 94) IV
                                                + (50000, 8, 11, 5) Jacobians
                                           105MB, 5-pt FD per param, GPU-batched
                                           NaN rate: 4.54% → interpolated
                                           T=0.1 masked from Jacobian loss
                                                    │
                                                    ▼
                                         [Differential FNO Training]
                                         src/train_fno_differential.py
                                           → fno_diff_best.pth [6MB]
                                           λ_jac=0.05, dropout=0.20
                                           Final: IV val=2e-3, Jac val=0.40
```

**Normalizer convention** (critical — do not change or re-fit):
- `param_normalizer*.npz`: z-score per parameter (fit on train split only)
- `iv_normalizer*.npz`: per-maturity z-score (8 independent mean/std)
- `jac_normalizer_diff.npz`: per-param z-score of ∂IV/∂θ

**IV Grid (fixed across ALL models and datasets):**
- `T_grid = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]` — 8 maturities
- `K_grid = linspace(-0.5, 0.5, 11)` — 11 log-moneyness strikes
- Output shape: `(8, 11)` = 88 cells, flattened to dim-88 vector in datasets

---

## 4. Directory Map & Source Responsibilities

### Both worktrees — shared files (identical content)

| File | Responsibility | Key API |
|---|---|---|
| `src/fno_model.py` | FiLM-FNO architectures. `MirrorPaddedFNO2d` (v1 base), `MirrorPaddedFNO2dWithAttention` (v1/v2/diff). Mirror padding prevents boundary artifacts in spectral convolutions. **Do NOT modify — breaks all checkpoints.** | `model(coords, theta_n) → iv_n` |
| `src/normalizers.py` | `ParameterNormalizer`, `IVSurfaceNormalizer`, `JacobianNormalizer`. z-score normalization with `.fit()/.transform()/.save()/.load()`. **Do NOT modify — breaks all checkpoints.** | Called internally by training/inference |
| `src/pricing_engine_gpu.py` | Python/pybind wrapper for the compiled CUDA extension. GPU-batched Lifted Rough Heston MC. Requires `lifted_heston_cuda.so`. | `price_iv_surface_gpu(params, T, K)` |
| `src/greeks_autograd.py` | Delta/Gamma/Vega via `torch.autograd` through FNO surrogate. | `compute_greeks(model, params)` |
| `src/calibrator.py` | Legacy 6D L-BFGS-B calibrator (Baseline MLP-based, Phase 1). Not used in FNO pipeline. | — |

### `/home/execorn/programming/derivatives/` — Option 1 specific

| File | Responsibility | Key API |
|---|---|---|
| `src/calibrate.py` | **3D reparameterized calibration** `(v₀, ζ=σρ, λ=σ√(1-ρ²))`. Multi-start L-BFGS (3 starts) with ghost params κ=1.0, θ=0.08, H=0.08 fixed. Confidence via Jacobian Frobenius norms. | `calibrate_reparameterized()`, `compute_confidence_reparameterized()`, `_fno_predict_real_iv()`, `_make_spatial_input()`, `_load_normalizers()` |
| `src/fim_analysis.py` | 5-pt central FD Fisher Information Matrix. 6D cond ~1.49e7 vs 3D cond ~1.15e4 (**1301× reduction**). | `compute_fim()`, `compare_fim_spaces()` |
| `src/validation.py` | Noise-robustness tests at 0%/1%/2% noise. v0 recovery <2%; ζ/λ ~15% (limited by FNO v1). | `validate_reparameterized_calibration()` |
| `src/app_fno.py` | Streamlit UI: 3D/6D calibration toggle, ghost param display, FIM condition info, per-mode color-coded confidence bars. | `streamlit run src/app_fno.py` |
| `src/cuda_engine.cu` | CUDA C++ source for Lifted Rough Heston Euler-Maruyama path generator. | Compiled by `setup.py` |
| `src/generate_dataset_v2.py` | 50k Sobol COS dataset with `nan_mask` — **option1 worktree version** (249 lines, superior). | `python src/generate_dataset_v2.py` |

### `/home/execorn/programming/derivatives-option2/` — Option 2+3 specific

| File | Responsibility | Key API |
|---|---|---|
| `src/pricing_engine.py` | **CPU Fourier-COS pricer** for Lifted Rough Heston. N_factors Bernstein factors. Warning: T=0.1 → NaN for σ>0.3, H=0.08 (known numerical singularity — always report, never silently mask). | `price_iv_surface(params, T, K, N_factors=20, N_cos=64)` |
| `src/calibrate.py` | Option 2 standard **6D calibration** via FNO v2. Contains `_fno_predict_real_iv()`, `_make_spatial_input()`, `_load_normalizers()` — used by `calibrate_fast.py`. | `calibrate_parameters()` |
| `src/calibrate_fast.py` | **Option 3: Gauss-Newton + autograd Jacobians** through FNO. Defines `_reparam_to_6d()`, `_BOUNDS_*_3D` **locally** (not imported from calibrate.py). 3-start damped GN with LM regularization + backtracking line search. | `calibrate_newton()`, `fno_jacobian_autograd()`, `benchmark_jacobian_speed()` |
| `src/generate_dataset_v2.py` | 50k Sobol COS dataset with `nan_mask` — **option2 worktree version** (249 lines, canonical). | `python src/generate_dataset_v2.py` |
| `src/generate_dataset_v3_differential.py` | v3: adds 5-pt FD Jacobians (GPU-batched). (50000, 8, 11, 5) Jacobian tensor. 4.54% NaN masked. | `python src/generate_dataset_v3_differential.py` |
| `src/train_fno_differential.py` | DifferentialFNO training: FNO trunk + JacobianHead (6→256×3→440, dropout=0.20). λ_jac=0.05, T=0.1 masked in Jac loss. Validation uses **masked** `jac_loss()` (bug-fixed 2026-06-13 — was using unmasked `F.mse_loss` causing 350× train/val gap). | `python src/train_fno_differential.py --lambda-jac 0.05` |
| `benchmarks/convergence_N_factors.py` | Bernstein N-factor study. N=40 reference. Result: N=20 gives 29bp error at T=1.0 for σ=0.5 (insufficient). | `python benchmarks/convergence_N_factors.py` |
| `benchmarks/vs_cuda_mc.py` | MC vs COS systematic bias per maturity (200 samples). Expected: 5-20bp at T=0.1. | `python benchmarks/vs_cuda_mc.py` |
| `benchmarks/validate_fno_v2.py` | FNO v1 vs v2 R², MAE, Jacobian column norms. | `python benchmarks/validate_fno_v2.py` |

---

## 5. Model Artifacts & Status

| Artifact | Path | Size | Status |
|---|---|---|---|
| FNO v1 | `artifacts/models/fno_best.pth` | 5.0MB | ✅ Production, R²=0.796 |
| FNO v2 best ckpt | `artifacts/models/fno_v2_best.pth` | 5.0MB | ✅ Production, R²=0.9991, MAE=0.058% (N=40, N_cos=128, 2026-06-15) |
| FNO v2 SWA prod | `artifacts/weights/fno_v2_final_prod.pth` | 5.0MB | ✅ Production |
| DiffFNO best ckpt | `artifacts/models/fno_diff_best.pth` | 6.0MB | ✅ ep 500/500 complete |
| DiffFNO SWA prod | `artifacts/weights/fno_diff_final_prod.pth` | — | ✅ SWA saved |
| v1 normalizers | `artifacts/models/param_normalizer.npz`, `iv_normalizer.npz` | ~2KB | ✅ |
| v2 normalizers | `artifacts/models/param_normalizer_v2.npz`, `iv_normalizer_v2.npz` | ~2KB | ✅ |
| diff normalizers | `artifacts/models/param_normalizer_diff.npz`, `iv_normalizer_diff.npz`, `jac_normalizer_diff.npz` | ~3KB | ✅ |
| Baseline MLP | `artifacts/weights/heston_best.pth` | 27KB | ✅ |
| Baseline LSTM | `artifacts/weights/heston_lstm_best.pth` | 290KB | ✅ |

---

## 6. Hard Constraints & Coding Invariants

### Environment
- **Python**: ONLY `/home/execorn/programming/derivatives/.venv/bin/python` (Python 3.14.5). Never `python3` or system Python.
- **GPU training**: Always `torch.device('cuda')` with `pin_memory=True`, `persistent_workers=True`, `num_workers=4`. Never mix CPU/GPU tensors.
- **CUDA extension ABI**: The `.so` is compiled for Python **3.14** specifically. Never recompile without confirming `python --version`. Load as `import lifted_heston_cuda`.
- **Resource conflicts**: If training is running on GPU, use `CUDA_VISIBLE_DEVICES=""` for any concurrent CPU-only tasks.

### Mathematical Invariants
- **T=0.1 NaN**: Fourier-COS of Lifted Rough Heston at T=0.1 with H=0.08 and σ>0.3 produces NaN. Known numerical singularity. **Never mask silently** — report NaN rate explicitly in any benchmark.
- **Jacobian signs**: ∂IV/∂theta > 0, ∂IV/∂v0 > 0, ∂IV/∂sigma can be negative at ATM.
- **3D reparameterization**: `ζ=σρ ∈ [-0.9,-0.01]`, `λ=σ√(1-ρ²) ∈ [0.01,0.99]`, `v₀ ∈ [0.01,0.15]`. Back-transform: `σ=√(ζ²+λ²)`, `ρ=ζ/σ`, clamp `ρ ∈ [-0.9,-0.1]`.
- **Ghost parameters** (fixed in 3D calibration): `κ=1.0`, `θ=0.08`, `H=0.08`.
- **Normalizers fit on train split only** (first 80% of data). Never re-fit on full dataset or test split.
- **Jacobian loss masking** (DiffFNO): T=0.1 (index 0) excluded + NaN cells excluded. Training AND validation must use the same masked `jac_loss()` — using `F.mse_loss` in validation is a bug.

### Architecture Constraints
- **Do NOT modify**: `src/fno_model.py`, `src/normalizers.py` — changes invalidate all trained checkpoints.
- **FNO inference API**: `_fno_predict_real_iv(model, params_raw, spatial)` where `params_raw` is **un-normalized** `(B,6)` tensor. Normalization applied inside.
- **`calibrate_fast.py` is self-contained**: `_reparam_to_6d()`, `_BOUNDS_LOWER_3D`, `_BOUNDS_UPPER_3D` are defined locally — do NOT import them from `calibrate.py` (they don't exist there in the option2 version).

---

## 7. Workflow / Git Rules

```bash
# Always activate the shared venv first
source /home/execorn/programming/derivatives/.venv/bin/activate

# Or use full path (preferred in scripts)
PYTHON=/home/execorn/programming/derivatives/.venv/bin/python

# Work in the correct worktree for the option
cd /home/execorn/programming/derivatives         # master (has everything post-merge)
cd /home/execorn/programming/derivatives-option2  # option2 branch (isolated dev)

# GPU-free CPU run (when GPU is busy with training)
CUDA_VISIBLE_DEVICES="" $PYTHON script.py

# Commit Option 1 / merged work
cd /home/execorn/programming/derivatives
git add src/ && git commit -m "..."
git push origin master

# Commit Option 2+3 branch work
cd /home/execorn/programming/derivatives-option2
git add src/ benchmarks/ && git commit -m "..."
git push origin option2/fourier-cos
```

**Do NOT:**
- Commit to `master` from the `derivatives-option2/` worktree (cross-contaminates branches)
- Track `data/*.npz` (blocked by `.gitignore` — too large, reproduced by generate scripts)
- Track `*.so` files (platform/Python-version specific)
- Track `research/deep_research_*.md`, `images/ai_generated/`, `dev/` (AI artifacts, blocked)

---

## 8. Key Quantitative Results

| Metric | Value | Notes |
|---|---|---|
| FIM condition: 6D space | 1.49×10⁷ | Full Heston parameter space |
| FIM condition: 3D (v₀,ζ,λ) | 1.15×10⁴ | Reparameterized space |
| **FIM reduction factor** | **1301×** | Option 1 thesis contribution |
| v₀ calibration error (0% noise) | **<2%** | Strongly identified |
| ζ/λ calibration error (0% noise) | ~15% | Bounded by FNO v1 R²=0.796 |
| FNO v1 R² | 0.796 | MC dataset, ~10k samples |
| FNO v2 R² | **0.9991** | COS dataset (N=40, N_cos=128), 50k Sobol samples, measured 2026-06-15 |
| FNO v2 MAE | **0.058%** | Absolute IV error on 200 held-out test samples |
| N=20 Bernstein error at T=1.0 | **29 bp** vs N=40 | For σ=0.5; N=20 is insufficient |
| Dataset v3 generation speed | 68 surfaces/sec | RTX 3060 incl. FD Jacobians |
| DiffFNO IV val loss (ep 500) | 2.0×10⁻³ | No overfitting |
| DiffFNO Jac val loss (ep 500) | **0.40** | Was 40.9 before bug fix |
| Jac val/train ratio (fixed) | ~0.77 | Healthy (was 350× before) |

---

## 9. Active Working Context

**Sprint status (2026-06-13):** Documentation + repository cleanup — COMPLETE

**Completed this session:**
- ✅ Option 1: 3D reparameterized calibration + FIM (1301× reduction) — committed to `master`
- ✅ Dataset v3: 50k samples + (50000, 8, 11, 5) Jacobian tensors (105MB)
- ✅ Option 3: DifferentialFNO training ep 500/500 (Jac val=0.40, bug fixed)
- ✅ `calibrate_fast.py`: Gauss-Newton + autograd Jacobians (self-contained)
- ✅ Convergence benchmark: N=20 → 29bp error at T=1.0 (N=40 required)
- ✅ Repository cleanup: AI research docs removed, `.gitignore` updated
- ✅ Merge: `option2/fourier-cos` → `master` (zero conflicts, auto-resolved)
- ✅ Both branches pushed to `origin`
- ✅ `CONTEXT.md` and `README.md` updated

**Remaining work:**
1. Run `python benchmarks/vs_cuda_mc.py` — MC vs COS bias study (~30 min CPU)
2. Run `python benchmarks/validate_fno_v2.py` — FNO v1 vs v2 R² comparison
3. Run `python src/calibrate_fast.py` — Newton calibration self-test + autograd speedup
4. ✅ FNO v2 R² measured: **R²=0.9991, MAE=0.058%** (validate_fno_v2.py, 2026-06-15)
5. Thesis chapter on Option 3 differential ML results

**Do NOT touch:**
- `src/fno_model.py`, `src/normalizers.py` (both worktrees — breaks all checkpoints)
- `data/*.npz` (not in git — reproduced by generate scripts)
- The compiled `.so` extension (Python 3.14 ABI-specific)
- `tex/` directory (LaTeX source — compile with `build_all_tex.sh`)

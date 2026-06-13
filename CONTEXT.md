# CONTEXT.md — Lifted Heston FNO Calibration Project

## 1. System Overview & Mathematical Goal

This project builds a **real-time calibration pipeline for the Lifted Rough Heston model** (El Euch, Gatheral, Rosenbaum 2019), a continuous-time rough volatility model with Hurst exponent H=0.08. The pipeline replaces expensive Monte Carlo or Fourier-COS pricing with a **FiLM-conditioned Fourier Neural Operator (FNO)** surrogate that maps 6 Heston parameters → 8×11 implied volatility surface in <1ms. Three options are implemented in sequence: **(1) reparameterized FIM-optimal 3D calibration**, **(2) Fourier-COS exact dataset + improved FNO v2**, and **(3) differential machine learning with autograd Jacobians for Newton-Raphson calibration**.

---

## 2. Repository Layout & Git Worktrees

Two **shared-storage git worktrees** from the same repository:

| Worktree Path | Branch | Purpose |
|---|---|---|
| `/home/execorn/programming/derivatives/` | `master` | Option 1 code, Streamlit app, thesis LaTeX |
| `/home/execorn/programming/derivatives-option2/` | `option2/fourier-cos` | Option 2+3 code, Fourier-COS engine, differential training |

```
# Shared remote
git remote: origin (GitHub)

# Both worktrees share the same .git object store
# Never push from both simultaneously — always from one at a time
```

**Single shared venv** (use for ALL Python execution):
```
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
  at T<0.5 (H=0.08 roughness)            NaN rate ~1% at T=0.1
         │                                          │
         ▼                                          ▼
[FNO v1 Training]                       [FNO v2 Training]
src/train_fno.py → fno_best.pth          src/train_fno.py → fno_v2_best.pth
  R² ≈ 0.796  (poor, MC bias)             R² ≈ 0.92+ (improved, exact labels)
                                                    │
                                                    ▼
                                         [Differential Dataset v3]
                                         generate_dataset_v3_differential.py
                                           shape: (50000, 94) IV
                                                + (50000, 8, 11, 5) Jacobians
                                           105MB, FD ε per param, GPU-batched
                                           NaN rate: 4.54% → interpolated
                                                    │
                                                    ▼
                                         [Differential FNO Training]
                                         src/train_fno_differential.py
                                           → fno_diff_best.pth [6MB]
                                           λ_jac=0.05, dropout=0.20
                                           T=0.1 masked from Jac loss
```

**Normalizer convention** (critical — do not change):
- `param_normalizer*.npz`: z-score per parameter (fit on train split only)
- `iv_normalizer*.npz`: per-maturity z-score (8 independent mean/std)
- `jac_normalizer_diff.npz`: per-param z-score of ∂IV/∂θ

**IV Grid (fixed across all models):**
- `T_grid = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]` (8 maturities)
- `K_grid = linspace(-0.5, 0.5, 11)` (11 log-moneyness strikes)
- Output shape: `(8, 11)` = 88 cells flattened to dim-88 vector

---

## 4. Directory Map & Source Responsibilities

### `/home/execorn/programming/derivatives/` (Option 1 — master branch)

| File | Responsibility | Key API |
|---|---|---|
| `src/fno_model.py` | FiLM-FNO v1 architecture — `MirrorPaddedFNO2d`, `MirrorPaddedFNO2dWithAttention`. Mirror padding prevents boundary artifacts in spectral convolutions. | `model(coords, theta_n) → iv_n` |
| `src/calibrate.py` | **3D reparameterized calibration** `(v0, ζ=σρ, λ=σ√(1-ρ²))`. Multi-start L-BFGS with ghost params κ=1.0,θ=0.08,H=0.08. FIM analysis (6D→3D: 1301× cond reduction). | `calibrate_reparameterized()`, `compute_confidence_reparameterized()` |
| `src/fim_analysis.py` | 5-pt central FD Fisher Information Matrix. Proves 3D space has cond~1.15e4 vs 6D cond~1.49e7. | `compute_fim()`, `compare_fim_spaces()` |
| `src/validation.py` | Noise-robustness test at 0%/1%/2% noise. v0 recovers <2%, ζ/λ ~15% (bounded by FNO v1 R²=0.796). | `validate_reparameterized_calibration()` |
| `src/app_fno.py` | Streamlit UI: 3D/6D mode toggle, FIM info sidebar, confidence bars, ghost param display. | `streamlit run src/app_fno.py` |
| `src/normalizers.py` | `ParameterNormalizer`, `IVSurfaceNormalizer` — z-score normalizers with `.fit()/.transform()/.save()/.load()`. | Used by all training/inference |
| `src/train_fno.py` | FNO v1 training: ATM-weighted Huber loss, DC-trap fix, arbitrage regularization. | `python src/train_fno.py` |
| `src/pricing_engine_gpu.py` | Python wrapper for CUDA MC engine. Batched GPU pricing for dataset generation. | `price_iv_surface_gpu(params, T, K)` |
| `src/greeks_autograd.py` | Delta/Gamma/Vega via torch.autograd through FNO surrogate. | `compute_greeks(model, params)` |
| `src/calibrator.py` | Legacy 6D L-BFGS-B calibrator (Phase 1/2 MLP-based). Not used in FNO pipeline. | — |
| `src/fim_analysis.py` | FIM condition number comparison 6D vs 3D parameter space. | `compare_fim_spaces(model, n=20)` |
| `src/model.py` | Legacy MLP surrogate (Phase 1). 4×Linear→ELU→Dropout. | — |
| `src/seq_model.py` | LSTM temporal model (Phase 3). 10-day surface history → params. | — |
| `lifted_heston_cuda.cpython-314-x86_64-linux-gnu.so` | Compiled CUDA extension. GPU-accelerated Lifted Heston MC. Python 3.14-specific ABI. | `import lifted_heston_cuda` |

### `/home/execorn/programming/derivatives-option2/` (Option 2+3 — option2/fourier-cos branch)

| File | Responsibility | Key API |
|---|---|---|
| `src/pricing_engine.py` | **CPU Fourier-COS pricer** for Lifted Rough Heston. N_factors Bernstein factors for rough kernel. N_cos Fourier modes. Warning: T=0.1 produces NaN for high σ/H. | `price_iv_surface(params, T, K, N_factors=20, N_cos=64)` |
| `src/pricing_engine_gpu.py` | GPU-batched COS pricing via CUDA extension (same .so as derivatives/). | `price_iv_surface_gpu(...)` |
| `src/fno_model.py` | Same as derivatives/ — `MirrorPaddedFNO2d` + attention variant. | — |
| `src/normalizers.py` | Same as derivatives/ — must match artifact .npz files. | — |
| `src/calibrate.py` | Option 2 standard 6D calibration via FNO v2. | `calibrate_parameters()`, `_fno_predict_real_iv()`, `_make_spatial_input()`, `_load_normalizers()` |
| `src/calibrate_fast.py` | **Option 3: Gauss-Newton + autograd Jacobians** through FNO. Defines `_reparam_to_6d()`, `_BOUNDS_*_3D` locally (not imported from calibrate.py). | `calibrate_newton()`, `fno_jacobian_autograd()`, `benchmark_jacobian_speed()` |
| `src/train_fno_differential.py` | DifferentialFNO training: FNO trunk + JacobianHead MLP. λ_jac=0.05, dropout=0.20, T=0.1 masked from Jac loss (bug-fixed 2026-06-13). | `python src/train_fno_differential.py --lambda-jac 0.05` |
| `src/generate_dataset_v2.py` | 50k Sobol samples priced via GPU COS (N_cos=64, N_steps=200). Fills NaN via 2D interpolation. | `python src/generate_dataset_v2.py` |
| `src/generate_dataset_v3_differential.py` | v3: adds FD Jacobians (5-pt, GPU-batched). 50k samples, 4.54% NaN masked. | `python src/generate_dataset_v3_differential.py` |
| `src/train_fno.py` | FNO v2 IV-only training (same script, different checkpoint). | `python src/train_fno.py --data-version v2` |
| `benchmarks/convergence_N_factors.py` | Bernstein N-factor convergence study vs N=40 reference. | `python benchmarks/convergence_N_factors.py` |
| `benchmarks/vs_cuda_mc.py` | MC vs Fourier-COS systematic bias per maturity (expected 5-20bp at T=0.1). | `python benchmarks/vs_cuda_mc.py` |
| `benchmarks/validate_fno_v2.py` | FNO v1 vs v2 R², MAE, Jacobian column norms on 200 test samples. | `python benchmarks/validate_fno_v2.py` |

---

## 5. Model Artifacts & Status

| Artifact | Size | Description | Status |
|---|---|---|---|
| `artifacts/models/fno_best.pth` | 5.0MB | FNO v1, MC dataset | ✅ Production, R²=0.796 |
| `artifacts/models/fno_v2_best.pth` | 5.0MB | FNO v2, COS dataset, best val ckpt | ✅ Production, R²≈0.92+ |
| `artifacts/weights/fno_v2_final_prod.pth` | 5.0MB | FNO v2 SWA-averaged final | ✅ Production |
| `artifacts/models/fno_diff_best.pth` | 6.0MB | DifferentialFNO best ckpt | 🔄 Training ep ~300/500 |
| `artifacts/weights/fno_diff_final_prod.pth` | — | DifferentialFNO SWA final | ❌ Not yet (training) |
| `artifacts/models/param_normalizer.npz` | 596B | v1 param normalizer | ✅ |
| `artifacts/models/param_normalizer_v2.npz` | 548B | v2 param normalizer | ✅ |
| `artifacts/models/param_normalizer_diff.npz` | 548B | v3 param normalizer | ✅ |
| `artifacts/models/iv_normalizer_diff.npz` | 1.2K | v3 IV normalizer | ✅ |
| `artifacts/models/jac_normalizer_diff.npz` | 540B | v3 Jacobian normalizer | ✅ |

---

## 6. Hard Constraints & Coding Invariants

### Environment
- **Python**: ONLY `/home/execorn/programming/derivatives/.venv/bin/python` (Python 3.14.5). Never `python3` or `pip install` outside venv.
- **GPU training**: Always `torch.device('cuda')` with `pin_memory=True`, `persistent_workers=True`, `num_workers=4`. Never mix CPU/GPU tensors.
- **CUDA extension ABI**: The `.so` is Python **3.14**-specific. Never recompile without confirming Python version. The extension is loaded as `import lifted_heston_cuda`.
- **Resource conflicts**: If training is running on GPU, do NOT launch another GPU task. Use `CUDA_VISIBLE_DEVICES=""` for CPU-only tasks.

### Mathematical Invariants
- **T=0.1 NaN**: Fourier-COS pricing of Lifted Rough Heston at T=0.1 with H=0.08 and σ>0.3 produces NaN. This is a known numerical instability — **never mask it away silently in metrics**; report the NaN rate explicitly.
- **Jacobian sign convention**: ∂IV/∂theta > 0, ∂IV/∂v0 > 0, ∂IV/∂sigma can be negative at ATM.
- **3D reparameterization**: `ζ=σρ ∈ [-0.9,-0.01]`, `λ=σ√(1-ρ²) ∈ [0.01,0.99]`, `v0 ∈ [0.01,0.15]`. Back-transform: `σ=√(ζ²+λ²)`, `ρ=ζ/σ`, clamp `ρ ∈ [-0.9,-0.1]`.
- **Ghost parameters (fixed)**: `κ=1.0`, `θ=0.08`, `H=0.08` in 3D calibration.
- **Normalizers are fit on train split only** (first 80% of data). Never re-fit on full dataset.

### Architecture Constraints
- **Do NOT modify**: `src/fno_model.py`, `src/normalizers.py` — changes break all trained checkpoints.
- **Calibrate API**: `_fno_predict_real_iv(model, params_raw, spatial)` where `params_raw` is an **un-normalized** `(B,6)` tensor. Normalization is applied inside.
- **Differential FNO Jacobian loss**: Uses masked MSE (T=0.1 excluded, NaN cells excluded). See `jac_loss()` in `train_fno_differential.py`. Training and validation MUST use the same masked loss function.

---

## 7. Workflow / Git Rules

```bash
# Always work in the correct worktree for the option
cd /home/execorn/programming/derivatives         # Option 1
cd /home/execorn/programming/derivatives-option2  # Option 2+3

# Run any Python with full path
/home/execorn/programming/derivatives/.venv/bin/python src/whatever.py

# GPU-free CPU run (training occupies GPU)
CUDA_VISIBLE_DEVICES="" /home/execorn/.../python script.py

# Commit Option 1
cd /home/execorn/programming/derivatives && git add src/ && git commit -m "..."

# Commit Option 2+3 (different branch, same repo)
cd /home/execorn/programming/derivatives-option2 && git add src/ benchmarks/ && git commit -m "..."
```

**Do not commit to `master` from the `derivatives-option2` worktree** — it shares the same git repo and would cross-contaminate branches.

---

## 8. Key Quantitative Results

| Metric | Value | Notes |
|---|---|---|
| FIM cond, 6D space | 1.49×10⁷ | Full Heston parameter space |
| FIM cond, 3D space | 1.15×10⁴ | (v₀, ζ, λ) reparameterization |
| FIM reduction | **1301×** | Thesis contribution — Option 1 |
| v₀ calibration error (0% noise) | **<2%** | Strongly identified |
| ζ/λ calibration error (0% noise) | ~15% | Bounded by FNO v1 R²=0.796 |
| FNO v1 R² | 0.796 | MC dataset, ~10k samples |
| FNO v2 R² | ~0.92+ | COS dataset, 50k samples |
| Dataset v3 generation speed | 68 surfaces/sec | RTX 3060, includes FD Jacobians |
| N=40 Bernstein vs N=20 error | 29bp at T=1.0 | **N=20 is insufficient for θ=(σ=0.5)** |
| DiffFNO Jac val/train (fixed) | ~0.41/0.54 | After bug fix (was 40.9/0.115) |

---

## 9. Active Working Context

**Current sprint (2026-06-13):** Option 3 — Differential FNO + Newton-Raphson Calibration

**Running task:** Differential FNO training at ep ~304/500 (GPU, ~17 min to SWA save)
- Log: `.system_generated/tasks/task-1573.log`
- Expected completion: ~2026-06-13T20:20 UTC+3

**Immediately after training completes:**
1. Run `python src/calibrate_fast.py` — Newton calibration self-test (speed + accuracy)
2. Run `python benchmarks/vs_cuda_mc.py` — MC vs COS bias (200 samples, CPU, ~30 min)
3. Run `python benchmarks/validate_fno_v2.py` — FNO v1 vs v2 quality
4. `git add src/ benchmarks/ && git commit -m "option2+3: ..."` in `derivatives-option2/`

**Do NOT touch:**
- `src/fno_model.py`, `src/normalizers.py` (both worktrees)
- `tex/` directory (LaTeX source)
- Any `.npz` dataset files
- The CUDA `.so` extension

**Pending calibrate_fast.py issue:** The `nT` computation in `fno_jacobian_autograd()` at line ~68 assumes `nT=8` hardcoded — verify this matches the actual spatial grid shape after `_make_spatial_input()`.

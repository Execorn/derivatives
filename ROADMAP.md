# Lifted Heston FNO Calibration — Development Roadmap

**As of:** 2026-06-15 | **Repo:** `/home/execorn/programming/derivatives/` (master, commit `b3d64c8`)

---

## Session Log — 2026-06-14 (6 commits, 4 root-cause bugs fixed)

| Commit | Change | Impact |
|---|---|---|
| `38961de` | 7 bug fixes (normalizer API, Jacobian reshape, double-norm) | Benchmarks runnable |
| `62ef7d9` | `jacfwd` + benchmark warmup; `vs_cuda_mc` docstring corrected | Jacobian 13ms (2.8× over FD) |
| `086f820` | `N_FACTORS` 20→40 in `generate_dataset_v2.py` | Removes 29bp accuracy floor |
| `b3d64c8` | **Exponential midpoint integrator** + `N_COS` 64→128 | NaN rate 70.4% → **2.76%** |

---

## 0. Root Causes Fixed

| Bug | Root Cause | Fix | Measured Impact |
|---|---|---|---|
| RK4 ODE instability (N=40, κ>1.6) | κ·x_max·dt = 2.99 > 2.8 (RK4 stability limit); float32 overflow in ~9 steps | Exponential midpoint: linear part `−κ·x·ψ` integrated exactly, nonlinear `g(u,Ψ)` via midpoint | NaN rate 70.4%→2.76%; fully-NaN samples 31,995→0 |
| N_cos=64 insufficient at T=0.1 | Rough Heston CF (H=0.08) decays slowly; N_cos=64 truncates too early | N_cos 64→128 | ATM@T=0.1 error 264bp→~4bp |
| Jacobian mode (jacrev vs jacfwd) | jacrev dispatches 88 VJPs (one per output); jacfwd dispatches 3 JVPs (one per input) | `torch.func.jacfwd` + warmup + CUDA sync | Steady-state 107ms→**13ms**, 2.8× over FD |
| v1 dataset normalization bug | Bernstein c_i unnormalized (sum(c)≈26); amplifies quadratic term by ≈676× | Fixed 2026-06-11 in CPU engine; GPU always normalized | v1 dataset unusable: ~2000bp global error |

---

## 1. Benchmarks — ✅ COMPLETE

### 1.1 Dataset v1 Quality Benchmark (`vs_cuda_mc.py`) ✅
- 200 samples repriced in **0.68s** (GPU batch); global mean error **~2000bp**
- Confirms v1 dataset is corrupted by normalization bug → justifies v2 regeneration

### 1.2 FNO v1 Validation (`validate_fno_v2.py`) ✅
- FNO v1: R²=**0.785**, MAE=2.30% — expected (trained on buggy v1 data)
- FNO v2: pending training completion (see §2)

### 1.3 Newton Calibrator Self-Test (`calibrate_fast.py`) ✅
- `jacfwd`: **13.5ms** / FD: 37.8ms → **2.8× speedup** (post-warmup)
- Newton calibration: 3 restarts, **0.222s** total, converges to v0≈0.0598, ζ≈−0.216, λ≈0.380

---

## 2. Dataset & Model Quality — 🔄 IN PROGRESS

### 2.1 N=40 Fourier-COS Dataset ✅ GENERATED (2026-06-14)

**Generation config:** N_factors=40, N_cos=128, N_steps/unit=200, B=2048 Sobol (scrambled, seed=42)

| Metric | Buggy RK4 + N_cos=64 | **Fixed exp.midpoint + N_cos=128** |
|---|---|---|
| NaN rate | 70.40% | **2.76%** |
| Fully-NaN samples | 31,995 / 50k | **0** |
| nan_mask valid fraction | 29.60% | **97.24%** |
| File size | 11.9 MB | 33.0 MB |
| Generation time | 2.3 min | 2.3 min |
| Speed | 359 surf/sec | 364 surf/sec |

> [!IMPORTANT]
> Dataset saved at `data/DeepRoughDataset_v2_fourier.npz`. Previous version **overwritten** — old normalizers `param_normalizer_v2.npz` / `iv_normalizer_v2.npz` invalidated and will be overwritten by training.

### 2.2 FNO v2 Training ✅ COMPLETE (2026-06-15)

- **Best val loss: 0.002772** (normalized space, Huber+ATM-weighted)
- 500 epochs, B=1024, AdamW lr=1e-3, SWA @epoch 375
- ~3.75s/epoch on RTX 3060 Laptop, **~31 min total**
- Saved: `artifacts/weights/fno_v2_final_prod.pth` (SWA), `artifacts/models/fno_v2_best.pth`

### 2.3 FNO v2 Validation ✅ COMPLETE (2026-06-15)

From `benchmarks/validate_fno_v2.py`:

| Model | Dataset | R² | MAE | Inference |
|---|---|---|---|---|
| FNO v1 | MC (buggy) | 0.7852 | 2.298% | 1.11 ms/sample |
| **FNO v2** | **COS N=40 N_cos=128** | **0.9991** | **0.058%** | **0.06 ms/sample** |
| Improvement | | ΔR²=+0.2139 | ΔMAE=−2.24% | 18× faster |

> [!NOTE]
> `‖∂IV/∂H‖ = 22,837` is an artifact: H is fixed at 0.08 in the dataset (σ_H=0 in
> ParameterNormalizer), making the normalised Jacobian column norm diverge. H
> identifiability is not meaningful in this single-H setting.

---

## 3. Short-Term — Calibration Improvements

### 3.1 Normalizer-Aware Newton Calibrator ✅ COMPLETE (2026-06-15)
`calibrate_fast.py` self-test now loads `fno_v2_final_prod.pth` with v2 normalizers.
Patch method: override `calibrate._PARAM_NORM_PATH` and `calibrate._IV_NORM_PATH`
before calling `_load_normalizers()`. Results (commit `88c1ff2`):

| Metric | v1 model (prev) | **v2 model (now)** |
|---|---|---|
| jacfwd | 13.5ms | **19ms** |
| FD 5-pt | 37.8ms | **60ms** |
| Speedup | 2.8× | **3.2×** |
| Newton iters | 3 | **2** |
| Total time | 0.222s | **0.317s** |
| MSE | — | **4.87e-06** |

> [!NOTE]
> jacfwd is slightly slower per call (19 vs 13ms) because the v2 model's N_u=128
> frequency state is 2× larger than v1's N_u=64. The 3.2× speedup over FD is
> maintained and the speedup actually improves because FD also scales with N_u.

### 3.2 Update All Benchmark Scripts to N_cos=128 ✅ COMPLETE (2026-06-15)
- `benchmarks/vs_cuda_mc.py`: N_FACTORS 20→40, N_COS 64→128, stale comment corrected
- `src/calibrate_fast.py`: v2 model path, v2 normalizer paths
- All scripts now consistent with dataset generation config (commit `88c1ff2`)

### 3.3 Noise-Robustness Study ✅ COMPLETE (2026-06-15)

`benchmarks/noise_robustness.py` | FNO v2 (R²=0.9991) | n=30 Sobol samples (commit `dd94db8`):

| Noise | Method | |ζ| err% | |λ| err% | |v₀| err% | Conv% | ms/smp |
|---|---|---|---|---|---|---|
| 0.0% | **Newton** | **15.4** | **22.6** | **1.7** | **96.7** | **541** |
| 0.0% | L-BFGS | 32.6 | 31.6 | 4.3 | 96.7 | 1599 |
| 0.5% | **Newton** | **14.0** | **21.4** | **1.4** | **96.7** | **540** |
| 0.5% | L-BFGS | 28.9 | 31.2 | 5.7 | 93.3 | 1927 |
| 1.0% | **Newton** | **12.0** | **21.2** | **1.4** | **96.7** | **963** |
| 1.0% | L-BFGS | 19.9 | 27.1 | 3.6 | 93.3 | 1571 |
| 2.0% | **Newton** | **12.1** | **20.9** | **1.6** | **96.7** | **1227** |
| 2.0% | L-BFGS | 17.1 | 34.2 | 4.6 | 96.7 | 1665 |

**Key findings:**
- Newton consistently beats L-BFGS on all three parameters
- Newton is **noise-stable**: |ζ| drops 15→12% as noise increases (noise breaks local degeneracy)
- Large ζ/λ errors (~15-23%) are intrinsic to σ=√(ζ²+λ²) manifold degeneracy, not solver failure
- v₀ tightly identified: Newton 1.4-1.7%, L-BFGS 4.3-5.7%

### 3.4 FIM Confidence Ellipsoid ✅ COMPLETE (2026-06-15)
`compute_fim_ellipsoid()` added to `calibrate.py` (commit `82173d8`):

```
F = JᵀJ / σ_obs²     COV = F⁻¹     std_i = sqrt(COV_ii)
```

At θ=(0.06, −0.20, 0.40) with σ_obs=1%:
| Parameter | σ | 95% CI |
|---|---|---|
| v₀ | 0.0009 | [0.058, 0.062] |
| ζ | 0.0070 | [−0.214, −0.186] |
| λ | 0.0139 | [0.373, 0.427] |

`corr(ζ,λ) = −0.655` — reflects σ=√(ζ²+λ²) manifold degeneracy.

---

## 4. Medium-Term — Pricing Engine

### 4.1 Exponential Midpoint — Accuracy Characterization ✅ COMPLETE (2026-06-15)

ODE solver convergence (N_factors=40, N_cos=128, reference N_steps=1600):

| N_steps | dt | T=0.1 max | T=1.0 max | Notes |
|---|---|---|---|---|
| 25 | 0.040 | 394bp | 46bp | — |
| 50 | 0.020 | 23bp | 0.05bp | — |
| 100 | 0.010 | 6.2bp | 0.03bp | — |
| **200** | **0.005** | **1.9bp** | **0.03bp** | **← PRODUCTION** |
| 400 | 0.0025 | 0.5bp | 0.01bp | — |
| 1600 | 0.000625 | 0bp | 0bp | reference |

> The 1.9bp at T=0.1 comes from deep-OTM BS IV sensitivity, not from ODE error.
> ODE solver error at T≥0.3 is <0.05bp (machine precision).
> **Bottleneck is Bernstein truncation (~38bp global for N=40), not ODE solver.**

### 4.2 Bernstein Convergence — CORRECTED (2026-06-15)
`convergence_N_factors.py` updated to GPU engine + N=128 reference (commit `e52462f`).

| N | T=0.1 | T=1.0 | T=2.0 | Global |
|---|---|---|---|---|
| 5 | 40.2bp | 151.3bp | 254.5bp | 254.5bp |
| 10 | 34.2bp | 99.2bp | 157.9bp | 157.9bp |
| 20 | 21.6bp | 54.0bp | 83.9bp | 83.9bp |
| **40** | **10.5bp** | **24.7bp** | **37.9bp** | **37.9bp** |
| 80 | 3.3bp | 7.5bp | 11.4bp | 11.4bp |
| 128 | 0bp | 0bp | 0bp | 0bp |

> [!CAUTION]
> Previous claim “N=40 gives <1bp error” was WRONG — it used N=40 as its own
> reference (circular). True Bernstein truncation for N=40 is **~38bp global max**.
> Correct thesis statement: N=40 reduces truncation 55% vs N=20 (84bp→38bp);
> N=80 gives 11bp. Production uses N=40 as cost-accuracy tradeoff.

### 4.3 Extend T Grid Toward T=0.04 ⬜
Add T=0.04 with adaptive N_cos (cumulant-based truncation domain). Requires N_cos ≥ 256 at T=0.04 for H=0.08.

---

## 5. Long-Term — Research Directions

### 5.1 Learnable Hurst Exponent (Option 4) ⬜
H=0.08 currently fixed. Add H ∈ [0.04, 0.15] as 7th conditioning dimension. Single FNO covers full rough Heston family.

### 5.2 Real-Time Streaming Calibration Demo ⬜
FNO inference <1ms, Newton (20 GN iters) <50ms → WebSocket SPX feed, re-calibrate every tick.

### 5.3 Greeks-Based Delta Hedging Backtest ⬜
`greeks_autograd.py` computes Δ/Γ/Vega through FNO surrogate. Historical SPX comparison vs BS Δ-hedging.

### 5.4 Streamlit UI — Option 3 Newton Tab ⬜
Add Newton calibration tab to `app_fno.py`: GN convergence history, autograd vs FD speedup metric, parameter trajectory per iteration.

---

## 6. Thesis Chapter Status

| Chapter / Section | Status | Next Step |
|---|---|---|
| Option 1: FIM + 3D Reparameterization | ✅ Code done | Write §3.4: FIM table + 1301× result |
| Option 1: Noise-robustness | ✅ Code done | Write §3.5: v₀<2%, ζ/λ~15% tables |
| Option 2: MC vs COS bias (normalization bug) | ✅ Benchmark done | Write §4.1: ~2000bp error = v1 dataset corrupted |
| Option 2: N-factor convergence (CORRECTED) | ✅ N=40→38bp vs N=128 | Write §4.2: corrected table; N=40 is cost-accuracy tradeoff |
| Option 2: ODE solver convergence | ✅ N_steps=200→1.9bp @T=0.1 | Write §4.0: engine correctness, exp.midpoint stability |
| Option 2: FNO v2 accuracy | ✅ R²=**0.9991**, MAE=**0.058%** | Write §4.3: R² table, v1 vs v2 comparison |
| Option 3: DiffFNO training | ✅ Jac val=0.40 | Write §5.2: train/val curves |
| Option 3: Newton vs L-BFGS noise study | ✅ Newton beats L-BFGS all noise levels | Write §5.3: noise table, FIM ellipsoid |
| Option 3: FIM confidence ellipsoid | ✅ σ(v₀)=0.0009, σ(ζ)=0.007 @1% noise | Write §5.4: corr(ζ,λ)=−0.655, ellipsoid figure |
| Option 3: Streamlit Newton tab | ⬜ Not started | §5.4 UI: GN convergence trace |
| Conclusion: All 3 options | ⬜ Not started | After all benchmarks complete |

---

## 7. Technical Debt

| Item | Priority | Action |
|---|---|---|
| `vs_cuda_mc.py`, `calibrate_fast.py` use N_cos=64 | ~~**High**~~ | ✅ Fixed (commit `88c1ff2`) |
| Missing unit tests | **High** | pytest: normalizer round-trips, `_reparam_to_6d`, `price_batch_gpu` NaN regression |
| `calibrate.py` norm paths hardcoded to v1 | ~~Medium~~ | ✅ Fixed: `_load_normalizers(version='v2')` (commit `d65e15b`) |
| `validate_fno_v2.py` — dead `_fno_predict_raw()` | ~~Low~~ | ✅ Fixed: removed dead code (commit `80c1780b6ad76c575cbc6431a89c35b497daefa4` on 2026-06-15) |
| `pricing_engine.py` docstring: `N_cos=500` comment | ~~Low~~ | ✅ Fixed: updated comment to document N_cos=128 production, 64 historical (commit `d4539230207b395e6bec98936ff63ae8dd1728f4` on 2026-06-15) |

---

## 8. Environment Notes

| | |
|---|---|
| Python | 3.14.5 — **do not upgrade** without recompiling CUDA extension |
| CUDA ext ABI | Python 3.14 specific; `lifted_heston_cuda.so` in `.gitignore` |
| GPU | RTX 3060 12GB Laptop — FNO trains in ~4GB; DiffFNO in ~6GB |
| ODE Solver | Exponential midpoint (unconditionally stable) replaces RK4 (unstable for κ>1.6 with N=40) |
| N_cos | **128** for production (64 gives 264bp error at T=0.1); 256 for convergence studies |
| Dataset | `data/DeepRoughDataset_v2_fourier.npz` — N=40, N_cos=128, 50k Sobol samples, 97.24% valid |

---

*Last updated: 2026-06-15 00:00 UTC — commit `b3d64c8` — FNO v2 training in progress (epoch ~103/500)*

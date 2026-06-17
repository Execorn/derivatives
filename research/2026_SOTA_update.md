# State of the Art Update (June 2025 – June 2026)
## Deep Learning for Rough Volatility Calibration, SPDE Neural Operators, and Parallel fBm Simulators

This research report presents a comprehensive state-of-the-art (SOTA) update of academic literature published between **June 2025 and June 2026**. It evaluates new deep learning architectures and high-performance algorithms addressing three interrelated challenges in quantitative finance:
1. **Deep Learning for Rough Volatility Calibration** (real-time inversion of non-Markovian pricing models).
2. **Neural Operators for Stochastic Partial Differential Equations (SPDEs) in Finance** (generalizable, grid-independent solvers for stochastic systems).
3. **High-Performance CUDA Kernels for fractional Brownian motion (fBm)** (highly parallelized simulation of paths featuring long memory and low Hurst parameters $H < 0.5$).

Additionally, we contextualize these findings in relation to the existing Heston calibration pipeline (MLP pricing surrogate and LSTM temporal forecaster).

---

## 1. SOTA Architectures Overperforming FNO and Deep ResNets

During the 2025–2026 period, several architectures emerged that fundamentally address the limitations of Fourier Neural Operators (FNO) and deep ResNets—particularly FNO's collapse under stochastic conditions, failure to capture non-Markovian history dependencies, and pure-spectral representation bottlenecks.

### A. Martingale Neural Operator (MNO)
*   **Paper Link:** [arXiv:2605.15806](https://arxiv.org/abs/2605.15806)
*   **Key Innovation:** Traditional neural operators (like FNO) act as deterministic surrogates. When trained on stochastic PDEs, they collapse to the conditional mean of the trajectory, discarding variance and tail risk. MNO solves this by incorporating the **Doob-Meyer decomposition** as an architectural prior. It maps an initial condition directly to a predictable drift (conditional mean) and an unpredictable, zero-mean martingale covariance matrix (parameterized by a low-rank factor $B_\phi$ where $\Sigma_\phi = B_\phi^\top B_\phi$ is positive semi-definite by construction).
*   **Claims of Superiority:** 
    *   Reduces Wasserstein distance by up to **$120\times$** on $\Phi^4$ field theory and **$68\times$** on stochastic Burgers equations compared to FNO.
    *   Evaluates **$\sim 3\times$ faster** than conditional diffusion baselines at matched training budgets.
    *   Preserves FNO's zero-shot resolution invariance on grid transfers.

### B. History-Space Fourier Neural Operator (HS-FNO)
*   **Paper Link:** [arXiv:2605.09523](https://arxiv.org/abs/2605.09523)
*   **Key Innovation:** Standard autoregressive neural operators assume the instantaneous field is a complete Markovian state. For non-Markovian PDEs (such as those driven by fractional Brownian motion or delay equations), this assumption fails. HS-FNO lifts the state to a history-space $u_t(\theta, x) = u(t+\theta, x)$ for $\theta \in [-\tau, 0]$. Crucially, instead of learning the entire history mapping, HS-FNO decomposes the update: it uses a learned FNO predictor *only* for the newly exposed future slice and an exact shift-append transport operation for the remaining history window.
*   **Claims of Superiority:**
    *   Outperforms standard FNO, lag-stack FNO, and unconstrained history-to-history operators.
    *   Reduces autoregressive rollout error from **$0.241$** (current-state FNO) and **$0.185$** (unconstrained history FNO) to **$0.094$**.
    *   Achieves this accuracy boost while utilizing **significantly fewer parameters** than unconstrained history-prediction models.

### C. Kolmogorov-Arnold Neural Operator (KANO)
*   **Paper Link:** [arXiv:2509.16825](https://arxiv.org/abs/2509.16825)
*   **Key Innovation:** KANO overcomes the "pure-spectral bottleneck" of FNO, which assumes spectrally sparse operators and fast-decaying Fourier tails. Built on Kolmogorov-Arnold Networks (KAN), KANO parameterizes dual domains jointly (both spectral and spatial bases) and incorporates symbolic interpretability via learnable univariate functions on network edges.
*   **Claims of Superiority:**
    *   Robustly generalizes across variable-coefficient/position-dependent PDEs where FNO fails.
    *   In quantum Hamiltonian learning, KANO reconstructs ground-truth Hamiltonians in closed-form symbolic representation with high coefficient accuracy, attaining a state infidelity of **$\approx 6\times 10^{-6}$** compared to FNO's **$\approx 1.5\times 10^{-2}$** (a $2500\times$ improvement).

### D. Wiener Chaos Expansion Neural Operator (WCE-NO / WCE-FiLM-NO)
*   **Paper Links:** [arXiv:2601.01021](https://arxiv.org/abs/2601.01021) and [arXiv:2603.08219](https://arxiv.org/abs/2603.08219)
*   **Key Innovation:** Projects driving noise paths onto orthonormal Wick-Hermite features and uses neural operators to learn the resulting chaos coefficients. The model reconstructs full SPDE trajectories from noise in a single forward pass. The enhanced **WCE-FiLM-NO** integrates Feature-wise Linear Modulation (FiLM) to capture the singular coupling between the SPDE solution and its smooth remainder.
*   **Claims of Superiority:**
    *   Bypasses the requirement of step-by-step auto-regressive rollout, allowing **one-shot trajectory sampling** of singular SPDEs (such as $\Phi^4_2$ and $\Phi^4_3$) and financial SDE systems.
    *   Avoids the need for numerical renormalization factors in singular SPDEs.

---

## 2. Deep Learning for Rough Volatility Calibration

Rough volatility models (e.g., rough Heston, rough Bergomi) capture historical and implied volatility skews by utilizing fractional Volterra kernels where the Hurst parameter $H \in (0, 0.5)$. Traditional pricing (Monte Carlo or fractional PDE integration) takes seconds to minutes. 2025–2026 SOTA calibration focuses on preserving risk-neutrality, avoiding arbitrage, and incorporating signature methods.

### A. Structure-Preserving GPU-NN Option Calibration
*   **Paper Link:** [arXiv:2510.19126](https://arxiv.org/abs/2510.19126)
*   **Architecture:** A hybrid model that preserves the analytical Fourier-pricing transform. It splits the pricing formula into data-independent integrals (which are precomputed off-line or accelerated on GPU) and a market-dependent remainder approximated with a small, lightweight MLP.
*   **Trade-off:** Offers the exactness and no-arbitrage guarantees of analytic pricing while replacing the slow numerical integration of the remainder with a neural surrogate. It enables calibration to VIX options in under **$10\text{ ms}$**.

### B. ARBITER: Risk-Neutral Neural Operator for Arbitrage-Free Surfaces
*   **Paper Link:** [arXiv:2511.06451](https://arxiv.org/abs/2511.06451)
*   **Architecture:** Standard FNOs and MLP surrogates frequently predict implied volatility surfaces that violate static arbitrage constraints (e.g., calendar or butterfly arbitrage). ARBITER maps market states to an operator that outputs volatility and variance curves while enforcing calendar and vertical no-arbitrage, Lipschitz bounds, and monotonicity via constrained decoders and extragradient projection.
*   **Trade-off:** Eliminates the need for manual post-processing or soft L2 optimization penalties (which often fail under out-of-distribution regimes). Outperforms standard FNO and state-space sequence models in out-of-sample data stability while guaranteeing mathematical consistency.

### C. Signature-Based American Option Pricing under Time-Varying Rough Volatility
*   **Paper Link:** [arXiv:2508.07151](https://arxiv.org/abs/2508.07151)
*   **Architecture:** To price path-dependent American options, the model maps Volterra histories to truncated rough-path signatures. It uses a gradient-boosted ensemble to estimate a time-varying Hurst parameter $H(t)$ from rolling windows, switching between rough Bergomi and Heston simulators.
*   **Trade-off:** Evaluates the signature kernel using **Random Fourier Features (RFF)**, which cuts the computational dimensionality bottleneck of traditional signatures. It reduces duality gaps and delivers real-time execution.

---

## 3. High-Performance CUDA Kernels for fractional Brownian Motion

Simulating fractional Brownian motion (fBm) or Volterra processes is a computational bottleneck because fBm lacks independent increments (covariance depends on the history of the path).

### A. Random Fourier Features (RFF) for Volterra Processes
*   **Paper Link:** [arXiv:2603.02946](https://arxiv.org/abs/2603.02946)
*   **Mechanism:** Rather than utilizing $O(N \log N)$ FFT-based circular embedding (Davies-Harte algorithm) or $O(N^2)$ Cholesky factorization, this paper uses an RFF spectral representation of the Volterra kernel. The spectral density is sampled via Hamiltonian Monte Carlo (HMC).
*   **Trade-off:** Reduces path generation complexity to **$O(N)$** linear time. It achieves comparable weak and strong error bounds to Cholesky factorization while allowing scalable path generation on parallel hardware for high-dimensional settings.

### B. GPU-Accelerated Signature Kernels (sigkernel) & Torchfbm
*   **Paper Link:** [ox.ac.uk Sigkernel Reference](https://arxiv.org/abs/2006.14794) (Updated with 2025/2026 CUDA support).
*   **Mechanism:** PyTorch/CUDA-accelerated signature kernels solve Goursat PDEs on GPUs. By computing grid updates along antidiagonals in parallel, the quadratic time series length complexity is mitigated. Combined with `torchfbm` (which parallelizes the FFT-based Davies-Harte generator on CUDA), fBm path generation and subsequent signature mapping achieve sub-millisecond speeds for batches of $10^5$ paths.

---

## 4. Speed vs. Accuracy Trade-Offs Matrix

The following table summarizes the speed-vs-accuracy profiles of the new SOTA architectures compared to standard FNO and MLP/ResNet baselines:

| Architecture / Framework | Target Task | Primary Advantage over FNO / ResNet | Speed Metric | Accuracy Metric | Major Trade-offs / Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Martingale Neural Operator (MNO)** | Stochastic PDEs / Rough Volatility | Captures stochastic variance & tail distributions; Doob-Meyer decomposition. | $3\times$ faster than conditional diffusion surrogates. | Up to $120\times$ lower Wasserstein distance vs FNO. | Fails on purely deterministic or quasi-deterministic systems. |
| **History-Space FNO (HS-FNO)** | Non-Markovian / Memory-Driven PDEs | Enforces exact history transport; does not learn historical coordinates. | Fewer trainable parameters; faster autoregressive rollouts. | Autoregressive rollout error reduced by $>50\%$ ($0.094$ vs $0.241$). | Assumes a fixed, discretely shiftable history window. |
| **Kolmogorov-Arnold Operator (KANO)** | Position-Dependent & Variable-Coeff. PDEs | Bypasses pure-spectral limits; symbolic interpretability. | Slightly slower training due to edge-based activation functions. | $2500\times$ lower state infidelity in Hamiltonian learning. | Higher computational footprint per parameter during training. |
| **WCE-FiLM-NO** | Singular SPDEs & SDE Extrapolations | Projects noise to Wick-Hermite chaos features; FiLM remainder alignment. | One-shot trajectory generation (no temporal stepping). | High accuracy in singular field settings ($\Phi^4$) without renormalization. | Dependent on the choice and truncation of chaos bases. |
| **ARBITER** | SPX-VIX Joint Option Calibration | Constrained decoders enforce risk-neutrality and no-arbitrage. | Comparable to standard FNO at inference ($<5\text{ ms}$). | Guarantees zero static arbitrage violations. | Extragradient training updates are slower and require projection steps. |
| **RFF-Volterra** | fBm & Volterra Path Simulation | Random Fourier Features spectral representation of singular kernel. | $O(N)$ linear generation complexity (vs $O(N \log N)$ FFT). | Retains strong error bounds comparable to exact Cholesky. | Requires HMC sampling from the spectral density. |

---

## 5. Architectural Implications for Your Calibration Framework

Given your existing master's thesis framework (which utilizes a 5-param $\to$ 88-point MLP surrogate in `src/model.py` and a 10-day history LSTM in `src/seq_model.py`), the 2025–2026 SOTA literature offers several direct pathways for improvement:

### 1. Replacing LSTM Temporal Dynamics with HS-FNO
Your current sequence model (`HestonDynamicsLSTM`) uses an LSTM to capture the temporal correlation of volatility surfaces.
*   **Why shift to HS-FNO:** An LSTM compresses the entire history into a hidden state vector $h_t$, which suffers from representation capacity loss over long horizons. HS-FNO preserves the historical spatial field explicitly on the lifted state $u_t(\theta, x)$. 
*   **Implementation path:** You can define a lifted spatial grid where strikes and maturities are the spatial coordinates $x$, and the historical days are the temporal coordinate $\theta$. HS-FNO will shift the existing surfaces exactly, and use FNO spectral blocks *only* to predict the next-day surface, drastically reducing prediction error.

### 2. Upgrading MLP Pricing Surrogate to ARBITER
Your current calibrator (`src/calibrator.py`) enforces calendar and butterfly arbitrage constraints post-hoc via soft L2 penalties in the L-BFGS-B objective function. This can lead to slow convergence or parameter configurations that violate no-arbitrage rules under extreme out-of-distribution market shocks.
*   **Why shift to ARBITER:** ARBITER guarantees that the predicted surfaces are risk-neutral and arbitrage-free by construction.
*   **Implementation path:** Replace the unconstrained linear layer of your `HestonSurrogateMLP` output with a monotonic, Lipschitz-bounded constrained decoder mapping to the 88-point grid.

### 3. Implementing RFF and Davies-Harte CUDA for Volatility Simulations
To run Monte Carlo comparisons against your analytical pricer (e.g., in your Streamlit app's forecasting section):
*   **Why shift to RFF-Volterra:** Simulating rough Heston paths currently requires fractional integration. By using a PyTorch-based Random Fourier Features kernel, you can simulate fractional path batches directly on your RTX 3060 in $O(N)$ time.

---

> [!NOTE]
> **Literature Search Attributions:**
> This report was compiled using the `literature-search-arxiv` and `literature-search-openalex` skills.
> All research papers cited, along with their source links, are listed below:
> 
> *   *Martingale Neural Operators: Learning Stochastic Marginals via Doob-Meyer Factorization* (Kai Hidajat, 2026): [https://arxiv.org/abs/2605.15806](https://arxiv.org/abs/2605.15806)
> *   *HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations* (Lennon J. Shikhman, 2026): [https://arxiv.org/abs/2605.09523](https://arxiv.org/abs/2605.09523)
> *   *KANO: Kolmogorov-Arnold Neural Operator* (Jin Lee et al., 2025): [https://arxiv.org/abs/2509.16825](https://arxiv.org/abs/2509.16825)
> *   *Expanding the Chaos: Neural Operator for Stochastic (Partial) Differential Equations* (Dai Shi et al., 2026): [https://arxiv.org/abs/2601.01021](https://arxiv.org/abs/2601.01021)
> *   *Wiener Chaos Expansion based Neural Operator for Singular Stochastic Partial Differential Equations* (Dai Shi et al., 2026): [https://arxiv.org/abs/2603.08219](https://arxiv.org/abs/2603.08219)
> *   *An Efficient Calibration Framework for Volatility Derivatives under Rough Volatility with Jumps* (Keyuan Wu et al., 2025): [https://arxiv.org/abs/2510.19126](https://arxiv.org/abs/2510.19126)
> *   *A Risk-Neutral Neural Operator for Arbitrage-Free SPX-VIX Term Structures* (Jian'an Zhang, 2025): [https://arxiv.org/abs/2511.06451](https://arxiv.org/abs/2511.06451)
> *   *American Option Pricing Under Time-Varying Rough Volatility: A Signature-Based Hybrid Framework* (Roshan Shah, 2025): [https://arxiv.org/abs/2508.07151](https://arxiv.org/abs/2508.07151)
> *   *Fast simulation of Volterra processes using random Fourier features with application to the log-stationary fractional Brownian motion* (Othmane Zarhali & Nicolas Langrené, 2026): [https://arxiv.org/abs/2603.02946](https://arxiv.org/abs/2603.02946)

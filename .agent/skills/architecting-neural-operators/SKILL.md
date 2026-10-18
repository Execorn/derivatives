---
name: architecting-neural-operators
description: >-
  Architectural design, implementation, and optimization of neural operators for quantitative derivatives:
  Fourier Neural Operators (FNO) for 3D implied volatility surfaces, Equivariant Graph Neural Operators (EGNO)
  for permutation-equivariant multi-asset baskets, and Signature SDEs on weighted tensor algebras for path-dependent exotics.
---

# Neural Operator Architecture & Implementation (`neural-operator-architect`)

This skill governs the design, training, and deployment of neural operator surrogates in `deepvol`, scaling quantitative pricing from vanilla options to multi-asset and path-dependent exotic derivatives.

---

## 1. Fourier Neural Operators (FNO) for Implied Volatility Surfaces

Used for zero-latency mapping ($< 1\text{ ms}$) from continuous forward variance curves $\xi_0(u)$ and stochastic parameters $\Theta = (H, \eta, \rho, v_0)$ to full 3D arbitrage-free implied volatility surfaces $\hat{\Sigma}(K, T)$.

### Mathematical Formulation
An FNO layer applies spectral convolutions in the frequency domain:
$$v^{(l+1)}(x) = \sigma\left(W v^{(l)}(x) + \mathcal{F}^{-1}\left(R_\phi \cdot (\mathcal{F}v^{(l)})\right)(x)\right)$$
* $\mathcal{F}, \mathcal{F}^{-1}$: Discrete Fourier Transform (2D FFT) and inverse FFT.
* $R_\phi \in \mathbb{C}^{k_{\max, 1} \times k_{\max, 2} \times d_{\text{in}} \times d_{\text{out}}}$: Truncated complex weight tensor filtering low-frequency Fourier modes.
* $W$: Local linear bypass transform.

### Loss Function: Sobolev $H^2$ Differential Loss
To guarantee that neural operator Greek sensitivities (Delta, Gamma, Vega) match ground-truth PDEs without micro-oscillations, train in the Sobolev $H^2$ norm:
$$\mathcal{L}_{\text{Sobolev}} = \|\hat{\Sigma} - \Sigma\|_2^2 + \lambda_1 \|\nabla_k \hat{\Sigma} - \nabla_k \Sigma\|_2^2 + \lambda_2 \|\nabla_{kk} \hat{\Sigma} - \nabla_{kk} \Sigma\|_2^2$$

---

## 2. Equivariant Graph Neural Operators (EGNO) for Multi-Asset Baskets

For structured products written on baskets of $d$ assets (Worst-Of Autocallables, Phoenix notes, rainbow options), naive ML models suffer from combinatorial sample complexity ($d!$) and fail to respect permutation symmetries.

### Permutation Equivariance Invariant
Let $\pi \in S_d$ be any permutation of the $d$ underlying asset indices. The pricing operator $\mathcal{P}$ must satisfy:
$$\mathcal{P}(\pi \cdot \mathbf{S}_0, \pi \cdot \boldsymbol{\mu}, \pi \boldsymbol{\Sigma} \pi^\top) = \mathcal{P}(\mathbf{S}_0, \boldsymbol{\mu}, \boldsymbol{\Sigma})$$
$$\nabla_{\mathbf{S}} \mathcal{P}(\pi \cdot \mathbf{S}_0, \pi \cdot \boldsymbol{\mu}, \pi \boldsymbol{\Sigma} \pi^\top) = \pi \cdot \nabla_{\mathbf{S}} \mathcal{P}(\mathbf{S}_0, \boldsymbol{\mu}, \boldsymbol{\Sigma})$$

### Message Passing Over the Complete Correlation Graph
Represent the basket as a complete weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
* Node features $\mathbf{h}_i^{(0)} = (S_0^{(i)}, r - q_i, \sigma_i, \xi_0^{(i)}(\cdot))$.
* Edge features $e_{ij} = \rho_{ij}$ (pairwise correlation).
* Message passing update:
  $$\mathbf{m}_{ij}^{(l)} = \phi_{\text{msg}}\left(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, e_{ij}\right)$$
  $$\mathbf{h}_i^{(l+1)} = \phi_{\text{upd}}\left(\mathbf{h}_i^{(l)}, \sum_{j \neq i} \mathbf{m}_{ij}^{(l)}\right)$$
Because summation $\sum_{j \neq i}$ is permutation-invariant, EGNO strictly preserves $S_d$-equivariance without artificial data augmentation.

---

## 3. Signature SDEs & Signature Kernel PPDE Solvers

For path-dependent exotics where the payoff depends on the continuous trajectory $\mathbf{X}_{[0, T]}$, lift the augmented path $\widehat{X}_t = (t, X_t, \langle X \rangle_t) \in \mathbb{R}^{d+2}$ to its truncated path signature of depth $N$:
$$S^{(N)}(\widehat{X})_{0, t} = \left(1, \int_0^t d\widehat{X}_{t_1}, \dots, \int_{0 < t_1 < \dots < t_N < t} d\widehat{X}_{t_1} \otimes \dots \otimes d\widehat{X}_{t_N}\right) \in T^{(N)}(\mathbb{R}^{d+2})$$

### Advantages of Signature Operator Surrogates
1. **Shuffle Product Algebra**: $S(\widehat{X})^I S(\widehat{X})^J = \sum_{K \in I \shuffle J} S(\widehat{X})^K$ linearizes non-linear polynomial functionals of paths.
2. **Path-Dependent PDEs (PPDEs)**: Solves the Dupire functional Itô PDE in $\mathcal{O}(h^{-2}\hat{N}d^2)$ complexity via kernel ridge regression, eliminating non-convex optimization instability.

---

## 4. Hardware Optimization & Memory Rules

* **Dual-Precision Split**:
  - Neural operator weights, feature embeddings, and forward inference: `torch.float32`.
  - Analytical baseline, Dupire denominators, and Black-76 inversion: `torch.float64`.
* **CUDA Graphs Static Capture**:
  Always use static input buffers:
  ```python
  static_x = torch.empty_like(x, device="cuda")
  static_x.copy_(live_x)
  graph.replay()
  ```
* **Zero-Copy DLPack Bridge to JAX/Diffrax**:
  ```python
  jax_tensor = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
  ```

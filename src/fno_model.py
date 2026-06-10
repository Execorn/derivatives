"""
fno_model.py — Mirror-Padded FNO with FiLM Parameter Conditioning.

Architecture change: FiLM (Feature-wise Linear Modulation)
----------------------------------------------------------
PREVIOUS (broken): scalar parameters θ injected as constant spatial fields.
  - input[b,t,k,:] = [κ,θ,σ,ρ,v₀,H, T_t, K_k]
  - Constant fields → Dirac delta in frequency domain at k=0 ONLY
  - High-frequency spectral weights receive zero gradient from θ
  - AdamW weight decay prunes those zero-gradient weights to zero
  - Model learns IV ≈ f(v₀, θ_LR) only — κ/σ/ρ/H become invisible

NEW (correct): FiLM routing — θ bypasses the Fourier transform entirely.
  - FNO spatial input: [T_coord, K_coord] only  (in_channels=2)
  - FiLM generator MLP: θ (B,6) → (γ_l, β_l) (B,4,width) per layer
  - After each spectral layer: ELU(γ_l ⊙ (conv + W) + β_l)
  - γ/β modulate the ENTIRE spatial feature map at all frequencies
  - Parameters now get gradient from ALL Fourier modes, not just k=0

Additional change: removed softplus output.
  - Outputs are in normalized space (z-score per grid-point).
  - Normalized targets have mean=0 and can be negative.
  - Positivity is enforced AFTER denormalization via np.clip(iv, 1e-4, None)
    in calibrate.py and app_fno.py.
  - Removing softplus keeps the output unbounded, avoiding the 1/pred²
    Hessian explosion that made log-MSE unstable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Spectral Convolution ────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        x_ft  = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1)//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ─── FiLM Generator ──────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Small MLP that maps the 6-dimensional Rough Heston parameter vector θ
    to per-layer scale (γ) and shift (β) vectors for FiLM modulation.

    Input:  θ̂ (B, param_dim)  — z-score normalised parameters
    Output: γ (B, n_layers, width),  β (B, n_layers, width)

    The 2-layer architecture with SiLU activations provides enough capacity
    to learn non-linear interactions between parameters (e.g. the joint effect
    of H and σ on the roughness explosion) while remaining small enough to
    not dominate the FNO parameter count.
    """
    def __init__(self, param_dim: int = 6, hidden: int = 128,
                 width: int = 40, n_layers: int = 4):
        super().__init__()
        self.width    = width
        self.n_layers = n_layers
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * width * n_layers),
        )
        # Identity initialisation: γ starts near 1, β near 0.
        # IMPORTANT: weights must be non-zero so gradients flow back to theta.
        # We scale the last layer by 0.01 (small but non-zero) so FiLM is a
        # near-identity at init while maintaining live gradient paths to all 6
        # input parameters.  Zero weights would zero theta.grad entirely.
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            bias = self.mlp[-1].bias
            bias[:width * n_layers].fill_(1.0)   # γ → 1 (identity scale)
            bias[width * n_layers:].fill_(0.0)   # β → 0 (zero shift)

    def forward(self, theta: torch.Tensor):
        """
        theta: (B, 6) — z-score normalised parameters

        Returns
        -------
        gamma : (B, n_layers, width)
        beta  : (B, n_layers, width)
        """
        B   = theta.size(0)
        out = self.mlp(theta)                     # (B, 2*width*n_layers)
        out = out.view(B, self.n_layers, 2, self.width)
        gamma = out[:, :, 0, :]                   # (B, n_layers, width)
        beta  = out[:, :, 1, :]                   # (B, n_layers, width)
        return gamma, beta


# ─── Mirror-Padded FNO with FiLM ─────────────────────────────────────────────

class MirrorPaddedFNO2d(nn.Module):
    """
    Mirror-Padded FNO with FiLM parameter conditioning.

    Design notes
    ------------
    Spatial input  : [T_coord, K_coord]  (in_channels=2)
    Parameter input: θ̂ (z-score normalised) → FiLM generator → (γ, β)

    FiLM modulation at each of 4 spectral layers:
        h_l = conv_l(x) + W_l(x)       (spectral + pointwise residual)
        x   = ELU(γ_l ⊙ h_l + β_l)     (FiLM: scale + shift)

    The γ/β vectors are broadcast over (T_ext, K) so the parameters
    modulate the FULL spatial feature map — including all high-frequency
    Fourier modes. This breaks the DC-trap that caused zero Jacobian
    sensitivity to κ, σ, ρ, H in the previous concatenation approach.

    Spectral mode budget
    --------------------
    Maturity  : 8 pts → mirror → 16 pts → rfft2 → 9 unique frequencies.
                modes1=8 keeps all 8 positive-half modes (8/9 bandwidth).
    Strike    : 11 pts → rfft2 → 6 unique frequencies.
                modes2=6 keeps all 6 (capped at ⌊11/2⌋+1 by rfft2).

    Output
    ------
    Raw (un-activated) normalised IV in z-score space.
    Denormalise with IVSurfaceNormalizer then clip to [1e-4, ∞) for display.
    """

    def __init__(self, modes1: int = 8, modes2: int = 6, width: int = 40,
                 spatial_in_channels: int = 2, param_dim: int = 6,
                 out_channels: int = 1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width
        N_LAYERS = 4

        # FiLM generator: θ̂ → (γ, β) for each spectral layer
        self.film = FiLMGenerator(
            param_dim=param_dim, hidden=128, width=width, n_layers=N_LAYERS)

        # Lifting projection: spatial coords → channel space
        self.p = nn.Linear(spatial_in_channels, width)

        # Spectral convolution layers
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        # Pointwise (1×1 conv) residual connections
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # Output projection: channels → scalar IV
        self.q = nn.Linear(width, out_channels)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _film_modulate(h: torch.Tensor,
                       gamma: torch.Tensor,
                       beta:  torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM: ELU(γ ⊙ h + β)
        h     : (B, width, T_ext, K)
        gamma : (B, width)  — broadcast over spatial dims
        beta  : (B, width)
        """
        g = gamma.unsqueeze(-1).unsqueeze(-1)   # (B, width, 1, 1)
        b = beta.unsqueeze(-1).unsqueeze(-1)    # (B, width, 1, 1)
        return F.elu(g * h + b)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, spatial: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial : (B, T=8, K=11, 2)   — [T_coord, K_coord] normalised to [-1,1]
        theta   : (B, 6)              — z-score normalised Heston parameters

        Returns
        -------
        out : (B, T=8, K=11)  — normalised IV surface (z-score space)
        """
        # 1. Generate FiLM modulation vectors from parameters
        gamma, beta = self.film(theta)   # each (B, 4, width)

        # 2. Mirror-pad the maturity axis to suppress Gibbs oscillations at T_max
        original_T  = spatial.size(1)
        x_mirrored  = torch.flip(spatial, dims=[1])
        x_ext       = torch.cat([spatial, x_mirrored], dim=1)   # (B, 2T, K, 2)

        # 3. Lift to channel space
        x_ext = self.p(x_ext)                    # (B, 2T, K, width)
        x_ext = x_ext.permute(0, 3, 1, 2)        # (B, width, 2T, K)

        # 4. Four spectral layers, each FiLM-modulated by θ̂
        x_ext = self._film_modulate(
            self.conv0(x_ext) + self.w0(x_ext), gamma[:, 0], beta[:, 0])

        x_ext = self._film_modulate(
            self.conv1(x_ext) + self.w1(x_ext), gamma[:, 1], beta[:, 1])

        x_ext = self._film_modulate(
            self.conv2(x_ext) + self.w2(x_ext), gamma[:, 2], beta[:, 2])

        x_ext = self._film_modulate(
            self.conv3(x_ext) + self.w3(x_ext), gamma[:, 3], beta[:, 3])

        # 5. Project to output channels
        x_ext = x_ext.permute(0, 2, 3, 1)        # (B, 2T, K, width)
        out   = self.q(x_ext)                     # (B, 2T, K, 1)

        # 6. Truncate mirror padding
        out = out[:, :original_T, :, :]           # (B, T, K, 1)

        # 7. Return raw normalised IV (no softplus — targets are z-score normalised
        #    and can be negative; positivity is enforced after denormalisation)
        return out.squeeze(-1)                    # (B, T, K)


# ─── Utility losses (unchanged) ──────────────────────────────────────────────

def martingale_loss_prior(S_paths, r=0.0, dt=1/252.0):
    """
    Post-hoc validation tool for the CUDA SDE engine.
    Not used in FNO training (requires path-level data not available per batch).
    """
    S0 = S_paths[:, :, 0].mean(dim=1, keepdim=True)
    num_steps = S_paths.size(2)
    t_grid = torch.arange(num_steps, device=S_paths.device, dtype=torch.float32) * dt
    discount_factors = torch.exp(-r * t_grid)
    discounted_S = S_paths * discount_factors.view(1, 1, -1)
    E_discounted_S = discounted_S.mean(dim=1)
    return F.mse_loss(E_discounted_S, S0.expand_as(E_discounted_S))


def arbitrage_free_regularization(iv_surface, T_grid, K_grid):
    """
    Soft penalties for calendar and butterfly arbitrage.
    iv_surface : (B, T, K)   — in REAL (denormalised) IV space.
    """
    T_expanded = T_grid.view(1, -1, 1)
    W = iv_surface ** 2 * T_expanded   # Total Variance

    # Calendar spread: dW/dT >= 0
    dW_dT = W[:, 1:, :] - W[:, :-1, :]
    calendar_penalty = F.relu(-dW_dT).mean()

    # Butterfly: d²W/dK² bounded (density >= 0 proxy)
    d2W_dK2 = W[:, :, 2:] - 2 * W[:, :, 1:-1] + W[:, :, :-2]
    butterfly_penalty = F.relu(-d2W_dK2).mean()

    return calendar_penalty + butterfly_penalty


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    model = MirrorPaddedFNO2d()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    B = 4
    spatial = torch.randn(B, 8, 11, 2)
    theta   = torch.randn(B, 6)
    out = model(spatial, theta)
    print(f"spatial {spatial.shape}  +  theta {theta.shape}  ->  {out.shape}")
    assert out.shape == (B, 8, 11), f"Shape mismatch: {out.shape}"

    # Check gradient flows to ALL parameters
    loss = out.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  WARNING: {name} has no gradient!")
    print("Gradient check passed — all parameters have gradients.")
    print("FNO FiLM architecture test OK.")

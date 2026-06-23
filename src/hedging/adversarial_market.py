"""
adversarial_market.py — Stylized Facts Alignment GAN (SFAG) and Minimax Robust Deep Hedging.
Generates joint stock return and volatility paths, optimizing them against a hedging policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional
from .deep_hedging import DeepHedgingEnv, estimate_gpd_tail_index_pwm, compute_acf_loss, compute_leverage_loss, compute_cfvc_loss


class WGAN_GP_Generator(nn.Module):
    """
    Generator network producing joint log-return and volatility paths.
    Output channel 0: Log-returns of the stock.
    Output channel 1: Volatility proxy process.
    """
    def __init__(self, latent_dim: int = 100, seq_len: int = 252, hidden_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Linear projection layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, seq_len * 2)  # 2 channels: returns and volatility
        )
        
        # 1D Convolution refinement blocks
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 2, kernel_size=5, padding=2)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Latent noise of shape (batch_size, latent_dim)
        
        Returns:
            paths: Tensor of shape (batch_size, 2, seq_len)
        """
        x = self.fc(z)  # (batch_size, seq_len * 2)
        x = x.view(x.shape[0], 2, self.seq_len)  # (batch_size, 2, seq_len)
        paths = self.conv(x)  # (batch_size, 2, seq_len)
        
        # Apply physical constraints:
        # Returns: unconstrained
        # Volatility: positive, constrained via softplus to avoid collapse
        ret = paths[:, 0, :]
        vol = torch.clamp(torch.nn.functional.softplus(paths[:, 1, :]), min=1e-4, max=2.0)
        
        return torch.stack([ret, vol], dim=1)


class WGAN_GP_Discriminator(nn.Module):
    """
    Discriminator network scoring realism of returns and volatility paths.
    """
    def __init__(self, seq_len: int = 252, hidden_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        
        # 1D CNN classifier
        self.conv = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate flattened dimension
        dummy = torch.zeros(1, 2, seq_len)
        dummy_out = self.conv(dummy)
        flat_dim = dummy_out.numel()
        
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Path tensor of shape (batch_size, 2, seq_len)
        
        Returns:
            scores: Real values of shape (batch_size, 1)
        """
        features = self.conv(x)
        features = features.view(features.shape[0], -1)
        return self.fc(features)


class StylizedFactsAlignmentGAN:
    """
    Coordinator managing the training of the Stylized Facts Alignment GAN (SFAG)
    and robust minimax deep hedging.
    """
    def __init__(
        self,
        generator: WGAN_GP_Generator,
        discriminator: WGAN_GP_Discriminator,
        latent_dim: int = 100,
        lambda_gp: float = 10.0,
        weights: List[float] = [1.0, 1.0, 1.0, 1.0]  # [GPD, ACF, Leverage, CFVC]
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        
        self.w_gpd = weights[0]
        self.w_acf = weights[1]
        self.w_lev = weights[2]
        self.w_cfvc = weights[3]

    def compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Computes 1-GP gradient penalty for WGAN-GP stability.
        """
        batch_size = real_samples.shape[0]
        device = real_samples.device
        dtype = real_samples.dtype
        
        # Random interpolation factor
        epsilon = torch.rand(batch_size, 1, 1, device=device, dtype=dtype)
        interpolates = epsilon * real_samples + (1.0 - epsilon) * fake_samples
        interpolates.requires_grad_(True)
        
        # Discriminator scores of interpolates
        d_interpolates = self.discriminator(interpolates)
        
        # Gradients of scores with respect to interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=-1)
        gp = torch.mean((gradient_norm - 1.0) ** 2)
        return gp

    def compute_stylized_fact_losses(
        self,
        fake_returns: torch.Tensor,
        real_returns: torch.Tensor,
        real_acf: torch.Tensor,
        real_leverage: float,
        real_cfvc_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the four stylized fact losses: GPD tail gap, ACF gap, Leverage gap, and CFVC matrix gap.
        """
        # A. GPD tail index estimation on positive tail
        # We also compute it on negative returns (negative tail) to capture tail asymmetry
        xi_real_pos = estimate_gpd_tail_index_pwm(real_returns, threshold_quantile=0.90)
        xi_fake_pos = estimate_gpd_tail_index_pwm(fake_returns, threshold_quantile=0.90)
        
        xi_real_neg = estimate_gpd_tail_index_pwm(-real_returns, threshold_quantile=0.90)
        xi_fake_neg = estimate_gpd_tail_index_pwm(-fake_returns, threshold_quantile=0.90)
        
        loss_gpd = torch.mean(torch.abs(xi_real_pos - xi_fake_pos)) + torch.mean(torch.abs(xi_real_neg - xi_fake_neg))
        
        # B. Autocorrelation (ACF) of absolute returns
        loss_acf = compute_acf_loss(fake_returns, real_acf)
        
        # C. Leverage Correlation
        loss_lev = compute_leverage_loss(fake_returns, real_leverage)
        
        # D. Coarse-to-Fine Volatility Correlation (CFVC)
        loss_cfvc = compute_cfvc_loss(fake_returns, real_cfvc_matrix)
        
        return loss_gpd, loss_acf, loss_lev, loss_cfvc


def convert_returns_to_prices(returns: torch.Tensor, vol_paths: torch.Tensor, S_0: float = 100.0) -> torch.Tensor:
    """
    Vectorized conversion of log-returns and volatility into price paths.
    The returns are cumulatively summed to build the stock price path.
    
    Returns:
        H: Price paths of shape (N_paths, seq_len + 1, 2)
           H[:, :, 0] is Stock spot price S_t
           H[:, :, 1] is Volatility proxy process
    """
    device = returns.device
    dtype = returns.dtype
    batch_size, seq_len = returns.shape
    
    # Stock price S = S_0 * exp(cumsum(returns))
    S = S_0 * torch.exp(torch.cumsum(returns, dim=-1))
    
    # Prepend S_0 to make paths start at t_0
    S_0_tensor = torch.full((batch_size, 1), S_0, device=device, dtype=dtype)
    S_full = torch.cat([S_0_tensor, S], dim=-1)  # (batch_size, seq_len + 1)
    
    # Prepend initial volatility V_0 to the vol paths
    V_0_tensor = vol_paths[:, 0:1]
    V_full = torch.cat([V_0_tensor, vol_paths], dim=-1)  # (batch_size, seq_len + 1)
    
    # Stack stock and vol into a single instrument paths tensor
    H = torch.stack([S_full, V_full], dim=-1)  # (batch_size, seq_len + 1, 2)
    return H


def train_robust_minimax_hedger(
    real_returns: torch.Tensor,
    real_acf: torch.Tensor,
    real_leverage: float,
    real_cfvc_matrix: torch.Tensor,
    generator: WGAN_GP_Generator,
    discriminator: WGAN_GP_Discriminator,
    policy: nn.Module,
    epochs: int = 50,
    batch_size: int = 256,
    critic_steps: int = 5,
    minimax_coeff: float = 0.05,  # scale of the adversarial hedging loss in generator
    device: str = "cuda",
    risk_measure: str = "entropic"
):
    """
    Runs the minimax robust deep hedging training loop.
    Alternates:
      1. Train discriminator on real vs generated paths + stylized facts alignment.
      2. Train generator to maximize discriminator score AND maximize hedging error.
      3. Train hedging policy to minimize hedging error on generated paths.
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    policy = policy.to(device)
    
    sfag = StylizedFactsAlignmentGAN(generator, discriminator, latent_dim=generator.latent_dim)
    
    opt_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_p = optim.Adam(policy.parameters(), lr=5e-4)
    
    num_paths = real_returns.shape[0]
    num_batches = num_paths // batch_size
    
    for epoch in range(epochs):
        gen_loss_val = 0.0
        disc_loss_val = 0.0
        hedge_loss_val = 0.0
        
        # Shuffle real returns
        indices = torch.randperm(num_paths, device=device)
        
        for b in range(num_batches):
            # Extract real batch
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            real_ret_batch = real_returns[batch_idx]
            
            # Since real data might not have a second channel, we construct a dummy vol channel
            # of rolling realized standard deviations for discriminator training
            unfolded = real_ret_batch.unfold(dimension=-1, size=5, step=1)
            real_vol_batch = torch.std(unfolded, dim=-1, unbiased=False)
            real_vol_batch = torch.cat([real_vol_batch[:, 0:1].repeat(1, 4), real_vol_batch], dim=-1)
            real_samples = torch.stack([real_ret_batch, real_vol_batch], dim=1)  # (batch_size, 2, seq_len)
            
            # --- 1. TRAIN DISCRIMINATOR ---
            for _ in range(critic_steps):
                opt_d.zero_grad()
                
                # Generate fake paths without tracking generator gradients
                with torch.no_grad():
                    z = torch.randn(batch_size, generator.latent_dim, device=device)
                    fake_samples = generator(z)  # (batch_size, 2, seq_len)
                
                # Score real and fake
                d_real = discriminator(real_samples)
                d_fake = discriminator(fake_samples.detach())
                
                # WGAN Loss with Gradient Penalty
                gp = sfag.compute_gradient_penalty(real_samples, fake_samples.detach())
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + sfag.lambda_gp * gp
                
                d_loss.backward()
                opt_d.step()
                disc_loss_val += d_loss.item() / critic_steps
                
            # --- 2. TRAIN GENERATOR (Minimax Adversary) ---
            opt_g.zero_grad()
            
            # Freeze policy parameters to optimize backpropagation
            policy_grad_states = [p.requires_grad for p in policy.parameters()]
            for p in policy.parameters():
                p.requires_grad = False
                
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_samples = generator(z)
            fake_ret = fake_samples[:, 0, :]
            fake_vol = fake_samples[:, 1, :]
            
            # WGAN generator loss
            d_fake = discriminator(fake_samples)
            g_loss_adv = -torch.mean(d_fake)
            
            # Stylized facts alignment loss
            l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(
                fake_ret, real_ret_batch, real_acf, real_leverage, real_cfvc_matrix
            )
            g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
            
            # Convert generated paths to prices for hedging evaluation
            H = convert_returns_to_prices(fake_ret, fake_vol)
            
            # Evaluate option payoff (e.g. Call option)
            S_T = H[:, -1, 0]
            payoff = torch.clamp(S_T - 100.0, min=0.0)
            
            # Sub-environment for minimax calculation
            # Cost coeffs: 1 bp on stock, 5 bps on vol option
            cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)
            env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0, risk_measure=risk_measure)
            
            # Run simulation with the CURRENT policy
            # (Generator tries to MAXIMIZE this loss to find worst-case paths)
            wealth, _, _ = env.simulate_hedging_episode(policy)
            hedge_loss = env.compute_loss(wealth)
            
            # Minimax loss: generator minimizes WGAN/SF loss and maximizes hedging loss
            # We subtract hedge_loss because G wants to maximize it
            g_total_loss = g_loss_adv + g_loss_sf - minimax_coeff * hedge_loss
            
            g_total_loss.backward()
            opt_g.step()
            gen_loss_val += g_total_loss.item()
            
            # Restore policy parameter requires_grad state
            for p, state in zip(policy.parameters(), policy_grad_states):
                p.requires_grad = state
            
            # --- 3. TRAIN HEDGER POLICY ---
            opt_p.zero_grad()
            
            # We generate fresh paths from the updated generator
            with torch.no_grad():
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_samples = generator(z)
                fake_ret = fake_samples[:, 0, :]
                fake_vol = fake_samples[:, 1, :]
                H = convert_returns_to_prices(fake_ret, fake_vol)
                S_T = H[:, -1, 0]
                payoff = torch.clamp(S_T - 100.0, min=0.0)
                
            env_hedger = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0, risk_measure=risk_measure)
            
            # Optimize policy parameters to MINIMIZE hedging loss
            wealth, _, _ = env_hedger.simulate_hedging_episode(policy)
            p_loss = env_hedger.compute_loss(wealth)
            
            p_loss.backward()
            opt_p.step()
            hedge_loss_val += p_loss.item()
            
        # Print progress every epoch to monitor training speed and metrics dynamically
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Disc: {disc_loss_val/num_batches:.4f} | Gen: {gen_loss_val/num_batches:.4f} | Hedge: {hedge_loss_val/num_batches:.4f}")
            
    print("Minimax Deep Hedging Training COMPLETE.")

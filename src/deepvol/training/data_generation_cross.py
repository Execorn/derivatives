"""src/deepvol/training/data_generation_cross.py

13-D Scrambled Sobol quasi-Monte Carlo sampler for triangulated FX/Commodity cross pricing.
Generates certified, non-arbitraged market states and standardized moneyness strikes.
Supports state-level disjoint train/val/test splits and stress holdouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    pillar_strikes,
)
from deepvol.market.svi_slice import (
    RawSVI,
    certify_slice,
    fit_raw_svi_5pt,
)


@dataclass(frozen=True)
class CrossDomain:
    """Hypercube domain boundaries for market state parameters.
    
    Ranges represent liquid G10 and high-volatility emerging market pairs.
    """
    sqrtT: Tuple[float, float] = (math.sqrt(1.0 / 365.0), math.sqrt(5.0))
    rho: Tuple[float, float] = (-0.95, 0.95)
    atm1: Tuple[float, float] = (0.05, 0.60)
    atm2: Tuple[float, float] = (0.03, 1.20)
    rr25: Tuple[float, float] = (-0.25, 0.25)
    bf25: Tuple[float, float] = (0.001, 0.15)
    rr10_over_rr25: Tuple[float, float] = (1.5, 2.3)
    bf10_over_bf25: Tuple[float, float] = (2.5, 4.5)
    z: Tuple[float, float] = (-7.5, 7.5)

    # Rates are not network inputs (excluded per §1.7 / Audit E4).
    # Sourced from FRED series SOFR/DFF and long-run RUB policy rate history:
    r_usd: Tuple[float, float] = (0.01, 0.06)
    r_rub: Tuple[float, float] = (0.05, 0.25)
    r_xau: Tuple[float, float] = (0.001, 0.025)


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for Sobol market state and moneyness sampler."""
    log2_states: int = 13          # 2^13 = 8192 market states
    strikes_per_state: int = 64    # 56 Beta(2,2) QMC + 8 fixed taper strikes
    seed: int = 20261011
    domain: CrossDomain = CrossDomain()


def beta22_quantile_exact(u: Tensor) -> Tensor:
    """Evaluates the exact Beta(2,2) quantile function via closed-form depressed cubic roots.
    
    For X ~ Beta(2,2), the CDF is F(x) = 3x^2 - 2x^3. Inverting 2x^3 - 3x^2 + u = 0 yields:
        x = 0.5 + sin((1/3) * asin(2u - 1))
    Centered standardized mapping:
        2x - 1 = 2 * sin((1/3) * asin(2u - 1))
    
    Args:
        u: Uniform quasirandom values in [0, 1].
        
    Returns:
        Mapped centered quantile in [-1.0, 1.0].
    """
    arg = torch.clamp(2.0 * u - 1.0, min=-1.0 + 1e-15, max=1.0 - 1e-15)
    return 2.0 * torch.sin(torch.asin(arg) / 3.0)


class CrossSobolSampler:
    """13-D Scrambled Sobol sampler for synthetic FX cross options."""

    def __init__(self, cfg: SamplerConfig = SamplerConfig()) -> None:
        self.cfg = cfg
        self._rejection_mask: Optional[Tensor] = None
        self._raw_states: Optional[Tensor] = None
        self._certified_states: Optional[Tensor] = None

    def sample_raw_states(self) -> Tensor:
        """Samples 2^log2_states raw market states in R^12 without rejection."""
        n_states = 1 << self.cfg.log2_states
        engine = torch.quasirandom.SobolEngine(dimension=12, scramble=True, seed=self.cfg.seed)
        u = engine.draw(n_states, dtype=torch.float64)

        d = self.cfg.domain
        sqrtT = d.sqrtT[0] + u[:, 0] * (d.sqrtT[1] - d.sqrtT[0])
        rho = d.rho[0] + u[:, 1] * (d.rho[1] - d.rho[0])

        # Leg 1 smile parameters
        atm1 = d.atm1[0] + u[:, 2] * (d.atm1[1] - d.atm1[0])
        rr_max1 = torch.minimum(torch.tensor(d.rr25[1] - 0.01, dtype=torch.float64), 0.28 * atm1)
        rr25_1 = (2.0 * u[:, 3] - 1.0) * rr_max1
        bf25_1 = torch.clamp(atm1 * (0.030 + 0.060 * u[:, 4]), min=d.bf25[0] + 0.001, max=d.bf25[1] - 0.01)
        ratio_rr1 = 1.6 + 0.4 * u[:, 5]
        ratio_bf1 = 2.6 + 0.6 * u[:, 6]
        rr10_1 = rr25_1 * ratio_rr1
        bf10_1 = bf25_1 * ratio_bf1

        # Leg 2 smile parameters
        atm2 = d.atm2[0] + u[:, 7] * (d.atm2[1] - d.atm2[0])
        rr_max2 = torch.minimum(torch.tensor(d.rr25[1] - 0.01, dtype=torch.float64), 0.28 * atm2)
        rr25_2 = (2.0 * u[:, 8] - 1.0) * rr_max2
        bf25_2 = torch.clamp(atm2 * (0.030 + 0.060 * u[:, 9]), min=d.bf25[0] + 0.001, max=d.bf25[1] - 0.01)
        ratio_rr2 = 1.6 + 0.4 * u[:, 10]
        ratio_bf2 = 2.6 + 0.6 * u[:, 11]
        rr10_2 = rr25_2 * ratio_rr2
        bf10_2 = bf25_2 * ratio_bf2

        states = torch.stack([
            sqrtT, rho,
            atm1, rr25_1, bf25_1, rr10_1, bf10_1,
            atm2, rr25_2, bf25_2, rr10_2, bf10_2,
        ], dim=-1)
        self._raw_states = states
        return states

    def filter_and_certify(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """Filters market states through Durrleman butterfly certification and P5 Lee bounds.
        
        Returns:
            Tuple of (certified_states, valid_mask).
        """
        n_states = states.shape[0]
        sqrtT = states[:, 0]
        T = sqrtT ** 2
        rho = states[:, 1]
        quotes1 = states[:, 2:7]
        quotes2 = states[:, 7:12]

        ones = torch.ones(n_states, dtype=torch.float64, device=states.device)
        leg1 = LegMarket(spot=ones, df_foreign=ones, df_domestic=ones, T=T)
        leg2 = LegMarket(spot=ones, df_foreign=ones, df_domestic=ones, T=T)

        # 1. Check Leg 1
        k1, sig1 = pillar_strikes(quotes1, leg1, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
        x1 = torch.log(k1 / leg1.forward.unsqueeze(-1))
        svi1, _ = fit_raw_svi_5pt(x1, sig1 ** 2 * T.unsqueeze(1))
        cert1 = certify_slice(svi1, quotes1[:, 0], T)

        # 2. Check Leg 2
        k2, sig2 = pillar_strikes(quotes2, leg2, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
        x2 = torch.log(k2 / leg2.forward.unsqueeze(-1))
        svi2, _ = fit_raw_svi_5pt(x2, sig2 ** 2 * T.unsqueeze(1))
        cert2 = certify_slice(svi2, quotes2[:, 0], T)

        # 3. Check Proposition P5 cross Lee bound
        b1, rho_s1 = svi1.params[:, 1], svi1.params[:, 2]
        b2, rho_s2 = svi2.params[:, 1], svi2.params[:, 2]
        L1 = b1 * (1.0 + torch.abs(rho_s1))
        L2 = b2 * (1.0 + torch.abs(rho_s2))

        s1_atm = quotes1[:, 0]
        s2_atm = quotes2[:, 0]
        s_cross2 = s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho * s1_atm * s2_atm
        beta1 = (s1_atm ** 2 + rho * s1_atm * s2_atm) / torch.clamp(s_cross2, min=1e-8)
        beta2 = 1.0 - beta1

        L_bar = L1 * torch.abs(beta1) + L2 * torch.abs(beta2) + 2.0 * torch.abs(rho) * torch.sqrt(torch.clamp(L1 * L2 * torch.abs(beta1 * beta2), min=0.0))
        p5_ok = L_bar <= 1.90

        valid_mask = cert1.ok & cert2.ok & p5_ok
        self._rejection_mask = ~valid_mask
        self._certified_states = states[valid_mask]
        return self._certified_states, valid_mask

    def states(self) -> Tensor:
        """Returns certified (N_valid, 12) float64 market states."""
        if self._certified_states is not None:
            return self._certified_states
        raw = self.sample_raw_states()
        certified, _ = self.filter_and_certify(raw)
        return certified

    def strikes(self, n_states: int) -> Tensor:
        """Generates (n_states, 64) standardized moneyness strikes z.
        
        Combines 56 Beta(2,2) QMC strikes with 8 fixed strikes in the taper zone |z| in [5, 7.5].
        """
        taper_strikes = torch.tensor([
            -7.0, -6.333333333333333, -5.666666666666667, -5.0,
            5.0, 5.666666666666667, 6.333333333333333, 7.0
        ], dtype=torch.float64)

        all_strikes = []
        for s_idx in range(n_states):
            state_seed = self.cfg.seed + 10007 * s_idx + 1
            eng1d = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=state_seed)
            u_56 = eng1d.draw(56, dtype=torch.float64).squeeze(-1)
            z_56 = 7.5 * beta22_quantile_exact(u_56)
            z_64 = torch.cat([z_56, taper_strikes], dim=0)
            z_sorted, _ = torch.sort(z_64)
            all_strikes.append(z_sorted)

        return torch.stack(all_strikes, dim=0)

    def split(self, n_states: int) -> Dict[str, Tensor]:
        """Creates state-level disjoint splits: 80% train, 10% val, 10% test, plus stress holdout."""
        certified = self.states()[:n_states]
        atm2 = certified[:, 7]
        rr25_1 = certified[:, 3]
        rr25_2 = certified[:, 8]

        is_stress = (atm2 > 0.8) | (torch.abs(rr25_1) > 0.18) | (torch.abs(rr25_2) > 0.18)
        stress_indices = torch.where(is_stress)[0]
        normal_indices = torch.where(~is_stress)[0]

        n_norm = len(normal_indices)
        n_train = int(0.80 * n_norm)
        n_val = int(0.10 * n_norm)

        train_indices = normal_indices[:n_train]
        val_indices = normal_indices[n_train:n_train + n_val]
        test_indices = normal_indices[n_train + n_val:]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "stress": stress_indices,
        }

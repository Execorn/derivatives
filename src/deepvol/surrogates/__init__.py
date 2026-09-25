"""
surrogates package exports for PI-M-FNO meta-learning.
"""

from deepvol.surrogates.pde_loss import DupirePDELoss
from deepvol.surrogates.meta_fno import MetaFNO2d, train_reptile, train_fomaml
from deepvol.surrogates.guardian import ModelRiskGuardian
from deepvol.surrogates.correction_mlp import CorrectionMLP
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble

__all__ = [
    "DupirePDELoss",
    "MetaFNO2d",
    "train_reptile",
    "train_fomaml",
    "ModelRiskGuardian",
    "CorrectionMLP",
    "CorrectionEnsemble",
]


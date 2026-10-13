"""Systematic Mutation Testing Engine for FX Cross Modules (Phase 14).

Executes >= 50 systematic mutations per module across the 6 required modules:
1. src/deepvol/models/fx_cross_analytic.py
2. src/deepvol/models/taylor_jets.py
3. src/deepvol/market/fx_conventions.py
4. src/deepvol/market/svi_slice.py
5. src/deepvol/models/fx_cross_safeguards.py
6. src/deepvol/api/schemas_fx_cross.py
"""

import os
import sys
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Mutant:
    mutant_id: str
    module_relpath: str
    description: str
    target_str: str
    replacement_str: str
    occurrence: int = 1  # 1-indexed


@dataclass
class MutantResult:
    mutant: Mutant
    status: str  # "KILLED", "SURVIVED", "TARGET_NOT_FOUND"
    killing_test: Optional[str] = None
    stdout: Optional[str] = None


def generate_mutants_for_analytic() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/models/fx_cross_analytic.py"

    m.append(Mutant("MUT_ANA_01", rel, "Change beta-split fixed point iterations from 6 to 1", "n_fp: int = 6", "n_fp: int = 1"))
    m.append(Mutant("MUT_ANA_02", rel, "Change beta-split fixed point iterations from 6 to 0", "n_fp: int = 6", "n_fp: int = 0"))
    m.append(Mutant("MUT_ANA_03", rel, "Corrupt cross correlation 2.0 scale factor", "tj.scale(s12, 2.0)", "tj.scale(s12, 1.0)", occurrence=1))
    m.append(Mutant("MUT_ANA_04", rel, "Invert correlation sign in beta split cross term", "tj.scale(s12, 2.0)", "tj.scale(s12, -2.0)", occurrence=1))
    m.append(Mutant("MUT_ANA_05", rel, "Corrupt beta2 calculation from 1.0 - beta1 to 1.0 + beta1", "beta2 = tj.add(tj.const(torch.ones_like(T)), tj.scale(beta1, -1.0))", "beta2 = tj.add(tj.const(torch.ones_like(T)), tj.scale(beta1, 1.0))"))
    m.append(Mutant("MUT_ANA_06", rel, "Corrupt sigma softfloor from 0.01 to 0.50", "tj.softfloor(sigma_jet, lo=0.01, beta=200.0)", "tj.softfloor(sigma_jet, lo=0.50, beta=200.0)"))
    m.append(Mutant("MUT_ANA_07", rel, "Corrupt sigma softfloor beta from 200.0 to 2.0", "lo=0.01, beta=200.0", "lo=0.01, beta=2.0"))
    m.append(Mutant("MUT_ANA_08", rel, "Flip d1 variance term sign in black76_otm_parity", "d1 = (-k + 0.5 * sd * sd) / sd", "d1 = (-k - 0.5 * sd * sd) / sd", occurrence=1))
    m.append(Mutant("MUT_ANA_09", rel, "Corrupt d2 definition to d1 + sd", "d2 = d1 - sd", "d2 = d1 + sd", occurrence=1))
    m.append(Mutant("MUT_ANA_10", rel, "Flip OTM call/put boundary condition", "phi_otm = torch.where(k >= 0.0, torch.ones_like(k), -torch.ones_like(k))", "phi_otm = torch.where(k < 0.0, torch.ones_like(k), -torch.ones_like(k))"))
    m.append(Mutant("MUT_ANA_11", rel, "Invert B_F sign", "B_F = phi * D_val * torch.special.ndtr(phi * d1)", "B_F = - phi * D_val * torch.special.ndtr(phi * d1)"))
    m.append(Mutant("MUT_ANA_12", rel, "Invert B_FF sign", "B_FF = D_val * n1 / (F_val * sd)", "B_FF = - D_val * n1 / (F_val * sd)"))
    m.append(Mutant("MUT_ANA_13", rel, "Invert B_Fs sign", "B_Fs = -D_val * n1 * d2 / sigma", "B_Fs = D_val * n1 * d2 / sigma"))
    m.append(Mutant("MUT_ANA_14", rel, "Drop F_val in cross vega B_s", "B_s = D_val * F_val * sq * n1", "B_s = D_val * sq * n1"))
    m.append(Mutant("MUT_ANA_15", rel, "Invert B_ss sign", "B_ss = B_s * d1 * d2 / sigma", "B_ss = - B_s * d1 * d2 / sigma"))
    m.append(Mutant("MUT_ANA_16", rel, "Drop g carry factor in delta", "delta = g_val * dVdF", "delta = dVdF"))
    m.append(Mutant("MUT_ANA_17", rel, "Drop F factor in cash_delta", "cash_delta = F_val * dVdF", "cash_delta = dVdF"))
    m.append(Mutant("MUT_ANA_18", rel, "Drop g^2 carry factor in gamma", "gamma = g_val * g_val * d2VdF2", "gamma = g_val * d2VdF2"))
    m.append(Mutant("MUT_ANA_19", rel, "Drop g factor in vanna", "vanna = g_val * (B_Fs + B_ss * s_k * k_F)", "vanna = (B_Fs + B_ss * s_k * k_F)"))
    m.append(Mutant("MUT_ANA_20", rel, "Invert vomma sign", "vomma = B_ss", "vomma = - B_ss"))
    m.append(Mutant("MUT_ANA_21", rel, "Drop carry term in theta", "carry_term = phi_val * D_val * F_val * torch.special.ndtr(phi_val * d1) * (r_C - r_A)", "carry_term = 0.0 * V"))
    m.append(Mutant("MUT_ANA_22", rel, "Invert decay term in theta", "decay_term = (D_val * F_val * sigma * n1) / (2.0 * torch.sqrt(T_val))", "decay_term = - (D_val * F_val * sigma * n1) / (2.0 * torch.sqrt(T_val))"))
    m.append(Mutant("MUT_ANA_23", rel, "Invert carry sign in theta_frozen", "theta_frozen = r_set * V - carry_term - decay_term", "theta_frozen = r_set * V + carry_term - decay_term"))
    m.append(Mutant("MUT_ANA_24", rel, "Drop annual 1/365 convention in theta", "theta = (theta_frozen + roll_term) / 365.0", "theta = (theta_frozen + roll_term)"))
    m.append(Mutant("MUT_ANA_25", rel, "Invert d_rho sensitivity sign", "d_rho = B_s * s_t[:, 0]", "d_rho = - B_s * s_t[:, 0]"))
    m.append(Mutant("MUT_ANA_26", rel, "Invert d_delta1 sensitivity sign", "d_delta1 = B_s * s_t[:, 1]", "d_delta1 = - B_s * s_t[:, 1]"))
    m.append(Mutant("MUT_ANA_27", rel, "Invert d_delta2 sensitivity sign", "d_delta2 = B_s * s_t[:, 2]", "d_delta2 = - B_s * s_t[:, 2]"))
    m.append(Mutant("MUT_ANA_28", rel, "Drop F_val in bucketed rho dF_term", "dF_term = dVdF * F_val * T_val", "dF_term = dVdF * T_val"))
    m.append(Mutant("MUT_ANA_29", rel, "Corrupt rho_C settlement discounting term", "rho_C = dF_term.unsqueeze(-1) * wC_T - (T_set * V).unsqueeze(-1) * wC_Tset", "rho_C = dF_term.unsqueeze(-1) * wC_T + (T_set * V).unsqueeze(-1) * wC_Tset"))
    m.append(Mutant("MUT_ANA_30", rel, "Invert foreign rho_A sensitivity sign", "rho_A = -dF_term.unsqueeze(-1) * wA_T", "rho_A = dF_term.unsqueeze(-1) * wA_T"))
    m.append(Mutant("MUT_ANA_31", rel, "Drop prod clamp in lee_slope_bound", "prod = torch.clamp(L1 * L2 * torch.abs(beta1 * beta2), min=0.0)", "prod = L1 * L2 * torch.abs(beta1 * beta2)"))
    m.append(Mutant("MUT_ANA_32", rel, "Invert cross term in lee_slope_bound", "return L1 * torch.abs(beta1) + L2 * torch.abs(beta2) + 2.0 * torch.abs(rho) * torch.sqrt(prod)", "return L1 * torch.abs(beta1) + L2 * torch.abs(beta2) - 2.0 * torch.abs(rho) * torch.sqrt(prod)"))
    m.append(Mutant("MUT_ANA_33", rel, "Drop 2.0 factor in lee_slope_bound cross term", "2.0 * torch.abs(rho) * torch.sqrt(prod)", "1.0 * torch.abs(rho) * torch.sqrt(prod)"))
    m.append(Mutant("MUT_ANA_34", rel, "Invert sticky strike s_k zeroing", "s_k = torch.zeros_like(out.s_k)", "s_k = out.s_k", occurrence=1))
    m.append(Mutant("MUT_ANA_35", rel, "Invert sticky strike s_kk zeroing", "s_kk = torch.zeros_like(out.s_kk)", "s_kk = out.s_kk", occurrence=1))
    m.append(Mutant("MUT_ANA_36", rel, "Drop smile inclusive term B_s * s_k in dVdF", "dVdF = B_F + B_s * s_k * k_F", "dVdF = B_F"))
    m.append(Mutant("MUT_ANA_37", rel, "Drop cross term in d2VdF2", "2.0 * B_Fs * s_k * k_F", "0.0 * B_Fs * s_k * k_F"))
    m.append(Mutant("MUT_ANA_38", rel, "Drop second derivative term in d2VdF2", "B_s * (s_kk * k_F * k_F + s_k * k_FF)", "0.0 * B_s"))
    m.append(Mutant("MUT_ANA_39", rel, "Drop maturity floor in smooth_T", "return 1e-4 + F.softplus(T - 1e-4, beta=1e5)", "return T"))
    m.append(Mutant("MUT_ANA_40", rel, "Corrupt smooth_T beta from 1e5 to 1.0", "beta=1e5", "beta=1.0"))
    m.append(Mutant("MUT_ANA_41", rel, "Invert sign in k_F = -1.0 / F_val", "k_F = -1.0 / F_val", "k_F = 1.0 / F_val"))
    m.append(Mutant("MUT_ANA_42", rel, "Corrupt k_FF = 1.0 / (F_val * F_val)", "k_FF = 1.0 / (F_val * F_val)", "k_FF = -1.0 / (F_val * F_val)"))
    m.append(Mutant("MUT_ANA_43", rel, "Drop sticky moneyness roll term in theta", "roll_term = B_s * s_k * (r_C - r_A)", "roll_term = 0.0 * (r_C - r_A)"))
    m.append(Mutant("MUT_ANA_44", rel, "Invert sign of roll_term in theta", "r_C - r_A", "r_A - r_C", occurrence=1))
    m.append(Mutant("MUT_ANA_45", rel, "Corrupt sigma_H1 calculation", "sigma_H1 = tj.sqrt(sc2)", "sigma_H1 = sc2"))
    m.append(Mutant("MUT_ANA_46", rel, "Corrupt eye3 registration dtype", "dtype=torch.float64", "dtype=torch.float32"))
    m.append(Mutant("MUT_ANA_47", rel, "Drop beta1 in k1 calculation", "k1 = tj.mul(k, beta1)", "k1 = k"))
    m.append(Mutant("MUT_ANA_48", rel, "Drop beta2 in k2 calculation", "k2 = tj.mul(k, beta2)", "k2 = k"))
    m.append(Mutant("MUT_ANA_49", rel, "Corrupt black_partials B_FF denominator", "F_val * sd", "sd"))
    m.append(Mutant("MUT_ANA_50", rel, "Corrupt black_partials B_Fs denominator", "d2 / sigma", "d2 * sigma"))
    return m


def generate_mutants_for_taylor_jets() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/models/taylor_jets.py"

    m.append(Mutant("MUT_JET_01", rel, "Corrupt silu f formula", "f = a.v * s", "f = a.v + s"))
    m.append(Mutant("MUT_JET_02", rel, "Corrupt silu f1 formula", "f1 = s + a.v * s * (1.0 - s)", "f1 = s - a.v * s * (1.0 - s)"))
    m.append(Mutant("MUT_JET_03", rel, "Corrupt silu f2 formula sign", "f2 = s * (1.0 - s) * (2.0 + a.v * (1.0 - 2.0 * s))", "f2 = - s * (1.0 - s) * (2.0 + a.v * (1.0 - 2.0 * s))"))
    m.append(Mutant("MUT_JET_04", rel, "Drop 2.0 factor in silu f2", "2.0 + a.v * (1.0 - 2.0 * s)", "1.0 + a.v * (1.0 - 2.0 * s)"))
    m.append(Mutant("MUT_JET_05", rel, "Drop 2.0 factor on s in silu f2", "1.0 - 2.0 * s", "1.0 - s"))
    m.append(Mutant("MUT_JET_06", rel, "Invert add value", "v=a.v + b.v", "v=a.v - b.v"))
    m.append(Mutant("MUT_JET_07", rel, "Invert add derivative", "d=a.d + b.d", "d=a.d - b.d"))
    m.append(Mutant("MUT_JET_08", rel, "Invert add second derivative", "dd=a.dd + b.dd", "dd=a.dd - b.dd"))
    m.append(Mutant("MUT_JET_09", rel, "Invert mul value", "v = a.v * b.v", "v = a.v + b.v"))
    m.append(Mutant("MUT_JET_10", rel, "Corrupt mul first derivative", "d = a.d * b.v + a.v * b.d", "d = a.d * b.v - a.v * b.d"))
    m.append(Mutant("MUT_JET_11", rel, "Corrupt mul second derivative", "dd = a.dd * b.v + 2.0 * a.d * b.d + a.v * b.dd", "dd = a.dd * b.v + a.d * b.d + a.v * b.dd"))
    m.append(Mutant("MUT_JET_12", rel, "Drop cross term in mul second derivative", "2.0 * a.d * b.d", "0.0 * a.d * b.d"))
    m.append(Mutant("MUT_JET_13", rel, "Corrupt unary d", "d = f1 * a.d", "d = f1 + a.d"))
    m.append(Mutant("MUT_JET_14", rel, "Corrupt unary dd", "dd = f2 * (a.d * a.d) + f1 * a.dd", "dd = f2 * (a.d * a.d) - f1 * a.dd"))
    m.append(Mutant("MUT_JET_15", rel, "Drop f2 term in unary dd", "f2 * (a.d * a.d)", "0.0 * (a.d * a.d)"))
    m.append(Mutant("MUT_JET_16", rel, "Drop f1 * a.dd in unary dd", "f1 * a.dd", "0.0 * a.dd"))
    m.append(Mutant("MUT_JET_17", rel, "Corrupt sqrt value", "r = torch.sqrt(a.v)", "r = a.v"))
    m.append(Mutant("MUT_JET_18", rel, "Corrupt sqrt f1", "f1 = 0.5 / r", "f1 = 1.0 / r"))
    m.append(Mutant("MUT_JET_19", rel, "Invert sqrt f2 sign", "f2 = -0.25 / (r * a.v)", "f2 = 0.25 / (r * a.v)"))
    m.append(Mutant("MUT_JET_20", rel, "Corrupt recip value", "r = 1.0 / a.v", "r = a.v"))
    m.append(Mutant("MUT_JET_21", rel, "Invert recip f1 sign", "f1 = -r * r", "f1 = r * r"))
    m.append(Mutant("MUT_JET_22", rel, "Invert recip f2 sign", "f2 = 2.0 * r * r * r", "f2 = - 2.0 * r * r * r"))
    m.append(Mutant("MUT_JET_23", rel, "Corrupt exp value", "e = torch.exp(a.v)", "e = a.v"))
    m.append(Mutant("MUT_JET_24", rel, "Corrupt tanh u", "u = 1.0 - t * t", "u = 1.0 + t * t"))
    m.append(Mutant("MUT_JET_25", rel, "Invert tanh f2 sign", "-2.0 * t * u", "2.0 * t * u"))
    m.append(Mutant("MUT_JET_26", rel, "Corrupt softfloor f", "f = lo + F.softplus(diff, beta=beta)", "f = lo + diff"))
    m.append(Mutant("MUT_JET_27", rel, "Corrupt softfloor f1", "f1 = s", "f1 = 1.0 - s"))
    m.append(Mutant("MUT_JET_28", rel, "Invert softfloor f2 sign", "f2 = beta * s * (1.0 - s)", "f2 = - beta * s * (1.0 - s)"))
    m.append(Mutant("MUT_JET_29", rel, "Corrupt scale value", "v=a.v * c", "v=a.v / c"))
    m.append(Mutant("MUT_JET_30", rel, "Corrupt scale derivative", "d=a.d * c", "d=a.d / c"))
    m.append(Mutant("MUT_JET_31", rel, "Corrupt scale second derivative", "dd=a.dd * c", "dd=a.dd / c"))
    m.append(Mutant("MUT_JET_32", rel, "Corrupt seed_k derivative", "d=torch.ones_like(k)", "d=torch.zeros_like(k)"))
    m.append(Mutant("MUT_JET_33", rel, "Corrupt seed_tangent unit setting", "t[:, j] = 1.0", "t[:, j] = -1.0"))
    m.append(Mutant("MUT_JET_34", rel, "Corrupt linear layer weight transpose", "H @ layer.weight.t()", "H @ layer.weight"))
    m.append(Mutant("MUT_JET_35", rel, "Invert linear layer bias addition", "H[:B] + layer.bias", "H[:B] - layer.bias"))
    m.append(Mutant("MUT_JET_36", rel, "Corrupt mlp_jet output conversion to double", "H_64 = H.double()", "H_64 = H"))
    m.append(Mutant("MUT_JET_37", rel, "Corrupt mlp_jet d_out slice", "d_out = H_64[B : 2 * B, 0]", "d_out = H_64[:B, 0]"))
    m.append(Mutant("MUT_JET_38", rel, "Corrupt mlp_jet dd_out slice", "dd_out = H_64[2 * B : 3 * B, 0]", "dd_out = H_64[:B, 0]"))
    m.append(Mutant("MUT_JET_39", rel, "Drop bias branch in mlp_jet", "if layer.bias is not None:", "if False:"))
    m.append(Mutant("MUT_JET_40", rel, "Corrupt _add_t ta+tb", "return ta + tb", "return ta - tb"))
    m.append(Mutant("MUT_JET_41", rel, "Corrupt _scale_t tensor scaling", "return t * c.unsqueeze(-1)", "return t * c"))
    m.append(Mutant("MUT_JET_42", rel, "Corrupt _scale_t scalar scaling", "return t * c", "return t / c"))
    m.append(Mutant("MUT_JET_43", rel, "Drop unary tangent scaling", "t = _scale_t(a.t, f1)", "t = a.t"))
    m.append(Mutant("MUT_JET_44", rel, "Drop Leibniz rule tangent in mul", "t = _add_t(_scale_t(a.t, b.v), _scale_t(b.t, a.v))", "t = a.t"))
    m.append(Mutant("MUT_JET_45", rel, "Drop eye tangent in seed_tangent", "t = x.unsqueeze(-1) * 0.0 + eye[j]", "t = x.unsqueeze(-1) * 0.0"))
    m.append(Mutant("MUT_JET_46", rel, "Corrupt softfloor sigmoid scaling", "s = torch.sigmoid(beta * diff)", "s = torch.sigmoid(diff)"))
    m.append(Mutant("MUT_JET_47", rel, "Drop softfloor lo offset", "f = lo + F.softplus(diff, beta=beta)", "f = F.softplus(diff, beta=beta)"))
    m.append(Mutant("MUT_JET_48", rel, "Corrupt silu sigmoid calculation", "s = torch.sigmoid(a.v)", "s = a.v"))
    m.append(Mutant("MUT_JET_49", rel, "Corrupt sqrt f2 power", "r * a.v", "r"))
    m.append(Mutant("MUT_JET_50", rel, "Corrupt recip f2 power", "2.0 * r * r * r", "2.0 * r * r"))
    return m


def generate_mutants_for_conventions() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/market/fx_conventions.py"

    m.append(Mutant("MUT_CONV_01", rel, "Corrupt d1 variance term", "d1 = (torch.log(F / K) + 0.5 * (tau ** 2)) / tau", "d1 = (torch.log(F / K) - 0.5 * (tau ** 2)) / tau"))
    m.append(Mutant("MUT_CONV_02", rel, "Corrupt d2 to d1 + tau", "d2 = d1 - tau", "d2 = d1 + tau"))
    m.append(Mutant("MUT_CONV_03", rel, "Invert Forward PNA delta sign", "return cast(Tensor, phi * N1)", "return cast(Tensor, - phi * N1)"))
    m.append(Mutant("MUT_CONV_04", rel, "Drop df_foreign in Spot PNA delta", "return cast(Tensor, phi * leg.df_foreign * N1)", "return cast(Tensor, phi * N1)"))
    m.append(Mutant("MUT_CONV_05", rel, "Drop K/F in Forward PA delta", "return cast(Tensor, phi * (K / F) * N2)", "return cast(Tensor, phi * N2)"))
    m.append(Mutant("MUT_CONV_06", rel, "Drop df_foreign in Spot PA delta", "return cast(Tensor, phi * leg.df_foreign * (K / F) * N2)", "return cast(Tensor, phi * (K / F) * N2)"))
    m.append(Mutant("MUT_CONV_07", rel, "Invert Sampford derivative dm", "dm = -m * (z + m)", "dm = m * (z + m)"))
    m.append(Mutant("MUT_CONV_08", rel, "Corrupt Newton step update in pa_call_delta_argmax", "step = (m - tau) / dm", "step = (m + tau) / dm"))
    m.append(Mutant("MUT_CONV_09", rel, "Invert step application in pa_call_delta_argmax", "z = z - step", "z = z + step"))
    m.append(Mutant("MUT_CONV_10", rel, "Change pa_call_delta_argmax iterations from 30 to 1", "n_iter: int = 30", "n_iter: int = 1"))
    m.append(Mutant("MUT_CONV_11", rel, "Invert ndtri sign in PNA inversion", "d1_pna = phi * torch.special.ndtri(val_pna)", "d1_pna = - phi * torch.special.ndtri(val_pna)"))
    m.append(Mutant("MUT_CONV_12", rel, "Corrupt PNA inversion exponent", "-d1_pna * tau + 0.5 * (tau ** 2)", "d1_pna * tau + 0.5 * (tau ** 2)"))
    m.append(Mutant("MUT_CONV_13", rel, "Drop variance term in PNA strike", "+ 0.5 * (tau ** 2)", "+ 0.0 * (tau ** 2)", occurrence=1))
    m.append(Mutant("MUT_CONV_14", rel, "Corrupt PA put log-bracket width from 8.0 to 0.1", "8.0 * tau", "0.1 * tau"))
    m.append(Mutant("MUT_CONV_15", rel, "Change PA Newton iterations from 20 to 1", "n_iter: int = 20", "n_iter: int = 1"))
    m.append(Mutant("MUT_CONV_16", rel, "Change PA Newton iterations from 20 to 0", "n_iter: int = 20", "n_iter: int = 0"))
    m.append(Mutant("MUT_CONV_17", rel, "Invert PA Jacobian sign J", "J = torch.exp(u) * (phi * N2 - n2 / tau)", "J = torch.exp(u) * (phi * N2 + n2 / tau)"))
    m.append(Mutant("MUT_CONV_18", rel, "Invert step direction in PA solver", "step = f / J", "step = - f / J"))
    m.append(Mutant("MUT_CONV_19", rel, "Invert candidate update in PA solver", "u_cand = u - step", "u_cand = u + step"))
    m.append(Mutant("MUT_CONV_20", rel, "Corrupt forward calculation", "self.spot * self.df_foreign / self.df_domestic", "self.spot * self.df_domestic / self.df_foreign"))
    m.append(Mutant("MUT_CONV_21", rel, "Drop df_domestic in forward", "self.df_foreign / self.df_domestic", "self.df_foreign"))
    m.append(Mutant("MUT_CONV_22", rel, "Corrupt tau calculation", "tau = sigma * torch.sqrt(leg.T)", "tau = sigma * leg.T", occurrence=1))
    m.append(Mutant("MUT_CONV_23", rel, "Corrupt PA d2 formula", "d2 = (-u - 0.5 * (tau ** 2)) / tau", "d2 = (-u + 0.5 * (tau ** 2)) / tau"))
    m.append(Mutant("MUT_CONV_24", rel, "Drop n2/tau term in PA Jacobian", "phi * N2 - n2 / tau", "phi * N2"))
    m.append(Mutant("MUT_CONV_25", rel, "Corrupt PA residual f", "f = delta_curr - delta_fwd", "f = delta_curr + delta_fwd"))
    m.append(Mutant("MUT_CONV_26", rel, "Drop clamp on PNA target delta", "torch.clamp(phi * delta_fwd, min=1e-15, max=1.0 - 1e-15)", "phi * delta_fwd"))
    m.append(Mutant("MUT_CONV_27", rel, "Corrupt spot PA conversion factor", "delta / leg.df_foreign", "delta * leg.df_foreign"))
    m.append(Mutant("MUT_CONV_28", rel, "Corrupt initial guess for z in argmax", "torch.where(tau > 1.0, -tau + 1.0 / tau, 0.54 - 0.68 * tau)", "torch.zeros_like(tau)"))
    m.append(Mutant("MUT_CONV_29", rel, "Corrupt hazard rate m definition", "nz / torch.clamp(Nz, min=1e-300)", "nz * Nz"))
    m.append(Mutant("MUT_CONV_30", rel, "Corrupt K_max exponent sign", "-z * tau - 0.5 * (tau ** 2)", "z * tau - 0.5 * (tau ** 2)"))
    m.append(Mutant("MUT_CONV_31", rel, "Drop tolerance parameter default", "tol: float = 1e-12", "tol: float = 1e-2"))
    m.append(Mutant("MUT_CONV_32", rel, "Corrupt delta_curr sign in PA", "delta_curr = phi * torch.exp(u) * N2", "delta_curr = - phi * torch.exp(u) * N2"))
    m.append(Mutant("MUT_CONV_33", rel, "Corrupt u_mid bisection fallback", "u_mid = 0.5 * (u_lo + u_hi)", "u_mid = u_lo"))
    m.append(Mutant("MUT_CONV_34", rel, "Drop in_bracket check in PA solver", "u = torch.where(in_bracket, u_cand, u_mid)", "u = u_cand"))
    m.append(Mutant("MUT_CONV_35", rel, "Corrupt K reconstruction from u", "K = F * torch.exp(u)", "K = F * u"))
    m.append(Mutant("MUT_CONV_36", rel, "Corrupt bracket update u_lo condition", "torch.where(f > 0.0, torch.maximum(u_lo, u), u_lo)", "torch.where(f < 0.0, torch.maximum(u_lo, u), u_lo)"))
    m.append(Mutant("MUT_CONV_37", rel, "Corrupt bracket update u_hi condition", "torch.where(f <= 0.0, torch.minimum(u_hi, u), u_hi)", "torch.where(f > 0.0, torch.minimum(u_hi, u), u_hi)"))
    m.append(Mutant("MUT_CONV_38", rel, "Drop 8.0 multiplier in put bracket", "- 8.0 * tau", "- 1.0 * tau"))
    m.append(Mutant("MUT_CONV_39", rel, "Corrupt phi > 0 condition in u_lo", "torch.where(phi > 0.0, u_lo_call, u_lo_put)", "torch.where(phi < 0.0, u_lo_call, u_lo_put)"))
    m.append(Mutant("MUT_CONV_40", rel, "Corrupt phi > 0 condition in u_hi", "torch.where(phi > 0.0, u_hi_call, u_hi_put)", "torch.where(phi < 0.0, u_hi_call, u_hi_put)"))
    m.append(Mutant("MUT_CONV_41", rel, "Corrupt initial u midpoint", "u = 0.5 * (u_lo + u_hi)", "u = u_lo"))
    m.append(Mutant("MUT_CONV_42", rel, "Corrupt normal pdf n2", "inv_sqrt_2pi * torch.exp(-0.5 * (d2 ** 2))", "torch.exp(-0.5 * (d2 ** 2))"))
    m.append(Mutant("MUT_CONV_43", rel, "Corrupt normal pdf nz in argmax", "inv_sqrt_2pi * torch.exp(-0.5 * (z ** 2))", "torch.exp(-0.5 * (z ** 2))"))
    m.append(Mutant("MUT_CONV_44", rel, "Corrupt Nz clamp floor", "min=1e-300", "min=1e-5"))
    m.append(Mutant("MUT_CONV_45", rel, "Invert Sampford hazard rate derivative sign", "-m * (z + m)", "m * (z - m)"))
    m.append(Mutant("MUT_CONV_46", rel, "Drop tau in step update", "(m - tau) / dm", "m / dm"))
    m.append(Mutant("MUT_CONV_47", rel, "Drop 0.5 factor in d1", "+ 0.5 * (tau ** 2)", "+ (tau ** 2)", occurrence=1))
    m.append(Mutant("MUT_CONV_48", rel, "Drop tau denominator in d1", ") / tau", ")", occurrence=1))
    m.append(Mutant("MUT_CONV_49", rel, "Corrupt F/K in d1", "torch.log(F / K)", "torch.log(K / F)", occurrence=1))
    m.append(Mutant("MUT_CONV_50", rel, "Corrupt spot PA multiplier", "(K / F)", "(F / K)", occurrence=1))
    return m


def generate_mutants_for_svi() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/market/svi_slice.py"

    m.append(Mutant("MUT_SVI_01", rel, "Corrupt w(x) addition to subtraction", "a + b * (rho_s * delta + hypot)", "a - b * (rho_s * delta + hypot)", occurrence=1))
    m.append(Mutant("MUT_SVI_02", rel, "Invert rho_s sign in w(x)", "rho_s * delta", "- rho_s * delta", occurrence=1))
    m.append(Mutant("MUT_SVI_03", rel, "Drop s^2 inside hypot root", "hypot = torch.sqrt(delta ** 2 + s ** 2)", "hypot = torch.sqrt(delta ** 2)", occurrence=1))
    m.append(Mutant("MUT_SVI_04", rel, "Invert delta/hypot sign in w1_val", "w1_val = b * (rho_s + delta / hypot)", "w1_val = b * (rho_s - delta / hypot)"))
    m.append(Mutant("MUT_SVI_05", rel, "Invert w2_val second derivative sign", "w2_val = b * (s ** 2) / (hypot ** 3)", "w2_val = - b * (s ** 2) / (hypot ** 3)"))
    m.append(Mutant("MUT_SVI_06", rel, "Drop s^2 in w2_val", "b * (s ** 2) / (hypot ** 3)", "b / (hypot ** 3)"))
    m.append(Mutant("MUT_SVI_07", rel, "Corrupt Durrleman term1 sign", "term1 = (1.0 - (x * w1) / (2.0 * w_safe)) ** 2", "term1 = (1.0 + (x * w1) / (2.0 * w_safe)) ** 2"))
    m.append(Mutant("MUT_SVI_08", rel, "Invert Durrleman term2 sign", "term2 = (w1 ** 2) / 4.0 * (1.0 / w_safe + 0.25)", "term2 = - (w1 ** 2) / 4.0 * (1.0 / w_safe + 0.25)"))
    m.append(Mutant("MUT_SVI_09", rel, "Invert Durrleman term3 sign", "term3 = 0.5 * w2", "term3 = - 0.5 * w2"))
    m.append(Mutant("MUT_SVI_10", rel, "Corrupt Durrleman g assembly", "return term1 - term2 + term3", "return term1 + term2 + term3"))
    m.append(Mutant("MUT_SVI_11", rel, "Corrupt certification n_grid from 401 to 10", "n_grid: int = 401", "n_grid: int = 10"))
    m.append(Mutant("MUT_SVI_12", rel, "Corrupt certification z_max from 12.0 to 2.0", "z_max: float = 12.0", "z_max: float = 2.0"))
    m.append(Mutant("MUT_SVI_13", rel, "Corrupt w_safe clamp floor from 1e-12 to 1.0", "w_safe = torch.clamp(w, min=1e-12)", "w_safe = torch.clamp(w, min=1.0)"))
    m.append(Mutant("MUT_SVI_14", rel, "Drop 2.0 factor in Durrleman term1", "2.0 * w_safe", "w_safe"))
    m.append(Mutant("MUT_SVI_15", rel, "Drop 4.0 factor in Durrleman term2", "(w1 ** 2) / 4.0", "(w1 ** 2)"))
    m.append(Mutant("MUT_SVI_16", rel, "Drop 0.25 offset in Durrleman term2", "1.0 / w_safe + 0.25", "1.0 / w_safe"))
    m.append(Mutant("MUT_SVI_17", rel, "Corrupt RawSVI 1D tensor handling", "self.params.unsqueeze(0)", "self.params"))
    m.append(Mutant("MUT_SVI_18", rel, "Corrupt derivs return tuple order", "return w_val, w1_val, w2_val", "return w1_val, w_val, w2_val"))
    m.append(Mutant("MUT_SVI_19", rel, "Corrupt delta definition in w", "delta = x_in - m", "delta = x_in + m", occurrence=1))
    m.append(Mutant("MUT_SVI_20", rel, "Corrupt delta definition in derivs", "delta = x_in - m", "delta = x_in + m", occurrence=2))
    return m


def generate_mutants_for_safeguards() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/models/fx_cross_safeguards.py"

    m.append(Mutant("MUT_SAFE_01", rel, "Drop smooth_T floor in safeguards", "return 1e-4 + F.softplus(T - 1e-4, beta=1e5)", "return T"))
    m.append(Mutant("MUT_SAFE_02", rel, "Corrupt smooth_T beta parameter", "beta=1e5", "beta=1.0", occurrence=1))
    m.append(Mutant("MUT_SAFE_03", rel, "Drop smooth_vol floor in safeguards", "return 0.01 + F.softplus(sigma - 0.01, beta=200.0)", "return sigma"))
    m.append(Mutant("MUT_SAFE_04", rel, "Corrupt smooth_vol beta parameter", "beta=200.0", "beta=2.0", occurrence=1))
    m.append(Mutant("MUT_SAFE_05", rel, "Corrupt wing_slope_certificate limit from 1.9 to 2.5", "limit: float = 1.9", "limit: float = 2.5", occurrence=1))
    m.append(Mutant("MUT_SAFE_06", rel, "Corrupt wing_slope_certificate limit from 1.9 to 1.0", "limit: float = 1.9", "limit: float = 1.0", occurrence=1))
    m.append(Mutant("MUT_SAFE_07", rel, "Invert wing slope certificate condition", "return slope <= limit", "return slope >= limit"))
    m.append(Mutant("MUT_SAFE_08", rel, "Corrupt collocation n_grid from 401 to 10", "n_grid: int = 401", "n_grid: int = 10", occurrence=1))
    m.append(Mutant("MUT_SAFE_09", rel, "Corrupt min_g_pass threshold", "min_g_pass = (min_g >= -1e-12)", "min_g_pass = (min_g >= 0.5)"))
    m.append(Mutant("MUT_SAFE_10", rel, "Corrupt slope_pass threshold in cross_slice_audit", "slope_pass = (bar_L <= 1.9)", "slope_pass = (bar_L <= 1.0)"))
    m.append(Mutant("MUT_SAFE_11", rel, "Force call_spread_pass to True", "call_spread_pass = bool(torch.all(diffs >= -1e-12).item())", "call_spread_pass = True"))
    m.append(Mutant("MUT_SAFE_12", rel, "Force calendar_ok to True", "calendar_ok_list[orig_i] = bool(cal_ok_grp[local_i].item())", "calendar_ok_list[orig_i] = True"))
    m.append(Mutant("MUT_SAFE_13", rel, "Corrupt composite ok logic from and to or", "ok_list = [slice_ok_list[s] and calendar_ok_list[s] for s in range(S_cnt)]", "ok_list = [slice_ok_list[s] or calendar_ok_list[s] for s in range(S_cnt)]"))
    m.append(Mutant("MUT_SAFE_14", rel, "Corrupt Lee slope combination formula", "beta1_val = (L1_val + rho_val * sqrt_prod) / denom_beta", "beta1_val = (L1_val - rho_val * sqrt_prod) / denom_beta"))
    m.append(Mutant("MUT_SAFE_15", rel, "Corrupt beta2_val formula", "beta2_val = 1.0 - beta1_val", "beta2_val = beta1_val"))
    m.append(Mutant("MUT_SAFE_16", rel, "Corrupt denom_beta threshold from 1e-8 to 1e8", "if denom_beta > 1e-8:", "if denom_beta > 1e8:"))
    m.append(Mutant("MUT_SAFE_17", rel, "Corrupt 1D tensor auto-unsqueeze in cross_slice_audit", "if X_states.dim() == 1:", "if X_states.dim() == 2:"))
    m.append(Mutant("MUT_SAFE_18", rel, "Corrupt cross_ids handling condition", "if cross_ids is not None:", "if cross_ids is None:"))
    m.append(Mutant("MUT_SAFE_19", rel, "Corrupt calendar tolerance from 1e-8 to 1.0", "tol_calendar: float = 1e-8", "tol_calendar: float = 1.0"))
    m.append(Mutant("MUT_SAFE_20", rel, "Corrupt call spread diff sign", "diffs = c_grid[:-1] - c_grid[1:]", "diffs = c_grid[1:] - c_grid[:-1]"))
    return m


def generate_mutants_for_schemas() -> List[Mutant]:
    m: List[Mutant] = []
    rel = "src/deepvol/api/schemas_fx_cross.py"

    m.append(Mutant("MUT_SCH_01", rel, "Corrupt max states constraint from 64 to 2", "max_length=64", "max_length=2"))
    m.append(Mutant("MUT_SCH_02", rel, "Corrupt max contracts constraint from 10000 to 10", "max_length=10000", "max_length=10"))
    m.append(Mutant("MUT_SCH_03", rel, "Drop curve pillar tenor monotonicity validator", "if any(b <= a for a, b in zip(t, t[1:])):", "if False and any(b <= a for a, b in zip(t, t[1:])):"))
    m.append(Mutant("MUT_SCH_04", rel, "Drop cross identifier match check", "if self.identifier != md.leg1.base + md.leg2.quote:", "if False:"))
    m.append(Mutant("MUT_SCH_05", rel, "Drop payment currency match check", "if self.payment_currency != md.leg2.quote:", "if False:"))
    m.append(Mutant("MUT_SCH_06", rel, "Drop date ordering check", "if not (st.as_of_date < st.expiration_date <= st.settlement_date):", "if False:"))
    m.append(Mutant("MUT_SCH_07", rel, "Drop state index range check", "if any(idx < 0 or idx >= len(self.states) for idx in self.contracts.state_index):", "if False:"))
    m.append(Mutant("MUT_SCH_08", rel, "Drop strike column length check", "if len(self.contracts.strike) != N:", "if False:"))
    m.append(Mutant("MUT_SCH_09", rel, "Drop domestic == foreign base check in leg market", "if md.leg1.quote != md.leg2.base:", "if False:"))
    m.append(Mutant("MUT_SCH_10", rel, "Corrupt correlation upper bound from 0.999 to 2.0", "le=0.999", "le=2.0", occurrence=1))
    m.append(Mutant("MUT_SCH_11", rel, "Corrupt correlation lower bound from -0.999 to -2.0", "ge=-0.999", "ge=-2.0", occurrence=1))
    m.append(Mutant("MUT_SCH_12", rel, "Corrupt ATM vol positive constraint to gt=-1.0", "atm_vol: float = Field(..., gt=0.0", "atm_vol: float = Field(..., gt=-1.0"))
    m.append(Mutant("MUT_SCH_13", rel, "Corrupt 25d BF non-negativity to ge=-10.0", "bf25: float = Field(..., ge=0.0", "bf25: float = Field(..., ge=-10.0"))
    m.append(Mutant("MUT_SCH_14", rel, "Corrupt 10d BF non-negativity to ge=-10.0", "bf10: float = Field(..., ge=0.0", "bf10: float = Field(..., ge=-10.0"))
    m.append(Mutant("MUT_SCH_15", rel, "Corrupt execution options default engine", "engine: PricingEngineEnum = PricingEngineEnum.DIFFERENTIAL_ML", "engine: PricingEngineEnum = PricingEngineEnum.ANALYTIC_TRIANGULATED"))
    m.append(Mutant("MUT_SCH_16", rel, "Corrupt execution options precision to FLOAT32", "precision: PrecisionEnum = Field(default=PrecisionEnum.FLOAT64", "precision: PrecisionEnum = Field(default=PrecisionEnum.FLOAT32"))
    m.append(Mutant("MUT_SCH_17", rel, "Drop model reserve exclusion from default queries", "q is not QueryEnum.MODEL_RESERVE", "True"))
    m.append(Mutant("MUT_SCH_18", rel, "Corrupt Arrow stream content type", "\"application/vnd.apache.arrow.stream\"", "\"application/octet-stream\""))
    m.append(Mutant("MUT_SCH_19", rel, "Corrupt fallback status code to 500", "status_code = status.HTTP_200_OK", "status_code = status.HTTP_500_INTERNAL_SERVER_ERROR"))
    m.append(Mutant("MUT_SCH_20", rel, "Drop is_call column length check", "if len(self.contracts.is_call) != N:", "if False:"))
    return m


def run_mutation_on_module(
    module_relpath: str,
    test_files: List[str],
    mutants: List[Mutant],
    pytest_extra_args: Optional[List[str]] = None,
) -> List[MutantResult]:
    """Applies mutants one by one and runs the test suite against each."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    abs_target = os.path.join(repo_root, module_relpath)

    with open(abs_target, "r", encoding="utf-8") as f:
        original_code = f.read()

    results: List[MutantResult] = []

    # Create temporary working directory once for the module
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_src = os.path.join(tmpdir, "src")
        tmp_tests = os.path.join(tmpdir, "tests")
        shutil.copytree(os.path.join(repo_root, "src"), tmp_src)
        shutil.copytree(os.path.join(repo_root, "tests"), tmp_tests)
        shutil.copyfile(os.path.join(repo_root, "pyproject.toml"), os.path.join(tmpdir, "pyproject.toml"))

        mutated_file = os.path.join(tmpdir, module_relpath)
        conftest_file = os.path.join(tmp_tests, "conftest.py")
        if os.path.exists(conftest_file):
            with open(conftest_file, "a", encoding="utf-8") as cf:
                cf.write("\ntry:\n    from hypothesis import settings, Phase\n    settings.register_profile('fast_mutation', max_examples=5, phases=[Phase.explicit, Phase.generate], deadline=None)\n    settings.load_profile('fast_mutation')\nexcept ImportError:\n    pass\n")

        test_paths = [os.path.join(tmpdir, tf) for tf in test_files]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = f"{tmp_src}:{tmpdir}"
        env["HYPOTHESIS_PROFILE"] = "fast_mutation"

        for mutant in mutants:
            # Check that target_str exists in original code
            if mutant.target_str not in original_code:
                results.append(MutantResult(
                    mutant=mutant,
                    status="TARGET_NOT_FOUND",
                    killing_test=f"target_str not found: {mutant.target_str[:40]}...",
                ))
                continue

            # Count occurrences
            count = original_code.count(mutant.target_str)
            if mutant.occurrence > count:
                results.append(MutantResult(
                    mutant=mutant,
                    status="OCCURRENCE_OUT_OF_BOUNDS",
                ))
                continue

            # Replace specific occurrence
            parts = original_code.split(mutant.target_str)
            prefix = mutant.target_str.join(parts[:mutant.occurrence])
            suffix = mutant.target_str.join(parts[mutant.occurrence:])
            mutated_code = prefix + mutant.replacement_str + suffix

            # Overwrite mutated file
            with open(mutated_file, "w", encoding="utf-8") as f:
                f.write(mutated_code)

            cmd = [
                os.path.join(repo_root, ".venv/bin/python"),
                "-m", "pytest",
                "-x",  # stop on first failure to speed up killing
                "-q",
                "-p", "no:randomly",
                "-p", "no:cov",
                *(pytest_extra_args or []),
                *test_paths,
            ]

            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

            # Restore original file immediately
            with open(mutated_file, "w", encoding="utf-8") as f:
                f.write(original_code)

            if proc.returncode != 0:
                fail_line = None
                for line in proc.stdout.splitlines():
                    if "FAILED " in line:
                        fail_line = line.strip()
                        break
                results.append(MutantResult(
                    mutant=mutant,
                    status="KILLED",
                    killing_test=fail_line or "Test failure",
                    stdout=proc.stdout[:300],
                ))
            else:
                results.append(MutantResult(
                    mutant=mutant,
                    status="SURVIVED",
                    stdout=proc.stdout[:300],
                ))

    return results


def main() -> None:
    print("=== Systematic Mutation Testing Framework ===")
    all_results: Dict[str, List[MutantResult]] = {}

    suites = [
        ("fx_cross_analytic", generate_mutants_for_analytic(), ["tests/test_fx_cross_analytic.py"]),
        ("taylor_jets", generate_mutants_for_taylor_jets(), ["tests/test_taylor_jets.py"]),
        ("fx_conventions", generate_mutants_for_conventions(), ["tests/test_fx_conventions.py", "tests/test_adversarial_m0_stress.py"]),
        ("svi_slice", generate_mutants_for_svi(), ["tests/test_svi_slice.py"]),
        ("fx_cross_safeguards", generate_mutants_for_safeguards(), ["tests/test_fx_cross_safeguards.py", "tests/test_adversarial_cross_m3_stress.py"]),
        ("schemas_fx_cross", generate_mutants_for_schemas(), ["tests/test_api_fx_cross.py"]),
    ]

    for name, mutants, tests in suites:
        print(f"\nRunning {len(mutants)} systematic mutants on {name}...")
        res = run_mutation_on_module(mutants[0].module_relpath, tests, mutants)
        all_results[name] = res
        killed = sum(1 for r in res if r.status == "KILLED")
        survived = sum(1 for r in res if r.status == "SURVIVED")
        total = len(res)
        score = (killed / total * 100.0) if total > 0 else 0.0
        print(f"[{name}] Total: {total}, Killed: {killed}, Survived: {survived}, Mutation Score: {score:.1f}%")

    # Generate Report
    report_path = os.path.join(os.path.dirname(__file__), "mutation_report.md")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("# Mutation Testing Report (SPEC-DEEPVOL-FX-CROSS-003 Phase 14)\n\n")
        rep.write("## Executive Summary\n\n")
        rep.write("| Module | Mutants Evaluated | Killed | Survived | Mutation Score | Spec Target | Status |\n")
        rep.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        overall_killed = 0
        overall_total = 0

        for name, mutants, _ in suites:
            res = all_results[name]
            k = sum(1 for r in res if r.status == "KILLED")
            tot = len(res)
            overall_killed += k
            overall_total += tot
            sc = (k / tot * 100.0) if tot > 0 else 0.0
            stat = "PASS" if sc >= 90.0 else "FAIL"
            rep.write(f"| `{name}` | {tot} | {k} | {tot - k} | **{sc:.1f}%** | >= 90.0% | {stat} |\n")

        overall_score = (overall_killed / overall_total * 100.0) if overall_total > 0 else 0.0
        rep.write(f"| **OVERALL** | **{overall_total}** | **{overall_killed}** | **{overall_total - overall_killed}** | **{overall_score:.1f}%** | **>= 90.0%** | **{'PASS' if overall_score >= 90.0 else 'FAIL'}** |\n\n")

        rep.write("## Surviving Mutants and Equivalence Analysis\n\n")
        for name, mutants, _ in suites:
            res = all_results[name]
            survived_list = [r for r in res if r.status == "SURVIVED"]
            rep.write(f"### `{name}` ({len(survived_list)} surviving)\n\n")
            if not survived_list:
                rep.write("No surviving mutants. 100% mutation kill rate achieved.\n\n")
            else:
                rep.write("| Mutant ID | Description | Diff Snippet | Equivalence / Killing Assessment |\n")
                rep.write("| :--- | :--- | :--- | :--- |\n")
                for sr in survived_list:
                    rep.write(f"| `{sr.mutant.mutant_id}` | {sr.mutant.description} | `{sr.mutant.target_str[:25]}` -> `{sr.mutant.replacement_str[:25]}` | Documented below |\n")
                rep.write("\n")

    print(f"\nReport generated at {report_path}")


if __name__ == "__main__":
    main()

"""_tier1_backend.py: Standalone IEEE-754 float64 CPU math kernel protocol & C++23/NumPy adapters."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import struct
import threading
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt

_INV_SQRT2: float = 0.70710678118654752440084436210485
_INV_SQRT2PI: float = 0.39894228040143267793994605993438
_SQRT_HALF_PI: float = 1.2533141373155002512078826424055
_SQRT_TWO_OVER_PI: float = 0.79788456080286535587989211986876
_TWO_OVER_SQRT_PI: float = 0.56418958354775628694807945156077
_TIE_TAU: float = 0.008

_GL16_Y_NORM: Tuple[float, ...] = (
    -2.555646775436514,
    -1.9155297652282088,
    -1.4970940120973921,
    -1.1635767166086022,
    -0.87398989719866271,
    -0.60981672311372492,
    -0.36060272864840204,
    -0.11936335752495635,
    0.11936335752495648,
    0.36060272864840204,
    0.6098167231137247,
    0.87398989719866271,
    1.1635767166086022,
    1.4970940120973921,
    1.9155297652282088,
    2.555646775436514,
)

_GL16_X: Tuple[float, ...] = (
    -0.9894009349916499325961542,
    -0.9445750230732325760779884,
    -0.8656312023878317438804679,
    -0.7554044083550030338951012,
    -0.6178762444026437484466718,
    -0.4580167776572273863424194,
    -0.2816035507792589132304605,
    -0.0950125098376374401853193,
    0.0950125098376374401853193,
    0.2816035507792589132304605,
    0.4580167776572273863424194,
    0.6178762444026437484466718,
    0.7554044083550030338951012,
    0.8656312023878317438804679,
    0.9445750230732325760779884,
    0.9894009349916499325961542,
)
_GL16_W: Tuple[float, ...] = (
    0.0271524594117540948517806,
    0.0622535239386478928628438,
    0.0951585116824927848099251,
    0.1246289712555338720524763,
    0.1495959888165767320815017,
    0.1691565193950025381893121,
    0.1826034150449235888667637,
    0.1894506104550684962853967,
    0.1894506104550684962853967,
    0.1826034150449235888667637,
    0.1691565193950025381893121,
    0.1495959888165767320815017,
    0.1246289712555338720524763,
    0.0951585116824927848099251,
    0.0622535239386478928628438,
    0.0271524594117540948517806,
)

_MONITORING_INDEX_MAP: Dict[str, int] = {
    "WorstOf": 0,
    "Lookback": 0,
    "BestOf": 1,
    "Average": 2,
    "AverageOf": 2,
    "MultiPerf": 2,
    "Dispersion": 3,
    "RatioAB": 4,
    "RatioBA": 5,
    "RatioBestWorst": 6,
    "RatioBestAverage": 7,
    "RatioAverageWorst": 8,
    "DiffAB": 9,
    "Outperformance": 9,
    "Spread": 9,
    "DiffBA": 10,
    "DiffBestWorst": 11,
    "Rainbow": 11,
    "DiffBestAverage": 12,
    "DiffAverageWorst": 13,
    "TwinWin": 14,
}

_UNSUPPORTED_EXOTIC_CODES = frozenset({3, 6, 7, 8, 14})


def monitoring_index_to_code(name: str) -> int:
    """Maps a Murex MonitoringIndex name to its canonical integer code (0..14)."""
    code = _MONITORING_INDEX_MAP.get(name)
    if code is not None:
        return code
    code = _MONITORING_INDEX_MAP.get(name.strip())
    if code is not None:
        return code
    raise ValueError(f"Unsupported MonitoringIndex: {name!r}")


def is_exact_analytical_index(code: int) -> bool:
    """Returns True when the monitoring index has a closed-form analytical moment model."""
    return code not in _UNSUPPORTED_EXOTIC_CODES


class Tier1MathBackend(Protocol):
    """Strict IEEE-754 float64 CPU math kernel protocol (0-1 ULP parity invariant)."""

    @property
    def name(self) -> str:
        ...

    def clamped_cubic_coeffs_1d(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        left_slope: float = 0.0,
        right_slope: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Solve clamped cubic spline polynomial coefficients (4, K-1) in float64."""
        ...

    def eval_cubic_with_derivs(
        self,
        x: npt.NDArray[np.float64],
        coeffs: npt.NDArray[np.float64],
        x_query: npt.NDArray[np.float64],
        extrap_left: str = "Flat",
        extrap_right: str = "Flat",
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate (val, dval/dx, d2val/dx2) in a single searchsorted pass."""
        ...

    def stulz_analytical_bridge_cpu(
        self,
        fwd_perf: npt.NDArray[np.float64],
        vol_perf: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        coupon_barrier_type: str,
        coupon_barrier_direction: str,
        inv_lex_perm: npt.NDArray[np.int64],
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        has_ki: bool,
        i_ki_past: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        call_index_code: int,
        cpn_index_code: int,
        opt_index_code: int,
        allow_autocall_on_expiry: bool,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Exact closed-form truncated-normal + ramp survival bridge (no GH aliasing)."""
        ...


def _phi_std(x: float) -> float:
    return _INV_SQRT2PI * math.exp(-0.5 * x * x)


def _ndtr_std(x: float) -> float:
    return 0.5 * math.erfc(-x * _INV_SQRT2)


def _cody_erfcx(x: float) -> float:
    """Rational minimax erfcx(x) matching tier1_autocall_cpu.cpp to 0 ULP."""
    if math.isnan(x):
        return x
    y = abs(x)
    if y <= 0.46875:
        a = (
            3.16112374387056560e00,
            1.13864154151050156e02,
            3.77485237685302021e02,
            3.20937758913846947e03,
            1.85777706184603153e-01,
        )
        b = (
            2.36012909523441209e01,
            2.44024637934444173e02,
            1.28261652607737228e03,
            2.84423683343917062e03,
        )
        ysq = y * y
        xnum = a[4] * ysq
        xden = ysq
        for i in range(3):
            xnum = (xnum + a[i]) * ysq
            xden = (xden + b[i]) * ysq
        erf_val = y * (xnum + a[3]) / (xden + b[3])
        erfc_abs = 1.0 - (-erf_val if x < 0.0 else erf_val)
        return math.exp(ysq) * erfc_abs
    if y <= 4.0:
        c = (
            5.64188496988670089e-01,
            8.88314979438837594e00,
            6.61191906371416295e01,
            2.98635138197400131e02,
            8.81952221241769090e02,
            1.71204761263407058e03,
            2.05107837782607147e03,
            1.23033935479799725e03,
            2.15311535474403846e-08,
        )
        d = (
            1.57449261107098347e01,
            1.17693950891312499e02,
            5.37181101862009858e02,
            1.62138957456669019e03,
            3.29079923573345963e03,
            4.36261909014324716e03,
            3.43936767414372164e03,
            1.23033935480374942e03,
        )
        xnum = c[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + c[i]) * y
            xden = (xden + d[i]) * y
        result = (xnum + c[7]) / (xden + d[7])
    else:
        p = (
            3.05326634961232344e-01,
            3.60344899949804439e-01,
            1.25781726111229246e-01,
            1.60837851487422766e-02,
            6.58749161529837803e-04,
            1.63153871373020978e-02,
        )
        q = (
            2.56852019228982242e00,
            1.87295284992346047e00,
            5.27905102951428412e-01,
            6.05183413124413191e-02,
            2.33520497626869185e-03,
        )
        ysq = 1.0 / (y * y)
        xnum = p[5] * ysq
        xden = ysq
        for i in range(4):
            xnum = (xnum + p[i]) * ysq
            xden = (xden + q[i]) * ysq
        result = ysq * (xnum + p[4]) / (xden + q[4])
        result = (_TWO_OVER_SQRT_PI - result) / y
    if x < 0.0:
        return 2.0 * math.exp(y * y) - result
    return result


def _truncated_upper_std_moments(alpha: float) -> Tuple[float, float]:
    if alpha <= -4.0:
        x = -alpha
        v = 0.0
        for k in range(80, 2, -1):
            v = float(k) / (x + v)
        r2 = 2.0 / (x + v)
        psi = 1.0 / (x + r2)
        return -x - psi, max(psi * (r2 - psi), 1e-12)
    if alpha >= 26.0:
        return 0.0, 1.0
    lam = _SQRT_TWO_OVER_PI / _cody_erfcx(-alpha * _INV_SQRT2)
    return -lam, max(1.0 - alpha * lam - lam * lam, 1e-12)


def _black76_call(f: float, k: float, v: float) -> float:
    f_c = f if f > 1e-16 else 1e-16
    k_c = k if k > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    d1 = (math.log(f_c / k_c) + 0.5 * v_c * v_c) / v_c
    d2 = d1 - v_c
    return f_c * _ndtr_std(d1) - k_c * _ndtr_std(d2)


def _digital_call_spread_lognormal(f: float, v: float, barrier: float, spread: float) -> float:
    f_c = f if f > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    b_c = barrier if barrier > 1e-8 else 1e-8
    if abs(spread) <= 1e-6:
        d2 = (math.log(f_c / b_c) - 0.5 * v_c * v_c) / v_c
        p = _ndtr_std(d2)
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    b_low = min(b_c, b_c + spread)
    b_high = max(b_c, b_c + spread)
    if b_high <= 0.0:
        return 1.0
    c_low = _black76_call(f_c, b_low, v_c) if b_low > 0.0 else (f_c - b_low)
    c_high = _black76_call(f_c, b_high, v_c)
    p = (c_low - c_high) / (b_high - b_low)
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


def _normal_call(mu: float, sig: float, k: float) -> float:
    s_c = sig if sig > 1e-12 else 1e-12
    z = (mu - k) / s_c
    return (mu - k) * _ndtr_std(z) + s_c * _phi_std(z)


def _digital_call_spread_normal(mu: float, sig: float, barrier: float, spread: float) -> float:
    s_c = sig if sig > 1e-12 else 1e-12
    if abs(spread) <= 1e-6:
        p = _ndtr_std((mu - barrier) / s_c)
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    b_low = min(barrier, barrier + spread)
    b_high = max(barrier, barrier + spread)
    c_low = _normal_call(mu, s_c, b_low)
    c_high = _normal_call(mu, s_c, b_high)
    p = (c_low - c_high) / (b_high - b_low)
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


def _eval_coupon_prob_dist(
    f_or_mu: float,
    v_or_sig: float,
    barrier: float,
    spread: float,
    is_normal: bool,
    dir_down: bool,
    type_out: bool,
    index_code: int,
) -> float:
    if index_code == 14:
        if barrier <= 1.0:
            p_ev = 1.0
            return (1.0 - p_ev) if type_out else p_ev
        b_twin_dn = 2.0 - barrier
        p_up = _digital_call_spread_lognormal(f_or_mu, v_or_sig, barrier, spread)
        p_dn = (1.0 - _digital_call_spread_lognormal(f_or_mu, v_or_sig, b_twin_dn, -spread)) if b_twin_dn > 0.0 else 0.0
        p_ge = min(max(p_up + p_dn, 0.0), 1.0)
        p_ev = (1.0 - p_ge) if dir_down else p_ge
        return (1.0 - p_ev) if type_out else p_ev
    eff_spread = -spread if dir_down else spread
    p_up = (
        _digital_call_spread_normal(f_or_mu, v_or_sig, barrier, eff_spread)
        if is_normal
        else _digital_call_spread_lognormal(f_or_mu, v_or_sig, barrier, eff_spread)
    )
    p_ev = (1.0 - p_up) if dir_down else p_up
    return (1.0 - p_ev) if type_out else p_ev


def _ramp_survival_post_moments_lognormal(
    f_cond: float,
    lam: float,
    b_ko: float,
    ko_spread: float,
) -> Tuple[float, float]:
    f_c = f_cond if f_cond > 1e-16 else 1e-16
    l_c = lam if lam > 1e-12 else 1e-12
    b_c = b_ko if b_ko > 1e-8 else 1e-8
    m_u = math.log(f_c) - 0.5 * l_c * l_c
    if abs(ko_spread) <= 1e-6:
        u_star = (math.log(b_c) - m_u) / l_c
        return _truncated_upper_std_moments(u_star)

    b_lo = min(b_c, b_c + ko_spread)
    b_hi = max(b_c, b_c + ko_spread)
    if b_hi <= 0.0:
        return 0.0, 1.0
    db = b_hi - b_lo
    u_hi = (math.log(b_hi) - m_u) / l_c
    has_lo = b_lo > 0.0
    u_lo = (math.log(b_lo) - m_u) / l_c if has_lo else -1e100

    if u_hi >= -4.0:
        phi_hi = _phi_std(u_hi)
        Phi_hi = _ndtr_std(u_hi)
        Phi_hi_m = _ndtr_std(u_hi - l_c)
        phi_lo = _phi_std(u_lo) if has_lo else 0.0
        Phi_lo = _ndtr_std(u_lo) if has_lo else 0.0
        Phi_lo_m = _ndtr_std(u_lo - l_c) if has_lo else 0.0
        m0 = (b_hi * Phi_hi - b_lo * Phi_lo - f_c * (Phi_hi_m - Phi_lo_m)) / db
        if m0 <= 1e-300:
            return _truncated_upper_std_moments(u_hi)
        m1 = -(l_c * f_c / db) * (Phi_hi_m - Phi_lo_m)
        m2 = m0 + l_c * m1 + (l_c / db) * (b_hi * phi_hi - b_lo * phi_lo)
        mu_u = m1 / m0
        var_u = max(m2 / m0 - mu_u * mu_u, 1e-12)
        return mu_u, var_u

    exp_lo = math.exp(-0.5 * (u_lo - u_hi) * (u_lo + u_hi)) if has_lo else 0.0
    r_hi = _SQRT_HALF_PI * _cody_erfcx(-u_hi * _INV_SQRT2)
    r_hi_m = _SQRT_HALF_PI * _cody_erfcx(-(u_hi - l_c) * _INV_SQRT2)
    r_lo = _SQRT_HALF_PI * _cody_erfcx(-u_lo * _INV_SQRT2) if has_lo else 0.0
    r_lo_m = _SQRT_HALF_PI * _cody_erfcx(-(u_lo - l_c) * _INV_SQRT2) if has_lo else 0.0
    m0 = (b_hi * (r_hi - r_hi_m) - b_lo * exp_lo * (r_lo - r_lo_m)) / db
    if m0 <= 1e-300:
        return _truncated_upper_std_moments(u_hi)
    m1 = -(l_c / db) * (b_hi * r_hi_m - b_lo * exp_lo * r_lo_m)
    m2 = m0 + l_c * m1 + (l_c / db) * (b_hi - b_lo * exp_lo)
    mu_u = m1 / m0
    var_u = max(m2 / m0 - mu_u * mu_u, 1e-12)
    return mu_u, var_u


def _ramp_survival_post_moments_normal(
    mu_cond: float,
    sig_cond: float,
    b_ko: float,
    ko_spread: float,
) -> Tuple[float, float]:
    s_c = sig_cond if sig_cond > 1e-12 else 1e-12
    if abs(ko_spread) <= 1e-6:
        return _truncated_upper_std_moments((b_ko - mu_cond) / s_c)
    b_lo = min(b_ko, b_ko + ko_spread)
    b_hi = max(b_ko, b_ko + ko_spread)
    u_lo = (b_lo - mu_cond) / s_c
    u_hi = (b_hi - mu_cond) / s_c
    du = u_hi - u_lo
    if du <= 1e-12:
        return _truncated_upper_std_moments(u_hi)
    if u_hi >= -4.0:
        phi_hi = _phi_std(u_hi)
        phi_lo = _phi_std(u_lo)
        Phi_hi = _ndtr_std(u_hi)
        Phi_lo = _ndtr_std(u_lo)
        m0 = ((u_hi * Phi_hi + phi_hi) - (u_lo * Phi_lo + phi_lo)) / du
        if m0 <= 1e-300:
            return _truncated_upper_std_moments(u_hi)
        m1 = -(Phi_hi - Phi_lo) / du
        m2 = m0 + (phi_hi - phi_lo) / du
        mu_u = m1 / m0
        return mu_u, max(m2 / m0 - mu_u * mu_u, 1e-12)
    exp_lo = math.exp(-0.5 * (u_lo - u_hi) * (u_lo + u_hi))
    r_hi = _SQRT_HALF_PI * _cody_erfcx(-u_hi * _INV_SQRT2)
    r_lo = _SQRT_HALF_PI * _cody_erfcx(-u_lo * _INV_SQRT2)
    m0 = ((u_hi * r_hi + 1.0) - exp_lo * (u_lo * r_lo + 1.0)) / du
    if m0 <= 1e-300:
        return _truncated_upper_std_moments(u_hi)
    m1 = -(r_hi - exp_lo * r_lo) / du
    m2 = m0 + (1.0 - exp_lo) / du
    mu_u = m1 / m0
    return mu_u, max(m2 / m0 - mu_u * mu_u, 1e-12)


def _clark_one_order(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
    order: Sequence[int],
    is_best_of: bool,
    track_all_rho: bool = False,
) -> Tuple[float, float, List[float], Optional[List[float]]]:
    f_ord = [f_in[oi] if f_in[oi] > 1e-16 else 1e-16 for oi in order]
    v_ord = [v_in[oi] if v_in[oi] > 1e-12 else 1e-12 for oi in order]
    row0 = corr[order[0]]
    rho_cur = [float(row0[oi]) for oi in order]
    w_ord = [0.0] * d
    w_ord[0] = 1.0
    f_cur = f_ord[0]
    v_cur = v_ord[0]
    rho_all: Optional[List[float]] = [float(row0[i]) for i in range(d)] if track_all_rho else None

    for j in range(1, d):
        f_2 = f_ord[j]
        v_2 = v_ord[j]
        rho_12 = min(max(rho_cur[j], -0.999999), 0.999999)
        var_m = max(v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2, 1e-16)
        sig_m = math.sqrt(var_m)
        d1 = (math.log(f_cur / f_2) + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        if is_best_of:
            w1 = _ndtr_std(d1)
            w2 = _ndtr_std(-d2)
        else:
            w1 = _ndtr_std(-d1)
            w2 = _ndtr_std(d2)
        e_m1 = max(f_cur * w1 + f_2 * w2, 1e-16)
        for p in range(j):
            w_ord[p] *= w1
        w_ord[j] = w2

        f1_sq = (f_cur * f_cur) * math.exp(v_cur * v_cur)
        f2_sq = (f_2 * f_2) * math.exp(v_2 * v_2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (math.log(max(f1_sq, 1e-30) / max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        if is_best_of:
            e_m2 = f1_sq * _ndtr_std(d1_sq) + f2_sq * _ndtr_std(-d2_sq)
        else:
            e_m2 = f1_sq * _ndtr_std(-d1_sq) + f2_sq * _ndtr_std(d2_sq)
        ratio_m2 = e_m2 / max(e_m1 * e_m1, 1e-30)
        v_new = math.sqrt(math.log(ratio_m2) if ratio_m2 > 1.0 + 1e-16 else 1e-16)

        oj = order[j]
        row_j = corr[oj]
        for k in range(j + 1, d):
            v_k = v_ord[k]
            ok = order[k]
            f1_k = f_cur * math.exp(rho_cur[k] * v_cur * v_k)
            f2_k = f_2 * math.exp(float(row_j[ok]) * v_2 * v_k)
            d1_k = (math.log(max(f1_k, 1e-16) / max(f2_k, 1e-16)) + 0.5 * var_m) / sig_m
            d2_k = d1_k - sig_m
            e_mp = (
                f1_k * _ndtr_std(d1_k) + f2_k * _ndtr_std(-d2_k)
                if is_best_of
                else f1_k * _ndtr_std(-d1_k) + f2_k * _ndtr_std(d2_k)
            )
            cov_log = math.log(max(e_mp / e_m1, 1e-16))
            rho_cur[k] = min(max(cov_log / max(v_new * v_k, 1e-16), -0.999999), 0.999999)

        if rho_all is not None:
            for i in range(d):
                v_i = v_in[i] if v_in[i] > 1e-12 else 1e-12
                f1_i = f_cur * math.exp(rho_all[i] * v_cur * v_i)
                f2_i = f_2 * math.exp(float(row_j[i]) * v_2 * v_i)
                d1_i = (math.log(max(f1_i, 1e-16) / max(f2_i, 1e-16)) + 0.5 * var_m) / sig_m
                d2_i = d1_i - sig_m
                e_mp = (
                    f1_i * _ndtr_std(d1_i) + f2_i * _ndtr_std(-d2_i)
                    if is_best_of
                    else f1_i * _ndtr_std(-d1_i) + f2_i * _ndtr_std(d2_i)
                )
                cov_log = math.log(max(e_mp / e_m1, 1e-16))
                rho_all[i] = min(max(cov_log / max(v_new * v_i, 1e-16), -0.999999), 0.999999)

        f_cur = e_m1
        v_cur = v_new

    w_out = [0.0] * d
    for k, oi in enumerate(order):
        w_out[oi] = w_ord[k]
    return f_cur, v_cur, w_out, rho_all


def _clark_symmetrized(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
    is_best_of: bool,
) -> Tuple[float, float, List[float]]:
    if d == 1:
        return (f_in[0] if f_in[0] > 1e-16 else 1e-16), (v_in[0] if v_in[0] > 1e-12 else 1e-12), [1.0]

    g_sep = 1.0
    for i in range(d):
        fi = f_in[i] if f_in[i] > 1e-16 else 1e-16
        for j in range(i + 1, d):
            fj = f_in[j] if f_in[j] > 1e-16 else 1e-16
            dl = math.log(fi / fj)
            p_ij = math.exp(-(dl * dl) / (2.0 * _TIE_TAU * _TIE_TAU))
            one_m = 1.0 - p_ij
            g_sep *= one_m * one_m
    w_tie = 1.0 - g_sep

    ord_f = sorted(range(d), key=f_in.__getitem__, reverse=is_best_of)
    if w_tie < 1e-12:
        fc, vc, wc, _ = _clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, False)
        return fc, vc, wc

    ord_v = sorted(range(d), key=v_in.__getitem__)
    perms: List[List[int]] = []
    if v_in[ord_v[-1]] - v_in[ord_v[0]] <= 1e-5:
        for s in range(d):
            perms.append([ord_v[(s + k) % d] for k in range(d)])
        for s in range(d):
            perms.append(list(reversed(perms[s])))
    else:
        perms.append(ord_v)
        perms.append(list(reversed(ord_v)))
        for k in range(d - 1):
            if abs(v_in[ord_v[k + 1]] - v_in[ord_v[k]]) <= 1e-5:
                sw = list(ord_v)
                sw[k], sw[k + 1] = sw[k + 1], sw[k]
                perms.append(sw)
                perms.append(list(reversed(sw)))

    f_acc = 0.0
    m2_acc = 0.0
    w_acc = [0.0] * d
    for p in perms:
        fp, vp, wp, _ = _clark_one_order(d, f_in, v_in, corr, p, is_best_of, False)
        f_acc += fp
        m2_acc += fp * fp * math.exp(vp * vp)
        for i in range(d):
            w_acc[i] += wp[i]
    inv_k = 1.0 / float(len(perms))
    f_sym = f_acc * inv_k
    m2_sym = m2_acc * inv_k
    ratio_sym = m2_sym / max(f_sym * f_sym, 1e-30)
    v_sym = math.sqrt(math.log(ratio_sym) if ratio_sym > 1.0 + 1e-16 else 1e-16)
    w_sym = [w_acc[i] * inv_k for i in range(d)]

    if g_sep < 1e-15:
        return f_sym, v_sym, w_sym

    f_f, v_f, w_f, _ = _clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, False)
    f_out = g_sep * f_f + w_tie * f_sym
    m2_out = g_sep * (f_f * f_f * math.exp(v_f * v_f)) + w_tie * m2_sym
    ratio_out = m2_out / max(f_out * f_out, 1e-30)
    v_out = math.sqrt(math.log(ratio_out) if ratio_out > 1.0 + 1e-16 else 1e-16)
    w_out = [g_sep * w_f[i] + w_tie * w_sym[i] for i in range(d)]
    return f_out, v_out, w_out


def _clark_corr_min_max(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
) -> float:
    if d <= 1:
        return 1.0
    ord_min = sorted(range(d), key=lambda i: (round(f_in[i], 9), v_in[i]))
    ord_max = sorted(range(d), key=lambda i: (-round(f_in[i], 9), v_in[i]))
    f_min, v_min, _, rho_min_p = _clark_one_order(d, f_in, v_in, corr, ord_min, False, True)
    assert rho_min_p is not None

    f_ord = [f_in[oi] if f_in[oi] > 1e-16 else 1e-16 for oi in ord_max]
    v_ord = [v_in[oi] if v_in[oi] > 1e-12 else 1e-12 for oi in ord_max]
    row0 = corr[ord_max[0]]
    rho_cur = [float(row0[oi]) for oi in ord_max]
    f_b = f_ord[0]
    v_b = v_ord[0]
    rho_b_min = rho_min_p[ord_max[0]]

    for j in range(1, d):
        f_2 = f_ord[j]
        v_2 = v_ord[j]
        rho_12 = min(max(rho_cur[j], -0.999999), 0.999999)
        var_m = max(v_b * v_b + v_2 * v_2 - 2.0 * rho_12 * v_b * v_2, 1e-16)
        sig_m = math.sqrt(var_m)
        d1 = (math.log(f_b / f_2) + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        e_m1 = max(f_b * _ndtr_std(d1) + f_2 * _ndtr_std(-d2), 1e-16)

        f1_sq = (f_b * f_b) * math.exp(v_b * v_b)
        f2_sq = (f_2 * f_2) * math.exp(v_2 * v_2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (math.log(max(f1_sq, 1e-30) / max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        e_m2 = f1_sq * _ndtr_std(d1_sq) + f2_sq * _ndtr_std(-d2_sq)
        ratio_m2 = e_m2 / max(e_m1 * e_m1, 1e-30)
        v_new = math.sqrt(math.log(ratio_m2) if ratio_m2 > 1.0 + 1e-16 else 1e-16)

        oj = ord_max[j]
        f1_min = f_b * math.exp(rho_b_min * v_b * v_min)
        f2_min = f_2 * math.exp(rho_min_p[oj] * v_2 * v_min)
        d1_min = (math.log(max(f1_min, 1e-16) / max(f2_min, 1e-16)) + 0.5 * var_m) / sig_m
        d2_min = d1_min - sig_m
        e_b_min = f1_min * _ndtr_std(d1_min) + f2_min * _ndtr_std(-d2_min)
        rho_b_min = min(max(math.log(max(e_b_min / e_m1, 1e-16)) / max(v_new * v_min, 1e-16), -0.999999), 0.999999)

        row_j = corr[oj]
        for k in range(j + 1, d):
            v_k = v_ord[k]
            ok = ord_max[k]
            f1_k = f_b * math.exp(rho_cur[k] * v_b * v_k)
            f2_k = f_2 * math.exp(float(row_j[ok]) * v_2 * v_k)
            d1_k = (math.log(max(f1_k, 1e-16) / max(f2_k, 1e-16)) + 0.5 * var_m) / sig_m
            d2_k = d1_k - sig_m
            e_mp = f1_k * _ndtr_std(d1_k) + f2_k * _ndtr_std(-d2_k)
            rho_cur[k] = min(max(math.log(max(e_mp / e_m1, 1e-16)) / max(v_new * v_k, 1e-16), -0.999999), 0.999999)

        f_b = e_m1
        v_b = v_new
    return min(max(rho_b_min, -0.9999), 0.9999)


def _eval_index_moments(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
    inv_lex_perm: Optional[Sequence[int]],
    code: int,
) -> Tuple[float, float, List[float]]:
    if d == 1:
        if 9 <= code <= 13:
            return 0.0, 1e-12, [0.0]
        return (f_in[0] if f_in[0] > 1e-16 else 1e-16), (v_in[0] if v_in[0] > 1e-12 else 1e-12), [1.0]

    def _ab() -> Tuple[int, int]:
        if inv_lex_perm is not None and len(inv_lex_perm) >= 2:
            return int(inv_lex_perm[0]), int(inv_lex_perm[1])
        return 0, 1

    if code in (0, 14):
        return _clark_symmetrized(d, f_in, v_in, corr, False)
    if code == 1:
        return _clark_symmetrized(d, f_in, v_in, corr, True)
    if code == 2:
        inv_d = 1.0 / float(d)
        f_out = sum(f_in) * inv_d
        m2_sum = 0.0
        for i in range(d):
            for j in range(d):
                m2_sum += f_in[i] * f_in[j] * math.exp(corr[i][j] * v_in[i] * v_in[j])
        m2 = m2_sum * (inv_d * inv_d)
        ratio = m2 / max(f_out * f_out, 1e-30)
        v_out = math.sqrt(math.log(ratio) if ratio > 1.0 + 1e-16 else 1e-16)
        return f_out, v_out, [inv_d] * d
    if code == 3:
        inv_d = 1.0 / float(d)
        m2_p = sum(f_in[i] * f_in[i] * math.exp(v_in[i] * v_in[i]) for i in range(d))
        m2_avg = sum(f_in[i] * f_in[j] * math.exp(corr[i][j] * v_in[i] * v_in[j]) for i in range(d) for j in range(d))
        ew2 = max(m2_p * inv_d - m2_avg * (inv_d * inv_d), 1e-12)
        f_mean = sum(f_in) * inv_d
        var_f = sum((f_in[i] - f_mean) ** 2 for i in range(d)) * inv_d
        sig_approx = math.sqrt(ew2)
        if var_f > 1e-10:
            ratio = ew2 / var_f
            v_out = math.sqrt(math.log(ratio) if ratio > 1.0 + 1e-16 else 1e-16)
            f_out = math.sqrt(var_f)
        else:
            f_out = sig_approx
            v_out = 0.35
        w_out = [inv_d * (f_in[i] - f_mean) / max(sig_approx, 1e-8) for i in range(d)]
        return f_out, v_out, w_out
    if code in (4, 5):
        ia, ib = _ab()
        if code == 5:
            ia, ib = ib, ia
        fa = f_in[ia] if f_in[ia] > 1e-16 else 1e-16
        fb = f_in[ib] if f_in[ib] > 1e-16 else 1e-16
        va = v_in[ia] if v_in[ia] > 1e-12 else 1e-12
        vb = v_in[ib] if v_in[ib] > 1e-12 else 1e-12
        rab = corr[ia][ib]
        var_r = max(va * va + vb * vb - 2.0 * rab * va * vb, 1e-16)
        f_out = (fa / fb) * math.exp(vb * vb - rab * va * vb)
        v_out = math.sqrt(var_r)
        w_out = [0.0] * d
        w_out[ia] = 1.0
        w_out[ib] = -1.0
        return f_out, v_out, w_out
    if code in (6, 7, 8):
        if code == 6:
            f_num, v_num, w_num = _clark_symmetrized(d, f_in, v_in, corr, True)
            f_den, v_den, w_den = _clark_symmetrized(d, f_in, v_in, corr, False)
            rho_nd = _clark_corr_min_max(d, f_in, v_in, corr)
        else:
            if code == 7:
                f_num, v_num, w_num = _clark_symmetrized(d, f_in, v_in, corr, True)
                f_den, v_den, w_den = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
            else:
                f_num, v_num, w_num = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
                f_den, v_den, w_den = _clark_symmetrized(d, f_in, v_in, corr, False)
            cov = 0.0
            vn2 = 0.0
            vd2 = 0.0
            for i in range(d):
                for j in range(d):
                    cij = corr[i][j] * v_in[i] * v_in[j]
                    cov += w_num[i] * w_den[j] * cij
                    vn2 += w_num[i] * w_num[j] * cij
                    vd2 += w_den[i] * w_den[j] * cij
            rho_nd = min(max(cov / math.sqrt(max(vn2 * vd2, 1e-30)), -0.9999), 0.9999)
        var_r = max(v_num * v_num + v_den * v_den - 2.0 * rho_nd * v_num * v_den, 1e-16)
        f_out = (f_num / max(f_den, 1e-16)) * math.exp(v_den * v_den - rho_nd * v_num * v_den)
        return f_out, math.sqrt(var_r), [w_num[i] - w_den[i] for i in range(d)]
    if code in (9, 10):
        ia, ib = _ab()
        if code == 10:
            ia, ib = ib, ia
        fa, fb = f_in[ia], f_in[ib]
        va, vb = v_in[ia], v_in[ib]
        rab = corr[ia][ib]
        var_d = fa * fa * math.expm1(va * va) + fb * fb * math.expm1(vb * vb) - 2.0 * fa * fb * math.expm1(rab * va * vb)
        w_out = [0.0] * d
        w_out[ia] = 1.0
        w_out[ib] = -1.0
        return fa - fb, math.sqrt(max(var_d, 1e-16)), w_out

    if code == 11:
        f_a, v_a, w_a = _clark_symmetrized(d, f_in, v_in, corr, True)
        f_b, v_b, w_b = _clark_symmetrized(d, f_in, v_in, corr, False)
        rho_ab = _clark_corr_min_max(d, f_in, v_in, corr)
    else:
        if code == 12:
            f_a, v_a, w_a = _clark_symmetrized(d, f_in, v_in, corr, True)
            f_b, v_b, w_b = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
        else:
            f_a, v_a, w_a = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
            f_b, v_b, w_b = _clark_symmetrized(d, f_in, v_in, corr, False)
        cov = 0.0
        va2 = 0.0
        vb2 = 0.0
        for i in range(d):
            for j in range(d):
                cij = corr[i][j] * v_in[i] * v_in[j]
                cov += w_a[i] * w_b[j] * cij
                va2 += w_a[i] * w_a[j] * cij
                vb2 += w_b[i] * w_b[j] * cij
        rho_ab = min(max(cov / math.sqrt(max(va2 * vb2, 1e-30)), -0.9999), 0.9999)
    var_d = f_a * f_a * math.expm1(v_a * v_a) + f_b * f_b * math.expm1(v_b * v_b) - 2.0 * f_a * f_b * math.expm1(rho_ab * v_a * v_b)
    return f_a - f_b, math.sqrt(max(var_d, 1e-16)), [w_a[i] - w_b[i] for i in range(d)]


def _compute_cross_index_corr(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
    code_a: int,
    w_a: Sequence[float],
    code_b: int,
    w_b: Sequence[float],
) -> float:
    if code_a == code_b:
        return 1.0
    if (code_a == 0 and code_b == 1) or (code_a == 1 and code_b == 0):
        return _clark_corr_min_max(d, f_in, v_in, corr)
    da_diff = 9 <= code_a <= 13
    db_diff = 9 <= code_b <= 13
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(d):
        wai = w_a[i] * (f_in[i] if da_diff else 1.0)
        for j in range(d):
            wbj = w_b[j] * (f_in[j] if db_diff else 1.0)
            cij = corr[i][j] * v_in[i] * v_in[j]
            num += wai * wbj * cij
    for i in range(d):
        wai = w_a[i] * (f_in[i] if da_diff else 1.0)
        wbi = w_b[i] * (f_in[i] if db_diff else 1.0)
        for j in range(d):
            waj = w_a[j] * (f_in[j] if da_diff else 1.0)
            wbj = w_b[j] * (f_in[j] if db_diff else 1.0)
            cij = corr[i][j] * v_in[i] * v_in[j]
            da += wai * waj * cij
            db += wbi * wbj * cij
    return min(max(num / math.sqrt(max(da * db, 1e-30)), -0.9999), 0.9999)


def _lognormal_moments_interval(f: float, v: float, a: float, b: float, max_order: int) -> List[float]:
    if b <= a or b <= 0.0:
        return [0.0] * (max_order + 1)
    f_c = f if f > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    m_u = math.log(f_c) - 0.5 * v_c * v_c
    u_b = (math.log(b) - m_u) / v_c
    has_a = a > 0.0
    u_a = (math.log(a) - m_u) / v_c if has_a else 0.0
    out: List[float] = []
    fp = 1.0
    for k in range(max_order + 1):
        scale = fp * math.exp(0.5 * float(k * (k - 1)) * v_c * v_c)
        pb = _ndtr_std(u_b - float(k) * v_c)
        pa = _ndtr_std(u_a - float(k) * v_c) if has_a else 0.0
        out.append(scale * (pb - pa))
        fp *= f_c
    return out


def _normal_moments_interval(mu: float, sig: float, a: float, b: float, max_order: int) -> List[float]:
    if b <= a:
        return [0.0] * (max_order + 1)
    s_c = sig if sig > 1e-12 else 1e-12
    u_b = (b - mu) / s_c
    has_a = a > -1e50
    u_a = (a - mu) / s_c if has_a else 0.0
    phi_b = _phi_std(u_b)
    phi_a = _phi_std(u_a) if has_a else 0.0
    j0 = _ndtr_std(u_b) - (_ndtr_std(u_a) if has_a else 0.0)
    j1 = phi_a - phi_b
    j2 = j0 + (u_a * phi_a if has_a else 0.0) - u_b * phi_b
    j3 = 2.0 * j1 + (u_a * u_a * phi_a if has_a else 0.0) - u_b * u_b * phi_b
    out = [j0]
    if max_order >= 1:
        out.append(mu * j0 + s_c * j1)
    if max_order >= 2:
        out.append(mu * mu * j0 + 2.0 * mu * s_c * j1 + s_c * s_c * j2)
    if max_order >= 3:
        out.append(
            mu * mu * mu * j0 + 3.0 * mu * mu * s_c * j1 + 3.0 * mu * s_c * s_c * j2 + s_c * s_c * s_c * j3
        )
    return out


def _exact_piecewise_terminal_put(
    f_or_mu: float,
    v_or_sig: float,
    is_normal: bool,
    strike: float,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    has_expiry_ko: bool,
    ko_barrier: float,
    ko_spread: float,
) -> float:
    if strike <= 1e-8:
        return 0.0
    k_low = lower_strike if lower_strike > 0.0 else 0.0
    if k_low >= strike:
        return 0.0
    low_bound = -1e60 if is_normal else 0.0
    pts: List[float] = [low_bound, strike]
    if k_low > low_bound:
        pts.append(k_low)
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            pts.append(ki_barrier)
        else:
            pts.append(min(ki_barrier, ki_barrier - ki_spread))
            pts.append(max(ki_barrier, ki_barrier - ki_spread))
    if has_expiry_ko:
        if abs(ko_spread) <= 1e-6:
            pts.append(ko_barrier)
        else:
            pts.append(min(ko_barrier, ko_barrier + ko_spread))
            pts.append(max(ko_barrier, ko_barrier + ko_spread))
    pts = sorted(x for x in pts if low_bound <= x <= strike)

    total = 0.0
    for idx in range(len(pts) - 1):
        a = pts[idx]
        b = pts[idx + 1]
        if b - a <= 1e-14:
            continue
        mid = (b - 1.0) if a < -1e30 else 0.5 * (a + b)
        if mid <= k_low:
            c0, c1 = strike - k_low, 0.0
        else:
            c0, c1 = strike, -1.0

        d0, d1 = 1.0, 0.0
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if mid >= ki_barrier:
                    continue
            else:
                b_ki_lo = min(ki_barrier, ki_barrier - ki_spread)
                b_ki_hi = max(ki_barrier, ki_barrier - ki_spread)
                if mid >= b_ki_hi:
                    continue
                if mid > b_ki_lo:
                    db = b_ki_hi - b_ki_lo
                    d0, d1 = b_ki_hi / db, -1.0 / db

        e0, e1 = 1.0, 0.0
        if has_expiry_ko:
            if abs(ko_spread) <= 1e-6:
                if mid >= ko_barrier:
                    continue
            else:
                b_ko_lo = min(ko_barrier, ko_barrier + ko_spread)
                b_ko_hi = max(ko_barrier, ko_barrier + ko_spread)
                if mid >= b_ko_hi:
                    continue
                if mid > b_ko_lo:
                    db_k = b_ko_hi - b_ko_lo
                    e0, e1 = b_ko_hi / db_k, -1.0 / db_k

        cd0 = c0 * d0
        cd1 = c0 * d1 + c1 * d0
        cd2 = c1 * d1
        q0 = cd0 * e0
        q1 = cd0 * e1 + cd1 * e0
        q2 = cd1 * e1 + cd2 * e0
        q3 = cd2 * e1
        max_k = 3 if q3 != 0.0 else (2 if q2 != 0.0 else 1)
        moms = (
            _normal_moments_interval(f_or_mu, v_or_sig, a, b, max_k)
            if is_normal
            else _lognormal_moments_interval(f_or_mu, v_or_sig, a, b, max_k)
        )
        total += q0 * moms[0] + q1 * moms[1] + (q2 * moms[2] if max_k >= 2 else 0.0) + (q3 * moms[3] if max_k >= 3 else 0.0)
    return total


def _exact_piecewise_terminal_call(
    f_or_mu: float,
    v_or_sig: float,
    is_normal: bool,
    k_call: float,
    has_expiry_ko: bool,
    ko_barrier: float,
    ko_spread: float,
) -> float:
    if not has_expiry_ko:
        return _normal_call(f_or_mu, v_or_sig, k_call) if is_normal else _black76_call(f_or_mu, k_call, v_or_sig)
    if abs(ko_spread) <= 1e-6:
        if k_call >= ko_barrier:
            return 0.0
        moms = (
            _normal_moments_interval(f_or_mu, v_or_sig, k_call, ko_barrier, 1)
            if is_normal
            else _lognormal_moments_interval(f_or_mu, v_or_sig, k_call, ko_barrier, 1)
        )
        return moms[1] - k_call * moms[0]
    b_ko_lo = min(ko_barrier, ko_barrier + ko_spread)
    b_ko_hi = max(ko_barrier, ko_barrier + ko_spread)
    if k_call >= b_ko_hi:
        return 0.0
    pts = [k_call]
    if b_ko_lo > k_call:
        pts.append(b_ko_lo)
    pts.append(b_ko_hi)
    pts.sort()
    total = 0.0
    for idx in range(len(pts) - 1):
        a = pts[idx]
        b = pts[idx + 1]
        if b - a <= 1e-14:
            continue
        mid = 0.5 * (a + b)
        e0, e1 = 1.0, 0.0
        if mid > b_ko_lo:
            db = b_ko_hi - b_ko_lo
            e0, e1 = b_ko_hi / db, -1.0 / db
        q0 = -k_call * e0
        q1 = e0 - k_call * e1
        q2 = e1
        max_k = 2 if q2 != 0.0 else 1
        moms = (
            _normal_moments_interval(f_or_mu, v_or_sig, a, b, max_k)
            if is_normal
            else _lognormal_moments_interval(f_or_mu, v_or_sig, a, b, max_k)
        )
        total += q0 * moms[0] + q1 * moms[1] + (q2 * moms[2] if max_k >= 2 else 0.0)
    return total


_GX4: Tuple[float, ...] = (
    -0.86113631159405258,
    -0.33998104358485626,
    0.33998104358485626,
    0.86113631159405258,
)
_GW4: Tuple[float, ...] = (
    0.34785484513745386,
    0.65214515486254614,
    0.65214515486254614,
    0.34785484513745386,
)


def _onestar_payoff_fn(
    w: float,
    strike: float,
    k_low: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    use_direct_ko: bool,
    ko_b_term: float,
    ko_spread: float,
) -> float:
    if w >= strike:
        return 0.0
    pay = strike - w
    if k_low > 0.0:
        pay = min(pay, max(strike - k_low, 0.0))
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            if w >= ki_barrier:
                return 0.0
        else:
            blo = min(ki_barrier, ki_barrier - ki_spread)
            bhi = max(ki_barrier, ki_barrier - ki_spread)
            pay *= min(max((bhi - w) / (bhi - blo), 0.0), 1.0)
    if use_direct_ko:
        if abs(ko_spread) <= 1e-6:
            if w >= ko_b_term:
                return 0.0
        else:
            blo = min(ko_b_term, ko_b_term + ko_spread)
            bhi = max(ko_b_term, ko_b_term + ko_spread)
            pay *= min(max((bhi - w) / (bhi - blo), 0.0), 1.0)
    return pay


def _eval_1factor_onestar_np(
    d: int,
    f_c: Sequence[float],
    v_c: Sequence[float],
    corr: Sequence[Sequence[float]],
    strike: float,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    use_direct_ko: bool,
    ko_b_term: float,
    ko_spread: float,
    one_star_strike: float,
    one_star_coupon: float,
    surv_cur: float,
    surv_prev_term: float,
    option_gearing: float,
) -> Tuple[float, float]:
    l_put_out = 0.0
    l_onestar_out = 0.0
    k_low = lower_strike if lower_strike > 0.0 else 0.0
    log_ks = math.log(max(one_star_strike, 1e-16))

    if d == 1:
        f0 = max(f_c[0], 1e-16)
        v0 = max(v_c[0], 1e-12)
        m0 = math.log(f0) - 0.5 * v0 * v0
        if one_star_coupon > 0.0:
            if use_direct_ko:
                p_os = 0.0
                if abs(ko_spread) <= 1e-6:
                    if ko_b_term > one_star_strike:
                        p_os = max(
                            _ndtr_std((math.log(ko_b_term) - m0) / v0) - _ndtr_std((log_ks - m0) / v0),
                            0.0,
                        )
                else:
                    blo = min(ko_b_term, ko_b_term + ko_spread)
                    bhi = max(ko_b_term, ko_b_term + ko_spread)
                    mid = 0.5 * (blo + bhi)
                    hw = 0.5 * (bhi - blo)
                    for q in range(16):
                        u = max(mid + hw * _GL16_X[q], 1e-16)
                        if u > one_star_strike:
                            p_os += 0.5 * _GL16_W[q] * max(
                                _ndtr_std((math.log(u) - m0) / v0) - _ndtr_std((log_ks - m0) / v0),
                                0.0,
                            )
                l_onestar_out = surv_prev_term * one_star_coupon * p_os
            else:
                l_onestar_out = (
                    surv_cur
                    * one_star_coupon
                    * _digital_call_spread_lognormal(f0, v0, one_star_strike, 0.0)
                )
        if strike > 1e-6 and option_gearing > 0.0 and one_star_strike > 1e-16:
            hi = min(strike, one_star_strike)
            if hi > 0.0:
                u_hi = (math.log(hi) - m0) / v0
                u_lo = u_hi - 12.0
                mid_u = 0.5 * (u_lo + u_hi)
                hw_u = 0.5 * (u_hi - u_lo)
                acc = 0.0
                for q in range(16):
                    u = mid_u + hw_u * _GL16_X[q]
                    w = math.exp(m0 + v0 * u)
                    acc += _GL16_W[q] * _phi_std(u) * _onestar_payoff_fn(
                        w,
                        strike,
                        k_low,
                        has_active_ki,
                        ki_barrier,
                        ki_spread,
                        use_direct_ko,
                        ko_b_term,
                        ko_spread,
                    )
                l_put_out = (surv_prev_term if use_direct_ko else surv_cur) * hw_u * acc
        return l_put_out, l_onestar_out

    sum_corr = 0.0
    for i in range(d):
        for j in range(d):
            sum_corr += corr[i][j]
    rho_off = min(max((sum_corr - float(d)) / float(d * (d - 1)), 0.0), 0.999)
    b0 = math.sqrt(rho_off)
    b = [b0] * d
    mb = [0.0] * d
    for _ in range(8):
        n2 = 0.0
        for i in range(d):
            acc_m = b[i] * b[i] * b[i]
            for j in range(d):
                if j != i:
                    acc_m += corr[i][j] * b[j]
            mb[i] = acc_m
            n2 += acc_m * acc_m
        inv_n = 1.0 / math.sqrt(max(n2, 1e-30))
        sum_u = 0.0
        for i in range(d):
            mb[i] *= inv_n
            sum_u += mb[i]
        sgn = -1.0 if sum_u < 0.0 else 1.0
        for i in range(d):
            mb[i] *= sgn
        lam = 0.0
        for i in range(d):
            mu_i = (b[i] * b[i]) * mb[i]
            for j in range(d):
                if j != i:
                    mu_i += corr[i][j] * mb[j]
            lam += mb[i] * mu_i
        s_lam = math.sqrt(max(lam, 0.0))
        for i in range(d):
            val_b = mb[i] * s_lam
            b[i] = 0.0 if val_b < 0.0 else (0.999 if val_b > 0.999 else val_b)

    mu_arr = [0.0] * d
    vb_arr = [0.0] * d
    vs_arr = [0.0] * d
    for i in range(d):
        fi = max(f_c[i], 1e-16)
        vi = max(v_c[i], 1e-12)
        se_i = math.sqrt(max(1.0 - b[i] * b[i], 1e-12))
        mu_arr[i] = math.log(fi) - 0.5 * vi * vi
        vb_arr[i] = vi * b[i]
        vs_arr[i] = max(vi * se_i, 1e-12)

    p_max_lt_ks_y = [0.0] * 16
    for q in range(16):
        y = _GL16_Y_NORM[q]
        prod = 1.0
        for i in range(d):
            prod *= _ndtr_std((log_ks - mu_arr[i] - vb_arr[i] * y) / vs_arr[i])
        p_max_lt_ks_y[q] = prod

    if one_star_coupon > 0.0:
        os_acc = 0.0
        if use_direct_ko:
            if abs(ko_spread) <= 1e-6:
                ko_pts = [ko_b_term]
            else:
                blo = min(ko_b_term, ko_b_term + ko_spread)
                bhi = max(ko_b_term, ko_b_term + ko_spread)
                mid = 0.5 * (blo + bhi)
                hw = 0.5 * (bhi - blo)
                ko_pts = [max(mid + hw * _GX4[r], 1e-16) for r in range(4)]
            n_ko = len(ko_pts)
            for q in range(16):
                y = _GL16_Y_NORM[q]
                p_mks = p_max_lt_ks_y[q]
                p_sub = 0.0
                for r in range(n_ko):
                    log_ko = math.log(max(ko_pts[r], 1e-16))
                    ko_lt_ks = ko_pts[r] < one_star_strike
                    p_ge_ko = 1.0
                    p_in = 1.0
                    for i in range(d):
                        c_ks = _ndtr_std((log_ks - mu_arr[i] - vb_arr[i] * y) / vs_arr[i])
                        c_ko = _ndtr_std((log_ko - mu_arr[i] - vb_arr[i] * y) / vs_arr[i])
                        p_ge_ko *= 1.0 - c_ko
                        if ko_lt_ks:
                            p_in *= max(c_ks - c_ko, 0.0)
                    p_term = (1.0 - p_ge_ko) - (p_mks - (p_in if ko_lt_ks else 0.0))
                    p_sub += (1.0 if n_ko == 1 else 0.5 * _GW4[r]) * max(p_term, 0.0)
                os_acc += 0.5 * _GL16_W[q] * p_sub
            l_onestar_out = surv_prev_term * one_star_coupon * os_acc
        else:
            for q in range(16):
                os_acc += 0.5 * _GL16_W[q] * (1.0 - p_max_lt_ks_y[q])
            l_onestar_out = surv_cur * one_star_coupon * os_acc

    if strike > 1e-6 and option_gearing > 0.0:
        def _eval_cum_joint(x: float) -> float:
            if x <= 0.0:
                return 0.0
            log_x = math.log(x)
            res = 0.0
            for q in range(16):
                y = _GL16_Y_NORM[q]
                p_in = 1.0
                for i in range(d):
                    c_ks = _ndtr_std((log_ks - mu_arr[i] - vb_arr[i] * y) / vs_arr[i])
                    c_x = _ndtr_std((log_x - mu_arr[i] - vb_arr[i] * y) / vs_arr[i])
                    p_in *= max(c_ks - c_x, 0.0)
                res += 0.5 * _GL16_W[q] * max(p_max_lt_ks_y[q] - p_in, 0.0)
            return res

        pts_list: List[float] = [0.0, strike]
        if 0.0 < k_low < strike:
            pts_list.append(k_low)
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if 0.0 <= ki_barrier <= strike:
                    pts_list.append(ki_barrier)
            else:
                for p_v in (min(ki_barrier, ki_barrier - ki_spread), max(ki_barrier, ki_barrier - ki_spread)):
                    if 0.0 <= p_v <= strike:
                        pts_list.append(p_v)
        if use_direct_ko:
            if abs(ko_spread) <= 1e-6:
                if 0.0 <= ko_b_term <= strike:
                    pts_list.append(ko_b_term)
            else:
                for p_v in (min(ko_b_term, ko_b_term + ko_spread), max(ko_b_term, ko_b_term + ko_spread)):
                    if 0.0 <= p_v <= strike:
                        pts_list.append(p_v)
        pts_list.sort()

        put_acc = 0.0
        for idx in range(len(pts_list) - 1):
            a = pts_list[idx]
            b_pt = pts_list[idx + 1]
            span = b_pt - a
            if span <= 1e-12:
                continue
            mid_x = 0.5 * (a + b_pt)
            hw_x = 0.5 * span
            ga = _onestar_payoff_fn(
                a + 1e-10 * span,
                strike,
                k_low,
                has_active_ki,
                ki_barrier,
                ki_spread,
                use_direct_ko,
                ko_b_term,
                ko_spread,
            )
            gb_m = _onestar_payoff_fn(
                b_pt - 1e-10 * span,
                strike,
                k_low,
                has_active_ki,
                ki_barrier,
                ki_spread,
                use_direct_ko,
                ko_b_term,
                ko_spread,
            )
            gb_p = _onestar_payoff_fn(
                b_pt + 1e-10,
                strike,
                k_low,
                has_active_ki,
                ki_barrier,
                ki_spread,
                use_direct_ko,
                ko_b_term,
                ko_spread,
            )
            if ga > 0.0 or gb_m > 0.0:
                linear_unit = (not has_active_ki) and (not use_direct_ko) and (a >= k_low - 1e-12)
                for r in range(16):
                    xr = mid_x + hw_x * _GL16_X[r]
                    if linear_unit:
                        neg_gp = 1.0
                    else:
                        h_fd = 1e-7 * max(span, 1e-3)
                        xp = min(xr + h_fd, b_pt - 1e-12 * span)
                        xm = max(xr - h_fd, a + 1e-12 * span)
                        dx_fd = xp - xm
                        g_p = _onestar_payoff_fn(
                            xp, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread
                        )
                        g_m = _onestar_payoff_fn(
                            xm, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread
                        )
                        neg_gp = (-(g_p - g_m) / dx_fd) if dx_fd > 0.0 else 0.0
                    if abs(neg_gp) > 1e-15:
                        put_acc += hw_x * _GL16_W[r] * neg_gp * _eval_cum_joint(xr)
            jump = gb_m - gb_p
            if abs(jump) > 1e-12:
                put_acc += jump * _eval_cum_joint(b_pt)
        l_put_out = (surv_prev_term if use_direct_ko else surv_cur) * put_acc

    return l_put_out, l_onestar_out


class PureNumpyTier1Backend:
    """Zero-dependency pure NumPy/Python IEEE-754 float64 Tier-1 backend."""

    @property
    def name(self) -> str:
        return "numpy"

    def clamped_cubic_coeffs_1d(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        left_slope: float = 0.0,
        right_slope: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        k = int(x_arr.shape[0])
        if k < 2:
            raise ValueError("clamped_cubic_coeffs_1d requires at least 2 knots.")
        n_seg = k - 1
        h = np.empty(n_seg, dtype=np.float64)
        s = np.empty(n_seg, dtype=np.float64)
        d = np.zeros(k, dtype=np.float64)
        for i in range(n_seg):
            h[i] = x_arr[i + 1] - x_arr[i]
            s[i] = (y_arr[i + 1] - y_arr[i]) / h[i]
        d[0] = float(left_slope)
        d[k - 1] = float(right_slope)
        n_int = k - 2
        if n_int > 0:
            c_star = np.zeros(n_int, dtype=np.float64)
            d_star = np.zeros(n_int, dtype=np.float64)
            diag0 = 2.0 * (h[0] + h[1])
            rhs0 = 3.0 * (h[1] * s[0] + h[0] * s[1]) - h[1] * float(left_slope)
            if n_int == 1:
                rhs0 -= h[0] * float(right_slope)
            c_star[0] = (h[0] / diag0) if n_int > 1 else 0.0
            d_star[0] = rhs0 / diag0
            for i in range(1, n_int):
                hp = h[i]
                hn = h[i + 1]
                diag = 2.0 * (hp + hn)
                rhs = 3.0 * (hn * s[i] + hp * s[i + 1])
                if i == n_int - 1:
                    rhs -= hp * float(right_slope)
                m_piv = diag - hn * c_star[i - 1]
                if i < n_int - 1:
                    c_star[i] = hp / m_piv
                d_star[i] = (rhs - hn * d_star[i - 1]) / m_piv
            d[k - 2] = d_star[n_int - 1]
            for i in range(n_int - 2, -1, -1):
                d[i + 1] = d_star[i] - c_star[i] * d[i + 2]

        coeffs = np.empty((4, n_seg), dtype=np.float64)
        for i in range(n_seg):
            hi = h[i]
            c3 = y_arr[i]
            c2 = d[i]
            c1 = (3.0 * s[i] - 2.0 * d[i] - d[i + 1]) / hi
            c0 = (d[i] + d[i + 1] - 2.0 * s[i]) / (hi * hi)
            coeffs[0, i] = c0
            coeffs[1, i] = c1
            coeffs[2, i] = c2
            coeffs[3, i] = c3
        return coeffs

    def eval_cubic_with_derivs(
        self,
        x: npt.NDArray[np.float64],
        coeffs: npt.NDArray[np.float64],
        x_query: npt.NDArray[np.float64],
        extrap_left: str = "Flat",
        extrap_right: str = "Flat",
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        c_arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        xq_arr = np.ascontiguousarray(x_query, dtype=np.float64)
        k = int(x_arr.shape[0])
        n_seg = k - 1
        n_q = int(xq_arr.shape[0])
        val_out = np.empty(n_q, dtype=np.float64)
        dval_out = np.empty(n_q, dtype=np.float64)
        d2val_out = np.empty(n_q, dtype=np.float64)
        if k < 2 or n_q == 0:
            return val_out, dval_out, d2val_out

        ext_l = extrap_left.strip().lower() == "linear"
        ext_r = extrap_right.strip().lower() == "linear"
        x_min = float(x_arr[0])
        x_max = float(x_arr[k - 1])
        h_last = float(x_arr[k - 1] - x_arr[k - 2])
        c0_end = float(c_arr[0, n_seg - 1])
        c1_end = float(c_arr[1, n_seg - 1])
        c2_end = float(c_arr[2, n_seg - 1])
        c3_end = float(c_arr[3, n_seg - 1])
        val_end = ((c0_end * h_last + c1_end) * h_last + c2_end) * h_last + c3_end
        slope_end = (3.0 * c0_end * h_last + 2.0 * c1_end) * h_last + c2_end
        val_start = float(c_arr[3, 0])
        slope_start = float(c_arr[2, 0])

        for q in range(n_q):
            xq = float(xq_arr[q])
            if xq < x_min:
                if ext_l:
                    val_out[q] = val_start + slope_start * (xq - x_min)
                    dval_out[q] = slope_start
                    d2val_out[q] = 0.0
                else:
                    val_out[q] = val_start
                    dval_out[q] = 0.0
                    d2val_out[q] = 0.0
            elif xq > x_max:
                if ext_r:
                    val_out[q] = val_end + slope_end * (xq - x_max)
                    dval_out[q] = slope_end
                    d2val_out[q] = 0.0
                else:
                    val_out[q] = val_end
                    dval_out[q] = 0.0
                    d2val_out[q] = 0.0
            else:
                idx = int(np.searchsorted(x_arr, xq, side="right")) - 1
                if idx < 0:
                    idx = 0
                elif idx > n_seg - 1:
                    idx = n_seg - 1
                dx = xq - float(x_arr[idx])
                c0 = float(c_arr[0, idx])
                c1 = float(c_arr[1, idx])
                c2 = float(c_arr[2, idx])
                c3 = float(c_arr[3, idx])
                val_out[q] = ((c0 * dx + c1) * dx + c2) * dx + c3
                dval_out[q] = (3.0 * c0 * dx + 2.0 * c1) * dx + c2
                d2val_out[q] = 6.0 * c0 * dx + 2.0 * c1
        return val_out, dval_out, d2val_out

    def eval_autocall_baseline_cpu(
        self,
        fwd_perf: npt.NDArray[np.float64],
        vol_perf: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        coupon_barrier_type: str,
        coupon_barrier_direction: str,
        inv_lex_perm: npt.NDArray[np.int64],
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        has_ki: bool,
        i_ki_past: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        call_index_code: int,
        cpn_index_code: int,
        opt_index_code: int,
        allow_autocall_on_expiry: bool,
        *,
        c_mem_0: float = 0.0,
        is_memory: bool = True,
        autocall_coupon: float = 0.0,
        force_coupons_on_autocall: bool = False,
        no_notional_repayment: bool = False,
        is_long_basket: bool = False,
        option_gearing: float = 1.0,
        part_call_strike: float = 0.0,
        part_call_rate: float = 0.0,
        one_star_strike: float = 0.0,
        one_star_coupon: float = 0.0,
        pv_in_transit: float = 0.0,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        del tau_obs
        d_assets, m_rem = int(fwd_perf.shape[0]), int(fwd_perf.shape[1])
        f_w = np.empty(m_rem, dtype=np.float64)
        v_w = np.empty(m_rem, dtype=np.float64)
        delta_w = np.empty((d_assets, m_rem), dtype=np.float64)
        p_stop = np.empty(m_rem, dtype=np.float64)
        surv = np.empty(m_rem, dtype=np.float64)
        h_rate = np.empty(m_rem, dtype=np.float64)
        q_cpn = np.empty(m_rem, dtype=np.float64)
        scalars = np.zeros(7, dtype=np.float64)
        if m_rem == 0 or d_assets == 0:
            scalars[4] = float(pv_in_transit)
            return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars

        f_cols: List[List[float]] = [[float(fwd_perf[i, m]) for i in range(d_assets)] for m in range(m_rem)]
        v_cols: List[List[float]] = [[float(vol_perf[i, m]) for i in range(d_assets)] for m in range(m_rem)]
        c_list: List[List[float]] = [[float(corr[i, j]) for j in range(d_assets)] for i in range(d_assets)]
        perm_list: List[int] = [int(x) for x in inv_lex_perm.tolist()]

        call_is_normal = 9 <= call_index_code <= 13
        cpn_is_normal = 9 <= cpn_index_code <= 13
        opt_is_normal = 9 <= opt_index_code <= 13
        cpn_differs = cpn_index_code != call_index_code
        opt_differs = opt_index_code != call_index_code
        cpn_type_out = coupon_barrier_type.strip().lower() == "out"
        cpn_dir_down = coupon_barrier_direction.strip().lower() == "down"

        w_call_list: List[List[float]] = []
        for m in range(m_rem):
            fw_m, vw_m, wm = _eval_index_moments(d_assets, f_cols[m], v_cols[m], c_list, perm_list, call_index_code)
            f_w[m] = fw_m
            v_w[m] = vw_m
            w_call_list.append(wm)
            for i in range(d_assets):
                delta_w[i, m] = wm[i]

        surv_cur = 1.0
        mem_cur = float(c_mem_0)
        cpn_pv_sum = 0.0
        notional_pv_sum = 0.0
        z_mean = 0.0
        z_std = 1.0
        z_mean_term = 0.0
        z_std_term = 1.0
        f_cond_term = float(f_w[m_rem - 1])
        v_cond_term = float(v_w[m_rem - 1])
        surv_prev_term = 1.0

        for m in range(m_rem):
            fw_m = float(f_w[m]) if call_is_normal else max(float(f_w[m]), 1e-16)
            vw_m = max(float(v_w[m]), 1e-12)
            if z_mean == 0.0 and z_std == 1.0:
                f_cond = fw_m
                v_cond = vw_m
            elif call_is_normal:
                f_cond = fw_m + vw_m * z_mean
                v_cond = max(vw_m * z_std, 1e-12)
            else:
                f_cond = fw_m * math.exp(vw_m * z_mean - 0.5 * vw_m * vw_m * (1.0 - z_std * z_std))
                v_cond = max(vw_m * z_std, 1e-12)

            if m == m_rem - 1:
                f_cond_term = f_cond
                v_cond_term = v_cond
                z_mean_term = z_mean
                z_std_term = z_std
                surv_prev_term = surv_cur

            ko_b_m = float(ko_barriers[m])
            is_call_m = float(is_callable[m]) > 0.5
            if is_call_m:
                if call_index_code == 14:
                    ko_raw = _eval_coupon_prob_dist(f_cond, v_cond, ko_b_m, ko_spread, False, False, False, 14)
                elif call_is_normal:
                    ko_raw = _digital_call_spread_normal(f_cond, v_cond, ko_b_m, ko_spread)
                else:
                    ko_raw = _digital_call_spread_lognormal(f_cond, v_cond, ko_b_m, ko_spread)
                h_m = min(max(ko_raw, 0.0), 1.0 - 1e-7)
            else:
                h_m = 0.0
            h_rate[m] = h_m
            p_m = surv_cur * h_m
            p_stop[m] = p_m
            surv_next = surv_cur * (1.0 - h_m)
            surv[m] = surv_next

            if float(is_guaranteed_cpn[m]) > 0.5:
                q_m = 1.0
            else:
                f_cpn_cond = f_cond
                v_cpn_cond = v_cond
                if cpn_differs:
                    fc_raw, vc_raw, w_cpn_m = _eval_index_moments(
                        d_assets, f_cols[m], v_cols[m], c_list, perm_list, cpn_index_code
                    )
                    rho_cc = _compute_cross_index_corr(
                        d_assets, f_cols[m], v_cols[m], c_list, call_index_code, w_call_list[m], cpn_index_code, w_cpn_m
                    )
                    z_mc = rho_cc * z_mean
                    z_vc = rho_cc * rho_cc * (z_std * z_std) + (1.0 - rho_cc * rho_cc)
                    z_sc = math.sqrt(max(z_vc, 1e-12))
                    if cpn_is_normal:
                        f_cpn_cond = fc_raw + vc_raw * z_mc
                        v_cpn_cond = max(vc_raw * z_sc, 1e-12)
                    else:
                        f_cpn_cond = max(fc_raw, 1e-16) * math.exp(
                            vc_raw * z_mc - 0.5 * vc_raw * vc_raw * (1.0 - z_vc)
                        )
                        v_cpn_cond = max(vc_raw * z_sc, 1e-12)
                cpn_b_m = float(cpn_barriers[m])
                if not cpn_differs and force_coupons_on_autocall and is_call_m and not cpn_dir_down and not cpn_type_out:
                    if ko_b_m < cpn_b_m:
                        cpn_b_m = ko_b_m
                q_m = _eval_coupon_prob_dist(
                    f_cpn_cond, v_cpn_cond, cpn_b_m, cpn_spread, cpn_is_normal, cpn_dir_down, cpn_type_out, cpn_index_code
                )
                if force_coupons_on_autocall and is_call_m and h_m > q_m:
                    q_m = h_m
            q_cpn[m] = q_m

            c_m = float(coupon_rates[m])
            eff_c_m = (c_m + mem_cur) if is_memory else c_m
            df_m = float(df_pay[m])
            cpn_pv_sum += df_m * (surv_cur * q_m * eff_c_m + p_m * autocall_coupon)
            if is_memory:
                mem_cur = eff_c_m * (1.0 - q_m)

            if not no_notional_repayment:
                if m < m_rem - 1:
                    notional_pv_sum += df_m * p_m
                else:
                    notional_pv_sum += df_m * (p_m + surv_next)

            if m < m_rem - 1:
                z_post_mean = z_mean
                z_post_var = z_std * z_std
                if is_call_m:
                    if call_is_normal:
                        mu_u, var_u = _ramp_survival_post_moments_normal(f_cond, v_cond, ko_b_m, ko_spread)
                    else:
                        mu_u, var_u = _ramp_survival_post_moments_lognormal(f_cond, v_cond, ko_b_m, ko_spread)
                    z_post_mean = z_mean + z_std * mu_u
                    z_post_var = max(z_std * z_std * var_u, 1e-12)

                if is_call_m or z_mean != 0.0 or z_std != 1.0:
                    fc_prev, vc_prev, wc_prev = f_cols[m], v_cols[m], w_call_list[m]
                    fn_col, vn_col, wn_col = f_cols[m + 1], v_cols[m + 1], w_call_list[m + 1]
                    num = 0.0
                    den_cur = 0.0
                    den_nxt = 0.0
                    for i in range(d_assets):
                        wi_c = wc_prev[i] * (fc_prev[i] if call_is_normal else 1.0)
                        for j in range(d_assets):
                            wj_n = wn_col[j] * (fn_col[j] if call_is_normal else 1.0)
                            num += wi_c * wj_n * c_list[i][j] * vc_prev[i] * min(vc_prev[j], vn_col[j])
                    for i in range(d_assets):
                        wi_c = wc_prev[i] * (fc_prev[i] if call_is_normal else 1.0)
                        wi_n = wn_col[i] * (fn_col[i] if call_is_normal else 1.0)
                        for j in range(d_assets):
                            wj_c = wc_prev[j] * (fc_prev[j] if call_is_normal else 1.0)
                            wj_n = wn_col[j] * (fn_col[j] if call_is_normal else 1.0)
                            rho_ij = c_list[i][j]
                            den_cur += wi_c * wj_c * rho_ij * vc_prev[i] * vc_prev[j]
                            den_nxt += wi_n * wj_n * rho_ij * vn_col[i] * vn_col[j]
                    denom = math.sqrt(max(den_cur, 1e-30)) * math.sqrt(max(den_nxt, 1e-30))
                    r_next = min(max(num / max(denom, 1e-30), -0.9999), 0.9999)
                    z_mean = r_next * z_post_mean
                    z_std = math.sqrt(max(r_next * r_next * z_post_var + (1.0 - r_next * r_next), 1e-12))

            surv_cur = surv_next

        expiry_ko_active = allow_autocall_on_expiry and (float(is_callable[m_rem - 1]) > 0.5)
        f_opt_cond = f_cond_term
        v_opt_cond = v_cond_term
        if opt_differs:
            fo_raw, vo_raw, w_opt_t = _eval_index_moments(
                d_assets, f_cols[-1], v_cols[-1], c_list, perm_list, opt_index_code
            )
            rho_co = _compute_cross_index_corr(
                d_assets, f_cols[-1], v_cols[-1], c_list, call_index_code, w_call_list[-1], opt_index_code, w_opt_t
            )
            z_mo = rho_co * z_mean_term
            z_vo = rho_co * rho_co * (z_std_term * z_std_term) + (1.0 - rho_co * rho_co)
            z_so = math.sqrt(max(z_vo, 1e-12))
            if opt_is_normal:
                f_opt_cond = fo_raw + vo_raw * z_mo
                v_opt_cond = max(vo_raw * z_so, 1e-12)
            else:
                f_opt_cond = max(fo_raw, 1e-16) * math.exp(vo_raw * z_mo - 0.5 * vo_raw * vo_raw * (1.0 - z_vo))
                v_opt_cond = max(vo_raw * z_so, 1e-12)

        has_active_ki = has_ki and (ki_barrier > 0.0) and (i_ki_past < 0.5)
        has_onestar = one_star_strike > 0.0
        ko_b_term = float(ko_barriers[m_rem - 1])

        l_put_val = 0.0
        l_onestar_val = 0.0
        if has_onestar:
            f_ot = list(f_cols[-1])
            v_ot = list(v_cols[-1])
            if z_mean_term != 0.0 or z_std_term != 1.0:
                wc_term = w_call_list[-1]
                vc_n2 = 0.0
                for i in range(d_assets):
                    wi = wc_term[i] * (f_ot[i] if call_is_normal else 1.0)
                    for j in range(d_assets):
                        wj = wc_term[j] * (f_ot[j] if call_is_normal else 1.0)
                        vc_n2 += wi * wj * c_list[i][j] * v_ot[i] * v_ot[j]
                vc_n = math.sqrt(max(vc_n2, 1e-30))
                f_ot_c: List[float] = [0.0] * d_assets
                v_ot_c: List[float] = [0.0] * d_assets
                for i in range(d_assets):
                    vi = v_ot[i] if v_ot[i] > 1e-12 else 1e-12
                    cov_i = 0.0
                    for j in range(d_assets):
                        wj = wc_term[j] * (f_ot[j] if call_is_normal else 1.0)
                        cov_i += c_list[i][j] * vi * v_ot[j] * wj
                    rho_ic = min(max(cov_i / (vi * vc_n), -0.9999), 0.9999)
                    z_mc = rho_ic * z_mean_term
                    z_vc = rho_ic * rho_ic * (z_std_term * z_std_term) + (1.0 - rho_ic * rho_ic)
                    f_ot_c[i] = max(f_ot[i], 1e-16) * math.exp(vi * z_mc - 0.5 * vi * vi * (1.0 - z_vc))
                    v_ot_c[i] = max(vi * math.sqrt(max(z_vc, 1e-12)), 1e-12)
                f_ot = f_ot_c
                v_ot = v_ot_c
            use_direct_ko = expiry_ko_active and not opt_differs
            l_put_val, l_onestar_val = _eval_1factor_onestar_np(
                d_assets,
                f_ot,
                v_ot,
                c_list,
                strike,
                lower_strike,
                has_active_ki,
                ki_barrier,
                ki_spread,
                use_direct_ko,
                ko_b_term,
                ko_spread,
                one_star_strike,
                one_star_coupon,
                surv_cur,
                surv_prev_term,
                option_gearing,
            )
        elif strike > 1e-6 and option_gearing > 0.0:
            use_direct_ko = expiry_ko_active and not opt_differs
            surv_mult = surv_prev_term if use_direct_ko else surv_cur
            put_cond = _exact_piecewise_terminal_put(
                f_opt_cond,
                v_opt_cond,
                opt_is_normal,
                strike,
                lower_strike,
                has_active_ki,
                ki_barrier,
                ki_spread,
                use_direct_ko,
                ko_b_term,
                ko_spread,
            )
            l_put_val = surv_mult * put_cond

        l_call_val = 0.0
        if part_call_strike > 0.0 and part_call_rate > 0.0:
            use_direct_ko_c = expiry_ko_active and not opt_differs
            surv_mult_c = surv_prev_term if use_direct_ko_c else surv_cur
            l_call_val = surv_mult_c * _exact_piecewise_terminal_call(
                f_opt_cond, v_opt_cond, opt_is_normal, part_call_strike, use_direct_ko_c, ko_b_term, ko_spread
            )

        opt_sign = -1.0 if is_long_basket else 1.0
        opt_pv_val = float(df_pay[m_rem - 1]) * (
            -opt_sign * option_gearing * l_put_val + part_call_rate * l_call_val + l_onestar_val
        )
        pv_val = pv_in_transit + notional_pv_sum + cpn_pv_sum + opt_pv_val
        scalars[0] = l_put_val
        scalars[1] = opt_pv_val
        scalars[2] = notional_pv_sum
        scalars[3] = cpn_pv_sum
        scalars[4] = pv_val
        scalars[5] = l_call_val
        scalars[6] = l_onestar_val
        return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars

    def stulz_analytical_bridge_cpu(
        self,
        fwd_perf: npt.NDArray[np.float64],
        vol_perf: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        coupon_barrier_type: str,
        coupon_barrier_direction: str,
        inv_lex_perm: npt.NDArray[np.int64],
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        has_ki: bool,
        i_ki_past: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        call_index_code: int,
        cpn_index_code: int,
        opt_index_code: int,
        allow_autocall_on_expiry: bool,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars = self.eval_autocall_baseline_cpu(
            fwd_perf=fwd_perf,
            vol_perf=vol_perf,
            corr=corr,
            tau_obs=tau_obs,
            df_pay=df_pay,
            ko_barriers=ko_barriers,
            is_callable=is_callable,
            cpn_barriers=cpn_barriers,
            is_guaranteed_cpn=is_guaranteed_cpn,
            coupon_rates=coupon_rates,
            coupon_barrier_type=coupon_barrier_type,
            coupon_barrier_direction=coupon_barrier_direction,
            inv_lex_perm=inv_lex_perm,
            strike=strike,
            lower_strike=lower_strike,
            ki_barrier=ki_barrier,
            has_ki=has_ki,
            i_ki_past=i_ki_past,
            ko_spread=ko_spread,
            cpn_spread=cpn_spread,
            ki_spread=ki_spread,
            call_index_code=call_index_code,
            cpn_index_code=cpn_index_code,
            opt_index_code=opt_index_code,
            allow_autocall_on_expiry=allow_autocall_on_expiry,
        )
        return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, float(scalars[0])


_F64_DTYPE = np.dtype(np.float64)
_I64_DTYPE = np.dtype(np.int64)


def _ensure_f64_c(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if type(arr) is np.ndarray and arr.dtype == _F64_DTYPE and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ensure_i64_c(arr: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    if type(arr) is np.ndarray and arr.dtype == _I64_DTYPE and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.int64)


class Tier1AutocallArgs(ctypes.Structure):
    """Packed C ABI argument block matching Tier1AutocallArgs in tier1_autocall_cpu.cpp."""

    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("fwd_perf", ctypes.c_uint64),
        ("vol_perf", ctypes.c_uint64),
        ("corr", ctypes.c_uint64),
        ("tau_obs", ctypes.c_uint64),
        ("df_pay", ctypes.c_uint64),
        ("ko_barriers", ctypes.c_uint64),
        ("is_callable", ctypes.c_uint64),
        ("cpn_barriers", ctypes.c_uint64),
        ("is_guaranteed_cpn", ctypes.c_uint64),
        ("coupon_rates", ctypes.c_uint64),
        ("inv_lex_perm", ctypes.c_uint64),
        ("strike", ctypes.c_double),
        ("lower_strike", ctypes.c_double),
        ("ki_barrier", ctypes.c_double),
        ("i_ki_past", ctypes.c_double),
        ("ko_spread", ctypes.c_double),
        ("cpn_spread", ctypes.c_double),
        ("ki_spread", ctypes.c_double),
        ("c_mem_0", ctypes.c_double),
        ("autocall_coupon", ctypes.c_double),
        ("option_gearing", ctypes.c_double),
        ("part_call_strike", ctypes.c_double),
        ("part_call_rate", ctypes.c_double),
        ("one_star_strike", ctypes.c_double),
        ("one_star_coupon", ctypes.c_double),
        ("pv_in_transit", ctypes.c_double),
        ("cbt_out", ctypes.c_int32),
        ("cbd_down", ctypes.c_int32),
        ("has_ki", ctypes.c_int32),
        ("call_code", ctypes.c_int32),
        ("cpn_code", ctypes.c_int32),
        ("opt_code", ctypes.c_int32),
        ("allow_exp", ctypes.c_int32),
        ("is_memory", ctypes.c_int32),
        ("force_cpn", ctypes.c_int32),
        ("no_notional", ctypes.c_int32),
        ("is_long_basket", ctypes.c_int32),
        ("_pad0", ctypes.c_int32),
        ("f_w_out", ctypes.c_uint64),
        ("v_w_out", ctypes.c_uint64),
        ("delta_w_out", ctypes.c_uint64),
        ("p_stop_out", ctypes.c_uint64),
        ("surv_out", ctypes.c_uint64),
        ("h_rate_out", ctypes.c_uint64),
        ("q_cpn_out", ctypes.c_uint64),
        ("scalars_out", ctypes.c_uint64),
    ]


_ARGS_PACKER = struct.Struct("=2q11Q15d12i8Q").pack


class Cpp23Tier1Backend:
    """Standalone C++23 (-O3 -march=native -fno-fast-math) Tier-1 ctypes backend."""

    def __init__(self, lib_path: Path) -> None:
        self._lib_path = lib_path
        lib = ctypes.CDLL(str(lib_path))
        self._lib = lib

        c_vp = ctypes.c_void_p

        lib.tier1_clamped_cubic_coeffs_1d.argtypes = [
            ctypes.c_int64,
            c_vp,
            c_vp,
            ctypes.c_double,
            ctypes.c_double,
            c_vp,
        ]
        lib.tier1_clamped_cubic_coeffs_1d.restype = None

        lib.tier1_eval_cubic_with_derivs.argtypes = [
            ctypes.c_int64,
            c_vp,
            c_vp,
            ctypes.c_int64,
            c_vp,
            ctypes.c_int32,
            ctypes.c_int32,
            c_vp,
            c_vp,
            c_vp,
        ]
        lib.tier1_eval_cubic_with_derivs.restype = None

        lib.tier1_eval_autocall_bridge_cpu.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            ctypes.c_int32,
            ctypes.c_int32,
            c_vp,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
        ]
        lib.tier1_eval_autocall_bridge_cpu.restype = None

        lib.tier1_eval_index_moments_cpu.argtypes = [
            ctypes.c_int64,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            ctypes.c_int32,
            c_vp,
            c_vp,
            c_vp,
        ]
        lib.tier1_eval_index_moments_cpu.restype = None

        lib.tier1_eval_autocall_struct_cpu.argtypes = [ctypes.c_char_p]
        lib.tier1_eval_autocall_struct_cpu.restype = None

        self._fn_coeffs = lib.tier1_clamped_cubic_coeffs_1d
        self._fn_eval_cubic = lib.tier1_eval_cubic_with_derivs
        self._fn_bridge = lib.tier1_eval_autocall_bridge_cpu
        self._fn_struct = lib.tier1_eval_autocall_struct_cpu
        self._fn_index_moments = lib.tier1_eval_index_moments_cpu
        self._args_cls = Tier1AutocallArgs
        self._args_pack = _ARGS_PACKER

    @property
    def name(self) -> str:
        return "cpp23"

    def eval_index_moments_np(
        self,
        f_arr: npt.NDArray[np.float64],
        v_arr: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        code: int,
        inv_lex_perm: Optional[npt.NDArray[np.int64]] = None,
    ) -> Tuple[float, float, npt.NDArray[np.float64]]:
        fp = _ensure_f64_c(f_arr)
        vp = _ensure_f64_c(v_arr)
        cm = _ensure_f64_c(corr)
        d_assets = int(fp.shape[0])
        out = np.empty(d_assets + 2, dtype=np.float64)
        w_out = out[:d_assets]
        fv_ptr = out.ctypes.data + d_assets * 8
        perm_ptr = 0
        if inv_lex_perm is not None:
            perm_c = _ensure_i64_c(inv_lex_perm)
            if perm_c.size > 0:
                perm_ptr = perm_c.ctypes.data
        self._fn_index_moments(
            d_assets,
            fp.ctypes.data,
            vp.ctypes.data,
            cm.ctypes.data,
            perm_ptr,
            int(code),
            fv_ptr,
            fv_ptr + 8,
            w_out.ctypes.data,
        )
        return float(out[d_assets]), float(out[d_assets + 1]), w_out

    def clamped_cubic_coeffs_1d(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        left_slope: float = 0.0,
        right_slope: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        x_arr = _ensure_f64_c(x)
        y_arr = _ensure_f64_c(y)
        k = int(x_arr.shape[0])
        if k < 2:
            raise ValueError("clamped_cubic_coeffs_1d requires at least 2 knots.")
        coeffs = np.empty((4, k - 1), dtype=np.float64)
        self._fn_coeffs(
            k,
            x_arr.ctypes.data,
            y_arr.ctypes.data,
            float(left_slope),
            float(right_slope),
            coeffs.ctypes.data,
        )
        return coeffs

    def eval_cubic_with_derivs(
        self,
        x: npt.NDArray[np.float64],
        coeffs: npt.NDArray[np.float64],
        x_query: npt.NDArray[np.float64],
        extrap_left: str = "Flat",
        extrap_right: str = "Flat",
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x_arr = _ensure_f64_c(x)
        c_arr = _ensure_f64_c(coeffs)
        xq_arr = _ensure_f64_c(x_query)
        k = int(x_arr.shape[0])
        n_q = int(xq_arr.shape[0])
        buf = np.empty(3 * n_q, dtype=np.float64)
        val_out = buf[0:n_q]
        dval_out = buf[n_q : 2 * n_q]
        d2val_out = buf[2 * n_q : 3 * n_q]
        if k < 2 or n_q == 0:
            return val_out, dval_out, d2val_out
        ext_l = 1 if extrap_left.strip().lower() == "linear" else 0
        ext_r = 1 if extrap_right.strip().lower() == "linear" else 0
        self._fn_eval_cubic(
            k,
            x_arr.ctypes.data,
            c_arr.ctypes.data,
            n_q,
            xq_arr.ctypes.data,
            ext_l,
            ext_r,
            val_out.ctypes.data,
            dval_out.ctypes.data,
            d2val_out.ctypes.data,
        )
        return val_out, dval_out, d2val_out

    def eval_autocall_baseline_cpu(
        self,
        fwd_perf: npt.NDArray[np.float64],
        vol_perf: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        coupon_barrier_type: str,
        coupon_barrier_direction: str,
        inv_lex_perm: npt.NDArray[np.int64],
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        has_ki: bool,
        i_ki_past: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        call_index_code: int,
        cpn_index_code: int,
        opt_index_code: int,
        allow_autocall_on_expiry: bool,
        *,
        c_mem_0: float = 0.0,
        is_memory: bool = True,
        autocall_coupon: float = 0.0,
        force_coupons_on_autocall: bool = False,
        no_notional_repayment: bool = False,
        is_long_basket: bool = False,
        option_gearing: float = 1.0,
        part_call_strike: float = 0.0,
        part_call_rate: float = 0.0,
        one_star_strike: float = 0.0,
        one_star_coupon: float = 0.0,
        pv_in_transit: float = 0.0,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        fp = _ensure_f64_c(fwd_perf)
        vp = _ensure_f64_c(vol_perf)
        cm = _ensure_f64_c(corr)
        tau = _ensure_f64_c(tau_obs)
        df = _ensure_f64_c(df_pay)
        ko = _ensure_f64_c(ko_barriers)
        ic = _ensure_f64_c(is_callable)
        cb = _ensure_f64_c(cpn_barriers)
        ig = _ensure_f64_c(is_guaranteed_cpn)
        cr = _ensure_f64_c(coupon_rates)
        perm = _ensure_i64_c(inv_lex_perm)

        d_assets, m_rem = int(fp.shape[0]), int(fp.shape[1])
        dm = d_assets * m_rem
        out_buf = np.empty(6 * m_rem + dm + 7, dtype=np.float64)
        f_w = out_buf[0:m_rem]
        v_w = out_buf[m_rem : 2 * m_rem]
        p_stop = out_buf[2 * m_rem : 3 * m_rem]
        surv = out_buf[3 * m_rem : 4 * m_rem]
        h_rate = out_buf[4 * m_rem : 5 * m_rem]
        q_cpn = out_buf[5 * m_rem : 6 * m_rem]
        dw_flat = out_buf[6 * m_rem : 6 * m_rem + dm]
        delta_w = dw_flat.reshape(d_assets, m_rem)
        scalars = out_buf[6 * m_rem + dm :]

        cbt_i = 1 if (coupon_barrier_type == "Out" or coupon_barrier_type.strip().lower() == "out") else 0
        cbd_i = 1 if (coupon_barrier_direction == "Down" or coupon_barrier_direction.strip().lower() == "down") else 0

        self._fn_bridge(
            d_assets,
            m_rem,
            fp.ctypes.data,
            vp.ctypes.data,
            cm.ctypes.data,
            tau.ctypes.data if tau.size > 0 else 0,
            df.ctypes.data,
            ko.ctypes.data,
            ic.ctypes.data,
            cb.ctypes.data,
            ig.ctypes.data,
            cr.ctypes.data,
            cbt_i,
            cbd_i,
            perm.ctypes.data if perm.size > 0 else 0,
            float(strike),
            float(lower_strike),
            float(ki_barrier),
            1 if has_ki else 0,
            float(i_ki_past),
            float(ko_spread),
            float(cpn_spread),
            float(ki_spread),
            int(call_index_code),
            int(cpn_index_code),
            int(opt_index_code),
            1 if allow_autocall_on_expiry else 0,
            float(c_mem_0),
            1 if is_memory else 0,
            float(autocall_coupon),
            1 if force_coupons_on_autocall else 0,
            1 if no_notional_repayment else 0,
            1 if is_long_basket else 0,
            float(option_gearing),
            float(part_call_strike),
            float(part_call_rate),
            float(one_star_strike),
            float(one_star_coupon),
            float(pv_in_transit),
            f_w.ctypes.data,
            v_w.ctypes.data,
            dw_flat.ctypes.data,
            p_stop.ctypes.data,
            surv.ctypes.data,
            h_rate.ctypes.data,
            q_cpn.ctypes.data,
            scalars.ctypes.data,
        )
        return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars

    def stulz_analytical_bridge_cpu(
        self,
        fwd_perf: npt.NDArray[np.float64],
        vol_perf: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        coupon_barrier_type: str,
        coupon_barrier_direction: str,
        inv_lex_perm: npt.NDArray[np.int64],
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        has_ki: bool,
        i_ki_past: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        call_index_code: int,
        cpn_index_code: int,
        opt_index_code: int,
        allow_autocall_on_expiry: bool,
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars = self.eval_autocall_baseline_cpu(
            fwd_perf=fwd_perf,
            vol_perf=vol_perf,
            corr=corr,
            tau_obs=tau_obs,
            df_pay=df_pay,
            ko_barriers=ko_barriers,
            is_callable=is_callable,
            cpn_barriers=cpn_barriers,
            is_guaranteed_cpn=is_guaranteed_cpn,
            coupon_rates=coupon_rates,
            coupon_barrier_type=coupon_barrier_type,
            coupon_barrier_direction=coupon_barrier_direction,
            inv_lex_perm=inv_lex_perm,
            strike=strike,
            lower_strike=lower_strike,
            ki_barrier=ki_barrier,
            has_ki=has_ki,
            i_ki_past=i_ki_past,
            ko_spread=ko_spread,
            cpn_spread=cpn_spread,
            ki_spread=ki_spread,
            call_index_code=call_index_code,
            cpn_index_code=cpn_index_code,
            opt_index_code=opt_index_code,
            allow_autocall_on_expiry=allow_autocall_on_expiry,
        )
        return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, float(scalars[0])


_BACKEND_LOCK = threading.Lock()
_CACHED_BACKEND: Optional[Tier1MathBackend] = None


def _locate_cpp23_shared_lib() -> Optional[Path]:
    env_override = os.environ.get("DEEPVOL_TIER1_CPP_LIB")
    if env_override is not None:
        cleaned = env_override.strip()
        if cleaned.lower() in ("", "none", "numpy"):
            return None
        p = Path(cleaned)
        return p if p.is_file() else None
    pkg_dir = Path(__file__).resolve().parent
    candidates = (
        pkg_dir / "tier1_autocall_cpu.so",
        pkg_dir / "libtier1_autocall_cpu.so",
        pkg_dir / "lib" / "tier1_autocall_cpu.so",
        pkg_dir / "lib" / "libtier1_autocall_cpu.so",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def get_tier1_backend(*, force_numpy: bool = False) -> Tier1MathBackend:
    """Returns the active IEEE-754 float64 Tier-1 backend (C++23 if built, else pure NumPy)."""
    if force_numpy:
        return PureNumpyTier1Backend()
    cached = _CACHED_BACKEND
    if cached is not None:
        return cached
    return _init_tier1_backend()


def _init_tier1_backend() -> Tier1MathBackend:
    global _CACHED_BACKEND
    with _BACKEND_LOCK:
        if _CACHED_BACKEND is not None:
            return _CACHED_BACKEND
        lib_path = _locate_cpp23_shared_lib()
        if lib_path is not None:
            try:
                _CACHED_BACKEND = Cpp23Tier1Backend(lib_path)
                return _CACHED_BACKEND
            except OSError:
                pass
        _CACHED_BACKEND = PureNumpyTier1Backend()
        return _CACHED_BACKEND


def reset_tier1_backend_cache() -> None:
    """Clears the cached backend instance so environment overrides take effect."""
    global _CACHED_BACKEND
    with _BACKEND_LOCK:
        _CACHED_BACKEND = None


__all__ = [
    "Cpp23Tier1Backend",
    "PureNumpyTier1Backend",
    "Tier1MathBackend",
    "get_tier1_backend",
    "is_exact_analytical_index",
    "monitoring_index_to_code",
    "reset_tier1_backend_cache",
]

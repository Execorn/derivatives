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
    "EquallyWeightedBasket": 2,
    "MultiPerf": 2,
    "Dispersion": 3,
    "RatioAB": 4,
    "Ratio": 4,
    "RatioBA": 5,
    "RatioBestWorst": 6,
    "RatioBestAverage": 7,
    "RatioAverageWorst": 8,
    "DiffAB": 9,
    "Outperformance": 9,
    "Spread": 9,
    "DiffBA": 10,
    "DiffBestWorst": 11,
    "SpreadBestWorst": 11,
    "Rainbow": 11,
    "DiffBestAverage": 12,
    "SpreadBestAverage": 12,
    "DiffAverageWorst": 13,
    "SpreadAverageWorst": 13,
    "TwinWin": 14,
    "SecondWorstOf": 15,
    "ThirdWorstOf": 16,
    "SecondBestOf": 17,
}

_UNSUPPORTED_EXOTIC_CODES = frozenset({3, 6, 7, 8, 14, 15, 16, 17})


def monitoring_index_to_code(name: str) -> int:
    """Maps a Murex MonitoringIndex name to its canonical integer code (0..17)."""
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


_GL8_X01 = (
    0.019855071751231912,
    0.10166676129318664,
    0.2372337950418355,
    0.40828267875217511,
    0.59171732124782483,
    0.7627662049581645,
    0.89833323870681336,
    0.98014492824876809,
)
_GL8_W01 = (
    0.050614268145188532,
    0.11119051722668721,
    0.15685332293894344,
    0.18134189168918083,
    0.18134189168918083,
    0.15685332293894344,
    0.11119051722668721,
    0.050614268145188532,
)
_LAG6_X = (
    0.22284660417926069,
    1.1889321016726226,
    2.9927363260593141,
    5.7751435691045101,
    9.8374674183825892,
    15.982873980601703,
)
_LAG6_W = (
    0.45896467394996476,
    0.41700083077212002,
    0.11337338207404488,
    0.010399197453149087,
    0.00026101720281493265,
    8.9854790642962134e-07,
)
_GL16_X_ARR = (
    -0.9894009349916499325961542, -0.9445750230732325760779884,
    -0.8656312023878317438804679, -0.7554044083550030338951012,
    -0.6178762444026437484466718, -0.4580167776572273863424194,
    -0.2816035507792589132304605, -0.0950125098376374401853193,
     0.0950125098376374401853193,  0.2816035507792589132304605,
     0.4580167776572273863424194,  0.6178762444026437484466718,
     0.7554044083550030338951012,  0.8656312023878317438804679,
     0.9445750230732325760779884,  0.9894009349916499325961542,
)
_GL16_YNORM = (
    -2.555646775436514, -1.9155297652282088, -1.4970940120973921, -1.1635767166086022,
    -0.87398989719866271, -0.60981672311372492, -0.36060272864840204, -0.11936335752495635,
     0.11936335752495648,  0.36060272864840204,  0.6098167231137247,  0.87398989719866271,
     1.1635767166086022,   1.4970940120973921,   1.9155297652282088,  2.555646775436514,
)


def _ndtr_scalar(x: float) -> float:
    return 0.5 * math.erfc(-x * 0.7071067811865475)


def _ndtri_scalar(p: float) -> float:
    if p <= 0.0:
        return -8.5
    if p >= 1.0:
        return 8.5
    a = (
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    )
    b = (
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    )
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    err = _ndtr_scalar(x) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    x -= u / (1.0 + 0.5 * x * u)
    return max(-8.5, min(8.5, x))


def _bvn_cdf_pos_scalar(x: float, y: float, r: float) -> float:
    if r >= 1.0 - 1e-12:
        return _ndtr_scalar(min(x, y))
    two_pi = 2.0 * math.pi
    if r < 0.925:
        as_v = math.asin(r)
        hs = 0.5 * (x * x + y * y)
        hk = x * y
        acc = 0.0
        for gx, gw in zip(_GL16_X_ARR, _GL16_W):
            sn = math.sin(0.5 * as_v * (gx + 1.0))
            acc += 0.5 * gw * math.exp((sn * hk - hs) / (1.0 - sn * sn))
        return acc * (as_v / two_pi) + _ndtr_scalar(x) * _ndtr_scalar(y)
    h = -x
    k = -y
    hk = h * k
    a = math.sqrt((1.0 - r) * (1.0 + r))
    b = abs(h - k)
    bs = b * b
    c = (4.0 - hk) / 8.0
    d = (12.0 - hk) / 16.0
    asr = -0.5 * (bs / (a * a) + hk)
    bvn = 0.0
    if asr > -100.0:
        bvn = a * math.exp(asr) * (1.0 - c * (bs - a * a) * (1.0 - d * bs / 5.0) / 3.0 + c * d * (a ** 4) / 5.0)
    if -hk < 100.0:
        bvn -= math.exp(-0.5 * hk) * math.sqrt(two_pi) * _ndtr_scalar(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
    a_half = 0.5 * a
    acc = 0.0
    for gx, gw in zip(_GL16_X_ARR, _GL16_W):
        xs = a_half * (gx + 1.0)
        xs2 = xs * xs
        rs = math.sqrt(max(1.0 - xs2, 1e-30))
        asr_q = -0.5 * (bs / max(xs2, 1e-30) + hk)
        if asr_q > -100.0:
            acc += gw * math.exp(asr_q) * (math.exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs2 * (1.0 + d * xs2)))
    bvn -= a_half * acc
    return _ndtr_scalar(-max(h, k)) - bvn / two_pi


def _bvn_cdf_genz_scalar(x: float, y: float, rho: float) -> float:
    if abs(rho) < 1e-14:
        return _ndtr_scalar(x) * _ndtr_scalar(y)
    res = _bvn_cdf_pos_scalar(x, y, rho) if rho >= 0.0 else (_ndtr_scalar(x) - _bvn_cdf_pos_scalar(x, -y, -rho))
    lo = max(_ndtr_scalar(x) + _ndtr_scalar(y) - 1.0, 0.0)
    hi = min(_ndtr_scalar(x), _ndtr_scalar(y))
    return max(lo, min(hi, res))


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
    eff_spread = -spread if dir_down else spread
    if index_code == 14:
        b_lo = min(barrier, barrier + eff_spread)
        b_hi = max(barrier, barrier + eff_spread)
        db = b_hi - b_lo
        p_ge = 1.0
        if b_hi > 1.0:
            f_c = f_or_mu if f_or_mu > 1e-16 else 1e-16
            v_c = v_or_sig if v_or_sig > 1e-12 else 1e-12
            a_dn = max(2.0 - b_hi, 0.0)
            b_dn = max(0.0, min(1.0, 2.0 - b_lo))
            a_up = max(b_lo, 1.0)
            p_surv = 0.0
            if db <= 1e-6:
                p_surv = _lognormal_moments_interval(f_c, v_c, a_dn, b_hi, 0)[0]
            else:
                inv_db = 1.0 / db
                if b_dn > a_dn + 1e-14:
                    m = _lognormal_moments_interval(f_c, v_c, a_dn, b_dn, 1)
                    p_surv += (-(2.0 - b_hi) * m[0] + m[1]) * inv_db
                if a_up > b_dn + 1e-14:
                    p_surv += _lognormal_moments_interval(f_c, v_c, b_dn, a_up, 0)[0]
                if b_hi > a_up + 1e-14:
                    m = _lognormal_moments_interval(f_c, v_c, a_up, b_hi, 1)
                    p_surv += (b_hi * m[0] - m[1]) * inv_db
            p_ge = min(max(1.0 - p_surv, 0.0), 1.0)
        p_ev = (1.0 - p_ge) if dir_down else p_ge
        return (1.0 - p_ev) if type_out else p_ev
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

    if u_hi > -20.0:
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

    x = -u_hi
    y_lo = (u_hi - u_lo) if has_lo else 1e100
    y_end = min(y_lo, 45.0 / x)
    inv_ratio = b_hi / db
    m0 = 0.0
    n1 = 0.0
    n2 = 0.0
    for gx, gw in zip(_GL8_X01, _GL8_W01):
        yq = y_end * gx
        wt = y_end * gw * math.exp(-yq * (x + 0.5 * yq)) * inv_ratio * (-math.expm1(-l_c * yq))
        m0 += wt
        n1 += wt * yq
        n2 += wt * yq * yq
    if has_lo and y_lo * (x + 0.5 * y_lo) < 55.0:
        alpha = x + y_lo
        inv_alpha = 1.0 / alpha
        pref = math.exp(-y_lo * (x + 0.5 * y_lo)) * inv_alpha
        for lx, lw in zip(_LAG6_X, _LAG6_W):
            sq = lx * inv_alpha
            yq = y_lo + sq
            wt = pref * lw * math.exp(-0.5 * sq * sq)
            m0 += wt
            n1 += wt * yq
            n2 += wt * yq * yq
    if m0 <= 1e-300:
        return _truncated_upper_std_moments(u_hi)
    ey = n1 / m0
    return -x - ey, max(n2 / m0 - ey * ey, 1e-12)


def _ramp_survival_post_moments_twinwin_lognormal(
    f_cond: float,
    lam: float,
    b_ko: float,
    ko_spread: float,
) -> Tuple[float, float]:
    f_c = f_cond if f_cond > 1e-16 else 1e-16
    l_c = lam if lam > 1e-12 else 1e-12
    b_lo = min(b_ko, b_ko + ko_spread)
    b_hi = max(b_ko, b_ko + ko_spread)
    if b_hi <= 1.0:
        return 0.0, 1e-12
    m_u = math.log(f_c) - 0.5 * l_c * l_c
    db = b_hi - b_lo
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0

    def _add_seg(seg_a: float, seg_b: float, c0: float, c1: float) -> None:
        nonlocal m0, m1, m2
        if seg_b <= seg_a + 1e-14:
            return
        u_b = (math.log(seg_b) - m_u) / l_c
        u_a = ((math.log(seg_a) - m_u) / l_c) if seg_a > 0.0 else -1e60
        jc = _normal_moments_interval(0.0, 1.0, u_a, u_b, 2)
        m0 += c0 * jc[0]
        m1 += c0 * jc[1]
        m2 += c0 * jc[2]
        if c1 != 0.0:
            jl = _normal_moments_interval(l_c, 1.0, u_a, u_b, 2)
            wc1 = c1 * f_c
            m0 += wc1 * jl[0]
            m1 += wc1 * jl[1]
            m2 += wc1 * jl[2]

    a_dn = max(2.0 - b_hi, 0.0)
    if db <= 1e-6:
        _add_seg(a_dn, b_hi, 1.0, 0.0)
    else:
        inv_db = 1.0 / db
        b_dn = max(0.0, min(1.0, 2.0 - b_lo))
        a_up = max(b_lo, 1.0)
        _add_seg(a_dn, b_dn, -(2.0 - b_hi) * inv_db, inv_db)
        _add_seg(b_dn, a_up, 1.0, 0.0)
        _add_seg(a_up, b_hi, b_hi * inv_db, -inv_db)

    if m0 <= 1e-300:
        u_top = (math.log(b_hi) - m_u) / l_c
        if m_u > math.log(b_hi):
            return _truncated_upper_std_moments(u_top)
        if a_dn > 0.0:
            mu_neg, v_neg = _truncated_upper_std_moments(-(math.log(a_dn) - m_u) / l_c)
            return -mu_neg, v_neg
        return 0.0, 1.0
    mu_u = m1 / m0
    return mu_u, max(m2 / m0 - mu_u * mu_u, 1e-12)


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
    if u_hi > -20.0:
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
    x = -u_hi
    y_end = min(du, 45.0 / x)
    m0 = 0.0
    n1 = 0.0
    n2 = 0.0
    for gx, gw in zip(_GL8_X01, _GL8_W01):
        yq = y_end * gx
        wt = y_end * gw * math.exp(-yq * (x + 0.5 * yq)) * (yq / du)
        m0 += wt
        n1 += wt * yq
        n2 += wt * yq * yq
    if du * (x + 0.5 * du) < 55.0:
        alpha = x + du
        inv_alpha = 1.0 / alpha
        pref = math.exp(-du * (x + 0.5 * du)) * inv_alpha
        for lx, lw in zip(_LAG6_X, _LAG6_W):
            sq = lx * inv_alpha
            yq = du + sq
            wt = pref * lw * math.exp(-0.5 * sq * sq)
            m0 += wt
            n1 += wt * yq
            n2 += wt * yq * yq
    if m0 <= 1e-300:
        return _truncated_upper_std_moments(u_hi)
    ey = n1 / m0
    return -x - ey, max(n2 / m0 - ey * ey, 1e-12)


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

    ord_f = sorted(
        range(d),
        key=lambda i: (-round(f_in[i], 9) if is_best_of else round(f_in[i], 9), v_in[i], i),
    )
    if w_tie < 1e-12:
        fc, vc, wc, _ = _clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, False)
        return fc, vc, wc

    f_acc = 0.0
    m2_acc = 0.0
    w_acc = [0.0] * d
    if d == 2:
        f_sym, v_sym, w_sym, _ = _clark_one_order(2, f_in, v_in, corr, (0, 1), is_best_of, False)
        m2_sym = f_sym * f_sym * math.exp(v_sym * v_sym)
    elif d == 3:
        for p in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            fp, vp, wp, _ = _clark_one_order(3, f_in, v_in, corr, p, is_best_of, False)
            f_acc += fp
            m2_acc += fp * fp * math.exp(vp * vp)
            for i in range(3):
                w_acc[i] += wp[i]
        inv3 = 1.0 / 3.0
        f_sym = f_acc * inv3
        m2_sym = m2_acc * inv3
        ratio_sym = m2_sym / max(f_sym * f_sym, 1e-30)
        v_sym = math.sqrt(math.log(ratio_sym) if ratio_sym > 1.0 + 1e-16 else 1e-16)
        w_sym = [w_acc[i] * inv3 for i in range(3)]
    elif d == 4:
        trees_d4 = ((0, 1, 2, 3), (2, 3, 0, 1), (0, 2, 1, 3), (1, 3, 0, 2), (0, 3, 1, 2), (1, 2, 0, 3))
        for a, b, c, dd in trees_d4:
            f_ab, v_ab, w_ab_v, rho_ab = _clark_one_order(2, (f_in[a], f_in[b]), (v_in[a], v_in[b]), ((1.0, corr[a][b]), (corr[b][a], 1.0)), (0, 1), is_best_of, False)
            del w_ab_v, rho_ab
            # Balanced 2+2 tree via sequential 1-step merges matching C++ clark_d4_tied.
            va = v_in[a] if v_in[a] > 1e-12 else 1e-12
            vb = v_in[b] if v_in[b] > 1e-12 else 1e-12
            vc = v_in[c] if v_in[c] > 1e-12 else 1e-12
            vd = v_in[dd] if v_in[dd] > 1e-12 else 1e-12
            f_ab, v_ab, wa, wb, rho_ab_cd = _clark_pair_merge_py(
                f_in[a], va, f_in[b], vb, corr[a][b],
                (vc, vd), (corr[a][c], corr[a][dd]), (corr[b][c], corr[b][dd]), is_best_of,
            )
            f_cd, v_cd, w_c_m, w_d_m, rho_cd_ab = _clark_pair_merge_py(
                f_in[c], vc, f_in[dd], vd, corr[c][dd],
                (v_ab,), (rho_ab_cd[0],), (rho_ab_cd[1],), is_best_of,
            )
            f_r, v_r, w_ab, w_cd, _ = _clark_pair_merge_py(
                f_ab, v_ab, f_cd, v_cd, rho_cd_ab[0], (), (), (), is_best_of,
            )
            f_acc += f_r
            m2_acc += f_r * f_r * math.exp(v_r * v_r)
            w_acc[a] += w_ab * wa
            w_acc[b] += w_ab * wb
            w_acc[c] += w_cd * w_c_m
            w_acc[dd] += w_cd * w_d_m
        inv6 = 1.0 / 6.0
        f_sym = f_acc * inv6
        m2_sym = m2_acc * inv6
        ratio_sym = m2_sym / max(f_sym * f_sym, 1e-30)
        v_sym = math.sqrt(math.log(ratio_sym) if ratio_sym > 1.0 + 1e-16 else 1e-16)
        w_sym = [w_acc[i] * inv6 for i in range(4)]
    elif d == 5:
        trees_d4 = ((0, 1, 2, 3), (2, 3, 0, 1), (0, 2, 1, 3), (1, 3, 0, 2), (0, 3, 1, 2), (1, 2, 0, 3))
        for e in range(5):
            rem4 = [i for i in range(5) if i != e]
            ve = v_in[e] if v_in[e] > 1e-12 else 1e-12
            for t in trees_d4:
                a, b, c, dd = rem4[t[0]], rem4[t[1]], rem4[t[2]], rem4[t[3]]
                va = v_in[a] if v_in[a] > 1e-12 else 1e-12
                vb = v_in[b] if v_in[b] > 1e-12 else 1e-12
                vc = v_in[c] if v_in[c] > 1e-12 else 1e-12
                vd = v_in[dd] if v_in[dd] > 1e-12 else 1e-12
                f_ab, v_ab, wa, wb, rho_ab_cde = _clark_pair_merge_py(
                    f_in[a], va, f_in[b], vb, corr[a][b],
                    (vc, vd, ve), (corr[a][c], corr[a][dd], corr[a][e]), (corr[b][c], corr[b][dd], corr[b][e]), is_best_of,
                )
                f_cd, v_cd, w_c_m, w_d_m, rho_cd_abe = _clark_pair_merge_py(
                    f_in[c], vc, f_in[dd], vd, corr[c][dd],
                    (v_ab, ve), (rho_ab_cde[0], corr[c][e]), (rho_ab_cde[1], corr[dd][e]), is_best_of,
                )
                f_abcd, v_abcd, w_ab, w_cd, rho_abcd_e = _clark_pair_merge_py(
                    f_ab, v_ab, f_cd, v_cd, rho_cd_abe[0],
                    (ve,), (rho_ab_cde[2],), (rho_cd_abe[1],), is_best_of,
                )
                f_r, v_r, w_abcd, w_e, _ = _clark_pair_merge_py(
                    f_abcd, v_abcd, f_in[e], ve, rho_abcd_e[0], (), (), (), is_best_of,
                )
                f_acc += f_r
                m2_acc += f_r * f_r * math.exp(v_r * v_r)
                w_acc[a] += w_abcd * w_ab * wa
                w_acc[b] += w_abcd * w_ab * wb
                w_acc[c] += w_abcd * w_cd * w_c_m
                w_acc[dd] += w_abcd * w_cd * w_d_m
                w_acc[e] += w_e
        inv30 = 1.0 / 30.0
        f_sym = f_acc * inv30
        m2_sym = m2_acc * inv30
        ratio_sym = m2_sym / max(f_sym * f_sym, 1e-30)
        v_sym = math.sqrt(math.log(ratio_sym) if ratio_sym > 1.0 + 1e-16 else 1e-16)
        w_sym = [w_acc[i] * inv30 for i in range(5)]
    else:
        perms: List[List[int]] = []
        for s in range(d):
            p_rot = [(s + k) % d for k in range(d)]
            perms.append(p_rot)
            perms.append(list(reversed(p_rot)))
        for p_seq in perms:
            fp, vp, wp, _ = _clark_one_order(d, f_in, v_in, corr, p_seq, is_best_of, False)
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


def _clark_pair_merge_py(
    f1_in: float,
    v1_in: float,
    f2_in: float,
    v2_in: float,
    rho_12: float,
    v_rem: Sequence[float],
    r1_rem: Sequence[float],
    r2_rem: Sequence[float],
    is_best_of: bool,
) -> Tuple[float, float, float, float, List[float]]:
    f1 = f1_in if f1_in > 1e-16 else 1e-16
    v1 = v1_in if v1_in > 1e-12 else 1e-12
    f2 = f2_in if f2_in > 1e-16 else 1e-16
    v2 = v2_in if v2_in > 1e-12 else 1e-12
    rho_c = min(max(rho_12, -0.999999), 0.999999)
    var_m = max(v1 * v1 + v2 * v2 - 2.0 * rho_c * v1 * v2, 1e-16)
    sig_m = math.sqrt(var_m)
    log_fr = math.log(f1 / f2)
    d1 = (log_fr + 0.5 * var_m) / sig_m
    d2 = d1 - sig_m
    w1 = _ndtr_std(d1) if is_best_of else _ndtr_std(-d1)
    w2 = _ndtr_std(-d2) if is_best_of else _ndtr_std(d2)
    e_m1 = max(f1 * w1 + f2 * w2, 1e-16)
    f1_sq = (f1 * f1) * math.exp(v1 * v1)
    f2_sq = (f2 * f2) * math.exp(v2 * v2)
    sig_m2 = 2.0 * sig_m
    var_m2 = sig_m2 * sig_m2
    d1_sq = (math.log(max(f1_sq, 1e-30) / max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2
    d2_sq = d1_sq - sig_m2
    e_m2 = (
        f1_sq * _ndtr_std(d1_sq) + f2_sq * _ndtr_std(-d2_sq)
        if is_best_of
        else f1_sq * _ndtr_std(-d1_sq) + f2_sq * _ndtr_std(d2_sq)
    )
    ratio_m2 = e_m2 / max(e_m1 * e_m1, 1e-30)
    v_new = math.sqrt(math.log(ratio_m2) if ratio_m2 > 1.0 + 1e-16 else 1e-16)
    rho_out: List[float] = []
    for k in range(len(v_rem)):
        vk = v_rem[k] if v_rem[k] > 1e-12 else 1e-12
        ea1 = r1_rem[k] * v1 * vk
        ea2 = r2_rem[k] * v2 * vk
        f1_k = f1 * math.exp(ea1)
        f2_k = f2 * math.exp(ea2)
        d1_k = (log_fr + (ea1 - ea2) + 0.5 * var_m) / sig_m
        d2_k = d1_k - sig_m
        e_mp = (
            f1_k * _ndtr_std(d1_k) + f2_k * _ndtr_std(-d2_k)
            if is_best_of
            else f1_k * _ndtr_std(-d1_k) + f2_k * _ndtr_std(d2_k)
        )
        cov_log = math.log(max(e_mp / e_m1, 1e-16))
        rho_out.append(min(max(cov_log / max(v_new * vk, 1e-16), -0.999999), 0.999999))
    return e_m1, v_new, w1, w2, rho_out


def _clark_corr_min_max(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
) -> float:
    if d <= 1:
        return 1.0
    ord_min = sorted(range(d), key=lambda i: (round(f_in[i], 9), v_in[i], i))
    ord_max = sorted(range(d), key=lambda i: (-round(f_in[i], 9), v_in[i], i))
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


def _fit_1factor_loadings_py(d: int, corr: Sequence[Sequence[float]]) -> List[float]:
    if d <= 1:
        return [0.0] if d == 1 else []
    sum_corr = sum(float(corr[i][j]) for i in range(d) for j in range(d))
    rho_off = min(max((sum_corr - float(d)) / float(d * (d - 1)), 0.0), 0.999)
    b = [math.sqrt(rho_off)] * d
    for _ in range(8):
        mb = [0.0] * d
        for i in range(d):
            row_i = corr[i]
            acc = b[i] ** 3
            for j in range(d):
                if j != i:
                    acc += float(row_i[j]) * b[j]
            mb[i] = acc
        n2 = sum(x * x for x in mb)
        inv_n = 1.0 / math.sqrt(max(n2, 1e-30))
        mb = [x * inv_n for x in mb]
        if sum(mb) < 0.0:
            mb = [-x for x in mb]
        lam = 0.0
        for i in range(d):
            row_i = corr[i]
            mu_i = (b[i] ** 2) * mb[i] + sum(float(row_i[j]) * mb[j] for j in range(d) if j != i)
            lam += mb[i] * mu_i
        s_lam = math.sqrt(max(lam, 0.0))
        b = [min(max(mb[i] * s_lam, 0.0), 0.999) for i in range(d)]
    return b


def _eval_index_moments(
    d: int,
    f_in: Sequence[float],
    v_in: Sequence[float],
    corr: Sequence[Sequence[float]],
    inv_lex_perm: Optional[Sequence[int]],
    code: int,
) -> Tuple[float, float, List[float]]:
    if d <= 0:
        return 0.0, 1e-12, []
    if d == 1:
        if 9 <= code <= 13:
            return 0.0, 1e-12, [0.0]
        return (f_in[0] if f_in[0] > 1e-16 else 1e-16), (v_in[0] if v_in[0] > 1e-12 else 1e-12), [1.0]

    def _ab() -> Tuple[int, int]:
        if inv_lex_perm is not None and len(inv_lex_perm) >= 2:
            ia_v = min(max(int(inv_lex_perm[0]), 0), d - 1)
            ib_v = min(max(int(inv_lex_perm[1]), 0), d - 1)
            return ia_v, ib_v
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
            vi = v_in[i] if v_in[i] > 1e-12 else 1e-12
            for j in range(d):
                vj = v_in[j] if v_in[j] > 1e-12 else 1e-12
                m2_sum += f_in[i] * f_in[j] * math.exp(corr[i][j] * vi * vj)
        m2 = m2_sum * (inv_d * inv_d)
        ratio = m2 / max(f_out * f_out, 1e-30)
        v_out = math.sqrt(math.log(ratio) if ratio > 1.0 + 1e-16 else 1e-16)
        return f_out, v_out, [inv_d] * d
    if code == 3:
        inv_d = 1.0 / float(d)
        v_pos = [v_in[i] if v_in[i] > 1e-12 else 1e-12 for i in range(d)]
        m2_p = sum(f_in[i] * f_in[i] * math.exp(v_pos[i] * v_pos[i]) for i in range(d))
        m2_avg = sum(f_in[i] * f_in[j] * math.exp(corr[i][j] * v_pos[i] * v_pos[j]) for i in range(d) for j in range(d))
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
                vi = v_in[i] if v_in[i] > 1e-12 else 1e-12
                for j in range(d):
                    vj = v_in[j] if v_in[j] > 1e-12 else 1e-12
                    cij = corr[i][j] * vi * vj
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
        va = v_in[ia] if v_in[ia] > 1e-12 else 1e-12
        vb = v_in[ib] if v_in[ib] > 1e-12 else 1e-12
        rab = corr[ia][ib]
        var_d = fa * fa * math.expm1(va * va) + fb * fb * math.expm1(vb * vb) - 2.0 * fa * fb * math.expm1(rab * va * vb)
        w_out = [0.0] * d
        w_out[ia] = 1.0
        w_out[ib] = -1.0
        return fa - fb, math.sqrt(max(var_d, 1e-16)), w_out

    if code in (11, 12, 13):
        if code == 11:
            f_a, v_a, w_a = _clark_symmetrized(d, f_in, v_in, corr, True)
            f_b, v_b, w_b = _clark_symmetrized(d, f_in, v_in, corr, False)
            rho_ab = _clark_corr_min_max(d, f_in, v_in, corr)
        elif code == 12:
            f_a, v_a, w_a = _clark_symmetrized(d, f_in, v_in, corr, True)
            f_b, v_b, w_b = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
        else:
            f_a, v_a, w_a = _eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2)
            f_b, v_b, w_b = _clark_symmetrized(d, f_in, v_in, corr, False)
        if code in (12, 13):
            cov = 0.0
            va2 = 0.0
            vb2 = 0.0
            for i in range(d):
                vi = v_in[i] if v_in[i] > 1e-12 else 1e-12
                for j in range(d):
                    vj = v_in[j] if v_in[j] > 1e-12 else 1e-12
                    cij = corr[i][j] * vi * vj
                    cov += w_a[i] * w_b[j] * cij
                    va2 += w_a[i] * w_a[j] * cij
                    vb2 += w_b[i] * w_b[j] * cij
            rho_ab = min(max(cov / math.sqrt(max(va2 * vb2, 1e-30)), -0.9999), 0.9999)
        var_d = f_a * f_a * math.expm1(v_a * v_a) + f_b * f_b * math.expm1(v_b * v_b) - 2.0 * f_a * f_b * math.expm1(rho_ab * v_a * v_b)
        return f_a - f_b, math.sqrt(max(var_d, 1e-16)), [w_a[i] - w_b[i] for i in range(d)]

    if code in (15, 16, 17):
        target_k = min(1, d - 1) if code == 15 else (min(2, d - 1) if code == 16 else max(0, d - 2))
        if target_k == 0:
            return _clark_symmetrized(d, f_in, v_in, corr, False)
        if target_k == d - 1:
            return _clark_symmetrized(d, f_in, v_in, corr, True)
        b = _fit_1factor_loadings_py(d, corr)
        fi_arr = [f_in[i] if f_in[i] > 1e-16 else 1e-16 for i in range(d)]
        vi_arr = [v_in[i] if v_in[i] > 1e-12 else 1e-12 for i in range(d)]
        mu_os = [math.log(fi_arr[i]) - 0.5 * vi_arr[i] ** 2 for i in range(d)]
        vb_os = [vi_arr[i] * b[i] for i in range(d)]
        vs_os = [max(vi_arr[i] * math.sqrt(max(1.0 - b[i] * b[i], 1e-12)), 1e-12) for i in range(d)]
        w_out = [0.0] * d
        f_sum = 0.0
        m2_sum = 0.0
        for qy in range(16):
            y = _GL16_YNORM[qy]
            wy = 0.5 * _GL16_W[qy]
            for i in range(d):
                base_shift = -0.5 * vi_arr[i] ** 2 + vb_os[i] * y
                for qz in range(16):
                    z = _GL16_YNORM[qz]
                    wz = 0.5 * _GL16_W[qz]
                    lx = mu_os[i] + vb_os[i] * y + vs_os[i] * z
                    unit_p = math.exp(base_shift + vs_os[i] * z)
                    p_val = fi_arr[i] * unit_p
                    dp = [0.0] * (target_k + 1)
                    dp[0] = 1.0
                    lim = 0
                    for j in range(d):
                        if j == i:
                            continue
                        prob = _ndtr_scalar((lx - mu_os[j] - vb_os[j] * y) / vs_os[j])
                        qrob = 1.0 - prob
                        next_lim = min(lim + 1, target_k)
                        for r in range(next_lim, 0, -1):
                            dp[r] = dp[r] * qrob + dp[r - 1] * prob
                        dp[0] *= qrob
                        lim = next_lim
                    pr = wy * wz * dp[target_k]
                    w_out[i] += pr * unit_p
                    f_sum += pr * p_val
                    m2_sum += pr * p_val * p_val
        f_out = max(f_sum, 1e-16)
        ratio = m2_sum / max(f_out * f_out, 1e-30)
        v_out = math.sqrt(math.log(ratio) if ratio > 1.0 + 1e-16 else 1e-16)
        return f_out, v_out, w_out

    return _clark_symmetrized(d, f_in, v_in, corr, False)


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
        vi = v_in[i] if v_in[i] > 1e-12 else 1e-12
        wai = w_a[i] * (f_in[i] if da_diff else 1.0)
        wbi = w_b[i] * (f_in[i] if db_diff else 1.0)
        for j in range(d):
            vj = v_in[j] if v_in[j] > 1e-12 else 1e-12
            waj = w_a[j] * (f_in[j] if da_diff else 1.0)
            wbj = w_b[j] * (f_in[j] if db_diff else 1.0)
            cij = corr[i][j] * vi * vj
            num += wai * wbj * cij
            da += wai * waj * cij
            db += wbi * wbj * cij
    return min(max(num / math.sqrt(max(da * db, 1e-30)), -0.9999), 0.9999)


def _lognormal_moments_interval(f: float, v: float, a: float, b: float, max_order: int) -> List[float]:
    if max_order < 0:
        return []
    max_order = min(max_order, 4)
    if b <= a or b <= 0.0:
        return [0.0] * (max_order + 1)
    f_c = f if f > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    m_u = math.log(f_c) - 0.5 * v_c * v_c
    log_b = math.log(b)
    u_b = (log_b - m_u) / v_c
    has_a = a > 0.0
    log_a = math.log(a) if has_a else 0.0
    u_a = (log_a - m_u) / v_c if has_a else 0.0
    out: List[float] = []
    fp = 1.0
    for k in range(max_order + 1):
        kf = float(k)
        zb = u_b - kf * v_c
        za = (u_a - kf * v_c) if has_a else 0.0
        if not has_a and zb <= -6.0:
            out.append(0.5 * math.exp(kf * log_b - 0.5 * u_b * u_b) * _cody_erfcx(-zb * _INV_SQRT2))
        elif has_a and zb <= -6.0:
            exp_ba = math.exp(-0.5 * (za - zb) * (za + zb))
            br = _cody_erfcx(-zb * _INV_SQRT2) - exp_ba * _cody_erfcx(-za * _INV_SQRT2)
            out.append(0.0 if br <= 0.0 else (0.5 * math.exp(kf * log_b - 0.5 * u_b * u_b) * br))
        elif has_a and za >= 6.0:
            exp_ab = math.exp(-0.5 * (zb - za) * (zb + za))
            br = _cody_erfcx(za * _INV_SQRT2) - exp_ab * _cody_erfcx(zb * _INV_SQRT2)
            out.append(0.0 if br <= 0.0 else (0.5 * math.exp(kf * log_a - 0.5 * u_a * u_a) * br))
        else:
            diff = (_ndtr_std(zb) - _ndtr_std(za)) if has_a else _ndtr_std(zb)
            if diff <= 0.0:
                out.append(0.0)
            else:
                scale = fp * math.exp(0.5 * float(k * (k - 1)) * v_c * v_c)
                out.append(scale * diff)
        fp *= f_c
    return out


def _normal_moments_interval(mu: float, sig: float, a: float, b: float, max_order: int) -> List[float]:
    if max_order < 0:
        return []
    if b <= a:
        return [0.0] * (max_order + 1)
    s_c = sig if sig > 1e-12 else 1e-12
    u_b = (b - mu) / s_c
    has_a = a > -1e50
    u_a = (a - mu) / s_c if has_a else 0.0
    if u_b <= -15.0:
        x = -u_b
        pref = _INV_SQRT2PI * math.exp(-0.5 * x * x)
        if pref <= 0.0:
            return [0.0] * (max_order + 1)
        du = (u_b - u_a) if has_a else 1e100
        zm = [0.0, 0.0, 0.0, 0.0]
        if has_a and du * x < 45.0:
            for gx, gw in zip(_GL8_X01, _GL8_W01):
                zq = du * gx
                wt = pref * du * gw * math.exp(-zq * (x + 0.5 * zq))
                zm[0] += wt
                zm[1] += wt * zq
                zm[2] += wt * zq * zq
                zm[3] += wt * zq * zq * zq
        else:
            inv_x = 1.0 / x
            p_scale = pref * inv_x
            for lx, lw in zip(_LAG6_X, _LAG6_W):
                zq = lx * inv_x
                wt = p_scale * lw * math.exp(-0.5 * zq * zq)
                zm[0] += wt
                zm[1] += wt * zq
                zm[2] += wt * zq * zq
                zm[3] += wt * zq * zq * zq
        out = [zm[0]]
        if max_order >= 1:
            out.append(b * zm[0] - s_c * zm[1])
        if max_order >= 2:
            out.append(b * b * zm[0] - 2.0 * b * s_c * zm[1] + s_c * s_c * zm[2])
        if max_order >= 3:
            out.append(b * b * b * zm[0] - 3.0 * b * b * s_c * zm[1] + 3.0 * b * s_c * s_c * zm[2] - s_c * s_c * s_c * zm[3])
        return out
    if has_a and u_a >= 15.0:
        out = _normal_moments_interval(-mu, s_c, -b, -a, max_order)
        if max_order >= 1:
            out[1] = -out[1]
        if max_order >= 3:
            out[3] = -out[3]
        return out
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
    has_lower_strike: Optional[bool] = None,
) -> float:
    if (not is_normal) and strike <= 1e-8:
        return 0.0
    use_low = (math.isfinite(lower_strike) and lower_strike > 0.0) if has_lower_strike is None else bool(has_lower_strike)
    k_low = lower_strike if use_low else 0.0
    if use_low and k_low >= strike:
        return 0.0
    low_bound = -1e60 if is_normal else 0.0
    pts: List[float] = [low_bound, strike]
    if use_low and k_low > low_bound:
        pts.append(k_low)
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            pts.append(ki_barrier)
        else:
            pts.append(min(ki_barrier, ki_barrier + ki_spread))
            pts.append(max(ki_barrier, ki_barrier + ki_spread))
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
        if use_low and mid <= k_low:
            c0, c1 = strike - k_low, 0.0
        else:
            c0, c1 = strike, -1.0

        d0, d1 = 1.0, 0.0
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if mid >= ki_barrier:
                    continue
            else:
                b_ki_lo = min(ki_barrier, ki_barrier + ki_spread)
                b_ki_hi = max(ki_barrier, ki_barrier + ki_spread)
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


def _exact_twinwin_terminal_put(
    f: float,
    v: float,
    strike: float,
    has_lower_strike: bool,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    has_expiry_ko: bool,
    ko_barrier: float,
    ko_spread: float,
) -> float:
    if strike <= 1.0:
        return 0.0
    k_low = lower_strike if has_lower_strike else 0.0
    if has_lower_strike and k_low >= strike:
        return 0.0
    b_ki_lo, b_ki_hi = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
    b_ko_lo, b_ko_hi = min(ko_barrier, ko_barrier + ko_spread), max(ko_barrier, ko_barrier + ko_spread)
    x_dn_min = max(2.0 - strike, 0.0)
    pts: List[float] = [x_dn_min, 1.0, strike]
    if has_lower_strike and k_low > 1.0:
        pts.extend((k_low, 2.0 - k_low))
    if has_active_ki:
        pts.extend((b_ki_lo, b_ki_hi, 2.0 - b_ki_hi, 2.0 - b_ki_lo))
    if has_expiry_ko:
        pts.extend((b_ko_lo, b_ko_hi, 2.0 - b_ko_hi, 2.0 - b_ko_lo))
    pts = sorted(x for x in pts if x_dn_min <= x <= strike)
    total = 0.0
    for idx in range(len(pts) - 1):
        a, b = pts[idx], pts[idx + 1]
        if b - a <= 1e-14:
            continue
        mid = 0.5 * (a + b)
        on_left = mid < 1.0
        w_tw = (2.0 - mid) if on_left else mid
        if w_tw >= strike:
            continue
        if has_lower_strike and w_tw <= k_low:
            c0, c1 = strike - k_low, 0.0
        elif on_left:
            c0, c1 = strike - 2.0, 1.0
        else:
            c0, c1 = strike, -1.0
        d0, d1 = 1.0, 0.0
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if w_tw >= ki_barrier:
                    continue
            else:
                if w_tw >= b_ki_hi:
                    continue
                if w_tw > b_ki_lo:
                    db = b_ki_hi - b_ki_lo
                    d0 = ((b_ki_hi - 2.0) / db) if on_left else (b_ki_hi / db)
                    d1 = (1.0 / db) if on_left else (-1.0 / db)
        e0, e1 = 1.0, 0.0
        if has_expiry_ko:
            if abs(ko_spread) <= 1e-6:
                if w_tw >= ko_barrier:
                    continue
            else:
                if w_tw >= b_ko_hi:
                    continue
                if w_tw > b_ko_lo:
                    db_k = b_ko_hi - b_ko_lo
                    e0 = ((b_ko_hi - 2.0) / db_k) if on_left else (b_ko_hi / db_k)
                    e1 = (1.0 / db_k) if on_left else (-1.0 / db_k)
        cd0, cd1, cd2 = c0 * d0, c0 * d1 + c1 * d0, c1 * d1
        q0, q1, q2, q3 = cd0 * e0, cd0 * e1 + cd1 * e0, cd1 * e1 + cd2 * e0, cd2 * e1
        max_k = 3 if q3 != 0.0 else (2 if q2 != 0.0 else 1)
        moms = _lognormal_moments_interval(f, v, a, b, max_k)
        total += q0 * moms[0] + q1 * moms[1] + (q2 * moms[2] if max_k >= 2 else 0.0) + (q3 * moms[3] if max_k >= 3 else 0.0)
    return total


def _exact_twinwin_terminal_call(
    f: float,
    v: float,
    k_call: float,
    has_expiry_ko: bool,
    ko_barrier: float,
    ko_spread: float,
) -> float:
    if not has_expiry_ko:
        total = 0.0
        b_left = min(1.0, max(2.0 - k_call, 0.0))
        if b_left > 1e-14:
            m = _lognormal_moments_interval(f, v, 0.0, b_left, 1)
            total += (2.0 - k_call) * m[0] - m[1]
        if k_call < 1.0:
            f_c = max(f, 1e-16)
            v_c = max(v, 1e-12)
            p_ge1 = _ndtr_std((math.log(f_c) - 0.5 * v_c * v_c) / v_c)
            total += _black76_call(f, 1.0, v) + (1.0 - k_call) * p_ge1
        else:
            total += _black76_call(f, k_call, v)
        return total
    b_ko_lo, b_ko_hi = min(ko_barrier, ko_barrier + ko_spread), max(ko_barrier, ko_barrier + ko_spread)
    if k_call >= b_ko_hi or b_ko_hi <= 1.0:
        return 0.0
    pts: List[float] = [0.0, 1.0, b_ko_hi]
    if 2.0 - b_ko_hi > 0.0:
        pts.append(2.0 - b_ko_hi)
    if k_call > 1.0:
        pts.append(k_call)
        if 2.0 - k_call > 0.0:
            pts.append(2.0 - k_call)
    if abs(ko_spread) > 1e-6:
        pts.append(b_ko_lo)
        if 2.0 - b_ko_lo > 0.0:
            pts.append(2.0 - b_ko_lo)
    pts = sorted(x for x in pts if 0.0 <= x <= b_ko_hi)
    total = 0.0
    for idx in range(len(pts) - 1):
        a, b = pts[idx], pts[idx + 1]
        if b - a <= 1e-14:
            continue
        mid = 0.5 * (a + b)
        on_left = mid < 1.0
        w_tw = (2.0 - mid) if on_left else mid
        if w_tw <= k_call:
            continue
        e0, e1 = 1.0, 0.0
        if abs(ko_spread) <= 1e-6:
            if w_tw >= ko_barrier:
                continue
        else:
            if w_tw >= b_ko_hi:
                continue
            if w_tw > b_ko_lo:
                db_k = b_ko_hi - b_ko_lo
                e0 = ((b_ko_hi - 2.0) / db_k) if on_left else (b_ko_hi / db_k)
                e1 = (1.0 / db_k) if on_left else (-1.0 / db_k)
        c0 = (2.0 - k_call) if on_left else -k_call
        c1 = -1.0 if on_left else 1.0
        q0, q1, q2 = c0 * e0, c0 * e1 + c1 * e0, c1 * e1
        moms = _lognormal_moments_interval(f, v, a, b, 2 if q2 != 0.0 else 1)
        total += q0 * moms[0] + q1 * moms[1] + (q2 * moms[2] if q2 != 0.0 else 0.0)
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
            blo = min(ki_barrier, ki_barrier + ki_spread)
            bhi = max(ki_barrier, ki_barrier + ki_spread)
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
) -> Tuple[float, float, float]:
    l_put_out = 0.0
    l_onestar_out = 0.0
    p_onestar_out = 0.0
    has_lower_strike = math.isfinite(lower_strike) and (lower_strike > 0.0)
    k_low = lower_strike if has_lower_strike else 0.0
    log_ks = math.log(max(one_star_strike, 1e-16))

    if d == 1:
        eps_os = 0.02
        f0 = max(f_c[0], 1e-16)
        v0 = max(v_c[0], 1e-12)
        os_lo = max(one_star_strike - 0.5 * eps_os, 0.0)
        os_hi = one_star_strike + 0.5 * eps_os
        inv_os_eps = 1.0 / (os_hi - os_lo)
        surv_mult = surv_prev_term if use_direct_ko else surv_cur

        if one_star_strike > 0.0:
            blo_ko = min(ko_b_term, ko_b_term + ko_spread)
            bhi_ko = max(ko_b_term, ko_b_term + ko_spread)
            hard_ko = abs(ko_spread) <= 1e-6
            ko_top = (ko_b_term if hard_ko else bhi_ko) if use_direct_ko else 1e30
            if os_lo < ko_top:
                pts_os: List[float] = [os_lo, ko_top]
                if os_hi < ko_top:
                    pts_os.append(os_hi)
                if use_direct_ko and not hard_ko and os_lo < blo_ko < ko_top:
                    pts_os.append(blo_ko)
                pts_os.sort()
                p_os = 0.0
                m_u = math.log(f0) - 0.5 * v0 * v0
                for idx in range(len(pts_os) - 1):
                    a = pts_os[idx]
                    b_pt = pts_os[idx + 1]
                    if b_pt - a <= 1e-14:
                        continue
                    mid = (a + 1.0) if b_pt > 1e20 else 0.5 * (a + b_pt)
                    s0, s1 = (-os_lo * inv_os_eps, inv_os_eps) if mid < os_hi else (1.0, 0.0)
                    if use_direct_ko and not hard_ko and mid > blo_ko:
                        db = bhi_ko - blo_ko
                        e0, e1 = bhi_ko / db, -1.0 / db
                    else:
                        e0, e1 = 1.0, 0.0
                    q0 = s0 * e0
                    q1 = s0 * e1 + s1 * e0
                    q2 = s1 * e1
                    if b_pt > 1e20:
                        p_os += q0 * _ndtr_std(-(math.log(max(a, 1e-16)) - m_u) / v0)
                    else:
                        moms = _lognormal_moments_interval(f0, v0, a, b_pt, 2 if q2 != 0.0 else 1)
                        p_os += q0 * moms[0] + q1 * moms[1] + (q2 * moms[2] if q2 != 0.0 else 0.0)
                p_onestar_out = surv_mult * max(p_os, 0.0)
                l_onestar_out = p_onestar_out * one_star_coupon

        if strike > 1e-6 and option_gearing > 0.0 and one_star_strike > 1e-16:
            if os_lo >= strike:
                l_put_out = surv_mult * _exact_piecewise_terminal_put(
                    f0, v0, False, strike, lower_strike,
                    has_active_ki, ki_barrier, ki_spread,
                    use_direct_ko, ko_b_term, ko_spread,
                    has_lower_strike=has_lower_strike,
                )
                return l_put_out, l_onestar_out, p_onestar_out

            hi = min(strike, os_hi)
            pts_p: List[float] = [0.0, hi]
            if has_lower_strike and k_low > 0.0:
                pts_p.append(k_low)
            if os_lo > 0.0:
                pts_p.append(os_lo)
            if has_active_ki:
                if abs(ki_spread) <= 1e-6:
                    pts_p.append(ki_barrier)
                else:
                    pts_p.extend((min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)))
            if use_direct_ko:
                if abs(ko_spread) <= 1e-6:
                    pts_p.append(ko_b_term)
                else:
                    pts_p.extend((min(ko_b_term, ko_b_term + ko_spread), max(ko_b_term, ko_b_term + ko_spread)))
            pts_p = sorted(x for x in pts_p if 0.0 <= x <= hi)
            total_put = 0.0
            for idx in range(len(pts_p) - 1):
                a = pts_p[idx]
                b_pt = pts_p[idx + 1]
                if b_pt - a <= 1e-14:
                    continue
                mid = 0.5 * (a + b_pt)
                c0, c1 = ((strike - k_low), 0.0) if (has_lower_strike and mid <= k_low) else (strike, -1.0)
                d0, d1 = 1.0, 0.0
                if has_active_ki:
                    if abs(ki_spread) <= 1e-6:
                        if mid >= ki_barrier:
                            continue
                    else:
                        blo, bhi = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
                        if mid >= bhi:
                            continue
                        if mid > blo:
                            d0, d1 = bhi / (bhi - blo), -1.0 / (bhi - blo)
                e0, e1 = 1.0, 0.0
                if use_direct_ko:
                    if abs(ko_spread) <= 1e-6:
                        if mid >= ko_b_term:
                            continue
                    else:
                        blo, bhi = min(ko_b_term, ko_b_term + ko_spread), max(ko_b_term, ko_b_term + ko_spread)
                        if mid >= bhi:
                            continue
                        if mid > blo:
                            e0, e1 = bhi / (bhi - blo), -1.0 / (bhi - blo)
                r0, r1 = (os_hi * inv_os_eps, -inv_os_eps) if mid > os_lo else (1.0, 0.0)
                cd0, cd1, cd2 = c0 * d0, c0 * d1 + c1 * d0, c1 * d1
                cde0, cde1, cde2, cde3 = cd0 * e0, cd0 * e1 + cd1 * e0, cd1 * e1 + cd2 * e0, cd2 * e1
                q0, q1 = cde0 * r0, cde0 * r1 + cde1 * r0
                q2, q3, q4 = cde1 * r1 + cde2 * r0, cde2 * r1 + cde3 * r0, cde3 * r1
                order = 4 if q4 != 0.0 else (3 if q3 != 0.0 else (2 if q2 != 0.0 else 1))
                moms = _lognormal_moments_interval(f0, v0, a, b_pt, order)
                seg_v = q0 * moms[0] + q1 * moms[1]
                if order >= 2:
                    seg_v += q2 * moms[2]
                if order >= 3:
                    seg_v += q3 * moms[3]
                if order >= 4:
                    seg_v += q4 * moms[4]
                total_put += seg_v
            l_put_out = surv_mult * max(total_put, 0.0)
        return l_put_out, l_onestar_out, p_onestar_out

    b = _fit_1factor_loadings_py(d, corr)

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

    if one_star_strike > 0.0:
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
            p_onestar_out = surv_prev_term * os_acc
        else:
            for q in range(16):
                os_acc += 0.5 * _GL16_W[q] * (1.0 - p_max_lt_ks_y[q])
            p_onestar_out = surv_cur * os_acc
        l_onestar_out = p_onestar_out * one_star_coupon

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
                for p_v in (min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)):
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
            next_span = (pts_list[idx + 2] - b_pt) if (idx + 2 < len(pts_list)) else 1.0
            probe_right = b_pt + min(1e-10, 0.25 * max(next_span, 0.0))
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
                probe_right,
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

    return l_put_out, l_onestar_out, p_onestar_out


class PureNumpyTier1Backend:
    """Zero-dependency pure NumPy/Python IEEE-754 float64 Tier-1 backend."""

    @property
    def name(self) -> str:
        return "numpy"

    def eval_index_moments_np(
        self,
        f_arr: npt.NDArray[np.float64],
        v_arr: npt.NDArray[np.float64],
        corr: npt.NDArray[np.float64],
        code: int,
        inv_lex_perm: Optional[npt.NDArray[np.int64]] = None,
    ) -> Tuple[float, float, npt.NDArray[np.float64]]:
        fp = np.ascontiguousarray(f_arr, dtype=np.float64)
        vp = np.ascontiguousarray(v_arr, dtype=np.float64)
        cm = np.ascontiguousarray(corr, dtype=np.float64)
        if fp.ndim != 1 or vp.shape != fp.shape:
            raise ValueError(f"Expected matching 1D f_arr and v_arr, got {fp.shape} vs {vp.shape}.")
        d_assets = int(fp.shape[0])
        if d_assets == 0:
            return 0.0, 0.0, np.zeros(0, dtype=np.float64)
        if cm.ndim != 2 or cm.shape != (d_assets, d_assets):
            raise ValueError(f"Expected corr shape ({d_assets}, {d_assets}), got {cm.shape}.")
        perm_list = [int(x) for x in inv_lex_perm.tolist()] if inv_lex_perm is not None else []
        mu_i, sig_i, w_i = _eval_index_moments(d_assets, fp.tolist(), vp.tolist(), cm.tolist(), perm_list, int(code))
        return float(mu_i), float(sig_i), np.asarray(w_i, dtype=np.float64)

    def clamped_cubic_coeffs_1d(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        left_slope: float = 0.0,
        right_slope: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("clamped_cubic_coeffs_1d expects matching 1D arrays x and y.")
        k = int(x_arr.shape[0])
        if k < 2:
            raise ValueError("clamped_cubic_coeffs_1d requires at least 2 knots.")
        n_seg = k - 1
        h = np.zeros(n_seg, dtype=np.float64)
        s = np.zeros(n_seg, dtype=np.float64)
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

        coeffs = np.zeros((4, n_seg), dtype=np.float64)
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
        if x_arr.ndim != 1 or xq_arr.ndim != 1:
            raise ValueError("eval_cubic_with_derivs expects 1D x and x_query arrays.")
        k = int(x_arr.shape[0])
        n_seg = k - 1
        n_q = int(xq_arr.shape[0])
        val_out = np.zeros(n_q, dtype=np.float64)
        dval_out = np.zeros(n_q, dtype=np.float64)
        d2val_out = np.zeros(n_q, dtype=np.float64)
        if k < 2 or n_q == 0:
            return val_out, dval_out, d2val_out
        if c_arr.ndim != 2 or c_arr.shape != (4, n_seg):
            raise ValueError(f"eval_cubic_with_derivs expects coeffs shape (4, {n_seg}), got {c_arr.shape}.")

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
            if math.isnan(xq):
                val_out[q] = math.nan
                dval_out[q] = math.nan
                d2val_out[q] = math.nan
                continue
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
        if fwd_perf.ndim != 2 or vol_perf.shape != fwd_perf.shape:
            raise ValueError(f"Expected matching 2D fwd_perf and vol_perf, got {fwd_perf.shape} vs {vol_perf.shape}.")
        d_assets, m_rem = int(fwd_perf.shape[0]), int(fwd_perf.shape[1])
        f_w = np.zeros(m_rem, dtype=np.float64)
        v_w = np.zeros(m_rem, dtype=np.float64)
        delta_w = np.zeros((d_assets, m_rem), dtype=np.float64)
        p_stop = np.zeros(m_rem, dtype=np.float64)
        surv = np.zeros(m_rem, dtype=np.float64)
        h_rate = np.zeros(m_rem, dtype=np.float64)
        q_cpn = np.zeros(m_rem, dtype=np.float64)
        scalars = np.zeros(8, dtype=np.float64)
        if m_rem == 0 or d_assets == 0:
            scalars[4] = float(pv_in_transit)
            return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars
        if corr.ndim != 2 or corr.shape != (d_assets, d_assets):
            raise ValueError(f"Expected corr shape ({d_assets}, {d_assets}), got {corr.shape}.")
        for arr_name, sched_arr in (
            ("df_pay", df_pay),
            ("ko_barriers", ko_barriers),
            ("is_callable", is_callable),
            ("cpn_barriers", cpn_barriers),
            ("is_guaranteed_cpn", is_guaranteed_cpn),
            ("coupon_rates", coupon_rates),
        ):
            if sched_arr.ndim < 1 or int(sched_arr.shape[0]) < m_rem:
                raise ValueError(f"Schedule {arr_name} shape {sched_arr.shape} too short for M={m_rem}.")

        f_cols: List[List[float]] = [[float(fwd_perf[i, m]) for i in range(d_assets)] for m in range(m_rem)]
        v_cols: List[List[float]] = [[float(vol_perf[i, m]) for i in range(d_assets)] for m in range(m_rem)]
        c_list: List[List[float]] = [[float(corr[i, j]) for j in range(d_assets)] for i in range(d_assets)]
        perm_list: List[int] = [int(x) for x in inv_lex_perm.tolist()]

        if is_long_basket and opt_index_code == 0:
            opt_index_code = 2
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
            h_raw_m = 0.0
            if is_call_m:
                if call_index_code == 14:
                    h_raw_m = _eval_coupon_prob_dist(f_cond, v_cond, ko_b_m, ko_spread, False, False, False, 14)
                elif call_is_normal:
                    h_raw_m = _digital_call_spread_normal(f_cond, v_cond, ko_b_m, ko_spread)
                else:
                    h_raw_m = _digital_call_spread_lognormal(f_cond, v_cond, ko_b_m, ko_spread)
                h_m = min(max(h_raw_m, 0.0), 1.0 - 1e-7)
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
                rho_ck_cond = 1.0
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
                    rho_ck_cond = min(max(rho_cc * z_std / z_sc, -0.999999), 0.999999)
                    if cpn_is_normal:
                        f_cpn_cond = fc_raw + vc_raw * z_mc
                        v_cpn_cond = max(vc_raw * z_sc, 1e-12)
                    else:
                        f_cpn_cond = max(fc_raw, 1e-16) * math.exp(
                            vc_raw * z_mc - 0.5 * vc_raw * vc_raw * (1.0 - z_vc)
                        )
                        v_cpn_cond = max(vc_raw * z_sc, 1e-12)
                cpn_b_m = float(cpn_barriers[m])
                q_base = min(
                    max(
                        _eval_coupon_prob_dist(
                            f_cpn_cond, v_cpn_cond, cpn_b_m, cpn_spread, cpn_is_normal, cpn_dir_down, cpn_type_out, cpn_index_code
                        ),
                        0.0,
                    ),
                    1.0,
                )
                q_m = q_base
                if force_coupons_on_autocall and is_call_m:
                    h_prob = min(max(h_raw_m, 0.0), 1.0)
                    if h_prob >= 1.0 - 1e-12 or q_base >= 1.0 - 1e-12:
                        q_m = 1.0
                    elif h_prob <= 1e-12:
                        q_m = q_base
                    elif q_base <= 1e-12:
                        q_m = h_prob
                    else:
                        s_c = -1.0 if (cpn_dir_down != cpn_type_out) else 1.0
                        p_ck = _bvn_cdf_genz_scalar(_ndtri_scalar(q_base), _ndtri_scalar(h_prob), s_c * rho_ck_cond)
                        q_m = min(max(q_base + h_prob - p_ck, max(q_base, h_prob)), min(q_base + h_prob, 1.0))
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
                    if call_index_code == 14:
                        mu_u, var_u = _ramp_survival_post_moments_twinwin_lognormal(f_cond, v_cond, ko_b_m, ko_spread)
                    elif call_is_normal:
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
                        vi_c = max(vc_prev[i], 1e-12)
                        for j in range(d_assets):
                            wj_n = wn_col[j] * (fn_col[j] if call_is_normal else 1.0)
                            vj_c = max(vc_prev[j], 1e-12)
                            vj_n = max(vn_col[j], 1e-12)
                            num += wi_c * wj_n * c_list[i][j] * vi_c * min(vj_c, vj_n)
                    for i in range(d_assets):
                        wi_c = wc_prev[i] * (fc_prev[i] if call_is_normal else 1.0)
                        wi_n = wn_col[i] * (fn_col[i] if call_is_normal else 1.0)
                        vi_c = max(vc_prev[i], 1e-12)
                        vi_n = max(vn_col[i], 1e-12)
                        for j in range(d_assets):
                            wj_c = wc_prev[j] * (fc_prev[j] if call_is_normal else 1.0)
                            wj_n = wn_col[j] * (fn_col[j] if call_is_normal else 1.0)
                            vj_c = max(vc_prev[j], 1e-12)
                            vj_n = max(vn_col[j], 1e-12)
                            rho_ij = c_list[i][j]
                            den_cur += wi_c * wj_c * rho_ij * vi_c * vj_c
                            den_nxt += wi_n * wj_n * rho_ij * vi_n * vj_n
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
        has_lower_strike = math.isfinite(lower_strike) and (lower_strike > 0.0)
        ko_b_term = float(ko_barriers[m_rem - 1])
        use_direct_ko = expiry_ko_active and not opt_differs
        surv_mult = surv_prev_term if use_direct_ko else surv_cur

        l_put_val = 0.0
        l_onestar_val = 0.0
        p_onestar_val = 0.0
        has_put = (option_gearing > 0.0) and (
            (not has_lower_strike or strike > lower_strike)
            if opt_is_normal
            else (strike > 1e-6 and (not has_lower_strike or strike > lower_strike))
        )
        if has_onestar:
            f_ot = list(f_cols[-1])
            v_ot = list(v_cols[-1])
            if z_mean_term != 0.0 or z_std_term != 1.0:
                wc_term = w_call_list[-1]
                vc_n2 = 0.0
                for i in range(d_assets):
                    wi = wc_term[i] * (f_ot[i] if call_is_normal else 1.0)
                    vi = max(v_ot[i], 1e-12)
                    for j in range(d_assets):
                        wj = wc_term[j] * (f_ot[j] if call_is_normal else 1.0)
                        vj = max(v_ot[j], 1e-12)
                        vc_n2 += wi * wj * c_list[i][j] * vi * vj
                vc_n = math.sqrt(max(vc_n2, 1e-30))
                f_ot_c: List[float] = [0.0] * d_assets
                v_ot_c: List[float] = [0.0] * d_assets
                for i in range(d_assets):
                    vi = max(v_ot[i], 1e-12)
                    cov_i = 0.0
                    for j in range(d_assets):
                        wj = wc_term[j] * (f_ot[j] if call_is_normal else 1.0)
                        vj = max(v_ot[j], 1e-12)
                        cov_i += c_list[i][j] * vi * vj * wj
                    rho_ic = min(max(cov_i / (vi * vc_n), -0.9999), 0.9999)
                    z_mc = rho_ic * z_mean_term
                    z_vc = rho_ic * rho_ic * (z_std_term * z_std_term) + (1.0 - rho_ic * rho_ic)
                    f_ot_c[i] = max(f_ot[i], 1e-16) * math.exp(vi * z_mc - 0.5 * vi * vi * (1.0 - z_vc))
                    v_ot_c[i] = max(vi * math.sqrt(max(z_vc, 1e-12)), 1e-12)
                f_ot = f_ot_c
                v_ot = v_ot_c
            l_put_val, l_onestar_val, p_onestar_val = _eval_1factor_onestar_np(
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
        elif has_put:
            if opt_index_code == 14:
                put_cond = _exact_twinwin_terminal_put(
                    f_opt_cond,
                    v_opt_cond,
                    strike,
                    has_lower_strike,
                    lower_strike,
                    has_active_ki,
                    ki_barrier,
                    ki_spread,
                    use_direct_ko,
                    ko_b_term,
                    ko_spread,
                )
            else:
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
                    has_lower_strike=has_lower_strike,
                )
            l_put_val = surv_mult * put_cond

        l_call_val = 0.0
        has_part_call = math.isfinite(part_call_strike) and (part_call_rate != 0.0) and (
            opt_is_normal or part_call_strike >= 0.0
        )
        if has_part_call:
            if opt_index_code == 14:
                l_call_val = surv_mult * _exact_twinwin_terminal_call(
                    f_opt_cond, v_opt_cond, part_call_strike, use_direct_ko, ko_b_term, ko_spread
                )
            else:
                l_call_val = surv_mult * _exact_piecewise_terminal_call(
                    f_opt_cond, v_opt_cond, opt_is_normal, part_call_strike, use_direct_ko, ko_b_term, ko_spread
                )

        opt_sign = 1.0
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
        scalars[7] = p_onestar_val
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
_U32_DTYPE = np.dtype(np.uint32)


def _ensure_f64_c(arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    if type(arr) is np.ndarray and arr.dtype == _F64_DTYPE and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ensure_i64_c(arr: npt.ArrayLike) -> npt.NDArray[np.int64]:
    if type(arr) is np.ndarray and arr.dtype == _I64_DTYPE and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.int64)


def _ensure_u32_c(arr: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    if type(arr) is np.ndarray and arr.dtype == _U32_DTYPE and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.uint32)


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


_ABI_VERSION = 20261124
_ABI_VERSION_V2 = 20261125


class Tier1AbiInfoC(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("autocall_args_size", ctypes.c_uint32),
        ("autocall_args_align", ctypes.c_uint32),
        ("crn_args_size", ctypes.c_uint32),
        ("crn_args_align", ctypes.c_uint32),
    ]


class Tier1AbiInfoV2C(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("autocall_args_size", ctypes.c_uint32),
        ("autocall_args_align", ctypes.c_uint32),
        ("crn_args_size", ctypes.c_uint32),
        ("crn_args_align", ctypes.c_uint32),
        ("platen_sde_aad_args_size", ctypes.c_uint32),
        ("platen_sde_aad_args_align", ctypes.c_uint32),
        ("sobol_gen_args_size", ctypes.c_uint32),
        ("sobol_gen_args_align", ctypes.c_uint32),
        ("vibrato_gamma_args_size", ctypes.c_uint32),
        ("vibrato_gamma_args_align", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
    ]


class PlatenSdeAadArgsC(ctypes.Structure):
    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_sub", ctypes.c_int64),
        ("eff_r", ctypes.c_int64),
        ("n_rep", ctypes.c_int64),
        ("lv_nt", ctypes.c_int64),
        ("lv_ny", ctypes.c_int64),
        ("lv_tables", ctypes.c_void_p),
        ("z_indep", ctypes.c_void_p),
        ("chol_base", ctypes.c_void_p),
        ("tau_obs", ctypes.c_void_p),
        ("r_native_obs", ctypes.c_void_p),
        ("r_tot_obs", ctypes.c_void_p),
        ("lv_t_grid", ctypes.c_void_p),
        ("lv_y_min", ctypes.c_void_p),
        ("lv_y_max", ctypes.c_void_p),
        ("bar_lx_post", ctypes.c_void_p),
        ("lx_norm_in", ctypes.c_void_p),
        ("q_shift_sub", ctypes.c_void_p),
        ("lx_period_out", ctypes.c_void_p),
        ("norm_x_out", ctypes.c_void_p),
        ("bar_lv_out", ctypes.c_void_p),
        ("bar_chol_out", ctypes.c_void_p),
        ("bar_q_shift_out", ctypes.c_void_p),
        ("run_forward", ctypes.c_int32),
        ("run_backward", ctypes.c_int32),
        ("compute_lv_aad", ctypes.c_int32),
        ("compute_chol_aad", ctypes.c_int32),
        ("compute_q_aad", ctypes.c_int32),
        ("has_q_shift_sub", ctypes.c_int32),
        ("out_lx_pre", ctypes.c_void_p),
        ("out_j_pre", ctypes.c_void_p),
        ("out_sig_pre", ctypes.c_void_p),
        ("vib_period_idx", ctypes.c_int32),
        ("vib_sub_idx", ctypes.c_int32),
        ("save_vib_pre", ctypes.c_int32),
        ("_pad_vib", ctypes.c_int32),
        ("vib_j_pre_max", ctypes.c_double),
        ("vib_dsig_dy_max", ctypes.c_double),
    ]


class SobolGenArgsC(ctypes.Structure):
    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_sub", ctypes.c_int64),
        ("eff_r", ctypes.c_int64),
        ("n_rep", ctypes.c_int64),
        ("sobol_dim", ctypes.c_int64),
        ("n_bits", ctypes.c_int64),
        ("dir_numbers", ctypes.c_void_p),
        ("digital_shifts", ctypes.c_void_p),
        ("bb_ortho_mat", ctypes.c_void_p),
        ("chol_mat", ctypes.c_void_p),
        ("z_indep_out", ctypes.c_void_p),
        ("z_corr_out", ctypes.c_void_p),
        ("use_gray_code", ctypes.c_int32),
        ("apply_bb_time", ctypes.c_int32),
        ("apply_corr", ctypes.c_int32),
        ("antithetic", ctypes.c_int32),
        ("_reserved", ctypes.c_int32 * 2),
    ]


class VibratoGammaArgsC(ctypes.Structure):
    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_paths_total", ctypes.c_int64),
        ("vib_step_start", ctypes.c_int64),
        ("vib_n_steps", ctypes.c_int64),
        ("perf_mat", ctypes.c_void_p),
        ("lx_pre", ctypes.c_void_p),
        ("j_pre", ctypes.c_void_p),
        ("sig_pre", ctypes.c_void_p),
        ("sv_vib", ctypes.c_void_p),
        ("chol_mat", ctypes.c_void_p),
        ("l_inv_t", ctypes.c_void_p),
        ("c_inv", ctypes.c_void_p),
        ("fwd_term", ctypes.c_void_p),
        ("cdl_m", ctypes.c_void_p),
        ("cdl2_last", ctypes.c_void_p),
        ("ko_barriers", ctypes.c_void_p),
        ("is_callable", ctypes.c_void_p),
        ("cpn_barriers", ctypes.c_void_p),
        ("is_guaranteed_cpn", ctypes.c_void_p),
        ("coupon_rates", ctypes.c_void_p),
        ("eps_vec", ctypes.c_void_p),
        ("inv_lex_perm", ctypes.c_void_p),
        ("frozen_term_out", ctypes.c_void_p),
        ("h_vib_out", ctypes.c_void_p),
        ("df_last", ctypes.c_double),
        ("h_vib", ctypes.c_double),
        ("ko_spread", ctypes.c_double),
        ("cpn_spread", ctypes.c_double),
        ("ki_spread", ctypes.c_double),
        ("one_star_strike", ctypes.c_double),
        ("one_star_eps", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("lower_strike", ctypes.c_double),
        ("ki_barrier", ctypes.c_double),
        ("option_gearing", ctypes.c_double),
        ("autocall_coupon", ctypes.c_double),
        ("c_mem_0", ctypes.c_double),
        ("i_ki_past", ctypes.c_double),
        ("part_call_strike", ctypes.c_double),
        ("part_call_rate", ctypes.c_double),
        ("one_star_coupon", ctypes.c_double),
        ("call_code", ctypes.c_int32),
        ("cpn_code", ctypes.c_int32),
        ("opt_code", ctypes.c_int32),
        ("has_ki", ctypes.c_int32),
        ("ki_active", ctypes.c_int32),
        ("has_one_star", ctypes.c_int32),
        ("is_memory", ctypes.c_int32),
        ("is_long_basket", ctypes.c_int32),
        ("no_notional", ctypes.c_int32),
        ("force_cpn_ac", ctypes.c_int32),
        ("cpn_dir_down", ctypes.c_int32),
        ("cpn_type_out", ctypes.c_int32),
        ("has_lower_strike", ctypes.c_int32),
        ("has_part_call", ctypes.c_int32),
    ]


assert ctypes.sizeof(Tier1AbiInfoV2C) == 48 and ctypes.alignment(Tier1AbiInfoV2C) == 4
assert ctypes.sizeof(PlatenSdeAadArgsC) == 272 and ctypes.alignment(PlatenSdeAadArgsC) == 8
assert ctypes.sizeof(SobolGenArgsC) == 128 and ctypes.alignment(SobolGenArgsC) == 8
assert ctypes.sizeof(VibratoGammaArgsC) == 392 and ctypes.alignment(VibratoGammaArgsC) == 8


class Cpp23Tier1Backend:
    """Standalone C++23 (-O3 -march=native -fno-fast-math) Tier-1 ctypes backend."""

    def __init__(self, lib_path: Path) -> None:
        self._lib_path = lib_path
        lib = ctypes.CDLL(str(lib_path))
        self._lib = lib

        abi_fn = lib.tier1_abi_info
        abi_fn.argtypes = []
        abi_fn.restype = Tier1AbiInfoC
        abi_info = abi_fn()
        if (
            int(abi_info.abi_version) != _ABI_VERSION
            or int(abi_info.autocall_args_size) != 336
            or int(abi_info.autocall_args_align) != 8
            or int(abi_info.crn_args_size) != 320
            or int(abi_info.crn_args_align) != 8
        ):
            raise RuntimeError(
                "Incompatible tier1_autocall_cpu ABI: "
                f"version={abi_info.abi_version}, "
                f"autocall=({abi_info.autocall_args_size},{abi_info.autocall_args_align}), "
                f"crn=({abi_info.crn_args_size},{abi_info.crn_args_align})"
            )

        abi_v2_fn = lib.tier1_abi_info_v2
        abi_v2_fn.argtypes = []
        abi_v2_fn.restype = Tier1AbiInfoV2C
        abi_v2: Tier1AbiInfoV2C = abi_v2_fn()
        if (
            int(abi_v2.abi_version) != _ABI_VERSION_V2
            or int(abi_v2.autocall_args_size) != 336
            or int(abi_v2.autocall_args_align) != 8
            or int(abi_v2.crn_args_size) != 320
            or int(abi_v2.crn_args_align) != 8
            or int(abi_v2.platen_sde_aad_args_size) != 272
            or int(abi_v2.platen_sde_aad_args_align) != 8
            or int(abi_v2.sobol_gen_args_size) != 128
            or int(abi_v2.sobol_gen_args_align) != 8
            or int(abi_v2.vibrato_gamma_args_size) != 392
            or int(abi_v2.vibrato_gamma_args_align) != 8
        ):
            raise RuntimeError(
                "Incompatible tier1_autocall_cpu ABI v2: "
                f"version={abi_v2.abi_version}, "
                f"platen=({abi_v2.platen_sde_aad_args_size},{abi_v2.platen_sde_aad_args_align}), "
                f"sobol=({abi_v2.sobol_gen_args_size},{abi_v2.sobol_gen_args_align}), "
                f"vibrato=({abi_v2.vibrato_gamma_args_size},{abi_v2.vibrato_gamma_args_align})"
            )
        self._abi_info_v2: Tier1AbiInfoV2C = abi_v2

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

        lib.tier1_eval_autocall_struct_cpu.argtypes = [ctypes.c_void_p]
        lib.tier1_eval_autocall_struct_cpu.restype = None

        lib.crn_step_1d_trajectories_cpp.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
        ]
        lib.crn_step_1d_trajectories_cpp.restype = None

        lib.crn_eval_payoff_loo_l2o_cpp.argtypes = [c_vp]
        lib.crn_eval_payoff_loo_l2o_cpp.restype = None

        lib.crn_platen_sde_aad_cpp.argtypes = [c_vp]
        lib.crn_platen_sde_aad_cpp.restype = ctypes.c_int32

        lib.crn_generate_sobol_normals_cpp.argtypes = [c_vp]
        lib.crn_generate_sobol_normals_cpp.restype = ctypes.c_int32

        lib.crn_eval_vibrato_gamma_cpp.argtypes = [c_vp]
        lib.crn_eval_vibrato_gamma_cpp.restype = ctypes.c_int32

        lib.crn_step_spot_bumps_cpp.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
            c_vp,
        ]
        lib.crn_step_spot_bumps_cpp.restype = ctypes.c_int32

        self._fn_coeffs = lib.tier1_clamped_cubic_coeffs_1d
        self._fn_eval_cubic = lib.tier1_eval_cubic_with_derivs
        self._fn_bridge = lib.tier1_eval_autocall_bridge_cpu
        self._fn_struct = lib.tier1_eval_autocall_struct_cpu
        self._fn_index_moments = lib.tier1_eval_index_moments_cpu
        self._fn_crn_step_1d = lib.crn_step_1d_trajectories_cpp
        self._fn_crn_step_spot_bumps = lib.crn_step_spot_bumps_cpp
        self._fn_crn_loo_l2o = lib.crn_eval_payoff_loo_l2o_cpp
        self._fn_platen_sde_aad = lib.crn_platen_sde_aad_cpp
        self._fn_sobol_gen = lib.crn_generate_sobol_normals_cpp
        self._fn_vibrato_gamma = lib.crn_eval_vibrato_gamma_cpp
        self._args_cls = Tier1AutocallArgs
        self._args_pack = _ARGS_PACKER

    def abi_info_v2(self) -> Tier1AbiInfoV2C:
        return self._abi_info_v2

    def crn_platen_sde_aad_cpu(
        self,
        *,
        lv_tables: npt.NDArray[np.float64],
        z_indep: npt.NDArray[np.float64],
        chol_base: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        r_native_obs: Optional[npt.NDArray[np.float64]],
        r_tot_obs: Optional[npt.NDArray[np.float64]],
        lv_t_grid: npt.NDArray[np.float64],
        lv_y_min: float | npt.NDArray[np.float64],
        lv_y_max: float | npt.NDArray[np.float64],
        n_sub: int,
        eff_r: int,
        n_rep: int,
        bar_lx_post: Optional[npt.NDArray[np.float64]] = None,
        lx_norm_in: Optional[npt.NDArray[np.float64]] = None,
        q_shift_sub: Optional[npt.NDArray[np.float64]] = None,
        run_forward: bool = True,
        run_backward: bool = False,
        compute_lv_aad: bool = True,
        compute_chol_aad: bool = True,
        compute_q_aad: bool = True,
        out_lx_pre: Optional[npt.NDArray[np.float64]] = None,
        out_j_pre: Optional[npt.NDArray[np.float64]] = None,
        out_sig_pre: Optional[npt.NDArray[np.float64]] = None,
        vib_period_idx: int = 0,
        vib_sub_idx: int = 0,
        save_vib_pre: int | bool = 0,
        vib_j_pre_max: float = 0.0,
        vib_dsig_dy_max: float = 0.0,
    ) -> Tuple[
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
    ]:
        lv_c = _ensure_f64_c(lv_tables)
        z_c = _ensure_f64_c(z_indep)
        chol_c = _ensure_f64_c(chol_base)
        tau_c = _ensure_f64_c(tau_obs)
        tgrid_c = _ensure_f64_c(lv_t_grid)
        d_assets, lv_nt, lv_ny = (int(lv_c.shape[0]), int(lv_c.shape[1]), int(lv_c.shape[2]))
        m_rem = int(tau_c.shape[0])
        n_total = int(eff_r) * int(n_rep)

        y_min_c = _ensure_f64_c(np.broadcast_to(np.asarray(lv_y_min, dtype=np.float64), (d_assets,)))
        y_max_c = _ensure_f64_c(np.broadcast_to(np.asarray(lv_y_max, dtype=np.float64), (d_assets,)))

        has_q_sub = q_shift_sub is not None
        q_sub_c = _ensure_f64_c(q_shift_sub) if q_shift_sub is not None else None
        rn_c = _ensure_f64_c(r_native_obs) if r_native_obs is not None else None
        rt_c = _ensure_f64_c(r_tot_obs) if r_tot_obs is not None else None
        if not has_q_sub and (rn_c is None or rt_c is None):
            raise ValueError("Provide either q_shift_sub or both r_native_obs and r_tot_obs.")

        lx_period_out = np.empty((m_rem, d_assets, n_total), dtype=np.float64) if run_forward else None
        norm_x_out = np.empty((m_rem, d_assets, n_total), dtype=np.float64) if run_forward else None
        bar_post_c = _ensure_f64_c(bar_lx_post) if bar_lx_post is not None else None
        lx_norm_c = _ensure_f64_c(lx_norm_in) if lx_norm_in is not None else None
        if run_backward and bar_post_c is None:
            raise ValueError("bar_lx_post is required when run_backward=True.")
        if run_backward and lx_norm_c is None and lx_period_out is None:
            raise ValueError("lx_norm_in is required when run_backward=True and run_forward=False.")

        save_vib_flag = int(save_vib_pre) if isinstance(save_vib_pre, bool) else int(save_vib_pre)
        if not run_forward:
            save_vib_flag = 0
        if save_vib_flag != 0:
            if out_lx_pre is None or out_j_pre is None or out_sig_pre is None:
                raise ValueError("out_lx_pre, out_j_pre, and out_sig_pre are required when save_vib_pre != 0.")
            expected_vib_shape = (d_assets, n_total)
            for name_buf, arr_buf in (
                ("out_lx_pre", out_lx_pre),
                ("out_j_pre", out_j_pre),
                ("out_sig_pre", out_sig_pre),
            ):
                if (
                    not isinstance(arr_buf, np.ndarray)
                    or arr_buf.dtype != _F64_DTYPE
                    or not arr_buf.flags.c_contiguous
                    or arr_buf.shape != expected_vib_shape
                ):
                    raise ValueError(
                        f"{name_buf} must be C-contiguous float64 of shape {expected_vib_shape}, "
                        f"got {getattr(arr_buf, 'shape', None)} dtype={getattr(arr_buf, 'dtype', None)}"
                    )

        bar_lv_out = np.empty((d_assets, lv_nt, lv_ny), dtype=np.float64) if (run_backward and compute_lv_aad) else None
        bar_chol_out = np.empty((d_assets, d_assets), dtype=np.float64) if (run_backward and compute_chol_aad) else None
        bar_q_out = np.empty((d_assets, m_rem * int(n_sub)), dtype=np.float64) if (run_backward and compute_q_aad) else None

        c_args = PlatenSdeAadArgsC(
            d_assets=d_assets,
            m_rem=m_rem,
            n_sub=int(n_sub),
            eff_r=int(eff_r),
            n_rep=int(n_rep),
            lv_nt=lv_nt,
            lv_ny=lv_ny,
            lv_tables=lv_c.ctypes.data,
            z_indep=z_c.ctypes.data,
            chol_base=chol_c.ctypes.data,
            tau_obs=tau_c.ctypes.data,
            r_native_obs=rn_c.ctypes.data if rn_c is not None else 0,
            r_tot_obs=rt_c.ctypes.data if rt_c is not None else 0,
            lv_t_grid=tgrid_c.ctypes.data,
            lv_y_min=y_min_c.ctypes.data,
            lv_y_max=y_max_c.ctypes.data,
            bar_lx_post=bar_post_c.ctypes.data if bar_post_c is not None else 0,
            lx_norm_in=lx_norm_c.ctypes.data if lx_norm_c is not None else 0,
            q_shift_sub=q_sub_c.ctypes.data if q_sub_c is not None else 0,
            lx_period_out=lx_period_out.ctypes.data if lx_period_out is not None else 0,
            norm_x_out=norm_x_out.ctypes.data if norm_x_out is not None else 0,
            bar_lv_out=bar_lv_out.ctypes.data if bar_lv_out is not None else 0,
            bar_chol_out=bar_chol_out.ctypes.data if bar_chol_out is not None else 0,
            bar_q_shift_out=bar_q_out.ctypes.data if bar_q_out is not None else 0,
            run_forward=1 if run_forward else 0,
            run_backward=1 if run_backward else 0,
            compute_lv_aad=1 if compute_lv_aad else 0,
            compute_chol_aad=1 if compute_chol_aad else 0,
            compute_q_aad=1 if compute_q_aad else 0,
            has_q_shift_sub=1 if has_q_sub else 0,
            out_lx_pre=out_lx_pre.ctypes.data if (save_vib_flag != 0 and out_lx_pre is not None) else 0,
            out_j_pre=out_j_pre.ctypes.data if (save_vib_flag != 0 and out_j_pre is not None) else 0,
            out_sig_pre=out_sig_pre.ctypes.data if (save_vib_flag != 0 and out_sig_pre is not None) else 0,
            vib_period_idx=int(vib_period_idx),
            vib_sub_idx=int(vib_sub_idx),
            save_vib_pre=save_vib_flag,
            _pad_vib=0,
            vib_j_pre_max=float(vib_j_pre_max),
            vib_dsig_dy_max=float(vib_dsig_dy_max),
        )
        rc = int(self._fn_platen_sde_aad(ctypes.byref(c_args)))
        if rc != 0:
            raise RuntimeError(f"crn_platen_sde_aad_cpp failed with status code {rc}.")
        return lx_period_out, norm_x_out, bar_lv_out, bar_chol_out, bar_q_out

    def crn_step_spot_bumps_cpu(
        self,
        *,
        lv_tables_pm: npt.NDArray[np.float64],
        z_corr: npt.NDArray[np.float64],
        tau_obs: npt.NDArray[np.float64],
        r_native_obs: npt.NDArray[np.float64],
        r_tot_obs: npt.NDArray[np.float64],
        lv_t_grid: npt.NDArray[np.float64],
        lv_y_min: float | npt.NDArray[np.float64],
        lv_y_max: float | npt.NDArray[np.float64],
        n_sub: int,
        eff_r: int,
        n_rep: int,
        out_norm_x_pm: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        lv_c = _ensure_f64_c(lv_tables_pm)
        z_c = _ensure_f64_c(z_corr)
        tau_c = _ensure_f64_c(tau_obs)
        rn_c = _ensure_f64_c(r_native_obs)
        rt_c = _ensure_f64_c(r_tot_obs)
        tgrid_c = _ensure_f64_c(lv_t_grid)
        d_assets, _, lv_nt, lv_ny = (int(lv_c.shape[0]), int(lv_c.shape[1]), int(lv_c.shape[2]), int(lv_c.shape[3]))
        m_rem = int(tau_c.shape[0])
        n_total = int(eff_r) * int(n_rep)
        y_min_c = _ensure_f64_c(np.broadcast_to(np.asarray(lv_y_min, dtype=np.float64), (d_assets,)))
        y_max_c = _ensure_f64_c(np.broadcast_to(np.asarray(lv_y_max, dtype=np.float64), (d_assets,)))
        out_shape = (2, d_assets, m_rem, n_total)
        if out_norm_x_pm is None:
            out_arr = np.empty(out_shape, dtype=np.float64)
        else:
            if (
                not isinstance(out_norm_x_pm, np.ndarray)
                or out_norm_x_pm.dtype != _F64_DTYPE
                or not out_norm_x_pm.flags.c_contiguous
                or out_norm_x_pm.shape != out_shape
            ):
                raise ValueError(f"out_norm_x_pm must be C-contiguous float64 of shape {out_shape}")
            out_arr = out_norm_x_pm
        rc = int(
            self._fn_crn_step_spot_bumps(
                d_assets,
                lv_nt,
                lv_ny,
                m_rem,
                int(n_sub),
                int(eff_r),
                int(n_rep),
                lv_c.ctypes.data,
                z_c.ctypes.data,
                tau_c.ctypes.data,
                rn_c.ctypes.data,
                rt_c.ctypes.data,
                tgrid_c.ctypes.data,
                y_min_c.ctypes.data,
                y_max_c.ctypes.data,
                out_arr.ctypes.data,
            )
        )
        if rc != 0:
            raise RuntimeError(f"crn_step_spot_bumps_cpp failed with status code {rc}.")
        return out_arr

    def crn_generate_sobol_normals_cpu(
        self,
        *,
        d_assets: int,
        m_rem: int,
        n_sub: int,
        eff_r: int,
        n_rep: int,
        dir_numbers: Optional[npt.NDArray[np.uint32]] = None,
        digital_shifts: Optional[npt.NDArray[np.uint32]] = None,
        bb_ortho_mat: Optional[npt.NDArray[np.float64]] = None,
        chol_mat: Optional[npt.NDArray[np.float64]] = None,
        n_bits: int = 30,
        use_gray_code: bool = True,
        apply_bb_time: bool = False,
        apply_corr: bool = False,
        antithetic: bool = False,
        return_indep: bool = True,
        return_corr: bool = False,
    ) -> Tuple[Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
        dir_c = _ensure_u32_c(dir_numbers) if dir_numbers is not None else None
        sobol_dim = int(dir_c.shape[0]) if dir_c is not None else 0
        shifts_c = _ensure_u32_c(digital_shifts) if digital_shifts is not None else None
        bb_c = _ensure_f64_c(bb_ortho_mat) if bb_ortho_mat is not None else None
        chol_c = _ensure_f64_c(chol_mat) if chol_mat is not None else None
        shape = (int(m_rem), int(n_sub), int(eff_r), int(n_rep), int(d_assets))
        z_indep_out = np.empty(shape, dtype=np.float64) if return_indep else None
        z_corr_out = np.empty(shape, dtype=np.float64) if (return_corr or apply_corr) else None
        if z_indep_out is None and z_corr_out is None:
            raise ValueError("At least one of return_indep or return_corr/apply_corr must be True.")

        c_args = SobolGenArgsC(
            d_assets=int(d_assets),
            m_rem=int(m_rem),
            n_sub=int(n_sub),
            eff_r=int(eff_r),
            n_rep=int(n_rep),
            sobol_dim=sobol_dim,
            n_bits=int(n_bits),
            dir_numbers=dir_c.ctypes.data if dir_c is not None else 0,
            digital_shifts=shifts_c.ctypes.data if shifts_c is not None else 0,
            bb_ortho_mat=bb_c.ctypes.data if bb_c is not None else 0,
            chol_mat=chol_c.ctypes.data if chol_c is not None else 0,
            z_indep_out=z_indep_out.ctypes.data if z_indep_out is not None else 0,
            z_corr_out=z_corr_out.ctypes.data if z_corr_out is not None else 0,
            use_gray_code=1 if use_gray_code else 0,
            apply_bb_time=1 if apply_bb_time else 0,
            apply_corr=1 if apply_corr else 0,
            antithetic=1 if antithetic else 0,
        )
        rc = int(self._fn_sobol_gen(ctypes.byref(c_args)))
        if rc != 0:
            raise RuntimeError(f"crn_generate_sobol_normals_cpp failed with status code {rc}.")
        return z_indep_out, z_corr_out

    def crn_eval_vibrato_gamma_cpu(
        self,
        *,
        perf_mat: npt.NDArray[np.float64],
        lx_pre: npt.NDArray[np.float64],
        j_pre: npt.NDArray[np.float64],
        sig_pre: npt.NDArray[np.float64],
        sv_vib: npt.NDArray[np.float64],
        chol_mat: npt.NDArray[np.float64],
        l_inv_t: npt.NDArray[np.float64],
        c_inv: npt.NDArray[np.float64],
        fwd_term: npt.NDArray[np.float64],
        cdl_m: npt.NDArray[np.float64],
        cdl2_last: npt.NDArray[np.float64],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guaranteed_cpn: npt.NDArray[np.float64],
        coupon_rates: npt.NDArray[np.float64],
        eps_vec: npt.NDArray[np.float64],
        inv_lex_perm: Optional[npt.NDArray[np.int64]],
        vib_step_start: int,
        vib_n_steps: int,
        df_last: float,
        h_vib: float,
        ko_spread: float,
        cpn_spread: float,
        ki_spread: float,
        one_star_strike: float,
        one_star_eps: float,
        strike: float,
        lower_strike: float,
        ki_barrier: float,
        option_gearing: float,
        autocall_coupon: float,
        c_mem_0: float,
        i_ki_past: float,
        part_call_strike: float,
        part_call_rate: float,
        one_star_coupon: float,
        call_code: int,
        cpn_code: int,
        opt_code: int,
        has_ki: bool,
        ki_active: bool,
        has_one_star: bool,
        is_memory: bool,
        is_long_basket: bool,
        no_notional: bool,
        force_cpn_ac: bool,
        cpn_dir_down: bool,
        cpn_type_out: bool,
        has_lower_strike: bool,
        has_part_call: bool,
        emit_frozen_term: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        perf_c = _ensure_f64_c(perf_mat)
        lx_pre_c = _ensure_f64_c(lx_pre)
        j_pre_c = _ensure_f64_c(j_pre)
        sig_pre_c = _ensure_f64_c(sig_pre)
        sv_vib_c = _ensure_f64_c(sv_vib)
        chol_c = _ensure_f64_c(chol_mat)
        linvt_c = _ensure_f64_c(l_inv_t)
        cinv_c = _ensure_f64_c(c_inv)
        fwd_c = _ensure_f64_c(fwd_term)
        cdlm_c = _ensure_f64_c(cdl_m)
        cdl2_c = _ensure_f64_c(cdl2_last)
        ko_c = _ensure_f64_c(ko_barriers)
        ic_c = _ensure_f64_c(is_callable)
        cpn_c = _ensure_f64_c(cpn_barriers)
        ig_c = _ensure_f64_c(is_guaranteed_cpn)
        cr_c = _ensure_f64_c(coupon_rates)
        eps_c = _ensure_f64_c(eps_vec)
        perm_c = _ensure_i64_c(inv_lex_perm) if inv_lex_perm is not None else None

        m_rem, d_assets, n_paths_total = (int(perf_c.shape[0]), int(perf_c.shape[1]), int(perf_c.shape[2]))
        h_vib_out = np.empty((d_assets, d_assets), dtype=np.float64)
        frozen_out = np.empty((4, n_paths_total), dtype=np.float64) if (emit_frozen_term and m_rem > 1) else None

        c_args = VibratoGammaArgsC(
            d_assets=d_assets,
            m_rem=m_rem,
            n_paths_total=n_paths_total,
            vib_step_start=int(vib_step_start),
            vib_n_steps=int(vib_n_steps),
            perf_mat=perf_c.ctypes.data,
            lx_pre=lx_pre_c.ctypes.data,
            j_pre=j_pre_c.ctypes.data,
            sig_pre=sig_pre_c.ctypes.data,
            sv_vib=sv_vib_c.ctypes.data,
            chol_mat=chol_c.ctypes.data,
            l_inv_t=linvt_c.ctypes.data,
            c_inv=cinv_c.ctypes.data,
            fwd_term=fwd_c.ctypes.data,
            cdl_m=cdlm_c.ctypes.data,
            cdl2_last=cdl2_c.ctypes.data,
            ko_barriers=ko_c.ctypes.data,
            is_callable=ic_c.ctypes.data,
            cpn_barriers=cpn_c.ctypes.data,
            is_guaranteed_cpn=ig_c.ctypes.data,
            coupon_rates=cr_c.ctypes.data,
            eps_vec=eps_c.ctypes.data,
            inv_lex_perm=perm_c.ctypes.data if perm_c is not None else 0,
            frozen_term_out=frozen_out.ctypes.data if frozen_out is not None else 0,
            h_vib_out=h_vib_out.ctypes.data,
            df_last=float(df_last),
            h_vib=float(h_vib),
            ko_spread=float(ko_spread),
            cpn_spread=float(cpn_spread),
            ki_spread=float(ki_spread),
            one_star_strike=float(one_star_strike),
            one_star_eps=float(one_star_eps),
            strike=float(strike),
            lower_strike=float(lower_strike),
            ki_barrier=float(ki_barrier),
            option_gearing=float(option_gearing),
            autocall_coupon=float(autocall_coupon),
            c_mem_0=float(c_mem_0),
            i_ki_past=float(i_ki_past),
            part_call_strike=float(part_call_strike),
            part_call_rate=float(part_call_rate),
            one_star_coupon=float(one_star_coupon),
            call_code=int(call_code),
            cpn_code=int(cpn_code),
            opt_code=int(opt_code),
            has_ki=1 if has_ki else 0,
            ki_active=1 if ki_active else 0,
            has_one_star=1 if has_one_star else 0,
            is_memory=1 if is_memory else 0,
            is_long_basket=1 if is_long_basket else 0,
            no_notional=1 if no_notional else 0,
            force_cpn_ac=1 if force_cpn_ac else 0,
            cpn_dir_down=1 if cpn_dir_down else 0,
            cpn_type_out=1 if cpn_type_out else 0,
            has_lower_strike=1 if has_lower_strike else 0,
            has_part_call=1 if has_part_call else 0,
        )
        rc = int(self._fn_vibrato_gamma(ctypes.byref(c_args)))
        if rc != 0:
            raise RuntimeError(f"crn_eval_vibrato_gamma_cpp failed with status code {rc}.")
        return h_vib_out, frozen_out

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
        if fp.ndim != 1 or vp.shape != fp.shape:
            raise ValueError(f"Expected matching 1D f_arr and v_arr, got {fp.shape} vs {vp.shape}.")
        d_assets = int(fp.shape[0])
        if d_assets == 0:
            return 0.0, 0.0, np.zeros(0, dtype=np.float64)
        if cm.ndim != 2 or cm.shape != (d_assets, d_assets):
            raise ValueError(f"Expected corr shape ({d_assets}, {d_assets}), got {cm.shape}.")
        out = np.zeros(d_assets + 2, dtype=np.float64)
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
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("clamped_cubic_coeffs_1d expects matching 1D arrays x and y.")
        k = int(x_arr.shape[0])
        if k < 2:
            raise ValueError("clamped_cubic_coeffs_1d requires at least 2 knots.")
        coeffs = np.zeros((4, k - 1), dtype=np.float64)
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
        if x_arr.ndim != 1 or xq_arr.ndim != 1:
            raise ValueError("eval_cubic_with_derivs expects 1D x and x_query arrays.")
        k = int(x_arr.shape[0])
        n_q = int(xq_arr.shape[0])
        buf = np.zeros(3 * n_q, dtype=np.float64)
        val_out = buf[0:n_q]
        dval_out = buf[n_q : 2 * n_q]
        d2val_out = buf[2 * n_q : 3 * n_q]
        if k < 2 or n_q == 0:
            return val_out, dval_out, d2val_out
        if c_arr.ndim != 2 or c_arr.shape != (4, k - 1):
            raise ValueError(f"eval_cubic_with_derivs expects coeffs shape (4, {k - 1}), got {c_arr.shape}.")
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

        if fp.ndim != 2 or vp.shape != fp.shape:
            raise ValueError(f"Expected matching 2D fwd_perf and vol_perf, got {fp.shape} vs {vp.shape}.")
        d_assets, m_rem = int(fp.shape[0]), int(fp.shape[1])
        dm = d_assets * m_rem
        out_buf = np.zeros(6 * m_rem + dm + 8, dtype=np.float64)
        f_w = out_buf[0:m_rem]
        v_w = out_buf[m_rem : 2 * m_rem]
        p_stop = out_buf[2 * m_rem : 3 * m_rem]
        surv = out_buf[3 * m_rem : 4 * m_rem]
        h_rate = out_buf[4 * m_rem : 5 * m_rem]
        q_cpn = out_buf[5 * m_rem : 6 * m_rem]
        dw_flat = out_buf[6 * m_rem : 6 * m_rem + dm]
        delta_w = dw_flat.reshape(d_assets, m_rem)
        scalars = out_buf[6 * m_rem + dm :]
        if d_assets == 0 or m_rem == 0:
            scalars[4] = float(pv_in_transit)
            return f_w, v_w, delta_w, p_stop, surv, h_rate, q_cpn, scalars
        if cm.ndim != 2 or cm.shape != (d_assets, d_assets):
            raise ValueError(f"Expected corr shape ({d_assets}, {d_assets}), got {cm.shape}.")
        for arr_name, sched_arr in (
            ("df_pay", df),
            ("ko_barriers", ko),
            ("is_callable", ic),
            ("cpn_barriers", cb),
            ("is_guaranteed_cpn", ig),
            ("coupon_rates", cr),
        ):
            if sched_arr.ndim < 1 or int(sched_arr.shape[0]) < m_rem:
                raise ValueError(f"Schedule {arr_name} shape {sched_arr.shape} too short for M={m_rem}.")

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
        if cleaned.lower() == "numpy":
            return None
        if not cleaned or cleaned.lower() == "none":
            raise RuntimeError(
                f"DEEPVOL_TIER1_CPP_LIB={env_override!r} disabled the C++23 backend; "
                "set DEEPVOL_TIER1_CPP_LIB=numpy explicitly if PureNumpy fallback is intended."
            )
        p = Path(cleaned)
        if not p.is_file():
            raise RuntimeError(f"DEEPVOL_TIER1_CPP_LIB points to missing shared library: {p}")
        return p
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
    """Returns the active IEEE-754 float64 Tier-1 backend (C++23, or PureNumPy if explicitly requested)."""
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
        env_override = os.environ.get("DEEPVOL_TIER1_CPP_LIB")
        if env_override is not None and env_override.strip().lower() == "numpy":
            _CACHED_BACKEND = PureNumpyTier1Backend()
            return _CACHED_BACKEND
        lib_path = _locate_cpp23_shared_lib()
        if lib_path is None:
            raise RuntimeError(
                "C++23 backend shared library tier1_autocall_cpu.so not found. "
                "Build and install src/deepvol/cpp or set DEEPVOL_TIER1_CPP_LIB=numpy explicitly."
            )
        try:
            _CACHED_BACKEND = Cpp23Tier1Backend(lib_path)
            return _CACHED_BACKEND
        except (OSError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load C++23 backend from {lib_path}: {exc}. "
                "Set DEEPVOL_TIER1_CPP_LIB=numpy explicitly if PureNumpy fallback is intended."
            ) from exc


def reset_tier1_backend_cache() -> None:
    """Clears the cached backend instance so environment overrides take effect."""
    global _CACHED_BACKEND
    with _BACKEND_LOCK:
        _CACHED_BACKEND = None


def _require_cpp23_backend() -> Cpp23Tier1Backend:
    backend = get_tier1_backend()
    if not isinstance(backend, Cpp23Tier1Backend):
        raise RuntimeError(
            f"Requested C++23 ABI v2 kernel requires Cpp23Tier1Backend, got {backend.name!r}."
        )
    return backend


def tier1_abi_info_v2() -> Tier1AbiInfoV2C:
    """Returns the validated ABI v2 handshake struct from tier1_autocall_cpu.so."""
    return _require_cpp23_backend().abi_info_v2()


def crn_platen_sde_aad_cpu(
    *,
    lv_tables: npt.NDArray[np.float64],
    z_indep: npt.NDArray[np.float64],
    chol_base: npt.NDArray[np.float64],
    tau_obs: npt.NDArray[np.float64],
    r_native_obs: Optional[npt.NDArray[np.float64]],
    r_tot_obs: Optional[npt.NDArray[np.float64]],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_min: float | npt.NDArray[np.float64],
    lv_y_max: float | npt.NDArray[np.float64],
    n_sub: int,
    eff_r: int,
    n_rep: int,
    bar_lx_post: Optional[npt.NDArray[np.float64]] = None,
    lx_norm_in: Optional[npt.NDArray[np.float64]] = None,
    q_shift_sub: Optional[npt.NDArray[np.float64]] = None,
    run_forward: bool = True,
    run_backward: bool = False,
    compute_lv_aad: bool = True,
    compute_chol_aad: bool = True,
    compute_q_aad: bool = True,
    out_lx_pre: Optional[npt.NDArray[np.float64]] = None,
    out_j_pre: Optional[npt.NDArray[np.float64]] = None,
    out_sig_pre: Optional[npt.NDArray[np.float64]] = None,
    vib_period_idx: int = 0,
    vib_sub_idx: int = 0,
    save_vib_pre: int | bool = 0,
    vib_j_pre_max: float = 0.0,
    vib_dsig_dy_max: float = 0.0,
) -> Tuple[
    Optional[npt.NDArray[np.float64]],
    Optional[npt.NDArray[np.float64]],
    Optional[npt.NDArray[np.float64]],
    Optional[npt.NDArray[np.float64]],
    Optional[npt.NDArray[np.float64]],
]:
    """Dispatches Platen Weak-2.0 forward + reverse-mode SDE AAD to C++23 AVX2/OpenMP."""
    return _require_cpp23_backend().crn_platen_sde_aad_cpu(
        lv_tables=lv_tables,
        z_indep=z_indep,
        chol_base=chol_base,
        tau_obs=tau_obs,
        r_native_obs=r_native_obs,
        r_tot_obs=r_tot_obs,
        lv_t_grid=lv_t_grid,
        lv_y_min=lv_y_min,
        lv_y_max=lv_y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        bar_lx_post=bar_lx_post,
        lx_norm_in=lx_norm_in,
        q_shift_sub=q_shift_sub,
        run_forward=run_forward,
        run_backward=run_backward,
        compute_lv_aad=compute_lv_aad,
        compute_chol_aad=compute_chol_aad,
        compute_q_aad=compute_q_aad,
        out_lx_pre=out_lx_pre,
        out_j_pre=out_j_pre,
        out_sig_pre=out_sig_pre,
        vib_period_idx=vib_period_idx,
        vib_sub_idx=vib_sub_idx,
        save_vib_pre=save_vib_pre,
        vib_j_pre_max=vib_j_pre_max,
        vib_dsig_dy_max=vib_dsig_dy_max,
    )


def crn_step_spot_bumps_cpu(
    *,
    lv_tables_pm: npt.NDArray[np.float64],
    z_corr: npt.NDArray[np.float64],
    tau_obs: npt.NDArray[np.float64],
    r_native_obs: npt.NDArray[np.float64],
    r_tot_obs: npt.NDArray[np.float64],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_min: float | npt.NDArray[np.float64],
    lv_y_max: float | npt.NDArray[np.float64],
    n_sub: int,
    eff_r: int,
    n_rep: int,
    out_norm_x_pm: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """Dispatches batched +/- spot-bump 1D Platen Weak-2.0 stepping to C++23 AVX2/OpenMP."""
    return _require_cpp23_backend().crn_step_spot_bumps_cpu(
        lv_tables_pm=lv_tables_pm,
        z_corr=z_corr,
        tau_obs=tau_obs,
        r_native_obs=r_native_obs,
        r_tot_obs=r_tot_obs,
        lv_t_grid=lv_t_grid,
        lv_y_min=lv_y_min,
        lv_y_max=lv_y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        out_norm_x_pm=out_norm_x_pm,
    )


def crn_generate_sobol_normals_cpu(
    *,
    d_assets: int,
    m_rem: int,
    n_sub: int,
    eff_r: int,
    n_rep: int,
    dir_numbers: Optional[npt.NDArray[np.uint32]] = None,
    digital_shifts: Optional[npt.NDArray[np.uint32]] = None,
    bb_ortho_mat: Optional[npt.NDArray[np.float64]] = None,
    chol_mat: Optional[npt.NDArray[np.float64]] = None,
    n_bits: int = 30,
    use_gray_code: bool = True,
    apply_bb_time: bool = False,
    apply_corr: bool = False,
    antithetic: bool = False,
    return_indep: bool = True,
    return_corr: bool = False,
) -> Tuple[Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
    """Dispatches Sobol Brownian-bridge normal generation to C++23 AVX2/OpenMP."""
    return _require_cpp23_backend().crn_generate_sobol_normals_cpu(
        d_assets=d_assets,
        m_rem=m_rem,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        dir_numbers=dir_numbers,
        digital_shifts=digital_shifts,
        bb_ortho_mat=bb_ortho_mat,
        chol_mat=chol_mat,
        n_bits=n_bits,
        use_gray_code=use_gray_code,
        apply_bb_time=apply_bb_time,
        apply_corr=apply_corr,
        antithetic=antithetic,
        return_indep=return_indep,
        return_corr=return_corr,
    )


def crn_eval_vibrato_gamma_cpu(
    *,
    perf_mat: npt.NDArray[np.float64],
    lx_pre: npt.NDArray[np.float64],
    j_pre: npt.NDArray[np.float64],
    sig_pre: npt.NDArray[np.float64],
    sv_vib: npt.NDArray[np.float64],
    chol_mat: npt.NDArray[np.float64],
    l_inv_t: npt.NDArray[np.float64],
    c_inv: npt.NDArray[np.float64],
    fwd_term: npt.NDArray[np.float64],
    cdl_m: npt.NDArray[np.float64],
    cdl2_last: npt.NDArray[np.float64],
    ko_barriers: npt.NDArray[np.float64],
    is_callable: npt.NDArray[np.float64],
    cpn_barriers: npt.NDArray[np.float64],
    is_guaranteed_cpn: npt.NDArray[np.float64],
    coupon_rates: npt.NDArray[np.float64],
    eps_vec: npt.NDArray[np.float64],
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
    vib_step_start: int,
    vib_n_steps: int,
    df_last: float,
    h_vib: float,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    one_star_strike: float,
    one_star_eps: float,
    strike: float,
    lower_strike: float,
    ki_barrier: float,
    option_gearing: float,
    autocall_coupon: float,
    c_mem_0: float,
    i_ki_past: float,
    part_call_strike: float,
    part_call_rate: float,
    one_star_coupon: float,
    call_code: int,
    cpn_code: int,
    opt_code: int,
    has_ki: bool,
    ki_active: bool,
    has_one_star: bool,
    is_memory: bool,
    is_long_basket: bool,
    no_notional: bool,
    force_cpn_ac: bool,
    cpn_dir_down: bool,
    cpn_type_out: bool,
    has_lower_strike: bool,
    has_part_call: bool,
    emit_frozen_term: bool = True,
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
    """Dispatches terminal antithetic Vibrato Gamma/Cross-Gamma to C++23 OpenMP."""
    return _require_cpp23_backend().crn_eval_vibrato_gamma_cpu(
        perf_mat=perf_mat,
        lx_pre=lx_pre,
        j_pre=j_pre,
        sig_pre=sig_pre,
        sv_vib=sv_vib,
        chol_mat=chol_mat,
        l_inv_t=l_inv_t,
        c_inv=c_inv,
        fwd_term=fwd_term,
        cdl_m=cdl_m,
        cdl2_last=cdl2_last,
        ko_barriers=ko_barriers,
        is_callable=is_callable,
        cpn_barriers=cpn_barriers,
        is_guaranteed_cpn=is_guaranteed_cpn,
        coupon_rates=coupon_rates,
        eps_vec=eps_vec,
        inv_lex_perm=inv_lex_perm,
        vib_step_start=vib_step_start,
        vib_n_steps=vib_n_steps,
        df_last=df_last,
        h_vib=h_vib,
        ko_spread=ko_spread,
        cpn_spread=cpn_spread,
        ki_spread=ki_spread,
        one_star_strike=one_star_strike,
        one_star_eps=one_star_eps,
        strike=strike,
        lower_strike=lower_strike,
        ki_barrier=ki_barrier,
        option_gearing=option_gearing,
        autocall_coupon=autocall_coupon,
        c_mem_0=c_mem_0,
        i_ki_past=i_ki_past,
        part_call_strike=part_call_strike,
        part_call_rate=part_call_rate,
        one_star_coupon=one_star_coupon,
        call_code=call_code,
        cpn_code=cpn_code,
        opt_code=opt_code,
        has_ki=has_ki,
        ki_active=ki_active,
        has_one_star=has_one_star,
        is_memory=is_memory,
        is_long_basket=is_long_basket,
        no_notional=no_notional,
        force_cpn_ac=force_cpn_ac,
        cpn_dir_down=cpn_dir_down,
        cpn_type_out=cpn_type_out,
        has_lower_strike=has_lower_strike,
        has_part_call=has_part_call,
        emit_frozen_term=emit_frozen_term,
    )


__all__ = [
    "Cpp23Tier1Backend",
    "PlatenSdeAadArgsC",
    "PureNumpyTier1Backend",
    "SobolGenArgsC",
    "Tier1AbiInfoC",
    "Tier1AbiInfoV2C",
    "Tier1MathBackend",
    "VibratoGammaArgsC",
    "crn_eval_vibrato_gamma_cpu",
    "crn_generate_sobol_normals_cpu",
    "crn_platen_sde_aad_cpu",
    "crn_step_spot_bumps_cpu",
    "get_tier1_backend",
    "is_exact_analytical_index",
    "monitoring_index_to_code",
    "reset_tier1_backend_cache",
    "tier1_abi_info_v2",
]

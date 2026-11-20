#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440084436210485;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kSqrtHalfPi = 1.2533141373155002512078826424055;
constexpr double kSqrtTwoOverPi = 0.79788456080286535587989211986876;
constexpr double kTwoOverSqrtPi = 0.56418958354775628694807945156077;
constexpr double kTieTau = 0.008;

inline double phi_std(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

inline double ndtr_std(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

// Rational minimax approximation to erfcx(x) = exp(x^2) * erfc(x) (Cody 1969).
inline double cody_erfcx(double x) noexcept {
    if (std::isnan(x)) {
        return x;
    }
    const double y = std::abs(x);
    double result = 0.0;
    if (y <= 0.46875) {
        static constexpr double a[5] = {
            3.16112374387056560e00, 1.13864154151050156e02,
            3.77485237685302021e02, 3.20937758913846947e03,
            1.85777706184603153e-01
        };
        static constexpr double b[4] = {
            2.36012909523441209e01, 2.44024637934444173e02,
            1.28261652607737228e03, 2.84423683343917062e03
        };
        const double ysq = y * y;
        double xnum = a[4] * ysq;
        double xden = ysq;
        for (int i = 0; i < 3; ++i) {
            xnum = (xnum + a[i]) * ysq;
            xden = (xden + b[i]) * ysq;
        }
        const double erf_val = y * (xnum + a[3]) / (xden + b[3]);
        const double erfc_abs = 1.0 - (x < 0.0 ? -erf_val : erf_val);
        return std::exp(ysq) * erfc_abs;
    } else if (y <= 4.0) {
        static constexpr double c[9] = {
            5.64188496988670089e-01, 8.88314979438837594e00,
            6.61191906371416295e01, 2.98635138197400131e02,
            8.81952221241769090e02, 1.71204761263407058e03,
            2.05107837782607147e03, 1.23033935479799725e03,
            2.15311535474403846e-08
        };
        static constexpr double d[8] = {
            1.57449261107098347e01, 1.17693950891312499e02,
            5.37181101862009858e02, 1.62138957456669019e03,
            3.29079923573345963e03, 4.36261909014324716e03,
            3.43936767414372164e03, 1.23033935480374942e03
        };
        double xnum = c[8] * y;
        double xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        result = (xnum + c[7]) / (xden + d[7]);
    } else {
        static constexpr double p[6] = {
            3.05326634961232344e-01, 3.60344899949804439e-01,
            1.25781726111229246e-01, 1.60837851487422766e-02,
            6.58749161529837803e-04, 1.63153871373020978e-02
        };
        static constexpr double q[5] = {
            2.56852019228982242e00, 1.87295284992346047e00,
            5.27905102951428412e-01, 6.05183413124413191e-02,
            2.33520497626869185e-03
        };
        const double ysq = 1.0 / (y * y);
        double xnum = p[5] * ysq;
        double xden = ysq;
        for (int i = 0; i < 4; ++i) {
            xnum = (xnum + p[i]) * ysq;
            xden = (xden + q[i]) * ysq;
        }
        result = ysq * (xnum + p[4]) / (xden + q[4]);
        result = (kTwoOverSqrtPi - result) / y;
    }
    if (x < 0.0) {
        return 2.0 * std::exp(y * y) - result;
    }
    return result;
}

// Continued fraction tail avoids catastrophic subtraction (alpha + lambda(alpha)) for alpha <= -6.
inline void truncated_upper_std_moments(double alpha, double& mu_out, double& var_out) noexcept {
    if (alpha <= -6.0) {
        const double x = -alpha;
        double v = 0.0;
        for (int k = 36; k >= 3; --k) {
            v = static_cast<double>(k) / (x + v);
        }
        const double r2 = 2.0 / (x + v);
        const double psi = 1.0 / (x + r2);
        mu_out = -x - psi;
        var_out = std::max(psi * (r2 - psi), 1e-12);
        return;
    }
    if (alpha >= 26.0) {
        mu_out = 0.0;
        var_out = 1.0;
        return;
    }
    const double lam = kSqrtTwoOverPi / cody_erfcx(-alpha * kInvSqrt2);
    mu_out = -lam;
    var_out = std::max(1.0 - alpha * lam - lam * lam, 1e-12);
}

inline double black76_call(double f, double k, double v) noexcept {
    const double f_c = f > 1e-16 ? f : 1e-16;
    const double k_c = k > 1e-16 ? k : 1e-16;
    const double v_c = v > 1e-12 ? v : 1e-12;
    const double d1 = (std::log(f_c / k_c) + 0.5 * v_c * v_c) / v_c;
    const double d2 = d1 - v_c;
    return f_c * ndtr_std(d1) - k_c * ndtr_std(d2);
}

inline double black76_put(double f, double k, double v) noexcept {
    const double f_c = f > 1e-16 ? f : 1e-16;
    const double k_c = k > 1e-16 ? k : 1e-16;
    const double v_c = v > 1e-12 ? v : 1e-12;
    const double d1 = (std::log(f_c / k_c) + 0.5 * v_c * v_c) / v_c;
    const double d2 = d1 - v_c;
    return k_c * ndtr_std(-d2) - f_c * ndtr_std(-d1);
}

inline double digital_call_spread_lognormal(double f, double v, double barrier, double spread) noexcept {
    const double f_c = f > 1e-16 ? f : 1e-16;
    const double v_c = v > 1e-12 ? v : 1e-12;
    const double b_c = barrier > 1e-8 ? barrier : 1e-8;
    if (std::abs(spread) <= 1e-6) {
        const double d2 = (std::log(f_c / b_c) - 0.5 * v_c * v_c) / v_c;
        return std::clamp(ndtr_std(d2), 0.0, 1.0);
    }
    const double b_low = std::min(b_c, b_c + spread);
    const double b_high = std::max(b_c, b_c + spread);
    if (b_high <= 0.0) {
        return 1.0;
    }
    const double c_low = (b_low > 0.0) ? black76_call(f_c, b_low, v_c) : (f_c - b_low);
    const double c_high = black76_call(f_c, b_high, v_c);
    return std::clamp((c_low - c_high) / (b_high - b_low), 0.0, 1.0);
}

inline double normal_call(double mu, double sig, double k) noexcept {
    const double s_c = sig > 1e-12 ? sig : 1e-12;
    const double z = (mu - k) / s_c;
    return (mu - k) * ndtr_std(z) + s_c * phi_std(z);
}

inline double digital_call_spread_normal(double mu, double sig, double barrier, double spread) noexcept {
    const double s_c = sig > 1e-12 ? sig : 1e-12;
    if (std::abs(spread) <= 1e-6) {
        return std::clamp(ndtr_std((mu - barrier) / s_c), 0.0, 1.0);
    }
    const double b_low = std::min(barrier, barrier + spread);
    const double b_high = std::max(barrier, barrier + spread);
    const double c_low = normal_call(mu, s_c, b_low);
    const double c_high = normal_call(mu, s_c, b_high);
    return std::clamp((c_low - c_high) / (b_high - b_low), 0.0, 1.0);
}

inline double eval_coupon_prob_dist(
    double f_or_mu,
    double v_or_sig,
    double barrier,
    double spread,
    bool is_normal,
    bool dir_down,
    bool type_out,
    int index_code
) noexcept {
    // TwinWin maps W = max(W_min, 2 - W_min) >= 1.0
    if (index_code == 14) {
        if (barrier <= 1.0) {
            double p_ev = 1.0;
            return type_out ? (1.0 - p_ev) : p_ev;
        }
        const double b_twin_dn = 2.0 - barrier;
        const double p_up = digital_call_spread_lognormal(f_or_mu, v_or_sig, barrier, spread);
        const double p_dn = (b_twin_dn > 0.0)
            ? (1.0 - digital_call_spread_lognormal(f_or_mu, v_or_sig, b_twin_dn, -spread))
            : 0.0;
        const double p_ge = std::clamp(p_up + p_dn, 0.0, 1.0);
        const double p_ev = dir_down ? (1.0 - p_ge) : p_ge;
        return type_out ? (1.0 - p_ev) : p_ev;
    }
    const double eff_spread = dir_down ? -spread : spread;
    const double p_up = is_normal
        ? digital_call_spread_normal(f_or_mu, v_or_sig, barrier, eff_spread)
        : digital_call_spread_lognormal(f_or_mu, v_or_sig, barrier, eff_spread);
    const double p_ev = dir_down ? (1.0 - p_up) : p_up;
    return type_out ? (1.0 - p_ev) : p_ev;
}

inline void ramp_survival_post_moments_lognormal(
    double f_cond,
    double lam,
    double b_ko,
    double ko_spread,
    double& mu_u_out,
    double& var_u_out
) noexcept {
    const double f_c = f_cond > 1e-16 ? f_cond : 1e-16;
    const double l_c = lam > 1e-12 ? lam : 1e-12;
    const double b_c = b_ko > 1e-8 ? b_ko : 1e-8;
    const double m_u = std::log(f_c) - 0.5 * l_c * l_c;

    if (std::abs(ko_spread) <= 1e-6) {
        const double u_star = (std::log(b_c) - m_u) / l_c;
        truncated_upper_std_moments(u_star, mu_u_out, var_u_out);
        return;
    }

    const double b_lo = std::min(b_c, b_c + ko_spread);
    const double b_hi = std::max(b_c, b_c + ko_spread);
    if (b_hi <= 0.0) {
        mu_u_out = 0.0;
        var_u_out = 1.0;
        return;
    }
    const double db = b_hi - b_lo;
    const double u_hi = (std::log(b_hi) - m_u) / l_c;
    const bool has_lo = b_lo > 0.0;
    const double u_lo = has_lo ? (std::log(b_lo) - m_u) / l_c : -1e100;

    if (u_hi >= -4.0) {
        const double phi_hi = phi_std(u_hi);
        const double Phi_hi = ndtr_std(u_hi);
        const double Phi_hi_m = ndtr_std(u_hi - l_c);
        const double phi_lo = has_lo ? phi_std(u_lo) : 0.0;
        const double Phi_lo = has_lo ? ndtr_std(u_lo) : 0.0;
        const double Phi_lo_m = has_lo ? ndtr_std(u_lo - l_c) : 0.0;
        const double m0 = (b_hi * Phi_hi - b_lo * Phi_lo - f_c * (Phi_hi_m - Phi_lo_m)) / db;
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double m1 = -(l_c * f_c / db) * (Phi_hi_m - Phi_lo_m);
        const double m2 = m0 + l_c * m1 + (l_c / db) * (b_hi * phi_hi - b_lo * phi_lo);
        mu_u_out = m1 / m0;
        var_u_out = std::max(m2 / m0 - mu_u_out * mu_u_out, 1e-12);
    } else {
        const double exp_lo = has_lo ? std::exp(-0.5 * (u_lo - u_hi) * (u_lo + u_hi)) : 0.0;
        const double r_hi = kSqrtHalfPi * cody_erfcx(-u_hi * kInvSqrt2);
        const double r_hi_m = kSqrtHalfPi * cody_erfcx(-(u_hi - l_c) * kInvSqrt2);
        const double r_lo = has_lo ? kSqrtHalfPi * cody_erfcx(-u_lo * kInvSqrt2) : 0.0;
        const double r_lo_m = has_lo ? kSqrtHalfPi * cody_erfcx(-(u_lo - l_c) * kInvSqrt2) : 0.0;
        const double m0 = (b_hi * (r_hi - r_hi_m) - b_lo * exp_lo * (r_lo - r_lo_m)) / db;
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double m1 = -(l_c / db) * (b_hi * r_hi_m - b_lo * exp_lo * r_lo_m);
        const double m2 = m0 + l_c * m1 + (l_c / db) * (b_hi - b_lo * exp_lo);
        mu_u_out = m1 / m0;
        var_u_out = std::max(m2 / m0 - mu_u_out * mu_u_out, 1e-12);
    }
}

inline void ramp_survival_post_moments_normal(
    double mu_cond,
    double sig_cond,
    double b_ko,
    double ko_spread,
    double& mu_u_out,
    double& var_u_out
) noexcept {
    const double s_c = sig_cond > 1e-12 ? sig_cond : 1e-12;
    if (std::abs(ko_spread) <= 1e-6) {
        const double u_star = (b_ko - mu_cond) / s_c;
        truncated_upper_std_moments(u_star, mu_u_out, var_u_out);
        return;
    }
    const double b_lo = std::min(b_ko, b_ko + ko_spread);
    const double b_hi = std::max(b_ko, b_ko + ko_spread);
    const double u_lo = (b_lo - mu_cond) / s_c;
    const double u_hi = (b_hi - mu_cond) / s_c;
    const double du = u_hi - u_lo;
    if (du <= 1e-12) {
        truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
        return;
    }
    if (u_hi >= -4.0) {
        const double phi_hi = phi_std(u_hi);
        const double phi_lo = phi_std(u_lo);
        const double Phi_hi = ndtr_std(u_hi);
        const double Phi_lo = ndtr_std(u_lo);
        const double m0 = ((u_hi * Phi_hi + phi_hi) - (u_lo * Phi_lo + phi_lo)) / du;
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double m1 = -(Phi_hi - Phi_lo) / du;
        const double m2 = m0 + (phi_hi - phi_lo) / du;
        mu_u_out = m1 / m0;
        var_u_out = std::max(m2 / m0 - mu_u_out * mu_u_out, 1e-12);
    } else {
        const double exp_lo = std::exp(-0.5 * (u_lo - u_hi) * (u_lo + u_hi));
        const double r_hi = kSqrtHalfPi * cody_erfcx(-u_hi * kInvSqrt2);
        const double r_lo = kSqrtHalfPi * cody_erfcx(-u_lo * kInvSqrt2);
        const double m0 = ((u_hi * r_hi + 1.0) - exp_lo * (u_lo * r_lo + 1.0)) / du;
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double m1 = -(r_hi - exp_lo * r_lo) / du;
        const double m2 = m0 + (1.0 - exp_lo) / du;
        mu_u_out = m1 / m0;
        var_u_out = std::max(m2 / m0 - mu_u_out * mu_u_out, 1e-12);
    }
}

void clark_one_order(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    const int* order,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double* w_out,
    double* rho_all_out
) noexcept {
    static constexpr int kStackD = 32;
    std::array<double, kStackD> f_stk{}, v_stk{}, w_stk{}, rc_stk{}, ra_stk{};
    std::vector<double> f_heap, v_heap, w_heap, rc_heap, ra_heap;
    double* f_ord = f_stk.data();
    double* v_ord = v_stk.data();
    double* w_ord = w_stk.data();
    double* rho_cur = rc_stk.data();
    double* rho_all = ra_stk.data();
    if (d > kStackD) {
        f_heap.resize(d);
        v_heap.resize(d);
        w_heap.assign(d, 0.0);
        rc_heap.resize(d);
        f_ord = f_heap.data();
        v_ord = v_heap.data();
        w_ord = w_heap.data();
        rho_cur = rc_heap.data();
        if (rho_all_out != nullptr) {
            ra_heap.resize(d);
            rho_all = ra_heap.data();
        }
    }
    for (int i = 0; i < d; ++i) {
        const int oi = order[i];
        f_ord[i] = f_in[oi] > 1e-16 ? f_in[oi] : 1e-16;
        v_ord[i] = v_in[oi] > 1e-12 ? v_in[oi] : 1e-12;
        w_ord[i] = 0.0;
        rho_cur[i] = corr[order[0] * d + oi];
    }
    w_ord[0] = 1.0;
    double f_cur = f_ord[0];
    double v_cur = v_ord[0];

    if (rho_all_out != nullptr) {
        for (int i = 0; i < d; ++i) {
            rho_all[i] = corr[order[0] * d + i];
        }
    }

    for (int j = 1; j < d; ++j) {
        const double f_2 = f_ord[j];
        const double v_2 = v_ord[j];
        double rho_12 = std::clamp(rho_cur[j], -0.999999, 0.999999);
        const double var_m = std::max(v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2, 1e-16);
        const double sig_m = std::sqrt(var_m);
        const double d1 = (std::log(f_cur / f_2) + 0.5 * var_m) / sig_m;
        const double d2 = d1 - sig_m;

        const double w1 = is_best_of ? ndtr_std(d1) : ndtr_std(-d1);
        const double w2 = is_best_of ? ndtr_std(-d2) : ndtr_std(d2);
        const double e_m1 = std::max(f_cur * w1 + f_2 * w2, 1e-16);

        for (int p = 0; p < j; ++p) {
            w_ord[p] *= w1;
        }
        w_ord[j] = w2;

        const double f1_sq = (f_cur * f_cur) * std::exp(v_cur * v_cur);
        const double f2_sq = (f_2 * f_2) * std::exp(v_2 * v_2);
        const double sig_m2 = 2.0 * sig_m;
        const double var_m2 = sig_m2 * sig_m2;
        const double d1_sq = (std::log(std::max(f1_sq, 1e-30) / std::max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2;
        const double d2_sq = d1_sq - sig_m2;
        const double e_m2 = is_best_of
            ? (f1_sq * ndtr_std(d1_sq) + f2_sq * ndtr_std(-d2_sq))
            : (f1_sq * ndtr_std(-d1_sq) + f2_sq * ndtr_std(d2_sq));

        const double ratio_m2 = e_m2 / std::max(e_m1 * e_m1, 1e-30);
        const double v_new = std::sqrt(ratio_m2 > 1.0 + 1e-16 ? std::log(ratio_m2) : 1e-16);

        const int oj = order[j];
        for (int k = j + 1; k < d; ++k) {
            const double v_k = v_ord[k];
            const int ok = order[k];
            const double f1_k = f_cur * std::exp(rho_cur[k] * v_cur * v_k);
            const double f2_k = f_2 * std::exp(corr[oj * d + ok] * v_2 * v_k);
            const double d1_k = (std::log(std::max(f1_k, 1e-16) / std::max(f2_k, 1e-16)) + 0.5 * var_m) / sig_m;
            const double d2_k = d1_k - sig_m;
            const double e_mp = is_best_of
                ? (f1_k * ndtr_std(d1_k) + f2_k * ndtr_std(-d2_k))
                : (f1_k * ndtr_std(-d1_k) + f2_k * ndtr_std(d2_k));
            const double cov_log = std::log(std::max(e_mp / e_m1, 1e-16));
            const double denom = std::max(v_new * v_k, 1e-16);
            rho_cur[k] = std::clamp(cov_log / denom, -0.999999, 0.999999);
        }

        if (rho_all_out != nullptr) {
            for (int i = 0; i < d; ++i) {
                const double v_i = v_in[i] > 1e-12 ? v_in[i] : 1e-12;
                const double f1_i = f_cur * std::exp(rho_all[i] * v_cur * v_i);
                const double f2_i = f_2 * std::exp(corr[oj * d + i] * v_2 * v_i);
                const double d1_i = (std::log(std::max(f1_i, 1e-16) / std::max(f2_i, 1e-16)) + 0.5 * var_m) / sig_m;
                const double d2_i = d1_i - sig_m;
                const double e_mp = is_best_of
                    ? (f1_i * ndtr_std(d1_i) + f2_i * ndtr_std(-d2_i))
                    : (f1_i * ndtr_std(-d1_i) + f2_i * ndtr_std(d2_i));
                const double cov_log = std::log(std::max(e_mp / e_m1, 1e-16));
                rho_all[i] = std::clamp(cov_log / std::max(v_new * v_i, 1e-16), -0.999999, 0.999999);
            }
        }

        f_cur = e_m1;
        v_cur = v_new;
    }

    f_out = f_cur;
    v_out = v_cur;
    if (w_out != nullptr) {
        for (int k = 0; k < d; ++k) {
            w_out[order[k]] = w_ord[k];
        }
    }
    if (rho_all_out != nullptr) {
        for (int i = 0; i < d; ++i) {
            rho_all_out[i] = rho_all[i];
        }
    }
}

void clark_symmetrized(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double* w_out
) noexcept {
    if (d == 1) {
        f_out = f_in[0] > 1e-16 ? f_in[0] : 1e-16;
        v_out = v_in[0] > 1e-12 ? v_in[0] : 1e-12;
        w_out[0] = 1.0;
        return;
    }

    double g_sep = 1.0;
    for (int i = 0; i < d; ++i) {
        const double fi = f_in[i] > 1e-16 ? f_in[i] : 1e-16;
        for (int j = i + 1; j < d; ++j) {
            const double fj = f_in[j] > 1e-16 ? f_in[j] : 1e-16;
            const double dl = std::log(fi / fj);
            const double p_ij = std::exp(-(dl * dl) / (2.0 * kTieTau * kTieTau));
            const double one_m = 1.0 - p_ij;
            g_sep *= (one_m * one_m);
        }
    }
    const double w_tie = 1.0 - g_sep;

    static constexpr int kStackD = 32;
    std::array<int, kStackD> ord_f_stk{};
    std::vector<int> ord_f_heap;
    int* ord_f = ord_f_stk.data();
    if (d > kStackD) {
        ord_f_heap.resize(d);
        ord_f = ord_f_heap.data();
    }
    for (int i = 0; i < d; ++i) ord_f[i] = i;
    std::stable_sort(ord_f, ord_f + d, [&](int a, int b) {
        return is_best_of ? (f_in[a] > f_in[b]) : (f_in[a] < f_in[b]);
    });

    if (w_tie < 1e-12) {
        clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, f_out, v_out, w_out, nullptr);
        return;
    }

    std::array<int, kStackD> ord_v_stk{};
    std::vector<int> ord_v_heap;
    int* ord_v = ord_v_stk.data();
    if (d > kStackD) {
        ord_v_heap.resize(d);
        ord_v = ord_v_heap.data();
    }
    for (int i = 0; i < d; ++i) ord_v[i] = i;
    std::stable_sort(ord_v, ord_v + d, [&](int a, int b) {
        return v_in[a] < v_in[b];
    });

    const double v_min_val = v_in[ord_v[0]];
    const double v_max_val = v_in[ord_v[d - 1]];

    double f_acc = 0.0;
    double m2_acc = 0.0;
    std::array<double, kStackD> w_acc_stk{}, w_tmp_stk{};
    std::vector<double> w_acc_heap, w_tmp_heap;
    double* w_acc = w_acc_stk.data();
    double* w_tmp = w_tmp_stk.data();
    if (d > kStackD) {
        w_acc_heap.assign(d, 0.0);
        w_tmp_heap.assign(d, 0.0);
        w_acc = w_acc_heap.data();
        w_tmp = w_tmp_heap.data();
    } else {
        for (int i = 0; i < d; ++i) w_acc[i] = 0.0;
    }

    std::array<int, kStackD> p_stk{};
    std::vector<int> p_heap;
    int* p_buf = p_stk.data();
    if (d > kStackD) {
        p_heap.resize(d);
        p_buf = p_heap.data();
    }

    int n_perms = 0;
    auto accum_perm = [&](const int* p_ord) {
        double fp = 0.0;
        double vp = 0.0;
        clark_one_order(d, f_in, v_in, corr, p_ord, is_best_of, fp, vp, w_tmp, nullptr);
        f_acc += fp;
        m2_acc += fp * fp * std::exp(vp * vp);
        for (int i = 0; i < d; ++i) w_acc[i] += w_tmp[i];
        ++n_perms;
    };

    if (v_max_val - v_min_val <= 1e-5) {
        for (int s = 0; s < d; ++s) {
            for (int k = 0; k < d; ++k) p_buf[k] = ord_v[(s + k) % d];
            accum_perm(p_buf);
            std::reverse(p_buf, p_buf + d);
            accum_perm(p_buf);
        }
    } else {
        accum_perm(ord_v);
        for (int i = 0; i < d; ++i) p_buf[i] = ord_v[d - 1 - i];
        accum_perm(p_buf);
        for (int k = 0; k < d - 1; ++k) {
            if (std::abs(v_in[ord_v[k + 1]] - v_in[ord_v[k]]) <= 1e-5) {
                for (int i = 0; i < d; ++i) p_buf[i] = ord_v[i];
                std::swap(p_buf[k], p_buf[k + 1]);
                accum_perm(p_buf);
                std::reverse(p_buf, p_buf + d);
                accum_perm(p_buf);
            }
        }
    }

    const double inv_k = 1.0 / static_cast<double>(n_perms);
    const double f_sym = f_acc * inv_k;
    const double m2_sym = m2_acc * inv_k;
    const double ratio_sym = m2_sym / std::max(f_sym * f_sym, 1e-30);
    const double v_sym = std::sqrt(ratio_sym > 1.0 + 1e-16 ? std::log(ratio_sym) : 1e-16);
    for (int i = 0; i < d; ++i) w_acc[i] *= inv_k;

    if (g_sep < 1e-15) {
        f_out = f_sym;
        v_out = v_sym;
        for (int i = 0; i < d; ++i) w_out[i] = w_acc[i];
        return;
    }

    double f_f = 0.0;
    double v_f = 0.0;
    std::array<double, kStackD> w_f_stk{};
    std::vector<double> w_f_heap;
    double* w_f = w_f_stk.data();
    if (d > kStackD) {
        w_f_heap.assign(d, 0.0);
        w_f = w_f_heap.data();
    }
    clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, f_f, v_f, w_f, nullptr);

    f_out = g_sep * f_f + w_tie * f_sym;
    const double m2_out = g_sep * (f_f * f_f * std::exp(v_f * v_f)) + w_tie * m2_sym;
    const double ratio_out = m2_out / std::max(f_out * f_out, 1e-30);
    v_out = std::sqrt(ratio_out > 1.0 + 1e-16 ? std::log(ratio_out) : 1e-16);
    for (int i = 0; i < d; ++i) {
        w_out[i] = g_sep * w_f[i] + w_tie * w_acc[i];
    }
}

double clark_corr_min_max(int d, const double* f_in, const double* v_in, const double* corr) noexcept {
    if (d <= 1) {
        return 1.0;
    }
    std::vector<int> ord_min(d), ord_max(d);
    for (int i = 0; i < d; ++i) {
        ord_min[i] = i;
        ord_max[i] = i;
    }
    // Use v-sorted canonical order when forwards tie to avoid arbitrary index bias.
    std::stable_sort(ord_min.begin(), ord_min.end(), [&](int a, int b) {
        if (std::abs(f_in[a] - f_in[b]) > 1e-9) return f_in[a] < f_in[b];
        return v_in[a] < v_in[b];
    });
    std::stable_sort(ord_max.begin(), ord_max.end(), [&](int a, int b) {
        if (std::abs(f_in[a] - f_in[b]) > 1e-9) return f_in[a] > f_in[b];
        return v_in[a] < v_in[b];
    });

    double f_min = 0.0, v_min = 0.0;
    std::vector<double> rho_min_p(d, 0.0);
    clark_one_order(d, f_in, v_in, corr, ord_min.data(), false, f_min, v_min, nullptr, rho_min_p.data());

    // Reduce BestOf while carrying correlation against W_min as an extra tracked variable.
    std::vector<double> f_ord(d), v_ord(d), rho_cur(d);
    for (int i = 0; i < d; ++i) {
        const int oi = ord_max[i];
        f_ord[i] = f_in[oi] > 1e-16 ? f_in[oi] : 1e-16;
        v_ord[i] = v_in[oi] > 1e-12 ? v_in[oi] : 1e-12;
        rho_cur[i] = corr[ord_max[0] * d + oi];
    }
    double f_b = f_ord[0];
    double v_b = v_ord[0];
    double rho_b_min = rho_min_p[ord_max[0]];

    for (int j = 1; j < d; ++j) {
        const double f_2 = f_ord[j];
        const double v_2 = v_ord[j];
        const double rho_12 = std::clamp(rho_cur[j], -0.999999, 0.999999);
        const double var_m = std::max(v_b * v_b + v_2 * v_2 - 2.0 * rho_12 * v_b * v_2, 1e-16);
        const double sig_m = std::sqrt(var_m);
        const double d1 = (std::log(f_b / f_2) + 0.5 * var_m) / sig_m;
        const double d2 = d1 - sig_m;
        const double e_m1 = std::max(f_b * ndtr_std(d1) + f_2 * ndtr_std(-d2), 1e-16);

        const double f1_sq = (f_b * f_b) * std::exp(v_b * v_b);
        const double f2_sq = (f_2 * f_2) * std::exp(v_2 * v_2);
        const double sig_m2 = 2.0 * sig_m;
        const double var_m2 = sig_m2 * sig_m2;
        const double d1_sq = (std::log(std::max(f1_sq, 1e-30) / std::max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2;
        const double d2_sq = d1_sq - sig_m2;
        const double e_m2 = f1_sq * ndtr_std(d1_sq) + f2_sq * ndtr_std(-d2_sq);
        const double ratio_m2 = e_m2 / std::max(e_m1 * e_m1, 1e-30);
        const double v_new = std::sqrt(ratio_m2 > 1.0 + 1e-16 ? std::log(ratio_m2) : 1e-16);

        const int oj = ord_max[j];
        const double f1_min = f_b * std::exp(rho_b_min * v_b * v_min);
        const double f2_min = f_2 * std::exp(rho_min_p[oj] * v_2 * v_min);
        const double d1_min = (std::log(std::max(f1_min, 1e-16) / std::max(f2_min, 1e-16)) + 0.5 * var_m) / sig_m;
        const double d2_min = d1_min - sig_m;
        const double e_b_min = f1_min * ndtr_std(d1_min) + f2_min * ndtr_std(-d2_min);
        rho_b_min = std::clamp(std::log(std::max(e_b_min / e_m1, 1e-16)) / std::max(v_new * v_min, 1e-16), -0.999999, 0.999999);

        for (int k = j + 1; k < d; ++k) {
            const double v_k = v_ord[k];
            const int ok = ord_max[k];
            const double f1_k = f_b * std::exp(rho_cur[k] * v_b * v_k);
            const double f2_k = f_2 * std::exp(corr[oj * d + ok] * v_2 * v_k);
            const double d1_k = (std::log(std::max(f1_k, 1e-16) / std::max(f2_k, 1e-16)) + 0.5 * var_m) / sig_m;
            const double d2_k = d1_k - sig_m;
            const double e_mp = f1_k * ndtr_std(d1_k) + f2_k * ndtr_std(-d2_k);
            rho_cur[k] = std::clamp(std::log(std::max(e_mp / e_m1, 1e-16)) / std::max(v_new * v_k, 1e-16), -0.999999, 0.999999);
        }
        f_b = e_m1;
        v_b = v_new;
    }
    return std::clamp(rho_b_min, -0.9999, 0.9999);
}

inline bool is_diff_index(int code) noexcept {
    return code >= 9 && code <= 13;
}

void eval_index_moments(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    const int64_t* inv_lex_perm,
    int code,
    double& f_out,
    double& v_out,
    double* w_out
) noexcept {
    if (d == 1) {
        f_out = f_in[0] > 1e-16 ? f_in[0] : 1e-16;
        v_out = v_in[0] > 1e-12 ? v_in[0] : 1e-12;
        w_out[0] = 1.0;
        if (is_diff_index(code)) {
            f_out = 0.0;
            v_out = 1e-12;
            w_out[0] = 0.0;
        }
        return;
    }

    auto resolve_ab = [&](int& idx_a, int& idx_b) {
        if (inv_lex_perm != nullptr) {
            idx_a = static_cast<int>(inv_lex_perm[0]);
            idx_b = static_cast<int>(inv_lex_perm[1]);
        } else {
            idx_a = 0;
            idx_b = 1;
        }
    };

    if (code == 0 || code == 14) {
        clark_symmetrized(d, f_in, v_in, corr, false, f_out, v_out, w_out);
        return;
    }
    if (code == 1) {
        clark_symmetrized(d, f_in, v_in, corr, true, f_out, v_out, w_out);
        return;
    }
    if (code == 2) {
        double f_sum = 0.0;
        for (int i = 0; i < d; ++i) f_sum += f_in[i];
        const double inv_d = 1.0 / static_cast<double>(d);
        f_out = f_sum * inv_d;
        double m2_sum = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                m2_sum += f_in[i] * f_in[j] * std::exp(corr[i * d + j] * v_in[i] * v_in[j]);
            }
        }
        const double m2 = m2_sum * (inv_d * inv_d);
        const double ratio = m2 / std::max(f_out * f_out, 1e-30);
        v_out = std::sqrt(ratio > 1.0 + 1e-16 ? std::log(ratio) : 1e-16);
        for (int i = 0; i < d; ++i) w_out[i] = inv_d;
        return;
    }
    if (code == 3) {
        // Dispersion: W = sqrt(mean(P_i^2) - mean(P)^2)
        double m2_p_sum = 0.0;
        double m2_avg_sum = 0.0;
        double f_sum = 0.0;
        const double inv_d = 1.0 / static_cast<double>(d);
        for (int i = 0; i < d; ++i) {
            f_sum += f_in[i];
            m2_p_sum += f_in[i] * f_in[i] * std::exp(v_in[i] * v_in[i]);
            for (int j = 0; j < d; ++j) {
                m2_avg_sum += f_in[i] * f_in[j] * std::exp(corr[i * d + j] * v_in[i] * v_in[j]);
            }
        }
        const double ew2 = std::max(m2_p_sum * inv_d - m2_avg_sum * (inv_d * inv_d), 1e-12);
        const double f_mean = f_sum * inv_d;
        double var_f = 0.0;
        for (int i = 0; i < d; ++i) {
            const double diff = f_in[i] - f_mean;
            var_f += diff * diff;
        }
        var_f *= inv_d;
        double sig_approx = std::sqrt(ew2);
        if (var_f > 1e-10) {
            const double ratio = ew2 / var_f;
            v_out = std::sqrt(ratio > 1.0 + 1e-16 ? std::log(ratio) : 1e-16);
            f_out = std::sqrt(var_f);
        } else {
            f_out = sig_approx;
            v_out = 0.35;
        }
        for (int i = 0; i < d; ++i) {
            w_out[i] = inv_d * (f_in[i] - f_mean) / std::max(sig_approx, 1e-8);
        }
        return;
    }
    if (code == 4 || code == 5) {
        int ia = 0, ib = 1;
        resolve_ab(ia, ib);
        if (code == 5) std::swap(ia, ib);
        const double fa = f_in[ia] > 1e-16 ? f_in[ia] : 1e-16;
        const double fb = f_in[ib] > 1e-16 ? f_in[ib] : 1e-16;
        const double va = v_in[ia] > 1e-12 ? v_in[ia] : 1e-12;
        const double vb = v_in[ib] > 1e-12 ? v_in[ib] : 1e-12;
        const double rab = corr[ia * d + ib];
        const double var_r = std::max(va * va + vb * vb - 2.0 * rab * va * vb, 1e-16);
        f_out = (fa / fb) * std::exp(vb * vb - rab * va * vb);
        v_out = std::sqrt(var_r);
        for (int i = 0; i < d; ++i) w_out[i] = 0.0;
        w_out[ia] = 1.0;
        w_out[ib] = -1.0;
        return;
    }
    if (code == 6 || code == 7 || code == 8) {
        double f_num = 0.0, v_num = 0.0, f_den = 0.0, v_den = 0.0;
        std::vector<double> w_num(d, 0.0), w_den(d, 0.0);
        if (code == 6) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_num, v_num, w_num.data());
            clark_symmetrized(d, f_in, v_in, corr, false, f_den, v_den, w_den.data());
        } else if (code == 7) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_num, v_num, w_num.data());
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_den, v_den, w_den.data());
        } else {
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_num, v_num, w_num.data());
            clark_symmetrized(d, f_in, v_in, corr, false, f_den, v_den, w_den.data());
        }
        double rho_nd = 0.0;
        if (code == 6) {
            rho_nd = clark_corr_min_max(d, f_in, v_in, corr);
        } else {
            double cov = 0.0, vn2 = 0.0, vd2 = 0.0;
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    const double cij = corr[i * d + j] * v_in[i] * v_in[j];
                    cov += w_num[i] * w_den[j] * cij;
                    vn2 += w_num[i] * w_num[j] * cij;
                    vd2 += w_den[i] * w_den[j] * cij;
                }
            }
            rho_nd = std::clamp(cov / std::sqrt(std::max(vn2 * vd2, 1e-30)), -0.9999, 0.9999);
        }
        const double var_r = std::max(v_num * v_num + v_den * v_den - 2.0 * rho_nd * v_num * v_den, 1e-16);
        f_out = (f_num / std::max(f_den, 1e-16)) * std::exp(v_den * v_den - rho_nd * v_num * v_den);
        v_out = std::sqrt(var_r);
        for (int i = 0; i < d; ++i) w_out[i] = w_num[i] - w_den[i];
        return;
    }
    if (code == 9 || code == 10) {
        int ia = 0, ib = 1;
        resolve_ab(ia, ib);
        if (code == 10) std::swap(ia, ib);
        const double fa = f_in[ia];
        const double fb = f_in[ib];
        const double va = v_in[ia];
        const double vb = v_in[ib];
        const double rab = corr[ia * d + ib];
        f_out = fa - fb;
        const double var_d = fa * fa * std::expm1(va * va) + fb * fb * std::expm1(vb * vb) - 2.0 * fa * fb * std::expm1(rab * va * vb);
        v_out = std::sqrt(std::max(var_d, 1e-16));
        for (int i = 0; i < d; ++i) w_out[i] = 0.0;
        w_out[ia] = 1.0;
        w_out[ib] = -1.0;
        return;
    }
    // DiffBestWorst (11), DiffBestAverage (12), DiffAverageWorst (13)
    double f_a = 0.0, v_a = 0.0, f_b = 0.0, v_b = 0.0;
    std::vector<double> w_a(d, 0.0), w_b(d, 0.0);
    if (code == 11) {
        clark_symmetrized(d, f_in, v_in, corr, true, f_a, v_a, w_a.data());
        clark_symmetrized(d, f_in, v_in, corr, false, f_b, v_b, w_b.data());
    } else if (code == 12) {
        clark_symmetrized(d, f_in, v_in, corr, true, f_a, v_a, w_a.data());
        eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_b, v_b, w_b.data());
    } else {
        eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_a, v_a, w_a.data());
        clark_symmetrized(d, f_in, v_in, corr, false, f_b, v_b, w_b.data());
    }
    double rho_ab = 0.0;
    if (code == 11) {
        rho_ab = clark_corr_min_max(d, f_in, v_in, corr);
    } else {
        double cov = 0.0, va2 = 0.0, vb2 = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                const double cij = corr[i * d + j] * v_in[i] * v_in[j];
                cov += w_a[i] * w_b[j] * cij;
                va2 += w_a[i] * w_a[j] * cij;
                vb2 += w_b[i] * w_b[j] * cij;
            }
        }
        rho_ab = std::clamp(cov / std::sqrt(std::max(va2 * vb2, 1e-30)), -0.9999, 0.9999);
    }
    f_out = f_a - f_b;
    const double var_d = f_a * f_a * std::expm1(v_a * v_a) + f_b * f_b * std::expm1(v_b * v_b) - 2.0 * f_a * f_b * std::expm1(rho_ab * v_a * v_b);
    v_out = std::sqrt(std::max(var_d, 1e-16));
    for (int i = 0; i < d; ++i) w_out[i] = w_a[i] - w_b[i];
}

double compute_cross_index_corr(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    int code_a,
    const double* w_a,
    int code_b,
    const double* w_b
) noexcept {
    if (code_a == code_b) {
        return 1.0;
    }
    if ((code_a == 0 && code_b == 1) || (code_a == 1 && code_b == 0)) {
        return clark_corr_min_max(d, f_in, v_in, corr);
    }
    double num = 0.0, da = 0.0, db = 0.0;
    const bool da_diff = is_diff_index(code_a);
    const bool db_diff = is_diff_index(code_b);
    for (int i = 0; i < d; ++i) {
        const double wai = w_a[i] * (da_diff ? f_in[i] : 1.0);
        for (int j = 0; j < d; ++j) {
            const double wbj = w_b[j] * (db_diff ? f_in[j] : 1.0);
            const double cij = corr[i * d + j] * v_in[i] * v_in[j];
            num += wai * wbj * cij;
        }
    }
    for (int i = 0; i < d; ++i) {
        const double wai = w_a[i] * (da_diff ? f_in[i] : 1.0);
        const double wbi = w_b[i] * (db_diff ? f_in[i] : 1.0);
        for (int j = 0; j < d; ++j) {
            const double waj = w_a[j] * (da_diff ? f_in[j] : 1.0);
            const double wbj = w_b[j] * (db_diff ? f_in[j] : 1.0);
            const double cij = corr[i * d + j] * v_in[i] * v_in[j];
            da += wai * waj * cij;
            db += wbi * wbj * cij;
        }
    }
    return std::clamp(num / std::sqrt(std::max(da * db, 1e-30)), -0.9999, 0.9999);
}

void lognormal_moments_interval(
    double f,
    double v,
    double a,
    double b,
    int max_order,
    double* out
) noexcept {
    if (b <= a || b <= 0.0) {
        for (int k = 0; k <= max_order; ++k) out[k] = 0.0;
        return;
    }
    const double f_c = f > 1e-16 ? f : 1e-16;
    const double v_c = v > 1e-12 ? v : 1e-12;
    const double m_u = std::log(f_c) - 0.5 * v_c * v_c;
    const double u_b = (std::log(b) - m_u) / v_c;
    const bool has_a = a > 0.0;
    const double u_a = has_a ? (std::log(a) - m_u) / v_c : 0.0;
    double fp = 1.0;
    for (int k = 0; k <= max_order; ++k) {
        const double scale = fp * std::exp(0.5 * static_cast<double>(k * (k - 1)) * v_c * v_c);
        const double pb = ndtr_std(u_b - static_cast<double>(k) * v_c);
        const double pa = has_a ? ndtr_std(u_a - static_cast<double>(k) * v_c) : 0.0;
        out[k] = scale * (pb - pa);
        fp *= f_c;
    }
}

void normal_moments_interval(
    double mu,
    double sig,
    double a,
    double b,
    int max_order,
    double* out
) noexcept {
    if (b <= a) {
        for (int k = 0; k <= max_order; ++k) out[k] = 0.0;
        return;
    }
    const double s_c = sig > 1e-12 ? sig : 1e-12;
    const double u_b = (b - mu) / s_c;
    const bool has_a = a > -1e50;
    const double u_a = has_a ? (a - mu) / s_c : 0.0;
    const double phi_b = phi_std(u_b);
    const double phi_a = has_a ? phi_std(u_a) : 0.0;
    const double j0 = ndtr_std(u_b) - (has_a ? ndtr_std(u_a) : 0.0);
    const double j1 = phi_a - phi_b;
    const double j2 = j0 + (has_a ? u_a * phi_a : 0.0) - u_b * phi_b;
    const double j3 = 2.0 * j1 + (has_a ? u_a * u_a * phi_a : 0.0) - u_b * u_b * phi_b;
    out[0] = j0;
    if (max_order >= 1) out[1] = mu * j0 + s_c * j1;
    if (max_order >= 2) out[2] = mu * mu * j0 + 2.0 * mu * s_c * j1 + s_c * s_c * j2;
    if (max_order >= 3) {
        out[3] = mu * mu * mu * j0 + 3.0 * mu * mu * s_c * j1 + 3.0 * mu * s_c * s_c * j2 + s_c * s_c * s_c * j3;
    }
}

double exact_piecewise_terminal_put(
    double f_or_mu,
    double v_or_sig,
    bool is_normal,
    double strike,
    double lower_strike,
    bool has_active_ki,
    double ki_barrier,
    double ki_spread,
    bool has_expiry_ko,
    double ko_barrier,
    double ko_spread
) noexcept {
    if (strike <= 1e-8) {
        return 0.0;
    }
    const double k_low = lower_strike > 0.0 ? lower_strike : 0.0;
    if (k_low >= strike) {
        return 0.0;
    }

    const double low_bound = is_normal ? -1e60 : 0.0;
    std::array<double, 8> pts{};
    int n_pts = 0;
    auto push_pt = [&](double x) {
        if (x >= low_bound && x <= strike && n_pts < 8) {
            pts[n_pts++] = x;
        }
    };
    push_pt(low_bound);
    push_pt(strike);
    if (k_low > low_bound) push_pt(k_low);
    if (has_active_ki) {
        if (std::abs(ki_spread) <= 1e-6) {
            push_pt(ki_barrier);
        } else {
            push_pt(std::min(ki_barrier, ki_barrier - ki_spread));
            push_pt(std::max(ki_barrier, ki_barrier - ki_spread));
        }
    }
    if (has_expiry_ko) {
        if (std::abs(ko_spread) <= 1e-6) {
            push_pt(ko_barrier);
        } else {
            push_pt(std::min(ko_barrier, ko_barrier + ko_spread));
            push_pt(std::max(ko_barrier, ko_barrier + ko_spread));
        }
    }
    for (int i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        int j = i - 1;
        while (j >= 0 && pts[j] > key) {
            pts[j + 1] = pts[j];
            --j;
        }
        pts[j + 1] = key;
    }

    double total = 0.0;
    for (int idx = 0; idx < n_pts - 1; ++idx) {
        const double a = pts[idx];
        const double b = pts[idx + 1];
        if (b - a <= 1e-14) continue;
        const double mid = (a < -1e30) ? (b - 1.0) : (0.5 * (a + b));

        double c0 = 0.0, c1 = 0.0;
        if (mid <= k_low) {
            c0 = strike - k_low;
            c1 = 0.0;
        } else {
            c0 = strike;
            c1 = -1.0;
        }

        double d0 = 1.0, d1 = 0.0;
        if (has_active_ki) {
            if (std::abs(ki_spread) <= 1e-6) {
                if (mid >= ki_barrier) continue;
            } else {
                const double b_ki_lo = std::min(ki_barrier, ki_barrier - ki_spread);
                const double b_ki_hi = std::max(ki_barrier, ki_barrier - ki_spread);
                if (mid >= b_ki_hi) continue;
                if (mid > b_ki_lo) {
                    const double db = b_ki_hi - b_ki_lo;
                    d0 = b_ki_hi / db;
                    d1 = -1.0 / db;
                }
            }
        }

        double e0 = 1.0, e1 = 0.0;
        if (has_expiry_ko) {
            if (std::abs(ko_spread) <= 1e-6) {
                if (mid >= ko_barrier) continue;
            } else {
                const double b_ko_lo = std::min(ko_barrier, ko_barrier + ko_spread);
                const double b_ko_hi = std::max(ko_barrier, ko_barrier + ko_spread);
                if (mid >= b_ko_hi) continue;
                if (mid > b_ko_lo) {
                    const double db = b_ko_hi - b_ko_lo;
                    e0 = b_ko_hi / db;
                    e1 = -1.0 / db;
                }
            }
        }

        const double cd0 = c0 * d0;
        const double cd1 = c0 * d1 + c1 * d0;
        const double cd2 = c1 * d1;
        const double q0 = cd0 * e0;
        const double q1 = cd0 * e1 + cd1 * e0;
        const double q2 = cd1 * e1 + cd2 * e0;
        const double q3 = cd2 * e1;
        const int max_k = (q3 != 0.0) ? 3 : ((q2 != 0.0) ? 2 : 1);
        double moms[4] = {0.0, 0.0, 0.0, 0.0};
        if (is_normal) {
            normal_moments_interval(f_or_mu, v_or_sig, a, b, max_k, moms);
        } else {
            lognormal_moments_interval(f_or_mu, v_or_sig, a, b, max_k, moms);
        }
        total += q0 * moms[0] + q1 * moms[1] + q2 * moms[2] + q3 * moms[3];
    }
    return total;
}

double exact_piecewise_terminal_call(
    double f_or_mu,
    double v_or_sig,
    bool is_normal,
    double k_call,
    bool has_expiry_ko,
    double ko_barrier,
    double ko_spread
) noexcept {
    if (!has_expiry_ko) {
        return is_normal ? normal_call(f_or_mu, v_or_sig, k_call) : black76_call(f_or_mu, k_call, v_or_sig);
    }
    if (std::abs(ko_spread) <= 1e-6) {
        if (k_call >= ko_barrier) return 0.0;
        double moms[2] = {0.0, 0.0};
        if (is_normal) normal_moments_interval(f_or_mu, v_or_sig, k_call, ko_barrier, 1, moms);
        else lognormal_moments_interval(f_or_mu, v_or_sig, k_call, ko_barrier, 1, moms);
        return moms[1] - k_call * moms[0];
    }
    const double b_ko_lo = std::min(ko_barrier, ko_barrier + ko_spread);
    const double b_ko_hi = std::max(ko_barrier, ko_barrier + ko_spread);
    if (k_call >= b_ko_hi) return 0.0;

    std::array<double, 4> pts{};
    int n_pts = 0;
    pts[n_pts++] = k_call;
    if (b_ko_lo > k_call) pts[n_pts++] = b_ko_lo;
    pts[n_pts++] = b_ko_hi;
    for (int i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        int j = i - 1;
        while (j >= 0 && pts[j] > key) {
            pts[j + 1] = pts[j];
            --j;
        }
        pts[j + 1] = key;
    }

    double total = 0.0;
    for (int idx = 0; idx < n_pts - 1; ++idx) {
        const double a = pts[idx];
        const double b = pts[idx + 1];
        if (b - a <= 1e-14) continue;
        const double mid = 0.5 * (a + b);
        double e0 = 1.0, e1 = 0.0;
        if (mid > b_ko_lo) {
            const double db = b_ko_hi - b_ko_lo;
            e0 = b_ko_hi / db;
            e1 = -1.0 / db;
        }
        const double q0 = -k_call * e0;
        const double q1 = e0 - k_call * e1;
        const double q2 = e1;
        double moms[3] = {0.0, 0.0, 0.0};
        const int max_k = (q2 != 0.0) ? 2 : 1;
        if (is_normal) normal_moments_interval(f_or_mu, v_or_sig, a, b, max_k, moms);
        else lognormal_moments_interval(f_or_mu, v_or_sig, a, b, max_k, moms);
        total += q0 * moms[0] + q1 * moms[1] + q2 * moms[2];
    }
    return total;
}

// 16-point Gauss-Legendre nodes/weights on [-1, 1] for smooth bivariate OneStar integration.
constexpr std::array<double, 16> kGL16X = {
    -0.9894009349916499325961542, -0.9445750230732325760779884,
    -0.8656312023878317438804679, -0.7554044083550030338951012,
    -0.6178762444026437484466718, -0.4580167776572273863424194,
    -0.2816035507792589132304605, -0.0950125098376374401853193,
     0.0950125098376374401853193,  0.2816035507792589132304605,
     0.4580167776572273863424194,  0.6178762444026437484466718,
     0.7554044083550030338951012,  0.8656312023878317438804679,
     0.9445750230732325760779884,  0.9894009349916499325961542
};
constexpr std::array<double, 16> kGL16W = {
    0.0271524594117540948517806, 0.0622535239386478928628438,
    0.0951585116824927848099251, 0.1246289712555338720524763,
    0.1495959888165767320815017, 0.1691565193950025381893121,
    0.1826034150449235888667637, 0.1894506104550684962853967,
    0.1894506104550684962853967, 0.1826034150449235888667637,
    0.1691565193950025381893121, 0.1495959888165767320815017,
    0.1246289712555338720524763, 0.0951585116824927848099251,
    0.0622535239386478928628438, 0.0271524594117540948517806
};

constexpr std::array<double, 16> kGL16YNorm = {
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
    2.555646775436514
};

inline double onestar_payoff_fn(
    double w,
    double strike,
    double k_low,
    bool has_active_ki,
    double ki_barrier,
    double ki_spread,
    bool use_direct_ko,
    double ko_b_term,
    double ko_spread
) noexcept {
    if (w >= strike) {
        return 0.0;
    }
    double pay = strike - w;
    if (k_low > 0.0) {
        pay = std::min(pay, std::max(strike - k_low, 0.0));
    }
    if (has_active_ki) {
        if (std::abs(ki_spread) <= 1e-6) {
            if (w >= ki_barrier) return 0.0;
        } else {
            const double blo = std::min(ki_barrier, ki_barrier - ki_spread);
            const double bhi = std::max(ki_barrier, ki_barrier - ki_spread);
            pay *= std::clamp((bhi - w) / (bhi - blo), 0.0, 1.0);
        }
    }
    if (use_direct_ko) {
        if (std::abs(ko_spread) <= 1e-6) {
            if (w >= ko_b_term) return 0.0;
        } else {
            const double blo = std::min(ko_b_term, ko_b_term + ko_spread);
            const double bhi = std::max(ko_b_term, ko_b_term + ko_spread);
            pay *= std::clamp((bhi - w) / (bhi - blo), 0.0, 1.0);
        }
    }
    return pay;
}

void eval_1factor_onestar(
    int d,
    const double* f_c,
    const double* v_c,
    const double* corr,
    double strike,
    double lower_strike,
    bool has_active_ki,
    double ki_barrier,
    double ki_spread,
    bool use_direct_ko,
    double ko_b_term,
    double ko_spread,
    double one_star_strike,
    double one_star_coupon,
    double surv_cur,
    double surv_prev_term,
    double option_gearing,
    double& l_put_out,
    double& l_onestar_out
) noexcept {
    l_put_out = 0.0;
    l_onestar_out = 0.0;
    const double k_low = lower_strike > 0.0 ? lower_strike : 0.0;
    const double log_ks = std::log(std::max(one_star_strike, 1e-16));

    if (d == 1) {
        const double f0 = std::max(f_c[0], 1e-16);
        const double v0 = std::max(v_c[0], 1e-12);
        const double m0 = std::log(f0) - 0.5 * v0 * v0;
        if (one_star_coupon > 0.0) {
            if (use_direct_ko) {
                double p_os = 0.0;
                if (std::abs(ko_spread) <= 1e-6) {
                    if (ko_b_term > one_star_strike) {
                        p_os = std::max(ndtr_std((std::log(ko_b_term) - m0) / v0) - ndtr_std((log_ks - m0) / v0), 0.0);
                    }
                } else {
                    const double blo = std::min(ko_b_term, ko_b_term + ko_spread);
                    const double bhi = std::max(ko_b_term, ko_b_term + ko_spread);
                    const double mid = 0.5 * (blo + bhi);
                    const double hw = 0.5 * (bhi - blo);
                    for (int q = 0; q < 16; ++q) {
                        const double u = std::max(mid + hw * kGL16X[q], 1e-16);
                        if (u > one_star_strike) {
                            p_os += 0.5 * kGL16W[q] * std::max(ndtr_std((std::log(u) - m0) / v0) - ndtr_std((log_ks - m0) / v0), 0.0);
                        }
                    }
                }
                l_onestar_out = surv_prev_term * one_star_coupon * p_os;
            } else {
                l_onestar_out = surv_cur * one_star_coupon * digital_call_spread_lognormal(f0, v0, one_star_strike, 0.0);
            }
        }
        if (strike > 1e-6 && option_gearing > 0.0 && one_star_strike > 1e-16) {
            const double hi = std::min(strike, one_star_strike);
            if (hi > 0.0) {
                const double u_hi = (std::log(hi) - m0) / v0;
                const double u_lo = u_hi - 12.0;
                const double mid_u = 0.5 * (u_lo + u_hi);
                const double hw_u = 0.5 * (u_hi - u_lo);
                double acc = 0.0;
                for (int q = 0; q < 16; ++q) {
                    const double u = mid_u + hw_u * kGL16X[q];
                    const double w = std::exp(m0 + v0 * u);
                    acc += kGL16W[q] * phi_std(u) * onestar_payoff_fn(w, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
                }
                l_put_out = (use_direct_ko ? surv_prev_term : surv_cur) * hw_u * acc;
            }
        }
        return;
    }

    static constexpr int kStackD = 32;
    std::array<double, kStackD> b_stk{}, mb_stk{}, se_stk{}, mu_stk{}, vb_stk{}, vs_stk{};
    std::vector<double> b_hp, mb_hp, se_hp, mu_hp, vb_hp, vs_hp;
    double* b = b_stk.data();
    double* mb = mb_stk.data();
    double* s_eps = se_stk.data();
    double* mu = mu_stk.data();
    double* vb = vb_stk.data();
    double* vs = vs_stk.data();
    if (d > kStackD) {
        b_hp.resize(d); mb_hp.resize(d); se_hp.resize(d); mu_hp.resize(d); vb_hp.resize(d); vs_hp.resize(d);
        b = b_hp.data(); mb = mb_hp.data(); s_eps = se_hp.data(); mu = mu_hp.data(); vb = vb_hp.data(); vs = vs_hp.data();
    }

    double sum_corr = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) sum_corr += corr[i * d + j];
    }
    const double rho_off = std::clamp((sum_corr - static_cast<double>(d)) / static_cast<double>(d * (d - 1)), 0.0, 0.999);
    const double b0 = std::sqrt(rho_off);
    for (int i = 0; i < d; ++i) b[i] = b0;

    for (int it = 0; it < 8; ++it) {
        double n2 = 0.0;
        for (int i = 0; i < d; ++i) {
            double acc = b[i] * b[i] * b[i];
            for (int j = 0; j < d; ++j) {
                if (j != i) acc += corr[i * d + j] * b[j];
            }
            mb[i] = acc;
            n2 += acc * acc;
        }
        const double inv_n = 1.0 / std::sqrt(std::max(n2, 1e-30));
        double sum_u = 0.0;
        for (int i = 0; i < d; ++i) {
            mb[i] *= inv_n;
            sum_u += mb[i];
        }
        const double sgn = (sum_u < 0.0) ? -1.0 : 1.0;
        for (int i = 0; i < d; ++i) mb[i] *= sgn;
        double lam = 0.0;
        for (int i = 0; i < d; ++i) {
            double mu_i = (b[i] * b[i]) * mb[i];
            for (int j = 0; j < d; ++j) {
                if (j != i) mu_i += corr[i * d + j] * mb[j];
            }
            lam += mb[i] * mu_i;
        }
        const double s_lam = std::sqrt(std::max(lam, 0.0));
        for (int i = 0; i < d; ++i) {
            b[i] = std::clamp(mb[i] * s_lam, 0.0, 0.999);
        }
    }

    for (int i = 0; i < d; ++i) {
        const double fi = std::max(f_c[i], 1e-16);
        const double vi = std::max(v_c[i], 1e-12);
        s_eps[i] = std::sqrt(std::max(1.0 - b[i] * b[i], 1e-12));
        mu[i] = std::log(fi) - 0.5 * vi * vi;
        vb[i] = vi * b[i];
        vs[i] = std::max(vi * s_eps[i], 1e-12);
    }

    std::array<double, 16> p_max_lt_ks_y{};
    for (int q = 0; q < 16; ++q) {
        const double y = kGL16YNorm[q];
        double prod = 1.0;
        for (int i = 0; i < d; ++i) {
            prod *= ndtr_std((log_ks - mu[i] - vb[i] * y) / vs[i]);
        }
        p_max_lt_ks_y[q] = prod;
    }

    if (one_star_coupon > 0.0) {
        double os_acc = 0.0;
        if (use_direct_ko) {
            std::array<double, 4> ko_pts{};
            int n_ko = 1;
            if (std::abs(ko_spread) <= 1e-6) {
                ko_pts[0] = ko_b_term;
            } else {
                const double blo = std::min(ko_b_term, ko_b_term + ko_spread);
                const double bhi = std::max(ko_b_term, ko_b_term + ko_spread);
                const double mid = 0.5 * (blo + bhi);
                const double hw = 0.5 * (bhi - blo);
                // 4-point GL on [-1, 1]
                static constexpr double gx4[4] = {-0.86113631159405258, -0.33998104358485626, 0.33998104358485626, 0.86113631159405258};
                for (int r = 0; r < 4; ++r) ko_pts[r] = std::max(mid + hw * gx4[r], 1e-16);
                n_ko = 4;
            }
            static constexpr double gw4[4] = {0.34785484513745386, 0.65214515486254614, 0.65214515486254614, 0.34785484513745386};
            for (int q = 0; q < 16; ++q) {
                const double y = kGL16YNorm[q];
                const double p_mks = p_max_lt_ks_y[q];
                double p_sub = 0.0;
                for (int r = 0; r < n_ko; ++r) {
                    const double log_ko = std::log(std::max(ko_pts[r], 1e-16));
                    const bool ko_lt_ks = ko_pts[r] < one_star_strike;
                    double p_ge_ko = 1.0;
                    double p_in = 1.0;
                    for (int i = 0; i < d; ++i) {
                        const double c_ks = ndtr_std((log_ks - mu[i] - vb[i] * y) / vs[i]);
                        const double c_ko = ndtr_std((log_ko - mu[i] - vb[i] * y) / vs[i]);
                        p_ge_ko *= (1.0 - c_ko);
                        if (ko_lt_ks) p_in *= std::max(c_ks - c_ko, 0.0);
                    }
                    const double p_term = (1.0 - p_ge_ko) - (p_mks - (ko_lt_ks ? p_in : 0.0));
                    p_sub += (n_ko == 1 ? 1.0 : 0.5 * gw4[r]) * std::max(p_term, 0.0);
                }
                os_acc += 0.5 * kGL16W[q] * p_sub;
            }
            l_onestar_out = surv_prev_term * one_star_coupon * os_acc;
        } else {
            for (int q = 0; q < 16; ++q) {
                os_acc += 0.5 * kGL16W[q] * (1.0 - p_max_lt_ks_y[q]);
            }
            l_onestar_out = surv_cur * one_star_coupon * os_acc;
        }
    }

    if (strike > 1e-6 && option_gearing > 0.0) {
        auto eval_cum_joint = [&](double x) noexcept -> double {
            if (x <= 0.0) return 0.0;
            const double log_x = std::log(x);
            double res = 0.0;
            for (int q = 0; q < 16; ++q) {
                const double y = kGL16YNorm[q];
                double p_in = 1.0;
                for (int i = 0; i < d; ++i) {
                    const double c_ks = ndtr_std((log_ks - mu[i] - vb[i] * y) / vs[i]);
                    const double c_x = ndtr_std((log_x - mu[i] - vb[i] * y) / vs[i]);
                    p_in *= std::max(c_ks - c_x, 0.0);
                }
                res += 0.5 * kGL16W[q] * std::max(p_max_lt_ks_y[q] - p_in, 0.0);
            }
            return res;
        };

        std::array<double, 8> pts{};
        int n_pts = 0;
        auto push_pt = [&](double x) {
            if (x >= 0.0 && x <= strike && n_pts < 8) pts[n_pts++] = x;
        };
        push_pt(0.0);
        push_pt(strike);
        if (k_low > 0.0 && k_low < strike) push_pt(k_low);
        if (has_active_ki) {
            if (std::abs(ki_spread) <= 1e-6) {
                push_pt(ki_barrier);
            } else {
                push_pt(std::min(ki_barrier, ki_barrier - ki_spread));
                push_pt(std::max(ki_barrier, ki_barrier - ki_spread));
            }
        }
        if (use_direct_ko) {
            if (std::abs(ko_spread) <= 1e-6) {
                push_pt(ko_b_term);
            } else {
                push_pt(std::min(ko_b_term, ko_b_term + ko_spread));
                push_pt(std::max(ko_b_term, ko_b_term + ko_spread));
            }
        }
        for (int i = 1; i < n_pts; ++i) {
            const double key = pts[i];
            int j = i - 1;
            while (j >= 0 && pts[j] > key) {
                pts[j + 1] = pts[j];
                --j;
            }
            pts[j + 1] = key;
        }

        double put_acc = 0.0;
        for (int idx = 0; idx < n_pts - 1; ++idx) {
            const double a = pts[idx];
            const double b_pt = pts[idx + 1];
            const double span = b_pt - a;
            if (span <= 1e-12) continue;
            const double mid_x = 0.5 * (a + b_pt);
            const double hw_x = 0.5 * span;
            const double ga = onestar_payoff_fn(a + 1e-10 * span, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            const double gb_m = onestar_payoff_fn(b_pt - 1e-10 * span, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            const double gb_p = onestar_payoff_fn(b_pt + 1e-10, strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            if (ga > 0.0 || gb_m > 0.0) {
                const bool linear_unit = !has_active_ki && !use_direct_ko && a >= k_low - 1e-12;
                for (int r = 0; r < 16; ++r) {
                    const double xr = mid_x + hw_x * kGL16X[r];
                    double neg_gp = 1.0;
                    if (!linear_unit) {
                        const double h_fd = 1e-7 * std::max(span, 1e-3);
                        const double g_p = onestar_payoff_fn(std::min(xr + h_fd, b_pt - 1e-12 * span), strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
                        const double g_m = onestar_payoff_fn(std::max(xr - h_fd, a + 1e-12 * span), strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
                        const double dx_fd = std::min(xr + h_fd, b_pt - 1e-12 * span) - std::max(xr - h_fd, a + 1e-12 * span);
                        neg_gp = dx_fd > 0.0 ? -(g_p - g_m) / dx_fd : 0.0;
                    }
                    if (std::abs(neg_gp) > 1e-15) {
                        put_acc += hw_x * kGL16W[r] * neg_gp * eval_cum_joint(xr);
                    }
                }
            }
            const double jump = gb_m - gb_p;
            if (std::abs(jump) > 1e-12) {
                put_acc += jump * eval_cum_joint(b_pt);
            }
        }
        l_put_out = (use_direct_ko ? surv_prev_term : surv_cur) * put_acc;
    }
}

}  // namespace

extern "C" {

void tier1_clamped_cubic_coeffs_1d(
    int64_t k,
    const double* x,
    const double* y,
    double left_slope,
    double right_slope,
    double* coeffs_out
) {
    if (k < 2) {
        return;
    }
    const int64_t n_seg = k - 1;
    std::vector<double> h(n_seg), s(n_seg), d(k, 0.0);
    for (int64_t i = 0; i < n_seg; ++i) {
        h[i] = x[i + 1] - x[i];
        s[i] = (y[i + 1] - y[i]) / h[i];
    }
    d[0] = left_slope;
    d[k - 1] = right_slope;

    const int64_t n_int = k - 2;
    if (n_int > 0) {
        std::vector<double> c_star(n_int, 0.0), d_star(n_int, 0.0);
        const double diag0 = 2.0 * (h[0] + h[1]);
        double rhs0 = 3.0 * (h[1] * s[0] + h[0] * s[1]) - h[1] * left_slope;
        if (n_int == 1) {
            rhs0 -= h[0] * right_slope;
        }
        c_star[0] = (n_int > 1) ? (h[0] / diag0) : 0.0;
        d_star[0] = rhs0 / diag0;

        for (int64_t i = 1; i < n_int; ++i) {
            const double hp = h[i];
            const double hn = h[i + 1];
            const double diag = 2.0 * (hp + hn);
            double rhs = 3.0 * (hn * s[i] + hp * s[i + 1]);
            if (i == n_int - 1) {
                rhs -= hp * right_slope;
            }
            const double m_piv = diag - hn * c_star[i - 1];
            if (i < n_int - 1) {
                c_star[i] = hp / m_piv;
            }
            d_star[i] = (rhs - hn * d_star[i - 1]) / m_piv;
        }
        d[k - 2] = d_star[n_int - 1];
        for (int64_t i = n_int - 2; i >= 0; --i) {
            d[i + 1] = d_star[i] - c_star[i] * d[i + 2];
        }
    }

    for (int64_t i = 0; i < n_seg; ++i) {
        const double hi = h[i];
        const double c3 = y[i];
        const double c2 = d[i];
        const double c1 = (3.0 * s[i] - 2.0 * d[i] - d[i + 1]) / hi;
        const double c0 = (d[i] + d[i + 1] - 2.0 * s[i]) / (hi * hi);
        coeffs_out[0 * n_seg + i] = c0;
        coeffs_out[1 * n_seg + i] = c1;
        coeffs_out[2 * n_seg + i] = c2;
        coeffs_out[3 * n_seg + i] = c3;
    }
}

void tier1_eval_cubic_with_derivs(
    int64_t k,
    const double* x,
    const double* coeffs,
    int64_t n_q,
    const double* x_query,
    int32_t extrap_left_linear,
    int32_t extrap_right_linear,
    double* val_out,
    double* dval_out,
    double* d2val_out
) {
    if (k < 2 || n_q <= 0) {
        return;
    }
    const int64_t n_seg = k - 1;
    const double x_min = x[0];
    const double x_max = x[k - 1];
    const double h_last = x[k - 1] - x[k - 2];
    const double c0_end = coeffs[0 * n_seg + (n_seg - 1)];
    const double c1_end = coeffs[1 * n_seg + (n_seg - 1)];
    const double c2_end = coeffs[2 * n_seg + (n_seg - 1)];
    const double c3_end = coeffs[3 * n_seg + (n_seg - 1)];
    const double val_end = ((c0_end * h_last + c1_end) * h_last + c2_end) * h_last + c3_end;
    const double slope_end = (3.0 * c0_end * h_last + 2.0 * c1_end) * h_last + c2_end;
    const double val_start = coeffs[3 * n_seg + 0];
    const double slope_start = coeffs[2 * n_seg + 0];

    for (int64_t q = 0; q < n_q; ++q) {
        const double xq = x_query[q];
        if (xq < x_min) {
            if (extrap_left_linear != 0) {
                val_out[q] = val_start + slope_start * (xq - x_min);
                dval_out[q] = slope_start;
                d2val_out[q] = 0.0;
            } else {
                val_out[q] = val_start;
                dval_out[q] = 0.0;
                d2val_out[q] = 0.0;
            }
        } else if (xq > x_max) {
            if (extrap_right_linear != 0) {
                val_out[q] = val_end + slope_end * (xq - x_max);
                dval_out[q] = slope_end;
                d2val_out[q] = 0.0;
            } else {
                val_out[q] = val_end;
                dval_out[q] = 0.0;
                d2val_out[q] = 0.0;
            }
        } else {
            const double* it = std::upper_bound(x, x + k, xq);
            int64_t idx = static_cast<int64_t>(it - x) - 1;
            if (idx < 0) idx = 0;
            if (idx > n_seg - 1) idx = n_seg - 1;
            const double dx = xq - x[idx];
            const double c0 = coeffs[0 * n_seg + idx];
            const double c1 = coeffs[1 * n_seg + idx];
            const double c2 = coeffs[2 * n_seg + idx];
            const double c3 = coeffs[3 * n_seg + idx];
            val_out[q] = ((c0 * dx + c1) * dx + c2) * dx + c3;
            dval_out[q] = (3.0 * c0 * dx + 2.0 * c1) * dx + c2;
            d2val_out[q] = 6.0 * c0 * dx + 2.0 * c1;
        }
    }
}

void tier1_eval_autocall_bridge_cpu(
    int64_t d_assets,
    int64_t m_rem,
    const double* fwd_perf,
    const double* vol_perf,
    const double* corr,
    const double* tau_obs,
    const double* df_pay,
    const double* ko_barriers,
    const double* is_callable,
    const double* cpn_barriers,
    const double* is_guaranteed_cpn,
    const double* coupon_rates,
    int32_t cpn_type_out,
    int32_t cpn_dir_down,
    const int64_t* inv_lex_perm,
    double strike,
    double lower_strike,
    double ki_barrier,
    int32_t has_ki,
    double i_ki_past,
    double ko_spread,
    double cpn_spread,
    double ki_spread,
    int32_t call_index_code,
    int32_t cpn_index_code,
    int32_t opt_index_code,
    int32_t allow_autocall_on_expiry,
    double c_mem_0,
    int32_t is_memory,
    double autocall_coupon,
    int32_t force_coupons_on_autocall,
    int32_t no_notional_repayment,
    int32_t is_long_basket,
    double option_gearing,
    double part_call_strike,
    double part_call_rate,
    double one_star_strike,
    double one_star_coupon,
    double pv_in_transit,
    double* f_w_out,
    double* v_w_out,
    double* delta_weights_out,
    double* p_stop_out,
    double* surv_out,
    double* h_rate_out,
    double* q_cpn_out,
    double* scalars_out
) {
    (void)tau_obs;
    if (m_rem <= 0 || d_assets <= 0) {
        for (int i = 0; i < 7; ++i) scalars_out[i] = 0.0;
        scalars_out[4] = pv_in_transit;
        return;
    }

    const int d = static_cast<int>(d_assets);
    const int m_tot = static_cast<int>(m_rem);
    const bool call_is_normal = is_diff_index(call_index_code);
    const bool cpn_is_normal = is_diff_index(cpn_index_code);
    const bool opt_is_normal = is_diff_index(opt_index_code);
    const bool cpn_differs = (cpn_index_code != call_index_code);
    const bool opt_differs = (opt_index_code != call_index_code);

    static constexpr int kStackD = 32;
    std::array<double, kStackD> fc_s{}, vc_s{}, wc_s{}, fn_s{}, vn_s{}, wn_s{}, wcp_s{}, wcc_s{}, fcp_s{}, vcp_s{}, fcm_s{}, vcm_s{}, wcm_s{}, fot_s{}, vot_s{}, wot_s{};
    std::vector<double> fc_h, vc_h, wc_h, fn_h, vn_h, wn_h, wcp_h, wcc_h, fcp_h, vcp_h, fcm_h, vcm_h, wcm_h, fot_h, vot_h, wot_h;
    double* f_col = fc_s.data();
    double* v_col = vc_s.data();
    double* w_col = wc_s.data();
    double* f_next = fn_s.data();
    double* v_next = vn_s.data();
    double* w_next = wn_s.data();
    double* w_call_prev = wcp_s.data();
    double* w_call_cur = wcc_s.data();
    double* f_call_prev = fcp_s.data();
    double* v_call_prev = vcp_s.data();
    double* f_cpn_m = fcm_s.data();
    double* v_cpn_m = vcm_s.data();
    double* w_cpn_m = wcm_s.data();
    double* f_opt_t = fot_s.data();
    double* v_opt_t = vot_s.data();
    double* w_opt_t = wot_s.data();
    if (d > kStackD) {
        fc_h.resize(d); vc_h.resize(d); wc_h.resize(d);
        fn_h.resize(d); vn_h.resize(d); wn_h.resize(d);
        wcp_h.resize(d); wcc_h.resize(d); fcp_h.resize(d); vcp_h.resize(d);
        fcm_h.resize(d); vcm_h.resize(d); wcm_h.resize(d);
        fot_h.resize(d); vot_h.resize(d); wot_h.resize(d);
        f_col = fc_h.data(); v_col = vc_h.data(); w_col = wc_h.data();
        f_next = fn_h.data(); v_next = vn_h.data(); w_next = wn_h.data();
        w_call_prev = wcp_h.data(); w_call_cur = wcc_h.data(); f_call_prev = fcp_h.data(); v_call_prev = vcp_h.data();
        f_cpn_m = fcm_h.data(); v_cpn_m = vcm_h.data(); w_cpn_m = wcm_h.data();
        f_opt_t = fot_h.data(); v_opt_t = vot_h.data(); w_opt_t = wot_h.data();
    }

    auto load_period = [&](int m, double* f_dst, double* v_dst) {
        for (int i = 0; i < d; ++i) {
            f_dst[i] = fwd_perf[i * m_tot + m];
            v_dst[i] = vol_perf[i * m_tot + m];
        }
    };

    for (int m = 0; m < m_tot; ++m) {
        load_period(m, f_col, v_col);
        double fw_m = 0.0, vw_m = 0.0;
        eval_index_moments(d, f_col, v_col, corr, inv_lex_perm, call_index_code, fw_m, vw_m, w_col);
        f_w_out[m] = fw_m;
        v_w_out[m] = vw_m;
        for (int i = 0; i < d; ++i) {
            delta_weights_out[i * m_tot + m] = w_col[i];
        }
    }

    double surv_cur = 1.0;
    double mem_cur = c_mem_0;
    double cpn_pv_sum = 0.0;
    double notional_pv_sum = 0.0;
    double z_mean = 0.0;
    double z_std = 1.0;
    double z_mean_term = 0.0;
    double z_std_term = 1.0;
    double f_cond_term = f_w_out[m_tot - 1];
    double v_cond_term = v_w_out[m_tot - 1];
    double surv_prev_term = 1.0;

    for (int m = 0; m < m_tot; ++m) {
        const double fw_m = call_is_normal ? f_w_out[m] : std::max(f_w_out[m], 1e-16);
        const double vw_m = std::max(v_w_out[m], 1e-12);

        double f_cond = fw_m;
        double v_cond = vw_m;
        if (z_mean != 0.0 || z_std != 1.0) {
            if (call_is_normal) {
                f_cond = fw_m + vw_m * z_mean;
                v_cond = std::max(vw_m * z_std, 1e-12);
            } else {
                f_cond = fw_m * std::exp(vw_m * z_mean - 0.5 * vw_m * vw_m * (1.0 - z_std * z_std));
                v_cond = std::max(vw_m * z_std, 1e-12);
            }
        }
        if (m == m_tot - 1) {
            f_cond_term = f_cond;
            v_cond_term = v_cond;
            z_mean_term = z_mean;
            z_std_term = z_std;
            surv_prev_term = surv_cur;
        }

        const double ko_b_m = ko_barriers[m];
        const bool is_call_m = is_callable[m] > 0.5;
        double h_m = 0.0;
        if (is_call_m) {
            double ko_raw = 0.0;
            if (call_index_code == 14) {
                ko_raw = eval_coupon_prob_dist(f_cond, v_cond, ko_b_m, ko_spread, false, false, false, 14);
            } else if (call_is_normal) {
                ko_raw = digital_call_spread_normal(f_cond, v_cond, ko_b_m, ko_spread);
            } else {
                ko_raw = digital_call_spread_lognormal(f_cond, v_cond, ko_b_m, ko_spread);
            }
            h_m = std::clamp(ko_raw, 0.0, 1.0 - 1e-7);
        }
        h_rate_out[m] = h_m;
        const double p_m = surv_cur * h_m;
        p_stop_out[m] = p_m;
        const double surv_next = surv_cur * (1.0 - h_m);
        surv_out[m] = surv_next;

        double q_m = 1.0;
        if (is_guaranteed_cpn[m] <= 0.5) {
            double f_cpn_cond = f_cond;
            double v_cpn_cond = v_cond;
            if (cpn_differs) {
                load_period(m, f_cpn_m, v_cpn_m);
                for (int i = 0; i < d; ++i) w_call_cur[i] = delta_weights_out[i * m_tot + m];
                double fc_raw = 0.0, vc_raw = 0.0;
                eval_index_moments(d, f_cpn_m, v_cpn_m, corr, inv_lex_perm, cpn_index_code, fc_raw, vc_raw, w_cpn_m);
                const double rho_cc = compute_cross_index_corr(
                    d, f_cpn_m, v_cpn_m, corr, call_index_code, w_call_cur, cpn_index_code, w_cpn_m
                );
                const double z_mc = rho_cc * z_mean;
                const double z_vc = rho_cc * rho_cc * (z_std * z_std) + (1.0 - rho_cc * rho_cc);
                const double z_sc = std::sqrt(std::max(z_vc, 1e-12));
                if (cpn_is_normal) {
                    f_cpn_cond = fc_raw + vc_raw * z_mc;
                    v_cpn_cond = std::max(vc_raw * z_sc, 1e-12);
                } else {
                    const double fc_pos = std::max(fc_raw, 1e-16);
                    f_cpn_cond = fc_pos * std::exp(vc_raw * z_mc - 0.5 * vc_raw * vc_raw * (1.0 - z_vc));
                    v_cpn_cond = std::max(vc_raw * z_sc, 1e-12);
                }
            }
            double cpn_b_m = cpn_barriers[m];
            if (!cpn_differs && force_coupons_on_autocall && is_call_m && cpn_dir_down == 0 && cpn_type_out == 0) {
                if (ko_b_m < cpn_b_m) cpn_b_m = ko_b_m;
            }
            q_m = eval_coupon_prob_dist(
                f_cpn_cond, v_cpn_cond, cpn_b_m, cpn_spread, cpn_is_normal,
                cpn_dir_down != 0, cpn_type_out != 0, cpn_index_code
            );
            if (force_coupons_on_autocall != 0 && is_call_m && h_m > q_m) {
                q_m = h_m;
            }
        }
        q_cpn_out[m] = q_m;

        const double c_m = coupon_rates[m];
        const double eff_c_m = (is_memory != 0) ? (c_m + mem_cur) : c_m;
        const double df_m = df_pay[m];
        cpn_pv_sum += df_m * (surv_cur * q_m * eff_c_m + p_m * autocall_coupon);
        if (is_memory != 0) {
            mem_cur = eff_c_m * (1.0 - q_m);
        }

        if (no_notional_repayment == 0) {
            if (m < m_tot - 1) {
                notional_pv_sum += df_m * p_m;
            } else {
                notional_pv_sum += df_m * (p_m + surv_next);
            }
        }

        if (m < m_tot - 1) {
            double z_post_mean = z_mean;
            double z_post_var = z_std * z_std;
            if (is_call_m) {
                double mu_u = 0.0, var_u = 1.0;
                if (call_is_normal) {
                    ramp_survival_post_moments_normal(f_cond, v_cond, ko_b_m, ko_spread, mu_u, var_u);
                } else {
                    ramp_survival_post_moments_lognormal(f_cond, v_cond, ko_b_m, ko_spread, mu_u, var_u);
                }
                z_post_mean = z_mean + z_std * mu_u;
                z_post_var = std::max(z_std * z_std * var_u, 1e-12);
            }

            if (is_call_m || z_mean != 0.0 || z_std != 1.0) {
                load_period(m, f_call_prev, v_call_prev);
                load_period(m + 1, f_next, v_next);
                for (int i = 0; i < d; ++i) {
                    w_call_prev[i] = delta_weights_out[i * m_tot + m];
                    w_next[i] = delta_weights_out[i * m_tot + (m + 1)];
                }
                double num = 0.0, den_cur = 0.0, den_nxt = 0.0;
                for (int i = 0; i < d; ++i) {
                    const double wi_c = w_call_prev[i] * (call_is_normal ? f_call_prev[i] : 1.0);
                    for (int j = 0; j < d; ++j) {
                        const double wj_n = w_next[j] * (call_is_normal ? f_next[j] : 1.0);
                        const double rho_ij = corr[i * d + j];
                        num += wi_c * wj_n * rho_ij * v_call_prev[i] * std::min(v_call_prev[j], v_next[j]);
                    }
                }
                for (int i = 0; i < d; ++i) {
                    const double wi_c = w_call_prev[i] * (call_is_normal ? f_call_prev[i] : 1.0);
                    const double wi_n = w_next[i] * (call_is_normal ? f_next[i] : 1.0);
                    for (int j = 0; j < d; ++j) {
                        const double wj_c = w_call_prev[j] * (call_is_normal ? f_call_prev[j] : 1.0);
                        const double wj_n = w_next[j] * (call_is_normal ? f_next[j] : 1.0);
                        const double rho_ij = corr[i * d + j];
                        den_cur += wi_c * wj_c * rho_ij * v_call_prev[i] * v_call_prev[j];
                        den_nxt += wi_n * wj_n * rho_ij * v_next[i] * v_next[j];
                    }
                }
                const double denom = std::sqrt(std::max(den_cur, 1e-30)) * std::sqrt(std::max(den_nxt, 1e-30));
                const double r_next = std::clamp(num / std::max(denom, 1e-30), -0.9999, 0.9999);
                z_mean = r_next * z_post_mean;
                z_std = std::sqrt(std::max(r_next * r_next * z_post_var + (1.0 - r_next * r_next), 1e-12));
            }
        }
        surv_cur = surv_next;
    }

    // Terminal option leg (condition on pre-expiry survival if AllowAutocallOnExpiry is active).
    const bool expiry_ko_active = (allow_autocall_on_expiry != 0) && (is_callable[m_tot - 1] > 0.5);

    double f_opt_cond = f_cond_term;
    double v_opt_cond = v_cond_term;
    double rho_call_opt = 1.0;
    if (opt_differs) {
        load_period(m_tot - 1, f_opt_t, v_opt_t);
        for (int i = 0; i < d; ++i) w_call_cur[i] = delta_weights_out[i * m_tot + (m_tot - 1)];
        double fo_raw = 0.0, vo_raw = 0.0;
        eval_index_moments(d, f_opt_t, v_opt_t, corr, inv_lex_perm, opt_index_code, fo_raw, vo_raw, w_opt_t);
        rho_call_opt = compute_cross_index_corr(
            d, f_opt_t, v_opt_t, corr, call_index_code, w_call_cur, opt_index_code, w_opt_t
        );
        const double z_mo = rho_call_opt * z_mean_term;
        const double z_vo = rho_call_opt * rho_call_opt * (z_std_term * z_std_term) + (1.0 - rho_call_opt * rho_call_opt);
        const double z_so = std::sqrt(std::max(z_vo, 1e-12));
        if (opt_is_normal) {
            f_opt_cond = fo_raw + vo_raw * z_mo;
            v_opt_cond = std::max(vo_raw * z_so, 1e-12);
        } else {
            f_opt_cond = std::max(fo_raw, 1e-16) * std::exp(vo_raw * z_mo - 0.5 * vo_raw * vo_raw * (1.0 - z_vo));
            v_opt_cond = std::max(vo_raw * z_so, 1e-12);
        }
    }

    const bool has_active_ki = (has_ki != 0) && (ki_barrier > 0.0) && (i_ki_past < 0.5);
    const bool has_onestar = one_star_strike > 0.0;
    const double ko_b_term = ko_barriers[m_tot - 1];

    double l_put_val = 0.0;
    double l_onestar_val = 0.0;
    if (has_onestar) {
        load_period(m_tot - 1, f_opt_t, v_opt_t);
        const double* f_os_in = f_opt_t;
        const double* v_os_in = v_opt_t;
        if (z_mean_term != 0.0 || z_std_term != 1.0) {
            for (int i = 0; i < d; ++i) w_call_cur[i] = delta_weights_out[i * m_tot + (m_tot - 1)];
            double vc_n2 = 0.0;
            for (int i = 0; i < d; ++i) {
                const double wi = w_call_cur[i] * (call_is_normal ? f_opt_t[i] : 1.0);
                for (int j = 0; j < d; ++j) {
                    const double wj = w_call_cur[j] * (call_is_normal ? f_opt_t[j] : 1.0);
                    vc_n2 += wi * wj * corr[i * d + j] * v_opt_t[i] * v_opt_t[j];
                }
            }
            const double vc_n = std::sqrt(std::max(vc_n2, 1e-30));
            for (int i = 0; i < d; ++i) {
                const double vi = std::max(v_opt_t[i], 1e-12);
                double cov_i = 0.0;
                for (int j = 0; j < d; ++j) {
                    const double wj = w_call_cur[j] * (call_is_normal ? f_opt_t[j] : 1.0);
                    cov_i += corr[i * d + j] * vi * v_opt_t[j] * wj;
                }
                const double rho_ic = std::clamp(cov_i / (vi * vc_n), -0.9999, 0.9999);
                const double z_mc = rho_ic * z_mean_term;
                const double z_vc = rho_ic * rho_ic * (z_std_term * z_std_term) + (1.0 - rho_ic * rho_ic);
                f_next[i] = std::max(f_opt_t[i], 1e-16) * std::exp(vi * z_mc - 0.5 * vi * vi * (1.0 - z_vc));
                v_next[i] = std::max(vi * std::sqrt(std::max(z_vc, 1e-12)), 1e-12);
            }
            f_os_in = f_next;
            v_os_in = v_next;
        }
        const bool use_direct_ko = expiry_ko_active && !opt_differs;
        eval_1factor_onestar(
            d, f_os_in, v_os_in, corr,
            strike, lower_strike,
            has_active_ki, ki_barrier, ki_spread,
            use_direct_ko, ko_b_term, ko_spread,
            one_star_strike, one_star_coupon,
            surv_cur, surv_prev_term, option_gearing,
            l_put_val, l_onestar_val
        );
    } else if (strike > 1e-6 && option_gearing > 0.0) {
        const bool use_direct_ko = expiry_ko_active && !opt_differs;
        const double surv_mult = use_direct_ko ? surv_prev_term : surv_cur;
        const double put_cond = exact_piecewise_terminal_put(
            f_opt_cond, v_opt_cond, opt_is_normal,
            strike, lower_strike,
            has_active_ki, ki_barrier, ki_spread,
            use_direct_ko, ko_b_term, ko_spread
        );
        l_put_val = surv_mult * put_cond;
    }

    double l_call_val = 0.0;
    if (part_call_strike > 0.0 && part_call_rate > 0.0) {
        const bool use_direct_ko = expiry_ko_active && !opt_differs;
        const double surv_mult = use_direct_ko ? surv_prev_term : surv_cur;
        l_call_val = surv_mult * exact_piecewise_terminal_call(
            f_opt_cond, v_opt_cond, opt_is_normal,
            part_call_strike, use_direct_ko, ko_b_term, ko_spread
        );
    }

    const double opt_sign = (is_long_basket != 0) ? -1.0 : 1.0;
    const double opt_pv_val = df_pay[m_tot - 1] * (
        -opt_sign * option_gearing * l_put_val
        + part_call_rate * l_call_val
        + l_onestar_val
    );
    const double pv_val = pv_in_transit + notional_pv_sum + cpn_pv_sum + opt_pv_val;

    scalars_out[0] = l_put_val;
    scalars_out[1] = opt_pv_val;
    scalars_out[2] = notional_pv_sum;
    scalars_out[3] = cpn_pv_sum;
    scalars_out[4] = pv_val;
    scalars_out[5] = l_call_val;
    scalars_out[6] = l_onestar_val;
}

void tier1_eval_index_moments_cpu(
    std::int64_t d_assets,
    const double* f_in,
    const double* v_in,
    const double* corr,
    const std::int64_t* inv_lex_perm,
    std::int32_t index_code,
    double* f_out,
    double* v_out,
    double* w_out
) noexcept {
    double fw = 0.0;
    double vw = 0.0;
    eval_index_moments(
        static_cast<int>(d_assets),
        f_in,
        v_in,
        corr,
        inv_lex_perm,
        static_cast<int>(index_code),
        fw,
        vw,
        w_out
    );
    *f_out = fw;
    *v_out = vw;
}

struct Tier1AutocallArgs {
    std::int64_t d_assets;
    std::int64_t m_rem;
    const double* fwd_perf;
    const double* vol_perf;
    const double* corr;
    const double* tau_obs;
    const double* df_pay;
    const double* ko_barriers;
    const double* is_callable;
    const double* cpn_barriers;
    const double* is_guaranteed_cpn;
    const double* coupon_rates;
    const std::int64_t* inv_lex_perm;
    double strike;
    double lower_strike;
    double ki_barrier;
    double i_ki_past;
    double ko_spread;
    double cpn_spread;
    double ki_spread;
    double c_mem_0;
    double autocall_coupon;
    double option_gearing;
    double part_call_strike;
    double part_call_rate;
    double one_star_strike;
    double one_star_coupon;
    double pv_in_transit;
    std::int32_t cbt_out;
    std::int32_t cbd_down;
    std::int32_t has_ki;
    std::int32_t call_code;
    std::int32_t cpn_code;
    std::int32_t opt_code;
    std::int32_t allow_exp;
    std::int32_t is_memory;
    std::int32_t force_cpn;
    std::int32_t no_notional;
    std::int32_t is_long_basket;
    std::int32_t _pad0;
    double* f_w_out;
    double* v_w_out;
    double* delta_w_out;
    double* p_stop_out;
    double* surv_out;
    double* h_rate_out;
    double* q_cpn_out;
    double* scalars_out;
};

void tier1_eval_autocall_struct_cpu(const Tier1AutocallArgs* a) noexcept {
    tier1_eval_autocall_bridge_cpu(
        a->d_assets,
        a->m_rem,
        a->fwd_perf,
        a->vol_perf,
        a->corr,
        a->tau_obs,
        a->df_pay,
        a->ko_barriers,
        a->is_callable,
        a->cpn_barriers,
        a->is_guaranteed_cpn,
        a->coupon_rates,
        a->cbt_out,
        a->cbd_down,
        a->inv_lex_perm,
        a->strike,
        a->lower_strike,
        a->ki_barrier,
        a->has_ki,
        a->i_ki_past,
        a->ko_spread,
        a->cpn_spread,
        a->ki_spread,
        a->call_code,
        a->cpn_code,
        a->opt_code,
        a->allow_exp,
        a->c_mem_0,
        a->is_memory,
        a->autocall_coupon,
        a->force_cpn,
        a->no_notional,
        a->is_long_basket,
        a->option_gearing,
        a->part_call_strike,
        a->part_call_rate,
        a->one_star_strike,
        a->one_star_coupon,
        a->pv_in_transit,
        a->f_w_out,
        a->v_w_out,
        a->delta_w_out,
        a->p_stop_out,
        a->surv_out,
        a->h_rate_out,
        a->q_cpn_out,
        a->scalars_out
    );
}

}  // extern "C"

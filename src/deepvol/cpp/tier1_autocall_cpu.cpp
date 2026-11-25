#include <immintrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

inline constexpr std::uint32_t kTier1AbiVersion = 20261124u;
inline constexpr std::uint32_t kTier1AbiVersionV2 = 20261125u;

struct Tier1AbiInfo {
    std::uint32_t abi_version;
    std::uint32_t autocall_args_size;
    std::uint32_t autocall_args_align;
    std::uint32_t crn_args_size;
    std::uint32_t crn_args_align;
};

static_assert(alignof(Tier1AbiInfo) == 4, "Tier1AbiInfo must be 4-byte aligned");
static_assert(sizeof(Tier1AbiInfo) == 20, "Exact 20B Tier1AbiInfo ABI");

struct CrnLooL2oBatchArgs {
    std::int64_t n_bumps;
    std::int64_t d_assets;
    std::int64_t m_rem;
    std::int64_t n_paths_total;
    const double* perf_base;
    const double* bump_perf_a;
    const double* bump_perf_b;
    const std::int32_t* bump_asset_a;
    const std::int32_t* bump_asset_b;
    const double* df_pay_mat;
    const double* w_fund_mat;
    const double* pv_in_transit;
    double* pv_out;
    const double* ko_barriers;
    const double* is_callable;
    const double* cpn_barriers;
    const double* is_guaranteed_cpn;
    const double* coupon_rates;
    double ko_spread;
    double cpn_spread;
    double ki_spread;
    double one_star_strike;
    double one_star_eps;
    double strike;
    double lower_strike;
    double ki_barrier;
    double option_gearing;
    double autocall_coupon;
    double c_mem_0;
    double i_ki_past;
    double part_call_strike;
    double part_call_rate;
    double one_star_coupon;
    std::int32_t call_code;
    std::int32_t cpn_code;
    std::int32_t opt_code;
    std::int32_t has_ki;
    std::int32_t is_memory;
    std::int32_t is_long_basket;
    std::int32_t no_notional;
    std::int32_t force_cpn_ac;
    std::int32_t cpn_dir_down;
    std::int32_t cpn_type_out;
    std::int32_t _reserved[4];
};

static_assert(alignof(CrnLooL2oBatchArgs) == 8, "Expected 8-byte natural alignment");
static_assert(sizeof(CrnLooL2oBatchArgs) == 320, "Exact 320B CRN batch ABI");

struct PlatenSdeAadArgs {
    std::int64_t d_assets;
    std::int64_t m_rem;
    std::int64_t n_sub;
    std::int64_t eff_r;
    std::int64_t n_rep;
    std::int64_t lv_nt;
    std::int64_t lv_ny;
    const double* lv_tables;
    const double* z_indep;
    const double* chol_base;
    const double* tau_obs;
    const double* r_native_obs;
    const double* r_tot_obs;
    const double* lv_t_grid;
    const double* lv_y_min;
    const double* lv_y_max;
    const double* bar_lx_post;
    const double* lx_norm_in;
    const double* q_shift_sub;
    double* lx_period_out;
    double* norm_x_out;
    double* bar_lv_out;
    double* bar_chol_out;
    double* bar_q_shift_out;
    std::int32_t run_forward;
    std::int32_t run_backward;
    std::int32_t compute_lv_aad;
    std::int32_t compute_chol_aad;
    std::int32_t compute_q_aad;
    std::int32_t has_q_shift_sub;
    double* out_lx_pre;
    double* out_j_pre;
    double* out_sig_pre;
    std::int32_t vib_period_idx;
    std::int32_t vib_sub_idx;
    std::int32_t save_vib_pre;
    std::int32_t _pad_vib;
    double vib_j_pre_max;
    double vib_dsig_dy_max;
};

static_assert(alignof(PlatenSdeAadArgs) == 8, "PlatenSdeAadArgs must be 8-byte aligned");
static_assert(sizeof(PlatenSdeAadArgs) == 272, "Exact 272B PlatenSdeAadArgs ABI");
static_assert(offsetof(PlatenSdeAadArgs, lv_tables) == 56, "Pointer block at 56B");
static_assert(offsetof(PlatenSdeAadArgs, run_forward) == 192, "Flag block at 192B");
static_assert(offsetof(PlatenSdeAadArgs, out_lx_pre) == 216, "Vibrato tail block at 216B");
static_assert(offsetof(PlatenSdeAadArgs, vib_j_pre_max) == 256, "Vibrato clamp block at 256B");

struct SobolGenArgs {
    std::int64_t d_assets;
    std::int64_t m_rem;
    std::int64_t n_sub;
    std::int64_t eff_r;
    std::int64_t n_rep;
    std::int64_t sobol_dim;
    std::int64_t n_bits;
    const std::uint32_t* dir_numbers;
    const std::uint32_t* digital_shifts;
    const double* bb_ortho_mat;
    const double* chol_mat;
    double* z_indep_out;
    double* z_corr_out;
    std::int32_t use_gray_code;
    std::int32_t apply_bb_time;
    std::int32_t apply_corr;
    std::int32_t antithetic;
    std::int32_t _reserved[2];
};

static_assert(alignof(SobolGenArgs) == 8, "SobolGenArgs must be 8-byte aligned");
static_assert(sizeof(SobolGenArgs) == 128, "Exact 128B SobolGenArgs ABI");
static_assert(offsetof(SobolGenArgs, dir_numbers) == 56, "Pointer block at 56B");
static_assert(offsetof(SobolGenArgs, use_gray_code) == 104, "Flag block at 104B");

struct VibratoGammaArgs {
    std::int64_t d_assets;
    std::int64_t m_rem;
    std::int64_t n_paths_total;
    std::int64_t vib_step_start;
    std::int64_t vib_n_steps;
    const double* perf_mat;
    const double* lx_pre;
    const double* j_pre;
    const double* sig_pre;
    const double* sv_vib;
    const double* chol_mat;
    const double* l_inv_t;
    const double* c_inv;
    const double* fwd_term;
    const double* cdl_m;
    const double* cdl2_last;
    const double* ko_barriers;
    const double* is_callable;
    const double* cpn_barriers;
    const double* is_guaranteed_cpn;
    const double* coupon_rates;
    const double* eps_vec;
    const std::int64_t* inv_lex_perm;
    double* frozen_term_out;
    double* h_vib_out;
    double df_last;
    double h_vib;
    double ko_spread;
    double cpn_spread;
    double ki_spread;
    double one_star_strike;
    double one_star_eps;
    double strike;
    double lower_strike;
    double ki_barrier;
    double option_gearing;
    double autocall_coupon;
    double c_mem_0;
    double i_ki_past;
    double part_call_strike;
    double part_call_rate;
    double one_star_coupon;
    std::int32_t call_code;
    std::int32_t cpn_code;
    std::int32_t opt_code;
    std::int32_t has_ki;
    std::int32_t ki_active;
    std::int32_t has_one_star;
    std::int32_t is_memory;
    std::int32_t is_long_basket;
    std::int32_t no_notional;
    std::int32_t force_cpn_ac;
    std::int32_t cpn_dir_down;
    std::int32_t cpn_type_out;
    std::int32_t has_lower_strike;
    std::int32_t has_part_call;
};

static_assert(alignof(VibratoGammaArgs) == 8, "VibratoGammaArgs must be 8-byte aligned");
static_assert(sizeof(VibratoGammaArgs) == 392, "Exact 392B VibratoGammaArgs ABI");
static_assert(offsetof(VibratoGammaArgs, perf_mat) == 40, "Pointer block at 40B");
static_assert(offsetof(VibratoGammaArgs, df_last) == 200, "Scalar block at 200B");
static_assert(offsetof(VibratoGammaArgs, call_code) == 336, "Flag block at 336B");

struct Tier1AbiInfoV2 {
    std::uint32_t abi_version;
    std::uint32_t autocall_args_size;
    std::uint32_t autocall_args_align;
    std::uint32_t crn_args_size;
    std::uint32_t crn_args_align;
    std::uint32_t platen_sde_aad_args_size;
    std::uint32_t platen_sde_aad_args_align;
    std::uint32_t sobol_gen_args_size;
    std::uint32_t sobol_gen_args_align;
    std::uint32_t vibrato_gamma_args_size;
    std::uint32_t vibrato_gamma_args_align;
    std::uint32_t _pad0;
};

static_assert(alignof(Tier1AbiInfoV2) == 4, "Tier1AbiInfoV2 is 4-byte aligned");
static_assert(sizeof(Tier1AbiInfoV2) == 48, "Exact 48B Tier1AbiInfoV2 ABI");

}  // extern "C"

struct alignas(64) Tier1StackArena {
    static constexpr std::size_t kMaxAssets = 32;
    std::array<double, kMaxAssets> f_col;
    std::array<double, kMaxAssets> v_col;
    std::array<double, kMaxAssets> w_col;
    std::array<double, kMaxAssets> f_next;
    std::array<double, kMaxAssets> v_next;
    std::array<double, kMaxAssets> w_next;
    std::array<double, kMaxAssets> f_cpn;
    std::array<double, kMaxAssets> v_cpn;
    std::array<double, kMaxAssets> w_cpn;
    std::array<double, kMaxAssets> f_opt;
    std::array<double, kMaxAssets> v_opt;
    std::array<double, kMaxAssets> w_opt;
    std::array<double, kMaxAssets * kMaxAssets> corr_sym;
    std::array<std::int32_t, kMaxAssets> ord_idx;
    std::array<std::int32_t, kMaxAssets> scratch_perm;
};

static_assert(alignof(Tier1StackArena) == 64, "64B cache-line alignment required for AVX2");
static_assert(sizeof(Tier1StackArena) == 11520, "Exact stack arena footprint check");

struct alignas(64) CrnThreadPoolArena {
    static constexpr std::size_t kMaxRep = 2048;
    static constexpr std::size_t kMaxLvY = 128;
    static constexpr std::size_t kMaxPeriods = 64;
    std::array<double, kMaxRep> log_x;
    std::array<double, kMaxRep> ex_buf;
    // Pad 4 extra elements so 4-wide gathers at the top knot never read out of bounds.
    std::array<double, kMaxLvY + 4> lv_slice;
    std::array<double, kMaxPeriods> surv_prev_sum;
    std::array<double, kMaxPeriods> p_stop_sum;
    std::array<double, kMaxPeriods> cpn_cf_sum;
};

static_assert(alignof(CrnThreadPoolArena) == 64, "64B alignment required for AVX2 simd loads");
static_assert(sizeof(CrnThreadPoolArena) == 35392, "Exact stack/TLS arena byte footprint");

struct CrnDynamicArenaFallback {
    std::vector<double> log_x;
    std::vector<double> ex_buf;
    std::vector<double> lv_slice;
    std::vector<double> surv_prev_sum;
    std::vector<double> p_stop_sum;
    std::vector<double> cpn_cf_sum;

    void ensure_capacity(std::size_t n_rep, std::size_t lv_ny, std::size_t m_rem) {
        if (n_rep > CrnThreadPoolArena::kMaxRep) {
            if (log_x.size() < n_rep) log_x.resize(n_rep);
            if (ex_buf.size() < n_rep) ex_buf.resize(n_rep);
        }
        if (lv_ny > CrnThreadPoolArena::kMaxLvY && lv_slice.size() < lv_ny + 4) {
            lv_slice.resize(lv_ny + 4);
        }
        if (m_rem > CrnThreadPoolArena::kMaxPeriods) {
            if (surv_prev_sum.size() < m_rem) surv_prev_sum.resize(m_rem);
            if (p_stop_sum.size() < m_rem) p_stop_sum.resize(m_rem);
            if (cpn_cf_sum.size() < m_rem) cpn_cf_sum.resize(m_rem);
        }
    }
};

namespace {

inline std::atomic<int>& crn_active_omp_calls() noexcept {
    static std::atomic<int> g_active_calls{0};
    return g_active_calls;
}

inline int crn_omp_thread_cap(int max_hw_threads) noexcept {
    return std::clamp(std::min(6, max_hw_threads), 1, 6);
}

inline int crn_hw_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Acquire a thread budget only when fewer than 2 parallel regions are already active.
struct OmpCallGuard {
    bool active = false;
    int n_threads = 1;

    explicit OmpCallGuard(bool want_parallel) noexcept {
        if (!want_parallel) {
            return;
        }
        const int cap = crn_omp_thread_cap(crn_hw_max_threads());
        if (cap <= 1) {
            return;
        }
        const int prev = crn_active_omp_calls().fetch_add(1, std::memory_order_acq_rel);
        if (prev < 2) {
            active = true;
            n_threads = cap;
        } else {
            crn_active_omp_calls().fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    ~OmpCallGuard() noexcept {
        if (active) {
            crn_active_omp_calls().fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    OmpCallGuard(const OmpCallGuard&) = delete;
    OmpCallGuard& operator=(const OmpCallGuard&) = delete;
};

inline double centered_onestar_ramp(double w_best, double k_os, double eps_os) noexcept {
    if (eps_os <= 1e-6) {
        return w_best >= k_os ? 1.0 : 0.0;
    }
    const double half = 0.5 * eps_os;
    const double inv_eps = 1.0 / eps_os;
    return std::clamp((w_best - (k_os - half)) * inv_eps, 0.0, 1.0);
}

constexpr double kInvSqrt2 = 0.70710678118654752440084436210485;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kSqrtHalfPi = 1.2533141373155002512078826424055;
constexpr double kSqrtTwoOverPi = 0.79788456080286535587989211986876;
constexpr double kTwoOverSqrtPi = 0.56418958354775628694807945156077;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kTieTau = 0.008;

// 16-point Gauss-Legendre nodes/weights on [-1, 1] and mapped standard normal quantiles.
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
    -2.555646775436514, -1.9155297652282088, -1.4970940120973921, -1.1635767166086022,
    -0.87398989719866271, -0.60981672311372492, -0.36060272864840204, -0.11936335752495635,
     0.11936335752495648,  0.36060272864840204,  0.6098167231137247,  0.87398989719866271,
     1.1635767166086022,   1.4970940120973921,   1.9155297652282088,  2.555646775436514
};

inline double phi_std(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

inline double ndtr_std(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

inline double ndtri_std(double p) noexcept {
    if (p <= 0.0) return -8.5;
    if (p >= 1.0) return 8.5;
    static constexpr double a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static constexpr double b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static constexpr double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static constexpr double d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    constexpr double plow = 0.02425;
    constexpr double phigh = 1.0 - plow;
    double x = 0.0;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= phigh) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double e = ndtr_std(x) - p;
    const double u = e * std::sqrt(kTwoPi) * std::exp(0.5 * x * x);
    x -= u / (1.0 + 0.5 * x * u);
    return std::clamp(x, -8.5, 8.5);
}

#if defined(__AVX2__) && defined(__FMA__)
inline __m256d v_abs_pd(__m256d x) noexcept {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
}

inline __m256d v_exp_pd(__m256d x) noexcept {
    const __m256d max_x = _mm256_set1_pd(709.782712893384);
    const __m256d min_x = _mm256_set1_pd(-745.1332191019412);
    const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256d ln2_hi = _mm256_set1_pd(6.93147180559890330e-01);
    const __m256d ln2_lo = _mm256_set1_pd(5.49792301870837116e-14);

    __m256d xs = _mm256_min_pd(_mm256_max_pd(x, min_x), max_x);
    __m256d kd = _mm256_round_pd(_mm256_mul_pd(xs, inv_ln2),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(kd, ln2_lo, _mm256_fnmadd_pd(kd, ln2_hi, xs));

    const __m256d one = _mm256_set1_pd(1.0);
    __m256d p = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 6227020800.0), r, _mm256_set1_pd(1.0 / 479001600.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 39916800.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 3628800.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 362880.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 40320.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 5040.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 720.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 120.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 24.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 6.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.5));
    p = _mm256_fmadd_pd(p, r, one);
    p = _mm256_fmadd_pd(p, r, one);

    // Split k = k1 + k2 so 1023 + k_half stays in normal exponent range for subnormals.
    const __m128i k32 = _mm256_cvtpd_epi32(kd);
    const __m128i k1_32 = _mm_srai_epi32(k32, 1);
    const __m128i k2_32 = _mm_sub_epi32(k32, k1_32);
    const __m256i bias = _mm256_set1_epi64x(1023);
    const __m256i s1_i = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(k1_32), bias), 52);
    const __m256i s2_i = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(k2_32), bias), 52);
    __m256d res = _mm256_mul_pd(_mm256_mul_pd(p, _mm256_castsi256_pd(s1_i)), _mm256_castsi256_pd(s2_i));

    __m256d ovf = _mm256_cmp_pd(x, max_x, _CMP_GT_OQ);
    __m256d udf = _mm256_cmp_pd(x, min_x, _CMP_LT_OQ);
    __m256d is_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    res = _mm256_blendv_pd(res, _mm256_set1_pd(std::numeric_limits<double>::infinity()), ovf);
    res = _mm256_blendv_pd(res, _mm256_setzero_pd(), udf);
    return _mm256_blendv_pd(res, x, is_nan);
}

inline __m256d v_log_pd(__m256d x) noexcept {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d dbl_min = _mm256_set1_pd(std::numeric_limits<double>::min());
    const __m256d two54 = _mm256_set1_pd(18014398509481984.0);
    const __m256d neg54 = _mm256_set1_pd(-54.0);

    __m256d is_pos = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
    __m256d is_sub = _mm256_and_pd(is_pos, _mm256_cmp_pd(x, dbl_min, _CMP_LT_OQ));
    __m256d x_norm = _mm256_blendv_pd(x, _mm256_mul_pd(x, two54), is_sub);
    __m256d sub_adj = _mm256_blendv_pd(zero, neg54, is_sub);

    const __m256i exp_mask = _mm256_set1_epi64x(0x7ffLL);
    const __m256i two52_i = _mm256_set1_epi64x(0x4330000000000000LL);
    const __m256d two52_plus_bias = _mm256_set1_pd(4503599627370496.0 + 1023.0);
    __m256i xi = _mm256_castpd_si256(x_norm);
    __m256i e_raw = _mm256_and_si256(_mm256_srli_epi64(xi, 52), exp_mask);
    __m256d e_dbl = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(e_raw, two52_i)), two52_plus_bias);

    const __m256i mant_mask = _mm256_set1_epi64x(0x000fffffffffffffLL);
    const __m256i one_bits = _mm256_set1_epi64x(0x3ff0000000000000LL);
    __m256d m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(xi, mant_mask), one_bits));

    const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730951);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d gt_s2 = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), gt_s2);
    e_dbl = _mm256_add_pd(e_dbl, _mm256_blendv_pd(zero, one, gt_s2));
    __m256d k = _mm256_add_pd(e_dbl, sub_adj);

    __m256d f = _mm256_sub_pd(m, one);
    __m256d s = _mm256_div_pd(f, _mm256_add_pd(m, one));
    __m256d s2 = _mm256_mul_pd(s, s);
    __m256d p = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 19.0), s2, _mm256_set1_pd(1.0 / 17.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 15.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 13.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 11.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 9.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 7.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 5.0));
    p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 3.0));

    __m256d two_s = _mm256_add_pd(s, s);
    __m256d log_m = _mm256_fmadd_pd(two_s, _mm256_mul_pd(s2, p), two_s);
    const __m256d ln2_hi = _mm256_set1_pd(6.93147180559890330e-01);
    const __m256d ln2_lo = _mm256_set1_pd(5.49792301870837116e-14);
    __m256d res = _mm256_fmadd_pd(k, ln2_hi, _mm256_fmadd_pd(k, ln2_lo, log_m));

    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d is_inf = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);
    __m256d is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
    __m256d is_bad = _mm256_or_pd(_mm256_cmp_pd(x, zero, _CMP_LT_OQ), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
    res = _mm256_blendv_pd(res, inf, is_inf);
    res = _mm256_blendv_pd(res, _mm256_set1_pd(-std::numeric_limits<double>::infinity()), is_zero);
    return _mm256_blendv_pd(res, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), is_bad);
}

inline void v_cody_erf_erfcx_parts(
    __m256d y, __m256d& erf_small, __m256d& erfcx_mid, __m256d& erfcx_hi
) noexcept {
    {
        __m256d ys = _mm256_min_pd(y, _mm256_set1_pd(0.46875));
        __m256d ysq = _mm256_mul_pd(ys, ys);
        __m256d xnum = _mm256_mul_pd(_mm256_set1_pd(1.85777706184603153e-01), ysq);
        __m256d xden = ysq;
        const double a[4] = {3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03};
        const double b[4] = {2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03};
        for (int i = 0; i < 3; ++i) {
            xnum = _mm256_mul_pd(_mm256_add_pd(xnum, _mm256_set1_pd(a[i])), ysq);
            xden = _mm256_mul_pd(_mm256_add_pd(xden, _mm256_set1_pd(b[i])), ysq);
        }
        erf_small = _mm256_div_pd(
            _mm256_mul_pd(ys, _mm256_add_pd(xnum, _mm256_set1_pd(a[3]))),
            _mm256_add_pd(xden, _mm256_set1_pd(b[3]))
        );
    }
    {
        __m256d ym = _mm256_min_pd(_mm256_max_pd(y, _mm256_set1_pd(0.46875)), _mm256_set1_pd(4.0));
        __m256d xnum = _mm256_mul_pd(_mm256_set1_pd(2.15311535474403846e-08), ym);
        __m256d xden = ym;
        const double c[8] = {
            5.64188496988670089e-01, 8.88314979438837594e00,
            6.61191906371416295e01, 2.98635138197400131e02,
            8.81952221241769090e02, 1.71204761263407058e03,
            2.05107837782607147e03, 1.23033935479799725e03
        };
        const double d[8] = {
            1.57449261107098347e01, 1.17693950891312499e02,
            5.37181101862009858e02, 1.62138957456669019e03,
            3.29079923573345963e03, 4.36261909014324716e03,
            3.43936767414372164e03, 1.23033935480374942e03
        };
        for (int i = 0; i < 7; ++i) {
            xnum = _mm256_mul_pd(_mm256_add_pd(xnum, _mm256_set1_pd(c[i])), ym);
            xden = _mm256_mul_pd(_mm256_add_pd(xden, _mm256_set1_pd(d[i])), ym);
        }
        erfcx_mid = _mm256_div_pd(
            _mm256_add_pd(xnum, _mm256_set1_pd(c[7])),
            _mm256_add_pd(xden, _mm256_set1_pd(d[7]))
        );
    }
    {
        __m256d yh = _mm256_max_pd(y, _mm256_set1_pd(4.0));
        __m256d ysq = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(yh, yh));
        __m256d xnum = _mm256_mul_pd(_mm256_set1_pd(1.63153871373020978e-02), ysq);
        __m256d xden = ysq;
        const double p[5] = {
            3.05326634961232344e-01, 3.60344899949804439e-01,
            1.25781726111229246e-01, 1.60837851487422766e-02,
            6.58749161529837803e-04
        };
        const double q[5] = {
            2.56852019228982242e00, 1.87295284992346047e00,
            5.27905102951428412e-01, 6.05183413124413191e-02,
            2.33520497626869185e-03
        };
        for (int i = 0; i < 4; ++i) {
            xnum = _mm256_mul_pd(_mm256_add_pd(xnum, _mm256_set1_pd(p[i])), ysq);
            xden = _mm256_mul_pd(_mm256_add_pd(xden, _mm256_set1_pd(q[i])), ysq);
        }
        __m256d frac = _mm256_div_pd(
            _mm256_mul_pd(ysq, _mm256_add_pd(xnum, _mm256_set1_pd(p[4]))),
            _mm256_add_pd(xden, _mm256_set1_pd(q[4]))
        );
        erfcx_hi = _mm256_div_pd(_mm256_sub_pd(_mm256_set1_pd(kTwoOverSqrtPi), frac), yh);
    }
}

[[maybe_unused]] inline __m256d v_erfcx_pd(__m256d x) noexcept {
    __m256d y = v_abs_pd(x);
    __m256d erf_s{}, e_mid{}, e_hi{};
    v_cody_erf_erfcx_parts(y, erf_s, e_mid, e_hi);

    __m256d is_neg = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);
    __m256d is_small = _mm256_cmp_pd(y, _mm256_set1_pd(0.46875), _CMP_LE_OQ);
    __m256d is_mid = _mm256_cmp_pd(y, _mm256_set1_pd(4.0), _CMP_LE_OQ);
    __m256d res_pos = _mm256_blendv_pd(e_hi, e_mid, is_mid);
    if (_mm256_movemask_pd(is_small) != 0) {
        __m256d ys = _mm256_min_pd(y, _mm256_set1_pd(0.46875));
        __m256d erf_s_signed = _mm256_blendv_pd(erf_s, _mm256_sub_pd(_mm256_setzero_pd(), erf_s), is_neg);
        __m256d e_small = _mm256_mul_pd(
            v_exp_pd(_mm256_mul_pd(ys, ys)),
            _mm256_sub_pd(_mm256_set1_pd(1.0), erf_s_signed)
        );
        res_pos = _mm256_blendv_pd(res_pos, e_small, is_small);
        __m256d neg_gt = _mm256_andnot_pd(is_small, is_neg);
        if (_mm256_movemask_pd(neg_gt) != 0) {
            __m256d exp_y2 = v_exp_pd(_mm256_mul_pd(y, y));
            __m256d res_neg = _mm256_fmsub_pd(_mm256_set1_pd(2.0), exp_y2, res_pos);
            res_pos = _mm256_blendv_pd(res_pos, res_neg, neg_gt);
        }
        return _mm256_blendv_pd(res_pos, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
    }
    if (_mm256_movemask_pd(is_neg) != 0) {
        __m256d exp_y2 = v_exp_pd(_mm256_mul_pd(y, y));
        __m256d res_neg = _mm256_fmsub_pd(_mm256_set1_pd(2.0), exp_y2, res_pos);
        res_pos = _mm256_blendv_pd(res_pos, res_neg, is_neg);
    }
    return _mm256_blendv_pd(res_pos, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
}

inline __m256d v_ndtr_pd(__m256d z) noexcept {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d az = v_abs_pd(z);
    __m256d y = _mm256_mul_pd(az, _mm256_set1_pd(kInvSqrt2));
    __m256d erf_s{}, e_mid{}, e_hi{};
    v_cody_erf_erfcx_parts(y, erf_s, e_mid, e_hi);

    __m256d is_small = _mm256_cmp_pd(y, _mm256_set1_pd(0.46875), _CMP_LE_OQ);
    __m256d is_mid = _mm256_cmp_pd(y, _mm256_set1_pd(4.0), _CMP_LE_OQ);
    __m256d r_big = _mm256_blendv_pd(e_hi, e_mid, is_mid);
    __m256d tail_big = _mm256_mul_pd(
        _mm256_mul_pd(half, v_exp_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(az, az)))),
        r_big
    );
    __m256d tail_small = _mm256_mul_pd(half, _mm256_sub_pd(one, erf_s));
    __m256d tail = _mm256_blendv_pd(tail_big, tail_small, is_small);
    __m256d pos_z = _mm256_cmp_pd(z, _mm256_setzero_pd(), _CMP_GE_OQ);
    __m256d res = _mm256_blendv_pd(tail, _mm256_sub_pd(one, tail), pos_z);
    return _mm256_blendv_pd(res, z, _mm256_cmp_pd(z, z, _CMP_UNORD_Q));
}
#endif

// Genz (2004) bivariate normal CDF P(X <= x, Y <= y) with corr rho.
inline double bvn_cdf_pos(double x, double y, double r) noexcept {
    if (r >= 1.0 - 1e-12) return ndtr_std(std::min(x, y));
    if (r < 0.925) {
        const double as = std::asin(r);
        const double hs = 0.5 * (x * x + y * y);
        const double hk = x * y;
        double acc = 0.0;
        for (std::size_t q = 0; q < 16; ++q) {
            const double sn = std::sin(0.5 * as * (kGL16X[q] + 1.0));
            acc += 0.5 * kGL16W[q] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
        }
        return acc * (as / kTwoPi) + ndtr_std(x) * ndtr_std(y);
    }
    const double h = -x;
    const double k = -y;
    const double hk = h * k;
    const double a = std::sqrt((1.0 - r) * (1.0 + r));
    const double b = std::abs(h - k);
    const double bs = b * b;
    const double c = (4.0 - hk) / 8.0;
    const double d = (12.0 - hk) / 16.0;
    const double asr = -0.5 * (bs / (a * a) + hk);
    double bvn = 0.0;
    if (asr > -100.0) {
        bvn = a * std::exp(asr) * (1.0 - c * (bs - a * a) * (1.0 - d * bs / 5.0) / 3.0 + c * d * a * a * a * a / 5.0);
    }
    if (-hk < 100.0) {
        bvn -= std::exp(-0.5 * hk) * std::sqrt(kTwoPi) * ndtr_std(-b / a) * b *
               (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
    }
    const double a_half = 0.5 * a;
    double acc = 0.0;
    for (std::size_t q = 0; q < 16; ++q) {
        const double xs = a_half * (kGL16X[q] + 1.0);
        const double xs2 = xs * xs;
        const double rs = std::sqrt(std::max(1.0 - xs2, 1e-30));
        const double asr_q = -0.5 * (bs / std::max(xs2, 1e-30) + hk);
        if (asr_q > -100.0) {
            acc += kGL16W[q] * std::exp(asr_q) *
                   (std::exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs2 * (1.0 + d * xs2)));
        }
    }
    bvn -= a_half * acc;
    return ndtr_std(-std::max(h, k)) - bvn / kTwoPi;
}

inline double bvn_cdf_genz(double x, double y, double rho) noexcept {
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
    if (std::abs(rho) < 1e-14) return ndtr_std(x) * ndtr_std(y);
    const double res = (rho >= 0.0) ? bvn_cdf_pos(x, y, rho) : (ndtr_std(x) - bvn_cdf_pos(x, -y, -rho));
    return std::clamp(res, std::max(ndtr_std(x) + ndtr_std(y) - 1.0, 0.0), std::min(ndtr_std(x), ndtr_std(y)));
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

// Continued fraction tail avoids catastrophic subtraction (alpha + lambda(alpha)) for alpha <= -4.
inline void truncated_upper_std_moments(double alpha, double& mu_out, double& var_out) noexcept {
    if (alpha <= -4.0) {
        const double x = -alpha;
        double v = 0.0;
        for (int k = 80; k >= 3; --k) {
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

constexpr std::array<double, 8> kGL8X01 = {
    0.019855071751231912, 0.10166676129318664, 0.2372337950418355, 0.40828267875217511,
    0.59171732124782483,  0.7627662049581645,  0.89833323870681336, 0.98014492824876809
};
constexpr std::array<double, 8> kGL8W01 = {
    0.050614268145188532, 0.11119051722668721, 0.15685332293894344, 0.18134189168918083,
    0.18134189168918083,  0.15685332293894344, 0.11119051722668721, 0.050614268145188532
};
constexpr std::array<double, 6> kLag6X = {
    0.22284660417926069, 1.1889321016726226, 2.9927363260593141,
    5.7751435691045101,  9.8374674183825892, 15.982873980601703
};
constexpr std::array<double, 6> kLag6W = {
    0.45896467394996476, 0.41700083077212002, 0.11337338207404488,
    0.010399197453149087, 0.00026101720281493265, 8.9854790642962134e-07
};

inline void lognormal_moments_interval(
    double f, double v, double a, double b, int max_order, double* out
) noexcept;
inline void normal_moments_interval(
    double mu, double sig, double a, double b, int max_order, double* out
) noexcept;

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
    const double eff_spread = dir_down ? -spread : spread;
    // TwinWin maps W = max(W_min, 2 - W_min) >= 1.0 onto survival band [max(2 - b_hi, 0), b_hi].
    if (index_code == 14) {
        const double b_lo = std::min(barrier, barrier + eff_spread);
        const double b_hi = std::max(barrier, barrier + eff_spread);
        const double db = b_hi - b_lo;
        double p_ge = 1.0;
        if (b_hi > 1.0) {
            const double f_c = f_or_mu > 1e-16 ? f_or_mu : 1e-16;
            const double v_c = v_or_sig > 1e-12 ? v_or_sig : 1e-12;
            const double a_dn = std::max(2.0 - b_hi, 0.0);
            const double b_dn = std::clamp(2.0 - b_lo, 0.0, 1.0);
            const double a_up = std::max(b_lo, 1.0);
            double p_surv = 0.0;
            if (db <= 1e-6) {
                double m0 = 0.0;
                lognormal_moments_interval(f_c, v_c, a_dn, b_hi, 0, &m0);
                p_surv = m0;
            } else {
                const double inv_db = 1.0 / db;
                double m[2] = {0.0, 0.0};
                if (b_dn > a_dn + 1e-14) {
                    lognormal_moments_interval(f_c, v_c, a_dn, b_dn, 1, m);
                    p_surv += (-(2.0 - b_hi) * m[0] + m[1]) * inv_db;
                }
                if (a_up > b_dn + 1e-14) {
                    lognormal_moments_interval(f_c, v_c, b_dn, a_up, 0, m);
                    p_surv += m[0];
                }
                if (b_hi > a_up + 1e-14) {
                    lognormal_moments_interval(f_c, v_c, a_up, b_hi, 1, m);
                    p_surv += (b_hi * m[0] - m[1]) * inv_db;
                }
            }
            p_ge = std::clamp(1.0 - p_surv, 0.0, 1.0);
        }
        const double p_ev = dir_down ? (1.0 - p_ge) : p_ge;
        return type_out ? (1.0 - p_ev) : p_ev;
    }
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
    } else if (u_hi > -20.0) {
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
    } else {
        // Shifted-Y = -(U + x) quadrature eliminates deep left-tail cancellation in erfcx differences.
        const double x = -u_hi;
        const double y_lo = has_lo ? (u_hi - u_lo) : 1e100;
        const double y_end = std::min(y_lo, 45.0 / x);
        const double inv_ratio = b_hi / db;
        double m0 = 0.0;
        double n1 = 0.0;
        double n2 = 0.0;
        for (std::size_t q = 0; q < 8; ++q) {
            const double yq = y_end * kGL8X01[q];
            const double wt = y_end * kGL8W01[q] * std::exp(-yq * (x + 0.5 * yq)) * inv_ratio * (-std::expm1(-l_c * yq));
            m0 += wt;
            n1 += wt * yq;
            n2 += wt * yq * yq;
        }
        if (has_lo && y_lo * (x + 0.5 * y_lo) < 55.0) {
            const double alpha = x + y_lo;
            const double inv_alpha = 1.0 / alpha;
            const double pref = std::exp(-y_lo * (x + 0.5 * y_lo)) * inv_alpha;
            for (std::size_t q = 0; q < 6; ++q) {
                const double sq = kLag6X[q] * inv_alpha;
                const double yq = y_lo + sq;
                const double wt = pref * kLag6W[q] * std::exp(-0.5 * sq * sq);
                m0 += wt;
                n1 += wt * yq;
                n2 += wt * yq * yq;
            }
        }
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double ey = n1 / m0;
        mu_u_out = -x - ey;
        var_u_out = std::max(n2 / m0 - ey * ey, 1e-12);
    }
}

inline void ramp_survival_post_moments_twinwin_lognormal(
    double f_cond,
    double lam,
    double b_ko,
    double ko_spread,
    double& mu_u_out,
    double& var_u_out
) noexcept {
    const double f_c = f_cond > 1e-16 ? f_cond : 1e-16;
    const double l_c = lam > 1e-12 ? lam : 1e-12;
    const double b_lo = std::min(b_ko, b_ko + ko_spread);
    const double b_hi = std::max(b_ko, b_ko + ko_spread);
    if (b_hi <= 1.0) {
        mu_u_out = 0.0;
        var_u_out = 1e-12;
        return;
    }
    const double m_u = std::log(f_c) - 0.5 * l_c * l_c;
    const double db = b_hi - b_lo;
    double m0 = 0.0, m1 = 0.0, m2 = 0.0;

    auto add_seg = [&](double seg_a, double seg_b, double c0, double c1) noexcept {
        if (seg_b <= seg_a + 1e-14) return;
        const double u_b = (std::log(seg_b) - m_u) / l_c;
        const double u_a = (seg_a > 0.0) ? ((std::log(seg_a) - m_u) / l_c) : -1e60;
        double jc[3] = {0.0, 0.0, 0.0};
        normal_moments_interval(0.0, 1.0, u_a, u_b, 2, jc);
        m0 += c0 * jc[0];
        m1 += c0 * jc[1];
        m2 += c0 * jc[2];
        if (c1 != 0.0) {
            double jl[3] = {0.0, 0.0, 0.0};
            normal_moments_interval(l_c, 1.0, u_a, u_b, 2, jl);
            const double w_c1 = c1 * f_c;
            m0 += w_c1 * jl[0];
            m1 += w_c1 * jl[1];
            m2 += w_c1 * jl[2];
        }
    };

    const double a_dn = std::max(2.0 - b_hi, 0.0);
    if (db <= 1e-6) {
        add_seg(a_dn, b_hi, 1.0, 0.0);
    } else {
        const double inv_db = 1.0 / db;
        const double b_dn = std::clamp(2.0 - b_lo, 0.0, 1.0);
        const double a_up = std::max(b_lo, 1.0);
        add_seg(a_dn, b_dn, -(2.0 - b_hi) * inv_db, inv_db);
        add_seg(b_dn, a_up, 1.0, 0.0);
        add_seg(a_up, b_hi, b_hi * inv_db, -inv_db);
    }
    if (m0 <= 1e-300) {
        const double u_top = (std::log(b_hi) - m_u) / l_c;
        if (m_u > std::log(b_hi)) {
            truncated_upper_std_moments(u_top, mu_u_out, var_u_out);
        } else if (a_dn > 0.0) {
            double mu_neg = 0.0;
            truncated_upper_std_moments(-(std::log(a_dn) - m_u) / l_c, mu_neg, var_u_out);
            mu_u_out = -mu_neg;
        } else {
            mu_u_out = 0.0;
            var_u_out = 1.0;
        }
        return;
    }
    mu_u_out = m1 / m0;
    var_u_out = std::max(m2 / m0 - mu_u_out * mu_u_out, 1e-12);
}

inline double shifted_y_j0(double t) noexcept {
    const double inv_t = 1.0 / t;
    const double inv_t2 = inv_t * inv_t;
    static constexpr double c[7] = {
        1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0
    };
    double acc = c[6];
    for (int n = 5; n >= 0; --n) {
        acc = acc * inv_t2 + c[n];
    }
    return acc * inv_t;
}

inline double shifted_y_jk(int k, double t) noexcept {
    const double inv_t = 1.0 / t;
    const double inv_t2 = inv_t * inv_t;
    static constexpr double c1[7] = {
        1.0, -3.0, 15.0, -105.0, 945.0, -10395.0, 135135.0
    };
    static constexpr double c2[7] = {
        2.0, -12.0, 90.0, -840.0, 9450.0, -124740.0, 1891890.0
    };
    static constexpr double c3[7] = {
        6.0, -60.0, 630.0, -7560.0, 103950.0, -1621620.0, 28378350.0
    };
    const double* c = (k == 1) ? c1 : ((k == 2) ? c2 : c3);
    double acc = c[6];
    for (int n = 5; n >= 0; --n) {
        acc = acc * inv_t2 + c[n];
    }
    double p = inv_t2;
    for (int i = 1; i < k; ++i) p *= inv_t;
    return acc * p;
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
    } else if (u_hi > -20.0) {
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
    } else {
        // Cancellation-free Shifted-Y = -(U + x) asymptotic integrals in deep left tail.
        const double x = -u_hi;
        const double eta = du * (x + 0.5 * du);
        double m0 = 0.0;
        double n1 = 0.0;
        double n2 = 0.0;
        if (eta >= 1e-3) {
            const double t0 = x;
            const double t1 = x + du;
            const double e_eta = std::exp(-eta);
            const double j1_0 = shifted_y_jk(1, t0);
            const double j2_0 = shifted_y_jk(2, t0);
            const double j3_0 = shifted_y_jk(3, t0);
            const double j1_1 = shifted_y_jk(1, t1);
            const double j2_1 = shifted_y_jk(2, t1);
            const double j3_1 = shifted_y_jk(3, t1);
            m0 = (j1_0 - e_eta * j1_1) / du;
            n1 = (j2_0 - e_eta * (j2_1 + du * j1_1)) / du;
            n2 = (j3_0 - e_eta * (j3_1 + 2.0 * du * j2_1 + du * du * j1_1)) / du;
        } else {
            // 3-point Gauss-Legendre avoids subtraction between nearly equal J_k(x) and e^{-eta} J_k(x+du).
            static constexpr double gl_x[3] = {
                0.11270166537925831148,
                0.50000000000000000000,
                0.88729833462074168852
            };
            static constexpr double gl_w[3] = {
                0.27777777777777777778,
                0.44444444444444444444,
                0.27777777777777777778
            };
            for (int q = 0; q < 3; ++q) {
                const double s_q = du * gl_x[q];
                const double t_q = x + s_q;
                const double we = gl_w[q] * std::exp(-(x * s_q + 0.5 * s_q * s_q));
                const double j0 = shifted_y_j0(t_q);
                const double j1 = shifted_y_jk(1, t_q);
                const double j2 = shifted_y_jk(2, t_q);
                m0 += we * j0;
                n1 += we * (j1 + s_q * j0);
                n2 += we * (j2 + 2.0 * s_q * j1 + s_q * s_q * j0);
            }
        }
        if (m0 <= 1e-300) {
            truncated_upper_std_moments(u_hi, mu_u_out, var_u_out);
            return;
        }
        const double ey = n1 / m0;
        mu_u_out = -x - ey;
        var_u_out = std::max(n2 / m0 - ey * ey, 1e-12);
    }
}

void clark_one_order_impl(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    const int* order,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double* w_out,
    double* rho_all_out,
    double* f_ord,
    double* v_ord,
    double* w_ord,
    double* rho_cur,
    double* rho_all
) noexcept {
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

[[gnu::noinline]] void clark_one_order_dyn(
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
    const auto du = static_cast<std::size_t>(d);
    std::vector<double> f_heap(du), v_heap(du), w_heap(du, 0.0), rc_heap(du), ra_heap(rho_all_out != nullptr ? du : 0);
    clark_one_order_impl(
        d, f_in, v_in, corr, order, is_best_of, f_out, v_out, w_heap.data(), rho_all_out,
        f_heap.data(), v_heap.data(), w_heap.data(), rc_heap.data(),
        rho_all_out != nullptr ? ra_heap.data() : nullptr
    );
    if (w_out != nullptr) {
        for (std::size_t i = 0; i < du; ++i) w_out[i] = w_heap[i];
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
    if (d <= kStackD) {
        std::array<double, kStackD> f_stk{}, v_stk{}, w_stk{}, rc_stk{}, ra_stk{};
        clark_one_order_impl(
            d, f_in, v_in, corr, order, is_best_of, f_out, v_out, w_out, rho_all_out,
            f_stk.data(), v_stk.data(), w_stk.data(), rc_stk.data(),
            rho_all_out != nullptr ? ra_stk.data() : nullptr
        );
        return;
    }
    clark_one_order_dyn(d, f_in, v_in, corr, order, is_best_of, f_out, v_out, w_out, rho_all_out);
}

inline void clark_pair_merge(
    double f1_in,
    double v1_in,
    double f2_in,
    double v2_in,
    double rho_12,
    int n_rem,
    const double* v_rem,
    const double* rho_1_rem,
    const double* rho_2_rem,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double& w1_out,
    double& w2_out,
    double* rho_m_rem_out
) noexcept {
    const double f1 = f1_in > 1e-16 ? f1_in : 1e-16;
    const double v1 = v1_in > 1e-12 ? v1_in : 1e-12;
    const double f2 = f2_in > 1e-16 ? f2_in : 1e-16;
    const double v2 = v2_in > 1e-12 ? v2_in : 1e-12;
    const double rho_c = std::clamp(rho_12, -0.999999, 0.999999);
    const double var_m = std::max(v1 * v1 + v2 * v2 - 2.0 * rho_c * v1 * v2, 1e-16);
    const double sig_m = std::sqrt(var_m);
    const double log_f_ratio = std::log(f1 / f2);
    const double d1 = (log_f_ratio + 0.5 * var_m) / sig_m;
    const double d2 = d1 - sig_m;
    const double w1 = is_best_of ? ndtr_std(d1) : ndtr_std(-d1);
    const double w2 = is_best_of ? ndtr_std(-d2) : ndtr_std(d2);
    const double e_m1 = std::max(f1 * w1 + f2 * w2, 1e-16);

    const double f1_sq = (f1 * f1) * std::exp(v1 * v1);
    const double f2_sq = (f2 * f2) * std::exp(v2 * v2);
    const double sig_m2 = 2.0 * sig_m;
    const double var_m2 = sig_m2 * sig_m2;
    const double d1_sq = (std::log(std::max(f1_sq, 1e-30) / std::max(f2_sq, 1e-30)) + 0.5 * var_m2) / sig_m2;
    const double d2_sq = d1_sq - sig_m2;
    const double e_m2 = is_best_of
        ? (f1_sq * ndtr_std(d1_sq) + f2_sq * ndtr_std(-d2_sq))
        : (f1_sq * ndtr_std(-d1_sq) + f2_sq * ndtr_std(d2_sq));
    const double ratio_m2 = e_m2 / std::max(e_m1 * e_m1, 1e-30);
    const double v_new = std::sqrt(ratio_m2 > 1.0 + 1e-16 ? std::log(ratio_m2) : 1e-16);

    f_out = e_m1;
    v_out = v_new;
    w1_out = w1;
    w2_out = w2;

    if (rho_m_rem_out != nullptr && n_rem > 0) {
        for (int k = 0; k < n_rem; ++k) {
            const double vk = v_rem[k] > 1e-12 ? v_rem[k] : 1e-12;
            const double exp_arg_1 = rho_1_rem[k] * v1 * vk;
            const double exp_arg_2 = rho_2_rem[k] * v2 * vk;
            const double f1_k = f1 * std::exp(exp_arg_1);
            const double f2_k = f2 * std::exp(exp_arg_2);
            const double d1_k = (log_f_ratio + (exp_arg_1 - exp_arg_2) + 0.5 * var_m) / sig_m;
            const double d2_k = d1_k - sig_m;
            const double e_mp = is_best_of
                ? (f1_k * ndtr_std(d1_k) + f2_k * ndtr_std(-d2_k))
                : (f1_k * ndtr_std(-d1_k) + f2_k * ndtr_std(d2_k));
            const double cov_log = std::log(std::max(e_mp / e_m1, 1e-16));
            rho_m_rem_out[k] = std::clamp(cov_log / std::max(v_new * vk, 1e-16), -0.999999, 0.999999);
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
template <int NRem>
inline void clark_pair_merge_avx2(
    __m256d f1_in,
    __m256d v1_in,
    __m256d f2_in,
    __m256d v2_in,
    __m256d rho_12,
    const __m256d* v_rem,
    const __m256d* rho_1_rem,
    const __m256d* rho_2_rem,
    bool is_best_of,
    __m256d& f_out,
    __m256d& v_out,
    __m256d& w1_out,
    __m256d& w2_out,
    __m256d* rho_m_rem_out
) noexcept {
    const __m256d eps_f = _mm256_set1_pd(1e-16);
    const __m256d eps_v = _mm256_set1_pd(1e-12);
    const __m256d eps_sq = _mm256_set1_pd(1e-30);
    const __m256d rho_max = _mm256_set1_pd(0.999999);
    const __m256d rho_min = _mm256_set1_pd(-0.999999);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d sgn = _mm256_set1_pd(is_best_of ? 1.0 : -1.0);
    const __m256d neg_sgn = _mm256_set1_pd(is_best_of ? -1.0 : 1.0);

    __m256d f1 = _mm256_max_pd(f1_in, eps_f);
    __m256d v1 = _mm256_max_pd(v1_in, eps_v);
    __m256d f2 = _mm256_max_pd(f2_in, eps_f);
    __m256d v2 = _mm256_max_pd(v2_in, eps_v);
    __m256d rc = _mm256_min_pd(_mm256_max_pd(rho_12, rho_min), rho_max);

    __m256d var_m = _mm256_max_pd(
        _mm256_fnmadd_pd(two, _mm256_mul_pd(_mm256_mul_pd(rc, v1), v2), _mm256_fmadd_pd(v1, v1, _mm256_mul_pd(v2, v2))),
        eps_f
    );
    __m256d sig_m = _mm256_sqrt_pd(var_m);
    __m256d inv_sig_m = _mm256_div_pd(_mm256_set1_pd(1.0), sig_m);
    __m256d log_fr = v_log_pd(_mm256_div_pd(f1, f2));
    __m256d d1 = _mm256_mul_pd(_mm256_fmadd_pd(half, var_m, log_fr), inv_sig_m);
    __m256d d2 = _mm256_sub_pd(d1, sig_m);

    __m256d w1 = v_ndtr_pd(_mm256_mul_pd(sgn, d1));
    __m256d w2 = v_ndtr_pd(_mm256_mul_pd(neg_sgn, d2));
    __m256d e_m1 = _mm256_max_pd(_mm256_fmadd_pd(f1, w1, _mm256_mul_pd(f2, w2)), eps_f);

    __m256d f1_sq = _mm256_max_pd(_mm256_mul_pd(_mm256_mul_pd(f1, f1), v_exp_pd(_mm256_mul_pd(v1, v1))), eps_sq);
    __m256d f2_sq = _mm256_max_pd(_mm256_mul_pd(_mm256_mul_pd(f2, f2), v_exp_pd(_mm256_mul_pd(v2, v2))), eps_sq);
    __m256d sig_m2 = _mm256_mul_pd(two, sig_m);
    __m256d var_m2 = _mm256_mul_pd(sig_m2, sig_m2);
    __m256d d1_sq = _mm256_div_pd(_mm256_fmadd_pd(half, var_m2, v_log_pd(_mm256_div_pd(f1_sq, f2_sq))), sig_m2);
    __m256d d2_sq = _mm256_sub_pd(d1_sq, sig_m2);
    __m256d w1_sq = v_ndtr_pd(_mm256_mul_pd(sgn, d1_sq));
    __m256d w2_sq = v_ndtr_pd(_mm256_mul_pd(neg_sgn, d2_sq));
    __m256d e_m2 = _mm256_fmadd_pd(f1_sq, w1_sq, _mm256_mul_pd(f2_sq, w2_sq));

    __m256d ratio_m2 = _mm256_div_pd(e_m2, _mm256_max_pd(_mm256_mul_pd(e_m1, e_m1), eps_sq));
    const __m256d one_plus_eps = _mm256_set1_pd(1.0 + 1e-16);
    __m256d has_var = _mm256_cmp_pd(ratio_m2, one_plus_eps, _CMP_GT_OQ);
    __m256d log_var = v_log_pd(_mm256_max_pd(ratio_m2, one_plus_eps));
    __m256d v_new = _mm256_sqrt_pd(_mm256_blendv_pd(eps_f, log_var, has_var));

    f_out = e_m1;
    v_out = v_new;
    w1_out = w1;
    w2_out = w2;

    if constexpr (NRem > 0) {
        for (int k = 0; k < NRem; ++k) {
            __m256d vk = _mm256_max_pd(v_rem[k], eps_v);
            __m256d ea1 = _mm256_mul_pd(_mm256_mul_pd(rho_1_rem[k], v1), vk);
            __m256d ea2 = _mm256_mul_pd(_mm256_mul_pd(rho_2_rem[k], v2), vk);
            __m256d f1_k = _mm256_mul_pd(f1, v_exp_pd(ea1));
            __m256d f2_k = _mm256_mul_pd(f2, v_exp_pd(ea2));
            __m256d d1_k = _mm256_mul_pd(
                _mm256_add_pd(_mm256_add_pd(log_fr, _mm256_sub_pd(ea1, ea2)), _mm256_mul_pd(half, var_m)),
                inv_sig_m
            );
            __m256d d2_k = _mm256_sub_pd(d1_k, sig_m);
            __m256d e_mp = _mm256_fmadd_pd(f1_k, v_ndtr_pd(_mm256_mul_pd(sgn, d1_k)),
                                           _mm256_mul_pd(f2_k, v_ndtr_pd(_mm256_mul_pd(neg_sgn, d2_k))));
            __m256d cov_log = v_log_pd(_mm256_max_pd(_mm256_div_pd(e_mp, e_m1), eps_f));
            __m256d denom_cov = _mm256_max_pd(_mm256_mul_pd(v_new, vk), eps_f);
            rho_m_rem_out[k] = _mm256_min_pd(_mm256_max_pd(_mm256_div_pd(cov_log, denom_cov), rho_min), rho_max);
        }
    }
}

inline void clark_d4_tied_avx2(
    const double* f_in,
    const double* v_in,
    const double* corr,
    bool is_best_of,
    double& f_sym,
    double& m2_sym,
    double* w_acc
) noexcept {
    static constexpr int kTreesD4[6][4] = {
        {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3},
        {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
    };
    double f_acc = 0.0;
    for (int base = 0; base < 6; base += 4) {
        const int n_act = std::min(4, 6 - base);
        alignas(32) double a_fa[4], a_va[4], a_fb[4], a_vb[4], a_fc[4], a_vc[4], a_fd[4], a_vd[4];
        alignas(32) double a_rab[4], a_rcd[4], a_rac[4], a_rad[4], a_rbc[4], a_rbd[4];
        int idx_a[4], idx_b[4], idx_c[4], idx_d[4];
        for (int l = 0; l < 4; ++l) {
            const int t = base + (l < n_act ? l : n_act - 1);
            const int ia = kTreesD4[t][0], ib = kTreesD4[t][1], ic = kTreesD4[t][2], id = kTreesD4[t][3];
            idx_a[l] = ia; idx_b[l] = ib; idx_c[l] = ic; idx_d[l] = id;
            a_fa[l] = f_in[ia]; a_va[l] = v_in[ia] > 1e-12 ? v_in[ia] : 1e-12;
            a_fb[l] = f_in[ib]; a_vb[l] = v_in[ib] > 1e-12 ? v_in[ib] : 1e-12;
            a_fc[l] = f_in[ic]; a_vc[l] = v_in[ic] > 1e-12 ? v_in[ic] : 1e-12;
            a_fd[l] = f_in[id]; a_vd[l] = v_in[id] > 1e-12 ? v_in[id] : 1e-12;
            a_rab[l] = corr[ia * 4 + ib]; a_rcd[l] = corr[ic * 4 + id];
            a_rac[l] = corr[ia * 4 + ic]; a_rad[l] = corr[ia * 4 + id];
            a_rbc[l] = corr[ib * 4 + ic]; a_rbd[l] = corr[ib * 4 + id];
        }
        __m256d vc = _mm256_load_pd(a_vc), vd = _mm256_load_pd(a_vd);
        __m256d v_cd_rem[2] = {vc, vd};
        __m256d r1_cd_rem[2] = {_mm256_load_pd(a_rac), _mm256_load_pd(a_rad)};
        __m256d r2_cd_rem[2] = {_mm256_load_pd(a_rbc), _mm256_load_pd(a_rbd)};
        __m256d f_ab, v_ab, wa, wb, rho_ab_cd[2];
        clark_pair_merge_avx2<2>(
            _mm256_load_pd(a_fa), _mm256_load_pd(a_va),
            _mm256_load_pd(a_fb), _mm256_load_pd(a_vb),
            _mm256_load_pd(a_rab), v_cd_rem, r1_cd_rem, r2_cd_rem, is_best_of,
            f_ab, v_ab, wa, wb, rho_ab_cd
        );
        __m256d f_cd, v_cd, wc, wd, rho_cd_ab;
        clark_pair_merge_avx2<1>(
            _mm256_load_pd(a_fc), vc, _mm256_load_pd(a_fd), vd, _mm256_load_pd(a_rcd),
            &v_ab, &rho_ab_cd[0], &rho_ab_cd[1], is_best_of,
            f_cd, v_cd, wc, wd, &rho_cd_ab
        );
        __m256d f_r, v_r, w_ab, w_cd;
        clark_pair_merge_avx2<0>(
            f_ab, v_ab, f_cd, v_cd, rho_cd_ab,
            nullptr, nullptr, nullptr, is_best_of,
            f_r, v_r, w_ab, w_cd, nullptr
        );
        __m256d m2_v = _mm256_mul_pd(_mm256_mul_pd(f_r, f_r), v_exp_pd(_mm256_mul_pd(v_r, v_r)));
        alignas(32) double s_f[4], s_m2[4], s_wa[4], s_wb[4], s_wc[4], s_wd[4];
        _mm256_store_pd(s_f, f_r);
        _mm256_store_pd(s_m2, m2_v);
        _mm256_store_pd(s_wa, _mm256_mul_pd(w_ab, wa));
        _mm256_store_pd(s_wb, _mm256_mul_pd(w_ab, wb));
        _mm256_store_pd(s_wc, _mm256_mul_pd(w_cd, wc));
        _mm256_store_pd(s_wd, _mm256_mul_pd(w_cd, wd));
        for (int l = 0; l < n_act; ++l) {
            f_acc += s_f[l];
            m2_sym += s_m2[l];
            w_acc[idx_a[l]] += s_wa[l];
            w_acc[idx_b[l]] += s_wb[l];
            w_acc[idx_c[l]] += s_wc[l];
            w_acc[idx_d[l]] += s_wd[l];
        }
    }
    constexpr double inv6 = 1.0 / 6.0;
    f_sym = f_acc * inv6;
    m2_sym *= inv6;
    for (int i = 0; i < 4; ++i) w_acc[i] *= inv6;
}

inline void clark_d5_tied_avx2(
    const double* f_in,
    const double* v_in,
    const double* corr,
    bool is_best_of,
    double& f_sym,
    double& m2_sym,
    double* w_acc
) noexcept {
    static constexpr int kTreesD4[6][4] = {
        {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3},
        {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
    };
    int flat_idx[30][5];
    int n_t = 0;
    for (int e = 0; e < 5; ++e) {
        int rem4[4]{};
        int rp = 0;
        for (int i = 0; i < 5; ++i) {
            if (i != e) rem4[rp++] = i;
        }
        for (int t = 0; t < 6; ++t) {
            flat_idx[n_t][0] = rem4[kTreesD4[t][0]];
            flat_idx[n_t][1] = rem4[kTreesD4[t][1]];
            flat_idx[n_t][2] = rem4[kTreesD4[t][2]];
            flat_idx[n_t][3] = rem4[kTreesD4[t][3]];
            flat_idx[n_t][4] = e;
            ++n_t;
        }
    }
    double f_acc = 0.0;
    for (int base = 0; base < 30; base += 4) {
        const int n_act = std::min(4, 30 - base);
        alignas(32) double a_fa[4], a_va[4], a_fb[4], a_vb[4], a_fc[4], a_vc[4], a_fd[4], a_vd[4], a_fe[4], a_ve[4];
        alignas(32) double a_rab[4], a_rcd[4], a_rac[4], a_rad[4], a_rae[4], a_rbc[4], a_rbd[4], a_rbe[4], a_rce[4], a_rde[4];
        int ia[4], ib[4], ic[4], id[4], ie[4];
        for (int l = 0; l < 4; ++l) {
            const int t = base + (l < n_act ? l : n_act - 1);
            const int x0 = flat_idx[t][0], x1 = flat_idx[t][1], x2 = flat_idx[t][2], x3 = flat_idx[t][3], x4 = flat_idx[t][4];
            ia[l] = x0; ib[l] = x1; ic[l] = x2; id[l] = x3; ie[l] = x4;
            a_fa[l] = f_in[x0]; a_va[l] = v_in[x0] > 1e-12 ? v_in[x0] : 1e-12;
            a_fb[l] = f_in[x1]; a_vb[l] = v_in[x1] > 1e-12 ? v_in[x1] : 1e-12;
            a_fc[l] = f_in[x2]; a_vc[l] = v_in[x2] > 1e-12 ? v_in[x2] : 1e-12;
            a_fd[l] = f_in[x3]; a_vd[l] = v_in[x3] > 1e-12 ? v_in[x3] : 1e-12;
            a_fe[l] = f_in[x4]; a_ve[l] = v_in[x4] > 1e-12 ? v_in[x4] : 1e-12;
            a_rab[l] = corr[x0 * 5 + x1]; a_rcd[l] = corr[x2 * 5 + x3];
            a_rac[l] = corr[x0 * 5 + x2]; a_rad[l] = corr[x0 * 5 + x3]; a_rae[l] = corr[x0 * 5 + x4];
            a_rbc[l] = corr[x1 * 5 + x2]; a_rbd[l] = corr[x1 * 5 + x3]; a_rbe[l] = corr[x1 * 5 + x4];
            a_rce[l] = corr[x2 * 5 + x4]; a_rde[l] = corr[x3 * 5 + x4];
        }
        __m256d vc = _mm256_load_pd(a_vc), vd = _mm256_load_pd(a_vd), ve = _mm256_load_pd(a_ve);
        __m256d v_cde[3] = {vc, vd, ve};
        __m256d r_a_cde[3] = {_mm256_load_pd(a_rac), _mm256_load_pd(a_rad), _mm256_load_pd(a_rae)};
        __m256d r_b_cde[3] = {_mm256_load_pd(a_rbc), _mm256_load_pd(a_rbd), _mm256_load_pd(a_rbe)};
        __m256d f_ab, v_ab, wa, wb, rho_ab_cde[3];
        clark_pair_merge_avx2<3>(
            _mm256_load_pd(a_fa), _mm256_load_pd(a_va),
            _mm256_load_pd(a_fb), _mm256_load_pd(a_vb),
            _mm256_load_pd(a_rab), v_cde, r_a_cde, r_b_cde, is_best_of,
            f_ab, v_ab, wa, wb, rho_ab_cde
        );
        __m256d v_rem2[2] = {v_ab, ve};
        __m256d r1_rem2[2] = {rho_ab_cde[0], _mm256_load_pd(a_rce)};
        __m256d r2_rem2[2] = {rho_ab_cde[1], _mm256_load_pd(a_rde)};
        __m256d f_cd, v_cd, wc, wd, rho_cd_abe[2];
        clark_pair_merge_avx2<2>(
            _mm256_load_pd(a_fc), vc, _mm256_load_pd(a_fd), vd, _mm256_load_pd(a_rcd),
            v_rem2, r1_rem2, r2_rem2, is_best_of,
            f_cd, v_cd, wc, wd, rho_cd_abe
        );
        __m256d f_abcd, v_abcd, w_ab, w_cd, rho_abcd_e;
        clark_pair_merge_avx2<1>(
            f_ab, v_ab, f_cd, v_cd, rho_cd_abe[0],
            &ve, &rho_ab_cde[2], &rho_cd_abe[1], is_best_of,
            f_abcd, v_abcd, w_ab, w_cd, &rho_abcd_e
        );
        __m256d f_r, v_r, w_abcd, w_e;
        clark_pair_merge_avx2<0>(
            f_abcd, v_abcd, _mm256_load_pd(a_fe), ve, rho_abcd_e,
            nullptr, nullptr, nullptr, is_best_of,
            f_r, v_r, w_abcd, w_e, nullptr
        );
        __m256d m2_v = _mm256_mul_pd(_mm256_mul_pd(f_r, f_r), v_exp_pd(_mm256_mul_pd(v_r, v_r)));
        alignas(32) double s_f[4], s_m2[4], s_wa[4], s_wb[4], s_wc[4], s_wd[4], s_we[4];
        __m256d w_ab_t = _mm256_mul_pd(w_abcd, w_ab);
        __m256d w_cd_t = _mm256_mul_pd(w_abcd, w_cd);
        _mm256_store_pd(s_f, f_r);
        _mm256_store_pd(s_m2, m2_v);
        _mm256_store_pd(s_wa, _mm256_mul_pd(w_ab_t, wa));
        _mm256_store_pd(s_wb, _mm256_mul_pd(w_ab_t, wb));
        _mm256_store_pd(s_wc, _mm256_mul_pd(w_cd_t, wc));
        _mm256_store_pd(s_wd, _mm256_mul_pd(w_cd_t, wd));
        _mm256_store_pd(s_we, w_e);
        for (int l = 0; l < n_act; ++l) {
            f_acc += s_f[l];
            m2_sym += s_m2[l];
            w_acc[ia[l]] += s_wa[l];
            w_acc[ib[l]] += s_wb[l];
            w_acc[ic[l]] += s_wc[l];
            w_acc[id[l]] += s_wd[l];
            w_acc[ie[l]] += s_we[l];
        }
    }
    constexpr double inv30 = 1.0 / 30.0;
    f_sym = f_acc * inv30;
    m2_sym *= inv30;
    for (int i = 0; i < 5; ++i) w_acc[i] *= inv30;
}
#endif

void clark_symmetrized_impl(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double* w_out,
    int* ord_f,
    int* p_buf,
    double* w_acc,
    double* w_tmp,
    double* w_f
) noexcept {
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

    for (int i = 0; i < d; ++i) ord_f[i] = i;
    std::stable_sort(ord_f, ord_f + d, [&](int a, int b) {
        if (std::abs(f_in[a] - f_in[b]) > 1e-9) {
            return is_best_of ? (f_in[a] > f_in[b]) : (f_in[a] < f_in[b]);
        }
        if (std::abs(v_in[a] - v_in[b]) > 1e-12) {
            return v_in[a] < v_in[b];
        }
        return a < b;
    });

    if (w_tie < 1e-12) {
        clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, f_out, v_out, w_out, nullptr);
        return;
    }

    double f_sym = 0.0;
    double m2_sym = 0.0;
    for (int i = 0; i < d; ++i) w_acc[i] = 0.0;

    if (d == 2) {
        double v_sym_d2 = 0.0;
        clark_pair_merge(
            f_in[0], v_in[0], f_in[1], v_in[1], corr[1],
            0, nullptr, nullptr, nullptr, is_best_of,
            f_sym, v_sym_d2, w_acc[0], w_acc[1], nullptr
        );
        m2_sym = f_sym * f_sym * std::exp(v_sym_d2 * v_sym_d2);
    } else if (d == 3) {
        static constexpr int kPerms3[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};
        double f_acc = 0.0;
        for (int p = 0; p < 3; ++p) {
            double fp = 0.0, vp = 0.0;
            clark_one_order(3, f_in, v_in, corr, kPerms3[p], is_best_of, fp, vp, w_tmp, nullptr);
            f_acc += fp;
            m2_sym += fp * fp * std::exp(vp * vp);
            for (int i = 0; i < 3; ++i) w_acc[i] += w_tmp[i];
        }
        constexpr double inv3 = 1.0 / 3.0;
        f_sym = f_acc * inv3;
        m2_sym *= inv3;
        for (int i = 0; i < 3; ++i) w_acc[i] *= inv3;
    } else if (d == 4) {
#if defined(__AVX2__) && defined(__FMA__)
        clark_d4_tied_avx2(f_in, v_in, corr, is_best_of, f_sym, m2_sym, w_acc);
#else
        static constexpr int kTreesD4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3},
            {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        double f_acc = 0.0;
        for (int t = 0; t < 6; ++t) {
            const int a = kTreesD4[t][0], b = kTreesD4[t][1], c = kTreesD4[t][2], dd = kTreesD4[t][3];
            const double va = v_in[a] > 1e-12 ? v_in[a] : 1e-12;
            const double vb = v_in[b] > 1e-12 ? v_in[b] : 1e-12;
            const double vc = v_in[c] > 1e-12 ? v_in[c] : 1e-12;
            const double vd = v_in[dd] > 1e-12 ? v_in[dd] : 1e-12;
            const double v_cd_arr[2] = {vc, vd};
            const double r_a_cd[2] = {corr[a * 4 + c], corr[a * 4 + dd]};
            const double r_b_cd[2] = {corr[b * 4 + c], corr[b * 4 + dd]};
            double f_ab = 0.0, v_ab = 0.0, wa = 0.0, wb = 0.0, rho_ab_cd[2]{};
            clark_pair_merge(
                f_in[a], va, f_in[b], vb, corr[a * 4 + b],
                2, v_cd_arr, r_a_cd, r_b_cd, is_best_of,
                f_ab, v_ab, wa, wb, rho_ab_cd
            );
            double f_cd = 0.0, v_cd = 0.0, wc = 0.0, wd = 0.0, rho_cd_ab = 0.0;
            clark_pair_merge(
                f_in[c], vc, f_in[dd], vd, corr[c * 4 + dd],
                1, &v_ab, &rho_ab_cd[0], &rho_ab_cd[1], is_best_of,
                f_cd, v_cd, wc, wd, &rho_cd_ab
            );
            double f_r = 0.0, v_r = 0.0, w_ab = 0.0, w_cd = 0.0;
            clark_pair_merge(
                f_ab, v_ab, f_cd, v_cd, rho_cd_ab,
                0, nullptr, nullptr, nullptr, is_best_of,
                f_r, v_r, w_ab, w_cd, nullptr
            );
            f_acc += f_r;
            m2_sym += f_r * f_r * std::exp(v_r * v_r);
            w_acc[a] += w_ab * wa;
            w_acc[b] += w_ab * wb;
            w_acc[c] += w_cd * wc;
            w_acc[dd] += w_cd * wd;
        }
        constexpr double inv6 = 1.0 / 6.0;
        f_sym = f_acc * inv6;
        m2_sym *= inv6;
        for (int i = 0; i < 4; ++i) w_acc[i] *= inv6;
#endif
    } else if (d == 5) {
#if defined(__AVX2__) && defined(__FMA__)
        clark_d5_tied_avx2(f_in, v_in, corr, is_best_of, f_sym, m2_sym, w_acc);
#else
        static constexpr int kTreesD4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3},
            {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        double f_acc = 0.0;
        for (int e = 0; e < 5; ++e) {
            int rem4[4]{};
            int rp = 0;
            for (int i = 0; i < 5; ++i) {
                if (i != e) rem4[rp++] = i;
            }
            const double ve = v_in[e] > 1e-12 ? v_in[e] : 1e-12;
            for (int t = 0; t < 6; ++t) {
                const int a = rem4[kTreesD4[t][0]];
                const int b = rem4[kTreesD4[t][1]];
                const int c = rem4[kTreesD4[t][2]];
                const int dd = rem4[kTreesD4[t][3]];
                const double va = v_in[a] > 1e-12 ? v_in[a] : 1e-12;
                const double vb = v_in[b] > 1e-12 ? v_in[b] : 1e-12;
                const double vc = v_in[c] > 1e-12 ? v_in[c] : 1e-12;
                const double vd = v_in[dd] > 1e-12 ? v_in[dd] : 1e-12;
                const double v_cde[3] = {vc, vd, ve};
                const double r_a_cde[3] = {corr[a * 5 + c], corr[a * 5 + dd], corr[a * 5 + e]};
                const double r_b_cde[3] = {corr[b * 5 + c], corr[b * 5 + dd], corr[b * 5 + e]};
                double f_ab = 0.0, v_ab = 0.0, wa = 0.0, wb = 0.0, rho_ab_cde[3]{};
                clark_pair_merge(
                    f_in[a], va, f_in[b], vb, corr[a * 5 + b],
                    3, v_cde, r_a_cde, r_b_cde, is_best_of,
                    f_ab, v_ab, wa, wb, rho_ab_cde
                );
                const double v_rem2[2] = {v_ab, ve};
                const double r1_rem2[2] = {rho_ab_cde[0], corr[c * 5 + e]};
                const double r2_rem2[2] = {rho_ab_cde[1], corr[dd * 5 + e]};
                double f_cd = 0.0, v_cd = 0.0, wc = 0.0, wd = 0.0, rho_cd_abe[2]{};
                clark_pair_merge(
                    f_in[c], vc, f_in[dd], vd, corr[c * 5 + dd],
                    2, v_rem2, r1_rem2, r2_rem2, is_best_of,
                    f_cd, v_cd, wc, wd, rho_cd_abe
                );
                double f_abcd = 0.0, v_abcd = 0.0, w_ab = 0.0, w_cd = 0.0, rho_abcd_e = 0.0;
                clark_pair_merge(
                    f_ab, v_ab, f_cd, v_cd, rho_cd_abe[0],
                    1, &ve, &rho_ab_cde[2], &rho_cd_abe[1], is_best_of,
                    f_abcd, v_abcd, w_ab, w_cd, &rho_abcd_e
                );
                double f_r = 0.0, v_r = 0.0, w_abcd = 0.0, w_e = 0.0;
                clark_pair_merge(
                    f_abcd, v_abcd, f_in[e], ve, rho_abcd_e,
                    0, nullptr, nullptr, nullptr, is_best_of,
                    f_r, v_r, w_abcd, w_e, nullptr
                );
                f_acc += f_r;
                m2_sym += f_r * f_r * std::exp(v_r * v_r);
                w_acc[a] += w_abcd * w_ab * wa;
                w_acc[b] += w_abcd * w_ab * wb;
                w_acc[c] += w_abcd * w_cd * wc;
                w_acc[dd] += w_abcd * w_cd * wd;
                w_acc[e] += w_e;
            }
        }
        constexpr double inv30 = 1.0 / 30.0;
        f_sym = f_acc * inv30;
        m2_sym *= inv30;
        for (int i = 0; i < 5; ++i) w_acc[i] *= inv30;
#endif
    } else {
        // Canonical cyclic + reverse shifts avoid data-dependent argsort(v).
        double f_acc = 0.0;
        int n_perms = 0;
        for (int s = 0; s < d; ++s) {
            for (int k = 0; k < d; ++k) p_buf[k] = (s + k) % d;
            double fp = 0.0, vp = 0.0;
            clark_one_order(d, f_in, v_in, corr, p_buf, is_best_of, fp, vp, w_tmp, nullptr);
            f_acc += fp;
            m2_sym += fp * fp * std::exp(vp * vp);
            for (int i = 0; i < d; ++i) w_acc[i] += w_tmp[i];
            std::reverse(p_buf, p_buf + d);
            clark_one_order(d, f_in, v_in, corr, p_buf, is_best_of, fp, vp, w_tmp, nullptr);
            f_acc += fp;
            m2_sym += fp * fp * std::exp(vp * vp);
            for (int i = 0; i < d; ++i) w_acc[i] += w_tmp[i];
            n_perms += 2;
        }
        const double inv_k = 1.0 / static_cast<double>(n_perms);
        f_sym = f_acc * inv_k;
        m2_sym *= inv_k;
        for (int i = 0; i < d; ++i) w_acc[i] *= inv_k;
    }

    const double ratio_sym = m2_sym / std::max(f_sym * f_sym, 1e-30);
    const double v_sym = std::sqrt(ratio_sym > 1.0 + 1e-16 ? std::log(ratio_sym) : 1e-16);

    if (g_sep < 1e-15) {
        f_out = f_sym;
        v_out = v_sym;
        for (int i = 0; i < d; ++i) w_out[i] = w_acc[i];
        return;
    }

    double f_f = 0.0;
    double v_f = 0.0;
    clark_one_order(d, f_in, v_in, corr, ord_f, is_best_of, f_f, v_f, w_f, nullptr);

    f_out = g_sep * f_f + w_tie * f_sym;
    const double m2_out = g_sep * (f_f * f_f * std::exp(v_f * v_f)) + w_tie * m2_sym;
    const double ratio_out = m2_out / std::max(f_out * f_out, 1e-30);
    v_out = std::sqrt(ratio_out > 1.0 + 1e-16 ? std::log(ratio_out) : 1e-16);
    for (int i = 0; i < d; ++i) {
        w_out[i] = g_sep * w_f[i] + w_tie * w_acc[i];
    }
}

[[gnu::noinline]] void clark_symmetrized_dyn(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    bool is_best_of,
    double& f_out,
    double& v_out,
    double* w_out
) noexcept {
    const auto du = static_cast<std::size_t>(d);
    std::vector<int> ord_f(du), p_buf(du);
    std::vector<double> w_acc(du, 0.0), w_tmp(du, 0.0), w_f(du, 0.0);
    clark_symmetrized_impl(
        d, f_in, v_in, corr, is_best_of, f_out, v_out, w_out,
        ord_f.data(), p_buf.data(), w_acc.data(), w_tmp.data(), w_f.data()
    );
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
    static constexpr int kStackD = 32;
    if (d <= kStackD) {
        std::array<int, kStackD> ord_f_stk{}, p_stk{};
        std::array<double, kStackD> w_acc_stk{}, w_tmp_stk{}, w_f_stk{};
        clark_symmetrized_impl(
            d, f_in, v_in, corr, is_best_of, f_out, v_out, w_out,
            ord_f_stk.data(), p_stk.data(), w_acc_stk.data(), w_tmp_stk.data(), w_f_stk.data()
        );
        return;
    }
    clark_symmetrized_dyn(d, f_in, v_in, corr, is_best_of, f_out, v_out, w_out);
}

double clark_corr_min_max_impl(
    int d,
    const double* f_in,
    const double* v_in,
    const double* corr,
    int* ord_min,
    int* ord_max,
    double* rho_min_p,
    double* f_ord,
    double* v_ord,
    double* rho_cur
) noexcept {
    for (int i = 0; i < d; ++i) {
        ord_min[i] = i;
        ord_max[i] = i;
        rho_min_p[i] = 0.0;
    }
    std::stable_sort(ord_min, ord_min + d, [&](int a, int b) {
        if (std::abs(f_in[a] - f_in[b]) > 1e-9) return f_in[a] < f_in[b];
        if (std::abs(v_in[a] - v_in[b]) > 1e-12) return v_in[a] < v_in[b];
        return a < b;
    });
    std::stable_sort(ord_max, ord_max + d, [&](int a, int b) {
        if (std::abs(f_in[a] - f_in[b]) > 1e-9) return f_in[a] > f_in[b];
        if (std::abs(v_in[a] - v_in[b]) > 1e-12) return v_in[a] < v_in[b];
        return a < b;
    });

    double f_min = 0.0, v_min = 0.0;
    clark_one_order(d, f_in, v_in, corr, ord_min, false, f_min, v_min, nullptr, rho_min_p);

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

[[gnu::noinline]] double clark_corr_min_max_dyn(int d, const double* f_in, const double* v_in, const double* corr) noexcept {
    const auto du = static_cast<std::size_t>(d);
    std::vector<int> ord_min(du), ord_max(du);
    std::vector<double> rho_min_p(du, 0.0), f_ord(du), v_ord(du), rho_cur(du);
    return clark_corr_min_max_impl(
        d, f_in, v_in, corr,
        ord_min.data(), ord_max.data(), rho_min_p.data(),
        f_ord.data(), v_ord.data(), rho_cur.data()
    );
}

double clark_corr_min_max(int d, const double* f_in, const double* v_in, const double* corr) noexcept {
    if (d <= 1) {
        return 1.0;
    }
    static constexpr int kStackD = 32;
    if (d <= kStackD) {
        std::array<int, kStackD> ord_min{}, ord_max{};
        std::array<double, kStackD> rho_min_p{}, f_ord{}, v_ord{}, rho_cur{};
        return clark_corr_min_max_impl(
            d, f_in, v_in, corr,
            ord_min.data(), ord_max.data(), rho_min_p.data(),
            f_ord.data(), v_ord.data(), rho_cur.data()
        );
    }
    return clark_corr_min_max_dyn(d, f_in, v_in, corr);
}

inline bool is_diff_index(int code) noexcept {
    return code >= 9 && code <= 13;
}

// 8-step power iteration extracts optimal rank-1 correlation loadings for 1-factor baskets.
void fit_1factor_loadings(int d, const double* corr, double* b_out) noexcept {
    if (d <= 1) {
        if (d == 1) b_out[0] = 0.0;
        return;
    }
    static constexpr int kStackD = 32;
    std::array<double, kStackD> mb_stk{};
    std::vector<double> mb_hp;
    double* mb = mb_stk.data();
    if (d > kStackD) [[unlikely]] {
        mb_hp.assign(static_cast<std::size_t>(d), 0.0);
        mb = mb_hp.data();
    }
    double sum_corr = 0.0;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) sum_corr += corr[i * d + j];
    }
    const double rho_off = std::clamp((sum_corr - static_cast<double>(d)) / static_cast<double>(d * (d - 1)), 0.0, 0.999);
    const double b0 = std::sqrt(rho_off);
    for (int i = 0; i < d; ++i) b_out[i] = b0;

    for (int it = 0; it < 8; ++it) {
        double n2 = 0.0;
        for (int i = 0; i < d; ++i) {
            double acc = b_out[i] * b_out[i] * b_out[i];
            for (int j = 0; j < d; ++j) {
                if (j != i) acc += corr[i * d + j] * b_out[j];
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
            double mu_i = (b_out[i] * b_out[i]) * mb[i];
            for (int j = 0; j < d; ++j) {
                if (j != i) mu_i += corr[i * d + j] * mb[j];
            }
            lam += mb[i] * mu_i;
        }
        const double s_lam = std::sqrt(std::max(lam, 0.0));
        for (int i = 0; i < d; ++i) {
            b_out[i] = std::clamp(mb[i] * s_lam, 0.0, 0.999);
        }
    }
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
    if (d <= 0) {
        f_out = 0.0;
        v_out = 1e-12;
        return;
    }
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
            idx_a = std::clamp(static_cast<int>(inv_lex_perm[0]), 0, d - 1);
            idx_b = std::clamp(static_cast<int>(inv_lex_perm[1]), 0, d - 1);
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
            const double vi = std::max(v_in[i], 1e-12);
            for (int j = 0; j < d; ++j) {
                const double vj = std::max(v_in[j], 1e-12);
                m2_sum += f_in[i] * f_in[j] * std::exp(corr[i * d + j] * vi * vj);
            }
        }
        const double m2 = m2_sum * (inv_d * inv_d);
        const double ratio = m2 / std::max(f_out * f_out, 1e-30);
        v_out = std::sqrt(ratio > 1.0 + 1e-16 ? std::log(ratio) : 1e-16);
        for (int i = 0; i < d; ++i) w_out[i] = inv_d;
        return;
    }
    if (code == 3) {
        double m2_p_sum = 0.0;
        double m2_avg_sum = 0.0;
        double f_sum = 0.0;
        const double inv_d = 1.0 / static_cast<double>(d);
        for (int i = 0; i < d; ++i) {
            const double vi = std::max(v_in[i], 1e-12);
            f_sum += f_in[i];
            m2_p_sum += f_in[i] * f_in[i] * std::exp(vi * vi);
            for (int j = 0; j < d; ++j) {
                const double vj = std::max(v_in[j], 1e-12);
                m2_avg_sum += f_in[i] * f_in[j] * std::exp(corr[i * d + j] * vi * vj);
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
        const double sig_approx = std::sqrt(ew2);
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
        static constexpr int kStackD = 32;
        std::array<double, kStackD> wn_stk{}, wd_stk{};
        std::vector<double> wn_hp, wd_hp;
        double* w_num_p = wn_stk.data();
        double* w_den_p = wd_stk.data();
        if (d > kStackD) [[unlikely]] {
            const auto du = static_cast<std::size_t>(d);
            wn_hp.assign(du, 0.0);
            wd_hp.assign(du, 0.0);
            w_num_p = wn_hp.data();
            w_den_p = wd_hp.data();
        } else {
            for (int i = 0; i < d; ++i) {
                w_num_p[i] = 0.0;
                w_den_p[i] = 0.0;
            }
        }
        if (code == 6) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_num, v_num, w_num_p);
            clark_symmetrized(d, f_in, v_in, corr, false, f_den, v_den, w_den_p);
        } else if (code == 7) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_num, v_num, w_num_p);
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_den, v_den, w_den_p);
        } else {
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_num, v_num, w_num_p);
            clark_symmetrized(d, f_in, v_in, corr, false, f_den, v_den, w_den_p);
        }
        double rho_nd = 0.0;
        if (code == 6) {
            rho_nd = clark_corr_min_max(d, f_in, v_in, corr);
        } else {
            double cov = 0.0, vn2 = 0.0, vd2 = 0.0;
            for (int i = 0; i < d; ++i) {
                const double vi = std::max(v_in[i], 1e-12);
                for (int j = 0; j < d; ++j) {
                    const double vj = std::max(v_in[j], 1e-12);
                    const double cij = corr[i * d + j] * vi * vj;
                    cov += w_num_p[i] * w_den_p[j] * cij;
                    vn2 += w_num_p[i] * w_num_p[j] * cij;
                    vd2 += w_den_p[i] * w_den_p[j] * cij;
                }
            }
            rho_nd = std::clamp(cov / std::sqrt(std::max(vn2 * vd2, 1e-30)), -0.9999, 0.9999);
        }
        const double var_r = std::max(v_num * v_num + v_den * v_den - 2.0 * rho_nd * v_num * v_den, 1e-16);
        f_out = (f_num / std::max(f_den, 1e-16)) * std::exp(v_den * v_den - rho_nd * v_num * v_den);
        v_out = std::sqrt(var_r);
        for (int i = 0; i < d; ++i) w_out[i] = w_num_p[i] - w_den_p[i];
        return;
    }
    if (code == 9 || code == 10) {
        int ia = 0, ib = 1;
        resolve_ab(ia, ib);
        if (code == 10) std::swap(ia, ib);
        const double fa = f_in[ia];
        const double fb = f_in[ib];
        const double va = v_in[ia] > 1e-12 ? v_in[ia] : 1e-12;
        const double vb = v_in[ib] > 1e-12 ? v_in[ib] : 1e-12;
        const double rab = corr[ia * d + ib];
        f_out = fa - fb;
        const double var_d = fa * fa * std::expm1(va * va) + fb * fb * std::expm1(vb * vb) - 2.0 * fa * fb * std::expm1(rab * va * vb);
        v_out = std::sqrt(std::max(var_d, 1e-16));
        for (int i = 0; i < d; ++i) w_out[i] = 0.0;
        w_out[ia] = 1.0;
        w_out[ib] = -1.0;
        return;
    }
    if (code == 11 || code == 12 || code == 13) {
        double f_a = 0.0, v_a = 0.0, f_b = 0.0, v_b = 0.0;
        static constexpr int kStackD = 32;
        std::array<double, kStackD> wa_stk{}, wb_stk{};
        std::vector<double> wa_hp, wb_hp;
        double* w_a_p = wa_stk.data();
        double* w_b_p = wb_stk.data();
        if (d > kStackD) [[unlikely]] {
            const auto du = static_cast<std::size_t>(d);
            wa_hp.assign(du, 0.0);
            wb_hp.assign(du, 0.0);
            w_a_p = wa_hp.data();
            w_b_p = wb_hp.data();
        } else {
            for (int i = 0; i < d; ++i) {
                w_a_p[i] = 0.0;
                w_b_p[i] = 0.0;
            }
        }
        if (code == 11) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_a, v_a, w_a_p);
            clark_symmetrized(d, f_in, v_in, corr, false, f_b, v_b, w_b_p);
        } else if (code == 12) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_a, v_a, w_a_p);
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_b, v_b, w_b_p);
        } else {
            eval_index_moments(d, f_in, v_in, corr, inv_lex_perm, 2, f_a, v_a, w_a_p);
            clark_symmetrized(d, f_in, v_in, corr, false, f_b, v_b, w_b_p);
        }
        double rho_ab = 0.0;
        if (code == 11) {
            rho_ab = clark_corr_min_max(d, f_in, v_in, corr);
        } else {
            double cov = 0.0, va2 = 0.0, vb2 = 0.0;
            for (int i = 0; i < d; ++i) {
                const double vi = std::max(v_in[i], 1e-12);
                for (int j = 0; j < d; ++j) {
                    const double vj = std::max(v_in[j], 1e-12);
                    const double cij = corr[i * d + j] * vi * vj;
                    cov += w_a_p[i] * w_b_p[j] * cij;
                    va2 += w_a_p[i] * w_a_p[j] * cij;
                    vb2 += w_b_p[i] * w_b_p[j] * cij;
                }
            }
            rho_ab = std::clamp(cov / std::sqrt(std::max(va2 * vb2, 1e-30)), -0.9999, 0.9999);
        }
        f_out = f_a - f_b;
        const double var_d = f_a * f_a * std::expm1(v_a * v_a) + f_b * f_b * std::expm1(v_b * v_b) - 2.0 * f_a * f_b * std::expm1(rho_ab * v_a * v_b);
        v_out = std::sqrt(std::max(var_d, 1e-16));
        for (int i = 0; i < d; ++i) w_out[i] = w_a_p[i] - w_b_p[i];
        return;
    }
    if (code == 15 || code == 16 || code == 17) {
        const int target_k = (code == 15) ? std::min(1, d - 1)
            : ((code == 16) ? std::min(2, d - 1) : std::max(0, d - 2));
        if (target_k == 0) {
            clark_symmetrized(d, f_in, v_in, corr, false, f_out, v_out, w_out);
            return;
        }
        if (target_k == d - 1) {
            clark_symmetrized(d, f_in, v_in, corr, true, f_out, v_out, w_out);
            return;
        }
        static constexpr int kStackD = 32;
        std::array<double, kStackD> b_stk{}, mu_stk{}, vb_stk{}, vs_stk{}, dp_stk{};
        std::vector<double> dyn_hp;
        double* b = b_stk.data();
        double* mu = mu_stk.data();
        double* vb = vb_stk.data();
        double* vs = vs_stk.data();
        double* dp = dp_stk.data();
        const auto du = static_cast<std::size_t>(d);
        if (d > kStackD) [[unlikely]] {
            dyn_hp.assign(5 * du + 1, 0.0);
            b = dyn_hp.data();
            mu = b + d;
            vb = mu + d;
            vs = vb + d;
            dp = vs + d;
        }
        fit_1factor_loadings(d, corr, b);
        for (int i = 0; i < d; ++i) {
            const double fi = std::max(f_in[i], 1e-16);
            const double vi = std::max(v_in[i], 1e-12);
            const double s_eps = std::sqrt(std::max(1.0 - b[i] * b[i], 1e-12));
            mu[i] = std::log(fi) - 0.5 * vi * vi;
            vb[i] = vi * b[i];
            vs[i] = std::max(vi * s_eps, 1e-12);
            w_out[i] = 0.0;
        }
        double f_sum = 0.0;
        double m2_sum = 0.0;
        for (std::size_t qy = 0; qy < 16; ++qy) {
            const double y = kGL16YNorm[qy];
            const double wy = 0.5 * kGL16W[qy];
            for (int i = 0; i < d; ++i) {
                const double fi = std::max(f_in[i], 1e-16);
                const double vi = std::max(v_in[i], 1e-12);
                const double base_shift = -0.5 * vi * vi + vb[i] * y;
                for (std::size_t qz = 0; qz < 16; ++qz) {
                    const double z = kGL16YNorm[qz];
                    const double wz = 0.5 * kGL16W[qz];
                    const double lx = mu[i] + vb[i] * y + vs[i] * z;
                    const double unit_p = std::exp(base_shift + vs[i] * z);
                    const double p_val = fi * unit_p;
                    for (int r = 0; r <= target_k; ++r) dp[r] = 0.0;
                    dp[0] = 1.0;
                    int lim = 0;
                    for (int j = 0; j < d; ++j) {
                        if (j == i) continue;
                        const double prob = ndtr_std((lx - mu[j] - vb[j] * y) / vs[j]);
                        const double qrob = 1.0 - prob;
                        const int next_lim = std::min(lim + 1, target_k);
                        for (int r = next_lim; r >= 1; --r) {
                            dp[r] = dp[r] * qrob + dp[r - 1] * prob;
                        }
                        dp[0] *= qrob;
                        lim = next_lim;
                    }
                    const double pr = wy * wz * dp[target_k];
                    w_out[i] += pr * unit_p;
                    f_sum += pr * p_val;
                    m2_sum += pr * p_val * p_val;
                }
            }
        }
        f_out = std::max(f_sum, 1e-16);
        const double ratio = m2_sum / std::max(f_out * f_out, 1e-30);
        v_out = std::sqrt(ratio > 1.0 + 1e-16 ? std::log(ratio) : 1e-16);
        return;
    }
    clark_symmetrized(d, f_in, v_in, corr, false, f_out, v_out, w_out);
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
        const double vi = std::max(v_in[i], 1e-12);
        const double wai = w_a[i] * (da_diff ? f_in[i] : 1.0);
        for (int j = 0; j < d; ++j) {
            const double vj = std::max(v_in[j], 1e-12);
            const double wbj = w_b[j] * (db_diff ? f_in[j] : 1.0);
            const double cij = corr[i * d + j] * vi * vj;
            num += wai * wbj * cij;
        }
    }
    for (int i = 0; i < d; ++i) {
        const double vi = std::max(v_in[i], 1e-12);
        const double wai = w_a[i] * (da_diff ? f_in[i] : 1.0);
        const double wbi = w_b[i] * (db_diff ? f_in[i] : 1.0);
        for (int j = 0; j < d; ++j) {
            const double vj = std::max(v_in[j], 1e-12);
            const double waj = w_a[j] * (da_diff ? f_in[j] : 1.0);
            const double wbj = w_b[j] * (db_diff ? f_in[j] : 1.0);
            const double cij = corr[i * d + j] * vi * vj;
            da += wai * waj * cij;
            db += wbi * wbj * cij;
        }
    }
    return std::clamp(num / std::sqrt(std::max(da * db, 1e-30)), -0.9999, 0.9999);
}

inline void lognormal_moments_interval(
    double f,
    double v,
    double a,
    double b,
    int max_order,
    double* out
) noexcept {
    if (max_order < 0) return;
    if (max_order > 4) max_order = 4;
    if (b <= a || b <= 0.0) {
        for (int k = 0; k <= max_order; ++k) out[k] = 0.0;
        return;
    }
    const double f_c = f > 1e-16 ? f : 1e-16;
    const double v_c = v > 1e-12 ? v : 1e-12;
    const double m_u = std::log(f_c) - 0.5 * v_c * v_c;
    const double log_b = std::log(b);
    const double u_b = (log_b - m_u) / v_c;
    const bool has_a = a > 0.0;
    const double log_a = has_a ? std::log(a) : 0.0;
    const double u_a = has_a ? (log_a - m_u) / v_c : 0.0;
    double fp = 1.0;
    for (int k = 0; k <= max_order; ++k) {
        const double kd = static_cast<double>(k);
        const double zb = u_b - kd * v_c;
        const double za = has_a ? (u_a - kd * v_c) : 0.0;
        if (!has_a && zb <= -6.0) {
            out[k] = 0.5 * std::exp(kd * log_b - 0.5 * u_b * u_b) * cody_erfcx(-zb * kInvSqrt2);
        } else if (has_a && zb <= -6.0) {
            const double exp_ba = std::exp(-0.5 * (za - zb) * (za + zb));
            const double br = cody_erfcx(-zb * kInvSqrt2) - exp_ba * cody_erfcx(-za * kInvSqrt2);
            out[k] = (br <= 0.0) ? 0.0 : (0.5 * std::exp(kd * log_b - 0.5 * u_b * u_b) * br);
        } else if (has_a && za >= 6.0) {
            const double exp_ab = std::exp(-0.5 * (zb - za) * (zb + za));
            const double br = cody_erfcx(za * kInvSqrt2) - exp_ab * cody_erfcx(zb * kInvSqrt2);
            out[k] = (br <= 0.0) ? 0.0 : (0.5 * std::exp(kd * log_a - 0.5 * u_a * u_a) * br);
        } else {
            const double diff = has_a ? (ndtr_std(zb) - ndtr_std(za)) : ndtr_std(zb);
            if (diff <= 0.0) {
                out[k] = 0.0;
            } else {
                const double scale = fp * std::exp(0.5 * kd * (kd - 1.0) * v_c * v_c);
                out[k] = scale * diff;
            }
        }
        fp *= f_c;
    }
}

inline void normal_moments_interval(
    double mu,
    double sig,
    double a,
    double b,
    int max_order,
    double* out
) noexcept {
    if (max_order < 0) return;
    if (b <= a) {
        for (int k = 0; k <= max_order; ++k) out[k] = 0.0;
        return;
    }
    const double s_c = sig > 1e-12 ? sig : 1e-12;
    const double u_b = (b - mu) / s_c;
    const bool has_a = a > -1e50;
    const double u_a = has_a ? (a - mu) / s_c : 0.0;

    // Deep left tail: expand around b with Z = u_b - U >= 0 to avoid mu*j0 - sig*phi cancellation.
    if (u_b <= -15.0) {
        const double x = -u_b;
        const double pref = kInvSqrt2Pi * std::exp(-0.5 * x * x);
        if (pref <= 0.0) {
            for (int k = 0; k <= max_order; ++k) out[k] = 0.0;
            return;
        }
        const double du = has_a ? (u_b - u_a) : 1e100;
        double zm[4] = {0.0, 0.0, 0.0, 0.0};
        if (has_a && du * x < 45.0) {
            for (std::size_t q = 0; q < 8; ++q) {
                const double zq = du * kGL8X01[q];
                const double wt = pref * du * kGL8W01[q] * std::exp(-zq * (x + 0.5 * zq));
                zm[0] += wt;
                zm[1] += wt * zq;
                zm[2] += wt * zq * zq;
                zm[3] += wt * zq * zq * zq;
            }
        } else {
            const double inv_x = 1.0 / x;
            const double p_scale = pref * inv_x;
            for (std::size_t q = 0; q < 6; ++q) {
                const double zq = kLag6X[q] * inv_x;
                const double wt = p_scale * kLag6W[q] * std::exp(-0.5 * zq * zq);
                zm[0] += wt;
                zm[1] += wt * zq;
                zm[2] += wt * zq * zq;
                zm[3] += wt * zq * zq * zq;
            }
        }
        out[0] = zm[0];
        if (max_order >= 1) out[1] = b * zm[0] - s_c * zm[1];
        if (max_order >= 2) out[2] = b * b * zm[0] - 2.0 * b * s_c * zm[1] + s_c * s_c * zm[2];
        if (max_order >= 3) {
            out[3] = b * b * b * zm[0] - 3.0 * b * b * s_c * zm[1] + 3.0 * b * s_c * s_c * zm[2] - s_c * s_c * s_c * zm[3];
        }
        return;
    }
    if (has_a && u_a >= 15.0) {
        normal_moments_interval(-mu, s_c, -b, -a, max_order, out);
        if (max_order >= 1) out[1] = -out[1];
        if (max_order >= 3) out[3] = -out[3];
        return;
    }

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
    bool has_lower_strike,
    double lower_strike,
    bool has_active_ki,
    double ki_barrier,
    double ki_spread,
    bool has_expiry_ko,
    double ko_barrier,
    double ko_spread
) noexcept {
    if (!is_normal && strike <= 1e-8) {
        return 0.0;
    }
    const double k_low = has_lower_strike ? lower_strike : 0.0;
    if (has_lower_strike && k_low >= strike) {
        return 0.0;
    }

    const double low_bound = is_normal ? -1e60 : 0.0;
    std::array<double, 8> pts{};
    std::size_t n_pts = 0;
    auto push_pt = [&](double x) {
        if (x >= low_bound && x <= strike && n_pts < pts.size()) {
            pts[n_pts++] = x;
        }
    };
    push_pt(low_bound);
    push_pt(strike);
    if (has_lower_strike && k_low > low_bound) push_pt(k_low);
    if (has_active_ki) {
        if (std::abs(ki_spread) <= 1e-6) {
            push_pt(ki_barrier);
        } else {
            push_pt(std::min(ki_barrier, ki_barrier + ki_spread));
            push_pt(std::max(ki_barrier, ki_barrier + ki_spread));
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
    for (std::size_t i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        std::size_t j = i;
        while (j > 0 && pts[j - 1] > key) {
            pts[j] = pts[j - 1];
            --j;
        }
        pts[j] = key;
    }

    double total = 0.0;
    for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
        const double a = pts[idx];
        const double b = pts[idx + 1];
        if (b - a <= 1e-14) continue;
        const double mid = (a < -1e30) ? (b - 1.0) : (0.5 * (a + b));

        double c0 = 0.0, c1 = 0.0;
        if (has_lower_strike && mid <= k_low) {
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
                const double b_ki_lo = std::min(ki_barrier, ki_barrier + ki_spread);
                const double b_ki_hi = std::max(ki_barrier, ki_barrier + ki_spread);
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
    std::size_t n_pts = 0;
    pts[n_pts++] = k_call;
    if (b_ko_lo > k_call) pts[n_pts++] = b_ko_lo;
    pts[n_pts++] = b_ko_hi;
    for (std::size_t i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        std::size_t j = i;
        while (j > 0 && pts[j - 1] > key) {
            pts[j] = pts[j - 1];
            --j;
        }
        pts[j] = key;
    }

    double total = 0.0;
    for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
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

double exact_twinwin_terminal_put(
    double f,
    double v,
    double strike,
    bool has_lower_strike,
    double lower_strike,
    bool has_active_ki,
    double ki_barrier,
    double ki_spread,
    bool has_expiry_ko,
    double ko_barrier,
    double ko_spread
) noexcept {
    // W_twin = max(W_min, 2 - W_min) >= 1.0, so strike <= 1.0 produces zero put value.
    if (strike <= 1.0) return 0.0;
    const double k_low = has_lower_strike ? lower_strike : 0.0;
    if (has_lower_strike && k_low >= strike) return 0.0;

    const double b_ki_lo = std::min(ki_barrier, ki_barrier + ki_spread);
    const double b_ki_hi = std::max(ki_barrier, ki_barrier + ki_spread);
    const double b_ko_lo = std::min(ko_barrier, ko_barrier + ko_spread);
    const double b_ko_hi = std::max(ko_barrier, ko_barrier + ko_spread);
    const double x_dn_min = std::max(2.0 - strike, 0.0);

    std::array<double, 16> pts{};
    std::size_t n_pts = 0;
    auto push_pt = [&](double x) {
        if (x >= x_dn_min && x <= strike && n_pts < pts.size()) {
            pts[n_pts++] = x;
        }
    };
    push_pt(x_dn_min);
    push_pt(1.0);
    push_pt(strike);
    if (has_lower_strike && k_low > 1.0) {
        push_pt(k_low);
        push_pt(2.0 - k_low);
    }
    if (has_active_ki) {
        push_pt(b_ki_lo); push_pt(b_ki_hi);
        push_pt(2.0 - b_ki_hi); push_pt(2.0 - b_ki_lo);
    }
    if (has_expiry_ko) {
        push_pt(b_ko_lo); push_pt(b_ko_hi);
        push_pt(2.0 - b_ko_hi); push_pt(2.0 - b_ko_lo);
    }
    for (std::size_t i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        std::size_t j = i;
        while (j > 0 && pts[j - 1] > key) {
            pts[j] = pts[j - 1];
            --j;
        }
        pts[j] = key;
    }

    double total = 0.0;
    for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
        const double a = pts[idx];
        const double b = pts[idx + 1];
        if (b - a <= 1e-14) continue;
        const double mid = 0.5 * (a + b);
        const bool on_left = mid < 1.0;
        const double w_tw = on_left ? (2.0 - mid) : mid;
        if (w_tw >= strike) continue;

        double c0 = 0.0, c1 = 0.0;
        if (has_lower_strike && w_tw <= k_low) {
            c0 = strike - k_low;
        } else if (on_left) {
            c0 = strike - 2.0;
            c1 = 1.0;
        } else {
            c0 = strike;
            c1 = -1.0;
        }

        double d0 = 1.0, d1 = 0.0;
        if (has_active_ki) {
            if (std::abs(ki_spread) <= 1e-6) {
                if (w_tw >= ki_barrier) continue;
            } else {
                if (w_tw >= b_ki_hi) continue;
                if (w_tw > b_ki_lo) {
                    const double db = b_ki_hi - b_ki_lo;
                    d0 = on_left ? ((b_ki_hi - 2.0) / db) : (b_ki_hi / db);
                    d1 = on_left ? (1.0 / db) : (-1.0 / db);
                }
            }
        }

        double e0 = 1.0, e1 = 0.0;
        if (has_expiry_ko) {
            if (std::abs(ko_spread) <= 1e-6) {
                if (w_tw >= ko_barrier) continue;
            } else {
                if (w_tw >= b_ko_hi) continue;
                if (w_tw > b_ko_lo) {
                    const double db = b_ko_hi - b_ko_lo;
                    e0 = on_left ? ((b_ko_hi - 2.0) / db) : (b_ko_hi / db);
                    e1 = on_left ? (1.0 / db) : (-1.0 / db);
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
        lognormal_moments_interval(f, v, a, b, max_k, moms);
        total += q0 * moms[0] + q1 * moms[1] + q2 * moms[2] + q3 * moms[3];
    }
    return total;
}

double exact_twinwin_terminal_call(
    double f,
    double v,
    double k_call,
    bool has_expiry_ko,
    double ko_barrier,
    double ko_spread
) noexcept {
    if (!has_expiry_ko) {
        double total = 0.0;
        const double b_left = std::min(1.0, std::max(2.0 - k_call, 0.0));
        if (b_left > 1e-14) {
            double m[2] = {0.0, 0.0};
            lognormal_moments_interval(f, v, 0.0, b_left, 1, m);
            total += (2.0 - k_call) * m[0] - m[1];
        }
        if (k_call < 1.0) {
            double p_ge1 = 0.0;
            const double f_c = std::max(f, 1e-16);
            const double v_c = std::max(v, 1e-12);
            p_ge1 = ndtr_std((std::log(f_c) - 0.5 * v_c * v_c) / v_c);
            total += black76_call(f, 1.0, v) + (1.0 - k_call) * p_ge1;
        } else {
            total += black76_call(f, k_call, v);
        }
        return total;
    }

    const double b_ko_lo = std::min(ko_barrier, ko_barrier + ko_spread);
    const double b_ko_hi = std::max(ko_barrier, ko_barrier + ko_spread);
    if (k_call >= b_ko_hi || b_ko_hi <= 1.0) return 0.0;

    std::array<double, 8> pts{};
    std::size_t n_pts = 0;
    auto push_pt = [&](double x) {
        if (x >= 0.0 && x <= b_ko_hi && n_pts < pts.size()) {
            pts[n_pts++] = x;
        }
    };
    push_pt(0.0);
    push_pt(1.0);
    push_pt(b_ko_hi);
    push_pt(std::max(2.0 - b_ko_hi, 0.0));
    if (k_call > 1.0) {
        push_pt(k_call);
        if (2.0 - k_call > 0.0) push_pt(2.0 - k_call);
    }
    if (std::abs(ko_spread) > 1e-6) {
        push_pt(b_ko_lo);
        if (2.0 - b_ko_lo > 0.0) push_pt(2.0 - b_ko_lo);
    }
    for (std::size_t i = 1; i < n_pts; ++i) {
        const double key = pts[i];
        std::size_t j = i;
        while (j > 0 && pts[j - 1] > key) {
            pts[j] = pts[j - 1];
            --j;
        }
        pts[j] = key;
    }

    double total = 0.0;
    for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
        const double a = pts[idx];
        const double b = pts[idx + 1];
        if (b - a <= 1e-14) continue;
        const double mid = 0.5 * (a + b);
        const bool on_left = mid < 1.0;
        const double w_tw = on_left ? (2.0 - mid) : mid;
        if (w_tw <= k_call) continue;

        double e0 = 1.0, e1 = 0.0;
        if (std::abs(ko_spread) <= 1e-6) {
            if (w_tw >= ko_barrier) continue;
        } else {
            if (w_tw >= b_ko_hi) continue;
            if (w_tw > b_ko_lo) {
                const double db = b_ko_hi - b_ko_lo;
                e0 = on_left ? ((b_ko_hi - 2.0) / db) : (b_ko_hi / db);
                e1 = on_left ? (1.0 / db) : (-1.0 / db);
            }
        }
        const double c0 = on_left ? (2.0 - k_call) : (-k_call);
        const double c1 = on_left ? -1.0 : 1.0;
        const double q0 = c0 * e0;
        const double q1 = c0 * e1 + c1 * e0;
        const double q2 = c1 * e1;
        double moms[3] = {0.0, 0.0, 0.0};
        lognormal_moments_interval(f, v, a, b, (q2 != 0.0) ? 2 : 1, moms);
        total += q0 * moms[0] + q1 * moms[1] + q2 * moms[2];
    }
    return total;
}

inline double onestar_payoff_fn(
    double w,
    double strike,
    bool has_lower_strike,
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
    if (has_lower_strike) {
        pay = std::min(pay, std::max(strike - k_low, 0.0));
    }
    if (has_active_ki) {
        if (std::abs(ki_spread) <= 1e-6) {
            if (w >= ki_barrier) return 0.0;
        } else {
            const double blo = std::min(ki_barrier, ki_barrier + ki_spread);
            const double bhi = std::max(ki_barrier, ki_barrier + ki_spread);
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
    bool has_lower_strike,
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
    double& l_onestar_out,
    double& p_onestar_out
) noexcept {
    l_put_out = 0.0;
    l_onestar_out = 0.0;
    p_onestar_out = 0.0;
    const double k_low = has_lower_strike ? lower_strike : 0.0;
    const double log_ks = std::log(std::max(one_star_strike, 1e-16));

    if (d == 1) {
        constexpr double kOneStarEpsD1 = 0.02;
        const double f0 = std::max(f_c[0], 1e-16);
        const double v0 = std::max(v_c[0], 1e-12);
        const double os_lo = std::max(one_star_strike - 0.5 * kOneStarEpsD1, 0.0);
        const double os_hi = one_star_strike + 0.5 * kOneStarEpsD1;
        const double inv_os_eps = 1.0 / (os_hi - os_lo);

        if (one_star_strike > 0.0) {
            const double blo_ko = std::min(ko_b_term, ko_b_term + ko_spread);
            const double bhi_ko = std::max(ko_b_term, ko_b_term + ko_spread);
            const bool hard_ko = std::abs(ko_spread) <= 1e-6;
            const double ko_top = use_direct_ko ? (hard_ko ? ko_b_term : bhi_ko) : 1e30;
            if (os_lo < ko_top) {
                std::array<double, 6> pts{};
                std::size_t n_pts = 0;
                auto push_pt = [&](double x) {
                    if (x >= os_lo && x <= ko_top && n_pts < pts.size()) pts[n_pts++] = x;
                };
                push_pt(os_lo);
                if (os_hi < ko_top) push_pt(os_hi);
                if (use_direct_ko && !hard_ko && blo_ko > os_lo && blo_ko < ko_top) push_pt(blo_ko);
                push_pt(ko_top);
                for (std::size_t i = 1; i < n_pts; ++i) {
                    const double key = pts[i];
                    std::size_t j = i;
                    while (j > 0 && pts[j - 1] > key) {
                        pts[j] = pts[j - 1];
                        --j;
                    }
                    pts[j] = key;
                }
                double p_os = 0.0;
                for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
                    const double a = pts[idx];
                    const double b = pts[idx + 1];
                    if (b - a <= 1e-14) continue;
                    const double mid = (b > 1e20) ? (a + 1.0) : (0.5 * (a + b));
                    double s0 = 1.0, s1 = 0.0;
                    if (mid < os_hi) {
                        s0 = -os_lo * inv_os_eps;
                        s1 = inv_os_eps;
                    }
                    double e0 = 1.0, e1 = 0.0;
                    if (use_direct_ko && !hard_ko && mid > blo_ko) {
                        const double db = bhi_ko - blo_ko;
                        e0 = bhi_ko / db;
                        e1 = -1.0 / db;
                    }
                    const double q0 = s0 * e0;
                    const double q1 = s0 * e1 + s1 * e0;
                    const double q2 = s1 * e1;
                    double moms[3] = {0.0, 0.0, 0.0};
                    if (b > 1e20) {
                        const double m_u = std::log(f0) - 0.5 * v0 * v0;
                        moms[0] = ndtr_std(-(std::log(std::max(a, 1e-16)) - m_u) / v0);
                        p_os += q0 * moms[0];
                    } else {
                        lognormal_moments_interval(f0, v0, a, b, (q2 != 0.0) ? 2 : 1, moms);
                        p_os += q0 * moms[0] + q1 * moms[1] + q2 * moms[2];
                    }
                }
                p_onestar_out = (use_direct_ko ? surv_prev_term : surv_cur) * std::max(p_os, 0.0);
                l_onestar_out = p_onestar_out * one_star_coupon;
            }
        }
        if (strike > 1e-6 && option_gearing > 0.0 && one_star_strike > 1e-16) {
            const double surv_mult = use_direct_ko ? surv_prev_term : surv_cur;
            if (os_lo >= strike) {
                l_put_out = surv_mult * exact_piecewise_terminal_put(
                    f0, v0, false, strike, has_lower_strike, lower_strike,
                    has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread
                );
                return;
            }
            const double hi = std::min(strike, os_hi);
            std::array<double, 12> pts{};
            std::size_t n_pts = 0;
            auto push_pt = [&](double x) {
                if (x >= 0.0 && x <= hi && n_pts < pts.size()) pts[n_pts++] = x;
            };
            push_pt(0.0);
            push_pt(hi);
            if (has_lower_strike && k_low > 0.0) push_pt(k_low);
            if (os_lo > 0.0) push_pt(os_lo);
            if (has_active_ki) {
                if (std::abs(ki_spread) <= 1e-6) push_pt(ki_barrier);
                else {
                    push_pt(std::min(ki_barrier, ki_barrier + ki_spread));
                    push_pt(std::max(ki_barrier, ki_barrier + ki_spread));
                }
            }
            if (use_direct_ko) {
                if (std::abs(ko_spread) <= 1e-6) push_pt(ko_b_term);
                else {
                    push_pt(std::min(ko_b_term, ko_b_term + ko_spread));
                    push_pt(std::max(ko_b_term, ko_b_term + ko_spread));
                }
            }
            for (std::size_t i = 1; i < n_pts; ++i) {
                const double key = pts[i];
                std::size_t j = i;
                while (j > 0 && pts[j - 1] > key) {
                    pts[j] = pts[j - 1];
                    --j;
                }
                pts[j] = key;
            }
            double total = 0.0;
            for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
                const double a = pts[idx];
                const double b = pts[idx + 1];
                if (b - a <= 1e-14) continue;
                const double mid = 0.5 * (a + b);
                const double c0 = (has_lower_strike && mid <= k_low) ? (strike - k_low) : strike;
                const double c1 = (has_lower_strike && mid <= k_low) ? 0.0 : -1.0;

                double d0 = 1.0, d1 = 0.0;
                if (has_active_ki) {
                    if (std::abs(ki_spread) <= 1e-6) {
                        if (mid >= ki_barrier) continue;
                    } else {
                        const double blo = std::min(ki_barrier, ki_barrier + ki_spread);
                        const double bhi = std::max(ki_barrier, ki_barrier + ki_spread);
                        if (mid >= bhi) continue;
                        if (mid > blo) { d0 = bhi / (bhi - blo); d1 = -1.0 / (bhi - blo); }
                    }
                }
                double e0 = 1.0, e1 = 0.0;
                if (use_direct_ko) {
                    if (std::abs(ko_spread) <= 1e-6) {
                        if (mid >= ko_b_term) continue;
                    } else {
                        const double blo = std::min(ko_b_term, ko_b_term + ko_spread);
                        const double bhi = std::max(ko_b_term, ko_b_term + ko_spread);
                        if (mid >= bhi) continue;
                        if (mid > blo) { e0 = bhi / (bhi - blo); e1 = -1.0 / (bhi - blo); }
                    }
                }
                double r0 = 1.0, r1 = 0.0;
                if (mid > os_lo) {
                    r0 = os_hi * inv_os_eps;
                    r1 = -inv_os_eps;
                }
                const double cd0 = c0 * d0;
                const double cd1 = c0 * d1 + c1 * d0;
                const double cd2 = c1 * d1;
                const double cde0 = cd0 * e0;
                const double cde1 = cd0 * e1 + cd1 * e0;
                const double cde2 = cd1 * e1 + cd2 * e0;
                const double cde3 = cd2 * e1;
                const double q0 = cde0 * r0;
                const double q1 = cde0 * r1 + cde1 * r0;
                const double q2 = cde1 * r1 + cde2 * r0;
                const double q3 = cde2 * r1 + cde3 * r0;
                const double q4 = cde3 * r1;
                const int max_k = (q4 != 0.0) ? 4 : ((q3 != 0.0) ? 3 : ((q2 != 0.0) ? 2 : 1));
                double moms[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
                lognormal_moments_interval(f0, v0, a, b, max_k, moms);
                total += q0 * moms[0] + q1 * moms[1] + q2 * moms[2] + q3 * moms[3] + q4 * moms[4];
            }
            l_put_out = surv_mult * std::max(total, 0.0);
        }
        return;
    }

    static constexpr int kStackD = 32;
    std::array<double, kStackD> b_stk{}, se_stk{}, mu_stk{}, vb_stk{}, vs_stk{};
    std::vector<double> dyn_hp;
    double* b = b_stk.data();
    double* s_eps = se_stk.data();
    double* mu = mu_stk.data();
    double* vb = vb_stk.data();
    double* vs = vs_stk.data();
    if (d > kStackD) [[unlikely]] {
        dyn_hp.resize(5 * static_cast<std::size_t>(d));
        b = dyn_hp.data();
        s_eps = b + d;
        mu = s_eps + d;
        vb = mu + d;
        vs = vb + d;
    }
    fit_1factor_loadings(d, corr, b);

    for (int i = 0; i < d; ++i) {
        const double fi = std::max(f_c[i], 1e-16);
        const double vi = std::max(v_c[i], 1e-12);
        s_eps[i] = std::sqrt(std::max(1.0 - b[i] * b[i], 1e-12));
        mu[i] = std::log(fi) - 0.5 * vi * vi;
        vb[i] = vi * b[i];
        vs[i] = std::max(vi * s_eps[i], 1e-12);
    }

    std::array<double, 16> p_max_lt_ks_y{};
    for (std::size_t q = 0; q < 16; ++q) {
        const double y = kGL16YNorm[q];
        double prod = 1.0;
        for (int i = 0; i < d; ++i) {
            prod *= ndtr_std((log_ks - mu[i] - vb[i] * y) / vs[i]);
        }
        p_max_lt_ks_y[q] = prod;
    }

    if (one_star_strike > 0.0) {
        double os_acc = 0.0;
        if (use_direct_ko) {
            std::array<double, 4> ko_pts{};
            std::size_t n_ko = 1;
            if (std::abs(ko_spread) <= 1e-6) {
                ko_pts[0] = ko_b_term;
            } else {
                const double blo = std::min(ko_b_term, ko_b_term + ko_spread);
                const double bhi = std::max(ko_b_term, ko_b_term + ko_spread);
                const double mid = 0.5 * (blo + bhi);
                const double hw = 0.5 * (bhi - blo);
                static constexpr std::array<double, 4> gx4 = {-0.86113631159405258, -0.33998104358485626, 0.33998104358485626, 0.86113631159405258};
                for (std::size_t r = 0; r < 4; ++r) ko_pts[r] = std::max(mid + hw * gx4[r], 1e-16);
                n_ko = 4;
            }
            static constexpr std::array<double, 4> gw4 = {0.34785484513745386, 0.65214515486254614, 0.65214515486254614, 0.34785484513745386};
            for (std::size_t q = 0; q < 16; ++q) {
                const double y = kGL16YNorm[q];
                const double p_mks = p_max_lt_ks_y[q];
                double p_sub = 0.0;
                for (std::size_t r = 0; r < n_ko; ++r) {
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
            p_onestar_out = surv_prev_term * os_acc;
        } else {
            for (std::size_t q = 0; q < 16; ++q) {
                os_acc += 0.5 * kGL16W[q] * (1.0 - p_max_lt_ks_y[q]);
            }
            p_onestar_out = surv_cur * os_acc;
        }
        l_onestar_out = p_onestar_out * one_star_coupon;
    }

    if (strike > 1e-6 && option_gearing > 0.0 && (!has_lower_strike || strike > k_low)) {
        auto eval_cum_joint = [&](double x) noexcept -> double {
            if (x <= 0.0) return 0.0;
            const double log_x = std::log(x);
            double res = 0.0;
            for (std::size_t q = 0; q < 16; ++q) {
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
        std::size_t n_pts = 0;
        auto push_pt = [&](double x) {
            if (x >= 0.0 && x <= strike && n_pts < pts.size()) pts[n_pts++] = x;
        };
        push_pt(0.0);
        push_pt(strike);
        if (has_lower_strike && k_low > 0.0 && k_low < strike) push_pt(k_low);
        if (has_active_ki) {
            if (std::abs(ki_spread) <= 1e-6) {
                push_pt(ki_barrier);
            } else {
                push_pt(std::min(ki_barrier, ki_barrier + ki_spread));
                push_pt(std::max(ki_barrier, ki_barrier + ki_spread));
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
        for (std::size_t i = 1; i < n_pts; ++i) {
            const double key = pts[i];
            std::size_t j = i;
            while (j > 0 && pts[j - 1] > key) {
                pts[j] = pts[j - 1];
                --j;
            }
            pts[j] = key;
        }

        double put_acc = 0.0;
        for (std::size_t idx = 0; idx + 1 < n_pts; ++idx) {
            const double a = pts[idx];
            const double b_pt = pts[idx + 1];
            const double span = b_pt - a;
            if (span <= 1e-12) continue;
            const double mid_x = 0.5 * (a + b_pt);
            const double hw_x = 0.5 * span;
            const double next_span = (idx + 2 < n_pts) ? (pts[idx + 2] - b_pt) : 1.0;
            const double ga = onestar_payoff_fn(a + 1e-10 * span, strike, has_lower_strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            const double gb_m = onestar_payoff_fn(b_pt - 1e-10 * span, strike, has_lower_strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            const double probe_right = b_pt + std::min(1e-10, 0.25 * std::max(next_span, 0.0));
            const double gb_p = onestar_payoff_fn(probe_right, strike, has_lower_strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
            if (ga > 0.0 || gb_m > 0.0) {
                const bool linear_unit = !has_active_ki && !use_direct_ko && (!has_lower_strike || a >= k_low - 1e-12);
                for (std::size_t r = 0; r < 16; ++r) {
                    const double xr = mid_x + hw_x * kGL16X[r];
                    double neg_gp = 1.0;
                    if (!linear_unit) {
                        const double h_fd = 1e-7 * std::max(span, 1e-3);
                        const double g_p = onestar_payoff_fn(std::min(xr + h_fd, b_pt - 1e-12 * span), strike, has_lower_strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
                        const double g_m = onestar_payoff_fn(std::max(xr - h_fd, a + 1e-12 * span), strike, has_lower_strike, k_low, has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_b_term, ko_spread);
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

void tier1_clamped_cubic_coeffs_impl(
    int64_t k,
    const double* x,
    const double* y,
    double left_slope,
    double right_slope,
    double* coeffs_out,
    double* h,
    double* s,
    double* d,
    double* c_star,
    double* d_star
) noexcept {
    const int64_t n_seg = k - 1;
    for (int64_t i = 0; i < n_seg; ++i) {
        h[i] = x[i + 1] - x[i];
        s[i] = (y[i + 1] - y[i]) / h[i];
    }
    d[0] = left_slope;
    d[k - 1] = right_slope;

    const int64_t n_int = k - 2;
    if (n_int > 0) {
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

[[gnu::noinline]] void tier1_clamped_cubic_coeffs_dyn(
    int64_t k,
    const double* x,
    const double* y,
    double left_slope,
    double right_slope,
    double* coeffs_out
) noexcept {
    const auto n_seg = static_cast<std::size_t>(k - 1);
    const auto n_int = static_cast<std::size_t>(std::max<int64_t>(k - 2, 1));
    std::vector<double> h(n_seg), s(n_seg), d(static_cast<std::size_t>(k), 0.0), c_star(n_int, 0.0), d_star(n_int, 0.0);
    tier1_clamped_cubic_coeffs_impl(
        k, x, y, left_slope, right_slope, coeffs_out,
        h.data(), s.data(), d.data(), c_star.data(), d_star.data()
    );
}

void tier1_clamped_cubic_coeffs_1d(
    int64_t k,
    const double* x,
    const double* y,
    double left_slope,
    double right_slope,
    double* coeffs_out
) {
    if (k < 2 || x == nullptr || y == nullptr || coeffs_out == nullptr) {
        return;
    }
    static constexpr int64_t kStackCubic = 64;
    if (k <= kStackCubic) {
        std::array<double, kStackCubic> h{}, s{}, d{}, c_star{}, d_star{};
        tier1_clamped_cubic_coeffs_impl(
            k, x, y, left_slope, right_slope, coeffs_out,
            h.data(), s.data(), d.data(), c_star.data(), d_star.data()
        );
        return;
    }
    tier1_clamped_cubic_coeffs_dyn(k, x, y, left_slope, right_slope, coeffs_out);
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
    if (k < 2 || n_q <= 0 || x == nullptr || coeffs == nullptr || x_query == nullptr ||
        val_out == nullptr || dval_out == nullptr || d2val_out == nullptr) {
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
        if (std::isnan(xq)) {
            val_out[q] = xq;
            dval_out[q] = xq;
            d2val_out[q] = xq;
            continue;
        }
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
    if (scalars_out == nullptr) {
        return;
    }
    if (m_rem <= 0 || d_assets <= 0 ||
        fwd_perf == nullptr || vol_perf == nullptr || corr == nullptr ||
        df_pay == nullptr || ko_barriers == nullptr || is_callable == nullptr ||
        cpn_barriers == nullptr || is_guaranteed_cpn == nullptr || coupon_rates == nullptr ||
        f_w_out == nullptr || v_w_out == nullptr || delta_weights_out == nullptr ||
        p_stop_out == nullptr || surv_out == nullptr || h_rate_out == nullptr || q_cpn_out == nullptr) {
        for (int i = 0; i < 8; ++i) scalars_out[i] = 0.0;
        scalars_out[4] = pv_in_transit;
        return;
    }

    const int d = static_cast<int>(d_assets);
    const int m_tot = static_cast<int>(m_rem);
    if (is_long_basket != 0 && opt_index_code == 0) {
        opt_index_code = 2;
    }
    const bool call_is_normal = is_diff_index(call_index_code);
    const bool cpn_is_normal = is_diff_index(cpn_index_code);
    const bool opt_is_normal = is_diff_index(opt_index_code);
    const bool cpn_differs = (cpn_index_code != call_index_code);
    const bool opt_differs = (opt_index_code != call_index_code);

    static constexpr int kStackD = static_cast<int>(Tier1StackArena::kMaxAssets);
    Tier1StackArena arena;
    std::vector<double> dyn_pool;
    double* f_col = arena.f_col.data();
    double* v_col = arena.v_col.data();
    double* w_col = arena.w_col.data();
    double* f_next = arena.f_next.data();
    double* v_next = arena.v_next.data();
    double* w_next = arena.w_next.data();
    double* w_call_prev = arena.w_col.data();
    double* w_call_cur = arena.w_col.data();
    double* f_call_prev = arena.f_col.data();
    double* v_call_prev = arena.v_col.data();
    double* f_cpn_m = arena.f_cpn.data();
    double* v_cpn_m = arena.v_cpn.data();
    double* w_cpn_m = arena.w_cpn.data();
    double* f_opt_t = arena.f_opt.data();
    double* v_opt_t = arena.v_opt.data();
    double* w_opt_t = arena.w_opt.data();
    if (d > kStackD) [[unlikely]] {
        dyn_pool.assign(12 * static_cast<std::size_t>(d), 0.0);
        double* base_p = dyn_pool.data();
        f_col = base_p;
        v_col = f_col + d;
        w_col = v_col + d;
        f_next = w_col + d;
        v_next = f_next + d;
        w_next = v_next + d;
        f_cpn_m = w_next + d;
        v_cpn_m = f_cpn_m + d;
        w_cpn_m = v_cpn_m + d;
        f_opt_t = w_cpn_m + d;
        v_opt_t = f_opt_t + d;
        w_opt_t = v_opt_t + d;
        w_call_prev = w_col;
        w_call_cur = w_col;
        f_call_prev = f_col;
        v_call_prev = v_col;
    }

    auto load_period = [&](int m, double* f_dst, double* v_dst) {
        for (int i = 0; i < d; ++i) {
            f_dst[i] = fwd_perf[i * m_tot + m];
            v_dst[i] = vol_perf[i * m_tot + m];
        }
    };

    bool has_near_tie = false;
    if (m_tot * d < 64 && m_tot >= 8 && d >= 2) {
        for (int m = 0; m < m_tot && !has_near_tie; ++m) {
            for (int i = 0; i < d && !has_near_tie; ++i) {
                const double fi = std::max(fwd_perf[i * m_tot + m], 1e-16);
                for (int j = i + 1; j < d; ++j) {
                    const double fj = std::max(fwd_perf[j * m_tot + m], 1e-16);
                    if (std::abs(std::log(fi / fj)) < 8.0 * kTieTau) {
                        has_near_tie = true;
                        break;
                    }
                }
            }
        }
    }

    {
        OmpCallGuard omp_guard(m_tot * d >= 64 || (m_tot >= 8 && has_near_tie));
        #pragma omp parallel if(omp_guard.active) num_threads(omp_guard.n_threads)
        {
            alignas(64) std::array<double, kStackD> fm_s, vm_s, wm_s;
            std::vector<double> fm_dyn;
            double* fm = fm_s.data();
            double* vm = vm_s.data();
            double* wm = wm_s.data();
            if (d > kStackD) [[unlikely]] {
                fm_dyn.assign(3 * static_cast<std::size_t>(d), 0.0);
                fm = fm_dyn.data();
                vm = fm + d;
                wm = vm + d;
            }
            #pragma omp for schedule(static)
            for (int m = 0; m < m_tot; ++m) {
                for (int i = 0; i < d; ++i) {
                    fm[i] = fwd_perf[i * m_tot + m];
                    vm[i] = vol_perf[i * m_tot + m];
                }
                double fw_m = 0.0, vw_m = 0.0;
                eval_index_moments(d, fm, vm, corr, inv_lex_perm, call_index_code, fw_m, vw_m, wm);
                f_w_out[m] = fw_m;
                v_w_out[m] = vw_m;
                for (int i = 0; i < d; ++i) {
                    delta_weights_out[i * m_tot + m] = wm[i];
                }
            }
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
        double h_raw_m = 0.0;
        double h_m = 0.0;
        if (is_call_m) {
            if (call_index_code == 14) {
                h_raw_m = eval_coupon_prob_dist(f_cond, v_cond, ko_b_m, ko_spread, false, false, false, 14);
            } else if (call_is_normal) {
                h_raw_m = digital_call_spread_normal(f_cond, v_cond, ko_b_m, ko_spread);
            } else {
                h_raw_m = digital_call_spread_lognormal(f_cond, v_cond, ko_b_m, ko_spread);
            }
            h_m = std::clamp(h_raw_m, 0.0, 1.0 - 1e-7);
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
            double rho_ck_cond = 1.0;
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
                rho_ck_cond = std::clamp(rho_cc * z_std / z_sc, -0.999999, 0.999999);
                if (cpn_is_normal) {
                    f_cpn_cond = fc_raw + vc_raw * z_mc;
                    v_cpn_cond = std::max(vc_raw * z_sc, 1e-12);
                } else {
                    const double fc_pos = std::max(fc_raw, 1e-16);
                    f_cpn_cond = fc_pos * std::exp(vc_raw * z_mc - 0.5 * vc_raw * vc_raw * (1.0 - z_vc));
                    v_cpn_cond = std::max(vc_raw * z_sc, 1e-12);
                }
            }
            const double q_base = std::clamp(
                eval_coupon_prob_dist(
                    f_cpn_cond, v_cpn_cond, cpn_barriers[m], cpn_spread, cpn_is_normal,
                    cpn_dir_down != 0, cpn_type_out != 0, cpn_index_code
                ),
                0.0, 1.0
            );
            q_m = q_base;
            if (force_coupons_on_autocall != 0 && is_call_m) {
                const double h_prob = std::clamp(h_raw_m, 0.0, 1.0);
                if (h_prob >= 1.0 - 1e-12 || q_base >= 1.0 - 1e-12) {
                    q_m = 1.0;
                } else if (h_prob <= 1e-12) {
                    q_m = q_base;
                } else if (q_base <= 1e-12) {
                    q_m = h_prob;
                } else {
                    const double s_c = ((cpn_dir_down != 0) != (cpn_type_out != 0)) ? -1.0 : 1.0;
                    const double z_c = ndtri_std(q_base);
                    const double z_k = ndtri_std(h_prob);
                    const double p_ck = bvn_cdf_genz(z_c, z_k, s_c * rho_ck_cond);
                    q_m = std::clamp(
                        q_base + h_prob - p_ck,
                        std::max(q_base, h_prob),
                        std::min(q_base + h_prob, 1.0)
                    );
                }
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
                if (call_index_code == 14) {
                    ramp_survival_post_moments_twinwin_lognormal(f_cond, v_cond, ko_b_m, ko_spread, mu_u, var_u);
                } else if (call_is_normal) {
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
                    const double vi_c = std::max(v_call_prev[i], 1e-12);
                    for (int j = 0; j < d; ++j) {
                        const double wj_n = w_next[j] * (call_is_normal ? f_next[j] : 1.0);
                        const double vj_c = std::max(v_call_prev[j], 1e-12);
                        const double vj_n = std::max(v_next[j], 1e-12);
                        const double rho_ij = corr[i * d + j];
                        num += wi_c * wj_n * rho_ij * vi_c * std::min(vj_c, vj_n);
                    }
                }
                for (int i = 0; i < d; ++i) {
                    const double wi_c = w_call_prev[i] * (call_is_normal ? f_call_prev[i] : 1.0);
                    const double wi_n = w_next[i] * (call_is_normal ? f_next[i] : 1.0);
                    const double vi_c = std::max(v_call_prev[i], 1e-12);
                    const double vi_n = std::max(v_next[i], 1e-12);
                    for (int j = 0; j < d; ++j) {
                        const double wj_c = w_call_prev[j] * (call_is_normal ? f_call_prev[j] : 1.0);
                        const double wj_n = w_next[j] * (call_is_normal ? f_next[j] : 1.0);
                        const double vj_c = std::max(v_call_prev[j], 1e-12);
                        const double vj_n = std::max(v_next[j], 1e-12);
                        const double rho_ij = corr[i * d + j];
                        den_cur += wi_c * wj_c * rho_ij * vi_c * vj_c;
                        den_nxt += wi_n * wj_n * rho_ij * vi_n * vj_n;
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

    const bool expiry_ko_active = (allow_autocall_on_expiry != 0) && (is_callable[m_tot - 1] > 0.5);

    double f_opt_cond = f_cond_term;
    double v_opt_cond = v_cond_term;
    if (opt_differs) {
        load_period(m_tot - 1, f_opt_t, v_opt_t);
        for (int i = 0; i < d; ++i) w_call_cur[i] = delta_weights_out[i * m_tot + (m_tot - 1)];
        double fo_raw = 0.0, vo_raw = 0.0;
        eval_index_moments(d, f_opt_t, v_opt_t, corr, inv_lex_perm, opt_index_code, fo_raw, vo_raw, w_opt_t);
        const double rho_call_opt = compute_cross_index_corr(
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
    const bool has_lower_strike = std::isfinite(lower_strike) && (lower_strike > 0.0);
    const double ko_b_term = ko_barriers[m_tot - 1];
    const bool use_direct_ko = expiry_ko_active && !opt_differs;
    const double surv_mult = use_direct_ko ? surv_prev_term : surv_cur;

    double l_put_val = 0.0;
    double l_onestar_val = 0.0;
    double p_onestar_val = 0.0;
    if (has_onestar) {
        load_period(m_tot - 1, f_opt_t, v_opt_t);
        const double* f_os_in = f_opt_t;
        const double* v_os_in = v_opt_t;
        if (z_mean_term != 0.0 || z_std_term != 1.0) {
            for (int i = 0; i < d; ++i) w_call_cur[i] = delta_weights_out[i * m_tot + (m_tot - 1)];
            double vc_n2 = 0.0;
            for (int i = 0; i < d; ++i) {
                const double wi = w_call_cur[i] * (call_is_normal ? f_opt_t[i] : 1.0);
                const double vi = std::max(v_opt_t[i], 1e-12);
                for (int j = 0; j < d; ++j) {
                    const double wj = w_call_cur[j] * (call_is_normal ? f_opt_t[j] : 1.0);
                    const double vj = std::max(v_opt_t[j], 1e-12);
                    vc_n2 += wi * wj * corr[i * d + j] * vi * vj;
                }
            }
            const double vc_n = std::sqrt(std::max(vc_n2, 1e-30));
            for (int i = 0; i < d; ++i) {
                const double vi = std::max(v_opt_t[i], 1e-12);
                double cov_i = 0.0;
                for (int j = 0; j < d; ++j) {
                    const double wj = w_call_cur[j] * (call_is_normal ? f_opt_t[j] : 1.0);
                    const double vj = std::max(v_opt_t[j], 1e-12);
                    cov_i += corr[i * d + j] * vi * vj * wj;
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
        eval_1factor_onestar(
            d, f_os_in, v_os_in, corr,
            strike, has_lower_strike, lower_strike,
            has_active_ki, ki_barrier, ki_spread,
            use_direct_ko, ko_b_term, ko_spread,
            one_star_strike, one_star_coupon,
            surv_cur, surv_prev_term, option_gearing,
            l_put_val, l_onestar_val, p_onestar_val
        );
    } else if (option_gearing > 0.0 && (opt_is_normal ? (!has_lower_strike || strike > lower_strike) : (strike > 1e-6 && (!has_lower_strike || strike > lower_strike)))) {
        double put_cond = 0.0;
        if (opt_index_code == 14) {
            put_cond = exact_twinwin_terminal_put(
                f_opt_cond, v_opt_cond,
                strike, has_lower_strike, lower_strike,
                has_active_ki, ki_barrier, ki_spread,
                use_direct_ko, ko_b_term, ko_spread
            );
        } else {
            put_cond = exact_piecewise_terminal_put(
                f_opt_cond, v_opt_cond, opt_is_normal,
                strike, has_lower_strike, lower_strike,
                has_active_ki, ki_barrier, ki_spread,
                use_direct_ko, ko_b_term, ko_spread
            );
        }
        l_put_val = surv_mult * put_cond;
    }

    double l_call_val = 0.0;
    const bool has_part_call = std::isfinite(part_call_strike) && (part_call_rate != 0.0) &&
                               (opt_is_normal || part_call_strike >= 0.0);
    if (has_part_call) {
        if (opt_index_code == 14) {
            l_call_val = surv_mult * exact_twinwin_terminal_call(
                f_opt_cond, v_opt_cond,
                part_call_strike, use_direct_ko, ko_b_term, ko_spread
            );
        } else {
            l_call_val = surv_mult * exact_piecewise_terminal_call(
                f_opt_cond, v_opt_cond, opt_is_normal,
                part_call_strike, use_direct_ko, ko_b_term, ko_spread
            );
        }
    }

    const double opt_pv_val = df_pay[m_tot - 1] * (
        -option_gearing * l_put_val
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
    scalars_out[7] = p_onestar_val;
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
    if (f_out == nullptr || v_out == nullptr) return;
    if (d_assets <= 0 || f_in == nullptr || v_in == nullptr || corr == nullptr || w_out == nullptr) {
        *f_out = 0.0;
        *v_out = 1e-12;
        return;
    }
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

static_assert(sizeof(Tier1AutocallArgs) == 336, "ABI mismatch with =2q11Q15d12i8Q");
static_assert(alignof(Tier1AutocallArgs) == 8, "Expected 8-byte natural alignment");
static_assert(offsetof(Tier1AutocallArgs, strike) == 104, "Scalar block must start at 104B");
static_assert(offsetof(Tier1AutocallArgs, cbt_out) == 224, "Flag block must start at 224B");
static_assert(offsetof(Tier1AutocallArgs, f_w_out) == 272, "Output pointer block must start at 272B");

Tier1AbiInfo tier1_abi_info() noexcept {
    return Tier1AbiInfo{
        kTier1AbiVersion,
        static_cast<std::uint32_t>(sizeof(Tier1AutocallArgs)),
        static_cast<std::uint32_t>(alignof(Tier1AutocallArgs)),
        static_cast<std::uint32_t>(sizeof(CrnLooL2oBatchArgs)),
        static_cast<std::uint32_t>(alignof(CrnLooL2oBatchArgs)),
    };
}

Tier1AbiInfoV2 tier1_abi_info_v2() noexcept {
    return Tier1AbiInfoV2{
        kTier1AbiVersionV2,
        static_cast<std::uint32_t>(sizeof(Tier1AutocallArgs)),
        static_cast<std::uint32_t>(alignof(Tier1AutocallArgs)),
        static_cast<std::uint32_t>(sizeof(CrnLooL2oBatchArgs)),
        static_cast<std::uint32_t>(alignof(CrnLooL2oBatchArgs)),
        static_cast<std::uint32_t>(sizeof(PlatenSdeAadArgs)),
        static_cast<std::uint32_t>(alignof(PlatenSdeAadArgs)),
        static_cast<std::uint32_t>(sizeof(SobolGenArgs)),
        static_cast<std::uint32_t>(alignof(SobolGenArgs)),
        static_cast<std::uint32_t>(sizeof(VibratoGammaArgs)),
        static_cast<std::uint32_t>(alignof(VibratoGammaArgs)),
        0u,
    };
}

void tier1_eval_autocall_struct_cpu(const Tier1AutocallArgs* a) noexcept {
    if (a == nullptr) {
        return;
    }
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

void crn_step_1d_trajectories_cpp(
    int B, int lv_nt, int lv_ny, int M, int n_sub, int R, int N_rep, int z_mode,
    double lv_y_min, double lv_y_max,
    const double* lv_tables,
    const double* z_flat,
    const double* tau_obs,
    const double* r_native_obs,
    const double* r_tot_obs,
    const double* lv_t_grid,
    double* out_norm_x
) noexcept {
    if (B <= 0 || lv_nt <= 0 || lv_ny <= 0 || M <= 0 || n_sub <= 0 || R <= 0 || N_rep <= 0 ||
        lv_tables == nullptr || z_flat == nullptr || tau_obs == nullptr ||
        r_native_obs == nullptr || r_tot_obs == nullptr || lv_t_grid == nullptr || out_norm_x == nullptr) {
        return;
    }
    const std::size_t n_rep_sz = static_cast<std::size_t>(N_rep);
    const std::size_t n_tot_sz = static_cast<std::size_t>(R) * n_rep_sz;
    const std::size_t lv_ny_sz = static_cast<std::size_t>(lv_ny);
    const std::size_t lv_plane_sz = static_cast<std::size_t>(lv_nt) * lv_ny_sz;
    const double t_max = lv_t_grid[lv_nt - 1];
    const double inv_dt_grid = (lv_nt > 1) ? (static_cast<double>(lv_nt - 1) / std::max(t_max, 1e-12)) : 0.0;
    const double inv_dy_grid = (lv_ny > 1) ? (static_cast<double>(lv_ny - 1) / std::max(lv_y_max - lv_y_min, 1e-12)) : 0.0;
    const double max_u_t = std::max(0.0, static_cast<double>(lv_nt - 1) - 1e-6);
    const double max_u_y = std::max(0.0, static_cast<double>(lv_ny - 1) - 1e-6);

    OmpCallGuard omp_guard(B * R >= 4);
    #pragma omp parallel if(omp_guard.active) num_threads(omp_guard.n_threads)
    {
        CrnThreadPoolArena arena{};
        CrnDynamicArenaFallback dyn{};
        const bool big_rep = n_rep_sz > CrnThreadPoolArena::kMaxRep;
        const bool big_ny = lv_ny_sz > CrnThreadPoolArena::kMaxLvY;
        if (big_rep || big_ny) [[unlikely]] {
            dyn.ensure_capacity(n_rep_sz, lv_ny_sz, 0);
        }
        double* log_x = big_rep ? dyn.log_x.data() : arena.log_x.data();
        double* ex_buf = big_rep ? dyn.ex_buf.data() : arena.ex_buf.data();
        double* lv_slice_start = big_ny ? dyn.lv_slice.data() : arena.lv_slice.data();
        std::vector<double> lv_slice_end_vec(lv_ny_sz + 4, 0.0);
        double* lv_slice_end = lv_slice_end_vec.data();

        auto fill_lv_slice = [&](const double* lv_b, double t_val, double* dst_slice) noexcept {
            const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
            const int it0 = static_cast<int>(u_t);
            const int it1 = std::min(it0 + 1, lv_nt - 1);
            const double wt = (lv_nt > 1) ? (u_t - static_cast<double>(it0)) : 0.0;
            const double* row0 = lv_b + static_cast<std::size_t>(it0) * lv_ny_sz;
            const double* row1 = lv_b + static_cast<std::size_t>(it1) * lv_ny_sz;
            #pragma omp simd
            for (int iy = 0; iy < lv_ny; ++iy) {
                dst_slice[iy] = (1.0 - wt) * row0[iy] + wt * row1[iy];
            }
            const double top_sig = dst_slice[lv_ny - 1];
            for (int pad = 0; pad < 4; ++pad) {
                dst_slice[lv_ny + pad] = top_sig;
            }
        };

        #pragma omp for collapse(2) schedule(static)
        for (int b = 0; b < B; ++b) {
            for (int r = 0; r < R; ++r) {
                const double* lv_b = lv_tables + static_cast<std::size_t>(z_mode == 1 ? 0 : b) * lv_plane_sz;
                const std::size_t rep_off = static_cast<std::size_t>(r) * n_rep_sz;

                #pragma omp simd
                for (int n = 0; n < N_rep; ++n) {
                    log_x[n] = 0.0;
                }

                double t_prev = 0.0;
                double rt_prev = 0.0;
                double rn_prev = 0.0;

                for (int m = 0; m < M; ++m) {
                    const double t_cur = tau_obs[m];
                    const double rt_cur = r_tot_obs[m];
                    const double rn_cur = r_native_obs[m];
                    const double d_tau = std::max(t_cur - t_prev, 1e-10);
                    const double dt = d_tau / static_cast<double>(n_sub);
                    const double sqrt_dt = std::sqrt(dt);
                    const double drt = rt_cur - rt_prev;
                    const double drn = rn_cur - rn_prev;

                    for (int s = 0; s < n_sub; ++s) {
                        const double alpha_s = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
                        const double t_mid = t_prev + alpha_s * d_tau;
                        const double t_start = std::max(t_mid - 0.5 * dt, 0.0);
                        const double t_end = t_start + dt;
                        const double q_shift_mid = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn);

                        fill_lv_slice(lv_b, t_start, lv_slice_start);
                        fill_lv_slice(lv_b, t_end, lv_slice_end);

                        const std::size_t z_step = (z_mode == 1)
                            ? ((static_cast<std::size_t>(b) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m)) * static_cast<std::size_t>(n_sub) + static_cast<std::size_t>(s)) * n_tot_sz + rep_off
                            : (static_cast<std::size_t>(m) * static_cast<std::size_t>(n_sub) + static_cast<std::size_t>(s)) * n_tot_sz + rep_off;
                        const double* z_ptr = z_flat + z_step;
                        const bool has_q = std::abs(q_shift_mid) > 1e-15;

                        int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                        const __m256d v_y_min = _mm256_set1_pd(lv_y_min);
                        const __m256d v_inv_dy = _mm256_set1_pd(inv_dy_grid);
                        const __m256d v_zero = _mm256_setzero_pd();
                        const __m256d v_max_uy = _mm256_set1_pd(max_u_y);
                        const __m256d v_q_mid = _mm256_set1_pd(has_q ? q_shift_mid : 0.0);
                        const __m256d v_dt = _mm256_set1_pd(dt);
                        const __m256d v_sqrt_dt = _mm256_set1_pd(sqrt_dt);
                        const __m256d v_half = _mm256_set1_pd(0.5);
                        const __m256d v_neg_half = _mm256_set1_pd(-0.5);
                        const __m256d v_quarter = _mm256_set1_pd(0.25);
                        const __m256d v_two = _mm256_set1_pd(2.0);
                        const __m256d v_one = _mm256_set1_pd(1.0);

                        auto lookup_slice_pd = [&](const double* slice_ptr, __m256d y_v) noexcept -> __m256d {
                            __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                            __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                            __m256d u_floor = _mm256_floor_pd(u_y);
                            __m128i iy0 = _mm256_cvttpd_epi32(u_floor);
                            __m256d wy = _mm256_sub_pd(u_y, u_floor);
                            __m256d s0 = _mm256_i32gather_pd(slice_ptr, iy0, 8);
                            __m256d s1 = _mm256_i32gather_pd(slice_ptr + 1, iy0, 8);
                            return _mm256_fmadd_pd(wy, _mm256_sub_pd(s1, s0), s0);
                        };

                        for (; n + 4 <= N_rep; n += 4) {
                            __m256d lx = _mm256_loadu_pd(log_x + n);
                            __m256d zd = _mm256_loadu_pd(z_ptr + n);
                            __m256d y0 = has_q ? _mm256_add_pd(lx, v_q_mid) : lx;
                            __m256d sig0 = lookup_slice_pd(lv_slice_start, y0);
                            __m256d a0 = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sig0, sig0));
                            __m256d dw = _mm256_mul_pd(v_sqrt_dt, zd);
                            __m256d sig0_sdt = _mm256_mul_pd(sig0, v_sqrt_dt);
                            __m256d base_y = _mm256_fmadd_pd(a0, v_dt, y0);
                            __m256d sigh = lookup_slice_pd(lv_slice_end, _mm256_fmadd_pd(sig0, dw, base_y));
                            __m256d sigp = lookup_slice_pd(lv_slice_end, _mm256_add_pd(base_y, sig0_sdt));
                            __m256d sigm = lookup_slice_pd(lv_slice_end, _mm256_sub_pd(base_y, sig0_sdt));
                            __m256d ah = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sigh, sigh));
                            __m256d drift_term = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0, ah), v_dt));
                            __m256d sum_sig = _mm256_add_pd(_mm256_add_pd(sigp, sigm), _mm256_mul_pd(v_two, sig0));
                            __m256d diff_term = _mm256_mul_pd(v_quarter, _mm256_mul_pd(sum_sig, dw));
                            __m256d z2m1 = _mm256_fmsub_pd(zd, zd, v_one);
                            __m256d mil_term = _mm256_mul_pd(
                                v_quarter,
                                _mm256_mul_pd(_mm256_sub_pd(sigp, sigm), _mm256_mul_pd(v_sqrt_dt, z2m1))
                            );
                            _mm256_storeu_pd(log_x + n, _mm256_add_pd(lx, _mm256_add_pd(drift_term, _mm256_add_pd(diff_term, mil_term))));
                        }
#endif
                        for (; n < N_rep; ++n) {
                            const double lx = log_x[n];
                            const double zd = z_ptr[n];
                            auto lookup_scalar = [&](const double* slice_ptr, double y_val) noexcept -> double {
                                const double u_raw = (lv_ny > 1) ? ((y_val - lv_y_min) * inv_dy_grid) : 0.0;
                                const double u_y = (u_raw > 0.0) ? ((u_raw < max_u_y) ? u_raw : max_u_y) : 0.0;
                                const int iy0 = static_cast<int>(u_y);
                                const int iy1 = std::min(iy0 + 1, lv_ny - 1);
                                const double wy = u_y - static_cast<double>(iy0);
                                return (1.0 - wy) * slice_ptr[iy0] + wy * slice_ptr[iy1];
                            };
                            const double q_s = has_q ? q_shift_mid : 0.0;
                            const double sig0 = lookup_scalar(lv_slice_start, lx + q_s);
                            const double a0 = -0.5 * sig0 * sig0;
                            const double dw = sqrt_dt * zd;
                            const double base_y = lx + a0 * dt + q_s;
                            const double sigh = lookup_scalar(lv_slice_end, base_y + sig0 * dw);
                            const double sigp = lookup_scalar(lv_slice_end, base_y + sig0 * sqrt_dt);
                            const double sigm = lookup_scalar(lv_slice_end, base_y - sig0 * sqrt_dt);
                            const double ah = -0.5 * sigh * sigh;
                            log_x[n] += 0.5 * (a0 + ah) * dt
                                + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                                + 0.25 * (sigp - sigm) * sqrt_dt * (zd * zd - 1.0);
                        }
                    }

                    double max_lx = -std::numeric_limits<double>::infinity();
                    for (int n = 0; n < N_rep; ++n) {
                        max_lx = std::max(max_lx, log_x[n]);
                    }
                    double sum_ex = 0.0;
                    int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                    const __m256d v_max_lx = _mm256_set1_pd(max_lx);
                    __m256d v_sum = _mm256_setzero_pd();
                    for (; n + 4 <= N_rep; n += 4) {
                        __m256d ex = v_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(log_x + n), v_max_lx));
                        _mm256_storeu_pd(ex_buf + n, ex);
                        v_sum = _mm256_add_pd(v_sum, ex);
                    }
                    alignas(32) double s4[4];
                    _mm256_store_pd(s4, v_sum);
                    sum_ex = s4[0] + s4[1] + s4[2] + s4[3];
#endif
                    for (; n < N_rep; ++n) {
                        const double ex = std::exp(log_x[n] - max_lx);
                        ex_buf[n] = ex;
                        sum_ex += ex;
                    }
                    const double mean_rep = std::max(sum_ex / static_cast<double>(N_rep), 1e-30);
                    const double inv_mean = 1.0 / mean_rep;
                    const double shift_log = max_lx + std::log(mean_rep);
                    double* out_ptr = out_norm_x
                        + (static_cast<std::size_t>(b) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m)) * n_tot_sz
                        + rep_off;

                    #pragma omp simd
                    for (int n_s = 0; n_s < N_rep; ++n_s) {
                        out_ptr[n_s] = ex_buf[n_s] * inv_mean;
                        log_x[n_s] -= shift_log;
                    }

                    t_prev = t_cur;
                    rt_prev = rt_cur;
                    rn_prev = rn_cur;
                }
            }
        }
    }
}

std::int32_t crn_step_spot_bumps_cpp(
    std::int32_t d_assets,
    std::int32_t lv_nt,
    std::int32_t lv_ny,
    std::int32_t m_rem,
    std::int32_t n_sub,
    std::int32_t eff_r,
    std::int32_t n_rep,
    const double* lv_tables_pm,
    const double* z_corr,
    const double* tau_obs,
    const double* r_native_obs,
    const double* r_tot_obs,
    const double* lv_t_grid,
    const double* lv_y_min,
    const double* lv_y_max,
    double* out_norm_x_pm
) noexcept {
    if (d_assets <= 0 || lv_nt <= 0 || lv_ny <= 0 || m_rem <= 0 || n_sub <= 0 ||
        eff_r <= 0 || n_rep <= 0 || lv_tables_pm == nullptr || z_corr == nullptr ||
        tau_obs == nullptr || r_native_obs == nullptr || r_tot_obs == nullptr ||
        lv_t_grid == nullptr || lv_y_min == nullptr || lv_y_max == nullptr ||
        out_norm_x_pm == nullptr) {
        return -1;
    }
    const int D = static_cast<int>(d_assets);
    const int M = static_cast<int>(m_rem);
    const int S = static_cast<int>(n_sub);
    const int R = static_cast<int>(eff_r);
    const int N_rep = static_cast<int>(n_rep);

    const std::size_t Du = static_cast<std::size_t>(D);
    const std::size_t Mu = static_cast<std::size_t>(M);
    const std::size_t Su = static_cast<std::size_t>(S);
    const std::size_t Ru = static_cast<std::size_t>(R);
    const std::size_t n_rep_sz = static_cast<std::size_t>(N_rep);
    const std::size_t n_tot_sz = Ru * n_rep_sz;
    const std::size_t lv_ny_sz = static_cast<std::size_t>(lv_ny);
    const std::size_t lv_plane_sz = static_cast<std::size_t>(lv_nt) * lv_ny_sz;
    const std::size_t out_asset_stride = Mu * n_tot_sz;
    const std::size_t out_pm_stride = Du * out_asset_stride;

    const double t_max = lv_t_grid[lv_nt - 1];
    const double inv_dt_grid = (lv_nt > 1) ? (static_cast<double>(lv_nt - 1) / std::max(t_max, 1e-12)) : 0.0;
    const double max_u_t = std::max(0.0, static_cast<double>(lv_nt - 1) - 1e-6);
    const double max_u_y = std::max(0.0, static_cast<double>(lv_ny - 1) - 1e-6);

    OmpCallGuard omp_guard(D * R >= 2);
    #pragma omp parallel if(omp_guard.active) num_threads(omp_guard.n_threads)
    {
        std::vector<double> lx_p(n_rep_sz, 0.0);
        std::vector<double> lx_m(n_rep_sz, 0.0);
        std::vector<double> ex_p(n_rep_sz, 0.0);
        std::vector<double> ex_m(n_rep_sz, 0.0);
        std::vector<double> zd_buf(n_rep_sz, 0.0);
        std::vector<double> sl0_p(lv_ny_sz + 4, 0.0);
        std::vector<double> sl1_p(lv_ny_sz + 4, 0.0);
        std::vector<double> sl0_m(lv_ny_sz + 4, 0.0);
        std::vector<double> sl1_m(lv_ny_sz + 4, 0.0);

        auto fill_lv_slice = [&](const double* lv_b, double t_val, double* dst_slice) noexcept {
            const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
            const int it0 = static_cast<int>(u_t);
            const int it1 = std::min(it0 + 1, lv_nt - 1);
            const double wt = (lv_nt > 1) ? (u_t - static_cast<double>(it0)) : 0.0;
            const double* row0 = lv_b + static_cast<std::size_t>(it0) * lv_ny_sz;
            const double* row1 = lv_b + static_cast<std::size_t>(it1) * lv_ny_sz;
            const double omt = 1.0 - wt;
            #pragma omp simd
            for (int iy = 0; iy < lv_ny; ++iy) {
                dst_slice[iy] = omt * row0[iy] + wt * row1[iy];
            }
            const double top_sig = dst_slice[lv_ny - 1];
            for (int pad = 0; pad < 4; ++pad) {
                dst_slice[lv_ny + pad] = top_sig;
            }
        };

        #pragma omp for collapse(2) schedule(static)
        for (int d = 0; d < D; ++d) {
            for (int r = 0; r < R; ++r) {
                const std::size_t du = static_cast<std::size_t>(d);
                const std::size_t ru = static_cast<std::size_t>(r);
                const double y_min_d = lv_y_min[du];
                const double y_max_d = lv_y_max[du];
                const double inv_dy_grid = (lv_ny > 1)
                    ? (static_cast<double>(lv_ny - 1) / std::max(y_max_d - y_min_d, 1e-12))
                    : 0.0;
                const double* lv_p = lv_tables_pm + (2u * du + 0u) * lv_plane_sz;
                const double* lv_m = lv_tables_pm + (2u * du + 1u) * lv_plane_sz;
                const double* rn_d = r_native_obs + du * Mu;
                const double* rt_d = r_tot_obs + du * Mu;

                std::fill(lx_p.begin(), lx_p.end(), 0.0);
                std::fill(lx_m.begin(), lx_m.end(), 0.0);

                double t_prev = 0.0;
                double rt_prev = 0.0;
                double rn_prev = 0.0;

                for (int m = 0; m < M; ++m) {
                    const std::size_t mu = static_cast<std::size_t>(m);
                    const double t_cur = tau_obs[mu];
                    const double rt_cur = rt_d[mu];
                    const double rn_cur = rn_d[mu];
                    const double d_tau = std::max(t_cur - t_prev, 1e-10);
                    const double dt = d_tau / static_cast<double>(S);
                    const double sqrt_dt = std::sqrt(dt);
                    const double drt = rt_cur - rt_prev;
                    const double drn = rn_cur - rn_prev;

                    for (int s = 0; s < S; ++s) {
                        const std::size_t su = static_cast<std::size_t>(s);
                        const double alpha_s = (static_cast<double>(s) + 0.5) / static_cast<double>(S);
                        const double t_mid = t_prev + alpha_s * d_tau;
                        const double t_start = std::max(t_mid - 0.5 * dt, 0.0);
                        const double t_end = t_start + dt;
                        const double q_shift_mid = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn);

                        fill_lv_slice(lv_p, t_start, sl0_p.data());
                        fill_lv_slice(lv_p, t_end, sl1_p.data());
                        fill_lv_slice(lv_m, t_start, sl0_m.data());
                        fill_lv_slice(lv_m, t_end, sl1_m.data());

                        // Gather z_corr[(m, s, r, n, d)] into unit-stride zd_buf.
                        const double* z_base = z_corr + (((mu * Su + su) * Ru + ru) * n_rep_sz) * Du + du;
                        for (int n = 0; n < N_rep; ++n) {
                            zd_buf[static_cast<std::size_t>(n)] = z_base[static_cast<std::size_t>(n) * Du];
                        }

                        int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                        const __m256d v_y_min = _mm256_set1_pd(y_min_d);
                        const __m256d v_inv_dy = _mm256_set1_pd(inv_dy_grid);
                        const __m256d v_zero = _mm256_setzero_pd();
                        const __m256d v_max_uy = _mm256_set1_pd(max_u_y);
                        const __m256d v_q_mid = _mm256_set1_pd(q_shift_mid);
                        const __m256d v_dt = _mm256_set1_pd(dt);
                        const __m256d v_sqrt_dt = _mm256_set1_pd(sqrt_dt);
                        const __m256d v_half = _mm256_set1_pd(0.5);
                        const __m256d v_neg_half = _mm256_set1_pd(-0.5);
                        const __m256d v_quarter = _mm256_set1_pd(0.25);
                        const __m256d v_two = _mm256_set1_pd(2.0);
                        const __m256d v_one = _mm256_set1_pd(1.0);

                        auto lookup_slice_pd = [&](const double* slice_ptr, __m256d y_v) noexcept -> __m256d {
                            if (lv_ny <= 1) return _mm256_set1_pd(slice_ptr[0]);
                            __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                            __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                            __m256d u_floor = _mm256_floor_pd(u_y);
                            __m128i iy0 = _mm256_cvttpd_epi32(u_floor);
                            __m256d wy = _mm256_sub_pd(u_y, u_floor);
                            __m256d s0 = _mm256_i32gather_pd(slice_ptr, iy0, 8);
                            __m256d s1 = _mm256_i32gather_pd(slice_ptr + 1, iy0, 8);
                            return _mm256_fmadd_pd(wy, _mm256_sub_pd(s1, s0), s0);
                        };

                        for (; n + 4 <= N_rep; n += 4) {
                            const __m256d zd = _mm256_loadu_pd(zd_buf.data() + n);
                            const __m256d dw = _mm256_mul_pd(v_sqrt_dt, zd);
                            const __m256d z2m1 = _mm256_fmsub_pd(zd, zd, v_one);
                            const __m256d sdt_z2m1 = _mm256_mul_pd(v_sqrt_dt, z2m1);

                            // +h trajectory
                            const __m256d lxp = _mm256_loadu_pd(lx_p.data() + n);
                            const __m256d y0p = _mm256_add_pd(lxp, v_q_mid);
                            const __m256d sig0p = lookup_slice_pd(sl0_p.data(), y0p);
                            const __m256d a0p = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sig0p, sig0p));
                            const __m256d s0p_sdt = _mm256_mul_pd(sig0p, v_sqrt_dt);
                            const __m256d base_yp = _mm256_fmadd_pd(a0p, v_dt, y0p);
                            const __m256d sighp = lookup_slice_pd(sl1_p.data(), _mm256_fmadd_pd(sig0p, dw, base_yp));
                            const __m256d sigpp = lookup_slice_pd(sl1_p.data(), _mm256_add_pd(base_yp, s0p_sdt));
                            const __m256d sigmp = lookup_slice_pd(sl1_p.data(), _mm256_sub_pd(base_yp, s0p_sdt));
                            const __m256d ahp = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sighp, sighp));
                            const __m256d driftp = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0p, ahp), v_dt));
                            const __m256d diffp = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(sigpp, sigmp), _mm256_mul_pd(v_two, sig0p)), dw));
                            const __m256d milp = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_sub_pd(sigpp, sigmp), sdt_z2m1));
                            _mm256_storeu_pd(lx_p.data() + n, _mm256_add_pd(lxp, _mm256_add_pd(driftp, _mm256_add_pd(diffp, milp))));

                            // -h trajectory
                            const __m256d lxm = _mm256_loadu_pd(lx_m.data() + n);
                            const __m256d y0m = _mm256_add_pd(lxm, v_q_mid);
                            const __m256d sig0m = lookup_slice_pd(sl0_m.data(), y0m);
                            const __m256d a0m = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sig0m, sig0m));
                            const __m256d s0m_sdt = _mm256_mul_pd(sig0m, v_sqrt_dt);
                            const __m256d base_ym = _mm256_fmadd_pd(a0m, v_dt, y0m);
                            const __m256d sighm = lookup_slice_pd(sl1_m.data(), _mm256_fmadd_pd(sig0m, dw, base_ym));
                            const __m256d sigpm = lookup_slice_pd(sl1_m.data(), _mm256_add_pd(base_ym, s0m_sdt));
                            const __m256d sigmm = lookup_slice_pd(sl1_m.data(), _mm256_sub_pd(base_ym, s0m_sdt));
                            const __m256d ahm = _mm256_mul_pd(v_neg_half, _mm256_mul_pd(sighm, sighm));
                            const __m256d driftm = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0m, ahm), v_dt));
                            const __m256d diffm = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(sigpm, sigmm), _mm256_mul_pd(v_two, sig0m)), dw));
                            const __m256d milm = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_sub_pd(sigpm, sigmm), sdt_z2m1));
                            _mm256_storeu_pd(lx_m.data() + n, _mm256_add_pd(lxm, _mm256_add_pd(driftm, _mm256_add_pd(diffm, milm))));
                        }
#endif
                        auto lookup_scalar = [&](const double* slice_ptr, double y_val) noexcept -> double {
                            if (lv_ny <= 1) return slice_ptr[0];
                            const double u_raw = (y_val - y_min_d) * inv_dy_grid;
                            const double u_y = std::clamp(u_raw, 0.0, max_u_y);
                            const int iy0 = static_cast<int>(u_y);
                            const int iy1 = std::min(iy0 + 1, lv_ny - 1);
                            const double wy = u_y - static_cast<double>(iy0);
                            return (1.0 - wy) * slice_ptr[iy0] + wy * slice_ptr[iy1];
                        };
                        for (; n < N_rep; ++n) {
                            const double zd = zd_buf[static_cast<std::size_t>(n)];
                            const double dw = sqrt_dt * zd;
                            const double z2m1 = zd * zd - 1.0;

                            const double lxp = lx_p[static_cast<std::size_t>(n)];
                            const double sig0p = lookup_scalar(sl0_p.data(), lxp + q_shift_mid);
                            const double a0p = -0.5 * sig0p * sig0p;
                            const double base_yp = lxp + a0p * dt + q_shift_mid;
                            const double sighp = lookup_scalar(sl1_p.data(), base_yp + sig0p * dw);
                            const double sigpp = lookup_scalar(sl1_p.data(), base_yp + sig0p * sqrt_dt);
                            const double sigmp = lookup_scalar(sl1_p.data(), base_yp - sig0p * sqrt_dt);
                            lx_p[static_cast<std::size_t>(n)] = lxp
                                + 0.5 * (a0p - 0.5 * sighp * sighp) * dt
                                + 0.25 * (sigpp + sigmp + 2.0 * sig0p) * dw
                                + 0.25 * (sigpp - sigmp) * sqrt_dt * z2m1;

                            const double lxm = lx_m[static_cast<std::size_t>(n)];
                            const double sig0m = lookup_scalar(sl0_m.data(), lxm + q_shift_mid);
                            const double a0m = -0.5 * sig0m * sig0m;
                            const double base_ym = lxm + a0m * dt + q_shift_mid;
                            const double sighm = lookup_scalar(sl1_m.data(), base_ym + sig0m * dw);
                            const double sigpm = lookup_scalar(sl1_m.data(), base_ym + sig0m * sqrt_dt);
                            const double sigmm = lookup_scalar(sl1_m.data(), base_ym - sig0m * sqrt_dt);
                            lx_m[static_cast<std::size_t>(n)] = lxm
                                + 0.5 * (a0m - 0.5 * sighm * sighm) * dt
                                + 0.25 * (sigpm + sigmm + 2.0 * sig0m) * dw
                                + 0.25 * (sigpm - sigmm) * sqrt_dt * z2m1;
                        }
                    }

                    auto martingalize_buf = [&](std::vector<double>& lxb, std::vector<double>& exb, double* out_ptr) noexcept {
                        double max_lx = -std::numeric_limits<double>::infinity();
                        for (int n = 0; n < N_rep; ++n) {
                            max_lx = std::max(max_lx, lxb[static_cast<std::size_t>(n)]);
                        }
                        double sum_ex = 0.0;
                        int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                        const __m256d v_max_lx = _mm256_set1_pd(max_lx);
                        __m256d v_sum = _mm256_setzero_pd();
                        for (; n + 4 <= N_rep; n += 4) {
                            __m256d ex = v_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(lxb.data() + n), v_max_lx));
                            _mm256_storeu_pd(exb.data() + n, ex);
                            v_sum = _mm256_add_pd(v_sum, ex);
                        }
                        alignas(32) double s4[4];
                        _mm256_store_pd(s4, v_sum);
                        sum_ex = s4[0] + s4[1] + s4[2] + s4[3];
#endif
                        for (; n < N_rep; ++n) {
                            const double ex = std::exp(lxb[static_cast<std::size_t>(n)] - max_lx);
                            exb[static_cast<std::size_t>(n)] = ex;
                            sum_ex += ex;
                        }
                        const double mean_rep = std::max(sum_ex / static_cast<double>(N_rep), 1e-30);
                        const double inv_mean = 1.0 / mean_rep;
                        const double shift_log = max_lx + std::log(mean_rep);
                        #pragma omp simd
                        for (int n_s = 0; n_s < N_rep; ++n_s) {
                            out_ptr[n_s] = exb[static_cast<std::size_t>(n_s)] * inv_mean;
                            lxb[static_cast<std::size_t>(n_s)] -= shift_log;
                        }
                    };

                    const std::size_t out_off = du * out_asset_stride + mu * n_tot_sz + ru * n_rep_sz;
                    martingalize_buf(lx_p, ex_p, out_norm_x_pm + 0u * out_pm_stride + out_off);
                    martingalize_buf(lx_m, ex_m, out_norm_x_pm + 1u * out_pm_stride + out_off);

                    t_prev = t_cur;
                    rt_prev = rt_cur;
                    rn_prev = rn_cur;
                }
            }
        }
    }
    return 0;
}

}  // extern "C"

namespace {

template <int Code>
void eval_loo_l2o_tile_specialized(
    int D,
    int t_len,
    bool has_a,
    bool has_b,
    int aa,
    int bb,
    int ia,
    int ib,
    double inv_D,
    const double* __restrict t_min1,
    const double* __restrict t_min2,
    const double* __restrict t_min3,
    const double* __restrict t_min4,
    const double* __restrict t_min5,
    const std::int32_t* __restrict t_i1,
    const std::int32_t* __restrict t_i2,
    const std::int32_t* __restrict t_i3,
    const std::int32_t* __restrict t_i4,
    const double* __restrict t_max1,
    const double* __restrict t_max2,
    const double* __restrict t_max3,
    const double* __restrict t_max4,
    const std::int32_t* __restrict t_j1,
    const std::int32_t* __restrict t_j2,
    const std::int32_t* __restrict t_j3,
    const double* __restrict t_sum1,
    const double* __restrict t_sum2,
    const double* __restrict pa_ptr,
    const double* __restrict pb_ptr,
    const double* __restrict base_a_ptr,
    const double* __restrict base_b_ptr,
    const double* __restrict base_ia_ptr,
    const double* __restrict base_ib_ptr,
    double* __restrict dst
) noexcept {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const int aa_eff = has_a ? aa : -1;
    const int bb_eff = has_b ? bb : -1;
    const bool pa_is_a = (has_a && aa == ia);
    const bool pa_is_b = (has_b && bb == ia);
    const bool pb_is_a = (has_a && aa == ib);
    const bool pb_is_b = (has_b && bb == ib);

    #pragma omp simd
    for (int t = 0; t < t_len; ++t) {
        const double va_raw = pa_ptr[t];
        const double vb_raw = pb_ptr[t];
        double out = 0.0;

        if constexpr (Code == 0 || Code == 14 || Code == 6 || Code == 8 || Code == 11 || Code == 13) {
            // Wait to compute w_min when needed below.
        }
        auto get_w_min = [&]() noexcept -> double {
            const double va_u = has_a ? va_raw : kInf;
            const double vb_u = has_b ? vb_raw : kInf;
            const int i1v = t_i1[t];
            const int i2v = t_i2[t];
            const bool ok1 = (i1v != aa_eff && i1v != bb_eff);
            const bool ok2 = (i2v != aa_eff && i2v != bb_eff);
            const double base_ex = ok1 ? t_min1[t] : (ok2 ? t_min2[t] : t_min3[t]);
            return std::min(base_ex, std::min(va_u, vb_u));
        };
        auto get_w_max = [&]() noexcept -> double {
            const double va_u = has_a ? va_raw : -kInf;
            const double vb_u = has_b ? vb_raw : -kInf;
            const int j1v = t_j1[t];
            const int j2v = t_j2[t];
            const bool ok1 = (j1v != aa_eff && j1v != bb_eff);
            const bool ok2 = (j2v != aa_eff && j2v != bb_eff);
            const double base_ex = ok1 ? t_max1[t] : (ok2 ? t_max2[t] : t_max3[t]);
            return std::max(base_ex, std::max(va_u, vb_u));
        };
        auto get_w_avg = [&]() noexcept -> double {
            double s1 = t_sum1[t];
            if (has_a) s1 += (va_raw - base_a_ptr[t]);
            if (has_b) s1 += (vb_raw - base_b_ptr[t]);
            return s1 * inv_D;
        };
        auto get_pa = [&]() noexcept -> double {
            return pa_is_a ? va_raw : (pa_is_b ? vb_raw : base_ia_ptr[t]);
        };
        auto get_pb = [&]() noexcept -> double {
            return pb_is_a ? va_raw : (pb_is_b ? vb_raw : base_ib_ptr[t]);
        };

        if constexpr (Code == 0) {
            out = get_w_min();
        } else if constexpr (Code == 1) {
            out = get_w_max();
        } else if constexpr (Code == 2) {
            out = get_w_avg();
        } else if constexpr (Code == 3) {
            if (D == 1) {
                out = 0.0;
            } else {
                const double w_avg = get_w_avg();
                double s2 = t_sum2[t];
                if (has_a) s2 += (va_raw * va_raw - base_a_ptr[t] * base_a_ptr[t]);
                if (has_b) s2 += (vb_raw * vb_raw - base_b_ptr[t] * base_b_ptr[t]);
                out = std::sqrt(std::max(s2 * inv_D - w_avg * w_avg, 0.0));
            }
        } else if constexpr (Code == 4) {
            const double pa_v = get_pa();
            out = (D == 1) ? pa_v : (pa_v / std::max(get_pb(), 1e-12));
        } else if constexpr (Code == 5) {
            const double pb_v = get_pb();
            out = (D == 1) ? pb_v : (pb_v / std::max(get_pa(), 1e-12));
        } else if constexpr (Code == 6) {
            out = (D == 1) ? 1.0 : (get_w_max() / std::max(get_w_min(), 1e-12));
        } else if constexpr (Code == 7) {
            out = (D == 1) ? 1.0 : (get_w_max() / std::max(get_w_avg(), 1e-12));
        } else if constexpr (Code == 8) {
            out = (D == 1) ? 1.0 : (get_w_avg() / std::max(get_w_min(), 1e-12));
        } else if constexpr (Code == 9) {
            out = (D == 1) ? 0.0 : (get_pa() - get_pb());
        } else if constexpr (Code == 10) {
            out = (D == 1) ? 0.0 : (get_pb() - get_pa());
        } else if constexpr (Code == 11) {
            out = (D == 1) ? 0.0 : (get_w_max() - get_w_min());
        } else if constexpr (Code == 12) {
            out = (D == 1) ? 0.0 : (get_w_max() - get_w_avg());
        } else if constexpr (Code == 13) {
            out = (D == 1) ? 0.0 : (get_w_avg() - get_w_min());
        } else if constexpr (Code == 14) {
            const double w_min = get_w_min();
            out = (w_min < 1.0) ? (2.0 - w_min) : w_min;
        } else if constexpr (Code == 15) {
            const double va_u = has_a ? va_raw : kInf;
            const double vb_u = has_b ? vb_raw : kInf;
            const double u_lo = std::min(va_u, vb_u);
            const double u_hi = std::max(va_u, vb_u);
            const int i1v = t_i1[t], i2v = t_i2[t], i3v = t_i3[t];
            const bool ok1 = (i1v != aa_eff && i1v != bb_eff);
            const bool ok2 = (i2v != aa_eff && i2v != bb_eff);
            const bool ok3 = (i3v != aa_eff && i3v != bb_eff);
            const double b0 = ok1 ? t_min1[t] : (ok2 ? t_min2[t] : t_min3[t]);
            const double b1 = ok1 ? (ok2 ? t_min2[t] : (ok3 ? t_min3[t] : t_min4[t]))
                                  : (ok2 ? (ok3 ? t_min3[t] : t_min4[t]) : t_min4[t]);
            out = std::min(std::max(b0, u_lo), std::min(b1, u_hi));
        } else if constexpr (Code == 16) {
            const double va_u = has_a ? va_raw : kInf;
            const double vb_u = has_b ? vb_raw : kInf;
            const double u_lo = std::min(va_u, vb_u);
            const double u_hi = std::max(va_u, vb_u);
            const int i1v = t_i1[t], i2v = t_i2[t], i3v = t_i3[t], i4v = t_i4[t];
            const bool ok1 = (i1v != aa_eff && i1v != bb_eff);
            const bool ok2 = (i2v != aa_eff && i2v != bb_eff);
            const bool ok3 = (i3v != aa_eff && i3v != bb_eff);
            const bool ok4 = (i4v != aa_eff && i4v != bb_eff);
            const int p2 = ok1 ? 1 : 0;
            const int p3 = p2 + (ok2 ? 1 : 0);
            const int p4 = p3 + (ok3 ? 1 : 0);
            const double b0 = ok1 ? t_min1[t] : (ok2 ? t_min2[t] : t_min3[t]);
            const double b1 = (ok2 && p2 == 1) ? t_min2[t] : ((ok3 && p3 == 1) ? t_min3[t] : t_min4[t]);
            const double b2 = (ok3 && p3 == 2) ? t_min3[t] : ((ok4 && p4 == 2) ? t_min4[t] : t_min5[t]);
            out = std::min(std::max(b0, u_hi), std::min(std::max(b1, u_lo), b2));
        } else if constexpr (Code == 17) {
            const double va_u = has_a ? va_raw : -kInf;
            const double vb_u = has_b ? vb_raw : -kInf;
            const double u_hi = std::max(va_u, vb_u);
            const double u_lo = std::min(va_u, vb_u);
            const int j1v = t_j1[t], j2v = t_j2[t], j3v = t_j3[t];
            const bool ok1 = (j1v != aa_eff && j1v != bb_eff);
            const bool ok2 = (j2v != aa_eff && j2v != bb_eff);
            const bool ok3 = (j3v != aa_eff && j3v != bb_eff);
            const double t0 = ok1 ? t_max1[t] : (ok2 ? t_max2[t] : t_max3[t]);
            const double t1 = ok1 ? (ok2 ? t_max2[t] : (ok3 ? t_max3[t] : t_max4[t]))
                                  : (ok2 ? (ok3 ? t_max3[t] : t_max4[t]) : t_max4[t]);
            out = std::max(std::min(t0, u_hi), std::max(t1, u_lo));
        }
        dst[t] = out;
    }
}

void dispatch_loo_l2o_tile_code(
    int eff_code,
    int D,
    int t_len,
    bool has_a,
    bool has_b,
    int aa,
    int bb,
    int ia,
    int ib,
    double inv_D,
    const double* t_min1, const double* t_min2, const double* t_min3, const double* t_min4, const double* t_min5,
    const std::int32_t* t_i1, const std::int32_t* t_i2, const std::int32_t* t_i3, const std::int32_t* t_i4,
    const double* t_max1, const double* t_max2, const double* t_max3, const double* t_max4,
    const std::int32_t* t_j1, const std::int32_t* t_j2, const std::int32_t* t_j3,
    const double* t_sum1, const double* t_sum2,
    const double* pa_ptr, const double* pb_ptr,
    const double* base_a_ptr, const double* base_b_ptr,
    const double* base_ia_ptr, const double* base_ib_ptr,
    double* dst
) noexcept {
#define DISPATCH_CODE_CASE(C) \
    case C: \
        eval_loo_l2o_tile_specialized<C>( \
            D, t_len, has_a, has_b, aa, bb, ia, ib, inv_D, \
            t_min1, t_min2, t_min3, t_min4, t_min5, t_i1, t_i2, t_i3, t_i4, \
            t_max1, t_max2, t_max3, t_max4, t_j1, t_j2, t_j3, \
            t_sum1, t_sum2, pa_ptr, pb_ptr, base_a_ptr, base_b_ptr, base_ia_ptr, base_ib_ptr, dst \
        ); \
        return;

    switch (eff_code) {
        DISPATCH_CODE_CASE(0)
        DISPATCH_CODE_CASE(1)
        DISPATCH_CODE_CASE(2)
        DISPATCH_CODE_CASE(3)
        DISPATCH_CODE_CASE(4)
        DISPATCH_CODE_CASE(5)
        DISPATCH_CODE_CASE(6)
        DISPATCH_CODE_CASE(7)
        DISPATCH_CODE_CASE(8)
        DISPATCH_CODE_CASE(9)
        DISPATCH_CODE_CASE(10)
        DISPATCH_CODE_CASE(11)
        DISPATCH_CODE_CASE(12)
        DISPATCH_CODE_CASE(13)
        DISPATCH_CODE_CASE(14)
        DISPATCH_CODE_CASE(15)
        DISPATCH_CODE_CASE(16)
        DISPATCH_CODE_CASE(17)
        default:
            eval_loo_l2o_tile_specialized<0>(
                D, t_len, has_a, has_b, aa, bb, ia, ib, inv_D,
                t_min1, t_min2, t_min3, t_min4, t_min5, t_i1, t_i2, t_i3, t_i4,
                t_max1, t_max2, t_max3, t_max4, t_j1, t_j2, t_j3,
                t_sum1, t_sum2, pa_ptr, pb_ptr, base_a_ptr, base_b_ptr, base_ia_ptr, base_ib_ptr, dst
            );
            return;
    }
#undef DISPATCH_CODE_CASE
}

template <int Code>
void eval_full_tile_specialized(
    int D,
    int t_len,
    std::size_t N,
    int ia,
    int ib,
    double inv_D,
    const double* __restrict full_m_base,
    const double* __restrict base_ia_ptr,
    const double* __restrict base_ib_ptr,
    double* __restrict dst
) noexcept {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    #pragma omp simd
    for (int t = 0; t < t_len; ++t) {
        const double v0 = full_m_base[static_cast<std::size_t>(t)];
        double out = 0.0;
        if constexpr (Code == 0 || Code == 14) {
            double w_min = v0;
            for (int d = 1; d < D; ++d) w_min = std::min(w_min, full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)]);
            out = (Code == 14 && w_min < 1.0) ? (2.0 - w_min) : w_min;
        } else if constexpr (Code == 1) {
            double w_max = v0;
            for (int d = 1; d < D; ++d) w_max = std::max(w_max, full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)]);
            out = w_max;
        } else if constexpr (Code == 2) {
            double s1 = v0;
            for (int d = 1; d < D; ++d) s1 += full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
            out = s1 * inv_D;
        } else if constexpr (Code == 3) {
            if (D == 1) {
                out = 0.0;
            } else {
                double s1 = v0, s2 = v0 * v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    s1 += vd; s2 += vd * vd;
                }
                const double w_avg = s1 * inv_D;
                out = std::sqrt(std::max(s2 * inv_D - w_avg * w_avg, 0.0));
            }
        } else if constexpr (Code == 4) {
            const double pa_v = base_ia_ptr[t];
            out = (D == 1) ? pa_v : (pa_v / std::max(base_ib_ptr[t], 1e-12));
        } else if constexpr (Code == 5) {
            const double pb_v = base_ib_ptr[t];
            out = (D == 1) ? pb_v : (pb_v / std::max(base_ia_ptr[t], 1e-12));
        } else if constexpr (Code == 6) {
            if (D == 1) { out = 1.0; }
            else {
                double w_min = v0, w_max = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_min = std::min(w_min, vd); w_max = std::max(w_max, vd);
                }
                out = w_max / std::max(w_min, 1e-12);
            }
        } else if constexpr (Code == 7) {
            if (D == 1) { out = 1.0; }
            else {
                double w_max = v0, s1 = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_max = std::max(w_max, vd); s1 += vd;
                }
                out = w_max / std::max(s1 * inv_D, 1e-12);
            }
        } else if constexpr (Code == 8) {
            if (D == 1) { out = 1.0; }
            else {
                double w_min = v0, s1 = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_min = std::min(w_min, vd); s1 += vd;
                }
                out = (s1 * inv_D) / std::max(w_min, 1e-12);
            }
        } else if constexpr (Code == 9) {
            out = (D == 1) ? 0.0 : (base_ia_ptr[t] - base_ib_ptr[t]);
        } else if constexpr (Code == 10) {
            out = (D == 1) ? 0.0 : (base_ib_ptr[t] - base_ia_ptr[t]);
        } else if constexpr (Code == 11) {
            if (D == 1) { out = 0.0; }
            else {
                double w_min = v0, w_max = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_min = std::min(w_min, vd); w_max = std::max(w_max, vd);
                }
                out = w_max - w_min;
            }
        } else if constexpr (Code == 12) {
            if (D == 1) { out = 0.0; }
            else {
                double w_max = v0, s1 = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_max = std::max(w_max, vd); s1 += vd;
                }
                out = w_max - (s1 * inv_D);
            }
        } else if constexpr (Code == 13) {
            if (D == 1) { out = 0.0; }
            else {
                double w_min = v0, s1 = v0;
                for (int d = 1; d < D; ++d) {
                    const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                    w_min = std::min(w_min, vd); s1 += vd;
                }
                out = (s1 * inv_D) - w_min;
            }
        } else if constexpr (Code == 15) {
            double m1 = v0, m2 = kInf;
            for (int d = 1; d < D; ++d) {
                const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                if (vd < m1) { m2 = m1; m1 = vd; }
                else if (vd < m2) { m2 = vd; }
            }
            out = m2;
        } else if constexpr (Code == 16) {
            double m1 = v0, m2 = kInf, m3 = kInf;
            for (int d = 1; d < D; ++d) {
                const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                if (vd < m1) { m3 = m2; m2 = m1; m1 = vd; }
                else if (vd < m2) { m3 = m2; m2 = vd; }
                else if (vd < m3) { m3 = vd; }
            }
            out = m3;
        } else if constexpr (Code == 17) {
            double x1 = v0, x2 = -kInf;
            for (int d = 1; d < D; ++d) {
                const double vd = full_m_base[static_cast<std::size_t>(d) * N + static_cast<std::size_t>(t)];
                if (vd > x1) { x2 = x1; x1 = vd; }
                else if (vd > x2) { x2 = vd; }
            }
            out = x2;
        }
        (void)ia; (void)ib;
        dst[t] = out;
    }
}

void dispatch_full_tile_code(
    int eff_code,
    int D,
    int t_len,
    std::size_t N,
    int ia,
    int ib,
    double inv_D,
    const double* full_m_base,
    const double* base_ia_ptr,
    const double* base_ib_ptr,
    double* dst
) noexcept {
#define DISPATCH_FULL_CASE(C) \
    case C: \
        eval_full_tile_specialized<C>(D, t_len, N, ia, ib, inv_D, full_m_base, base_ia_ptr, base_ib_ptr, dst); \
        return;

    switch (eff_code) {
        DISPATCH_FULL_CASE(0)
        DISPATCH_FULL_CASE(1)
        DISPATCH_FULL_CASE(2)
        DISPATCH_FULL_CASE(3)
        DISPATCH_FULL_CASE(4)
        DISPATCH_FULL_CASE(5)
        DISPATCH_FULL_CASE(6)
        DISPATCH_FULL_CASE(7)
        DISPATCH_FULL_CASE(8)
        DISPATCH_FULL_CASE(9)
        DISPATCH_FULL_CASE(10)
        DISPATCH_FULL_CASE(11)
        DISPATCH_FULL_CASE(12)
        DISPATCH_FULL_CASE(13)
        DISPATCH_FULL_CASE(14)
        DISPATCH_FULL_CASE(15)
        DISPATCH_FULL_CASE(16)
        DISPATCH_FULL_CASE(17)
        default:
            eval_full_tile_specialized<0>(D, t_len, N, ia, ib, inv_D, full_m_base, base_ia_ptr, base_ib_ptr, dst);
            return;
    }
#undef DISPATCH_FULL_CASE
}

inline int canonicalize_tile_code(int code, int D) noexcept {
    if (D <= 1 && (code == 15 || code == 16 || code == 17)) return 0;
    if (D == 2) {
        if (code == 15 || code == 16) return 1;
        if (code == 17) return 0;
    }
    if (D == 3) {
        if (code == 16) return 1;
        if (code == 17) return 15;
    }
    return code;
}

inline double c2_s5(double t) noexcept {
    const double u = std::clamp(t, 0.0, 1.0);
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

inline double c2_ds5(double t) noexcept {
    if (t <= 0.0 || t >= 1.0) return 0.0;
    const double u = t;
    const double om = 1.0 - u;
    return 30.0 * u * u * om * om;
}

struct BilinearLvEval {
    double val;
    double dy;
    int it0;
    int it1;
    double wt;
    int iy0;
    int iy1;
    double wy;
};

inline BilinearLvEval eval_lv_bilinear_with_deriv(
    const double* lv_d,
    int lv_nt,
    int lv_ny,
    double t_val,
    double y_val,
    double inv_dt_grid,
    double y_min,
    double inv_dy_grid,
    double max_u_t,
    double max_u_y
) noexcept {
    const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
    const int it0 = static_cast<int>(u_t);
    const int it1 = std::min(it0 + 1, lv_nt - 1);
    const double wt = u_t - static_cast<double>(it0);

    const double u_y_raw = (lv_ny > 1) ? ((y_val - y_min) * inv_dy_grid) : 0.0;
    const bool in_y = (lv_ny > 1) && (u_y_raw > 0.0) && (u_y_raw < max_u_y);
    const double u_y = (lv_ny > 1) ? std::clamp(u_y_raw, 0.0, max_u_y) : 0.0;
    const int iy0 = static_cast<int>(u_y);
    const int iy1 = std::min(iy0 + 1, lv_ny - 1);
    const double wy = u_y - static_cast<double>(iy0);

    const std::size_t row0 = static_cast<std::size_t>(it0) * static_cast<std::size_t>(lv_ny);
    const std::size_t row1 = static_cast<std::size_t>(it1) * static_cast<std::size_t>(lv_ny);
    const double v00 = lv_d[row0 + static_cast<std::size_t>(iy0)];
    const double v01 = lv_d[row0 + static_cast<std::size_t>(iy1)];
    const double v10 = lv_d[row1 + static_cast<std::size_t>(iy0)];
    const double v11 = lv_d[row1 + static_cast<std::size_t>(iy1)];

    const double v0 = (1.0 - wt) * v00 + wt * v10;
    const double v1 = (1.0 - wt) * v01 + wt * v11;
    const double val = (1.0 - wy) * v0 + wy * v1;
    const double dy = in_y ? ((v1 - v0) * inv_dy_grid) : 0.0;
    return {val, dy, it0, it1, wt, iy0, iy1, wy};
}

inline double eval_lv_bilinear_val(
    const double* lv_d,
    int lv_nt,
    int lv_ny,
    double t_val,
    double y_val,
    double inv_dt_grid,
    double y_min,
    double inv_dy_grid,
    double max_u_t,
    double max_u_y
) noexcept {
    const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
    const int it0 = static_cast<int>(u_t);
    const int it1 = std::min(it0 + 1, lv_nt - 1);
    const double wt = u_t - static_cast<double>(it0);

    const double u_y = (lv_ny > 1) ? std::clamp((y_val - y_min) * inv_dy_grid, 0.0, max_u_y) : 0.0;
    const int iy0 = static_cast<int>(u_y);
    const int iy1 = std::min(iy0 + 1, lv_ny - 1);
    const double wy = u_y - static_cast<double>(iy0);

    const std::size_t row0 = static_cast<std::size_t>(it0) * static_cast<std::size_t>(lv_ny);
    const std::size_t row1 = static_cast<std::size_t>(it1) * static_cast<std::size_t>(lv_ny);
    const double v0 = (1.0 - wt) * lv_d[row0 + static_cast<std::size_t>(iy0)]
                    + wt * lv_d[row1 + static_cast<std::size_t>(iy0)];
    const double v1 = (1.0 - wt) * lv_d[row0 + static_cast<std::size_t>(iy1)]
                    + wt * lv_d[row1 + static_cast<std::size_t>(iy1)];
    return (1.0 - wy) * v0 + wy * v1;
}

inline std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double u64_to_unit_open(std::uint64_t x) noexcept {
    const double u = (static_cast<double>(x >> 11) + 0.5) * 0x1.0p-53;
    return std::clamp(u, 1e-12, 1.0 - 1e-12);
}

struct IdxBaseStats {
    double min1;
    double min2;
    double min3;
    double max1;
    double max2;
    double s1;
    double s2;
    double pa;
    double pb;
    int idx_min1;
    int idx_min2;
    int idx_min3;
    int idx_max1;
    int idx_max2;
};

inline double select_idx_from_stats(
    const IdxBaseStats& st,
    double inv_d,
    int d_assets,
    int code
) noexcept {
    const double w_avg = st.s1 * inv_d;
    switch (code) {
        case 0: return st.min1;
        case 1: return st.max1;
        case 2: return w_avg;
        case 3: return (d_assets <= 1) ? 0.0 : std::sqrt(std::max(st.s2 * inv_d - w_avg * w_avg, 0.0));
        case 4: return (d_assets < 2) ? st.pa : (st.pa / std::max(st.pb, 1e-12));
        case 5: return (d_assets < 2) ? st.pa : (st.pb / std::max(st.pa, 1e-12));
        case 6: return st.max1 / std::max(st.min1, 1e-12);
        case 7: return st.max1 / std::max(w_avg, 1e-12);
        case 8: return w_avg / std::max(st.min1, 1e-12);
        case 9: return (d_assets < 2) ? 0.0 : (st.pa - st.pb);
        case 10: return (d_assets < 2) ? 0.0 : (st.pb - st.pa);
        case 11: return st.max1 - st.min1;
        case 12: return st.max1 - w_avg;
        case 13: return w_avg - st.min1;
        case 14: return (st.min1 < 1.0) ? (2.0 - st.min1) : st.min1;
        case 15: return (d_assets > 1) ? st.min2 : st.min1;
        case 16: return (d_assets >= 3) ? st.min3 : ((d_assets == 2) ? st.min2 : st.min1);
        case 17: return (d_assets > 1) ? st.max2 : st.max1;
        default: return st.min1;
    }
}

inline double accum_idx_grad_scalar(
    double bp,
    double bar_w,
    double pd,
    int d,
    int d_assets,
    double inv_d,
    int ia,
    int ib,
    const IdxBaseStats& st,
    double w_avg,
    double w_disp,
    int code
) noexcept {
    constexpr double kTiny = 1e-12;
    switch (code) {
        case 0: return (d == st.idx_min1) ? (bp + bar_w) : bp;
        case 1: return (d == st.idx_max1) ? (bp + bar_w) : bp;
        case 2: return bp + bar_w * inv_d;
        case 3:
            if (d_assets > 1 && w_disp > kTiny) {
                return bp + bar_w * (pd - w_avg) * inv_d / std::max(w_disp, kTiny);
            }
            return bp;
        case 4: {
            if (d_assets < 2) return (d == ia) ? (bp + bar_w) : bp;
            const double vb_c = std::max(st.pb, kTiny);
            if (d == ia) bp += bar_w / vb_c;
            if (d == ib && st.pb > kTiny) bp -= bar_w * st.pa / (vb_c * vb_c);
            return bp;
        }
        case 5: {
            if (d_assets < 2) return (d == ia) ? (bp + bar_w) : bp;
            const double va_c = std::max(st.pa, kTiny);
            if (d == ib) bp += bar_w / va_c;
            if (d == ia && st.pa > kTiny) bp -= bar_w * st.pb / (va_c * va_c);
            return bp;
        }
        case 6: {
            const double m_c = std::max(st.min1, kTiny);
            if (d == st.idx_max1) bp += bar_w / m_c;
            if (d == st.idx_min1 && st.min1 > kTiny) bp -= bar_w * st.max1 / (m_c * m_c);
            return bp;
        }
        case 7: {
            const double a_c = std::max(w_avg, kTiny);
            if (w_avg > kTiny) bp -= bar_w * st.max1 * inv_d / (a_c * a_c);
            if (d == st.idx_max1) bp += bar_w / a_c;
            return bp;
        }
        case 8: {
            const double m_c = std::max(st.min1, kTiny);
            bp += bar_w * inv_d / m_c;
            if (d == st.idx_min1 && st.min1 > kTiny) bp -= bar_w * w_avg / (m_c * m_c);
            return bp;
        }
        case 9:
            if (d_assets < 2) return bp;
            if (d == ia) bp += bar_w;
            if (d == ib) bp -= bar_w;
            return bp;
        case 10:
            if (d_assets < 2) return bp;
            if (d == ib) bp += bar_w;
            if (d == ia) bp -= bar_w;
            return bp;
        case 11:
            if (d == st.idx_max1) bp += bar_w;
            if (d == st.idx_min1) bp -= bar_w;
            return bp;
        case 12:
            if (d == st.idx_max1) bp += bar_w;
            return bp - bar_w * inv_d;
        case 13:
            bp += bar_w * inv_d;
            if (d == st.idx_min1) bp -= bar_w;
            return bp;
        case 14: {
            const double s_tw = (st.min1 < 1.0) ? -1.0 : 1.0;
            return (d == st.idx_min1) ? (bp + bar_w * s_tw) : bp;
        }
        case 15: {
            const int t_idx = (d_assets > 1) ? st.idx_min2 : st.idx_min1;
            return (d == t_idx) ? (bp + bar_w) : bp;
        }
        case 16: {
            const int t_idx = (d_assets >= 3) ? st.idx_min3 : ((d_assets == 2) ? st.idx_min2 : st.idx_min1);
            return (d == t_idx) ? (bp + bar_w) : bp;
        }
        case 17: {
            const int t_idx = (d_assets > 1) ? st.idx_max2 : st.idx_max1;
            return (d == t_idx) ? (bp + bar_w) : bp;
        }
        default:
            return bp;
    }
}

}  // namespace

extern "C" {

void crn_eval_payoff_loo_l2o_cpp(const CrnLooL2oBatchArgs* args) noexcept {
    if (args == nullptr || args->n_bumps <= 0 || args->d_assets <= 0 || args->m_rem <= 0 || args->n_paths_total <= 0) {
        return;
    }
    if (args->pv_out == nullptr || args->df_pay_mat == nullptr ||
        args->ko_barriers == nullptr || args->is_callable == nullptr ||
        args->cpn_barriers == nullptr || args->is_guaranteed_cpn == nullptr || args->coupon_rates == nullptr) {
        return;
    }
    const int B = static_cast<int>(args->n_bumps);
    const int D = static_cast<int>(args->d_assets);
    const int M = static_cast<int>(args->m_rem);
    const std::size_t N = static_cast<std::size_t>(args->n_paths_total);
    const double inv_N = 1.0 / static_cast<double>(N);
    const double inv_D = 1.0 / static_cast<double>(D);

    const double* perf_base = args->perf_base;
    const double* bump_a_ptr = args->bump_perf_a;
    const double* bump_b_ptr = args->bump_perf_b;
    const std::int32_t* asset_a = args->bump_asset_a;
    const std::int32_t* asset_b = args->bump_asset_b;

    bool any_needs_base = false;
    bool invalid_bump = false;
    for (int b = 0; b < B; ++b) {
        const int aa = (asset_a != nullptr) ? static_cast<int>(asset_a[b]) : -1;
        if (aa == -2) {
            if (bump_a_ptr == nullptr) {
                invalid_bump = true;
                break;
            }
        } else {
            any_needs_base = true;
            if (perf_base == nullptr || aa >= D || aa < -2) {
                invalid_bump = true;
                break;
            }
        }
    }
    if (invalid_bump || (any_needs_base && perf_base == nullptr)) {
        for (int b = 0; b < B; ++b) args->pv_out[b] = 0.0;
        return;
    }

    const int call_code = args->call_code;
    const int cpn_code = args->cpn_code;
    const bool is_long_basket = (args->is_long_basket != 0);
    const int opt_code = args->opt_code;
    const int eff_call = canonicalize_tile_code(call_code, D);
    const int eff_cpn = canonicalize_tile_code(cpn_code, D);
    const int eff_opt = canonicalize_tile_code(opt_code, D);

    const int ia = std::clamp((args->_reserved[0] > 0) ? (args->_reserved[0] - 1) : 0, 0, D - 1);
    const int ib = std::clamp((args->_reserved[1] > 0) ? (args->_reserved[1] - 1) : (D > 1 ? 1 : 0), 0, D - 1);

    const bool has_ki = args->has_ki != 0;
    const bool is_mem = args->is_memory != 0;
    const bool no_notional = args->no_notional != 0;
    const bool force_ac = args->force_cpn_ac != 0;
    const bool dir_down = args->cpn_dir_down != 0;
    const bool type_out = args->cpn_type_out != 0;
    const bool has_one_star = args->one_star_strike > 0.0;
    const bool opt_is_normal = is_diff_index(opt_code);
    const bool has_lower_strike = std::isfinite(args->lower_strike) && (args->lower_strike > 0.0);
    const double k_put = opt_is_normal ? args->strike : std::max(args->strike, 0.0);
    const double k_low = has_lower_strike ? args->lower_strike : 0.0;
    const bool has_put = is_long_basket || (
        (args->option_gearing > 0.0) &&
        (opt_is_normal ? (!has_lower_strike || k_put > k_low) : (k_put > 0.0 && (!has_lower_strike || k_put > k_low)))
    );
    const bool has_part_call = std::isfinite(args->part_call_strike) && (args->part_call_rate != 0.0) &&
                               (opt_is_normal || args->part_call_strike >= 0.0);
    const bool has_w_fund = args->w_fund_mat != nullptr;

    const bool hard_ko = std::abs(args->ko_spread) <= 1e-6;
    const double eff_cpn_sp = dir_down ? -args->cpn_spread : args->cpn_spread;
    const bool hard_cpn = std::abs(eff_cpn_sp) <= 1e-6;
    const bool hard_ki = std::abs(args->ki_spread) <= 1e-6;
    const double b_ki = args->ki_barrier;
    const double blo_ki = std::min(b_ki, b_ki + args->ki_spread);
    const double bhi_ki = std::max(b_ki, b_ki + args->ki_spread);
    const double inv_w_ki = 1.0 / std::max(bhi_ki - blo_ki, 1e-12);
    const double blo_ki_islb = std::min(k_put, k_put + args->ki_spread);
    const double bhi_ki_islb = std::max(k_put, k_put + args->ki_spread);
    const double inv_w_ki_islb = 1.0 / std::max(bhi_ki_islb - blo_ki_islb, 1e-12);
    const double k_call = args->part_call_strike;
    const double k_os = args->one_star_strike;
    const double eps_os = args->one_star_eps;
    const double os_cpn = args->one_star_coupon;
    const double c_mem_0 = args->c_mem_0;
    const double i_ki_past = args->i_ki_past;
    const double ac_cpn = args->autocall_coupon;
    const double option_gearing = args->option_gearing;
    const double part_call_rate = args->part_call_rate;

    const double* ko_arr = args->ko_barriers;
    const double* is_call_arr = args->is_callable;
    const double* cpn_arr = args->cpn_barriers;
    const double* is_guar_arr = args->is_guaranteed_cpn;
    const double* cpn_rates = args->coupon_rates;
    const double* df_pay_mat = args->df_pay_mat;
    const double* w_fund_mat = args->w_fund_mat;
    const double* pv_in_tr = args->pv_in_transit;

    auto uses_min5 = [](int c) noexcept { return c == 16; };
    auto uses_min4 = [](int c) noexcept { return c == 15 || c == 16; };
    auto uses_min3 = [](int c) noexcept {
        return c == 0 || c == 6 || c == 8 || c == 11 || c == 13 || c == 14 || c == 15 || c == 16;
    };
    auto uses_max4 = [](int c) noexcept { return c == 17; };
    auto uses_max3 = [](int c) noexcept {
        return c == 1 || c == 6 || c == 7 || c == 11 || c == 12 || c == 17;
    };
    auto uses_sum1 = [](int c) noexcept {
        return c == 2 || c == 3 || c == 7 || c == 8 || c == 12 || c == 13;
    };
    const bool need_min3 = any_needs_base && (uses_min3(eff_call) || uses_min3(eff_cpn) || uses_min3(eff_opt) || is_long_basket);
    const bool need_min4 = any_needs_base && (uses_min4(eff_call) || uses_min4(eff_cpn) || uses_min4(eff_opt));
    const bool need_min5 = any_needs_base && (uses_min5(eff_call) || uses_min5(eff_cpn) || uses_min5(eff_opt));
    const bool need_max3 = any_needs_base && (uses_max3(eff_call) || uses_max3(eff_cpn) || uses_max3(eff_opt) || has_one_star);
    const bool need_max4 = any_needs_base && (uses_max4(eff_call) || uses_max4(eff_cpn) || uses_max4(eff_opt));
    const bool need_sum1 = any_needs_base && (uses_sum1(eff_call) || uses_sum1(eff_cpn) || uses_sum1(eff_opt) || is_long_basket);
    const bool need_sum2 = any_needs_base && (eff_call == 3 || eff_cpn == 3 || eff_opt == 3);
    const bool need_any_base = need_min3 || need_max3 || need_sum1 || need_sum2;

    static constexpr int kTile = 32;
    static constexpr int kBTile = 16;
    const std::size_t n_tiles = (N + static_cast<std::size_t>(kTile) - 1) / static_cast<std::size_t>(kTile);
    const bool cache_all_m = (B > kBTile);
    const std::size_t m_slots = cache_all_m ? static_cast<std::size_t>(M) : 1u;

    const std::size_t bump_stride = static_cast<std::size_t>(3 * M + 4);
    const std::size_t total_acc = static_cast<std::size_t>(B) * bump_stride;
    std::vector<double> global_acc(total_acc, 0.0);

    OmpCallGuard omp_guard(static_cast<std::size_t>(B) * N * static_cast<std::size_t>(M) >= 65536);
    const int n_threads = omp_guard.active ? omp_guard.n_threads : 1;
    std::vector<double> per_thread_acc(static_cast<std::size_t>(n_threads) * total_acc, 0.0);

    alignas(64) static const double kDummyZero[kTile]{};

    #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
    {
#ifdef _OPENMP
        const int tid = omp_guard.active ? omp_get_thread_num() : 0;
#else
        const int tid = 0;
#endif
        double* thr_acc = per_thread_acc.data() + static_cast<std::size_t>(tid) * total_acc;
        const std::size_t period_stride = static_cast<std::size_t>(kTile);

        std::vector<double> t_min1(need_min3 ? m_slots * period_stride : 0);
        std::vector<double> t_min2(need_min3 ? m_slots * period_stride : 0);
        std::vector<double> t_min3(need_min3 ? m_slots * period_stride : 0);
        std::vector<double> t_min4(need_min4 ? m_slots * period_stride : 0);
        std::vector<double> t_min5(need_min5 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_i1(need_min3 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_i2(need_min3 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_i3(need_min4 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_i4(need_min5 ? m_slots * period_stride : 0);

        std::vector<double> t_max1(need_max3 ? m_slots * period_stride : 0);
        std::vector<double> t_max2(need_max3 ? m_slots * period_stride : 0);
        std::vector<double> t_max3(need_max3 ? m_slots * period_stride : 0);
        std::vector<double> t_max4(need_max4 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_j1(need_max3 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_j2(need_max3 ? m_slots * period_stride : 0);
        std::vector<std::int32_t> t_j3(need_max4 ? m_slots * period_stride : 0);

        std::vector<double> t_sum1(need_sum1 ? m_slots * period_stride : 0);
        std::vector<double> t_sum2(need_sum2 ? m_slots * period_stride : 0);

        alignas(64) std::array<std::array<double, kTile>, kBTile> surv_prev;
        alignas(64) std::array<std::array<double, kTile>, kBTile> c_mem;
        alignas(64) std::array<double, kTile> w_call_buf{}, w_cpn_buf{}, w_opt_buf{}, w_best_buf{}, w_avg_buf{}, w_wof_buf{};

        #pragma omp for schedule(static)
        for (std::size_t tile_idx = 0; tile_idx < n_tiles; ++tile_idx) {
            const std::size_t n0 = tile_idx * static_cast<std::size_t>(kTile);
            const int t_len = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(kTile), N - n0));

            for (int b0 = 0; b0 < B; b0 += kBTile) {
                const int b_end = std::min(b0 + kBTile, B);
                const int b_len = b_end - b0;
                for (int bl = 0; bl < b_len; ++bl) {
                    const auto blu = static_cast<std::size_t>(bl);
                    for (int t = 0; t < t_len; ++t) {
                        const auto tu = static_cast<std::size_t>(t);
                        surv_prev[blu][tu] = 1.0;
                        if (is_mem) c_mem[blu][tu] = c_mem_0;
                    }
                }

                for (int m = 0; m < M; ++m) {
                    const std::size_t m_off = cache_all_m ? (static_cast<std::size_t>(m) * period_stride) : 0u;
                    const std::size_t perf_m_off = static_cast<std::size_t>(m) * static_cast<std::size_t>(D) * N + n0;

                    if (need_any_base && (!cache_all_m || b0 == 0)) {
                        if (need_min3) {
                            for (int t = 0; t < t_len; ++t) {
                                const std::size_t idx = m_off + static_cast<std::size_t>(t);
                                t_min1[idx] = std::numeric_limits<double>::infinity();
                                t_min2[idx] = std::numeric_limits<double>::infinity();
                                t_min3[idx] = std::numeric_limits<double>::infinity();
                                t_i1[idx] = -1;
                                t_i2[idx] = -1;
                                if (need_min4) {
                                    t_min4[idx] = std::numeric_limits<double>::infinity();
                                    t_i3[idx] = -1;
                                }
                                if (need_min5) {
                                    t_min5[idx] = std::numeric_limits<double>::infinity();
                                    t_i4[idx] = -1;
                                }
                            }
                        }
                        if (need_max3) {
                            for (int t = 0; t < t_len; ++t) {
                                const std::size_t idx = m_off + static_cast<std::size_t>(t);
                                t_max1[idx] = -std::numeric_limits<double>::infinity();
                                t_max2[idx] = -std::numeric_limits<double>::infinity();
                                t_max3[idx] = -std::numeric_limits<double>::infinity();
                                t_j1[idx] = -1;
                                t_j2[idx] = -1;
                                if (need_max4) {
                                    t_max4[idx] = -std::numeric_limits<double>::infinity();
                                    t_j3[idx] = -1;
                                }
                            }
                        }
                        if (need_sum1) {
                            for (int t = 0; t < t_len; ++t) t_sum1[m_off + static_cast<std::size_t>(t)] = 0.0;
                        }
                        if (need_sum2) {
                            for (int t = 0; t < t_len; ++t) t_sum2[m_off + static_cast<std::size_t>(t)] = 0.0;
                        }

                        for (int d = 0; d < D; ++d) {
                            const double* p_d = perf_base + perf_m_off + static_cast<std::size_t>(d) * N;
                            for (int t = 0; t < t_len; ++t) {
                                const double v = p_d[t];
                                const std::size_t idx = m_off + static_cast<std::size_t>(t);
                                if (need_min5) {
                                    if (v < t_min1[idx]) {
                                        t_min5[idx] = t_min4[idx]; t_i4[idx] = t_i3[idx];
                                        t_min4[idx] = t_min3[idx]; t_i3[idx] = t_i2[idx];
                                        t_min3[idx] = t_min2[idx]; t_i2[idx] = t_i1[idx];
                                        t_min2[idx] = t_min1[idx]; t_min1[idx] = v; t_i1[idx] = d;
                                    } else if (v < t_min2[idx]) {
                                        t_min5[idx] = t_min4[idx]; t_i4[idx] = t_i3[idx];
                                        t_min4[idx] = t_min3[idx]; t_i3[idx] = t_i2[idx];
                                        t_min3[idx] = t_min2[idx]; t_min2[idx] = v; t_i2[idx] = d;
                                    } else if (v < t_min3[idx]) {
                                        t_min5[idx] = t_min4[idx]; t_i4[idx] = t_i3[idx];
                                        t_min4[idx] = t_min3[idx]; t_min3[idx] = v; t_i3[idx] = d;
                                    } else if (v < t_min4[idx]) {
                                        t_min5[idx] = t_min4[idx]; t_min4[idx] = v; t_i4[idx] = d;
                                    } else if (v < t_min5[idx]) {
                                        t_min5[idx] = v;
                                    }
                                } else if (need_min4) {
                                    if (v < t_min1[idx]) {
                                        t_min4[idx] = t_min3[idx]; t_i3[idx] = t_i2[idx];
                                        t_min3[idx] = t_min2[idx]; t_i2[idx] = t_i1[idx];
                                        t_min2[idx] = t_min1[idx]; t_min1[idx] = v; t_i1[idx] = d;
                                    } else if (v < t_min2[idx]) {
                                        t_min4[idx] = t_min3[idx]; t_i3[idx] = t_i2[idx];
                                        t_min3[idx] = t_min2[idx]; t_min2[idx] = v; t_i2[idx] = d;
                                    } else if (v < t_min3[idx]) {
                                        t_min4[idx] = t_min3[idx]; t_min3[idx] = v; t_i3[idx] = d;
                                    } else if (v < t_min4[idx]) {
                                        t_min4[idx] = v;
                                    }
                                } else if (need_min3) {
                                    if (v < t_min1[idx]) {
                                        t_min3[idx] = t_min2[idx]; t_min2[idx] = t_min1[idx];
                                        t_i2[idx] = t_i1[idx]; t_min1[idx] = v; t_i1[idx] = d;
                                    } else if (v < t_min2[idx]) {
                                        t_min3[idx] = t_min2[idx]; t_min2[idx] = v; t_i2[idx] = d;
                                    } else if (v < t_min3[idx]) {
                                        t_min3[idx] = v;
                                    }
                                }
                                if (need_max4) {
                                    if (v > t_max1[idx]) {
                                        t_max4[idx] = t_max3[idx]; t_j3[idx] = t_j2[idx];
                                        t_max3[idx] = t_max2[idx]; t_j2[idx] = t_j1[idx];
                                        t_max2[idx] = t_max1[idx]; t_max1[idx] = v; t_j1[idx] = d;
                                    } else if (v > t_max2[idx]) {
                                        t_max4[idx] = t_max3[idx]; t_j3[idx] = t_j2[idx];
                                        t_max3[idx] = t_max2[idx]; t_max2[idx] = v; t_j2[idx] = d;
                                    } else if (v > t_max3[idx]) {
                                        t_max4[idx] = t_max3[idx]; t_max3[idx] = v; t_j3[idx] = d;
                                    } else if (v > t_max4[idx]) {
                                        t_max4[idx] = v;
                                    }
                                } else if (need_max3) {
                                    if (v > t_max1[idx]) {
                                        t_max3[idx] = t_max2[idx]; t_max2[idx] = t_max1[idx];
                                        t_j2[idx] = t_j1[idx]; t_max1[idx] = v; t_j1[idx] = d;
                                    } else if (v > t_max2[idx]) {
                                        t_max3[idx] = t_max2[idx]; t_max2[idx] = v; t_j2[idx] = d;
                                    } else if (v > t_max3[idx]) {
                                        t_max3[idx] = v;
                                    }
                                }
                                if (need_sum1) t_sum1[idx] += v;
                                if (need_sum2) t_sum2[idx] += v * v;
                            }
                        }
                    }

                    const bool is_call_m = is_call_arr[m] > 0.5;
                    const bool is_guar_m = is_guar_arr[m] > 0.5;
                    const bool need_cpn_m = !is_guar_m;
                    const bool need_opt_m = (m == M - 1);

                    const double b_ko = ko_arr[m];
                    const double blo_ko = std::min(b_ko, b_ko + args->ko_spread);
                    const double bhi_ko = std::max(b_ko, b_ko + args->ko_spread);
                    const double inv_w_ko = 1.0 / std::max(bhi_ko - blo_ko, 1e-12);
                    const double b_cp = cpn_arr[m];
                    const double blo_cp = std::min(b_cp, b_cp + eff_cpn_sp);
                    const double bhi_cp = std::max(b_cp, b_cp + eff_cpn_sp);
                    const double inv_w_cp = 1.0 / std::max(bhi_cp - blo_cp, 1e-12);
                    const double c_rate = cpn_rates[m];

                    const double* t_m1_p = need_min3 ? (t_min1.data() + m_off) : kDummyZero;
                    const double* t_m2_p = need_min3 ? (t_min2.data() + m_off) : kDummyZero;
                    const double* t_m3_p = need_min3 ? (t_min3.data() + m_off) : kDummyZero;
                    const double* t_m4_p = need_min4 ? (t_min4.data() + m_off) : kDummyZero;
                    const double* t_m5_p = need_min5 ? (t_min5.data() + m_off) : kDummyZero;
                    const std::int32_t* t_i1_p = need_min3 ? (t_i1.data() + m_off) : nullptr;
                    const std::int32_t* t_i2_p = need_min3 ? (t_i2.data() + m_off) : nullptr;
                    const std::int32_t* t_i3_p = need_min4 ? (t_i3.data() + m_off) : nullptr;
                    const std::int32_t* t_i4_p = need_min5 ? (t_i4.data() + m_off) : nullptr;
                    const double* t_x1_p = need_max3 ? (t_max1.data() + m_off) : kDummyZero;
                    const double* t_x2_p = need_max3 ? (t_max2.data() + m_off) : kDummyZero;
                    const double* t_x3_p = need_max3 ? (t_max3.data() + m_off) : kDummyZero;
                    const double* t_x4_p = need_max4 ? (t_max4.data() + m_off) : kDummyZero;
                    const std::int32_t* t_j1_p = need_max3 ? (t_j1.data() + m_off) : nullptr;
                    const std::int32_t* t_j2_p = need_max3 ? (t_j2.data() + m_off) : nullptr;
                    const std::int32_t* t_j3_p = need_max4 ? (t_j3.data() + m_off) : nullptr;
                    const double* t_s1_p = need_sum1 ? (t_sum1.data() + m_off) : kDummyZero;
                    const double* t_s2_p = need_sum2 ? (t_sum2.data() + m_off) : kDummyZero;

                    for (int bl = 0; bl < b_len; ++bl) {
                        const int b = b0 + bl;
                        const auto blu = static_cast<std::size_t>(bl);
                        double* const surv_b = surv_prev[blu].data();
                        double* const cmem_b = c_mem[blu].data();

                        const int aa = (asset_a != nullptr) ? static_cast<int>(asset_a[b]) : -1;
                        int bb = (asset_b != nullptr) ? static_cast<int>(asset_b[b]) : -1;
                        if (bb == aa) bb = -1;
                        const bool is_full = (aa == -2 && bump_a_ptr != nullptr);
                        const bool has_a = (!is_full && aa >= 0 && aa < D && bump_a_ptr != nullptr);
                        const bool has_b = (!is_full && bb >= 0 && bb < D && bump_b_ptr != nullptr);

                        const double* full_m_base = is_full
                            ? (bump_a_ptr + (static_cast<std::size_t>(b) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m)) * static_cast<std::size_t>(D) * N + n0)
                            : nullptr;
                        const double* pa_ptr = has_a
                            ? (bump_a_ptr + (static_cast<std::size_t>(b) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m)) * N + n0)
                            : kDummyZero;
                        const double* pb_ptr = has_b
                            ? (bump_b_ptr + (static_cast<std::size_t>(b) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m)) * N + n0)
                            : kDummyZero;
                        const double* base_a_ptr = has_a ? (perf_base + perf_m_off + static_cast<std::size_t>(aa) * N) : kDummyZero;
                        const double* base_b_ptr = has_b ? (perf_base + perf_m_off + static_cast<std::size_t>(bb) * N) : kDummyZero;
                        const double* base_ia_ptr = is_full ? (full_m_base + static_cast<std::size_t>(ia) * N)
                            : (perf_base != nullptr ? (perf_base + perf_m_off + static_cast<std::size_t>(ia) * N) : kDummyZero);
                        const double* base_ib_ptr = is_full ? (full_m_base + static_cast<std::size_t>(ib) * N)
                            : (perf_base != nullptr ? (perf_base + perf_m_off + static_cast<std::size_t>(ib) * N) : kDummyZero);

                        auto eval_code_tile = [&](int ecode, double* dst) noexcept {
                            if (is_full) {
                                dispatch_full_tile_code(
                                    ecode, D, t_len, N, ia, ib, inv_D,
                                    full_m_base, base_ia_ptr, base_ib_ptr, dst
                                );
                            } else {
                                dispatch_loo_l2o_tile_code(
                                    ecode, D, t_len, has_a, has_b, aa, bb, ia, ib, inv_D,
                                    t_m1_p, t_m2_p, t_m3_p, t_m4_p, t_m5_p, t_i1_p, t_i2_p, t_i3_p, t_i4_p,
                                    t_x1_p, t_x2_p, t_x3_p, t_x4_p, t_j1_p, t_j2_p, t_j3_p,
                                    t_s1_p, t_s2_p, pa_ptr, pb_ptr, base_a_ptr, base_b_ptr, base_ia_ptr, base_ib_ptr, dst
                                );
                            }
                        };

                        if (is_call_m || (need_cpn_m && eff_cpn == eff_call) || (need_opt_m && eff_opt == eff_call)) {
                            eval_code_tile(eff_call, w_call_buf.data());
                        }
                        const double* wc_ptr = w_call_buf.data();
                        const double* wcp_ptr = w_call_buf.data();
                        if (need_cpn_m && eff_cpn != eff_call) {
                            eval_code_tile(eff_cpn, w_cpn_buf.data());
                            wcp_ptr = w_cpn_buf.data();
                        }
                        const double* wo_ptr = w_call_buf.data();
                        const double* w_avg_ptr = w_avg_buf.data();
                        const double* w_wof_ptr = w_wof_buf.data();
                        if (need_opt_m) {
                            if (eff_opt == eff_call) {
                                wo_ptr = w_call_buf.data();
                            } else if (need_cpn_m && eff_opt == eff_cpn) {
                                wo_ptr = w_cpn_buf.data();
                            } else {
                                eval_code_tile(eff_opt, w_opt_buf.data());
                                wo_ptr = w_opt_buf.data();
                            }
                            if (has_one_star) {
                                const bool call_computed_best =
                                    (eff_call == 1) && (is_call_m || (need_cpn_m && eff_cpn == 1) || (eff_opt == 1));
                                const bool cpn_computed_best = (eff_cpn == 1) && need_cpn_m;
                                if (call_computed_best) {
                                    for (int t = 0; t < t_len; ++t) w_best_buf[static_cast<std::size_t>(t)] = wc_ptr[t];
                                } else if (cpn_computed_best) {
                                    for (int t = 0; t < t_len; ++t) w_best_buf[static_cast<std::size_t>(t)] = wcp_ptr[t];
                                } else if (eff_opt == 1) {
                                    for (int t = 0; t < t_len; ++t) w_best_buf[static_cast<std::size_t>(t)] = wo_ptr[t];
                                } else {
                                    eval_code_tile(1, w_best_buf.data());
                                }
                            }
                            if (is_long_basket) {
                                const bool call_computed_avg =
                                    (eff_call == 2) && (is_call_m || (need_cpn_m && eff_cpn == 2) || (eff_opt == 2));
                                const bool cpn_computed_avg = (eff_cpn == 2) && need_cpn_m;
                                if (call_computed_avg) {
                                    w_avg_ptr = wc_ptr;
                                } else if (cpn_computed_avg) {
                                    w_avg_ptr = wcp_ptr;
                                } else if (eff_opt == 2) {
                                    w_avg_ptr = wo_ptr;
                                } else {
                                    eval_code_tile(2, w_avg_buf.data());
                                    w_avg_ptr = w_avg_buf.data();
                                }

                                const bool call_computed_wof =
                                    (eff_call == 0) && (is_call_m || (need_cpn_m && eff_cpn == 0) || (eff_opt == 0));
                                const bool cpn_computed_wof = (eff_cpn == 0) && need_cpn_m;
                                if (call_computed_wof) {
                                    w_wof_ptr = wc_ptr;
                                } else if (cpn_computed_wof) {
                                    w_wof_ptr = wcp_ptr;
                                } else if (eff_opt == 0) {
                                    w_wof_ptr = wo_ptr;
                                } else {
                                    eval_code_tile(0, w_wof_buf.data());
                                    w_wof_ptr = w_wof_buf.data();
                                }
                            }
                        }

                        double s_surv = 0.0;
                        double s_p = 0.0;
                        double s_cpn = 0.0;

                        #pragma omp simd reduction(+:s_surv, s_p, s_cpn)
                        for (int t = 0; t < t_len; ++t) {
                            double g_ko = 0.0;
                            if (is_call_m) {
                                const double wc = wc_ptr[t];
                                g_ko = hard_ko ? (wc >= b_ko ? 1.0 : 0.0) : std::clamp((wc - blo_ko) * inv_w_ko, 0.0, 1.0);
                            }
                            double g_cpn = 1.0;
                            if (!is_guar_m) {
                                const double wcp = wcp_ptr[t];
                                if (hard_cpn) {
                                    const bool hit_dir = dir_down ? (wcp <= b_cp) : (wcp >= b_cp);
                                    g_cpn = type_out ? (hit_dir ? 0.0 : 1.0) : (hit_dir ? 1.0 : 0.0);
                                } else {
                                    const double ramp_up = std::clamp((wcp - blo_cp) * inv_w_cp, 0.0, 1.0);
                                    const double hit_g = dir_down ? (1.0 - ramp_up) : ramp_up;
                                    g_cpn = type_out ? (1.0 - hit_g) : hit_g;
                                }
                            }
                            const double eff_g_cpn = force_ac ? std::max(g_cpn, g_ko) : g_cpn;
                            const double sp = surv_b[t];
                            const double p_m = sp * g_ko;
                            double cpn_path = 0.0;
                            if (is_mem) {
                                const double eff_c = c_rate + cmem_b[t];
                                cpn_path = sp * eff_g_cpn * eff_c + p_m * ac_cpn;
                                cmem_b[t] = sp * eff_c * (1.0 - eff_g_cpn);
                            } else {
                                cpn_path = sp * eff_g_cpn * c_rate + p_m * ac_cpn;
                            }
                            s_surv += sp;
                            s_p += p_m;
                            s_cpn += cpn_path;
                            surv_b[t] = std::max(sp - p_m, 0.0);
                        }

                        double* b_acc = thr_acc + static_cast<std::size_t>(b) * bump_stride;
                        b_acc[m] += s_surv;
                        b_acc[M + m] += s_p;
                        b_acc[2 * M + m] += s_cpn;

                        if (need_opt_m) {
                            double s_term = 0.0;
                            double s_put = 0.0;
                            double s_call = 0.0;
                            double s_os = 0.0;
                            #pragma omp simd reduction(+:s_term, s_put, s_call, s_os)
                            for (int t = 0; t < t_len; ++t) {
                                const double sp_term = surv_b[t];
                                s_term += sp_term;
                                const double wo = wo_ptr[t];
                                const double star_ind = has_one_star
                                    ? centered_onestar_ramp(w_best_buf[static_cast<std::size_t>(t)], k_os, eps_os)
                                    : 0.0;
                                if (is_long_basket) {
                                    const double avg_t = w_avg_ptr[t];
                                    const double wof_t = w_wof_ptr[t];
                                    double ki_wof_k = i_ki_past;
                                    if (!has_ki || i_ki_past < 0.5) {
                                        const double g_ki_k = hard_ki
                                            ? (wof_t < k_put ? 1.0 : 0.0)
                                            : std::clamp((bhi_ki_islb - wof_t) * inv_w_ki_islb, 0.0, 1.0);
                                        ki_wof_k = has_ki ? std::max(i_ki_past, g_ki_k) : g_ki_k;
                                    }
                                    s_put += sp_term * ki_wof_k * (1.0 - star_ind) * (1.0 - avg_t);
                                } else {
                                    double i_ki = i_ki_past;
                                    if (has_ki) {
                                        const double g_ki = hard_ki
                                            ? (wo < b_ki ? 1.0 : 0.0)
                                            : std::clamp((bhi_ki - wo) * inv_w_ki, 0.0, 1.0);
                                        i_ki = std::max(i_ki, g_ki);
                                    }
                                    if (has_put) {
                                        double put_intr = std::max(k_put - wo, 0.0);
                                        if (has_lower_strike) {
                                            put_intr -= std::max(k_low - wo, 0.0);
                                        }
                                        const double ki_gate = has_ki ? i_ki : 1.0;
                                        s_put += sp_term * ki_gate * (1.0 - star_ind) * put_intr;
                                    }
                                }
                                if (has_part_call) {
                                    s_call += sp_term * std::max(wo - k_call, 0.0);
                                }
                                if (has_one_star && os_cpn != 0.0) {
                                    s_os += sp_term * star_ind * os_cpn;
                                }
                            }
                            b_acc[3 * M + 0] += s_term;
                            b_acc[3 * M + 1] += s_put;
                            b_acc[3 * M + 2] += s_call;
                            b_acc[3 * M + 3] += s_os;
                        }
                    }
                }
            }
        }
    }

    for (int tid = 0; tid < n_threads; ++tid) {
        const double* t_acc = per_thread_acc.data() + static_cast<std::size_t>(tid) * total_acc;
        for (std::size_t idx = 0; idx < total_acc; ++idx) {
            global_acc[idx] += t_acc[idx];
        }
    }

    for (int b = 0; b < B; ++b) {
        const double* b_acc = global_acc.data() + static_cast<std::size_t>(b) * bump_stride;
        const double* b_surv_sum = b_acc;
        const double* b_p_sum = b_acc + M;
        const double* b_cpn_sum = b_acc + 2 * M;
        const double surv_term_sum = b_acc[3 * M + 0];
        const double l_put_sum = b_acc[3 * M + 1];
        const double l_call_sum = b_acc[3 * M + 2];
        const double l_onestar_sum = b_acc[3 * M + 3];

        const double* df_b = df_pay_mat + static_cast<std::size_t>(b) * static_cast<std::size_t>(M);
        double notional_pv = 0.0;
        if (!no_notional) {
            for (int m = 0; m < M; ++m) {
                notional_pv += df_b[m] * (b_p_sum[m] * inv_N);
            }
            notional_pv += df_b[M - 1] * (surv_term_sum * inv_N);
        }
        double coupon_pv = 0.0;
        for (int m = 0; m < M; ++m) {
            coupon_pv += df_b[m] * (b_cpn_sum[m] * inv_N);
        }
        const double put_mult = is_long_basket ? -1.0 : -option_gearing;
        const double option_pv = df_b[M - 1] * (
            put_mult * (l_put_sum * inv_N)
            + part_call_rate * (l_call_sum * inv_N)
            + (l_onestar_sum * inv_N)
        );
        double funding_pv = 0.0;
        if (has_w_fund) {
            const double* wf_b = w_fund_mat + static_cast<std::size_t>(b) * static_cast<std::size_t>(M);
            for (int m = 0; m < M; ++m) {
                funding_pv += (b_surv_sum[m] * inv_N) * wf_b[m];
            }
        }
        const double in_tr = (pv_in_tr != nullptr) ? pv_in_tr[b] : 0.0;
        args->pv_out[b] = notional_pv + coupon_pv + option_pv - funding_pv + in_tr;
    }
}

std::int32_t crn_platen_sde_aad_cpp(const PlatenSdeAadArgs* args) noexcept {
    if (args == nullptr) return -1;
    if (args->d_assets <= 0 || args->m_rem <= 0 || args->n_sub <= 0 ||
        args->eff_r <= 0 || args->n_rep <= 0 || args->lv_nt <= 0 || args->lv_ny <= 0) {
        return -1;
    }
    const bool has_q_sub = (args->has_q_shift_sub != 0 && args->q_shift_sub != nullptr);
    if (args->lv_tables == nullptr || args->z_indep == nullptr || args->chol_base == nullptr ||
        args->tau_obs == nullptr || args->lv_t_grid == nullptr ||
        args->lv_y_min == nullptr || args->lv_y_max == nullptr) {
        return -1;
    }
    if (!has_q_sub && (args->r_native_obs == nullptr || args->r_tot_obs == nullptr)) {
        return -1;
    }
    const bool run_fwd = args->run_forward != 0;
    const bool run_bwd = args->run_backward != 0;
    if (run_fwd && args->lx_period_out == nullptr) {
        return -1;
    }
    const int vib_mode = (run_fwd && args->save_vib_pre != 0)
        ? ((args->save_vib_pre == 2 || args->vib_sub_idx < 0) ? 2 : 1)
        : 0;
    if (vib_mode != 0) {
        if (args->out_lx_pre == nullptr || args->out_j_pre == nullptr || args->out_sig_pre == nullptr ||
            args->vib_period_idx < 0 || args->vib_period_idx >= args->m_rem) {
            return -1;
        }
        if (vib_mode == 1 && (args->vib_sub_idx < 0 || args->vib_sub_idx >= args->n_sub)) {
            return -1;
        }
    }
    const double* lx_norm = (args->lx_norm_in != nullptr) ? args->lx_norm_in : args->lx_period_out;
    if (run_bwd && (args->bar_lx_post == nullptr || lx_norm == nullptr)) {
        return -1;
    }
    const bool compute_lv = run_bwd && (args->compute_lv_aad != 0);
    const bool compute_chol = run_bwd && (args->compute_chol_aad != 0);
    const bool compute_q = run_bwd && (args->compute_q_aad != 0);
    if ((compute_lv && args->bar_lv_out == nullptr) ||
        (compute_chol && args->bar_chol_out == nullptr) ||
        (compute_q && args->bar_q_shift_out == nullptr)) {
        return -1;
    }

    const int D = static_cast<int>(args->d_assets);
    const int M = static_cast<int>(args->m_rem);
    const int n_sub = static_cast<int>(args->n_sub);
    const int eff_r = static_cast<int>(args->eff_r);
    const int n_rep = static_cast<int>(args->n_rep);
    const int lv_nt = static_cast<int>(args->lv_nt);
    const int lv_ny = static_cast<int>(args->lv_ny);
    const int vib_period_idx = (vib_mode != 0) ? static_cast<int>(args->vib_period_idx) : -1;
    const int vib_sub_idx = (vib_mode != 0) ? static_cast<int>(args->vib_sub_idx) : -1;
    const double vib_j_pre_max = (args->vib_j_pre_max > 0.0) ? args->vib_j_pre_max : ((args->vib_j_pre_max < 0.0) ? 0.0 : 6.0);
    const double vib_dsig_dy_max = (args->vib_dsig_dy_max > 0.0) ? args->vib_dsig_dy_max : ((args->vib_dsig_dy_max < 0.0) ? 0.0 : 4.0);
    const bool clamp_vib_j = vib_j_pre_max > 0.0;
    const bool clamp_vib_dsig = vib_dsig_dy_max > 0.0;

    const std::size_t Du = static_cast<std::size_t>(D);
    const std::size_t Mu = static_cast<std::size_t>(M);
    const std::size_t Su = static_cast<std::size_t>(n_sub);
    const std::size_t Ru = static_cast<std::size_t>(eff_r);
    const std::size_t Nrep_u = static_cast<std::size_t>(n_rep);
    const std::size_t Ntot_u = Ru * Nrep_u;
    const std::size_t lv_nt_u = static_cast<std::size_t>(lv_nt);
    const std::size_t lv_ny_u = static_cast<std::size_t>(lv_ny);
    const std::size_t lv_asset_stride = lv_nt_u * lv_ny_u;

    const double t_max = args->lv_t_grid[lv_nt - 1];
    const double inv_dt_grid = (lv_nt > 1) ? (static_cast<double>(lv_nt - 1) / std::max(t_max, 1e-12)) : 0.0;
    const double max_u_t = (lv_nt > 1) ? (static_cast<double>(lv_nt - 1) - 1e-6) : 0.0;
    const double max_u_y = (lv_ny > 1) ? (static_cast<double>(lv_ny - 1) - 1e-6) : 0.0;

    std::vector<double> inv_dy_vec(Du, 0.0);
    for (int d = 0; d < D; ++d) {
        if (lv_ny > 1) {
            inv_dy_vec[static_cast<std::size_t>(d)] =
                static_cast<double>(lv_ny - 1) / std::max(args->lv_y_max[d] - args->lv_y_min[d], 1e-12);
        }
    }

    const std::size_t work_size = Mu * Su * Du * Ntot_u;
    OmpCallGuard omp_guard(work_size >= 8192);
    const int n_threads = omp_guard.active ? omp_guard.n_threads : 1;

    // Precompute correlated normals [M, n_sub, D, eff_r, n_rep] so inner Platen step is unit-stride along n_rep.
    std::vector<double> z_corr(work_size);
    const double* z_indep = args->z_indep;
    const double* chol_base = args->chol_base;
    #pragma omp parallel for collapse(2) schedule(static) if(omp_guard.active) num_threads(n_threads)
    for (int ms = 0; ms < M * n_sub; ++ms) {
        for (int r = 0; r < eff_r; ++r) {
            const std::size_t ms_u = static_cast<std::size_t>(ms);
            const std::size_t r_u = static_cast<std::size_t>(r);
            const std::size_t in_base = (ms_u * Ru + r_u) * Nrep_u * Du;
            const std::size_t out_ms_base = ms_u * Du * Ntot_u + r_u * Nrep_u;
            for (int d = 0; d < D; ++d) {
                const double* c_row = chol_base + static_cast<std::size_t>(d) * Du;
                double* dst = z_corr.data() + out_ms_base + static_cast<std::size_t>(d) * Ntot_u;
                for (int n = 0; n < n_rep; ++n) {
                    const double* zin = z_indep + in_base + static_cast<std::size_t>(n) * Du;
                    double acc = 0.0;
                    for (int k = 0; k < D; ++k) {
                        acc += zin[k] * c_row[k];
                    }
                    dst[n] = acc;
                }
            }
        }
    }

    if (run_fwd) {
        #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
        {
            std::vector<double> lx(Nrep_u);
            std::vector<double> ex_s(Nrep_u);
            std::vector<double> j_lx(vib_mode != 0 ? Nrep_u : 0u, 1.0);
            std::vector<double> lv_slice_start(lv_ny_u + 4u);
            std::vector<double> lv_slice_end(lv_ny_u + 4u);
            std::vector<double> lv_slice_mid(vib_mode != 0 ? (lv_ny_u + 4u) : 0u);
            std::vector<double> sig_sq_acc(vib_mode == 2 ? Nrep_u : 0u, 0.0);

            #pragma omp for collapse(2) schedule(static)
            for (int d = 0; d < D; ++d) {
                for (int r = 0; r < eff_r; ++r) {
                    const std::size_t du = static_cast<std::size_t>(d);
                    const std::size_t ru = static_cast<std::size_t>(r);
                    const double* lv_d = args->lv_tables + du * lv_asset_stride;
                    const double y_min_d = args->lv_y_min[d];
                    const double inv_dy_d = inv_dy_vec[du];

                    auto fill_lv_slice = [&](double t_val, double* dst) noexcept {
                        const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
                        const int it0 = static_cast<int>(u_t);
                        const int it1 = std::min(it0 + 1, lv_nt - 1);
                        const double wt = (lv_nt > 1) ? (u_t - static_cast<double>(it0)) : 0.0;
                        const double omt = 1.0 - wt;
                        const double* r0 = lv_d + static_cast<std::size_t>(it0) * lv_ny_u;
                        const double* r1 = lv_d + static_cast<std::size_t>(it1) * lv_ny_u;
                        #pragma omp simd
                        for (int iy = 0; iy < lv_ny; ++iy) {
                            dst[iy] = omt * r0[iy] + wt * r1[iy];
                        }
                        const double top_sig = dst[lv_ny - 1];
                        for (int pad = 0; pad < 4; ++pad) {
                            dst[lv_ny + pad] = top_sig;
                        }
                    };

                    auto lookup_scalar = [&](const double* sl, double y_v) noexcept -> double {
                        if (lv_ny <= 1) return sl[0];
                        const double u_y = std::clamp((y_v - y_min_d) * inv_dy_d, 0.0, max_u_y);
                        const int iy0 = static_cast<int>(u_y);
                        const int iy1 = std::min(iy0 + 1, lv_ny - 1);
                        const double wy = u_y - static_cast<double>(iy0);
                        return (1.0 - wy) * sl[iy0] + wy * sl[iy1];
                    };

                    auto lookup_scalar_with_deriv = [&](const double* sl, double y_v, double& d_out) noexcept -> double {
                        if (lv_ny <= 1) {
                            d_out = 0.0;
                            return sl[0];
                        }
                        const double u_raw = (y_v - y_min_d) * inv_dy_d;
                        const bool in_y = (u_raw > 0.0) && (u_raw < max_u_y);
                        const double u_y = std::clamp(u_raw, 0.0, max_u_y);
                        const int iy0 = static_cast<int>(u_y);
                        const int iy1 = std::min(iy0 + 1, lv_ny - 1);
                        const double wy = u_y - static_cast<double>(iy0);
                        const double s0 = sl[iy0];
                        const double s1 = sl[iy1];
                        d_out = in_y ? ((s1 - s0) * inv_dy_d) : 0.0;
                        return s0 + wy * (s1 - s0);
                    };

                    std::fill(lx.begin(), lx.end(), 0.0);
                    if (vib_mode != 0) {
                        std::fill(j_lx.begin(), j_lx.end(), 1.0);
                    }
                    double t_prev = 0.0;
                    double rt_prev = 0.0;
                    double rn_prev = 0.0;

                    for (int m = 0; m < M; ++m) {
                        const std::size_t mu = static_cast<std::size_t>(m);
                        const double t_cur = args->tau_obs[m];
                        const double d_tau = std::max(t_cur - t_prev, 1e-10);
                        const double dt = d_tau / static_cast<double>(n_sub);
                        const double sqrt_dt = std::sqrt(dt);
                        const double rt_cur = has_q_sub ? 0.0 : args->r_tot_obs[du * Mu + mu];
                        const double rn_cur = has_q_sub ? 0.0 : args->r_native_obs[du * Mu + mu];
                        const double drt = rt_cur - rt_prev;
                        const double drn = rn_cur - rn_prev;

                        const bool track_j_m = (vib_mode == 1 && m <= vib_period_idx) || (vib_mode == 2 && m < vib_period_idx);

                        if (vib_mode == 2 && m == vib_period_idx) {
                            const std::size_t v_off = du * Ntot_u + ru * Nrep_u;
                            double* v_lx_dst = args->out_lx_pre + v_off;
                            double* v_j_dst = args->out_j_pre + v_off;
                            double* v_sig_dst = args->out_sig_pre + v_off;
                            std::fill(sig_sq_acc.begin(), sig_sq_acc.end(), 0.0);
                            for (int sv_k = 0; sv_k < n_sub; ++sv_k) {
                                const double alpha_k = (static_cast<double>(sv_k) + 0.5) / static_cast<double>(n_sub);
                                const double t_mid_k = t_prev + alpha_k * d_tau;
                                const double q_k = has_q_sub
                                    ? args->q_shift_sub[(du * Mu + mu) * Su + static_cast<std::size_t>(sv_k)]
                                    : ((rt_prev + alpha_k * drt) - (rn_prev + alpha_k * drn));
                                fill_lv_slice(t_mid_k, lv_slice_mid.data());
                                for (int n = 0; n < n_rep; ++n) {
                                    const double sk = lookup_scalar(lv_slice_mid.data(), lx[static_cast<std::size_t>(n)] + q_k);
                                    sig_sq_acc[static_cast<std::size_t>(n)] += sk * sk;
                                }
                            }
                            const double inv_nsub_f = 1.0 / static_cast<double>(n_sub);
                            for (int n = 0; n < n_rep; ++n) {
                                const std::size_t nu = static_cast<std::size_t>(n);
                                v_lx_dst[n] = lx[nu];
                                const double jv = j_lx[nu];
                                v_j_dst[n] = clamp_vib_j ? std::clamp(jv, -vib_j_pre_max, vib_j_pre_max) : jv;
                                v_sig_dst[n] = std::sqrt(sig_sq_acc[nu] * inv_nsub_f);
                            }
                        }

                        for (int s = 0; s < n_sub; ++s) {
                            const double alpha_s = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
                            const double t_mid = t_prev + alpha_s * d_tau;
                            const double t_start = std::max(t_mid - 0.5 * dt, 0.0);
                            const double t_end = t_start + dt;
                            const double q_mid = has_q_sub
                                ? args->q_shift_sub[(du * Mu + mu) * Su + static_cast<std::size_t>(s)]
                                : ((rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn));

                            if (vib_mode == 1 && m == vib_period_idx && s == vib_sub_idx) {
                                fill_lv_slice(t_mid, lv_slice_mid.data());
                                const std::size_t v_off = du * Ntot_u + ru * Nrep_u;
                                double* v_lx_dst = args->out_lx_pre + v_off;
                                double* v_j_dst = args->out_j_pre + v_off;
                                double* v_sig_dst = args->out_sig_pre + v_off;
                                for (int n = 0; n < n_rep; ++n) {
                                    const std::size_t nu = static_cast<std::size_t>(n);
                                    const double lxv = lx[nu];
                                    const double jv = j_lx[nu];
                                    v_lx_dst[n] = lxv;
                                    v_j_dst[n] = clamp_vib_j ? std::clamp(jv, -vib_j_pre_max, vib_j_pre_max) : jv;
                                    v_sig_dst[n] = lookup_scalar(lv_slice_mid.data(), lxv + q_mid);
                                }
                            }

                            fill_lv_slice(t_start, lv_slice_start.data());
                            fill_lv_slice(t_end, lv_slice_end.data());

                            const double* zd_ptr = z_corr.data() +
                                ((mu * Su + static_cast<std::size_t>(s)) * Du + du) * Ntot_u + ru * Nrep_u;
                            const bool track_j_step = track_j_m && !(vib_mode == 1 && m == vib_period_idx && s >= vib_sub_idx);

                            int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                            const __m256d v_y_min = _mm256_set1_pd(y_min_d);
                            const __m256d v_inv_dy = _mm256_set1_pd(inv_dy_d);
                            const __m256d v_zero = _mm256_setzero_pd();
                            const __m256d v_max_uy = _mm256_set1_pd(max_u_y);
                            const __m256d v_q_mid = _mm256_set1_pd(q_mid);
                            const __m256d v_dt = _mm256_set1_pd(dt);
                            const __m256d v_sdt = _mm256_set1_pd(sqrt_dt);
                            const __m256d v_half = _mm256_set1_pd(0.5);
                            const __m256d v_nhalf = _mm256_set1_pd(-0.5);
                            const __m256d v_quarter = _mm256_set1_pd(0.25);
                            const __m256d v_two = _mm256_set1_pd(2.0);
                            const __m256d v_one = _mm256_set1_pd(1.0);
                            const double* sl0 = lv_slice_start.data();
                            const double* sl1 = lv_slice_end.data();

                            auto lookup_slice_pd = [&](const double* sl, __m256d y_v) noexcept -> __m256d {
                                if (lv_ny <= 1) return _mm256_set1_pd(sl[0]);
                                __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                                __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                                __m256d u_fl = _mm256_floor_pd(u_y);
                                __m128i iy0 = _mm256_cvttpd_epi32(u_fl);
                                __m256d wy = _mm256_sub_pd(u_y, u_fl);
                                __m256d s0 = _mm256_i32gather_pd(sl, iy0, 8);
                                __m256d s1 = _mm256_i32gather_pd(sl + 1, iy0, 8);
                                return _mm256_fmadd_pd(wy, _mm256_sub_pd(s1, s0), s0);
                            };

                            if (!track_j_step) {
                                for (; n + 4 <= n_rep; n += 4) {
                                    __m256d lx_v = _mm256_loadu_pd(lx.data() + n);
                                    __m256d zd = _mm256_loadu_pd(zd_ptr + n);
                                    __m256d y0 = _mm256_add_pd(lx_v, v_q_mid);
                                    __m256d sig0 = lookup_slice_pd(sl0, y0);
                                    __m256d a0 = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sig0, sig0));
                                    __m256d dw = _mm256_mul_pd(v_sdt, zd);
                                    __m256d sig0_sdt = _mm256_mul_pd(sig0, v_sdt);
                                    __m256d base_y = _mm256_fmadd_pd(a0, v_dt, y0);
                                    __m256d sigh = lookup_slice_pd(sl1, _mm256_fmadd_pd(sig0, dw, base_y));
                                    __m256d sigp = lookup_slice_pd(sl1, _mm256_add_pd(base_y, sig0_sdt));
                                    __m256d sigm = lookup_slice_pd(sl1, _mm256_sub_pd(base_y, sig0_sdt));
                                    __m256d ah = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sigh, sigh));
                                    __m256d drift = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0, ah), v_dt));
                                    __m256d sum_sig = _mm256_add_pd(_mm256_add_pd(sigp, sigm), _mm256_mul_pd(v_two, sig0));
                                    __m256d diff = _mm256_mul_pd(v_quarter, _mm256_mul_pd(sum_sig, dw));
                                    __m256d mil = _mm256_mul_pd(
                                        v_quarter,
                                        _mm256_mul_pd(_mm256_sub_pd(sigp, sigm), _mm256_mul_pd(v_sdt, _mm256_fmsub_pd(zd, zd, v_one)))
                                    );
                                    _mm256_storeu_pd(lx.data() + n, _mm256_add_pd(lx_v, _mm256_add_pd(drift, _mm256_add_pd(diff, mil))));
                                }
                            } else {
                                const __m256d v_dsig_hi = _mm256_set1_pd(vib_dsig_dy_max);
                                const __m256d v_dsig_lo = _mm256_set1_pd(-vib_dsig_dy_max);
                                const __m256d v_j_hi = _mm256_set1_pd(vib_j_pre_max);
                                const __m256d v_j_lo = _mm256_set1_pd(-vib_j_pre_max);
                                auto lookup_slice_pd_deriv = [&](const double* sl, __m256d y_v, __m256d& d_out) noexcept -> __m256d {
                                    if (lv_ny <= 1) {
                                        d_out = v_zero;
                                        return _mm256_set1_pd(sl[0]);
                                    }
                                    __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                                    __m256d in_y = _mm256_and_pd(
                                        _mm256_cmp_pd(uy_raw, v_zero, _CMP_GT_OQ),
                                        _mm256_cmp_pd(uy_raw, v_max_uy, _CMP_LT_OQ)
                                    );
                                    __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                                    __m256d u_fl = _mm256_floor_pd(u_y);
                                    __m128i iy0 = _mm256_cvttpd_epi32(u_fl);
                                    __m256d wy = _mm256_sub_pd(u_y, u_fl);
                                    __m256d s0 = _mm256_i32gather_pd(sl, iy0, 8);
                                    __m256d s1 = _mm256_i32gather_pd(sl + 1, iy0, 8);
                                    __m256d ds = _mm256_sub_pd(s1, s0);
                                    __m256d d_raw = _mm256_and_pd(in_y, _mm256_mul_pd(ds, v_inv_dy));
                                    d_out = clamp_vib_dsig ? _mm256_min_pd(_mm256_max_pd(d_raw, v_dsig_lo), v_dsig_hi) : d_raw;
                                    return _mm256_fmadd_pd(wy, ds, s0);
                                };
                                for (; n + 4 <= n_rep; n += 4) {
                                    __m256d lx_v = _mm256_loadu_pd(lx.data() + n);
                                    __m256d zd = _mm256_loadu_pd(zd_ptr + n);
                                    __m256d y0 = _mm256_add_pd(lx_v, v_q_mid);
                                    __m256d d0, dh, dp, dm;
                                    __m256d sig0 = lookup_slice_pd_deriv(sl0, y0, d0);
                                    __m256d a0 = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sig0, sig0));
                                    __m256d dw = _mm256_mul_pd(v_sdt, zd);
                                    __m256d sig0_sdt = _mm256_mul_pd(sig0, v_sdt);
                                    __m256d base_y = _mm256_fmadd_pd(a0, v_dt, y0);
                                    __m256d sigh = lookup_slice_pd_deriv(sl1, _mm256_fmadd_pd(sig0, dw, base_y), dh);
                                    __m256d sigp = lookup_slice_pd_deriv(sl1, _mm256_add_pd(base_y, sig0_sdt), dp);
                                    __m256d sigm = lookup_slice_pd_deriv(sl1, _mm256_sub_pd(base_y, sig0_sdt), dm);
                                    __m256d ah = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sigh, sigh));
                                    __m256d zc2m1 = _mm256_fmsub_pd(zd, zd, v_one);
                                    __m256d drift = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0, ah), v_dt));
                                    __m256d sum_sig = _mm256_add_pd(_mm256_add_pd(sigp, sigm), _mm256_mul_pd(v_two, sig0));
                                    __m256d diff = _mm256_mul_pd(v_quarter, _mm256_mul_pd(sum_sig, dw));
                                    __m256d mil = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_sub_pd(sigp, sigm), _mm256_mul_pd(v_sdt, zc2m1)));
                                    _mm256_storeu_pd(lx.data() + n, _mm256_add_pd(lx_v, _mm256_add_pd(drift, _mm256_add_pd(diff, mil))));

                                    __m256d d_a0_dx = _mm256_mul_pd(_mm256_sub_pd(v_zero, sig0), d0);
                                    __m256d d_base = _mm256_fmadd_pd(d_a0_dx, v_dt, v_one);
                                    __m256d d_hat_dx = _mm256_fmadd_pd(d0, dw, d_base);
                                    __m256d d_p_dx = _mm256_fmadd_pd(d0, v_sdt, d_base);
                                    __m256d d_m_dx = _mm256_fnmadd_pd(d0, v_sdt, d_base);
                                    __m256d dp_x = _mm256_mul_pd(dp, d_p_dx);
                                    __m256d dm_x = _mm256_mul_pd(dm, d_m_dx);
                                    __m256d j_t1 = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_fnmadd_pd(_mm256_mul_pd(sigh, dh), d_hat_dx, d_a0_dx), v_dt));
                                    __m256d j_t2 = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(dp_x, dm_x), _mm256_mul_pd(v_two, d0)), dw));
                                    __m256d j_t3 = _mm256_mul_pd(v_quarter, _mm256_mul_pd(_mm256_sub_pd(dp_x, dm_x), _mm256_mul_pd(v_sdt, zc2m1)));
                                    __m256d step_j = _mm256_add_pd(v_one, _mm256_add_pd(j_t1, _mm256_add_pd(j_t2, j_t3)));
                                    if (clamp_vib_j) {
                                        step_j = _mm256_min_pd(_mm256_max_pd(step_j, _mm256_set1_pd(0.25)), _mm256_set1_pd(2.5));
                                    }
                                    __m256d j_next = _mm256_mul_pd(_mm256_loadu_pd(j_lx.data() + n), step_j);
                                    if (clamp_vib_j) {
                                        j_next = _mm256_min_pd(_mm256_max_pd(j_next, v_j_lo), v_j_hi);
                                    }
                                    _mm256_storeu_pd(j_lx.data() + n, j_next);
                                }
                            }
#endif
                            for (; n < n_rep; ++n) {
                                const double zd = zd_ptr[n];
                                const double lx_v = lx[static_cast<std::size_t>(n)];
                                const double dw = sqrt_dt * zd;
                                const double zc2m1 = zd * zd - 1.0;
                                if (!track_j_step) {
                                    const double sig0 = lookup_scalar(lv_slice_start.data(), lx_v + q_mid);
                                    const double a0 = -0.5 * sig0 * sig0;
                                    const double base_y = lx_v + a0 * dt + q_mid;
                                    const double sigh = lookup_scalar(lv_slice_end.data(), base_y + sig0 * dw);
                                    const double sigp = lookup_scalar(lv_slice_end.data(), base_y + sig0 * sqrt_dt);
                                    const double sigm = lookup_scalar(lv_slice_end.data(), base_y - sig0 * sqrt_dt);
                                    const double ah = -0.5 * sigh * sigh;
                                    lx[static_cast<std::size_t>(n)] = lx_v
                                        + 0.5 * (a0 + ah) * dt
                                        + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                                        + 0.25 * (sigp - sigm) * sqrt_dt * zc2m1;
                                } else {
                                    double d0 = 0.0, dh = 0.0, dp = 0.0, dm = 0.0;
                                    const double sig0 = lookup_scalar_with_deriv(lv_slice_start.data(), lx_v + q_mid, d0);
                                    const double a0 = -0.5 * sig0 * sig0;
                                    const double base_y = lx_v + a0 * dt + q_mid;
                                    const double sigh = lookup_scalar_with_deriv(lv_slice_end.data(), base_y + sig0 * dw, dh);
                                    const double sigp = lookup_scalar_with_deriv(lv_slice_end.data(), base_y + sig0 * sqrt_dt, dp);
                                    const double sigm = lookup_scalar_with_deriv(lv_slice_end.data(), base_y - sig0 * sqrt_dt, dm);
                                    if (clamp_vib_dsig) {
                                        d0 = std::clamp(d0, -vib_dsig_dy_max, vib_dsig_dy_max);
                                        dh = std::clamp(dh, -vib_dsig_dy_max, vib_dsig_dy_max);
                                        dp = std::clamp(dp, -vib_dsig_dy_max, vib_dsig_dy_max);
                                        dm = std::clamp(dm, -vib_dsig_dy_max, vib_dsig_dy_max);
                                    }
                                    const double ah = -0.5 * sigh * sigh;
                                    lx[static_cast<std::size_t>(n)] = lx_v
                                        + 0.5 * (a0 + ah) * dt
                                        + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                                        + 0.25 * (sigp - sigm) * sqrt_dt * zc2m1;

                                    const double d_a0_dx = -sig0 * d0;
                                    const double d_hat_dx = 1.0 + d_a0_dx * dt + d0 * dw;
                                    const double d_p_dx = 1.0 + d_a0_dx * dt + d0 * sqrt_dt;
                                    const double d_m_dx = 1.0 + d_a0_dx * dt - d0 * sqrt_dt;
                                    double step_j = 1.0
                                        + 0.5 * (d_a0_dx - sigh * dh * d_hat_dx) * dt
                                        + 0.25 * (dp * d_p_dx + dm * d_m_dx + 2.0 * d0) * dw
                                        + 0.25 * (dp * d_p_dx - dm * d_m_dx) * sqrt_dt * zc2m1;
                                    if (clamp_vib_j) {
                                        step_j = std::clamp(step_j, 0.25, 2.5);
                                    }
                                    const double j_next = j_lx[static_cast<std::size_t>(n)] * step_j;
                                    j_lx[static_cast<std::size_t>(n)] = clamp_vib_j
                                        ? std::clamp(j_next, -vib_j_pre_max, vib_j_pre_max)
                                        : j_next;
                                }
                            }
                        }

                        double max_lx = -std::numeric_limits<double>::infinity();
                        for (int n = 0; n < n_rep; ++n) {
                            max_lx = std::max(max_lx, lx[static_cast<std::size_t>(n)]);
                        }
                        double sum_ex = 0.0;
                        int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                        const __m256d v_max_lx = _mm256_set1_pd(max_lx);
                        __m256d v_sum_ex = _mm256_setzero_pd();
                        for (; n + 4 <= n_rep; n += 4) {
                            __m256d ex = v_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(lx.data() + n), v_max_lx));
                            _mm256_storeu_pd(ex_s.data() + n, ex);
                            v_sum_ex = _mm256_add_pd(v_sum_ex, ex);
                        }
                        alignas(32) double s4_ex[4];
                        _mm256_store_pd(s4_ex, v_sum_ex);
                        sum_ex = s4_ex[0] + s4_ex[1] + s4_ex[2] + s4_ex[3];
#endif
                        for (; n < n_rep; ++n) {
                            const double ev = std::exp(lx[static_cast<std::size_t>(n)] - max_lx);
                            ex_s[static_cast<std::size_t>(n)] = ev;
                            sum_ex += ev;
                        }
                        const double mean_rep = std::max(sum_ex / static_cast<double>(n_rep), 1e-30);
                        const double shift = max_lx + std::log(mean_rep);
                        const std::size_t out_off = (mu * Du + du) * Ntot_u + ru * Nrep_u;
                        double* lp_dst = args->lx_period_out + out_off;
                        double* nx_dst = (args->norm_x_out != nullptr) ? (args->norm_x_out + out_off) : nullptr;
                        const double inv_mean = 1.0 / mean_rep;
                        #pragma omp simd
                        for (int n_s = 0; n_s < n_rep; ++n_s) {
                            const double normed = lx[static_cast<std::size_t>(n_s)] - shift;
                            lx[static_cast<std::size_t>(n_s)] = normed;
                            lp_dst[n_s] = normed;
                            if (nx_dst != nullptr) {
                                nx_dst[n_s] = ex_s[static_cast<std::size_t>(n_s)] * inv_mean;
                            }
                        }
                        if (track_j_m && m < vib_period_idx) {
                            const double inv_rep = 1.0 / static_cast<double>(n_rep);
                            const double omega_scale = inv_mean * inv_rep;
                            double mean_j_dev = 0.0;
                            for (int n_j = 0; n_j < n_rep; ++n_j) {
                                const std::size_t nu = static_cast<std::size_t>(n_j);
                                mean_j_dev += ex_s[nu] * (j_lx[nu] - 1.0);
                            }
                            mean_j_dev *= omega_scale;
                            for (int n_j = 0; n_j < n_rep; ++n_j) {
                                const std::size_t nu = static_cast<std::size_t>(n_j);
                                const double j_norm = 1.0 + ((j_lx[nu] - 1.0) - mean_j_dev);
                                j_lx[nu] = clamp_vib_j ? std::clamp(j_norm, -vib_j_pre_max, vib_j_pre_max) : j_norm;
                            }
                        }
                        t_prev = t_cur;
                        rt_prev = rt_cur;
                        rn_prev = rn_cur;
                    }
                }
            }
        }
    }

    if (run_bwd) {
        const std::size_t n_lv_tot = Du * lv_asset_stride;
        const std::size_t n_chol_tot = Du * Du;
        const std::size_t n_q_tot = Du * Mu * Su;
        std::vector<double> per_thr_lv(compute_lv ? (static_cast<std::size_t>(n_threads) * n_lv_tot) : 0u, 0.0);
        std::vector<double> per_thr_chol(compute_chol ? (static_cast<std::size_t>(n_threads) * n_chol_tot) : 0u, 0.0);
        std::vector<double> per_thr_q(compute_q ? (static_cast<std::size_t>(n_threads) * n_q_tot) : 0u, 0.0);

        const double inv_n_rep = 1.0 / static_cast<double>(n_rep);

        #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
        {
#ifdef _OPENMP
            const int tid = omp_guard.active ? omp_get_thread_num() : 0;
#else
            const int tid = 0;
#endif
            double* thr_lv = compute_lv ? (per_thr_lv.data() + static_cast<std::size_t>(tid) * n_lv_tot) : nullptr;
            double* thr_chol = compute_chol ? (per_thr_chol.data() + static_cast<std::size_t>(tid) * n_chol_tot) : nullptr;
            double* thr_q = compute_q ? (per_thr_q.data() + static_cast<std::size_t>(tid) * n_q_tot) : nullptr;

            std::vector<double> bar_x(Nrep_u);
            std::vector<double> bar_tot(Nrep_u);
            std::vector<double> lx_sub_pre(Su * Nrep_u);
            std::vector<double> lx_run(Nrep_u);
            std::vector<double> lv_slice_start(lv_ny_u + 4u);
            std::vector<double> lv_slice_end(lv_ny_u + 4u);
            std::vector<double> bar_sl0(compute_lv ? lv_ny_u : 0u, 0.0);
            std::vector<double> bar_sl1(compute_lv ? lv_ny_u : 0u, 0.0);

            #pragma omp for collapse(2) schedule(static)
            for (int d = 0; d < D; ++d) {
                for (int r = 0; r < eff_r; ++r) {
                    const std::size_t du = static_cast<std::size_t>(d);
                    const std::size_t ru = static_cast<std::size_t>(r);
                    const double* lv_d = args->lv_tables + du * lv_asset_stride;
                    double* thr_lv_d = compute_lv ? (thr_lv + du * lv_asset_stride) : nullptr;
                    const double y_min_d = args->lv_y_min[d];
                    const double inv_dy_d = inv_dy_vec[du];

                    auto fill_lv_slice_meta = [&](double t_val, double* dst, int& it0_out, int& it1_out, double& wt_out) noexcept {
                        const double u_t = (lv_nt > 1) ? std::clamp(t_val * inv_dt_grid, 0.0, max_u_t) : 0.0;
                        const int it0 = static_cast<int>(u_t);
                        const int it1 = std::min(it0 + 1, lv_nt - 1);
                        const double wt = (lv_nt > 1) ? (u_t - static_cast<double>(it0)) : 0.0;
                        const double omt = 1.0 - wt;
                        const double* r0 = lv_d + static_cast<std::size_t>(it0) * lv_ny_u;
                        const double* r1 = lv_d + static_cast<std::size_t>(it1) * lv_ny_u;
                        #pragma omp simd
                        for (int iy = 0; iy < lv_ny; ++iy) {
                            dst[iy] = omt * r0[iy] + wt * r1[iy];
                        }
                        const double top_sig = dst[lv_ny - 1];
                        for (int pad = 0; pad < 4; ++pad) {
                            dst[lv_ny + pad] = top_sig;
                        }
                        it0_out = it0;
                        it1_out = it1;
                        wt_out = wt;
                    };

                    std::fill(bar_x.begin(), bar_x.end(), 0.0);

                    for (int m = M - 1; m >= 0; --m) {
                        const std::size_t mu = static_cast<std::size_t>(m);
                        const double t_prev = (m == 0) ? 0.0 : args->tau_obs[m - 1];
                        const double t_cur = args->tau_obs[m];
                        const double d_tau = std::max(t_cur - t_prev, 1e-10);
                        const double dt = d_tau / static_cast<double>(n_sub);
                        const double sqrt_dt = std::sqrt(dt);

                        const double rt_prev = (has_q_sub || m == 0) ? 0.0 : args->r_tot_obs[du * Mu + static_cast<std::size_t>(m - 1)];
                        const double rt_cur = has_q_sub ? 0.0 : args->r_tot_obs[du * Mu + mu];
                        const double rn_prev = (has_q_sub || m == 0) ? 0.0 : args->r_native_obs[du * Mu + static_cast<std::size_t>(m - 1)];
                        const double rn_cur = has_q_sub ? 0.0 : args->r_native_obs[du * Mu + mu];
                        const double drt = rt_cur - rt_prev;
                        const double drn = rn_cur - rn_prev;

                        const std::size_t m_off = (mu * Du + du) * Ntot_u + ru * Nrep_u;
                        const double* bar_post_p = args->bar_lx_post + m_off;
                        const double* lx_norm_m = lx_norm + m_off;

                        double sum_bar_tot = 0.0;
                        for (int n = 0; n < n_rep; ++n) {
                            const double bt = bar_post_p[n] + bar_x[static_cast<std::size_t>(n)];
                            bar_tot[static_cast<std::size_t>(n)] = bt;
                            sum_bar_tot += bt;
                        }
                        {
                            int n_o = 0;
#if defined(__AVX2__) && defined(__FMA__)
                            const __m256d v_sum_tot = _mm256_set1_pd(sum_bar_tot * inv_n_rep);
                            for (; n_o + 4 <= n_rep; n_o += 4) {
                                const __m256d bt_v = _mm256_loadu_pd(bar_tot.data() + n_o);
                                const __m256d ex_v = v_exp_pd(_mm256_loadu_pd(lx_norm_m + n_o));
                                _mm256_storeu_pd(bar_x.data() + n_o, _mm256_fnmadd_pd(ex_v, v_sum_tot, bt_v));
                            }
#endif
                            for (; n_o < n_rep; ++n_o) {
                                const double omega = std::exp(lx_norm_m[n_o]) * inv_n_rep;
                                bar_x[static_cast<std::size_t>(n_o)] = bar_tot[static_cast<std::size_t>(n_o)] - omega * sum_bar_tot;
                            }
                        }

                        if (m == 0) {
                            std::fill(lx_run.begin(), lx_run.end(), 0.0);
                        } else {
                            const double* prev_norm = lx_norm + (static_cast<std::size_t>(m - 1) * Du + du) * Ntot_u + ru * Nrep_u;
                            std::copy(prev_norm, prev_norm + Nrep_u, lx_run.begin());
                        }

                        // Forward rematerialization of substep pre-states within period m (AVX2).
                        for (int s = 0; s < n_sub; ++s) {
                            const std::size_t su = static_cast<std::size_t>(s);
                            std::copy(lx_run.begin(), lx_run.end(), lx_sub_pre.data() + su * Nrep_u);
                            const double alpha_s = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
                            const double t_start = std::max(t_prev + alpha_s * d_tau - 0.5 * dt, 0.0);
                            const double t_end = t_start + dt;
                            const double q_mid = has_q_sub
                                ? args->q_shift_sub[(du * Mu + mu) * Su + su]
                                : ((rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn));
                            const double* zd_ptr = z_corr.data() + ((mu * Su + su) * Du + du) * Ntot_u + ru * Nrep_u;

                            int it0_tmp = 0, it1_tmp = 0;
                            double wt_tmp = 0.0;
                            fill_lv_slice_meta(t_start, lv_slice_start.data(), it0_tmp, it1_tmp, wt_tmp);
                            fill_lv_slice_meta(t_end, lv_slice_end.data(), it0_tmp, it1_tmp, wt_tmp);

                            int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                            const __m256d v_y_min = _mm256_set1_pd(y_min_d);
                            const __m256d v_inv_dy = _mm256_set1_pd(inv_dy_d);
                            const __m256d v_zero = _mm256_setzero_pd();
                            const __m256d v_max_uy = _mm256_set1_pd(max_u_y);
                            const __m256d v_q_mid = _mm256_set1_pd(q_mid);
                            const __m256d v_dt = _mm256_set1_pd(dt);
                            const __m256d v_sdt = _mm256_set1_pd(sqrt_dt);
                            const __m256d v_half = _mm256_set1_pd(0.5);
                            const __m256d v_nhalf = _mm256_set1_pd(-0.5);
                            const __m256d v_quarter = _mm256_set1_pd(0.25);
                            const __m256d v_two = _mm256_set1_pd(2.0);
                            const __m256d v_one = _mm256_set1_pd(1.0);
                            const double* sl0 = lv_slice_start.data();
                            const double* sl1 = lv_slice_end.data();

                            auto lookup_slice_pd = [&](const double* sl, __m256d y_v) noexcept -> __m256d {
                                if (lv_ny <= 1) return _mm256_set1_pd(sl[0]);
                                __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                                __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                                __m256d u_fl = _mm256_floor_pd(u_y);
                                __m128i iy0 = _mm256_cvttpd_epi32(u_fl);
                                __m256d wy = _mm256_sub_pd(u_y, u_fl);
                                __m256d s0 = _mm256_i32gather_pd(sl, iy0, 8);
                                __m256d s1 = _mm256_i32gather_pd(sl + 1, iy0, 8);
                                return _mm256_fmadd_pd(wy, _mm256_sub_pd(s1, s0), s0);
                            };

                            for (; n + 4 <= n_rep; n += 4) {
                                __m256d lxv = _mm256_loadu_pd(lx_run.data() + n);
                                __m256d zd = _mm256_loadu_pd(zd_ptr + n);
                                __m256d y0 = _mm256_add_pd(lxv, v_q_mid);
                                __m256d sig0 = lookup_slice_pd(sl0, y0);
                                __m256d a0 = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sig0, sig0));
                                __m256d dw = _mm256_mul_pd(v_sdt, zd);
                                __m256d sig0_sdt = _mm256_mul_pd(sig0, v_sdt);
                                __m256d base_y = _mm256_fmadd_pd(a0, v_dt, y0);
                                __m256d sigh = lookup_slice_pd(sl1, _mm256_fmadd_pd(sig0, dw, base_y));
                                __m256d sigp = lookup_slice_pd(sl1, _mm256_add_pd(base_y, sig0_sdt));
                                __m256d sigm = lookup_slice_pd(sl1, _mm256_sub_pd(base_y, sig0_sdt));
                                __m256d ah = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sigh, sigh));
                                __m256d drift = _mm256_mul_pd(v_half, _mm256_mul_pd(_mm256_add_pd(a0, ah), v_dt));
                                __m256d sum_sig = _mm256_add_pd(_mm256_add_pd(sigp, sigm), _mm256_mul_pd(v_two, sig0));
                                __m256d diff = _mm256_mul_pd(v_quarter, _mm256_mul_pd(sum_sig, dw));
                                __m256d mil = _mm256_mul_pd(
                                    v_quarter,
                                    _mm256_mul_pd(_mm256_sub_pd(sigp, sigm), _mm256_mul_pd(v_sdt, _mm256_fmsub_pd(zd, zd, v_one)))
                                );
                                _mm256_storeu_pd(lx_run.data() + n, _mm256_add_pd(lxv, _mm256_add_pd(drift, _mm256_add_pd(diff, mil))));
                            }
#endif
                            auto lookup_scalar_bwd = [&](const double* sl, double y_v) noexcept -> double {
                                if (lv_ny <= 1) return sl[0];
                                const double u_y = std::clamp((y_v - y_min_d) * inv_dy_d, 0.0, max_u_y);
                                const int iy0 = static_cast<int>(u_y);
                                const int iy1 = std::min(iy0 + 1, lv_ny - 1);
                                const double wy = u_y - static_cast<double>(iy0);
                                return (1.0 - wy) * sl[iy0] + wy * sl[iy1];
                            };
                            for (; n < n_rep; ++n) {
                                const double lxv = lx_run[static_cast<std::size_t>(n)];
                                const double zd = zd_ptr[n];
                                const double dw = sqrt_dt * zd;
                                const double sig0 = lookup_scalar_bwd(lv_slice_start.data(), lxv + q_mid);
                                const double a0 = -0.5 * sig0 * sig0;
                                const double base_y = lxv + a0 * dt + q_mid;
                                const double sigh = lookup_scalar_bwd(lv_slice_end.data(), base_y + sig0 * dw);
                                const double sigp = lookup_scalar_bwd(lv_slice_end.data(), base_y + sig0 * sqrt_dt);
                                const double sigm = lookup_scalar_bwd(lv_slice_end.data(), base_y - sig0 * sqrt_dt);
                                lx_run[static_cast<std::size_t>(n)] = lxv
                                    + 0.5 * (a0 - 0.5 * sigh * sigh) * dt
                                    + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                                    + 0.25 * (sigp - sigm) * sqrt_dt * (zd * zd - 1.0);
                            }
                        }

                        // Reverse-mode substep adjoints (AVX2 + slice-buffered LV scatter).
                        for (int s = n_sub - 1; s >= 0; --s) {
                            const std::size_t su = static_cast<std::size_t>(s);
                            const double* lx_pre = lx_sub_pre.data() + su * Nrep_u;
                            const double alpha_s = (static_cast<double>(s) + 0.5) / static_cast<double>(n_sub);
                            const double t_start = std::max(t_prev + alpha_s * d_tau - 0.5 * dt, 0.0);
                            const double t_end = t_start + dt;
                            const double q_mid = has_q_sub
                                ? args->q_shift_sub[(du * Mu + mu) * Su + su]
                                : ((rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn));
                            const double* zd_ptr = z_corr.data() + ((mu * Su + su) * Du + du) * Ntot_u + ru * Nrep_u;
                            const double* zin_step = z_indep + ((mu * Su + su) * Ru + ru) * Nrep_u * Du;

                            int it0_0 = 0, it1_0 = 0, it0_1 = 0, it1_1 = 0;
                            double wt_0 = 0.0, wt_1 = 0.0;
                            fill_lv_slice_meta(t_start, lv_slice_start.data(), it0_0, it1_0, wt_0);
                            fill_lv_slice_meta(t_end, lv_slice_end.data(), it0_1, it1_1, wt_1);
                            if (compute_lv) {
                                std::fill(bar_sl0.begin(), bar_sl0.end(), 0.0);
                                std::fill(bar_sl1.begin(), bar_sl1.end(), 0.0);
                            }

                            double q_acc_step = 0.0;
                            int n = 0;
#if defined(__AVX2__) && defined(__FMA__)
                            const __m256d v_y_min = _mm256_set1_pd(y_min_d);
                            const __m256d v_inv_dy = _mm256_set1_pd(inv_dy_d);
                            const __m256d v_zero = _mm256_setzero_pd();
                            const __m256d v_max_uy = _mm256_set1_pd(max_u_y);
                            const __m256d v_q_mid = _mm256_set1_pd(q_mid);
                            const __m256d v_dt = _mm256_set1_pd(dt);
                            const __m256d v_sdt = _mm256_set1_pd(sqrt_dt);
                            const __m256d v_half = _mm256_set1_pd(0.5);
                            const __m256d v_nhalf = _mm256_set1_pd(-0.5);
                            const __m256d v_quarter = _mm256_set1_pd(0.25);
                            const __m256d v_two = _mm256_set1_pd(2.0);
                            const __m256d v_one = _mm256_set1_pd(1.0);
                            const __m256d v_dsig_hi = _mm256_set1_pd(vib_dsig_dy_max);
                            const __m256d v_dsig_lo = _mm256_set1_pd(-vib_dsig_dy_max);
                            const double* sl0 = lv_slice_start.data();
                            const double* sl1 = lv_slice_end.data();
                            __m256d v_q_acc = _mm256_setzero_pd();

                            auto eval_slice_bwd_pd = [&](
                                const double* sl,
                                __m256d y_v,
                                __m256d& dy_out,
                                __m128i& iy0_out,
                                __m256d& wy_out
                            ) noexcept -> __m256d {
                                if (lv_ny <= 1) {
                                    dy_out = v_zero;
                                    iy0_out = _mm_setzero_si128();
                                    wy_out = v_zero;
                                    return _mm256_set1_pd(sl[0]);
                                }
                                __m256d uy_raw = _mm256_mul_pd(_mm256_sub_pd(y_v, v_y_min), v_inv_dy);
                                __m256d in_y = _mm256_and_pd(
                                    _mm256_cmp_pd(uy_raw, v_zero, _CMP_GT_OQ),
                                    _mm256_cmp_pd(uy_raw, v_max_uy, _CMP_LT_OQ)
                                );
                                __m256d u_y = _mm256_min_pd(_mm256_max_pd(uy_raw, v_zero), v_max_uy);
                                __m256d u_fl = _mm256_floor_pd(u_y);
                                iy0_out = _mm256_cvttpd_epi32(u_fl);
                                wy_out = _mm256_sub_pd(u_y, u_fl);
                                __m256d s0 = _mm256_i32gather_pd(sl, iy0_out, 8);
                                __m256d s1 = _mm256_i32gather_pd(sl + 1, iy0_out, 8);
                                __m256d ds = _mm256_sub_pd(s1, s0);
                                __m256d d_raw = _mm256_and_pd(in_y, _mm256_mul_pd(ds, v_inv_dy));
                                dy_out = clamp_vib_dsig ? _mm256_min_pd(_mm256_max_pd(d_raw, v_dsig_lo), v_dsig_hi) : d_raw;
                                return _mm256_fmadd_pd(wy_out, ds, s0);
                            };

                            for (; n + 4 <= n_rep; n += 4) {
                                const __m256d lxp = _mm256_loadu_pd(lx_pre + n);
                                const __m256d zd = _mm256_loadu_pd(zd_ptr + n);
                                const __m256d dw = _mm256_mul_pd(v_sdt, zd);
                                const __m256d y0 = _mm256_add_pd(lxp, v_q_mid);

                                __m256d dy0, wy0;
                                __m128i iy0_0;
                                const __m256d sig0 = eval_slice_bwd_pd(sl0, y0, dy0, iy0_0, wy0);
                                const __m256d a0 = _mm256_mul_pd(v_nhalf, _mm256_mul_pd(sig0, sig0));
                                const __m256d base_y = _mm256_fmadd_pd(a0, v_dt, y0);
                                const __m256d sig0_sdt = _mm256_mul_pd(sig0, v_sdt);

                                __m256d dyh, wyh, dyp, wyp, dym, wym;
                                __m128i iy0_h, iy0_p, iy0_m;
                                const __m256d sigh = eval_slice_bwd_pd(sl1, _mm256_fmadd_pd(sig0, dw, base_y), dyh, iy0_h, wyh);
                                const __m256d sigp = eval_slice_bwd_pd(sl1, _mm256_add_pd(base_y, sig0_sdt), dyp, iy0_p, wyp);
                                const __m256d sigm = eval_slice_bwd_pd(sl1, _mm256_sub_pd(base_y, sig0_sdt), dym, iy0_m, wym);

                                const __m256d bx = _mm256_loadu_pd(bar_x.data() + n);
                                const __m256d zc2m1 = _mm256_fmsub_pd(zd, zd, v_one);
                                const __m256d bar_lvh = _mm256_mul_pd(bx, _mm256_mul_pd(v_nhalf, _mm256_mul_pd(v_dt, sigh)));
                                const __m256d sdt_z = _mm256_mul_pd(v_sdt, zc2m1);
                                const __m256d bar_lvp = _mm256_mul_pd(bx, _mm256_mul_pd(v_quarter, _mm256_add_pd(dw, sdt_z)));
                                const __m256d bar_lvm = _mm256_mul_pd(bx, _mm256_mul_pd(v_quarter, _mm256_sub_pd(dw, sdt_z)));
                                const __m256d bar_lv0_dir = _mm256_mul_pd(bx, _mm256_fmadd_pd(_mm256_mul_pd(v_nhalf, v_dt), sig0, _mm256_mul_pd(v_half, dw)));

                                const __m256d bar_lx_hat = _mm256_mul_pd(bar_lvh, dyh);
                                const __m256d bar_lx_p = _mm256_mul_pd(bar_lvp, dyp);
                                const __m256d bar_lx_m = _mm256_mul_pd(bar_lvm, dym);

                                if (compute_chol) {
                                    const __m256d sum_pm = _mm256_add_pd(_mm256_add_pd(sigp, sigm), _mm256_mul_pd(v_two, sig0));
                                    const __m256d diff_pm = _mm256_sub_pd(sigp, sigm);
                                    const __m256d t_zc = _mm256_fmadd_pd(
                                        _mm256_mul_pd(v_quarter, sum_pm), v_sdt,
                                        _mm256_mul_pd(_mm256_mul_pd(v_half, diff_pm), _mm256_mul_pd(v_sdt, zd))
                                    );
                                    const __m256d bar_zc = _mm256_fmadd_pd(bx, t_zc, _mm256_mul_pd(bar_lx_hat, sig0_sdt));
                                    alignas(32) double bzc_arr[4];
                                    _mm256_store_pd(bzc_arr, bar_zc);
                                    double* chol_row = thr_chol + du * Du;
                                    for (int lane = 0; lane < 4; ++lane) {
                                        const double bzc_l = bzc_arr[lane];
                                        const double* zin_n = zin_step + static_cast<std::size_t>(n + lane) * Du;
                                        for (int j = 0; j < D; ++j) {
                                            chol_row[j] += bzc_l * zin_n[j];
                                        }
                                    }
                                }

                                const __m256d bar_a0 = _mm256_mul_pd(v_dt, _mm256_add_pd(bar_lx_hat, _mm256_add_pd(bar_lx_p, bar_lx_m)));
                                const __m256d bar_lv0 = _mm256_add_pd(
                                    _mm256_fnmadd_pd(bar_a0, sig0, bar_lv0_dir),
                                    _mm256_fmadd_pd(bar_lx_hat, dw, _mm256_mul_pd(_mm256_sub_pd(bar_lx_p, bar_lx_m), v_sdt))
                                );

                                if (compute_lv) {
                                    alignas(16) std::int32_t i0_a[4], ih_a[4], ip_a[4], im_a[4];
                                    alignas(32) double w0_a[4], wh_a[4], wp_a[4], wm_a[4];
                                    alignas(32) double b0_a[4], bh_a[4], bp_a[4], bm_a[4];
                                    _mm_store_si128(reinterpret_cast<__m128i*>(i0_a), iy0_0);
                                    _mm_store_si128(reinterpret_cast<__m128i*>(ih_a), iy0_h);
                                    _mm_store_si128(reinterpret_cast<__m128i*>(ip_a), iy0_p);
                                    _mm_store_si128(reinterpret_cast<__m128i*>(im_a), iy0_m);
                                    _mm256_store_pd(w0_a, wy0);
                                    _mm256_store_pd(wh_a, wyh);
                                    _mm256_store_pd(wp_a, wyp);
                                    _mm256_store_pd(wm_a, wym);
                                    _mm256_store_pd(b0_a, bar_lv0);
                                    _mm256_store_pd(bh_a, bar_lvh);
                                    _mm256_store_pd(bp_a, bar_lvp);
                                    _mm256_store_pd(bm_a, bar_lvm);
                                    const int iy_inc = (lv_ny > 1) ? 1 : 0;
                                    for (int lane = 0; lane < 4; ++lane) {
                                        const std::size_t c0_0 = static_cast<std::size_t>(i0_a[lane]);
                                        const std::size_t c1_0 = c0_0 + static_cast<std::size_t>(iy_inc);
                                        const double bv0 = b0_a[lane];
                                        const double wyv0 = w0_a[lane];
                                        bar_sl0[c0_0] += bv0 * (1.0 - wyv0);
                                        bar_sl0[c1_0] += bv0 * wyv0;

                                        const std::size_t c0_h = static_cast<std::size_t>(ih_a[lane]);
                                        const std::size_t c1_h = c0_h + static_cast<std::size_t>(iy_inc);
                                        const double bvh = bh_a[lane];
                                        const double wyvh = wh_a[lane];
                                        bar_sl1[c0_h] += bvh * (1.0 - wyvh);
                                        bar_sl1[c1_h] += bvh * wyvh;

                                        const std::size_t c0_p = static_cast<std::size_t>(ip_a[lane]);
                                        const std::size_t c1_p = c0_p + static_cast<std::size_t>(iy_inc);
                                        const double bvp = bp_a[lane];
                                        const double wyvp = wp_a[lane];
                                        bar_sl1[c0_p] += bvp * (1.0 - wyvp);
                                        bar_sl1[c1_p] += bvp * wyvp;

                                        const std::size_t c0_m = static_cast<std::size_t>(im_a[lane]);
                                        const std::size_t c1_m = c0_m + static_cast<std::size_t>(iy_inc);
                                        const double bvm = bm_a[lane];
                                        const double wyvm = wm_a[lane];
                                        bar_sl1[c0_m] += bvm * (1.0 - wyvm);
                                        bar_sl1[c1_m] += bvm * wyvm;
                                    }
                                }

                                const __m256d bar_y = _mm256_fmadd_pd(bar_lv0, dy0, _mm256_add_pd(bar_lx_hat, _mm256_add_pd(bar_lx_p, bar_lx_m)));
                                if (compute_q) {
                                    v_q_acc = _mm256_add_pd(v_q_acc, bar_y);
                                }
                                _mm256_storeu_pd(bar_x.data() + n, _mm256_add_pd(bx, bar_y));
                            }
                            if (compute_q) {
                                alignas(32) double q4[4];
                                _mm256_store_pd(q4, v_q_acc);
                                q_acc_step += q4[0] + q4[1] + q4[2] + q4[3];
                            }
#endif
                            auto eval_slice_bwd_scalar = [&](const double* sl, double y_v, double& dy_out, int& iy0_out, int& iy1_out, double& wy_out) noexcept -> double {
                                if (lv_ny <= 1) {
                                    dy_out = 0.0;
                                    iy0_out = 0;
                                    iy1_out = 0;
                                    wy_out = 0.0;
                                    return sl[0];
                                }
                                const double u_raw = (y_v - y_min_d) * inv_dy_d;
                                const bool in_y = (u_raw > 0.0) && (u_raw < max_u_y);
                                const double u_y = std::clamp(u_raw, 0.0, max_u_y);
                                iy0_out = static_cast<int>(u_y);
                                iy1_out = std::min(iy0_out + 1, lv_ny - 1);
                                wy_out = u_y - static_cast<double>(iy0_out);
                                const double s0 = sl[iy0_out];
                                const double s1 = sl[iy1_out];
                                const double d_raw = in_y ? ((s1 - s0) * inv_dy_d) : 0.0;
                                dy_out = clamp_vib_dsig ? std::clamp(d_raw, -vib_dsig_dy_max, vib_dsig_dy_max) : d_raw;
                                return s0 + wy_out * (s1 - s0);
                            };
                            for (; n < n_rep; ++n) {
                                const double lxp = lx_pre[n];
                                const double zd = zd_ptr[n];
                                const double dw = sqrt_dt * zd;
                                double dy0 = 0.0, wy0 = 0.0, dyh = 0.0, wyh = 0.0, dyp = 0.0, wyp = 0.0, dym = 0.0, wym = 0.0;
                                int iy0_0 = 0, iy1_0 = 0, iy0_h = 0, iy1_h = 0, iy0_p = 0, iy1_p = 0, iy0_m = 0, iy1_m = 0;
                                const double sig0 = eval_slice_bwd_scalar(lv_slice_start.data(), lxp + q_mid, dy0, iy0_0, iy1_0, wy0);
                                const double a0 = -0.5 * sig0 * sig0;
                                const double base_y = lxp + a0 * dt + q_mid;
                                const double sigh = eval_slice_bwd_scalar(lv_slice_end.data(), base_y + sig0 * dw, dyh, iy0_h, iy1_h, wyh);
                                const double sigp = eval_slice_bwd_scalar(lv_slice_end.data(), base_y + sig0 * sqrt_dt, dyp, iy0_p, iy1_p, wyp);
                                const double sigm = eval_slice_bwd_scalar(lv_slice_end.data(), base_y - sig0 * sqrt_dt, dym, iy0_m, iy1_m, wym);

                                const double bx = bar_x[static_cast<std::size_t>(n)];
                                const double zc2m1 = zd * zd - 1.0;
                                const double bar_lvh = bx * (-0.5 * dt * sigh);
                                const double bar_lvp = bx * (0.25 * (dw + sqrt_dt * zc2m1));
                                const double bar_lvm = bx * (0.25 * (dw - sqrt_dt * zc2m1));
                                const double bar_lv0_dir = bx * (-0.5 * dt * sig0 + 0.5 * dw);

                                const double bar_lx_hat = bar_lvh * dyh;
                                const double bar_lx_p = bar_lvp * dyp;
                                const double bar_lx_m = bar_lvm * dym;

                                if (compute_chol) {
                                    const double bar_zc =
                                        bx * (0.25 * (sigp + sigm + 2.0 * sig0) * sqrt_dt + 0.5 * (sigp - sigm) * sqrt_dt * zd)
                                        + bar_lx_hat * (sig0 * sqrt_dt);
                                    const double* zin_n = zin_step + static_cast<std::size_t>(n) * Du;
                                    double* chol_row = thr_chol + du * Du;
                                    for (int j = 0; j < D; ++j) {
                                        chol_row[j] += bar_zc * zin_n[j];
                                    }
                                }

                                const double bar_a0 = dt * (bar_lx_hat + bar_lx_p + bar_lx_m);
                                const double bar_lv0 = bar_lv0_dir + bar_a0 * (-sig0) + bar_lx_hat * dw + (bar_lx_p - bar_lx_m) * sqrt_dt;

                                if (compute_lv) {
                                    bar_sl0[static_cast<std::size_t>(iy0_0)] += bar_lv0 * (1.0 - wy0);
                                    bar_sl0[static_cast<std::size_t>(iy1_0)] += bar_lv0 * wy0;
                                    bar_sl1[static_cast<std::size_t>(iy0_h)] += bar_lvh * (1.0 - wyh);
                                    bar_sl1[static_cast<std::size_t>(iy1_h)] += bar_lvh * wyh;
                                    bar_sl1[static_cast<std::size_t>(iy0_p)] += bar_lvp * (1.0 - wyp);
                                    bar_sl1[static_cast<std::size_t>(iy1_p)] += bar_lvp * wyp;
                                    bar_sl1[static_cast<std::size_t>(iy0_m)] += bar_lvm * (1.0 - wym);
                                    bar_sl1[static_cast<std::size_t>(iy1_m)] += bar_lvm * wym;
                                }

                                const double bar_y = bar_lv0 * dy0 + bar_lx_hat + bar_lx_p + bar_lx_m;
                                if (compute_q) {
                                    q_acc_step += bar_y;
                                }
                                bar_x[static_cast<std::size_t>(n)] = bx + bar_y;
                            }

                            if (compute_lv) {
                                const double omt0 = 1.0 - wt_0;
                                double* dst0_a = thr_lv_d + static_cast<std::size_t>(it0_0) * lv_ny_u;
                                double* dst0_b = thr_lv_d + static_cast<std::size_t>(it1_0) * lv_ny_u;
                                const double omt1 = 1.0 - wt_1;
                                double* dst1_a = thr_lv_d + static_cast<std::size_t>(it0_1) * lv_ny_u;
                                double* dst1_b = thr_lv_d + static_cast<std::size_t>(it1_1) * lv_ny_u;
                                for (int iy = 0; iy < lv_ny; ++iy) {
                                    const double v0 = bar_sl0[static_cast<std::size_t>(iy)];
                                    dst0_a[iy] += v0 * omt0;
                                    dst0_b[iy] += v0 * wt_0;
                                    const double v1 = bar_sl1[static_cast<std::size_t>(iy)];
                                    dst1_a[iy] += v1 * omt1;
                                    dst1_b[iy] += v1 * wt_1;
                                }
                            }
                            if (compute_q) {
                                thr_q[(du * Mu + mu) * Su + su] += q_acc_step;
                            }
                        }
                    }
                }
            }
        }

        if (compute_lv) {
            std::fill(args->bar_lv_out, args->bar_lv_out + n_lv_tot, 0.0);
            for (int tid = 0; tid < n_threads; ++tid) {
                const double* src = per_thr_lv.data() + static_cast<std::size_t>(tid) * n_lv_tot;
                for (std::size_t idx = 0; idx < n_lv_tot; ++idx) {
                    args->bar_lv_out[idx] += src[idx];
                }
            }
        }
        if (compute_chol) {
            std::fill(args->bar_chol_out, args->bar_chol_out + n_chol_tot, 0.0);
            for (int tid = 0; tid < n_threads; ++tid) {
                const double* src = per_thr_chol.data() + static_cast<std::size_t>(tid) * n_chol_tot;
                for (std::size_t idx = 0; idx < n_chol_tot; ++idx) {
                    args->bar_chol_out[idx] += src[idx];
                }
            }
        }
        if (compute_q) {
            std::fill(args->bar_q_shift_out, args->bar_q_shift_out + n_q_tot, 0.0);
            for (int tid = 0; tid < n_threads; ++tid) {
                const double* src = per_thr_q.data() + static_cast<std::size_t>(tid) * n_q_tot;
                for (std::size_t idx = 0; idx < n_q_tot; ++idx) {
                    args->bar_q_shift_out[idx] += src[idx];
                }
            }
        }
    }

    return 0;
}

std::int32_t crn_generate_sobol_normals_cpp(const SobolGenArgs* args) noexcept {
    if (args == nullptr) return -1;
    if (args->d_assets <= 0 || args->m_rem <= 0 || args->n_sub <= 0 ||
        args->eff_r <= 0 || args->n_rep <= 0) {
        return -1;
    }
    if (args->z_indep_out == nullptr && args->z_corr_out == nullptr) {
        return -1;
    }
    if (args->apply_bb_time != 0 && args->bb_ortho_mat == nullptr) {
        return -1;
    }
    if (args->apply_corr != 0 && args->z_corr_out != nullptr && args->chol_mat == nullptr) {
        return -1;
    }

    const int D = static_cast<int>(args->d_assets);
    const int M = static_cast<int>(args->m_rem);
    const int n_sub = static_cast<int>(args->n_sub);
    const int eff_r = static_cast<int>(args->eff_r);
    const int n_rep = static_cast<int>(args->n_rep);
    const int sobol_dim = static_cast<int>(std::max<std::int64_t>(args->sobol_dim, 0));
    const int n_bits = static_cast<int>((args->n_bits > 0 && args->n_bits <= 32) ? args->n_bits : 30);

    const std::size_t Du = static_cast<std::size_t>(D);
    const std::size_t Mu = static_cast<std::size_t>(M);
    const std::size_t Su = static_cast<std::size_t>(n_sub);
    const std::size_t Ru = static_cast<std::size_t>(eff_r);
    const std::size_t Nrep_u = static_cast<std::size_t>(n_rep);
    const std::size_t n_bits_u = static_cast<std::size_t>(n_bits);
    const std::uint32_t bit_mask = (n_bits >= 32) ? 0xFFFFFFFFu : ((1u << n_bits) - 1u);
    const double inv_pow2 = std::ldexp(1.0, -n_bits);
    const double inv_n_sub = 1.0 / static_cast<double>(n_sub);
    const double inv_sqrt_sub = std::sqrt(inv_n_sub);

    const bool use_gray = args->use_gray_code != 0;
    const bool apply_bb = args->apply_bb_time != 0;
    const bool apply_corr = (args->apply_corr != 0) && (args->z_corr_out != nullptr);
    const bool antithetic = args->antithetic != 0;
    const int n_gen = antithetic ? ((n_rep + 1) / 2) : n_rep;

    const std::size_t n_dims_total = Mu * (Su + 1u) * Du;
    const std::size_t out_total = Mu * Su * Ru * Nrep_u * Du;

    std::vector<double> indep_tmp;
    double* z_indep = args->z_indep_out;
    if (z_indep == nullptr) {
        indep_tmp.resize(out_total);
        z_indep = indep_tmp.data();
    }

    auto fill_dim_sobol_normals = [&](int dim, int r, double* dst_n) noexcept {
        if (args->dir_numbers != nullptr && dim < sobol_dim) {
            const std::uint32_t* dir_row = args->dir_numbers + static_cast<std::size_t>(dim) * n_bits_u;
            const std::uint32_t shift_val = (args->digital_shifts != nullptr)
                ? (args->digital_shifts[static_cast<std::size_t>(r) * n_dims_total + static_cast<std::size_t>(dim)] & bit_mask)
                : 0u;
            if (use_gray && n_gen > 0) {
                std::uint32_t acc = shift_val;
                double u0 = std::clamp((static_cast<double>(acc) + 0.5) * inv_pow2, 1e-12, 1.0 - 1e-12);
                dst_n[0] = ndtri_std(u0);
                for (int n_idx = 1; n_idx < n_gen; ++n_idx) {
                    const int bit_idx = std::countr_zero(static_cast<std::uint32_t>(n_idx));
                    if (bit_idx < n_bits) {
                        acc ^= dir_row[bit_idx];
                    }
                    const double u = std::clamp((static_cast<double>(acc) + 0.5) * inv_pow2, 1e-12, 1.0 - 1e-12);
                    dst_n[n_idx] = ndtri_std(u);
                }
            } else {
                for (int n_idx = 0; n_idx < n_gen; ++n_idx) {
                    std::uint32_t code = static_cast<std::uint32_t>(n_idx);
                    std::uint32_t acc = shift_val;
                    for (int b = 0; b < n_bits && code != 0u; ++b, code >>= 1) {
                        if ((code & 1u) != 0u) {
                            acc ^= dir_row[b];
                        }
                    }
                    const double u = std::clamp((static_cast<double>(acc) + 0.5) * inv_pow2, 1e-12, 1.0 - 1e-12);
                    dst_n[n_idx] = ndtri_std(u);
                }
            }
            return;
        }
        const std::uint64_t base_seed = static_cast<std::uint64_t>(0x9E3779B97F4A7C15ULL)
            ^ (static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(0xBF58476D1CE4E5B9ULL))
            ^ (static_cast<std::uint64_t>(dim) * static_cast<std::uint64_t>(0x94D049BB133111EBULL));
        for (int n_idx = 0; n_idx < n_gen; ++n_idx) {
            std::uint64_t rng_state = base_seed ^ static_cast<std::uint64_t>(n_idx);
            const double u = u64_to_unit_open(splitmix64_next(rng_state));
            dst_n[n_idx] = ndtri_std(u);
        }
    };

    OmpCallGuard omp_guard(out_total >= 8192);
    const int n_threads = omp_guard.active ? omp_guard.n_threads : 1;
    const std::size_t n_gen_u = static_cast<std::size_t>(n_gen);

    // Step 1: Generate Sobol + Brownian-bridge substeps into z_indep [M, n_sub, eff_r, n_rep, D].
    #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
    {
        std::vector<double> zc_buf(n_gen_u);
        std::vector<double> zf_buf(Su * n_gen_u);

        #pragma omp for collapse(2) schedule(static)
        for (int r = 0; r < eff_r; ++r) {
            for (int m = 0; m < M; ++m) {
                const std::size_t ru = static_cast<std::size_t>(r);
                const std::size_t mu = static_cast<std::size_t>(m);
                const int coarse_base = m * D;
                const int fine_base = (M + m * n_sub) * D;

                for (int d = 0; d < D; ++d) {
                    const std::size_t du = static_cast<std::size_t>(d);
                    fill_dim_sobol_normals(coarse_base + d, r, zc_buf.data());
                    if (n_sub == 1) {
                        const std::size_t base_out = ((mu * Su + 0u) * Ru + ru) * Nrep_u * Du + du;
                        for (int n = 0; n < n_gen; ++n) {
                            z_indep[base_out + static_cast<std::size_t>(n) * Du] = zc_buf[static_cast<std::size_t>(n)];
                        }
                    } else {
                        for (int s = 0; s < n_sub; ++s) {
                            fill_dim_sobol_normals(fine_base + s * D + d, r, zf_buf.data() + static_cast<std::size_t>(s) * n_gen_u);
                        }
                        for (int n = 0; n < n_gen; ++n) {
                            const std::size_t nu = static_cast<std::size_t>(n);
                            double z_sum = 0.0;
                            for (int s = 0; s < n_sub; ++s) {
                                z_sum += zf_buf[static_cast<std::size_t>(s) * n_gen_u + nu];
                            }
                            const double delta = zc_buf[nu] * inv_sqrt_sub - z_sum * inv_n_sub;
                            for (int s = 0; s < n_sub; ++s) {
                                const std::size_t out_off = ((mu * Su + static_cast<std::size_t>(s)) * Ru + ru) * Nrep_u * Du + nu * Du + du;
                                z_indep[out_off] = zf_buf[static_cast<std::size_t>(s) * n_gen_u + nu] + delta;
                            }
                        }
                    }
                }

                if (antithetic && n_gen < n_rep) {
                    for (int s = 0; s < n_sub; ++s) {
                        const std::size_t step_off = ((mu * Su + static_cast<std::size_t>(s)) * Ru + ru) * Nrep_u * Du;
                        for (int n = n_gen; n < n_rep; ++n) {
                            const int src_n = n - n_gen;
                            double* dst_n = z_indep + step_off + static_cast<std::size_t>(n) * Du;
                            const double* src_p = z_indep + step_off + static_cast<std::size_t>(src_n) * Du;
                            for (int d = 0; d < D; ++d) {
                                dst_n[d] = -src_p[d];
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Optional temporal rotation across M periods via bb_ortho_mat [M, M].
    if (apply_bb && M > 1) {
        const double* bb = args->bb_ortho_mat;
        const std::size_t rep_d = Nrep_u * Du;
        #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
        {
            std::vector<double> z_mean(Mu * rep_d);
            std::vector<double> z_rot(Mu * rep_d);

            #pragma omp for schedule(static)
            for (int r = 0; r < eff_r; ++r) {
                const std::size_t ru = static_cast<std::size_t>(r);
                for (int m = 0; m < M; ++m) {
                    const std::size_t mu = static_cast<std::size_t>(m);
                    double* zm = z_mean.data() + mu * rep_d;
                    const double* src0 = z_indep + ((mu * Su + 0u) * Ru + ru) * rep_d;
                    for (std::size_t i = 0; i < rep_d; ++i) {
                        zm[i] = src0[i];
                    }
                    for (int s = 1; s < n_sub; ++s) {
                        const double* srcs = z_indep + ((mu * Su + static_cast<std::size_t>(s)) * Ru + ru) * rep_d;
                        for (std::size_t i = 0; i < rep_d; ++i) {
                            zm[i] += srcs[i];
                        }
                    }
                    for (std::size_t i = 0; i < rep_d; ++i) {
                        zm[i] *= inv_n_sub;
                    }
                }

                for (int m = 0; m < M; ++m) {
                    const std::size_t mu = static_cast<std::size_t>(m);
                    const double* bb_row = bb + mu * Mu;
                    double* zr = z_rot.data() + mu * rep_d;
                    const double b0 = bb_row[0];
                    const double* zm0 = z_mean.data();
                    for (std::size_t i = 0; i < rep_d; ++i) {
                        zr[i] = b0 * zm0[i];
                    }
                    for (int k = 1; k < M; ++k) {
                        const double bk = bb_row[k];
                        const double* zmk = z_mean.data() + static_cast<std::size_t>(k) * rep_d;
                        for (std::size_t i = 0; i < rep_d; ++i) {
                            zr[i] += bk * zmk[i];
                        }
                    }
                }

                for (int m = 0; m < M; ++m) {
                    const std::size_t mu = static_cast<std::size_t>(m);
                    const double* zm = z_mean.data() + mu * rep_d;
                    const double* zr = z_rot.data() + mu * rep_d;
                    for (int s = 0; s < n_sub; ++s) {
                        double* dst = z_indep + ((mu * Su + static_cast<std::size_t>(s)) * Ru + ru) * rep_d;
                        for (std::size_t i = 0; i < rep_d; ++i) {
                            dst[i] += zr[i] - zm[i];
                        }
                    }
                }
            }
        }
    }

    // Step 3: Optional cross-asset Cholesky correlation z_corr_out = z_indep @ chol_mat.T.
    if (args->z_corr_out != nullptr) {
        if (apply_corr) {
            const double* chol = args->chol_mat;
            double* z_corr_out = args->z_corr_out;
            #pragma omp parallel for collapse(2) schedule(static) if(omp_guard.active) num_threads(n_threads)
            for (int ms = 0; ms < M * n_sub; ++ms) {
                for (int r = 0; r < eff_r; ++r) {
                    const std::size_t base = (static_cast<std::size_t>(ms) * Ru + static_cast<std::size_t>(r)) * Nrep_u * Du;
                    for (int n = 0; n < n_rep; ++n) {
                        const double* zin = z_indep + base + static_cast<std::size_t>(n) * Du;
                        double* zout = z_corr_out + base + static_cast<std::size_t>(n) * Du;
                        for (int d = 0; d < D; ++d) {
                            const double* c_row = chol + static_cast<std::size_t>(d) * Du;
                            double acc = 0.0;
                            for (int k = 0; k <= d; ++k) {
                                acc += zin[k] * c_row[k];
                            }
                            // Support non-lower-triangular square factors gracefully if caller passed one.
                            for (int k = d + 1; k < D; ++k) {
                                if (c_row[k] != 0.0) acc += zin[k] * c_row[k];
                            }
                            zout[d] = acc;
                        }
                    }
                }
            }
        } else if (args->z_corr_out != z_indep) {
            std::copy(z_indep, z_indep + out_total, args->z_corr_out);
        }
    }

    return 0;
}

std::int32_t crn_eval_vibrato_gamma_cpp(const VibratoGammaArgs* args) noexcept {
    if (args == nullptr) return -1;
    if (args->d_assets <= 0 || args->m_rem <= 0 || args->n_paths_total <= 0 ||
        args->vib_step_start < 0 || args->vib_n_steps <= 0) {
        return -1;
    }
    if (args->perf_mat == nullptr || args->lx_pre == nullptr || args->j_pre == nullptr ||
        args->sig_pre == nullptr || args->sv_vib == nullptr || args->chol_mat == nullptr ||
        args->l_inv_t == nullptr || args->c_inv == nullptr || args->fwd_term == nullptr ||
        args->cdl_m == nullptr || args->cdl2_last == nullptr ||
        args->ko_barriers == nullptr || args->is_callable == nullptr ||
        args->cpn_barriers == nullptr || args->is_guaranteed_cpn == nullptr ||
        args->coupon_rates == nullptr || args->eps_vec == nullptr || args->h_vib_out == nullptr) {
        return -1;
    }

    const int D = static_cast<int>(args->d_assets);
    const int M = static_cast<int>(args->m_rem);
    const std::size_t N = static_cast<std::size_t>(args->n_paths_total);
    const int vib_start = static_cast<int>(args->vib_step_start);
    const int vib_n_steps = static_cast<int>(args->vib_n_steps);
    const std::size_t Du = static_cast<std::size_t>(D);
    const int m_last = M - 1;

    const int ia = std::clamp(
        (args->inv_lex_perm != nullptr && D > 0) ? static_cast<int>(args->inv_lex_perm[0]) : 0,
        0, D - 1);
    const int ib = std::clamp(
        (args->inv_lex_perm != nullptr && D > 1) ? static_cast<int>(args->inv_lex_perm[1]) : 0,
        0, D - 1);

    const bool is_long_basket = args->is_long_basket != 0;
    const int call_code = canonicalize_tile_code(args->call_code, D);
    const int cpn_code = canonicalize_tile_code(args->cpn_code, D);
    const int eff_opt_raw = args->opt_code;
    const int eff_opt_code = canonicalize_tile_code(eff_opt_raw, D);

    const bool has_ki = args->has_ki != 0;
    const bool ki_active = args->ki_active != 0;
    const bool has_one_star = args->has_one_star != 0;
    const bool is_mem = args->is_memory != 0;
    const bool no_notional = args->no_notional != 0;
    const bool force_cpn_ac = args->force_cpn_ac != 0;
    const bool cpn_dir_down = args->cpn_dir_down != 0;
    const bool cpn_type_out = args->cpn_type_out != 0;
    const bool has_lower_strike = args->has_lower_strike != 0;
    const bool has_part_call = args->has_part_call != 0;

    const double ko_spread = args->ko_spread;
    const double cpn_spread = args->cpn_spread;
    const double ki_spread = args->ki_spread;
    const double one_star_strike = args->one_star_strike;
    const double one_star_eps = args->one_star_eps;
    const double strike = args->strike;
    const double lower_strike = args->lower_strike;
    const double ki_barrier = args->ki_barrier;
    const double option_gearing = args->option_gearing;
    const double ac_cpn = args->autocall_coupon;
    const double c_mem_0 = args->c_mem_0;
    const double i_ki_past = args->i_ki_past;
    const double part_call_strike = args->part_call_strike;
    const double part_call_rate = args->part_call_rate;
    const double one_star_coupon = args->one_star_coupon;
    const double df_last = args->df_last;
    const double h_vib = args->h_vib;
    const double sqrt_h_vib = std::sqrt(std::max(h_vib, 1e-30));
    const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(vib_n_steps));
    const double inv_d = 1.0 / static_cast<double>(D);
    const double scale = 1.0 / static_cast<double>(N);

    const bool hard_ko_sp = std::abs(ko_spread) > 1e-6;
    const double eff_csp = cpn_dir_down ? -cpn_spread : cpn_spread;
    const bool hard_cpn_sp = std::abs(eff_csp) > 1e-6;
    const bool hard_ki_sp = std::abs(ki_spread) > 1e-6;
    const double blo_ki_k = std::min(ki_barrier, ki_barrier + ki_spread);
    const double bhi_ki_k = std::max(ki_barrier, ki_barrier + ki_spread);
    const double inv_sp_ko = 1.0 / std::max(std::abs(ko_spread), 1e-12);
    const double inv_sp_cp = 1.0 / std::max(std::abs(eff_csp), 1e-12);
    const double inv_sp_ki = 1.0 / std::max(bhi_ki_k - blo_ki_k, 1e-12);
    const double blo_ki_islb = std::min(strike, strike + ki_spread);
    const double bhi_ki_islb = std::max(strike, strike + ki_spread);
    const double inv_sp_ki_islb = 1.0 / std::max(bhi_ki_islb - blo_ki_islb, 1e-12);
    const double inv_sp_os = 1.0 / std::max(one_star_eps, 1e-12);
    const double sign_cpn_k = (cpn_dir_down ? -1.0 : 1.0) * (cpn_type_out ? -1.0 : 1.0);
    const double term_not = no_notional ? 0.0 : 1.0;

    auto eval_stats_at_period = [&](int m, std::size_t n) noexcept -> IdxBaseStats {
        IdxBaseStats st{};
        st.min1 = 1e300;
        st.min2 = 1e300;
        st.min3 = 1e300;
        st.max1 = -1e300;
        st.max2 = -1e300;
        st.idx_min1 = -1;
        st.idx_min2 = -1;
        st.idx_min3 = -1;
        st.idx_max1 = -1;
        st.idx_max2 = -1;
        st.pb = 1.0;
        const std::size_t m_off = static_cast<std::size_t>(m) * Du;
        for (int d = 0; d < D; ++d) {
            const double pd = args->perf_mat[(m_off + static_cast<std::size_t>(d)) * N + n];
            if (pd < st.min1) {
                st.min3 = st.min2; st.idx_min3 = st.idx_min2;
                st.min2 = st.min1; st.idx_min2 = st.idx_min1;
                st.min1 = pd; st.idx_min1 = d;
            } else if (pd < st.min2) {
                st.min3 = st.min2; st.idx_min3 = st.idx_min2;
                st.min2 = pd; st.idx_min2 = d;
            } else if (pd < st.min3) {
                st.min3 = pd; st.idx_min3 = d;
            }
            if (pd > st.max1) {
                st.max2 = st.max1; st.idx_max2 = st.idx_max1;
                st.max1 = pd; st.idx_max1 = d;
            } else if (pd > st.max2) {
                st.max2 = pd; st.idx_max2 = d;
            }
            st.s1 += pd;
            st.s2 += pd * pd;
            if (d == ia) st.pa = pd;
            if (d == ib) st.pb = pd;
        }
        return st;
    };

    auto eval_terminal_opt = [&](const IdxBaseStats& st, double wo, double wb, double os_ind) noexcept -> double {
        double ki_g = 1.0;
        if (is_long_basket) {
            if (has_ki && !ki_active) {
                ki_g = i_ki_past;
            } else {
                const double w_wof = st.min1;
                ki_g = hard_ki_sp
                    ? std::clamp((bhi_ki_islb - w_wof) * inv_sp_ki_islb, 0.0, 1.0)
                    : ((w_wof < strike) ? 1.0 : 0.0);
                if (has_ki) ki_g = std::max(i_ki_past, ki_g);
            }
        } else if (ki_active) {
            ki_g = hard_ki_sp
                ? std::clamp((bhi_ki_k - wo) * inv_sp_ki, 0.0, 1.0)
                : ((wo < ki_barrier) ? 1.0 : 0.0);
        } else if (has_ki) {
            ki_g = i_ki_past;
        }

        double opt = 0.0;
        if (is_long_basket) {
            const double w_avg = st.s1 * inv_d;
            opt -= ki_g * (1.0 - os_ind) * (1.0 - w_avg);
        } else {
            const bool put_mask = has_lower_strike ? ((strike > lower_strike) && (option_gearing > 0.0)) : (option_gearing > 0.0);
            if (put_mask) {
                double put_intr = std::max(strike - wo, 0.0);
                if (has_lower_strike) {
                    put_intr -= std::max(lower_strike - wo, 0.0);
                }
                opt -= option_gearing * ki_g * (1.0 - os_ind) * put_intr;
            }
        }
        if (has_part_call && part_call_rate != 0.0) {
            opt += part_call_rate * std::max(wo - part_call_strike, 0.0);
        }
        if (has_one_star && one_star_coupon != 0.0) {
            (void)wb;
            opt += os_ind * one_star_coupon;
        }
        return opt;
    };

    auto eval_last_indicators = [&](const IdxBaseStats& st, double& gk_out, double& egc_out, double& opt_out, double& os_out) noexcept {
        const double wc = select_idx_from_stats(st, inv_d, D, call_code);
        const double wcp = (cpn_code == call_code) ? wc : select_idx_from_stats(st, inv_d, D, cpn_code);
        const double wo = (eff_opt_code == call_code) ? wc : ((eff_opt_code == cpn_code) ? wcp : select_idx_from_stats(st, inv_d, D, eff_opt_code));
        const double wb = has_one_star ? st.max1 : 0.0;

        const double b_ko_l = args->ko_barriers[m_last];
        const bool is_c_l = args->is_callable[m_last] > 0.5;
        double gk_l = 0.0;
        if (is_c_l) {
            if (hard_ko_sp) {
                gk_l = std::clamp((wc - std::min(b_ko_l, b_ko_l + ko_spread)) * inv_sp_ko, 0.0, 1.0);
            } else {
                gk_l = (wc >= b_ko_l) ? 1.0 : 0.0;
            }
        }
        const double b_cp_l = args->cpn_barriers[m_last];
        const bool is_g_l = args->is_guaranteed_cpn[m_last] > 0.5;
        double gc_l = 1.0;
        if (!is_g_l) {
            if (hard_cpn_sp) {
                const double blo_cp = std::min(b_cp_l, b_cp_l + eff_csp);
                const double r_up = std::clamp((wcp - blo_cp) * inv_sp_cp, 0.0, 1.0);
                const double hit_s = cpn_dir_down ? (1.0 - r_up) : r_up;
                gc_l = cpn_type_out ? (1.0 - hit_s) : hit_s;
            } else {
                const bool hit_h = cpn_dir_down ? (wcp <= b_cp_l) : (wcp >= b_cp_l);
                gc_l = cpn_type_out ? (hit_h ? 0.0 : 1.0) : (hit_h ? 1.0 : 0.0);
            }
        }
        const double egc_l = force_cpn_ac ? std::max(gc_l, gk_l) : gc_l;
        double os_l = 0.0;
        if (has_one_star) {
            if (one_star_eps <= 1e-6) {
                os_l = (wb >= one_star_strike) ? 1.0 : 0.0;
            } else {
                os_l = std::clamp((wb - (one_star_strike - 0.5 * one_star_eps)) * inv_sp_os, 0.0, 1.0);
            }
        }
        gk_out = gk_l;
        egc_out = egc_l;
        os_out = os_l;
        opt_out = eval_terminal_opt(st, wo, wb, os_l);
    };

    const bool write_frozen = (args->frozen_term_out != nullptr) && (M > 1);
    const std::size_t n_h_tot = Du * Du;

    OmpCallGuard omp_guard(N * Du * static_cast<std::size_t>(std::max(M + vib_n_steps, 1)) >= 4096);
    const int n_threads = omp_guard.active ? omp_guard.n_threads : 1;
    std::vector<double> per_thr_h(static_cast<std::size_t>(n_threads) * n_h_tot, 0.0);

    #pragma omp parallel if(omp_guard.active) num_threads(n_threads)
    {
#ifdef _OPENMP
        const int tid = omp_guard.active ? omp_get_thread_num() : 0;
#else
        const int tid = 0;
#endif
        double* thr_h = per_thr_h.data() + static_cast<std::size_t>(tid) * n_h_tot;
        std::vector<double> dlns(Du, 0.0);
        std::vector<double> dcm(is_mem ? Du : 0u, 0.0);
        std::vector<double> zk_mean(Du, 0.0);
        std::vector<double> xi(Du, 0.0);
        std::vector<double> inv_s(Du, 0.0);
        std::vector<double> j_last(Du, 0.0);
        std::vector<double> j_pre_v(Du, 0.0);
        std::vector<double> w1_raw(Du, 0.0);
        std::vector<double> w1(Du, 0.0);
        std::vector<double> w1_egc(is_mem ? Du : 0u, 0.0);

        #pragma omp for schedule(static)
        for (std::size_t n = 0; n < N; ++n) {
            double surv_pre = 1.0;
            double c_mem = c_mem_0;
            std::fill(dlns.begin(), dlns.end(), 0.0);
            if (is_mem) std::fill(dcm.begin(), dcm.end(), 0.0);

            for (int m = 0; m < m_last; ++m) {
                if (surv_pre <= 1e-24) break;
                const IdxBaseStats st = eval_stats_at_period(m, n);
                const double w_call = select_idx_from_stats(st, inv_d, D, call_code);
                const double w_cpn = (cpn_code == call_code) ? w_call : select_idx_from_stats(st, inv_d, D, cpn_code);

                const double eps_m = args->eps_vec[m];
                const double b_ko = args->ko_barriers[m];
                const double lo_ko = hard_ko_sp ? std::min(b_ko, b_ko + ko_spread) : (b_ko - 0.5 * eps_m);
                const double hi_ko = hard_ko_sp ? std::max(b_ko, b_ko + ko_spread) : (b_ko + 0.5 * eps_m);
                const double inv_w_ko = 1.0 / std::max(hi_ko - lo_ko, 1e-12);
                const double t_ko = (w_call - lo_ko) * inv_w_ko;
                const bool is_call_m = args->is_callable[m] > 0.5;
                const double g_k = is_call_m ? c2_s5(t_ko) : 0.0;
                const double dg_k_dw = is_call_m ? (c2_ds5(t_ko) * inv_w_ko) : 0.0;
                const double coef_surv_m = -dg_k_dw / std::max(1.0 - g_k, 1e-8);

                const double b_cp = args->cpn_barriers[m];
                const double lo_cp = hard_cpn_sp ? std::min(b_cp, b_cp + eff_csp) : (b_cp - 0.5 * eps_m);
                const double hi_cp = hard_cpn_sp ? std::max(b_cp, b_cp + eff_csp) : (b_cp + 0.5 * eps_m);
                const double inv_w_cp = 1.0 / std::max(hi_cp - lo_cp, 1e-12);
                const double t_cp = (w_cpn - lo_cp) * inv_w_cp;
                const double ramp_c2 = c2_s5(t_cp);
                const double dramp_c2_dw = c2_ds5(t_cp) * inv_w_cp;
                const double hit_c2 = cpn_dir_down ? (1.0 - ramp_c2) : ramp_c2;
                const double ind_c2 = cpn_type_out ? (1.0 - hit_c2) : hit_c2;
                const bool is_guar_m = args->is_guaranteed_cpn[m] > 0.5;
                const double g_c = is_guar_m ? 1.0 : ind_c2;
                const double dg_c_dw = is_guar_m ? 0.0 : (dramp_c2_dw * sign_cpn_k);

                double eff_gc = g_c;
                double w_c_max = 1.0;
                double w_k_max = 0.0;
                if (force_cpn_ac) {
                    eff_gc = std::max(g_c, g_k);
                    w_c_max = (g_c > g_k) ? 1.0 : ((g_c < g_k) ? 0.0 : 0.5);
                    w_k_max = 1.0 - w_c_max;
                }

                const double c_rate = args->coupon_rates[m];
                const double eff_c_m = is_mem ? (c_rate + c_mem) : 0.0;

                const double w_avg_v = st.s1 * inv_d;
                const double w_disp_v = std::sqrt(std::max(st.s2 * inv_d - w_avg_v * w_avg_v, 0.0));
                const std::size_t m_off = static_cast<std::size_t>(m) * Du;
                for (int d = 0; d < D; ++d) {
                    const std::size_t du = static_cast<std::size_t>(d);
                    const double pd = args->perf_mat[(m_off + du) * N + n];
                    const double cdl_md = args->cdl_m[m_off + du];
                    const double bp_d = accum_idx_grad_scalar(
                        0.0, coef_surv_m, pd, d, D, inv_d, ia, ib, st, w_avg_v, w_disp_v, call_code);
                    const double prev_dlns = dlns[du];
                    if (is_mem) {
                        double bp_cm = 0.0;
                        if (cpn_code == call_code) {
                            const double bar_wc_cm = force_cpn_ac ? (w_c_max * dg_c_dw + w_k_max * dg_k_dw) : dg_c_dw;
                            bp_cm = accum_idx_grad_scalar(
                                0.0, bar_wc_cm, pd, d, D, inv_d, ia, ib, st, w_avg_v, w_disp_v, call_code);
                        } else {
                            bp_cm = accum_idx_grad_scalar(
                                0.0, w_c_max * dg_c_dw, pd, d, D, inv_d, ia, ib, st, w_avg_v, w_disp_v, cpn_code);
                            if (force_cpn_ac) {
                                bp_cm = accum_idx_grad_scalar(
                                    bp_cm, w_k_max * dg_k_dw, pd, d, D, inv_d, ia, ib, st, w_avg_v, w_disp_v, call_code);
                            }
                        }
                        const double d_eff_gc_d = bp_cm * pd * cdl_md;
                        // V3 literal: c_mem_next = surv_pre * eff_c_m * (1 - eff_gc)
                        dcm[du] = surv_pre * (
                            prev_dlns * eff_c_m * (1.0 - eff_gc)
                            + dcm[du] * (1.0 - eff_gc)
                            - eff_c_m * d_eff_gc_d
                        );
                    }
                    dlns[du] = prev_dlns + bp_d * pd * cdl_md;
                }

                if (is_mem) {
                    c_mem = surv_pre * eff_c_m * (1.0 - eff_gc);
                }
                surv_pre = std::max(surv_pre - surv_pre * g_k, 0.0);
            }

            if (write_frozen) {
                const IdxBaseStats st_b = eval_stats_at_period(m_last, n);
                double gk_l = 0.0, egc_l = 0.0, opt_l = 0.0, os_l = 0.0;
                eval_last_indicators(st_b, gk_l, egc_l, opt_l, os_l);
                args->frozen_term_out[0u * N + n] = gk_l;
                args->frozen_term_out[1u * N + n] = egc_l;
                args->frozen_term_out[2u * N + n] = opt_l;
                args->frozen_term_out[3u * N + n] = os_l;
            }

            if (surv_pre <= 1e-24) {
                continue;
            }

            for (int k = 0; k < D; ++k) {
                double z_sum = 0.0;
                for (int s_v = 0; s_v < vib_n_steps; ++s_v) {
                    const std::size_t st_i = static_cast<std::size_t>(vib_start + s_v);
                    z_sum += args->sv_vib[(st_i * Du + static_cast<std::size_t>(k)) * N + n];
                }
                zk_mean[static_cast<std::size_t>(k)] = z_sum * inv_sqrt_n;
            }

            const double eff_c_l = is_mem ? (args->coupon_rates[m_last] + c_mem) : args->coupon_rates[m_last];
            double f0 = 0.0, fp = 0.0, fm = 0.0, egc_p = 0.0, egc_m = 0.0;
            for (int sgn = 0; sgn < 3; ++sgn) {
                const double mult = (sgn == 0) ? 0.0 : ((sgn == 1) ? 1.0 : -1.0);
                IdxBaseStats st{};
                st.min1 = 1e300;
                st.min2 = 1e300;
                st.min3 = 1e300;
                st.max1 = -1e300;
                st.max2 = -1e300;
                st.idx_min1 = -1;
                st.idx_min2 = -1;
                st.idx_min3 = -1;
                st.idx_max1 = -1;
                st.idx_max2 = -1;
                st.pb = 1.0;

                for (int d = 0; d < D; ++d) {
                    const std::size_t du = static_cast<std::size_t>(d);
                    const double lxp = args->lx_pre[du * N + n];
                    const double sigp = std::max(args->sig_pre[du * N + n], 1e-4);
                    double xd = lxp - 0.5 * sigp * sigp * h_vib;
                    if (sgn != 0) {
                        const double* crow = args->chol_mat + du * Du;
                        double zc_d = 0.0;
                        for (int k = 0; k < D; ++k) {
                            zc_d += crow[k] * zk_mean[static_cast<std::size_t>(k)];
                        }
                        xd += mult * sigp * sqrt_h_vib * zc_d;
                    }
                    const double pd = args->fwd_term[du] * std::exp(std::clamp(xd, -30.0, 30.0));
                    if (pd < st.min1) {
                        st.min3 = st.min2; st.idx_min3 = st.idx_min2;
                        st.min2 = st.min1; st.idx_min2 = st.idx_min1;
                        st.min1 = pd; st.idx_min1 = d;
                    } else if (pd < st.min2) {
                        st.min3 = st.min2; st.idx_min3 = st.idx_min2;
                        st.min2 = pd; st.idx_min2 = d;
                    } else if (pd < st.min3) {
                        st.min3 = pd; st.idx_min3 = d;
                    }
                    if (pd > st.max1) {
                        st.max2 = st.max1; st.idx_max2 = st.idx_max1;
                        st.max1 = pd; st.idx_max1 = d;
                    } else if (pd > st.max2) {
                        st.max2 = pd; st.idx_max2 = d;
                    }
                    st.s1 += pd;
                    st.s2 += pd * pd;
                    if (d == ia) st.pa = pd;
                    if (d == ib) st.pb = pd;
                }

                double gk_s = 0.0, egc_s = 0.0, opt_s = 0.0, os_s = 0.0;
                eval_last_indicators(st, gk_s, egc_s, opt_s, os_s);
                (void)os_s;
                const double f_val = gk_s * (term_not + ac_cpn) + egc_s * eff_c_l + (1.0 - gk_s) * (term_not + opt_s);
                if (sgn == 0) {
                    f0 = f_val;
                } else if (sgn == 1) {
                    fp = f_val;
                    egc_p = egc_s;
                } else {
                    fm = f_val;
                    egc_m = egc_s;
                }
            }

            const double df_diff = 0.5 * (fp - fm);
            const double df_hess = 0.5 * (fp + fm) - f0;
            const double degc_diff = is_mem ? (0.5 * (egc_p - egc_m)) : 0.0;
            const std::size_t m_last_off = static_cast<std::size_t>(m_last) * Du;

            for (int i = 0; i < D; ++i) {
                const std::size_t iu = static_cast<std::size_t>(i);
                const double sig_i = std::max(args->sig_pre[iu * N + n], 1e-4);
                const double isi = 1.0 / std::max(sig_i * sqrt_h_vib, 1e-6);
                const double jp = std::clamp(args->j_pre[iu * N + n], -8.0, 8.0);
                const double ji = jp * args->cdl_m[m_last_off + iu];
                const double* lrow = args->l_inv_t + iu * Du;
                double xi_i = 0.0;
                for (int k = 0; k < D; ++k) {
                    xi_i += lrow[k] * zk_mean[static_cast<std::size_t>(k)];
                }
                inv_s[iu] = isi;
                j_pre_v[iu] = jp;
                j_last[iu] = ji;
                xi[iu] = xi_i;
                w1_raw[iu] = df_diff * xi_i * isi * jp;
                w1[iu] = df_diff * xi_i * isi * ji;
                if (is_mem) w1_egc[iu] = degc_diff * xi_i * isi * ji;
            }

            const double w_surv = df_last * surv_pre;
            for (int i = 0; i < D; ++i) {
                const std::size_t iu = static_cast<std::size_t>(i);
                const double dlns_i = dlns[iu];
                const double dcm_i = is_mem ? dcm[iu] : 0.0;
                const double cdl2_i = args->cdl2_last[iu];
                for (int j = i; j < D; ++j) {
                    const std::size_t ju = static_cast<std::size_t>(j);
                    const double cinv_ij = args->c_inv[iu * Du + ju];
                    const double w2_ij = df_hess * (xi[iu] * xi[ju] - cinv_ij) * inv_s[iu] * inv_s[ju] * j_last[iu] * j_last[ju];
                    const double cmem_cross = is_mem ? (dcm_i * w1_egc[ju] + dcm[ju] * w1_egc[iu]) : 0.0;
                    const double diag_corr = (i == j) ? (w1_raw[iu] * cdl2_i) : 0.0;
                    const double h_ij = w_surv * (w2_ij + dlns_i * w1[ju] + dlns[ju] * w1[iu] + cmem_cross + diag_corr);
                    thr_h[iu * Du + ju] += h_ij;
                }
            }
        }
    }

    std::fill(args->h_vib_out, args->h_vib_out + n_h_tot, 0.0);
    for (int tid = 0; tid < n_threads; ++tid) {
        const double* src = per_thr_h.data() + static_cast<std::size_t>(tid) * n_h_tot;
        for (int i = 0; i < D; ++i) {
            for (int j = i; j < D; ++j) {
                const std::size_t ij = static_cast<std::size_t>(i) * Du + static_cast<std::size_t>(j);
                args->h_vib_out[ij] += src[ij];
            }
        }
    }
    for (int i = 0; i < D; ++i) {
        for (int j = i; j < D; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * Du + static_cast<std::size_t>(j);
            const double val = args->h_vib_out[ij] * scale;
            args->h_vib_out[ij] = val;
            if (i != j) {
                args->h_vib_out[static_cast<std::size_t>(j) * Du + static_cast<std::size_t>(i)] = val;
            }
        }
    }

    return 0;
}

}  // extern "C"

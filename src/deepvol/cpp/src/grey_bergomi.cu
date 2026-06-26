#include "grey_bergomi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

// Helper to check cuFFT errors
#define CUFFT_CHECK(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult code, const char *file, int line, bool abort=true) {
   if (code != CUFFT_SUCCESS) {
      char err_msg[256];
      snprintf(err_msg, sizeof(err_msg), "cuFFT Error %d at %s:%d", code, file, line);
      throw std::runtime_error(err_msg);
   }
}

inline int get_M(int N) {
    int val = 2 * N - 2;
    if (val < 2) val = 2;
    int M = 1;
    while (M < val) {
        M *= 2;
    }
    return M;
}

// Element-wise Mittag-Leffler evaluation on the GPU.
//
// Computes E_beta(z) = sum_{k=0}^{inf} z^k / Gamma(beta*k + 1)
// using two regimes based on val_check = z^(1/beta):
//
//   Regime 1 (val_check <= 35.0): Power series in log-space via lgamma.
//     - Uses lgamma instead of tgamma to avoid overflow at large k*beta.
//     - Convergence criterion: term < tol * sum (relative), k > 0.
//     - Accurate to machine epsilon for the tested range.
//
//   Regime 2 (val_check > 35.0): 4-term asymptotic expansion:
//     E_beta(z) ~ (1/beta) * exp(z^{1/beta}) - sum_{j=1}^{4} z^{-j} / Gamma(1 - beta*j)
//     - Poles of Gamma(1 - beta*j) at non-positive integers are handled by setting
//       the corresponding term to 0.0 (residue vanishes at those poles).
//     - REC-7 ACCURACY NOTE: Only 4 terms are used. For beta near 1.0 and intermediate
//       z in (35, 100), the expansion may lose 1-2 ULP relative to the power series
//       because poles of Gamma(1 - beta*j) cluster near negative integers.
//       Validated to < 1e-7 relative error for beta in [0.5, 0.95] via test_grey_cuda.py.
//       If beta > 0.95 and z in (35, 100) is required, consider increasing to 6 terms
//       or switching the threshold to val_check <= 50.0.
//
// Parameters:
//   z        : input argument (must be >= 0; z <= 0 returns 1.0 exactly)
//   beta     : fractional order in (0, 1]
//   max_iter : maximum power-series terms (default 500)
//   tol      : relative convergence tolerance (default 1e-9)
__device__ double mittag_leffler_eval(double z, double beta, int max_iter, double tol) {
    if (z <= 0.0) {
        return 1.0;
    }
    double val_check = pow(z, 1.0 / beta);
    if (val_check <= 35.0) {
        double sum = 0.0;
        double log_z = log(z);
        for (int k = 0; k < max_iter; ++k) {
            double val = k * log_z - lgamma(beta * k + 1.0);
            double term = exp(val);
            sum += term;
            if (term < tol * sum && k > 0) {
                break;
            }
        }
        return sum;
    } else {
        double sum_terms = 0.0;
        for (int j = 1; j <= 4; ++j) {
            double arg = 1.0 - beta * j;
            double gamma_val;
            // Handle poles of Gamma function at non-positive integers
            if (fabs(arg - round(arg)) < 1e-9 && arg <= 0.0) {
                gamma_val = INFINITY;
            } else {
                gamma_val = tgamma(arg);
            }
            double term = (gamma_val == INFINITY || isinf(gamma_val)) ? 0.0 : (pow(z, -j) / gamma_val);
            sum_terms += term;
        }
        return (1.0 / beta) * exp(val_check) - sum_terms;
    }
}

__global__ void mittag_leffler_kernel(
    const double* __restrict__ z,
    double* __restrict__ out,
    double beta,
    int max_iter,
    double tol,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = mittag_leffler_eval(z[idx], beta, max_iter, tol);
}

torch::Tensor mittag_leffler_cuda(
    torch::Tensor z,
    double beta,
    int max_iter,
    double tol
) {
    const at::cuda::CUDAGuard device_guard(z.device());
    auto original_dtype = z.scalar_type();
    auto z_double = z.to(torch::kDouble).contiguous();
    auto out_double = torch::empty_like(z_double);

    int numel = z_double.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    mittag_leffler_kernel<<<blocks, threads>>>(
        z_double.data_ptr<double>(),
        out_double.data_ptr<double>(),
        beta,
        max_iter,
        tol,
        numel
    );

    return out_double.to(original_dtype);
}

// Covariance vector initialization kernel
__global__ void initialize_covariance_kernel(
    double* __restrict__ C,
    const double* __restrict__ params,
    double dt,
    int M,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * M;
    if (idx >= total_elements) return;

    int b = idx / M;
    int k = idx % M;
    double H = params[b * 5 + 1]; // H is the second parameter (index 1)
    double gamma_H = tgamma(H + 0.5);
    double scale = 1.0 / (gamma_H * gamma_H);

    double val = 0.0;
    int k_eff = (k <= M / 2) ? k : (M - k);
    double dt_2H = pow(dt, 2.0 * H);
    if (k_eff == 0) {
        val = dt_2H;
    } else {
        val = 0.5 * dt_2H * (pow(k_eff + 1.0, 2.0 * H) + pow(k_eff - 1.0, 2.0 * H) - 2.0 * pow(k_eff, 2.0 * H));
    }
    C[idx] = val * scale;
}

// Eigenvalue extraction and clamping kernel
__global__ void extract_lambda_kernel(
    const cufftDoubleComplex* __restrict__ C_complex,
    double* __restrict__ Lambda,
    int M_half_plus_1,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * M_half_plus_1;
    if (idx >= total_elements) return;

    double val = C_complex[idx].x; // Real part is the eigenvalue
    Lambda[idx] = val > 0.0 ? val : 0.0;
}

// Noise scaling kernel for fBm and Brownian increments preparation
__global__ void prepare_fft_inputs_kernel(
    const double* __restrict__ noise_real,
    const double* __restrict__ noise_imag,
    const double* __restrict__ Lambda,
    cufftDoubleComplex* __restrict__ in_X,
    cufftDoubleComplex* __restrict__ in_B,
    int paths,
    int M_half_plus_1,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * paths * M_half_plus_1;
    if (idx >= total_elements) return;

    int p = idx / M_half_plus_1;
    int j = idx % M_half_plus_1;
    int b = p / paths;

    double lam = Lambda[b * M_half_plus_1 + j];
    double r = noise_real[idx];
    double i = noise_imag[idx];

    cufftDoubleComplex val_X;
    cufftDoubleComplex val_B;

    if (j == 0 || j == M_half_plus_1 - 1) {
        val_X.x = sqrt(lam) * r;
        val_X.y = 0.0;
        val_B.x = r;
        val_B.y = 0.0;
    } else {
        double sqrt_lam_half = sqrt(lam * 0.5);
        double sqrt_half = sqrt(0.5);
        val_X.x = sqrt_lam_half * r;
        val_X.y = sqrt_lam_half * i;
        val_B.x = sqrt_half * r;
        val_B.y = sqrt_half * i;
    }

    in_X[idx] = val_X;
    in_B[idx] = val_B;
}

// Grey Bergomi path simulation kernel
__global__ void simulate_paths_kernel(
    const double* __restrict__ out_X,
    const double* __restrict__ out_B,
    const double* __restrict__ V_rand,
    const double* __restrict__ W_rand,
    const double* __restrict__ Z_perp,
    const double* __restrict__ params,
    double* __restrict__ S,
    double* __restrict__ V,
    double* __restrict__ B_H,
    int B,
    int paths,
    int steps,
    int M,
    double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_paths = B * paths;
    if (idx >= total_paths) return;

    int b = idx / paths;

    double v0 = params[b * 5 + 0];
    double H = params[b * 5 + 1];
    double eta = params[b * 5 + 2];
    double rho = params[b * 5 + 3];
    double beta = params[b * 5 + 4];

    double c = sqrt(2.0 * H) / tgamma(H + 0.5);
    double b_coeff = (eta * eta * c * c) / (4.0 * H);

    double V_val = V_rand[idx];
    double W_val = W_rand[idx];

    // Kanter's representation for M-Wright variable Y_beta
    double sin_V = sin(V_val);
    double sin_beta_V = sin(beta * V_val);
    double sin_one_minus_beta_V = sin((1.0 - beta) * V_val);

    double Y_beta = pow(W_val, 1.0 - beta) * sin_V /
                    (pow(sin_beta_V, beta) * pow(sin_one_minus_beta_V, 1.0 - beta));

    double fbm_val = 0.0;
    double log_S = 0.0;

    int path_offset = idx * (steps + 1);
    S[path_offset + 0] = 1.0;
    V[path_offset + 0] = v0;
    B_H[path_offset + 0] = 0.0;

    double sqrt_one_minus_rho_sq = sqrt(1.0 - rho * rho);

    for (int s = 0; s < steps; ++s) {
        double dX = out_X[idx * M + s] / sqrt((double)M);
        double dB = (out_B[idx * M + s] / sqrt((double)M)) * sqrt(dt);

        fbm_val += dX;
        B_H[path_offset + s + 1] = fbm_val;

        // Compute variance for the next step
        double t = (s + 1) * dt;
        double z = b_coeff * pow(t, 2.0 * H);
        double E_beta_val = mittag_leffler_eval(z, beta, 500, 1e-9);
        double exponent = eta * sqrt(Y_beta) * fbm_val;
        if (exponent > 150.0) {
            exponent = 150.0;
        }
        V[path_offset + s + 1] = v0 * exp(exponent) / E_beta_val;

        // Stock log-price update (Euler-Maruyama)
        double V_prev = V[path_offset + s];
        double dW = rho * dB + sqrt_one_minus_rho_sq * sqrt(dt) * Z_perp[idx * steps + s];
        log_S += -0.5 * V_prev * dt + sqrt(V_prev) * dW;
        S[path_offset + s + 1] = exp(log_S);
    }
}

std::vector<torch::Tensor> generate_grey_paths_cuda(
    torch::Tensor params,
    int steps,
    int paths,
    double T,
    double dt
) {
    const at::cuda::CUDAGuard device_guard(params.device());

    bool is_1d = (params.dim() == 1);
    torch::Tensor params_2d = is_1d ? params.unsqueeze(0) : params;
    int B = params_2d.size(0);

    params_2d = params_2d.to(torch::kDouble).to(torch::kCUDA).contiguous();

    int M = get_M(steps);

    // 1. Initialize covariance vector C on GPU
    torch::Tensor C = torch::empty({B, M}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    {
        int total_elements = B * M;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        initialize_covariance_kernel<<<blocks, threads>>>(
            C.data_ptr<double>(),
            params_2d.data_ptr<double>(),
            dt,
            M,
            B
        );
    }

    // 2. Compute eigenvalues Lambda using R2C cuFFT on C
    int M_half_plus_1 = M / 2 + 1;
    torch::Tensor C_complex = torch::empty({B, M_half_plus_1}, torch::TensorOptions().dtype(torch::kComplexDouble).device(params_2d.device()));

    cufftHandle plan_r2c;
    CUFFT_CHECK(cufftPlan1d(&plan_r2c, M, CUFFT_D2Z, B));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, 
                             (cufftDoubleReal*)C.data_ptr<double>(), 
                             (cufftDoubleComplex*)C_complex.data_ptr<c10::complex<double>>()));
    CUFFT_CHECK(cufftDestroy(plan_r2c));

    // Extract and clamp Lambda
    torch::Tensor Lambda = torch::empty({B, M_half_plus_1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    {
        int total_elements = B * M_half_plus_1;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        extract_lambda_kernel<<<blocks, threads>>>(
            (cufftDoubleComplex*)C_complex.data_ptr<c10::complex<double>>(),
            Lambda.data_ptr<double>(),
            M_half_plus_1,
            B
        );
    }

    // 3. Generate random noise
    int total_paths = B * paths;
    torch::Tensor noise_real = torch::randn({total_paths, M_half_plus_1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    torch::Tensor noise_imag = torch::randn({total_paths, M_half_plus_1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));

    torch::Tensor in_X = torch::empty({total_paths, M_half_plus_1}, torch::TensorOptions().dtype(torch::kComplexDouble).device(params_2d.device()));
    torch::Tensor in_B = torch::empty({total_paths, M_half_plus_1}, torch::TensorOptions().dtype(torch::kComplexDouble).device(params_2d.device()));

    {
        int total_elements = total_paths * M_half_plus_1;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        prepare_fft_inputs_kernel<<<blocks, threads>>>(
            noise_real.data_ptr<double>(),
            noise_imag.data_ptr<double>(),
            Lambda.data_ptr<double>(),
            (cufftDoubleComplex*)in_X.data_ptr<c10::complex<double>>(),
            (cufftDoubleComplex*)in_B.data_ptr<c10::complex<double>>(),
            paths,
            M_half_plus_1,
            B
        );
    }

    // 4. Perform backward FFT (Z2D) to get fBm and Brownian increments
    torch::Tensor out_X = torch::empty({total_paths, M}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    torch::Tensor out_B = torch::empty({total_paths, M}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));

    cufftHandle plan_z2d;
    CUFFT_CHECK(cufftPlan1d(&plan_z2d, M, CUFFT_Z2D, total_paths));
    CUFFT_CHECK(cufftExecZ2D(plan_z2d, 
                             (cufftDoubleComplex*)in_X.data_ptr<c10::complex<double>>(), 
                             (cufftDoubleReal*)out_X.data_ptr<double>()));
    CUFFT_CHECK(cufftExecZ2D(plan_z2d, 
                             (cufftDoubleComplex*)in_B.data_ptr<c10::complex<double>>(), 
                             (cufftDoubleReal*)out_B.data_ptr<double>()));
    CUFFT_CHECK(cufftDestroy(plan_z2d));

    // 5. Generate other random variables
    torch::Tensor V_rand = torch::rand({total_paths}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device())) * M_PI;
    torch::Tensor W_rand = -torch::log(1.0 - torch::rand({total_paths}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device())));
    torch::Tensor Z_perp = torch::randn({total_paths, steps}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));

    // Allocate output path tensors in double precision (SoA layout)
    torch::Tensor S = torch::empty({B, paths, steps + 1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    torch::Tensor V = torch::empty({B, paths, steps + 1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));
    torch::Tensor B_H = torch::empty({B, paths, steps + 1}, torch::TensorOptions().dtype(torch::kDouble).device(params_2d.device()));

    {
        int threads = 256;
        int blocks = (total_paths + threads - 1) / threads;
        simulate_paths_kernel<<<blocks, threads>>>(
            out_X.data_ptr<double>(),
            out_B.data_ptr<double>(),
            V_rand.data_ptr<double>(),
            W_rand.data_ptr<double>(),
            Z_perp.data_ptr<double>(),
            params_2d.data_ptr<double>(),
            S.data_ptr<double>(),
            V.data_ptr<double>(),
            B_H.data_ptr<double>(),
            B,
            paths,
            steps,
            M,
            dt
        );
    }

    // Apply empirical martingale correction to restore exactly E[S_t] = 1.0
    // S has shape (B, paths, steps + 1). Compute mean along paths (dim 1)
    torch::Tensor S_mean = S.mean(1, /*keepdim=*/true);
    S.div_(S_mean);

    // Convert output path states to float32 at the boundary
    torch::Tensor S_out = is_1d ? S.squeeze(0) : S;
    torch::Tensor V_out = is_1d ? V.squeeze(0) : V;
    torch::Tensor B_H_out = is_1d ? B_H.squeeze(0) : B_H;

    return {S_out.to(torch::kFloat32), V_out.to(torch::kFloat32), B_H_out.to(torch::kFloat32)};
}

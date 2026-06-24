#include <torch/extension.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector>

// Define the number of auxiliary factors for the Lifted Heston model.
// Set to 20 as per Master's thesis requirements for industrial-grade accuracy.
constexpr int N_FACTORS = 20;

// Thread block size for optimal occupancy.
constexpr int THREADS_PER_BLOCK = 256;

// __launch_bounds__ helps the nvcc compiler optimize register allocation
// to prevent spilling to slow DRAM (local memory).
__global__ 
__launch_bounds__(THREADS_PER_BLOCK, 4)
void lifted_heston_kernel(
    int num_paths,
    int num_steps,
    float dt,
    float S0,
    float V0,
    float rho,
    float kappa,
    float theta,
    float sigma,
    const float* __restrict__ c_weights_global,
    const float* __restrict__ x_speeds_global,
    unsigned long long seed,
    unsigned long long call_index,  // FIX: per-call offset to prevent path correlation
    float* __restrict__ prices_out) 
{
    // 1. Shared Memory for Bernstein Weights
    // Broadcast read pattern: all threads in a warp read the same index i in the
    // inner loop (a warp-level broadcast), which costs exactly 1 shared memory
    // transaction per cycle — no bank conflicts.
    __shared__ float c_weights[N_FACTORS];
    __shared__ float x_speeds[N_FACTORS];

    int tid = threadIdx.x;
    if (tid < N_FACTORS) {
        c_weights[tid] = c_weights_global[tid];
        x_speeds[tid] = x_speeds_global[tid];
    }
    __syncthreads();

    // Global path index
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (path_idx >= num_paths) return;

    // 2. Initialize RNG (Philox4_32_10)
    // CONTRACT: seed identifies the Monte Carlo experiment; call_index is incremented
    // by the Python caller on every invocation to guarantee distinct subsequences
    // across repeated calibration calls.  Subsequence = path_idx gives per-path
    // independence within a single call.  Offset = call_index * num_steps gives
    // non-overlapping streams across calls (Philox period >> 2^64).
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)path_idx, call_index * (unsigned long long)num_steps, &state);

    // 3. Local State Initialization
    float S_t = S0;
    float V_t = V0;
    float U_t[N_FACTORS];

    // Initialize auxiliary factors to 0 (U_i(0) = 0 by definition of the lifting)
    #pragma unroll
    for (int i = 0; i < N_FACTORS; ++i) {
        U_t[i] = 0.0f;
    }

    float sqrt_dt = sqrtf(dt);
    float sqrt_1_minus_rho2 = sqrtf(1.0f - rho * rho);

    // 4. Monte Carlo Simulation Loop
    for (int t = 0; t < num_steps; ++t) {
        // Generate correlated random variables (2 at a time via Philox)
        float4 rand_vals = curand_normal4(&state);
        float Z1 = rand_vals.x; // Variance Brownian
        float Z2 = rand_vals.y; // Independent standard normal
        
        // Correlated Brownian increment for Asset
        float Z_S = rho * Z1 + sqrt_1_minus_rho2 * Z2;

        // Full-truncation reflection for variance positivity
        float V_clipped = fmaxf(V_t, 0.0f);
        float sqrt_V = sqrtf(V_clipped);

        // Advance Asset Price (log-Euler for geometric SDE, drift-free under risk-neutral measure)
        // dS_t = S_t * sqrt(V_t) * dW^S  =>  log-exact step:
        S_t *= expf(-0.5f * V_clipped * dt + sqrt_V * sqrt_dt * Z_S);

        // Advance Auxiliary Factors with Semi-Implicit Euler
        // Lifted Heston SDE for factor i:
        //   dU_i = [-x_i * U_i + kappa*(theta - V_t)] dt + sigma*sqrt(V_t)*dW^V
        // Splitting: implicit on stiff -x_i*U_i term, explicit on the remainder.
        // This gives the update:
        //   U_i^{n+1} = [U_i^n + kappa*(theta - V^n)*dt + sigma*sqrt(V^n)*sqrt(dt)*Z1]
        //               / (1 + x_i * dt)
        //
        // FIXED (was: drift_vol_part = -kappa*V_clipped*dt — missing theta entirely,
        // causing V to drift toward -inf instead of mean-reverting to theta).
        float V_next = V0;  // g_0 baseline (stationary approximation: g_0 ~ V_0)
        float mean_rev_drift = kappa * (theta - V_clipped) * dt;
        float diffusion      = sigma * sqrt_V * sqrt_dt * Z1;

        #pragma unroll
        for (int i = 0; i < N_FACTORS; ++i) {
            U_t[i] = (U_t[i] + mean_rev_drift + diffusion) / (1.0f + x_speeds[i] * dt);
            V_next += c_weights[i] * U_t[i];
        }
        
        V_t = V_next;
    }

    // 5. Store Final Price
    prices_out[path_idx] = S_t;
}

// C++ Interface
torch::Tensor simulate_lifted_heston(
    int num_paths,
    int num_steps,
    float dt,
    float S0,
    float V0,
    float rho,
    float kappa,
    float theta,
    float sigma,
    torch::Tensor c_weights,
    torch::Tensor x_speeds,
    int64_t seed,
    int64_t call_index = 0)  // FIX: increment per calibration call to avoid path correlation
{
    // Ensure inputs are contiguous on CUDA
    TORCH_CHECK(c_weights.is_cuda() && c_weights.is_contiguous(), "c_weights must be a contiguous CUDA tensor");
    TORCH_CHECK(x_speeds.is_cuda() && x_speeds.is_contiguous(), "x_speeds must be a contiguous CUDA tensor");
    TORCH_CHECK(c_weights.size(0) == N_FACTORS, "c_weights must have exactly N_FACTORS elements");
    TORCH_CHECK(num_steps > 0, "num_steps must be positive");
    TORCH_CHECK(dt > 0.0f, "dt must be positive");

    // Output tensor for terminal prices
    auto prices_out = torch::empty({num_paths}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    int blocks = (num_paths + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    lifted_heston_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        num_paths,
        num_steps,
        dt,
        S0,
        V0,
        rho,
        kappa,
        theta,
        sigma,
        c_weights.data_ptr<float>(),
        x_speeds.data_ptr<float>(),
        static_cast<unsigned long long>(seed),
        static_cast<unsigned long long>(call_index),
        prices_out.data_ptr<float>()
    );

    // Synchronize to ensure completion before Python resumes
    cudaDeviceSynchronize();

    return prices_out;
}

// Pybind11 Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simulate_lifted_heston", &simulate_lifted_heston,
          "Lifted Heston Monte Carlo Simulation (CUDA)",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("dt"),
          py::arg("S0"), py::arg("V0"), py::arg("rho"),
          py::arg("kappa"), py::arg("theta"), py::arg("sigma"),
          py::arg("c_weights"), py::arg("x_speeds"),
          py::arg("seed"), py::arg("call_index") = 0);
}

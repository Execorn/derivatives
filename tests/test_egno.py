import pytest
import torch
import numpy as np
from deepvol.surrogates.egno import EGNO, EGNOLayer, monte_carlo_basket_price
from deepvol.utils.gpu_lock import acquire_gpu_lock
import scipy.stats as stats

def black_scholes_call(S, K, T, r, q, sigma):
    """Analytical Black-Scholes European Call Price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return price

def test_egno_shape_consistency():
    """Verify that EGNO accepts inputs of varying batch sizes and asset counts, and outputs [B, 1]."""
    hidden_dim = 32
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=hidden_dim, num_layers=2)
    model.eval()
    
    # Test different batch sizes and asset counts
    for B in [1, 4, 16]:
        for N in [2, 5, 10]:
            x = torch.rand(B, N, 4)
            edge_attr = torch.rand(B, N, N, 1)
            g = torch.rand(B, 2)
            
            with torch.no_grad():
                out = model(x, edge_attr, g)
                
            assert out.shape == (B, 1), f"Expected shape {(B, 1)}, got {out.shape} for B={B}, N={N}"

def test_egno_permutation_invariance():
    """Verify that permuting the order of assets (nodes and corresponding edges) does not change the final price."""
    hidden_dim = 32
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=hidden_dim, num_layers=2)
    model.eval()
    
    B, N = 4, 5
    x = torch.rand(B, N, 4)
    edge_attr = torch.rand(B, N, N, 1)
    g = torch.rand(B, 2)
    
    # Define a random permutation of asset indices
    perm = torch.randperm(N)
    
    # Permute node features
    x_perm = x[:, perm, :]
    
    # Permute edge features (both rows and columns)
    edge_attr_perm = edge_attr[:, perm, :, :][:, :, perm, :]
    
    with torch.no_grad():
        out_orig = model(x, edge_attr, g)
        out_perm = model(x_perm, edge_attr_perm, g)
        
    # Check that prices are identical
    torch.testing.assert_close(out_orig, out_perm, rtol=1e-5, atol=1e-5)

def test_egno_layer_permutation_equivariance():
    """Verify that EGNOLayer is permutation equivariant for node and edge updates."""
    node_dim, edge_dim, global_dim, hidden_dim = 8, 4, 2, 16
    layer = EGNOLayer(node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim, hidden_dim=hidden_dim)
    layer.eval()
    
    B, N = 2, 4
    h = torch.rand(B, N, node_dim)
    e = torch.rand(B, N, N, edge_dim)
    g = torch.rand(B, global_dim)
    
    # Permutation
    perm = torch.randperm(N)
    h_perm = h[:, perm, :]
    e_perm = e[:, perm, :, :][:, :, perm, :]
    
    with torch.no_grad():
        h_out, e_out = layer(h, e, g)
        h_out_perm, e_out_perm = layer(h_perm, e_perm, g)
        
    # Apply permutation to original output to verify equivariance
    h_out_expected = h_out[:, perm, :]
    e_out_expected = e_out[:, perm, :, :][:, :, perm, :]
    
    torch.testing.assert_close(h_out_perm, h_out_expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(e_out_perm, e_out_expected, rtol=1e-5, atol=1e-5)

def test_egno_double_precision():
    """Verify that EGNO supports float64 inputs and performs double precision computations."""
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=16, num_layers=2)
    model.eval()
    
    x = torch.rand(2, 3, 4, dtype=torch.float64)
    edge_attr = torch.rand(2, 3, 3, 1, dtype=torch.float64)
    g = torch.rand(2, 2, dtype=torch.float64)
    
    with torch.no_grad():
        out = model(x, edge_attr, g)
        
    assert out.dtype == torch.float64, f"Expected float64 output, got {out.dtype}"
    assert model.node_proj.weight.dtype == torch.float64, "Model parameters should be converted to double precision"

def test_egno_volatility_clamping():
    """Verify that input volatility values below 0.01 (100 bps) are clamped to 0.01 to prevent singularities."""
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=16, num_layers=2)
    model.eval()
    
    B, N = 2, 3
    # Case 1: Volatility = 0.005 (below 0.01)
    x1 = torch.tensor([
        [[100.0, 0.005, 0.05, 0.0], [100.0, 0.02, 0.05, 0.0], [100.0, 0.03, 0.05, 0.0]],
        [[100.0, 0.015, 0.05, 0.0], [100.0, 0.002, 0.05, 0.0], [100.0, 0.04, 0.05, 0.0]]
    ], dtype=torch.float32)
    
    # Case 2: Volatility = 0.01 (exact clamped value)
    x2 = torch.tensor([
        [[100.0, 0.01, 0.05, 0.0], [100.0, 0.02, 0.05, 0.0], [100.0, 0.03, 0.05, 0.0]],
        [[100.0, 0.015, 0.05, 0.0], [100.0, 0.01, 0.05, 0.0], [100.0, 0.04, 0.05, 0.0]]
    ], dtype=torch.float32)
    
    edge_attr = torch.rand(B, N, N, 1)
    g = torch.rand(B, 2)
    
    with torch.no_grad():
        out1 = model(x1, edge_attr, g)
        out2 = model(x2, edge_attr, g)
        
    torch.testing.assert_close(out1, out2, rtol=1e-6, atol=1e-6)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_egno_cuda_compatibility():
    """Verify that EGNO can run on CUDA and produces outputs consistent with the CPU version."""
    acquire_gpu_lock()
    
    hidden_dim = 32
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=hidden_dim, num_layers=2)
    
    # CPU forward pass
    model.eval()
    x = torch.rand(4, 5, 4)
    edge_attr = torch.rand(4, 5, 5, 1)
    g = torch.rand(4, 2)
    
    with torch.no_grad():
        out_cpu = model(x, edge_attr, g)
        
    # GPU forward pass
    model_cuda = model.cuda()
    x_cuda = x.cuda()
    edge_attr_cuda = edge_attr.cuda()
    g_cuda = g.cuda()
    
    with torch.no_grad():
        out_cuda = model_cuda(x_cuda, edge_attr_cuda, g_cuda)
        
    torch.testing.assert_close(out_cpu, out_cuda.cpu(), rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_egno_torch_compile():
    """Verify that EGNO can be compiled using torch.compile and produces matching outputs."""
    acquire_gpu_lock()
    
    model = EGNO(node_in_dim=4, edge_in_dim=1, global_in_dim=2, hidden_dim=16, num_layers=2)
    model.cuda()
    model.eval()
    
    x = torch.rand(2, 3, 4).cuda()
    edge_attr = torch.rand(2, 3, 3, 1).cuda()
    g = torch.rand(2, 2).cuda()
    
    # Standard forward
    with torch.no_grad():
        out_standard = model(x, edge_attr, g)
        
    # Compiled forward
    compiled_model = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        out_compiled = compiled_model(x, edge_attr, g)
        
    torch.testing.assert_close(out_standard, out_compiled, rtol=1e-5, atol=1e-5)


# ─── Monte Carlo Pricer Tests ────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_monte_carlo_basket_price_perfect_correlation_limit():
    """
    Validate Monte Carlo pricer against analytical Black-Scholes Call price.
    In the limit of rho_ij = 1.0 (perfect correlation) and equal asset weights/vols,
    the multi-asset basket behaves exactly as a single-asset Black-Scholes call.
    """
    acquire_gpu_lock()
    
    S0 = 100.0
    vol = 0.20
    r_val = 0.05
    q_val = 0.02
    K_val = 100.0
    T_val = 1.0
    N = 3 # 3 assets
    
    spots = torch.tensor([[S0] * N], dtype=torch.float64)
    vols = torch.tensor([[vol] * N], dtype=torch.float64)
    r = torch.tensor([[r_val]], dtype=torch.float64)
    q = torch.tensor([[q_val] * N], dtype=torch.float64)
    K = torch.tensor([[K_val]], dtype=torch.float64)
    T = torch.tensor([[T_val]], dtype=torch.float64)
    
    # Correlation matrix: all ones (perfect correlation)
    correlations = torch.ones((1, N, N), dtype=torch.float64)
    w = torch.tensor([[1.0 / N] * N], dtype=torch.float64)
    
    # Run Monte Carlo
    with torch.no_grad():
        mc_price = monte_carlo_basket_price(
            spots=spots,
            vols=vols,
            r=r,
            q=q,
            correlations=correlations,
            w=w,
            K=K,
            T=T,
            num_paths=1048576, # 1M paths
            block_size=8192,
            device="cuda"
        )
        
    # Analytical Black-Scholes price
    bs_price = black_scholes_call(S=S0, K=K_val, T=T_val, r=r_val, q=q_val, sigma=vol)
    
    # Verify that the MC price is within 3 standard errors of the analytical price
    # Standard error for 1M paths is very small (approx 0.01)
    diff = abs(mc_price.item() - bs_price)
    assert diff < 0.03, f"MC price {mc_price.item():.4f} differed from analytical BS price {bs_price:.4f} by {diff:.4f}"

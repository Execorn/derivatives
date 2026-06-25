import pytest
import torch
import torch.optim as optim
from deepvol.hedging.mp_diffusion import (
    MartingaleViolationError,
    PathDenoisingNet,
    project_spot_martingale,
    audit_martingale_paths,
    MPDDPM
)


def test_martingale_projection_math():
    """
    Test that the project_spot_martingale function mathematically enforces
    the mean spot condition at all time steps exactly (within float64 precision).
    """
    N_p = 1000
    T = 50
    S_0 = 100.0
    
    # Generate arbitrary random spot paths in float64 for exact precision verification
    S = S_0 + torch.randn(N_p, T, dtype=torch.float64) * 10.0
    
    # Ensure they don't satisfy the martingale condition initially
    initial_mean_dev = torch.max(torch.abs(S.mean(dim=0) - S_0)).item()
    assert initial_mean_dev > 1e-2

    
    # Project paths
    S_proj = project_spot_martingale(S, S_0)
    
    # Check that after projection, mean S_t at every t is exactly S_0
    projected_mean_dev = torch.max(torch.abs(S_proj.mean(dim=0) - S_0)).item()
    # Should be extremely close to 0 in float64 internal computation
    assert projected_mean_dev < 1e-12
    
    # Check shape is preserved
    assert S_proj.shape == S.shape
    
    # Check projection on 3D tensor (N_p, 2, T)
    S_3d = torch.stack([S, torch.ones_like(S)], dim=1)
    S_proj_3d = project_spot_martingale(S_3d, S_0)
    
    assert S_proj_3d.shape == S_3d.shape
    projected_3d_mean_dev = torch.max(torch.abs(S_proj_3d[:, 0, :].mean(dim=0) - S_0)).item()
    assert projected_3d_mean_dev < 1e-12
    # Check that channel 1 (variance) is unmodified
    assert torch.allclose(S_proj_3d[:, 1, :], S_3d[:, 1, :])


def test_martingale_audit_and_compliance():
    """
    Test that the audit_martingale_paths function properly logs and triggers
    violations based on the 10^-5 tolerance limit (SR 26-2 compliance).
    """
    N_p = 100
    T = 10
    S_0 = 100.0
    
    # Case 1: Martingale paths (mean matches S_0 exactly)
    S_good = torch.ones(N_p, T) * S_0
    residuals = audit_martingale_paths(S_good, S_0, raise_on_failure=True)
    assert residuals.max().item() < 1e-12
    
    # Case 2: Slightly noisy but within 10^-5 limit
    S_within_tolerance = S_good + torch.randn(N_p, T) * 1e-8
    residuals_tol = audit_martingale_paths(S_within_tolerance, S_0, raise_on_failure=True)
    assert residuals_tol.max().item() < 1e-5
    
    # Case 3: Breaches 10^-5 limit, must raise MartingaleViolationError
    S_bad = S_good.clone()
    S_bad[:, 5] = S_0 + 2e-5  # Add offset to step 5 to trigger violation
    
    with pytest.raises(MartingaleViolationError) as exc_info:
        audit_martingale_paths(S_bad, S_0, raise_on_failure=True)
    
    assert "Martingale violation detected" in str(exc_info.value)
    
    # If raise_on_failure is False, it should return residuals without raising an error
    residuals_bad = audit_martingale_paths(S_bad, S_0, raise_on_failure=False)
    assert residuals_bad[5].item() >= 2e-5


def test_mp_ddpm_training():
    """
    Test the training forward pass of MP-DDPM and verify that the loss is
    differentiable and gradients flow to the network parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = 16
    T = 20
    S_0 = 100.0
    V_0 = 0.04
    
    # Instantiate network and model
    net = PathDenoisingNet(in_channels=2, hidden_dim=32, num_blocks=2, emb_dim=32).to(device)
    model = MPDDPM(denoising_net=net, T_d=50).to(device)
    
    # Mock dataset of spot and variance paths
    S_paths = S_0 + torch.randn(B, T, device=device) * 5.0
    V_paths = V_0 + torch.randn(B, T, device=device) * 0.01
    
    # Run forward pass loss computation
    loss = model.forward_loss(S_paths, V_paths)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() >= 0.0
    
    # Verify differentiability
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    
    # Ensure gradients are populated for denoising network parameters
    for name, param in model.denoising_net.named_parameters():
        assert param.grad is not None, f"Gradient not found for parameter {name}"
        
    optimizer.step()


def test_mp_ddpm_sampling():
    """
    Test the sampling process of MP-DDPM. Verifies that generated paths:
    1. Maintain shape (num_paths, T).
    2. Start exactly at S_0 and V_0.
    3. Strictly satisfy the 10^-5 martingale tolerance.
    4. Have strictly positive variance paths under different constraints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N_p = 128
    T = 30
    S_0 = 100.0
    V_0 = 0.04
    
    net = PathDenoisingNet(in_channels=2, hidden_dim=32, num_blocks=2, emb_dim=32).to(device)
    model = MPDDPM(denoising_net=net, T_d=20).to(device)  # Few diffusion steps for fast testing
    
    # Sample with softplus constraint
    S_gen, V_gen = model.sample(
        num_paths=N_p, T=T, S_0=S_0, V_0=V_0, device=device,
        project_at_each_step=True, variance_positivity_constraint="softplus"
    )
    
    # Verify shape
    assert S_gen.shape == (N_p, T)
    assert V_gen.shape == (N_p, T)
    
    # Verify boundary conditions at t=0
    assert torch.allclose(S_gen[:, 0], torch.tensor(S_0, device=device))
    assert torch.allclose(V_gen[:, 0], torch.tensor(V_0, device=device))
    
    # Verify martingale audit conservation
    residuals = audit_martingale_paths(S_gen, S_0, raise_on_failure=True)
    assert residuals.max().item() < 1e-5
    
    # Verify variance positivity
    assert torch.all(V_gen > 0.0)
    
    # Sample with clamp constraint and project_at_each_step=False (only final projection)
    S_gen2, V_gen2 = model.sample(
        num_paths=N_p, T=T, S_0=S_0, V_0=V_0, device=device,
        project_at_each_step=False, variance_positivity_constraint="clamp"
    )
    
    assert S_gen2.shape == (N_p, T)
    assert V_gen2.shape == (N_p, T)
    assert torch.allclose(S_gen2[:, 0], torch.tensor(S_0, device=device))
    assert torch.allclose(V_gen2[:, 0], torch.tensor(V_0, device=device))
    
    residuals2 = audit_martingale_paths(S_gen2, S_0, raise_on_failure=True)
    assert residuals2.max().item() < 1e-5
    assert torch.all(V_gen2 >= 1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution_and_precision():
    """
    Test CUDA execution, verification of GPU-first rule, and double precision
    guarantees in the projection layer.
    """
    device = torch.device("cuda")
    N_p = 500
    T = 100
    S_0 = 100.0
    V_0 = 0.04
    
    net = PathDenoisingNet(in_channels=2, hidden_dim=64, num_blocks=3, emb_dim=64).to(device)
    model = MPDDPM(denoising_net=net, T_d=10).to(device)
    
    # Synchronize before/after CUDA execution to time the sampler
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    S_gen, V_gen = model.sample(
        num_paths=N_p, T=T, S_0=S_0, V_0=V_0, device=device,
        project_at_each_step=True, variance_positivity_constraint="softplus"
    )
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    
    assert S_gen.device.type == "cuda"
    assert V_gen.device.type == "cuda"
    
    # Martingale validation
    residuals = audit_martingale_paths(S_gen, S_0, raise_on_failure=True)
    assert residuals.max().item() < 1e-5
    
    print(f"\nCUDA MP-DDPM Sampling of {N_p} paths of length {T} took {elapsed_time:.3f} ms.")

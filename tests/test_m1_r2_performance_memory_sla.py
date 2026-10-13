"""tests/test_m1_r2_performance_memory_sla.py

Milestone 1 Iteration 2 Performance & Memory SLA Benchmark Test Suite:
1. Memory Budget Verification:
   - Peak GPU tensor memory during 1601-node quadrature forward pass (<= 165 MB target).
   - Peak GPU tensor memory during label_state_batch autograd backward propagation with M=1 (<= 256 MB ceiling).
   - Stress challenge on strike scaling M in {1, 2, 4, 8}.
2. Throughput & Scaling SLA:
   - Benchmark t_aad vs t_fwd across 201-node and 1601-node quadrature.
   - Verify t_aad < 10.0 * t_fwd computational efficiency scaling.
3. CUDAGraphs Memory Safety:
   - Verify that cloning sigma_f64 in PIVOTImpliedVolFunction.forward prevents static buffer overwrite.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest
import torch

from deepvol.hedging.pivot_iv import PIVOTImpliedVolFunction
from deepvol.market.svi_slice import RawSVI
from deepvol.models.fx_cross_copula import (
    CopulaConfig,
    CrossState,
    GaussianCopulaCrossEngine,
)
from deepvol.training.generate_cross_shards import label_state_batch

# Ensure tests dir is on sys.path
tests_dir = str(Path(__file__).parent.resolve())
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from test_cross_data_generation import _oracle_copula_cross_primal  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_engine_price_peak_memory_target_165mb() -> None:
    """Verify that Copula 1601-node forward pass meets the <= 165 MB target."""
    dev = torch.device("cuda")
    torch.cuda.empty_cache()

    cfg = CopulaConfig(n_nodes=1601, strike_chunk=1, device="cuda")
    engine = GaussianCopulaCrossEngine(cfg)

    torch.cuda.reset_peak_memory_stats(dev)
    cstate = CrossState(
        F1=torch.ones(1, device=dev, dtype=torch.float64, requires_grad=True),
        F2=torch.ones(1, device=dev, dtype=torch.float64, requires_grad=True),
        T=torch.tensor([0.5], device=dev, dtype=torch.float64),
        svi1=RawSVI(torch.tensor([[0.04, 0.1, -0.2, 0.0, 0.1]], device=dev, dtype=torch.float64)),
        svi2=RawSVI(torch.tensor([[0.04, 0.1, -0.2, 0.0, 0.1]], device=dev, dtype=torch.float64)),
        rho=torch.tensor([0.0], device=dev, dtype=torch.float64, requires_grad=True),
    )

    _ = engine.price(cstate, torch.tensor([[1.0]], device=dev, dtype=torch.float64, requires_grad=True))
    fwd_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    # 156.51 MB measured vs 165 MB target
    assert fwd_peak_mb <= 165.0, (
        f"Forward peak memory ({fwd_peak_mb:.2f} MB) exceeds target 165.0 MB"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_label_state_batch_peak_memory_ceiling_256mb() -> None:
    """Verify that label_state_batch at S=1, M=1 stays strictly <= 256 MB ceiling."""
    dev = "cuda"
    torch.cuda.empty_cache()

    st = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050,
    ]], dtype=torch.float64, device=dev)
    z = torch.tensor([[0.0]], dtype=torch.float64, device=dev)

    cfg = CopulaConfig(n_nodes=1601, strike_chunk=1, device=dev)
    engine = GaussianCopulaCrossEngine(cfg)

    # Ensure clean allocator state before measuring
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    labels = label_state_batch(st, z, engine)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    assert labels["dc_dx"].shape == (1, 1, 13)
    assert labels["deps_dx"].shape == (1, 1, 13)
    # Ceiling check: must strictly satisfy <= 256.0 MB ceiling
    assert peak_mb <= 256.0, f"Peak memory ({peak_mb:.2f} MB) exceeds 256.0 MB ceiling"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_label_state_batch_strike_scaling_stays_within_256mb_ceiling() -> None:
    """Verify that autograd strike chunking maintains peak memory <= 165.0 MB target across M in {1, 2, 4, 8}."""
    dev = "cuda"
    st = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050,
    ]], dtype=torch.float64, device=dev)

    cfg = CopulaConfig(n_nodes=1601, strike_chunk=1, device=dev)
    engine = GaussianCopulaCrossEngine(cfg)

    for M in [1, 2, 4, 8]:
        z = torch.linspace(-0.5, 0.5, M, dtype=torch.float64, device=dev).unsqueeze(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        labels = label_state_batch(st, z, engine)
        peak_m = torch.cuda.max_memory_allocated() / (1024 ** 2)
        assert labels["dc_dx"].shape == (1, M, 13)
        assert labels["deps_dx"].shape == (1, M, 13)
        assert peak_m <= 165.0, f"M={M} peak memory ({peak_m:.2f} MB) exceeds target 165.0 MB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_generate_shards_production_1601_nodes_peak_vram_under_256mb(tmp_path: Path) -> None:
    """Verifies that generate_shards executes a production 1601-node batch at M=64 strikes
    under the strict <= 256 MB hard ceiling (target <= 165 MB).
    """
    import gc
    import json
    import numpy as np
    from deepvol.training.data_generation_cross import SamplerConfig
    from deepvol.training.generate_cross_shards import ShardConfig, compute_sha256, generate_shards

    dev = "cuda"
    torch.cuda.empty_cache()
    gc.collect()

    out_dir = tmp_path / "prod_shards"
    cfg = ShardConfig(
        out_dir=out_dir,
        records_per_shard=64,
        sampler=SamplerConfig(log2_states=2, strikes_per_state=64, seed=20261011),
        copula=CopulaConfig(n_nodes=1601, strike_chunk=1, device=dev),
    )

    torch.cuda.reset_peak_memory_stats(dev)
    manifest_p = generate_shards(cfg)
    peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    # 1. Verify strict memory ceiling
    assert peak_vram_mb <= 256.0, (
        f"Production shard generation peak VRAM ({peak_vram_mb:.2f} MB) exceeded 256.0 MB ceiling!"
    )

    # 2. Verify manifest integrity
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["copula_nodes"] == 1601
    assert len(manifest["file_checksums"]) > 0

    # 3. Verify shard records and finite values
    for rel_path, expected_hash in manifest["file_checksums"].items():
        shard_path = out_dir / rel_path
        actual_hash = compute_sha256(shard_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"

        data = np.load(shard_path)
        for key in ["x", "c", "dc_dx", "eps_star", "deps_dx", "sigma_h1", "sigma_cop"]:
            arr = data[key]
            assert np.all(np.isfinite(arr)), f"NaN/Inf detected in {rel_path} field {key}"
            assert len(arr) == 64
        assert data["x"].shape == (64, 13)
        assert data["dc_dx"].shape == (64, 13)
        assert data["deps_dx"].shape == (64, 13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_aad_scaling_ratio_under_10x_sla() -> None:
    """Verify that AAD gradient computation runtime is strictly < 10x forward runtime."""
    dev = "cuda"
    engine = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=201, device=dev))

    st = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050,
    ]], dtype=torch.float64, device=dev)
    z = torch.tensor([[0.5]], dtype=torch.float64, device=dev)

    # Warmup
    _ = label_state_batch(st, z, engine)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _oracle_copula_cross_primal(st[0], torch.tensor(0.5, dtype=torch.float64, device=dev), engine)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    label_state_batch(st, z, engine)
    torch.cuda.synchronize()
    t_aad = time.perf_counter() - t0

    ratio = t_aad / max(t_fwd, 1e-6)
    assert ratio < 10.0, (
        f"AAD execution scaling ratio ({ratio:.2f}x) exceeds 10x SLA limit (t_fwd={t_fwd:.3f}s, t_aad={t_aad:.3f}s)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory SLA tests")
def test_cudagraphs_cloning_prevents_buffer_corruption() -> None:
    """Verify that compiled PIVOT solver with clone() operates safely across sequential autograd passes."""
    dev = "cuda"

    prices = [
        torch.tensor([0.05, 0.10], dtype=torch.float64, device=dev, requires_grad=True),
        torch.tensor([0.08, 0.12], dtype=torch.float64, device=dev, requires_grad=True),
        torch.tensor([0.03, 0.15], dtype=torch.float64, device=dev, requires_grad=True),
    ]
    S = torch.tensor(1.0, dtype=torch.float64, device=dev)
    K = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    T = torch.tensor(1.0, dtype=torch.float64, device=dev)

    # Multiple successive forward evaluations storing graph outputs
    iv_outputs = []
    for p in prices:
        iv = PIVOTImpliedVolFunction.apply(p, S, K, T, 0.0, 0.0, True, 1e-4)  # type: ignore[no-untyped-call]
        iv_outputs.append(iv)

    # Reversed-order backward execution stresses static buffer reuse
    for i, iv in enumerate(reversed(iv_outputs)):
        loss = (iv ** 2).sum()
        loss.backward()
        p_orig = list(reversed(prices))[i]
        assert p_orig.grad is not None
        assert torch.all(torch.isfinite(p_orig.grad)), f"NaN/Inf gradient in backward pass {i}"

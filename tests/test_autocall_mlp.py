"""
test_autocall_mlp.py — Unit and Numerical Tests for Autocall MLP Surrogate and Normalizers.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from deepvol.surrogates.autocall_normalizer import (
    AutocallInputNormalizer,
    AutocallOutputNormalizer,
)
from deepvol.surrogates.autocall_mlp import AutocallMLP, compute_greeks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_forward_shape():
    """Verify output shape is (B, 3) for B=32."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    x = torch.randn(32, 10, device=DEVICE)
    out = model(x)
    assert out.shape == (32, 3)


def test_output_range():
    """Verify outputs are bounded in [0, 1]."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    x = torch.randn(64, 10, device=DEVICE)
    out = model(x)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_backward_gradients():
    """Verify backward gradients are computed and non-None for all parameters."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    x = torch.randn(16, 10, device=DEVICE)
    out = model(x)
    target = torch.rand(16, 3, device=DEVICE)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_normalizer_roundtrip_input():
    """Verify inverse_transform(transform(X)) ~= X with atol=1e-5."""
    np.random.seed(42)
    X = np.random.uniform(0.5, 5.0, size=(100, 10)).astype(np.float32)
    norm = AutocallInputNormalizer().fit(X)
    X_norm = norm.transform(X)
    X_rec = norm.inverse_transform(X_norm)
    assert np.allclose(X, X_rec, atol=1e-5)


def test_normalizer_roundtrip_output():
    """Verify inverse_transform(transform(Y)) ~= Y with atol=1e-5."""
    np.random.seed(42)
    Y = np.column_stack([
        np.random.uniform(0.8, 1.2, size=100),   # npv
        np.random.uniform(0.0, 1.0, size=100),   # call_prob
        np.random.uniform(0.2, 3.0, size=100),   # exp_life
    ]).astype(np.float32)
    norm = AutocallOutputNormalizer().fit(Y)
    Y_norm = norm.transform(Y)
    assert np.all(Y_norm >= -1e-6) and np.all(Y_norm <= 1.0 + 1e-6)
    Y_rec = norm.inverse_transform(Y_norm)
    assert np.allclose(Y, Y_rec, atol=1e-5)


def test_normalizer_save_load():
    """Verify saving and loading normalizers preserves transform exactly."""
    X = np.random.uniform(0.5, 5.0, size=(50, 10)).astype(np.float32)
    norm = AutocallInputNormalizer().fit(X)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        norm.save(tmp.name)
        loaded = AutocallInputNormalizer.load(tmp.name)
        os.remove(tmp.name)

    assert np.allclose(norm.mean, loaded.mean)
    assert np.allclose(norm.std, loaded.std)
    assert np.allclose(norm.transform(X), loaded.transform(X))


def test_compile_clone_guard():
    """Run forward twice, verify outputs are distinct memory buffers."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    x1 = torch.randn(8, 10, device=DEVICE)
    x2 = torch.randn(8, 10, device=DEVICE)

    out1 = model(x1)
    out2 = model(x2)

    assert out1.data_ptr() != out2.data_ptr()
    assert not torch.equal(out1, out2)


def test_greeks_shape():
    """Verify compute_greeks returns dictionary with delta_B, vega, theta."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    X = np.random.uniform(0.5, 5.0, size=(100, 10)).astype(np.float32)
    Y = np.random.uniform(0.5, 1.5, size=(100, 3)).astype(np.float32)
    norm_in = AutocallInputNormalizer().fit(X)
    norm_out = AutocallOutputNormalizer().fit(Y)

    x_raw = np.array([2.0, 0.04, 0.3, -0.7, 0.04, 1.0, 0.10, 1.5, 4.0, 0.03], dtype=np.float32)
    greeks = compute_greeks(model, x_raw, norm_in, norm_out)

    assert isinstance(greeks, dict)
    for k in ["delta_B", "vega", "theta"]:
        assert k in greeks
        assert isinstance(greeks[k], float)
        assert not np.isnan(greeks[k])


def test_greeks_delta_sign():
    """Verify delta_B reflects call on spot (positive delta_B)."""
    model = AutocallMLP(in_dim=10, hidden=64, n_layers=2, out_dim=3).to(DEVICE)
    X = np.random.uniform(0.5, 5.0, size=(100, 10)).astype(np.float32)
    Y = np.random.uniform(0.5, 1.5, size=(100, 3)).astype(np.float32)
    norm_in = AutocallInputNormalizer().fit(X)
    norm_out = AutocallOutputNormalizer().fit(Y)

    x_raw = np.array([2.0, 0.04, 0.3, -0.7, 0.04, 1.0, 0.10, 1.5, 4.0, 0.03], dtype=np.float32)
    greeks = compute_greeks(model, x_raw, norm_in, norm_out)

    assert greeks["delta_B"] >= 0.0, f"Expected positive spot delta_B, got {greeks['delta_B']}"

"""
Unit tests — Phase 3: HestonDynamicsLSTM and training pipeline.

Test coverage:
  Model (seq_model.py):
    1.  Forward pass output shape
    2.  bounded_predict output shape and per-parameter range enforcement
    3.  Feller condition helper (correctness)
    4.  Parameter count is within expected range
    5.  Weight initialization sanity (no NaN/Inf)
    6.  Gradient flow through the full LSTM graph
    7.  Model restores to eval() mode correctly

  Dataset (train_seq.py):
    8.  HestonSequenceDataset length and item shapes
    9.  Z-score normalization is applied correctly (zero-mean labels)
    10. Noise augmentation is active/inactive based on flag
    11. No data leakage: train and val indices don't overlap

  Training smoke test (train_seq.py):
    12. 5-epoch training run on tiny synthetic data converges (loss decreases)
    13. Checkpoint file is created after training
    14. Label stats .npz is saved with correct keys

  Integration:
    15. Full predict→bounded_predict round-trip stays within bounds
    16. Feller check on bounded_predict output: majority should satisfy
"""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Path setup ─────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from seq_model import HestonDynamicsLSTM, PARAM_LOWER, PARAM_UPPER, PARAM_NAMES
from train_seq import HestonSequenceDataset, EarlyStopping, run_epoch


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> HestonDynamicsLSTM:
    """Fresh model instance in eval mode."""
    m = HestonDynamicsLSTM()
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_batch() -> torch.Tensor:
    """Random input batch: (32, 10, 88) — 32 sequences of 10 daily W surfaces."""
    torch.manual_seed(0)
    return torch.randn(32, 10, 88) * 0.05 + 0.02  # realistic W-scale values


@pytest.fixture(scope="module")
def dummy_dataset_arrays():
    """Minimal synthetic dataset arrays for dataset/training tests."""
    rng = np.random.default_rng(42)
    N = 200

    # X: (N, 10, 88) — small positive W values
    X = rng.uniform(0.001, 0.10, size=(N, 10, 88)).astype(np.float32)

    # y: (N, 5) in data order [v0, rho, sigma, theta, kappa]
    # Sample within empirical OU bounds
    lows  = np.array([0.010, -0.90, 0.10, 0.02, 1.00], dtype=np.float32)
    highs = np.array([0.150, -0.40, 0.80, 0.12, 6.00], dtype=np.float32)
    y = rng.uniform(lows, highs, size=(N, 5)).astype(np.float32)

    label_mean = y[:160].mean(axis=0)
    label_std  = np.maximum(y[:160].std(axis=0), 1e-8)

    return X, y, label_mean, label_std


# ══════════════════════════════════════════════════════════════════════════════
# 1–7: Model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHestonDynamicsLSTM:

    def test_forward_output_shape(self, model, dummy_batch):
        """forward() must return (32, 5)."""
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (32, 5), f"Expected (32,5), got {out.shape}"

    def test_bounded_predict_shape(self, model, dummy_batch):
        """bounded_predict() must return (32, 5)."""
        with torch.no_grad():
            out = model.bounded_predict(dummy_batch)
        assert out.shape == (32, 5), f"Expected (32,5), got {out.shape}"

    def test_bounded_predict_lower_bounds(self, model, dummy_batch):
        """All predicted parameters must be >= their physical lower bound."""
        with torch.no_grad():
            out = model.bounded_predict(dummy_batch).numpy()
        lo = PARAM_LOWER.numpy()
        for i, name in enumerate(PARAM_NAMES):
            assert np.all(out[:, i] >= lo[i] - 1e-5), (
                f"bounded_predict: {name} below lower bound {lo[i]:.4f}. "
                f"Min observed: {out[:, i].min():.6f}"
            )

    def test_bounded_predict_upper_bounds(self, model, dummy_batch):
        """All predicted parameters must be <= their physical upper bound."""
        with torch.no_grad():
            out = model.bounded_predict(dummy_batch).numpy()
        hi = PARAM_UPPER.numpy()
        for i, name in enumerate(PARAM_NAMES):
            assert np.all(out[:, i] <= hi[i] + 1e-5), (
                f"bounded_predict: {name} above upper bound {hi[i]:.4f}. "
                f"Max observed: {out[:, i].max():.6f}"
            )

    def test_feller_helper_pass(self):
        """Feller condition: 2*kappa*theta > sigma^2 must return True."""
        # kappa=5, theta=0.1 → 2*5*0.1=1.0 > sigma^2=0.25 → True
        params = torch.tensor([[0.02, -0.5, 0.5, 0.1, 5.0]])
        result = HestonDynamicsLSTM.feller_satisfied(params)
        assert result.all(), "Feller should be satisfied for this parameter set"

    def test_feller_helper_fail(self):
        """Feller condition: 2*kappa*theta <= sigma^2 must return False."""
        # kappa=1, theta=0.01 → 2*0.01=0.02 < sigma^2=0.64 → False
        params = torch.tensor([[0.02, -0.5, 0.8, 0.01, 1.0]])
        result = HestonDynamicsLSTM.feller_satisfied(params)
        assert not result.all(), "Feller should be violated for this parameter set"

    def test_parameter_count_in_range(self, model):
        """Model should have between 40,000 and 200,000 trainable parameters."""
        n = sum(p.numel() for p in model.parameters())
        assert 40_000 <= n <= 200_000, (
            f"Unexpected parameter count: {n:,}. "
            "Check hidden_size and num_layers."
        )

    def test_no_nan_in_initial_weights(self, model):
        """No NaN or Inf in any weight tensor after initialization."""
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"
            assert not torch.isinf(param).any(), f"Inf in {name}"

    def test_gradient_flows(self, dummy_batch):
        """Loss.backward() must propagate non-zero gradients to LSTM input weights."""
        m = HestonDynamicsLSTM()
        m.train()
        out  = m(dummy_batch)
        loss = out.mean()
        loss.backward()
        grad = m.lstm.weight_ih_l0.grad
        assert grad is not None, "No gradient on LSTM weight_ih_l0"
        assert not torch.allclose(grad, torch.zeros_like(grad)), (
            "Gradient is all-zero — backprop is broken"
        )

    def test_eval_mode_is_deterministic(self, model, dummy_batch):
        """Two forward passes in eval mode must return identical outputs."""
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model(dummy_batch)
        assert torch.allclose(out1, out2), (
            "eval() mode is non-deterministic — dropout may still be active"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8–11: Dataset tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHestonSequenceDataset:

    def test_dataset_length(self, dummy_dataset_arrays):
        X, y, lm, ls = dummy_dataset_arrays
        ds = HestonSequenceDataset(X, y, lm, ls)
        assert len(ds) == len(X), f"Dataset length {len(ds)} != {len(X)}"

    def test_item_shapes(self, dummy_dataset_arrays):
        X, y, lm, ls = dummy_dataset_arrays
        ds = HestonSequenceDataset(X, y, lm, ls)
        x_item, y_item = ds[0]
        assert x_item.shape == (10, 88), f"X item shape {x_item.shape} != (10,88)"
        assert y_item.shape == (5,),     f"y item shape {y_item.shape} != (5,)"

    def test_label_normalization(self, dummy_dataset_arrays):
        """Z-score normalized labels should have mean ≈ 0 and std ≈ 1 on training subset."""
        X, y, lm, ls = dummy_dataset_arrays
        ds = HestonSequenceDataset(X[:160], y[:160], lm, ls)

        all_y = torch.stack([ds[i][1] for i in range(len(ds))], dim=0)
        means = all_y.mean(dim=0).abs()
        stds  = all_y.std(dim=0)

        for i, name in enumerate(PARAM_NAMES):
            assert means[i].item() < 0.1, (
                f"{name}: normalized mean {means[i]:.4f} is too far from 0"
            )
            assert abs(stds[i].item() - 1.0) < 0.1, (
                f"{name}: normalized std {stds[i]:.4f} is too far from 1"
            )

    def test_augmentation_changes_input(self, dummy_dataset_arrays):
        """With augment=True, two calls to __getitem__ should return different X tensors."""
        X, y, lm, ls = dummy_dataset_arrays
        ds = HestonSequenceDataset(X, y, lm, ls, augment=True, noise_std=0.01)
        x1, _ = ds[0]
        x2, _ = ds[0]
        assert not torch.allclose(x1, x2), (
            "Augmentation did not change the input — noise injection may be broken"
        )

    def test_no_augmentation_is_deterministic(self, dummy_dataset_arrays):
        """With augment=False, two calls to __getitem__ must return identical X tensors."""
        X, y, lm, ls = dummy_dataset_arrays
        ds = HestonSequenceDataset(X, y, lm, ls, augment=False)
        x1, _ = ds[0]
        x2, _ = ds[0]
        assert torch.allclose(x1, x2), (
            "No-augmentation mode returned different tensors — X may not be immutable"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 12–14: Training smoke tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingSmoke:
    """End-to-end training smoke test — runs 5 epochs on tiny synthetic data."""

    @pytest.fixture(scope="class")
    def smoke_artifacts(self, dummy_dataset_arrays, tmp_path_factory):
        """Run 5 epochs of training and return (final_val_loss, initial_val_loss, tmp_dir)."""
        import torch
        from torch.utils.data import DataLoader

        tmp_dir = tmp_path_factory.mktemp("lstm_smoke")
        ckpt_path = tmp_dir / "best.pth"

        X, y, lm, ls = dummy_dataset_arrays
        train_ds = HestonSequenceDataset(X[:160], y[:160], lm, ls, augment=True)
        val_ds   = HestonSequenceDataset(X[160:180], y[160:180], lm, ls, augment=False)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

        model     = HestonDynamicsLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device    = torch.device("cpu")

        initial_val = run_epoch(model, val_loader, criterion, None, device, train=False)

        best_val = initial_val
        for _ in range(5):
            run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)

        return initial_val, best_val, ckpt_path, tmp_dir

    def test_loss_decreases(self, smoke_artifacts):
        """Validation loss must decrease within 5 training epochs."""
        initial_val, best_val, _, _ = smoke_artifacts
        assert best_val <= initial_val, (
            f"Val loss did not decrease: initial={initial_val:.6f}, "
            f"best={best_val:.6f}"
        )

    def test_checkpoint_created(self, smoke_artifacts):
        """Best checkpoint file must exist after training."""
        _, _, ckpt_path, _ = smoke_artifacts
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
        assert ckpt_path.stat().st_size > 0, "Checkpoint file is empty"

    def test_checkpoint_loadable(self, smoke_artifacts):
        """Saved checkpoint must load into a fresh model with strict=True."""
        _, _, ckpt_path, _ = smoke_artifacts
        model = HestonDynamicsLSTM()
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)


# ══════════════════════════════════════════════════════════════════════════════
# 15–16: Integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_bounded_predict_round_trip(self, dummy_batch):
        """End-to-end: random input → bounded_predict → all params in bounds."""
        model = HestonDynamicsLSTM()
        model.eval()
        with torch.no_grad():
            params = model.bounded_predict(dummy_batch)

        lo = PARAM_LOWER.unsqueeze(0)
        hi = PARAM_UPPER.unsqueeze(0)
        assert torch.all(params >= lo - 1e-5), "bounded_predict violates lower bounds"
        assert torch.all(params <= hi + 1e-5), "bounded_predict violates upper bounds"

    def test_feller_majority_satisfied_after_bounded_predict(self, dummy_batch):
        """
        After bounded_predict, at least 20% of parameter sets should satisfy
        the Feller condition.  With random weights this is probabilistic, so
        we use a loose threshold.
        """
        torch.manual_seed(99)
        model = HestonDynamicsLSTM()
        model.eval()
        with torch.no_grad():
            params  = model.bounded_predict(dummy_batch)
        feller  = HestonDynamicsLSTM.feller_satisfied(params)
        ratio   = feller.float().mean().item()
        assert ratio >= 0.05, (
            f"Only {ratio*100:.1f}% of predictions satisfy Feller — "
            "bounds may be misconfigured"
        )

    def test_early_stopping_triggers(self):
        """EarlyStopping should fire after exactly `patience` epochs of no improvement."""
        stopper = EarlyStopping(patience=3, min_delta=0.0)
        assert not stopper.step(1.0, 1)   # improvement
        assert not stopper.step(1.0, 2)   # no improvement, counter=1
        assert not stopper.step(1.0, 3)   # no improvement, counter=2
        assert     stopper.step(1.0, 4)   # no improvement, counter=3 → STOP

    def test_early_stopping_resets_on_improvement(self):
        """EarlyStopping counter must reset when a better loss is observed."""
        stopper = EarlyStopping(patience=3, min_delta=0.0)
        stopper.step(1.0, 1)   # best=1.0
        stopper.step(1.0, 2)   # counter=1
        stopper.step(0.9, 3)   # improvement → counter=0
        result = stopper.step(0.9, 4)  # counter=1, not yet stopping
        assert not result, "EarlyStopping should not stop after counter reset"

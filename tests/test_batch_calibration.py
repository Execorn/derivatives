"""
Tests for §P2-B5 GPU Batch Calibration.

Uses mocked market data to avoid network calls. Verifies the dataclass,
I/O round-trip, DataFrame conversion, and calibration pipeline.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from deepvol.calibration.batch_calibration import (
    CalibrationResult,
    calibrate_batch,
    calibrate_single,
    load_results,
    results_to_dataframe,
    save_results,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_result(
    date: str = "2024-01-02",
    currency: str = "SPX",
    converged: bool = True,
    rmse_bps: float = 30.0,
) -> CalibrationResult:
    return CalibrationResult(
        date=date,
        currency=currency,
        params={"kappa": 2.5, "theta": 0.08, "sigma": 0.5,
                "rho": -0.5, "v0": 0.08, "H": 0.08},
        rmse_bps=rmse_bps,
        runtime_ms=120.0,
        converged=converged,
        surface=np.full((8, 11), 0.25),
    )


# ── CalibrationResult dataclass ────────────────────────────────────────────────

def test_calibration_result_creation():
    r = make_result()
    assert r.date == "2024-01-02"
    assert r.currency == "SPX"
    assert r.converged is True
    assert r.rmse_bps == 30.0
    assert r.surface.shape == (8, 11)


def test_calibration_result_default_surface_none():
    r = CalibrationResult(
        date="2024-01-02", currency="SPX",
        params={"kappa": 1.0, "theta": 0.08, "sigma": 0.5,
                "rho": -0.5, "v0": 0.08, "H": 0.08},
        rmse_bps=50.0, runtime_ms=100.0, converged=True,
    )
    assert r.surface is None


def test_calibration_result_to_dict():
    r   = make_result()
    d   = r.to_dict()
    assert d["date"] == "2024-01-02"
    assert "params" in d
    assert "surface" in d
    assert isinstance(d["surface"], list)   # np.ndarray → list for JSON


def test_calibration_result_from_dict_roundtrip():
    r1 = make_result()
    d  = r1.to_dict()
    r2 = CalibrationResult.from_dict(d)
    assert r1.date      == r2.date
    assert r1.currency  == r2.currency
    assert r1.converged == r2.converged
    assert abs(r1.rmse_bps - r2.rmse_bps) < 1e-9
    assert r2.surface is not None
    assert r2.surface.shape == (8, 11)
    np.testing.assert_allclose(r1.surface, r2.surface)


# ── results_to_dataframe ───────────────────────────────────────────────────────

def test_results_to_dataframe_columns():
    results = [
        make_result("2024-01-02", converged=True),
        make_result("2024-08-05", converged=False, rmse_bps=120.0),
    ]
    df = results_to_dataframe(results)
    expected_cols = {"date", "currency", "kappa", "theta", "sigma",
                     "rho", "v0", "H", "rmse_bps", "runtime_ms", "converged"}
    assert expected_cols <= set(df.columns)
    assert len(df) == 2


def test_results_to_dataframe_values():
    r  = make_result("2024-01-02", converged=True, rmse_bps=42.0)
    df = results_to_dataframe([r])
    assert df.iloc[0]["date"] == "2024-01-02"
    assert abs(df.iloc[0]["rmse_bps"] - 42.0) < 1e-9
    assert df.iloc[0]["converged"] == True
    assert abs(df.iloc[0]["kappa"] - 2.5) < 1e-9


def test_results_to_dataframe_empty():
    import pandas as pd
    df = results_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── save_results / load_results ────────────────────────────────────────────────

def test_save_load_roundtrip():
    results = [
        make_result("2024-01-02", converged=True,  rmse_bps=28.0),
        make_result("2024-08-05", converged=False, rmse_bps=115.0),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        save_results(results, path)
        loaded = load_results(path)

        assert len(loaded) == 2
        for orig, load in zip(results, loaded):
            assert orig.date      == load.date
            assert orig.converged == load.converged
            assert abs(orig.rmse_bps - load.rmse_bps) < 1e-9
            np.testing.assert_allclose(orig.surface, load.surface)
    finally:
        Path(path).unlink(missing_ok=True)


def test_save_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "sub" / "dir" / "results.json")
        save_results([make_result()], path)
        assert Path(path).exists()


def test_load_results_json_is_valid():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
        json.dump([], f)

    try:
        loaded = load_results(path)
        assert loaded == []
    finally:
        Path(path).unlink(missing_ok=True)


# ── calibrate_single with synthetic surface ────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_surface():
    """Generate a (8,11) surface from known FNO params."""
    import torch
    import deepvol.calibration.calibrate_bfgs as calibrate
    calibrate._load_normalizers("v3")
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer

    weights  = Path(__file__).parents[1] / "artifacts/weights/fno_v3_final_prod.pth"
    pn_path  = Path(__file__).parents[1] / "artifacts/models/param_normalizer_v3.npz"
    yn_path  = Path(__file__).parents[1] / "artifacts/models/iv_normalizer_v3.npz"

    device   = torch.device("cpu")
    model    = MirrorPaddedFNO2d()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    pn = ParameterNormalizer.load(str(pn_path))
    yn = IVSurfaceNormalizer.load(str(yn_path))

    theta = np.array([1.0, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=np.float32)
    theta_t = torch.tensor(theta).unsqueeze(0).to(device)

    mats = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    strs = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    T, K = np.meshgrid(mats, strs, indexing="ij")
    spatial = torch.tensor(
        np.stack([T, K], axis=-1)[None], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        norm = pn.transform_tensor(theta_t)
        pred = model(spatial, norm)
        iv   = yn.inverse_transform_tensor(pred).squeeze(0).clamp(min=1e-4).numpy()

    return iv


def test_calibrate_single_with_surface(synthetic_surface):
    """calibrate_single should work when given a pre-computed surface."""
    result = calibrate_single(
        date_str="2024-01-02",
        currency="SPX",
        device="cpu",
        target_surface=synthetic_surface,
    )
    assert isinstance(result, CalibrationResult)
    assert result.date == "2024-01-02"
    assert np.isfinite(result.rmse_bps)
    assert result.runtime_ms > 0
    # Surface was FNO-generated → should calibrate to a low RMSE
    assert result.rmse_bps < 200.0, f"RMSE too high: {result.rmse_bps:.1f} bps"


def test_calibrate_single_params_in_bounds(synthetic_surface):
    from deepvol.calibration.joint_calibration import BOUNDS
    result = calibrate_single("2024-01-02", target_surface=synthetic_surface, device="cpu")
    for name, (lo, hi) in BOUNDS.items():
        val = result.params[name]
        assert lo <= val <= hi, f"{name}={val:.4f} outside [{lo}, {hi}]"


# ── calibrate_batch ───────────────────────────────────────────────────────────

def test_calibrate_batch_returns_sorted(synthetic_surface):
    dates = ["2024-08-05", "2024-01-02"]   # out of order
    surfaces = {d: synthetic_surface for d in dates}

    results = calibrate_batch(
        dates, currency="SPX", max_workers=2, device="cpu",
        target_surfaces=surfaces, verbose=False,
    )
    assert len(results) == 2
    assert results[0].date <= results[1].date   # sorted by date


def test_calibrate_batch_all_have_results(synthetic_surface):
    dates = ["2024-01-02", "2024-08-05"]
    surfaces = {d: synthetic_surface for d in dates}

    results = calibrate_batch(
        dates, currency="SPX", max_workers=2, device="cpu",
        target_surfaces=surfaces, verbose=False,
    )
    assert all(isinstance(r, CalibrationResult) for r in results)
    assert all(np.isfinite(r.rmse_bps) for r in results)


def test_calibrate_batch_single_date(synthetic_surface):
    results = calibrate_batch(
        ["2024-01-02"], device="cpu",
        target_surfaces={"2024-01-02": synthetic_surface},
        verbose=False,
    )
    assert len(results) == 1
    assert results[0].date == "2024-01-02"

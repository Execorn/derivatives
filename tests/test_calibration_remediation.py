import numpy as np
import pytest
import torch
from deepvol.calibration.interface import calibrate, CalibrationResult
from deepvol.calibration.calibrate_newton import calibrate_newton_h

def test_calibrate_fno_newton_dispatch():
    """Verify that calling calibrate with model_name='fno' and method='newton'
    correctly dispatches to calibrate_newton with T_grid and K_grid, without raising TypeError."""
    T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    target_iv = np.full((8, 11), 0.25, dtype=np.float32)
    
    res = calibrate(
        market_iv_surface=target_iv,
        model_name="fno",
        method="newton",
        T_grid=T_grid,
        K_grid=K_grid,
        max_iter=1
    )
    
    assert isinstance(res, CalibrationResult)
    assert res.parameters is not None

def test_calibrate_lbfgs_reparameterized_dispatch():
    """Verify that calling calibrate with method='l-bfgs' and reparameterized=True
    correctly dispatches without passing init_params, preventing TypeError."""
    T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    target_iv = np.full((8, 11), 0.25, dtype=np.float32)
    
    res = calibrate(
        market_iv_surface=target_iv,
        model_name="fno",
        method="l-bfgs",
        reparameterized=True,
        T_grid=T_grid,
        K_grid=K_grid,
        max_iter=1
    )
    
    assert isinstance(res, CalibrationResult)
    assert res.parameters is not None

def test_calibrate_lbfgs_default_guess_6d():
    """Verify that when reparameterized=False, the default guess is 6D, avoiding RuntimeError."""
    T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    target_iv = np.full((8, 11), 0.25, dtype=np.float32)
    
    res = calibrate(
        market_iv_surface=target_iv,
        model_name="fno",
        method="l-bfgs",
        reparameterized=False,
        T_grid=T_grid,
        K_grid=K_grid,
        max_iter=1
    )
    
    assert isinstance(res, CalibrationResult)
    assert len(res.parameters) == 6

def test_lbfgs_tuple_return_handling(monkeypatch):
    """Verify that if calibrate_parameters returns a 3-tuple, it is correctly handled."""
    import deepvol.calibration.calibrate_bfgs as calibrate_bfgs
    
    mock_params = np.array([1.0, 0.08, 0.3, -0.6, 0.04, 0.08], dtype=np.float32)
    mock_history = [0.1, 0.05, 0.02]
    mock_elapsed = 1.5
    
    def mock_calibrate_parameters(*args, **kwargs):
        return (mock_params, mock_history, mock_elapsed)
        
    monkeypatch.setattr(calibrate_bfgs, "calibrate_parameters", mock_calibrate_parameters)
    
    class DummyModel:
        def to(self, device): pass
        def eval(self): pass
        
    res = calibrate(
        market_iv_surface=np.full((8, 9), 0.25, dtype=np.float32),
        model_name="fno",
        method="l-bfgs",
        reparameterized=False,
        model=DummyModel(),
    )
    
    assert isinstance(res, CalibrationResult)
    assert np.allclose(res.parameters, mock_params)
    assert res.rmse == 0.02
    assert res.elapsed_time == mock_elapsed
    assert res.status == "converged"
    assert res.info == {"loss_history": mock_history}

def test_calibrate_newton_h_status(monkeypatch):
    """Verify that calibrate_newton_h includes 'status' in its return dictionary."""
    import deepvol.calibration.calibrate_newton as newton_module
    
    def mock_load_normalizers(*args, **kwargs):
        pass
        
    def mock_fno_predict_real_iv(*args, **kwargs):
        return torch.full((72,), 0.25, dtype=torch.float32)
        
    monkeypatch.setattr(newton_module, "_load_normalizers", mock_load_normalizers)
    monkeypatch.setattr(newton_module, "_fno_predict_real_iv", mock_fno_predict_real_iv)
    
    class DummyModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))
            
    T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
    K_grid = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], dtype=np.float32)
    target_iv = np.full((8, 9), 0.25, dtype=np.float32)
    
    res = calibrate_newton_h(
        model=DummyModel(),
        iv_target=target_iv,
        T_grid=T_grid,
        K_grid=K_grid,
        max_iter=2
    )
    
    assert "status" in res
    assert res["status"] in ("converged", "failed")

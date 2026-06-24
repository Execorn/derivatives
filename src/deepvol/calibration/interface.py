import time
import torch
import numpy as np
from typing import Dict, Any, Union, NamedTuple
from deepvol.utils.path_helpers import get_weights_path
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

class CalibrationResult(NamedTuple):
    parameters: Union[np.ndarray, torch.Tensor]  # Calibrated model parameters
    rmse: float                                  # Root Mean Squared Error of fit
    elapsed_time: float                          # Time taken for calibration in seconds
    status: str                                  # Status message (e.g., "converged", "failed")
    info: Dict[str, Any]                         # Additional info (e.g., iterations, gradients)

def _get_default_model(model_name: str, device: torch.device) -> MirrorPaddedFNO2d:
    """Instantiate and load the correct FNO model for the given model name."""
    model_name = model_name.lower()
    if model_name == "heston":
        param_dim = 5
        filename = "fno_heston_final_prod.pth"
    elif model_name == "sabr":
        param_dim = 3
        filename = "fno_sabr_final_prod.pth"
    elif model_name == "ssvi":
        param_dim = 11
        filename = "fno_ssvi_final_prod.pth"
    elif model_name in ("rbergomi", "rough_bergomi"):
        param_dim = 4
        filename = "fno_rbergomi_final_prod.pth"
    else:  # fno, rough_heston
        param_dim = 6
        filename = "fno_v2_final_prod.pth"
        
    model = MirrorPaddedFNO2d(param_dim=param_dim)
    weights_path = get_weights_path(filename)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def calibrate(
    market_iv_surface: Union[np.ndarray, torch.Tensor],
    model_name: str,
    method: str = "newton",
    device: str = "cpu",
    **kwargs
) -> CalibrationResult:
    """
    Calibrate a volatility model to a market implied volatility surface.
    
    Parameters:
        market_iv_surface (array-like): Grid of market implied volatilities.
        model_name (str): The name of the model ('sabr', 'heston', 'ssvi', 'rbergomi', 'fno', etc.).
        method (str): Optimization method ('newton', 'l-bfgs', etc.).
        device (str): PyTorch device context ('cpu' or 'cuda').
        **kwargs: Additional parameters passed to the underlying calibrator.
        
    Returns:
        CalibrationResult: A named tuple containing calibrated parameters and metadata.
    """
    t_start = time.time()
    model_name = model_name.lower()
    method = method.lower()
    
    device_obj = torch.device(device)
    
    # 1. Resolve/Load the FNO model
    model = kwargs.pop("model", None)
    if model is None:
        model = _get_default_model(model_name, device_obj)
    else:
        model.to(device_obj)
        model.eval()


    # 2. Format input surface to numpy array
    if isinstance(market_iv_surface, torch.Tensor):
        iv_np = market_iv_surface.detach().cpu().numpy()
    else:
        iv_np = np.asarray(market_iv_surface, dtype=np.float32)

    # 3. Default grids if needed
    T_grid = kwargs.pop("T_grid", np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32))
    K_grid = kwargs.pop("K_grid", np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], dtype=np.float32))
    
    # 4. Dispatch based on method and model_name
    if method == "newton":
        from deepvol.calibration.calibrate_newton import (
            calibrate_heston,
            calibrate_sabr,
            calibrate_ssvi,
            calibrate_rbergomi,
            calibrate_newton,
            calibrate_newton_h
        )
        
        if model_name == "heston":
            res = calibrate_heston(model, iv_np, T_grid, K_grid, **kwargs)
        elif model_name == "sabr":
            res = calibrate_sabr(model, iv_np, T_grid, K_grid, **kwargs)
        elif model_name == "ssvi":
            res = calibrate_ssvi(model, iv_np, T_grid, K_grid, **kwargs)
        elif model_name == "rbergomi":
            res = calibrate_rbergomi(model, iv_np, T_grid, K_grid, **kwargs)
        elif model_name in ("fno", "rough_heston"):
            # Check if learnable H or standard Newton
            use_h = kwargs.pop("learnable_h", False)
            if use_h:
                res = calibrate_newton_h(model, iv_np, T_grid, K_grid, **kwargs)
            else:
                res = calibrate_newton(model, iv_np, T_grid, K_grid, **kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name} for method {method}")
            
    elif method in ("l-bfgs", "bfgs", "lbfgs"):
        from deepvol.calibration.calibrate_bfgs import (
            calibrate_parameters,
            calibrate_reparameterized
        )
        
        use_reparam = kwargs.pop("reparameterized", False)
        init_params = kwargs.pop("init_params", None)
        if init_params is None:
            if use_reparam:
                init_params = np.array([0.04, -0.4, 0.4], dtype=np.float32) # length 3 for [v0, zeta, lambda]
            else:
                init_params = np.array([1.0, 0.08, 0.3, -0.6, 0.04, 0.08], dtype=np.float32)  # length 6 for [kappa, theta, sigma, rho, v0, H]
            
        if use_reparam:
            res = calibrate_reparameterized(model, iv_np, T_grid, K_grid, **kwargs)
        else:
            res = calibrate_parameters(model, iv_np, init_params, T_grid, K_grid, **kwargs)
            
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # 5. Extract values and construct CalibrationResult
    elapsed = time.time() - t_start
    
    if isinstance(res, tuple):
        # res is (final_params, history, elapsed) from calibrate_parameters
        params = res[0]
        rmse = float(res[1][-1]) if len(res[1]) > 0 else 0.0
        elapsed_val = res[2]
        status = "converged"
        info = {"loss_history": res[1]}
        return CalibrationResult(
            parameters=params,
            rmse=rmse,
            elapsed_time=elapsed_val,
            status=status,
            info=info
        )

    # Standardise return formats
    params = res.get("params") if res.get("params") is not None else res.get("parameters") if res.get("parameters") is not None else res.get("param_vector") if res.get("param_vector") is not None else res.get("theta_hat")
    if params is None and "v0" in res and "zeta" in res and "lambda" in res:
        params = np.array([res["v0"], res["zeta"], res["lambda"]], dtype=np.float32)
    rmse = float(res.get("rmse", 0.0))
    status = res.get("status", "converged")
    
    info = {k: v for k, v in res.items() if k not in ("params", "parameters", "rmse", "status")}
    
    return CalibrationResult(
        parameters=params,
        rmse=rmse,
        elapsed_time=elapsed,
        status=status,
        info=info
    )

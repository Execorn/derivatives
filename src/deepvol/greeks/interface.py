import torch
import numpy as np
from typing import Dict, Union, Any
from deepvol.utils.path_helpers import get_weights_path
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

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

def compute_greeks(
    model_name: str,
    parameters: Union[np.ndarray, torch.Tensor],
    spot: float,
    strikes: Union[np.ndarray, torch.Tensor],
    maturities: Union[np.ndarray, torch.Tensor],
    device: str = "cpu",
    **kwargs
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Calculate Greeks (Delta, Gamma, Vega, Theta, Vanna, Volga) for a given model.
    
    Parameters:
        model_name (str): Name of the option pricing model ('fno', 'schwartz_smith', 'bs', etc.).
        parameters (array-like): Model parameters.
        spot (float): Current price of the underlying asset.
        strikes (array-like): Option strike prices.
        maturities (array-like): Option maturities.
        device (str): PyTorch device context ('cpu' or 'cuda').
        
    Returns:
        Dict[str, array-like]: Dictionary of calculated Greeks.
    """
    model_name = model_name.lower()
    device_obj = torch.device(device)
    
    # Ensure parameter tensors
    if isinstance(parameters, np.ndarray):
        params_t = torch.tensor(parameters, dtype=torch.float32, device=device_obj)
    else:
        params_t = parameters.to(device_obj)

    if model_name in ("fno", "rough_heston"):
        from deepvol.greeks.portfolio_greeks import fno_surface_greeks
        from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
        
        model = kwargs.pop("model", None)
        if model is None:
            model = _get_default_model(model_name, device_obj)
        else:
            model.to(device_obj)
            model.eval()


        # Load normalizers v2
        from deepvol.calibration.calibrate_bfgs import _PARAM_NORM_PATH, _IV_NORM_PATH, _load_normalizers, _param_norm, _iv_norm
        _load_normalizers(version="v2")
        pn = _param_norm
        yn = _iv_norm
        
        interest_rate = kwargs.pop("r", 0.05)
        
        # Format grids
        T_grid = np.asarray(maturities, dtype=np.float32)
        K_grid = np.asarray(strikes, dtype=np.float32)
        
        greeks_dict = fno_surface_greeks(
            model=model,
            theta=params_t,
            pn=pn,
            yn=yn,
            S=spot,
            r=interest_rate,
            T_grid=T_grid,
            K_grid=K_grid
        )
        return greeks_dict
        
    elif model_name == "schwartz_smith":
        from deepvol.models.schwartz_smith import schwartz_smith_greeks_pt
        
        # schwartz_smith_greeks_pt(spot, strikes, maturities, params, target_greek="all", ...)
        # parameters should be a list/tensor of 5 parameters [kappa, mu_y, sigma_x, sigma_y, rho_xy]
        res = schwartz_smith_greeks_pt(
            S0=spot,
            K=torch.as_tensor(strikes, dtype=torch.float32, device=device_obj),
            T=torch.as_tensor(maturities, dtype=torch.float32, device=device_obj),
            params=params_t,
            target_greek=kwargs.pop("target_greek", "all"),
            **kwargs
        )
        return res
        
    elif model_name in ("bs", "black_scholes"):
        from deepvol.greeks.portfolio_greeks import bs_greeks
        
        # Compute BS greeks element-wise or list-wise
        # Expects: spot, strikes, maturities, r, vol, option_type
        r = kwargs.pop("r", 0.05)
        vol = kwargs.pop("vol", 0.2)
        option_type = kwargs.pop("option_type", "call")
        
        strikes_arr = np.atleast_1d(strikes)
        maturities_arr = np.atleast_1d(maturities)
        
        # We can assume a grid if they have different shapes or iterate
        res_dict = {
            "price": [], "delta": [], "gamma": [], "vega": [],
            "theta": [], "rho": [], "vanna": [], "volga": []
        }
        
        for t in maturities_arr:
            t_row = {k: [] for k in res_dict.keys()}
            for k in strikes_arr:
                g = bs_greeks(S=spot, K=float(k), T=float(t), r=r, sigma_iv=vol, option_type=option_type, **kwargs)
                for key in t_row.keys():
                    t_row[key].append(g.get(key, 0.0))
            for key in res_dict.keys():
                res_dict[key].append(t_row[key])
                
        # Convert to numpy arrays
        final_dict = {}
        for key, val in res_dict.items():
            final_dict[key] = np.array(val, dtype=np.float32)
            
        return final_dict
        
    else:
        raise ValueError(f"Unknown model_name: {model_name} for compute_greeks")

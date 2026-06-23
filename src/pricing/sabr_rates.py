"""
sabr_rates.py — Displaced SABR pricing and calibration for Interest Rate Swaptions.
"""

import numpy as np
import scipy.optimize as opt

def sabr_iv_normal_internal(F, K, T, alpha, beta, rho, nu):
    """
    Hagan (2002) approximate normal (Bachelier) implied volatility.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    nu = np.asarray(nu, dtype=float)
    
    # Broadcast to same shape
    F, K, T, alpha, beta, rho, nu = np.broadcast_arrays(F, K, T, alpha, beta, rho, nu)
    shape = F.shape
    
    # Normal SABR allows negative rates/strikes only if beta == 0.
    # If beta > 0, we require F > 0, K > 0.
    valid = np.where(beta > 0.0, (F > 0.0) & (K > 0.0), True) & (T > 0.0) & (alpha > 0.0)
    out = np.full(shape, np.nan)
    
    if not np.any(valid):
        if len(shape) == 0:
            return float(out)
        return out
        
    F_v = F[valid]
    K_v = K[valid]
    T_v = T[valid]
    alpha_v = alpha[valid]
    beta_v = beta[valid]
    rho_v = rho[valid]
    nu_v = nu[valid]
    
    res = np.zeros_like(F_v)
    atm = np.abs(F_v - K_v) < 1e-8
    not_atm = ~atm
    
    # ATM Case
    if np.any(atm):
        F_atm = F_v[atm]
        T_atm = T_v[atm]
        alpha_atm = alpha_v[atm]
        beta_atm = beta_v[atm]
        rho_atm = rho_v[atm]
        nu_atm = nu_v[atm]
        
        term1 = alpha_atm * (F_atm ** beta_atm)
        T1 = np.where(beta_atm > 0, -beta_atm * (2.0 - beta_atm) * (alpha_atm ** 2) / (24.0 * (F_atm ** (2.0 - 2.0 * beta_atm))), 0.0)
        T2 = np.where(beta_atm > 0, rho_atm * beta_atm * nu_atm * alpha_atm / (4.0 * (F_atm ** (1.0 - beta_atm))), 0.0)
        T3 = ((2.0 - 3.0 * rho_atm ** 2) / 24.0) * (nu_atm ** 2)
        
        res[atm] = term1 * (1.0 + (T1 + T2 + T3) * T_atm)
        
    # Non-ATM Case
    if np.any(not_atm):
        F_natm = F_v[not_atm]
        K_natm = K_v[not_atm]
        T_natm = T_v[not_atm]
        alpha_natm = alpha_v[not_atm]
        beta_natm = beta_v[not_atm]
        rho_natm = rho_v[not_atm]
        nu_natm = nu_v[not_atm]
        
        is_beta_zero = beta_natm == 0.0
        is_beta_pos = ~is_beta_zero
        
        I_0 = np.zeros_like(F_natm)
        if np.any(is_beta_zero):
            I_0[is_beta_zero] = F_natm[is_beta_zero] - K_natm[is_beta_zero]
        if np.any(is_beta_pos):
            F_p = F_natm[is_beta_pos]
            K_p = K_natm[is_beta_pos]
            beta_p = beta_natm[is_beta_pos]
            I_0_pos = np.where(beta_p == 1.0, np.log(F_p / K_p), (F_p ** (1.0 - beta_p) - K_p ** (1.0 - beta_p)) / (1.0 - beta_p))
            I_0[is_beta_pos] = I_0_pos
            
        zeta = (nu_natm / alpha_natm) * I_0
        
        val = (np.sqrt(1.0 - 2.0 * rho_natm * zeta + zeta ** 2) + zeta - rho_natm) / (1.0 - rho_natm)
        val = np.maximum(val, 1e-15)
        D_zeta = np.log(val)
        D_zeta_safe = np.where(np.abs(zeta) < 1e-8, 1.0, D_zeta)
        zeta_over_D = np.where(np.abs(zeta) < 1e-8, 1.0, zeta / D_zeta_safe)
        
        term1 = alpha_natm * (F_natm - K_natm) / I_0
        fk = F_natm * K_natm
        T1 = np.zeros_like(F_natm)
        T2 = np.zeros_like(F_natm)
        if np.any(is_beta_pos):
            fk_p = fk[is_beta_pos]
            beta_p = beta_natm[is_beta_pos]
            alpha_p = alpha_natm[is_beta_pos]
            rho_p = rho_natm[is_beta_pos]
            nu_p = nu_natm[is_beta_pos]
            T1[is_beta_pos] = -beta_p * (2.0 - beta_p) * (alpha_p ** 2) / (24.0 * (fk_p ** (1.0 - beta_p)))
            T2[is_beta_pos] = rho_p * beta_p * nu_p * alpha_p / (4.0 * (fk_p ** ((1.0 - beta_p) / 2.0)))
            
        T3 = ((2.0 - 3.0 * rho_natm ** 2) / 24.0) * (nu_natm ** 2)
        res[not_atm] = term1 * zeta_over_D * (1.0 + (T1 + T2 + T3) * T_natm)
        
    out[valid] = res
    if len(shape) == 0:
        return float(out)
    return out

def sabr_iv_lognormal_internal(F, K, T, alpha, beta, rho, nu):
    """
    Hagan (2002) approximate lognormal (Black) implied volatility.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    nu = np.asarray(nu, dtype=float)
    
    # Broadcast to same shape
    F, K, T, alpha, beta, rho, nu = np.broadcast_arrays(F, K, T, alpha, beta, rho, nu)
    shape = F.shape
    
    valid = (F > 0.0) & (K > 0.0) & (T > 0.0) & (alpha > 0.0)
    out = np.full(shape, np.nan)
    
    if not np.any(valid):
        if len(shape) == 0:
            return float(out)
        return out
        
    F_v = F[valid]
    K_v = K[valid]
    T_v = T[valid]
    alpha_v = alpha[valid]
    beta_v = beta[valid]
    rho_v = rho[valid]
    nu_v = nu[valid]
    
    res = np.zeros_like(F_v)
    atm = np.abs(F_v - K_v) < 1e-8
    not_atm = ~atm
    
    # ATM Case
    if np.any(atm):
        F_atm = F_v[atm]
        T_atm = T_v[atm]
        alpha_atm = alpha_v[atm]
        beta_atm = beta_v[atm]
        rho_atm = rho_v[atm]
        nu_atm = nu_v[atm]
        
        one_minus_beta = 1.0 - beta_atm
        term1 = alpha_atm / (F_atm ** one_minus_beta)
        num_c1 = ((one_minus_beta ** 2) / 24.0) * (alpha_atm ** 2) / (F_atm ** (2.0 * one_minus_beta))
        num_c2 = 0.25 * rho_atm * beta_atm * nu_atm * alpha_atm / (F_atm ** one_minus_beta)
        num_c3 = ((2.0 - 3.0 * rho_atm ** 2) / 24.0) * (nu_atm ** 2)
        res[atm] = term1 * (1.0 + (num_c1 + num_c2 + num_c3) * T_atm)
        
    # Non-ATM Case
    if np.any(not_atm):
        F_natm = F_v[not_atm]
        K_natm = K_v[not_atm]
        T_natm = T_v[not_atm]
        alpha_natm = alpha_v[not_atm]
        beta_natm = beta_v[not_atm]
        rho_natm = rho_v[not_atm]
        nu_natm = nu_v[not_atm]
        
        one_minus_beta = 1.0 - beta_natm
        fk = F_natm * K_natm
        log_fk = np.log(F_natm / K_natm)
        fk_pow = fk ** (one_minus_beta / 2.0)
        
        denom = 1.0 + ((one_minus_beta ** 2) / 24.0) * (log_fk ** 2) + ((one_minus_beta ** 4) / 1920.0) * (log_fk ** 4)
        term1 = alpha_natm / (fk_pow * denom)
        
        z = (nu_natm / alpha_natm) * fk_pow * log_fk
        
        val = (np.sqrt(1.0 - 2.0 * rho_natm * z + z ** 2) + z - rho_natm) / (1.0 - rho_natm)
        val = np.maximum(val, 1e-15)
        xz = np.log(val)
        xz_safe = np.where(np.abs(z) < 1e-8, 1.0, xz)
        z_over_xz = np.where(np.abs(z) < 1e-8, 1.0, z / xz_safe)
        
        num_c1 = ((one_minus_beta ** 2) / 24.0) * (alpha_natm ** 2) / (fk ** one_minus_beta)
        num_c2 = 0.25 * rho_natm * beta_natm * nu_natm * alpha_natm / fk_pow
        num_c3 = ((2.0 - 3.0 * rho_natm ** 2) / 24.0) * (nu_natm ** 2)
        res[not_atm] = term1 * z_over_xz * (1.0 + (num_c1 + num_c2 + num_c3) * T_natm)
        
    out[valid] = res
    if len(shape) == 0:
        return float(out)
    return out

def displaced_sabr_vol(F, K, T, alpha, beta, rho, nu, shift, vol_type='normal'):
    """
    Compute displaced/shifted SABR implied volatility.
    
    Parameters
    ----------
    F : float or ndarray
        Forward price/rate.
    K : float or ndarray
        Strike price/rate.
    T : float or ndarray
        Time to maturity.
    alpha : float or ndarray
        SABR initial volatility.
    beta : float or ndarray
        SABR CEV exponent.
    rho : float or ndarray
        SABR correlation.
    nu : float or ndarray
        SABR vol-of-vol.
    shift : float or ndarray
        Displacement/shift parameter.
    vol_type : str
        'normal' or 'lognormal'.
        
    Returns
    -------
    iv : float or ndarray
        Displaced SABR implied volatility.
    """
    # Shift forward and strike
    F_s = np.asarray(F, dtype=float) + shift
    K_s = np.asarray(K, dtype=float) + shift
    
    if vol_type.lower() == 'normal':
        return sabr_iv_normal_internal(F_s, K_s, T, alpha, beta, rho, nu)
    elif vol_type.lower() == 'lognormal':
        return sabr_iv_lognormal_internal(F_s, K_s, T, alpha, beta, rho, nu)
    else:
        raise ValueError(f"Unknown vol_type: {vol_type}")

def solve_alpha_from_atm(F, T, beta, rho, nu, shift, atm_vol, vol_type):
    """
    Solve for alpha from ATM volatility by solving a cubic equation for A.
    
    Parameters
    ----------
    F : float
        Forward rate.
    T : float
        Time to maturity.
    beta : float
        SABR beta.
    rho : float
        SABR rho.
    nu : float
        SABR nu.
    shift : float
        Shift parameter.
    atm_vol : float
        ATM volatility.
    vol_type : str
        'normal' or 'lognormal'.
        
    Returns
    -------
    alpha : float
        The solved alpha parameter.
    """
    F_s = F + shift
    if F_s <= 0.0:
        return 1e-4  # Safe fallback if shifted forward is non-positive
        
    # Coefficients of cubic equation C3*A^3 + C2*A^2 + C1*A + C0 = 0
    if vol_type.lower() == 'normal':
        C3 = -beta * (2.0 - beta) * T / (24.0 * F_s**2)
        C2 = rho * beta * nu * T / (4.0 * F_s)
        C1 = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2 * T
        C0 = -atm_vol
    elif vol_type.lower() == 'lognormal':
        C3 = (1.0 - beta)**2 * T / 24.0
        C2 = rho * beta * nu * T / 4.0
        C1 = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2 * T
        C0 = -atm_vol
    else:
        raise ValueError(f"Unknown vol_type: {vol_type}")
        
    # Solve cubic equation for A using Brent's method over [1e-10, 10.0]
    def cubic_eq(A):
        return C3 * A**3 + C2 * A**2 + C1 * A + C0
        
    try:
        # Dynamically find an upper bound that brackets the root
        high = max(2.0 * atm_vol, 1e-5)
        while cubic_eq(high) < 0.0 and high < 10.0:
            high *= 2.0
            
        if cubic_eq(high) > 0.0:
            A_sol = opt.brentq(cubic_eq, 1e-10, high, xtol=1e-8)
        else:
            A_sol = -C0 / C1
            A_sol = np.clip(A_sol, 1e-10, 10.0)
    except Exception:
        A_sol = -C0 / C1
        A_sol = np.clip(A_sol, 1e-10, 10.0)
        
    # Translate A back to alpha
    if vol_type.lower() == 'normal':
        alpha = A_sol / (F_s**beta)
    else:
        alpha = A_sol * (F_s**(1.0 - beta))
        
    return float(alpha)

def calibrate_sabr_node(F, strikes, market_vols, T, beta, shift, vol_type='normal'):
    """
    Calibrate SABR parameters (alpha, rho, nu) to fit market volatilities at a single grid node.
    Uses 2D calibration (reducing dimensionality via ATM volatility relation)
    with a fallback to full 3D calibration.
    
    Parameters
    ----------
    F : float
        Forward price/rate.
    strikes : ndarray
        1D array of strikes.
    market_vols : ndarray
        1D array of market volatilities.
    T : float
        Time to maturity.
    beta : float
        SABR CEV exponent.
    shift : float
        SABR displacement/shift.
    vol_type : str
        'normal' or 'lognormal'.
        
    Returns
    -------
    alpha, rho, nu : tuple
        Calibrated SABR parameters.
    """
    F = float(F)
    T = float(T)
    beta = float(beta)
    shift = float(shift)
    
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    
    # Good default initialization
    atm_idx = np.argmin(np.abs(strikes - F))
    atm_vol = market_vols[atm_idx]
    
    F_s = F + shift
    if vol_type.lower() == 'normal':
        # ATM normal vol approx: alpha * F_s**beta
        alpha_init = atm_vol / (F_s ** beta)
    else:
        # ATM lognormal vol approx: alpha / F_s**(1-beta)
        alpha_init = atm_vol * (F_s ** (1.0 - beta))
        
    alpha_init = np.clip(alpha_init, 1e-4, 5.0)
    rho_init = 0.0
    nu_init = 0.3
    
    # Try 2D optimization: optimization variables are [rho, nu]
    bounds_2d = ([-0.999, 1e-5], [0.999, 5.0])
    x0_2d = [rho_init, nu_init]
    
    def residuals_2d(params_2d):
        rho_val, nu_val = params_2d
        alpha_val = solve_alpha_from_atm(F, T, beta, rho_val, nu_val, shift, atm_vol, vol_type)
        model_vols = displaced_sabr_vol(F, strikes, T, alpha_val, beta, rho_val, nu_val, shift, vol_type=vol_type)
        res = model_vols - market_vols
        res = np.where(np.isnan(res), 1e6, res)
        return res
        
    success_2d = False
    alpha_cal, rho_cal, nu_cal = alpha_init, rho_init, nu_init
    
    try:
        res_2d = opt.least_squares(residuals_2d, x0_2d, bounds=bounds_2d, method='trf')
        if res_2d.success:
            rho_2d, nu_2d = res_2d.x
            alpha_2d = solve_alpha_from_atm(F, T, beta, rho_2d, nu_2d, shift, atm_vol, vol_type)
            fit_vols = displaced_sabr_vol(F, strikes, T, alpha_2d, beta, rho_2d, nu_2d, shift, vol_type=vol_type)
            mse = np.mean((fit_vols - market_vols)**2)
            if mse < 1e-4 and alpha_2d > 1e-5 and alpha_2d < 10.0:
                alpha_cal, rho_cal, nu_cal = alpha_2d, rho_2d, nu_2d
                success_2d = True
    except Exception:
        success_2d = False
        
    if not success_2d:
        # Fallback to 3D calibration
        x0 = [alpha_init, rho_init, nu_init]
        bounds = ([1e-5, -0.999, 1e-5], [10.0, 0.999, 5.0])
        
        def residuals(params):
            alpha_val, rho_val, nu_val = params
            model_vols = displaced_sabr_vol(F, strikes, T, alpha_val, beta, rho_val, nu_val, shift, vol_type=vol_type)
            res = model_vols - market_vols
            res = np.where(np.isnan(res), 1e6, res)
            return res
            
        res = opt.least_squares(residuals, x0, bounds=bounds, method='trf')
        alpha_cal, rho_cal, nu_cal = res.x
        
    return alpha_cal, rho_cal, nu_cal

def calibrate_node_worker(task):
    """
    Top-level worker function for parallel SABR node calibration.
    """
    i, j, F, strikes, market_vols_node, T, b_node, sh_node, vol_type = task
    a, r, n = calibrate_sabr_node(F, strikes, market_vols_node, T, b_node, sh_node, vol_type=vol_type)
    return i, j, a, r, n

def bilinear_interpolate(x, y, x_grid, y_grid, z_values):
    """
    Vectorized bilinear interpolation on a 2D grid with flat extrapolation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast_shapes(x.shape, y.shape)
    x = np.broadcast_to(x, shape)
    y = np.broadcast_to(y, shape)
    
    x_clip = np.clip(x, x_grid[0], x_grid[-1])
    y_clip = np.clip(y, y_grid[0], y_grid[-1])
    
    i1 = np.searchsorted(x_grid, x_clip) - 1
    i1 = np.clip(i1, 0, len(x_grid) - 2)
    i2 = i1 + 1
    
    j1 = np.searchsorted(y_grid, y_clip) - 1
    j1 = np.clip(j1, 0, len(y_grid) - 2)
    j2 = j1 + 1
    
    x1 = x_grid[i1]
    x2 = x_grid[i2]
    y1 = y_grid[j1]
    y2 = y_grid[j2]
    
    z11 = z_values[i1, j1]
    z12 = z_values[i1, j2]
    z21 = z_values[i2, j1]
    z22 = z_values[i2, j2]
    
    denom = (x2 - x1) * (y2 - y1)
    w11 = (x2 - x_clip) * (y2 - y_clip) / denom
    w21 = (x_clip - x1) * (y2 - y_clip) / denom
    w12 = (x2 - x_clip) * (y_clip - y1) / denom
    w22 = (x_clip - x1) * (y_clip - y1) / denom
    
    return w11 * z11 + w21 * z21 + w12 * z12 + w22 * z22

class SwaptionVolCube:
    """
    Volatility cube storing market swaption coordinates, forwards, and calibrated SABR parameters.
    Supports parameter interpolation, smile extraction, and option pricing.
    """
    def __init__(self, expiries, tenors, relative_strikes):
        """
        Parameters
        ----------
        expiries : ndarray
            1D array of option expiries.
        tenors : ndarray
            1D array of swap tenors.
        relative_strikes : ndarray
            1D array of relative strikes/moneyness in bps.
        """
        self.expiries = np.sort(np.asarray(expiries, dtype=float))
        self.tenors = np.sort(np.asarray(tenors, dtype=float))
        self.relative_strikes = np.asarray(relative_strikes, dtype=float)
        
        # Grid parameters to be populated during calibration
        self.alpha = None
        self.beta = None
        self.rho = None
        self.nu = None
        self.shift = None
        self.forward_rates = None
        self.vol_type = None
        
    def calibrate(self, market_cube_vols, forward_rates, beta=0.5, shift=0.01, vol_type='normal', parallel=False):
        """
        Perform node-by-node calibration of SABR parameters to market volatilities.
        
        Parameters
        ----------
        market_cube_vols : ndarray
            3D array of shape (len(expiries), len(tenors), len(relative_strikes)).
        forward_rates : ndarray
            2D array of shape (len(expiries), len(tenors)).
        beta : float or ndarray
            CEV exponent (constant or grid).
        shift : float or ndarray
            Shift/displacement parameter (constant or grid).
        vol_type : str
            'normal' or 'lognormal'.
        parallel : bool
            Whether to run calibration in parallel across multiple CPU cores.
        """
        self.forward_rates = np.asarray(forward_rates, dtype=float)
        self.vol_type = vol_type
        
        num_exp = len(self.expiries)
        num_ten = len(self.tenors)
        
        # Format beta and shift into grid arrays if they are scalar
        if np.isscalar(beta):
            self.beta = np.full((num_exp, num_ten), beta)
        else:
            self.beta = np.asarray(beta, dtype=float)
            
        if np.isscalar(shift):
            self.shift = np.full((num_exp, num_ten), shift)
        else:
            self.shift = np.asarray(shift, dtype=float)
            
        self.alpha = np.zeros((num_exp, num_ten))
        self.rho = np.zeros((num_exp, num_ten))
        self.nu = np.zeros((num_exp, num_ten))
        
        tasks = []
        for i, T in enumerate(self.expiries):
            for j, tenor in enumerate(self.tenors):
                F = self.forward_rates[i, j]
                # Strikes are F + relative_strikes * 1e-4
                strikes = F + self.relative_strikes * 1e-4
                market_vols_node = market_cube_vols[i, j, :]
                
                b_node = self.beta[i, j]
                sh_node = self.shift[i, j]
                tasks.append((i, j, F, strikes, market_vols_node, T, b_node, sh_node, vol_type))
                
        if parallel:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(calibrate_node_worker, tasks))
            for i, j, a, r, n in results:
                self.alpha[i, j] = a
                self.rho[i, j] = r
                self.nu[i, j] = n
        else:
            for task in tasks:
                i, j, a, r, n = calibrate_node_worker(task)
                self.alpha[i, j] = a
                self.rho[i, j] = r
                self.nu[i, j] = n
                
    def interpolate_params(self, T_exp, T_tenor):
        """
        Bilinear interpolation of calibrated SABR parameters at arbitrary expiry and tenor.
        
        Parameters
        ----------
        T_exp : float or ndarray
            Target option expiry.
        T_tenor : float or ndarray
            Target swap tenor.
            
        Returns
        -------
        alpha, beta, rho, nu, shift : tuple
            Interpolated SABR parameters.
        """
        if self.alpha is None:
            raise ValueError("Vol cube must be calibrated before parameters can be interpolated.")
            
        alpha = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.alpha)
        beta = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.beta)
        rho = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.rho)
        nu = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.nu)
        shift = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.shift)
        
        # Clip parameters to valid ranges to prevent numeric instability
        rho = np.clip(rho, -0.999, 0.999)
        alpha = np.maximum(alpha, 1e-5)
        nu = np.maximum(nu, 1e-5)
        
        return alpha, beta, rho, nu, shift
        
    def get_smile(self, T_exp, T_tenor, strikes, vol_type='normal'):
        """
        Get interpolated SABR implied volatilities for a specific smile.
        
        Parameters
        ----------
        T_exp : float
            Option expiry.
        T_tenor : float
            Swap tenor.
        strikes : ndarray
            Requested strikes (absolute rates).
        vol_type : str
            'normal' or 'lognormal'.
            
        Returns
        -------
        vols : ndarray
            Interpolated SABR volatilities.
        """
        # Interpolate forward rate for the node
        F = bilinear_interpolate(T_exp, T_tenor, self.expiries, self.tenors, self.forward_rates)
        
        # Interpolate SABR parameters
        alpha, beta, rho, nu, shift = self.interpolate_params(T_exp, T_tenor)
        
        # Compute implied vols
        return displaced_sabr_vol(F, strikes, T_exp, alpha, beta, rho, nu, shift, vol_type=vol_type)
        
    def price_swaption(self, T_exp, T_tenor, strike, forward, option_type='call', vol_type='normal'):
        """
        Price a swaption using the interpolated volatility.
        
        Parameters
        ----------
        T_exp : float
            Option expiry.
        T_tenor : float
            Swap tenor.
        strike : float
            Swaption strike rate.
        forward : float
            Forward swap rate.
        option_type : str
            'call' or 'put' (case-insensitive).
        vol_type : str
            'normal' or 'lognormal'.
            
        Returns
        -------
        price : float
            Option price.
        """
        # Interpolate parameters
        alpha, beta, rho, nu, shift = self.interpolate_params(T_exp, T_tenor)
        
        # Get SABR vol
        iv = displaced_sabr_vol(forward, strike, T_exp, alpha, beta, rho, nu, shift, vol_type=vol_type)
        
        # Price option
        if vol_type.lower() == 'normal':
            from .bachelier import bachelier_price
            return bachelier_price(forward, strike, T_exp, iv, option_type=option_type)
        elif vol_type.lower() == 'lognormal':
            from .bachelier import shifted_black_price
            return shifted_black_price(forward, strike, T_exp, iv, shift, option_type=option_type)
        else:
            raise ValueError(f"Unknown vol_type: {vol_type}")

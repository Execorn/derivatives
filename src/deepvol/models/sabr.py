"""
sabr.py — Analytical formulas and surfaces for SABR and SSVI models.

Contains:
- sabr_iv_normal: Hagan approximate normal IV (general beta, with special beta=0 case).
- sabr_iv_lognormal: Hagan approximate lognormal IV.
- ssvi_total_variance: SSVI power-law total variance computation.
- sabr_iv_surface: Implied volatility surface for SABR.
- ssvi_iv_surface: Implied volatility surface for SSVI.
"""

import numpy as np


def sabr_iv_lognormal(F, K, T, alpha, beta, rho, nu):
    """
    Hagan (2002) approximate lognormal (Black) implied volatility.
    
    Parameters
    ----------
    F : float or ndarray
        Forward price.
    K : float or ndarray
        Strike price.
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
        
    Returns
    -------
    iv : float or ndarray
        Lognormal implied volatility.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    nu = np.asarray(nu, dtype=float)
    
    # Broadcast to same shape to support multi-dimensional and scalar inputs consistently
    F, K, T, alpha, beta, rho, nu = np.broadcast_arrays(F, K, T, alpha, beta, rho, nu)
    shape = F.shape
    
    # SABR parameter validation
    if np.any(alpha <= 0): raise ValueError("alpha must be > 0")
    if np.any((beta < 0) | (beta > 1)): raise ValueError("beta must be in [0, 1]")
    if np.any(nu <= 0): raise ValueError("nu must be > 0")
    if np.any(np.abs(rho) > 1.0): raise ValueError("|rho| must be <= 1")
    
    # Lognormal SABR requires F > 0, K > 0, and T > 0
    valid = (F > 0.0) & (K > 0.0) & (T > 0.0)
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
    
    # ATM check
    atm = np.abs(F_v - K_v) < 1e-8 * np.maximum(np.abs(F_v), 1e-10)
    not_atm = ~atm
    
    # ── ATM Case ──
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
        
    # ── Non-ATM Case ──
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
        
        # Denominator of the first term:
        # 1 + (1-beta)^2 / 24 * log(f/k)^2 + (1-beta)^4 / 1920 * log(f/k)^4
        denom = 1.0 + ((one_minus_beta ** 2) / 24.0) * (log_fk ** 2) + ((one_minus_beta ** 4) / 1920.0) * (log_fk ** 4)
        term1 = alpha_natm / (fk_pow * denom)
        
        # z
        z = (nu_natm / alpha_natm) * fk_pow * log_fk
        
        # z / x(z)
        # Avoid 0/0 floating-point cancellation in standard SABR formula by using the mathematically equivalent conjugate formula when rho is near 1
        denom_rho = 1.0 - rho_natm
        standard_val = (np.sqrt(1.0 - 2.0 * rho_natm * z + z ** 2) + z - rho_natm) / np.where(np.abs(denom_rho) < 1e-12, 1e-12, denom_rho)
        denom_conj = np.sqrt(1.0 - 2.0 * rho_natm * z + z ** 2) - z + rho_natm
        safe_denom_conj = np.where(np.abs(denom_conj) < 1e-12, 1e-12, denom_conj)
        conjugate_val = (1.0 + rho_natm) / safe_denom_conj
        val = np.where(rho_natm > 0.9, conjugate_val, standard_val)
        val = np.maximum(val, 1e-15)
        xz = np.log(val)
        xz_safe = np.where(np.abs(z) < 1e-8, 1.0, xz)
        z_over_xz = np.where(np.abs(z) < 1e-8, 1.0, z / xz_safe)
        
        # correction term
        num_c1 = ((one_minus_beta ** 2) / 24.0) * (alpha_natm ** 2) / (fk ** one_minus_beta)
        num_c2 = 0.25 * rho_natm * beta_natm * nu_natm * alpha_natm / fk_pow
        num_c3 = ((2.0 - 3.0 * rho_natm ** 2) / 24.0) * (nu_natm ** 2)
        
        res[not_atm] = term1 * z_over_xz * (1.0 + (num_c1 + num_c2 + num_c3) * T_natm)
        
    out[valid] = res
    
    if len(shape) == 0:
        return float(out)
    return out


def sabr_iv_normal(F, K, T, alpha, beta, rho, nu):
    """
    Hagan (2002) approximate normal (Bachelier) implied volatility.
    
    Parameters
    ----------
    F : float or ndarray
        Forward price.
    K : float or ndarray
        Strike price.
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
        
    Returns
    -------
    iv : float or ndarray
        Normal implied volatility.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    nu = np.asarray(nu, dtype=float)
    
    # Broadcast to same shape to support multi-dimensional and scalar inputs consistently
    F, K, T, alpha, beta, rho, nu = np.broadcast_arrays(F, K, T, alpha, beta, rho, nu)
    shape = F.shape
    
    # SABR parameter validation
    if np.any(alpha <= 0): raise ValueError("alpha must be > 0")
    if np.any((beta < 0) | (beta > 1)): raise ValueError("beta must be in [0, 1]")
    if np.any(nu <= 0): raise ValueError("nu must be > 0")
    if np.any(np.abs(rho) > 1.0): raise ValueError("|rho| must be <= 1")
    
    # Normal SABR allows negative rates/strikes only if beta == 0.
    # If beta > 0, we require F > 0, K > 0.
    valid = np.where(beta > 0.0, (F > 0.0) & (K > 0.0), True) & (T > 0.0)
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
    
    atm = np.abs(F_v - K_v) < 1e-8 * np.maximum(np.abs(F_v), 1e-10)
    not_atm = ~atm
    
    # ── ATM Case ──
    if np.any(atm):
        F_atm = F_v[atm]
        T_atm = T_v[atm]
        alpha_atm = alpha_v[atm]
        beta_atm = beta_v[atm]
        rho_atm = rho_v[atm]
        nu_atm = nu_v[atm]
        
        term1 = alpha_atm * (F_atm ** beta_atm)
        
        # If beta == 0, we must avoid taking negative numbers to negative powers,
        # but the terms with negative bases are scaled by beta, so they vanish.
        T1 = np.where(beta_atm > 0, -beta_atm * (2.0 - beta_atm) * (alpha_atm ** 2) / (24.0 * (F_atm ** (2.0 - 2.0 * beta_atm))), 0.0)
        T2 = np.where(beta_atm > 0, rho_atm * beta_atm * nu_atm * alpha_atm / (4.0 * (F_atm ** (1.0 - beta_atm))), 0.0)
        T3 = ((2.0 - 3.0 * rho_atm ** 2) / 24.0) * (nu_atm ** 2)
        
        res[atm] = term1 * (1.0 + (T1 + T2 + T3) * T_atm)
        
    # ── Non-ATM Case ──
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
            
        # zeta
        zeta = (nu_natm / alpha_natm) * I_0
        
        # D(zeta)
        # Avoid 0/0 floating-point cancellation in standard SABR formula by using the mathematically equivalent conjugate formula when rho is near 1
        denom_rho = 1.0 - rho_natm
        standard_val = (np.sqrt(1.0 - 2.0 * rho_natm * zeta + zeta ** 2) + zeta - rho_natm) / np.where(np.abs(denom_rho) < 1e-12, 1e-12, denom_rho)
        denom_conj = np.sqrt(1.0 - 2.0 * rho_natm * zeta + zeta ** 2) - zeta + rho_natm
        safe_denom_conj = np.where(np.abs(denom_conj) < 1e-12, 1e-12, denom_conj)
        conjugate_val = (1.0 + rho_natm) / safe_denom_conj
        val = np.where(rho_natm > 0.9, conjugate_val, standard_val)
        val = np.maximum(val, 1e-15)
        D_zeta = np.log(val)
        D_zeta_safe = np.where(np.abs(zeta) < 1e-8, 1.0, D_zeta)
        zeta_over_D = np.where(np.abs(zeta) < 1e-8, 1.0, zeta / D_zeta_safe)
        
        # first term
        term1 = alpha_natm * (F_natm - K_natm) / I_0
        
        # correction terms
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


def ssvi_total_variance(k, theta, rho, eta, gamma):
    """
    SSVI power-law total variance surface computation.
    
    w(k, theta) = theta/2 * [1 + rho * phi(theta) * k + sqrt((phi(theta) * k + rho)^2 + 1 - rho^2)]
    where phi(theta) = eta / (theta^gamma * (1 + theta)^(1-gamma))
    
    Parameters
    ----------
    k : float or ndarray
        Log-forward moneyness.
    theta : float or ndarray
        ATM total variance.
    rho : float or ndarray
        Correlation parameter.
    eta : float or ndarray
        SSVI vol-of-vol scaling.
    gamma : float or ndarray
        SSVI power-law exponent.
    """
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    eta = np.asarray(eta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    
    # Broadcast to same shape to support multi-dimensional and scalar inputs consistently
    k, theta, rho, eta, gamma = np.broadcast_arrays(k, theta, rho, eta, gamma)
    shape = k.shape
    
    # Ensure theta >= 0
    valid = theta >= 0.0
    out = np.full(shape, np.nan)
    
    if not np.any(valid):
        if len(shape) == 0:
            return float(out)
        return out
        
    k_v = k[valid]
    theta_v = theta[valid]
    rho_v = rho[valid]
    eta_v = eta[valid]
    gamma_v = gamma[valid]
    
    res = np.zeros_like(theta_v)
    
    # ATM total variance == 0 implies total variance is 0
    nonzero = theta_v > 0.0
    
    if np.any(nonzero):
        k_nz = k_v[nonzero]
        theta_nz = theta_v[nonzero]
        rho_nz = rho_v[nonzero]
        eta_nz = eta_v[nonzero]
        gamma_nz = gamma_v[nonzero]
        
        # phi(theta)
        phi = eta_nz / (theta_nz ** gamma_nz * (1.0 + theta_nz) ** (1.0 - gamma_nz))
        phi = np.clip(phi, 0.0, 1e6)
        
        inside = (phi * k_nz + rho_nz) ** 2 + 1.0 - rho_nz ** 2
        inside = np.maximum(inside, 0.0)
        
        res[nonzero] = 0.5 * theta_nz * (1.0 + rho_nz * phi * k_nz + np.sqrt(inside))
        
    out[valid] = res
    
    if len(shape) == 0:
        return float(out)
    return out


def sabr_iv_surface(F, T_grid, k_grid, alpha, beta, rho, nu, iv_type="lognormal"):
    """
    Computes SABR implied volatility surface on a grid, supporting both scalar and batch inputs.
    """
    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid, indexing='ij')
    K_mesh = F * np.exp(k_mesh)
    
    alpha = np.asarray(alpha)
    is_batched = alpha.ndim > 0
    
    if is_batched:
        K_mesh = K_mesh[np.newaxis, :, :]
        T_mesh = T_mesh[np.newaxis, :, :]
        alpha = alpha[:, np.newaxis, np.newaxis]
        beta = np.asarray(beta)[:, np.newaxis, np.newaxis]
        rho = np.asarray(rho)[:, np.newaxis, np.newaxis]
        nu = np.asarray(nu)[:, np.newaxis, np.newaxis]
        
    if iv_type == "lognormal":
        res = sabr_iv_lognormal(F, K_mesh, T_mesh, alpha, beta, rho, nu)
    elif iv_type == "normal":
        res = sabr_iv_normal(F, K_mesh, T_mesh, alpha, beta, rho, nu)
    else:
        raise ValueError(f"Unknown iv_type: {iv_type}")
        
    return res


def ssvi_iv_surface(T_grid, k_grid, theta_grid, rho, eta, gamma):
    """
    Computes SSVI implied volatility surface on a grid, supporting both scalar and batch inputs.
    """
    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid, indexing='ij')
    
    theta_grid = np.asarray(theta_grid)
    rho = np.asarray(rho)
    is_batched = rho.ndim > 0
    
    if is_batched:
        T_mesh = T_mesh[np.newaxis, :, :]
        k_mesh = k_mesh[np.newaxis, :, :]
        theta_mesh = theta_grid[:, :, np.newaxis]
        rho = rho[:, np.newaxis, np.newaxis]
        eta = np.asarray(eta)[:, np.newaxis, np.newaxis]
        gamma = np.asarray(gamma)[:, np.newaxis, np.newaxis]
    else:
        theta_mesh, _ = np.meshgrid(theta_grid, k_grid, indexing='ij')
        
    w = ssvi_total_variance(k_mesh, theta_mesh, rho, eta, gamma)
    
    w_safe = np.maximum(w, 0.0)
    T_safe = np.maximum(T_mesh, 1e-15)
    iv = np.sqrt(w_safe / T_safe)
    return iv

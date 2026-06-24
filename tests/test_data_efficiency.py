import os
import sys
import pytest
import numpy as np
import torch
import torch.nn.functional as F
import gpytorch
import chaospy as cp
from sklearn.linear_model import Ridge, RidgeCV, MultiTaskLassoCV

# Ensure deepvol package is importable
import deepvol.benchmarks.data_efficiency as de
from deepvol.benchmarks.data_efficiency import BatchSVGPModel, _make_spatial_input, compute_metrics
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from deepvol.surrogates.normalizers import ParameterNormalizerHeston, IVSurfaceNormalizer

def test_numpy_bool_monkeypatch():
    # Verify that np.bool is monkeypatched to np.bool_
    assert np.bool is np.bool_
    
    # Verify numpy.ma compatibility with the monkeypatch
    import numpy.ma as ma
    data = np.array([1, 2, 3])
    mask = np.array([True, False, True], dtype=np.bool)
    masked_arr = ma.array(data, mask=mask)
    
    assert masked_arr[0] is ma.masked
    assert masked_arr[1] == 2
    assert masked_arr[2] is ma.masked
    assert masked_arr.sum() == 2
    assert np.all(masked_arr.mask == [True, False, True])

def test_data_loading_and_splitting_normalizers():
    # Create dummy dataset parameters
    N_total = 50
    dummy_params = np.random.uniform(0.1, 1.0, (N_total, 5))
    dummy_iv = np.random.uniform(0.1, 0.6, (N_total, 8, 11))
    
    # Train/Validation Split (80% / 20%) -> 40 Train, 10 Val
    X_tr, X_va = dummy_params[:40], dummy_params[40:]
    Y_tr, Y_va = dummy_iv[:40], dummy_iv[40:]
    
    # Test fitting and transformation of normalizers
    param_normalizer = ParameterNormalizerHeston().fit(X_tr)
    iv_normalizer = IVSurfaceNormalizer().fit(Y_tr)
    
    X_tr_n = param_normalizer.transform(X_tr)
    X_va_n = param_normalizer.transform(X_va)
    Y_tr_n = iv_normalizer.transform(Y_tr)
    Y_va_n = iv_normalizer.transform(Y_va)
    
    assert X_tr_n.shape == (40, 5)
    assert X_va_n.shape == (10, 5)
    assert Y_tr_n.shape == (40, 8, 11)
    assert Y_va_n.shape == (10, 8, 11)
    
    # Verify round-trip mapping
    X_tr_recon = param_normalizer.inverse_transform(X_tr_n)
    Y_tr_recon = iv_normalizer.inverse_transform(Y_tr_n)
    assert np.allclose(X_tr_recon, X_tr, atol=1e-5)
    assert np.allclose(Y_tr_recon, Y_tr, atol=1e-5)

def test_train_val_split_bounds():
    # Test the splitting logic and boundary fallbacks present in main()
    def get_split(n_samples, total_available):
        N = min(n_samples, total_available)
        if N < 13:
            N = min(13, total_available)
        split = int(0.8 * N)
        if split < 10:
            split = 10
        if split > N:
            split = N
        return N, split
        
    # Standard case
    N, split = get_split(100, 1000)
    assert N == 100
    assert split == 80
    
    # Small N (less than 13) bound
    N, split = get_split(8, 100)
    assert N == 13
    assert split == 10
    
    # Small total_available
    N, split = get_split(100, 5)
    assert N == 5
    assert split == 5

def test_gpe_svgp_logic():
    n_params = 5
    n_grid_points = 88 # 8 * 11
    num_inducing = 20
    batch_size = 10
    
    inducing_points = torch.randn(n_grid_points, num_inducing, n_params)
    model = BatchSVGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_grid_points]))
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=100)
    
    # Forward pass through SVGP
    x = torch.randn(n_grid_points, batch_size, n_params)
    output = model(x)
    assert isinstance(output, gpytorch.distributions.MultivariateNormal)
    assert output.mean.shape == (n_grid_points, batch_size)
    assert output.covariance_matrix.shape == (n_grid_points, batch_size, batch_size)
    
    # Loss computation (ELBO)
    y = torch.randn(n_grid_points, batch_size)
    loss = -mll(output, y).sum()
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    
    # Likelihood prediction
    pred = likelihood(output)
    assert pred.mean.shape == (n_grid_points, batch_size)

def test_pce_expansion_and_fitting():
    n_params = 5
    n_grid_points = 10
    pce_order = 2
    
    pce_dist = cp.Iid(cp.Uniform(-np.sqrt(3), np.sqrt(3)), n_params)
    pce_expansion = cp.generate_expansion(pce_order, pce_dist)
    num_terms = len(pce_expansion)
    
    # Test underdetermined (samples < terms)
    num_samples_under = max(5, num_terms - 5)
    X_tr_under = np.random.randn(num_samples_under, n_params)
    Y_tr_under = np.random.randn(num_samples_under, n_grid_points)
    
    cv_folds_under = min(3, X_tr_under.shape[0])
    is_underdetermined_under = X_tr_under.shape[0] < num_terms
    use_cv_under = (not is_underdetermined_under) and (cv_folds_under >= 2)
    
    assert is_underdetermined_under
    assert not use_cv_under
    
    design_matrix_under = pce_expansion(*(X_tr_under.T)).T
    assert design_matrix_under.shape == (num_samples_under, num_terms)
    
    # OLS fit
    coef_ols = np.linalg.lstsq(design_matrix_under, Y_tr_under, rcond=None)[0]
    assert coef_ols.shape == (num_terms, n_grid_points)
    
    # Ridge fallback (when not use_cv)
    ridge_fallback = Ridge(alpha=1.0)
    ridge_fallback.fit(design_matrix_under, Y_tr_under)
    pred_ridge_fallback = ridge_fallback.predict(design_matrix_under)
    assert pred_ridge_fallback.shape == Y_tr_under.shape
    
    # Lasso fallback (when not use_cv, falls back to Ridge)
    lasso_fallback = Ridge(alpha=1.0)
    lasso_fallback.fit(design_matrix_under, Y_tr_under)
    pred_lasso_fallback = lasso_fallback.predict(design_matrix_under)
    assert pred_lasso_fallback.shape == Y_tr_under.shape
    
    # Test well-determined (samples >= terms)
    num_samples_well = num_terms + 10
    X_tr_well = np.random.randn(num_samples_well, n_params)
    Y_tr_well = np.random.randn(num_samples_well, n_grid_points)
    
    cv_folds_well = min(3, X_tr_well.shape[0])
    is_underdetermined_well = X_tr_well.shape[0] < num_terms
    use_cv_well = (not is_underdetermined_well) and (cv_folds_well >= 2)
    
    assert not is_underdetermined_well
    assert use_cv_well
    
    design_matrix_well = pce_expansion(*(X_tr_well.T)).T
    
    # RidgeCV
    ridge_cv = RidgeCV(cv=cv_folds_well)
    ridge_cv.fit(design_matrix_well, Y_tr_well)
    pred_ridge_cv = ridge_cv.predict(design_matrix_well)
    assert pred_ridge_cv.shape == Y_tr_well.shape
    
    # MultiTaskLassoCV (small max_iter for speed)
    lasso_cv = MultiTaskLassoCV(cv=cv_folds_well, max_iter=10)
    lasso_cv.fit(design_matrix_well, Y_tr_well)
    pred_lasso_cv = lasso_cv.predict(design_matrix_well)
    assert pred_lasso_cv.shape == Y_tr_well.shape

def test_fno_training_step_and_loss():
    n_params = 5
    nT, nK = 8, 11
    T_grid = np.linspace(0.1, 2.0, nT)
    K_grid = np.linspace(0.8, 1.2, nK)
    
    device = torch.device("cpu")
    fno_model = MirrorPaddedFNO2d(param_dim=n_params).to(device)
    
    batch_size = 4
    X_b = torch.randn(batch_size, n_params, requires_grad=False)
    Y_b = torch.randn(batch_size, nT, nK, requires_grad=False)
    
    fno_spatial = _make_spatial_input(T_grid, K_grid, device)
    sp = fno_spatial.expand(batch_size, -1, -1, -1)
    
    pred = fno_model(sp, X_b)
    assert pred.shape == (batch_size, nT, nK)
    
    # Huber loss
    loss_huber = F.huber_loss(pred, Y_b, delta=1.0)
    
    # Arbitrage regularization + negative prediction penalty
    mean_t = torch.zeros((nT, nK), dtype=torch.float32, device=device)
    std_t = torch.ones((nT, nK), dtype=torch.float32, device=device)
    pred_denorm = pred * std_t + mean_t
    
    t_grid_tensor = torch.tensor(T_grid, dtype=torch.float32, device=device)
    k_grid_tensor = torch.tensor(K_grid, dtype=torch.float32, device=device)
    
    loss_arb = arbitrage_free_regularization(pred_denorm, t_grid_tensor, k_grid_tensor)
    loss_neg = F.relu(-pred_denorm).mean()
    
    total_loss = loss_huber + 1e-4 * loss_arb + 1.0 * loss_neg
    
    assert total_loss.ndim == 0
    assert not torch.isnan(total_loss)
    
    # Test backward pass
    optimizer = torch.optim.AdamW(fno_model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    
    has_grad = False
    for param in fno_model.parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any()
    assert has_grad
    
    optimizer.step()

def test_entire_smoke_test_execution(monkeypatch, tmp_path):
    import sys
    
    # Generate dummy npz dataset to feed to data_efficiency.py main function
    dummy_path = tmp_path / "dummy_dataset.npz"
    N_total = 110
    
    params = np.random.uniform(0.1, 1.0, (N_total, 5))
    iv = np.random.uniform(0.1, 0.6, (N_total, 8, 11))
    T_grid = np.linspace(0.1, 2.0, 8)
    K_grid = np.linspace(0.8, 1.2, 11)
    
    np.savez(dummy_path, params=params, iv=iv, T_grid=T_grid, K_grid=K_grid)
    
    # Mock sys.argv
    test_args = [
        "data_efficiency.py",
        "--smoke",
        "--dataset_path", str(dummy_path),
        "--model", "all"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Call the main function of the benchmark script
    # This should run the full GPE, PCE, and FNO smoke workflows successfully
    de.main()

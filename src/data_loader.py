import os
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Grid constants (8 maturities x 11 strikes)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
# Standard moneyness grid (log(F/K))
MONEYNESS = np.linspace(-0.5, 0.5, 11)

def generate_deep_rough_lhs(n_samples=100_000, seed=42):
    """
    Generates the 'Deep Rough' dataset using Latin Hypercube Sampling (LHS).
    6D Parameter Space: (kappa, theta, sigma, rho, v0, H)
    Bounds:
        kappa: [0.1, 5.0]
        theta: [0.01, 0.15]
        sigma: [0.1, 1.0]
        rho:   [-0.9, -0.1]
        v0:    [0.01, 0.15]
        H:     [0.02, 0.15]  <- Deep Rough Regime
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    sample_unit = sampler.random(n=n_samples)
    
    l_bounds = np.array([0.1, 0.01, 0.1, -0.9, 0.01, 0.02])
    u_bounds = np.array([5.0, 0.15, 1.0, -0.1, 0.15, 0.15])
    
    # Scale to bounds
    samples = qmc.scale(sample_unit, l_bounds, u_bounds)
    
    # Columns: kappa, theta, sigma, rho, v0, H
    df = pd.DataFrame(samples, columns=['kappa', 'theta', 'sigma', 'rho', 'v0', 'H'])
    return df

def process_and_scale_data(features_df, targets_df, scalers_dir="artifacts/scalers"):
    """
    Applies Total Variance transformation (W = IV^2 * T), fits scalers,
    and returns PyTorch-ready scaled tensors.
    """
    # Total Variance Transformation W = IV^2 * T
    T_vector = np.repeat(MATURITIES, len(MONEYNESS))
    targets_w = targets_df.values ** 2 * T_vector
    targets_w_df = pd.DataFrame(targets_w, columns=targets_df.columns)
    
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = StandardScaler()
    
    X_scaled = feature_scaler.fit_transform(features_df)
    y_scaled = target_scaler.fit_transform(targets_w_df)
    
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(scalers_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.pkl"))
    np.save(os.path.join(scalers_dir, "T_vector.npy"), T_vector)
    
    return X_scaled, y_scaled

def get_dataloaders(features_df, targets_df, batch_size=32, test_size=0.15, random_state=42, scalers_dir="artifacts/scalers"):
    X_scaled, y_scaled = process_and_scale_data(features_df, targets_df, scalers_dir)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Generating 100k Deep Rough LHS Dataset...")
    params_df = generate_deep_rough_lhs(100_000)
    print("Head of parameters:")
    print(params_df.head())
    
    # We will simulate the IV surface for these parameters using the CUDA engine in a separate script.
    # The dataloader is ready to accept the features and generated targets.
    print("LHS Generation completed successfully.")

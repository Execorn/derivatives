"""
Heston Surrogate MLP — Data Loading and Preprocessing.

Loads the raw Heston training dataset, applies the Total Variance
transformation (W = IV² × T), fits MinMax/Standard scalers, and
returns PyTorch DataLoaders ready for training.

The scaler objects are persisted to artifacts/scalers/ so that the
same transformations can be applied at calibration and inference time
without re-loading the full dataset.

Usage:
    python src/data_loader.py  (smoke-test: prints batch shapes)
"""

import os
import gzip
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Grid constants (must match training data layout: 8 maturities × 11 strikes)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])


def load_and_scale_data(filepath, scalers_dir="artifacts/scalers"):
    """
    Loads Heston dataset from a gzip numpy file into a Pandas DataFrame,
    separates features and targets, scales them, and saves the scalers.

    Features (columns 0:5) are scaled to [-1, 1] using MinMaxScaler.
    Targets are transformed to Total Variance (W = IV² × T) before
    being scaled using StandardScaler. This stabilizes training on
    the edges of the volatility surface.
    """
    # Load binary gzip data using numpy
    with gzip.GzipFile(filepath, "r") as f:
        data = np.load(f)

    # Per requirements, convert and process with Pandas
    df = pd.DataFrame(data)

    # 5 Heston Parameters (data order: v0, rho, sigma, theta, kappa)
    features_df = df.iloc[:, :5]
    # 88 IV points (8 maturities x 11 strikes)
    targets_df = df.iloc[:, 5:]

    # ── Total Variance Transformation ──────────────────────────────────────
    # W = IV² × T  (each maturity spans 11 strike columns)
    T_vector = np.repeat(MATURITIES, 11)  # shape (88,)
    targets_w = targets_df.values ** 2 * T_vector
    targets_df = pd.DataFrame(targets_w, columns=targets_df.columns)

    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = StandardScaler()

    # Fit and transform (StandardScaler is now fit on Total Variance W)
    X_scaled = feature_scaler.fit_transform(features_df)
    y_scaled = target_scaler.fit_transform(targets_df)

    # Save scalers and maturity grid
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(scalers_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.pkl"))
    np.save(os.path.join(scalers_dir, "T_vector.npy"), T_vector)

    return X_scaled, y_scaled


def get_dataloaders(
    filepath, batch_size=32, test_size=0.15, random_state=42, scalers_dir="artifacts/scalers"
):
    """
    Creates PyTorch DataLoaders for train and test sets.
    """
    X_scaled, y_scaled = load_and_scale_data(filepath, scalers_dir)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    filepath = "data/HestonTrainSet.txt.gz"
    print(f"Loading and testing data from {filepath}...")

    train_loader, test_loader = get_dataloaders(filepath, scalers_dir="artifacts/scalers")

    for batch_X, batch_y in train_loader:
        print(f"Train batch X shape: {batch_X.shape}")
        print(f"Train batch y shape: {batch_y.shape}")
        break

    for batch_X, batch_y in test_loader:
        print(f"Test batch X shape: {batch_X.shape}")
        print(f"Test batch y shape: {batch_y.shape}")
        break

    print("Data loader tested successfully. Scalers saved to artifacts/scalers/.")

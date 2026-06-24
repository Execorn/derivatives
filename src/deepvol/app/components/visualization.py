"""
visualization.py — Interactive plotting component for the dashboard.
Uses Plotly to render 3D surface comparisons, 2D smile fits, and optimization timelines.
"""

import plotly.graph_objects as go
import numpy as np

def plot_3d_surfaces(T_grid, K_grid, market_iv, reconstructed_iv=None, market_name="Market / Target", reconstructed_name="Calibrated Model"):
    """
    Renders an interactive 3D comparison of two implied volatility surfaces.
    """
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
    
    fig = go.Figure()
    
    # Target / Market surface in Blues
    fig.add_trace(go.Surface(
        x=K_mesh, y=T_mesh, z=market_iv,
        colorscale="Blues", opacity=0.7, name=market_name, showscale=False
    ))
    
    if reconstructed_iv is not None:
        # Calibrated / Reconstructed surface in Reds
        fig.add_trace(go.Surface(
            x=K_mesh, y=T_mesh, z=reconstructed_iv,
            colorscale="Reds", opacity=0.7, name=reconstructed_name, showscale=False
        ))
        
    fig.update_layout(
        scene=dict(
            xaxis_title="Log-Moneyness (k)",
            yaxis_title="Maturity (T)",
            zaxis_title="Implied Volatility (σ)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=550
    )
    return fig

def plot_parameter_trajectory(history: list[np.ndarray], param_names: list[str]):
    """
    Plots the convergence trajectory of parameters during optimization.
    """
    history_arr = np.array(history)  # Shape (iterations, param_dim)
    n_steps = len(history_arr)
    
    fig = go.Figure()
    for j, name in enumerate(param_names):
        fig.add_trace(go.Scatter(
            x=np.arange(1, n_steps + 1),
            y=history_arr[:, j],
            mode="lines+markers",
            name=name
        ))
        
    fig.update_layout(
        xaxis_title="Gauss-Newton Iteration",
        yaxis_title="Parameter Value",
        height=350,
        margin=dict(l=0, r=0, b=40, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_smile_slice(T_grid, K_grid, market_iv, reconstructed_iv=None, t_idx=0, market_name="Market", reconstructed_name="Calibrated"):
    """
    Compares 2D volatility smiles for a selected maturity slice.
    """
    T = T_grid[t_idx]
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=K_grid, y=market_iv[t_idx, :],
        mode="lines+markers",
        name=f"{market_name} (T={T:.2f})",
        line=dict(color="#00d4ff", width=2)
    ))
    
    if reconstructed_iv is not None:
        fig.add_trace(go.Scatter(
            x=K_grid, y=reconstructed_iv[t_idx, :],
            mode="lines+markers",
            name=f"{reconstructed_name} (T={T:.2f})",
            line=dict(color="#ff3366", width=2, dash="dash")
        ))
        
    fig.update_layout(
        xaxis_title="Log-Moneyness",
        yaxis_title="Implied Volatility",
        height=350,
        margin=dict(l=0, r=0, b=40, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

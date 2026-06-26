"""
policy.py — Deep Hedging Policy network and differentiable transaction cost functions.
Supports LSTM recurrent state updates and smooth/differentiable transaction costs.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


@torch.compile(mode="reduce-overhead")
def proportional_transaction_cost(
    delta_diff: torch.Tensor,
    S: torch.Tensor,
    c_fee: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Computes the proportional transaction cost.
    
    Formula:
        C_trans(delta_diff) = c_fee * |delta_t - delta_{t-1}| * S_t
        
    Parameters:
        delta_diff: Tensor representing the change in holdings, delta_t - delta_{t-1}.
        S: Current spot price S_t of the underlying asset.
        c_fee: Proportional fee coefficient.
        
    Returns:
        The proportional transaction cost.
    """
    cost = c_fee * torch.abs(delta_diff) * S
    return cost.clone()


@torch.compile(mode="reduce-overhead")
def huber_transaction_cost(
    delta_diff: torch.Tensor,
    S: torch.Tensor,
    c_fee: Union[float, torch.Tensor],
    d: float = 0.01
) -> torch.Tensor:
    """
    Computes the Huber-style smooth transaction cost.
    
    Formula:
        H(delta_diff, S_t, c_fee, d) = c_fee * S_t * (delta_diff)^2 / (2 * d) if |delta_diff| <= d
                                      c_fee * S_t * (|delta_diff| - d / 2) otherwise
                                      
    Parameters:
        delta_diff: Tensor representing the change in holdings, delta_t - delta_{t-1}.
        S: Current spot price S_t of the underlying asset.
        c_fee: Proportional fee coefficient.
        d: Transition threshold parameter.
        
    Returns:
        The Huber-style transaction cost.
    """
    abs_diff = torch.abs(delta_diff)
    cost_small = c_fee * S * (delta_diff ** 2) / (2.0 * d)
    cost_large = c_fee * S * (abs_diff - 0.5 * d)
    cost = torch.where(abs_diff <= d, cost_small, cost_large)
    return cost.clone()


@torch.compile(mode="reduce-overhead")
def sqrt_transaction_cost(
    delta_diff: torch.Tensor,
    S: torch.Tensor,
    c_fee: Union[float, torch.Tensor],
    eps_c: float = 1e-6
) -> torch.Tensor:
    """
    Computes the square-root smooth transaction cost.
    
    Formula:
        C_sqrt(delta_diff, S_t, c_fee, eps_c) = c_fee * S_t * sqrt((delta_diff)^2 + eps_c)
        
    Parameters:
        delta_diff: Tensor representing the change in holdings, delta_t - delta_{t-1}.
        S: Current spot price S_t of the underlying asset.
        c_fee: Proportional fee coefficient.
        eps_c: Smoothing parameter.
        
    Returns:
        The square-root transaction cost.
    """
    cost = c_fee * S * torch.sqrt(delta_diff ** 2 + eps_c)
    return cost.clone()


class DeepHedgingPolicy(nn.Module):
    """
    LSTM Recurrent policy network for Deep Hedging.
    
    Takes features input tensor of shape [Batch, Seq_Len, Input_Dim] or [Batch, Input_Dim]
    and returns the target asset holdings delta_t constrained to the interval [0.01, 0.99] or [0, 1] via sigmoid.
    Operates in single-precision float32.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        range_limit: str = "0.01_0.99"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if range_limit not in ("0.01_0.99", "0_1"):
            raise ValueError(f"range_limit must be '0.01_0.99' or '0_1', got '{range_limit}'")
        self.range_limit = range_limit
        
        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.to(torch.float32)

    @torch.compile(mode="reduce-overhead")
    def _step(
        self,
        x: torch.Tensor,
        h: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes a single step using the LSTM cell and applies constraints.
        
        Parameters:
            x: Input tensor for current step, shape [Batch, Input_Dim].
            h: Recurrent states, tuple of (h_c, c_c), each shape [Batch, Hidden_Dim].
            
        Returns:
            delta: Holdings ratio delta_t, shape [Batch, Output_Dim].
            h_next: Updated recurrent states, tuple of (h_c_next, c_c_next).
        """
        # Ensure correct type/device alignment for safety
        x = x.to(dtype=torch.float32)
        h_c, c_c = h[0].to(dtype=torch.float32), h[1].to(dtype=torch.float32)
        
        h_c_next, c_c_next = self.lstm_cell(x, (h_c, c_c))
        out = self.fc(h_c_next)
        
        if self.range_limit == "0.01_0.99":
            delta = 0.01 + 0.98 * torch.sigmoid(out)
            # Clamping safely to prevent singularities at boundary
            delta = torch.clamp(delta, min=0.01, max=0.99)
        else:  # "0_1"
            delta = torch.sigmoid(out)
            delta = torch.clamp(delta, min=0.0, max=1.0)
            
        return delta.clone(), (h_c_next.clone(), c_c_next.clone())

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for either sequential or single-step features.
        
        Parameters:
            x: Input tensor of shape [Batch, Seq_Len, Input_Dim] or [Batch, Input_Dim].
            h: Optional recurrent states (h_c, c_c).
            
        Returns:
            delta: Target holdings delta, shape [Batch, Seq_Len, Output_Dim] or [Batch, Output_Dim].
            h_next: Updated recurrent states.
        """
        x = x.to(dtype=torch.float32)
        
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            if h is None:
                h_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=torch.float32)
                c_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=torch.float32)
                h = (h_c, c_c)
            else:
                h = (h[0].to(dtype=torch.float32), h[1].to(dtype=torch.float32))
                
            outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                delta_t, h = self._step(x_t, h)
                outputs.append(delta_t)
                
            outputs = torch.stack(outputs, dim=1)
            return outputs.clone(), (h[0].clone(), h[1].clone())
            
        elif x.dim() == 2:
            if h is None:
                batch_size = x.shape[0]
                h_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=torch.float32)
                c_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=torch.float32)
                h = (h_c, c_c)
            else:
                h = (h[0].to(dtype=torch.float32), h[1].to(dtype=torch.float32))
                
            delta_t, h_next = self._step(x, h)
            return delta_t.clone(), (h_next[0].clone(), h_next[1].clone())
            
        else:
            raise ValueError(f"Input tensor x must have 2 or 3 dimensions, got shape {x.shape}")

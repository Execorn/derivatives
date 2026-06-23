"""
deep_hedging.py — Vectorized environment and LSTM policy for model-free reinforcement learning hedging.
Supports multiple hedging instruments, transaction costs, and entropic/quadratic risk measures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional


class HedgingPolicy(nn.Module):
    """
    Fully Recurrent LSTM-based hedging policy network.
    Outputs the hedge ratio (position) delta_t for all hedging instruments at time t.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM cell for recurrent history tracking
        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        
        # Linear layer mapping hidden state to trading action
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: Input tensor of shape (batch_size, input_dim)
        h: Optional tuple of (h_c, c_c) LSTM states, each of shape (batch_size, hidden_dim)
        
        Returns:
            delta: Hedge ratio action of shape (batch_size, output_dim)
            h_next: Updated LSTM state tuple
        """
        if h is None:
            # Initialize hidden state on the same device as input
            batch_size = x.shape[0]
            h_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            c_c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            h = (h_c, c_c)
            
        h_next = self.lstm_cell(x, h)
        delta = self.fc(h_next[0])
        
        # We squash delta to a reasonable trading limit, e.g. [-2.0, 2.0]
        delta = 2.0 * torch.tanh(delta)
        
        return delta, h_next


class DeepHedgingEnv:
    """
    Vectorized deep hedging environment managing asset price paths, portfolio wealth,
    transaction costs, and option payoff evaluation.
    """
    def __init__(
        self,
        H: torch.Tensor,
        payoff: torch.Tensor,
        cost_coeffs: torch.Tensor,
        risk_aversion: float = 1.0,
        risk_measure: str = "entropic",
        strike: float = 100.0,
        expiry: float = 1.0,
        t_grid: Optional[torch.Tensor] = None
    ):
        """
        Parameters:
            H: Tensor of shape (N_paths, N_t + 1, d) - prices of hedging instruments.
               H[:, :, 0] is assumed to be the underlying stock spot S_t.
            payoff: Tensor of shape (N_paths,) - terminal payoff of the derivative.
            cost_coeffs: Tensor of shape (d,) - proportional cost coefficients (e.g. 0.0001).
            risk_aversion: Risk aversion parameter lambda.
            risk_measure: "entropic" (exponential utility) or "quad" (mean squared error).
            strike: Strike price K of the option.
            expiry: Maturity T of the option.
            t_grid: Optional tensor of shape (N_t + 1,) representing the simulation time steps.
        """
        self.H = H
        self.payoff = payoff
        self.cost_coeffs = cost_coeffs.to(device=H.device, dtype=H.dtype)
        self.risk_aversion = risk_aversion
        self.risk_measure = risk_measure.lower()
        self.strike = strike
        self.expiry = expiry
        
        self.N_paths, self.N_t_plus_1, self.d = H.shape
        self.N_t = self.N_t_plus_1 - 1
        self.dt = expiry / self.N_t
        
        if t_grid is None:
            self.t_grid = torch.arange(self.N_t_plus_1, device=H.device, dtype=H.dtype) * self.dt
        else:
            self.t_grid = t_grid.to(device=H.device, dtype=H.dtype)
            
    def get_state(self, k: int, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Constructs the state feature vector for time step k.
        State features: [log(S_k / K), T - t_k, vol_proxy (local standard deviation), prev_delta]
        """
        if hasattr(self, "_precomputed_log_moneyness") and self._precomputed_log_moneyness is not None:
            log_moneyness = self._precomputed_log_moneyness[:, k]
            time_to_expiry_tensor = self._precomputed_time_to_expiry[:, k]
            vol_proxy = self._precomputed_vol_proxy[:, k]
        else:
            S_k = self.H[:, k, 0:1]  # (N_paths, 1)
            log_moneyness = torch.log(torch.clamp(S_k / self.strike, min=1e-5))
            time_to_expiry = self.expiry - self.t_grid[k]
            time_to_expiry_tensor = torch.full_like(S_k, time_to_expiry)
            
            # Volatility proxy: local rolling standard deviation of underlying log returns
            # For simplicity, if k < 5, we use a default volatility proxy (0.2),
            # otherwise we calculate standard deviation of past 5 log returns.
            if k < 5:
                vol_proxy = torch.full_like(S_k, 0.2)
            else:
                past_S = self.H[:, k-5:k+1, 0]  # (N_paths, 6)
                log_returns = torch.log(torch.clamp(past_S[:, 1:] / torch.clamp(past_S[:, :-1], min=1e-5), min=1e-5))  # (N_paths, 5)
                vol_proxy = torch.std(log_returns, dim=-1, keepdim=True) * np.sqrt(252)
            
        # Concatenate features: log_moneyness, time_to_expiry, vol_proxy, and all dimensions of prev_delta
        # shape: (N_paths, 3 + d)
        state = torch.cat([log_moneyness, time_to_expiry_tensor, vol_proxy, prev_delta], dim=-1)
        return state


    def simulate_hedging_episode(self, policy: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates the entire hedging episode, calculating wealth and transaction costs.
        
        Returns:
            wealth: Final portfolio wealth of shape (N_paths,)
            total_costs: Total transaction costs of shape (N_paths,)
            all_deltas: Tensor of shape (N_paths, N_t, d) representing the hedging path.
        """
        device = self.H.device
        dtype = self.H.dtype
        
        # Precompute state features if enabled
        if getattr(self, "precompute", True):
            S = self.H[:, :, 0:1]
            self._precomputed_log_moneyness = torch.log(torch.clamp(S / self.strike, min=1e-5))
            
            time_to_expiry_all = self.expiry - self.t_grid
            self._precomputed_time_to_expiry = time_to_expiry_all.view(1, -1, 1).expand(self.N_paths, -1, -1)
            
            self._precomputed_vol_proxy = torch.full_like(S, 0.2)
            if self.N_t >= 5:
                S_0 = self.H[:, :, 0]
                log_returns = torch.log(torch.clamp(S_0[:, 1:] / torch.clamp(S_0[:, :-1], min=1e-5), min=1e-5))  # (N_paths, N_t)
                windows = log_returns.unfold(dimension=-1, size=5, step=1)  # (N_paths, N_t - 4, 5)
                vol_proxy_windows = torch.std(windows, dim=-1, keepdim=True) * np.sqrt(252)  # (N_paths, N_t - 4, 1)
                self._precomputed_vol_proxy[:, 5:self.N_t + 1] = vol_proxy_windows
        else:
            self._precomputed_log_moneyness = None
            self._precomputed_time_to_expiry = None
            self._precomputed_vol_proxy = None
            
        # Initial wealth is set to 0.0 (indifference pricing baseline)
        wealth = torch.zeros(self.N_paths, device=device, dtype=dtype)
        total_costs = torch.zeros(self.N_paths, device=device, dtype=dtype)
        
        # Previous trading position is 0
        prev_delta = torch.zeros(self.N_paths, self.d, device=device, dtype=dtype)
        lstm_state = None
        
        deltas = []
        
        for k in range(self.N_t):
            # 1. Get current environment state
            state = self.get_state(k, prev_delta)
            
            # 2. Query policy for actions (hedge ratio delta)
            delta, lstm_state = policy(state, lstm_state)  # delta shape: (N_paths, d)
            deltas.append(delta)
            
            # 3. Calculate rebalancing cost
            # c_k = sum_i cost_coeff_i * H_k^i * |delta_k^i - prev_delta_k^i|
            delta_diff = torch.abs(delta - prev_delta)
            step_costs = torch.sum(self.cost_coeffs.unsqueeze(0) * self.H[:, k, :] * delta_diff, dim=-1)
            total_costs = total_costs + step_costs
            
            # 4. Update wealth at the start of step k+1 using pricing changes
            # W_{k+1} = W_k + delta_k * (H_{k+1} - H_k) - step_costs
            price_change = self.H[:, k+1, :] - self.H[:, k, :]
            trading_gain = torch.sum(delta * price_change, dim=-1)
            wealth = wealth + trading_gain - step_costs
            
            prev_delta = delta
            
        # 5. Unwind portfolio to 0 at maturity (T)
        # Cost to unwind position: sum_i cost_coeff_i * H_N^i * |0 - prev_delta^i|
        terminal_unwind_cost = torch.sum(self.cost_coeffs.unsqueeze(0) * self.H[:, -1, :] * torch.abs(prev_delta), dim=-1)
        total_costs = total_costs + terminal_unwind_cost
        wealth = wealth - terminal_unwind_cost
        
        all_deltas = torch.stack(deltas, dim=1)  # shape: (N_paths, N_t, d)
        
        # Clean up precomputed features
        self._precomputed_log_moneyness = None
        self._precomputed_time_to_expiry = None
        self._precomputed_vol_proxy = None
        
        return wealth, total_costs, all_deltas

    def compute_loss(self, wealth: torch.Tensor) -> torch.Tensor:
        """
        Computes the hedging risk measure loss.
        """
        # Hedging error: wealth - payoff
        hedging_error = wealth - self.payoff
        
        if self.risk_measure == "entropic":
            # Entropy / Exponential utility loss: E[exp(-lambda * HE)]
            loss = torch.mean(torch.exp(-self.risk_aversion * hedging_error))
        elif self.risk_measure == "quad":
            # Mean Squared Error: E[HE^2]
            loss = torch.mean(hedging_error ** 2)
        else:
            raise ValueError(f"Unknown risk measure: {self.risk_measure}")
            
        return loss


def train_deep_hedger(
    env: DeepHedgingEnv,
    policy: nn.Module,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    device: str = "cuda"
) -> List[float]:
    """
    Trains the hedging policy using pathwise backpropagation (BPTT).
    """
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    losses = []
    
    # We partition paths into mini-batches to optimize VRAM
    num_paths = env.N_paths
    num_batches = (num_paths + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        
        # Shuffle path indices
        indices = torch.randperm(num_paths, device=device)
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            if len(batch_idx) == 0:
                continue
                
            optimizer.zero_grad()
            
            # Slice H and payoff for the current batch
            batch_H = env.H[batch_idx]
            batch_payoff = env.payoff[batch_idx]
            
            # Create a local sub-env for this batch
            if env.__class__.__name__ == "BarrierHedgingEnv":
                sub_env = env.__class__(
                    H=batch_H,
                    cost_coeffs=env.cost_coeffs,
                    strike=env.strike,
                    barrier=env.barrier,
                    expiry=env.expiry,
                    risk_aversion=env.risk_aversion,
                    risk_measure=env.risk_measure,
                    t_grid=env.t_grid
                )
            else:
                sub_env = DeepHedgingEnv(
                    H=batch_H,
                    payoff=batch_payoff,
                    cost_coeffs=env.cost_coeffs,
                    risk_aversion=env.risk_aversion,
                    risk_measure=env.risk_measure,
                    strike=env.strike,
                    expiry=env.expiry,
                    t_grid=env.t_grid
                )
            
            # Run simulation episode
            wealth, _, _ = sub_env.simulate_hedging_episode(policy)
            
            # Compute loss
            loss = sub_env.compute_loss(wealth)
            
            # Backpropagate gradients pathwise
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_idx)
            
        avg_loss = epoch_loss / num_paths
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {avg_loss:.6f}")
            
    return losses


def estimate_gpd_tail_index_pwm(returns, threshold_quantile=0.95):
    """
    Differentiably estimates the GPD tail index xi using the
    Probability Weighted Moments (PWM) method on the right tail.
    
    returns: Tensor of shape (batch_size, sequence_length)
    """
    batch_size, seq_len = returns.shape
    
    # Calculate exact number of exceedances per path
    n_exceed = int(round((1.0 - threshold_quantile) * seq_len))
    
    # Guard 1: Handle case where no exceedances exist to avoid tensor shape mismatch and NaN
    if n_exceed <= 1:
        return torch.zeros(batch_size, device=returns.device, dtype=returns.dtype)
        
    q = torch.quantile(returns, threshold_quantile, dim=-1, keepdim=True)
    deviations = returns - q
    sorted_dev, _ = torch.sort(deviations, dim=-1, descending=False)
    y = sorted_dev[:, -n_exceed:]  # shape: (batch_size, n_exceed)
    
    # Ensure all exceedances are positive (clamped to epsilon)
    y = torch.clamp(y, min=1e-6)
    
    # Compute Probability Weighted Moments (PWM)
    i = torch.arange(1, n_exceed + 1, device=returns.device, dtype=torch.float32)
    weight = 1.0 - (i - 0.35) / n_exceed  # shape: (n_exceed,)
    
    a0 = torch.mean(y, dim=-1)  # shape: (batch_size,)
    a1 = torch.sum(weight * y, dim=-1) / n_exceed  # shape: (batch_size,)
    
    # Shape parameter xi
    # Guard 2: Prevent near-zero division by clamping denominator absolute value
    denominator = a0 - 2.0 * a1
    denom_sign = torch.sign(denominator)
    denom_sign = torch.where(denom_sign == 0.0, torch.tensor(1.0, device=returns.device), denom_sign)
    denominator = denom_sign * torch.clamp(torch.abs(denominator), min=1e-8)
    
    xi = 2.0 - a0 / denominator
    
    return xi



def compute_autocorrelation(x, lag):
    """
    Differentiably computes the autocorrelation of a time series x at a given lag.
    
    x: Tensor of shape (batch_size, sequence_length)
    """
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-8
    
    x_centered = x - mean
    
    # Compute covariance at lag
    cov = torch.mean(x_centered[:, :-lag] * x_centered[:, lag:], dim=-1, keepdim=True)
    corr = cov / var
    return corr.squeeze(-1)


def compute_acf_loss(returns, target_acf, max_lag=20):
    """
    Computes Volatility Clustering Loss (LACF) by comparing autocorrelation of
    absolute returns up to max_lag.
    
    returns: Tensor of shape (batch_size, sequence_length)
    target_acf: Tensor of shape (max_lag,) representing the real market ACF
    """
    abs_ret = torch.abs(returns)
    acf_list = []
    for lag in range(1, max_lag + 1):
        acf_list.append(compute_autocorrelation(abs_ret, lag))
    
    gen_acf = torch.stack(acf_list, dim=-1)
    loss = torch.mean((gen_acf - target_acf.unsqueeze(0)) ** 2)
    return loss


def compute_leverage_loss(returns, target_leverage, vol_window=20):
    """
    Computes Leverage Effect Loss (LLev) by comparing correlation between
    past returns and future realized volatility.
    
    returns: Tensor of shape (batch_size, sequence_length)
    target_leverage: Scalar representing the real market leverage correlation
    """
    batch_size, seq_len = returns.shape
    
    # Compute future realized volatility over vol_window
    unfolded = returns.unfold(dimension=-1, size=vol_window, step=1)
    future_vol = torch.std(unfolded, dim=-1, unbiased=False)
    
    past_returns = returns[:, :future_vol.shape[-1]]
    
    mean_ret = torch.mean(past_returns, dim=-1, keepdim=True)
    mean_vol = torch.mean(future_vol, dim=-1, keepdim=True)
    
    var_ret = torch.var(past_returns, dim=-1, keepdim=True, unbiased=False) + 1e-8
    var_vol = torch.var(future_vol, dim=-1, keepdim=True, unbiased=False) + 1e-8
    
    cov = torch.mean((past_returns - mean_ret) * (future_vol - mean_vol), dim=-1, keepdim=True)
    leverage_corr = cov / torch.sqrt(var_ret * var_vol)
    
    loss = torch.mean((leverage_corr.squeeze(-1) - target_leverage) ** 2)
    return loss


def compute_cfvc_loss(returns, target_corr_matrix, scales=[5, 20, 60, 120]):
    """
    Computes Coarse-to-Fine Volatility Correlation Loss (LCFVC).
    
    returns: Tensor of shape (batch_size, sequence_length)
    target_corr_matrix: Tensor of shape (M, M) representing real market correlation
    """
    batch_size, seq_len = returns.shape
    
    # Filter scales that are strictly less than seq_len
    valid_indices = [i for i, s in enumerate(scales) if s < seq_len]
    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=returns.device, dtype=returns.dtype)
        
    valid_scales = [scales[i] for i in valid_indices]
    sub_target_corr = target_corr_matrix[valid_indices][:, valid_indices]
    
    vols = []
    min_len = seq_len - max(valid_scales) + 1
    
    for scale in valid_scales:
        unfolded = returns.unfold(dimension=-1, size=scale, step=1)
        vol = torch.sqrt(torch.mean(unfolded ** 2, dim=-1) + 1e-8)
        vols.append(vol[:, -min_len:])
        
    vols_tensor = torch.stack(vols, dim=1)  # shape: (batch_size, num_valid_scales, min_len)
    
    vols_centered = vols_tensor - torch.mean(vols_tensor, dim=-1, keepdim=True)
    vols_std = torch.std(vols_tensor, dim=-1, keepdim=True, unbiased=False) + 1e-8
    vols_norm = vols_centered / vols_std
    
    corr_matrix = torch.matmul(vols_norm, vols_norm.transpose(-1, -2)) / min_len
    diff = corr_matrix - sub_target_corr.unsqueeze(0)
    loss = torch.mean(torch.norm(diff, p='fro', dim=(-2, -1)))
    return loss


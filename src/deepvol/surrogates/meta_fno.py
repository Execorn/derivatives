"""
meta_fno.py — Physics-Informed Meta-Learning FNO (PI-M-FNO) Model and Meta-Learning Engine.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from deepvol.surrogates.fno_model import SpectralConv2d
from deepvol.surrogates.pde_loss import DupirePDELoss


class MetaFNO2d(nn.Module):
    """
    Fourier Neural Operator with meta-learning capabilities and frozen-core adaptation.

    Formula References:
        Li, Z. et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations.
        arXiv:2010.08890.
    """

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 64):
        super().__init__()
        self.width = width

        # Input layer mapping (K, T, sigma_loc) to hidden width
        self.fc0 = nn.Linear(3, self.width)

        # Spectral Convolution Core (Frozen during online adaptation)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)

        # Pointwise Conv 1D equivalent (using Conv2D with 1x1 kernel)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Output MLP Head (Adapted online during crisis events)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    @torch.compile(mode="reduce-overhead")
    def _forward_core_compiled(self, grid_inputs: torch.Tensor) -> torch.Tensor:
        """
        Compiled internal method running the frozen core layers.
        """
        x = self.fc0(grid_inputs)  # [Batch, N_K, N_T, width]
        x = x.permute(0, 3, 1, 2)  # [Batch, width, N_K, N_T]

        x1 = self.conv0(x) + self.w0(x)
        x1 = torch.tanh(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = torch.tanh(x2)
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = torch.tanh(x3)

        x3 = x3.permute(0, 2, 3, 1)  # [Batch, N_K, N_T, width]
        return x3

    def forward_core(self, grid_inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs the frozen core layers of the FNO and returns extracted spatial features.
        Clones the tensor outside of torch.compile() to prevent CUDAGraphs buffer overwrites.
        """
        return self._forward_core_compiled(grid_inputs).clone()

    @torch.compile(mode="reduce-overhead")
    def _forward_mlp_compiled(self, core_features: torch.Tensor) -> torch.Tensor:
        """
        Compiled internal method running the adaptable output MLP layers.
        """
        x4 = torch.tanh(self.fc1(core_features))  # [Batch, N_K, N_T, 128]
        out = self.fc2(x4).squeeze(-1)  # [Batch, N_K, N_T]
        return out

    def forward_mlp(self, core_features: torch.Tensor) -> torch.Tensor:
        """
        Runs only the adaptable output MLP layers.
        Clones the tensor outside of torch.compile() to prevent CUDAGraphs buffer overwrites.
        """
        return self._forward_mlp_compiled(core_features).clone()

    def forward(self, grid_inputs: torch.Tensor) -> torch.Tensor:
        """
        Combined forward pass.
        """
        core_feats = self.forward_core(grid_inputs)
        return self.forward_mlp(core_feats)

    def get_adaptable_parameters(self) -> List[nn.Parameter]:
        """
        Returns only the output MLP parameters for fast online adaptation (Frozen Core strategy).
        """
        return list(self.fc1.parameters()) + list(self.fc2.parameters())


def train_reptile(
    model: MetaFNO2d,
    tasks: List[Dict[str, torch.Tensor]],
    pde_loss_fn: DupirePDELoss,
    inner_lr: float = 1e-3,
    outer_lr: float = 0.1,
    inner_steps: int = 3,
    meta_epochs: int = 10,
    meta_batch_size: int = 4,
) -> None:
    """
    Offline training loop using the Reptile meta-learning algorithm.

    Formula References:
        Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms.
        arXiv:1803.02999.
    """
    adaptable_params = model.get_adaptable_parameters()

    for epoch in range(meta_epochs):
        indices = torch.randperm(len(tasks)).tolist()

        for i in range(0, len(tasks), meta_batch_size):
            batch_indices = indices[i : i + meta_batch_size]
            if not batch_indices:
                continue

            # 1. Save current meta weights (phi)
            phi_data = [p.data.clone() for p in adaptable_params]
            meta_updates = [torch.zeros_like(p) for p in adaptable_params]

            # 2. Loop over each task in the batch
            for idx in batch_indices:
                task = tasks[idx]

                # Load meta-parameters as start weights
                for p, val in zip(adaptable_params, phi_data):
                    p.data.copy_(val)

                # Pre-compute core features once per task
                with torch.no_grad():
                    core_features = model.forward_core(task["grid_inputs"])

                # Task-level inner optimization on MLP only
                optimizer = torch.optim.SGD(adaptable_params, lr=inner_lr)
                for step in range(inner_steps):
                    optimizer.zero_grad()
                    C_pred = model.forward_mlp(core_features)
                    loss = pde_loss_fn(
                        C_pred,
                        task["K"],
                        task["T"],
                        task["sigma_loc"],
                        task["r"],
                        task["q"],
                    )
                    loss.backward()
                    optimizer.step()

                # Compute difference: theta^(k) - phi
                for m_up, p, p_init in zip(meta_updates, adaptable_params, phi_data):
                    m_up.add_(p.data - p_init)

            # 3. Restore parameters to meta weights (phi)
            for p, val in zip(adaptable_params, phi_data):
                p.data.copy_(val)

            # 4. Perform outer-loop meta-weight update: phi = phi + beta * (sum(theta_b - phi) / B)
            with torch.no_grad():
                for p, m_up in zip(adaptable_params, meta_updates):
                    p.data.add_(outer_lr * m_up / len(batch_indices))


def train_fomaml(
    model: MetaFNO2d,
    tasks: List[Dict[str, torch.Tensor]],
    pde_loss_fn: DupirePDELoss,
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
    inner_steps: int = 3,
    meta_epochs: int = 10,
    meta_batch_size: int = 4,
) -> None:
    """
    Offline training loop using the First-Order MAML (FOMAML) meta-learning algorithm.

    Formula References:
        Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation
        of Deep Networks. arXiv:1703.03400.
    """
    adaptable_params = model.get_adaptable_parameters()

    for epoch in range(meta_epochs):
        indices = torch.randperm(len(tasks)).tolist()

        for i in range(0, len(tasks), meta_batch_size):
            batch_indices = indices[i : i + meta_batch_size]
            if not batch_indices:
                continue

            phi_data = [p.data.clone() for p in adaptable_params]
            meta_grads = [torch.zeros_like(p) for p in adaptable_params]

            for idx in batch_indices:
                task = tasks[idx]

                # Load meta-parameters
                for p, val in zip(adaptable_params, phi_data):
                    p.data.copy_(val)

                # Pre-compute core features once per task
                with torch.no_grad():
                    core_features = model.forward_core(task["grid_inputs"])

                # Inner loop gradient updates
                optimizer = torch.optim.SGD(adaptable_params, lr=inner_lr)
                for step in range(inner_steps):
                    optimizer.zero_grad()
                    C_pred = model.forward_mlp(core_features)
                    loss = pde_loss_fn(
                        C_pred,
                        task["K"],
                        task["T"],
                        task["sigma_loc"],
                        task["r"],
                        task["q"],
                    )
                    loss.backward()
                    optimizer.step()

                # Compute gradient at adapted weights for FOMAML (query loss step)
                optimizer.zero_grad()
                C_pred = model.forward_mlp(core_features)
                loss = pde_loss_fn(
                    C_pred,
                    task["K"],
                    task["T"],
                    task["sigma_loc"],
                    task["r"],
                    task["q"],
                )
                loss.backward()

                # Accumulate gradients (first-order approximation)
                for m_grad, p in zip(meta_grads, adaptable_params):
                    if p.grad is not None:
                        m_grad.add_(p.grad.data)

            # Restore parameters to meta weights (phi)
            for p, val in zip(adaptable_params, phi_data):
                p.data.copy_(val)

            # Outer update: phi = phi - outer_lr * (sum(grads) / B)
            with torch.no_grad():
                for p, m_grad in zip(adaptable_params, meta_grads):
                    p.data.sub_(outer_lr * m_grad / len(batch_indices))

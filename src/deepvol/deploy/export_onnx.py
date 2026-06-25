import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

# Resolve project path and inject src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.utils.path_helpers import get_weights_path

def build_fft_matrices(H, W):
    """
    Compute real-valued linear transformation matrices representing 2D RFFT and 2D IRFFT.
    This allows tracing FNO spectral convolutions to ONNX without complex number support.
    """
    W_f = W // 2 + 1
    N = H * W
    M = H * W_f
    
    # 1. RFFT matrices
    M_rfft_real = torch.zeros(M, N, dtype=torch.float32)
    M_rfft_imag = torch.zeros(M, N, dtype=torch.float32)
    
    for k in range(N):
        e_k = torch.zeros(N, dtype=torch.float32)
        e_k[k] = 1.0
        e_k = e_k.view(1, 1, H, W)
        
        y = torch.fft.rfft2(e_k)
        M_rfft_real[:, k] = y.real.flatten()
        M_rfft_imag[:, k] = y.imag.flatten()
        
    # 2. IRFFT matrix
    M_irfft = torch.zeros(N, 2 * M, dtype=torch.float32)
    
    # Real part impulses
    for k in range(M):
        z_k = torch.zeros(M, dtype=torch.complex64)
        z_k[k] = 1.0 + 0.0j
        z_k = z_k.view(1, 1, H, W_f)
        
        x_rec = torch.fft.irfft2(z_k, s=(H, W))
        M_irfft[:, k] = x_rec.flatten()
        
    # Imag part impulses
    for k in range(M):
        z_k = torch.zeros(M, dtype=torch.complex64)
        z_k[k] = 0.0 + 1.0j
        z_k = z_k.view(1, 1, H, W_f)
        
        x_rec = torch.fft.irfft2(z_k, s=(H, W))
        M_irfft[:, M + k] = x_rec.flatten()
        
    return M_rfft_real, M_rfft_imag, M_irfft

class SpectralConv2dDeployment(nn.Module):
    """
    Real-split equivalent of SpectralConv2d using grouped 1x1 convolutions.
    """
    def __init__(self, original_conv, H, W):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.modes1 = original_conv.modes1
        self.modes2 = original_conv.modes2
        
        self.H = H
        self.W = W
        self.W_f = W // 2 + 1
        
        # Build static transformation matrices
        M_rfft_real, M_rfft_imag, M_irfft = build_fft_matrices(H, W)
        self.register_buffer("M_rfft_real", M_rfft_real)
        self.register_buffer("M_rfft_imag", M_rfft_imag)
        self.register_buffer("M_irfft", M_irfft)
        
        # Setup Conv2d blocks for part 1 (low frequencies)
        seq_len1 = self.modes1 * self.modes2
        self.conv_part1 = nn.Conv2d(
            in_channels=seq_len1 * 2 * self.in_channels,
            out_channels=seq_len1 * 2 * self.out_channels,
            kernel_size=1,
            groups=seq_len1,
            bias=False
        )
        
        # Map original complex weights to real-split block matrix weights for part 1
        w1 = original_conv.weights1.permute(2, 3, 1, 0).reshape(seq_len1, self.out_channels, self.in_channels)
        w1_real = w1.real
        w1_imag = w1.imag
        conv_weight1 = torch.zeros(seq_len1 * 2 * self.out_channels, 2 * self.in_channels, 1, 1)
        for g in range(seq_len1):
            conv_weight1[g * 2 * self.out_channels : g * 2 * self.out_channels + self.out_channels, :self.in_channels, 0, 0] = w1_real[g]
            conv_weight1[g * 2 * self.out_channels : g * 2 * self.out_channels + self.out_channels, self.in_channels:, 0, 0] = -w1_imag[g]
            conv_weight1[g * 2 * self.out_channels + self.out_channels : (g+1) * 2 * self.out_channels, :self.in_channels, 0, 0] = w1_imag[g]
            conv_weight1[g * 2 * self.out_channels + self.out_channels : (g+1) * 2 * self.out_channels, self.in_channels:, 0, 0] = w1_real[g]
        self.conv_part1.weight.data.copy_(conv_weight1)
        
        # Setup Conv2d blocks for part 2 (high frequencies)
        seq_len2 = self.modes1 * self.modes2
        self.conv_part2 = nn.Conv2d(
            in_channels=seq_len2 * 2 * self.in_channels,
            out_channels=seq_len2 * 2 * self.out_channels,
            kernel_size=1,
            groups=seq_len2,
            bias=False
        )
        
        # Map original complex weights to real-split block matrix weights for part 2
        w2 = original_conv.weights2.permute(2, 3, 1, 0).reshape(seq_len2, self.out_channels, self.in_channels)
        w2_real = w2.real
        w2_imag = w2.imag
        conv_weight2 = torch.zeros(seq_len2 * 2 * self.out_channels, 2 * self.in_channels, 1, 1)
        for g in range(seq_len2):
            conv_weight2[g * 2 * self.out_channels : g * 2 * self.out_channels + self.out_channels, :self.in_channels, 0, 0] = w2_real[g]
            conv_weight2[g * 2 * self.out_channels : g * 2 * self.out_channels + self.out_channels, self.in_channels:, 0, 0] = -w2_imag[g]
            conv_weight2[g * 2 * self.out_channels + self.out_channels : (g+1) * 2 * self.out_channels, :self.in_channels, 0, 0] = w2_imag[g]
            conv_weight2[g * 2 * self.out_channels + self.out_channels : (g+1) * 2 * self.out_channels, self.in_channels:, 0, 0] = w2_real[g]
        self.conv_part2.weight.data.copy_(conv_weight2)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. RFFT using matrix multiplication
        x_flat = x.view(B, self.in_channels, self.H * self.W)
        x_ft_real = torch.matmul(x_flat, self.M_rfft_real.T).view(B, self.in_channels, self.H, self.W_f)
        x_ft_imag = torch.matmul(x_flat, self.M_rfft_imag.T).view(B, self.in_channels, self.H, self.W_f)
        
        # 2. Slice and apply part 1
        x_ft_real_1 = x_ft_real[:, :, :self.modes1, :self.modes2]
        x_ft_imag_1 = x_ft_imag[:, :, :self.modes1, :self.modes2]
        
        x_stacked_1 = torch.stack([x_ft_real_1, x_ft_imag_1], dim=1)
        x_permuted_1 = x_stacked_1.permute(0, 3, 4, 1, 2)
        x_conv_in_1 = x_permuted_1.reshape(B, self.modes1 * self.modes2 * 2 * self.in_channels, 1, 1)
        
        out_conv_1 = self.conv_part1(x_conv_in_1)
        out_reshaped_1 = out_conv_1.view(B, self.modes1, self.modes2, 2, self.out_channels)
        out_permuted_1 = out_reshaped_1.permute(0, 3, 4, 1, 2)
        
        part1_real = out_permuted_1[:, 0]
        part1_imag = out_permuted_1[:, 1]
        
        # 3. Slice and apply part 2
        x_ft_real_2 = x_ft_real[:, :, -self.modes1:, :self.modes2]
        x_ft_imag_2 = x_ft_imag[:, :, -self.modes1:, :self.modes2]
        
        x_stacked_2 = torch.stack([x_ft_real_2, x_ft_imag_2], dim=1)
        x_permuted_2 = x_stacked_2.permute(0, 3, 4, 1, 2)
        x_conv_in_2 = x_permuted_2.reshape(B, self.modes1 * self.modes2 * 2 * self.in_channels, 1, 1)
        
        out_conv_2 = self.conv_part2(x_conv_in_2)
        out_reshaped_2 = out_conv_2.view(B, self.modes1, self.modes2, 2, self.out_channels)
        out_permuted_2 = out_reshaped_2.permute(0, 3, 4, 1, 2)
        
        part2_real = out_permuted_2[:, 0]
        part2_imag = out_permuted_2[:, 1]
        
        # 4. Reconstruct columns
        modes1_eff = min(self.modes1, self.H)
        
        slice1_real = part1_real[:, :, :self.H - modes1_eff]
        column_real = torch.cat([slice1_real, part2_real], dim=2)
        
        slice1_imag = part1_imag[:, :, :self.H - modes1_eff]
        column_imag = torch.cat([slice1_imag, part2_imag], dim=2)
        
        # 5. IRFFT using matrix multiplication
        column_real_flat = column_real.reshape(B, self.out_channels, self.H * self.W_f)
        column_imag_flat = column_imag.reshape(B, self.out_channels, self.H * self.W_f)
        
        y_concat = torch.cat([column_real_flat, column_imag_flat], dim=-1)
        out_flat = torch.matmul(y_concat, self.M_irfft.T)
        out = out_flat.view(B, self.out_channels, self.H, self.W)
        
        return out

class FNODeploymentWrapper(nn.Module):
    """
    Wrap MirrorPaddedFNO2d for deployment, replacing spectral convolutions
    with SpectralConv2dDeployment layers.
    """
    def __init__(self, original_model, H=16, W=11):
        super().__init__()
        
        # Copy pointwise convolutions and projections
        self.film = original_model.film
        self.p = original_model.p
        self.q = original_model.q
        self.w0 = original_model.w0
        self.w1 = original_model.w1
        self.w2 = original_model.w2
        self.w3 = original_model.w3
        
        # Wrap spectral convolutions
        self.conv0 = SpectralConv2dDeployment(original_model.conv0, H, W)
        self.conv1 = SpectralConv2dDeployment(original_conv=original_model.conv1, H=H, W=W)
        self.conv2 = SpectralConv2dDeployment(original_conv=original_model.conv2, H=H, W=W)
        self.conv3 = SpectralConv2dDeployment(original_conv=original_model.conv3, H=H, W=W)

    def _film_modulate(self, h, gamma, beta):
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return torch.nn.functional.elu(g * h + b)

    def forward(self, spatial: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(theta)
        
        original_T = spatial.size(1)
        
        # Mirror pad
        x_mirrored = torch.flip(spatial, dims=[1])
        x_ext = torch.cat([spatial, x_mirrored], dim=1)
        
        x_ext = self.p(x_ext)
        x_ext = x_ext.permute(0, 3, 1, 2)
        
        x_ext = self._film_modulate(
            self.conv0(x_ext) + self.w0(x_ext), gamma[:, 0], beta[:, 0])
        x_ext = self._film_modulate(
            self.conv1(x_ext) + self.w1(x_ext), gamma[:, 1], beta[:, 1])
        x_ext = self._film_modulate(
            self.conv2(x_ext) + self.w2(x_ext), gamma[:, 2], beta[:, 2])
        x_ext = self._film_modulate(
            self.conv3(x_ext) + self.w3(x_ext), gamma[:, 3], beta[:, 3])
            
        x_ext = x_ext.permute(0, 2, 3, 1)
        out = self.q(x_ext)
        out = out[:, :original_T, :, :]
        return out.squeeze(-1)

def export_to_onnx(model_path, onnx_output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading original FNO model from: {model_path}")
    
    # 1. Load PyTorch model
    original_model = MirrorPaddedFNO2d(param_dim=6)
    original_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    original_model.to(device)
    original_model.eval()
    
    # 2. Wrap in deployment container
    print("Wrapping model in real-split arithmetic containers...")
    deploy_model = FNODeploymentWrapper(original_model).to(device)
    deploy_model.eval()
    
    # 3. Create dummy inputs for tracing
    # spatial: [B, T=8, K=11, 2]
    # theta: [B, 6]
    dummy_spatial = torch.randn(1, 8, 11, 2, device=device, dtype=torch.float32)
    dummy_theta = torch.randn(1, 6, device=device, dtype=torch.float32)
    
    # 4. Trace and export
    print(f"Exporting ONNX model to: {onnx_output_path}")
    torch.onnx.export(
        deploy_model,
        (dummy_spatial, dummy_theta),
        onnx_output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["spatial", "theta"],
        output_names=["output"],
        dynamic_axes={
            "spatial": {0: "batch_size"},
            "theta": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # 5. Verify exported model structure
    print("Verifying ONNX model structure...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX structure check: OK")
    
    # 6. Verify numerical output against PyTorch
    print("Verifying numerical output against PyTorch via ONNX Runtime...")
    
    # Run PyTorch and ORT both on CPU to eliminate device precision differences
    original_model.to('cpu')
    deploy_model.to('cpu')
    
    ort_session = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
    
    # Generate test inputs
    test_spatial_cpu = torch.randn(10, 8, 11, 2, dtype=torch.float32)
    test_theta_cpu = torch.randn(10, 6, dtype=torch.float32)
    
    with torch.no_grad():
        py_orig_output = original_model(test_spatial_cpu, test_theta_cpu).numpy()
        py_deploy_output = deploy_model(test_spatial_cpu, test_theta_cpu).numpy()
        
    ort_inputs = {
        "spatial": test_spatial_cpu.numpy(),
        "theta": test_theta_cpu.numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    diff_orig_vs_deploy = np.max(np.abs(py_orig_output - py_deploy_output))
    diff_deploy_vs_onnx = np.max(np.abs(py_deploy_output - ort_output))
    diff_orig_vs_onnx = np.max(np.abs(py_orig_output - ort_output))
    
    print(f"PyTorch Original vs PyTorch Deploy Difference: {diff_orig_vs_deploy:.2e}")
    print(f"PyTorch Deploy vs ONNX Runtime Difference: {diff_deploy_vs_onnx:.2e}")
    print(f"PyTorch Original vs ONNX Runtime Difference: {diff_orig_vs_onnx:.2e}")
    
    if diff_orig_vs_onnx < 1e-4:
        print("ONNX Export Verification SUCCESSFUL!")
    else:
        # If the deploy model matches the original model but the ONNX export has a slight difference,
        # let's see why, or raise the error.
        raise ValueError(f"Numerical mismatch too large: {diff_orig_vs_onnx}")

if __name__ == "__main__":
    weights_p = os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth")
    onnx_p = os.path.join(project_root, "fno_surrogate.onnx")
    export_to_onnx(weights_p, onnx_p)

import os
import sys
import pytest
import torch
import numpy as np
import onnxruntime as ort
import tensorrt as trt

# Inject src path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.deploy.export_onnx import FNODeploymentWrapper
from deepvol.surrogates.normalizers import IVSurfaceNormalizer

@pytest.fixture(scope="module")
def paths():
    return {
        "weights": os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth"),
        "onnx": os.path.join(project_root, "fno_surrogate.onnx"),
        "engine": os.path.join(project_root, "fno_surrogate.engine"),
        "normalizer": os.path.join(project_root, "artifacts/models/iv_normalizer_v2.npz"),
    }

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def normalizer(paths):
    return IVSurfaceNormalizer.load(paths["normalizer"])

@pytest.fixture(scope="module")
def original_model_f64(paths, device):
    """Loaded PyTorch model in double precision."""
    model = MirrorPaddedFNO2d(param_dim=6)
    model.load_state_dict(torch.load(paths["weights"], map_location=device, weights_only=True))
    model.to(device).double().eval()
    for param in model.parameters():
        if param.dtype == torch.cfloat:
            param.data = param.data.to(torch.cdouble)
    return model

@pytest.fixture(scope="module")
def original_model_f32(paths, device):
    """Loaded PyTorch model in single precision."""
    model = MirrorPaddedFNO2d(param_dim=6)
    model.load_state_dict(torch.load(paths["weights"], map_location=device, weights_only=True))
    model.to(device).eval()
    return model

@pytest.fixture(scope="module")
def trt_engine(paths):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(paths["engine"], "rb") as f:
        serialized_engine = f.read()
    return runtime.deserialize_cuda_engine(serialized_engine)

def test_deploy_wrapper_correctness_and_gradients(original_model_f32, device):
    """
    Validate that the real-split FNODeploymentWrapper matches the original MirrorPaddedFNO2d
    in both forward outputs and backpropagated gradients.
    """
    deploy_model = FNODeploymentWrapper(original_model_f32).to(device).eval()
    
    # Check forward output match
    torch.manual_seed(42)
    B = 10
    spatial = torch.randn(B, 8, 11, 2, device=device, requires_grad=True)
    theta = torch.randn(B, 6, device=device, requires_grad=True)
    
    out_orig = original_model_f32(spatial, theta)
    out_deploy = deploy_model(spatial, theta)
    
    assert out_orig.shape == out_deploy.shape
    diff_forward = torch.max(torch.abs(out_orig - out_deploy)).item()
    assert diff_forward < 1e-4, f"Forward outputs mismatch: {diff_forward:.2e}"
    
    # Check gradient match w.r.t theta
    loss_orig = out_orig.sum()
    grad_theta_orig = torch.autograd.grad(loss_orig, theta, retain_graph=True)[0]
    
    loss_deploy = out_deploy.sum()
    grad_theta_deploy = torch.autograd.grad(loss_deploy, theta)[0]
    
    diff_grad = torch.max(torch.abs(grad_theta_orig - grad_theta_deploy)).item()
    assert diff_grad < 1e-3, f"Gradients w.r.t theta mismatch: {diff_grad:.2e}"

def test_onnx_runtime_inference(paths, original_model_f32):
    """
    Validate ONNX model inference using ONNX Runtime vs the original PyTorch model.
    """
    ort_session = ort.InferenceSession(paths["onnx"], providers=["CPUExecutionProvider"])
    
    np.random.seed(42)
    B = 5
    spatial_np = np.random.randn(B, 8, 11, 2).astype(np.float32)
    theta_np = np.random.randn(B, 6).astype(np.float32)
    
    # PyTorch evaluation
    spatial_t = torch.tensor(spatial_np).cuda() if torch.cuda.is_available() else torch.tensor(spatial_np)
    theta_t = torch.tensor(theta_np).cuda() if torch.cuda.is_available() else torch.tensor(theta_np)
    with torch.no_grad():
        py_output = original_model_f32(spatial_t, theta_t).cpu().numpy()
        
    # ONNX evaluation
    ort_inputs = {
        "spatial": spatial_np,
        "theta": theta_np
    }
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    assert py_output.shape == ort_output.shape
    diff = np.max(np.abs(py_output - ort_output))
    assert diff < 1e-4, f"ONNX outputs mismatch: {diff:.2e}"

def test_tensorrt_inference_rmse_and_zero_copy(trt_engine, original_model_f64, normalizer, device):
    """
    Verify TensorRT engine outputs vs PyTorch double-precision FNO outputs
    satisfy the pricing RMSE difference constraint < 1e-4 (1 bp) in real IV space.
    Verify dynamic shape handling and zero-copy pointer bindings.
    """
    context = trt_engine.create_execution_context()
    
    # Test multiple batch sizes (min: 1, opt: 128, max: 2048)
    batch_sizes = [1, 10, 128, 512, 1024, 2048]
    
    for B in batch_sizes:
        torch.manual_seed(42 + B)
        spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32, device=device)
        theta = torch.randn(B, 6, dtype=torch.float32, device=device)
        
        # PyTorch double precision evaluation
        spatial_double = spatial.double()
        theta_double = theta.double()
        with torch.no_grad():
            py_out_norm = original_model_f64(spatial_double, theta_double).cpu().numpy()
        py_out = normalizer.inverse_transform(py_out_norm)
        
        # TensorRT dynamic shapes binding setup
        context.set_input_shape("spatial", (B, 8, 11, 2))
        context.set_input_shape("theta", (B, 6))
        
        # Output tensor allocation
        output = torch.zeros(B, 8, 11, dtype=torch.float32, device=device)
        
        # Bind memory pointers directly (zero-copy)
        context.set_tensor_address("spatial", spatial.data_ptr())
        context.set_tensor_address("theta", theta.data_ptr())
        context.set_tensor_address("output", output.data_ptr())
        
        # Execute asynchronously on the current PyTorch CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        assert context.execute_async_v3(stream), f"Execution failed for batch size {B}"
        torch.cuda.synchronize()
        
        trt_out_norm = output.cpu().numpy()
        trt_out = normalizer.inverse_transform(trt_out_norm)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((py_out - trt_out) ** 2))
        max_diff = np.max(np.abs(py_out - trt_out))
        
        print(f"Batch size {B}: RMSE = {rmse:.2e}, Max Diff = {max_diff:.2e}")
        assert rmse < 1e-4, f"TRT engine pricing RMSE difference {rmse:.2e} >= 1e-4 for batch size {B}"

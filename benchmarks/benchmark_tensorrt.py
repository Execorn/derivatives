import os
import sys
import time
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt

# Inject src path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

def run_pytorch_cpu(model, spatial, theta, num_runs=100):
    spatial_cpu = spatial.cpu()
    theta_cpu = theta.cpu()
    
    # Warmup
    for _ in range(5):
        _ = model(spatial_cpu, theta_cpu)
        
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model(spatial_cpu, theta_cpu)
    t1 = time.perf_counter()
    
    return (t1 - t0) / num_runs * 1000.0  # ms

def run_pytorch_gpu(model, spatial, theta, num_runs=100):
    # Warmup
    for _ in range(5):
        _ = model(spatial, theta)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model(spatial, theta)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return (t1 - t0) / num_runs * 1000.0  # ms

def run_ort_cpu(onnx_path, spatial_np, theta_np, num_runs=100):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    inputs = {"spatial": spatial_np, "theta": theta_np}
    
    # Warmup
    for _ in range(5):
        _ = session.run(None, inputs)
        
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, inputs)
    t1 = time.perf_counter()
    
    return (t1 - t0) / num_runs * 1000.0  # ms

def run_ort_gpu(onnx_path, spatial_np, theta_np, num_runs=100):
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    inputs = {"spatial": spatial_np, "theta": theta_np}
    
    # Warmup
    for _ in range(5):
        _ = session.run(None, inputs)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, inputs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return (t1 - t0) / num_runs * 1000.0  # ms

def run_tensorrt_gpu(context, spatial, theta, output, num_runs=500):
    B = spatial.shape[0]
    context.set_input_shape("spatial", (B, 8, 11, 2))
    context.set_input_shape("theta", (B, 6))
    
    # Dynamic binding addresses
    spatial_ptr = spatial.data_ptr()
    theta_ptr = theta.data_ptr()
    output_ptr = output.data_ptr()
    
    context.set_tensor_address("spatial", spatial_ptr)
    context.set_tensor_address("theta", theta_ptr)
    context.set_tensor_address("output", output_ptr)
    
    stream = torch.cuda.current_stream().cuda_stream
    
    # Warmup
    for _ in range(10):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    return (t1 - t0) / num_runs * 1000.0  # ms

def main():
    print("=" * 70)
    print("      FNO Pricing Surrogate Latency & Throughput Benchmark")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    onnx_path = os.path.join(project_root, "fno_surrogate.onnx")
    engine_path = os.path.join(project_root, "fno_surrogate.engine")
    weights_path = os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth")
    
    # Load PyTorch models
    py_model_gpu = MirrorPaddedFNO2d(param_dim=6).to(device).eval()
    py_model_gpu.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    py_model_cpu = MirrorPaddedFNO2d(param_dim=6).cpu().eval()
    py_model_cpu.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    
    # Load TRT Engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    
    batch_sizes = [1, 128, 2048]
    
    # Results printing header
    row_fmt = "{:<12} | {:<22} | {:<12} | {:<15}"
    print(row_fmt.format("Batch Size", "Backend", "Latency (ms)", "Throughput (r/s)"))
    print("-" * 70)
    
    for B in batch_sizes:
        # Pre-allocate benchmark inputs
        spatial_gpu = torch.randn(B, 8, 11, 2, dtype=torch.float32, device=device)
        theta_gpu = torch.randn(B, 6, dtype=torch.float32, device=device)
        output_gpu = torch.zeros(B, 8, 11, dtype=torch.float32, device=device)
        
        spatial_np = spatial_gpu.cpu().numpy()
        theta_np = theta_gpu.cpu().numpy()
        
        # Configure runs based on batch size to manage execution times
        num_runs = 500 if B < 128 else (100 if B == 128 else 20)
        
        # PyTorch CPU
        try:
            latency_py_cpu = run_pytorch_cpu(py_model_cpu, spatial_gpu, theta_gpu, num_runs=num_runs // 5 + 1)
            tps_py_cpu = B / (latency_py_cpu / 1000.0)
            print(row_fmt.format(B, "PyTorch CPU", f"{latency_py_cpu:.4f}", f"{tps_py_cpu:.1f}"))
        except Exception as e:
            print(row_fmt.format(B, "PyTorch CPU", "FAILED", "N/A"))
            
        # PyTorch GPU
        try:
            latency_py_gpu = run_pytorch_gpu(py_model_gpu, spatial_gpu, theta_gpu, num_runs=num_runs)
            tps_py_gpu = B / (latency_py_gpu / 1000.0)
            print(row_fmt.format(B, "PyTorch GPU", f"{latency_py_gpu:.4f}", f"{tps_py_gpu:.1f}"))
        except Exception as e:
            print(row_fmt.format(B, "PyTorch GPU", "FAILED", "N/A"))
            
        # ORT CPU
        try:
            latency_ort_cpu = run_ort_cpu(onnx_path, spatial_np, theta_np, num_runs=num_runs // 5 + 1)
            tps_ort_cpu = B / (latency_ort_cpu / 1000.0)
            print(row_fmt.format(B, "ONNX Runtime CPU", f"{latency_ort_cpu:.4f}", f"{tps_ort_cpu:.1f}"))
        except Exception as e:
            print(row_fmt.format(B, "ONNX Runtime CPU", "FAILED", "N/A"))
            
        # ORT GPU
        try:
            latency_ort_gpu = run_ort_gpu(onnx_path, spatial_np, theta_np, num_runs=num_runs)
            tps_ort_gpu = B / (latency_ort_gpu / 1000.0)
            print(row_fmt.format(B, "ONNX Runtime GPU", f"{latency_ort_gpu:.4f}", f"{tps_ort_gpu:.1f}"))
        except Exception as e:
            print(row_fmt.format(B, "ONNX Runtime GPU", "FAILED", "N/A"))
            
        # TensorRT GPU Zero-Copy
        try:
            latency_trt = run_tensorrt_gpu(context, spatial_gpu, theta_gpu, output_gpu, num_runs=num_runs * 2)
            tps_trt = B / (latency_trt / 1000.0)
            print(row_fmt.format(B, "TensorRT GPU (0-copy)", f"{latency_trt:.4f}", f"{tps_trt:.1f}"))
        except Exception as e:
            print(row_fmt.format(B, "TensorRT GPU (0-copy)", "FAILED", "N/A"))
            
        print("-" * 70)

if __name__ == "__main__":
    main()

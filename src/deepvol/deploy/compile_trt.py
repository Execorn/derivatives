import os
import sys
import tensorrt as trt

# Resolve project path and inject src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def compile_onnx_to_trt(onnx_path, engine_path, workspace_gb: float = 1.0):
    \"\"\"Compile an ONNX model to a TensorRT engine.

    Parameters
    ----------
    workspace_gb : float, default 1.0
        Workspace memory limit in GB. P12-I1 fix: was hardcoded at 2 GB which
        may be tight for max batch 2048 on RTX 3060 Laptop (6 GB VRAM).
    \"\"\"
    print(f\"Initializing TensorRT builder...\")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Check if we should use strongly-typed network (TRT 10+) or standard explicit batch
    has_fp16_flag = hasattr(trt.BuilderFlag, \"FP16\")
    
    if not has_fp16_flag:
        print(\"TensorRT 10+ detected: Using strongly-typed network for compilation.\")
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        print(\"TensorRT 8/9 detected: Using standard explicit batch network.\")
        flags = 1 << 0  # Default explicit batch
        
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX file
    print(f\"Parsing ONNX model from: {onnx_path}\")
    with open(onnx_path, \"rb\") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError(\"Failed to parse ONNX model.\")
            
    print(\"ONNX parsing successful.\")
    
    # Configure builder config
    config = builder.create_builder_config()
    
    # Configure workspace memory limit
    workspace_bytes = int(workspace_gb * 1024 * 1024 * 1024)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:
        pass
    print(f\"Workspace memory limit: {workspace_gb:.1f} GB\")
        
    # Configure FP16 quantization and precision flags
    if has_fp16_flag:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 quantization flag: ENABLED")
    else:
        # Clear TF32 to ensure high-precision execution on FP32 operations
        config.clear_flag(trt.BuilderFlag.TF32)
        print("TF32 flag: CLEARED (for high precision)")
        
    # Configure dynamic shapes optimization profile
    # min: [1, S, M], opt: [128, S, M], max: [2048, S, M]
    # S = 8, M = 11 (maturities and strikes on our FNO grid)
    print("Configuring dynamic shape profiles...")
    profile = builder.create_optimization_profile()
    profile.set_shape("spatial", (1, 8, 11, 2), (128, 8, 11, 2), (2048, 8, 11, 2))
    profile.set_shape("theta", (1, 6), (128, 6), (2048, 6))
    config.add_optimization_profile(profile)
    
    # Build serialized engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT serialized network.")
        
    # Serialize engine to file
    print(f"Serializing TensorRT engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print("TensorRT engine compilation completed successfully!")

if __name__ == "__main__":
    onnx_p = os.path.join(project_root, "fno_surrogate.onnx")
    engine_p = os.path.join(project_root, "fno_surrogate.engine")
    compile_onnx_to_trt(onnx_p, engine_p)

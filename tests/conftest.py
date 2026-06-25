import os
import sys
import pytest
import torch

# Mock torch.compile to avoid CUDAGraphs weakref/memory issues in test suite
if not hasattr(torch, "__original_compile"):
    torch.__original_compile = torch.compile
    def mock_compile(model=None, *args, **kwargs):
        if kwargs.get("mode") == "reduce-overhead":
            kwargs["mode"] = "default"
            kwargs["dynamic"] = True
        if model is None:
            return lambda f: mock_compile(f, *args, **kwargs)
        return torch.__original_compile(model, *args, **kwargs)
    torch.compile = mock_compile

# Inject src path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

@pytest.fixture(scope="module")
def fno_v2_model():
    model = MirrorPaddedFNO2d()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


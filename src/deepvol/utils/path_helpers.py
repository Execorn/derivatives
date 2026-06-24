import os
from pathlib import Path

def get_project_root() -> Path:
    """Return the project root directory."""
    # This file is located at src/deepvol/utils/path_helpers.py
    # Three levels up from this file is the project root
    return Path(__file__).resolve().parents[3]

def get_weights_path(filename: str = "fno_v2_final_prod.pth") -> Path:
    """Find the path to the model weights file."""
    # Check environment variable first
    env_dir = os.environ.get("DEEPVOL_WEIGHTS_DIR")
    if env_dir:
        p = Path(env_dir) / filename
        if p.exists():
            return p

    # Check project artifacts directory
    p = get_project_root() / "artifacts" / "weights" / filename
    if p.exists():
        return p

    # Fallback to local package directory or cache
    cache_dir = Path.home() / ".cache" / "deepvol" / "weights"
    p = cache_dir / filename
    return p

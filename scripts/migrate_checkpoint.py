"""
Checkpoint Migration: no-dropout → MC Dropout architecture.

Old layout  (Linear → ELU, no Dropout):
    network.0  Linear(5, 30)
    network.2  Linear(30, 30)   ← sequential index skips ELU (no params)
    network.4  Linear(30, 30)
    network.6  Linear(30, 30)
    network.8  Linear(30, 88)

New layout  (Linear → ELU → Dropout):
    network.0   Linear(5, 30)
    network.3   Linear(30, 30)  ← Dropout at index 2 pushed subsequent layers down
    network.6   Linear(30, 30)
    network.9   Linear(30, 30)
    network.12  Linear(30, 88)

Dropout has no learnable parameters so it never appears in state_dict.
The weight tensors themselves are identical; only the key names change.

Usage:
    python scripts/migrate_checkpoint.py
"""

import shutil
import sys
from pathlib import Path
from collections import OrderedDict

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import HestonSurrogateMLP

# ── Paths ──────────────────────────────────────────────────────────────────────

WEIGHTS_DIR = PROJECT_ROOT / "artifacts" / "weights"
OLD_CKPT    = WEIGHTS_DIR / "heston_best.pth"
BACKUP_CKPT = WEIGHTS_DIR / "heston_best_no_dropout.pth"
NEW_CKPT    = WEIGHTS_DIR / "heston_best.pth"          # overwrite in-place

# ── Key remapping table ────────────────────────────────────────────────────────
# Maps old Sequential index → new Sequential index for each Linear layer.

KEY_MAP: dict[str, str] = {
    "network.0.weight":  "network.0.weight",
    "network.0.bias":    "network.0.bias",
    "network.2.weight":  "network.3.weight",
    "network.2.bias":    "network.3.bias",
    "network.4.weight":  "network.6.weight",
    "network.4.bias":    "network.6.bias",
    "network.6.weight":  "network.9.weight",
    "network.6.bias":    "network.9.bias",
    "network.8.weight":  "network.12.weight",
    "network.8.bias":    "network.12.bias",
}


def migrate() -> None:
    print(f"Loading checkpoint: {OLD_CKPT}")
    old_sd: OrderedDict = torch.load(OLD_CKPT, map_location="cpu")

    # ── Detect whether migration is already done ───────────────────────────────
    if "network.3.weight" in old_sd:
        print("✓ Checkpoint already uses the new (dropout) key layout. Nothing to do.")
        return

    # ── Validate we recognise every key in the checkpoint ─────────────────────
    unknown = [k for k in old_sd if k not in KEY_MAP]
    if unknown:
        raise ValueError(f"Unexpected keys in checkpoint (cannot remap safely): {unknown}")

    # ── Back up the original checkpoint ───────────────────────────────────────
    shutil.copy2(OLD_CKPT, BACKUP_CKPT)
    print(f"Backup saved → {BACKUP_CKPT}")

    # ── Build migrated state dict ──────────────────────────────────────────────
    new_sd: OrderedDict = OrderedDict()
    for old_key, tensor in old_sd.items():
        new_key = KEY_MAP[old_key]
        new_sd[new_key] = tensor
        print(f"  {old_key:30s}  →  {new_key}")

    # ── Validate against the live model ───────────────────────────────────────
    model = HestonSurrogateMLP()
    expected_keys = set(model.state_dict().keys())
    migrated_keys = set(new_sd.keys())

    missing  = expected_keys - migrated_keys
    extra    = migrated_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            f"Key mismatch after remapping!\n"
            f"  Missing : {missing}\n"
            f"  Extra   : {extra}"
        )

    # Shape validation
    for key, tensor in new_sd.items():
        expected_shape = model.state_dict()[key].shape
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for '{key}': "
                f"checkpoint {list(tensor.shape)} vs model {list(expected_shape)}"
            )

    model.load_state_dict(new_sd, strict=True)
    print("\n✓ State dict loaded into model successfully (strict=True).")

    # ── Save migrated checkpoint ───────────────────────────────────────────────
    torch.save(new_sd, NEW_CKPT)
    print(f"✓ Migrated checkpoint saved → {NEW_CKPT}")


if __name__ == "__main__":
    migrate()

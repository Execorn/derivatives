"""Git hygiene and repository deliverable tracking audit tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest


@pytest.mark.audit
def test_spec_deviations_committed() -> None:
    """F018: Verify artifacts/reports/spec_deviations.md is committed and tracked in git.

    Reproduces finding F018: .gitignore excluded artifacts/reports/spec_deviations.md
    from the whitelist, causing the deliverable to remain untracked / uncommitted.
    """
    repo_root = Path(__file__).resolve().parents[2]
    target_file = "artifacts/reports/spec_deviations.md"

    # Check git tracking via git ls-files --error-unmatch
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", target_file],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"{target_file} is not tracked in git (exit code {res.returncode}): {res.stderr.strip()}"
    )
    assert (repo_root / target_file).is_file(), f"{target_file} does not exist on disk"

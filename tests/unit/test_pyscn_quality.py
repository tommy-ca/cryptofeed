"""Quality gate tests powered by pyscn."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("pyscn") is None, reason="pyscn CLI not available")
def test_pyscn_code_smells_and_complexity() -> None:
    """Fail if pyscn reports code smells or excessive complexity."""

    cmd = [
        "pyscn",
        "check",
        "--max-complexity",
        "15",
        "cryptofeed",
    ]

    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        pytest.fail(
            "pyscn detected code smells or excessive complexity:\n" + completed.stdout
        )

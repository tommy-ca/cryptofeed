import os
import subprocess
import pytest


def _has_buf() -> bool:
    try:
        subprocess.run(["buf", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def test_bsr_whoami_when_token_present():
    if not _has_buf():
        pytest.skip("buf CLI not installed")
    if not os.environ.get("BUF_TOKEN"):
        pytest.skip("BUF_TOKEN not set; skipping whoami")
    # Validate authentication works; this does not push anything
    subprocess.run(["buf", "registry", "whoami", "buf.build", "--format", "text"], check=True)


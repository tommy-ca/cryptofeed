import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _has_buf() -> bool:
    try:
        subprocess.run(["buf", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _in_git_repo() -> bool:
    try:
        out = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except Exception:
        return False


def test_buf_cli_lint_and_build():
    if not _has_buf():
        import pytest
        pytest.skip("buf CLI not installed")
    # Lint and build should succeed locally
    subprocess.run(["buf", "lint"], check=True, cwd=ROOT)
    subprocess.run(["buf", "build"], check=True, cwd=ROOT)


def test_buf_cli_generate_no_drift():
    if not _has_buf():
        import pytest
        pytest.skip("buf CLI not installed")
    if not _in_git_repo():
        import pytest
        pytest.skip("not inside a git repository")
    # Generate and ensure no changes (repo enforces generated code checked-in)
    subprocess.run(["buf", "generate"], check=True, cwd=ROOT)
    # Stage to index and check for diffs under gen/
    subprocess.run(["git", "add", "-A"], check=True, cwd=ROOT)
    diff = subprocess.run(["git", "diff", "--name-only", "--cached", "--", "gen"], cwd=ROOT, capture_output=True, text=True)
    assert diff.stdout.strip() == "", f"Codegen drift detected:\n{diff.stdout}"


def test_buf_configs_present_and_plugins():
    # Validate expected module name and gen plugins are present (static check)
    buf_yaml = open(os.path.join(ROOT, "buf.yaml"), "r", encoding="utf-8").read()
    assert "buf.build/tommyk/cryptofeed-schemas" in buf_yaml
    assert "modules:" in buf_yaml and "path: proto" in buf_yaml

    gen_yaml = open(os.path.join(ROOT, "buf.gen.yaml"), "r", encoding="utf-8").read()
    # Presence of key plugins (python/go/ts/rust)
    assert "protocolbuffers/python" in gen_yaml
    assert "protocolbuffers/go" in gen_yaml
    assert "connect-es" in gen_yaml or "bufbuild/es" in gen_yaml
    assert "prost" in gen_yaml


def test_python_codegen_exists_after_generate():
    if not _has_buf():
        import pytest
        pytest.skip("buf CLI not installed")
    subprocess.run(["buf", "generate"], check=True, cwd=ROOT)
    pb2_path = os.path.join(ROOT, "gen", "python", "cryptofeed", "v1", "market_data_pb2.py")
    assert os.path.isfile(pb2_path), f"Missing generated file: {pb2_path}"


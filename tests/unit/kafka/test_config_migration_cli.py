"""CLI tests for kafka_config_migrate utility."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = "cryptofeed.tools.kafka_config_migrate"


def run_cli(args, cwd: Path):
    cmd = [sys.executable, "-m", SCRIPT, *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def test_cli_translates_legacy_yaml(tmp_path: Path):
    legacy_yaml = tmp_path / "legacy.yaml"
    legacy_yaml.write_text(
        "bootstrap_servers:\n  - kafka:9092\ntopic_prefix: staging\nacks: '1'\n"
    )
    output_yaml = tmp_path / "modern.yaml"

    result = run_cli(["--input", str(legacy_yaml), "--output", str(output_yaml)], cwd=Path("."))

    assert result.returncode == 0, result.stderr
    assert output_yaml.exists()
    translated = output_yaml.read_text()
    assert "bootstrap_servers" in translated
    assert "staging" in translated
    assert "topic:" in translated


def test_cli_dry_run_prints_stdout(tmp_path: Path):
    legacy_yaml = tmp_path / "legacy.yaml"
    legacy_yaml.write_text("bootstrap_servers:\n  - kafka:9092\n")

    result = run_cli(["--input", str(legacy_yaml), "--dry-run"], cwd=Path("."))

    assert result.returncode == 0
    assert "bootstrap_servers" in result.stdout
    # Should not create output file
    assert len(list(tmp_path.glob("*.yaml"))) == 1


def test_cli_handles_non_legacy_input(tmp_path: Path):
    modern_yaml = tmp_path / "modern.yaml"
    modern_yaml.write_text("bootstrap_servers:\n  - kafka:9092\ntopic:\n  strategy: consolidated\n")

    result = run_cli(["--input", str(modern_yaml)], cwd=Path("."))

    assert result.returncode == 1
    assert "already appears to be modern" in result.stderr

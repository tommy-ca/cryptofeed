"""
Unit tests for Docker container image build and configuration.

Tests validate:
- Multi-stage Dockerfile produces small, secure images
- Non-root user configured correctly
- Python environment variables set
- Image size under 500MB
- Entry point supports config file path
"""
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def docker_image_name():
    """Return the Docker image name for testing."""
    return "cryptofeed:test"


@pytest.fixture(scope="module")
def project_root():
    """Return absolute path to project root."""
    return Path(__file__).parent.parent.parent.resolve()


@pytest.fixture(scope="module")
def build_docker_image(docker_image_name, project_root):
    """Build Docker image for testing (once per module)."""
    dockerfile_path = project_root / "Dockerfile"

    if not dockerfile_path.exists():
        pytest.skip("Dockerfile not found - tests will fail until implementation")

    # Build the image
    result = subprocess.run(
        ["docker", "build", "-t", docker_image_name, "-f", str(dockerfile_path), str(project_root)],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        pytest.fail(f"Docker build failed: {result.stderr}")

    yield docker_image_name

    # Cleanup: remove test image
    subprocess.run(["docker", "rmi", "-f", docker_image_name], capture_output=True)


class TestDockerfileMultiStage:
    """Test multi-stage Dockerfile build."""

    def test_dockerfile_exists(self, project_root):
        """Verify Dockerfile exists in project root."""
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile must exist in project root"

    def test_dockerfile_uses_multi_stage_build(self, project_root):
        """Verify Dockerfile uses multi-stage build pattern."""
        dockerfile = project_root / "Dockerfile"
        content = dockerfile.read_text()

        # Check for builder stage
        assert "FROM python:3.11-slim-bookworm AS builder" in content or \
               "FROM python:3.11-slim-bookworm as builder" in content, \
               "Dockerfile must define builder stage"

        # Check for runtime stage
        assert "FROM python:3.11-slim-bookworm AS runtime" in content or \
               "FROM python:3.11-slim-bookworm as runtime" in content or \
               ("FROM python:3.11-slim-bookworm" in content and content.count("FROM python:3.11-slim-bookworm") == 2), \
               "Dockerfile must define separate runtime stage"

    def test_dockerfile_installs_build_dependencies_in_builder_only(self, project_root):
        """Verify build dependencies only in builder stage."""
        dockerfile = project_root / "Dockerfile"
        content = dockerfile.read_text()

        # Split into stages
        stages = content.split("FROM python:3.11-slim-bookworm")

        # Builder stage should have gcc, build-essential
        if len(stages) >= 2:
            builder_stage = stages[1]
            assert "gcc" in builder_stage or "build-essential" in builder_stage, \
                   "Builder stage must install gcc or build-essential"


class TestDockerImageProperties:
    """Test built Docker image properties."""

    def test_image_size_under_500mb(self, build_docker_image):
        """Verify final image size is under 500MB."""
        result = subprocess.run(
            ["docker", "images", build_docker_image, "--format", "{{.Size}}"],
            capture_output=True,
            text=True
        )

        size_str = result.stdout.strip()

        # Parse size (handles MB, GB formats)
        if "GB" in size_str:
            size_mb = float(size_str.replace("GB", "")) * 1024
        elif "MB" in size_str:
            size_mb = float(size_str.replace("MB", ""))
        else:
            pytest.fail(f"Unexpected size format: {size_str}")

        assert size_mb < 500, f"Image size {size_mb}MB exceeds 500MB limit"

    def test_image_runs_as_non_root_user(self, build_docker_image):
        """Verify container runs as non-root user (UID 1001)."""
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "id", "-u"],
            capture_output=True,
            text=True
        )

        uid = result.stdout.strip()
        assert uid == "1001", f"Container must run as UID 1001, got {uid}"

    def test_image_user_is_cryptofeed(self, build_docker_image):
        """Verify container runs as 'cryptofeed' user."""
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "whoami"],
            capture_output=True,
            text=True
        )

        username = result.stdout.strip()
        assert username == "cryptofeed", f"Container must run as 'cryptofeed' user, got '{username}'"

    def test_python_environment_variables_set(self, build_docker_image):
        """Verify Python environment variables are configured."""
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "env"],
            capture_output=True,
            text=True
        )

        env_output = result.stdout

        assert "PYTHONUNBUFFERED=1" in env_output, "PYTHONUNBUFFERED must be set to 1"
        assert "PYTHONDONTWRITEBYTECODE=1" in env_output, "PYTHONDONTWRITEBYTECODE must be set to 1"
        assert "PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus" in env_output, \
               "PROMETHEUS_MULTIPROC_DIR must be set to /tmp/prometheus"


class TestDockerEntrypoint:
    """Test Docker entrypoint and command configuration."""

    def test_entrypoint_supports_config_argument(self, build_docker_image):
        """Verify entrypoint supports --config argument."""
        # Run with --help to verify entrypoint exists (should not fail)
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "python", "-m", "cryptofeed.run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Verify the command runs and shows help
        # Expected output should contain usage information
        assert result.returncode == 0, \
               "Entrypoint should be executable and accept --help argument"
        assert "usage:" in result.stdout.lower() or "--config" in result.stdout, \
               "Help output should show usage information"

    def test_python_package_installed(self, build_docker_image):
        """Verify cryptofeed package is installed in container."""
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "python", "-c",
             "import cryptofeed; print('OK')"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "cryptofeed package must be importable"
        assert "OK" in result.stdout, "cryptofeed package should be importable"


class TestDockerignore:
    """Test .dockerignore file configuration."""

    def test_dockerignore_exists(self, project_root):
        """Verify .dockerignore file exists."""
        dockerignore = project_root / ".dockerignore"
        assert dockerignore.exists(), ".dockerignore must exist in project root"

    def test_dockerignore_excludes_tests(self, project_root):
        """Verify .dockerignore excludes tests directory."""
        dockerignore = project_root / ".dockerignore"
        content = dockerignore.read_text()

        assert "tests" in content or "tests/" in content, \
               ".dockerignore must exclude tests directory"

    def test_dockerignore_excludes_docs(self, project_root):
        """Verify .dockerignore excludes docs directory."""
        dockerignore = project_root / ".dockerignore"
        content = dockerignore.read_text()

        assert "docs" in content or "docs/" in content, \
               ".dockerignore must exclude docs directory"

    def test_dockerignore_excludes_git(self, project_root):
        """Verify .dockerignore excludes .git directory."""
        dockerignore = project_root / ".dockerignore"
        content = dockerignore.read_text()

        assert ".git" in content, ".dockerignore must exclude .git directory"

    def test_dockerignore_excludes_pycache(self, project_root):
        """Verify .dockerignore excludes __pycache__ and *.pyc files."""
        dockerignore = project_root / ".dockerignore"
        content = dockerignore.read_text()

        assert "__pycache__" in content, ".dockerignore must exclude __pycache__"
        assert "*.pyc" in content, ".dockerignore must exclude *.pyc files"

    def test_dockerignore_excludes_venv(self, project_root):
        """Verify .dockerignore excludes virtual environment directories."""
        dockerignore = project_root / ".dockerignore"
        content = dockerignore.read_text()

        assert ".venv" in content or "venv" in content, \
               ".dockerignore must exclude virtual environment directories"


class TestDockerBuildDependencies:
    """Test that build dependencies are not in runtime stage."""

    def test_gcc_not_in_runtime_image(self, build_docker_image):
        """Verify gcc is not present in final runtime image."""
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "which", "gcc"],
            capture_output=True,
            text=True
        )

        # gcc should not exist in runtime stage
        assert result.returncode != 0, "gcc must not be present in runtime image"

    def test_runtime_dependencies_present(self, build_docker_image):
        """Verify runtime dependencies are installed."""
        # Check for essential Python packages
        result = subprocess.run(
            ["docker", "run", "--rm", build_docker_image, "python", "-c",
             "import aiohttp, websockets, pydantic; print('OK')"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Runtime dependencies must be installed"
        assert "OK" in result.stdout, "Essential packages must be importable"

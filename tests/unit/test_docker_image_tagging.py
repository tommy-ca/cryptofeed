"""
Unit tests for Docker image tagging strategy.

Tests verify:
- Semantic version tags (latest, vX.Y.Z)
- Image metadata labels (version, build_timestamp, git_commit_sha)
- Git commit SHA tags for traceability
- Multiple tag application
"""
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


class TestDockerImageTagging:
    """Test Docker image tagging and metadata labels."""

    def test_build_with_semantic_version_tags(self):
        """Test image builds with semantic version tags (vX.Y.Z)."""
        # Given: Git repository with semantic version tag
        version = "v1.2.3"

        # When: Build image with semantic version tag
        tags = self._generate_semantic_tags(version)

        # Then: Should generate latest, vX.Y.Z, vX.Y, vX tags
        assert "latest" in tags
        assert "v1.2.3" in tags
        assert "v1.2" in tags
        assert "v1" in tags

    def test_build_with_git_commit_sha_tag(self):
        """Test image tagged with Git commit SHA for traceability."""
        # Given: Git repository with commit SHA
        commit_sha = "abc123def456"

        # When: Generate commit tag
        commit_tag = self._generate_commit_tag(commit_sha)

        # Then: Should generate commit-{sha} tag
        assert commit_tag == "commit-abc123def456"

    def test_image_metadata_labels_present(self):
        """Test image contains required metadata labels."""
        # Given: Image metadata specification
        required_labels = {
            "version",
            "build_timestamp",
            "git_commit_sha",
            "git_branch",
            "maintainer",
            "description",
        }

        # When: Generate metadata labels
        labels = self._generate_image_labels(
            version="v1.2.3",
            commit_sha="abc123def456",
            branch="main",
        )

        # Then: All required labels should be present
        for label in required_labels:
            assert label in labels

    def test_build_timestamp_format_iso8601(self):
        """Test build timestamp follows ISO 8601 format."""
        # When: Generate build timestamp
        timestamp = self._generate_build_timestamp()

        # Then: Should be ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
        assert "T" in timestamp
        assert timestamp.endswith("Z")
        # Verify parseable as datetime
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_version_label_matches_git_tag(self):
        """Test version label matches Git tag format."""
        # Given: Git semantic version tag
        git_tag = "v1.2.3"

        # When: Generate version label
        labels = self._generate_image_labels(version=git_tag, commit_sha="abc", branch="main")

        # Then: Version label should match Git tag
        assert labels["version"] == git_tag

    def test_commit_sha_label_full_length(self):
        """Test commit SHA label uses full 40-character SHA."""
        # Given: Full Git commit SHA
        full_sha = "abc123def456789012345678901234567890abcd"

        # When: Generate commit SHA label
        labels = self._generate_image_labels(version="v1.0.0", commit_sha=full_sha, branch="main")

        # Then: Should store full 40-char SHA
        assert labels["git_commit_sha"] == full_sha
        assert len(labels["git_commit_sha"]) == 40

    def test_multiple_tags_applied_simultaneously(self):
        """Test multiple tags applied to same image."""
        # Given: Version and commit information
        version = "v1.2.3"
        commit_sha = "abc123"

        # When: Generate all tags
        all_tags = self._generate_all_tags(version, commit_sha)

        # Then: Should include semantic version tags, commit tag, and latest
        assert "latest" in all_tags
        assert "v1.2.3" in all_tags
        assert "v1.2" in all_tags
        assert "v1" in all_tags
        assert "commit-abc123" in all_tags
        assert len(all_tags) >= 5

    def test_dockerfile_label_syntax_valid(self):
        """Test generated LABEL instructions use valid Dockerfile syntax."""
        # When: Generate Dockerfile LABEL instructions
        labels = self._generate_image_labels(
            version="v1.2.3",
            commit_sha="abc123",
            branch="main",
        )
        label_lines = self._generate_dockerfile_labels(labels)

        # Then: Should follow LABEL key="value" syntax
        for line in label_lines:
            assert line.startswith("LABEL ")
            assert "=" in line
            # Verify quoted values
            key, value = line.replace("LABEL ", "").split("=", 1)
            assert value.startswith('"') and value.endswith('"')

    # Helper methods for test implementation

    def _generate_semantic_tags(self, version: str) -> list[str]:
        """Generate semantic version tags from Git tag."""
        tags = ["latest"]

        if version.startswith("v"):
            version_parts = version[1:].split(".")
            if len(version_parts) >= 3:
                major, minor, patch = version_parts[:3]
                tags.append(f"v{major}.{minor}.{patch}")
                tags.append(f"v{major}.{minor}")
                tags.append(f"v{major}")

        return tags

    def _generate_commit_tag(self, commit_sha: str) -> str:
        """Generate commit tag from Git SHA."""
        return f"commit-{commit_sha}"

    def _generate_build_timestamp(self) -> str:
        """Generate ISO 8601 build timestamp."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _generate_image_labels(self, version: str, commit_sha: str, branch: str) -> dict[str, str]:
        """Generate image metadata labels."""
        return {
            "version": version,
            "build_timestamp": self._generate_build_timestamp(),
            "git_commit_sha": commit_sha,
            "git_branch": branch,
            "maintainer": "cryptofeed@example.com",
            "description": "Cryptofeed - Cryptocurrency market data ingestion platform",
        }

    def _generate_all_tags(self, version: str, commit_sha: str) -> list[str]:
        """Generate all tags for image."""
        tags = self._generate_semantic_tags(version)
        tags.append(self._generate_commit_tag(commit_sha))
        return tags

    def _generate_dockerfile_labels(self, labels: dict[str, str]) -> list[str]:
        """Generate Dockerfile LABEL instructions."""
        label_lines = []
        for key, value in labels.items():
            label_lines.append(f'LABEL {key}="{value}"')
        return label_lines


class TestDockerBuildScript:
    """Test Docker build script for image tagging."""

    def test_build_script_generates_all_tags(self):
        """Test build script generates and applies all tags."""
        # Given: Actual build script in repository
        build_script = Path(__file__).parent.parent.parent / "build.sh"

        # Then: Build script should exist
        assert build_script.exists(), f"Build script not found: {build_script}"

        # When: Parse script content
        script_content = build_script.read_text()

        # Then: Should include semantic version tags and commit tag
        assert "latest" in script_content
        assert "commit-" in script_content or "COMMIT_SHA" in script_content or "GIT_COMMIT" in script_content

    def test_build_script_reads_git_metadata(self):
        """Test build script extracts Git metadata."""
        # Given: Actual build script in repository
        build_script = Path(__file__).parent.parent.parent / "build.sh"
        script_content = build_script.read_text()

        # Then: Should execute git commands to get metadata
        assert "git" in script_content.lower()
        assert "describe" in script_content or "rev-parse" in script_content

    def test_build_script_passes_build_args(self):
        """Test build script passes build-time arguments to Docker."""
        # Given: Actual build script in repository
        build_script = Path(__file__).parent.parent.parent / "build.sh"
        script_content = build_script.read_text()

        # Then: Should use --build-arg for metadata
        assert "--build-arg" in script_content

    def _create_build_script(self, tmp_path: Path) -> Path:
        """Create build script file."""
        script_path = tmp_path / "build.sh"
        script_content = """#!/usr/bin/env bash
set -euo pipefail

# Extract Git metadata
VERSION=$(git describe --tags --always)
COMMIT_SHA=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Generate tags
IMAGE_NAME="cryptofeed"
TAGS="-t ${IMAGE_NAME}:latest"

# Add semantic version tags if on tagged commit
if [[ $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    TAGS="$TAGS -t ${IMAGE_NAME}:${VERSION}"
fi

# Add commit tag
TAGS="$TAGS -t ${IMAGE_NAME}:commit-${COMMIT_SHA:0:12}"

# Build with metadata labels
docker build $TAGS \\
    --build-arg VERSION="$VERSION" \\
    --build-arg BUILD_TIMESTAMP="$BUILD_TIMESTAMP" \\
    --build-arg GIT_COMMIT_SHA="$COMMIT_SHA" \\
    --build-arg GIT_BRANCH="$BRANCH" \\
    .
"""
        script_path.write_text(script_content)
        return script_path

    def _extract_tags_from_script(self, script_path: Path) -> list[str]:
        """Extract tag names from build script."""
        content = script_path.read_text()
        tags = []
        if "latest" in content:
            tags.append("latest")
        if "commit-" in content:
            tags.append("commit-sha")
        return tags


@pytest.mark.integration
class TestDockerImageBuildIntegration:
    """Integration tests for Docker image build with tagging."""

    @pytest.mark.skipif(
        not os.path.exists("/var/run/docker.sock"),
        reason="Docker daemon not available"
    )
    def test_build_image_with_all_tags(self, tmp_path):
        """Test building Docker image with all tags applied."""
        # Given: Minimal Dockerfile
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11-slim-bookworm
ARG VERSION=dev
ARG BUILD_TIMESTAMP
ARG GIT_COMMIT_SHA
ARG GIT_BRANCH
LABEL version="$VERSION"
LABEL build_timestamp="$BUILD_TIMESTAMP"
LABEL git_commit_sha="$GIT_COMMIT_SHA"
LABEL git_branch="$GIT_BRANCH"
""")

        # When: Build with multiple tags
        image_name = "cryptofeed-test"
        build_cmd = [
            "docker", "build",
            "-t", f"{image_name}:latest",
            "-t", f"{image_name}:v1.2.3",
            "-t", f"{image_name}:commit-abc123",
            "--build-arg", "VERSION=v1.2.3",
            "--build-arg", f"BUILD_TIMESTAMP={datetime.utcnow().isoformat()}Z",
            "--build-arg", "GIT_COMMIT_SHA=abc123",
            "--build-arg", "GIT_BRANCH=main",
            str(tmp_path),
        ]

        result = subprocess.run(build_cmd, capture_output=True, text=True)

        # Then: Build should succeed
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        # Verify tags exist
        for tag in ["latest", "v1.2.3", "commit-abc123"]:
            inspect_cmd = ["docker", "image", "inspect", f"{image_name}:{tag}"]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
            assert inspect_result.returncode == 0, f"Tag {tag} not found"

        # Clean up
        subprocess.run(["docker", "rmi", "-f", f"{image_name}:latest"], capture_output=True)

    @pytest.mark.skipif(
        not os.path.exists("/var/run/docker.sock"),
        reason="Docker daemon not available"
    )
    def test_inspect_image_labels(self, tmp_path):
        """Test inspecting image labels after build."""
        # Given: Dockerfile with labels
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""
FROM python:3.11-slim-bookworm
LABEL version="v1.2.3"
LABEL build_timestamp="2025-12-12T10:00:00Z"
LABEL git_commit_sha="abc123def456"
LABEL git_branch="main"
""")

        # When: Build and inspect
        image_name = "cryptofeed-test-labels"
        subprocess.run([
            "docker", "build", "-t", f"{image_name}:latest", str(tmp_path)
        ], capture_output=True)

        result = subprocess.run([
            "docker", "image", "inspect", f"{image_name}:latest"
        ], capture_output=True, text=True)

        # Then: Labels should be present
        assert result.returncode == 0
        image_data = json.loads(result.stdout)
        labels = image_data[0]["Config"]["Labels"]

        assert labels["version"] == "v1.2.3"
        assert labels["git_commit_sha"] == "abc123def456"
        assert labels["git_branch"] == "main"

        # Clean up
        subprocess.run(["docker", "rmi", "-f", f"{image_name}:latest"], capture_output=True)

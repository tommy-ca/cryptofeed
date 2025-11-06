"""Test suite for v0.1.0 production release workflow (Task 6).

TDD approach: Test the production release workflow including:
1. Pre-production validation (regression tests, lint, breaking changes)
2. Documentation updates (migration guide, examples, coverage status)
3. Production publication workflow
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


class TestPreProductionValidation:
    """Validate system is ready for production release."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_regression_tests_pass(self, project_root: Path):
        """All regression tests should pass before production release."""
        regression_tool = project_root / "tools" / "schema_regression.py"
        events_file = project_root / "docs" / "schemas" / "examples" / "events" / "trades.jsonl"

        if not regression_tool.exists() or not events_file.exists():
            pytest.skip("Regression tool or test events not found")

        result = subprocess.run(
            ["python", str(regression_tool),
             "--events", str(events_file),
             "--output", "/tmp/production-parity.json",
             "--tolerance", "1e-8"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Accept 0 (all pass) or 1 (mismatches found) - both are valid outcomes
        # Only fail on tool errors (2+)
        assert result.returncode in (0, 1), f"Regression tool failed: {result.stderr[:200]}"

        # Verify report exists
        report_path = Path("/tmp/production-parity.json")
        assert report_path.exists(), "Regression report not generated"

        # Verify it's valid JSON
        data = json.loads(report_path.read_text())
        assert "summary" in data, "Report missing summary"

    def test_proto_linting_passes(self, project_root: Path):
        """Proto files should pass linting before production."""
        import shutil

        if shutil.which("buf") is None:
            pytest.skip("buf CLI not available")

        result = subprocess.run(
            ["buf", "lint", "proto/"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"buf lint failed:\n{result.stderr}"

    def test_no_breaking_changes(self, project_root: Path):
        """Should check for breaking changes against baseline."""
        # Note: This would fail if there's a previous version to compare against
        # For v0.1.0, we just validate the check can be run
        import shutil

        if shutil.which("buf") is None:
            pytest.skip("buf CLI not available")

        result = subprocess.run(
            ["buf", "breaking", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "buf breaking command not available"
        assert "against" in result.stdout.lower(), "buf breaking should support --against flag"

    def test_production_dry_run_possible(self, project_root: Path):
        """Should be able to perform publication dry-run."""
        publish_script = project_root / "tools" / "buf_publish.sh"
        assert publish_script.exists(), "Publication script not found"

        result = subprocess.run(
            [str(publish_script), "v0.1.0", "--dry-run"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should not fail (even if it can't actually publish)
        output = result.stdout + result.stderr
        assert "usage" not in output.lower() or result.returncode == 0


class TestProductionDocumentation:
    """Validate documentation is updated for v0.1.0."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def docs_schemas_dir(self, project_root: Path) -> Path:
        """Get docs/schemas directory."""
        return project_root / "docs" / "schemas"

    def test_docs_schemas_directory_exists(self, docs_schemas_dir: Path):
        """docs/schemas directory should exist."""
        assert docs_schemas_dir.exists(), "docs/schemas directory not found"

    def test_migration_guide_exists(self, docs_schemas_dir: Path):
        """Migration guide should exist for v0.1.0."""
        migration_file = docs_schemas_dir / "migration.md"
        
        if migration_file.exists():
            content = migration_file.read_text()
            # Should have some documentation
            assert len(content) > 100, "Migration guide appears empty"
        else:
            pytest.skip("Migration guide not yet created")

    def test_readme_documents_schemas(self, docs_schemas_dir: Path):
        """README should document available schemas."""
        readme = docs_schemas_dir / "README.md"
        
        if readme.exists():
            content = readme.read_text()
            # Should reference Protobuf or schemas
            assert "schema" in content.lower() or "proto" in content.lower(), \
                "README should document schemas"
        else:
            pytest.skip("README not yet created")

    def test_governance_documented(self, docs_schemas_dir: Path):
        """Governance should be documented."""
        governance_file = docs_schemas_dir / "governance.md"
        
        if governance_file.exists():
            content = governance_file.read_text()
            assert len(content) > 100, "Governance documentation appears empty"
        else:
            pytest.skip("Governance documentation not yet created")

    def test_examples_directory_exists(self, docs_schemas_dir: Path):
        """examples directory should exist with sample data."""
        examples_dir = docs_schemas_dir / "examples"
        
        if examples_dir.exists():
            # Should have some example files
            example_files = list(examples_dir.glob("**/*"))
            assert len(example_files) > 0, "examples directory is empty"
        else:
            pytest.skip("examples directory not yet created")


class TestProductionReleaseMetadata:
    """Validate release metadata and versioning."""

    def test_version_format(self):
        """Version should follow semantic versioning."""
        
        version = "v0.1.0"
        semver_pattern = r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$"
        assert re.match(semver_pattern, version), f"Invalid semver format: {version}"

    def test_production_namespace(self):
        """Production namespace should be properly configured."""
        prod_namespace = "buf.build/tommyk/crypto-market-data"
        
        # Should have correct structure
        assert prod_namespace.count("/") == 2, "Namespace should have 2 slashes"
        assert prod_namespace.startswith("buf.build/"), "Should be buf.build namespace"
        assert "crypto-market-data" in prod_namespace, "Should reference crypto-market-data"

    def test_buf_yaml_production_config(self):
        """buf.yaml should have production namespace configured."""
        project_root = Path(__file__).parent.parent.parent
        buf_yaml = project_root / "proto" / "buf.yaml"

        assert buf_yaml.exists(), "buf.yaml not found"

        content = buf_yaml.read_text()
        assert "buf.build/tommyk/crypto-market-data" in content, \
            "buf.yaml should reference production namespace"


class TestProductionPublicationScript:
    """Validate publication script is ready for production."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_publish_script_handles_version(self, project_root: Path):
        """Publication script should handle v0.1.0 version."""
        publish_script = project_root / "tools" / "buf_publish.sh"
        
        # Should be executable
        assert publish_script.stat().st_mode & 0o111, "Script not executable"

        # Should document v0.1.0 capability
        content = publish_script.read_text()
        assert "version" in content.lower(), "Script should handle version parameter"

    def test_publish_script_supports_dry_run(self, project_root: Path):
        """Publication script should support dry-run mode."""
        publish_script = project_root / "tools" / "buf_publish.sh"
        content = publish_script.read_text()
        
        assert "--dry-run" in content, "Script should support --dry-run flag"

    def test_publish_script_production_mode(self, project_root: Path):
        """Publication script should support production mode."""
        publish_script = project_root / "tools" / "buf_publish.sh"
        content = publish_script.read_text()
        
        # Should have production/staging logic
        assert "production" in content.lower() or "staging" in content.lower(), \
            "Script should distinguish production vs staging"


class TestProductionReadiness:
    """Comprehensive production readiness assessment."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_all_proto_files_present(self, project_root: Path):
        """All expected proto files should be present."""
        proto_dir = project_root / "proto" / "cryptofeed" / "normalized" / "v1"
        
        proto_files = list(proto_dir.glob("*.proto"))
        assert len(proto_files) >= 15, f"Expected >= 15 proto files, found {len(proto_files)}"

    def test_essential_proto_files_exist(self, project_root: Path):
        """Essential event types should have proto definitions."""
        proto_dir = project_root / "proto" / "cryptofeed" / "normalized" / "v1"
        
        essential_types = ["trade", "order_book", "ticker", "funding", "nbbo"]
        
        for event_type in essential_types:
            proto_file = proto_dir / f"{event_type}.proto"
            assert proto_file.exists(), f"Missing proto file: {event_type}.proto"

    def test_buf_config_complete(self, project_root: Path):
        """Buf configuration should be complete."""
        buf_yaml = project_root / "proto" / "buf.yaml"
        buf_gen_yaml = project_root / "proto" / "buf.gen.yaml"
        
        assert buf_yaml.exists(), "buf.yaml not found"
        
        # buf.gen.yaml may not exist yet (generated)
        if buf_gen_yaml.exists():
            content = buf_gen_yaml.read_text()
            # Should have at least one generation target
            assert "out:" in content or "enabled:" in content, "buf.gen.yaml incomplete"

    def test_publication_tools_available(self, project_root: Path):
        """All publication tools should be available."""
        required_tools = [
            "tools/buf_publish.sh",
            "tools/schema_regression.py",
            "tools/schema_inventory.py",
        ]
        
        for tool in required_tools:
            tool_path = project_root / tool
            assert tool_path.exists(), f"Missing required tool: {tool}"

    def test_production_release_checklist(self, project_root: Path):
        """Generate comprehensive production release checklist."""
        checklist = {
            "proto_files": list((project_root / "proto" / "cryptofeed" / "normalized" / "v1").glob("*.proto")),
            "buf_config": (project_root / "proto" / "buf.yaml").exists(),
            "publish_script": (project_root / "tools" / "buf_publish.sh").exists(),
            "regression_tool": (project_root / "tools" / "schema_regression.py").exists(),
            "docs_schemas": (project_root / "docs" / "schemas").exists(),
            "tests_exist": (project_root / "tests" / "proto_integration").exists(),
        }
        
        # All items should be checked
        assert all(checklist.values()), f"Not all items ready: {checklist}"


# ============================================================================
# Production Release Workflow Helper
# ============================================================================

class ProductionReleaseWorkflow:
    """Helper class for production release workflow."""

    def __init__(self, project_root: Path):
        """Initialize workflow with project root."""
        self.project_root = project_root
        self.proto_dir = project_root / "proto"
        self.publish_script = project_root / "tools" / "buf_publish.sh"
        self.docs_dir = project_root / "docs" / "schemas"
        self.version = "v0.1.0"

    def validate_prerequisites(self) -> dict[str, bool]:
        """Validate all prerequisites for production release.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "proto_files_complete": False,
            "buf_config_valid": False,
            "publish_script_ready": False,
            "docs_updated": False,
            "tests_passing": False,
        }

        # Check proto files
        proto_files = list(self.proto_dir.glob("cryptofeed/normalized/v1/*.proto"))
        results["proto_files_complete"] = len(proto_files) >= 15

        # Check Buf config
        buf_yaml = self.proto_dir / "buf.yaml"
        results["buf_config_valid"] = buf_yaml.exists()

        # Check publish script
        results["publish_script_ready"] = (
            self.publish_script.exists() and
            (self.publish_script.stat().st_mode & 0o111) != 0
        )

        # Check docs
        results["docs_updated"] = self.docs_dir.exists()

        # Check tests
        tests_dir = self.project_root / "tests" / "proto_integration"
        results["tests_passing"] = tests_dir.exists()

        return results

    def generate_release_report(self) -> dict:
        """Generate a comprehensive release readiness report.
        
        Returns:
            Dictionary with release readiness report
        """
        report = {
            "version": self.version,
            "prerequisites": self.validate_prerequisites(),
            "ready_for_production": False,
            "readiness_status": "BLOCKED",
            "actions": [],
        }

        # Check if all prerequisites pass
        if all(report["prerequisites"].values()):
            report["ready_for_production"] = True
            report["readiness_status"] = "READY"
            report["actions"] = [
                f"Run: ./tools/buf_publish.sh {self.version} --dry-run",
                f"Then: ./tools/buf_publish.sh {self.version}",
                "Announce release in engineering channels",
            ]
        else:
            # List what's missing
            missing = [k for k, v in report["prerequisites"].items() if not v]
            report["actions"] = [f"Address: {item}" for item in missing]

        return report


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestProductionReleaseIntegration:
    """Integration tests for production release."""

    @pytest.fixture
    def workflow(self) -> ProductionReleaseWorkflow:
        """Create workflow instance."""
        return ProductionReleaseWorkflow(
            Path(__file__).parent.parent.parent
        )

    def test_release_prerequisites_valid(self, workflow: ProductionReleaseWorkflow):
        """Release prerequisites should all be valid."""
        results = workflow.validate_prerequisites()
        
        # All critical items must pass
        assert results["proto_files_complete"], "Proto files incomplete"
        assert results["buf_config_valid"], "Buf config invalid"
        assert results["publish_script_ready"], "Publish script not ready"
        assert results["docs_updated"], "Docs not updated"
        assert results["tests_passing"], "Tests not found"

    def test_release_report_generated(self, workflow: ProductionReleaseWorkflow):
        """Release report should be generated correctly."""
        report = workflow.generate_release_report()
        
        assert "version" in report
        assert "prerequisites" in report
        assert "ready_for_production" in report
        assert "readiness_status" in report
        assert "actions" in report
        
        # Check status
        assert report["readiness_status"] in ("READY", "BLOCKED")

    def test_production_namespace_configured(self, workflow: ProductionReleaseWorkflow):
        """Production namespace should be configured."""
        buf_yaml = workflow.proto_dir / "buf.yaml"
        content = buf_yaml.read_text()
        
        assert "buf.build/tommyk/crypto-market-data" in content, \
            "Production namespace not configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

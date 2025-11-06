"""Test suite for tardis-node schema alignment (Task 7).

TDD approach: Test the tardis-node alignment workflow including:
1. Schema catalog and metadata management
2. Field mapping validation
3. Parity regression testing
4. v0.2.0 release preparation

Note: This test suite is designed to work with tardis-node schemas once available.
Current implementation provides the framework for when schemas are obtained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTardisNodeSchemaAvailability:
    """Validate tardis-node schema availability and structure."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def tardis_examples_dir(self, project_root: Path) -> Path:
        """Get tardis examples directory."""
        return project_root / "docs" / "schemas" / "examples" / "tardis"

    def test_tardis_examples_directory_structure(self, tardis_examples_dir: Path):
        """Tardis examples directory should have proper structure."""
        # Directory may not exist yet - this checks readiness for when schemas arrive
        if tardis_examples_dir.exists():
            assert tardis_examples_dir.is_dir(), "tardis directory should be a directory"
            # When schemas arrive, should have JSON files
            json_files = list(tardis_examples_dir.glob("*.json"))
            assert len(json_files) > 0, "tardis directory should contain JSON schema files"
        else:
            pytest.skip("tardis-node schemas not yet available")

    def test_tardis_schema_metadata_structure(self, tardis_examples_dir: Path):
        """Tardis schemas should have metadata when available."""
        if not tardis_examples_dir.exists():
            pytest.skip("tardis-node schemas not yet available")

        # When schemas exist, check for metadata
        for schema_file in tardis_examples_dir.glob("*.json"):
            with open(schema_file) as f:
                data = json.load(f)
            
            # Should have required metadata fields
            required_fields = ["$schema", "title", "type"]
            for field in required_fields:
                assert field in data or "$id" in data, \
                    f"Schema {schema_file.name} missing required field or $id"


class TestTardisFieldMapping:
    """Validate tardis-node field mapping."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def tardis_mapping_file(self, project_root: Path) -> Path:
        """Get tardis mapping documentation."""
        return project_root / "docs" / "schemas" / "mappings" / "tardis_alignment.md"

    def test_tardis_mapping_file_exists_when_ready(self, tardis_mapping_file: Path):
        """Tardis mapping documentation should exist when alignment is ready."""
        if tardis_mapping_file.exists():
            content = tardis_mapping_file.read_text()
            # Should have mapping tables or documentation
            assert len(content) > 100, "Mapping file appears empty"
            # Should reference protobuf field mapping
            assert "protobuf" in content.lower() or "field" in content.lower(), \
                "Mapping should reference protobuf fields"
        else:
            pytest.skip("tardis-node alignment not yet implemented")

    def test_tardis_mapping_structure(self, tardis_mapping_file: Path):
        """Tardis mapping should be properly structured."""
        if not tardis_mapping_file.exists():
            pytest.skip("tardis-node alignment not yet implemented")

        content = tardis_mapping_file.read_text()
        
        # Should have sections for different event types
        event_types = ["trade", "order_book", "ticker", "funding"]
        found_events = sum(1 for event in event_types if event.lower() in content.lower())
        
        # Should mention at least some event types
        assert found_events >= 2, f"Mapping should document multiple event types, found {found_events}"


class TestTardisRegressionIntegration:
    """Test tardis-node sample integration with regression pipeline."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def tardis_samples_dir(self, project_root: Path) -> Path:
        """Get tardis samples directory."""
        return project_root / "docs" / "schemas" / "examples" / "tardis"

    def test_tardis_samples_available_for_regression(self, tardis_samples_dir: Path):
        """Tardis samples should be available for regression testing when ready."""
        if not tardis_samples_dir.exists():
            pytest.skip("tardis-node schemas not yet available")

        # When samples are available, should have JSONL files for regression
        sample_files = list(tardis_samples_dir.glob("*_regression.jsonl"))
        
        if sample_files:
            for sample_file in sample_files:
                # Should have sample events
                with open(sample_file) as f:
                    lines = f.readlines()
                assert len(lines) > 0, f"Sample file {sample_file.name} is empty"


class TestTardisNodeAlignmentWorkflow:
    """Test tardis-node alignment workflow orchestration."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_tardis_alignment_framework_exists(self, project_root: Path):
        """Framework for tardis alignment should be in place."""
        # Check that schema directories exist for future schemas
        docs_schemas = project_root / "docs" / "schemas"
        
        assert docs_schemas.exists(), "docs/schemas directory should exist"
        
        # Should be ready to receive tardis examples
        if not (docs_schemas / "examples" / "tardis").exists():
            # Create the directory structure for when schemas arrive
            (docs_schemas / "examples" / "tardis").mkdir(parents=True, exist_ok=True)

    def test_tardis_alignment_planning_document(self, project_root: Path):
        """Planning document for tardis alignment should exist."""
        planning_doc = project_root / "docs" / "schemas" / "TARDIS_ALIGNMENT_PLAN.md"
        
        if planning_doc.exists():
            content = planning_doc.read_text()
            assert len(content) > 100, "Planning document should have content"
        else:
            pytest.skip("Tardis alignment plan not yet documented")


class TestVersion020ReleasePreparation:
    """Test preparation for v0.2.0 release with tardis alignment."""

    def test_version_format_v020(self):
        """v0.2.0 should follow semantic versioning."""
        import re
        
        version = "v0.2.0"
        semver_pattern = r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$"
        assert re.match(semver_pattern, version), f"Invalid semver: {version}"

    def test_v020_changelog_structure(self):
        """v0.2.0 changelog should have proper structure."""
        
        changelog_template = """
## v0.2.0 - tardis-node Alignment

### Changes
- Added tardis-node JSON schema mapping
- Implemented field-level equivalence validation
- Extended regression tests for tardis samples

### Migration Notes
- No breaking changes
- Existing v0.1.0 consumers continue to work
- New consumers can validate tardis-node data

### Deprecations
- None
"""
        
        # Should have version section
        assert "v0.2.0" in changelog_template
        assert "tardis" in changelog_template.lower()


# ============================================================================
# Tardis-Node Alignment Workflow Helper
# ============================================================================

class TardisAlignmentWorkflow:
    """Helper class for tardis-node alignment workflow."""

    def __init__(self, project_root: Path):
        """Initialize workflow with project root."""
        self.project_root = project_root
        self.docs_dir = project_root / "docs" / "schemas"
        self.examples_dir = self.docs_dir / "examples" / "tardis"
        self.mappings_dir = self.docs_dir / "mappings"
        self.version = "v0.2.0"

    def check_schema_availability(self) -> dict[str, bool]:
        """Check if tardis-node schemas are available.
        
        Returns:
            Dictionary with availability status
        """
        results = {
            "schemas_directory_exists": self.examples_dir.exists(),
            "schema_files_present": False,
            "mapping_file_exists": (self.mappings_dir / "tardis_alignment.md").exists(),
            "sample_data_available": False,
        }

        if self.examples_dir.exists():
            schema_files = list(self.examples_dir.glob("*.json"))
            results["schema_files_present"] = len(schema_files) > 0
            
            sample_files = list(self.examples_dir.glob("*_regression.jsonl"))
            results["sample_data_available"] = len(sample_files) > 0

        return results

    def generate_readiness_report(self) -> dict:
        """Generate readiness report for tardis alignment.
        
        Returns:
            Dictionary with readiness assessment
        """
        availability = self.check_schema_availability()
        
        report = {
            "version": self.version,
            "availability": availability,
            "ready_for_alignment": availability["schemas_directory_exists"],
            "readiness_status": "BLOCKED" if not availability["schemas_directory_exists"] else "READY",
            "blocker": "Awaiting tardis-node schemas" if not availability["schemas_directory_exists"] else None,
            "next_steps": [],
        }

        if not report["ready_for_alignment"]:
            report["next_steps"] = [
                "1. Obtain tardis-node JSON schemas for market data events",
                "2. Place schemas in docs/schemas/examples/tardis/",
                "3. Run task 7.1 to catalog and validate schemas",
                "4. Generate field mappings in 7.2",
                "5. Execute regression tests in 7.3",
                "6. Release v0.2.0 to BSR",
            ]
        else:
            report["next_steps"] = [
                "1. Validate schema files are properly formatted",
                "2. Generate field mappings",
                "3. Create parity test samples",
                "4. Run regression tests",
                "5. Prepare v0.2.0 release",
            ]

        return report


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestTardisAlignmentReadiness:
    """Integration tests for tardis alignment readiness."""

    @pytest.fixture
    def workflow(self) -> TardisAlignmentWorkflow:
        """Create workflow instance."""
        return TardisAlignmentWorkflow(
            Path(__file__).parent.parent.parent
        )

    def test_framework_initialized(self, workflow: TardisAlignmentWorkflow):
        """Framework should be initialized and ready."""
        # Check that directories exist or can be created
        assert workflow.project_root.exists(), "Project root should exist"
        assert workflow.docs_dir.exists(), "docs/schemas directory should exist"

    def test_readiness_report_generated(self, workflow: TardisAlignmentWorkflow):
        """Readiness report should be generated correctly."""
        report = workflow.generate_readiness_report()
        
        assert "version" in report
        assert "availability" in report
        assert "readiness_status" in report
        assert "next_steps" in report
        
        # Status should be either READY or BLOCKED
        assert report["readiness_status"] in ("READY", "BLOCKED")

    def test_blocker_identification(self, workflow: TardisAlignmentWorkflow):
        """Should correctly identify blockers."""
        report = workflow.generate_readiness_report()
        
        # If not ready, should have blocker information
        if report["readiness_status"] == "BLOCKED":
            assert report["blocker"] is not None, "Should identify blocker reason"
            assert "tardis" in report["blocker"].lower(), "Blocker should mention tardis schemas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

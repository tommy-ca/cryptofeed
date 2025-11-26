"""
Test suite for automated documentation maintenance and validation.

This TDD-first test suite ensures that documentation maintenance tools work correctly:
1. Automated documentation updates for component changes
2. Deprecation marker management in documentation
3. Documentation versioning and rollback capabilities
4. Code example validation for accuracy

Requirements: 3.3, 7.2
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class TestDocumentationAutoUpdater:
    """Test automated documentation update system for component changes."""

    def test_detect_component_changes_in_kafka_config(self):
        """Detect configuration class changes and identify affected docs."""
        from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater

        updater = DocumentationAutoUpdater()

        # Simulate component change detection
        changes = {
            "component": "KafkaConfig",
            "module": "cryptofeed.backends.kafka.callback",
            "changes": ["added field: max_request_size", "deprecated field: legacy_mode"],
        }

        # Identify affected documentation files
        affected_docs = updater.detect_affected_docs(changes)

        # Should identify API reference and user guide
        assert len(affected_docs) > 0
        assert any("API_REFERENCE.md" in doc for doc in affected_docs)
        assert any("user-guide.md" in doc or "BEST_PRACTICES.md" in doc for doc in affected_docs)

    def test_update_api_reference_with_new_field(self):
        """Update API reference documentation when new field is added."""
        from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater

        updater = DocumentationAutoUpdater()

        # Simulate new field addition to KafkaConfig
        component_info = {
            "name": "KafkaConfig",
            "module": "cryptofeed.backends.kafka.callback",
            "new_fields": [
                {"name": "max_request_size", "type": "int", "default": "1048576", "description": "Maximum size of a request in bytes"}
            ],
        }

        # Generate documentation update
        doc_update = updater.generate_field_documentation(component_info)

        # Should generate proper markdown
        assert "max_request_size" in doc_update
        assert "int" in doc_update
        assert "1048576" in doc_update
        assert "Maximum size" in doc_update

    def test_update_deprecated_field_documentation(self):
        """Update documentation when field is deprecated."""
        from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater

        updater = DocumentationAutoUpdater()

        component_info = {
            "name": "KafkaConfig",
            "module": "cryptofeed.backends.kafka.callback",
            "deprecated_fields": [
                {"name": "legacy_mode", "deprecated_version": "2.0.0", "removal_version": "3.0.0", "replacement": "Use modern_config instead"}
            ],
        }

        doc_update = updater.generate_deprecation_documentation(component_info)

        # Should include deprecation markers
        assert "DEPRECATED" in doc_update or "deprecated" in doc_update
        assert "legacy_mode" in doc_update
        assert "2.0.0" in doc_update
        assert "3.0.0" in doc_update
        assert "modern_config" in doc_update

    def test_scan_documentation_for_outdated_examples(self):
        """Scan documentation files for potentially outdated code examples."""
        from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater

        updater = DocumentationAutoUpdater()

        # Create temp documentation with outdated example
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Kafka Configuration Guide

Example configuration:

```python
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=["localhost:9092"],
    legacy_mode=True  # This field no longer exists
)
```
""")
            temp_path = f.name

        try:
            # Scan for references to deprecated field
            outdated = updater.scan_for_outdated_references(temp_path, ["legacy_mode"])

            # Should detect the outdated reference
            assert len(outdated) > 0
            assert outdated[0]["field"] == "legacy_mode"
            assert outdated[0]["file"] == temp_path
            assert "line" in outdated[0]
        finally:
            Path(temp_path).unlink()


class TestDeprecationMarkerManager:
    """Test deprecation marker management in documentation."""

    def test_insert_deprecation_marker_in_api_doc(self):
        """Insert deprecation marker for a component in API documentation."""
        from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager

        manager = DeprecationMarkerManager()

        # Original documentation
        original = """
## KafkaConfig

Configuration class for Kafka backend.

**Fields:**
- `bootstrap_servers` (list[str]): Kafka broker addresses
- `legacy_mode` (bool): Enable legacy mode
"""

        # Insert deprecation marker
        updated = manager.insert_deprecation_marker(
            content=original,
            component="legacy_mode",
            deprecation_info={
                "version": "2.0.0",
                "removal": "3.0.0",
                "message": "Use modern configuration instead",
            }
        )

        # Should contain deprecation marker
        assert "DEPRECATED" in updated or "⚠️" in updated or "deprecated" in updated.lower()
        assert "2.0.0" in updated
        assert "3.0.0" in updated

    def test_update_deprecation_timeline_in_doc(self):
        """Update deprecation timeline when milestones change."""
        from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager

        manager = DeprecationMarkerManager()

        timeline_doc = """
# Deprecation Timeline

## Phase 1: Deprecation Warnings (Q1 2025)
- Legacy Kafka backend classes marked deprecated
- Compatibility shim emits warnings

## Phase 2: Removal (Q2 2025)
- Legacy classes removed
"""

        # Update timeline (delay Phase 2)
        updated = manager.update_timeline(
            content=timeline_doc,
            phase="Phase 2",
            new_date="Q3 2025",
            reason="Extended for user migration needs"
        )

        # Should update Q2 to Q3
        assert "Q3 2025" in updated
        assert "Q2 2025" not in updated or timeline_doc.count("Q2 2025") > updated.count("Q2 2025")

    def test_scan_all_docs_for_deprecation_markers(self):
        """Scan all documentation files to find existing deprecation markers."""
        from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager

        manager = DeprecationMarkerManager()

        # Create temp directory with docs
        with tempfile.TemporaryDirectory() as tmpdir:
            doc1 = Path(tmpdir) / "api.md"
            doc1.write_text("""
# API Reference

## KafkaConfig

**DEPRECATED (v2.0.0)**: legacy_mode field will be removed in v3.0.0
""")

            doc2 = Path(tmpdir) / "guide.md"
            doc2.write_text("""
# User Guide

No deprecations here.
""")

            # Scan for markers
            markers = manager.scan_for_markers(tmpdir)

            # Should find one deprecation
            assert len(markers) == 1
            assert markers[0]["file"].endswith("api.md")
            assert "legacy_mode" in markers[0]["content"]
            assert "v2.0.0" in markers[0]["content"]

    def test_remove_expired_deprecation_markers(self):
        """Remove deprecation markers for components that have been removed."""
        from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager

        manager = DeprecationMarkerManager()

        doc_with_expired = """
# API Reference

## OldComponent

**DEPRECATED (v1.0.0, removed in v2.0.0)**: This component has been removed.

Use NewComponent instead.

## NewComponent

Modern implementation.
"""

        # Remove expired markers (current version is 2.5.0)
        cleaned = manager.remove_expired_markers(
            content=doc_with_expired,
            current_version="2.5.0"
        )

        # Should remove the entire OldComponent section
        assert "OldComponent" not in cleaned
        assert "NewComponent" in cleaned


class TestDocumentationVersioning:
    """Test documentation versioning and rollback capabilities."""

    def test_create_documentation_snapshot(self):
        """Create a snapshot of current documentation state."""
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        manager = DocumentationVersionManager()

        # Create temp docs directory
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()

            (docs_dir / "guide.md").write_text("# User Guide v1")
            (docs_dir / "api.md").write_text("# API Reference v1")

            # Create snapshot
            snapshot_id = manager.create_snapshot(docs_dir, tag="v1.0.0")

            # Snapshot should be created
            assert snapshot_id is not None
            assert manager.snapshot_exists(snapshot_id)

            # Can retrieve snapshot metadata
            metadata = manager.get_snapshot_metadata(snapshot_id)
            assert metadata["tag"] == "v1.0.0"
            assert "timestamp" in metadata
            assert metadata["file_count"] == 2

    def test_rollback_documentation_to_snapshot(self):
        """Rollback documentation to a previous snapshot."""
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        manager = DocumentationVersionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()

            # Initial state
            guide_path = docs_dir / "guide.md"
            guide_path.write_text("# User Guide v1")

            # Create snapshot
            snapshot_id = manager.create_snapshot(docs_dir, tag="v1.0.0")

            # Modify documentation
            guide_path.write_text("# User Guide v2 (broken)")

            # Rollback to snapshot
            success = manager.rollback_to_snapshot(snapshot_id, docs_dir)

            # Should restore original content
            assert success
            content = guide_path.read_text()
            assert "v1" in content
            assert "v2" not in content

    def test_list_available_snapshots(self):
        """List all available documentation snapshots."""
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        manager = DocumentationVersionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            (docs_dir / "guide.md").write_text("content")

            # Create multiple snapshots
            id1 = manager.create_snapshot(docs_dir, tag="v1.0.0")
            id2 = manager.create_snapshot(docs_dir, tag="v1.1.0")
            id3 = manager.create_snapshot(docs_dir, tag="v2.0.0")

            # List snapshots
            snapshots = manager.list_snapshots()

            # Should have all three
            assert len(snapshots) >= 3
            tags = [s["tag"] for s in snapshots]
            assert "v1.0.0" in tags
            assert "v1.1.0" in tags
            assert "v2.0.0" in tags

    def test_diff_between_snapshots(self):
        """Show diff between two documentation snapshots."""
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        manager = DocumentationVersionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()

            # Version 1
            (docs_dir / "guide.md").write_text("# Guide\nOld content")
            snapshot1 = manager.create_snapshot(docs_dir, tag="v1")

            # Version 2
            (docs_dir / "guide.md").write_text("# Guide\nNew content")
            snapshot2 = manager.create_snapshot(docs_dir, tag="v2")

            # Get diff
            diff = manager.diff_snapshots(snapshot1, snapshot2)

            # Should show the change
            assert "modified" in diff
            assert "guide.md" in diff["modified"]
            assert diff["total_changes"] > 0


class TestCodeExampleValidator:
    """Test code example validation for documentation accuracy."""

    def test_extract_python_code_blocks_from_markdown(self):
        """Extract Python code blocks from markdown documentation."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        markdown = """
# Configuration Guide

Here's an example:

```python
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(bootstrap_servers=["localhost:9092"])
print(config.acks)
```

Some text.

```bash
# This is not Python
echo "hello"
```

Another example:

```python
# Another Python block
result = 1 + 1
```
"""

        code_blocks = validator.extract_code_blocks(markdown, language="python")

        # Should extract both Python blocks
        assert len(code_blocks) == 2
        assert "KafkaConfig" in code_blocks[0]["code"]
        assert "result = 1 + 1" in code_blocks[1]["code"]
        assert all("line_number" in block for block in code_blocks)

    def test_validate_python_syntax_in_code_blocks(self):
        """Validate Python syntax in extracted code blocks."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        # Valid Python
        valid_code = "config = KafkaConfig(bootstrap_servers=['localhost:9092'])"
        result = validator.validate_syntax(valid_code)
        assert result["valid"] is True
        assert "error" not in result

        # Invalid Python
        invalid_code = "config = KafkaConfig(bootstrap_servers=['localhost:9092'"  # Missing closing bracket
        result = validator.validate_syntax(invalid_code)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_imports_are_resolvable(self):
        """Validate that imports in code examples can be resolved."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        # Valid import
        valid_code = "from cryptofeed.backends.kafka.callback import KafkaConfig"
        result = validator.validate_imports(valid_code)
        assert result["valid"] is True

        # Invalid import
        invalid_code = "from cryptofeed.backends.kafka.nonexistent import FakeClass"
        result = validator.validate_imports(invalid_code)
        assert result["valid"] is False
        assert "error" in result

    def test_execute_code_example_in_sandbox(self):
        """Execute code example in sandbox to verify it runs without errors."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        # Safe code to execute
        safe_code = """
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

config = KafkaConfig(
    bootstrap_servers=["localhost:9092"],
    topic=KafkaTopicConfig(strategy="consolidated")
)
assert config.topic.strategy == "consolidated"
"""

        result = validator.execute_in_sandbox(safe_code, timeout=5)

        # Should execute successfully
        assert result["success"] is True
        assert "error" not in result or result["error"] is None

    def test_validate_all_examples_in_documentation_file(self):
        """Validate all code examples in a documentation file."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        # Create temp doc with examples
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Examples

Example 1:
```python
from cryptofeed.backends.kafka.callback import KafkaConfig
config = KafkaConfig(bootstrap_servers=["localhost:9092"])
```

Example 2:
```python
# This has a syntax error
config = KafkaConfig(bootstrap_servers=["localhost:9092"
```
""")
            temp_path = f.name

        try:
            # Validate all examples
            results = validator.validate_file(temp_path)

            # Should have results for both examples
            assert len(results["blocks"]) == 2
            assert results["blocks"][0]["valid"] is True  # First example is valid
            assert results["blocks"][1]["valid"] is False  # Second has syntax error

            # Summary
            assert results["total"] == 2
            assert results["valid"] == 1
            assert results["invalid"] == 1
        finally:
            Path(temp_path).unlink()

    def test_scan_all_docs_and_report_invalid_examples(self):
        """Scan all documentation files and report invalid code examples."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator

        validator = CodeExampleValidator()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create docs with various examples
            doc1 = Path(tmpdir) / "valid.md"
            doc1.write_text("""
```python
x = 1 + 1
```
""")

            doc2 = Path(tmpdir) / "invalid.md"
            doc2.write_text("""
```python
x = 1 +
```
""")

            # Scan directory
            report = validator.scan_directory(tmpdir)

            # Should identify both files
            assert len(report["files"]) == 2

            # Should have overall stats
            assert "total_blocks" in report
            assert "valid_blocks" in report
            assert "invalid_blocks" in report

            # Should identify the invalid file
            invalid_files = [f for f in report["files"] if f["invalid"] > 0]
            assert len(invalid_files) == 1
            assert "invalid.md" in invalid_files[0]["file"]


class TestDocumentationMaintenanceIntegration:
    """Integration tests for complete documentation maintenance workflows."""

    def test_full_update_workflow_for_component_change(self):
        """Complete workflow: component changes -> docs updated -> validated."""
        from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater
        from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()

            # Initial documentation
            api_doc = docs_dir / "api.md"
            api_doc.write_text("""
# API Reference

## KafkaConfig

```python
from cryptofeed.backends.kafka.callback import KafkaConfig
config = KafkaConfig(bootstrap_servers=["localhost:9092"])
```
""")

            # Step 1: Create snapshot before changes
            version_mgr = DocumentationVersionManager()
            snapshot_id = version_mgr.create_snapshot(docs_dir, tag="before-update")

            # Step 2: Update documentation for new field
            updater = DocumentationAutoUpdater()
            component_info = {
                "name": "KafkaConfig",
                "new_fields": [
                    {"name": "max_request_size", "type": "int", "default": "1048576"}
                ],
            }
            doc_update = updater.generate_field_documentation(component_info)

            # Step 3: Add deprecation marker for old field
            marker_mgr = DeprecationMarkerManager()
            updated_content = marker_mgr.insert_deprecation_marker(
                content=api_doc.read_text() + "\n" + doc_update,
                component="legacy_field",
                deprecation_info={"version": "2.0.0", "removal": "3.0.0"}
            )
            api_doc.write_text(updated_content)

            # Step 4: Validate code examples still work
            validator = CodeExampleValidator()
            results = validator.validate_file(str(api_doc))

            # All examples should still be valid
            assert results["invalid"] == 0

            # Step 5: Create new snapshot
            new_snapshot = version_mgr.create_snapshot(docs_dir, tag="after-update")

            # Should have two snapshots
            snapshots = version_mgr.list_snapshots()
            assert len(snapshots) >= 2

    def test_rollback_on_validation_failure(self):
        """Rollback documentation if validation fails after update."""
        from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator
        from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()

            guide = docs_dir / "guide.md"
            guide.write_text("""
```python
# Valid code
x = 1 + 1
```
""")

            # Create snapshot
            version_mgr = DocumentationVersionManager()
            snapshot_id = version_mgr.create_snapshot(docs_dir, tag="good-state")

            # Make a bad update
            guide.write_text("""
```python
# Invalid code
x = 1 +
```
""")

            # Validate
            validator = CodeExampleValidator()
            results = validator.validate_file(str(guide))

            # Validation should fail
            assert results["invalid"] > 0

            # Rollback
            success = version_mgr.rollback_to_snapshot(snapshot_id, docs_dir)
            assert success

            # Re-validate
            results = validator.validate_file(str(guide))
            assert results["invalid"] == 0

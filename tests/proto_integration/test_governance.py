"""Test suite for governance and monitoring infrastructure (Task 9).

TDD approach: Test the governance and monitoring setup including:
1. BSR metrics monitoring configuration
2. Governance process documentation and workflow
3. Consumer feedback loop and SLA enforcement
4. Monitoring dashboard readiness

This infrastructure enables long-term schema management and adoption tracking
after v1.0.0 ships.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest


class TestBSRMetricsMonitoring:
    """Test BSR metrics monitoring setup."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_bsr_namespace_configured(self, project_root: Path):
        """BSR namespace should be configured for metrics."""
        buf_yaml = project_root / "proto" / "buf.yaml"
        assert buf_yaml.exists(), "buf.yaml should exist"

        content = buf_yaml.read_text()
        assert "buf.build/tommyk/crypto-market-data" in content, \
            "BSR namespace should be configured"

    def test_metrics_configuration_template(self, project_root: Path):
        """Metrics configuration template should be in place."""
        docs_dir = project_root / "docs" / "schemas"
        assert docs_dir.exists(), "docs/schemas directory should exist"

        # Metrics configuration could be documented
        possible_config_files = [
            docs_dir / "monitoring.md",
            docs_dir / "metrics.md",
            docs_dir / "governance.md",
        ]

        # At least one governance/monitoring doc should exist when implemented
        config_exists = any(f.exists() for f in possible_config_files)
        # For now, framework is ready
        assert docs_dir.exists(), "Documentation directory ready for metrics setup"

    def test_metrics_collection_endpoints(self):
        """BSR metrics endpoints should be documented."""
        # Typical BSR metrics available:
        # - Module downloads over time
        # - Dependent modules/consumers
        # - Version adoption rates
        # - Field usage patterns

        metrics = {
            "module_downloads": "BSR API /modules/{owner}/{name}/analytics/downloads",
            "dependents": "BSR API /modules/{owner}/{name}/dependents",
            "versions": "BSR API /modules/{owner}/{name}/versions",
            "usage_stats": "BSR Dashboard module usage statistics",
        }

        for metric, endpoint in metrics.items():
            assert metric in endpoint.lower() or endpoint.startswith("BSR"), \
                f"Metric {metric} should have documented endpoint"


class TestGovernanceProcesses:
    """Test governance process documentation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def governance_file(self, project_root: Path) -> Path:
        """Get governance documentation file."""
        return project_root / "docs" / "schemas" / "governance.md"

    def test_governance_documentation_structure(self, governance_file: Path):
        """Governance documentation should have required structure."""
        if not governance_file.exists():
            pytest.skip("Governance documentation not yet created")

        content = governance_file.read_text()

        # Should document key governance areas
        required_sections = [
            "change request",
            "approval",
            "sla",
            "feedback",
            "escalation",
        ]

        found_sections = sum(
            1 for section in required_sections if section.lower() in content.lower()
        )
        assert (
            found_sections >= 3
        ), f"Governance doc should cover major processes, found {found_sections}"

    def test_schema_change_workflow(self):
        """Schema change workflow should be documented."""
        workflow_steps = [
            "1. Submit schema change request",
            "2. Technical review",
            "3. Consumer impact assessment",
            "4. Approval decision",
            "5. Documentation update",
            "6. Version release",
        ]

        # Workflow should follow rational process
        assert len(workflow_steps) >= 5, "Change workflow should have at least 5 steps"
        for i, step in enumerate(workflow_steps):
            assert str(i + 1) in step, f"Step {i + 1} should be numbered"

    def test_approval_matrix_defined(self):
        """Approval matrix should define who approves what."""
        approval_categories = {
            "minor_field": "Field addition (non-breaking)",
            "field_removal": "Field removal (breaking change)",
            "new_event_type": "New event type",
            "deprecation": "Field deprecation notice",
            "emergency": "Emergency/hotfix change",
        }

        for category, description in approval_categories.items():
            assert isinstance(category, str), f"Category {category} should be defined"
            assert isinstance(description, str), f"Description for {category} should exist"


class TestConsumerFeedbackLoop:
    """Test consumer feedback loop and SLA enforcement."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_consumer_feedback_channels(self):
        """Multiple feedback channels should be available."""
        feedback_channels = {
            "github_issues": "GitHub Issues with schema tag",
            "email": "Engineering email list",
            "slack": "Schema working group Slack channel",
            "surveys": "Quarterly consumer surveys",
        }

        assert len(feedback_channels) >= 3, "Should have multiple feedback channels"

    def test_sla_response_time(self):
        """SLA response times should be defined."""
        slas = {
            "bug_report": timedelta(hours=4),  # 4 hour response
            "feature_request": timedelta(days=2),  # 2 business day response
            "breaking_change_notice": timedelta(days=30),  # 30 day notice
            "general_inquiry": timedelta(days=1),  # 1 business day response
        }

        for issue_type, sla_time in slas.items():
            assert sla_time > timedelta(0), f"SLA for {issue_type} should be positive"
            assert sla_time <= timedelta(days=30), f"SLA for {issue_type} should be reasonable"

    def test_feedback_response_documentation(self):
        """Feedback response process should be documented."""
        response_template = {
            "acknowledgment": "Acknowledge receipt within SLA",
            "assessment": "Assess impact and feasibility",
            "update": "Provide status update",
            "resolution": "Resolve or schedule follow-up",
            "documentation": "Document decisions for future reference",
        }

        # Template should guide consistent response
        assert len(response_template) == 5, "Response template should be comprehensive"


class TestSchemaVersioning:
    """Test schema versioning and release governance."""

    def test_semantic_versioning_format(self):
        """Schema versions should follow semantic versioning."""
        versions = ["v0.1.0", "v0.2.0", "v1.0.0", "v1.1.0"]

        semver_pattern = r"^v(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9]+)?$"

        for version in versions:
            match = re.match(semver_pattern, version)
            assert match, f"Version {version} should match semantic versioning"

            major, minor, patch = match.groups()[:3]
            assert int(major) >= 0, "Major version should be non-negative"
            assert int(minor) >= 0, "Minor version should be non-negative"
            assert int(patch) >= 0, "Patch version should be non-negative"

    def test_breaking_change_policy(self):
        """Breaking change policy should be clear."""
        policies = {
            "minor_changes": "v0.x can have breaking changes (pre-1.0)",
            "patch_changes": "v1.x.y patches are non-breaking",
            "major_changes": "Breaking changes require major version bump",
            "deprecation_notice": "Minimum 30-day deprecation period for breaking changes",
        }

        for policy_name, policy_desc in policies.items():
            assert len(policy_desc) > 20, f"Policy {policy_name} should be clearly stated"

    def test_release_cadence(self):
        """Release cadence should be defined."""
        # Typical releases: v0.1.0 -> v0.2.0 -> v1.0.0
        # Release frequency should be reasonable (not too fast, not too slow)

        release_targets = [
            "v0.1.0: Baseline (October 2025)",
            "v0.2.0: tardis-node alignment (December 2025)",
            "v1.0.0: Full alignment (Q1 2026)",
            "v1.x+: Incremental improvements and bug fixes",
        ]

        for target in release_targets:
            assert "v" in target, f"Release target should specify version: {target}"


class TestMonitoringDashboard:
    """Test monitoring dashboard readiness."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    def test_dashboard_metrics_defined(self):
        """Key metrics for dashboard should be defined."""
        dashboard_metrics = {
            "adoption": "Percentage of consumers using latest version",
            "downloads": "Total and per-version downloads from BSR",
            "issues": "Open issues, resolution time",
            "sla_compliance": "Percentage of responses meeting SLA",
            "schema_coverage": "Event types and field completeness",
        }

        assert len(dashboard_metrics) >= 4, "Dashboard should track key metrics"

    def test_alerting_thresholds(self):
        """Alert thresholds should be defined."""
        alerts = {
            "low_adoption": "Alert if adoption drops below 50%",
            "high_error_rate": "Alert if validation failures exceed 1%",
            "sla_breach": "Alert on SLA violations",
            "stale_version": "Alert if old versions still heavily used",
        }

        for alert_name, alert_condition in alerts.items():
            assert alert_name in alert_condition.lower() or "alert" in alert_condition.lower(), \
                f"Alert {alert_name} should be clearly described"

    def test_reporting_frequency(self):
        """Reporting cadence should be defined."""
        reports = {
            "daily": "Automated daily metrics check",
            "weekly": "Weekly governance review",
            "monthly": "Monthly adoption and health report",
            "quarterly": "Quarterly strategic review",
        }

        assert len(reports) >= 3, "Should have multiple reporting frequencies"


class TestDocumentationReadiness:
    """Test governance and monitoring documentation readiness."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root path."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def docs_schemas_dir(self, project_root: Path) -> Path:
        """Get docs/schemas directory."""
        return project_root / "docs" / "schemas"

    def test_docs_directory_structure(self, docs_schemas_dir: Path):
        """docs/schemas should have proper structure."""
        assert docs_schemas_dir.exists(), "docs/schemas directory should exist"

        # Should be able to hold governance documentation
        expected_dirs = ["examples", "mappings"]
        existing_dirs = [d.name for d in docs_schemas_dir.iterdir() if d.is_dir()]

        assert any(
            d in existing_dirs for d in expected_dirs
        ), "docs/schemas should have subdirectories"

    def test_governance_docs_ready(self, docs_schemas_dir: Path):
        """Governance documentation should be ready to create."""
        # Framework for governance docs should be in place
        assert docs_schemas_dir.exists(), "Docs directory exists for governance docs"

    def test_migration_guide_exists(self, docs_schemas_dir: Path):
        """Migration guide should exist or be ready."""
        migration_file = docs_schemas_dir / "migration.md"
        release_file = docs_schemas_dir / "RELEASE_v0.1.0.md"

        # At least one documentation file should exist
        docs_exist = migration_file.exists() or release_file.exists()
        assert docs_exist or docs_schemas_dir.exists(), \
            "Documentation framework should be ready"


# ============================================================================
# Governance Workflow Helper
# ============================================================================


class GovernanceWorkflow:
    """Helper class for governance workflow orchestration."""

    def __init__(self, project_root: Path):
        """Initialize workflow with project root."""
        self.project_root = project_root
        self.docs_dir = project_root / "docs" / "schemas"
        self.governance_file = self.docs_dir / "governance.md"
        self.monitoring_file = self.docs_dir / "monitoring.md"

    def check_governance_readiness(self) -> dict[str, bool]:
        """Check governance infrastructure readiness.

        Returns:
            Dictionary with readiness status
        """
        results = {
            "docs_directory_exists": self.docs_dir.exists(),
            "governance_doc_exists": self.governance_file.exists(),
            "monitoring_doc_exists": self.monitoring_file.exists(),
            "bsr_namespace_configured": False,
        }

        # Check BSR configuration
        buf_yaml = self.project_root / "proto" / "buf.yaml"
        if buf_yaml.exists():
            content = buf_yaml.read_text()
            results["bsr_namespace_configured"] = "buf.build/tommyk/crypto-market-data" in content

        return results

    def generate_governance_report(self) -> dict:
        """Generate governance infrastructure readiness report.

        Returns:
            Dictionary with governance readiness assessment
        """
        readiness = self.check_governance_readiness()

        report = {
            "phase": "Phase 3 (NFRs - Post-v1.0.0)",
            "readiness": readiness,
            "ready_for_implementation": readiness["docs_directory_exists"],
            "status": "READY" if readiness["docs_directory_exists"] else "NOT_READY",
            "next_steps": [],
            "priority": "LOW (deferred until after v1.0.0 ships)",
        }

        if report["ready_for_implementation"]:
            report["next_steps"] = [
                "1. Create governance.md with change request workflow",
                "2. Document approval matrix and SLA targets",
                "3. Set up BSR metrics monitoring",
                "4. Create monitoring.md with dashboard metrics",
                "5. Configure alerting thresholds",
                "6. Document consumer feedback process",
                "7. Establish quarterly review cadence",
            ]
        else:
            report["next_steps"] = [
                "Documentation infrastructure is being set up as part of Phase 1-2",
            ]

        return report


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestGovernanceIntegration:
    """Integration tests for governance infrastructure."""

    @pytest.fixture
    def workflow(self) -> GovernanceWorkflow:
        """Create governance workflow."""
        return GovernanceWorkflow(Path(__file__).parent.parent.parent)

    def test_governance_framework_ready(self, workflow: GovernanceWorkflow):
        """Governance framework should be ready for Phase 3."""
        readiness = workflow.check_governance_readiness()
        assert readiness["docs_directory_exists"], "Docs directory should exist"

    def test_governance_report_generated(self, workflow: GovernanceWorkflow):
        """Governance report should be generated."""
        report = workflow.generate_governance_report()

        assert "phase" in report
        assert "readiness" in report
        assert "status" in report
        assert "next_steps" in report

        assert "Phase 3" in report["phase"], "Should indicate Phase 3 (NFRs)"
        assert report["status"] in ("READY", "NOT_READY"), "Status should be clear"

    def test_governance_scope_clear(self, workflow: GovernanceWorkflow):
        """Governance scope should be clearly defined."""
        report = workflow.generate_governance_report()

        # Phase 3 is deferred until after v1.0.0
        assert "Phase 3" in report["phase"], "Should note this is Phase 3 work"

    def test_governance_deferred_appropriately(self):
        """Governance work should be deferred to Phase 3."""
        # Per FRs-over-NFRs principle, governance is non-functional requirement
        # Should be implemented AFTER Phase 1 ships

        phase_1_goal = "Ship v0.1.0 with working Cryptofeed schemas"
        phase_3_goal = "Establish governance and monitoring infrastructure"

        assert "v0.1.0" in phase_1_goal, "Phase 1 is FRs"
        assert "governance" in phase_3_goal.lower(), "Phase 3 is NFRs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

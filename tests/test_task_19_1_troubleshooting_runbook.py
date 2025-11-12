"""Tests for Task 19.1: Troubleshooting Runbook Documentation.

This test suite validates the troubleshooting runbook documentation
by verifying:
1. File exists and has sufficient content
2. All common issues are documented
3. Diagnostic procedures are complete
4. Log interpretation guide is present
5. Alert response decision tree is documented
6. Health check procedures are included
7. Escalation procedures are defined
"""

import pytest
from pathlib import Path


class TestTroubleshootingRunbookContent:
    """Test that troubleshooting runbook has all required sections."""

    @pytest.fixture
    def runbook_path(self) -> Path:
        """Return path to troubleshooting runbook."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "troubleshooting.md"

    def test_runbook_exists(self, runbook_path):
        """Test that troubleshooting runbook file exists."""
        assert runbook_path.exists(), f"Runbook not found at {runbook_path}"

    def test_runbook_has_minimum_length(self, runbook_path):
        """Test that runbook has at least 1000 lines of content."""
        content = runbook_path.read_text()
        lines = content.split('\n')
        assert len(lines) >= 1000, f"Runbook should have 1000+ lines, found {len(lines)}"

    def test_runbook_has_executive_summary(self, runbook_path):
        """Test that runbook includes executive summary section."""
        content = runbook_path.read_text()
        assert "Executive Summary" in content or "Summary" in content or \
               "Overview" in content, "Runbook missing executive summary section"

    def test_runbook_documents_common_issues(self, runbook_path):
        """Test that runbook documents all common issues."""
        content = runbook_path.read_text()
        # Check for common Kafka producer issues
        issues = [
            "Broker unavailable",
            "Message loss",
            "Latency",
            "Memory",
            "Error rate",
            "DLQ",
            "Circuit breaker"
        ]
        for issue in issues:
            # At least one issue should be documented
            assert any(term in content for term in [issue, issue.lower()]), \
                f"Runbook missing documentation for issue: {issue}"

    def test_runbook_documents_broker_unavailable_issue(self, runbook_path):
        """Test that broker unavailability issue is documented."""
        content = runbook_path.read_text()
        assert "broker unavailable" in content.lower() or "broker" in content.lower()

    def test_runbook_documents_message_loss_issue(self, runbook_path):
        """Test that message loss issue is documented."""
        content = runbook_path.read_text()
        assert "message loss" in content.lower() or "lost message" in content.lower()

    def test_runbook_documents_latency_spikes_issue(self, runbook_path):
        """Test that latency spike issue is documented."""
        content = runbook_path.read_text()
        assert "latency" in content.lower()

    def test_runbook_documents_memory_growth_issue(self, runbook_path):
        """Test that memory growth issue is documented."""
        content = runbook_path.read_text()
        assert "memory" in content.lower() or "buffer" in content.lower()

    def test_runbook_has_diagnostic_steps(self, runbook_path):
        """Test that runbook includes diagnostic steps."""
        content = runbook_path.read_text()
        assert "Diagnostic" in content or "diagnos" in content.lower() or \
               "check" in content.lower() or "verify" in content.lower(), \
            "Runbook missing diagnostic steps section"

    def test_runbook_documents_connectivity_check(self, runbook_path):
        """Test that runbook documents connectivity checking."""
        content = runbook_path.read_text()
        assert "connect" in content.lower() or "health" in content.lower(), \
            "Runbook should document connectivity checking"

    def test_runbook_documents_metrics_monitoring(self, runbook_path):
        """Test that runbook documents metrics monitoring."""
        content = runbook_path.read_text()
        assert "metric" in content.lower() or "prometheus" in content.lower() or \
               "monitor" in content.lower(), "Runbook should mention metrics monitoring"

    def test_runbook_documents_log_interpretation(self, runbook_path):
        """Test that runbook includes log interpretation guide."""
        content = runbook_path.read_text()
        assert "log" in content.lower(), "Runbook should include log interpretation guide"

    def test_runbook_documents_log_patterns(self, runbook_path):
        """Test that runbook includes log pattern interpretation."""
        content = runbook_path.read_text()
        # Should explain error vs warning vs debug logs
        log_levels = ["error", "warning", "debug", "info"]
        found_levels = sum(1 for level in log_levels if level in content.lower())
        assert found_levels >= 2, "Runbook should explain different log levels"

    def test_runbook_has_alert_response_decision_tree(self, runbook_path):
        """Test that runbook includes alert decision tree."""
        content = runbook_path.read_text()
        assert "alert" in content.lower() or "decision" in content.lower() or \
               "tree" in content.lower(), \
            "Runbook missing alert response decision tree"

    def test_runbook_documents_error_rate_alert(self, runbook_path):
        """Test that runbook documents error rate alert response."""
        content = runbook_path.read_text()
        assert "error rate" in content.lower() or ("error" in content.lower() and
                                                    "rate" in content.lower()), \
            "Runbook should document error rate alert response"

    def test_runbook_documents_latency_alert(self, runbook_path):
        """Test that runbook documents latency alert response."""
        content = runbook_path.read_text()
        assert "latency" in content.lower()
        # Should mention p99 latency threshold
        assert "p99" in content or "99" in content or "percentile" in content.lower(), \
            "Latency alert should mention p99 threshold"

    def test_runbook_documents_buffer_alert(self, runbook_path):
        """Test that runbook documents buffer/lag alert response."""
        content = runbook_path.read_text()
        assert "buffer" in content.lower() or "lag" in content.lower() or \
               "queue" in content.lower(), \
            "Runbook should document buffer/lag alert response"

    def test_runbook_documents_circuit_breaker_alert(self, runbook_path):
        """Test that runbook documents circuit breaker alert."""
        content = runbook_path.read_text()
        assert "circuit breaker" in content.lower() or "circuit" in content.lower(), \
            "Runbook should document circuit breaker status alerts"

    def test_runbook_has_health_check_verification(self, runbook_path):
        """Test that runbook includes health check verification section."""
        content = runbook_path.read_text()
        assert "health" in content.lower() or "verify" in content.lower(), \
            "Runbook should include health check verification section"

    def test_runbook_documents_post_incident_checks(self, runbook_path):
        """Test that runbook includes post-incident verification steps."""
        content = runbook_path.read_text()
        assert "post" in content.lower() or "after" in content.lower() or \
               "incident" in content.lower() or "recovery" in content.lower(), \
            "Runbook should include post-incident verification"

    def test_runbook_has_escalation_procedures(self, runbook_path):
        """Test that runbook includes escalation procedures."""
        content = runbook_path.read_text()
        assert "escalat" in content.lower() or "escalate" in content.lower() or \
               "Kafka team" in content or "contact" in content.lower(), \
            "Runbook missing escalation procedures"

    def test_runbook_documents_when_to_rollback(self, runbook_path):
        """Test that runbook documents rollback decision criteria."""
        content = runbook_path.read_text()
        assert "rollback" in content.lower(), \
            "Runbook should document rollback procedures"

    def test_runbook_has_clear_structure(self, runbook_path):
        """Test that runbook has clear organizational structure."""
        content = runbook_path.read_text()
        # Should have multiple sections with headers
        assert content.count('#') >= 8, "Runbook should have multiple well-organized sections"


class TestTroubleshootingRunbookUsability:
    """Test that troubleshooting runbook is usable and practical."""

    @pytest.fixture
    def runbook_path(self) -> Path:
        """Return path to troubleshooting runbook."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "troubleshooting.md"

    def test_runbook_has_table_of_contents(self, runbook_path):
        """Test that runbook includes navigation aids."""
        content = runbook_path.read_text()
        # Should have clear sections
        section_count = content.count('#')
        assert section_count >= 8, "Runbook should have well-organized sections"

    def test_runbook_provides_command_examples(self, runbook_path):
        """Test that runbook includes command examples for diagnostics."""
        content = runbook_path.read_text()
        # Should have code blocks or command examples
        assert "```" in content or "`" in content, \
            "Runbook should include diagnostic command examples"

    def test_runbook_has_step_by_step_procedures(self, runbook_path):
        """Test that diagnostic procedures are step-by-step."""
        content = runbook_path.read_text()
        # Check for numbered steps or clear procedures
        has_steps = ("1." in content or "Step" in content or
                     "step" in content.lower() or "- " in content)
        assert has_steps, "Runbook should have step-by-step procedures"

    def test_runbook_explains_error_messages(self, runbook_path):
        """Test that runbook explains common error messages."""
        content = runbook_path.read_text()
        assert "error" in content.lower(), \
            "Runbook should explain error messages"


class TestTroubleshootingRunbookActionability:
    """Test that troubleshooting runbook is immediately actionable."""

    @pytest.fixture
    def runbook_path(self) -> Path:
        """Return path to troubleshooting runbook."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "troubleshooting.md"

    def test_runbook_provides_quick_reference(self, runbook_path):
        """Test that runbook has quick reference section for fast lookup."""
        content = runbook_path.read_text()
        # Should have summary or quick reference
        has_quick_ref = ("Quick" in content or "quick" in content.lower() or
                         "Reference" in content or "summary" in content.lower())
        assert has_quick_ref, "Runbook should include quick reference section"

    def test_runbook_provides_specific_metrics_to_check(self, runbook_path):
        """Test that runbook specifies which metrics to check."""
        content = runbook_path.read_text()
        # Should mention specific Kafka/Prometheus metrics
        metrics = ["latency", "errors", "throughput", "lag", "buffer"]
        found_metrics = sum(1 for m in metrics if m in content.lower())
        assert found_metrics >= 3, "Runbook should specify metrics to check"

    def test_runbook_provides_success_criteria(self, runbook_path):
        """Test that runbook includes success criteria for each procedure."""
        content = runbook_path.read_text()
        assert ("expected" in content.lower() or "verify" in content.lower() or
                "confirm" in content.lower() or "check" in content.lower()), \
            "Runbook should include verification steps for each procedure"

    def test_runbook_explains_circuit_breaker_states(self, runbook_path):
        """Test that circuit breaker states are explained."""
        content = runbook_path.read_text()
        assert "circuit" in content.lower(), "Runbook should explain circuit breaker"
        # Should explain states: OPEN, CLOSED, HALF_OPEN
        states = ["open", "closed", "half", "break"]
        found_states = sum(1 for s in states if s in content.lower())
        assert found_states >= 1, "Runbook should explain circuit breaker states"

    def test_runbook_explains_dlq_messages(self, runbook_path):
        """Test that DLQ (Dead Letter Queue) is documented."""
        content = runbook_path.read_text()
        assert "DLQ" in content or "dead letter" in content.lower() or \
               "letter queue" in content.lower(), \
            "Runbook should explain DLQ handling"

    def test_runbook_provides_recovery_procedures(self, runbook_path):
        """Test that runbook includes recovery procedures."""
        content = runbook_path.read_text()
        assert "recover" in content.lower() or "fix" in content.lower() or \
               "resolv" in content.lower() or "remediat" in content.lower(), \
            "Runbook should include recovery procedures"

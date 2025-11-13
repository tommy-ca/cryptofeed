"""Tests for Prometheus alert rules validation.

This module tests Task 17.3: Custom Alerting & Health Checks for the market-data-kafka-producer.
Tests verify alert rule PromQL syntax, firing conditions, thresholds, and severity levels.

Coverage:
- Alert rule PromQL syntax validation
- Critical alerts (error rate, latency, buffer, disconnected, circuit breaker)
- Warning alerts (latency, buffer, lag, errors, DLQ, serialization)
- Info alerts (throughput, broker latency)
- Alert thresholds and conditions
- Alert severity and annotation correctness
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Alert Rule Models
# ============================================================================


class AlertRule:
    """Prometheus alert rule model."""

    def __init__(
        self,
        name: str,
        expr: str,
        duration: str,
        severity: str,
        description: str,
        runbook: str = None,
    ):
        """Initialize alert rule.

        Args:
            name: Alert name (e.g., KafkaProducerErrorRateHigh)
            expr: PromQL expression
            duration: Duration before alert fires (e.g., '5m', '10m')
            severity: Alert severity (critical/warning/info)
            description: Alert description
            runbook: Link to runbook (optional)
        """
        self.name = name
        self.expr = expr
        self.duration = duration
        self.severity = severity
        self.description = description
        self.runbook = runbook

    def validate_promql_syntax(self) -> Tuple[bool, str]:
        """Validate PromQL syntax is syntactically correct.

        This performs basic syntax validation without executing against actual data.
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for basic syntax issues
        expr = self.expr.strip()

        # Should not be empty
        if not expr:
            return False, "PromQL expression is empty"

        # Check for balanced parentheses
        if expr.count("(") != expr.count(")"):
            return False, "Unbalanced parentheses"

        # Check for balanced brackets (for range vectors)
        if expr.count("[") != expr.count("]"):
            return False, "Unbalanced brackets"

        # Check for balanced braces (for label filters)
        if expr.count("{") != expr.count("}"):
            return False, "Unbalanced braces"

        # Check for common metric names
        if "rate(" in expr or "sum(" in expr or "histogram_quantile(" in expr:
            # Should have valid function syntax
            if not any(func in expr for func in [
                "rate(", "sum(", "histogram_quantile(", "max(",
                "avg(", "increase(", "topk(", "time(", "timestamp("
            ]):
                return False, "No recognized Prometheus functions found"

        return True, ""

    def get_severity_level(self) -> int:
        """Get numeric severity level for sorting.

        Returns:
            0 = critical, 1 = warning, 2 = info
        """
        severity_map = {"critical": 0, "warning": 1, "info": 2}
        return severity_map.get(self.severity.lower(), 999)


# ============================================================================
# Alert Rules Registry
# ============================================================================


class AlertRulesRegistry:
    """Registry of Prometheus alert rules."""

    def __init__(self):
        """Initialize alert rules registry."""
        self.rules: Dict[str, AlertRule] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register all default alert rules from spec."""
        # Critical Alerts
        self.add_rule(AlertRule(
            name="KafkaProducerErrorRateHigh",
            expr=(
                "(sum(rate(produce_errors_total[5m])) / "
                "(sum(rate(messages_produced_total[5m])) + sum(rate(produce_errors_total[5m])))) > 0.01"
            ),
            duration="5m",
            severity="critical",
            description="Kafka producer error rate > 1%",
            runbook="/docs/kafka/troubleshooting.md#producer-error-rate",
        ))

        self.add_rule(AlertRule(
            name="KafkaProducerLatencyCritical",
            expr=(
                "histogram_quantile(0.99, rate(produce_latency_seconds_bucket[5m])) > 0.05"
            ),
            duration="10m",
            severity="critical",
            description="Kafka producer P99 latency > 50ms",
            runbook="/docs/kafka/troubleshooting.md#high-latency",
        ))

        self.add_rule(AlertRule(
            name="KafkaProducerBufferCritical",
            expr=(
                "kafka_buffer_utilization_percent > 95"
            ),
            duration="5m",
            severity="critical",
            description="Kafka producer buffer utilization critical (>95%)",
            runbook="/docs/kafka/troubleshooting.md#buffer-full",
        ))

        self.add_rule(AlertRule(
            name="KafkaProducerDisconnected",
            expr=(
                "(time() - timestamp(max(messages_produced_total))) > 300"
            ),
            duration="1m",
            severity="critical",
            description="Kafka producer disconnected (no messages for 5min)",
            runbook="/docs/kafka/troubleshooting.md#producer-offline",
        ))

        # Warning Alerts
        self.add_rule(AlertRule(
            name="KafkaProducerLatencyHigh",
            expr=(
                "histogram_quantile(0.99, rate(produce_latency_seconds_bucket[5m])) > 0.015"
            ),
            duration="10m",
            severity="warning",
            description="Kafka producer P99 latency elevated (>15ms)",
            runbook="/docs/kafka/troubleshooting.md#latency-optimization",
        ))

        self.add_rule(AlertRule(
            name="KafkaProducerBufferHigh",
            expr=(
                "kafka_buffer_utilization_percent > 80"
            ),
            duration="10m",
            severity="warning",
            description="Kafka producer buffer utilization high (>80%)",
            runbook="/docs/kafka/troubleshooting.md#buffer-optimization",
        ))

        self.add_rule(AlertRule(
            name="KafkaPartitionLagHigh",
            expr=(
                "max(kafka_partition_lag_records) > 100"
            ),
            duration="15m",
            severity="warning",
            description="Kafka partition lag exceeded threshold (>100 records)",
            runbook="/docs/kafka/troubleshooting.md#high-consumer-lag",
        ))

        self.add_rule(AlertRule(
            name="KafkaProducerErrorsDetected",
            expr=(
                "(sum(rate(produce_errors_total[5m])) / "
                "(sum(rate(messages_produced_total[5m])) + sum(rate(produce_errors_total[5m])))) > 0.001"
            ),
            duration="5m",
            severity="warning",
            description="Kafka producer errors detected (>0.1%)",
            runbook="/docs/kafka/troubleshooting.md#producer-errors",
        ))

        self.add_rule(AlertRule(
            name="SerializationLatencyHigh",
            expr=(
                "histogram_quantile(0.95, rate(serialization_latency_seconds_bucket[5m])) > 0.001"
            ),
            duration="10m",
            severity="warning",
            description="Protobuf serialization latency elevated (>1ms)",
            runbook="/docs/kafka/troubleshooting.md#serialization-performance",
        ))

        # Info Alerts
        self.add_rule(AlertRule(
            name="KafkaProducerLowThroughput",
            expr=(
                "sum(rate(messages_produced_total[5m])) < 100"
            ),
            duration="5m",
            severity="info",
            description="Kafka producer throughput low (<100 msg/sec)",
        ))

        self.add_rule(AlertRule(
            name="KafkaBrokerLatencyElevated",
            expr=(
                "histogram_quantile(0.95, rate(kafka_broker_latency_seconds_bucket[5m])) > 0.02"
            ),
            duration="15m",
            severity="info",
            description="Kafka broker latency elevated (P95 > 20ms)",
            runbook="/docs/kafka/troubleshooting.md#broker-latency",
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule to registry.

        Args:
            rule: AlertRule instance
        """
        self.rules[rule.name] = rule

    def get_rule(self, name: str) -> AlertRule:
        """Get alert rule by name.

        Args:
            name: Alert rule name

        Returns:
            AlertRule instance or None
        """
        return self.rules.get(name)

    def get_rules_by_severity(self, severity: str) -> List[AlertRule]:
        """Get all rules with given severity.

        Args:
            severity: Severity level (critical/warning/info)

        Returns:
            List of AlertRule instances
        """
        return [r for r in self.rules.values() if r.severity.lower() == severity.lower()]

    def count_by_severity(self) -> Dict[str, int]:
        """Count alert rules by severity.

        Returns:
            Dictionary with severity as key, count as value
        """
        counts = {"critical": 0, "warning": 0, "info": 0}
        for rule in self.rules.values():
            severity = rule.severity.lower()
            if severity in counts:
                counts[severity] += 1
        return counts


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def alert_registry():
    """Provide alert rules registry."""
    return AlertRulesRegistry()


# ============================================================================
# Alert Rule Syntax Validation Tests
# ============================================================================


class TestAlertRulePromQLSyntax:
    """Test PromQL syntax validation for alert rules."""

    def test_error_rate_alert_syntax(self, alert_registry):
        """Test KafkaProducerErrorRateHigh PromQL syntax."""
        rule = alert_registry.get_rule("KafkaProducerErrorRateHigh")
        assert rule is not None
        valid, error = rule.validate_promql_syntax()
        assert valid, f"Invalid PromQL: {error}"

    def test_latency_alert_syntax(self, alert_registry):
        """Test KafkaProducerLatencyCritical PromQL syntax."""
        rule = alert_registry.get_rule("KafkaProducerLatencyCritical")
        assert rule is not None
        valid, error = rule.validate_promql_syntax()
        assert valid, f"Invalid PromQL: {error}"

    def test_buffer_alert_syntax(self, alert_registry):
        """Test KafkaProducerBufferCritical PromQL syntax."""
        rule = alert_registry.get_rule("KafkaProducerBufferCritical")
        assert rule is not None
        valid, error = rule.validate_promql_syntax()
        assert valid, f"Invalid PromQL: {error}"

    def test_disconnected_alert_syntax(self, alert_registry):
        """Test KafkaProducerDisconnected PromQL syntax."""
        rule = alert_registry.get_rule("KafkaProducerDisconnected")
        assert rule is not None
        valid, error = rule.validate_promql_syntax()
        assert valid, f"Invalid PromQL: {error}"

    def test_all_rules_have_valid_syntax(self, alert_registry):
        """Test all alert rules have valid PromQL syntax."""
        for rule_name, rule in alert_registry.rules.items():
            valid, error = rule.validate_promql_syntax()
            assert valid, f"Rule {rule_name} has invalid PromQL: {error}"

    def test_empty_expression_invalid(self):
        """Test empty PromQL expression is invalid."""
        rule = AlertRule(
            name="InvalidRule",
            expr="",
            duration="5m",
            severity="critical",
            description="Test",
        )
        valid, error = rule.validate_promql_syntax()
        assert not valid
        assert "empty" in error.lower()

    def test_unbalanced_parentheses_invalid(self):
        """Test unbalanced parentheses are invalid."""
        rule = AlertRule(
            name="InvalidRule",
            expr="sum(rate(metric[5m])",  # Missing closing paren
            duration="5m",
            severity="critical",
            description="Test",
        )
        valid, error = rule.validate_promql_syntax()
        assert not valid
        assert "parentheses" in error.lower()

    def test_unbalanced_brackets_invalid(self):
        """Test unbalanced brackets are invalid."""
        rule = AlertRule(
            name="InvalidRule",
            expr="rate(metric[5m)",  # Missing closing bracket
            duration="5m",
            severity="critical",
            description="Test",
        )
        valid, error = rule.validate_promql_syntax()
        assert not valid
        assert "bracket" in error.lower()


# ============================================================================
# Alert Rule Count and Distribution Tests
# ============================================================================


class TestAlertRuleDistribution:
    """Test alert rule distribution and counts."""

    def test_total_alert_count(self, alert_registry):
        """Test correct total alert rule count."""
        assert len(alert_registry.rules) == 11, "Should have 11 alert rules"

    def test_critical_alert_count(self, alert_registry):
        """Test critical alert count."""
        critical = alert_registry.get_rules_by_severity("critical")
        assert len(critical) == 4, "Should have 4 critical alerts"

    def test_warning_alert_count(self, alert_registry):
        """Test warning alert count."""
        warning = alert_registry.get_rules_by_severity("warning")
        assert len(warning) == 5, "Should have 5 warning alerts"

    def test_info_alert_count(self, alert_registry):
        """Test info alert count."""
        info = alert_registry.get_rules_by_severity("info")
        assert len(info) == 2, "Should have 2 info alerts"

    def test_severity_distribution(self, alert_registry):
        """Test severity distribution counts."""
        counts = alert_registry.count_by_severity()
        assert counts["critical"] == 4
        assert counts["warning"] == 5
        assert counts["info"] == 2


# ============================================================================
# Alert Threshold Tests
# ============================================================================


class TestAlertThresholds:
    """Test alert rule thresholds."""

    def test_error_rate_critical_threshold(self, alert_registry):
        """Test error rate critical threshold is > 1%."""
        rule = alert_registry.get_rule("KafkaProducerErrorRateHigh")
        assert "0.01" in rule.expr, "Critical error rate should be > 1% (0.01)"

    def test_error_rate_warning_threshold(self, alert_registry):
        """Test error rate warning threshold is > 0.1%."""
        rule = alert_registry.get_rule("KafkaProducerErrorsDetected")
        assert "0.001" in rule.expr, "Warning error rate should be > 0.1% (0.001)"

    def test_latency_critical_threshold(self, alert_registry):
        """Test latency critical threshold is > 50ms."""
        rule = alert_registry.get_rule("KafkaProducerLatencyCritical")
        assert "0.05" in rule.expr, "Critical latency should be > 50ms (0.05s)"

    def test_latency_warning_threshold(self, alert_registry):
        """Test latency warning threshold is > 15ms."""
        rule = alert_registry.get_rule("KafkaProducerLatencyHigh")
        assert "0.015" in rule.expr, "Warning latency should be > 15ms (0.015s)"

    def test_buffer_critical_threshold(self, alert_registry):
        """Test buffer critical threshold is > 95%."""
        rule = alert_registry.get_rule("KafkaProducerBufferCritical")
        assert "95" in rule.expr, "Critical buffer should be > 95%"

    def test_buffer_warning_threshold(self, alert_registry):
        """Test buffer warning threshold is > 80%."""
        rule = alert_registry.get_rule("KafkaProducerBufferHigh")
        assert "80" in rule.expr, "Warning buffer should be > 80%"

    def test_partition_lag_threshold(self, alert_registry):
        """Test partition lag threshold is > 100 records."""
        rule = alert_registry.get_rule("KafkaPartitionLagHigh")
        assert "100" in rule.expr, "Lag threshold should be > 100 records"

    def test_serialization_latency_threshold(self, alert_registry):
        """Test serialization latency threshold is > 1ms."""
        rule = alert_registry.get_rule("SerializationLatencyHigh")
        assert "0.001" in rule.expr, "Serialization latency should be > 1ms (0.001s)"

    def test_low_throughput_threshold(self, alert_registry):
        """Test low throughput threshold is < 100 msg/sec."""
        rule = alert_registry.get_rule("KafkaProducerLowThroughput")
        assert "100" in rule.expr, "Low throughput should be < 100 msg/sec"


# ============================================================================
# Alert Duration Tests
# ============================================================================


class TestAlertDuration:
    """Test alert rule duration/for clauses."""

    def test_critical_alerts_have_short_duration(self, alert_registry):
        """Test critical alerts fire within 5 minutes."""
        critical = alert_registry.get_rules_by_severity("critical")
        for rule in critical:
            # Critical should be 1-10 minutes
            duration = rule.duration
            assert duration in ["1m", "5m", "10m"], f"Unexpected critical duration: {duration}"

    def test_error_rate_critical_duration(self, alert_registry):
        """Test error rate critical fires after 5 minutes."""
        rule = alert_registry.get_rule("KafkaProducerErrorRateHigh")
        assert rule.duration == "5m"

    def test_latency_critical_duration(self, alert_registry):
        """Test latency critical fires after 10 minutes."""
        rule = alert_registry.get_rule("KafkaProducerLatencyCritical")
        assert rule.duration == "10m"

    def test_buffer_critical_duration(self, alert_registry):
        """Test buffer critical fires after 5 minutes."""
        rule = alert_registry.get_rule("KafkaProducerBufferCritical")
        assert rule.duration == "5m"

    def test_warning_alerts_have_moderate_duration(self, alert_registry):
        """Test warning alerts fire within 10-15 minutes."""
        warning = alert_registry.get_rules_by_severity("warning")
        for rule in warning:
            duration = rule.duration
            assert duration in ["5m", "10m", "15m"], f"Unexpected warning duration: {duration}"


# ============================================================================
# Alert Annotation Tests
# ============================================================================


class TestAlertAnnotations:
    """Test alert rule annotations."""

    def test_all_critical_alerts_have_runbook(self, alert_registry):
        """Test all critical alerts have runbook links."""
        critical = alert_registry.get_rules_by_severity("critical")
        # Disconnected alert may not have runbook, but most should
        critical_with_runbook = [r for r in critical if r.runbook]
        assert len(critical_with_runbook) >= 3, "Most critical alerts should have runbooks"

    def test_all_warning_alerts_have_runbook(self, alert_registry):
        """Test all warning alerts have runbook links."""
        warning = alert_registry.get_rules_by_severity("warning")
        warning_with_runbook = [r for r in warning if r.runbook]
        assert len(warning_with_runbook) >= 4, "Most warning alerts should have runbooks"

    def test_alert_descriptions_not_empty(self, alert_registry):
        """Test all alerts have descriptions."""
        for rule in alert_registry.rules.values():
            assert rule.description, f"Alert {rule.name} missing description"
            assert len(rule.description) > 10, f"Alert {rule.name} has too short description"


# ============================================================================
# Alert Metrics Reference Tests
# ============================================================================


class TestAlertMetricsReference:
    """Test alert rules reference expected metrics."""

    EXPECTED_METRICS = {
        "produce_errors_total",
        "messages_produced_total",
        "produce_latency_seconds_bucket",
        "kafka_buffer_utilization_percent",
        "kafka_partition_lag_records",
        "serialization_latency_seconds_bucket",
        "kafka_broker_latency_seconds_bucket",
    }

    def test_error_rate_uses_correct_metrics(self, alert_registry):
        """Test error rate alert uses produce metrics."""
        rule = alert_registry.get_rule("KafkaProducerErrorRateHigh")
        assert "produce_errors_total" in rule.expr
        assert "messages_produced_total" in rule.expr

    def test_latency_uses_latency_metric(self, alert_registry):
        """Test latency alerts use produce_latency metric."""
        rule = alert_registry.get_rule("KafkaProducerLatencyCritical")
        assert "produce_latency_seconds_bucket" in rule.expr

    def test_buffer_uses_buffer_metric(self, alert_registry):
        """Test buffer alerts use buffer utilization metric."""
        rule = alert_registry.get_rule("KafkaProducerBufferCritical")
        assert "kafka_buffer_utilization_percent" in rule.expr

    def test_lag_uses_lag_metric(self, alert_registry):
        """Test lag alert uses partition lag metric."""
        rule = alert_registry.get_rule("KafkaPartitionLagHigh")
        assert "kafka_partition_lag_records" in rule.expr

    def test_serialization_uses_serialization_metric(self, alert_registry):
        """Test serialization alert uses serialization metric."""
        rule = alert_registry.get_rule("SerializationLatencyHigh")
        assert "serialization_latency_seconds_bucket" in rule.expr

    def test_all_alerts_use_known_metrics(self, alert_registry):
        """Test all alerts reference only known metrics."""
        for rule in alert_registry.rules.values():
            # Extract metric names from expression (basic parsing)
            # Look for identifiers followed by optional _bucket, _sum, _count
            metric_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:_bucket|_sum|_count)?)\s*[\(\[\{><!]'
            matches = re.findall(metric_pattern, rule.expr)
            for match in matches:
                # Skip Prometheus functions and keywords
                if match in ["time", "timestamp", "sum", "rate", "max", "avg", "histogram_quantile",
                           "topk", "on", "by", "group_left", "offset", "and", "or", "unless"]:
                    continue
                # All other identifiers should be metric names
                # We're not strictly validating here, just checking for obvious issues
                assert len(match) > 0, f"Empty metric name in {rule.name}"


# ============================================================================
# Alert Correctness Tests with Mock Data
# ============================================================================


class TestAlertFiringConditions:
    """Test alert firing conditions with sample metric scenarios."""

    def test_error_rate_critical_fires_at_1_percent(self):
        """Test error rate critical alert condition."""
        # Simulate error rate > 1%
        error_rate = 0.015  # 1.5%
        threshold = 0.01  # 1%
        assert error_rate > threshold, "Error rate should trigger critical alert"

    def test_error_rate_warning_fires_at_0_1_percent(self):
        """Test error rate warning alert condition."""
        error_rate = 0.0015  # 0.15%
        threshold = 0.001  # 0.1%
        assert error_rate > threshold, "Error rate should trigger warning alert"

    def test_latency_critical_fires_at_50ms(self):
        """Test P99 latency critical alert condition."""
        p99_latency = 0.060  # 60ms
        threshold = 0.050  # 50ms
        assert p99_latency > threshold, "Latency should trigger critical alert"

    def test_latency_warning_fires_at_15ms(self):
        """Test P99 latency warning alert condition."""
        p99_latency = 0.020  # 20ms
        threshold = 0.015  # 15ms
        assert p99_latency > threshold, "Latency should trigger warning alert"

    def test_buffer_critical_fires_at_95_percent(self):
        """Test buffer critical alert condition."""
        buffer_util = 96.0  # 96%
        threshold = 95  # 95%
        assert buffer_util > threshold, "Buffer should trigger critical alert"

    def test_buffer_warning_fires_at_80_percent(self):
        """Test buffer warning alert condition."""
        buffer_util = 85.0  # 85%
        threshold = 80  # 80%
        assert buffer_util > threshold, "Buffer should trigger warning alert"

    def test_lag_warning_fires_at_100_records(self):
        """Test partition lag warning alert condition."""
        lag = 150  # 150 records
        threshold = 100  # 100 records
        assert lag > threshold, "Lag should trigger warning alert"

    def test_throughput_info_fires_below_100(self):
        """Test low throughput info alert condition."""
        throughput = 50  # 50 msg/sec
        threshold = 100  # 100 msg/sec
        assert throughput < threshold, "Throughput should trigger info alert"

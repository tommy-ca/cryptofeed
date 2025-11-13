"""
Task 26: Production Stability Monitoring Tests (TDD Approach)

Tests for per-exchange metric tracking, anomaly detection, daily reporting,
and escalation procedures during Week 3 of Phase 5 migration.

Test Strategy:
1. Per-exchange metric collection (6 metrics)
2. Anomaly detection algorithms
3. Daily stability report generation
4. Escalation logic (Severity 1-3)
5. 3-day rollback window management
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# 1. Per-Exchange Metric Collection Tests
# ============================================================================


@dataclass
class PerExchangeMetrics:
    """6 key metrics tracked per exchange during migration."""
    exchange_name: str
    timestamp: datetime

    # Metric 1: Consumer lag (seconds)
    consumer_lag_seconds: float

    # Metric 2: Error rate (percent)
    error_rate_percent: float

    # Metric 3: Throughput (messages/second)
    throughput_msg_per_sec: float

    # Metric 4: Latency percentiles (milliseconds)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    # Metric 5: Message loss (percent)
    message_loss_percent: float

    # Metric 6: Data integrity (hash match percent)
    data_integrity_percent: float

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "exchange": self.exchange_name,
            "timestamp": self.timestamp,
            "consumer_lag_seconds": self.consumer_lag_seconds,
            "error_rate_percent": self.error_rate_percent,
            "throughput_msg_per_sec": self.throughput_msg_per_sec,
            "latency_percentiles": {
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
            },
            "message_loss_percent": self.message_loss_percent,
            "data_integrity_percent": self.data_integrity_percent,
        }


class TestPerExchangeMetricCollection:
    """Test per-exchange metric collection."""

    def test_metric_collection_initialization(self):
        """Test metric collection initializes."""
        metrics = PerExchangeMetrics(
            exchange_name="coinbase",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        assert metrics.exchange_name == "coinbase"
        assert metrics.consumer_lag_seconds == 2.5
        assert metrics.error_rate_percent == 0.05

    def test_metric_summary_report(self):
        """Test metric summary report."""
        now = datetime.utcnow()
        metrics = PerExchangeMetrics(
            exchange_name="binance",
            timestamp=now,
            consumer_lag_seconds=1.8,
            error_rate_percent=0.08,
            throughput_msg_per_sec=150000,
            latency_p50_ms=0.9,
            latency_p95_ms=2.8,
            latency_p99_ms=4.2,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        summary = metrics.get_summary()
        assert summary["exchange"] == "binance"
        assert summary["timestamp"] == now
        assert summary["consumer_lag_seconds"] == 1.8
        assert summary["latency_percentiles"]["p99_ms"] == 4.2

    def test_multiple_exchange_metrics(self):
        """Test collecting metrics for multiple exchanges."""
        exchanges = ["coinbase", "binance", "kraken", "okx"]
        metrics_list = []

        for i, exchange in enumerate(exchanges):
            metrics = PerExchangeMetrics(
                exchange_name=exchange,
                timestamp=datetime.utcnow(),
                consumer_lag_seconds=2.0 + (i * 0.5),
                error_rate_percent=0.05,
                throughput_msg_per_sec=120000 + (i * 10000),
                latency_p50_ms=1.0,
                latency_p95_ms=3.0,
                latency_p99_ms=4.5,
                message_loss_percent=0.0,
                data_integrity_percent=100.0
            )
            metrics_list.append(metrics)

        assert len(metrics_list) == 4
        assert metrics_list[0].exchange_name == "coinbase"
        assert metrics_list[3].exchange_name == "okx"


# ============================================================================
# 2. Anomaly Detection Tests
# ============================================================================


class AnomalyDetector:
    """Detects anomalies in per-exchange metrics."""

    def __init__(self):
        self.thresholds = {
            "consumer_lag_seconds": 10.0,  # Alert if >10s
            "error_rate_percent": 0.5,     # Alert if >0.5%
            "throughput_msg_per_sec": 50000,  # Alert if <50k
            "latency_p99_ms": 10.0,         # Alert if >10ms
            "message_loss_percent": 0.1,    # Alert if >0.1%
            "data_integrity_percent": 99.5, # Alert if <99.5%
        }

    def detect_anomalies(self, metrics: PerExchangeMetrics) -> List[str]:
        """Detect anomalies in metrics."""
        anomalies = []

        if metrics.consumer_lag_seconds > self.thresholds["consumer_lag_seconds"]:
            anomalies.append(
                f"Consumer lag {metrics.consumer_lag_seconds}s exceeds "
                f"{self.thresholds['consumer_lag_seconds']}s threshold"
            )

        if metrics.error_rate_percent > self.thresholds["error_rate_percent"]:
            anomalies.append(
                f"Error rate {metrics.error_rate_percent}% exceeds "
                f"{self.thresholds['error_rate_percent']}% threshold"
            )

        if metrics.throughput_msg_per_sec < self.thresholds["throughput_msg_per_sec"]:
            anomalies.append(
                f"Throughput {metrics.throughput_msg_per_sec} msg/s below "
                f"{self.thresholds['throughput_msg_per_sec']} threshold"
            )

        if metrics.latency_p99_ms > self.thresholds["latency_p99_ms"]:
            anomalies.append(
                f"Latency p99 {metrics.latency_p99_ms}ms exceeds "
                f"{self.thresholds['latency_p99_ms']}ms threshold"
            )

        if metrics.message_loss_percent > self.thresholds["message_loss_percent"]:
            anomalies.append(
                f"Message loss {metrics.message_loss_percent}% exceeds "
                f"{self.thresholds['message_loss_percent']}% threshold"
            )

        if metrics.data_integrity_percent < self.thresholds["data_integrity_percent"]:
            anomalies.append(
                f"Data integrity {metrics.data_integrity_percent}% below "
                f"{self.thresholds['data_integrity_percent']}% threshold"
            )

        return anomalies


class TestAnomalyDetection:
    """Test anomaly detection algorithms."""

    def test_no_anomalies_when_healthy(self):
        """Test no anomalies detected when metrics healthy."""
        detector = AnomalyDetector()
        metrics = PerExchangeMetrics(
            exchange_name="coinbase",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        anomalies = detector.detect_anomalies(metrics)
        assert len(anomalies) == 0

    def test_detect_consumer_lag_anomaly(self):
        """Test detecting consumer lag anomaly."""
        detector = AnomalyDetector()
        metrics = PerExchangeMetrics(
            exchange_name="binance",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=15.0,  # EXCEEDS 10s THRESHOLD
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        anomalies = detector.detect_anomalies(metrics)
        assert len(anomalies) == 1
        assert "Consumer lag" in anomalies[0]

    def test_detect_error_rate_anomaly(self):
        """Test detecting error rate anomaly."""
        detector = AnomalyDetector()
        metrics = PerExchangeMetrics(
            exchange_name="kraken",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=1.0,  # EXCEEDS 0.5% THRESHOLD
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        anomalies = detector.detect_anomalies(metrics)
        assert len(anomalies) == 1
        assert "Error rate" in anomalies[0]

    def test_detect_multiple_anomalies(self):
        """Test detecting multiple anomalies."""
        detector = AnomalyDetector()
        metrics = PerExchangeMetrics(
            exchange_name="okx",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=20.0,  # EXCEEDS THRESHOLD
            error_rate_percent=1.5,      # EXCEEDS THRESHOLD
            throughput_msg_per_sec=30000, # BELOW THRESHOLD
            latency_p50_ms=1.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        anomalies = detector.detect_anomalies(metrics)
        assert len(anomalies) == 3


# ============================================================================
# 3. Daily Stability Report Tests
# ============================================================================


@dataclass
class DailyStabilityReport:
    """Daily report of per-exchange metrics and anomalies."""
    report_date: datetime
    exchange_metrics: Dict[str, PerExchangeMetrics] = field(default_factory=dict)
    anomalies_by_exchange: Dict[str, List[str]] = field(default_factory=dict)
    alert_history: List[Dict[str, Any]] = field(default_factory=list)
    migration_status: str = ""

    def add_metrics(self, metrics: PerExchangeMetrics) -> None:
        """Add metrics for an exchange."""
        self.exchange_metrics[metrics.exchange_name] = metrics
        self.anomalies_by_exchange[metrics.exchange_name] = []

    def add_anomalies(self, exchange_name: str, anomalies: List[str]) -> None:
        """Add anomalies for an exchange."""
        self.anomalies_by_exchange[exchange_name] = anomalies
        for anomaly in anomalies:
            self.alert_history.append({
                "exchange": exchange_name,
                "anomaly": anomaly,
                "timestamp": datetime.utcnow()
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get report summary."""
        total_exchanges = len(self.exchange_metrics)
        healthy_exchanges = sum(
            1 for anomalies in self.anomalies_by_exchange.values()
            if len(anomalies) == 0
        )

        return {
            "report_date": self.report_date,
            "total_exchanges": total_exchanges,
            "healthy_exchanges": healthy_exchanges,
            "exchanges_with_issues": total_exchanges - healthy_exchanges,
            "total_alerts": len(self.alert_history),
            "metrics_by_exchange": {
                name: metrics.get_summary()
                for name, metrics in self.exchange_metrics.items()
            },
            "anomalies": self.anomalies_by_exchange,
            "migration_status": self.migration_status
        }

    def is_stable(self) -> bool:
        """Check if system is stable (no anomalies)."""
        return all(
            len(anomalies) == 0
            for anomalies in self.anomalies_by_exchange.values()
        )


class TestDailyStabilityReport:
    """Test daily stability report generation."""

    def test_report_initialization(self):
        """Test report initialization."""
        now = datetime.utcnow()
        report = DailyStabilityReport(report_date=now)

        assert report.report_date == now
        assert len(report.exchange_metrics) == 0
        assert len(report.anomalies_by_exchange) == 0

    def test_add_metrics_and_generate_report(self):
        """Test adding metrics and generating report."""
        report = DailyStabilityReport(report_date=datetime.utcnow())
        detector = AnomalyDetector()

        # Add metrics for 3 exchanges
        for i, exchange in enumerate(["coinbase", "binance", "kraken"]):
            metrics = PerExchangeMetrics(
                exchange_name=exchange,
                timestamp=datetime.utcnow(),
                consumer_lag_seconds=2.0 + (i * 0.5),
                error_rate_percent=0.05,
                throughput_msg_per_sec=125000,
                latency_p50_ms=1.0,
                latency_p95_ms=3.0,
                latency_p99_ms=4.5,
                message_loss_percent=0.0,
                data_integrity_percent=100.0
            )
            report.add_metrics(metrics)
            anomalies = detector.detect_anomalies(metrics)
            report.add_anomalies(exchange, anomalies)

        summary = report.get_summary()
        assert summary["total_exchanges"] == 3
        assert summary["healthy_exchanges"] == 3
        assert summary["exchanges_with_issues"] == 0
        assert report.is_stable() is True

    def test_report_with_anomalies(self):
        """Test report when anomalies detected."""
        report = DailyStabilityReport(report_date=datetime.utcnow())
        detector = AnomalyDetector()

        # Add metrics with anomaly
        metrics_bad = PerExchangeMetrics(
            exchange_name="okx",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=15.0,  # ANOMALY
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )
        report.add_metrics(metrics_bad)
        anomalies = detector.detect_anomalies(metrics_bad)
        report.add_anomalies("okx", anomalies)

        # Add healthy metrics
        metrics_good = PerExchangeMetrics(
            exchange_name="bybit",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )
        report.add_metrics(metrics_good)
        anomalies = detector.detect_anomalies(metrics_good)
        report.add_anomalies("bybit", anomalies)

        summary = report.get_summary()
        assert summary["total_exchanges"] == 2
        assert summary["healthy_exchanges"] == 1
        assert summary["exchanges_with_issues"] == 1
        assert report.is_stable() is False


# ============================================================================
# 4. Escalation Logic Tests (Severity 1-3)
# ============================================================================


class SeverityLevel(Enum):
    """Escalation severity levels."""
    SEVERITY_1 = "severity_1"  # Pause migrations, investigate
    SEVERITY_2 = "severity_2"  # Monitor closely, may resume
    SEVERITY_3 = "severity_3"  # Informational, continue


@dataclass
class EscalationDecision:
    """Escalation decision based on metrics."""
    exchange_name: str
    severity: SeverityLevel
    reason: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_summary(self) -> Dict[str, Any]:
        """Get escalation summary."""
        return {
            "exchange": self.exchange_name,
            "severity": self.severity.value,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp
        }


class EscalationEngine:
    """Determines escalation level based on metrics."""

    def evaluate(self, metrics: PerExchangeMetrics,
                 anomalies: List[str]) -> Optional[EscalationDecision]:
        """Evaluate if escalation needed."""

        # Severity 1: Critical issues (pause migrations)
        if metrics.consumer_lag_seconds > 20.0:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_1,
                reason=f"Consumer lag {metrics.consumer_lag_seconds}s critically high",
                recommended_action="Pause migrations immediately, investigate root cause"
            )

        if metrics.error_rate_percent > 1.0:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_1,
                reason=f"Error rate {metrics.error_rate_percent}% critically high",
                recommended_action="Pause migrations, review error logs"
            )

        if metrics.message_loss_percent > 0.1:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_1,
                reason=f"Message loss {metrics.message_loss_percent}% detected",
                recommended_action="Initiate data recovery procedure"
            )

        # Severity 2: High-risk issues (monitor closely)
        if metrics.consumer_lag_seconds > 10.0:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_2,
                reason=f"Consumer lag {metrics.consumer_lag_seconds}s elevated",
                recommended_action="Monitor closely, prepare rollback if worsens"
            )

        if metrics.error_rate_percent > 0.5:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_2,
                reason=f"Error rate {metrics.error_rate_percent}% elevated",
                recommended_action="Increase monitoring frequency, review error patterns"
            )

        if metrics.latency_p99_ms > 10.0:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_2,
                reason=f"Latency p99 {metrics.latency_p99_ms}ms elevated",
                recommended_action="Monitor latency trend, check network/broker health"
            )

        # Severity 3: Informational (continue monitoring)
        if len(anomalies) > 0:
            return EscalationDecision(
                exchange_name=metrics.exchange_name,
                severity=SeverityLevel.SEVERITY_3,
                reason=f"Minor anomalies detected: {', '.join(anomalies[:2])}",
                recommended_action="Continue normal monitoring, no action required"
            )

        return None


class TestEscalationLogic:
    """Test escalation logic (Severity 1-3)."""

    def test_severity_1_critical_lag(self):
        """Test Severity 1 escalation for critical lag."""
        engine = EscalationEngine()
        metrics = PerExchangeMetrics(
            exchange_name="coinbase",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=30.0,  # CRITICAL
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        decision = engine.evaluate(metrics, [])
        assert decision is not None
        assert decision.severity == SeverityLevel.SEVERITY_1
        assert "Pause migrations" in decision.recommended_action

    def test_severity_1_message_loss(self):
        """Test Severity 1 escalation for message loss."""
        engine = EscalationEngine()
        metrics = PerExchangeMetrics(
            exchange_name="binance",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.5,  # CRITICAL
            data_integrity_percent=100.0
        )

        decision = engine.evaluate(metrics, [])
        assert decision is not None
        assert decision.severity == SeverityLevel.SEVERITY_1
        assert "data recovery" in decision.recommended_action

    def test_severity_2_elevated_lag(self):
        """Test Severity 2 escalation for elevated lag."""
        engine = EscalationEngine()
        metrics = PerExchangeMetrics(
            exchange_name="kraken",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=12.0,  # ELEVATED
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        decision = engine.evaluate(metrics, [])
        assert decision is not None
        assert decision.severity == SeverityLevel.SEVERITY_2
        assert "prepare rollback" in decision.recommended_action

    def test_severity_3_minor_anomaly(self):
        """Test Severity 3 escalation for minor anomalies."""
        engine = EscalationEngine()
        metrics = PerExchangeMetrics(
            exchange_name="okx",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=8.0,
            error_rate_percent=0.08,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        decision = engine.evaluate(metrics, ["Minor anomaly"])
        assert decision is not None
        assert decision.severity == SeverityLevel.SEVERITY_3
        assert "no action required" in decision.recommended_action

    def test_no_escalation_when_healthy(self):
        """Test no escalation when metrics healthy."""
        engine = EscalationEngine()
        metrics = PerExchangeMetrics(
            exchange_name="bybit",
            timestamp=datetime.utcnow(),
            consumer_lag_seconds=2.5,
            error_rate_percent=0.05,
            throughput_msg_per_sec=125000,
            latency_p50_ms=1.0,
            latency_p95_ms=3.0,
            latency_p99_ms=4.5,
            message_loss_percent=0.0,
            data_integrity_percent=100.0
        )

        decision = engine.evaluate(metrics, [])
        assert decision is None


# ============================================================================
# 5. 3-Day Rollback Window Management Tests
# ============================================================================


@dataclass
class RollbackWindow:
    """3-day rollback window for each exchange."""
    exchange_name: str
    migration_started_at: datetime
    rollback_deadline: Optional[datetime] = None

    def __post_init__(self):
        """Calculate rollback deadline (3 days from migration start)."""
        if self.rollback_deadline is None:
            self.rollback_deadline = self.migration_started_at + timedelta(days=3)

    def is_rollback_available(self) -> bool:
        """Check if rollback still available."""
        return datetime.utcnow() <= self.rollback_deadline

    def time_remaining_hours(self) -> float:
        """Get remaining rollback time in hours."""
        if not self.is_rollback_available():
            return 0.0

        remaining = self.rollback_deadline - datetime.utcnow()
        return remaining.total_seconds() / 3600

    def get_summary(self) -> Dict[str, Any]:
        """Get rollback window summary."""
        return {
            "exchange": self.exchange_name,
            "migration_started_at": self.migration_started_at,
            "rollback_deadline": self.rollback_deadline,
            "rollback_available": self.is_rollback_available(),
            "time_remaining_hours": self.time_remaining_hours()
        }


class TestRollbackWindow:
    """Test 3-day rollback window management."""

    def test_rollback_window_initialization(self):
        """Test rollback window initializes correctly."""
        start_time = datetime.utcnow()
        window = RollbackWindow(
            exchange_name="coinbase",
            migration_started_at=start_time
        )

        assert window.exchange_name == "coinbase"
        assert window.migration_started_at == start_time
        # Deadline should be 3 days later
        expected_deadline = start_time + timedelta(days=3)
        assert (window.rollback_deadline - expected_deadline).total_seconds() < 1

    def test_rollback_available_within_window(self):
        """Test rollback available within 3-day window."""
        start_time = datetime.utcnow() - timedelta(hours=12)
        window = RollbackWindow(
            exchange_name="binance",
            migration_started_at=start_time
        )

        assert window.is_rollback_available() is True
        assert window.time_remaining_hours() > 36

    def test_rollback_unavailable_after_window(self):
        """Test rollback unavailable after 3-day window."""
        # Simulate migration 4 days ago
        start_time = datetime.utcnow() - timedelta(days=4)
        window = RollbackWindow(
            exchange_name="kraken",
            migration_started_at=start_time
        )

        assert window.is_rollback_available() is False
        assert window.time_remaining_hours() == 0.0

    def test_rollback_window_near_deadline(self):
        """Test rollback window near deadline."""
        # Simulate migration started 2.9 days ago
        start_time = datetime.utcnow() - timedelta(days=2.9)
        window = RollbackWindow(
            exchange_name="okx",
            migration_started_at=start_time
        )

        assert window.is_rollback_available() is True
        remaining = window.time_remaining_hours()
        assert 0 < remaining < 3  # Less than 3 hours remaining


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestTask26EndToEnd:
    """End-to-end tests for Task 26: Production Stability Monitoring."""

    def test_daily_monitoring_workflow(self):
        """Test complete daily monitoring workflow."""
        report = DailyStabilityReport(report_date=datetime.utcnow())
        detector = AnomalyDetector()
        engine = EscalationEngine()

        # Simulate monitoring 5 exchanges
        exchanges_data = [
            ("coinbase", 2.5, 0.05, 125000, 4.5),
            ("binance", 3.0, 0.06, 130000, 4.2),
            ("kraken", 12.0, 0.08, 120000, 5.5),   # ELEVATED LAG (>10s)
            ("okx", 2.8, 0.05, 128000, 4.8),
            ("bybit", 2.2, 0.04, 135000, 3.9),
        ]

        escalations = []

        for exchange_name, lag, error_rate, throughput, p99 in exchanges_data:
            metrics = PerExchangeMetrics(
                exchange_name=exchange_name,
                timestamp=datetime.utcnow(),
                consumer_lag_seconds=lag,
                error_rate_percent=error_rate,
                throughput_msg_per_sec=throughput,
                latency_p50_ms=1.0,
                latency_p95_ms=3.0,
                latency_p99_ms=p99,
                message_loss_percent=0.0,
                data_integrity_percent=100.0
            )

            report.add_metrics(metrics)
            anomalies = detector.detect_anomalies(metrics)
            report.add_anomalies(exchange_name, anomalies)

            decision = engine.evaluate(metrics, anomalies)
            if decision:
                escalations.append(decision)

        # Verify report
        summary = report.get_summary()
        assert summary["total_exchanges"] == 5
        assert summary["healthy_exchanges"] == 4
        assert summary["exchanges_with_issues"] == 1

        # Verify escalations
        assert len(escalations) == 1
        assert escalations[0].exchange_name == "kraken"
        assert escalations[0].severity == SeverityLevel.SEVERITY_2

    def test_rollback_window_tracking_per_exchange(self):
        """Test 3-day rollback window tracking per exchange."""
        windows = {}

        # Start migration for 3 exchanges at different times
        # Coinbase started 2 days ago, Binance 1 day ago, Kraken just now
        for i, exchange in enumerate(["coinbase", "binance", "kraken"]):
            hours_ago = 48 - (24*i)  # 48, 24, 0
            start_time = datetime.utcnow() - timedelta(hours=hours_ago)
            windows[exchange] = RollbackWindow(
                exchange_name=exchange,
                migration_started_at=start_time
            )

        # Verify all still have rollback available
        for exchange, window in windows.items():
            assert window.is_rollback_available() is True
            remaining = window.time_remaining_hours()
            assert remaining > 0

        # Verify time remaining decreases with age
        # Oldest (coinbase) has least time remaining
        coinbase_remaining = windows["coinbase"].time_remaining_hours()
        binance_remaining = windows["binance"].time_remaining_hours()
        kraken_remaining = windows["kraken"].time_remaining_hours()

        # Coinbase (oldest) < Binance < Kraken (newest)
        assert coinbase_remaining < binance_remaining < kraken_remaining


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

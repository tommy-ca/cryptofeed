"""
Test suite for deprecation timeline tracking and communication system.

Tests Requirements 7.1, 7.2, 7.3, 7.5 from kafka-backend-maintenance spec.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json


class TestDeprecationTimeline:
    """Tests for timeline milestone tracking and validation."""

    def test_timeline_has_required_milestones(self):
        """Timeline must include all critical migration milestones."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

        timeline = DeprecationTimeline.load_default()

        # Must have these milestone phases
        required_phases = [
            "deprecation_warnings",
            "migration_tools",
            "documentation",
            "monitoring",
            "shim_removal",
            "legacy_cleanup",
        ]

        for phase in required_phases:
            assert phase in timeline.milestones, f"Missing milestone: {phase}"

    def test_milestones_have_target_dates(self):
        """Each milestone must have a target completion date."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

        timeline = DeprecationTimeline.load_default()

        for milestone_name, milestone in timeline.milestones.items():
            assert milestone.target_date is not None, f"{milestone_name} missing target date"
            assert isinstance(milestone.target_date, datetime)

    def test_milestones_ordered_chronologically(self):
        """Milestones must be in chronological order."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

        timeline = DeprecationTimeline.load_default()

        dates = [m.target_date for m in timeline.milestones.values()]
        assert dates == sorted(dates), "Milestones not in chronological order"

    def test_milestone_progress_tracking(self):
        """Milestones can track completion progress."""
        from cryptofeed.backends.kafka.deprecation import Milestone

        milestone = Milestone(
            name="test_phase",
            description="Test milestone",
            target_date=datetime.now() + timedelta(days=14),
            status="in_progress",
            completion_percentage=50,
        )

        assert milestone.status == "in_progress"
        assert milestone.completion_percentage == 50
        assert not milestone.is_complete

        milestone.mark_complete()
        assert milestone.is_complete
        assert milestone.completion_percentage == 100

    def test_timeline_validation_warns_on_overdue(self):
        """Timeline validation should identify overdue milestones."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline, Milestone

        past_date = datetime.now() - timedelta(days=7)
        timeline = DeprecationTimeline(
            milestones={
                "overdue_phase": Milestone(
                    name="overdue_phase",
                    description="Past due milestone",
                    target_date=past_date,
                    status="pending",
                    completion_percentage=0,
                )
            }
        )

        validation_result = timeline.validate()
        # Warnings are non-fatal, only errors make validation invalid
        assert validation_result.is_valid
        assert len(validation_result.warnings) > 0
        assert "overdue" in validation_result.warnings[0].lower()

    def test_timeline_export_to_markdown(self):
        """Timeline can be exported as markdown for documentation."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

        timeline = DeprecationTimeline.load_default()
        markdown = timeline.to_markdown()

        assert "# Kafka Backend Deprecation Timeline" in markdown
        assert "Phase" in markdown
        assert "Target Date" in markdown
        assert "Status" in markdown

    def test_timeline_persists_to_file(self):
        """Timeline can be saved and loaded from file."""
        from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

        timeline = DeprecationTimeline.load_default()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            timeline.save(f.name)
            saved_path = f.name

        try:
            loaded = DeprecationTimeline.load(saved_path)
            assert len(loaded.milestones) == len(timeline.milestones)
            for key in timeline.milestones:
                assert key in loaded.milestones
        finally:
            Path(saved_path).unlink()


class TestCommunicationSystem:
    """Tests for multi-channel timeline update communication."""

    def test_communication_channels_registered(self):
        """Communication system supports multiple notification channels."""
        from cryptofeed.backends.kafka.deprecation import CommunicationSystem

        comm = CommunicationSystem()

        # Should support these channels
        channels = comm.get_registered_channels()
        assert "documentation" in channels
        assert "release_notes" in channels
        assert "deprecation_warnings" in channels

    def test_send_timeline_update_to_documentation(self):
        """Timeline updates can be sent to documentation channel."""
        from cryptofeed.backends.kafka.deprecation import CommunicationSystem, TimelineUpdate

        comm = CommunicationSystem()
        update = TimelineUpdate(
            milestone_name="deprecation_warnings",
            old_status="pending",
            new_status="complete",
            message="Deprecation warnings implemented and tested",
            timestamp=datetime.now(),
        )

        result = comm.send_update(update, channels=["documentation"])
        assert result.success
        assert "documentation" in result.channels_notified

    def test_send_timeline_update_to_all_channels(self):
        """Timeline updates can be broadcast to all channels."""
        from cryptofeed.backends.kafka.deprecation import CommunicationSystem, TimelineUpdate

        comm = CommunicationSystem()
        update = TimelineUpdate(
            milestone_name="shim_removal",
            old_status="pending",
            new_status="in_progress",
            message="Compatibility shim removal has begun",
            timestamp=datetime.now(),
        )

        result = comm.send_update(update, channels="all")
        assert result.success
        assert len(result.channels_notified) >= 3

    def test_communication_history_tracked(self):
        """Communication system maintains history of sent updates."""
        from cryptofeed.backends.kafka.deprecation import CommunicationSystem, TimelineUpdate

        comm = CommunicationSystem()
        update1 = TimelineUpdate(
            milestone_name="phase1",
            old_status="pending",
            new_status="in_progress",
            message="Phase 1 started",
            timestamp=datetime.now(),
        )
        update2 = TimelineUpdate(
            milestone_name="phase1",
            old_status="in_progress",
            new_status="complete",
            message="Phase 1 completed",
            timestamp=datetime.now() + timedelta(days=7),
        )

        comm.send_update(update1, channels=["documentation"])
        comm.send_update(update2, channels=["documentation"])

        history = comm.get_history()
        assert len(history) == 2
        assert history[0].milestone_name == "phase1"
        assert history[1].new_status == "complete"


class TestDecisionLog:
    """Tests for architectural decision records (ADR) for Kafka backend."""

    def test_decision_log_creates_adr(self):
        """Decision log can create new architectural decision records."""
        from cryptofeed.backends.kafka.deprecation import DecisionLog, DecisionRecord

        log = DecisionLog()
        decision = DecisionRecord(
            id="ADR-001",
            title="Deprecate Legacy Kafka Backend",
            status="accepted",
            context="Legacy backend lacks protobuf support and modern features",
            decision="Deprecate legacy backend in favor of modular implementation",
            consequences=["Users must migrate", "Maintenance burden reduced"],
            date=datetime.now(),
        )

        log.add_decision(decision)
        assert log.get_decision("ADR-001") == decision

    def test_decision_log_exports_to_markdown(self):
        """Decision records can be exported as markdown ADR documents."""
        from cryptofeed.backends.kafka.deprecation import DecisionLog, DecisionRecord

        log = DecisionLog()
        decision = DecisionRecord(
            id="ADR-002",
            title="Remove Compatibility Shim",
            status="proposed",
            context="Shim adds confusion and maintenance overhead",
            decision="Remove kafka_callback.py after 90-day migration period",
            consequences=["Clean import paths", "Breaking change for slow adopters"],
            date=datetime.now(),
        )
        log.add_decision(decision)

        markdown = log.to_markdown()
        assert "# ADR-002" in markdown
        assert "Remove Compatibility Shim" in markdown
        assert "Status:" in markdown or "## Status" in markdown
        assert "proposed" in markdown
        assert "## Context" in markdown
        assert "## Decision" in markdown
        assert "## Consequences" in markdown

    def test_decision_log_lists_all_decisions(self):
        """Decision log can list all recorded decisions."""
        from cryptofeed.backends.kafka.deprecation import DecisionLog, DecisionRecord

        log = DecisionLog()
        log.add_decision(
            DecisionRecord(
                id="ADR-001",
                title="Decision 1",
                status="accepted",
                context="Context 1",
                decision="Decision 1",
                consequences=[],
                date=datetime.now(),
            )
        )
        log.add_decision(
            DecisionRecord(
                id="ADR-002",
                title="Decision 2",
                status="accepted",
                context="Context 2",
                decision="Decision 2",
                consequences=[],
                date=datetime.now(),
            )
        )

        decisions = log.list_decisions()
        assert len(decisions) == 2
        assert decisions[0].id == "ADR-001"
        assert decisions[1].id == "ADR-002"

    def test_decision_log_persists_to_directory(self):
        """Decision log can save ADRs to a directory structure."""
        from cryptofeed.backends.kafka.deprecation import DecisionLog, DecisionRecord

        log = DecisionLog()
        log.add_decision(
            DecisionRecord(
                id="ADR-003",
                title="Test Decision",
                status="accepted",
                context="Test context",
                decision="Test decision",
                consequences=["Test consequence"],
                date=datetime.now(),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log.save_to_directory(tmpdir)
            adr_file = Path(tmpdir) / "ADR-003-test-decision.md"
            assert adr_file.exists()

            content = adr_file.read_text()
            assert "# ADR-003" in content
            assert "Test Decision" in content


class TestProgressReporting:
    """Tests for usage statistics and migration progress reporting."""

    def test_progress_report_tracks_legacy_usage(self):
        """Progress reporting tracks legacy backend usage metrics."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        report.record_legacy_usage("TradeKafka", context={"exchange": "coinbase"})
        report.record_legacy_usage("BookKafka", context={"exchange": "binance"})

        stats = report.get_legacy_usage_stats()
        assert stats["total_legacy_usage"] == 2
        assert "TradeKafka" in stats["classes_used"]
        assert "BookKafka" in stats["classes_used"]

    def test_progress_report_tracks_modern_usage(self):
        """Progress reporting tracks modern backend adoption."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        report.record_modern_usage("KafkaCallback", context={"exchange": "coinbase"})
        report.record_modern_usage("KafkaProtobufCallback", context={"exchange": "kraken"})

        stats = report.get_modern_usage_stats()
        assert stats["total_modern_usage"] == 2
        assert "KafkaCallback" in stats["classes_used"]

    def test_progress_report_calculates_migration_percentage(self):
        """Progress report calculates migration completion percentage."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        # 20% modern, 80% legacy = 20% migrated
        for _ in range(8):
            report.record_legacy_usage("TradeKafka", context={})
        for _ in range(2):
            report.record_modern_usage("KafkaCallback", context={})

        migration_pct = report.get_migration_percentage()
        assert migration_pct == pytest.approx(20.0, abs=0.1)

    def test_progress_report_exports_to_markdown(self):
        """Progress report can be exported as markdown."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        report.record_legacy_usage("TradeKafka", context={})
        report.record_modern_usage("KafkaCallback", context={})

        markdown = report.to_markdown()
        assert "# Kafka Backend Migration Progress" in markdown
        assert "Legacy Usage" in markdown
        assert "Modern Usage" in markdown
        assert "Migration Percentage" in markdown

    def test_progress_report_generates_timeline_update_recommendation(self):
        """Progress report can recommend timeline adjustments based on adoption."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        # Simulate slow adoption (95% still on legacy)
        for _ in range(95):
            report.record_legacy_usage("TradeKafka", context={})
        for _ in range(5):
            report.record_modern_usage("KafkaCallback", context={})

        recommendation = report.get_timeline_recommendation()
        assert recommendation.should_extend_timeline
        assert recommendation.recommended_extension_days > 0
        assert "slow adoption" in recommendation.reason.lower()

    def test_progress_report_exports_to_json(self):
        """Progress report can be exported as JSON for automation."""
        from cryptofeed.backends.kafka.deprecation import ProgressReport

        report = ProgressReport()
        report.record_legacy_usage("TradeKafka", context={})
        report.record_modern_usage("KafkaCallback", context={})

        json_data = report.to_json()
        parsed = json.loads(json_data)
        assert "legacy_usage" in parsed
        assert "modern_usage" in parsed
        assert "migration_percentage" in parsed

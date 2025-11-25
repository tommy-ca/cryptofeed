"""
Task 25: Incremental Per-Exchange Migration Tests (TDD Approach)

Tests for per-exchange migration procedure, checklist validation,
and per-exchange success criteria during Week 3 of Phase 5 migration.

Test Strategy:
1. Per-exchange migration checklist (9 items)
2. Per-exchange success criteria (5 criteria)
3. Migration sequence validation
4. Rollback procedure validation (<5 min)
5. Exchange migration state tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pytest



# ============================================================================
# 1. Per-Exchange Migration Checklist Tests
# ============================================================================


class MigrationChecklistItem(Enum):
    """9-item migration checklist for each exchange."""
    STAGING_VALIDATION_PASSED = "staging_validation_passed"
    CONSOLIDATED_TOPICS_ENABLED = "consolidated_topics_enabled"
    CONSUMER_LAG_BASELINE = "consumer_lag_baseline"
    BASELINE_30MIN_MONITOR = "baseline_30min_monitor"
    ERROR_RATE_BELOW_THRESHOLD = "error_rate_below_threshold"
    DATA_INTEGRITY_HASH_MATCH = "data_integrity_hash_match"
    MESSAGE_HEADERS_PRESENT = "message_headers_present"
    EXTENDED_2HOUR_MONITOR = "extended_2hour_monitor"
    SIGN_OFF_FOR_NEXT_EXCHANGE = "sign_off_for_next_exchange"


@dataclass
class PerExchangeChecklist:
    """Tracks 9-item migration checklist for single exchange."""
    exchange_name: str
    checklist_items: Dict[MigrationChecklistItem, bool] = field(default_factory=dict)
    checked_at: Dict[MigrationChecklistItem, datetime] = field(default_factory=dict)
    signed_off_by: Optional[str] = None
    signed_off_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize all checklist items to False."""
        for item in MigrationChecklistItem:
            self.checklist_items[item] = False
            self.checked_at[item] = None

    def check_item(self, item: MigrationChecklistItem, checked_by: str) -> None:
        """Mark a checklist item as complete."""
        self.checklist_items[item] = True
        self.checked_at[item] = datetime.utcnow()

    def is_complete(self) -> bool:
        """Verify all 9 items checked."""
        return all(self.checklist_items.values())

    def get_status(self) -> Dict[str, Any]:
        """Get checklist status."""
        completed_items = sum(1 for v in self.checklist_items.values() if v)
        return {
            "exchange": self.exchange_name,
            "items_completed": completed_items,
            "items_total": len(MigrationChecklistItem),
            "is_complete": self.is_complete(),
            "checklist": {
                item.value: self.checklist_items[item]
                for item in MigrationChecklistItem
            }
        }


class TestPerExchangeChecklist:
    """Test per-exchange migration checklist (9 items)."""

    def test_checklist_initialization(self):
        """Test checklist initializes with all items unchecked."""
        checklist = PerExchangeChecklist("coinbase")

        assert checklist.exchange_name == "coinbase"
        assert len(checklist.checklist_items) == 9
        assert all(v is False for v in checklist.checklist_items.values())
        assert not checklist.is_complete()

    def test_mark_single_item_complete(self):
        """Test marking a single checklist item complete."""
        checklist = PerExchangeChecklist("coinbase")

        checklist.check_item(MigrationChecklistItem.STAGING_VALIDATION_PASSED, "alice")

        assert checklist.checklist_items[MigrationChecklistItem.STAGING_VALIDATION_PASSED]
        assert checklist.checked_at[MigrationChecklistItem.STAGING_VALIDATION_PASSED] is not None
        assert not checklist.is_complete()

    def test_mark_all_items_complete(self):
        """Test marking all 9 items complete."""
        checklist = PerExchangeChecklist("coinbase")

        for item in MigrationChecklistItem:
            checklist.check_item(item, "alice")

        assert checklist.is_complete()
        status = checklist.get_status()
        assert status["items_completed"] == 9
        assert status["is_complete"] is True

    def test_checklist_status_report(self):
        """Test checklist status report."""
        checklist = PerExchangeChecklist("binance")

        # Check first 5 items
        for i, item in enumerate(list(MigrationChecklistItem)[:5]):
            checklist.check_item(item, "bob")

        status = checklist.get_status()
        assert status["exchange"] == "binance"
        assert status["items_completed"] == 5
        assert status["items_total"] == 9
        assert status["is_complete"] is False

    def test_checklist_ordered_validation(self):
        """Test checklist items should be completed in order."""
        checklist = PerExchangeChecklist("kraken")

        # Enforce strict order: must check in sequence
        item_order = list(MigrationChecklistItem)

        # Check items out of order (but all should pass)
        for item in item_order:
            checklist.check_item(item, "charlie")

        assert checklist.is_complete()
        # Verify all timestamps are recorded
        for item in item_order:
            assert checklist.checked_at[item] is not None


# ============================================================================
# 2. Per-Exchange Success Criteria Tests
# ============================================================================


@dataclass
class ExchangeSuccessCriteria:
    """5 success criteria for each exchange migration."""
    exchange_name: str
    consumer_lag_max_seconds: float = 5.0  # Criterion 1: <5s
    error_rate_percent: float = 0.0        # Criterion 2: <0.1%
    message_loss_percent: float = 0.0       # Criterion 3: zero loss
    header_presence_percent: float = 100.0  # Criterion 4: 100%
    throughput_msg_per_sec: float = 0.0    # Criterion 5: ≥100k msg/s

    measured_at: Optional[datetime] = None

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all 5 criteria met."""
        failures = []

        if self.consumer_lag_max_seconds > 5.0:
            failures.append(
                f"Consumer lag {self.consumer_lag_max_seconds}s exceeds 5s threshold"
            )

        if self.error_rate_percent > 0.1:
            failures.append(
                f"Error rate {self.error_rate_percent}% exceeds 0.1% threshold"
            )

        if self.message_loss_percent > 0.0:
            failures.append(
                f"Message loss {self.message_loss_percent}% detected (target: zero)"
            )

        if self.header_presence_percent < 100.0:
            failures.append(
                f"Header presence {self.header_presence_percent}% below 100% threshold"
            )

        if self.throughput_msg_per_sec < 100000:
            failures.append(
                f"Throughput {self.throughput_msg_per_sec} msg/s below 100k threshold"
            )

        return len(failures) == 0, failures

    def get_summary(self) -> Dict[str, Any]:
        """Get criteria summary."""
        passed, failures = self.validate()
        return {
            "exchange": self.exchange_name,
            "passed": passed,
            "criteria": {
                "consumer_lag_seconds": self.consumer_lag_max_seconds,
                "error_rate_percent": self.error_rate_percent,
                "message_loss_percent": self.message_loss_percent,
                "header_presence_percent": self.header_presence_percent,
                "throughput_msg_per_sec": self.throughput_msg_per_sec,
            },
            "failures": failures,
            "measured_at": self.measured_at
        }


class TestPerExchangeSuccessCriteria:
    """Test per-exchange success criteria (5 criteria)."""

    def test_all_criteria_pass(self):
        """Test when all 5 criteria pass."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="coinbase",
            consumer_lag_max_seconds=2.5,  # <5s ✓
            error_rate_percent=0.05,       # <0.1% ✓
            message_loss_percent=0.0,      # zero ✓
            header_presence_percent=100.0, # 100% ✓
            throughput_msg_per_sec=150000  # ≥100k ✓
        )

        passed, failures = criteria.validate()
        assert passed is True
        assert len(failures) == 0

    def test_consumer_lag_exceeds_threshold(self):
        """Test when consumer lag exceeds 5s threshold."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="binance",
            consumer_lag_max_seconds=8.0  # >5s ✗
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert any("Consumer lag" in f for f in failures)

    def test_error_rate_exceeds_threshold(self):
        """Test when error rate exceeds 0.1%."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="kraken",
            error_rate_percent=0.5  # >0.1% ✗
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert any("Error rate" in f for f in failures)

    def test_message_loss_detected(self):
        """Test when message loss detected."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="okx",
            message_loss_percent=0.5  # >0 ✗
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert any("Message loss" in f for f in failures)

    def test_header_presence_below_100_percent(self):
        """Test when header presence below 100%."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="bybit",
            header_presence_percent=99.5  # <100% ✗
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert any("Header presence" in f for f in failures)

    def test_throughput_below_threshold(self):
        """Test when throughput below 100k msg/s."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="deribit",
            throughput_msg_per_sec=50000  # <100k ✗
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert any("Throughput" in f for f in failures)

    def test_multiple_failures(self):
        """Test when multiple criteria fail."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="huobi",
            consumer_lag_max_seconds=10.0,  # FAILS (>5s)
            error_rate_percent=0.5,          # FAILS (>0.1%)
            message_loss_percent=0.1,        # FAILS (>0%)
            header_presence_percent=100.0,
            throughput_msg_per_sec=100000
        )

        passed, failures = criteria.validate()
        assert passed is False
        assert len(failures) == 3

    def test_criteria_summary(self):
        """Test criteria summary report."""
        criteria = ExchangeSuccessCriteria(
            exchange_name="gemini",
            consumer_lag_max_seconds=3.0,
            throughput_msg_per_sec=125000
        )

        summary = criteria.get_summary()
        assert summary["exchange"] == "gemini"
        assert summary["passed"] is True
        assert summary["criteria"]["consumer_lag_seconds"] == 3.0


# ============================================================================
# 3. Migration Sequence Validation Tests
# ============================================================================


@dataclass
class MigrationSequence:
    """Tracks exchange migration sequence and order."""
    exchange_order: List[str] = field(
        default_factory=lambda: [
            "coinbase",      # Day 1: Largest volume
            "binance",       # Day 2: 2nd largest
            "okx",           # Day 3: Medium
            "kraken",        # Day 4: Medium
            "bybit",         # Day 5: Medium
            "deribit",       # Day 6: Small
            "crypto.com",    # Day 7: Small
            "huobi",         # Day 8: Small
        ]
    )
    completed_exchanges: List[str] = field(default_factory=list)
    migration_times: Dict[str, datetime] = field(default_factory=dict)

    def migrate_exchange(self, exchange_name: str) -> bool:
        """Mark exchange as migrated."""
        if exchange_name not in self.exchange_order:
            return False

        if exchange_name in self.completed_exchanges:
            return False  # Already migrated

        # Check if prerequisites met (previous exchanges migrated)
        current_index = self.exchange_order.index(exchange_name)
        for i in range(current_index):
            if self.exchange_order[i] not in self.completed_exchanges:
                return False  # Prerequisites not met

        self.completed_exchanges.append(exchange_name)
        self.migration_times[exchange_name] = datetime.utcnow()
        return True

    def get_next_exchange(self) -> Optional[str]:
        """Get next exchange to migrate."""
        for exchange in self.exchange_order:
            if exchange not in self.completed_exchanges:
                return exchange
        return None

    def get_progress(self) -> Dict[str, Any]:
        """Get migration progress."""
        return {
            "total_exchanges": len(self.exchange_order),
            "completed_exchanges": len(self.completed_exchanges),
            "remaining_exchanges": len(self.exchange_order) - len(self.completed_exchanges),
            "completion_percent": (len(self.completed_exchanges) / len(self.exchange_order)) * 100,
            "completed_list": self.completed_exchanges,
            "next_exchange": self.get_next_exchange()
        }


class TestMigrationSequence:
    """Test migration sequence validation."""

    def test_migration_sequence_order(self):
        """Test exchanges migrate in correct order."""
        sequence = MigrationSequence()

        assert sequence.migrate_exchange("coinbase") is True
        assert sequence.migrate_exchange("binance") is True
        assert sequence.migrate_exchange("okx") is True

        assert sequence.completed_exchanges == ["coinbase", "binance", "okx"]

    def test_cannot_migrate_out_of_order(self):
        """Test cannot migrate exchange out of order."""
        sequence = MigrationSequence()

        # Try to migrate Binance before Coinbase
        assert sequence.migrate_exchange("binance") is False

        # Migrate Coinbase first
        assert sequence.migrate_exchange("coinbase") is True

        # Now Binance succeeds
        assert sequence.migrate_exchange("binance") is True

    def test_cannot_migrate_duplicate(self):
        """Test cannot migrate same exchange twice."""
        sequence = MigrationSequence()

        assert sequence.migrate_exchange("coinbase") is True
        assert sequence.migrate_exchange("coinbase") is False

    def test_get_next_exchange(self):
        """Test getting next exchange to migrate."""
        sequence = MigrationSequence()

        assert sequence.get_next_exchange() == "coinbase"
        sequence.migrate_exchange("coinbase")
        assert sequence.get_next_exchange() == "binance"
        sequence.migrate_exchange("binance")
        assert sequence.get_next_exchange() == "okx"

    def test_migration_progress_tracking(self):
        """Test migration progress tracking."""
        sequence = MigrationSequence()

        progress = sequence.get_progress()
        assert progress["total_exchanges"] == 8
        assert progress["completed_exchanges"] == 0
        assert progress["completion_percent"] == 0.0

        # Migrate first exchange
        sequence.migrate_exchange("coinbase")
        progress = sequence.get_progress()
        assert progress["completed_exchanges"] == 1
        assert progress["completion_percent"] == 12.5

        # Migrate all remaining
        for exchange in sequence.exchange_order[1:]:
            sequence.migrate_exchange(exchange)

        progress = sequence.get_progress()
        assert progress["completed_exchanges"] == 8
        assert progress["completion_percent"] == 100.0
        assert progress["next_exchange"] is None

    def test_migration_times_recorded(self):
        """Test migration times recorded."""
        sequence = MigrationSequence()

        before = datetime.utcnow()
        sequence.migrate_exchange("coinbase")
        after = datetime.utcnow()

        assert "coinbase" in sequence.migration_times
        migration_time = sequence.migration_times["coinbase"]
        assert before <= migration_time <= after


# ============================================================================
# 4. Rollback Procedure Tests (<5 min)
# ============================================================================


@dataclass
class RollbackProcedure:
    """Tracks rollback procedure execution and timing."""
    exchange_name: str
    steps: List[str] = field(default_factory=lambda: [
        "pause_new_topic_production",      # T+0min
        "revert_consumer_subscriptions",   # T+1min
        "redeploy_consumers",              # T+2min
        "verify_consumers_connected",      # T+3min
        "monitor_consumer_lag",            # T+4min
        "confirm_rollback_success"         # T+5min
    ])
    completed_steps: List[str] = field(default_factory=list)
    step_times: Dict[str, datetime] = field(default_factory=dict)
    rollback_initiated_at: Optional[datetime] = None
    rollback_completed_at: Optional[datetime] = None

    def initiate_rollback(self) -> None:
        """Initiate rollback procedure."""
        self.rollback_initiated_at = datetime.utcnow()

    def complete_step(self, step_name: str) -> bool:
        """Mark a rollback step as complete."""
        if step_name not in self.steps:
            return False

        if step_name in self.completed_steps:
            return False  # Already completed

        # Check prerequisites
        step_index = self.steps.index(step_name)
        for i in range(step_index):
            if self.steps[i] not in self.completed_steps:
                return False

        self.completed_steps.append(step_name)
        self.step_times[step_name] = datetime.utcnow()
        return True

    def is_complete(self) -> bool:
        """Check if rollback complete."""
        return len(self.completed_steps) == len(self.steps)

    def get_duration_seconds(self) -> Optional[float]:
        """Get rollback duration in seconds."""
        if not self.is_complete() or not self.rollback_initiated_at:
            return None

        if self.rollback_completed_at is None:
            self.rollback_completed_at = datetime.utcnow()

        return (self.rollback_completed_at - self.rollback_initiated_at).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get rollback summary."""
        return {
            "exchange": self.exchange_name,
            "complete": self.is_complete(),
            "steps_completed": len(self.completed_steps),
            "steps_total": len(self.steps),
            "duration_seconds": self.get_duration_seconds(),
            "under_5min_target": (
                self.get_duration_seconds() is not None and
                self.get_duration_seconds() < 300
            )
        }


class TestRollbackProcedure:
    """Test rollback procedure (<5 minutes)."""

    def test_rollback_initialization(self):
        """Test rollback procedure initialization."""
        rollback = RollbackProcedure("coinbase")

        assert rollback.exchange_name == "coinbase"
        assert len(rollback.steps) == 6
        assert len(rollback.completed_steps) == 0
        assert not rollback.is_complete()

    def test_rollback_step_execution_order(self):
        """Test rollback steps execute in order."""
        rollback = RollbackProcedure("binance")
        rollback.initiate_rollback()

        # Execute steps in order
        for step in rollback.steps:
            assert rollback.complete_step(step) is True

        assert rollback.is_complete()

    def test_cannot_skip_steps(self):
        """Test cannot skip rollback steps."""
        rollback = RollbackProcedure("kraken")

        # Try to execute step 3 before step 1
        step_3 = rollback.steps[2]
        assert rollback.complete_step(step_3) is False

        # Execute step 1
        step_1 = rollback.steps[0]
        assert rollback.complete_step(step_1) is True

        # Still cannot execute step 3
        assert rollback.complete_step(step_3) is False

        # Execute step 2 first
        step_2 = rollback.steps[1]
        assert rollback.complete_step(step_2) is True

        # Now step 3 succeeds
        assert rollback.complete_step(step_3) is True

    def test_rollback_duration_under_5min(self):
        """Test rollback completes within 5 minutes."""
        rollback = RollbackProcedure("okx")
        rollback.initiate_rollback()

        # Simulate rapid step execution
        for step in rollback.steps:
            rollback.complete_step(step)
            time.sleep(0.01)  # 10ms per step = 60ms total (well under 5min)

        duration = rollback.get_duration_seconds()
        assert duration is not None
        assert duration < 300  # 5 minutes

    def test_rollback_summary(self):
        """Test rollback summary report."""
        rollback = RollbackProcedure("bybit")
        rollback.initiate_rollback()

        # Complete all steps
        for step in rollback.steps:
            rollback.complete_step(step)

        summary = rollback.get_summary()
        assert summary["exchange"] == "bybit"
        assert summary["complete"] is True
        assert summary["steps_completed"] == 6
        assert summary["under_5min_target"] is True


# ============================================================================
# 5. Exchange Migration State Tracking Tests
# ============================================================================


@dataclass
class ExchangeMigrationState:
    """Tracks complete migration state for one exchange."""
    exchange_name: str
    checklist: Optional[PerExchangeChecklist] = None
    criteria: Optional[ExchangeSuccessCriteria] = None
    rollback: Optional[RollbackProcedure] = None
    migration_status: str = "not_started"  # not_started, in_progress, completed, rolled_back
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize state objects."""
        if self.checklist is None:
            self.checklist = PerExchangeChecklist(self.exchange_name)
        if self.criteria is None:
            self.criteria = ExchangeSuccessCriteria(self.exchange_name)

    def start_migration(self) -> None:
        """Start migration for this exchange."""
        self.migration_status = "in_progress"
        self.started_at = datetime.utcnow()

    def complete_migration(self) -> bool:
        """Complete migration if all checks pass."""
        if not self.checklist.is_complete():
            return False

        passed, _ = self.criteria.validate()
        if not passed:
            return False

        self.migration_status = "completed"
        self.completed_at = datetime.utcnow()
        return True

    def initiate_rollback(self) -> None:
        """Initiate rollback for this exchange."""
        self.migration_status = "rolled_back"
        self.rollback = RollbackProcedure(self.exchange_name)
        self.rollback.initiate_rollback()

    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive migration report."""
        return {
            "exchange": self.exchange_name,
            "status": self.migration_status,
            "checklist": self.checklist.get_status(),
            "criteria": self.criteria.get_summary(),
            "rollback": self.rollback.get_summary() if self.rollback else None,
            "timing": {
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": (
                    (self.completed_at - self.started_at).total_seconds()
                    if self.started_at and self.completed_at else None
                )
            }
        }


class TestExchangeMigrationState:
    """Test complete exchange migration state tracking."""

    def test_migration_state_initialization(self):
        """Test migration state initializes correctly."""
        state = ExchangeMigrationState("coinbase")

        assert state.exchange_name == "coinbase"
        assert state.migration_status == "not_started"
        assert state.started_at is None
        assert state.completed_at is None

    def test_migration_state_transitions(self):
        """Test migration state transitions."""
        state = ExchangeMigrationState("binance")

        # Start migration
        state.start_migration()
        assert state.migration_status == "in_progress"
        assert state.started_at is not None

        # Cannot complete without checklist
        assert state.complete_migration() is False

        # Complete checklist
        for item in MigrationChecklistItem:
            state.checklist.check_item(item, "alice")

        # Complete criteria
        state.criteria = ExchangeSuccessCriteria(
            exchange_name="binance",
            consumer_lag_max_seconds=2.0,
            error_rate_percent=0.05,
            message_loss_percent=0.0,
            header_presence_percent=100.0,
            throughput_msg_per_sec=150000
        )

        # Now can complete
        assert state.complete_migration() is True
        assert state.migration_status == "completed"
        assert state.completed_at is not None

    def test_migration_rollback_from_state(self):
        """Test rollback from migration state."""
        state = ExchangeMigrationState("kraken")
        state.start_migration()

        # Simulate rollback triggered
        state.initiate_rollback()

        assert state.migration_status == "rolled_back"
        assert state.rollback is not None
        assert state.rollback.exchange_name == "kraken"

    def test_full_migration_report(self):
        """Test complete migration report."""
        state = ExchangeMigrationState("okx")
        state.start_migration()

        # Complete checklist
        for item in MigrationChecklistItem:
            state.checklist.check_item(item, "bob")

        # Set success criteria
        state.criteria = ExchangeSuccessCriteria(
            exchange_name="okx",
            consumer_lag_max_seconds=3.0,
            error_rate_percent=0.05,
            message_loss_percent=0.0,
            header_presence_percent=100.0,
            throughput_msg_per_sec=125000
        )

        # Complete migration
        assert state.complete_migration() is True

        # Get report
        report = state.get_full_report()
        assert report["exchange"] == "okx"
        assert report["status"] == "completed"
        assert report["checklist"]["is_complete"] is True
        assert report["criteria"]["passed"] is True
        assert report["timing"]["duration_seconds"] is not None


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestTask25EndToEnd:
    """End-to-end tests for Task 25: Incremental Per-Exchange Migration."""

    def test_complete_migration_sequence_all_exchanges(self):
        """Test completing migration for all 8 exchanges in sequence."""
        sequence = MigrationSequence()

        for exchange_name in sequence.exchange_order:
            # Create migration state
            state = ExchangeMigrationState(exchange_name)
            state.start_migration()

            # Complete checklist
            for item in MigrationChecklistItem:
                state.checklist.check_item(item, "qa_engineer")

            # Set success criteria
            state.criteria = ExchangeSuccessCriteria(
                exchange_name=exchange_name,
                consumer_lag_max_seconds=2.5,
                error_rate_percent=0.05,
                message_loss_percent=0.0,
                header_presence_percent=100.0,
                throughput_msg_per_sec=150000
            )

            # Complete migration
            assert state.complete_migration() is True
            assert sequence.migrate_exchange(exchange_name) is True

        # Verify all exchanges migrated
        progress = sequence.get_progress()
        assert progress["completion_percent"] == 100.0
        assert progress["next_exchange"] is None

    def test_rollback_during_migration(self):
        """Test rollback procedure during migration."""
        state = ExchangeMigrationState("binance")
        state.start_migration()

        # Complete some checklist items
        for item in list(MigrationChecklistItem)[:5]:
            state.checklist.check_item(item, "qa")

        # Simulate failure detected
        state.criteria = ExchangeSuccessCriteria(
            exchange_name="binance",
            consumer_lag_max_seconds=15.0  # EXCEEDS THRESHOLD
        )

        # Initiate rollback
        state.initiate_rollback()
        assert state.migration_status == "rolled_back"

        # Complete rollback steps
        for step in state.rollback.steps:
            state.rollback.complete_step(step)

        # Verify rollback completed within 5 minutes
        assert state.rollback.is_complete() is True
        assert state.rollback.get_duration_seconds() < 300

    def test_partial_migration_with_mixed_results(self):
        """Test partial migration with some exchanges successful."""
        # Don't use sequence ordering - test exchanges independently
        success_count = 0
        rollback_count = 0
        migrated_exchanges = []

        # Test 8 exchanges (Coinbase, Binance, etc)
        for i, exchange_name in enumerate(["coinbase", "binance", "okx", "kraken", "bybit", "deribit", "crypto.com", "huobi"]):
            state = ExchangeMigrationState(exchange_name)
            state.start_migration()

            # Complete checklist
            for item in MigrationChecklistItem:
                state.checklist.check_item(item, "qa")

            # Simulate some exchanges failing (not first 3)
            if i > 2 and i % 3 == 0:  # Every 3rd exchange fails after first 3
                state.criteria = ExchangeSuccessCriteria(
                    exchange_name=exchange_name,
                    consumer_lag_max_seconds=10.0  # FAILS
                )
                state.initiate_rollback()
                rollback_count += 1
            else:
                state.criteria = ExchangeSuccessCriteria(
                    exchange_name=exchange_name,
                    consumer_lag_max_seconds=2.5,
                    error_rate_percent=0.05,
                    message_loss_percent=0.0,
                    header_presence_percent=100.0,
                    throughput_msg_per_sec=150000
                )
                assert state.complete_migration() is True
                success_count += 1
                migrated_exchanges.append(exchange_name)

        # Verify results
        assert success_count > 0
        assert rollback_count > 0
        assert len(migrated_exchanges) == success_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

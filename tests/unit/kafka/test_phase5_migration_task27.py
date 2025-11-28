"""
Task 27: Legacy Topic Archival & Cleanup Tests (TDD Approach)

Tests for legacy topic archival, deletion procedures, backup verification,
and archive manifest creation during Week 4 of Phase 5 migration.

Test Strategy:
1. Backup creation and integrity verification (hash comparison)
2. Deletion prerequisites (no consumers, zero new messages)
3. Dry-run deletion verification
4. Cleanup verification (disk space, partition count reduction)
5. Archive manifest and restoration procedures
6. Audit trail logging
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest

# Silence deprecation warnings from datetime.utcnow in legacy fixtures
warnings.filterwarnings(
    "ignore",
    message=r".*utcnow\(\).*",
    category=DeprecationWarning,
)

# Pytest-level filter for datetime.utcnow deprecation warnings from legacy fixtures
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*utcnow\\(\\) is deprecated.*:DeprecationWarning"
)


# ============================================================================
# 1. Backup Creation and Integrity Tests
# ============================================================================


@dataclass
class BackupManifest:
    """Manifest entry for archived topic."""
    topic_name: str
    message_count: int
    backup_location: str  # S3 path, GCS path, or local path
    backup_size_bytes: int
    checksum_sha256: str  # Integrity verification
    compression_ratio: float  # Original vs compressed
    archived_at: datetime
    retention_days: int = 30


@dataclass
class ArchiveMetadata:
    """Complete archive metadata for all legacy topics."""
    archive_date: datetime
    total_topics: int
    total_messages: int
    total_size_bytes: int
    backup_manifests: List[BackupManifest] = field(default_factory=list)
    restoration_procedure: str = ""
    audit_trail: List[str] = field(default_factory=list)

    def add_backup(self, manifest: BackupManifest) -> None:
        """Add backup entry to archive."""
        self.backup_manifests.append(manifest)
        self.total_messages += manifest.message_count
        self.total_size_bytes += manifest.backup_size_bytes
        self.audit_trail.append(
            f"[{datetime.utcnow().isoformat()}] Archived {manifest.topic_name}: "
            f"{manifest.message_count} messages, {manifest.backup_size_bytes} bytes"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get archive summary."""
        return {
            "archive_date": self.archive_date,
            "total_topics": len(self.backup_manifests),
            "total_messages": self.total_messages,
            "total_size_bytes": self.total_size_bytes,
            "total_size_gb": round(self.total_size_bytes / (1024**3), 2),
            "manifests": [
                {
                    "topic": m.topic_name,
                    "messages": m.message_count,
                    "size_bytes": m.backup_size_bytes,
                    "location": m.backup_location,
                    "checksum": m.checksum_sha256,
                    "compression_ratio": round(m.compression_ratio, 2),
                    "archived_at": m.archived_at.isoformat(),
                    "retention_days": m.retention_days,
                }
                for m in self.backup_manifests
            ],
            "retention_policy": "30-day retention for incident investigation",
            "audit_log": self.audit_trail,
        }


class TestBackupCreation:
    """Test backup creation and integrity verification."""

    def test_backup_manifest_initialization(self):
        """Test backup manifest initializes."""
        manifest = BackupManifest(
            topic_name="cryptofeed.trades.coinbase.btc-usd",
            message_count=1000000,
            backup_location="s3://backups/kafka-legacy/cryptofeed.trades.coinbase.btc-usd.tar.gz",
            backup_size_bytes=63000000,
            checksum_sha256="abc123def456",
            compression_ratio=0.63,
            archived_at=datetime.utcnow(),
        )

        assert manifest.topic_name == "cryptofeed.trades.coinbase.btc-usd"
        assert manifest.message_count == 1000000
        assert manifest.compression_ratio == 0.63

    def test_archive_metadata_initialization(self):
        """Test archive metadata initializes."""
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        assert metadata.total_topics == 0
        assert len(metadata.backup_manifests) == 0

    def test_add_backup_to_archive(self):
        """Test adding backup to archive metadata."""
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        manifest = BackupManifest(
            topic_name="cryptofeed.trades.coinbase.btc-usd",
            message_count=1000000,
            backup_location="s3://backups/kafka-legacy/topic1.tar.gz",
            backup_size_bytes=63000000,
            checksum_sha256="abc123",
            compression_ratio=0.63,
            archived_at=datetime.utcnow(),
        )

        metadata.add_backup(manifest)

        assert len(metadata.backup_manifests) == 1
        assert metadata.total_messages == 1000000
        assert metadata.total_size_bytes == 63000000

    def test_archive_multiple_backups(self):
        """Test archiving multiple topic backups."""
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        topics = ["topic1", "topic2", "topic3"]
        for i, topic in enumerate(topics):
            manifest = BackupManifest(
                topic_name=topic,
                message_count=1000000 + (i * 100000),
                backup_location=f"s3://backups/kafka-legacy/{topic}.tar.gz",
                backup_size_bytes=63000000 + (i * 10000000),
                checksum_sha256=f"hash_{i}",
                compression_ratio=0.63,
                archived_at=datetime.utcnow(),
            )
            metadata.add_backup(manifest)

        summary = metadata.get_summary()
        assert summary["total_topics"] == 3
        assert summary["total_messages"] == 3300000  # 1M + 1.1M + 1.2M
        assert len(summary["manifests"]) == 3

    def test_checksum_integrity_verification(self):
        """Test checksum integrity verification."""
        # Simulate computing checksum for backup
        backup_data = b"message1\nmessage2\nmessage3\n" * 100
        computed_checksum = hashlib.sha256(backup_data).hexdigest()

        manifest = BackupManifest(
            topic_name="cryptofeed.trades.test",
            message_count=300,
            backup_location="s3://backups/test.tar.gz",
            backup_size_bytes=len(backup_data),
            checksum_sha256=computed_checksum,
            compression_ratio=0.70,
            archived_at=datetime.utcnow(),
        )

        # Verify checksum matches
        assert manifest.checksum_sha256 == hashlib.sha256(backup_data).hexdigest()

    def test_audit_trail_logging(self):
        """Test audit trail logging during backup."""
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        manifest1 = BackupManifest(
            topic_name="topic1",
            message_count=1000,
            backup_location="s3://backups/topic1.tar.gz",
            backup_size_bytes=63000,
            checksum_sha256="hash1",
            compression_ratio=0.63,
            archived_at=datetime.utcnow(),
        )
        metadata.add_backup(manifest1)

        manifest2 = BackupManifest(
            topic_name="topic2",
            message_count=2000,
            backup_location="s3://backups/topic2.tar.gz",
            backup_size_bytes=126000,
            checksum_sha256="hash2",
            compression_ratio=0.63,
            archived_at=datetime.utcnow(),
        )
        metadata.add_backup(manifest2)

        assert len(metadata.audit_trail) == 2
        assert "Archived topic1" in metadata.audit_trail[0]
        assert "Archived topic2" in metadata.audit_trail[1]


# ============================================================================
# 2. Deletion Prerequisites Tests
# ============================================================================


@dataclass
class DeletionPrerequisite:
    """Single prerequisite check."""
    check_name: str
    passed: bool
    details: str
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def get_summary(self) -> Dict[str, Any]:
        """Get check summary."""
        return {
            "check": self.check_name,
            "passed": self.passed,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class DeletionPrerequisiteValidator:
    """Validates deletion prerequisites (3-day window)."""
    checks: List[DeletionPrerequisite] = field(default_factory=list)
    last_check_passed: Optional[datetime] = None

    def check_no_active_consumers(self, active_consumers: int) -> DeletionPrerequisite:
        """Verify no active consumers reading legacy topics."""
        passed = active_consumers == 0
        check = DeletionPrerequisite(
            check_name="no_active_consumers",
            passed=passed,
            details=f"Active consumers: {active_consumers} (expected 0)"
        )
        self.checks.append(check)
        return check

    def check_zero_new_messages(self, messages_in_24h: int) -> DeletionPrerequisite:
        """Verify zero new messages in past 24h."""
        passed = messages_in_24h == 0
        check = DeletionPrerequisite(
            check_name="zero_new_messages_24h",
            passed=passed,
            details=f"Messages in 24h: {messages_in_24h} (expected 0)"
        )
        self.checks.append(check)
        return check

    def check_retention_verified(self, backup_count: int) -> DeletionPrerequisite:
        """Verify all data retained in backups."""
        passed = backup_count > 0
        check = DeletionPrerequisite(
            check_name="retention_verified",
            passed=passed,
            details=f"Backups verified: {backup_count} topics archived"
        )
        self.checks.append(check)
        return check

    def check_restoration_procedure_documented(self, doc_exists: bool) -> DeletionPrerequisite:
        """Verify restoration procedure is documented."""
        check = DeletionPrerequisite(
            check_name="restoration_procedure_documented",
            passed=doc_exists,
            details="Restoration procedure available" if doc_exists else "Missing"
        )
        self.checks.append(check)
        return check

    def all_checks_passed(self) -> bool:
        """Check if all prerequisites passed."""
        return all(check.passed for check in self.checks)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checks."""
        return {
            "total_checks": len(self.checks),
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "all_passed": self.all_checks_passed(),
            "checks": [c.get_summary() for c in self.checks],
        }


class TestDeletionPrerequisites:
    """Test deletion prerequisites validation."""

    def test_no_active_consumers_check(self):
        """Test checking for active consumers."""
        validator = DeletionPrerequisiteValidator()
        check = validator.check_no_active_consumers(0)

        assert check.passed is True
        assert "Active consumers: 0" in check.details

    def test_active_consumers_fails_check(self):
        """Test check fails when active consumers present."""
        validator = DeletionPrerequisiteValidator()
        check = validator.check_no_active_consumers(5)

        assert check.passed is False
        assert "Active consumers: 5" in check.details

    def test_zero_new_messages_check(self):
        """Test checking for zero new messages."""
        validator = DeletionPrerequisiteValidator()
        check = validator.check_zero_new_messages(0)

        assert check.passed is True

    def test_new_messages_fails_check(self):
        """Test check fails when new messages detected."""
        validator = DeletionPrerequisiteValidator()
        check = validator.check_zero_new_messages(100)

        assert check.passed is False

    def test_retention_verified_check(self):
        """Test retention verification."""
        validator = DeletionPrerequisiteValidator()
        check = validator.check_retention_verified(10000)

        assert check.passed is True
        assert "10000 topics" in check.details

    def test_all_prerequisites_met(self):
        """Test all prerequisites met."""
        validator = DeletionPrerequisiteValidator()
        validator.check_no_active_consumers(0)
        validator.check_zero_new_messages(0)
        validator.check_retention_verified(10000)
        validator.check_restoration_procedure_documented(True)

        assert validator.all_checks_passed() is True

    def test_one_prerequisite_failed(self):
        """Test one prerequisite fails."""
        validator = DeletionPrerequisiteValidator()
        validator.check_no_active_consumers(0)
        validator.check_zero_new_messages(100)  # FAILS
        validator.check_retention_verified(10000)
        validator.check_restoration_procedure_documented(True)

        assert validator.all_checks_passed() is False
        summary = validator.get_summary()
        assert summary["checks_passed"] == 3


# ============================================================================
# 3. Dry-Run Deletion Tests
# ============================================================================


class DeletionStatus(Enum):
    """Status of deletion operation."""
    PENDING = "pending"
    DRY_RUN_PASSED = "dry_run_passed"
    READY_FOR_ACTUAL = "ready_for_actual"
    ACTUAL_COMPLETED = "actual_completed"


@dataclass
class DeletionOperation:
    """Tracks topic deletion operation."""
    topic_name: str
    status: DeletionStatus
    dry_run_passed: bool = False
    actual_deleted: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def simulate_dry_run(self) -> bool:
        """Simulate dry-run deletion (no actual deletion)."""
        # In real scenario, would use Kafka AdminClient with dry_run=true
        # For now, always succeed
        self.dry_run_passed = True
        self.status = DeletionStatus.DRY_RUN_PASSED
        return True

    def execute_actual_deletion(self) -> bool:
        """Execute actual topic deletion."""
        if not self.dry_run_passed:
            self.error_message = "Dry-run not passed, cannot proceed"
            return False

        self.actual_deleted = True
        self.status = DeletionStatus.ACTUAL_COMPLETED
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get deletion operation summary."""
        return {
            "topic": self.topic_name,
            "status": self.status.value,
            "dry_run_passed": self.dry_run_passed,
            "actual_deleted": self.actual_deleted,
            "error": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class TestDryRunDeletion:
    """Test dry-run deletion verification."""

    def test_dry_run_deletion_initialization(self):
        """Test dry-run deletion initializes."""
        operation = DeletionOperation(
            topic_name="cryptofeed.trades.coinbase.btc-usd",
            status=DeletionStatus.PENDING,
        )

        assert operation.topic_name == "cryptofeed.trades.coinbase.btc-usd"
        assert operation.status == DeletionStatus.PENDING
        assert operation.dry_run_passed is False

    def test_simulate_dry_run(self):
        """Test simulating dry-run deletion."""
        operation = DeletionOperation(
            topic_name="test.topic",
            status=DeletionStatus.PENDING,
        )

        result = operation.simulate_dry_run()

        assert result is True
        assert operation.dry_run_passed is True
        assert operation.status == DeletionStatus.DRY_RUN_PASSED

    def test_execute_actual_deletion_after_dry_run(self):
        """Test actual deletion after dry-run passes."""
        operation = DeletionOperation(
            topic_name="test.topic",
            status=DeletionStatus.PENDING,
        )

        operation.simulate_dry_run()
        result = operation.execute_actual_deletion()

        assert result is True
        assert operation.actual_deleted is True
        assert operation.status == DeletionStatus.ACTUAL_COMPLETED

    def test_cannot_delete_without_dry_run(self):
        """Test cannot delete without dry-run passing."""
        operation = DeletionOperation(
            topic_name="test.topic",
            status=DeletionStatus.PENDING,
        )

        result = operation.execute_actual_deletion()

        assert result is False
        assert operation.actual_deleted is False
        assert "Dry-run not passed" in operation.error_message

    def test_multiple_deletions_workflow(self):
        """Test multiple topic deletions workflow."""
        topics = ["topic1", "topic2", "topic3"]
        operations = []

        for topic in topics:
            op = DeletionOperation(
                topic_name=topic,
                status=DeletionStatus.PENDING,
            )
            op.simulate_dry_run()
            assert op.status == DeletionStatus.DRY_RUN_PASSED

            op.execute_actual_deletion()
            assert op.status == DeletionStatus.ACTUAL_COMPLETED
            operations.append(op)

        assert len(operations) == 3
        assert all(op.actual_deleted for op in operations)


# ============================================================================
# 4. Cleanup Verification Tests
# ============================================================================


@dataclass
class CleanupVerification:
    """Verifies cleanup results after deletion."""
    disk_space_freed_gb: float
    partition_count_before: int
    partition_count_after: int
    rebalancing_complete: bool
    under_replicated_partitions: int

    def get_disk_space_freed_percent(self) -> float:
        """Get percentage of disk space freed."""
        # Simulating that legacy topics took ~2TB before
        total_legacy_size_gb = 2048
        return (self.disk_space_freed_gb / total_legacy_size_gb) * 100

    def is_cleanup_successful(self) -> bool:
        """Check if cleanup is successful."""
        return (
            self.disk_space_freed_gb > 0
            and self.partition_count_after < self.partition_count_before
            and self.rebalancing_complete
            and self.under_replicated_partitions == 0
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get cleanup verification summary."""
        return {
            "disk_space_freed_gb": round(self.disk_space_freed_gb, 2),
            "disk_space_freed_percent": round(
                self.get_disk_space_freed_percent(), 2
            ),
            "partition_count_reduction": (
                self.partition_count_before - self.partition_count_after
            ),
            "partition_count_before": self.partition_count_before,
            "partition_count_after": self.partition_count_after,
            "rebalancing_complete": self.rebalancing_complete,
            "under_replicated_partitions": self.under_replicated_partitions,
            "cleanup_successful": self.is_cleanup_successful(),
        }


class TestCleanupVerification:
    """Test cleanup verification after deletion."""

    def test_cleanup_verification_initialization(self):
        """Test cleanup verification initializes."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=500.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        assert cleanup.disk_space_freed_gb == 500.0
        assert cleanup.partition_count_before == 100000

    def test_disk_space_freed_calculation(self):
        """Test disk space freed percentage calculation."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=1024.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        freed_percent = cleanup.get_disk_space_freed_percent()
        # 1024 / 2048 = 0.5 = 50%
        assert freed_percent == 50.0

    def test_successful_cleanup(self):
        """Test successful cleanup verification."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=1500.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        assert cleanup.is_cleanup_successful() is True

    def test_cleanup_fails_no_disk_freed(self):
        """Test cleanup fails when no disk space freed."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=0.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        assert cleanup.is_cleanup_successful() is False

    def test_cleanup_fails_rebalancing_not_complete(self):
        """Test cleanup fails when rebalancing incomplete."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=500.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=False,  # FAILS
            under_replicated_partitions=5,
        )

        assert cleanup.is_cleanup_successful() is False

    def test_partition_count_reduction(self):
        """Test partition count reduction calculation."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=500.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        summary = cleanup.get_summary()
        expected_reduction = 100000 - 168
        assert summary["partition_count_reduction"] == expected_reduction


# ============================================================================
# 5. End-to-End Archival & Cleanup Tests
# ============================================================================


class TestTask27EndToEnd:
    """End-to-end tests for Task 27: Legacy Topic Archival & Cleanup."""

    def test_complete_archival_workflow(self):
        """Test complete archival workflow."""
        # 1. Create archive metadata
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        # 2. Archive 3 legacy topics
        for i in range(1, 4):
            manifest = BackupManifest(
                topic_name=f"cryptofeed.trades.exchange{i}.symbol{i}",
                message_count=1000000 * i,
                backup_location=f"s3://backups/kafka-legacy/topic{i}.tar.gz",
                backup_size_bytes=63000000 * i,
                checksum_sha256=f"hash_{i:06d}",
                compression_ratio=0.63,
                archived_at=datetime.utcnow(),
            )
            metadata.add_backup(manifest)

        # 3. Verify archive
        assert len(metadata.backup_manifests) == 3
        assert metadata.total_messages == 6000000  # 1M + 2M + 3M
        assert len(metadata.audit_trail) == 3

    def test_deletion_with_prerequisites(self):
        """Test deletion with prerequisite validation."""
        # 1. Validate prerequisites
        validator = DeletionPrerequisiteValidator()
        validator.check_no_active_consumers(0)
        validator.check_zero_new_messages(0)
        validator.check_retention_verified(10000)
        validator.check_restoration_procedure_documented(True)

        assert validator.all_checks_passed() is True

        # 2. Execute dry-run
        operation = DeletionOperation(
            topic_name="cryptofeed.trades.legacy.all",
            status=DeletionStatus.PENDING,
        )
        operation.simulate_dry_run()
        assert operation.dry_run_passed is True

        # 3. Execute actual deletion
        result = operation.execute_actual_deletion()
        assert result is True
        assert operation.status == DeletionStatus.ACTUAL_COMPLETED

    def test_cleanup_verification_workflow(self):
        """Test complete cleanup verification."""
        cleanup = CleanupVerification(
            disk_space_freed_gb=1500.0,
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        assert cleanup.is_cleanup_successful() is True
        summary = cleanup.get_summary()
        assert summary["cleanup_successful"] is True
        assert summary["partition_count_reduction"] == 99832

    def test_multi_step_archival_deletion_cleanup(self):
        """Test multi-step archival, deletion, and cleanup."""
        # STEP 1: Archive legacy topics
        metadata = ArchiveMetadata(
            archive_date=datetime.utcnow(),
            total_topics=0,
            total_messages=0,
            total_size_bytes=0,
        )

        for i in range(1, 6):  # 5 topics
            manifest = BackupManifest(
                topic_name=f"topic_{i}",
                message_count=1000000,
                backup_location=f"s3://backups/topic_{i}.tar.gz",
                backup_size_bytes=63000000,
                checksum_sha256=f"hash_{i}",
                compression_ratio=0.63,
                archived_at=datetime.utcnow(),
            )
            metadata.add_backup(manifest)

        # STEP 2: Validate deletion prerequisites
        validator = DeletionPrerequisiteValidator()
        validator.check_no_active_consumers(0)
        validator.check_zero_new_messages(0)
        validator.check_retention_verified(len(metadata.backup_manifests))
        validator.check_restoration_procedure_documented(True)

        assert validator.all_checks_passed() is True

        # STEP 3: Execute dry-run deletions
        deletion_ops = []
        for manifest in metadata.backup_manifests:
            op = DeletionOperation(
                topic_name=manifest.topic_name,
                status=DeletionStatus.PENDING,
            )
            op.simulate_dry_run()
            deletion_ops.append(op)

        assert all(op.dry_run_passed for op in deletion_ops)

        # STEP 4: Execute actual deletions
        for op in deletion_ops:
            result = op.execute_actual_deletion()
            assert result is True

        # STEP 5: Verify cleanup
        cleanup = CleanupVerification(
            disk_space_freed_gb=315.0,  # 5 topics * 63GB each
            partition_count_before=100000,
            partition_count_after=168,
            rebalancing_complete=True,
            under_replicated_partitions=0,
        )

        assert cleanup.is_cleanup_successful() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

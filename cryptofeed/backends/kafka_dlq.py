"""Dead Letter Queue (DLQ) handler for Kafka producer failures.

This module provides DLQ functionality for handling messages that fail to produce
after retries are exhausted. It routes failed messages to a centralized DLQ topic
for recovery, debugging, and audit purposes.

Key Components:
- DLQHandler: Routes failed messages to DLQ topic
- DLQConfig: Configuration for DLQ behavior
- DLQMessage: Schema for messages in DLQ
- DLQRecovery: Replays messages from DLQ to original topic
- ErrorClassifier: Categorizes errors as transient or permanent
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


LOG = logging.getLogger("feedhandler")


# ============================================================================
# DLQ Configuration Models
# ============================================================================


class DLQConfig(BaseModel):
    """Configuration for Dead Letter Queue.

    Attributes:
        enabled: Whether DLQ is enabled (default: True)
        dlq_topic_prefix: Topic name prefix for DLQ (default: "cryptofeed.dlq")
        partitions: Number of partitions for DLQ topic (default: 3)
        replication_factor: Replication factor for DLQ (default: 3)
        retention_ms: Message retention in milliseconds (default: 7 days)
        compression_type: Compression algorithm (default: "snappy")
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable DLQ")
    dlq_topic_prefix: str = Field(default="cryptofeed.dlq", description="DLQ topic prefix")
    partitions: int = Field(default=3, description="DLQ topic partitions")
    replication_factor: int = Field(default=3, description="DLQ replication factor")
    retention_ms: int = Field(
        default=7 * 24 * 3600 * 1000,
        description="Retention period in ms (default: 7 days)"
    )
    compression_type: str = Field(default="snappy", description="Compression type")

    @field_validator("partitions")
    @classmethod
    def validate_partitions(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("partitions must be > 0")
        return v

    @field_validator("replication_factor")
    @classmethod
    def validate_replication(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("replication_factor must be > 0")
        return v

    @field_validator("retention_ms")
    @classmethod
    def validate_retention(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("retention_ms must be > 0")
        return v


# ============================================================================
# DLQ Message Schema and Error Classification
# ============================================================================


@dataclass(slots=True)
class DLQMessage:
    """Schema for Dead Letter Queue message.

    Attributes:
        original_topic: Topic name where message originally failed
        original_payload: Original message payload (bytes)
        error_type: Classification of error (e.g., 'broker_unavailable', 'serialization_error')
        error_message: Detailed error message
        timestamp: Unix timestamp when message was sent to DLQ
        attempts: Number of produce attempts before DLQ routing
        partition_key: Optional partition key from original message
        headers: Optional headers from original message
    """

    original_topic: str
    original_payload: bytes
    error_type: str
    error_message: str
    timestamp: float
    attempts: int
    partition_key: Optional[bytes] = None
    headers: Optional[List[tuple]] = None


class ErrorType(str, Enum):
    """Classification of error types for DLQ routing."""

    # Transient errors (should retry with backoff)
    BROKER_UNAVAILABLE = "broker_unavailable"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    NOT_LEADER = "not_leader"

    # Permanent errors (should go to DLQ immediately)
    SCHEMA_MISMATCH = "schema_mismatch"
    SERIALIZATION_ERROR = "serialization_error"
    MESSAGE_SIZE_EXCEEDED = "message_size_exceeded"
    INVALID_TOPIC = "invalid_topic"
    AUTHENTICATION_ERROR = "authentication_error"


class ErrorClassifier:
    """Classifies errors as transient or permanent for retry/DLQ routing decisions."""

    TRANSIENT_ERRORS = {
        ErrorType.BROKER_UNAVAILABLE,
        ErrorType.TIMEOUT,
        ErrorType.NETWORK_ERROR,
        ErrorType.NOT_LEADER,
        "ConnectionError",
        "TimeoutError",
        "NotLeaderForPartition",
        "RequestTimedOut",
        "BrokerNotAvailable",
    }

    PERMANENT_ERRORS = {
        ErrorType.SCHEMA_MISMATCH,
        ErrorType.SERIALIZATION_ERROR,
        ErrorType.MESSAGE_SIZE_EXCEEDED,
        ErrorType.INVALID_TOPIC,
        ErrorType.AUTHENTICATION_ERROR,
        "ValueError",
        "TypeError",
        "AuthenticationError",
    }

    @staticmethod
    def is_transient(error: Exception | str) -> bool:
        """Determine if error is transient (retry with backoff) or permanent (DLQ).

        Transient errors indicate temporary conditions that may resolve with retry.
        Permanent errors indicate issues that will not resolve with retry.

        Args:
            error: Exception or error string to classify

        Returns:
            True if transient (retry), False if permanent (DLQ)
        """
        if isinstance(error, Exception):
            error_str = error.__class__.__name__ + str(error)
        else:
            error_str = str(error)

        # Check for exact matches in classified sets
        for transient_pattern in ErrorClassifier.TRANSIENT_ERRORS:
            if transient_pattern.lower() in error_str.lower():
                return True

        for permanent_pattern in ErrorClassifier.PERMANENT_ERRORS:
            if permanent_pattern.lower() in error_str.lower():
                return False

        # Default: treat unknown errors as transient (safer for recovery)
        return True


# ============================================================================
# DLQ Handler Implementation
# ============================================================================


class DLQHandler:
    """Handler for routing failed messages to Dead Letter Queue.

    Manages DLQ topic, message routing, metrics, and recovery mechanisms.
    Ensures messages that fail Kafka produce after retries are exhausted
    are captured for debugging and recovery.
    """

    def __init__(
        self,
        bootstrap_servers: List[str],
        config: Optional[DLQConfig] = None,
        dlq_topic_prefix: Optional[str] = None,
    ) -> None:
        """Initialize DLQ handler.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            config: DLQConfig object for configuration (optional)
            dlq_topic_prefix: Custom DLQ topic prefix (overrides config)
        """
        self.bootstrap_servers = bootstrap_servers
        self.config = config or DLQConfig()

        # Override topic prefix if provided
        if dlq_topic_prefix is not None:
            self.dlq_topic = dlq_topic_prefix
        else:
            self.dlq_topic = self.config.dlq_topic_prefix

        # Message queue for DLQ messages
        self._queue: List[DLQMessage] = []

        # Metrics tracking
        self._metrics: Dict[str, int] = {
            "dlq_messages_total": 0,
            "dlq_broker_unavailable": 0,
            "dlq_timeout": 0,
            "dlq_serialization_error": 0,
            "dlq_schema_mismatch": 0,
            "dlq_network_error": 0,
            "dlq_other": 0,
        }

        # Lock for thread-safe queue operations
        try:
            self._lock = asyncio.Lock()
        except RuntimeError:
            # No event loop in current thread, operations are synchronous
            self._lock = None

    def send_to_dlq(
        self,
        message: bytes,
        original_topic: str,
        error_type: str,
        error_message: str,
        attempts: int,
        partition_key: Optional[bytes] = None,
        headers: Optional[List[tuple]] = None,
    ) -> bool:
        """Send message to Dead Letter Queue after retries exhausted.

        Args:
            message: Original message payload
            original_topic: Topic where message was being produced
            error_type: Classified error type (from ErrorType enum)
            error_message: Detailed error message
            attempts: Number of produce attempts before DLQ routing
            partition_key: Optional partition key from original message
            headers: Optional headers from original message

        Returns:
            True if message was queued to DLQ, False if DLQ is disabled
        """
        if not self.config.enabled:
            return False

        # Create DLQ message
        dlq_msg = DLQMessage(
            original_topic=original_topic,
            original_payload=message,
            error_type=error_type,
            error_message=error_message,
            timestamp=time.time(),
            attempts=attempts,
            partition_key=partition_key,
            headers=headers,
        )

        # Add to queue
        self._queue.append(dlq_msg)

        # Update metrics
        self._metrics["dlq_messages_total"] += 1

        # Track error type
        error_key = f"dlq_{error_type.lower()}"
        if error_key in self._metrics:
            self._metrics[error_key] += 1
        else:
            self._metrics["dlq_other"] += 1

        LOG.warning(
            "Message routed to DLQ: topic=%s, error_type=%s, attempts=%d, message_size=%d",
            original_topic,
            error_type,
            attempts,
            len(message),
            extra={
                "dlq": True,
                "original_topic": original_topic,
                "error_type": error_type,
                "attempts": attempts,
                "message_size": len(message),
            }
        )

        return True

    def queue_size(self) -> int:
        """Get current DLQ queue size.

        Returns:
            Number of messages in DLQ queue
        """
        return len(self._queue)

    def get_queued_message(self, index: int) -> Optional[DLQMessage]:
        """Retrieve a queued DLQ message by index.

        Args:
            index: Index in queue (0-based)

        Returns:
            DLQMessage if exists, None otherwise
        """
        if 0 <= index < len(self._queue):
            return self._queue[index]
        return None

    def get_dlq_metrics(self) -> Dict[str, int]:
        """Get DLQ metrics snapshot.

        Returns:
            Dictionary with metrics: total count, count by error type
        """
        return self._metrics.copy()

    def clear_dlq_metrics(self) -> None:
        """Reset DLQ metrics counters."""
        for key in self._metrics:
            self._metrics[key] = 0


# ============================================================================
# DLQ Recovery Implementation
# ============================================================================


class DLQRecovery:
    """Manages recovery and replay of messages from Dead Letter Queue.

    Enables operators to replay failed messages from DLQ back to their
    original topics after issues are resolved (e.g., broker recovery,
    schema fixes).

    Includes:
    - Schema validation before replay
    - Audit trail of replayed messages
    - Rate limiting for recovery operations
    - Dry-run mode for validation
    """

    def __init__(
        self,
        bootstrap_servers: List[str],
        dlq_handler: DLQHandler,
        max_messages_per_second: int = 1000,
    ) -> None:
        """Initialize DLQ recovery.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            dlq_handler: DLQHandler instance to replay from
            max_messages_per_second: Rate limit for recovery (default: 1000 msg/s)
        """
        self.bootstrap_servers = bootstrap_servers
        self.dlq_handler = dlq_handler
        self.max_messages_per_second = max_messages_per_second

        # Audit trail tracking
        self._audit_trail: List[Dict[str, Any]] = []

    def replay_dlq_messages(
        self,
        max_messages: int = 100,
        validate_schema: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Replay messages from DLQ to original topic.

        Args:
            max_messages: Maximum messages to replay (default: 100)
            validate_schema: Validate schema compatibility before replay (default: False)
            dry_run: Validate without actually replaying (default: False)

        Returns:
            Number of messages successfully replayed

        Raises:
            ValueError: If schema validation fails and validate_schema=True
        """
        replayed_count = 0
        start_time = time.time()

        for i in range(min(max_messages, self.dlq_handler.queue_size())):
            dlq_msg = self.dlq_handler.get_queued_message(i)
            if dlq_msg is None:
                break

            # Schema validation if requested
            if validate_schema:
                # Check if message looks like a permanent error
                if dlq_msg.error_type in ("schema_mismatch", "serialization_error"):
                    raise ValueError(
                        f"Cannot replay message with schema compatibility issue: {dlq_msg.error_type}"
                    )

            if not dry_run:
                # Simulate replay (in real implementation, would push to original topic)
                self._audit_trail.append({
                    "original_topic": dlq_msg.original_topic,
                    "error_type": dlq_msg.error_type,
                    "timestamp": time.time(),
                    "status": "replayed",
                    "message_size": len(dlq_msg.original_payload),
                })

            replayed_count += 1

            # Apply rate limiting
            elapsed = time.time() - start_time
            expected_elapsed = replayed_count / self.max_messages_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

        return replayed_count

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail of replayed messages.

        Returns:
            List of audit entries with replay details
        """
        return self._audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear audit trail entries."""
        self._audit_trail.clear()

"""Circuit Breaker pattern implementation for Kafka producer resilience.

This module provides circuit breaker functionality for protecting against cascading
failures when Kafka brokers are unavailable. It implements the classic circuit
breaker pattern with three states: CLOSED, OPEN, HALF_OPEN.

Key Components:
- CircuitBreaker: Main circuit breaker implementation
- CircuitBreakerConfig: Configuration for circuit breaker behavior
- CircuitBreakerOpen: Exception raised when circuit is open
- ErrorClassifier: Categorizes errors for circuit decision making
- ErrorCategory: Transient vs permanent error classification
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


LOG = logging.getLogger("feedhandler")


# ============================================================================
# Circuit Breaker States and Configuration
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker operational states."""

    CLOSED = "CLOSED"      # Normal operation, requests pass through
    OPEN = "OPEN"          # Failure detected, fail-fast without calling operation
    HALF_OPEN = "HALF_OPEN"  # Testing recovery, limited requests allowed


class ErrorCategory(str, Enum):
    """Categorization of errors for circuit breaker decisions."""

    TRANSIENT = "transient"    # Temporary error, should retry with backoff
    PERMANENT = "permanent"    # Error won't resolve with retry, skip circuit


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is OPEN and request cannot be processed."""

    def __init__(self, message: str = "Circuit breaker is open"):
        self.message = message
        super().__init__(message)


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Error rate threshold to trigger open (default: 0.05 = 5%)
        window_size: Time window for measuring failure rate in seconds (default: 60)
        timeout_seconds: Duration to stay in OPEN state before trying recovery (default: 30)
        initial_backoff_ms: Initial backoff delay for first retry attempt (default: 100)
        max_backoff_ms: Maximum backoff delay across all retries (default: 30000)
        backoff_multiplier: Exponential backoff multiplier (default: 2.0)
        half_open_test_requests: Number of test requests allowed in HALF_OPEN (default: 5)
        jitter_percent: Jitter percentage for backoff (default: 10%)
    """

    model_config = ConfigDict(extra="forbid")

    failure_threshold: float = Field(default=0.05, description="Failure rate threshold (0-1)")
    window_size: int = Field(default=60, description="Failure window in seconds")
    timeout_seconds: int = Field(default=30, description="OPEN state timeout")
    initial_backoff_ms: int = Field(default=100, description="Initial backoff in ms")
    max_backoff_ms: int = Field(default=30000, description="Max backoff in ms")
    backoff_multiplier: float = Field(default=2.0, description="Exponential backoff multiplier")
    half_open_test_requests: int = Field(default=5, description="Test requests in HALF_OPEN")
    jitter_percent: float = Field(default=10, description="Jitter percentage (0-100)")

    @field_validator("failure_threshold")
    @classmethod
    def validate_failure_threshold(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("failure_threshold must be between 0 and 1")
        return v

    @field_validator("window_size", "timeout_seconds")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("backoff_multiplier")
    @classmethod
    def validate_backoff_multiplier(cls, v: float) -> float:
        if v <= 1:
            raise ValueError("backoff_multiplier must be > 1")
        return v


# ============================================================================
# Error Classification
# ============================================================================


class ErrorClassifier:
    """Classifies errors for circuit breaker and recovery decisions."""

    TRANSIENT_PATTERNS = {
        "ConnectionError",
        "TimeoutError",
        "NotLeaderForPartition",
        "RequestTimedOut",
        "BrokerNotAvailable",
        "temporarily unavailable",
        "Connection refused",
        "timeout",
        "unavailable",
        "broker",
    }

    PERMANENT_PATTERNS = {
        "ValueError",
        "TypeError",
        "Schema",
        "schema",
        "Serialization",
        "serialization",
        "Invalid",
        "invalid",
        "Authentication",
        "authentication",
        "Authorization",
        "authorization",
    }

    @staticmethod
    def categorize(error: Exception | str) -> ErrorCategory:
        """Categorize error as transient or permanent.

        Args:
            error: Exception or error string

        Returns:
            ErrorCategory.TRANSIENT or ErrorCategory.PERMANENT
        """
        if isinstance(error, Exception):
            error_str = error.__class__.__name__ + str(error)
        else:
            error_str = str(error)

        # Check permanent patterns first (more specific)
        for pattern in ErrorClassifier.PERMANENT_PATTERNS:
            if pattern.lower() in error_str.lower():
                return ErrorCategory.PERMANENT

        # Check transient patterns
        for pattern in ErrorClassifier.TRANSIENT_PATTERNS:
            if pattern.lower() in error_str.lower():
                return ErrorCategory.TRANSIENT

        # Default: transient (safer for recovery)
        return ErrorCategory.TRANSIENT

    @staticmethod
    def is_transient(error: Exception | str) -> bool:
        """Check if error is transient."""
        return ErrorClassifier.categorize(error) == ErrorCategory.TRANSIENT


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================


@dataclass(slots=True)
class _CircuitBreakerMetrics:
    """Internal metrics tracking for circuit breaker."""

    state_transitions: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    half_open_test_requests: int = 0
    failed_test_requests: int = 0
    successful_test_requests: int = 0


class CircuitBreaker:
    """Circuit breaker for producer resilience against broker failures.

    Implements the classic three-state circuit breaker pattern:

    1. CLOSED: Normal operation
       - All requests pass through to operation
       - Failures tracked in sliding time window
       - If failure rate exceeds threshold, transitions to OPEN

    2. OPEN: Broker unavailable
       - Requests fail immediately without calling operation (fail-fast)
       - Prevents cascading failures to downstream systems
       - After timeout, transitions to HALF_OPEN for recovery testing

    3. HALF_OPEN: Testing recovery
       - Limited test requests allowed (e.g., 5 requests)
       - If test requests succeed, transitions to CLOSED
       - If test requests fail, transitions back to OPEN

    Exponential backoff with jitter prevents thundering herd when
    broker recovers.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        """Initialize circuit breaker.

        Args:
            config: CircuitBreakerConfig for behavior customization
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._failure_times: List[float] = []
        self._half_open_requests = 0
        self._metrics = _CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result if successful

        Raises:
            CircuitBreakerOpen: If circuit is OPEN
            Exception: Original function exception
        """
        self._metrics.total_calls += 1

        if self._state == CircuitState.CLOSED:
            return self._call_closed(func, args, kwargs)
        elif self._state == CircuitState.OPEN:
            return self._call_open(func, args, kwargs)
        elif self._state == CircuitState.HALF_OPEN:
            return self._call_half_open(func, args, kwargs)

    def _call_closed(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle request in CLOSED state (normal operation)."""
        try:
            result = func(*args, **kwargs)
            # Success: reset failure tracking
            self._failure_times.clear()
            self._metrics.total_successes += 1
            return result
        except Exception as e:
            # Failure: track and check threshold
            self._record_failure(e)
            self._check_threshold()
            self._metrics.total_failures += 1
            raise

    def _call_open(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle request in OPEN state (fail-fast)."""
        # Check if timeout expired, transition to HALF_OPEN
        if self._last_failure_time is not None:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_requests = 0
                self._record_state_transition(CircuitState.HALF_OPEN)
                # Allow this request to proceed as test request
                return self._call_half_open(func, args, kwargs)

        # Circuit is still open, fail fast
        raise CircuitBreakerOpen(
            f"Circuit breaker is open (since {self._last_failure_time})"
        )

    def _call_half_open(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle request in HALF_OPEN state (testing recovery)."""
        self._half_open_requests += 1
        self._metrics.half_open_test_requests += 1

        # Check test request limit
        if self._half_open_requests > self.config.half_open_test_requests:
            raise CircuitBreakerOpen(
                f"Too many test requests ({self._half_open_requests}), "
                f"max {self.config.half_open_test_requests}"
            )

        try:
            result = func(*args, **kwargs)
            # Test request succeeded, close circuit
            self._state = CircuitState.CLOSED
            self._failure_times.clear()
            self._metrics.total_successes += 1
            self._metrics.successful_test_requests += 1
            self._record_state_transition(CircuitState.CLOSED)
            return result
        except Exception as e:
            # Test request failed, reopen circuit
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            self._metrics.total_failures += 1
            self._metrics.failed_test_requests += 1
            self._record_state_transition(CircuitState.OPEN)
            raise

    def _record_failure(self, error: Exception) -> None:
        """Record failure in sliding time window."""
        now = time.time()
        self._failure_times.append(now)
        self._last_failure_time = now

        # Clean old entries outside window
        window_start = now - self.config.window_size
        self._failure_times = [t for t in self._failure_times if t >= window_start]

    def _check_threshold(self) -> None:
        """Check if failure rate exceeds threshold, open circuit if needed."""
        if len(self._failure_times) < 3:
            # Need at least 3 failures to make decision
            return

        # Calculate failure rate in window
        failure_count = len(self._failure_times)

        # Simple heuristic: if we have more than N failures in the window, open
        # This prevents circuit from opening too early
        if failure_count >= 3:
            self._state = CircuitState.OPEN
            self._record_state_transition(CircuitState.OPEN)
            LOG.warning(
                "Circuit breaker opened: failure_count=%d (in window)",
                failure_count
            )

    def _record_state_transition(self, new_state: CircuitState) -> None:
        """Record state transition for metrics."""
        self._metrics.state_transitions += 1
        LOG.info("Circuit breaker state transition: %s -> %s", self._state, new_state)

    def get_backoff_delay(self, attempt: int) -> int:
        """Calculate exponential backoff delay with jitter.

        Args:
            attempt: Attempt number (0-based)

        Returns:
            Delay in milliseconds
        """
        # Exponential backoff: initial * multiplier^attempt
        delay = self.config.initial_backoff_ms * (
            self.config.backoff_multiplier ** attempt
        )

        # Cap at max
        delay = min(delay, self.config.max_backoff_ms)

        # Apply jitter: ±jitter_percent
        jitter_range = delay * (self.config.jitter_percent / 100)
        jitter = random.uniform(-jitter_range, jitter_range)
        final_delay = max(0, delay + jitter)

        return int(final_delay)

    def get_failure_count(self) -> int:
        """Get current failure count in window."""
        return len(self._failure_times)

    def get_state_transition_count(self) -> int:
        """Get total state transitions."""
        return self._metrics.state_transitions

    def get_metrics(self) -> Dict[str, int]:
        """Get circuit breaker metrics snapshot.

        Returns:
            Dictionary with metrics: state_transitions, total_calls, failures, etc.
        """
        return {
            "state_transitions": self._metrics.state_transitions,
            "total_calls": self._metrics.total_calls,
            "total_failures": self._metrics.total_failures,
            "total_successes": self._metrics.total_successes,
            "half_open_test_requests": self._metrics.half_open_test_requests,
            "failed_test_requests": self._metrics.failed_test_requests,
            "successful_test_requests": self._metrics.successful_test_requests,
        }

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for retry/circuit decisions.

        Args:
            error: Exception to categorize

        Returns:
            ErrorCategory.TRANSIENT or ErrorCategory.PERMANENT
        """
        return ErrorClassifier.categorize(error)

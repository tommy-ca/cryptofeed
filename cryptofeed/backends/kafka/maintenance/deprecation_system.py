"""
Maintenance utilities for Kafka backend operations.

This module provides centralized deprecation warning management,
usage tracking, and operational excellence tools for the Kafka
backend ecosystem.
"""

import inspect
import logging
import time
import warnings
from typing import Dict, Any, Optional
from threading import Lock


LOG = logging.getLogger("feedhandler")


class DeprecationWarningSystem:
    """
    Centralized deprecation warning management for all Kafka backend components.

    This class provides consistent deprecation warning emission with actionable
    migration guidance, usage tracking for analytics, and integration with the
    cryptofeed logging infrastructure.

    Implements singleton pattern to ensure consistent warning handling
    across the entire Kafka backend ecosystem.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "DeprecationWarningSystem":
        """Singleton implementation with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the deprecation warning system."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._usage_stats: Dict[str, int] = {}
        self._usage_meta: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._initialized = True
        LOG.debug("DeprecationWarningSystem initialized")

    def emit_class_warning(
        self, class_name: str, replacement: str, stacklevel: int
    ) -> None:
        """
        Emit deprecation warning for legacy class usage.

        Args:
            class_name: Name of the deprecated class
            replacement: Import path or class name to use instead
            stacklevel: Stacklevel to surface the warning at caller site
        """
        warning_message = (
            f"{class_name} is deprecated and will be removed in a future release. "
            f"Use {replacement} instead. "
            f"See migration guide: https://docs.cryptofeed.ai/kafka-migration"
        )

        # Emit Python deprecation warning
        warnings.warn(
            warning_message,
            DeprecationWarning,
            stacklevel=stacklevel,
        )

        # Log to cryptofeed logging system
        LOG.warning(
            f"Legacy class usage detected: {class_name} -> {replacement}. "
            f"Track usage for migration planning."
        )

        # Track usage for analytics
        self.track_usage(
            class_name, {"type": "class_deprecation", "replacement": replacement}
        )

    def emit_import_warning(self, old_path: str, new_path: str, stacklevel: int) -> None:
        """
        Emit deprecation warning for legacy import path usage.

        Args:
            old_path: Deprecated import path
            new_path: New import path to use instead
            stacklevel: Stacklevel to surface the warning at caller site
        """
        warning_message = (
            f"Import path '{old_path}' is deprecated and will be removed in a future release. "
            f"Import from {new_path} instead. "
            f"Update your import statements to continue receiving updates."
        )

        # Emit Python deprecation warning
        warnings.warn(
            warning_message,
            DeprecationWarning,
            stacklevel=stacklevel,
        )

        # Log to cryptofeed logging system
        LOG.warning(
            f"Legacy import path detected: {old_path} -> {new_path}. "
            f"Update imports to use new backend location."
        )

        # Track usage for analytics
        self.track_usage(old_path, {"type": "import_deprecation", "new_path": new_path})

    def track_usage(self, component: str, context: Dict[str, Any]) -> None:
        """
        Track usage patterns for migration planning and analytics.

        Args:
            component: Name of the component being used (class, import path, etc.)
            context: Additional context information (symbol, exchange, config, etc.)
        """
        # Enrich context with timestamp
        enriched_context = {"timestamp": time.time(), "component": component, **context}

        # Update usage statistics
        with self._lock:
            self._usage_stats[component] = self._usage_stats.get(component, 0) + 1
            self._usage_meta[component] = {
                "last_timestamp": enriched_context["timestamp"],
                "last_context": {k: v for k, v in enriched_context.items() if k not in {"timestamp", "component"}},
            }

        # Log usage for analytics collection
        self._log_usage(component, enriched_context)

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current usage statistics for all tracked components.

        Returns:
            Dictionary mapping component names to usage counts
        """
        with self._lock:
            return self._usage_stats.copy()

    def get_usage_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed usage report including counts and last-seen context.

        Returns:
            Dict mapping component -> {'count': int, 'last_timestamp': float, 'last_context': dict}
        """
        with self._lock:
            report: Dict[str, Dict[str, Any]] = {}
            for component, count in self._usage_stats.items():
                meta = self._usage_meta.get(component, {})
                report[component] = {
                    "count": count,
                    "last_timestamp": meta.get("last_timestamp"),
                    "last_context": meta.get("last_context", {}),
                }
            return report

    def emit_usage_report(self, logger: logging.Logger | None = None) -> None:
        """
        Emit usage report to provided logger (defaults to feedhandler).
        """
        logger = logger or LOG
        report = self.get_usage_report()
        logger.info("Kafka legacy usage report: %s", report)

    def reset_usage_stats(self) -> None:
        """Reset usage statistics (primarily for testing)."""
        with self._lock:
            self._usage_stats.clear()
        LOG.debug("Usage statistics reset")

    def _log_usage(self, component: str, context: Dict[str, Any]) -> None:
        """
        Log usage information for analytics collection.

        Args:
            component: Component name
            context: Enriched context information
        """
        # Log structured usage information
        LOG.info(
            f"Kafka backend usage tracked: component={component}, context={context}"
        )

        # In a production environment, this could send metrics to:
        # - Prometheus/Grafana for dashboard visualization
        # - Time-series database for trend analysis
        # - Alerting system for unexpected usage patterns
        # For now, we log to the existing cryptofeed logging system


# Global instance for easy access
_deprecation_system: Optional[DeprecationWarningSystem] = None


def _resolve_user_stacklevel() -> int:
    """
    Determine the stacklevel that points to user code, skipping internal frames.

    Skips frames belonging to the kafka maintenance/legacy modules and any
    site-packages modules (e.g., pytest wrappers) so warnings surface on the
    actual caller in user space.
    """
    stack = inspect.stack()
    for idx, frame_info in enumerate(stack[1:], start=2):
        filename = frame_info.filename
        if "cryptofeed/backends/kafka" in filename:
            continue
        if "site-packages" in filename:
            continue
        return idx
    return max(2, len(stack))


def get_deprecation_warning_system() -> DeprecationWarningSystem:
    """
    Get the global deprecation warning system instance.

    Returns:
        Singleton instance of DeprecationWarningSystem
    """
    global _deprecation_system
    if _deprecation_system is None:
        _deprecation_system = DeprecationWarningSystem()
    return _deprecation_system


def emit_class_deprecation_warning(class_name: str, replacement: str) -> None:
    """
    Convenience function to emit class deprecation warnings.

    Args:
        class_name: Name of the deprecated class
        replacement: Import path or class name to use instead
    """
    stacklevel = _resolve_user_stacklevel()
    get_deprecation_warning_system().emit_class_warning(
        class_name, replacement, stacklevel=stacklevel
    )


def emit_import_deprecation_warning(old_path: str, new_path: str) -> None:
    """
    Convenience function to emit import deprecation warnings.

    Args:
        old_path: Deprecated import path
        new_path: New import path to use instead
    """
    stacklevel = _resolve_user_stacklevel()
    get_deprecation_warning_system().emit_import_warning(
        old_path, new_path, stacklevel=stacklevel
    )


def track_kafka_usage(component: str, context: Dict[str, Any]) -> None:
    """
    Convenience function to track Kafka backend usage.

    Args:
        component: Name of the component being used
        context: Additional context information
    """
    get_deprecation_warning_system().track_usage(component, context)

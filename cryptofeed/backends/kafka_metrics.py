"""
Compatibility shim for the Kafka metrics exporter.
"""

from __future__ import annotations

import warnings

from cryptofeed.backends.kafka.metrics import *  # noqa: F401,F403

warnings.warn(
    "cryptofeed.backends.kafka_metrics is deprecated; import from "
    "cryptofeed.backends.kafka.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

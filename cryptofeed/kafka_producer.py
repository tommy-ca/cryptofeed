"""
Compatibility shim for legacy Kafka producer imports.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cryptofeed.kafka_producer is deprecated; import from "
    "cryptofeed.backends.kafka.producer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cryptofeed.backends.kafka.producer import *  # noqa: F401,F403

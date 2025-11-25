"""
Compatibility shim for legacy Kafka config imports.
"""

from __future__ import annotations

import warnings

from cryptofeed.backends.kafka.config import *  # noqa: F401,F403

warnings.warn(
    "cryptofeed.kafka_config is deprecated; import from "
    "cryptofeed.backends.kafka.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

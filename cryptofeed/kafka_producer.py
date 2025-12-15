"""
Compatibility shim for legacy Kafka producer imports.
"""

from __future__ import annotations

import warnings

from cryptofeed.backends.kafka.backend import KafkaProducer, DeliveryReport  # noqa: F401

warnings.warn(
    "cryptofeed.kafka_producer is deprecated; import from "
    "cryptofeed.backends.kafka.backend instead.",
    DeprecationWarning,
    stacklevel=2,
)

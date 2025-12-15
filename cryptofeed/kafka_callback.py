"""
Compatibility shim for legacy Kafka callback imports.
"""

from __future__ import annotations

from cryptofeed.backends.kafka.base import _SUPPORTED_METHODS  # noqa: F401
from cryptofeed.backends.kafka.callback import *  # noqa: F401,F403
from cryptofeed.backends.kafka.headers import (  # noqa: F401
    MessageHeaders,
    OptionalHeaders,
    HeaderEnricher,
)
from cryptofeed.backends.kafka.partitioner import (  # noqa: F401
    Partitioner,
    PartitionerFactory,
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
)
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback  # noqa: F401

# Use the new deprecation warning system for consistent messaging
from cryptofeed.backends.kafka.deprecation import emit_import_deprecation_warning

emit_import_deprecation_warning(
    "cryptofeed.kafka_callback", "cryptofeed.backends.kafka.callback"
)

"""
Protobuf-only Kafka backend that locks serialization format and schema headers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from cryptofeed.backends.protobuf.helpers import serialize_to_protobuf
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION

from .base import KafkaQueuedMessage
from .callback import KafkaCallback
from .headers import HeaderEnricher


class KafkaProtobufCallback(KafkaCallback):
    """
    Kafka callback specialized for protobuf payloads.

    - Locks serialization_format to "protobuf"
    - Always emits application/x-protobuf content-type headers
    - Stamps schema_version metadata on every message
    """

    def __init__(
        self,
        *,
        schema_version: str | None = None,
        bootstrap_servers: Iterable[str] | None = None,
        kafka_config: Any | None = None,
        acks: str | None = "all",
        enable_idempotence: bool | None = True,
        connection_timeout_ms: int = 5000,
        producer_factory=None,
        numeric_type=float,
        none_to=None,
        queue_maxsize: int = 0,
        enable_batch_drain: bool = True,
        batch_drain_size: int = 50,
        enable_partition_key_cache: bool = True,
        partition_key_cache_size: int = 1000,
        enable_header_precomputation: bool = True,
        drain_frequency_ms: int = 10,
        metrics_exporter=None,
        metrics_enabled: bool = True,
        metrics_producer_id: str | None = None,
        **config: Any,
    ) -> None:
        self._schema_version = schema_version or DEFAULT_SCHEMA_VERSION

        super().__init__(
            bootstrap_servers=bootstrap_servers,
            kafka_config=kafka_config,
            acks=acks,
            enable_idempotence=enable_idempotence,
            connection_timeout_ms=connection_timeout_ms,
            producer_factory=producer_factory,
            serialization_format="protobuf",
            numeric_type=numeric_type,
            none_to=none_to,
            queue_maxsize=queue_maxsize,
            enable_batch_drain=enable_batch_drain,
            batch_drain_size=batch_drain_size,
            enable_partition_key_cache=enable_partition_key_cache,
            partition_key_cache_size=partition_key_cache_size,
            enable_header_precomputation=enable_header_precomputation,
            drain_frequency_ms=drain_frequency_ms,
            metrics_exporter=metrics_exporter,
            metrics_enabled=metrics_enabled,
            metrics_producer_id=metrics_producer_id,
            **config,
        )

        # Override header enricher to ensure schema_version metadata matches protobuf schema
        self._header_enricher = HeaderEnricher(
            content_type="application/x-protobuf",
            schema_version=self._schema_version,
            serialization_format=self.serialization_format,
        )

    def _serialize_payload(self, obj: Any, receipt_timestamp: Optional[float]):
        payload = serialize_to_protobuf(obj)
        headers = [
            (b"content-type", b"application/x-protobuf"),
        ]
        return payload, headers


__all__ = ["KafkaProtobufCallback"]

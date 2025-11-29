from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest


# Import KafkaCallback and related components used across KafkaCallback tests
from cryptofeed.backends.kafka.callback import KafkaCallback
from cryptofeed.backends.kafka.topic_manager import TopicManager
from cryptofeed.backends.kafka.headers import MessageHeaders, OptionalHeaders, HeaderEnricher
from cryptofeed.backends.kafka.partitioner import (
    Partitioner,
    PartitionerFactory,
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
)
from cryptofeed.backends.kafka.config import (
    KafkaConfig,
    KafkaTopicConfig,
    KafkaPartitionConfig,
    KafkaProducerConfig,
)


@dataclass
class _RecordedMessage:
    """Record of a produced message for verification."""

    topic: str
    key: Optional[bytes]
    value: bytes
    headers: List[tuple[bytes, bytes]]


class _StubProducer:
    """In-memory producer for testing without Kafka brokers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_RecordedMessage] = []
        self.poll_count = 0

    def list_topics(self, timeout: Optional[float] = None):
        self.connected = True
        return {"topics": {}}

    def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[tuple[bytes, bytes]]] = None,
        on_delivery=None,
    ):
        """Record produced message and optionally call delivery callback."""
        headers = headers or []
        self.messages.append(
            _RecordedMessage(topic=topic, key=key, value=value, headers=headers)
        )
        if on_delivery:
            # Simulate successful delivery
            msg = Mock()
            msg.topic.return_value = topic
            msg.partition.return_value = 0
            msg.offset.return_value = len(self.messages) - 1
            on_delivery(None, msg)

    def poll(self, timeout: float):
        self.poll_count += 1
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


def _producer_factory(cls):
    """Create a producer factory function."""

    def _factory(config):
        return cls(config)

    return _factory

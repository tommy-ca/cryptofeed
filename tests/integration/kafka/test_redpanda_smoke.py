"""Minimal Redpanda smoke test (produce → consume) without cryptofeed stack.

Purpose: validate docker-compose Redpanda fixture and client libs before
running heavier cryptofeed → Kafka pipelines.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest
from confluent_kafka import Consumer, Producer


@pytest.mark.integration
def test_redpanda_produce_consume_roundtrip(redpanda: str) -> None:
    """Ensure Redpanda accepts produce/consume using confluent-kafka clients."""

    topic = f"cf-smoke-{uuid4().hex[:8]}"
    group_id = f"cf-smoke-g-{uuid4().hex[:8]}"

    producer = Producer({"bootstrap.servers": redpanda})
    producer.produce(topic, key=b"k", value=b"hello-redpanda-smoke")
    producer.flush(5)

    consumer = Consumer(
        {
            "bootstrap.servers": redpanda,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])

    msg = None
    end = time.time() + 15
    try:
        while time.time() < end:
            polled = consumer.poll(1.0)
            if polled and not polled.error():
                msg = polled
                break
    finally:
        consumer.close()

    assert msg is not None, "No message consumed from Redpanda smoke topic"
    assert msg.value() == b"hello-redpanda-smoke"
    assert msg.topic() == topic


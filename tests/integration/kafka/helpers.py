"""Shared helpers for Kafka/Redpanda integration tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

from confluent_kafka import Consumer


@dataclass
class ConsumedRecord:
    value: bytes
    headers: Dict[bytes, bytes]
    topic: str
    key: bytes | None


def consume_one(
    bootstrap: str,
    topic: str,
    *,
    timeout_s: float = 10.0,
    group_id: str = "cf-e2e-proto",
    offset_reset: str = "earliest",
) -> ConsumedRecord:
    """Consume a single record from the given topic within `timeout_s`."""

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": offset_reset,
        }
    )
    consumer.subscribe([topic])

    msg = None
    end_time = time.time() + timeout_s
    try:
        while time.time() < end_time:
            msg = consumer.poll(0.5)
            if msg and not msg.error():
                break
    finally:
        consumer.close()

    if msg is None or msg.error():
        raise AssertionError(f"No message consumed from Kafka for topic {topic}")

    def _clean(x: object) -> bytes:
        if isinstance(x, bytes):
            if x.startswith(b"b'") and x.endswith(b"'"):
                return x[2:-1]
            return x
        if isinstance(x, str) and x.startswith("b'") and x.endswith("'"):
            x = x[2:-1]
        return str(x).encode()

    header_dict = {_clean(k): _clean(v) for k, v in msg.headers() or []}
    return ConsumedRecord(
        value=msg.value(), headers=header_dict, topic=msg.topic(), key=msg.key()
    )

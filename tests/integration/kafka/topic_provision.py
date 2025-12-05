"""Topic auto-provision helpers for Kafka/Redpanda integration tests."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Sequence

import pytest


@asynccontextmanager
async def _admin_client(bootstrap: str):
    """Yield a started AIOKafkaAdminClient, skipping tests if unavailable.

    aiokafka requires an explicit `start()` before admin operations; without
    it CreateTopics will fail with IncompatibleBrokerVersion.
    """

    admin_mod = pytest.importorskip("aiokafka.admin")
    errors_mod = pytest.importorskip("aiokafka.errors")

    client = admin_mod.AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        await client.start()
        yield client, admin_mod, errors_mod
    finally:
        await client.close()


async def ensure_topics_exist(
    bootstrap: str,
    topics: Sequence[str],
    *,
    partitions: int | None = None,
    replication: int | None = None,
    timeout_s: float = 10.0,
) -> list[str]:
    """Idempotently create Kafka topics for E2E tests.

    Returns the list of topics that were created (empty if all already exist).
    Skips the calling test on broker connectivity or admin errors.
    """

    # Allow env overrides for CI/local tuning; defaults suit single-node Redpanda.
    partitions = partitions or int(os.getenv("KAFKA_E2E_TOPIC_PARTITIONS", "1"))
    replication = replication or int(os.getenv("KAFKA_E2E_TOPIC_REPLICATION", "1"))

    async with _admin_client(bootstrap) as (client, admin_mod, errors_mod):
        new_topic_cls = admin_mod.NewTopic
        topic_error = errors_mod.TopicAlreadyExistsError
        kafka_error = errors_mod.KafkaError

        try:
            existing = await client.list_topics()
        except kafka_error as exc:  # pragma: no cover - network/env issues
            pytest.skip(f"Kafka topic provisioning failed (list): {exc}")

        missing = [t for t in topics if t not in existing]
        if not missing:
            return []

        new_topics = [
            new_topic_cls(name=t, num_partitions=partitions, replication_factor=replication)
            for t in missing
        ]

        try:
            await client.create_topics(
                new_topics=new_topics,
                timeout_ms=int(timeout_s * 1000),
            )
        except topic_error:
            return []
        except kafka_error as exc:  # pragma: no cover - network/env issues
            pytest.skip(f"Kafka topic provisioning failed (create): {exc}")

        return missing

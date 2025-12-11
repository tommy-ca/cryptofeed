"""Topic auto-provision helpers for Kafka/Redpanda integration tests."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
import asyncio
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
    configs: dict[str, str] | None = None,
    timeout_s: float = 10.0,
) -> list[str]:
    """Idempotently create Kafka topics for E2E tests.

    Returns the list of topics that were created (empty if all already exist).
    Skips the calling test on broker connectivity or admin errors.
    """

    # Allow env overrides for CI/local tuning; defaults suit single-node Redpanda.
    partitions = partitions or int(os.getenv("KAFKA_E2E_TOPIC_PARTITIONS", "1"))
    replication = replication or int(os.getenv("KAFKA_E2E_TOPIC_REPLICATION", "1"))

    configs = configs or {}

    async with _admin_client(bootstrap) as (client, admin_mod, errors_mod):
        new_topic_cls = admin_mod.NewTopic
        topic_error = errors_mod.TopicAlreadyExistsError
        kafka_error = errors_mod.KafkaError
        connectivity_errors = (
            errors_mod.KafkaConnectionError,
            errors_mod.NoBrokersAvailable,
            errors_mod.IncompatibleBrokerVersion,
            errors_mod.KafkaUnavailableError,
            errors_mod.MetadataEmptyBrokerList,
            errors_mod.BrokerNotAvailableError,
        )

        try:
            existing = await client.list_topics()
        except connectivity_errors as exc:  # pragma: no cover - network/env issues
            pytest.skip(f"Kafka topic provisioning failed (list): {exc}")
        except kafka_error:
            raise

        missing = [t for t in topics if t not in existing]
        if not missing:
            return []

        new_topics = [
            new_topic_cls(
                name=t,
                num_partitions=partitions,
                replication_factor=replication,
                topic_configs=configs,
            )
            for t in missing
        ]

        try:
            await client.create_topics(
                new_topics=new_topics,
                timeout_ms=int(timeout_s * 1000),
            )
        except topic_error:
            missing = []
        except connectivity_errors as exc:  # pragma: no cover - network/env issues
            pytest.skip(f"Kafka topic provisioning failed (create): {exc}")
        except kafka_error:
            raise

        # Verify topics exist and match basic expectations; retry briefly for broker propagation.
        async def _verify() -> None:
            meta = await client.describe_topics(topics)
            by_name = {m["topic"]: m for m in meta}
            for t in topics:
                if t not in by_name:
                    raise RuntimeError(f"Topic {t} missing after create")
                partitions_meta = by_name[t].get("partitions") or []
                if partitions_meta and partitions is not None:
                    if len(partitions_meta) != partitions:
                        raise RuntimeError(
                            f"Topic {t} partitions {len(partitions_meta)} != requested {partitions}"
                        )
                for p in partitions_meta:
                    replicas = p.get("replicas") or []
                    if replication is not None and replicas:
                        if len(replicas) != replication:
                            raise RuntimeError(
                                f"Topic {t} replication {len(replicas)} != requested {replication}"
                            )

        for _ in range(3):
            try:
                await _verify()
                break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.5)
        else:
            raise RuntimeError("Topic verification failed after creation")

        return missing

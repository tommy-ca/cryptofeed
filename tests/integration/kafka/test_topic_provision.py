"""Tests for topic auto-provision helper using Redpanda."""

from __future__ import annotations

import uuid

import pytest

from tests.integration.kafka.topic_provision import _admin_client, ensure_topics_exist


@pytest.mark.asyncio
@pytest.mark.integration
async def test_topic_provision_idempotent(redpanda, monkeypatch):
    base = f"cf-e2e-provision-{uuid.uuid4().hex}"
    topics = [f"{base}-a", f"{base}-b"]

    # Ensure env-driven defaults are honored
    monkeypatch.setenv("KAFKA_E2E_TOPIC_PARTITIONS", "1")
    monkeypatch.setenv("KAFKA_E2E_TOPIC_REPLICATION", "1")

    created = await ensure_topics_exist(redpanda, topics)
    assert set(created) == set(topics)

    # Idempotent on second call
    created_again = await ensure_topics_exist(redpanda, topics)
    assert created_again == []

    # Verify partition/replication counts via admin metadata
    async with _admin_client(redpanda) as (client, _admin_mod, _errors_mod):
        metadata = await client.describe_topics(topics)
    for topic_md in metadata:
        partitions = topic_md["partitions"]
        assert len(partitions) == 1
        for partition in partitions:
            # replication factor equals number of replicas per partition
            assert len(partition["replicas"]) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_topic_provision_consolidated(redpanda):
    topics = ["cryptofeed.trade", "cryptofeed.l2_book"]

    created = await ensure_topics_exist(redpanda, topics)
    assert set(created) == set(topics) or created == []

    created_again = await ensure_topics_exist(redpanda, topics)
    assert created_again == []

    async with _admin_client(redpanda) as (client, _admin_mod, _errors_mod):
        metadata = await client.describe_topics(topics)
    for topic_md in metadata:
        # At least one partition and replica present
        assert topic_md["partitions"]
        for partition in topic_md["partitions"]:
            assert partition["replicas"]
"""Tests for topic auto-provision helper using Redpanda."""

from __future__ import annotations

import uuid

import pytest

from aiokafka.admin.config_resource import ConfigResource, ConfigResourceType

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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_topic_provision_respects_partitions_env(redpanda, monkeypatch):
    base = f"cf-e2e-provision-{uuid.uuid4().hex}"
    topics = [f"{base}-p"]

    monkeypatch.setenv("KAFKA_E2E_TOPIC_PARTITIONS", "2")
    monkeypatch.setenv("KAFKA_E2E_TOPIC_REPLICATION", "1")

    await ensure_topics_exist(redpanda, topics)

    async with _admin_client(redpanda) as (client, _admin_mod, _errors_mod):
        metadata = await client.describe_topics(topics)
    assert len(metadata[0]["partitions"]) == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_topic_provision_applies_configs(redpanda):
    base = f"cf-e2e-provision-{uuid.uuid4().hex}"
    topic = f"{base}-cfg"

    await ensure_topics_exist(
        redpanda,
        [topic],
        configs={"retention.ms": "60000", "cleanup.policy": "delete"},
    )

    async with _admin_client(redpanda) as (client, _admin_mod, errors_mod):
        try:
            resp_list = await client.describe_configs(
                [ConfigResource(ConfigResourceType.TOPIC, topic)]
            )
        except errors_mod.KafkaError as exc:  # pragma: no cover - env specific
            pytest.skip(f"describe_configs unavailable: {exc}")

    cfg_entries = (
        resp_list[0].to_object()
        .get("resources", [{}])[0]
        .get("config_entries", [])
    )
    cfg_map = {
        entry.get("config_names"): entry.get("config_value")
        for entry in cfg_entries
        if isinstance(entry, dict) and "config_names" in entry
    }

    assert cfg_map.get("retention.ms") == "60000"
    assert cfg_map.get("cleanup.policy") == "delete"

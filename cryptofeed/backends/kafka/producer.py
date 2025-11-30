"""Kafka producer abstraction built on confluent-kafka.

This module provides a thin wrapper around ``confluent_kafka.Producer``
with connection verification, error translation and convenience helpers
required by the Market Data Kafka Producer specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from confluent_kafka import KafkaException, Producer


LOG = logging.getLogger("feedhandler")


def _normalize_bootstrap_servers(servers: Iterable[str]) -> str:
    parts = [str(server).strip() for server in servers if str(server).strip()]
    if not parts:
        raise ValueError("bootstrap_servers must not be empty")
    return ",".join(parts)


@dataclass(slots=True)
class DeliveryReport:
    topic: str
    partition: int
    offset: int


class KafkaProducer:
    """Kafka producer helper with connection checks and delivery semantics."""

    def __init__(
        self,
        bootstrap_servers: Sequence[str],
        *,
        acks: str | None = "all",
        enable_idempotence: bool | None = True,
        producer_factory: Callable[[Mapping[str, Any]], Producer] | None = None,
        connection_timeout_ms: int = 5000,
        logger: logging.Logger | None = None,
        **config: Any,
    ) -> None:
        self._bootstrap_servers = list(bootstrap_servers)
        self._acks = acks
        self._enable_idempotence = enable_idempotence
        self._config = dict(config)
        self._producer_factory = producer_factory or Producer
        self._connection_timeout_ms = connection_timeout_ms
        self._producer: Producer | None = None
        self._logger = logger or LOG
        self._connected: bool = False

    @property
    def bootstrap_servers(self) -> Sequence[str]:
        return list(self._bootstrap_servers)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = dict(self._config)
        config["bootstrap.servers"] = _normalize_bootstrap_servers(self._bootstrap_servers)
        if self._acks is not None:
            config["acks"] = self._acks
        if self._enable_idempotence is not None:
            config["enable.idempotence"] = "true" if self._enable_idempotence else "false"
        return config

    def connect(self) -> None:
        if self._producer is None:
            config = self._build_config()
            try:
                self._producer = self._producer_factory(config)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ConnectionError("failed to initialize Kafka producer") from exc

        try:
            timeout_sec = self._connection_timeout_ms / 1000 if self._connection_timeout_ms else None
            self._producer.list_topics(timeout=timeout_sec)
        except KafkaException as exc:
            self._connected = False
            message = str(exc.args[0] if exc.args else exc)
            raise ConnectionError(f"kafka brokers unreachable: {message}") from exc
        else:
            self._connected = True
            self._logger.info(
                "Kafka producer connected [bootstrap=%s, idempotent=%s]",
                ",".join(self._bootstrap_servers),
                bool(self._enable_idempotence),
            )

    def produce(
        self,
        topic: str,
        value: bytes,
        *,
        key: Optional[bytes] = None,
        headers: Optional[list[tuple[str, bytes]]] = None,
        on_delivery: Optional[Callable[[DeliveryReport], None]] = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("producer not connected")

        def _delivery_callback(err, msg):
            if err is not None:
                self._logger.error("Kafka delivery error: %s", err)
                return
            if on_delivery is not None:
                on_delivery(
                    DeliveryReport(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                )

        self._producer.produce(
            topic,
            value=value,
            key=key,
            headers=headers,
            on_delivery=_delivery_callback if on_delivery else None,
        )

    def poll(self, timeout: float = 0.0) -> int:
        if self._producer is None:
            return 0
        return self._producer.poll(timeout)

    def flush(self, timeout: Optional[float] = None) -> int:
        if self._producer is None:
            return 0
        return self._producer.flush(timeout)

    def close(self, timeout: Optional[float] = None) -> None:
        if self._producer is not None:
            try:
                flush_timeout = 5.0 if timeout is None else timeout
                self._producer.flush(flush_timeout)
            finally:
                self._producer = None
                self._connected = False

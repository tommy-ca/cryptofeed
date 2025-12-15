"""
Compatibility partitioner helpers (Phase 2 shim).

Implements the legacy Partitioner classes using the shared normalization utilities
and the inlined partition-key logic from `cryptofeed.backends.kafka.callback`.
"""

from __future__ import annotations

from typing import Any

from .normalization import normalize_exchange, normalize_symbol


class Partitioner:
    """Base partitioner interface."""

    def get_partition_key(self, message: Any) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError


class SymbolPartitioner(Partitioner):
    """Partition by normalized symbol."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return normalize_symbol(getattr(message, "symbol", None)).encode("utf-8")


class CompositePartitioner(Partitioner):
    """Partition by normalized exchange-symbol combination."""

    def get_partition_key(self, message: Any) -> bytes | None:
        exchange = normalize_exchange(getattr(message, "exchange", None))
        symbol = normalize_symbol(getattr(message, "symbol", None))
        return f"{exchange}-{symbol}".encode("utf-8")


class ExchangePartitioner(Partitioner):
    """Partition by normalized exchange."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return normalize_exchange(getattr(message, "exchange", None)).encode("utf-8")


class RoundRobinPartitioner(Partitioner):
    """Round robin (no partition key)."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return None


class PartitionerFactory:
    """Factory for creating partitioners by strategy name."""

    @staticmethod
    def create(strategy: str | None = "composite") -> Partitioner:
        strategy_lower = (strategy or "composite").lower()
        if strategy_lower == "symbol":
            return SymbolPartitioner()
        if strategy_lower == "exchange":
            return ExchangePartitioner()
        if strategy_lower == "round_robin":
            return RoundRobinPartitioner()
        if strategy_lower == "composite":
            return CompositePartitioner()
        raise ValueError(f"Unknown partitioner strategy: {strategy}")


__all__ = [
    "Partitioner",
    "SymbolPartitioner",
    "CompositePartitioner",
    "ExchangePartitioner",
    "RoundRobinPartitioner",
    "PartitionerFactory",
]

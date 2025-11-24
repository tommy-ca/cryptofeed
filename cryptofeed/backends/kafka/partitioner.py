"""
Partition key strategy implementations for Kafka callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


class Partitioner(ABC):
    """Abstract base class for partition key strategies."""

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol).strip().upper().replace("_", "-").lower()

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        return str(exchange).strip().lower()

    @abstractmethod
    def get_partition_key(self, message: Any) -> Optional[bytes]:
        """Generate partition key for a message."""


class SymbolPartitioner(Partitioner):
    """Symbol-based partition key strategy."""

    def get_partition_key(self, message: Any) -> bytes:
        symbol = getattr(message, "symbol", "")
        normalized = self._normalize_symbol(symbol)
        return normalized.encode("utf-8")


class CompositePartitioner(Partitioner):
    """Composite exchange+symbol strategy (default)."""

    def get_partition_key(self, message: Any) -> bytes:
        exchange = getattr(message, "exchange", "")
        symbol = getattr(message, "symbol", "")
        normalized_exchange = self._normalize_exchange(exchange)
        normalized_symbol = self._normalize_symbol(symbol)
        return f"{normalized_exchange}-{normalized_symbol}".encode("utf-8")


class ExchangePartitioner(Partitioner):
    """Exchange-based partition key strategy."""

    def get_partition_key(self, message: Any) -> bytes:
        exchange = getattr(message, "exchange", "")
        normalized = self._normalize_exchange(exchange)
        return normalized.encode("utf-8")


class RoundRobinPartitioner(Partitioner):
    """Round-robin strategy that defers to Kafka."""

    def get_partition_key(self, message: Any) -> Optional[bytes]:
        return None


class PartitionerFactory:
    """Factory for creating partitioner instances by strategy name."""

    _PARTITIONERS: Dict[str, Type[Partitioner]] = {
        "symbol": SymbolPartitioner,
        "composite": CompositePartitioner,
        "exchange": ExchangePartitioner,
        "round_robin": RoundRobinPartitioner,
    }

    @staticmethod
    def create(strategy: str = "composite") -> Partitioner:
        strategy_lower = strategy.lower() if strategy else "composite"
        if strategy_lower not in PartitionerFactory._PARTITIONERS:
            supported = ", ".join(sorted(PartitionerFactory._PARTITIONERS.keys()))
            raise ValueError(
                f"Unknown partitioner strategy: {strategy}. Supported strategies: {supported}"
            )
        return PartitionerFactory._PARTITIONERS[strategy_lower]()


__all__ = [
    "Partitioner",
    "SymbolPartitioner",
    "CompositePartitioner",
    "ExchangePartitioner",
    "RoundRobinPartitioner",
    "PartitionerFactory",
]

"""Regression tests for legacy Kafka backend functional equivalence."""

from __future__ import annotations

import warnings

import pytest

# Load legacy kafka.py directly (shadowed by package)
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_KAFKA_PATH = REPO_ROOT / "cryptofeed/backends/kafka.py"
spec = importlib.util.spec_from_file_location("legacy_kafka", LEGACY_KAFKA_PATH)
legacy_kafka = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_kafka)


class TestLegacyKafkaFunctions:
    def setup_method(self):
        # fresh instance for each test
        self.legacy = legacy_kafka.TradeKafka(key="trades")

    def test_topic_json_backward_compat(self):
        data = {
            "exchange": "BINANCE",
            "symbol": "BTC/USDT",
            "feed": "trades",
        }
        topic = self.legacy.topic(data)
        assert topic == "trades-BINANCE-BTC/USDT"

    def test_topic_protobuf_bytes(self):
        topic = self.legacy.topic(b"\x08\x01")
        assert topic == "cryptofeed.market.trades.protobuf"

    def test_partition_key_from_symbol(self):
        data = {"symbol": "ETH-USD"}
        key = self.legacy.partition_key(data)
        assert key == b"ETH-USD"

    def test_partition_key_none_without_symbol(self):
        key = self.legacy.partition_key({})
        assert key is None

    def test_book_kafka_snapshot_params_preserved(self):
        book = legacy_kafka.BookKafka(snapshots_only=True, snapshot_interval=250)
        assert book.snapshots_only is True
        assert book.snapshot_interval == 250

    def test_deprecation_warning_emitted(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            legacy_kafka.TradeKafka()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TradeKafka" in str(w[0].message)

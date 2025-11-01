"""Compression benchmarks for protobuf serialization."""
from decimal import Decimal

import pytest

from cryptofeed.serializers import JSONSerializer, ProtobufSerializer
from cryptofeed.types import Trade


pytest.importorskip("lz4.frame")
pytest.importorskip("zstandard")

from lz4 import frame as lz4frame
import zstandard as zstd


@pytest.fixture
def sample_trade() -> Trade:
    return Trade(
        exchange="binance",
        symbol="BTC-USDT",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("50000.12"),
        timestamp=1_700_000_000.012345,
        id="trade-1",
    )


def _serialize_trade(trade: Trade):
    json_bytes = JSONSerializer().serialize(trade)
    proto_bytes = ProtobufSerializer().serialize(trade)
    return json_bytes, proto_bytes


def _assert_codec_behavior(
    codec_name: str,
    compressed: bytes,
    original: bytes,
    reference_json_size: int,
    max_ratio: float,
):
    assert len(compressed) < len(original), f"{codec_name} should shrink protobuf payload"
    ratio_vs_json = len(compressed) / reference_json_size
    assert ratio_vs_json <= max_ratio, (
        f"{codec_name} compression ratio too high: {ratio_vs_json:.2f} (limit {max_ratio:.2f})"
    )


def test_uncompressed_size_ratio(sample_trade: Trade):
    json_bytes, proto_bytes = _serialize_trade(sample_trade)

    ratio_vs_json = len(proto_bytes) / len(json_bytes)
    assert ratio_vs_json <= 0.55


def test_lz4_compression_ratio(sample_trade: Trade):
    json_bytes, proto_bytes = _serialize_trade(sample_trade)

    compressed = lz4frame.compress(proto_bytes)
    restored = lz4frame.decompress(compressed)

    assert restored == proto_bytes
    _assert_codec_behavior("lz4", compressed, proto_bytes, len(json_bytes), max_ratio=0.50)


def test_zstd_compression_ratio(sample_trade: Trade):
    json_bytes, proto_bytes = _serialize_trade(sample_trade)

    compressor = zstd.ZstdCompressor(level=3)
    compressed = compressor.compress(proto_bytes)
    restored = zstd.ZstdDecompressor().decompress(compressed)

    assert restored == proto_bytes
    _assert_codec_behavior("zstd", compressed, proto_bytes, len(json_bytes), max_ratio=0.45)

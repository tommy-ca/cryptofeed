"""Tests for optional compression codecs used with protobuf payloads."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf
from cryptofeed.types import Trade


def _sample_payload(multiplier: int = 32) -> bytes:
    trade = Trade(
        exchange="COINBASE",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("1.0"),
        price=Decimal("48000.00"),
        timestamp=1700000200.0,
        id="compression-test",
        type="spot",
        raw=None,
    )
    payload = serialize_to_protobuf(trade)
    return payload * multiplier


def test_lz4_compression_roundtrip():
    lz4 = pytest.importorskip("lz4.frame")

    payload = _sample_payload()
    compressed = lz4.compress(payload)
    assert len(compressed) < len(payload)
    decompressed = lz4.decompress(compressed)
    assert decompressed == payload


def test_zstd_compression_roundtrip():
    zstandard = pytest.importorskip("zstandard")

    compressor = zstandard.ZstdCompressor()
    decompressor = zstandard.ZstdDecompressor()

    payload = _sample_payload()
    compressed = compressor.compress(payload)
    assert len(compressed) < len(payload)
    decompressed = decompressor.decompress(compressed)
    assert decompressed == payload

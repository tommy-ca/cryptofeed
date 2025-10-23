from __future__ import annotations

from decimal import Decimal

import pytest

from cryptofeed import json_utils as ju


def test_loads_preserves_decimal_default():
    payload = ju.loads("{\"price\": 1.2345}")
    assert isinstance(payload["price"], Decimal)
    assert payload["price"] == Decimal("1.2345")


def test_loads_accepts_bytes_and_forced_float():
    payload = ju.loads(b"{\"price\": 1.2345}", parse_float=float)
    assert isinstance(payload["price"], float)


def test_dumps_handles_decimal_and_non_ascii():
    text = ju.dumps({"price": Decimal("1.2345"), "snow": "☃"})
    assert "☃" in text
    assert '"price":"1.2345"' in text


def test_dumps_ensure_ascii_uses_stdlib():
    text = ju.dumps({"snow": "☃"}, ensure_ascii=True)
    assert text == '{"snow": "\\u2603"}'


def test_dumps_bytes_round_trip_decimal():
    data = ju.dumps_bytes({"price": Decimal("1.23")})
    assert isinstance(data, bytes)
    decoded = ju.loads(data.decode("utf-8"))
    assert decoded["price"] == "1.23"


def test_json_namespace_wrapper():
    text = ju.json.dumps({"value": Decimal("2.5")})
    parsed = ju.json.loads(text)
    assert parsed["value"] == "2.5"


def test_load_map_handles_iterables():
    items = [("a", 1), ("b", 2)]
    assert ju.load_map(items) == {"a": 1, "b": 2}


def test_dumps_bytes_fallback_without_orjson(monkeypatch):
    # Force stdlib path by disabling orjson for the duration of the test
    monkeypatch.setattr(ju, "_orjson", None, raising=False)
    data = ju.dumps_bytes({"value": Decimal("3.14")})
    assert isinstance(data, bytes)
    assert ju.loads(data)["value"] == "3.14"

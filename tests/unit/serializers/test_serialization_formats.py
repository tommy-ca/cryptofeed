import os

import pytest

from cryptofeed.serializers.formats import (
    CALLBACK_FORMAT_ENV_VAR,
    DEFAULT_SERIALIZATION_FORMAT,
    get_serialization_format_from_env,
    resolve_serialization_format,
    validate_serialization_format,
)


def test_validate_serialization_format_accepts_supported_values():
    assert validate_serialization_format("json") == "json"
    assert validate_serialization_format("Protobuf") == "protobuf"


def test_validate_serialization_format_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid serialization format"):
        validate_serialization_format("avro")


def test_get_serialization_format_from_env(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)
    assert get_serialization_format_from_env() is None

    monkeypatch.setenv(CALLBACK_FORMAT_ENV_VAR, "PROTOBUF")
    assert get_serialization_format_from_env() == "protobuf"


def test_resolve_serialization_format_precedence(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)
    assert resolve_serialization_format(None) == DEFAULT_SERIALIZATION_FORMAT

    assert resolve_serialization_format("Protobuf") == "protobuf"

    monkeypatch.setenv(CALLBACK_FORMAT_ENV_VAR, "json")
    assert resolve_serialization_format("protobuf") == "json"


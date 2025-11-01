import pytest

import cryptofeed.proto_bindings as bindings


def test_schema_version_constant():
    assert bindings.SCHEMA_VERSION == "v0.1.0"


def test_validate_bindings_no_missing_modules():
    bindings.validate_bindings()


def test_validate_bindings_error_for_missing(monkeypatch):
    monkeypatch.delitem(bindings.__dict__, 'trade_pb2', raising=True)

    with pytest.raises(ImportError):
        bindings.validate_bindings()

    # Restore for downstream tests
    from importlib import import_module

    module = import_module('gen.python.cryptofeed.normalized.v1.trade_pb2')
    bindings.trade_pb2 = module

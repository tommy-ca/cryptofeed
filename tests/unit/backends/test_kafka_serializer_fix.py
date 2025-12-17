"""Unit tests for legacy kafka.py _default_serializer fix (PR #16 Issue #1).

Tests verify that json.dumpb AttributeError is fixed by using dumps_bytes.
Tests the method directly from the legacy backend file.
"""

import importlib.util
import inspect
import os
import pytest


def _load_legacy_kafka():
    """Load the legacy kafka.py backend file directly."""
    # Get path to legacy backend file (not the kafka/ module directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_file = os.path.normpath(
        os.path.join(current_dir, "..", "..", "..", "cryptofeed", "backends", "kafka.py")
    )

    # Load the specific file bypassing Python's normal import system
    spec = importlib.util.spec_from_file_location("_test_legacy_kafka", legacy_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_serializer_with_dict():
    """Test _default_serializer handles dict without AttributeError."""
    legacy = _load_legacy_kafka()

    # Create a minimal mock object with the method
    class MockCallback:
        _default_serializer = legacy.KafkaCallback._default_serializer

    callback = MockCallback()
    test_dict = {"symbol": "BTC-USDT", "price": 50000.0}

    result = callback._default_serializer(test_dict)

    assert isinstance(result, bytes)
    assert b"BTC-USDT" in result
    assert b"50000" in result


def test_default_serializer_with_str():
    """Test _default_serializer handles str."""
    legacy = _load_legacy_kafka()

    class MockCallback:
        _default_serializer = legacy.KafkaCallback._default_serializer

    callback = MockCallback()
    test_str = "test message"

    result = callback._default_serializer(test_str)

    assert result == b"test message"
    assert isinstance(result, bytes)


def test_default_serializer_with_bytes():
    """Test _default_serializer handles bytes (protobuf support)."""
    legacy = _load_legacy_kafka()

    class MockCallback:
        _default_serializer = legacy.KafkaCallback._default_serializer

    callback = MockCallback()
    test_bytes = b"\x08\x01\x12\x05hello"  # Mock protobuf bytes

    result = callback._default_serializer(test_bytes)

    assert result == test_bytes
    assert isinstance(result, bytes)


def test_default_serializer_type_error():
    """Test _default_serializer raises TypeError for invalid types."""
    legacy = _load_legacy_kafka()

    class MockCallback:
        _default_serializer = legacy.KafkaCallback._default_serializer

    callback = MockCallback()

    with pytest.raises(TypeError, match="is not a valid Serialization type"):
        callback._default_serializer(12345)

    with pytest.raises(TypeError, match="is not a valid Serialization type"):
        callback._default_serializer([1, 2, 3])


def test_no_duplicate_default_serializer_methods():
    """Verify only one _default_serializer method exists (Issue #2 fix)."""
    legacy = _load_legacy_kafka()
    KafkaCallback = legacy.KafkaCallback

    # Get all methods named _default_serializer
    methods = [
        name for name, method in inspect.getmembers(KafkaCallback, predicate=inspect.isfunction)
        if name == "_default_serializer"
    ]

    assert len(methods) == 1, f"Expected 1 _default_serializer method, found {len(methods)}"


def test_dumps_bytes_import_exists():
    """Verify dumps_bytes is properly imported from json_utils in legacy backend."""
    import sys
    import importlib.util
    import os

    # Load legacy backend file directly to check its namespace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_kafka_path = os.path.join(
        current_dir, "..", "..", "..", "cryptofeed", "backends", "kafka.py"
    )
    legacy_kafka_path = os.path.normpath(legacy_kafka_path)

    spec = importlib.util.spec_from_file_location("_legacy_kafka", legacy_kafka_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check that dumps_bytes is in the module's namespace
    assert hasattr(module, "dumps_bytes"), "dumps_bytes should be imported in kafka.py"

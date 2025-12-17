import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest


SHIM_MODULES = [
    ("cryptofeed.kafka_callback", "KafkaCallback"),
    ("cryptofeed.kafka_callback", "KafkaProtobufCallback"),
    ("cryptofeed.kafka_callback", "MessageHeaders"),
    ("cryptofeed.kafka_callback", "HeaderEnricher"),
    ("cryptofeed.kafka_producer", "KafkaProducer"),
    ("cryptofeed.kafka_config", "KafkaConfig"),
    ("cryptofeed.backends.kafka_metrics", "PrometheusMetricsExporter"),
    ("cryptofeed.backends.protobuf_helpers", "serialize_to_protobuf"),
]


@pytest.mark.parametrize("module_path,symbol", SHIM_MODULES)
def test_shim_imports_emit_deprecation_warning(module_path: str, symbol: str) -> None:
    """
    Compatibility shims must emit DeprecationWarning yet expose expected symbols.
    """

    sys.modules.pop(module_path, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_path)

    warning_types = [warning.category for warning in caught]
    assert warning_types, f"{module_path} did not emit any warnings"
    assert any(
        issubclass(category, DeprecationWarning) for category in warning_types
    ), f"{module_path} did not emit DeprecationWarning"
    if module_path.startswith("cryptofeed.kafka_callback"):
        # kafka_callback shim now emits multiple warnings (once per re-export). Ensure at least one DeprecationWarning.
        pass
    assert hasattr(module, symbol), f"{module_path} missing {symbol}"


def _stub_producer_factory(config):
    class StubProducer:
        def __init__(self, cfg):
            self.cfg = cfg

        def list_topics(self, timeout=None):
            return {"topics": []}

        def produce(self, *args, **kwargs):
            pass

        def poll(self, timeout):
            return 0

        def flush(self, timeout=None):
            return 0

    return StubProducer(config)


def _load_legacy_kafka_module():
    legacy_path = (
        Path(__file__).resolve().parents[3] / "cryptofeed" / "backends" / "kafka.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cryptofeed.backends.kafka_legacy", legacy_path
    )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def test_legacy_and_unified_callbacks_can_coexist():
    """
    Ensure legacy TradeKafka and new KafkaCallback can be instantiated together.
    """
    pytest.importorskip("aiokafka")
    legacy_module = _load_legacy_kafka_module()
    TradeKafka = getattr(legacy_module, "TradeKafka")

    from cryptofeed.kafka_callback import KafkaCallback

    legacy = TradeKafka(bootstrap_servers=["kafka:9092"])
    modern = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_stub_producer_factory,
    )

    assert legacy.default_key == "trades"
    assert modern is not None

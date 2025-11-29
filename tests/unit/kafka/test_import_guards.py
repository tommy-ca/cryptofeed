from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "cryptofeed"


def _iter_code_files():
    for path in CODE_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        # Allow anything inside the modern Kafka package plus shims
        if str(rel).startswith("cryptofeed/backends/kafka"):
            continue
        if rel in {
            Path("cryptofeed/kafka_callback.py"),
            Path("cryptofeed/backends/kafka.py"),
        }:
            continue
        yield rel, path


def test_no_legacy_kafka_imports_outside_allowed_paths():
    violations = []
    for rel, path in _iter_code_files():
        text = path.read_text(encoding="utf-8")
        if "import cryptofeed.backends.kafka" in text or "from cryptofeed.backends import kafka" in text:
            violations.append(rel)
    assert not violations, f"Legacy kafka imports found outside allowed paths: {violations}"


def test_no_kafka_callback_protobuf_usage_in_code():
    allowed = {
        Path("cryptofeed/backends/kafka/protobuf_callback.py"),
    }
    violations = []
    for rel, path in _iter_code_files():
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "KafkaCallback" not in text:
            continue
        if 'serialization_format="protobuf"' in text or "serialization_format='protobuf'" in text:
            violations.append(rel)
    assert not violations, f"KafkaCallback protobuf usage found in code: {violations}"

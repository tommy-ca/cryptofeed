"""Legacy entrypoint for KafkaCallback integration tests.

The original monolithic KafkaCallback integration test suite has been
split into category-specific modules under ``tests/unit/kafka/``:

- test_kafka_callback_pipeline_core.py
- test_kafka_callback_message_types.py
- test_kafka_callback_errors_and_config.py
- test_kafka_callback_partitioning_and_headers.py
- test_kafka_callback_backward_compat.py
- test_kafka_callback_performance.py

Use those modules directly for focused test runs.
"""

from __future__ import annotations

import pytest


# Preserve the slow marker so that any direct references to this module
# still respect the original collection semantics.
pytestmark = pytest.mark.slow

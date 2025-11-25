"""
Compatibility shim for legacy Kafka callback imports.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "cryptofeed.kafka_callback is deprecated; import from "
    "cryptofeed.backends.kafka.callback instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cryptofeed.backends.kafka.base import _SUPPORTED_METHODS  # noqa: F401
from cryptofeed.backends.kafka.callback import *  # noqa: F401,F403
from cryptofeed.backends.kafka.headers import (  # noqa: F401
    MessageHeaders,
    OptionalHeaders,
    HeaderEnricher,
)
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback  # noqa: F401

"""
Shim module for backward compatibility with legacy protobuf binding imports.
"""

from __future__ import annotations

import warnings

from cryptofeed.backends.protobuf.bindings import *  # noqa: F401,F403

warnings.warn(
    "cryptofeed.proto_bindings is deprecated; "
    "use cryptofeed.backends.protobuf.bindings instead.",
    DeprecationWarning,
    stacklevel=2,
)

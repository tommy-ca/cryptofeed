"""
Cryptofeed package initialization.

Extends the package path so generated protobuf modules under ``gen/python``
can be imported as ``cryptofeed.normalized.v1``.
"""

from pathlib import Path

from cryptofeed.feedhandler import FeedHandler

__all__ = ["FeedHandler"]

_GEN_PACKAGE = Path(__file__).resolve().parent.parent / "gen" / "python" / "cryptofeed"
if _GEN_PACKAGE.exists():
    __path__.append(str(_GEN_PACKAGE))  # type: ignore[name-defined]

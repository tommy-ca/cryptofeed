from __future__ import annotations

import pytest

from cryptofeed.defines import BACKPACK
from cryptofeed.config import Config
from cryptofeed.exchanges import EXCHANGE_MAP
from cryptofeed.exchanges.backpack.feed import BackpackFeed
from cryptofeed.feedhandler import FeedHandler


def test_backpack_registered_as_native_feed():
    assert BACKPACK in EXCHANGE_MAP
    assert EXCHANGE_MAP[BACKPACK] is BackpackFeed


def test_backpack_ccxt_feed_identifier_rejected():
    handler = FeedHandler()
    with pytest.raises(ValueError, match="Backpack ccxt integration has been removed"):
        handler.add_feed("BACKPACK_CCXT")


def test_config_loader_warns_backpack_ccxt_references():
    config_dict = {
        "exchanges": {
            "backpack_ccxt": {}
        }
    }

    cfg = Config(config=config_dict)
    assert getattr(cfg, "_legacy_backpack_warning_emitted", False)

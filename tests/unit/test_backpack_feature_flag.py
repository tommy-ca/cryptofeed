from __future__ import annotations

import importlib

import pytest

from cryptofeed.config import Config
from cryptofeed.defines import BACKPACK
from cryptofeed.exchanges.backpack_toggle import (
    is_backpack_native_enabled,
    maybe_enable_from_config,
    set_backpack_native_enabled,
)


@pytest.fixture(autouse=True)
def reset_backpack_flag():
    set_backpack_native_enabled(False)
    importlib.reload(__import__('cryptofeed.exchanges', fromlist=['dummy']))
    yield
    set_backpack_native_enabled(False)
    importlib.reload(__import__('cryptofeed.exchanges', fromlist=['dummy']))


def _get_exchange_map():
    module = importlib.reload(__import__('cryptofeed.exchanges', fromlist=['EXCHANGE_MAP']))
    return module.EXCHANGE_MAP


def test_maybe_enable_from_config_enables_native():
    cfg = Config({'exchanges': {'backpack': {'native_enabled': True}}})
    maybe_enable_from_config(cfg)
    module = importlib.reload(__import__('cryptofeed.exchanges', fromlist=['EXCHANGE_MAP']))
    from cryptofeed.exchanges.backpack.feed import BackpackFeed
    assert is_backpack_native_enabled() is True
    assert module.EXCHANGE_MAP[BACKPACK] is BackpackFeed


def test_maybe_enable_from_config_disables_native():
    cfg = Config({'exchanges': {'backpack': {'native_enabled': False}}})
    maybe_enable_from_config(cfg)
    module = importlib.reload(__import__('cryptofeed.exchanges', fromlist=['EXCHANGE_MAP']))
    from cryptofeed.exchanges.backpack.feed import BackpackFeed
    assert is_backpack_native_enabled() is False
    assert module.EXCHANGE_MAP[BACKPACK] is BackpackFeed

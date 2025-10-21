from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from cryptofeed.feedhandler import FeedHandler
from cryptofeed.exchanges.backpack.config import BackpackConfig
from cryptofeed.exchanges.backpack.feed import BackpackFeedDependencies


class _StubRestClient:
    async def close(self):
        return None


class _StubWsSession:
    async def open(self):
        return None

    async def read(self):
        return "{}"

    async def send(self, _):
        return None

    async def close(self):
        return None


class _StubSymbolService:
    async def ensure(self):
        return None

    def native_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "_")

    def all_markets(self):
        return []


@dataclass
class _DependencyRecorder:
    rest_configs: List[BackpackConfig]
    ws_configs: List[BackpackConfig]

    def __init__(self):
        self.rest_configs = []
        self.ws_configs = []

    def build(self) -> BackpackFeedDependencies:
        return BackpackFeedDependencies(
            rest_client_factory=self._record_rest,
            ws_session_factory=self._record_ws,
            symbol_service=_StubSymbolService(),
        )

    def _record_rest(self, config: BackpackConfig):
        self.rest_configs.append(config)
        return _StubRestClient()

    def _record_ws(self, config: BackpackConfig):
        self.ws_configs.append(config)
        return _StubWsSession()


def _add_backpack_feed(handler: FeedHandler, *, config=None, recorder=None):
    recorder = recorder or _DependencyRecorder()
    handler.add_feed(
        "BACKPACK",
        config=config,
        dependencies=recorder.build(),
        symbols=["BTC-USDT"],
        channels=["trades"],
    )
    return handler.feeds[-1], recorder


def test_feedhandler_uses_backpack_section_from_root_config():
    recorder = _DependencyRecorder()
    handler = FeedHandler(
        config={
            "log": {"level": "WARNING", "filename": "feedhandler.log"},
            "backpack": {"use_sandbox": True},
        }
    )

    feed, recorder = _add_backpack_feed(handler, recorder=recorder)
    connection = feed.connect()[0][0]
    asyncio.run(connection._open())
    asyncio.run(connection.close())

    assert isinstance(feed.exchange_config, BackpackConfig)
    assert feed.exchange_config.use_sandbox is True
    assert recorder.rest_configs[0] is feed.exchange_config
    assert recorder.ws_configs[0] is feed.exchange_config


def test_feedhandler_supports_backpack_config_under_exchanges():
    recorder = _DependencyRecorder()
    handler = FeedHandler(
        config={
            "log": {"level": "WARNING", "filename": "feedhandler.log"},
            "exchanges": {"BACKPACK": {"use_sandbox": True}},
        }
    )

    feed, recorder = _add_backpack_feed(handler, recorder=recorder)
    connection = feed.connect()[0][0]
    asyncio.run(connection._open())
    asyncio.run(connection.close())

    assert feed.exchange_config.use_sandbox is True
    assert recorder.rest_configs[0] is feed.exchange_config
    assert recorder.ws_configs[0] is feed.exchange_config


def test_feedhandler_accepts_explicit_backpackconfig_override():
    recorder = _DependencyRecorder()
    handler = FeedHandler()
    override = BackpackConfig(use_sandbox=True)

    feed, recorder = _add_backpack_feed(handler, config=override, recorder=recorder)
    connection = feed.connect()[0][0]
    asyncio.run(connection._open())
    asyncio.run(connection.close())

    assert feed.exchange_config is override
    assert recorder.rest_configs[0] is override
    assert recorder.ws_configs[0] is override


def test_feedhandler_validates_mapping_override_into_backpackconfig():
    recorder = _DependencyRecorder()
    handler = FeedHandler()

    feed, recorder = _add_backpack_feed(handler, config={"use_sandbox": True}, recorder=recorder)
    connection = feed.connect()[0][0]
    asyncio.run(connection._open())
    asyncio.run(connection.close())

    assert isinstance(feed.exchange_config, BackpackConfig)
    assert feed.exchange_config.use_sandbox is True
    assert recorder.rest_configs[0] is feed.exchange_config
    assert recorder.ws_configs[0] is feed.exchange_config

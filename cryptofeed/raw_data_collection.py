"""
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import asyncio
import atexit
from collections import defaultdict
import functools
import ast
from contextlib import contextmanager

from aiofile import AIOFile

from cryptofeed.defines import HUOBI, UPBIT, OKX, OKCOIN
from cryptofeed.exchanges import EXCHANGE_MAP
from cryptofeed.json_utils import loads as json_loads, dumps as json_dumps


class _PlaybackFakeWS:
    def __init__(self, filenames):
        self.conn_type = "wss"
        self.uuid = "1"
        self.cache = defaultdict(list)

        for filename in filenames:
            if "http" not in filename:
                continue
            with open(filename, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    if not line.startswith("http"):
                        continue
                    file_url, data = line.split(" -> ")
                    _, msg = data.split(": ", 1)
                    self.cache[file_url].append(msg)

    async def write(self, *args, **kwargs):
        pass

    async def read(self, url, **kwargs):
        data = self.cache[url].pop(0)
        if "header:" in data:
            payload, header = data.split(" header: ")
            return payload, json_loads(header.strip())
        return data


def _load_subscription_data(filenames):
    symbol_data = []
    subscription = None

    for path in filenames:
        if "ws" in path or "http" in path:
            continue
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if "configuration" in line:
                    subscription = json_loads(line.split(": ", 1)[1])
                if line == "\n":
                    continue
                payload = line.split(": ", 1)[1]
                symbol_data.append(json_loads(payload.strip()))

    return symbol_data, subscription


def _make_symbol_helper(symbol_data):
    def symbol_helper(*args, **kwargs):
        return symbol_data.pop(0)

    return symbol_helper


def _augment_callbacks(subscription, callbacks, callback_stats):
    async def internal_cb(*args, **kwargs):
        callback_stats[kwargs["cb_type"]] += 1

    def tracker(cb_type):
        return functools.partial(internal_cb, cb_type=cb_type)

    if not subscription:
        return callbacks or {}

    if callbacks is None:
        return {cb_type: tracker(cb_type) for cb_type in subscription.keys()}

    updated = {}
    for cb_type, cb in callbacks.items():
        if isinstance(cb, list):
            updated[cb_type] = cb + [tracker(cb_type)]
        else:
            updated[cb_type] = [cb, tracker(cb_type)]

    for cb_type in subscription.keys():
        if cb_type not in updated:
            updated[cb_type] = [tracker(cb_type)]

    return updated


def _convert_subscription(feed, ws_subscription, subscription):
    if not ws_subscription or not subscription:
        return ws_subscription

    exchange_sub = {}
    for chan in ws_subscription:
        exchange_channel = feed.std_channel_to_exchange(chan)
        symbols = [
            feed.std_symbol_to_exchange_symbol(symbol) for symbol in subscription[chan]
        ]
        exchange_sub[exchange_channel] = symbols
    return exchange_sub


def _filter_ws_files(filenames):
    return [filename for filename in filenames if ".ws." in filename]


@contextmanager
def _patched_http_reads(ws, symbol_helper):
    from cryptofeed.connection import HTTPAsyncConn, HTTPSync

    http_async_conn_read = HTTPAsyncConn.read
    http_sync_read = HTTPSync.read
    HTTPAsyncConn.read = ws.read
    HTTPSync.read = symbol_helper
    try:
        yield
    finally:
        HTTPAsyncConn.read = http_async_conn_read
        HTTPSync.read = http_sync_read


async def _replay_ws_files(filenames, handler, ws, feed) -> int:
    counter = 0
    if handler is None:
        return counter

    for filename in filenames:
        counter = await _replay_ws_file(filename, handler, ws, feed, counter)
    return counter


async def _replay_ws_file(filename, handler, ws, feed, counter: int) -> int:
    with open(filename, "r") as fp:
        for line in fp:
            if line == "\n":
                continue
            prefix = line[:3]
            if prefix == "wss":
                continue
            if prefix == "htt":
                counter += 1
                continue

            timestamp, message = line.split(": ", 1)
            counter += 1
            message = _normalize_message(filename, message)

            try:
                await handler(message, ws, timestamp)
            except Exception:
                print("Playback failed on message:", message)
                feed.stop()
                await feed.shutdown()
                raise
    return counter


def _normalize_message(filename: str, message: str) -> str:
    if OKCOIN in filename or OKX in filename:
        if message.startswith("b'") or message.startswith('b"'):
            return bytes_string_to_bytes(message)
        return message
    if HUOBI in filename:
        return bytes_string_to_bytes(message)
    if UPBIT in filename and (message.startswith("b'") or message.startswith('b"')):
        return message.strip()[2:-1]
    return message


def bytes_string_to_bytes(string):
    tree = ast.parse(string)
    return tree.body[0].value.s


def playback(
    feed: str, filenames: list, callbacks: dict = None, config: str = "config.yaml"
):
    return asyncio.run(_playback(feed, filenames, callbacks, config))


async def _playback(feed: str, filenames: list, callbacks: dict, config: str):
    callback_stats = defaultdict(int)
    ws = _PlaybackFakeWS(filenames)
    symbol_data, subscription = _load_subscription_data(filenames)
    symbol_helper = _make_symbol_helper(symbol_data)

    ws.subscription = subscription or {}

    callbacks = _augment_callbacks(subscription, callbacks, callback_stats)

    with _patched_http_reads(ws, symbol_helper):
        feed_instance = EXCHANGE_MAP[feed](
            candle_closed_only=False,
            config=config,
            subscription=subscription,
            callbacks=callbacks,
        )

        ws.subscription = _convert_subscription(
            feed_instance, ws.subscription, subscription
        )
        connections = feed_instance.connect()

        handler = None
        for _, subscribe_fn, handler, _ in connections:
            await subscribe_fn(ws)

        ws_files = _filter_ws_files(filenames)

        try:
            counter = await _replay_ws_files(ws_files, handler, ws, feed_instance)
        finally:
            feed_instance.stop()
            await feed_instance.shutdown()

    return {"messages_processed": counter, "callbacks": dict(callback_stats)}


class AsyncFileCallback:
    def __init__(self, path, length=10000, rotate=1024 * 1024 * 100):
        self.path = path
        self.length = length
        self.data = defaultdict(list)
        self.rotate = rotate
        self.count = defaultdict(int)
        self.pointer = defaultdict(int)
        atexit.register(self.__del__)

    def __del__(self):
        self.stop()

    def stop(self):
        for uuid in list(self.data.keys()):
            with open(f"{self.path}/{uuid}.{self.count[uuid]}", "a") as fp:
                fp.write("\n".join(self.data[uuid]) + "\n")
                self.data[uuid] = []
                fp.flush()

    def write_header(self, uuid, data):
        with open(f"{self.path}/{uuid}.{0}", "a") as fp:
            fp.write(f"configuration: {data}\n")
            fp.flush()

    async def write(self, uuid):
        p = f"{self.path}/{uuid}.{self.count[uuid]}"
        async with AIOFile(p, mode="a") as fp:
            r = await fp.write(
                "\n".join(self.data[uuid]) + "\n", offset=self.pointer[uuid]
            )
            self.pointer[uuid] += r
            self.data[uuid] = []
            await fp.fsync()

        if self.pointer[uuid] >= self.rotate:
            self.count[uuid] += 1
            self.pointer[uuid] = 0

    async def __call__(
        self,
        data: str,
        timestamp: float,
        uuid: str,
        endpoint: str = None,
        send: str = None,
        connect: str = None,
        header: str = None,
    ):
        if endpoint:
            if header:
                self.data[uuid].append(
                    f"{endpoint} -> {timestamp}: {data} header: {json_dumps(header)}"
                )
            else:
                data = data.replace("\n", "")
                self.data[uuid].append(f"{endpoint} -> {timestamp}: {data}")
        elif send:
            self.data[uuid].append(f"{send} <- {timestamp}: {data}")
        elif connect:
            self.data[uuid].append(f"{connect} <-> {timestamp}")
        else:
            self.data[uuid].append(f"{timestamp}: {data}")

        if len(self.data[uuid]) >= self.length:
            await asyncio.create_task(self.write(uuid))

    def sync_callback(
        self,
        data: str,
        timestamp: float,
        uuid: str,
        endpoint: str = None,
        send: str = None,
        connect: str = None,
        header: str = None,
    ):
        if endpoint:
            if header:
                w = w = (
                    f"{endpoint} -> {timestamp}: {data} header: {json_dumps(header)}"
                )
            else:
                data = data.replace("\n", "")
                w = f"{endpoint} -> {timestamp}: {data}"
        elif send:
            w = f"{send} <- {timestamp}: {data}"
        elif connect:
            w = f"{connect} <-> {timestamp}"
        else:
            w = f"{timestamp}: {data}"

        with open(f"{self.path}/{uuid}.{0}", "a") as fp:
            fp.write(w + "\n")
            fp.flush()

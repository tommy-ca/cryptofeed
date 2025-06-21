"""Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com.

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import asyncio
import glob
import random

from check_raw_dump import main as check_dump
import uvloop

from cryptofeed.defines import (
    BINANCE,
    BINANCE_FUTURES,
    BINANCE_TR,
    BINANCE_US,
    BITFINEX,
    CANDLES,
    EXX,
    L2_BOOK,
    TICKER,
    TRADES,
)
from cryptofeed.exchanges import EXCHANGE_MAP
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.raw_data_collection import AsyncFileCallback


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def stop():
    loop = asyncio.get_event_loop()
    loop.stop()


def main(only_exchange=None):
    skip = [EXX]
    files = glob.glob("*")
    for f in files:
        for e in EXCHANGE_MAP:
            if e + "." in f:
                skip.append(e.split(".")[0])

    loop = asyncio.get_event_loop()
    for exch_str, exchange in (
        EXCHANGE_MAP.items() if only_exchange is None else [(only_exchange, EXCHANGE_MAP[only_exchange])]
    ):
        if exch_str in skip:
            continue

        fh = FeedHandler(
            raw_data_collection=AsyncFileCallback("./"),
            config={
                "uvloop": False,
                "log": {"filename": "feedhandler.log", "level": "WARNING"},
                "rest": {"log": {"filename": "rest.log", "level": "WARNING"}},
            },
        )
        info = exchange.info()
        channels = list(set.intersection(set(info["channels"]["websocket"]), set([L2_BOOK, TRADES, TICKER, CANDLES])))
        sample_size = 10
        if exch_str in (BINANCE_US, BINANCE_TR, BINANCE):
            # books of size 5000 count significantly against rate limits
            sample_size = 4
        while True:
            try:
                symbols = random.sample(info["symbols"], sample_size)

                if exch_str == BINANCE_FUTURES:
                    symbols = [s for s in symbols if "PINDEX" not in s]
                elif exch_str == BITFINEX:
                    symbols = [s for s in symbols if "-" in s]

            except ValueError:
                sample_size -= 1
            else:
                break

        fh.add_feed(exchange(symbols=symbols, channels=channels))
        fh.run(start_loop=False)

        loop.call_later(31, stop)
        loop.run_forever()

        fh.stop(loop=loop)
        del fh

    for exch_str, _ in EXCHANGE_MAP.items():
        for file in glob.glob(exch_str + "*"):
            try:
                check_dump(file)
            except Exception as e:
                continue


if __name__ == "__main__":
    main("BIT.COM")

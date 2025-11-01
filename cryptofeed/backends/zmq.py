'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from collections import defaultdict

import zmq
import zmq.asyncio
from cryptofeed.json_utils import json

from cryptofeed.backends.backend import BackendQueue, BackendBookCallback, BackendCallback


class ZMQCallback(BackendQueue):
    def __init__(self, host='127.0.0.1', port=5555, none_to=None, numeric_type=float, key=None, dynamic_key=True, **kwargs):
        self.url = "tcp://{}:{}".format(host, port)
        self.key = key if key else self.default_key
        self.numeric_type = numeric_type
        self.none_to = none_to
        self.dynamic_key = dynamic_key
        self.running = True

    async def __call__(self, dtype, receipt_timestamp: float):
        fmt = self.serialization_format

        if fmt == 'json':
            await BackendCallback.__call__(self, dtype, receipt_timestamp)
            return

        serializer = self._get_serializer(fmt)
        payload = serializer.serialize(dtype)
        metadata = self._build_dict_payload(dtype, receipt_timestamp)

        message = {
            'format': fmt,
            'content_type': serializer.content_type(),
            'payload': payload,
            'metadata': metadata,
        }

        await self.write(message)

    async def writer(self):
        ctx = zmq.asyncio.Context.instance()
        con = ctx.socket(zmq.PUB)
        con.connect(self.url)
        while self.running:
            async with self.read_queue() as updates:
                for update in updates:
                    if isinstance(update, dict) and update.get('format') == 'protobuf':
                        metadata = update['metadata']
                        topic = f"{metadata['exchange']}-{self.key}-{metadata['symbol']}" if self.dynamic_key else self.key
                        header = json.dumps({
                            'format': update['format'],
                            'content_type': update['content_type'],
                            'metadata': metadata,
                        }).encode()
                        await con.send_multipart([topic.encode(), header, update['payload']])
                        continue

                    if self.dynamic_key:
                        message = f'{update["exchange"]}-{self.key}-{update["symbol"]} {json.dumps(update)}'
                    else:
                        message = f'{self.key} {json.dumps(update)}'
                    await con.send_string(message)


class TradeZMQ(ZMQCallback, BackendCallback):
    default_key = 'trades'


class TickerZMQ(ZMQCallback, BackendCallback):
    default_key = 'ticker'


class FundingZMQ(ZMQCallback, BackendCallback):
    default_key = 'funding'


class BookZMQ(ZMQCallback, BackendBookCallback):
    default_key = 'book'

    def __init__(self, *args, snapshots_only=False, snapshot_interval=1000, **kwargs):
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, **kwargs)


class OpenInterestZMQ(ZMQCallback, BackendCallback):
    default_key = 'open_interest'


class LiquidationsZMQ(ZMQCallback, BackendCallback):
    default_key = 'liquidations'


class CandlesZMQ(ZMQCallback, BackendCallback):
    default_key = 'candles'


class BalancesZMQ(ZMQCallback, BackendCallback):
    default_key = 'balances'


class PositionsZMQ(ZMQCallback, BackendCallback):
    default_key = 'positions'


class OrderInfoZMQ(ZMQCallback, BackendCallback):
    default_key = 'order_info'


class FillsZMQ(ZMQCallback, BackendCallback):
    default_key = 'fills'


class TransactionsZMQ(ZMQCallback, BackendCallback):
    default_key = 'transactions'

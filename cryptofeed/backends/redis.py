'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from collections import defaultdict
import base64

from redis import asyncio as aioredis
from cryptofeed.json_utils import json

from cryptofeed.backends.backend import BackendBookCallback, BackendCallback, BackendQueue


class RedisCallback(BackendQueue):
    def __init__(self, host='127.0.0.1', port=6379, socket=None, key=None, none_to='None', numeric_type=float, serialization_format=None, **kwargs):
        """
        setting key lets you override the prefix on the
        key used in redis. The defaults are related to the data
        being stored, i.e. trade, funding, etc
        """
        prefix = 'redis://'
        if socket:
            prefix = 'unix://'
            port = None

        self.redis = f"{prefix}{host}" + f":{port}" if port else ""
        self.key = key if key else self.default_key
        self.numeric_type = numeric_type
        self.none_to = none_to
        self.running = True
        if serialization_format is not None:
            self.set_serialization_format(serialization_format)

    def _prepare_json_record(self, update: dict) -> dict:
        if isinstance(update, dict) and update.get('format') == 'protobuf':
            encoded = base64.b64encode(update['payload']).decode('ascii')
            return {
                'format': 'protobuf',
                'content_type': update['content_type'],
                'metadata': update['metadata'],
                'payload_b64': encoded,
            }
        return update

    async def __call__(self, dtype, receipt_timestamp: float):
        # Use parent class serialization handling
        await BackendCallback.__call__(self, dtype, receipt_timestamp)

    def _prepare_stream_record(self, update: dict) -> dict:
        if isinstance(update, dict) and update.get('format') == 'protobuf':
            return {
                'format': 'protobuf',
                'content_type': update['content_type'],
                'metadata': json.dumps(update['metadata']),
                'payload': update['payload'],
            }
        return update


class RedisZSetCallback(RedisCallback):
    def __init__(self, host='127.0.0.1', port=6379, socket=None, key=None, numeric_type=float, score_key='timestamp', **kwargs):
        """
        score_key: str
            the value at this key will be used to store the data in the ZSet in redis. The
            default is timestamp. If you wish to look up the data by a different value,
            use this to change it. It must be a numeric value.
        """
        self.score_key = score_key
        super().__init__(host=host, port=port, socket=socket, key=key, numeric_type=numeric_type, **kwargs)

    async def writer(self):
        conn = aioredis.from_url(self.redis)

        while self.running:
            async with self.read_queue() as updates:
                async with conn.pipeline(transaction=False) as pipe:
                    for update in updates:
                        record = self._prepare_json_record(update)
                        pipe = pipe.zadd(
                            f"{self.key}-{record['metadata']['exchange']}-{record['metadata']['symbol']}" if record.get('format') == 'protobuf' else f"{self.key}-{update['exchange']}-{update['symbol']}",
                            {json.dumps(record): (record['metadata']['receipt_timestamp'] if record.get('format') == 'protobuf' else update[self.score_key])},
                            nx=True,
                        )
                    await pipe.execute()

        await conn.close()
        await conn.connection_pool.disconnect()


class RedisStreamCallback(RedisCallback):
    async def writer(self):
        conn = aioredis.from_url(self.redis)

        while self.running:
            async with self.read_queue() as updates:
                async with conn.pipeline(transaction=False) as pipe:
                    for update in updates:
                        if isinstance(update, dict) and update.get('format') == 'protobuf':
                            record = self._prepare_stream_record(update)
                            metadata = json.loads(record['metadata'])
                            stream_key = f"{self.key}-{metadata['exchange']}-{metadata['symbol']}"
                        else:
                            record = update
                            stream_key = f"{self.key}-{update['exchange']}-{update['symbol']}"
                            if 'delta' in record:
                                record['delta'] = json.dumps(record['delta'])
                            elif 'book' in record:
                                record['book'] = json.dumps(record['book'])
                            elif 'closed' in record:
                                record['closed'] = str(record['closed'])

                        pipe = pipe.xadd(stream_key, record)
                    await pipe.execute()

        await conn.close()
        await conn.connection_pool.disconnect()


class RedisKeyCallback(RedisCallback):

    async def writer(self):
        conn = aioredis.from_url(self.redis)

        while self.running:
            async with self.read_queue() as updates:
                update = list(updates)[-1]
                if update:
                    record = self._prepare_json_record(update)
                    if record.get('format') == 'protobuf':
                        metadata = record['metadata']
                        key = f"{self.key}-{metadata['exchange']}-{metadata['symbol']}"
                    else:
                        key = f"{self.key}-{update['exchange']}-{update['symbol']}"
                    await conn.set(key, json.dumps(record))

        await conn.close()
        await conn.connection_pool.disconnect()


class TradeRedis(RedisZSetCallback, BackendCallback):
    default_key = 'trades'


class TradeStream(RedisStreamCallback, BackendCallback):
    default_key = 'trades'


class FundingRedis(RedisZSetCallback, BackendCallback):
    default_key = 'funding'


class FundingStream(RedisStreamCallback, BackendCallback):
    default_key = 'funding'


class BookRedis(RedisZSetCallback, BackendBookCallback):
    default_key = 'book'

    def __init__(self, *args, snapshots_only=False, snapshot_interval=1000, score_key='receipt_timestamp', **kwargs):
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, score_key=score_key, **kwargs)


class BookStream(RedisStreamCallback, BackendBookCallback):
    default_key = 'book'

    def __init__(self, *args, snapshots_only=False, snapshot_interval=1000, **kwargs):
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, **kwargs)


class BookSnapshotRedisKey(RedisKeyCallback, BackendBookCallback):
    default_key = 'book'

    def __init__(self, *args, snapshot_interval=1000, score_key='receipt_timestamp', **kwargs):
        kwargs['snapshots_only'] = True
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, score_key=score_key, **kwargs)


class TickerRedis(RedisZSetCallback, BackendCallback):
    default_key = 'ticker'


class TickerStream(RedisStreamCallback, BackendCallback):
    default_key = 'ticker'


class OpenInterestRedis(RedisZSetCallback, BackendCallback):
    default_key = 'open_interest'


class OpenInterestStream(RedisStreamCallback, BackendCallback):
    default_key = 'open_interest'


class LiquidationsRedis(RedisZSetCallback, BackendCallback):
    default_key = 'liquidations'


class LiquidationsStream(RedisStreamCallback, BackendCallback):
    default_key = 'liquidations'


class CandlesRedis(RedisZSetCallback, BackendCallback):
    default_key = 'candles'


class CandlesStream(RedisStreamCallback, BackendCallback):
    default_key = 'candles'


class OrderInfoRedis(RedisZSetCallback, BackendCallback):
    default_key = 'order_info'


class OrderInfoStream(RedisStreamCallback, BackendCallback):
    default_key = 'order_info'


class TransactionsRedis(RedisZSetCallback, BackendCallback):
    default_key = 'transactions'


class TransactionsStream(RedisStreamCallback, BackendCallback):
    default_key = 'transactions'


class BalancesRedis(RedisZSetCallback, BackendCallback):
    default_key = 'balances'


class BalancesStream(RedisStreamCallback, BackendCallback):
    default_key = 'balances'


class FillsRedis(RedisZSetCallback, BackendCallback):
    default_key = 'fills'


class FillsStream(RedisStreamCallback, BackendCallback):
    default_key = 'fills'

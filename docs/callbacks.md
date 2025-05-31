## Using Callbacks

Cryptofeed is a library that uses asyncio to handle asynchronous events. When `fh.run()` is called, the main program thread of execution will block until an exception is hit or the user terminates the program (There is a slight exception to this, if `run` is called with the kwarg `start_loop=False` the feedhandler will not be started, the user can add more tasks/coroutines, and will then be responsible for starting the event loop later). Because the program is effectively blocked on the event loop, the user needs to define callbacks that will handle data from cryptofeed. Only data you register for will be delivered via these callbacks.

### Callback Types

There are two types of callbacks supported in cryptofeed, *raw* and *backend*. The raw callbacks deliver the data directly to the specified function. Backend callbacks take the data and do something else with it (typically store or send). Some examples of the backend callbacks are Redis, Postgres and TCP. You might use the Redis or Postgres callbacks to store the data, and you could use the TCP callback to send data to another application for processing.

The raw callbacks are defined [here](../cryptofeed/callback.py). They are:

* Trade
* Ticker
* Book
* Open Interest
* Funding
* Liquidation
* Candles
* Index
* L1Book (aka Top of Book)
* Order Info
* User Fills
* Transactions
* Balances

It's important to note that if your choose to use the raw callbacks and your callbacks are async functions, you do not need to use these wrappers (like is commonly shown in the example code). You can use your callback functions without wrapping them in `TradeCallback`, `TickerCallback`, etc.

Every callback has the same signature, two positional arguments, the data object and the receipt timestamp. The data object differs by data type. The data objects are defined in [types.pyx](../cryptofeed/types.pyx)


### Backends

The backends are defined [here](../cryptofeed/backends/). Currently the following are supported:

* Arctic
* ElasticSearch
* GCP Pub/Sub
* InfluxDB
* Kafka
* MongoDB
* Postgres
* QuestDB
* RabbitMQ
* Redis
* Redis Streams
* TCP/UDP/UDS sockets
* VictoriaMetrics
* ZMQ
* NATS

There are also a handful of wrappers defined [here](../cryptofeed/backends/aggregate.py) that can be used in conjunction with these and raw callbacks to convert data to OHLCV, throttle data, etc.

### NATS Backend

The NATS backend allows publishing data from Cryptofeed to NATS subjects.

**Installation**

To use the NATS backend, you need to install the necessary extra dependencies:

```bash
pip install cryptofeed[nats]
```

This will install the `nats-py` library.

**Configuration**

The NATS backend is configured within the `callbacks` section of your Cryptofeed configuration.

Example using a Python dictionary:

```python
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.backends.nats import TradeNATS, BookNATS

config = {
    'log': {
        'filename': 'feedhandler.log',
        'level': 'INFO'
    },
    'callbacks': {
        TRADES: TradeNATS(addr=['nats://localhost:4222', 'nats://another_server:4222'], subject_prefix='crypto.feed'),
        L2_BOOK: BookNATS(addr='nats://localhost:4222', subject_prefix='crypto.l2book')
    }
}
```

Example using `config.yaml`:

```yaml
log:
  filename: feedhandler.log
  level: INFO
callbacks:
  TRADES:
    class: TradeNATS
    addr: ['nats://localhost:4222', 'nats://another_server:4222'] # Can be a single string or a list of strings
    subject_prefix: crypto.feed # Optional, defaults to 'cryptofeed'
  L2_BOOK:
    class: BookNATS
    addr: nats://localhost:4222
    subject_prefix: crypto.l2book # Optional, defaults to 'cryptofeed'
    # Other options like 'book_depth', 'max_depth' can be added for BookNATS
```

**Configuration Parameters:**

*   `class`: The specific NATS callback class to use (e.g., `TradeNATS`, `BookNATS`).
*   `addr`: (Required) A NATS server URL string or a list of NATS server URL strings. Defaults to `'nats://localhost:4222'` if not provided in the specific callback constructor, but it's best to specify it explicitly.
*   `subject_prefix`: (Optional) A prefix for all NATS subjects published by this callback. Defaults to `'cryptofeed'`.
*   Other parameters specific to the data type (e.g., `book_depth` for `BookNATS`) can also be passed.

**NATS Subject Naming**

The NATS subjects are constructed using the following pattern:

`<subject_prefix>-<data_type>-<exchange>-<symbol>`

Where:
*   `<subject_prefix>` is the configured prefix (e.g., `crypto.feed`).
*   `<data_type>` is derived from the callback class (e.g., `trades` for `TradeNATS`, `book` for `BookNATS`).
*   `<exchange>` is the name of the exchange (e.g., `coinbase`).
*   `<symbol>` is the trading symbol (e.g., `BTC-USD`).

For example, a trade update for BTC-USD from Coinbase with `subject_prefix='crypto.feed'` would be published to: `crypto.feed-trades-coinbase-BTC-USD`.

**Available NATS Callback Classes:**

The following NATS-specific callback classes are available in `cryptofeed.backends.nats`:

*   `TradeNATS`
*   `BookNATS`
*   `TickerNATS`
*   `FundingNATS`
*   `OpenInterestNATS`
*   `LiquidationsNATS`
*   `CandlesNATS`
*   `OrderInfoNATS`
*   `TransactionsNATS`
*   `BalancesNATS`
*   `FillsNATS`

These classes inherit the appropriate base callback functionality (e.g., `BackendCallback`, `BackendBookCallback`) and handle publishing to NATS.

### Performance Considerations

Do not do anything computationally intensive in your callbacks, or this will greatly impact the performance of cryptofeed. Data should be quickly processed and passed along to another process/application/etc or a backend callback should be used to forward the data elsewhere. If possible, use async libraries in your callbacks!

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
* InfluxDB v3
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

There are also a handful of wrappers defined [here](../cryptofeed/backends/aggregate.py) that can be used in conjunction with these and raw callbacks to convert data to OHLCV, throttle data, etc. 

### Performance Considerations

Do not do anything computationally intensive in your callbacks, or this will greatly impact the performance of cryptofeed. Data should be quickly processed and passed along to another process/application/etc or a backend callback should be used to forward the data elsewhere. If possible, use async libraries in your callbacks!

## InfluxDB v3

This backend uses the official `influxdb3-python` client library. Ensure it is installed in your environment (`pip install influxdb3-python`).

The InfluxDB v3 backend allows you to stream real-time cryptocurrency data directly into an InfluxDB v3 instance (Cloud or OSS). The integration with the `influxdb3-python` client means that batching of data points and retry mechanisms are handled efficiently by the client library itself.

**Key Features:**
- Writes data using InfluxDB `Point` objects, converted from Cryptofeed data types.
- Supports various data types (trades, order books, tickers, etc.) through specific callback classes.
- Configurable InfluxDB v3 connection parameters.

**Configuration Parameters:**

When initializing an `InfluxDB3Callback` subclass (e.g., `TradeInflux3`, `BookInflux3`), the following parameters are used:

-   `addr` (str): The full HTTP(S) address of your InfluxDB v3 instance (e.g., `"http://localhost:8086"` for OSS, or `"https://<your-region>.cloud2.influxdata.com"` for Cloud). This is used as the `host` parameter for the `InfluxDBClient3`.
-   `token` (str): Your InfluxDB API token. This token must have write permissions to the specified `database`.
-   `database` (str): The name of the database (bucket in InfluxDB v3 terminology) where data will be written.
-   `org` (str, optional): Your InfluxDB organization ID or name. This is typically required for InfluxDB Cloud and may be needed for InfluxDB OSS depending on your setup.
-   `key` (str, optional): Advanced users can use this parameter to override the `default_key` set by specific subclasses (like `TradeInflux3`) or to provide a custom measurement prefix if using `InfluxDB3Callback` directly. Subclasses such as `TradeInflux3` automatically set this to their respective `default_key` (e.g., `'trades'`).
-   `batch_size` (int, optional): The number of data points to collect in a batch before writing to InfluxDB. Defaults to `5000`. This is handled by the `influxdb3-python` client's `WriteOptions`.
-   `flush_interval` (int, optional): The maximum time in milliseconds to wait before writing a batch, even if `batch_size` isn't reached. Defaults to `10000`. This is handled by the `influxdb3-python` client's `WriteOptions`.

**Available Callback Classes:**

A suite of callback classes is provided for different data types. These inherit from `InfluxDB3Callback` and automatically configure the appropriate `key` for measurements:
-   `TradeInflux3`
-   `FundingInflux3`
-   `BookInflux3` (handles L2 order book data)
-   `TickerInflux3`
-   `OpenInterestInflux3`
-   `LiquidationsInflux3`
-   `CandlesInflux3`
-   `OrderInfoInflux3`
-   `TransactionsInflux3`
-   `BalancesInflux3`
-   `FillsInflux3`

Each class automatically sets a `default_key` (e.g., `'trades'`, `'book'`) which is used to construct the measurement name in InfluxDB (e.g., `trades-COINBASE`, `book-BINANCE`).

**Example Usage:**

```python
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends import TradeInflux3, BookInflux3 # Assuming __init__.py is updated
from cryptofeed.defines import COINBASE, L2_BOOK, TRADES

# InfluxDB v3 Configuration
INFLUXDB_ADDRESS = "http://localhost:8086"  # Replace with your InfluxDB v3 address
INFLUXDB_DATABASE = "cryptofeed_data"       # Replace with your database name
INFLUXDB_TOKEN = "your_api_token"           # Replace with your InfluxDB API token
INFLUXDB_ORG = "your_organization"          # Replace with your InfluxDB organization

def main():
    fh = FeedHandler()

    # Instantiate InfluxDB v3 callbacks
    trade_cb = TradeInflux3(
        addr=INFLUXDB_ADDRESS,
        database=INFLUXDB_DATABASE,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        # Optional: configure client-side batching (handled by influxdb3-python client)
        # batch_size=5000,       # Default is 5000
        # flush_interval=10000,  # Default is 10000ms
    )

    book_cb = BookInflux3(
        addr=INFLUXDB_ADDRESS,
        database=INFLUXDB_DATABASE,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        # Optional: configure client-side batching
        # batch_size=5000,
        # flush_interval=10000, # in milliseconds
        # Book-specific options:
        snapshots_only=False,    # If True, send only snapshots. If False, send deltas and periodic snapshots.
        snapshot_interval=1000   # If snapshots_only=False, send snapshot every 1000 deltas.
                                 # If snapshots_only=True, send snapshot every 1000 seconds.
    )

    # Add subscriptions
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[TRADES], callbacks=[trade_cb])
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[L2_BOOK], callbacks=[book_cb])

    fh.run()

if __name__ == '__main__':
    main()
```

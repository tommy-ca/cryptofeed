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

The InfluxDB v3 backend allows you to stream real-time cryptocurrency data directly into an InfluxDB v3 instance (Cloud or OSS). It uses the InfluxDB v3 `/api/v2/write` endpoint with line protocol.

**Key Features:**
- Writes data in InfluxDB line protocol.
- Supports various data types (trades, order books, tickers, etc.) through specific callback classes.
- Configurable InfluxDB v3 connection parameters.

**Configuration Parameters:**

When initializing an InfluxDB v3 callback (e.g., `TradeInflux3`, `BookInflux3`), the following parameters are used:

-   `addr` (str): The full HTTP(S) address of your InfluxDB v3 instance (e.g., `"http://localhost:8086"` for OSS, or `"https://<your-region>.cloud2.influxdata.com"` for Cloud).
-   `token` (str): Your InfluxDB API token. This token must have write permissions to the specified database.
-   `database` (str): The name of the database (bucket in InfluxDB v3 terminology) where data will be written.
-   `org` (str, optional): Your InfluxDB organization ID or name. This is typically required for InfluxDB Cloud and may be needed for InfluxDB OSS depending on your setup.

**Available Callback Classes:**

A suite of callback classes is provided for different data types:
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

Each class automatically sets a `default_key` (e.g., 'trades', 'book') which is used to construct the measurement name in InfluxDB (e.g., `trades-COINBASE`, `book-BINANCE`).

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
        org=INFLUXDB_ORG
    )

    book_cb = BookInflux3(
        addr=INFLUXDB_ADDRESS,
        database=INFLUXDB_DATABASE,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG
        # For BookInflux3, you can also pass:
        # snapshots_only=False, snapshot_interval=1000
    )

    # Add subscriptions
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[TRADES], callbacks=[trade_cb])
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[L2_BOOK], callbacks=[book_cb])

    fh.run()

if __name__ == '__main__':
    main()
```

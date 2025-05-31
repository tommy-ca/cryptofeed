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
* Iceberg

There are also a handful of wrappers defined [here](../cryptofeed/backends/aggregate.py) that can be used in conjunction with these and raw callbacks to convert data to OHLCV, throttle data, etc.

### Apache Iceberg Backend

The Apache Iceberg backend allows you to store cryptofeed data into Iceberg tables, enabling robust data lakehouse capabilities. It leverages `pyiceberg` to interact with Iceberg catalogs (supporting local filesystem, S3, GCS, and others) and `pyarrow` for data serialization.

**Installation**

To use the Iceberg backend, you need to install the required dependencies:
```bash
pip install pyiceberg>=0.6.0 pyarrow>=10.0.0
```
Depending on your chosen catalog (S3, GCS), you might also need additional libraries like `boto3` (for S3) or `google-cloud-storage` (for GCS). `pyiceberg` often lists these as extras, e.g., `pip install pyiceberg[s3]` or `pyiceberg[gcs]`.

**Configuration**

The primary class for this backend is `cryptofeed.backends.iceberg.IcebergCallback`. Specific data types have their own callback classes inheriting from it (e.g., `TickerIceberg`, `TradeIceberg`).

Key configuration parameters for `IcebergCallback` and its children:

*   `path` (str): **Required**. The URI for the Iceberg catalog. This determines the type of catalog and its location.
    *   For a local filesystem catalog: `/path/to/your/iceberg_warehouse` or `relative/path/warehouse`. The last component of this path (e.g., `iceberg_warehouse`) is used as the catalog identifier when constructing table names.
    *   For S3: `s3://your-s3-bucket/path/to/warehouse/`
    *   For GCS: `gcs://your-gcs-bucket/path/to/warehouse/`
*   `key` (str): Optional. The data type being stored (e.g., `TICKER`, `TRADES`). This defaults to the `default_key` of the specific callback class (like `TickerIceberg.default_key` is `TICKER`). This key is used as the base name for the Iceberg table. For example, if the catalog identifier derived from `path` is `my_catalog` and `key` is `ticker`, the table will be `my_catalog.ticker`.
*   `gcs_project_id` (str): Optional. Required if using a GCS catalog to specify the Google Cloud Project ID.
*   `**kwargs`: Additional keyword arguments are passed directly to `pyiceberg.catalog.load_catalog()`. This is how you provide credentials and other configurations for S3, GCS, or other catalog types.
    *   **For S3:**
        *   `s3.access-key-id`: Your S3 access key ID.
        *   `s3.secret-access-key`: Your S3 secret access key.
        *   `s3.region`: The AWS region for the S3 bucket (e.g., `us-west-2`).
        *   `s3.endpoint-override`: The S3 endpoint URL, necessary for S3-compatible storage like MinIO (e.g., `http://localhost:9000`).
        *   And other S3 properties supported by PyIceberg's FsspecFileIO.
    *   **For GCS:**
        *   `gcs.token`: GCS token, if not relying on default credentials.
        *   And other GCS properties.
    *   Refer to the [PyIceberg documentation](https://pyiceberg.apache.org/configuration/) for all available catalog configuration options.

**Table Naming**

Iceberg tables are created with names in the format: `catalog_identifier.table_base_name`.
*   `catalog_identifier`: Derived from the last component of the `path` argument. For example, if `path` is `/opt/data/my_iceberg_catalog`, the identifier is `my_iceberg_catalog`.
*   `table_base_name`: This is taken from the `key` argument (e.g., `ticker`, `trades`).

**Example Usage**

```python
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Coinbase
from cryptofeed.defines import TICKER, TRADES
from cryptofeed.backends.iceberg import TickerIceberg, TradeIceberg

def main():
    f = FeedHandler()

    # Configure for a local Iceberg catalog in the 'iceberg_warehouse' directory
    # The catalog identifier will be 'iceberg_warehouse'
    # Ticker data will go to 'iceberg_warehouse.ticker'
    # Trade data will go to 'iceberg_warehouse.trades'
    iceberg_config_local = {
        'path': 'iceberg_warehouse'
    }

    # Example for S3 (ensure S3 bucket 'my-crypto-data' and prefix 'iceberg_catalog/' exist)
    # iceberg_config_s3 = {
    #     'path': 's3://my-crypto-data/iceberg_catalog/',
    #     's3.access-key-id': 'YOUR_AWS_ACCESS_KEY_ID',
    #     's3.secret-access-key': 'YOUR_AWS_SECRET_ACCESS_KEY',
    #     's3.region': 'us-east-1'
    # }

    f.add_feed(Coinbase(channels=[TICKER], symbols=['BTC-USD'], callbacks={TICKER: TickerIceberg(**iceberg_config_local)}))
    f.add_feed(Coinbase(channels=[TRADES], symbols=['ETH-USD'], callbacks={TRADES: TradeIceberg(**iceberg_config_local)}))

    # To use the S3 example:
    # f.add_feed(Coinbase(channels=[TICKER], symbols=['BTC-USD'], callbacks={TICKER: TickerIceberg(**iceberg_config_s3)}))

    f.run()

if __name__ == '__main__':
    main()
```

**Available Callback Classes**

The following specific callback classes are available in `cryptofeed.backends.iceberg`:

*   `TickerIceberg`
*   `TradeIceberg`
*   `FundingIceberg`
*   `OpenInterestIceberg`
*   `LiquidationsIceberg`
*   `CandlesIceberg`
*   `OrderInfoIceberg`
*   `TransactionsIceberg`
*   `BalancesIceberg`
*   `FillsIceberg`

Each class defaults to the appropriate `key` for its data type.

### Performance Considerations

Do not do anything computationally intensive in your callbacks, or this will greatly impact the performance of cryptofeed. Data should be quickly processed and passed along to another process/application/etc or a backend callback should be used to forward the data elsewhere. If possible, use async libraries in your callbacks!

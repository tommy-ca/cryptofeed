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
* Iceberg

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

### Iceberg Backend

The Iceberg backend allows publishing data from Cryptofeed to Apache Iceberg tables. It buffers data in memory and writes it in batches. Tables are automatically created with predefined schemas if they do not exist.

**Installation**

To use the Iceberg backend, you need to install the necessary extra dependencies:

```bash
pip install cryptofeed[iceberg]
```

This will install `pyiceberg`, `pyarrow`, and `pandas`. Depending on your Iceberg catalog and storage, you might need additional packages (e.g., `boto3` for S3, `psycopg2-binary` for a PostgreSQL-backed catalog). Refer to the PyIceberg documentation for catalog-specific requirements.

**Configuration**

The Iceberg backend is configured within the `callbacks` section of your Cryptofeed configuration.

Example using a Python dictionary:

```python
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.backends.iceberg import TradeIceberg, BookIceberg

# Example for a REST catalog
rest_catalog_config = {
    "name": "my_rest_catalog", # Optional: local name for the catalog instance
    "uri": "http://localhost:8181", # REST catalog URI
    "s3.endpoint-url": "http://minio:9000", # Example if warehouse is S3 via MinIO
    "s3.access-key-id": "YOUR_ACCESS_KEY",
    "s3.secret-access-key": "YOUR_SECRET_KEY",
    "warehouse": "s3a://my-bucket/iceberg_warehouse/"
}

# Example for a Hive catalog
hive_catalog_config = {
    "name": "my_hive_catalog",
    "uri": "thrift://localhost:9083", # Hive Metastore URI
    "warehouse": "s3a://my-bucket/iceberg_warehouse/" # Example S3 warehouse path
    # Add s3.endpoint-url, keys, etc. if using S3 with Hive
}


config = {
    'log': {
        'filename': 'feedhandler.log',
        'level': 'INFO'
    },
    'callbacks': {
        TRADES: TradeIceberg(
            catalog_config=rest_catalog_config,
            database_name='crypto_data',
            table_prefix='cf',
            batch_size=500
        ),
        L2_BOOK: BookIceberg(
            catalog_config=rest_catalog_config,
            database_name='crypto_data',
            table_prefix='cf_book',
            batch_size=200,
            # pandas_kwargs={'columns': ['custom_col_order']} # Optional
        )
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
    class: TradeIceberg
    catalog_config:
      name: "my_rest_catalog" # Optional: local name for the catalog instance
      uri: "http://localhost:8181" # REST catalog URI
      s3.endpoint-url: "http://minio:9000" # Example for S3 via MinIO
      s3.access-key-id: "YOUR_ACCESS_KEY"
      s3.secret-access-key: "YOUR_SECRET_KEY"
      warehouse: "s3a://my-bucket/iceberg_warehouse/"
    database_name: crypto_data
    table_prefix: cf             # Table will be cf_trades
    batch_size: 500
  L2_BOOK:
    class: BookIceberg
    catalog_config: # Can reuse catalog_config or define another
      name: "my_rest_catalog"
      uri: "http://localhost:8181"
      # ... other catalog properties
    database_name: crypto_data
    table_prefix: cf_book        # Table will be cf_book_orderbooks
    batch_size: 200
    # pandas_kwargs:
    #   columns: ['exchange', 'symbol', 'timestamp', ...] # To enforce column order/selection
```

**Configuration Parameters:**

*   `class`: The specific Iceberg callback class (e.g., `TradeIceberg`, `BookIceberg`).
*   `catalog_config`: (Required) A dictionary containing properties to initialize the PyIceberg catalog (e.g., `uri`, `warehouse`, S3 credentials, etc.). The specific keys and values depend heavily on your chosen Iceberg catalog type (REST, Hive, Nessie, SQL). Consult the PyIceberg documentation for `pyiceberg.catalog.load_catalog()` and your catalog's specific configuration.
*   `database_name`: (Optional) The Iceberg namespace (database) where tables will be managed. Defaults to `'default'`. The backend will attempt to create this namespace if it doesn't exist.
*   `table_prefix`: (Optional) A prefix for table names. The full table name is formed as `<table_prefix>_<data_type_key>` (e.g., `cryptofeed_trades`). Defaults to `'cryptofeed'`.
*   `batch_size`: (Optional) The number of records to buffer in memory before writing to an Iceberg table. Defaults to `1000`.
*   `pandas_kwargs`: (Optional) A dictionary of keyword arguments passed to `pandas.DataFrame.from_records()` when creating DataFrames from buffered data. This can be used to control aspects like column selection or indexing. By default, subclasses set `columns` based on their predefined schema.

**Table Management and Schemas**

Tables are automatically created by the backend if they do not already exist in the specified database. Each data-type specific callback (like `TradeIceberg`) has a predefined `pyarrow.Schema` that dictates the table structure.

**Book Data (`BookIceberg`)**

The `BookIceberg` callback stores order book snapshots. Bids and asks are stored in a structured format within the table, specifically as a `list` of `structs`, where each struct contains `price` and `size` fields (both floats). This allows for querying individual price levels. The schema also includes a `delta` boolean field to distinguish full snapshots from records originating from delta updates (though all records written by `BookIceberg` represent the state of the book or changes at a point in time).

**Available Iceberg Callback Classes:**

The following Iceberg-specific callback classes are available in `cryptofeed.backends.iceberg`:

*   `TradeIceberg`
*   `TickerIceberg`
*   `BookIceberg`
*   `FundingIceberg`
*   `OpenInterestIceberg`
*   `LiquidationsIceberg`
*   `CandlesIceberg`
*   `OrderInfoIceberg`
*   `TransactionsIceberg`
*   `BalancesIceberg`
*   `FillsIceberg`

These classes manage the buffering, schema definition, and writing of their respective data types to Iceberg tables.

### Performance Considerations

Do not do anything computationally intensive in your callbacks, or this will greatly impact the performance of cryptofeed. Data should be quickly processed and passed along to another process/application/etc or a backend callback should be used to forward the data elsewhere. If possible, use async libraries in your callbacks!

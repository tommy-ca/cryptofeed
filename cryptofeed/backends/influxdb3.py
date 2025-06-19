import logging
import asyncio
import json # Added for book data serialization
from influxdb_client_3 import InfluxDBClient3, Point, WriteOptions, write_client_options, WritePrecision, InfluxDBError

from cryptofeed.backends.backend import BackendQueue, BackendCallback, BackendBookCallback

LOG = logging.getLogger('feedhandler')


class InfluxDB3Callback(BackendQueue):
    def __init__(self, addr: str, database: str, token: str, org: str = None,
                 key: str = 'data', batch_size: int = 5000, flush_interval: int = 10000, **kwargs):
        """
        Parameters
        ----------
        addr: str
            Address for InfluxDBv3. e.g., 'http://localhost:8086' or InfluxDB Cloud URL.
        database: str
            Database name (bucket) to write to.
        token: str
            API token for InfluxDB.
        org: str, optional
            Organization ID or name. Required for InfluxDB Cloud, optional for OSS.
        key: str, optional
            Default key (measurement prefix) for data. Defaults to 'data'.
            Subclasses for specific data types (trades, book, etc.) will typically override this.
        batch_size: int, optional
            Number of data points to batch before writing. Defaults to 5000.
        flush_interval: int, optional
            Time in milliseconds to wait before flushing the batch. Defaults to 10000.
        kwargs:
            numeric_type (type, default=float): Type for numeric values.
            none_to (any, default=None): Value to convert None to.
        """
        super().__init__(**kwargs) # Pass relevant kwargs to BackendQueue if any
        self.key = key
        self.numeric_type = kwargs.get('numeric_type', float)
        self.none_to = kwargs.get('none_to', None)
        self.running = True # Explicitly set running state

        # Define WriteOptions for the InfluxDB client
        write_opts = WriteOptions(
            batch_size=batch_size,
            flush_interval=flush_interval,  # in ms
            jitter_interval=2000,           # in ms
            retry_interval=5000,            # in ms
            max_retries=3,
            max_retry_delay=30000,          # in ms
            exponential_base=2
        )

        # Define error and retry logging callbacks for write_client_options
        # These must be synchronous functions.
        def log_error_sync(conf, data_str, exception):
            LOG.error(f"InfluxDB3 async write error. Conf: {conf}, Data: {data_str}, Exc: {exception}")

        def log_retry_sync(conf, data_str, exception):
            LOG.warning(f"InfluxDB3 async write retry. Conf: {conf}, Data: {data_str}, Exc: {exception}")

        wco = write_client_options(
            success_callback=None,  # Or a simple log if needed: lambda conf, data_str: LOG.debug("Write success")
            error_callback=log_error_sync,
            retry_callback=log_retry_sync,
            write_options=write_opts
        )

        # Initialize InfluxDBClient3
        # Ensure addr is just the host part if the client expects that.
        # The InfluxDBClient3 host parameter is typically "http(s)://host:port"
        try:
            self.client = InfluxDBClient3(
                host=addr,
                database=database,
                token=token,
                org=org,
                write_client_options=wco,
                write_precision=WritePrecision.NS # Nanosecond precision is standard for financial data
                # Add other relevant options like timeout if available in constructor
            )
            LOG.info("%s: InfluxDB3 client initialized for %s, org %s, database %s", self.id, addr, org, database)
        except Exception as e:
            LOG.error("%s: Failed to initialize InfluxDB3 client: %s", self.id, e, exc_info=True)
            # Potentially raise an error or set a state indicating failure
            self.client = None # Ensure client is None if init fails
            raise  # Re-raise the exception to signal failure to the caller

    def _data_to_point(self, data_obj, receipt_timestamp: float) -> Point:
        """
        Converts a Cryptofeed data object to an InfluxDB Point.

        Parameters
        ----------
        data_obj: object
            The Cryptofeed data object (e.g., Trade, Book, Ticker).
            Can be a dataclass instance or a dictionary.
        receipt_timestamp: float
            The timestamp when the data was received by Cryptofeed.

        Returns
        -------
        Point or None
            An InfluxDB Point object, or None if the data cannot be processed.
        """
        # Determine exchange and symbol
        if hasattr(data_obj, 'exchange'):
            exchange = data_obj.exchange
        elif isinstance(data_obj, dict) and 'exchange' in data_obj:
            exchange = data_obj['exchange']
        elif hasattr(data_obj, 'feed'): # Some objects use 'feed'
            exchange = data_obj.feed
        elif isinstance(data_obj, dict) and 'feed' in data_obj:
            exchange = data_obj['feed']
        else:
            LOG.warning("%s: 'exchange' or 'feed' not found in data object. Using 'unknown_exchange'. Data: %s", self.id, data_obj)
            exchange = "unknown_exchange"

        if hasattr(data_obj, 'symbol'):
            symbol = data_obj.symbol
        elif isinstance(data_obj, dict) and 'symbol' in data_obj:
            symbol = data_obj['symbol']
        else:
            LOG.warning("%s: 'symbol' not found in data object for exchange %s. Using 'unknown_symbol'. Data: %s", self.id, exchange, data_obj)
            symbol = "unknown_symbol"

        measurement_name = f"{self.key}-{exchange}"
        point = Point(measurement_name)
        point.tag("symbol", symbol)

        # Determine timestamp (prefer object's timestamp, fallback to receipt_timestamp)
        ts_to_use = getattr(data_obj, 'timestamp', None)
        if ts_to_use is None and isinstance(data_obj, dict): # Check dict if not on attr
             ts_to_use = data_obj.get('timestamp')
        if ts_to_use is None:
            ts_to_use = receipt_timestamp

        try:
            timestamp_ns = int(float(ts_to_use) * 1_000_000_000)
            point.time(timestamp_ns, WritePrecision.NS)
        except (ValueError, TypeError) as e:
            LOG.error("%s: Invalid timestamp value '%s': %s. Point will use server time.", self.id, ts_to_use, e)
            # Point will use server timestamp if .time() is not called or fails

        # Convert data object to dictionary for field processing
        if hasattr(data_obj, 'to_dict'):
            # Assuming to_dict handles numeric_type and none_to internally if needed,
            # or we rely on influxdb-client-python's type handling.
            # For now, let's get a plain dict.
            data_dict = data_obj.to_dict()
        elif isinstance(data_obj, dict):
            data_dict = data_obj
        else: # Fallback to __dict__ for other objects
            data_dict = data_obj.__dict__

        has_fields = False
        for field_key, value in data_dict.items():
            if field_key in ('exchange', 'feed', 'symbol', 'timestamp', 'receipt_timestamp', 'raw'):
                continue

            if value is None and self.none_to is None: # Skip None if not converting
                continue
            elif value is None and self.none_to is not None:
                value = self.none_to

            # Special handling for book data
            if self.key == 'book' and field_key in ('bids', 'asks'):
                import json # Ensure json is imported
                try:
                    point.field(field_key, json.dumps(value))
                    has_fields = True
                except (TypeError, OverflowError) as e: # json.dumps can fail
                    LOG.error("%s: Could not JSON serialize book data for field '%s': %s. Data: %s", self.id, field_key, e, value)
                continue
            elif self.key == 'book' and field_key == 'delta' and value is not None: # Delta could be a dict/list
                 point.field(field_key, str(value)) # Convert delta to string
                 has_fields = True
                 continue

            # General field handling
            if isinstance(value, (str, float, int, bool)):
                point.field(field_key, value)
                has_fields = True
            elif hasattr(value, 'quantize'): # Handles Decimal objects
                point.field(field_key, float(value)) # Convert Decimal to float
                has_fields = True
            else: # Attempt to convert other types to string
                try:
                    point.field(field_key, str(value))
                    has_fields = True
                    LOG.debug("%s: Converted field '%s' to string for InfluxDB: %s", self.id, field_key, value)
                except Exception as e:
                    LOG.warning("%s: Could not convert value for field '%s' to a suitable InfluxDB type: %s. Value: %s, Type: %s", self.id, field_key, e, value, type(value))

        if not has_fields:
            LOG.debug("%s: No fields generated for point from data: %s. Skipping point.", self.id, data_dict)
            return None

        return point

    async def writer(self):
        """
        Background task that reads from the internal queue and writes data to InfluxDB.
        This method is started by BackendQueue.
        """
        loop = asyncio.get_event_loop()
        LOG.info("%s: InfluxDB3 writer task started.", self.id)

        while self.running:
            try:
                # BackendQueue.read_queue() is an async context manager yielding a list of items
                async with self.read_queue() as updates:
                    if not updates: # Can happen if queue is empty and timeout occurs (if configured)
                        if not self.running: # If stopping, exit.
                            break
                        continue

                    points_to_write = []
                    for item in updates: # item is QueueItem(data=data_object, timestamp=receipt_timestamp)
                        if item is None: # Sentinel from close() or stop()
                            LOG.debug("%s: Writer received sentinel, checking running state: %s.", self.id, self.running)
                            if not self.running:
                                break # Exit outer while loop
                            continue # Continue inner for loop (though usually sentinel is alone)

                        point = self._data_to_point(item.data, item.timestamp)
                        if point:
                            points_to_write.append(point)

                    if not self.running and not points_to_write: # If stopping and no more points, exit.
                        break

                    if points_to_write:
                        try:
                            # self.client.write is a synchronous method from influxdb_client_3.
                            # Run it in an executor to avoid blocking the asyncio event loop.
                            await loop.run_in_executor(None, self.client.write, points_to_write)
                            LOG.debug("%s: Successfully wrote %d points to InfluxDB.", self.id, len(points_to_write))
                        except InfluxDBError as ide: # Specific error from the InfluxDB client
                            LOG.error("%s: InfluxDBError writing %d points to InfluxDB: %s", self.id, len(points_to_write), ide, exc_info=True)
                            # Error callback in WriteClientOptions should also log this.
                        except Exception as e: # Catch other unexpected errors during write
                            LOG.error("%s: Unexpected error writing %d points to InfluxDB: %s", self.id, len(points_to_write), e, exc_info=True)

                    if not self.running and not updates: # Final check if already stopped and queue was flushed
                        break

            except asyncio.CancelledError:
                LOG.info("%s: InfluxDB3 writer task cancelled.", self.id)
                break # Exit loop on cancellation
            except Exception as e: # Catch errors from read_queue or other logic
                LOG.error("%s: Error in InfluxDB3 writer loop: %s", self.id, e, exc_info=True)
                if not self.running:
                    break
                await asyncio.sleep(1) # Avoid tight loop on persistent error from queue

        LOG.info("%s: InfluxDB3 writer task finished.", self.id)


    async def close(self):
        """
        Closes the InfluxDB3 client and stops the callback.
        """
        LOG.info("%s: Closing InfluxDB3Callback.", self.id)
        self.running = False # Signal any running loops to stop

        # BackendQueue's stop method might handle unblocking the queue and worker.
        # For BackendQueue, stopping is usually done by setting self.running = False.
        # The writer loop will then exit after processing remaining queue items.
        # We explicitly signal the queue to unblock if it's waiting.
        if hasattr(self, 'queue') and self.queue is not None:
             await self.queue.put(None) # Sentinel to unblock the writer task

        if self.client:
            LOG.info("%s: Closing InfluxDB3 client.", self.id)
            try:
                # The client's close method is synchronous.
                # Run it in an executor to avoid blocking the asyncio event loop.
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.client.close)
                LOG.info("%s: InfluxDB3 client closed successfully.", self.id)
            except Exception as e:
                LOG.error("%s: Error closing InfluxDB3 client: %s", self.id, e, exc_info=True)
            finally:
                self.client = None # Ensure client is marked as closed

# Example usage (for testing purposes, will be removed or moved)
# if __name__ == '__main__':


# Specific data type callback classes

class TradeInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'trades'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class FundingInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'funding'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class BookInflux3(InfluxDB3Callback, BackendBookCallback):
    default_key = 'book'

    def __init__(self, addr: str, database: str, token: str, org: str = None, snapshots_only=False, snapshot_interval=1000, **kwargs):
        # Call InfluxDB3Callback's __init__
        # It now accepts batch_size, flush_interval from kwargs if provided.
        InfluxDB3Callback.__init__(
            self,
            addr=addr,
            database=database,
            token=token,
            org=org,
            key=self.default_key,
            **kwargs # Pass all other kwargs, including potential batch_size, flush_interval
        )

        # Call BackendBookCallback's __init__
        BackendBookCallback.__init__(
            self,
            # default_key parameter is not strictly needed here if we rely on BookInflux3.default_key
            # for InfluxDB3Callback's key. BackendBookCallback's own default_key will be 'book'.
            snapshots_only=snapshots_only,
            snapshot_interval=snapshot_interval,
            **kwargs # Pass kwargs here too for future-proofing or other base class needs
        )

class TickerInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'ticker'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class OpenInterestInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'open_interest'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class LiquidationsInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'liquidations'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class CandlesInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'candles'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class OrderInfoInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'order_info'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class TransactionsInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'transactions'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class BalancesInflux3(InfluxDB3Callback, BackendCallback):
    default_key = 'balances'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

class FillsInflux3(InfluxDB3Callback, BackendCallback): # Assuming Fills is like Trades, uses BackendCallback
    default_key = 'fills'

    def __init__(self, addr: str, database: str, token: str, org: str = None, **kwargs):
        super().__init__(addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)

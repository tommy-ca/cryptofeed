import logging
import asyncio # Added for asyncio.sleep, asyncio.CancelledError
import aiohttp # Added for aiohttp.ClientSession

from cryptofeed.backends.http import HTTPCallback
from cryptofeed.backends.backend import BackendCallback, BackendBookCallback

LOG = logging.getLogger('feedhandler')


class InfluxDB3Callback(HTTPCallback):
    def _escape_key(self, key: str) -> str:
        """Escape keys (measurement, tag keys, field keys) for InfluxDB line protocol."""
        return str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    def _escape_string_value(self, value: str) -> str:
        """Escape string field values for InfluxDB line protocol."""
        return str(value).replace("\\", "\\\\").replace("\"", "\\\"")

    def __init__(self, addr: str, database: str, token: str, org: str = None, key: str = 'data', **kwargs):
        """
        Parameters
        ----------
        addr: str
            Address for InfluxDBv3. e.g., 'http://localhost:8086'
        database: str
            Database name to write to.
        token: str
            API token for InfluxDB.
        org: str, optional
            Organization ID. Required for InfluxDB Cloud, optional for OSS.
        key: str, optional
            Default key (measurement prefix) for data. Defaults to 'data'.
            Subclasses for specific data types (trades, book, etc.) will typically override this.
        """
        # Construct the write URL first
        # For InfluxDB v3, the /api/v2/write endpoint is used for line protocol writes.
        # The 'bucket' parameter in the query string is used for the database name.
        # Precision is typically 'ns' for nanoseconds.
        url_parts = [addr, "/api/v2/write?precision=ns"]
        if org:
            url_parts.append(f"&org={org}")
        url_parts.append(f"&bucket={database}")
        write_url = "".join(url_parts)

        super().__init__(write_url, key=key, **kwargs) # Pass key to parent
        self.write_url = write_url # Overrides addr from parent if it was just the base
        self.database = database
        self.token = token
        self.org = org
        self.key = key # Store the key for use in format method

        self.headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "text/plain",
            # **self.headers - HTTPCallback might not have self.headers initialized when super().__init__ is called.
            # It's safer to assume HTTPCallback.headers is the base and we override/add to it.
            # HTTPCallback.__init__ does self.headers = kwargs.get('headers', {})
        }
        if 'headers' in kwargs: # Merge any headers passed in kwargs
            self.headers.update(kwargs['headers'])

        # Session management is now done by this class's loop, not HTTPCallback._worker
        # HTTPCallback.__init__ sets self.session = None.
        # self.running is set by Callback.__init__

    async def http_connect(self):
        """
        Override HTTPCallback.http_connect to prevent starting the _read_worker_task,
        as InfluxDB3Callback will manage its own write loop and session.
        The session is created here if needed using aiohttp.ClientSession.
        """
        if getattr(self, 'session', None) is None or self.session.closed: # Check if self.session even exists yet
            LOG.debug("%s: HTTP session %s, creating new session", self.id, 'closed' if getattr(self, 'session', None) else 'is None')
            # Ensure self.headers is fully initialized here
            if not hasattr(self, 'headers'): # Should be set in __init__
                self.headers = {
                    "Authorization": f"Token {self.token}",
                    "Content-Type": "text/plain"
                }
                LOG.warning("%s: self.headers was not initialized before http_connect. Using default.", self.id)

            self.session = aiohttp.ClientSession(headers=self.headers)
            LOG.debug("%s: New HTTP session created.", self.id)
        # Explicitly DO NOT START self._read_worker_task (i.e., HTTPCallback._worker)

    async def run(self):
        """
        Main execution loop for the callback.
        Overrides Callback.run() to use our custom writer_loop.
        """
        LOG.info("%s: Starting InfluxDB3Callback run loop.", self.id)
        # self.http_connect() will be called by writer_loop as needed.
        await self.writer_loop()

    async def close(self):
        """
        Closes the callback and its resources.
        Signal the writer_loop to stop and ensure the session is closed.
        """
        LOG.info("%s: Closing InfluxDB3Callback initiated.", self.id)
        self.running = False # Signal writer_loop to stop

        # Unblock the read_queue if writer_loop is waiting on it.
        # Base Callback.stop() or a direct call to self.queue_add_sentinel() can do this.
        # This should ideally be handled by how Callback.stop() works with self.read_queue().
        # If self.writer_loop_task exists, can await/cancel it.
        # For now, relying on self.running = False and read_queue unblocking.

        # The session will be closed by the finally block in writer_loop.
        # If writer_loop is not running or stuck, we might need a direct close here.
        # Add a failsafe for session closing if writer_loop didn't handle it.
        if self.session and not self.session.closed:
            LOG.warning("%s: Session still open during close(). Forcing closure.", self.id)
            await self.session.close()
            self.session = None
        LOG.info("%s: InfluxDB3Callback close process finished.", self.id)


    async def format(self, data: dict, timestamp: float) -> str:
        """
        Format data into InfluxDB line protocol.

        Parameters
        ----------
        data: dict
            The data dictionary to format. Expected to contain keys like 'exchange', 'symbol',
            and other fields to be written to InfluxDB.
        timestamp: float
            The primary timestamp for the data point (e.g., data.timestamp from cryptofeed objects).
            This will be converted to nanoseconds. If None, receipt_timestamp from `data` dict is used.

        Returns
        -------
        str
            A string formatted in InfluxDB line protocol.
        """
        tags = []
        fields = []

        # Determine measurement name
        exchange = data.get('exchange', data.get('feed'))
        if not self.key:
            LOG.warning("InfluxDB3Callback: self.key is not set, measurement name might be incomplete.")
            measurement_key = "data" # Default if no key
        else:
            measurement_key = self.key

        if exchange:
            measurement = self._escape_key(f"{measurement_key}-{exchange}")
        else:
            measurement = self._escape_key(measurement_key)
            LOG.debug("InfluxDB3Callback: 'exchange' or 'feed' not found in data for measurement construction.")

        # Process tags
        # Symbol is a common tag.
        if 'symbol' in data:
            tags.append(f"symbol={self._escape_key(data['symbol'])}")

        # Process fields
        for k, v in data.items():
            # Skip keys already used or not suitable for fields
            if k in ('exchange', 'feed', 'symbol', 'timestamp', 'receipt_timestamp', 'raw'):
                continue
            if v is None: # Don't write None values
                continue

            escaped_k = self._escape_key(k)
            if isinstance(v, bool):
                fields.append(f"{escaped_k}={'true' if v else 'false'}")
            elif isinstance(v, int):
                fields.append(f"{escaped_k}={v}i")
            elif isinstance(v, float):
                # Ensure no trailing zeros that might be lost in string conversion for some locales
                fields.append(f"{escaped_k}={v}")
            elif isinstance(v, str):
                fields.append(f"{escaped_k}=\"{self._escape_string_value(v)}\"")
            else:
                # For other types, attempt to convert to string and quote
                fields.append(f"{escaped_k}=\"{self._escape_string_value(str(v))}\"")

        if not fields:
            # InfluxDB requires at least one field
            LOG.warning(f"InfluxDB3Callback: No fields generated for data: {data}. Skipping line.")
            return ""

        # Determine timestamp
        final_timestamp_ns_str = ""
        # Prioritize the timestamp argument, then data['timestamp'], then data['receipt_timestamp']
        ts_to_convert = None
        if timestamp is not None:
            ts_to_convert = timestamp
        elif data.get('timestamp') is not None:
            ts_to_convert = data['timestamp']
        elif data.get('receipt_timestamp') is not None:
            ts_to_convert = data['receipt_timestamp']

        if ts_to_convert is not None:
            try:
                # Convert to nanoseconds
                final_timestamp_ns_str = str(int(float(ts_to_convert) * 1_000_000_000))
            except ValueError:
                LOG.warning(f"InfluxDB3Callback: Could not convert timestamp '{ts_to_convert}' to float.")

        # Assemble line protocol string
        # measurement[,tag_set] field_set [timestamp]
        line = measurement
        if tags:
            line += f",{','.join(tags)}"
        line += f" {','.join(fields)}"
        if final_timestamp_ns_str:
            line += f" {final_timestamp_ns_str}"

        return line

    async def writer_loop(self):
        """
        Continuously reads from the main data queue, formats, and writes to InfluxDB.
        This acts as the primary processing loop for this callback.
        It manages its own HTTP session.
        """
        LOG.info("%s: InfluxDB3 writer loop starting.", self.id)
        # Ensure session is ready, or create it. Uses our overridden http_connect.
        # self.http_write (from HTTPCallback) will call self.http_connect if session is invalid.
        # So, an explicit call to self.http_connect() here is good for clarity but not strictly necessary
        # if http_write is robust enough. Let's ensure it is.
        await self.http_connect()


        try:
            while self.running:
                try:
                    # read_queue() is from the base Callback class (AsyncTimedQueue)
                    # It yields QueueItem(data=data_object, timestamp=receipt_timestamp)
                    item = await self.read_queue()

                    if item is None: # Sentinel value indicating queue is stopping or flushed by stop().
                        LOG.debug("%s: Received None from read_queue. Running state: %s", self.id, self.running)
                        if not self.running: # If stop() was called, self.running is False
                            break # Exit loop gracefully
                        # If self.running is True, None might be from a manual flush not related to stopping.
                        # Or, queue timeout in AsyncTimedQueue with ret_none=True.
                        # For now, assume None means stop or nothing to do.
                        continue

                    update_obj = item.data      # The actual data (e.g., Trade, L2Book objects)
                    receipt_ts = item.timestamp # Timestamp of receipt by cryptofeed

                    # Determine the primary timestamp for InfluxDB (usually event timestamp)
                    # getattr is used as 'timestamp' may not exist on all data objects.
                    primary_event_timestamp = getattr(update_obj, 'timestamp', None)

                    # Prepare data_dict for format()
                    if isinstance(update_obj, dict):
                        data_dict_to_format = update_obj
                    else:
                        # Convert dataclass/object to dict. Use a copy to avoid modifying original object's dict.
                        data_dict_to_format = dict(update_obj.__dict__)

                    # Ensure receipt_timestamp is available in the dict if format() needs it as a fallback.
                    # The format() method prioritizes its 'timestamp' argument (primary_event_timestamp),
                    # then 'timestamp' in dict, then 'receipt_timestamp' in dict.
                    if 'receipt_timestamp' not in data_dict_to_format:
                         data_dict_to_format['receipt_timestamp'] = receipt_ts
                    if 'timestamp' not in data_dict_to_format and primary_event_timestamp is not None:
                        # Ensure primary_event_timestamp is also in dict if format checks dict first for 'timestamp'
                        data_dict_to_format['timestamp'] = primary_event_timestamp

                    line_protocol_string = await self.format(data_dict_to_format, primary_event_timestamp)

                    if line_protocol_string:
                        # self.http_write is from HTTPCallback. It uses self.session (created by our http_connect),
                        # self.headers (set in our __init__), and includes retry logic.
                        await self.http_write(line_protocol_string)

                    self.read_queue_task_done() # Notify queue that item processing is complete

                except asyncio.CancelledError:
                    LOG.info("%s: Writer loop task was cancelled.", self.id)
                    self.running = False # Ensure outer loop terminates
                    raise # Propagate cancellation
                except Exception as e:
                    LOG.error("%s: Error in InfluxDB3 writer_loop's inner processing: %s", self.id, e, exc_info=True)
                    if not self.running: # If error occurred during shutdown
                        LOG.info("%s: Exiting writer_loop due to error during shutdown.", self.id)
                        break
                    await asyncio.sleep(1) # Avoid tight loop on persistent errors

        except asyncio.CancelledError:
            LOG.info("%s: Writer_loop was cancelled externally (e.g. during shutdown).", self.id)
        finally:
            LOG.info("%s: InfluxDB3 writer_loop finishing. Closing HTTP session.", self.id)
            if self.session and not self.session.closed:
                await self.session.close()
                LOG.debug("%s: HTTP session closed by writer_loop.", self.id)
            self.session = None # Mark as closed and None
            LOG.info("%s: InfluxDB3 writer_loop fully stopped.", self.id)

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
        # Explicitly call __init__ for both parent classes to ensure complete setup.
        # InfluxDB3Callback for its main logic, HTTP setup, and writer loop.
        InfluxDB3Callback.__init__(self, addr=addr, database=database, token=token, org=org, key=self.default_key, **kwargs)
        # BackendBookCallback for attributes like snapshots_only, snapshot_interval.
        BackendBookCallback.__init__(self, default_key=self.default_key, snapshots_only=snapshots_only, snapshot_interval=snapshot_interval)

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
#     class TradeInfluxDB3(InfluxDB3Callback):
#         def __init__(self, addr, database, token, org=None, **kwargs):
#             super().__init__(addr, database, token, org, **kwargs)
#             self.key = "trades" # Example key for trades

#         async def format(self, data: dict, timestamp: float) -> str:
#             # Simplified example for a trade
#             # {
#             #     'symbol': 'BTC-USD', 'side': 'buy', 'amount': 0.1, 'price': 50000,
#             #     'id': '12345', 'feed': 'COINBASE', 'timestamp': 1620000000.0
#             # }
#             tags = f"symbol={data['symbol']},side={data['side']}"
#             fields = f"price={data['price']},amount={data['amount']},id=\"{data['id']}\""
#             # InfluxDB expects timestamp in nanoseconds
#             ts_ns = int(timestamp * 1_000_000_000)
#             return f"{self.key},{tags} {fields} {ts_ns}"

#     async def main():
#         # Configuration (replace with your actual InfluxDB details)
#         influx_addr = "http://localhost:8086" # Your InfluxDB address
#         influx_database = "cryptofeed"       # Your database/bucket
#         influx_token = "your_influx_token"   # Your InfluxDB token
#         influx_org = "your_influx_org"       # Your InfluxDB organization (optional for OSS)

#         # Create an instance of TradeInfluxDB3
#         callback = TradeInfluxDB3(addr=influx_addr, database=influx_database, token=influx_token, org=influx_org)

#         # Example trade data
#         trade_data = {
#             'symbol': 'BTC-USD', 'side': 'buy', 'amount': 0.1, 'price': 50000.0,
#             'id': '12345xyz', 'feed': 'COINBASE', 'timestamp': 1620000000.123456
#         }
#
#         # Format the data
#         formatted_trade = await callback.format(trade_data, trade_data['timestamp'])
#         print(f"Formatted: {formatted_trade}")

#         # Write the data (assuming InfluxDB is running and configured)
#         try:
#             await callback.writer([formatted_trade])
#             print("Data written successfully (mock).")
#         except Exception as e:
#             print(f"Error writing data: {e}")
#
#         await callback.close_session() # Close the aiohttp session

#     import asyncio
#     asyncio.run(main())

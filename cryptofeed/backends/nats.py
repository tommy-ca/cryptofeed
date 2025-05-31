"""
NATS Backend for Cryptofeed

This backend publishes data from Cryptofeed to NATS (Neural Autonomic Transport System)
subjects. It allows for configurable subject naming and handles NATS connection
management and error handling.
"""
import asyncio
import logging

import nats
from nats.errors import ConnectionClosedError, TimeoutError as NatsTimeoutError, NoServersError

from cryptofeed.backends.backend import BackendQueue, BackendCallback, BackendBookCallback

LOG = logging.getLogger('feedhandler')


class NATSCallback(BackendQueue):
    """
    NATS Backend Base Class

    Parameters
    ----------
    addr : str, list[str], optional
        NATS server URL or list of URLs. Defaults to 'nats://localhost:4222'.
    subject_prefix : str, optional
        Prefix for NATS subjects. Defaults to 'cryptofeed'.
    jetstream_mode : bool, optional
        If True, enables NATS JetStream publishing. Defaults to False (core NATS publish).
    jetstream_timeout : float, optional
        Timeout in seconds for JetStream operations (e.g., publish ack wait, context creation).
        Defaults to 2.0 seconds.
    **kwargs :
        Additional keyword arguments.
    """
    def __init__(self, addr='nats://localhost:4222', subject_prefix='cryptofeed',
                 jetstream_mode: bool = False, jetstream_timeout: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.addr = addr
        self.subject_prefix = subject_prefix
        self.jetstream_mode = jetstream_mode
        self.jetstream_timeout = jetstream_timeout
        self.nc = None
        self.js = None # JetStream context

    async def _connect(self):
        """
        Connects to the NATS server.
        """
        if self.nc and self.nc.is_connected:
            # If also using JetStream, ensure JetStream context is also valid or recreated
            if self.jetstream_mode and not self.js: # JS context might be lost even if nc is connected briefly
                try:
                    LOG.info("NATSBackend: Attempting to get JetStream context.")
                    self.js = self.nc.jetstream(timeout=self.jetstream_timeout)
                    LOG.info("NATSBackend: JetStream context obtained.")
                except Exception as e_js:
                    LOG.error("NATSBackend: Failed to get JetStream context during reconnect: %s", e_js)
                    self.js = None # Ensure it's None if failed
                    # We might need to raise this or handle it to force a full reconnect if JS is critical
            return

        try:
            LOG.info("NATSBackend: Connecting to NATS server(s) at %s", self.addr)
            # Reset JetStream context before new connection
            self.js = None
            self.nc = await nats.connect(servers=self.addr, error_cb=self._error_cb, closed_cb=self._closed_cb, reconnected_cb=self._reconnected_cb)
            LOG.info("NATSBackend: Connected to NATS server(s) at %s", self.nc.connected_url.netloc if self.nc.connected_url else self.addr)

            if self.jetstream_mode:
                LOG.info("NATSBackend: JetStream mode enabled. Getting JetStream context.")
                try:
                    self.js = self.nc.jetstream(timeout=self.jetstream_timeout)
                    LOG.info("NATSBackend: JetStream context obtained successfully.")
                except Exception as e_js:
                    LOG.error("NATSBackend: Failed to get JetStream context: %s. JetStream publishing will likely fail.", e_js)
                    # Depending on strictness, could raise an error here or let it fail on publish
                    self.js = None # Ensure it's None if failed
        except NoServersError as e:
            LOG.error("NATSBackend: No NATS servers available for connection: %s", e)
            raise
        except Exception as e:
            LOG.error("NATSBackend: Error connecting to NATS: %s", e)
            raise

    async def _error_cb(self, e):
        LOG.error("NATSBackend: NATS Error: %s. JetStream context may need to be re-established on reconnect.", e)
        # If JetStream was active, it might be invalidated by some errors.
        # Setting self.js to None here could trigger re-acquisition in _connect or writer.
        self.js = None


    async def _closed_cb(self):
        LOG.warning("NATSBackend: NATS connection closed. JetStream context invalidated.")
        self.js = None

    async def _reconnected_cb(self):
        LOG.info("NATSBackend: Reconnected to NATS server at %s", self.nc.connected_url.netloc if self.nc.connected_url else "unknown")
        # Re-establish JetStream context if in JetStream mode
        if self.jetstream_mode and self.nc and self.nc.is_connected:
            try:
                LOG.info("NATSBackend: Re-establishing JetStream context after reconnect.")
                self.js = self.nc.jetstream(timeout=self.jetstream_timeout)
                LOG.info("NATSBackend: JetStream context re-established successfully.")
            except Exception as e:
                LOG.error("NATSBackend: Failed to re-establish JetStream context after reconnect: %s", e)
                self.js = None


    async def writer(self):

    async def _closed_cb(self):
        LOG.warning("NATSBackend: NATS connection closed.")

    async def _reconnected_cb(self):
        LOG.info("NATSBackend: Reconnected to NATS server at %s", self.nc.connected_url.netloc if self.nc.connected_url else "unknown")


    async def writer(self):
        """
        NATS writer method.

        Connects to NATS, consumes data from the queue, and publishes it.
        Handles connection errors and retries.
        """
        retry_delay = 1  # seconds
        max_retry_delay = 60

        while True:
            try:
                if not self.nc or not self.nc.is_connected or (self.jetstream_mode and not self.js):
                    await self._connect() # This will also attempt to set up self.js if jetstream_mode is True
                    if self.jetstream_mode and not self.js:
                        LOG.error("NATSBackend: JetStream context not available after connect. Retrying connection.")
                        raise ConnectionClosedError("JetStream context unavailable.")


                async with self.read_queue() as updates:
                    for update in updates:
                        data_type = update['data_type']
                        exchange = update['exchange']
                        symbol = update['symbol']
                        data = update['data']

                        subject_name = f"{self.subject_prefix}-{data_type}-{exchange}-{symbol}"
                        payload_bytes = self.serializer(data).encode()

                        try:
                            if self.jetstream_mode:
                                if not self.js: # Should have been caught by check above, but as safeguard
                                    LOG.error("NATSBackend: JetStream context lost. Attempting to reconnect and get context.")
                                    await self._connect()
                                    if not self.js:
                                        raise ConnectionClosedError("JetStream context could not be re-established.")
                                # For JetStream, always await ack for now as per simplified plan
                                await self.js.publish(subject=subject_name, payload=payload_bytes, timeout=self.jetstream_timeout)
                            else:
                                await self.nc.publish(subject_name, payload_bytes)
                        except ConnectionClosedError: # Core NATS or JetStream connection issue
                            LOG.warning("NATSBackend: Connection closed while publishing. Attempting to reconnect...")
                            await self._connect() # Reconnect and retry publishing
                            # Retry logic after reconnect
                            if self.jetstream_mode:
                                if not self.js: raise ConnectionClosedError("JetStream context unavailable post-reconnect for retry.")
                                await self.js.publish(subject=subject_name, payload=payload_bytes, timeout=self.jetstream_timeout)
                            else:
                                await self.nc.publish(subject_name, payload_bytes)
                        except NatsTimeoutError as e_timeout: # Specific to JetStream ack wait or core NATS publish timeout
                            LOG.warning("NATSBackend: Timeout publishing to %s (JetStream: %s): %s. Message may not have been delivered or ack lost.",
                                        subject_name, self.jetstream_mode, e_timeout)
                            # For JetStream, a timeout means ack was not received. The message might have been published.
                            # No automatic retry here to avoid duplicates unless we are sure.
                            # If core NATS publish timed out (less common for nc.publish unless server is slow)
                            # a retry might be considered, but current code doesn't for core NATS timeout on publish.
                            # The old code had a retry for nc.publish timeout, let's reconsider if that's needed.
                            # For now, log and continue for timeouts on publish.
                        except Exception as e:
                            LOG.error("NATSBackend: Error publishing to NATS (JetStream: %s) subject %s: %s",
                                      self.jetstream_mode, subject_name, e, exc_info=True)
                            # Depending on the error, might need to reconnect or take other actions

                retry_delay = 1 # Reset retry delay on successful publish cycle

            except NoServersError:
                LOG.error("NATSBackend: No NATS servers available. Retrying in %d seconds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            except ConnectionClosedError:
                LOG.warning("NATSBackend: NATS connection closed unexpectedly. Retrying in %d seconds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            except NatsTimeoutError:
                LOG.warning("NATSBackend: NATS connection timeout. Retrying in %d seconds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            except Exception as e:
                LOG.error("NATSBackend: Unhandled exception in writer: %s. Retrying in %d seconds...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            finally:
                if self.nc and self.nc.is_connected and not self.nc.is_draining:
                    pass # Keep connection open unless specific error forces closure
                elif self.nc and (self.nc.is_closed or self.nc.is_draining):
                    LOG.info("NATSBackend: Attempting to close NATS connection.")
                    try:
                        if self.nc and not self.nc.is_draining and not self.nc.is_closed:
                             await self.nc.drain() # Ensure all buffered messages are sent
                             LOG.info("NATSBackend: NATS connection drained and closed.")
                        elif self.nc and (self.nc.is_closed or self.nc.is_draining):
                             LOG.info("NATSBackend: NATS connection already closed or draining.")
                    except Exception as e:
                        LOG.error("NATSBackend: Error draining NATS connection: %s", e)
                    self.nc = None
                    self.js = None # Clear JetStream context as well


class TradeNATS(NATSCallback, BackendCallback):
    default_key = 'trades'


class FundingNATS(NATSCallback, BackendCallback):
    default_key = 'funding'


class BookNATS(NATSCallback, BackendBookCallback):
    default_key = 'book'


class TickerNATS(NATSCallback, BackendCallback):
    default_key = 'ticker'


class OpenInterestNATS(NATSCallback, BackendCallback):
    default_key = 'open_interest'


class LiquidationsNATS(NATSCallback, BackendCallback):
    default_key = 'liquidations'


class CandlesNATS(NATSCallback, BackendCallback):
    default_key = 'candles'


class OrderInfoNATS(NATSCallback, BackendCallback):
    default_key = 'order_info'


class TransactionsNATS(NATSCallback, BackendCallback):
    default_key = 'transactions'


class BalancesNATS(NATSCallback, BackendCallback):
    default_key = 'balances'


class FillsNATS(NATSCallback, BackendCallback):
    default_key = 'fills'

# Example of how data is expected to be structured for publishing
# update = {
#     'data_type': 'trades',  # Or 'book', 'ticker', etc.
#     'exchange': 'coinbase',
#     'symbol': 'BTC-USD',
#     'data': {...}  # The actual trade, book, or ticker data
# }

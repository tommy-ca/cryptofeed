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
    **kwargs :
        Additional keyword arguments.
    """
    def __init__(self, addr='nats://localhost:4222', subject_prefix='cryptofeed', **kwargs):
        super().__init__(**kwargs)
        self.addr = addr
        self.subject_prefix = subject_prefix
        self.nc = None

    async def _connect(self):
        """
        Connects to the NATS server.
        """
        if self.nc and self.nc.is_connected:
            return

        try:
            LOG.info("NATSBackend: Connecting to NATS server(s) at %s", self.addr)
            self.nc = await nats.connect(servers=self.addr, error_cb=self._error_cb, closed_cb=self._closed_cb, reconnected_cb=self._reconnected_cb)
            LOG.info("NATSBackend: Connected to NATS server(s) at %s", self.nc.connected_url.netloc if self.nc.connected_url else self.addr)
        except NoServersError as e:
            LOG.error("NATSBackend: No NATS servers available for connection: %s", e)
            raise
        except Exception as e:
            LOG.error("NATSBackend: Error connecting to NATS: %s", e)
            raise

    async def _error_cb(self, e):
        LOG.error("NATSBackend: NATS Error: %s", e)

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
                if not self.nc or not self.nc.is_connected:
                    await self._connect()

                async with self.read_queue() as updates:
                    for update in updates:
                        data_type = update['data_type']
                        exchange = update['exchange']
                        symbol = update['symbol']
                        data = update['data']

                        subject = f"{self.subject_prefix}-{data_type}-{exchange}-{symbol}"

                        try:
                            await self.nc.publish(subject, self.serializer(data).encode())
                        except ConnectionClosedError:
                            LOG.warning("NATSBackend: Connection closed while publishing. Attempting to reconnect...")
                            await self._connect() # Reconnect and retry publishing
                            await self.nc.publish(subject, self.serializer(data).encode())
                        except NatsTimeoutError:
                            LOG.warning("NATSBackend: Timeout while publishing to %s. Retrying...", subject)
                            # Basic retry, consider more sophisticated backoff
                            await asyncio.sleep(1)
                            await self.nc.publish(subject, self.serializer(data).encode())
                        except Exception as e:
                            LOG.error("NATSBackend: Error publishing to NATS: %s", e)
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
                        if not self.nc.is_draining: # Avoid draining if already closed.
                             await self.nc.drain() # Ensure all buffered messages are sent
                        LOG.info("NATSBackend: NATS connection drained and closed.")
                    except Exception as e:
                        LOG.error("NATSBackend: Error draining NATS connection: %s", e)
                    self.nc = None # Reset connection object


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

'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import asyncio
import logging
from socket import error as socket_error
import time
from typing import Awaitable
import zlib
import contextlib

from websockets import ConnectionClosed

from cryptofeed.connection import AsyncConnection
from cryptofeed.exceptions import ExhaustedRetries
from cryptofeed.defines import HUOBI, HUOBI_DM, HUOBI_SWAP, OKCOIN, OKX


LOG = logging.getLogger('feedhandler')


class ConnectionHandler:
    def __init__(self, conn: AsyncConnection, subscribe: Awaitable, handler: Awaitable, authenticate: Awaitable, retries: int, timeout=120, timeout_interval=30, exceptions=None, log_on_error=False, start_delay=0):
        self.conn = conn
        self.subscribe = subscribe
        self.handler = handler
        self.authenticate = authenticate
        self.retries = retries
        self.exceptions = exceptions
        self.log_on_error = log_on_error
        self.timeout = timeout
        self.timeout_interval = timeout_interval
        self.running = True
        self.start_delay = start_delay
        self._watcher_task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop):
        loop.create_task(self._create_connection())

    async def _watcher(self):
        while self.conn.is_open and self.running:
            if self.conn.last_message:
                if time.time() - self.conn.last_message > self.timeout:
                    LOG.warning("%s: received no messages within timeout, restarting connection", self.conn.uuid)
                    await self.conn.close()
                    break
            await asyncio.sleep(self.timeout_interval)

    async def _create_connection(self):
        await asyncio.sleep(self.start_delay)
        retries = 0
        delay = 1

        while self._within_retry_budget(retries) and self.running:
            try:
                await self._establish_connection()
                retries = 0
                delay = 1
            except (ConnectionClosed, ConnectionAbortedError, ConnectionResetError, socket_error) as exc:
                await self._handle_retry(exc, delay, LOG.warning, include_exc_message=True)
                retries += 1
                delay *= 2
            except Exception as exc:  # pragma: no cover - defensive
                await self._handle_retry(exc, delay, LOG.error, include_exc_message=False)
                retries += 1
                delay *= 2

        if not self.running:
            LOG.info('%s: terminate the connection handler because not running', self.conn.uuid)
            return

        LOG.error('%s: failed to reconnect after %d retries - exiting', self.conn.uuid, retries)
        raise ExhaustedRetries()

    def _within_retry_budget(self, retries: int) -> bool:
        return self.retries == -1 or retries <= self.retries

    async def _establish_connection(self) -> None:
        async with self.conn.connect() as connection:
            await self.authenticate(connection)
            await self.subscribe(connection)
            if self.timeout != -1:
                loop = asyncio.get_running_loop()
                self._watcher_task = loop.create_task(self._watcher())
            await self._handler(connection, self.handler)
        await self._cancel_watcher()

    async def _cancel_watcher(self) -> None:
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            with contextlib.suppress(Exception):
                await self._watcher_task
        self._watcher_task = None

    async def _handle_retry(self, exc: Exception, delay: float, log_method, *, include_exc_message: bool) -> None:
        if self._should_raise(exc):
            raise
        if include_exc_message:
            log_method("%s: encountered connection issue %s - reconnecting in %.1f seconds...", self.conn.uuid, str(exc), delay, exc_info=True)
        else:
            log_method("%s: encountered an exception, reconnecting in %.1f seconds", self.conn.uuid, delay, exc_info=True)
        await asyncio.sleep(delay)

    def _should_raise(self, exc: Exception) -> bool:
        if not self.exceptions:
            return False
        for ignored in self.exceptions:
            if isinstance(exc, ignored):
                LOG.warning("%s: encountered exception %s, which is on the ignore list. Raising", self.conn.uuid, str(exc))
                return True
        return False

    async def _handler(self, connection, handler):
        try:
            async for message in connection.read():
                if not self.running:
                    await connection.close()
                    return
                await handler(message, connection, self.conn.last_message)
        except Exception:
            if not self.running:
                return
            if self.log_on_error:
                if connection.uuid in {HUOBI, HUOBI_DM, HUOBI_SWAP}:
                    message = zlib.decompress(message, 16 + zlib.MAX_WBITS)
                elif connection.uuid in {OKCOIN, OKX}:
                    message = zlib.decompress(message, -15)
                LOG.error("%s: error handling message %s", connection.uuid, message)
            # exception will be logged with traceback when connection handler
            # retries the connection
            raise

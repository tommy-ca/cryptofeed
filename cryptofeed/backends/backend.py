"""
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import asyncio
import logging
from asyncio.queues import Queue
from multiprocessing import Pipe, Process
from contextlib import asynccontextmanager
from typing import Union, cast
from abc import ABC, abstractmethod

from cryptofeed.backends.protobuf_helpers import (
    serialize_to_protobuf,
)


LOG = logging.getLogger("feedhandler")

SHUTDOWN_SENTINEL = "STOP"


class BackendQueue:
    def start(self, loop: asyncio.AbstractEventLoop, multiprocess=False):
        if hasattr(self, "started") and self.started:
            # prevent a backend callback from starting more than 1 writer and creating more than 1 queue
            return
        self.multiprocess = multiprocess
        if self.multiprocess:
            self.queue = Pipe(duplex=False)
            self.worker = Process(
                target=BackendQueue.worker, args=(self.writer,), daemon=True
            )
            cast(Process, self.worker).start()
        else:
            self.queue = Queue()
            self.worker = loop.create_task(self.writer())
        self.started = True

    async def stop(self):
        if self.multiprocess:
            self.queue[1].send(SHUTDOWN_SENTINEL)
            cast(Process, self.worker).join()
        else:
            await cast(Queue, self.queue).put(SHUTDOWN_SENTINEL)
        self.running = False

    @staticmethod
    def worker(writer):
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(writer())
        except KeyboardInterrupt:
            pass

    async def writer(self):
        raise NotImplementedError

    async def write(self, data):
        if self.multiprocess:
            self.queue[1].send(data)
        else:
            await cast(Queue, self.queue).put(data)

    @asynccontextmanager
    async def read_queue(self) -> list:
        if self.multiprocess:
            msg = self.queue[0].recv()
            if msg == SHUTDOWN_SENTINEL:
                self.running = False
                yield []
            else:
                yield [msg]
        else:
            queue = cast(Queue, self.queue)
            current_depth = queue.qsize()
            if current_depth == 0:
                update = await queue.get()
                if update == SHUTDOWN_SENTINEL:
                    yield []
                else:
                    yield [update]
                queue.task_done()
            else:
                ret = []
                count = 0
                while current_depth > count:
                    update = await queue.get()
                    count += 1
                    if update == SHUTDOWN_SENTINEL:
                        self.running = False
                        break
                    ret.append(update)

                yield ret

                for _ in range(count):
                    queue.task_done()


class BackendCallback(ABC):
    """
    Base class for backend callbacks with pluggable serialization support.

    Supports both JSON (default, backward compatible) and Protobuf serialization formats.
    The serialization_format parameter can be set via:
    - Constructor parameter: serialization_format='protobuf'
    - YAML configuration: serialization_format: protobuf
    - Environment variable: CRYPTOFEED_SERIALIZATION_FORMAT=protobuf

    Design Principles:
    - Backward Compatibility: Defaults to JSON (existing to_dict() behavior)
    - Format Selection: Simple per-callback configuration
    - Minimal Overhead: Direct serialization calls in backends/
    """

    _explicit_serialization_format: str | None = None
    _serialization_log_state: tuple[str, str] | None = None
    _serialization_locked: bool = False

    def __init__(self, numeric_type=float, none_to=None):
        """Initialize backend callback with serialization parameters."""
        self.numeric_type = numeric_type
        self.none_to = none_to

    @abstractmethod
    async def write(self, data):
        """Write data to the backend. Must be implemented by subclasses."""
        pass

    def set_serialization_format(self, format_name: str | None) -> None:
        """Persist an explicit serialization format override for this callback."""

        if getattr(self, "_serialization_locked", False):
            if format_name is None and self._explicit_serialization_format is None:
                return
            if format_name is not None:
                normalized = self._validate_format(format_name)
                if self._explicit_serialization_format == normalized:
                    return
            raise RuntimeError(
                f"{self.__class__.__name__}: serialization format already locked"
            )

        if format_name is None:
            self._explicit_serialization_format = None
        else:
            self._explicit_serialization_format = self._validate_format(format_name)
            self._serialization_locked = True

    @staticmethod
    def _validate_format(format_name: str) -> str:
        """Validate and normalize serialization format."""
        normalized = format_name.lower().strip()
        if normalized not in ("json", "protobuf"):
            raise ValueError(
                f"Invalid serialization format '{format_name}'. "
                f"Valid formats: json, protobuf"
            )
        return normalized

    @staticmethod
    def _get_format_from_env() -> str | None:
        """Get serialization format from environment variable."""
        import os

        env_value = os.environ.get("CRYPTOFEED_SERIALIZATION_FORMAT")
        deprecated_value = (
            os.environ.get("CRYPTOFEED_CALLBACK_FORMAT") if env_value is None else None
        )

        if env_value:
            return BackendCallback._validate_format(env_value)

        if deprecated_value:
            LOG.warning(
                "CRYPTOFEED_CALLBACK_FORMAT is deprecated; use CRYPTOFEED_SERIALIZATION_FORMAT instead"
            )
            return BackendCallback._validate_format(deprecated_value)
        return None

    @property
    def serialization_format(self) -> str:
        """Active serialization format after applying env overrides."""

        preferred = getattr(self, "_explicit_serialization_format", None)

        env_value = self._get_format_from_env()
        if env_value is not None:
            resolved = env_value
            source = "env"
        elif preferred is not None:
            resolved = preferred
            source = "explicit"
        else:
            resolved = "json"
            source = "default"

        if getattr(self, "_serialization_log_state", None) != (source, resolved):
            LOG.info(
                "%s: serialization_format=%s (source=%s)",
                self.__class__.__name__,
                resolved,
                source,
            )
            self._serialization_log_state = (source, resolved)

        return resolved

    def _build_dict_payload(self, dtype, receipt_timestamp: float) -> dict:
        """Normalize data objects into dictionaries for JSON/backward paths."""

        data = dtype.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
        if not getattr(dtype, "timestamp", None):
            data["timestamp"] = receipt_timestamp
        data["receipt_timestamp"] = receipt_timestamp
        return data

    async def __call__(self, dtype, receipt_timestamp: float):
        """Default implementation: emit JSON-compatible dictionaries or protobuf."""

        if self.serialization_format == "protobuf":
            # Protobuf serialization: use consolidated helpers from backends
            payload = serialize_to_protobuf(dtype)
        else:
            # JSON serialization: use existing to_dict() path
            payload = self._build_dict_payload(dtype, receipt_timestamp)

        await self.write(payload)


class BackendBookCallback(BackendCallback):
    def __init__(
        self,
        snapshots_only=False,
        snapshot_interval=1000,
        numeric_type=float,
        none_to=None,
    ):
        """Initialize book callback with snapshot parameters."""
        super().__init__(numeric_type=numeric_type, none_to=none_to)
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = {}

    async def _write_snapshot(self, book, receipt_timestamp: float):
        data = book.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
        del data["delta"]
        if not book.timestamp:
            data["timestamp"] = receipt_timestamp
        data["receipt_timestamp"] = receipt_timestamp
        await self.write(data)

    async def __call__(self, book, receipt_timestamp: float):
        if self.snapshots_only:
            await self._write_snapshot(book, receipt_timestamp)
        else:
            data = book.to_dict(
                delta=book.delta is not None,
                numeric_type=self.numeric_type,
                none_to=self.none_to,
            )
            if not book.timestamp:
                data["timestamp"] = receipt_timestamp
            data["receipt_timestamp"] = receipt_timestamp

            if book.delta is None:
                del data["delta"]
            else:
                self.snapshot_count[book.symbol] += 1
            await self.write(data)
            if (
                self.snapshot_interval <= self.snapshot_count[book.symbol]
                and book.delta
            ):
                await self._write_snapshot(book, receipt_timestamp)
                self.snapshot_count[book.symbol] = 0

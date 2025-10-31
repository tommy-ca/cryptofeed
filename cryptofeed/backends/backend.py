'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import asyncio
from asyncio.queues import Queue
from multiprocessing import Pipe, Process
from contextlib import asynccontextmanager


SHUTDOWN_SENTINEL = 'STOP'


class BackendQueue:
    def start(self, loop: asyncio.AbstractEventLoop, multiprocess=False):
        if hasattr(self, 'started') and self.started:
            # prevent a backend callback from starting more than 1 writer and creating more than 1 queue
            return
        self.multiprocess = multiprocess
        if self.multiprocess:
            self.queue = Pipe(duplex=False)
            self.worker = Process(target=BackendQueue.worker, args=(self.writer,), daemon=True)
            self.worker.start()
        else:
            self.queue = Queue()
            self.worker = loop.create_task(self.writer())
        self.started = True

    async def stop(self):
        if self.multiprocess:
            self.queue[1].send(SHUTDOWN_SENTINEL)
            self.worker.join()
        else:
            await self.queue.put(SHUTDOWN_SENTINEL)
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
            await self.queue.put(data)

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
            current_depth = self.queue.qsize()
            if current_depth == 0:
                update = await self.queue.get()
                if update == SHUTDOWN_SENTINEL:
                    yield []
                else:
                    yield [update]
                self.queue.task_done()
            else:
                ret = []
                count = 0
                while current_depth > count:
                    update = await self.queue.get()
                    count += 1
                    if update == SHUTDOWN_SENTINEL:
                        self.running = False
                        break
                    ret.append(update)

                yield ret

                for _ in range(count):
                    self.queue.task_done()


class BackendCallback:
    """
    Base class for backend callbacks with pluggable serialization support.
    
    Supports both JSON (default, backward compatible) and Protobuf serialization formats.
    The serialization_format parameter can be set via:
    - Constructor parameter: serialization_format='protobuf'
    - YAML configuration: serialization_format: protobuf
    - Environment variable: CRYPTOFEED_SERIALIZATION_FORMAT=protobuf (future)
    
    Design Principles:
    - Backward Compatibility: Defaults to JSON (existing to_dict() behavior)
    - Dependency Inversion: Depends on Serializer abstraction, not concrete classes
    - Open/Closed: Open for new serialization formats via subclassing
    """
    
    def _get_serializer(self, format_name: str):
        """
        Factory method for serializer selection.
        
        Args:
            format_name: 'json' or 'protobuf'
            
        Returns:
            Serializer instance
            
        Raises:
            ValueError: If format_name is invalid
        """
        from cryptofeed.serializers import JSONSerializer
        
        if format_name == 'json':
            return JSONSerializer()
        elif format_name == 'protobuf':
            from cryptofeed.serializers.protobuf import ProtobufSerializer
            return ProtobufSerializer()
        else:
            raise ValueError(
                f"Invalid serialization format '{format_name}'. "
                f"Valid formats: json, protobuf"
            )
    
    async def __call__(self, dtype, receipt_timestamp: float):
        """
        Process data type and write to backend.
        
        For backward compatibility, continues to use to_dict() and pass
        dictionaries to write(). Protobuf serialization will be integrated
        in backends that support it (Kafka, Redis) via value_serializer.
        """
        data = dtype.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
        if not dtype.timestamp:
            data['timestamp'] = receipt_timestamp
        data['receipt_timestamp'] = receipt_timestamp
        await self.write(data)


class BackendBookCallback:
    async def _write_snapshot(self, book, receipt_timestamp: float):
        data = book.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
        del data['delta']
        if not book.timestamp:
            data['timestamp'] = receipt_timestamp
        data['receipt_timestamp'] = receipt_timestamp
        await self.write(data)

    async def __call__(self, book, receipt_timestamp: float):
        if self.snapshots_only:
            await self._write_snapshot(book, receipt_timestamp)
        else:
            data = book.to_dict(delta=book.delta is not None, numeric_type=self.numeric_type, none_to=self.none_to)
            if not book.timestamp:
                data['timestamp'] = receipt_timestamp
            data['receipt_timestamp'] = receipt_timestamp

            if book.delta is None:
                del data['delta']
            else:
                self.snapshot_count[book.symbol] += 1
            await self.write(data)
            if self.snapshot_interval <= self.snapshot_count[book.symbol] and book.delta:
                await self._write_snapshot(book, receipt_timestamp)
                self.snapshot_count[book.symbol] = 0

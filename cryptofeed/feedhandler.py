'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import asyncio
from collections.abc import Mapping
from cryptofeed.connection import Connection
import logging
import signal
from signal import SIGABRT, SIGINT, SIGTERM
import sys
from typing import List

try:
    # unix / macos only
    from signal import SIGHUP
    SIGNALS = (SIGABRT, SIGINT, SIGTERM, SIGHUP)
except ImportError:
    SIGNALS = (SIGABRT, SIGINT, SIGTERM)

from yapic import json

from cryptofeed.config import Config
from cryptofeed.defines import L2_BOOK
from cryptofeed.feed import Feed
from cryptofeed.log import get_logger
from cryptofeed.nbbo import NBBO
from cryptofeed.exchanges import EXCHANGE_MAP
from cryptofeed.proxy import ProxySettings, init_proxy_system, load_proxy_settings


LOG = logging.getLogger('feedhandler')


def setup_signal_handlers(loop):
    """
    This must be run from the loop in the main thread
    """
    def handle_stop_signals(*args):
        raise SystemExit
    if sys.platform.startswith('win'):
        # NOTE: asyncio loop.add_signal_handler() not supported on windows
        for sig in SIGNALS:
            signal.signal(sig, handle_stop_signals)
    else:
        for sig in SIGNALS:
            loop.add_signal_handler(sig, handle_stop_signals)


class FeedHandler:
    def __init__(self, config=None, raw_data_collection=None, proxy_settings=None):
        """
        config: str, dict or None
            if str, absolute path (including file name) of the config file. If not provided, config can also be a dictionary of values, or
            can be None, which will default options. See docs/config.md for more information.
        raw_data_collection: callback (see AsyncFileCallback) or None
            if set, enables collection of raw data from exchanges. ALL https/wss traffic from the exchanges will be collected.
        proxy_settings: ProxySettings, dict, or None
            optional explicit proxy configuration. Environment variables take precedence over config and explicit settings.
        """
        self.feeds = []
        self.config = Config(config=config)
        self.raw_data_collection = None
        self.running = False
        if raw_data_collection:
            Connection.raw_data_callback = raw_data_collection
            self.raw_data_collection = raw_data_collection

        if not self.config.log.disabled:
            get_logger('feedhandler', self.config.log.filename, self.config.log.level)

        if self.config.log_msg:
            LOG.info(self.config.log_msg)

        if self.config.uvloop:
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                LOG.info('FH: uvloop initalized')
            except ImportError:
                LOG.info("FH: uvloop not initialized")

        self._initialize_proxy_system(proxy_settings)

    def _initialize_proxy_system(self, explicit_settings):
        """Initialize proxy system using env → YAML → explicit precedence."""

        def _coerce_to_plain_dict(value):
            if isinstance(value, Mapping):
                return {k: _coerce_to_plain_dict(v) for k, v in value.items()}
            return value

        def _normalize_root_config(value):
            if isinstance(value, str):
                return {
                    'enabled': True,
                    'default': {
                        'http': value,
                        'websocket': value,
                    },
                }
            return value

        env_settings = load_proxy_settings()

        config_settings = None
        if 'proxy' in self.config:
            raw_proxy_config = _normalize_root_config(self.config['proxy'])
            if raw_proxy_config:
                config_settings = ProxySettings(**_coerce_to_plain_dict(raw_proxy_config))

        explicit_proxy_settings = None
        if explicit_settings is not None:
            normalized_explicit = _normalize_root_config(explicit_settings)
            if isinstance(explicit_settings, ProxySettings):
                explicit_proxy_settings = explicit_settings
            elif isinstance(normalized_explicit, Mapping):
                explicit_proxy_settings = ProxySettings(**_coerce_to_plain_dict(normalized_explicit))
            else:
                raise TypeError('proxy_settings must be a ProxySettings instance or mapping')

        if env_settings.model_fields_set:
            settings = env_settings
        elif config_settings is not None:
            settings = config_settings
        elif explicit_proxy_settings is not None:
            settings = explicit_proxy_settings
        else:
            settings = env_settings

        init_proxy_system(settings)

    def add_feed(self, feed, loop=None, **kwargs):
        """
        feed: str or class
            the feed (exchange) to add to the handler
        loop: event loop
            the event loop to use for the feed (only when the feedhandler is running)
        kwargs: dict
            if a string is used for the feed, kwargs will be passed to the
            newly instantiated object
        """
        if isinstance(feed, str):
            feed_key = feed.upper()
            if feed_key == "BACKPACK_CCXT":
                raise ValueError(
                    "Backpack ccxt integration has been removed. Configure the native 'BACKPACK' feed instead."
                )

            if feed_key in EXCHANGE_MAP:
                feed_cls = EXCHANGE_MAP[feed_key]
                config_override = kwargs.pop("config", None)

                if feed_key == "BACKPACK":
                    from cryptofeed.config import Config as FeedConfig
                    from cryptofeed.exchanges.backpack.config import BackpackConfig

                    handler_config_arg = self.config

                    if isinstance(config_override, BackpackConfig):
                        backpack_config = config_override
                    elif isinstance(config_override, Mapping):
                        backpack_config = self._resolve_backpack_config(config_override)
                    elif config_override is not None:
                        handler_config_arg = config_override
                        resolution_source = handler_config_arg
                        if not isinstance(resolution_source, FeedConfig):
                            try:
                                resolution_source = FeedConfig(config=resolution_source)
                            except Exception:
                                resolution_source = self.config
                        backpack_config = self._resolve_backpack_config(resolution_source)
                    else:
                        backpack_config = self._resolve_backpack_config(None)

                    backpack_override = kwargs.pop("backpack_config", None)
                    if isinstance(backpack_override, BackpackConfig):
                        backpack_config = backpack_override

                    self.feeds.append(
                        (
                            feed_cls(
                                config=handler_config_arg,
                                backpack_config=backpack_config,
                                **kwargs,
                            )
                        )
                    )
                else:
                    config_value = config_override if config_override is not None else self.config
                    self.feeds.append((feed_cls(config=config_value, **kwargs)))
            else:
                raise ValueError("Invalid feed specified")
        else:
            self.feeds.append((feed))
        if self.raw_data_collection:
            self.raw_data_collection.write_header(self.feeds[-1].id, json.dumps(self.feeds[-1]._feed_config))

        if self.running:
            if loop is None:
                loop = asyncio.get_event_loop()

            self.feeds[-1].start(loop)

    def _resolve_backpack_config(self, explicit):
        """
        Derive a BackpackConfig instance from explicit overrides or handler config.
        """
        from cryptofeed.config import Config, AttrDict
        from cryptofeed.exchanges.backpack.config import BackpackConfig

        if isinstance(explicit, BackpackConfig):
            return explicit

        def _to_plain_mapping(value):
            if isinstance(value, AttrDict):
                return {k: _to_plain_mapping(v) for k, v in value.items()}
            if isinstance(value, Config):
                return _to_plain_mapping(value.config)
            if isinstance(value, Mapping):
                return {k: _to_plain_mapping(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_plain_mapping(v) for v in value]
            return value

        def _candidate_from(value):
            if value is None:
                return None
            if isinstance(value, BackpackConfig):
                return value
            if isinstance(value, Config):
                value = value.config
            if isinstance(value, AttrDict):
                value = dict(value)
            if isinstance(value, Mapping):
                return value
            return None

        candidates: list[tuple[object, bool]] = []
        seen_candidates: set[int] = set()

        explicit_candidate = _candidate_from(explicit)
        explicit_is_mapping = isinstance(explicit, Mapping) and not isinstance(explicit, Config)
        if isinstance(explicit_candidate, BackpackConfig):
            return explicit_candidate
        if explicit_candidate is not None and explicit_is_mapping and id(explicit_candidate) not in seen_candidates:
            candidates.append((explicit_candidate, True))
            seen_candidates.add(id(explicit_candidate))

        def _lookup_backpack_section(config_source):
            if not isinstance(config_source, Mapping):
                return None

            def _match_key(source: Mapping, target: str):
                for key, value in source.items():
                    if isinstance(key, str) and key.casefold() == target:
                        return value
                return None

            direct = _match_key(config_source, "backpack")
            if direct:
                return direct

            exchanges = None
            for key, value in config_source.items():
                if isinstance(key, str) and key.casefold() == "exchanges":
                    exchanges = value
                    break
            if isinstance(exchanges, Mapping):
                return _match_key(exchanges, "backpack")

            return None

        root_plain = _to_plain_mapping(self.config.config if isinstance(self.config, Config) else self.config)
        section_from_explicit = _candidate_from(
            _lookup_backpack_section(explicit_candidate) if isinstance(explicit_candidate, Mapping) else None
        )
        if section_from_explicit is not None and id(section_from_explicit) not in seen_candidates:
            candidates.append((section_from_explicit, True))
            seen_candidates.add(id(section_from_explicit))

        section_from_root = _candidate_from(
            _lookup_backpack_section(root_plain) if isinstance(root_plain, Mapping) else None
        )
        if section_from_root is not None and id(section_from_root) not in seen_candidates:
            candidates.append((section_from_root, False))
            seen_candidates.add(id(section_from_root))

        from pydantic import ValidationError

        if not candidates:
            return BackpackConfig()

        allowed_keys = set(BackpackConfig.model_fields.keys())
        errors: list[ValidationError] = []
        for candidate, is_explicit in candidates:
            if isinstance(candidate, BackpackConfig):
                return candidate
            plain_candidate = _to_plain_mapping(candidate)
            if isinstance(plain_candidate, Mapping) and plain_candidate:
                invalid_keys = set(plain_candidate.keys()).difference(allowed_keys)
                if invalid_keys:
                    if is_explicit:
                        raise ValueError(
                            f"Backpack configuration contains unsupported keys: {sorted(invalid_keys)}"
                        )
                    continue
                try:
                    return BackpackConfig.model_validate(plain_candidate)
                except ValidationError as exc:
                    if is_explicit:
                        raise exc
                    errors.append(exc)
                    continue

        if errors:
            raise errors[0]

        return BackpackConfig()

    def add_nbbo(self, feeds: List[Feed], symbols: List[str], callback, config=None):
        """
        feeds: list of feed classes
            list of feeds (exchanges) that comprises the NBBO
        symbols: list str
            the trading symbols
        callback: function pointer
            the callback to be invoked when a new tick is calculated for the NBBO
        config: dict, str, or None
            optional information to pass to each exchange that is part of the NBBO feed
        """
        cb = NBBO(callback, symbols)
        for feed in feeds:
            self.add_feed(feed(channels=[L2_BOOK], symbols=symbols, callbacks={L2_BOOK: cb}, config=config))

    def run(self, start_loop: bool = True, install_signal_handlers: bool = True, exception_handler=None):
        """
        start_loop: bool, default True
            if false, will not start the event loop.
        install_signal_handlers: bool, default True
            if True, will install the signal handlers on the event loop. This
            can only be done from the main thread's loop, so if running cryptofeed on
            a child thread, this must be set to false, and setup_signal_handlers must
            be called from the main/parent thread's event loop
        exception_handler: asyncio exception handler function pointer
            a custom exception handler for asyncio
        """
        self.running = True
        loop = asyncio.get_event_loop()
        # Good to enable when debugging or without code change: export PYTHONASYNCIODEBUG=1)
        # loop.set_debug(True)

        if install_signal_handlers:
            setup_signal_handlers(loop)

        for feed in self.feeds:
            feed.start(loop)

        if not start_loop:
            return

        try:
            if exception_handler:
                loop.set_exception_handler(exception_handler)
            loop.run_forever()
        except SystemExit:
            LOG.info('FH: System Exit received - shutting down')
        except Exception as why:
            LOG.exception('FH: Unhandled %r - shutting down', why)
        finally:
            self.stop(loop=loop)
            self.close(loop=loop)

        LOG.info('FH: leaving run()')

    def _stop(self, loop=None):
        self.running = False
        if not loop:
            loop = asyncio.get_event_loop()

        LOG.info('FH: shutdown connections handlers in feeds')
        for feed in self.feeds:
            feed.stop()

        if self.raw_data_collection:
            LOG.info('FH: shutting down raw data collection')
            self.raw_data_collection.stop()

        LOG.info('FH: create the tasks to properly shutdown the backends (to flush the local cache)')
        shutdown_tasks = []
        for feed in self.feeds:
            task = loop.create_task(feed.shutdown())
            try:
                task.set_name(f'shutdown_feed_{feed.id}')
            except AttributeError:
                # set_name only in 3.8+
                pass
            shutdown_tasks.append(task)

        LOG.info('FH: wait %s backend tasks until termination', len(shutdown_tasks))
        return shutdown_tasks

    async def stop_async(self, loop=None):
        shutdown_tasks = self._stop(loop=loop)
        await asyncio.gather(*shutdown_tasks)

    def stop(self, loop=None):
        shutdown_tasks = self._stop(loop=loop)
        loop.run_until_complete(asyncio.gather(*shutdown_tasks))

    def close(self, loop=None):
        """Stop the asynchronous generators and close the event loop."""
        if not loop:
            loop = asyncio.get_event_loop()

        LOG.info('FH: stop the AsyncIO loop')
        loop.stop()
        LOG.info('FH: run the AsyncIO event loop one last time')
        loop.run_forever()

        pending = asyncio.all_tasks(loop=loop)
        LOG.info('FH: cancel the %s pending tasks', len(pending))
        for task in pending:
            task.cancel()

        LOG.info('FH: run the pending tasks until complete')
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        LOG.info('FH: shutdown asynchronous generators')
        loop.run_until_complete(loop.shutdown_asyncgens())

        LOG.info('FH: close the AsyncIO loop')
        loop.close()

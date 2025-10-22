"""
Simple Proxy System MVP - START SMALL Implementation

Following engineering principles from CLAUDE.md:
- START SMALL: MVP functionality only
- FRs over NFRs: Core proxy support, deferred enterprise features  
- Pydantic v2: Type-safe configuration with validation
- YAGNI: No external managers, HA, monitoring until proven needed
- KISS: Simple ProxyInjector instead of complex resolver hierarchy
"""
from __future__ import annotations

import aiohttp
import websockets
import logging
from typing import Optional, Literal, Dict, Tuple, Callable
from urllib.parse import urlparse
from weakref import ref as weakref_ref, ReferenceType

from cryptofeed.proxy_config import (
    ConnectionProxies,
    ProxyConfig,
    ProxyPoolConfig,
    ProxySettings,
    ProxyUrlConfig,
)
from cryptofeed.proxy_pool import (
    LeastConnectionsSelector,
    ProxyPool,
    ProxySelector,
    RoundRobinSelector,
    RandomSelector,
    TCPHealthChecker,
    HealthCheckResult,
)
from pydantic import BaseModel, Field, ConfigDict


__all__ = [
    'ProxySettings',
    'ProxyConfig',
    'ProxyPoolConfig',
    'ProxyUrlConfig',
    'ConnectionProxies',
    'ProxyPool',
    'ProxySelector',
    'RoundRobinSelector',
    'RandomSelector',
    'LeastConnectionsSelector',
    'TCPHealthChecker',
    'HealthCheckResult',
    'HealthCheckConfig',
    'ProxyInjector',
    'get_proxy_injector',
    'init_proxy_system',
    'load_proxy_settings',
    'log_proxy_usage',
]


LOG = logging.getLogger('feedhandler')


class HealthCheckConfig(BaseModel):
    """Health check configuration for proxy pools."""
    model_config = ConfigDict(extra='forbid')
    
    enabled: bool = Field(default=True, description="Enable health checking")
    method: Literal['tcp', 'http', 'ping'] = Field(default='tcp', description="Health check method")
    interval_seconds: int = Field(default=30, ge=5, le=300, description="Check interval in seconds")
    timeout_seconds: int = Field(default=5, ge=1, le=30, description="Check timeout in seconds")
    retry_count: int = Field(default=3, ge=1, le=10, description="Number of retries on failure")







class ProxyInjector:
    """Simple proxy injection for HTTP and WebSocket connections."""
    
    def __init__(self, proxy_settings: ProxySettings):
        self.settings = proxy_settings
        self._pool_cache: Dict[int, Tuple[ReferenceType[ProxyConfig], ProxyPool]] = {}
        self._leased_proxies: Dict[Tuple[int, int], Tuple[ProxyPool, ProxyUrlConfig]] = {}
        self._lease_counter: int = 0

    def _next_lease_id(self) -> int:
        self._lease_counter += 1
        return self._lease_counter

    def _get_proxy_pool(self, proxy_config: ProxyConfig) -> ProxyPool:
        """Get or create a ProxyPool for the given proxy configuration."""
        key = id(proxy_config)
        cached = self._pool_cache.get(key)

        if cached:
            proxy_ref, pool = cached
            if proxy_ref() is proxy_config:
                return pool
            if proxy_ref() is None:
                self._pool_cache.pop(key, None)

        pool = ProxyPool(proxy_config.pool)

        def _cleanup(_ref):
            self._pool_cache.pop(key, None)

        self._pool_cache[key] = (weakref_ref(proxy_config, _cleanup), pool)
        return pool

    def lease_proxy(self, exchange_id: str, connection_type: Literal['http', 'websocket']) -> Tuple[Optional[str], Callable[[], None]]:
        proxy_config = self.settings.get_proxy(exchange_id, connection_type)
        if not proxy_config:
            return None, lambda: None
        return self._lease_from_config(proxy_config)

    def _lease_from_config(self, proxy_config: ProxyConfig) -> Tuple[Optional[str], Callable[[], None]]:
        if proxy_config.pool:
            pool = self._get_proxy_pool(proxy_config)
            selected_proxy = pool.select_proxy()
            lease_id = self._next_lease_id()
            lease_key = (id(proxy_config), lease_id)
            self._leased_proxies[lease_key] = (pool, selected_proxy)

            def release() -> None:
                entry = self._leased_proxies.pop(lease_key, None)
                if entry:
                    entry[0].release_proxy(entry[1])

            return selected_proxy.url, release

        return proxy_config.url, lambda: None

    def get_http_proxy_url(self, exchange_id: str) -> Optional[str]:
        """Get HTTP proxy URL for exchange if configured."""
        url, release = self.lease_proxy(exchange_id, 'http')
        release()
        return url
    
    def apply_http_proxy(self, session: aiohttp.ClientSession, exchange_id: str) -> None:
        """Apply HTTP proxy to aiohttp session if configured."""
        # Note: aiohttp proxy is set at ClientSession creation time, not after
        # This method is kept for interface compatibility
        # Use get_http_proxy_url() during session creation instead
        pass
    
    async def create_websocket_connection(self, url: str, exchange_id: str, **kwargs):
        """Create WebSocket connection with proxy if configured."""
        proxy_config = self.settings.get_proxy(exchange_id, 'websocket')

        if not proxy_config:
            return await websockets.connect(url, **kwargs)

        connect_kwargs = dict(kwargs)
        resolved_proxy_url, release = self._lease_from_config(proxy_config)

        if not resolved_proxy_url:
            return await websockets.connect(url, **kwargs)

        log_proxy_usage(transport='websocket', exchange_id=exchange_id, proxy_url=resolved_proxy_url)

        connect_kwargs.pop('proxy', None)

        parsed_destination = urlparse(url)
        if not parsed_destination.hostname:
            release()
            raise ValueError(f"Invalid WebSocket URL without hostname: {url}")

        dest_port = parsed_destination.port
        if dest_port is None:
            dest_port = 443 if parsed_destination.scheme == 'wss' else 80

        async def _create_proxied_socket():
            parsed_proxy = urlparse(resolved_proxy_url)
            if not parsed_proxy.hostname or not parsed_proxy.port:
                raise ValueError(f"Invalid proxy URL: {resolved_proxy_url}")

            scheme = (parsed_proxy.scheme or '').lower()

            try:
                from python_socks import ProxyType
                from python_socks.async_.asyncio import Proxy as AsyncProxy
            except ModuleNotFoundError as exc:
                raise ImportError("python-socks library required for proxy WebSocket support. Install with: pip install python-socks") from exc

            rdns = scheme.endswith('h') or scheme.endswith('4a')

            if scheme.startswith('socks5'):
                proxy_type = ProxyType.SOCKS5
            elif scheme.startswith('socks4'):
                proxy_type = ProxyType.SOCKS4
            elif scheme in ('http', 'https'):
                proxy_type = ProxyType.HTTP
            else:
                raise ValueError(f"Unsupported proxy scheme for WebSocket tunneling: {scheme}")

            proxy = AsyncProxy(
                proxy_type=proxy_type,
                host=parsed_proxy.hostname,
                port=parsed_proxy.port,
                username=parsed_proxy.username,
                password=parsed_proxy.password,
                rdns=rdns,
            )

            timeout = connect_kwargs.get('open_timeout')
            return await proxy.connect(
                dest_host=parsed_destination.hostname,
                dest_port=dest_port,
                timeout=timeout,
            )

        try:
            sock = await _create_proxied_socket()
        except Exception:
            release()
            raise

        try:
            connection = await websockets.connect(url, sock=sock, **connect_kwargs)
        except Exception:
            try:
                sock.close()
            finally:
                release()
            raise

        close_attr = getattr(connection, 'close', None)
        if callable(close_attr):

            async def _wrapped_close(*close_args, **close_kwargs):
                try:
                    return await close_attr(*close_args, **close_kwargs)
                finally:
                    release()

            connection.close = _wrapped_close  # type: ignore[attr-defined]
            return connection

        # Fallback for unexpected connection types
        release()
        return connection


# Global proxy injector instance (singleton pattern simplified)
_proxy_injector: Optional[ProxyInjector] = None


def get_proxy_injector() -> Optional[ProxyInjector]:
    """Get global proxy injector instance."""
    return _proxy_injector


def init_proxy_system(settings: ProxySettings) -> None:
    """Initialize proxy system with settings."""
    global _proxy_injector
    if not settings.enabled:
        _proxy_injector = None
        return
    _proxy_injector = ProxyInjector(settings)


def load_proxy_settings() -> ProxySettings:
    """Load proxy settings from environment or configuration."""
    return ProxySettings()


def _proxy_endpoint_components(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    host = parsed.hostname or ''
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return parsed.scheme, host


def log_proxy_usage(*, transport: str, exchange_id: Optional[str], proxy_url: str) -> None:
    scheme, endpoint = _proxy_endpoint_components(proxy_url)
    LOG.info("proxy: transport=%s exchange=%s scheme=%s endpoint=%s", transport, exchange_id or 'default', scheme, endpoint)

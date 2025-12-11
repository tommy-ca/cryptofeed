"""Unit tests for proxy/pool resolution in Binance E2E context.

Test scope: Task 6.1 - Validate proxy/pool resolution
- Single HTTP proxy configuration
- Single SOCKS proxy configuration
- Proxy pool configuration (multiple proxies)
- Pool selection returns valid proxy
- Direct mode (no proxy configuration)
- Coverage for both Binance spot and futures
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from cryptofeed.proxy import ProxySettings, init_proxy_system, get_proxy_injector, load_proxy_settings


class TestSingleProxyResolution:
    """Test single proxy configuration resolution."""

    def setup_method(self):
        """Clear proxy env vars before each test."""
        self.env_backup = {}
        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore environment after each test."""
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        for key, value in self.env_backup.items():
            os.environ[key] = value

        init_proxy_system(ProxySettings(enabled=False))

    def test_single_http_proxy_binance_spot(self):
        """Single HTTP proxy for Binance spot should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = "http://proxy.example.com:8080"

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # get_http_proxy_url should return configured proxy
        http_url = injector.get_http_proxy_url("binance")
        assert http_url == "http://proxy.example.com:8080"

    def test_single_http_proxy_binance_futures(self):
        """Single HTTP proxy for Binance futures should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL"] = "http://futures-proxy.example.com:8080"

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # get_http_proxy_url should return configured proxy
        http_url = injector.get_http_proxy_url("binance_futures")
        assert http_url == "http://futures-proxy.example.com:8080"

    def test_single_socks_proxy_binance_spot_ws(self):
        """Single SOCKS proxy for Binance spot WebSocket should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = "socks5://user:pass@proxy.example.com:1080"

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # lease_proxy should return configured SOCKS proxy
        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            parsed = urlparse(ws_url)
            assert parsed.scheme == "socks5"
            assert parsed.hostname == "proxy.example.com"
            assert parsed.port == 1080
        finally:
            if release:
                release()

    def test_single_socks_proxy_binance_futures_ws(self):
        """Single SOCKS proxy for Binance futures WebSocket should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL"] = "socks5://futures-ws-proxy.example.com:1080"

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # lease_proxy should return configured SOCKS proxy
        ws_url, release = injector.lease_proxy("binance_futures", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            parsed = urlparse(ws_url)
            assert parsed.scheme == "socks5"
            assert parsed.hostname == "futures-ws-proxy.example.com"
            assert parsed.port == 1080
        finally:
            if release:
                release()


class TestProxyPoolResolution:
    """Test proxy pool configuration and selection."""

    def setup_method(self):
        """Clear proxy env vars before each test."""
        self.env_backup = {}
        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore environment after each test."""
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        for key, value in self.env_backup.items():
            os.environ[key] = value

        init_proxy_system(ProxySettings(enabled=False))

    def test_http_pool_binance_spot(self):
        """HTTP proxy pool for Binance spot should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        pool_config = {
            "proxies": [
                {"url": "http://p1.example.com:8080", "weight": 1},
                {"url": "http://p2.example.com:8080", "weight": 1}
            ],
            "strategy": "round_robin"
        }
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL"] = json.dumps(pool_config)

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # Pool selection should return one of the configured proxies
        http_url = injector.get_http_proxy_url("binance")
        assert http_url is not None
        assert http_url.startswith("http://")
        # Should be one of the two configured proxies
        assert "p1.example.com:8080" in http_url or "p2.example.com:8080" in http_url

    def test_ws_pool_binance_futures(self):
        """WebSocket proxy pool for Binance futures should resolve correctly."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        pool_config = {
            "proxies": [
                {"url": "socks5://ws1.example.com:1080", "weight": 1},
                {"url": "socks5://ws2.example.com:1080", "weight": 1}
            ],
            "strategy": "round_robin"
        }
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL"] = json.dumps(pool_config)

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # Pool selection should return one of the configured proxies
        ws_url, release = injector.lease_proxy("binance_futures", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            # Should be one of the two configured proxies
            assert "ws1.example.com:1080" in ws_url or "ws2.example.com:1080" in ws_url
        finally:
            if release:
                release()

    def test_pool_selection_does_not_crash(self):
        """Pool selection should not crash with valid configuration."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        pool_config = {
            "proxies": [
                {"url": "http://stable1.example.com:8080", "weight": 2},
                {"url": "http://stable2.example.com:8080", "weight": 1}
            ],
            "strategy": "round_robin"
        }
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL"] = json.dumps(pool_config)

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        # Should not crash when selecting from pool multiple times
        for _ in range(10):
            http_url = injector.get_http_proxy_url("binance")
            assert http_url is not None
            assert http_url.startswith("http://")

    def test_pool_selection_returns_at_least_one_proxy(self):
        """Pool selection must return at least one proxy (not None)."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        pool_config = {
            "proxies": [
                {"url": "socks5://single.example.com:1080", "weight": 1}
            ],
            "strategy": "round_robin"
        }
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL"] = json.dumps(pool_config)

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            # Must return a proxy, not None
            assert ws_url is not None
            assert ws_url == "socks5://single.example.com:1080"
        finally:
            if release:
                release()


class TestDirectMode:
    """Test direct mode (no proxy configuration)."""

    def setup_method(self):
        """Clear proxy env vars before each test."""
        self.env_backup = {}
        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore environment after each test."""
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        for key, value in self.env_backup.items():
            os.environ[key] = value

        init_proxy_system(ProxySettings(enabled=False))

    def test_direct_mode_binance_spot(self):
        """When no proxy is configured for Binance spot, should return None."""
        # Ensure no proxy env vars
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key)

        settings = load_proxy_settings()
        assert not settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()

        # In direct mode, should return None
        if injector:
            http_url = injector.get_http_proxy_url("binance")
            assert http_url is None

            ws_url, release = injector.lease_proxy("binance", "websocket")
            try:
                assert ws_url is None or release is None
            finally:
                if release:
                    release()

    def test_direct_mode_binance_futures(self):
        """When no proxy is configured for Binance futures, should return None."""
        # Ensure no proxy env vars
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key)

        settings = load_proxy_settings()
        assert not settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()

        # In direct mode, should return None
        if injector:
            http_url = injector.get_http_proxy_url("binance_futures")
            assert http_url is None

            ws_url, release = injector.lease_proxy("binance_futures", "websocket")
            try:
                assert ws_url is None or release is None
            finally:
                if release:
                    release()

    def test_no_regression_when_proxy_disabled(self):
        """When proxy is explicitly disabled, prior assertions should remain unchanged."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "false"

        settings = load_proxy_settings()
        assert not settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()

        # Should behave like direct mode
        if injector:
            for exchange in ["binance", "binance_futures"]:
                http_url = injector.get_http_proxy_url(exchange)
                assert http_url is None

                ws_url, release = injector.lease_proxy(exchange, "websocket")
                try:
                    assert ws_url is None or release is None
                finally:
                    if release:
                        release()


class TestBothProtocolsCoverage:
    """Test both HTTP and WebSocket proxy resolution for completeness."""

    def setup_method(self):
        """Clear proxy env vars before each test."""
        self.env_backup = {}
        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore environment after each test."""
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        for key, value in self.env_backup.items():
            os.environ[key] = value

        init_proxy_system(ProxySettings(enabled=False))

    def test_both_http_and_ws_binance_spot(self):
        """Both HTTP and WS proxies should resolve for Binance spot."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = "http://http-proxy.example.com:8080"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = "socks5://ws-proxy.example.com:1080"

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        # HTTP proxy resolution
        http_url = injector.get_http_proxy_url("binance")
        assert http_url == "http://http-proxy.example.com:8080"

        # WebSocket proxy resolution
        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            assert "ws-proxy.example.com:1080" in ws_url
        finally:
            if release:
                release()

    def test_both_http_and_ws_binance_futures(self):
        """Both HTTP and WS proxies should resolve for Binance futures."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL"] = "http://futures-http.example.com:8080"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL"] = "socks5://futures-ws.example.com:1080"

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        # HTTP proxy resolution
        http_url = injector.get_http_proxy_url("binance_futures")
        assert http_url == "http://futures-http.example.com:8080"

        # WebSocket proxy resolution
        ws_url, release = injector.lease_proxy("binance_futures", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            assert "futures-ws.example.com:1080" in ws_url
        finally:
            if release:
                release()

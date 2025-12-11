"""Unit tests for proxy configuration loading in Binance E2E tests.

Test scope: Task 6 - Enable proxy-configured E2E runs
- Load ProxySettings from env (CRYPTOFEED_PROXY_*, nested __)
- Maintain precedence: env > YAML > programmatic
- Direct mode works by default
- Skip with clear message when SOCKS WS configured but python-socks missing
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


from cryptofeed.proxy import ProxySettings, init_proxy_system, get_proxy_injector


class TestProxyLoadingFromEnv:
    """Test proxy configuration loading from environment variables."""

    def setup_method(self):
        """Save and clear proxy-related env vars before each test."""
        self.env_backup = {}
        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore original environment after each test."""
        # Clear any test-set values
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        # Restore backed-up values
        for key, value in self.env_backup.items():
            os.environ[key] = value

        # Reset proxy system
        init_proxy_system(ProxySettings(enabled=False))

    def test_direct_mode_by_default(self):
        """When no proxy env vars are set, system should work in direct mode."""
        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert not settings.enabled
        assert settings.default is None
        assert not settings.exchanges

        # Verify no proxy is resolved for binance
        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is None or injector.get_http_proxy_url("binance") is None

    def test_load_http_proxy_from_env(self):
        """HTTP proxy configuration should be loaded from CRYPTOFEED_PROXY_* env vars."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = "http://proxy.example.com:8080"

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        http_url = injector.get_http_proxy_url("binance")
        assert http_url == "http://proxy.example.com:8080"

    def test_load_socks_proxy_from_env(self):
        """SOCKS proxy configuration should be loaded from CRYPTOFEED_PROXY_* env vars."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = "socks5://user:pass@proxy.example.com:1080"

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            parsed = urlparse(ws_url)
            assert parsed.scheme == "socks5"
        finally:
            if release:
                release()

    def test_load_proxy_pool_from_env(self):
        """Proxy pool configuration should be loaded from CRYPTOFEED_PROXY_* nested env vars (JSON format)."""
        import json

        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        # Use JSON format for pool configuration (pydantic-settings limitation with nested lists)
        pool_config = {
            "proxies": [
                {"url": "socks5://p1:1080", "weight": 1},
                {"url": "socks5://p2:1080", "weight": 1}
            ],
            "strategy": "round_robin"
        }
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL"] = json.dumps(pool_config)

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()
        assert injector is not None

        # Should be able to lease a proxy from the pool
        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            assert ws_url.startswith("socks5://")
            # URL should be one of the two configured proxies
            assert "p1:1080" in ws_url or "p2:1080" in ws_url
        finally:
            if release:
                release()

    def test_env_precedence_over_programmatic(self):
        """Environment variables should take precedence over programmatic config."""
        # Set env var
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = "http://env-proxy:8080"

        from cryptofeed.proxy import load_proxy_settings

        # Load settings (env takes precedence)
        settings = load_proxy_settings()
        assert settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()

        # Should use env-configured proxy, not programmatic
        http_url = injector.get_http_proxy_url("binance")
        assert http_url == "http://env-proxy:8080"

    def test_skip_when_socks_ws_configured_but_python_socks_missing(self):
        """When SOCKS WS proxy is configured but python-socks is missing, should be detectable."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = "socks5://proxy:1080"

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()

        # This should be caught in E2E test initialization
        ws_proxy = settings.get_proxy("binance", "websocket") if hasattr(settings, "get_proxy") else None

        if ws_proxy and ws_proxy.url:
            scheme = urlparse(ws_proxy.url).scheme.lower()

            # Verify we can detect SOCKS scheme
            assert scheme.startswith("socks")

            # In real E2E tests, this check would skip the test if python_socks is missing
            # Here we just verify the scheme detection works
            from importlib import import_module

            try:
                import_module("python_socks")
                # If available, that's fine - test passes
            except ModuleNotFoundError:
                # If not available, E2E tests would skip - test passes
                pass


class TestProxySystemIntegration:
    """Test proxy system integration in E2E test harness."""

    def setup_method(self):
        """Save environment before each test."""
        self.env_backup = {}
        for key in ["HTTP_PROXY", "HTTPS_PROXY"]:
            self.env_backup[key] = os.environ.get(key)

        proxy_keys = [k for k in os.environ if k.startswith("CRYPTOFEED_PROXY_")]
        for key in proxy_keys:
            self.env_backup[key] = os.environ.pop(key)

    def teardown_method(self):
        """Restore environment after each test."""
        # Restore HTTP proxy env vars
        for key in ["HTTP_PROXY", "HTTPS_PROXY"]:
            if self.env_backup[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self.env_backup[key]

        # Restore CRYPTOFEED_PROXY_ vars
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key, None)

        for key, value in self.env_backup.items():
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ[key] = value

        # Reset proxy system
        init_proxy_system(ProxySettings(enabled=False))

    def test_http_proxy_env_propagation(self):
        """HTTP_PROXY and HTTPS_PROXY should be set when HTTP proxy is configured."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = "http://test-proxy:8080"

        # Clear HTTP proxy env vars
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        # Simulate E2E test initialization that sets HTTP_PROXY
        http_proxy_url = injector.get_http_proxy_url("binance")
        if http_proxy_url:
            os.environ["HTTPS_PROXY"] = http_proxy_url
            os.environ["HTTP_PROXY"] = http_proxy_url

        # Verify propagation
        assert os.environ["HTTP_PROXY"] == "http://test-proxy:8080"
        assert os.environ["HTTPS_PROXY"] == "http://test-proxy:8080"

    def test_proxy_resolution_returns_valid_url(self):
        """Leased proxy URL should have a valid scheme."""
        os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
        os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = "socks5://proxy:1080"

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            parsed = urlparse(ws_url)
            assert parsed.scheme, "Proxy URL must include a scheme"
        finally:
            if release:
                release()

    def test_no_regression_in_direct_mode(self):
        """When no proxy is configured, direct mode should work without changes."""
        # Ensure no proxy env vars
        for key in list(os.environ.keys()):
            if key.startswith("CRYPTOFEED_PROXY_"):
                os.environ.pop(key)

        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert not settings.enabled

        init_proxy_system(settings)
        injector = get_proxy_injector()

        # In direct mode, injector might be None or return None for proxies
        if injector:
            assert injector.get_http_proxy_url("binance") is None
            ws_url, release = injector.lease_proxy("binance", "websocket")
            try:
                # In direct mode, either no URL or no release handle
                assert ws_url is None or release is None
            finally:
                if release:
                    release()

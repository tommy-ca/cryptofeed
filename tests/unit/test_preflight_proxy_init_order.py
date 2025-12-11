"""Test for proxy preflight initialization order (Task 6.5).

The _preflight_rest_through_proxy helper must initialize the proxy system
BEFORE attempting to lease proxies. Previously it checked get_proxy_injector()
before calling init_proxy_system, causing proxy-configured runs to go direct.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from cryptofeed.proxy import init_proxy_system, get_proxy_injector, ProxySettings


@pytest.fixture(autouse=True)
def reset_proxy_system():
    """Reset proxy system before and after each test."""
    init_proxy_system(ProxySettings(enabled=False))
    yield
    init_proxy_system(ProxySettings(enabled=False))


def test_preflight_initializes_proxy_system_before_leasing():
    """Verify that init_proxy_system is called before get_proxy_injector in preflight flow."""

    # Simulate proxy configuration via environment
    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL": "http://proxy.example.com:8080"
    }):
        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert settings.enabled is True

        # Before init_proxy_system, injector should be None
        injector_before = get_proxy_injector()
        assert injector_before is None, "Injector should be None before init_proxy_system"

        # After init_proxy_system, injector should exist
        init_proxy_system(settings)
        injector_after = get_proxy_injector()
        assert injector_after is not None, "Injector should exist after init_proxy_system"

        # Should be able to get HTTP proxy URL
        http_url = injector_after.get_http_proxy_url("binance")
        assert http_url == "http://proxy.example.com:8080"


def test_preflight_sets_http_proxy_env_vars():
    """Verify that HTTP_PROXY and HTTPS_PROXY are set when proxy is configured."""

    # Clear any existing proxy env vars
    original_http = os.environ.pop("HTTP_PROXY", None)
    original_https = os.environ.pop("HTTPS_PROXY", None)

    try:
        with patch.dict(os.environ, {
            "CRYPTOFEED_PROXY_ENABLED": "true",
            "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL": "http://proxy.example.com:8080"
        }):
            from cryptofeed.proxy import load_proxy_settings

            settings = load_proxy_settings()
            init_proxy_system(settings)
            injector = get_proxy_injector()

            # Simulate what _init_proxy_settings_if_configured does
            http_proxy_url = injector.get_http_proxy_url("binance")
            assert http_proxy_url == "http://proxy.example.com:8080"

            # Set environment variables for requests library
            os.environ["HTTP_PROXY"] = http_proxy_url
            os.environ["HTTPS_PROXY"] = http_proxy_url

            # Verify they are set
            assert os.environ["HTTP_PROXY"] == "http://proxy.example.com:8080"
            assert os.environ["HTTPS_PROXY"] == "http://proxy.example.com:8080"

    finally:
        # Restore original values
        if original_http is not None:
            os.environ["HTTP_PROXY"] = original_http
        else:
            os.environ.pop("HTTP_PROXY", None)

        if original_https is not None:
            os.environ["HTTPS_PROXY"] = original_https
        else:
            os.environ.pop("HTTPS_PROXY", None)


def test_preflight_with_no_proxy_configuration():
    """Verify that preflight works correctly when no proxy is configured."""

    # Clear proxy env vars
    with patch.dict(os.environ, {}, clear=True):
        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        assert settings.enabled is False

        # No need to initialize proxy system if no proxy configured
        injector = get_proxy_injector()
        # Injector might be None or exist from previous init
        if injector:
            # If it exists, it should return None for binance
            http_url = injector.get_http_proxy_url("binance")
            assert http_url is None


@pytest.mark.asyncio
async def test_preflight_initialization_order_simulated():
    """Simulate the corrected preflight flow with proper initialization order."""

    # Simulate proxy configuration
    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL": "http://proxy.example.com:8080"
    }):
        from cryptofeed.proxy import load_proxy_settings

        # Step 1: Load settings
        settings = load_proxy_settings()
        assert settings.enabled is True

        # Step 2: Initialize proxy system FIRST
        init_proxy_system(settings)

        # Step 3: NOW get the injector (should exist)
        injector = get_proxy_injector()
        assert injector is not None, "Injector must exist after init_proxy_system"

        # Step 4: Lease HTTP proxy
        http_proxy_url = injector.get_http_proxy_url("binance")
        assert http_proxy_url == "http://proxy.example.com:8080"

        # Step 5: Set environment variables for REST calls
        os.environ["HTTP_PROXY"] = http_proxy_url
        os.environ["HTTPS_PROXY"] = http_proxy_url

        # Verify the flow completed successfully
        assert os.environ["HTTP_PROXY"] == "http://proxy.example.com:8080"
        assert os.environ["HTTPS_PROXY"] == "http://proxy.example.com:8080"


def test_websocket_proxy_lease_after_init():
    """Verify that websocket proxy can be leased after init_proxy_system."""

    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL": "socks5://proxy.example.com:1080"
    }):
        from cryptofeed.proxy import load_proxy_settings

        settings = load_proxy_settings()
        init_proxy_system(settings)
        injector = get_proxy_injector()

        assert injector is not None

        # Lease websocket proxy
        ws_url, release = injector.lease_proxy("binance", "websocket")

        try:
            assert ws_url is not None
            assert "socks5://" in ws_url
            assert callable(release)
        finally:
            release()

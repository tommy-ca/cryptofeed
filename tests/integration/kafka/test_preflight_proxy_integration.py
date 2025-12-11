"""Integration test for preflight proxy initialization (Task 6.5).

Verifies that _preflight_rest_through_proxy correctly initializes the proxy
system before attempting to lease proxies, ensuring proxy-configured runs
use the proxy correctly.
"""

import os
import pytest
from unittest.mock import patch
from cryptofeed.proxy import init_proxy_system, ProxySettings


@pytest.fixture(autouse=True)
def reset_proxy_system():
    """Reset proxy system before and after each test."""
    init_proxy_system(ProxySettings(enabled=False))
    yield
    init_proxy_system(ProxySettings(enabled=False))


@pytest.mark.asyncio
async def test_preflight_with_proxy_configured():
    """Test that preflight correctly initializes proxy system when proxy is configured."""

    # Set up proxy configuration
    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL": "http://proxy.example.com:8080",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL": "socks5://proxy.example.com:1080",
    }):
        # Import the test module functions
        from tests.integration.kafka.test_binance_kafka_protobuf_pipeline import (
            _init_proxy_settings_if_configured,
        )
        from cryptofeed.proxy import get_proxy_injector

        # Initialize proxy settings
        has_proxy = _init_proxy_settings_if_configured()
        assert has_proxy is True, "Should have proxy configuration"

        # Verify injector exists after initialization
        injector = get_proxy_injector()
        assert injector is not None, "Injector should exist after init_proxy_settings_if_configured"

        # Verify we can get HTTP proxy URL
        http_url = injector.get_http_proxy_url("binance")
        assert http_url == "http://proxy.example.com:8080"

        # Verify HTTP[S]_PROXY environment variables are set
        assert os.environ.get("HTTP_PROXY") == "http://proxy.example.com:8080"
        assert os.environ.get("HTTPS_PROXY") == "http://proxy.example.com:8080"


@pytest.mark.asyncio
async def test_preflight_without_proxy_configured():
    """Test that preflight works correctly when no proxy is configured."""

    # Clear all proxy-related environment variables
    proxy_env_vars = [k for k in os.environ.keys() if k.startswith("CRYPTOFEED_PROXY_")]
    with patch.dict(os.environ, {k: "" for k in proxy_env_vars}, clear=False):
        # Import the test module functions
        from tests.integration.kafka.test_binance_kafka_protobuf_pipeline import (
            _init_proxy_settings_if_configured,
        )
        from cryptofeed.proxy import get_proxy_injector

        # Initialize (should detect no proxy)
        has_proxy = _init_proxy_settings_if_configured()
        assert has_proxy is False, "Should not have proxy configuration"

        # Injector might exist from previous tests but should return None for binance
        injector = get_proxy_injector()
        if injector:
            http_url = injector.get_http_proxy_url("binance")
            assert http_url is None, "Should not have HTTP proxy for binance"


@pytest.mark.asyncio
async def test_preflight_futures_with_proxy():
    """Test that futures preflight correctly initializes proxy system."""

    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL": "http://proxy.example.com:8080",
    }):
        # Import the futures test module functions
        from tests.integration.kafka.test_binance_futures_kafka_protobuf_pipeline import (
            _init_proxy_settings_if_configured,
        )
        from cryptofeed.proxy import get_proxy_injector

        # Initialize proxy settings
        has_proxy = _init_proxy_settings_if_configured()
        assert has_proxy is True

        # Verify injector exists
        injector = get_proxy_injector()
        assert injector is not None

        # Verify we can get HTTP proxy URL for binance_futures
        http_url = injector.get_http_proxy_url("binance_futures")
        assert http_url == "http://proxy.example.com:8080"


@pytest.mark.asyncio
async def test_preflight_initialization_order_prevents_none_injector():
    """Verify that the fix prevents getting None injector when proxy is configured."""

    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL": "http://proxy.example.com:8080",
    }):
        from tests.integration.kafka.test_binance_kafka_protobuf_pipeline import (
            _init_proxy_settings_if_configured,
        )
        from cryptofeed.proxy import get_proxy_injector, load_proxy_settings

        # Load settings to verify proxy is configured
        settings = load_proxy_settings()
        assert settings.enabled is True

        # Before calling _init_proxy_settings_if_configured, injector should be None
        # (assuming clean state from fixture)
        injector_before = get_proxy_injector()
        assert injector_before is None or injector_before.get_http_proxy_url("binance") is None

        # After calling _init_proxy_settings_if_configured, injector should exist
        has_proxy = _init_proxy_settings_if_configured()
        assert has_proxy is True

        # Now injector should exist and return proxy URL
        injector_after = get_proxy_injector()
        assert injector_after is not None
        http_url = injector_after.get_http_proxy_url("binance")
        assert http_url == "http://proxy.example.com:8080"


@pytest.mark.asyncio
async def test_preflight_with_pool_configuration():
    """Test that preflight works with proxy pool configuration using JSON format."""

    import json

    # Use JSON format for pool configuration (Pydantic-friendly)
    pool_config = json.dumps({
        "proxies": [
            {"url": "socks5://p1.example.com:1080", "weight": 1},
            {"url": "socks5://p2.example.com:1080", "weight": 1}
        ],
        "strategy": "round_robin"
    })

    with patch.dict(os.environ, {
        "CRYPTOFEED_PROXY_ENABLED": "true",
        "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL": pool_config,
    }):
        from tests.integration.kafka.test_binance_kafka_protobuf_pipeline import (
            _init_proxy_settings_if_configured,
        )
        from cryptofeed.proxy import get_proxy_injector

        # Initialize proxy settings
        has_proxy = _init_proxy_settings_if_configured()
        assert has_proxy is True

        # Verify injector exists
        injector = get_proxy_injector()
        assert injector is not None

        # Verify we can lease a proxy from the pool
        ws_url, release = injector.lease_proxy("binance", "websocket")
        try:
            assert ws_url is not None
            assert "socks5://" in ws_url
            assert callable(release)
        finally:
            release()

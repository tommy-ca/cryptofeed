"""Unit tests for schema registry client proxy and timeout support.

Task 6.8b: Schema registry client migration to aiohttp + ProxyInjector
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    ConfluentSchemaRegistry,
)


def create_mock_aiohttp_response(status=200, json_data=None):
    """Helper to create properly mocked aiohttp response with context managers."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data or {})
    mock_response.raise_for_status = MagicMock()

    # Create async context manager for session.request()
    mock_request_context = AsyncMock()
    mock_request_context.__aenter__.return_value = mock_response
    mock_request_context.__aexit__.return_value = None

    return mock_request_context


def create_mock_aiohttp_session(response_context):
    """Helper to create properly mocked aiohttp ClientSession."""
    mock_session = AsyncMock()
    mock_session.request = MagicMock(return_value=response_context)

    # Create async context manager for ClientSession()
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__.return_value = mock_session
    mock_session_context.__aexit__.return_value = None

    return mock_session_context, mock_session


@pytest.fixture
def registry_config():
    """Confluent schema registry config for testing."""
    return SchemaRegistryConfig(
        registry_type="confluent",
        url="http://localhost:8081",
        username="test_user",
        password="test_password",
    )


@pytest.mark.asyncio
async def test_register_schema_with_http_proxy(registry_config):
    """Test schema registration with HTTP proxy."""
    registry = ConfluentSchemaRegistry(registry_config)

    # Mock ProxyInjector to return HTTP proxy
    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector_instance = MagicMock()
        mock_injector_instance.get_http_proxy_url.return_value = "http://proxy.example.com:8080"
        mock_injector.return_value = mock_injector_instance

        # Mock aiohttp session
        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            response_context = create_mock_aiohttp_response(200, {"id": 123})
            session_context, mock_session = create_mock_aiohttp_session(response_context)
            mock_session_class.return_value = session_context

            # Execute registration
            schema_id = await registry.register_schema_async(
                subject="test-subject",
                schema="syntax = \"proto3\";",
                schema_type="PROTOBUF"
            )

            # Verify schema ID returned
            assert schema_id == 123

            # Verify proxy was requested
            mock_injector_instance.get_http_proxy_url.assert_called_once_with("schema_registry")

            # Verify request was called
            assert mock_session.request.called
            request_call = mock_session.request.call_args
            assert request_call[0][0] == "POST"
            assert request_call[0][1].endswith("/subjects/test-subject/versions")


@pytest.mark.asyncio
async def test_register_schema_with_socks_proxy(registry_config):
    """Test schema registration with SOCKS5 proxy."""
    registry = ConfluentSchemaRegistry(registry_config)

    # Mock ProxyInjector to return SOCKS proxy
    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector_instance = MagicMock()
        mock_injector_instance.get_http_proxy_url.return_value = "socks5://proxy.example.com:1080"
        mock_injector.return_value = mock_injector_instance

        # Mock ProxyConnector
        with patch("cryptofeed.backends.kafka_schema.ProxyConnector") as mock_connector_class:
            mock_connector = MagicMock()
            mock_connector_class.from_url.return_value = mock_connector

            # Mock aiohttp session
            with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
                response_context = create_mock_aiohttp_response(200, {"id": 456})
                session_context, mock_session = create_mock_aiohttp_session(response_context)
                mock_session_class.return_value = session_context

                # Execute registration
                schema_id = await registry.register_schema_async(
                    subject="test-subject",
                    schema="syntax = \"proto3\";",
                    schema_type="PROTOBUF"
                )

                # Verify schema ID returned
                assert schema_id == 456

                # Verify SOCKS connector was created
                mock_connector_class.from_url.assert_called_once_with("socks5://proxy.example.com:1080")

                # Verify session was created with connector
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs.get("connector") == mock_connector


@pytest.mark.asyncio
async def test_get_schema_with_timeout_enforcement(registry_config, monkeypatch):
    """Test schema retrieval with custom timeout."""
    # Set custom timeout via environment variable
    monkeypatch.setenv("CF_SCHEMA_REGISTRY_TIMEOUT", "5")

    registry = ConfluentSchemaRegistry(registry_config)

    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector.return_value = None  # No proxy

        # Mock aiohttp session with timeout
        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            with patch("cryptofeed.backends.kafka_schema.ClientTimeout") as mock_timeout_class:
                mock_timeout = MagicMock()
                mock_timeout_class.return_value = mock_timeout

                response_context = create_mock_aiohttp_response(200, {
                    "schema": "syntax = \"proto3\";",
                    "schemaType": "PROTOBUF"
                })
                session_context, mock_session = create_mock_aiohttp_session(response_context)
                mock_session_class.return_value = session_context

                # Execute get schema
                result = await registry.get_schema_by_id_async(123)

                # Verify timeout was set to 5 seconds
                mock_timeout_class.assert_called_once_with(total=5.0)

                # Verify result
                assert result["schema"] == "syntax = \"proto3\";"


@pytest.mark.asyncio
async def test_direct_mode_no_proxy_regression(registry_config):
    """Test schema operations work without proxy configuration (direct mode)."""
    registry = ConfluentSchemaRegistry(registry_config)

    # Mock ProxyInjector returns None (no proxy configured)
    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector.return_value = None

        # Mock aiohttp session
        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            response_context = create_mock_aiohttp_response(200, {"id": 789})
            session_context, mock_session = create_mock_aiohttp_session(response_context)
            mock_session_class.return_value = session_context

            # Execute registration without proxy
            schema_id = await registry.register_schema_async(
                subject="direct-test",
                schema="syntax = \"proto3\";",
                schema_type="PROTOBUF"
            )

            # Verify schema ID returned
            assert schema_id == 789

            # Verify session was created without connector
            call_kwargs = mock_session_class.call_args[1]
            assert call_kwargs.get("connector") is None


@pytest.mark.asyncio
async def test_auth_header_preservation_with_proxy(registry_config):
    """Test HTTPBasicAuth is preserved when using proxy."""
    registry = ConfluentSchemaRegistry(registry_config)

    # Mock ProxyInjector to return HTTP proxy
    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector_instance = MagicMock()
        mock_injector_instance.get_http_proxy_url.return_value = "http://proxy.example.com:8080"
        mock_injector.return_value = mock_injector_instance

        # Mock aiohttp with BasicAuth
        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            with patch("cryptofeed.backends.kafka_schema.BasicAuth") as mock_auth_class:
                mock_auth = MagicMock()
                mock_auth_class.return_value = mock_auth

                response_context = create_mock_aiohttp_response(200, {"is_compatible": True})
                session_context, mock_session = create_mock_aiohttp_session(response_context)
                mock_session_class.return_value = session_context

                # Execute compatibility check
                is_compatible = await registry.check_compatibility_async(
                    subject="auth-test",
                    schema="syntax = \"proto3\";"
                )

                # Verify BasicAuth was created with credentials
                mock_auth_class.assert_called_once_with("test_user", "test_password")

                # Verify session was created with auth
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs.get("auth") == mock_auth

                # Verify result
                assert is_compatible is True


@pytest.mark.asyncio
async def test_compatibility_check_with_proxy(registry_config):
    """Test compatibility check with HTTP proxy."""
    registry = ConfluentSchemaRegistry(registry_config)

    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector_instance = MagicMock()
        mock_injector_instance.get_http_proxy_url.return_value = "http://proxy.example.com:8080"
        mock_injector.return_value = mock_injector_instance

        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            response_context = create_mock_aiohttp_response(200, {"is_compatible": True})
            session_context, mock_session = create_mock_aiohttp_session(response_context)
            mock_session_class.return_value = session_context

            # Execute compatibility check
            result = await registry.check_compatibility_async(
                subject="compat-test",
                schema="syntax = \"proto3\";"
            )

            # Verify compatibility result
            assert result is True

            # Verify proxy was used
            mock_injector_instance.get_http_proxy_url.assert_called_once()


@pytest.mark.asyncio
async def test_set_compatibility_mode_with_proxy(registry_config):
    """Test set compatibility mode with proxy."""
    registry = ConfluentSchemaRegistry(registry_config)

    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector_instance = MagicMock()
        mock_injector_instance.get_http_proxy_url.return_value = "http://proxy.example.com:8080"
        mock_injector.return_value = mock_injector_instance

        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            response_context = create_mock_aiohttp_response(200, {"compatibility": "BACKWARD"})
            session_context, mock_session = create_mock_aiohttp_session(response_context)
            mock_session_class.return_value = session_context

            # Execute set compatibility mode
            await registry.set_compatibility_mode_async(
                subject="mode-test",
                mode="BACKWARD"
            )

            # Verify PUT was called
            assert mock_session.request.called
            request_call = mock_session.request.call_args
            assert request_call[0][0] == "PUT"
            assert request_call[0][1].endswith("/config/mode-test")


@pytest.mark.asyncio
async def test_default_timeout_when_not_configured(registry_config, monkeypatch):
    """Test default 10s timeout is used when CF_SCHEMA_REGISTRY_TIMEOUT not set."""
    # Ensure env var is not set
    monkeypatch.delenv("CF_SCHEMA_REGISTRY_TIMEOUT", raising=False)
    monkeypatch.delenv("CRYPTOFEED_SCHEMA_REGISTRY_TIMEOUT", raising=False)

    registry = ConfluentSchemaRegistry(registry_config)

    with patch("cryptofeed.backends.kafka_schema.get_proxy_injector") as mock_injector:
        mock_injector.return_value = None

        with patch("cryptofeed.backends.kafka_schema.ClientSession") as mock_session_class:
            with patch("cryptofeed.backends.kafka_schema.ClientTimeout") as mock_timeout_class:
                mock_timeout = MagicMock()
                mock_timeout_class.return_value = mock_timeout

                response_context = create_mock_aiohttp_response(200, {"id": 999})
                session_context, mock_session = create_mock_aiohttp_session(response_context)
                mock_session_class.return_value = session_context

                # Execute registration
                await registry.register_schema_async(
                    subject="default-timeout-test",
                    schema="syntax = \"proto3\";",
                    schema_type="PROTOBUF"
                )

                # Verify default 10s timeout was used
                mock_timeout_class.assert_called_once_with(total=10.0)

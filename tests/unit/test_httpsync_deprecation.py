"""
Unit tests for HTTPSync deprecation and proxy/timeout support.

Task 6.8c: HTTPSync deprecation/migration
- Verify HTTPSync.read/write support proxy+timeout via env vars
- Verify deprecation warnings are emitted
- Ensure regression tests for proxy application
"""
import os
import warnings
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from cryptofeed.connection import HTTPSync


class TestHTTPSyncProxySupport:
    """Test that HTTPSync.read/write support proxy and timeout configuration."""

    def test_httpsync_read_with_http_proxy(self):
        """HTTPSync.read should respect HTTP_PROXY environment variable."""
        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://proxy.example.com:8080",
            "CRYPTOFEED_HTTP_TIMEOUT": "5"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                # Mock the async context manager chain
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(return_value='{"result": "ok"}')
                mock_resp.raise_for_status = MagicMock()  # Will be called by _Resp wrapper

                mock_get = AsyncMock()
                mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_get.__aexit__ = AsyncMock()

                mock_session = MagicMock()
                mock_session.get = MagicMock(return_value=mock_get)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()

                mock_session_class.return_value = mock_session

                http_sync = HTTPSync()
                result = http_sync.read("https://api.example.com/endpoint", json=True)

                assert result is not None
                # Verify session was created with timeout
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs["timeout"].total == 5.0

                # Verify get was called with proxy
                mock_session.get.assert_called_once()
                get_call_kwargs = mock_session.get.call_args[1]
                assert get_call_kwargs["proxy"] == "http://proxy.example.com:8080"

                # Verify raise_for_status was called on the response (via _Resp wrapper)
                # The actual call happens on line 76 in connection.py within _Resp
                assert mock_resp.raise_for_status.called

    def test_httpsync_read_with_socks_proxy(self):
        """HTTPSync.read should support SOCKS proxy via aiohttp_socks."""
        with patch.dict(os.environ, {
            "HTTP_PROXY": "socks5://proxy.example.com:1080",
            "CRYPTOFEED_HTTP_TIMEOUT": "10"
        }):
            with patch("aiohttp_socks.ProxyConnector") as mock_connector_class:
                with patch("aiohttp.ClientSession") as mock_session_class:
                    # Mock connector
                    mock_connector = MagicMock()
                    mock_connector_class.from_url.return_value = mock_connector

                    # Mock the async context manager chain
                    mock_resp = MagicMock()
                    mock_resp.status = 200
                    mock_resp.text = AsyncMock(return_value='{"result": "ok"}')
                    mock_resp.raise_for_status = MagicMock()

                    mock_get = AsyncMock()
                    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
                    mock_get.__aexit__ = AsyncMock()

                    mock_session = MagicMock()
                    mock_session.get = MagicMock(return_value=mock_get)
                    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session.__aexit__ = AsyncMock()

                    mock_session_class.return_value = mock_session

                    http_sync = HTTPSync()
                    result = http_sync.read("https://api.example.com/endpoint", json=True)

                    assert result is not None
                    # Verify ProxyConnector was created
                    mock_connector_class.from_url.assert_called_once_with("socks5://proxy.example.com:1080")

                    # Verify session was created with connector
                    mock_session_class.assert_called_once()
                    call_kwargs = mock_session_class.call_args[1]
                    assert call_kwargs["connector"] == mock_connector
                    assert call_kwargs["timeout"].total == 10.0

                    # Verify get was called without proxy kwarg (using connector instead)
                    mock_session.get.assert_called_once()
                    get_call_kwargs = mock_session.get.call_args[1]
                    assert get_call_kwargs["proxy"] is None

                    # Verify response was processed correctly
                    assert mock_resp.raise_for_status.called

    def test_httpsync_read_direct_mode_no_proxy(self):
        """HTTPSync.read should work in direct mode when no proxy configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Set only timeout
            os.environ["CRYPTOFEED_HTTP_TIMEOUT"] = "10"

            with patch("aiohttp.ClientSession") as mock_session_class:
                # Mock the async context manager chain
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(return_value='{"result": "ok"}')
                mock_resp.raise_for_status = MagicMock()

                mock_get = AsyncMock()
                mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_get.__aexit__ = AsyncMock()

                mock_session = MagicMock()
                mock_session.get = MagicMock(return_value=mock_get)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()

                mock_session_class.return_value = mock_session

                http_sync = HTTPSync()
                result = http_sync.read("https://api.example.com/endpoint", json=True)

                assert result is not None
                # Verify session created without connector
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs.get("connector") is None

                # Verify get called without proxy
                mock_session.get.assert_called_once()
                get_call_kwargs = mock_session.get.call_args[1]
                assert get_call_kwargs["proxy"] is None

                # Verify response was processed
                assert mock_resp.raise_for_status.called

    def test_httpsync_write_with_proxy(self):
        """HTTPSync.write should respect proxy configuration."""
        with patch.dict(os.environ, {
            "HTTPS_PROXY": "http://proxy.example.com:8080",
            "CF_HTTP_TIMEOUT": "8"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                # Mock the async context manager chain
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(return_value='{"result": "created"}')
                mock_resp.raise_for_status = MagicMock()

                mock_post = AsyncMock()
                mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_post.__aexit__ = AsyncMock()

                mock_session = MagicMock()
                mock_session.post = MagicMock(return_value=mock_post)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()

                mock_session_class.return_value = mock_session

                http_sync = HTTPSync()
                result = http_sync.write(
                    "https://api.example.com/endpoint",
                    data={"key": "value"},
                    is_data_json=True,
                    json=True
                )

                assert result is not None
                # Verify timeout applied
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs["timeout"].total == 8.0

                # Verify proxy applied
                mock_session.post.assert_called_once()
                post_call_kwargs = mock_session.post.call_args[1]
                assert post_call_kwargs["proxy"] == "http://proxy.example.com:8080"
                assert post_call_kwargs["json"] == {"key": "value"}

                # Verify response was processed
                assert mock_resp.raise_for_status.called

    def test_httpsync_timeout_enforcement(self):
        """HTTPSync should enforce timeout from environment."""
        # This test verifies that timeout configuration is extracted from env vars
        # The actual timeout enforcement is handled by aiohttp.ClientTimeout
        with patch.dict(os.environ, {
            "CRYPTOFEED_HTTP_TIMEOUT": "3"
        }):
            with patch("aiohttp.ClientSession") as mock_session_class:
                # Mock successful response to verify timeout configuration
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(return_value='{"result": "ok"}')
                mock_resp.raise_for_status = MagicMock()

                mock_get = AsyncMock()
                mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_get.__aexit__ = AsyncMock()

                mock_session = MagicMock()
                mock_session.get = MagicMock(return_value=mock_get)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()

                mock_session_class.return_value = mock_session

                http_sync = HTTPSync()
                http_sync.read("https://api.example.com/endpoint")

                # Verify timeout was configured correctly
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs["timeout"].total == 3.0


class TestHTTPSyncDeprecation:
    """Test HTTPSync deprecation warnings."""

    def test_httpsync_read_emits_deprecation_warning(self):
        """HTTPSync.read should emit DeprecationWarning on use."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Mock successful response
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.text = AsyncMock(return_value='{"result": "ok"}')
            mock_resp.raise_for_status = MagicMock()

            mock_get = AsyncMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_get.__aexit__ = AsyncMock()

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_get)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_session_class.return_value = mock_session

            http_sync = HTTPSync()

            # Verify deprecation warning is emitted
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                http_sync.read("https://api.example.com/endpoint")

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "HTTPSync is deprecated" in str(w[0].message)
                assert "HTTPAsyncConn" in str(w[0].message)

    def test_httpsync_write_emits_deprecation_warning(self):
        """HTTPSync.write should emit DeprecationWarning on use."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Mock successful response
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.text = AsyncMock(return_value='{"result": "created"}')
            mock_resp.raise_for_status = MagicMock()

            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post.__aexit__ = AsyncMock()

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_session_class.return_value = mock_session

            http_sync = HTTPSync()

            # Verify deprecation warning is emitted
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                http_sync.write("https://api.example.com/endpoint", data={"test": "data"})

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "HTTPSync is deprecated" in str(w[0].message)
                assert "HTTPAsyncConn" in str(w[0].message)


class TestHTTPSyncUsageAudit:
    """Document HTTPSync usage in codebase for migration planning."""

    def test_httpsync_usage_inventory(self):
        """Verify HTTPSync usage is limited to expected locations."""
        import subprocess

        # Find all HTTPSync usages in production code (exclude tests)
        result = subprocess.run(
            [
                "grep", "-r", "HTTPSync",
                "--include=*.py",
                "/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/cryptofeed/",
                "--exclude-dir=tests"
            ],
            capture_output=True,
            text=True
        )

        usage_lines = [
            line for line in result.stdout.split('\n')
            if line and 'connection.py' not in line  # Exclude definition
        ]

        # Expected usages:
        # 1. cryptofeed/exchange.py: class-level http_sync = HTTPSync()
        # 2. cryptofeed/raw_data_collection.py: monkeypatch helper for testing
        expected_files = {'exchange.py', 'raw_data_collection.py'}

        found_files = set()
        for line in usage_lines:
            filepath = line.split(':')[0]
            filename = filepath.split('/')[-1]
            if filename not in expected_files:
                pytest.fail(
                    f"Unexpected HTTPSync usage in {filename}: {line}"
                )
            found_files.add(filename)

        # Document findings
        assert 'exchange.py' in found_files or len(found_files) == 0, \
            "HTTPSync usage should be limited to known locations"

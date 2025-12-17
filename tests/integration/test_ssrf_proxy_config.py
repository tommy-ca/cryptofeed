"""
Integration tests for SSRF prevention in proxy configuration loading (REQ-2).

Tests the complete proxy.yaml loading pipeline with malicious configurations.
"""
import pytest
from pathlib import Path
from cryptofeed.run import load_proxy_mapping


@pytest.fixture
def temp_proxy_yaml(tmp_path):
    """Fixture to create temporary proxy.yaml files."""
    def _create_yaml(content: str) -> Path:
        yaml_file = tmp_path / "proxy.yaml"
        yaml_file.write_text(content)
        return yaml_file
    return _create_yaml


class TestMaliciousProxyConfigurations:
    """Test malicious proxy.yaml files are rejected."""

    def test_metadata_endpoint_in_global_blocked(self, temp_proxy_yaml):
        """Verify AWS metadata endpoint in global config is rejected."""
        malicious_yaml = temp_proxy_yaml("""
global:
  http: http://169.254.169.254/latest/meta-data/
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        assert "global.http" in str(exc_info.value)
        assert "blocked IP range" in str(exc_info.value)
        assert "169.254.169.254" in str(exc_info.value)

    def test_file_uri_in_exchange_blocked(self, temp_proxy_yaml):
        """Verify file:// URI in exchange config is rejected."""
        malicious_yaml = temp_proxy_yaml("""
exchanges:
  binance:
    http: file:///etc/passwd
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        assert "exchanges.binance.http" in str(exc_info.value)
        assert "Invalid proxy scheme 'file'" in str(exc_info.value)

    def test_private_ip_in_exchange_blocked(self, temp_proxy_yaml):
        """Verify private IP in exchange config is rejected."""
        malicious_yaml = temp_proxy_yaml("""
exchanges:
  okx:
    socks5: http://192.168.1.1:8080/
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        assert "exchanges.okx.socks5" in str(exc_info.value)
        assert "blocked IP range" in str(exc_info.value)

    def test_localhost_in_global_blocked(self, temp_proxy_yaml):
        """Verify localhost in global config is rejected."""
        malicious_yaml = temp_proxy_yaml("""
global:
  http: http://localhost:8080/
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        assert "global.http" in str(exc_info.value)
        assert "blocked hostname" in str(exc_info.value)
        assert "localhost" in str(exc_info.value)

    def test_legacy_format_with_private_ip_blocked(self, temp_proxy_yaml):
        """Verify legacy string format with private IP is rejected."""
        malicious_yaml = temp_proxy_yaml("""
exchanges:
  binance: http://10.0.0.1:8080/
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        assert "exchanges.binance" in str(exc_info.value)
        assert "blocked IP range" in str(exc_info.value)

    def test_multiple_exchanges_first_invalid_rejected(self, temp_proxy_yaml):
        """Verify first invalid URL stops loading (fail fast)."""
        malicious_yaml = temp_proxy_yaml("""
exchanges:
  binance:
    http: http://169.254.169.254/
  okx:
    http: http://proxy.example.com:8080/
""")
        with pytest.raises(ValueError) as exc_info:
            load_proxy_mapping(str(malicious_yaml))
        # Should fail on binance (first in dict order)
        assert "exchanges.binance.http" in str(exc_info.value)


class TestLegitimateProxyConfigurations:
    """Test legitimate proxy.yaml files load successfully."""

    def test_valid_global_and_exchange_proxies(self, temp_proxy_yaml):
        """Verify legitimate proxy configs load without errors."""
        valid_yaml = temp_proxy_yaml("""
global:
  http: http://proxy.example.com:8080
  socks5: socks5://socks.example.com:1080

exchanges:
  binance:
    http: http://binance-proxy.example.com:8080
  okx:
    socks5: socks5://okx-proxy.example.com:1080
""")
        config = load_proxy_mapping(str(valid_yaml))
        assert config is not None
        assert config['global']['http'] == 'http://proxy.example.com:8080'
        assert config['exchanges']['binance']['http'] == 'http://binance-proxy.example.com:8080'

    def test_public_ip_proxies_allowed(self, temp_proxy_yaml):
        """Verify public IP addresses pass validation."""
        valid_yaml = temp_proxy_yaml("""
global:
  http: http://8.8.8.8:8080
exchanges:
  binance:
    http: http://1.1.1.1:8080
""")
        config = load_proxy_mapping(str(valid_yaml))
        assert config is not None
        assert config['global']['http'] == 'http://8.8.8.8:8080'

    def test_empty_proxy_values_allowed(self, temp_proxy_yaml):
        """Verify empty/null proxy values are allowed (proxy disabled)."""
        valid_yaml = temp_proxy_yaml("""
global:
  http:
exchanges:
  binance:
    http: ""
""")
        config = load_proxy_mapping(str(valid_yaml))
        assert config is not None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Verify nonexistent proxy.yaml returns None (not an error)."""
        nonexistent_path = tmp_path / "does_not_exist.yaml"
        config = load_proxy_mapping(str(nonexistent_path))
        assert config is None

    def test_legacy_string_format_with_valid_url(self, temp_proxy_yaml):
        """Verify legacy string format works with valid URLs."""
        valid_yaml = temp_proxy_yaml("""
exchanges:
  binance: http://proxy.example.com:8080
  okx: socks5://socks.example.com:1080
""")
        config = load_proxy_mapping(str(valid_yaml))
        assert config is not None
        assert config['exchanges']['binance'] == 'http://proxy.example.com:8080'
        assert config['exchanges']['okx'] == 'socks5://socks.example.com:1080'

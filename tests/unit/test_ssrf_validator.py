"""
Unit tests for SSRF URL validation (REQ-2).

Tests the validate_proxy_url() function's three-layer defense:
1. Scheme whitelist (http, https, socks4, socks5, socks5h)
2. IP range validation (private, loopback, link-local, metadata)
3. Hostname pattern matching (localhost variants, metadata endpoints)
"""
import pytest
from cryptofeed.run import validate_proxy_url


class TestSchemeValidation:
    """Test Layer 1: Scheme whitelist validation."""

    @pytest.mark.parametrize("valid_url", [
        "http://proxy.example.com:8080",
        "https://secure-proxy.example.com:443",
        "socks4://socks4-proxy.example.com:1080",
        "socks5://socks-proxy.example.com:1080",
        "socks5h://tor-proxy.example.com:9050",
    ])
    def test_allowed_schemes_pass_validation(self, valid_url):
        """Verify legitimate proxy schemes pass validation."""
        # Should not raise
        validate_proxy_url(valid_url)

    @pytest.mark.parametrize("blocked_url,blocked_scheme", [
        ("file:///etc/passwd", "file"),
        ("ftp://internal.example.com/", "ftp"),
        ("gopher://old-service.example.com/", "gopher"),
        ("data:text/plain,hello", "data"),
        ("javascript:alert(1)", "javascript"),
    ])
    def test_blocked_schemes_raise_error(self, blocked_url, blocked_scheme):
        """Verify non-proxy schemes are rejected with clear error."""
        with pytest.raises(ValueError) as exc_info:
            validate_proxy_url(blocked_url)
        assert "Invalid proxy scheme" in str(exc_info.value)
        assert blocked_scheme in str(exc_info.value)
        assert "Allowed schemes:" in str(exc_info.value)

    def test_empty_url_allowed(self):
        """Empty URLs should pass (proxy disabled case)."""
        # Should not raise
        validate_proxy_url("")
        validate_proxy_url(None)


class TestIPRangeValidation:
    """Test Layer 2: IP range blacklist validation."""

    @pytest.mark.parametrize("blocked_url,reason", [
        ("http://10.0.0.1:8080/", "10.0.0.0/8"),        # Private Class A
        ("http://172.16.5.10/", "172.16.0.0/12"),       # Private Class B
        ("http://192.168.1.1/", "192.168.0.0/16"),      # Private Class C
        ("http://127.0.0.1:9050/", "127.0.0.0/8"),      # Loopback
        ("http://169.254.169.254/", "169.254.0.0/16"),  # AWS metadata
        ("http://169.254.1.1/", "169.254.0.0/16"),      # Link-local
    ])
    def test_private_ips_blocked(self, blocked_url, reason):
        """Verify private IP ranges are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_proxy_url(blocked_url)
        assert "blocked IP range" in str(exc_info.value)
        assert "SSRF prevention" in str(exc_info.value)

    @pytest.mark.parametrize("blocked_url", [
        "http://[::1]:8080/",           # IPv6 loopback
        "http://[fe80::1]/",            # IPv6 link-local
    ])
    def test_ipv6_blocked_addresses(self, blocked_url):
        """Verify IPv6 blocked addresses are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_proxy_url(blocked_url)
        assert "blocked IP range" in str(exc_info.value)

    @pytest.mark.parametrize("public_url", [
        "http://8.8.8.8:8080/",         # Google DNS
        "http://1.1.1.1:8080/",         # Cloudflare DNS
        "http://[2001:4860:4860::8888]/",  # Google IPv6 DNS
    ])
    def test_public_ips_allowed(self, public_url):
        """Verify public IP addresses pass validation."""
        # Should not raise
        validate_proxy_url(public_url)


class TestHostnamePatternMatching:
    """Test Layer 3: Hostname pattern validation."""

    @pytest.mark.parametrize("blocked_url,blocked_hostname", [
        ("http://localhost:8080/", "localhost"),
        ("http://LOCALHOST:8080/", "localhost"),        # Case insensitive
        ("http://metadata.google.internal/", "metadata.google.internal"),
        ("http://169.254.169.254/", "169.254.169.254"),  # Also matches hostname check
    ])
    def test_blocked_hostnames_rejected(self, blocked_url, blocked_hostname):
        """Verify blocked hostname patterns are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_proxy_url(blocked_url)
        assert "blocked" in str(exc_info.value).lower()
        assert "SSRF prevention" in str(exc_info.value)

    @pytest.mark.parametrize("legitimate_url", [
        "http://proxy.example.com:8080/",
        "http://secure-proxy.example.net/",
        "http://socks.mycompany.com:1080/",
        "https://vpn.example.org:443/",
    ])
    def test_legitimate_hostnames_allowed(self, legitimate_url):
        """Verify legitimate proxy hostnames pass validation."""
        # Should not raise
        validate_proxy_url(legitimate_url)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_url_without_hostname(self):
        """URLs without hostname should pass (edge case)."""
        # Should not raise
        validate_proxy_url("http://")

    def test_malformed_url_raises_error(self):
        """Malformed URLs should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_proxy_url("not-a-url")
        # Should fail on scheme validation (no scheme = not in whitelist)
        assert "Invalid proxy scheme" in str(exc_info.value)

    @pytest.mark.parametrize("url_with_creds", [
        "http://user:pass@proxy.example.com:8080/",
        "socks5://admin:secret@socks.example.com:1080/",
    ])
    def test_urls_with_credentials_allowed(self, url_with_creds):
        """URLs with embedded credentials should pass if otherwise valid."""
        # Should not raise
        validate_proxy_url(url_with_creds)

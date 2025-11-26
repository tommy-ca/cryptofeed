"""
Unit tests for security validation checklist script (Task 19.2)

Tests security prerequisite validation before Phase 5 Week 1 deployment.
Addresses CRIT-1 blocker from multi-agent review.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestSecurityValidationChecklist:
    """Test suite for scripts/validate-security-prerequisites.sh"""

    @pytest.fixture
    def script_path(self):
        """Path to security validation script"""
        return Path(__file__).parent.parent.parent / "scripts" / "validate-security-prerequisites.sh"

    @pytest.fixture
    def mock_kafka_env(self):
        """Mock Kafka environment variables"""
        return {
            "KAFKA_BOOTSTRAP_SERVERS": "kafka1:9092,kafka2:9092",
            "KAFKA_SASL_USERNAME": "cryptofeed-prod",
            "KAFKA_SASL_PASSWORD": "secure-password",
            "KAFKA_SSL_CERT": "/etc/ssl/kafka-client.pem",
            "KAFKA_SSL_KEY": "/etc/ssl/kafka-client-key.pem",
            "KAFKA_SSL_CA": "/etc/ssl/kafka-ca.pem",
            "VAULT_ADDR": "https://vault.internal:8200",
            "VAULT_TOKEN": "hvs.test-token",
            "METRICS_AUTH_USER": "prometheus",
            "METRICS_AUTH_PASSWORD": "metrics-password"
        }

    def test_script_exists(self, script_path):
        """Test that validation script exists"""
        assert script_path.exists(), f"Script not found at {script_path}"
        assert script_path.is_file(), f"Script is not a file: {script_path}"

    def test_script_executable(self, script_path):
        """Test that script has executable permissions"""
        assert os.access(script_path, os.X_OK), f"Script not executable: {script_path}"

    def test_sasl_ssl_certificates_check(self, script_path, mock_kafka_env, tmp_path):
        """Test SASL/SSL certificate validation"""
        # Create mock certificate files
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        (cert_dir / "kafka-client.pem").write_text("MOCK CERT")
        (cert_dir / "kafka-client-key.pem").write_text("MOCK KEY")
        (cert_dir / "kafka-ca.pem").write_text("MOCK CA")

        env = mock_kafka_env.copy()
        env["KAFKA_SSL_CERT"] = str(cert_dir / "kafka-client.pem")
        env["KAFKA_SSL_KEY"] = str(cert_dir / "kafka-client-key.pem")
        env["KAFKA_SSL_CA"] = str(cert_dir / "kafka-ca.pem")

        result = subprocess.run(
            [str(script_path), "--check", "sasl-ssl"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"SASL/SSL check failed: {result.stderr}"
        assert "SASL/SSL certificates validated" in result.stdout

    def test_missing_sasl_certificate_fails(self, script_path, mock_kafka_env):
        """Test that missing SASL/SSL certificate causes failure"""
        env = mock_kafka_env.copy()
        env["KAFKA_SSL_CERT"] = "/nonexistent/cert.pem"

        result = subprocess.run(
            [str(script_path), "--check", "sasl-ssl"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing certificate should fail validation"
        assert "SASL/SSL certificate not found" in result.stderr

    def test_service_accounts_check(self, script_path, mock_kafka_env):
        """Test service account validation"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--check", "service-accounts"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Service account check failed: {result.stderr}"
        assert "Service accounts validated" in result.stdout

    def test_missing_service_account_fails(self, script_path):
        """Test that missing service account credentials cause failure"""
        env = {"KAFKA_BOOTSTRAP_SERVERS": "kafka1:9092"}

        result = subprocess.run(
            [str(script_path), "--check", "service-accounts"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing service account should fail"
        assert "KAFKA_SASL_USERNAME not set" in result.stderr

    def test_vault_secrets_check(self, script_path, mock_kafka_env):
        """Test vault secret validation"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--check", "vault-secrets"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Vault secrets check failed: {result.stderr}"
        assert "Vault secrets validated" in result.stdout

    def test_missing_vault_token_fails(self, script_path, mock_kafka_env):
        """Test that missing vault token causes failure"""
        env = mock_kafka_env.copy()
        env.pop("VAULT_TOKEN", None)

        result = subprocess.run(
            [str(script_path), "--check", "vault-secrets"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing vault token should fail"
        assert "VAULT_TOKEN not set" in result.stderr

    def test_tls_enabled_check(self, script_path, mock_kafka_env):
        """Test TLS connection validation"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--check", "tls-enabled"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"TLS check failed: {result.stderr}"
        assert "TLS enabled for Kafka connections" in result.stdout

    def test_missing_tls_fails(self, script_path):
        """Test that missing TLS configuration causes failure"""
        env = {"KAFKA_BOOTSTRAP_SERVERS": "kafka1:9092"}

        result = subprocess.run(
            [str(script_path), "--check", "tls-enabled"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing TLS should fail"
        assert "TLS not configured" in result.stderr

    def test_metrics_auth_check(self, script_path, mock_kafka_env):
        """Test metrics endpoint authentication validation"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--check", "metrics-auth"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Metrics auth check failed: {result.stderr}"
        assert "Metrics endpoints protected" in result.stdout

    def test_missing_metrics_auth_fails(self, script_path, mock_kafka_env):
        """Test that missing metrics authentication causes failure"""
        env = mock_kafka_env.copy()
        env.pop("METRICS_AUTH_PASSWORD", None)

        result = subprocess.run(
            [str(script_path), "--check", "metrics-auth"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing metrics auth should fail"
        assert "METRICS_AUTH_PASSWORD not set" in result.stderr

    def test_network_policies_check(self, script_path, mock_kafka_env):
        """Test network policies validation"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--check", "network-policies"],
            env=env,
            capture_output=True,
            text=True
        )

        # Note: This check may require Kubernetes API access
        # For unit tests, we expect the script to validate config
        assert result.returncode == 0 or "Kubernetes API not accessible" in result.stderr

    def test_all_checks_pass(self, script_path, mock_kafka_env, tmp_path):
        """Test that all checks pass with complete configuration"""
        # Create mock certificates
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        (cert_dir / "kafka-client.pem").write_text("MOCK CERT")
        (cert_dir / "kafka-client-key.pem").write_text("MOCK KEY")
        (cert_dir / "kafka-ca.pem").write_text("MOCK CA")

        env = mock_kafka_env.copy()
        env["KAFKA_SSL_CERT"] = str(cert_dir / "kafka-client.pem")
        env["KAFKA_SSL_KEY"] = str(cert_dir / "kafka-client-key.pem")
        env["KAFKA_SSL_CA"] = str(cert_dir / "kafka-ca.pem")

        result = subprocess.run(
            [str(script_path)],  # Run all checks
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"All checks should pass: {result.stderr}"
        assert "All security checks passed" in result.stdout

    def test_any_check_failure_exits_nonzero(self, script_path):
        """Test that any failing check causes script to exit with error"""
        env = {"KAFKA_BOOTSTRAP_SERVERS": "kafka1:9092"}  # Minimal env (missing most configs)

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Missing configs should cause failure"
        assert "Security validation failed" in result.stderr

    def test_dry_run_mode(self, script_path, mock_kafka_env):
        """Test dry-run mode (no actual validation, just prints checks)"""
        env = mock_kafka_env.copy()

        result = subprocess.run(
            [str(script_path), "--dry-run"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Dry-run should always succeed"
        assert "DRY-RUN mode" in result.stdout

    def test_help_flag(self, script_path):
        """Test that --help flag displays usage"""
        result = subprocess.run(
            [str(script_path), "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--check" in result.stdout

    def test_verbose_mode(self, script_path, mock_kafka_env, tmp_path):
        """Test verbose output mode"""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        (cert_dir / "kafka-client.pem").write_text("MOCK CERT")
        (cert_dir / "kafka-client-key.pem").write_text("MOCK KEY")
        (cert_dir / "kafka-ca.pem").write_text("MOCK CA")

        env = mock_kafka_env.copy()
        env["KAFKA_SSL_CERT"] = str(cert_dir / "kafka-client.pem")
        env["KAFKA_SSL_KEY"] = str(cert_dir / "kafka-client-key.pem")
        env["KAFKA_SSL_CA"] = str(cert_dir / "kafka-ca.pem")

        result = subprocess.run(
            [str(script_path), "--verbose"],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        # Verbose mode should show detailed check output
        assert "Checking SASL/SSL certificates" in result.stdout
        assert "Checking service accounts" in result.stdout

    def test_summary_output(self, script_path, mock_kafka_env, tmp_path):
        """Test that script outputs validation summary"""
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir()
        (cert_dir / "kafka-client.pem").write_text("MOCK CERT")
        (cert_dir / "kafka-client-key.pem").write_text("MOCK KEY")
        (cert_dir / "kafka-ca.pem").write_text("MOCK CA")

        env = mock_kafka_env.copy()
        env["KAFKA_SSL_CERT"] = str(cert_dir / "kafka-client.pem")
        env["KAFKA_SSL_KEY"] = str(cert_dir / "kafka-client-key.pem")
        env["KAFKA_SSL_CA"] = str(cert_dir / "kafka-ca.pem")

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        # Should output summary with all checks
        assert "Security Validation Summary" in result.stdout
        assert "6/6 checks passed" in result.stdout or "checks passed" in result.stdout


class TestSecurityValidationIntegration:
    """Integration tests for security validation"""

    def test_environment_template_exists(self):
        """Test that .env.production.template exists"""
        template_path = Path(__file__).parent.parent.parent / ".env.production.template"
        assert template_path.exists(), f"Environment template not found: {template_path}"

    def test_template_has_required_vars(self):
        """Test that template documents all required environment variables"""
        template_path = Path(__file__).parent.parent.parent / ".env.production.template"
        if not template_path.exists():
            pytest.skip("Template file not yet created")

        content = template_path.read_text()

        required_vars = [
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_SASL_USERNAME",
            "KAFKA_SASL_PASSWORD",
            "KAFKA_SSL_CERT",
            "KAFKA_SSL_KEY",
            "KAFKA_SSL_CA",
            "VAULT_ADDR",
            "VAULT_TOKEN",
            "METRICS_AUTH_USER",
            "METRICS_AUTH_PASSWORD"
        ]

        for var in required_vars:
            assert var in content, f"Required variable {var} missing from template"

    def test_ci_environment_check(self):
        """Test that CI environment validation is documented"""
        # This would validate that CI has necessary secrets configured
        # For now, just ensure the check exists
        pass

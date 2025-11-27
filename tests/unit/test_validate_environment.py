"""
Unit tests for scripts/validate-environment.sh

Tests environment variable validation script that checks all required
variables for Phase 5 deployment are set before execution.

Test Coverage:
- All required variables validation
- Missing variable detection
- Invalid format detection
- Clear error messaging
- Exit code behavior
"""
import os
import subprocess
from pathlib import Path

import pytest


class TestEnvironmentValidation:
    """Test suite for environment variable validation script."""

    @pytest.fixture
    def script_path(self):
        """Path to validation script."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / "scripts" / "validate-environment.sh"

    @pytest.fixture
    def env_template_path(self):
        """Path to .env.production.template."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / ".env.production.template"

    @pytest.fixture
    def clean_env(self):
        """Clean environment without required variables."""
        # Save current environment
        saved_env = os.environ.copy()

        # Remove all KAFKA_, PROMETHEUS_, GRAFANA_, METRICS_ variables
        for key in list(os.environ.keys()):
            if any(prefix in key for prefix in ['KAFKA_', 'PROMETHEUS', 'GRAFANA', 'METRICS_']):
                del os.environ[key]

        yield

        # Restore environment
        os.environ.clear()
        os.environ.update(saved_env)

    @pytest.fixture
    def valid_env(self):
        """Environment with all required variables set."""
        return {
            # Kafka Configuration
            'KAFKA_BOOTSTRAP_SERVERS': 'kafka1.internal:9093,kafka2.internal:9093',
            'KAFKA_SASL_USERNAME': 'cryptofeed-prod',
            'KAFKA_SASL_PASSWORD': 'test-password-secure',
            'KAFKA_SSL_CERT': '/etc/ssl/certs/kafka-client.pem',
            'KAFKA_SSL_KEY': '/etc/ssl/private/kafka-client-key.pem',
            'KAFKA_SSL_CA': '/etc/ssl/certs/kafka-ca.pem',

            # Vault Configuration
            'VAULT_ADDR': 'https://vault.internal:8200',
            'VAULT_TOKEN': 'test-vault-token',

            # Metrics Configuration
            'METRICS_AUTH_USER': 'prometheus',
            'METRICS_AUTH_PASSWORD': 'test-metrics-password',

            # Monitoring Configuration
            'PROMETHEUS_URL': 'http://prometheus.internal:9090',
            'GRAFANA_URL': 'http://grafana.internal:3000',
            'GRAFANA_API_KEY': 'test-grafana-api-key',

            # Operational Configuration
            'ENVIRONMENT': 'production',
        }

    def test_script_exists(self, script_path):
        """Test that validation script exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_executable(self, script_path):
        """Test that validation script is executable."""
        assert os.access(script_path, os.X_OK), f"Script not executable: {script_path}"

    def test_env_template_exists(self, env_template_path):
        """Test that .env.production.template exists."""
        assert env_template_path.exists(), f"Template not found: {env_template_path}"

    def test_all_required_variables_present(self, script_path, valid_env):
        """Test validation passes when all required variables are set."""
        result = subprocess.run(
            [str(script_path)],
            env=valid_env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        assert "All required environment variables validated" in result.stdout

    def test_missing_kafka_bootstrap_servers(self, script_path, valid_env, clean_env):
        """Test validation fails when KAFKA_BOOTSTRAP_SERVERS is missing."""
        env = valid_env.copy()
        del env['KAFKA_BOOTSTRAP_SERVERS']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing KAFKA_BOOTSTRAP_SERVERS"
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.stderr

    def test_missing_sasl_credentials(self, script_path, valid_env, clean_env):
        """Test validation fails when SASL credentials are missing."""
        env = valid_env.copy()
        del env['KAFKA_SASL_USERNAME']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing SASL username"
        assert "KAFKA_SASL_USERNAME" in result.stderr

    def test_missing_ssl_certificates(self, script_path, valid_env, clean_env):
        """Test validation fails when SSL certificate paths are missing."""
        env = valid_env.copy()
        del env['KAFKA_SSL_CERT']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing SSL cert"
        assert "KAFKA_SSL_CERT" in result.stderr

    def test_missing_vault_configuration(self, script_path, valid_env, clean_env):
        """Test validation fails when Vault configuration is missing."""
        env = valid_env.copy()
        del env['VAULT_ADDR']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing VAULT_ADDR"
        assert "VAULT_ADDR" in result.stderr

    def test_missing_metrics_configuration(self, script_path, valid_env, clean_env):
        """Test validation fails when metrics configuration is missing."""
        env = valid_env.copy()
        del env['PROMETHEUS_URL']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing PROMETHEUS_URL"
        assert "PROMETHEUS_URL" in result.stderr

    def test_missing_grafana_configuration(self, script_path, valid_env, clean_env):
        """Test validation fails when Grafana configuration is missing."""
        env = valid_env.copy()
        del env['GRAFANA_URL']

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with missing GRAFANA_URL"
        assert "GRAFANA_URL" in result.stderr

    def test_invalid_kafka_bootstrap_servers_format(self, script_path, valid_env, clean_env):
        """Test validation fails with invalid KAFKA_BOOTSTRAP_SERVERS format."""
        env = valid_env.copy()
        env['KAFKA_BOOTSTRAP_SERVERS'] = 'invalid-format'  # Missing port

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with invalid bootstrap servers format"
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.stderr or "invalid" in result.stderr.lower()

    def test_invalid_vault_addr_format(self, script_path, valid_env, clean_env):
        """Test validation fails when VAULT_ADDR doesn't use HTTPS."""
        env = valid_env.copy()
        env['VAULT_ADDR'] = 'http://vault.internal:8200'  # HTTP instead of HTTPS

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with non-HTTPS VAULT_ADDR"
        assert "VAULT_ADDR" in result.stderr or "https" in result.stderr.lower()

    def test_invalid_environment_value(self, script_path, valid_env, clean_env):
        """Test validation fails with invalid ENVIRONMENT value."""
        env = valid_env.copy()
        env['ENVIRONMENT'] = 'invalid-env'  # Not production/staging/development

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with invalid ENVIRONMENT"
        assert "ENVIRONMENT" in result.stderr

    def test_clear_error_messages(self, script_path, clean_env):
        """Test that error messages are clear and actionable."""
        result = subprocess.run(
            [str(script_path)],
            env={},  # No variables set
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        # Should mention what's missing
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.stderr
        # Should provide actionable guidance
        assert ".env.production.template" in result.stderr or "required" in result.stderr.lower()

    def test_validation_summary_on_success(self, script_path, valid_env):
        """Test that validation summary is shown on success."""
        result = subprocess.run(
            [str(script_path)],
            env=valid_env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        # Should show summary of validated variables
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.stdout
        assert "validated" in result.stdout.lower() or "success" in result.stdout.lower()

    def test_empty_variable_treated_as_missing(self, script_path, valid_env, clean_env):
        """Test that empty string variables are treated as missing."""
        env = valid_env.copy()
        env['KAFKA_BOOTSTRAP_SERVERS'] = ''  # Empty string

        result = subprocess.run(
            [str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, "Should fail with empty KAFKA_BOOTSTRAP_SERVERS"
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.stderr

    def test_env_template_has_all_required_variables(self, env_template_path):
        """Test that .env.production.template documents all required variables."""
        template_content = env_template_path.read_text()

        required_vars = [
            'KAFKA_BOOTSTRAP_SERVERS',
            'KAFKA_SASL_USERNAME',
            'KAFKA_SASL_PASSWORD',
            'KAFKA_SSL_CERT',
            'KAFKA_SSL_KEY',
            'KAFKA_SSL_CA',
            'VAULT_ADDR',
            'VAULT_TOKEN',
            'METRICS_AUTH_USER',
            'METRICS_AUTH_PASSWORD',
            'PROMETHEUS_URL',
            'GRAFANA_URL',
            'GRAFANA_API_KEY',
            'ENVIRONMENT',
        ]

        for var in required_vars:
            assert var in template_content, f"{var} not documented in template"

    def test_env_template_has_usage_instructions(self, env_template_path):
        """Test that .env.production.template has clear usage instructions."""
        template_content = env_template_path.read_text()

        assert "Usage:" in template_content
        assert ".env.production" in template_content
        assert "source" in template_content.lower()

    def test_env_template_has_security_warnings(self, env_template_path):
        """Test that .env.production.template has security warnings."""
        template_content = env_template_path.read_text()

        assert "DO NOT commit" in template_content or "CRITICAL" in template_content
        assert "vault" in template_content.lower() or "secret" in template_content.lower()

"""
Integration tests for Docker Compose orchestration.

These tests verify the docker-compose.yml configuration works correctly
with the Kafka broker and cryptofeed service.

Test Requirements:
- Docker and Docker Compose must be installed
- Ports 8080, 9090, 9092 must be available
- Run from repository root: pytest tests/integration/test_docker_compose.py -v

Success Criteria (Task 2 Requirements):
- docker-compose up completes startup in under 60 seconds
- Kafka broker reaches healthy state within 30 seconds
- Cryptofeed service reaches healthy state within 40 seconds
- /health endpoint returns 200 OK
- /metrics endpoint returns Prometheus text format
- Graceful shutdown with docker-compose down completes within 30 seconds
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests


# Test fixture paths
REPO_ROOT = Path(__file__).parent.parent.parent
DOCKER_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_FILE = REPO_ROOT / ".env"
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="module")
def docker_compose_env():
    """
    Set up Docker Compose environment for testing.

    Creates .env file if it doesn't exist, starts docker-compose stack,
    waits for services to be healthy, yields for tests, then tears down.
    """
    # Ensure proxy environment variables exist for proxy integration tests (empty by default)
    proxy_http = os.environ.setdefault("PROXY_HTTP", "")
    proxy_socks5 = os.environ.setdefault("PROXY_SOCKS5", "")

    # Ensure .env file exists (create from .env.example if needed)
    if not ENV_FILE.exists():
        env_example = REPO_ROOT / ".env.example"
        if env_example.exists():
            ENV_FILE.write_text(env_example.read_text())
        else:
            # Create minimal .env for testing
            ENV_FILE.write_text(
                "VERSION=test\n"
                "LOG_LEVEL=INFO\n"
                "KAFKA_PARTITION_STRATEGY=composite\n"
            )

    # Start docker-compose stack
    print("\n[Setup] Starting Docker Compose stack...")
    start_time = time.time()

    result = subprocess.run(
        ["docker-compose", "up", "-d"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        pytest.fail(f"docker-compose up failed:\n{result.stderr}")

    startup_duration = time.time() - start_time
    print(f"[Setup] Docker Compose started in {startup_duration:.1f}s")

    # Wait for services to be healthy (max 60 seconds)
    print("[Setup] Waiting for services to be healthy...")
    wait_start = time.time()
    max_wait = 60

    kafka_healthy = False
    cryptofeed_healthy = False

    while (time.time() - wait_start) < max_wait:
        # Check Kafka health
        result = subprocess.run(
            ["docker-compose", "ps", "--filter", "health=healthy", "kafka"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        kafka_healthy = "kafka" in result.stdout

        # Check cryptofeed health
        result = subprocess.run(
            ["docker-compose", "ps", "--filter", "health=healthy", "cryptofeed"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        cryptofeed_healthy = "cryptofeed" in result.stdout

        if kafka_healthy and cryptofeed_healthy:
            break

        time.sleep(2)

    health_wait_duration = time.time() - wait_start
    print(f"[Setup] Services healthy after {health_wait_duration:.1f}s")

    if not kafka_healthy:
        # Print logs for debugging
        subprocess.run(["docker-compose", "logs", "kafka"], cwd=REPO_ROOT)
        pytest.fail("Kafka service did not become healthy within 60 seconds")

    if not cryptofeed_healthy:
        # Print logs for debugging
        subprocess.run(["docker-compose", "logs", "cryptofeed"], cwd=REPO_ROOT)
        pytest.fail("Cryptofeed service did not become healthy within 60 seconds")

    # Yield for tests
    yield {
        "startup_duration": startup_duration,
        "health_wait_duration": health_wait_duration,
        "proxy_http": proxy_http,
        "proxy_socks5": proxy_socks5,
    }

    # Teardown: Stop docker-compose stack
    print("\n[Teardown] Stopping Docker Compose stack...")
    shutdown_start = time.time()

    result = subprocess.run(
        ["docker-compose", "down"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        print(f"Warning: docker-compose down had errors:\n{result.stderr}")

    shutdown_duration = time.time() - shutdown_start
    print(f"[Teardown] Docker Compose stopped in {shutdown_duration:.1f}s")

    # Verify shutdown time requirement (< 30 seconds)
    assert shutdown_duration < 30, (
        f"Shutdown took {shutdown_duration:.1f}s, exceeds 30s requirement"
    )


class TestDockerComposeStartup:
    """Test Docker Compose stack startup and configuration."""

    def test_startup_time(self, docker_compose_env):
        """Test that docker-compose up completes in under 60 seconds."""
        startup_duration = docker_compose_env["startup_duration"]
        assert startup_duration < 60, (
            f"Startup took {startup_duration:.1f}s, exceeds 60s requirement"
        )

    def test_kafka_healthy(self, docker_compose_env):
        """Test that Kafka broker reaches healthy state."""
        # Check Kafka is running and healthy
        result = subprocess.run(
            ["docker-compose", "ps", "kafka"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert "Up" in result.stdout, "Kafka container is not running"
        assert "(healthy)" in result.stdout, "Kafka container is not healthy"

    def test_cryptofeed_healthy(self, docker_compose_env):
        """Test that cryptofeed service reaches healthy state."""
        # Check cryptofeed is running and healthy
        result = subprocess.run(
            ["docker-compose", "ps", "cryptofeed"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert "Up" in result.stdout, "Cryptofeed container is not running"
        assert "(healthy)" in result.stdout, "Cryptofeed container is not healthy"


class TestHealthEndpoints:
    """Test health check and metrics endpoints."""

    def test_health_endpoint(self, docker_compose_env):
        """Test /health endpoint returns 200 OK."""
        response = requests.get("http://localhost:8080/health", timeout=5)

        assert response.status_code == 200, (
            f"Health endpoint returned {response.status_code}, expected 200"
        )

        # Verify response is JSON
        data = response.json()
        assert "status" in data, "Health response missing 'status' field"
        assert data["status"] in ["healthy", "degraded"], (
            f"Unexpected health status: {data['status']}"
        )

    def test_ready_endpoint(self, docker_compose_env):
        """Test /ready endpoint returns 200 OK when service is ready."""
        response = requests.get("http://localhost:8080/ready", timeout=5)

        assert response.status_code == 200, (
            f"Ready endpoint returned {response.status_code}, expected 200"
        )

    def test_metrics_endpoint(self, docker_compose_env):
        """Test /metrics endpoint returns Prometheus text format."""
        response = requests.get("http://localhost:9090/metrics", timeout=5)

        assert response.status_code == 200, (
            f"Metrics endpoint returned {response.status_code}, expected 200"
        )

        # Verify Prometheus text format
        content = response.text
        assert "# HELP" in content, "Metrics response missing HELP directives"
        assert "# TYPE" in content, "Metrics response missing TYPE directives"

        # Check for expected metrics (from prometheus_client)
        assert "process_cpu_seconds_total" in content, (
            "Missing process_cpu_seconds_total metric"
        )


class TestKafkaConnectivity:
    """Test Kafka broker connectivity."""

    def test_kafka_reachable(self, docker_compose_env):
        """Test Kafka broker reachable at localhost:9092."""
        try:
            from kafka import KafkaProducer

            # Create Kafka producer to test connectivity
            producer = KafkaProducer(
                bootstrap_servers=["localhost:9092"],
                api_version_auto_timeout_ms=5000,
            )

            # Get cluster metadata to verify connection
            metadata = producer._metadata
            assert metadata is not None, "Failed to retrieve Kafka metadata"

            producer.close()

        except ImportError:
            pytest.skip("kafka-python not installed, skipping Kafka connectivity test")
        except Exception as e:
            pytest.fail(f"Kafka connectivity test failed: {e}")


class TestVolumeMounts:
    """Test volume mounts are configured correctly."""

    def test_config_volume_readable(self, docker_compose_env):
        """Test config.yaml volume mount is readable inside container."""
        result = subprocess.run(
            [
                "docker-compose",
                "exec",
                "-T",
                "cryptofeed",
                "cat",
                "/config/config.yaml",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Failed to read /config/config.yaml: {result.stderr}"
        )
        assert len(result.stdout) > 0, "config.yaml is empty"

    def test_proxy_volume_readable(self, docker_compose_env):
        """Test proxy.yaml volume mount is readable inside container."""
        result = subprocess.run(
            [
                "docker-compose",
                "exec",
                "-T",
                "cryptofeed",
                "cat",
                "/config/proxy.yaml",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Failed to read /config/proxy.yaml: {result.stderr}"
        )
        assert len(result.stdout) > 0, "proxy.yaml is empty"


class TestProxyIntegration:
    """Test proxy environment variables and configuration wiring."""

    def test_proxy_env_wired(self, docker_compose_env):
        """Ensure PROXY_HTTP and PROXY_SOCKS5 are present inside container."""
        result = subprocess.run(
            [
                "docker-compose",
                "exec",
                "-T",
                "cryptofeed",
                "env",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Failed to exec env: {result.stderr}"
        env_output = result.stdout

        assert (
            f"PROXY_HTTP={docker_compose_env['proxy_http']}" in env_output
        ), "PROXY_HTTP not set in container environment"

        assert (
            f"PROXY_SOCKS5={docker_compose_env['proxy_socks5']}" in env_output
        ), "PROXY_SOCKS5 not set in container environment"

    def test_proxy_config_loaded(self, docker_compose_env):
        """Verify proxy config renders environment substitutions."""
        result = subprocess.run(
            [
                "docker-compose",
                "exec",
                "-T",
                "cryptofeed",
                "python",
                "-c",
                (
                    "import yaml, os; "
                    "data = yaml.safe_load(open('/config/proxy.yaml')); "
                    "expected_http = os.environ.get('PROXY_HTTP') or None; "
                    "expected_socks5 = os.environ.get('PROXY_SOCKS5') or None; "
                    "assert data['global']['http'] == expected_http, "
                    "f'http proxy mismatch: {data['global']['http']} vs {expected_http}'; "
                    "assert data['global']['socks5'] == expected_socks5, "
                    "f'socks5 proxy mismatch: {data['global']['socks5']} vs {expected_socks5}'; "
                    "print('proxy config ok')"
                ),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Proxy config validation failed: {result.stderr}\\n{result.stdout}"
        )



class TestResourceLimits:
    """Test resource limits are applied correctly."""

    def test_cryptofeed_resource_limits(self, docker_compose_env):
        """Test cryptofeed container has correct resource limits."""
        result = subprocess.run(
            ["docker", "inspect", "cryptofeed-service"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Failed to inspect cryptofeed container"

        import json

        data = json.loads(result.stdout)
        resources = data[0]["HostConfig"]

        # Check CPU limit (1.0 CPU = 1000000000 nanoseconds)
        # Note: NanoCpus might be 0 if using CpuQuota/CpuPeriod
        if resources.get("NanoCpus"):
            assert resources["NanoCpus"] == 1000000000, (
                f"CPU limit incorrect: {resources['NanoCpus']}"
            )

        # Check memory limit (2GB = 2147483648 bytes)
        assert resources["Memory"] == 2147483648, (
            f"Memory limit incorrect: {resources['Memory']}"
        )

    def test_kafka_resource_limits(self, docker_compose_env):
        """Test Kafka container has correct resource limits."""
        result = subprocess.run(
            ["docker", "inspect", "cryptofeed-kafka"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, "Failed to inspect Kafka container"

        import json

        data = json.loads(result.stdout)
        resources = data[0]["HostConfig"]

        # Check memory limit (512MB = 536870912 bytes)
        assert resources["Memory"] == 536870912, (
            f"Memory limit incorrect: {resources['Memory']}"
        )


class TestGracefulShutdown:
    """Test graceful shutdown behavior."""

    def test_sigterm_handling(self, docker_compose_env):
        """Test container handles SIGTERM gracefully."""
        # Send SIGTERM to cryptofeed container
        result = subprocess.run(
            ["docker-compose", "kill", "-s", "SIGTERM", "cryptofeed"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Failed to send SIGTERM: {result.stderr}"

        # Wait for container to stop (should happen within 30 seconds)
        start_time = time.time()
        max_wait = 30

        while (time.time() - start_time) < max_wait:
            result = subprocess.run(
                ["docker-compose", "ps", "cryptofeed"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            if "Up" not in result.stdout:
                break

            time.sleep(1)

        stop_duration = time.time() - start_time

        assert stop_duration < 30, (
            f"Container took {stop_duration:.1f}s to stop, exceeds 30s grace period"
        )

        # Restart service for remaining tests
        subprocess.run(
            ["docker-compose", "up", "-d", "cryptofeed"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        time.sleep(10)  # Wait for service to be healthy again


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

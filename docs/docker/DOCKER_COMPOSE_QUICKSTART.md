# Docker Compose Quickstart Guide

This guide walks you through running Cryptofeed with Docker Compose for local development and testing.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose V2 (included with Docker Desktop)
- 4GB RAM available for containers
- Ports available: 8080, 9090, 9092

## Quick Start (5 Minutes)

### 1. Clone Repository and Navigate to Root

```bash
cd /path/to/cryptofeed
```

### 2. Create Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your exchange API keys (optional for testing)
nano .env
```

**Minimal Configuration** (No API keys needed for testing):

```bash
# .env
VERSION=dev
LOG_LEVEL=INFO
KAFKA_PARTITION_STRATEGY=composite
```

### 3. Verify Configuration Files

Ensure these files exist:

```bash
ls -l config/config.yaml
ls -l config/proxy.yaml
```

Both files are already created with example configurations.

### 4. Start the Stack

```bash
docker-compose up -d
```

Expected output:

```
[+] Running 3/3
 ✔ Network cryptofeed-network     Created
 ✔ Container cryptofeed-kafka     Started
 ✔ Container cryptofeed-service   Started
```

### 5. Verify Services are Healthy

```bash
# Check service status
docker-compose ps

# Expected output:
# NAME                  STATUS              PORTS
# cryptofeed-kafka      Up (healthy)        0.0.0.0:9092->9092/tcp
# cryptofeed-service    Up (healthy)        0.0.0.0:8080->8080/tcp, 0.0.0.0:9090->9090/tcp
```

### 6. Test Health Endpoints

```bash
# Test health endpoint
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-12-12T20:30:00Z",
#   "components": {
#     "kafka": "connected",
#     "exchanges": []
#   },
#   "uptime_seconds": 45.2
# }

# Test readiness endpoint
curl http://localhost:8080/ready

# Test metrics endpoint
curl http://localhost:9090/metrics
```

### 7. View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f cryptofeed
docker-compose logs -f kafka
```

### 8. Stop the Stack

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (clean slate)
docker-compose down -v
```

## Configuration

### Environment Variables (.env file)

Key environment variables you can configure:

```bash
# Image version and build metadata
VERSION=dev
BUILD_TIMESTAMP=2025-12-12T20:00:00Z
GIT_COMMIT_SHA=abc123
GIT_BRANCH=main

# Kafka configuration
KAFKA_PARTITION_STRATEGY=composite  # composite, symbol, exchange, round_robin

# Exchange API keys (optional, for real exchange connections)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_API_PASSPHRASE=your_passphrase

# Proxy settings (optional, for geo-restricted exchanges)
PROXY_HTTP=http://proxy.example.com:8080
PROXY_SOCKS5=socks5://proxy.example.com:1080

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Cryptofeed Configuration (config/config.yaml)

Edit `config/config.yaml` to configure:

- Exchange subscriptions (which exchanges and symbols to track)
- Kafka backend settings
- Logging preferences
- Performance tuning

Example minimal configuration:

```yaml
log:
  level: INFO

kafka:
  bootstrap_servers:
    - kafka:29092
  topic_strategy: consolidated
  partition_strategy: composite
```

### Proxy Configuration (config/proxy.yaml)

Edit `config/proxy.yaml` to configure proxy settings:

```yaml
global:
  http: ${PROXY_HTTP}
  socks5: ${PROXY_SOCKS5}

exchanges:
  binance:
    http: http://us-proxy.example.com:8080
```

## Testing

### Run Integration Tests

```bash
# Install test dependencies
pip install pytest requests kafka-python

# Run Docker Compose integration tests
pytest tests/integration/test_docker_compose.py -v

# Run with detailed output
pytest tests/integration/test_docker_compose.py -v --tb=short
```

### Manual Testing Checklist

- [ ] Services start in under 60 seconds
- [ ] Kafka reaches healthy state within 30 seconds
- [ ] Cryptofeed reaches healthy state within 40 seconds
- [ ] Health endpoint returns 200 OK
- [ ] Metrics endpoint returns Prometheus format
- [ ] Services shut down gracefully in under 30 seconds
- [ ] Volume mounts are readable inside containers
- [ ] Resource limits are applied correctly

## Troubleshooting

### Services Not Starting

**Check logs:**

```bash
docker-compose logs kafka
docker-compose logs cryptofeed
```

**Common issues:**

- Ports already in use (check with `netstat -tulpn | grep -E '8080|9090|9092'`)
- Insufficient memory (increase Docker memory limit to 4GB+)
- Configuration file errors (validate YAML syntax)

### Kafka Connection Errors

**Symptoms:**

- Cryptofeed logs show "Failed to connect to Kafka"
- Health check returns "unhealthy" status

**Solutions:**

1. Verify Kafka is healthy: `docker-compose ps kafka`
2. Check Kafka logs: `docker-compose logs kafka`
3. Test Kafka connectivity:

```bash
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### Health Check Failures

**Cryptofeed health check failing:**

```bash
# Check if health server is running
docker-compose exec cryptofeed curl localhost:8080/health

# Check for Python errors
docker-compose logs cryptofeed | grep -i error
```

### Volume Mount Issues

**Config files not readable:**

```bash
# Verify files exist on host
ls -l config/config.yaml config/proxy.yaml

# Check permissions
chmod 644 config/config.yaml config/proxy.yaml

# Test reading inside container
docker-compose exec cryptofeed cat /config/config.yaml
```

### Resource Limit Issues

**Containers getting OOM killed:**

```bash
# Check Docker resource limits
docker stats

# Increase memory limits in docker-compose.yml
```

## Advanced Usage

### Custom Exchanges Configuration

Edit `config/config.yaml` to add exchange subscriptions:

```yaml
# Subscribe to Binance SPOT trades and order book
binance:
  channels:
    - trades
    - l2_book
  symbols:
    - BTC-USDT
    - ETH-USDT

# Subscribe to Coinbase Pro
coinbase:
  channels:
    - trades
  symbols:
    - BTC-USD
```

### Multi-Exchange Setup

Run Cryptofeed with multiple exchanges:

```yaml
# config/config.yaml
binance:
  channels: [trades, l2_book]
  symbols: [BTC-USDT, ETH-USDT]

coinbase:
  channels: [trades]
  symbols: [BTC-USD, ETH-USD]

kraken:
  channels: [trades, ticker]
  symbols: [BTC-USD, ETH-USD]
```

### Production-like Testing

Use production-like configuration for testing:

```bash
# Use consolidated topics (default)
KAFKA_PARTITION_STRATEGY=composite docker-compose up -d

# Use per-symbol topics
KAFKA_PARTITION_STRATEGY=symbol docker-compose up -d

# Enable debug logging
LOG_LEVEL=DEBUG docker-compose up -d
```

### Kafka Consumer Testing

Test consuming messages from Kafka topics:

```bash
# Install kafka-python
pip install kafka-python

# Run consumer script
python examples/kafka_consumer.py
```

Example consumer script:

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'cryptofeed.trades',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
)

for message in consumer:
    print(f"Received: {message.value}")
```

## Next Steps

1. **Configure exchanges**: Add API keys and subscriptions in `config/config.yaml`
2. **Test with real data**: Enable exchange connections and verify Kafka messages
3. **Monitor metrics**: Set up Prometheus to scrape `http://localhost:9090/metrics`
4. **Deploy to k3s**: Follow [K3s Deployment Guide](../deployment/K3S_DEPLOYMENT_GUIDE.md)

## Additional Resources

- [Dockerfile Reference](./DOCKERFILE_REFERENCE.md)
- [Configuration Guide](./CONFIGURATION_GUIDE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)

## Support

For issues and questions:

- Check existing [GitHub Issues](https://github.com/bmoscon/cryptofeed/issues)
- Review [troubleshooting guide](./TROUBLESHOOTING.md)
- Open a new issue with logs and configuration details

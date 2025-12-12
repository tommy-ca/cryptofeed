# Docker Deployment Guide

This directory contains documentation for building, securing, and deploying cryptofeed container images.

## Quick Start

### Build Container Image

```bash
docker build -t cryptofeed:latest .
```

### Run Security Scan

```bash
./scripts/trivy-scan.sh --image cryptofeed:latest
```

### Run Container Locally

```bash
docker run -d \
  --name cryptofeed \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd)/config/config.yaml:/config/config.yaml:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e BINANCE_API_KEY=${BINANCE_API_KEY} \
  -e BINANCE_API_SECRET=${BINANCE_API_SECRET} \
  cryptofeed:latest
```

### Check Health

```bash
curl http://localhost:8080/health
curl http://localhost:9090/metrics
```

## Documentation

- **[SECURITY.md](SECURITY.md)** - Security scanning, CVE remediation, and container hardening
- **Build Guide** - Coming soon (multi-stage Dockerfile optimization)
- **Deployment Guide** - Coming soon (Docker Compose and k3s deployment)

## Container Image Details

### Base Image

- **Image**: `python:3.11-slim-bookworm`
- **Size**: ~300MB (multi-stage optimized)
- **User**: non-root (UID 1001)

### Exposed Ports

- **8080**: Health check endpoint (`/health`, `/ready`)
- **9090**: Prometheus metrics endpoint (`/metrics`)

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker endpoints | Yes | - |
| `HEALTH_PORT` | Health check HTTP port | No | 8080 |
| `PROMETHEUS_MULTIPROC_DIR` | Metrics directory | No | /tmp/prometheus |
| `{EXCHANGE}_API_KEY` | Exchange API key | No | - |
| `{EXCHANGE}_API_SECRET` | Exchange API secret | No | - |

### Volume Mounts

- `/config/config.yaml` - Cryptofeed configuration file (read-only)
- `/config/proxy.yaml` - Proxy configuration (read-only)
- `/tmp` - Temporary files (writable)

## Security

### Image Scanning

All images are scanned with Trivy before deployment:

```bash
./scripts/trivy-scan.sh --image cryptofeed:latest --report trivy-report.json
```

**Scan fails if CRITICAL or HIGH severity CVEs detected.**

### Security Features

- Non-root user (UID 1001)
- Multi-stage build (no build tools in runtime image)
- Minimal base image (Debian slim)
- Read-only root filesystem (when deployed to k8s)
- Secrets via environment variables (never in image layers)

See [SECURITY.md](SECURITY.md) for complete security documentation.

## Build Workflow

### Local Build

```bash
# Build image
docker build -t cryptofeed:local .

# Scan for vulnerabilities
./scripts/trivy-scan.sh --image cryptofeed:local

# Run locally
docker run -d --name cryptofeed cryptofeed:local
```

### CI/CD Build (GitHub Actions example)

```yaml
name: Build and Scan

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t cryptofeed:${{ github.sha }} .

      - name: Run Trivy scan
        run: |
          ./scripts/trivy-scan.sh \
            --image cryptofeed:${{ github.sha }} \
            --report trivy-report.json

      - name: Upload scan report
        uses: actions/upload-artifact@v3
        with:
          name: trivy-report
          path: trivy-report.json
```

## Troubleshooting

### Image Build Fails

**Problem**: Docker build fails with "no space left on device"

**Solution**:
```bash
docker system prune -a
docker volume prune
```

### Security Scan Fails

**Problem**: Trivy scan detects CRITICAL CVEs

**Solution**: See [SECURITY.md - CVE Remediation Process](SECURITY.md#cve-remediation-process)

### Health Check Fails

**Problem**: Container starts but `/health` returns 503

**Solution**:
```bash
# Check container logs
docker logs cryptofeed

# Verify Kafka connectivity
docker exec cryptofeed curl -f http://localhost:8080/health
```

### Trivy Not Installed

**Problem**: `trivy: command not found`

**Solution**:
```bash
# macOS
brew install trivy

# Linux (Debian/Ubuntu)
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy
```

See [Trivy Installation Guide](https://aquasecurity.github.io/trivy/latest/getting-started/installation/)

## Next Steps

1. **Build and scan image**: `docker build -t cryptofeed:latest . && ./scripts/trivy-scan.sh --image cryptofeed:latest`
2. **Deploy with Docker Compose**: See deployment guide (coming soon)
3. **Deploy to k3s**: See Kubernetes deployment guide (coming soon)

## Support

- **Issues**: [GitHub Issues](https://github.com/cryptofeed/cryptofeed/issues)
- **Security**: See [SECURITY.md](SECURITY.md#reporting-security-issues)
- **Documentation**: [Main Docs](../../README.md)

# Implementation Tasks

## Overview

This task list implements the multi-exchange Docker deployment requirements, transforming cryptofeed from a locally-run Python application into containerized microservices deployable via Docker Compose (development) and k3s (production). The implementation focuses exclusively on Phase 1 functional requirements: container image build, Docker Compose orchestration, k3s basic deployment, configuration management, health checks, metrics exposure, multi-exchange integration, basic security, and documentation.

Advanced non-functional requirements (HPA, Grafana dashboards, CI/CD automation, External Secrets Operator, disaster recovery, multi-environment overlays) are explicitly deferred to Phase 2 or future enhancements. This FR-first approach delivers core deployment capabilities rapidly while maintaining simplicity and operational clarity.

The implementation follows an incremental progression: build container image → test locally with Docker Compose → deploy to k3s staging → integrate with Kafka and proxy system → harden security → document operations. Each task builds on previous outputs, ensuring continuous validation and zero orphaned code.

## Tasks

- [x] 1. Build production-ready container image with multi-stage optimization
  - Create multi-stage Dockerfile with builder stage and runtime stage
  - Use python:3.11-slim-bookworm as base image for Debian package compatibility
  - Install build dependencies (gcc, build-essential, librdkafka-dev) in builder stage only
  - Compile Python wheels for runtime dependencies (aiohttp, websockets, ccxt, pydantic, protobuf, confluent-kafka, prometheus_client)
  - Copy compiled wheels to runtime stage and install without build tools
  - Configure non-root user (UID 1001) with restricted permissions in runtime stage
  - Set Python environment variables (PYTHONUNBUFFERED=1, PYTHONDONTWRITEBYTECODE=1, PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus)
  - Define entrypoint using Python module execution supporting config file path via --config argument
  - Create .dockerignore file excluding tests, docs, .git, .venv, __pycache__, *.pyc
  - Verify final image size is under 500MB
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.9, 1.10_

- [x] 1.1 Implement health check HTTP server for k3s probes
  - Create aiohttp HTTP server exposing /health, /ready, and /metrics endpoints on port 8080
  - Integrate with existing KafkaHealthCheck from cryptofeed/backends/kafka/health.py
  - Aggregate health status from Kafka producer connection state and exchange WebSocket connections
  - Return JSON response with status (healthy/degraded/unhealthy), timestamp, component health details, and uptime_seconds
  - Define readiness criteria: Kafka producer initialized and connected, at least 1 exchange connection active, config.yaml loaded successfully
  - Define liveness criteria: HTTP server responsive within 1 second, no fatal exceptions in asyncio event loop, Kafka producer not in failed state
  - Support configurable health check port via HEALTH_PORT environment variable (default 8080)
  - Handle graceful shutdown on SIGTERM by closing WebSocket connections, flushing Kafka producer buffers, and exiting within 30 seconds
  - Test health endpoints return correct HTTP status codes (200 for healthy, 503 for unhealthy)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.8, 5.9_

- [x] 1.2 Integrate security scanning with Trivy
  - Add Trivy security scanner to Dockerfile build workflow
  - Configure Trivy to scan final runtime image for CVE vulnerabilities
  - Set scan to fail build if critical or high severity CVEs detected
  - Generate security scan report in JSON format with vulnerability details
  - Document CVE remediation process for updating base image or dependencies
  - Test Trivy scan passes for freshly built image with zero critical CVEs
  - _Requirements: 1.6, 8.5_

- [x] 1.3 Configure image tagging strategy
  - Define semantic version tags (latest, vX.Y.Z format aligned with Git tags)
  - Apply image metadata labels (version, build_timestamp, git_commit_sha)
  - Tag images with Git commit SHA for traceability (cryptofeed:commit-abc123)
  - Document image versioning convention in deployment guide
  - Test image builds tagged correctly with multiple tags
  - _Requirements: 1.8_

- [x] 2. Create Docker Compose development orchestration
  - Create docker-compose.yml with cryptofeed service and Kafka service in KRaft mode
  - Use apache/kafka-native:latest image for Kafka with KRaft configuration (no Zookeeper)
  - Define cryptofeed service with build context pointing to Dockerfile
  - Configure environment variables for Kafka bootstrap servers (kafka:9092), exchange API keys, proxy URLs
  - Setup volume mounts for config.yaml and proxy.yaml as read-only volumes
  - Configure service dependencies with depends_on ensuring Kafka starts before cryptofeed
  - Define Kafka healthcheck using kafka-broker-api-versions command with 10s interval
  - Define cryptofeed healthcheck using curl to /health endpoint with 30s interval and 40s start_period
  - Expose ports: 8080 (health), 9090 (metrics), 9092 (Kafka)
  - Apply resource limits: cryptofeed (1 CPU, 2GB RAM), Kafka (512MB RAM)
  - Create .env.example file documenting required environment variables (BINANCE_API_KEY, COINBASE_API_KEY, etc.)
  - Test docker-compose up completes startup in under 60 seconds
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

- [x] 2.1 Configure proxy system integration in Docker Compose
  - Mount proxy.yaml configuration file from host ./config/proxy.yaml to container /config/proxy.yaml
  - Define PROXY_HTTP and PROXY_SOCKS5 environment variables in docker-compose.yml
  - Configure per-exchange proxy overrides in proxy.yaml using existing ProxySettings schema
  - Test proxy configuration loaded correctly via cryptofeed proxy system
  - Validate geo-restricted exchange connectivity (e.g., Binance US) using configured proxies
  - _Requirements: 2.10, 7.3, 7.7_

- [x] 2.2 Create integration tests for Docker Compose stack
  - Write pytest test verifying docker-compose up brings all services to healthy state
  - Test Kafka broker reachable at localhost:9092 using kafka-python client
  - Test cryptofeed /health endpoint returns 200 OK at localhost:8080/health
  - Test /metrics endpoint returns Prometheus text format at localhost:9090/metrics
  - Test graceful shutdown with docker-compose down completes within 30 seconds
  - Test volume mounts readable inside cryptofeed container
  - _Requirements: 2.4, 2.6, 2.9_

- [ ] 3. Build k3s production deployment manifests
  - Create Deployment manifest for cryptofeed in namespace cryptofeed-prod
  - Configure replicas: 3 for high availability
  - Define RollingUpdate strategy with maxUnavailable: 1 and maxSurge: 1
  - Set resource requests: 500m CPU, 1Gi memory per pod
  - Set resource limits: 2000m CPU, 4Gi memory per pod
  - Configure readiness probe: HTTP GET /ready port 8080, initialDelaySeconds 10, periodSeconds 10, failureThreshold 3
  - Configure liveness probe: HTTP GET /health port 8080, initialDelaySeconds 30, periodSeconds 30, failureThreshold 3
  - Configure startup probe: HTTP GET /health port 8080, initialDelaySeconds 0, periodSeconds 5, failureThreshold 12 (60s max startup)
  - Define preStop lifecycle hook with 10 second sleep for connection draining
  - Set terminationGracePeriodSeconds: 30
  - Add pod labels app=cryptofeed and exchange={exchange_name}
  - Add pod annotations for Prometheus scraping (prometheus.io/scrape: true, prometheus.io/port: 9090, prometheus.io/path: /metrics)
  - Create ClusterIP Service exposing port 8080 (http) and 9090 (metrics)
  - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.7_

- [ ] 3.1 Create k3s namespace and RBAC resources
  - Create Namespace manifest for cryptofeed-prod
  - Create ServiceAccount cryptofeed in cryptofeed-prod namespace
  - Create Role with read-only access to ConfigMaps and Secrets in cryptofeed-prod namespace
  - Create RoleBinding associating ServiceAccount with Role
  - Configure Deployment to use ServiceAccount cryptofeed
  - Test kubectl apply creates namespace and RBAC resources successfully
  - _Requirements: 3.6_

- [ ] 3.2 Configure k3s Service with built-in ServiceLB
  - Create ClusterIP Service for internal pod-to-pod communication
  - Document k3s ServiceLB (Klipper) usage for LoadBalancer type services if external access needed
  - Configure service selector matching Deployment pod labels (app=cryptofeed)
  - Define service ports: http (8080), metrics (9090)
  - Test service DNS resolution from within k3s cluster (cryptofeed.cryptofeed-prod.svc.cluster.local)
  - _Requirements: 3.8, 3.9, 5.9, 5.10_

- [ ] 3.3 Create k3s deployment integration tests
  - Write test applying Deployment manifest to Kind or k3s test cluster
  - Verify pods reach Running state within 60 seconds
  - Verify readiness probes pass and pods marked Ready
  - Test rolling update by changing image tag and verifying zero downtime
  - Test pod restart on liveness probe failure by killing health endpoint
  - Test graceful shutdown by deleting pod and verifying SIGTERM handling
  - _Requirements: 3.5, 3.3_

- [ ] 4. Implement configuration management with ConfigMaps and Secrets
  - Create ConfigMap manifest cryptofeed-config storing config.yaml content
  - Define ConfigMap data fields: exchanges (list), symbols (per-exchange dict), kafka.bootstrap_servers, kafka.acks, kafka.compression_type, proxy.http, proxy.socks5, log_level
  - Mark production ConfigMaps as immutable: true to prevent accidental changes
  - Create Secret manifest cryptofeed-api-keys storing exchange API keys
  - Define Secret data fields: BINANCE_API_KEY, BINANCE_API_SECRET, COINBASE_API_KEY, COINBASE_API_SECRET, KAFKA_SASL_USERNAME, KAFKA_SASL_PASSWORD (base64 encoded)
  - Configure Deployment to mount ConfigMap as volume at /config/config.yaml
  - Configure Deployment to inject Secret values as environment variables
  - Document hot-reload strategy: production requires new ConfigMap + Deployment rolling update; development can use mutable ConfigMaps with kubelet sync delay
  - Test ConfigMap and Secret mounting works correctly in pod
  - Test pod restart triggered by ConfigMap/Secret update (via Deployment annotation change)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.9, 4.10_

- [ ] 4.1 Create configuration validation schema
  - Define JSON schema for config.yaml validating exchange names, Kafka endpoints, proxy URLs
  - Implement validation script rejecting invalid exchange names (must be in supported exchange list)
  - Validate Kafka bootstrap_servers format (comma-separated host:port)
  - Validate proxy URLs conform to http:// or socks5:// schemes
  - Fail fast on startup if configuration validation fails with descriptive error message
  - Test validation script catches invalid configurations
  - _Requirements: 4.7_

- [ ] 4.2 Document secrets management best practices
  - Document Kubernetes native Secrets encryption at rest (etcd encryption configuration)
  - Document RBAC configuration restricting Secret access to cryptofeed ServiceAccount only
  - Document secret rotation procedure: create new Secret, update Deployment to reference new Secret, trigger rolling update, delete old Secret
  - Document .env file usage for Docker Compose with .gitignore exclusion
  - Document never logging API keys, secrets, or passphrases in plain text
  - _Requirements: 4.8, 8.2, 8.3, 8.4_

- [ ] 5. Extend Prometheus metrics exporter for container observability
  - Extend existing PrometheusMetricsExporter in cryptofeed/backends/kafka/metrics.py
  - Add container-specific labels to all metrics: pod_name, namespace, node_name
  - Configure prometheus_client multiprocess mode using PROMETHEUS_MULTIPROC_DIR environment variable
  - Expose /metrics endpoint at port 9090 in health check HTTP server
  - Emit existing Kafka producer metrics: messages_produced_total, produce_latency_seconds (histogram), produce_errors_total
  - Emit WebSocket connection metrics: active_connections, reconnection_attempts, connection_uptime_seconds
  - Test metrics endpoint returns Prometheus text exposition format
  - Test metrics include pod_name, namespace, node_name labels from Kubernetes downward API
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.10_

- [ ] 5.1 Configure Prometheus scrape configuration for k3s
  - Create Prometheus scrape_configs using kubernetes_sd_configs with role: pod
  - Filter namespaces to cryptofeed-prod only
  - Add relabel_configs extracting prometheus.io/scrape, prometheus.io/port, prometheus.io/path annotations
  - Add relabel_configs extracting pod_name, namespace, node_name from Kubernetes metadata
  - Test Prometheus successfully scrapes cryptofeed pods and metrics appear in Prometheus UI
  - Document Prometheus configuration in deployment guide
  - _Requirements: 6.6, 6.8_

- [ ] 5.2 Add structured JSON logging for ELK/Loki integration
  - Configure cryptofeed logging to emit JSON format with fields: level, timestamp, message, exchange, symbol, pod_name, namespace
  - Use existing loguru integration for structured logging
  - Ensure no API keys or secrets logged in plain text
  - Test logs parseable by JSON parser
  - Document log aggregation integration patterns for ELK stack or Loki
  - _Requirements: 6.9_

- [ ] 6. Test metrics exposure and collection
  - Write integration test verifying /metrics endpoint accessible from k3s pod
  - Test Prometheus scrapes metrics successfully and metrics appear in query results
  - Test metrics include expected labels (exchange, symbol, pod_name, namespace)
  - Test counter metrics increment over time (messages_produced_total)
  - Test histogram metrics capture latency buckets (produce_latency_seconds)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.10_

- [ ] 7. Integrate Kafka Protobuf backend and proxy system
  - Configure cryptofeed to load Kafka bootstrap servers from ConfigMap environment variable KAFKA_BOOTSTRAP_SERVERS
  - Enable Protobuf serialization for Trade, OrderBook, Ticker, Funding events using existing protobuf-callback-serialization implementation
  - Configure KafkaCallback with idempotent producer settings (enable.idempotence=true, acks=all)
  - Setup consolidated topic publishing (cryptofeed.trades, cryptofeed.l2_book, cryptofeed.ticker, cryptofeed.funding) with exchange/symbol in message headers
  - Configure partition strategy selection via KAFKA_PARTITION_STRATEGY environment variable (composite, symbol, exchange, round-robin)
  - Load proxy configuration from ConfigMap and initialize ProxyInjector with per-exchange HTTP/SOCKS5 proxy URLs
  - Inject exchange API keys from Secret environment variables into cryptofeed Exchange initialization
  - Implement exponential backoff retry on Kafka connection failures with maximum 10 attempts before marking container unhealthy
  - Test Kafka producer publishes Protobuf messages successfully
  - Test proxy system routes exchange connections through configured proxies
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_

- [ ] 7.1 Create multi-exchange integration tests
  - Write test deploying cryptofeed with 3+ exchanges (Binance, Coinbase, Kraken)
  - Test Kafka consumer receives Protobuf messages from all exchanges
  - Test message headers contain exchange, symbol, data_type fields
  - Test partition strategy distributes messages correctly across Kafka partitions
  - Test proxy configuration applied per exchange (validate via connection logs)
  - Test Docker Compose and k3s deployments both work with multi-exchange configuration
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.10_

- [ ] 8. Implement container security hardening
  - Configure Dockerfile USER directive setting UID 1001 and non-root user cryptofeed
  - Configure SecurityContext in Deployment manifest: runAsNonRoot: true, runAsUser: 1001, fsGroup: 1001
  - Configure SecurityContext: readOnlyRootFilesystem: true with writable /tmp volume using emptyDir
  - Configure SecurityContext: allowPrivilegeEscalation: false
  - Drop all Linux capabilities in SecurityContext (capabilities.drop: ["ALL"])
  - Verify Dockerfile excludes API keys and credentials from all image layers
  - Test Secret mounting as environment variables from Kubernetes Secrets
  - Test pod fails to start if runAsRoot attempted
  - Document RBAC configuration for least-privilege ServiceAccount access
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6, 8.7, 8.9_

- [ ] 8.1 Create security validation tests
  - Write test verifying container runs as non-root user (UID 1001)
  - Test root filesystem is read-only except /tmp
  - Test Trivy security scan passes with zero critical CVEs
  - Test Secret values not logged in container logs
  - Test RBAC prevents access to Secrets in other namespaces
  - _Requirements: 8.1, 8.5, 8.6, 8.8, 8.9_

- [ ] 9. Create comprehensive deployment documentation
  - Write Docker Compose quickstart guide with step-by-step instructions (setup in under 5 minutes)
  - Write k3s deployment guide covering k3s installation, namespace creation, ConfigMap/Secret creation, Deployment application
  - Create troubleshooting runbook for common issues: container crash loops, Kafka connection failures, proxy connection errors, health check failures
  - Create architecture diagrams showing container flow (Exchange API → Cryptofeed → Kafka), k3s pod architecture, Prometheus metrics flow
  - Provide sample ConfigMaps for 3+ exchanges (Binance, Coinbase, Kraken) with inline comments explaining each field
  - Document k3s built-in components (local-path provisioner, ServiceLB/Klipper, Traefik ingress, metrics-server)
  - Document health check endpoints (/health, /ready) with expected responses and status codes
  - Document available Prometheus metrics with descriptions (messages_produced_total, produce_latency_seconds, kafka_broker_latency_seconds, etc.)
  - Document secrets management with Kubernetes Secrets, .env files for Docker Compose, RBAC configuration
  - Document container security features (non-root user, read-only filesystem, dropped capabilities)
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10_

- [ ] 9.1 Create deployment architecture diagrams
  - Create diagram showing Docker Compose architecture (cryptofeed container, Kafka KRaft container, volume mounts, port mappings)
  - Create diagram showing k3s deployment architecture (Deployment, Service, ConfigMap, Secret, Prometheus scraping)
  - Create diagram showing data flow (Exchange WebSocket → Cryptofeed → Proxy → Kafka → Consumers)
  - Create diagram showing health check probe flow (k3s → /health endpoint → KafkaHealthCheck → response)
  - Export diagrams as PNG/SVG and embed in documentation
  - _Requirements: 9.4_

## Requirements Coverage Summary

All 9 Phase 1 requirements are mapped to implementation tasks:

- **Requirement 1**: Container Image Build System → Tasks 1, 1.1, 1.2, 1.3
- **Requirement 2**: Docker Compose Development Orchestration → Tasks 2, 2.1, 2.2
- **Requirement 3**: k3s Production Deployment → Tasks 3, 3.1, 3.2, 3.3
- **Requirement 4**: Configuration Management → Tasks 4, 4.1, 4.2
- **Requirement 5**: Health Checks and Service Discovery → Tasks 1.1, 3.2
- **Requirement 6**: Metrics Exposure → Tasks 5, 5.1, 5.2, 6
- **Requirement 7**: Multi-Exchange Integration with Kafka and Proxy → Tasks 7, 7.1
- **Requirement 8**: Container Security and Basic Secrets Management → Tasks 8, 8.1, 4.2
- **Requirement 9**: Deployment Documentation and Operational Guides → Tasks 9, 9.1

**Total**: 9 major tasks with 14 sub-tasks (23 total tasks), covering all 9 Phase 1 functional requirements with incremental progression from local development to production deployment.

**Deferred to Phase 2 (NFRs - NO TASKS IN THIS PLAN)**:
- Horizontal Pod Autoscaler (HPA)
- Advanced observability (Grafana dashboards, alerting)
- CI/CD automation (GitHub Actions, ArgoCD, Helm charts)
- Zero-downtime rolling updates (canary, blue-green)
- External Secrets Operator integration
- Disaster recovery and multi-region deployment
- Multi-environment configuration overlays (Kustomize, Helm)

These Phase 2 enhancements are documented in requirements.md "Future Enhancements (NFRs - Phase 2)" section and will be implemented after Phase 1 functional requirements are delivered and validated in production.

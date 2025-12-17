# Requirements Document

## Introduction

This specification defines the requirements for containerizing cryptofeed as Docker-based microservices capable of ingesting market data from 40+ cryptocurrency exchanges (Binance, Coinbase, Kraken, OKX, Bybit, etc.) across multiple product types (spot, futures, perpetuals, options) into a Kafka Protobuf backend. The deployment architecture prioritizes Docker Compose for local development and k3s (lightweight Kubernetes) for staging and production environments.

The containerization strategy aligns with cryptofeed's ingestion-layer-only architecture: containers ingest and normalize exchange data, serialize to Protobuf, and publish to Kafka topics. Storage, analytics, and retention remain the responsibility of downstream consumers (Flink, DuckDB, Spark, etc.). This separation of concerns enables flexible deployment models ranging from single-container multi-exchange configurations to horizontally scaled per-exchange isolation with dedicated resource limits.

**Platform Strategy:**
- **Docker Compose**: Primary local development environment with minimal setup (< 5 minutes)
- **k3s (Lightweight Kubernetes)**: Staging and production deployments with simplified operations
  - k3s is a certified Kubernetes distribution optimized for resource-constrained environments
  - Single binary installation (~512MB RAM vs 2GB+ for full Kubernetes)
  - Built-in components: local-path storage provisioner, ServiceLB load balancer, Traefik ingress
  - Production-grade but designed for edge, IoT, CI, and development use cases

**Functional Requirements First (FRs Over NFRs):**
This specification prioritizes functional requirements that deliver core deployment capabilities:
- Container image build and deployment
- Docker Compose orchestration for local development
- k3s basic deployment (Deployments, Services, ConfigMaps, Secrets)
- Health checks and metrics exposure
- Multi-exchange integration with Kafka and proxy system
- Basic security (non-root containers, secret management)
- Deployment documentation

Advanced non-functional requirements (NFRs) are deferred to Phase 2 or marked as future enhancements:
- Horizontal Pod Autoscaler (HPA)
- Advanced observability (Grafana dashboards, alerts)
- CI/CD automation
- Zero-downtime rolling updates (basic rolling updates included as FR)
- Disaster recovery and backup strategies
- Advanced security (External Secrets Operator, network policies)

**Key Deployment Models:**
- **Single Container Multi-Exchange**: Vertical scaling for low-volume environments
- **Per-Exchange Containers**: Horizontal scaling with exchange-level isolation
- **Exchange Groups**: Resource pooling for related exchanges

**Dependencies:**
- market-data-kafka-producer (Kafka backend integration)
- proxy-system-complete (geo-restriction handling)
- protobuf-callback-serialization (message serialization)
- normalized-data-schema-crypto (data schemas)

## Requirements

### Requirement 1: Container Image Build System
**Objective:** As a platform engineer, I want a production-ready Dockerfile with multi-stage build optimization, so that cryptofeed containers are small, secure, and deployable across environments.

#### Acceptance Criteria
1. WHEN Dockerfile is built THEN Container Image Build System SHALL produce an image smaller than 500MB with Python 3.11+ runtime
2. WHEN multi-stage build executes THEN Container Image Build System SHALL separate build dependencies from runtime dependencies in distinct stages
3. IF production build is triggered THEN Container Image Build System SHALL install only runtime dependencies (aiohttp, websockets, ccxt, pydantic, protobuf, kafka-python)
4. WHEN base image is selected THEN Container Image Build System SHALL use official python:3.11-slim-bookworm or python:3.11-alpine as base
5. WHERE cryptofeed package is installed THE Container Image Build System SHALL install package in editable mode from source directory
6. WHEN security scanning is performed THEN Container Image Build System SHALL produce images with zero critical CVE vulnerabilities
7. IF Docker buildx is available THEN Container Image Build System SHALL support multi-architecture builds (linux/amd64, linux/arm64)
8. WHEN image is tagged THEN Container Image Build System SHALL apply semantic version tags (latest, vX.Y.Z, vX.Y, vX)
9. WHERE configuration files are needed THE Container Image Build System SHALL copy config.yaml template and proxy configuration examples
10. WHEN entrypoint is defined THEN Container Image Build System SHALL use Python script or shell script supporting environment variable configuration

### Requirement 2: Docker Compose Development Orchestration
**Objective:** As a developer, I want Docker Compose orchestration for local development and staging environments, so that I can test multi-exchange deployments with minimal setup and iteration speed.

#### Acceptance Criteria
1. WHEN docker-compose.yml is executed THEN Docker Compose Development Orchestration SHALL start cryptofeed service with 3+ exchange configurations
2. IF Kafka backend is required THEN Docker Compose Development Orchestration SHALL include Kafka broker and Zookeeper services with health checks
3. WHERE environment variables are needed THE Docker Compose Development Orchestration SHALL define variables for exchange API keys, proxy URLs, and Kafka endpoints
4. WHEN services start THEN Docker Compose Development Orchestration SHALL ensure dependency order (Zookeeper → Kafka → cryptofeed)
5. IF volume mounts are configured THEN Docker Compose Development Orchestration SHALL mount local config.yaml and proxy configuration files
6. WHEN logs are accessed THEN Docker Compose Development Orchestration SHALL expose container logs via docker-compose logs command
7. WHERE service discovery is needed THE Docker Compose Development Orchestration SHALL use Docker networking with service names as DNS entries
8. IF resource limits are specified THEN Docker Compose Development Orchestration SHALL apply CPU and memory limits per service (e.g., 1 CPU, 2GB RAM)
9. WHEN development environment starts THEN Docker Compose Development Orchestration SHALL complete startup in less than 60 seconds
10. WHERE proxy system integration is needed THE Docker Compose Development Orchestration SHALL configure per-exchange HTTP/SOCKS5 proxy settings

### Requirement 3: k3s Production Deployment
**Objective:** As a platform engineer, I want k3s manifests for staging and production deployment with basic pod management, so that cryptofeed runs reliably in lightweight Kubernetes environments.

#### Acceptance Criteria
1. WHEN k3s Deployment manifest is applied THEN k3s Production Deployment SHALL create cryptofeed pods with replica count ≥ 1
2. IF StatefulSet is required THEN k3s Production Deployment SHALL use StatefulSet for exchanges requiring persistent state or ordered deployments
3. WHEN basic rolling update is triggered THEN k3s Production Deployment SHALL apply RollingUpdate strategy with maxUnavailable: 1 and maxSurge: 1
4. IF resource requests are defined THEN k3s Production Deployment SHALL set requests (0.5 CPU, 1GB RAM) per pod
5. WHEN pod lifecycle is managed THEN k3s Production Deployment SHALL configure readiness and liveness probes with HTTP or TCP checks
6. WHERE namespace isolation is needed THE k3s Production Deployment SHALL deploy cryptofeed to dedicated namespace (e.g., cryptofeed-prod)
7. IF service exposure is required THEN k3s Production Deployment SHALL create ClusterIP Service for internal communication
8. WHEN k3s built-in components are used THEN k3s Production Deployment SHALL leverage local-path provisioner for storage if needed
9. WHERE load balancing is required THE k3s Production Deployment SHALL use k3s ServiceLB (Klipper) for internal load balancing
10. IF ingress is configured THEN k3s Production Deployment SHALL use k3s built-in Traefik for HTTP ingress

### Requirement 4: Configuration Management
**Objective:** As a DevOps engineer, I want declarative configuration management via ConfigMaps and environment variables, so that exchange and proxy settings can be updated without rebuilding images.

#### Acceptance Criteria
1. WHEN ConfigMap is created THEN Configuration Management SHALL store config.yaml with exchange subscriptions, symbols, and Kafka endpoints
2. IF environment variables are injected THEN Configuration Management SHALL support per-exchange API keys via Kubernetes Secrets or Docker secrets
3. WHERE proxy configuration is needed THE Configuration Management SHALL define proxy URLs, pools, and per-exchange overrides in ConfigMap
4. WHEN configuration is updated THEN Configuration Management SHALL trigger pod restart within 30 seconds
5. IF multiple environments are managed THEN Configuration Management SHALL support environment-specific ConfigMaps (dev, staging, prod)
6. WHEN secrets are managed THEN Configuration Management SHALL use Kubernetes native secrets for k3s deployments
7. WHERE validation is required THE Configuration Management SHALL reject invalid configuration with descriptive error messages
8. IF sensitive data is handled THEN Configuration Management SHALL never log API keys, secrets, or passphrases in plain text
9. WHEN exchange-specific settings are defined THEN Configuration Management SHALL support per-exchange rate limits, timeout values, and retry policies
10. WHERE Docker Compose is used THE Configuration Management SHALL support .env files for local secret management excluded from version control

### Requirement 5: Health Checks and Service Discovery
**Objective:** As a reliability engineer, I want automated service discovery and health checks, so that unhealthy containers are detected and restarted automatically.

#### Acceptance Criteria
1. WHEN cryptofeed container starts THEN Health Checks and Service Discovery SHALL expose HTTP health endpoint (e.g., /health) returning 200 OK when healthy
2. IF k3s is used THEN Health Checks and Service Discovery SHALL configure readiness probe checking /ready endpoint with initialDelaySeconds: 10
3. WHERE liveness probe is configured THE Health Checks and Service Discovery SHALL configure probe checking /health endpoint with periodSeconds: 30 and failureThreshold: 3
4. WHEN health check fails THEN Health Checks and Service Discovery SHALL trigger container restart by k3s or Docker Compose
5. IF graceful shutdown is required THEN Health Checks and Service Discovery SHALL handle SIGTERM signal by closing WebSocket connections and flushing Kafka buffers within 30 seconds
6. WHEN startup probe is configured THEN Health Checks and Service Discovery SHALL allow up to 60 seconds for cryptofeed to initialize before liveness checks begin
7. WHERE dependencies are unavailable THE Health Checks and Service Discovery SHALL report degraded status without failing liveness probe
8. IF health check endpoint is accessed THEN Health Checks and Service Discovery SHALL return JSON payload with status, exchange connection states, and Kafka producer health
9. WHEN DNS-based discovery is used THEN Health Checks and Service Discovery SHALL register cryptofeed service with internal DNS (Docker network or k3s DNS)
10. WHERE Docker Compose is used THE Health Checks and Service Discovery SHALL support depends_on with health check conditions

### Requirement 6: Metrics Exposure
**Objective:** As a platform engineer, I want Prometheus metrics exposure, so that I can monitor cryptofeed performance and troubleshoot issues.

#### Acceptance Criteria
1. WHEN cryptofeed container runs THEN Metrics Exposure SHALL expose Prometheus metrics endpoint at /metrics
2. IF Prometheus scraping is configured THEN Metrics Exposure SHALL emit metrics for message ingestion rate (messages/second per exchange)
3. WHERE Kafka integration is active THE Metrics Exposure SHALL emit metrics for Kafka producer throughput, latency (p50, p99), and error rate
4. WHEN WebSocket connections are active THEN Metrics Exposure SHALL emit metrics for connection count, reconnection attempts, and connection uptime
5. IF resource usage is tracked THEN Metrics Exposure SHALL emit metrics for CPU usage, memory usage, and network I/O per container
6. WHEN k3s is used THEN Metrics Exposure SHALL expose metrics in Prometheus format compatible with k3s built-in metrics-server
7. WHERE custom metrics are needed THE Metrics Exposure SHALL support user-defined Prometheus gauges and counters via callback hooks
8. IF Docker Compose is used THEN Metrics Exposure SHALL make metrics accessible via container port mapping
9. WHEN logs are collected THEN Metrics Exposure SHALL emit structured JSON logs compatible with ELK stack or Loki
10. WHERE metrics scraping is configured THE Metrics Exposure SHALL annotate k3s pods with prometheus.io/scrape and prometheus.io/port annotations

### Requirement 7: Multi-Exchange Integration with Kafka and Proxy System
**Objective:** As a platform engineer, I want seamless integration with market-data-kafka-producer and proxy-system-complete, so that containerized cryptofeed publishes Protobuf-serialized messages to Kafka via configured proxies.

#### Acceptance Criteria
1. WHEN cryptofeed container starts THEN Multi-Exchange Integration SHALL connect to Kafka broker using endpoints from ConfigMap or environment variables
2. IF Protobuf serialization is enabled THEN Multi-Exchange Integration SHALL serialize Trade, OrderBook, Ticker, and Funding events to Protobuf format before Kafka publish
3. WHERE proxy system is configured THE Multi-Exchange Integration SHALL load proxy configuration from ConfigMap and apply per-exchange HTTP/SOCKS5 proxies
4. WHEN Kafka producer is initialized THEN Multi-Exchange Integration SHALL use idempotent producer configuration for message reliability
5. IF topic management is required THEN Multi-Exchange Integration SHALL publish to consolidated topics (e.g., cryptofeed.trades, cryptofeed.l2_book) with exchange/symbol in message headers
6. WHEN partition strategy is selected THEN Multi-Exchange Integration SHALL support configurable partitioning (composite, symbol-based, exchange-based, round-robin)
7. WHERE proxy pools are configured THE Multi-Exchange Integration SHALL rotate proxies per exchange using pool-aware selection strategies
8. IF authentication is required THEN Multi-Exchange Integration SHALL inject exchange API keys from Kubernetes Secrets or Docker secrets into cryptofeed configuration
9. WHEN Kafka connection fails THEN Multi-Exchange Integration SHALL implement exponential backoff retry with maximum 10 attempts before marking container unhealthy
10. WHERE Docker Compose is used THE Multi-Exchange Integration SHALL configure Kafka broker hostname as service name (e.g., kafka:9092)

### Requirement 8: Container Security and Basic Secrets Management
**Objective:** As a security engineer, I want secure container runtime and basic secrets management, so that API keys and credentials are never exposed in logs or image layers.

#### Acceptance Criteria
1. WHEN container image is built THEN Container Security SHALL run as non-root user (UID > 1000)
2. IF Dockerfile is created THEN Container Security SHALL never include API keys, secrets, or credentials in image layers
3. WHERE k3s is used THE Container Security SHALL mount secrets as environment variables or volume mounts (never in ConfigMaps)
4. WHEN Docker Compose is used THEN Container Security SHALL use Docker secrets or .env files excluded from version control
5. IF security scanning is performed THEN Container Security SHALL pass Trivy or Clair vulnerability scans with zero high/critical CVEs
6. WHEN runtime security is enforced THEN Container Security SHALL apply read-only root filesystem with writable /tmp volume
7. WHERE minimal capabilities are needed THE Container Security SHALL drop all Linux capabilities except required ones (NET_BIND_SERVICE if needed)
8. IF image is distributed THEN Container Security SHALL sign container images with Docker Content Trust or cosign
9. WHEN k3s deployment is created THEN Container Security SHALL set securityContext with runAsNonRoot: true and allowPrivilegeEscalation: false
10. WHERE secrets rotation is considered THE Container Security SHALL support manual secret rotation via ConfigMap/Secret updates triggering pod restart

### Requirement 9: Deployment Documentation and Operational Guides
**Objective:** As a platform engineer, I want comprehensive deployment documentation and operational guides, so that team members can deploy, operate, and troubleshoot cryptofeed containers without deep expertise.

#### Acceptance Criteria
1. WHEN documentation is published THEN Deployment Documentation SHALL provide quickstart guide deploying cryptofeed to Docker Compose in < 5 minutes
2. IF k3s deployment is documented THEN Deployment Documentation SHALL provide step-by-step guide deploying to k3s with kubectl
3. WHERE troubleshooting is needed THE Deployment Documentation SHALL provide runbook for common issues (container crash loops, Kafka connection failures, proxy errors)
4. WHEN architecture is explained THEN Deployment Documentation SHALL include architecture diagrams showing container flow, Kafka integration, and proxy system
5. IF configuration examples are provided THEN Deployment Documentation SHALL include sample ConfigMaps for 3+ exchanges with comments explaining each field
6. WHEN k3s specifics are documented THEN Deployment Documentation SHALL explain k3s built-in components (local-path provisioner, ServiceLB, Traefik, metrics-server)
7. WHERE Docker Compose is documented THE Deployment Documentation SHALL provide example docker-compose.yml with Kafka, Zookeeper, and cryptofeed services
8. IF health checks are documented THEN Deployment Documentation SHALL explain /health and /ready endpoint usage and expected responses
9. WHEN metrics are documented THEN Deployment Documentation SHALL list available Prometheus metrics and their meanings
10. WHERE security is documented THE Deployment Documentation SHALL explain secrets management, non-root containers, and basic hardening practices

## Future Enhancements (NFRs - Phase 2)

The following requirements are deferred to Phase 2 or marked as future enhancements to maintain FR-first delivery:

### Future: Advanced Observability (NFR)
- Grafana dashboards with pre-built panels for ingestion rate, Kafka lag, error rate, and resource usage
- Prometheus alert rules for high error rate (>1%), Kafka lag (>10000 messages), and container restarts
- Distributed tracing integration with OpenTelemetry or Jaeger
- Advanced log aggregation with ELK stack or Loki integration

### Future: Horizontal Pod Autoscaling (NFR)
- HPA based on CPU utilization (target: 70%) with scaling between 1 and 50 replicas
- Custom metrics autoscaling based on Kafka producer lag or ingestion rate
- Vertical pod autoscaling (VPA) for dynamic resource adjustment
- Scale-to-zero for inactive exchanges during off-market hours

### Future: Advanced Resource Management (NFR)
- Per-exchange resource limits and tier-based profiles (tier-1: 2 CPU/4GB, tier-2: 1 CPU/2GB)
- Pod priority classes and preemption policies for critical exchanges
- Namespace-level resource quotas and limit ranges
- Pod disruption budgets (PDB) ensuring minimum availability during disruptions

### Future: Zero-Downtime Rolling Updates (NFR)
- Advanced RollingUpdate strategy with maxUnavailable: 0 and canary deployments
- Blue-green deployment support with traffic switching
- Automated rollback on failed health checks
- Pre-stop hooks for graceful shutdown with connection draining

### Future: CI/CD Automation (NFR)
- Automated Docker image build on code commits to main branch
- Container registry integration (Docker Hub, ECR, GCR) with semantic versioning
- Automated deployment to staging environment with integration tests
- GitOps integration with ArgoCD or Flux for declarative deployments
- Smoke tests validating critical exchange connections post-deployment

### Future: Advanced Security (NFR)
- Integration with HashiCorp Vault or AWS Secrets Manager for dynamic secret injection
- External Secrets Operator for Kubernetes secret synchronization
- Network policies restricting egress traffic to Kafka brokers and exchange API endpoints only
- AppArmor or SELinux security profiles restricting container capabilities
- Automated secret rotation without container restarts

### Future: Disaster Recovery and High Availability (NFR)
- Multi-region deployment with image and manifest replication
- Backup and restore procedures for ConfigMaps, Secrets, and StatefulSet volumes
- Quarterly disaster recovery drills validating RTO (15 min) and RPO (0 data loss)
- Blue-green cluster migration with DNS or load balancer switchover
- k3s etcd backup to off-cluster storage (S3, GCS)

### Future: Multi-Environment Configuration Overlays (NFR)
- Kustomize overlays for base, dev, staging, and prod environments
- Helm charts with environment-specific values files
- Environment-specific resource limits, exchange lists, and Kafka endpoints
- Automated configuration validation and drift detection

---
status: pending
priority: p2
issue_id: "001"
tags: [docker, implementation, code-review, multi-exchange-docker-deployment]
dependencies: []
---

# Missing cryptofeed.run Module Implementation

The `docker-compose.yml` and `Dockerfile` reference `python -m cryptofeed.run --config /config/config.yaml` but this module doesn't exist in the codebase yet.

## Problem Statement

The Docker container cannot start because the entry point `cryptofeed.run` module is not implemented. This blocks Task 2 testing and validation.

**Impact:**
- Docker Compose stack cannot run successfully
- Integration tests will fail
- Quick start guide instructions don't work
- Task 2 is incomplete without a working entry point

## Findings

**Missing implementation:**
- `cryptofeed/run.py` does not exist (checked with file glob)
- Dockerfile CMD references: `python -m cryptofeed.run --config /config/config.yaml` (line 131)
- No alternative entry point module exists

**Required functionality:**
1. Load configuration from YAML file
2. Parse command-line arguments (--config flag)
3. Initialize health check HTTP server
4. Initialize exchange connections based on config
5. Setup Kafka backend callbacks
6. Handle graceful shutdown on SIGTERM
7. Start asyncio event loop

**Evidence:**
- `Dockerfile:131` - CMD references `cryptofeed.run`
- `docker-compose.yml:129` - No override provided
- `config/config.yaml` exists but no loader

## Proposed Solutions

### Option 1: Create Minimal cryptofeed.run Module

**Approach:** Implement basic module with configuration loading, health server, and graceful shutdown.

**Pros:**
- Simple implementation (< 200 LOC)
- Follows existing cryptofeed patterns
- Can leverage existing components (health_server.py already created)
- Allows Docker Compose testing immediately

**Cons:**
- Minimal functionality initially
- May need iteration for full feature support

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Create Full-Featured cryptofeed.run with All Integrations

**Approach:** Complete implementation with all features (exchanges, Kafka, proxy, metrics, logging).

**Pros:**
- Production-ready immediately
- Comprehensive feature coverage
- Fully tested integration

**Cons:**
- Larger scope (6-8 hours)
- Risk of scope creep
- Blocks Task 2 completion longer

**Effort:** 6-8 hours

**Risk:** Medium

---

### Option 3: Use Existing Example as Entry Point

**Approach:** Adapt an existing example script to work as the module entry point.

**Pros:**
- Leverages proven working code
- Quick path to working solution

**Cons:**
- Examples may not have all required features
- May require significant refactoring
- Less clean architecture

**Effort:** 1-2 hours

**Risk:** Medium

## Recommended Action

**To be filled during triage.**

Recommended: **Option 1** - Create minimal cryptofeed.run module focusing on:
1. Configuration loading (YAML parser)
2. Health server integration (already implemented in cryptofeed/health_server.py)
3. Graceful shutdown handlers
4. Basic exchange initialization
5. Kafka backend setup

This provides quickest path to working Docker Compose stack while maintaining quality.

## Technical Details

**Files to create:**
- `cryptofeed/run.py` - Main entry point module
- `cryptofeed/__main__.py` - Python module runner (optional, for `python -m cryptofeed.run`)

**Integration points:**
- `cryptofeed/health_server.py` - Already exists, needs integration
- `config/config.yaml` - Configuration schema
- `cryptofeed/backends/kafka.py` or `cryptofeed/kafka_callback.py` - Backend integration
- Exchange connectors - Various exchange modules

**Required dependencies:**
- PyYAML (for config parsing)
- argparse (for CLI arguments)
- asyncio (for event loop)
- signal (for graceful shutdown)

## Resources

- **Spec:** `.kiro/specs/multi-exchange-docker-deployment/tasks.md` Task 2
- **Dockerfile:** `Dockerfile:131` CMD specification
- **Health Server:** `cryptofeed/health_server.py` (already implemented)
- **Config:** `config/config.yaml` (example configuration)
- **Docker Compose:** `docker-compose.yml:129`

## Acceptance Criteria

- [ ] `cryptofeed/run.py` module created
- [ ] Accepts `--config` command-line argument
- [ ] Loads and validates YAML configuration file
- [ ] Starts health check HTTP server on port 8080
- [ ] Initializes at least one exchange connection (for testing)
- [ ] Sets up Kafka backend callback (if configured)
- [ ] Handles SIGTERM gracefully (shutdown within 30s)
- [ ] Docker container starts successfully: `docker-compose up`
- [ ] Health endpoint returns 200 OK: `curl localhost:8080/health`
- [ ] Integration test passes: `pytest tests/integration/test_docker_compose.py::TestDockerComposeStartup`
- [ ] Documentation updated if needed

## Work Log

### 2025-12-12 - Code Review Discovery

**By:** Claude Code

**Actions:**
- Reviewed docker-compose.yml implementation
- Checked for cryptofeed.run module existence
- Identified missing entry point as blocking issue
- Analyzed requirements from Dockerfile and compose config
- Drafted 3 solution approaches

**Learnings:**
- Health server already implemented separately (good foundation)
- Configuration schema exists in config/config.yaml
- Entry point needs minimal features to unblock Task 2 testing
- Can leverage existing health_server.py module

---

## Notes

- **Blocks:** Task 2 validation and testing
- **Priority:** P2 because Task 2 can be documented/tested manually without running container, but should be fixed before Task 2.1
- **Next Task:** After this is resolved, proceed with Task 2.1 (proxy integration) and Task 2.2 (integration tests)

# Cryptofeed Engineering Principles & AI Development Guide

## Active Specifications

Detailed status available in [`docs/specs/SPEC_STATUS.md`](docs/specs/SPEC_STATUS.md). Refer to `AGENTS.md` for overview of available agent workflows and command usage.

### ✅ Completed Specifications
- `proxy-system-complete`: ✅ COMPLETED (Jan 22, 2025) - Full proxy system implementation with transparent HTTP/SOCKS proxy support, consolidated documentation, 40 passing tests
  - **Implementation**: Core proxy system in `cryptofeed/proxy.py` with connection integration
  - **Testing**: 28 unit tests + 12 integration tests (all passing)
  - **Documentation**: `docs/proxy/README.md`, `docs/proxy/technical-specification.md`, `docs/proxy/user-guide.md`, `docs/proxy/architecture.md`
  - **Test Command**: `pytest tests/unit/test_proxy_mvp.py tests/integration/test_proxy_integration.py -v`

- `normalized-data-schema-crypto`: ✅ COMPLETE (Oct 20, 2025) - Phase 1 (v0.1.0) baseline schemas ready for production release
  - **Phase 1 (v0.1.0)**: 14/14 tasks complete, 46/46 tests passing, ready to merge and publish
  - **Phase 3 (Governance)**: 3/3 tasks complete, 42/42 tests passing, infrastructure ready
  - **Overall**: 68% complete (17/25 tasks), 119/119 tests passing, approved for merge
  - **Status**: Awaiting merge to main, then publication to Buf registry
  - **Documentation**: `docs/specs/normalized-data-schema/status.md`

- `ccxt-generic-pro-exchange`: ✅ COMPLETE (Oct 26, 2025) - Generic CCXT/CCXT-Pro abstraction for long-tail exchanges
  - **Implementation**: 1,612 LOC across 11 modules, 66 test files, 8/8 tasks complete
  - **Status**: Production ready, requires documentation update
  - **Next Step**: Create production integration guide and configuration examples

- `backpack-exchange-integration`: ✅ COMPLETE (Oct 26, 2025) - Native Cryptofeed Backpack connector with ED25519 auth
  - **Implementation**: 1,503 LOC across 11 modules, 59 test files, 10/10 tasks complete
  - **Approach**: Native Cryptofeed (not CCXT-based), exceptional quality (5/5 review score)
  - **Status**: Production ready, native integration guide pending
  - **Next Step**: Create native integration guide and ED25519 troubleshooting documentation

- `protobuf-callback-serialization`: ✅ COMPLETE (Nov 2, 2025) - Backend-only binary serialization for data feed callbacks
  - **Scope**: Protobuf serialization for 14 data types, BackendCallback integration with Kafka/Redis/ZMQ support
  - **Implementation**: 484 LOC in `cryptofeed/backends/protobuf_helpers.py`, 6 atomic commits
  - **Status**: PRODUCTION READY - Backend-only minimal implementation (500 LOC total)
  - **Key Achievement**: All protobuf logic consolidated in backends/, serializers/ and proto_wrappers/ deleted
  - **Testing**: 144+ tests passing, backward compatible (JSON default)
  - **Performance**: 2.1µs latency, 539k msg/s throughput, 63% smaller messages
  - **Next Step**: Merge to main, unblock market-data-kafka-producer

- `market-data-kafka-producer`: ✅ COMPLETE (Nov 13, 2025) - High-performance Kafka producer for protobuf-serialized market data with Phase 5 production execution plan
  - **Scope**: Kafka backend integration, topic management, exactly-once semantics, monitoring. Storage (Iceberg/DuckDB) delegated to consumers.
  - **Implementation**: 1,754 LOC in `cryptofeed/kafka_callback.py` and `cryptofeed/backends/kafka.py`
  - **Status**: ✅ PHASE 5 EXECUTION COMPLETE - Production-ready for immediate deployment
  - **Phase 1-4 (Core)**: 1,754 LOC, 628+ tests passing (100% pass rate), 7-8/10 code quality
  - **Phase 5 (Production Execution)**: 282 tests created, 261 passing (92.6%), 21 skipped (Kafka cluster), 0 failing
  - **Key Achievements**:
    - ✅ Consolidated topics (O(20)) as default, per-symbol (O(10K)) as option
    - ✅ 4 partition strategies (Composite, Symbol, Exchange, RoundRobin) with factory pattern
    - ✅ Message headers with routing metadata (exchange, symbol, data_type, schema_version)
    - ✅ Exactly-once semantics via idempotent producer + broker deduplication
    - ✅ Comprehensive error handling with exception boundaries (no silent failures)
    - ✅ Legacy backend (cryptofeed/backends/kafka.py) marked deprecated with migration guidance
    - ✅ 7-phase comprehensive review (status, requirements, design, gap analysis, implementation, documentation, code quality)
    - ✅ 4 atomic commits with Phase 5 execution materials merged to master
    - ✅ 10 measurable success criteria defined and validated (message loss zero, lag <5s, error <0.1%, latency p99 <5ms, throughput ≥100k msg/s, data integrity 100%, monitoring functional, rollback <5min, topic count O(20), headers 100%)
    - ✅ Complete team handoff package (roles, responsibilities, escalation procedures)
    - ✅ Consumer migration templates (Flink, Python async, Custom minimal)
    - ✅ Grafana monitoring dashboard (8 panels) + alert rules (8 rules)
    - ✅ Per-exchange migration procedure with automation framework
  - **Testing**: 628+ tests (Phase 1-4: 346 unit + 18 integration + 32 performance; Phase 5: 282 tests across 9 tasks)
  - **Code Quality**: 7-8/10 (post-critical fixes), performance 9.9/10
  - **Documentation**: Comprehensive (5,867+ specification lines + 3,847 test code lines)
    - Design (1,270 lines), requirements (304 lines), tasks (979 lines)
    - Phase 5 execution materials: 4-week timeline, task specifications, quick reference, visual timeline, operational runbook, team handoff
    - User guides: 7 comprehensive guides (162 KB) + consumer templates
  - **Atomic Commits** (Phase 5 Execution): 3197624e (spec), 70f7f575 (materials), f8753f35 (handoff), merged to master
  - **Risk Assessment**: LOW (0 blockers, 5 identified risks with mitigations)
  - **Confidence Level**: HIGH (95%)
  - **Next Step**: Teams can now execute Phase 5 production migration following PHASE_5_EXECUTION_PLAN.md (4-week Blue-Green cutover)

### 🚧 In Progress Specifications
(None - all active specs have either completed or are awaiting approval)

### 📋 Planning Phase
- `unified-exchange-feed-architecture`: Design generated (Oct 20, 2025) - Unify native and CCXT integrations behind shared contracts
  - **Status**: Design generated but NOT YET approved, blocks task generation
  - **Dependencies**: CCXT generic and Backpack specs (in progress)
  - **Next Step**: Review and approve design before proceeding

---

## Architecture: Ingestion Layer

Cryptofeed is positioned as a pure data ingestion layer. Storage and analytics are delegated to downstream consumers.

### Dependency Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Cryptofeed Ingestion Layer (IN-SCOPE)                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Exchange     │───▶│ Normalized   │───▶│ Protobuf     │  │
│  │ Connectors   │    │ Data Schema  │    │ Serialization│  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                   │          │
└───────────────────────────────────────────────────┼──────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │ Kafka Topics     │
                                          │ (Protobuf msgs)  │
                                          └────────┬─────────┘
                                                   │
                ┌─────────────┬──────────────────┬──────────────┬──────────────┬──────────────┐
                ▼             ▼                  ▼              ▼              ▼              ▼
          ┌──────────┐ ┌──────────┐      ┌──────────┐   ┌──────────┐   ┌──────────┐ ┌──────────┐
          │ Flink    │ │QuixStreams│     │ DuckDB   │   │ Custom   │   │ Iceberg  │ │ Spark    │
          │ → Iceberg│ │CryptofeedSrc    │ Consumer │   │ Consumer │   │ Direct   │ │ → Parquet│
          └──────────┘ └──────────┘      └──────────┘   └──────────┘   └──────────┘ └──────────┘

          Consumer Responsibility (OUT-OF-SCOPE):
          - Read Kafka topics
          - Deserialize protobuf (CryptofeedSource handles for QuixStreams)
          - Implement storage (Iceberg, Parquet, DuckDB)
          - Implement analytics (aggregations, queries)
          - Implement retention policies
```

### Specifications Alignment

| Spec | Phase | Scope | Boundary |
|------|-------|-------|----------|
| **Spec 0** | Complete | Protobuf schemas (.proto files) | Schema definition |
| **Spec 1** | In Progress | Serialization (`to_proto()` methods) | Kafka message production |
| **Spec 3** | Initialized | Kafka producer integration | Kafka topic publication |
| **Consumer** | External | Storage, analytics, retention | Everything after Kafka |

**Key Principle**: Cryptofeed stops at Kafka. Consumers handle everything downstream.

### ⏸️ Paused/Disabled Specifications
- `cryptofeed-lakehouse-architecture`: Disabled (user request) - Data lakehouse architecture with real-time ingestion and analytics
  - **Status**: Can be reactivated anytime, all phases (requirements, design, tasks) prepared and approved
  - **Dependencies**: Can leverage normalized-data-schema-crypto once merged

- `proxy-pool-system`: Disabled (paused) - Proxy pool management and rotation (extends proxy-system-complete)
  - **Status**: Requirements, design, tasks all approved, awaiting external service roadmap clarification
  - **Note**: Related to external-proxy-service spec

- `external-proxy-service`: Disabled (deferred) - Service-oriented proxy management with external service delegation
  - **Status**: High priority, 4-6 weeks effort, awaiting proxy roadmap realignment
  - **Note**: Depends on proxy-pool-system alignment

## Core Engineering Principles

### Ingestion Layer Only (Separation of Concerns)
- **Scope**: Cryptofeed focuses exclusively on data ingestion and normalization
- **Producer Role**: Publish protobuf-serialized messages to Kafka topics
- **Consumer Responsibility**: Downstream consumers implement storage, analytics, and persistence
- **Storage Agnostic**: No opinions on lakehouse technology (Apache Iceberg, DuckDB, Parquet, etc.)
- **Query Independence**: Query engines (Flink, Spark, Trino, DuckDB) are consumer choices
- **Benefits**: Clear separation of concerns, flexible storage backends, reduced maintenance burden

### SOLID Principles
- **Single Responsibility**: Each class/module has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes must be substitutable for base classes
- **Interface Segregation**: Clients shouldn't depend on interfaces they don't use
- **Dependency Inversion**: Depend on abstractions, not concretions

### KISS (Keep It Simple, Stupid)
- Prefer well-scoped conventional commits (feat:, fix:, chore:, etc.) to keep history searchable
- Document behavioral changes in the subject; leave refactors/docs/tests as chore/test/docs prefixes
- Avoid multi-purpose commits—split when scope spans unrelated areas
- Tie commits to spec/task IDs when available for traceability

### KISS (Keep It Simple, Stupid)
- Prefer simple solutions over complex ones
- Avoid premature optimization
- Write code that is easy to understand and maintain
- Minimize cognitive load for future developers

### Conventional Commits
- Use `feat:`, `fix:`, `chore:`, `docs:`, etc., to label intent and surface change type quickly
- Keep commit scope tight—one functional concern per commit, split unrelated work
- Reference spec/task IDs when available to maintain traceability
- Describe the user-facing behavior change in the subject; reserve details for the body if needed

### DRY (Don't Repeat Yourself)
- Extract common functionality into reusable components
- Use configuration over duplication
- Share metadata/transport logic across derived feeds
- Avoid duplicated rate limit logic

### YAGNI (You Aren't Gonna Need It)
- Implement only what's needed now
- Defer features until they're actually required
- Keep configuration surface minimal
- Avoid building for hypothetical future requirements

### FRs Over NFRs
- Deliver functional requirements before tuning non-functional concerns
- Capture NFR gaps as follow-up work instead of blocking feature delivery
- Align prioritization with user impact, revisiting NFRs once core behavior ships
- Treat performance, resiliency, and compliance targets as iterative enhancements unless explicitly critical

### Compound Engineering with Parallel Work Streams
- **Decompose Outcomes:** Split large initiatives into discrete, value-focused streams that can progress independently without blocking shared milestones.
- **Bounded Interfaces:** Define clear contracts (APIs, schema versions, specs) so parallel teams can integrate asynchronously with minimal coordination overhead.
- **Synchronization Cadence:** Establish short, recurring integration checkpoints to surface cross-stream risks early while preserving autonomous execution between checkpoints.
- **Shared Context Hubs:** Maintain living documents (specs, ADRs, dashboards) that aggregate decisions and status across streams to avoid redundant alignment meetings.
- **Risk Balancing:** Pair high-complexity streams with stabilization or hardening tracks to ensure compound delivery doesn’t sacrifice reliability.
- **Capacity Guardrails:** Reserve buffer capacity for emergent interdependencies or support needs, preventing one stream’s blockers from derailing overall delivery.

## Development Standards

### NO MOCKS
- Use real implementations with test fixtures
- Prefer integration tests over heavily mocked unit tests
- Test against actual exchange APIs when possible
- Use ccxt sandbox or permissive endpoints for testing

### NO LEGACY
- Remove deprecated code aggressively
- Don't maintain backward compatibility for internal APIs
- Upgrade dependencies regularly
- Clean architecture without legacy workarounds

### NO COMPATIBILITY
- Target latest Python versions
- Use modern language features
- Don't support outdated exchange API versions
- Break APIs when it improves design

### START SMALL
- Begin with MVP implementations
- Support minimal viable feature set first
- Add complexity only when justified
- Iterative development over big bang releases

### CONSISTENT NAMING WITHOUT PREFIXES
- Use clear, descriptive names
- Avoid Hungarian notation or type prefixes
- Consistent verb tenses (get/set, fetch/push)
- Domain-specific terminology over generic names

## Agentic Coding Best Practices

### Research-Plan-Execute Workflow
1. **Research Phase**: Read relevant files, understand context
2. **Planning Phase**: Outline solution architecture
3. **Execution Phase**: Implement with continuous verification
4. **Validation Phase**: Test and verify implementation

### Test-Driven Development (TDD)
- Write tests first based on expected behavior
- Run tests to confirm they fail
- Implement minimal code to pass tests
- Refactor without changing test behavior
- Never modify tests to fit implementation

### Context Engineering
- Maintain project context in CLAUDE.md
- Use specific, actionable instructions
- Provide file paths and screenshots for UI work
- Reference existing patterns and conventions
- Clear context between major tasks

### Steering & Spec Prerequisites
- Before editing code, read `.kiro/steering/product.md`, `.kiro/steering/tech.md`, and `.kiro/steering/structure.md` to understand project-wide compound engineering and AI agent guidelines.
- For any feature-level change, locate the relevant spec(s) under `.kiro/specs/` and read their **Compound Engineering Alignment** and **AI Agentic Implementation Constraints** sections before implementing.
- Treat steering docs as global governance and spec docs as local contracts; do not cross spec boundaries (schemas, serialization, Kafka producer, E2E flows, consumers) without updating or creating the appropriate spec.

### Iterative Development
- Make small, verifiable changes
- Commit frequently with descriptive messages
- Use subagents for complex verification tasks
- Review code changes continuously
- Maintain clean git history

## Context Engineering Principles

### Information Architecture
- **Prioritize by Relevance**: Most important information first
- **Logical Categorization**: Group related context together
- **Progressive Detail**: Start essential, add layers gradually
- **Clear Relationships**: Show dependencies and connections

### Dynamic Context Systems
- **Runtime Context**: Generate context on-demand for tasks
- **State Management**: Track conversation and project state
- **Memory Integration**: Combine short-term and long-term knowledge
- **Tool Integration**: Provide relevant tool and API context

### Context Optimization
- **Precision Over Volume**: Quality information over quantity
- **Format Consistency**: Structured, scannable information
- **Relevance Filtering**: Include only task-relevant context
- **Context Window Management**: Efficient use of available space

## Cryptofeed-Specific Guidelines

### Exchange Integration
- Use ccxt for standardized exchange APIs
- Follow existing emitter/queue patterns
- Implement proper rate limiting and backoff
- Handle regional restrictions with proxy support

### Data Normalization
- Convert timestamps to consistent float seconds
- Use Decimal for price/quantity precision
- Preserve sequence numbers for gap detection
- Normalize symbols via ccxt helpers

### Error Handling
- Surface HTTP errors with actionable messages
- Provide fallback modes (REST-only, alternative endpoints)
- Log warnings for experimental features
- Implement graceful degradation

### Configuration
- Use YAML configuration files
- Support environment variable interpolation
- Provide clear examples and documentation
- Allow per-deployment customization

### Architecture Patterns
```
CcxtGenericFeed
 ├─ CcxtMetadataCache   → ccxt.exchange.load_markets()
 ├─ CcxtRestTransport   → ccxt.async_support.exchange.fetch_*()
 └─ CcxtWsTransport     → ccxt.pro.exchange.watch_*()
      ↳ CcxtEmitter     → existing BackendQueue/Metrics
```

## Testing Strategy

### Unit Testing
- Mock ccxt transports for isolated testing
- Test symbol normalization and data transformation
- Verify queue integration and error handling
- Assert configuration parsing and validation

### Integration Testing
- Test against live exchange APIs (sandbox when available)
- Verify trade/L2 callback sequences
- Test with actual proxy configurations
- Record sample payloads for regression testing

### Regression Testing
- Maintain docker-compose test harnesses
- Test across ccxt version updates
- Verify backward compatibility of configurations
- Automated testing in CI/CD pipeline

## Common Commands

### Development
```bash
# Run tests
python -m pytest tests/ -v

# Code quality gate (smells + complexity)
pyscn check --max-complexity 15 cryptofeed

# Type checking
mypy cryptofeed/

# Linting
ruff check cryptofeed/
ruff format cryptofeed/

# Scoped formatting (only changed files)
./tools/format-utils.sh format-staged    # Format only staged changes
./tools/format-utils.sh format-unstaged  # Format only unstaged changes
./tools/format-utils.sh format-all       # Format all changes
./tools/format-utils.sh dry-run-unstaged # Preview what would be formatted
python tools/format-changed.py --unstaged  # Direct script usage

# Install development dependencies
pip install -e ".[dev]"
```

### Exchange Testing
```bash
# Test specific exchange integration
python -m pytest tests/integration/test_backpack.py -v

# Run with live data (requires credentials)
BACKPACK_API_KEY=xxx python examples/backpack_live.py
```

### Documentation
```bash
# Build docs
cd docs && make html

# Serve docs locally
cd docs/_build/html && python -m http.server 8000
```

## AI Development Workflow

### Task Initialization
1. Read this CLAUDE.md file for context
2. Examine relevant specification files in `docs/specs/`
3. Review existing implementation patterns
4. Plan approach using established principles

### Implementation Process
1. Write tests first (TDD approach)
2. Implement minimal viable solution
3. Iterate with continuous testing
4. Refactor for clarity and maintainability
5. Document configuration and usage

### Quality Assurance
1. Run full test suite
2. Check type annotations
3. Verify code formatting
4. Test with real exchange data
5. Update documentation as needed

### Code Review Checklist
- [ ] Follows SOLID principles
- [ ] Implements TDD approach
- [ ] No mocks in production code
- [ ] Consistent naming conventions
- [ ] Proper error handling
- [ ] Type annotations present
- [ ] Tests cover edge cases
- [ ] Documentation updated
- [ ] No legacy compatibility code
- [ ] Configuration examples provided

## Project Structure

```
cryptofeed/
├── adapters/           # ccxt integration adapters
├── exchanges/          # exchange-specific implementations
├── defines.py          # constants and enums
├── types.py           # type definitions
└── utils.py           # utility functions

docs/
├── specs/             # detailed specifications
├── examples/          # usage examples
└── api/              # API documentation

tests/
├── unit/             # isolated unit tests
├── integration/      # live exchange tests
└── fixtures/         # test data and mocks
```

## Performance Considerations

### Memory Management
- Use slots for data classes
- Implement proper cleanup in transports
- Monitor memory usage in long-running feeds
- Use generators for large data streams

### Network Optimization
- Implement connection pooling
- Use persistent WebSocket connections
- Batch REST API requests when possible
- Implement proper rate limiting

### Data Processing
- Use Decimal for financial calculations
- Minimize data copying in hot paths
- Implement efficient order book management
- Cache metadata to reduce API calls

---

*This document serves as the primary context for AI-assisted development in the Cryptofeed project. Update regularly as patterns and practices evolve.*


# AI-DLC and Spec-Driven Development

Kiro-style Spec Driven Development implementation on AI-DLC (AI Development Life Cycle)

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `/kiro/spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in English. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `/kiro/steering`, `/kiro/steering-custom`
- Phase 1 (Specification):
  - `/kiro/spec-init "description"`
  - `/kiro/spec-requirements {feature}`
  - `/kiro/validate-gap {feature}` (optional: for existing codebase)
  - `/kiro/spec-design {feature} [-y]`
  - `/kiro/validate-design {feature}` (optional: design review)
  - `/kiro/spec-tasks {feature} [-y]`
- Phase 2 (Implementation): `/kiro/spec-impl {feature} [tasks]`
  - `/kiro/validate-impl {feature}` (optional: after implementation)
- Progress check: `/kiro/spec-status {feature}` (use anytime)

### Command Definitions & Subagents
- All `kiro:spec-*` command definitions live in `.claude/commands/kiro/`. These files describe the workflow steps and checks you must perform manually—they are instructions, not executable binaries. Follow them verbatim when acting as the CLI.
- Commands that delegate work to specialized agents reference the implementations under `.claude/agents/kiro/` (for example `spec-design-agent`, `spec-tasks-agent`). These agent files likewise contain procedural instructions you must follow; no automatic process will run unless you explicitly perform the documented steps.

### Running `kiro:spec-*` Commands (Manual Emulation)
1. Run commands from the repository root so relative paths resolve correctly.
2. Use the zsh-style syntax (`kiro:spec-design feature-name`) as documented, but remember you are emulating the command by following the instructions in `.claude/commands/kiro/`.
3. Perform every validation and file read/write step described in the command definition, including invoking subagent instructions when specified.
4. Follow the approval gating shown in the command definitions (e.g., requirements must exist before design generation, design must be approved before tasks).
5. Consult `.claude/commands/kiro/spec-status.md` for the required checks when reporting progress.

```bash
# Typical workflow
kiro:spec-init "feature description"
kiro:spec-requirements my-feature
kiro:spec-design my-feature      # add -y to auto-approve requirements
kiro:spec-tasks my-feature       # add -y to auto-approve design
kiro:spec-impl my-feature 1.1    # executes task 1.1
kiro:spec-status my-feature      # view progress anytime
```

## Development Rules
- 3-phase approval workflow: Requirements → Design → Tasks → Implementation
- Human review required each phase; use `-y` only for intentional fast-track
- Keep steering current and verify alignment with `/kiro/spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- Load entire `.kiro/steering/` as project memory
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `/kiro/steering-custom`)


# AI-DLC and Spec-Driven Development

Kiro-style Spec Driven Development implementation on AI-DLC (AI Development Life Cycle)

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `/kiro:spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in English. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `/kiro:steering`, `/kiro:steering-custom`
- Phase 1 (Specification):
  - `/kiro:spec-init "description"`
  - `/kiro:spec-requirements {feature}`
  - `/kiro:validate-gap {feature}` (optional: for existing codebase)
  - `/kiro:spec-design {feature} [-y]`
  - `/kiro:validate-design {feature}` (optional: design review)
  - `/kiro:spec-tasks {feature} [-y]`
- Phase 2 (Implementation): `/kiro:spec-impl {feature} [tasks]`
  - `/kiro:validate-impl {feature}` (optional: after implementation)
- Progress check: `/kiro:spec-status {feature}` (use anytime)

## Development Rules
- 3-phase approval workflow: Requirements → Design → Tasks → Implementation
- Human review required each phase; use `-y` only for intentional fast-track
- Keep steering current and verify alignment with `/kiro:spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- Load entire `.kiro/steering/` as project memory
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `/kiro:steering-custom`)

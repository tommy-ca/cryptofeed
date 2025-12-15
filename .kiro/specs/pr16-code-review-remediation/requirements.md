# Requirements Document

## Project Description (Input)
PR #16 Code Review Remediation: Address critical findings from comprehensive multi-agent code review including data integrity issues (schema field population), security vulnerabilities (SSRF in proxy config), scope management (PR split strategy), code quality improvements (DRY violations), and complexity reduction (YAGNI compliance). Scope includes implementing missing protobuf field converters, URL validation for proxy configurations, normalization utility consolidation, and optional PR restructuring into focused deliverables.

## Introduction

This specification addresses five categories of critical and important findings from the comprehensive multi-agent code review of PR #16. The issues were identified by specialized review agents (Data Integrity Guardian, Security Sentinel, Kieran Rails Reviewer, Pattern Recognition Specialist, Code Simplicity Reviewer) and documented in TODO files 005-009.

The specification follows priority-based delivery:

**P1 Critical Issues (BLOCKS MERGE)**:
1. **Data Integrity**: 31% of protobuf schema fields never populated (todos/005)
2. **Security**: SSRF vulnerability in proxy configuration loading (todos/006)
3. **Process**: 364 files in single PR requiring split (todos/007)

**P2 Important Issues**:
4. **Code Quality**: DRY violation in normalization logic (todos/008)
5. **Complexity**: 2,000+ LOC of YAGNI violations (todos/009)

All requirements use EARS (Easy Approach to Requirements Syntax) format for testability and clarity. The Cryptofeed Protobuf System is the primary subject for software requirements.

## Requirements

### Requirement 1: Schema Field Population (P1 Critical - Data Integrity)

**Objective:** As a data consumer, I want all protobuf schema fields to be populated with exchange data when available, so that no metadata is silently lost during ingestion.

**Priority:** P1 (BLOCKS MERGE)
**Rationale:** The protobuf v2beta1 schema defines optional fields (maker, event_time, match_id, liquidity_flag for Trade; event_time, last_update_id for OrderBook) that are never populated, resulting in 31% silent data loss.

**Traceability:** todos/005-pending-p1-new-schema-fields-not-populated.md

#### Acceptance Criteria

1. WHEN the Trade class is instantiated THEN the Cryptofeed Protobuf System SHALL include attributes for maker, event_time, match_id, and liquidity_flag

2. WHEN the OrderBook class is instantiated THEN the Cryptofeed Protobuf System SHALL include attributes for event_time and last_update_id

3. WHEN Binance WebSocket message contains 'm' field (maker) THEN the Cryptofeed Protobuf System SHALL extract and populate Trade.maker attribute

4. WHEN Binance WebSocket message contains 'E' field (event_time) THEN the Cryptofeed Protobuf System SHALL extract and populate Trade.event_time attribute

5. WHEN Binance WebSocket message contains 'a' field (match_id) THEN the Cryptofeed Protobuf System SHALL extract and populate Trade.match_id attribute

6. WHEN exchange WebSocket message contains order book event timestamp THEN the Cryptofeed Protobuf System SHALL extract and populate OrderBook.event_time attribute

7. WHEN exchange WebSocket message contains last_update_id field THEN the Cryptofeed Protobuf System SHALL extract and populate OrderBook.last_update_id attribute

8. WHEN trade_to_proto() converter receives Trade object with maker attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf Trade.maker field

9. WHEN trade_to_proto() converter receives Trade object with event_time attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf Trade.event_time field with microsecond precision

10. WHEN trade_to_proto() converter receives Trade object with match_id attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf Trade.match_id field

11. WHEN trade_to_proto() converter receives Trade object with liquidity_flag attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf Trade.liquidity_flag field

12. WHEN orderbook_to_proto() converter receives OrderBook object with event_time attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf OrderBook.event_time field

13. WHEN orderbook_to_proto() converter receives OrderBook object with last_update_id attribute THEN the Cryptofeed Protobuf System SHALL populate the protobuf OrderBook.last_update_id field

14. WHEN field data is unavailable from exchange THEN the Cryptofeed Protobuf System SHALL leave the optional protobuf field unset (not populate with default values)

15. WHEN consumer reads protobuf Trade message from Binance THEN the Cryptofeed Protobuf System SHALL have populated maker, event_time, and match_id fields (if exchange provided data)

16. WHEN unit tests execute for new Trade fields THEN the Cryptofeed Protobuf System SHALL verify field population for each data type (maker: bool, event_time: timestamp, match_id: string, liquidity_flag: string)

17. WHEN integration tests execute via Kafka THEN the Cryptofeed Protobuf System SHALL confirm end-to-end field transmission from exchange to consumer

18. WHERE field availability varies per exchange THE Cryptofeed Protobuf System SHALL document which exchanges support which fields in schema mapping documentation

19. WHEN all schema fields are processed THEN the Cryptofeed Protobuf System SHALL report zero silent data loss (all extracted fields are transmitted)

---

### Requirement 2: SSRF Prevention in Proxy Configuration (P1 Critical - Security)

**Objective:** As a security engineer, I want proxy URLs to be validated before use, so that Server-Side Request Forgery attacks are prevented.

**Priority:** P1 (BLOCKS MERGE)
**Rationale:** Proxy configuration loading accepts arbitrary URLs without validation, enabling SSRF attacks against cloud metadata services, internal networks, and local files (CVSS 7.5 High).

**Traceability:** todos/006-pending-p1-ssrf-vulnerability-proxy-config.md

#### Acceptance Criteria

1. WHEN load_proxy_mapping() is called with proxy.yaml THEN the Cryptofeed Proxy System SHALL validate all proxy URLs before accepting configuration

2. WHEN proxy URL contains scheme 'http' or 'https' or 'socks4' or 'socks5' or 'socks5h' THEN the Cryptofeed Proxy System SHALL accept the URL for further validation

3. WHEN proxy URL contains scheme 'file' or 'ftp' or 'gopher' or any non-proxy scheme THEN the Cryptofeed Proxy System SHALL reject the URL with clear error message

4. WHEN proxy URL hostname resolves to private IP range (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) THEN the Cryptofeed Proxy System SHALL reject the URL with security error

5. WHEN proxy URL hostname resolves to loopback address (127.0.0.0/8, ::1) THEN the Cryptofeed Proxy System SHALL reject the URL with security error

6. WHEN proxy URL hostname resolves to link-local address (169.254.0.0/16, fe80::/10) THEN the Cryptofeed Proxy System SHALL reject the URL with security error

7. WHEN proxy URL hostname is 'localhost' or '127.0.0.1' or '::1' THEN the Cryptofeed Proxy System SHALL reject the URL with security error

8. WHEN proxy URL contains cloud metadata endpoint (169.254.169.254) THEN the Cryptofeed Proxy System SHALL reject the URL with SSRF prevention message

9. WHEN proxy URL validation fails THEN the Cryptofeed Proxy System SHALL raise ValueError with specific reason (scheme/IP/hostname blocked)

10. WHEN proxy URL uses URL encoding to bypass validation THEN the Cryptofeed Proxy System SHALL detect and reject the attempt

11. WHEN validate_proxy_url() function is called with valid proxy URL THEN the Cryptofeed Proxy System SHALL return True

12. WHEN validate_proxy_url() function is called with invalid proxy URL THEN the Cryptofeed Proxy System SHALL return False and log error

13. WHEN global proxy configuration contains invalid URL THEN the Cryptofeed Proxy System SHALL raise ValueError before applying any configuration

14. WHEN per-exchange proxy configuration contains invalid URL THEN the Cryptofeed Proxy System SHALL raise ValueError before applying any configuration

15. WHEN unit tests execute SSRF test cases THEN the Cryptofeed Proxy System SHALL reject all blocked URL patterns (http://169.254.169.254/, http://localhost:8080/, http://10.0.0.1/, http://192.168.1.1/, file:///etc/passwd, ftp://internal/)

16. WHEN integration test loads malicious proxy.yaml THEN the Cryptofeed Proxy System SHALL raise ValueError and prevent configuration load

17. WHEN documentation is updated THEN the Cryptofeed Proxy System SHALL include security notes about proxy validation and SSRF prevention

18. WHEN security scan executes against proxy configuration loading THEN the Cryptofeed Proxy System SHALL pass with zero SSRF vulnerabilities detected

---

### Requirement 3: PR Scope Management (P1 Critical - Process)

**Objective:** As a code reviewer, I want pull requests to be focused and reviewable, so that meaningful review is possible within reasonable timeframes and deployment risks are minimized.

**Priority:** P1 (BLOCKS MERGE)
**Rationale:** PR #16 contains 364 files (58,461 additions, 14,514 deletions) representing 7 independent features, making thorough review impossible and creating significant merge/revert risks.

**Traceability:** todos/007-pending-p1-excessive-scope-split-pr.md

#### Acceptance Criteria

1. WHEN a pull request is created THEN the PR Management Process SHALL ensure file count is less than 100 files per PR

2. WHEN a pull request is created THEN the PR Management Process SHALL ensure line additions are less than 5,000 lines per PR

3. WHEN PR #16.1 (Core Kafka Module Structure) is created THEN the PR Management Process SHALL include only base.py, producer.py, topic_manager.py, partitioner.py and their tests (approximately 60 files, 800 LOC)

4. WHEN PR #16.2 (Protobuf Consolidation) is created THEN the PR Management Process SHALL include only cryptofeed/backends/protobuf/ package and schema updates (approximately 70 files, 1,200 LOC)

5. WHEN PR #16.3 (Configuration Management) is created THEN the PR Management Process SHALL include only config.py, Pydantic models, and validation logic (approximately 35 files, 500 LOC)

6. WHEN PR #16.4 (Metrics & Observability) is created THEN the PR Management Process SHALL include only metrics.py, health checks, and health server (approximately 45 files, 900 LOC)

7. WHEN PR #16.5 (Deprecation System) is created THEN the PR Management Process SHALL include only deprecation.py, migration tools, and timeline management (approximately 25 files, 750 LOC)

8. WHEN PR #16.6 (Legacy Compatibility Shims) is created THEN the PR Management Process SHALL include only compatibility layer and backward compatibility tests (approximately 35 files, 300 LOC)

9. WHEN PR #16.7 (Documentation Updates) is created THEN the PR Management Process SHALL include only documentation reorganization and migration guides (approximately 90 files, docs only)

10. WHEN split PRs are created THEN the PR Management Process SHALL ensure dependency graph is documented (which PRs block which)

11. WHEN split PRs are sequenced THEN the PR Management Process SHALL establish 1 PR per week timeline over 4 weeks

12. WHEN each split PR is submitted THEN the PR Management Process SHALL verify independent test coverage exists

13. WHEN all split PRs are merged THEN the PR Management Process SHALL execute final integration test

14. WHEN PR #16 is addressed THEN the PR Management Process SHALL close or mark original PR as draft/WIP

15. WHEN commits are organized THEN the PR Management Process SHALL cherry-pick commits into appropriate feature branches (kafka-backend-1 through kafka-backend-7)

16. WHEN split PRs are merged THEN the PR Management Process SHALL use squash merge to main with preserved commit messages

17. WHEN final PR is merged THEN the PR Management Process SHALL tag as "kafka-backend-refactor-complete"

18. WHEN review timeframe is estimated THEN the PR Management Process SHALL ensure each PR is reviewable in 1-2 hours (less than 100 files)

19. WHEN regression is detected THEN the PR Management Process SHALL ensure no functionality loss vs. original PR #16

---

### Requirement 4: Normalization Code Deduplication (P2 Important - Code Quality)

**Objective:** As a maintainer, I want string normalization logic to be centralized, so that changes only need to be made in one place and behavior remains consistent.

**Priority:** P2 (Important, not blocking)
**Rationale:** String normalization for exchanges and symbols is duplicated across 3 files (topic_manager.py, partitioner.py, headers.py) with slight variations, violating DRY principle.

**Traceability:** todos/008-pending-p2-code-duplication-normalization.md

#### Acceptance Criteria

1. WHEN normalization module is created THEN the Cryptofeed Kafka Backend SHALL create cryptofeed/backends/kafka/normalization.py with 2 utility functions

2. WHEN normalize_symbol() function is implemented THEN the Cryptofeed Kafka Backend SHALL convert symbol to lowercase, replace '/' and '_' with '-', strip whitespace, and return 'unknown' for None/empty

3. WHEN normalize_exchange() function is implemented THEN the Cryptofeed Kafka Backend SHALL convert exchange to lowercase, strip whitespace, and return 'unknown' for None/empty

4. WHEN normalize_symbol() receives 'BTC/USD' THEN the Cryptofeed Kafka Backend SHALL return 'btc-usd'

5. WHEN normalize_symbol() receives 'BTC_USD' THEN the Cryptofeed Kafka Backend SHALL return 'btc-usd'

6. WHEN normalize_symbol() receives ' ETH-BTC ' THEN the Cryptofeed Kafka Backend SHALL return 'eth-btc'

7. WHEN normalize_symbol() receives None or empty string THEN the Cryptofeed Kafka Backend SHALL return 'unknown'

8. WHEN normalize_exchange() receives 'Binance' THEN the Cryptofeed Kafka Backend SHALL return 'binance'

9. WHEN normalize_exchange() receives ' OKX ' THEN the Cryptofeed Kafka Backend SHALL return 'okx'

10. WHEN normalize_exchange() receives None or empty string THEN the Cryptofeed Kafka Backend SHALL return 'unknown'

11. WHEN topic_manager.py is refactored THEN the Cryptofeed Kafka Backend SHALL import and use normalize_symbol and normalize_exchange from normalization module

12. WHEN partitioner.py is refactored THEN the Cryptofeed Kafka Backend SHALL import and use normalize_symbol and normalize_exchange from normalization module

13. WHEN headers.py is refactored THEN the Cryptofeed Kafka Backend SHALL import and use normalize_symbol and normalize_exchange from normalization module

14. WHEN normalization functions are consolidated THEN the Cryptofeed Kafka Backend SHALL remove duplicate _normalize_symbol and _normalize_exchange functions from topic_manager.py, partitioner.py, and headers.py

15. WHEN unit tests are created THEN the Cryptofeed Kafka Backend SHALL verify 10+ test cases for each normalization function (whitespace, mixed separators, None, empty, case variations)

16. WHEN consistency tests execute THEN the Cryptofeed Kafka Backend SHALL verify identical output across topic generation, partition key generation, and header encoding

17. WHEN existing integration tests execute THEN the Cryptofeed Kafka Backend SHALL pass all tests unchanged (behavior preservation)

18. WHEN docstrings are added THEN the Cryptofeed Kafka Backend SHALL document normalization rules and provide examples for both functions

---

### Requirement 5: Complexity Reduction (P2 Important - YAGNI Compliance)

**Objective:** As a developer, I want the codebase to contain only necessary code, so that maintenance burden is minimized and cognitive load is reduced.

**Priority:** P2 (Important, not blocking)
**Rationale:** PR #16 includes 2,000+ lines of code (57% of additions) for hypothetical future features, violating YAGNI principle and increasing complexity without delivering current value.

**Traceability:** todos/009-pending-p2-unnecessary-complexity-yagni-violations.md

#### Acceptance Criteria

1. WHEN deprecation.py is simplified THEN the Cryptofeed Kafka Backend SHALL retain only 2 warning functions (23 LOC total) and remove 504 LOC of unused timeline infrastructure

2. WHEN maintenance/__init__.py is evaluated THEN the Cryptofeed Kafka Backend SHALL delete entire file (135 LOC of no-op bridge patterns)

3. WHEN migration.py is relocated THEN the Cryptofeed Kafka Backend SHALL move to tools/migrate_kafka_config.py and remove from runtime imports (229 LOC from production package)

4. WHEN partitioner.py is simplified THEN the Cryptofeed Kafka Backend SHALL replace factory pattern and 4 strategy classes with 15-line inline function using if/elif (80 LOC reduction)

5. WHEN headers.py is inlined THEN the Cryptofeed Kafka Backend SHALL consolidate into 20-line function in callback.py (354 LOC reduction from separate module)

6. WHEN health.py is simplified THEN the Cryptofeed Kafka Backend SHALL reduce to basic health check function (approximately 30 lines, 100 LOC reduction)

7. WHEN config.py is flattened THEN the Cryptofeed Kafka Backend SHALL consolidate 4 Pydantic classes into 1 dataclass (270 LOC reduction)

8. WHEN metrics.py is simplified THEN the Cryptofeed Kafka Backend SHALL remove wrapper abstractions and use prometheus_client directly (250 LOC reduction)

9. WHEN module consolidation occurs THEN the Cryptofeed Kafka Backend SHALL merge base.py, producer.py, and topic_manager.py into backend.py (700 LOC overhead reduction)

10. WHEN Phase 1 simplification completes (delete dead code) THEN the Cryptofeed Kafka Backend SHALL achieve 868 LOC reduction with zero functional impact

11. WHEN Phase 2 simplification completes (inline trivial abstractions) THEN the Cryptofeed Kafka Backend SHALL achieve additional 500 LOC reduction

12. WHEN Phase 3 simplification completes (consolidate modules) THEN the Cryptofeed Kafka Backend SHALL achieve additional 700 LOC reduction

13. WHEN all simplification phases complete THEN the Cryptofeed Kafka Backend SHALL reduce total LOC from 3,576 to approximately 730 (79.6% reduction)

14. WHEN module count is evaluated THEN the Cryptofeed Kafka Backend SHALL reduce from 15 files to 4 files (73.3% reduction)

15. WHEN test coverage is evaluated THEN the Cryptofeed Kafka Backend SHALL reduce test count from 170+ to approximately 40 tests (76.5% reduction) while maintaining behavioral coverage

16. WHEN final structure is implemented THEN the Cryptofeed Kafka Backend SHALL consist of backend.py (500 LOC), config.py (60 LOC), _deprecated.py (150 LOC), and __init__.py (20 LOC)

17. WHEN all existing tests execute THEN the Cryptofeed Kafka Backend SHALL pass all tests (behavior preservation)

18. WHEN code review is performed THEN the Cryptofeed Kafka Backend SHALL demonstrate 50%+ reduction in review time due to simplified codebase

19. WHEN YAGNI compliance is verified THEN the Cryptofeed Kafka Backend SHALL align with CLAUDE.md principles: "Implement only what's needed now", "Prefer simple solutions over complex ones", "Begin with MVP implementations"

---

## Cross-Requirement Dependencies

### P1 Requirements Dependencies
- **REQ-1 (Schema Fields)** depends on protobuf schema being stable (no breaking changes during implementation)
- **REQ-2 (SSRF Prevention)** is independent and can be implemented immediately
- **REQ-3 (PR Split)** blocks all other requirements from merging (must be addressed first)

### P2 Requirements Dependencies
- **REQ-4 (Normalization)** should be implemented before REQ-5 (consolidation will be simpler with DRY compliance)
- **REQ-5 (Complexity)** benefits from REQ-4 completion (normalization module is part of simplified structure)

### Recommended Implementation Sequence
1. **REQ-3** (PR Split) - Immediate (blocks merge)
2. **REQ-2** (SSRF Prevention) - Immediate (security critical)
3. **REQ-1** (Schema Fields) - Week 1-2 (data integrity)
4. **REQ-4** (Normalization) - Week 2-3 (code quality foundation)
5. **REQ-5** (Complexity) - Week 3-4 (builds on REQ-4, comprehensive refactor)

## Success Criteria

This specification is considered complete when:

1. All P1 requirements (REQ-1, REQ-2, REQ-3) are implemented and verified
2. All P2 requirements (REQ-4, REQ-5) are implemented and verified
3. Zero silent data loss in protobuf message transmission
4. Zero SSRF vulnerabilities in proxy configuration
5. All PRs are < 100 files and independently reviewable
6. Normalization logic exists in single module
7. Codebase LOC reduced by 50%+ through YAGNI compliance
8. All existing integration tests pass
9. Code review process demonstrates 50%+ time reduction
10. Security scan passes with zero critical/high findings

## Non-Functional Requirements

### Performance
- Schema field population SHOULD NOT add more than 5% latency overhead to converter functions
- SSRF validation SHOULD complete in less than 1ms per URL

### Maintainability
- Normalization logic MUST exist in single source file
- Module structure SHOULD be understandable by new developers within 30 minutes

### Security
- Proxy URL validation MUST prevent all SSRF attack vectors documented in OWASP guidelines
- Security regression tests MUST run in CI/CD pipeline

### Testability
- Each requirement MUST have dedicated unit tests
- Integration tests MUST verify end-to-end behavior
- Test coverage MUST remain above 85% after simplification

## Compliance & Standards

- **EARS Format**: All acceptance criteria use EARS syntax (WHEN-THEN, IF-THEN, WHILE-THE, WHERE-THE)
- **CLAUDE.md Alignment**: Requirements align with YAGNI, KISS, DRY, START SMALL principles
- **Security**: SSRF prevention follows OWASP guidelines (CWE-918)
- **Code Review**: PR size limits follow Google Engineering Practices (<500 LOC recommended)

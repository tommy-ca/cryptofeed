# Implementation Plan

## Task Overview

This implementation plan addresses PR #16 code review remediation across 5 requirements, prioritizing P1 critical issues (data integrity, security, PR scope) before P2 improvements (code quality, complexity reduction). Tasks follow TDD workflow, include validation checkpoints, and respect dependency sequencing.

**Total Estimated Effort**: 12-15 days (240-300 hours across team)
**Critical Path**: REQ-3 (PR Split) → REQ-2 (SSRF) → REQ-1 (Schema Fields) → REQ-4 (Normalization) → REQ-5 (Complexity)

---

## P1 Critical Requirements (Blocks Merge)

### REQ-2: SSRF Prevention in Proxy Configuration

- [x] 1. Implement URL validation defense-in-depth system
  - Build comprehensive URL validator with three security layers to prevent SSRF attacks
  - Implement scheme whitelist allowing only proxy protocols (http, https, socks4, socks5, socks5h)
  - Create IP range blacklist covering private networks, loopback, link-local, and cloud metadata endpoints
  - Add hostname pattern matching to block localhost variants and metadata services
  - Design validation to resist DNS rebinding and URL encoding bypass attempts
  - _Requirements: REQ-2.1, REQ-2.2, REQ-2.3, REQ-2.9, REQ-2.10_
  - _Estimated Effort: 3 hours_
  - _Risk: Low_
  - _Dependencies: None (foundation task)_
  - **Status: COMPLETE** - All 45 tests passing (34 unit + 11 integration)

- [x] 1.1 Create validate_proxy_url() function with scheme whitelist
  - Implement scheme validation layer using allowed set (http, https, socks4, socks5, socks5h)
  - Parse URLs using urllib.parse.urlparse for normalized scheme extraction
  - Reject file://, ftp://, gopher://, and all non-proxy schemes with specific error messages
  - Handle empty URLs gracefully (return without error for proxy disabled case)
  - Add comprehensive docstring documenting security properties and validation layers
  - _Requirements: REQ-2.2, REQ-2.3_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/run.py_
  - **Status: COMPLETE**

- [x] 1.2 Add IP range validation layer using ipaddress module
  - Define BLOCKED_IP_RANGES constant with ipaddress.ip_network objects for RFC 1918, RFC 1122, RFC 3927, RFC 4291 ranges
  - Parse hostname as IP address and check membership in blocked ranges
  - Handle both IPv4 and IPv6 addresses (127.0.0.1, ::1, fe80::/10)
  - Provide specific error messages indicating which blocked range was matched
  - Include SSRF prevention context in error messages for security awareness
  - _Requirements: REQ-2.4, REQ-2.5, REQ-2.6, REQ-2.8_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/run.py_
  - **Status: COMPLETE**

- [x] 1.3 Implement hostname pattern matching for localhost and metadata endpoints
  - Create BLOCKED_HOSTNAMES set with lowercase patterns (localhost, 127.0.0.1, ::1, metadata.google.internal)
  - Normalize hostname to lowercase before comparison to prevent case bypass
  - Handle non-IP hostnames that failed IP parsing in previous layer
  - Raise ValueError with blocked hostname and SSRF prevention message
  - Document additional metadata endpoints to block in comments for future extension
  - _Requirements: REQ-2.7, REQ-2.8_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/run.py_
  - **Status: COMPLETE**

- [x] 1.4 Integrate validator into load_proxy_mapping() configuration loader
  - Call validate_proxy_url() for all URLs in global proxy section
  - Call validate_proxy_url() for all URLs in per-exchange proxy sections
  - Handle both dict-style configs (proxy_type: url) and legacy string-style configs
  - Provide clear error paths indicating section location (global.http, exchanges.binance.socks5)
  - Fail fast on first invalid URL to prevent partial configuration state
  - _Requirements: REQ-2.1, REQ-2.9, REQ-2.13, REQ-2.14_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/run.py_
  - **Status: COMPLETE**

- [x] 2. Create comprehensive SSRF security test suite
  - Develop unit tests covering all blocked URL patterns and attack vectors
  - Build integration tests with malicious proxy configuration files
  - Verify legitimate proxy URLs pass validation without false positives
  - Test URL encoding bypass attempts and DNS rebinding scenarios
  - Validate error messages provide clear security context
  - _Requirements: REQ-2.15, REQ-2.16, REQ-2.18_
  - _Estimated Effort: 2 hours_
  - _Risk: Low_
  - _Dependencies: Task 1 (validator implementation)_
  - **Status: COMPLETE** - 45 tests passing (34 unit + 11 integration)

- [x] 2.1 Write unit tests for scheme validation layer
  - Test rejection of file://, ftp://, gopher://, and other non-proxy schemes
  - Verify acceptance of http, https, socks4, socks5, socks5h schemes
  - Test empty URL handling (should pass without error)
  - Verify error messages include scheme name and allowed schemes list
  - Use pytest parametrize for comprehensive scheme coverage
  - _Requirements: REQ-2.2, REQ-2.3, REQ-2.15_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/test_ssrf_validator.py_
  - **Status: COMPLETE**

- [x] 2.2 Write unit tests for IP range validation layer
  - Test blocking of private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
  - Test blocking of loopback addresses (127.0.0.0/8, ::1)
  - Test blocking of link-local and metadata endpoint (169.254.169.254, 169.254.0.0/16)
  - Test blocking of IPv6 link-local (fe80::/10)
  - Verify public IP addresses pass validation (8.8.8.8, 1.1.1.1)
  - _Requirements: REQ-2.4, REQ-2.5, REQ-2.6, REQ-2.8, REQ-2.15_
  - _Estimated Effort: 0.75 hours_
  - _File: tests/unit/test_ssrf_validator.py_
  - **Status: COMPLETE**

- [x] 2.3 Write unit tests for hostname pattern matching layer
  - Test blocking of localhost, 127.0.0.1, ::1 hostnames
  - Test blocking of cloud metadata endpoints (metadata.google.internal)
  - Test case-insensitive matching (Localhost, LOCALHOST should be blocked)
  - Verify legitimate hostnames pass (proxy.example.com)
  - Test hostnames that are not IPs (non-IP parsing path)
  - _Requirements: REQ-2.7, REQ-2.8, REQ-2.15_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/test_ssrf_validator.py_
  - **Status: COMPLETE**

- [x] 2.4 Write integration test with malicious proxy.yaml configuration
  - Create temporary proxy.yaml with SSRF attack patterns (metadata endpoint, file URI, private IP)
  - Verify load_proxy_mapping() raises ValueError on malicious configs
  - Test error messages include section path (global.http, exchanges.binance.socks5)
  - Verify legitimate proxy configs load successfully
  - Test both global and per-exchange validation failures
  - _Requirements: REQ-2.16_
  - _Estimated Effort: 0.25 hours_
  - _File: tests/integration/test_ssrf_proxy_config.py_
  - **Status: COMPLETE**

- [x] 3. Update documentation with SSRF prevention security notes
  - Add security warnings to proxy configuration documentation
  - Document blocked URL patterns and rationale for each category
  - Provide examples of legitimate vs. malicious proxy configurations
  - Include troubleshooting guide for validation errors
  - Reference OWASP SSRF guidelines and CWE-918 in documentation
  - _Requirements: REQ-2.17_
  - _Estimated Effort: 1 hour_
  - _Risk: Low_
  - _Dependencies: Task 1, Task 2 (implementation and tests complete)_

- [x] 3.1 Add SSRF security section to proxy user guide
  - Document the three validation layers (scheme, IP, hostname)
  - Explain why each category is blocked (metadata endpoints, local files, internal networks)
  - Provide clear examples of rejected URLs with error messages
  - Include guidance on configuring legitimate proxies to avoid false positives
  - Add reference links to OWASP SSRF attack documentation
  - _Requirements: REQ-2.17_
  - _Estimated Effort: 0.5 hours_
  - _File: docs/proxy/user-guide.md_

- [x] 3.2 Update proxy.yaml example with security comments
  - Add inline comments explaining URL validation requirements
  - Include examples of valid proxy configurations (public IPs, DNS names)
  - Warn against using localhost, private IPs, or cloud metadata endpoints
  - Document supported schemes (http, https, socks4, socks5, socks5h)
  - Note that file:// and ftp:// schemes are blocked for security
  - _Requirements: REQ-2.17_
  - _Estimated Effort: 0.25 hours_
  - _File: config/proxy.yaml_

- [x] 3.3 Add SSRF vulnerability to security runbook
  - Document CVE details (CVSS 7.5 High, CWE-918)
  - List all blocked patterns with attack scenario explanations
  - Provide incident response procedure if SSRF attempt detected
  - Include monitoring guidance (rejection rate alerts, security log review)
  - Reference validation code locations for security audits
  - _Requirements: REQ-2.17, REQ-2.18_
  - _Estimated Effort: 0.25 hours_
  - _File: docs/security/ssrf-prevention.md (new file)_

---

### REQ-3: PR Scope Management (Split 364-File PR)

- [ ] 4. Execute PR split strategy with dependency-aware sequencing
  - Split PR #16 (364 files) into 7 focused PRs following dependency graph
  - Ensure each PR is independently reviewable (<100 files, <5,000 LOC)
  - Organize commits by feature area using cherry-pick from original PR
  - Maintain test coverage and CI passing for each independent PR
  - Document dependency relationships to guide merge sequence
  - _Requirements: REQ-3.1, REQ-3.2, REQ-3.10, REQ-3.11, REQ-3.18_
  - _Estimated Effort: 12 hours (across 4 weeks)_
  - _Risk: Medium (coordination overhead, merge conflicts)_
  - _Dependencies: None (process task)_

- [ ] 4.1 Create PR #16.1 - Core Kafka Module Structure (Week 1, no dependencies)
  - Extract base.py, producer.py, topic_manager.py, partitioner.py from PR #16
  - Cherry-pick relevant commits for core module functionality
  - Include unit tests for each extracted module
  - Verify CI passes independently (approximately 60 files, 800 LOC)
  - Open PR with clear description of scope and zero dependencies
  - _Requirements: REQ-3.3_
  - _Estimated Effort: 2 hours_
  - _Branch: kafka-backend-1_

- [ ] 4.2 Create PR #16.2 - Protobuf Consolidation (Week 1, parallel with #16.1)
  - Extract cryptofeed/backends/protobuf/ package and schema updates
  - Cherry-pick commits related to protobuf converter refactoring
  - Include comprehensive protobuf serialization tests
  - Verify independent CI success (approximately 70 files, 1,200 LOC)
  - Open PR documenting no dependency on #16.1 (parallel execution)
  - _Requirements: REQ-3.4_
  - _Estimated Effort: 2 hours_
  - _Branch: kafka-backend-2_

- [ ] 4.3 Create PR #16.3 - Configuration Management (Week 2, depends on #16.1)
  - Extract config.py, Pydantic models, and validation logic
  - Base branch on main after #16.1 merge to include base classes
  - Cherry-pick configuration-related commits
  - Include config validation tests (approximately 35 files, 500 LOC)
  - Document dependency on #16.1 in PR description
  - _Requirements: REQ-3.5_
  - _Estimated Effort: 1.5 hours_
  - _Branch: kafka-backend-3 (from main after #16.1)_

- [ ] 4.4 Create PR #16.4 - Metrics & Observability (Week 2, depends on #16.1)
  - Extract metrics.py, health.py, health_server.py modules
  - Base branch on main after #16.1 merge to hook into callbacks
  - Cherry-pick metrics and health check commits
  - Include Prometheus integration tests (approximately 45 files, 900 LOC)
  - Document dependency on #16.1 callback hooks
  - _Requirements: REQ-3.6_
  - _Estimated Effort: 1.5 hours_
  - _Branch: kafka-backend-4 (from main after #16.1)_

- [ ] 4.5 Create PR #16.5 - Deprecation System (Week 3, depends on #16.3)
  - Extract simplified deprecation.py (23 LOC warning functions only)
  - Base branch on main after #16.3 merge for config migration utilities
  - Include migration.py in this PR (not moved to tools/ yet to maintain migration capability)
  - Add deprecation warning tests (approximately 25 files, 750 LOC)
  - Document dependency on configuration management
  - _Requirements: REQ-3.7_
  - _Estimated Effort: 1 hour_
  - _Branch: kafka-backend-5 (from main after #16.3)_

- [ ] 4.6 Create PR #16.6 - Legacy Compatibility Shims (Week 3, depends on #16.1, #16.3, #16.4)
  - Extract __init__.py compatibility layer and backward compatibility tests
  - Base branch on main after all core features merged (#16.1, #16.3, #16.4)
  - Include comprehensive legacy API tests to prevent regressions
  - Verify backward compatibility with existing code (approximately 35 files, 300 LOC)
  - Document dependencies on all core interfaces
  - _Requirements: REQ-3.8_
  - _Estimated Effort: 1.5 hours_
  - _Branch: kafka-backend-6 (from main after #16.1, #16.3, #16.4)_

- [ ] 4.7 Create PR #16.7 - Documentation Updates (Week 4, depends on all above)
  - Extract documentation reorganization and migration guides
  - Base branch on main after all implementation PRs merged
  - Include API documentation updates reflecting final state
  - Add migration examples and user guides (approximately 90 files, docs only)
  - Tag final merge as kafka-backend-refactor-complete
  - _Requirements: REQ-3.9_
  - _Estimated Effort: 2 hours_
  - _Branch: kafka-backend-7 (from main after all PRs)_

- [ ] 5. Validate PR split execution quality and merge safety
  - Verify all 7 PRs meet size constraints (<100 files, <5,000 LOC each)
  - Confirm dependency graph documented and followed during merge sequence
  - Run integration tests after each PR merge to verify cumulative state
  - Validate zero functionality regression compared to original PR #16
  - Execute final integration test after all PRs merged
  - _Requirements: REQ-3.12, REQ-3.13, REQ-3.17, REQ-3.19_
  - _Estimated Effort: 3 hours (spread across 4 weeks)_
  - _Risk: Low (validation checkpoints)_
  - _Dependencies: Task 4 (all PRs created and sequenced)_

- [ ] 5.1 Create CI workflow to enforce PR size constraints
  - Add GitHub Actions workflow checking file count (<100) and line additions (<5,000)
  - Fail CI if PR exceeds size limits with clear error message
  - Include exception mechanism for documentation-only PRs
  - Test workflow with sample PR exceeding limits (should fail)
  - Document PR size policy in CONTRIBUTING.md
  - _Requirements: REQ-3.1, REQ-3.2, REQ-3.18_
  - _Estimated Effort: 1 hour_
  - _File: .github/workflows/pr-size-check.yml (new file)_

- [ ] 5.2 Document dependency graph and merge sequence timeline
  - Create visual dependency graph showing PR relationships (using Mermaid)
  - Document 4-week merge timeline with parallel and sequential phases
  - Specify rollback procedure for each PR (independent revert capability)
  - Include integration test checkpoints after each merge
  - Provide team coordination guidance for parallel PR reviews
  - _Requirements: REQ-3.10, REQ-3.11_
  - _Estimated Effort: 1 hour_
  - _File: docs/kafka-backend-refactor/pr-split-plan.md (new file)_

- [ ] 5.3 Execute final integration test validating complete refactor
  - Run full integration test suite after PR #16.7 merged
  - Compare behavior against original PR #16 branch (functionality parity)
  - Verify all 7 PR features integrated correctly (no missing pieces)
  - Test backward compatibility with legacy code using old APIs
  - Confirm zero regressions in existing functionality
  - _Requirements: REQ-3.13, REQ-3.19_
  - _Estimated Effort: 1 hour_
  - _File: tests/integration/test_kafka_refactor_complete.py (new file)_

---

### REQ-1: Schema Field Population (Data Integrity)

- [x] 6. Extend Trade and OrderBook types with protobuf v2beta1 fields
  - Add optional attributes to Trade class for maker, event_time, match_id, liquidity_flag
  - Add optional attributes to OrderBook class for event_time, last_update_id
  - Use Cython cdef public declarations for performance
  - Ensure backward compatibility (existing code not passing new fields continues working)
  - Update type stubs for IDE autocomplete and type checking
  - _Requirements: REQ-1.1, REQ-1.2_
  - _Estimated Effort: 2 hours_
  - _Risk: Low_
  - _Dependencies: None (type layer foundation)_

- [x] 6.1 Add new Trade class attributes with optional types
  - Extend Trade Cython class with cdef public declarations (maker: bool, event_time: float, match_id: str, liquidity_flag: str)
  - Update __init__ method signature to accept optional new fields with None defaults
  - Preserve existing Trade constructor interface (no breaking changes)
  - Add docstring documentation for each new field with exchange examples
  - Verify Cython compilation succeeds with new attributes
  - _Requirements: REQ-1.1, REQ-1.3, REQ-1.4, REQ-1.5_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/types.pyx_

- [x] 6.2 Add new OrderBook class attributes with optional types
  - Extend OrderBook Cython class with cdef public declarations (event_time: float, last_update_id: int)
  - Update __init__ method signature to accept optional new fields with None defaults
  - Preserve existing OrderBook constructor interface (backward compatibility)
  - Add docstring documentation for each new field with exchange-specific notes
  - Test Cython compilation and type hints generation
  - _Requirements: REQ-1.2, REQ-1.6, REQ-1.7_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/types.pyx_

- [x] 6.3 Write unit tests for new type attributes
  - Test Trade instantiation with maker=True/False, event_time, match_id, liquidity_flag
  - Test OrderBook instantiation with event_time, last_update_id
  - Verify optional fields default to None when not provided
  - Test all new fields with realistic exchange data values
  - Ensure existing tests pass without modification (backward compatibility)
  - _Requirements: REQ-1.16_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/test_types_schema_fields.py (new file)_

- [x] 7. Implement Binance exchange field extraction
  - Extract maker flag from Binance WebSocket 'm' field in trade messages
  - Extract event timestamp from Binance 'E' field (convert milliseconds to seconds)
  - Extract match ID from Binance 'a' field (aggregate trade ID)
  - Extract order book event timestamp and last update ID from book messages
  - Handle missing fields gracefully (set to None if exchange doesn't provide)
  - _Requirements: REQ-1.3, REQ-1.4, REQ-1.5, REQ-1.6, REQ-1.7, REQ-1.15_
  - _Estimated Effort: 2 hours_
  - _Risk: Low_
  - _Dependencies: Task 6 (types extended)_
  - **Status: COMPLETE** - All 12 tests passing, no regressions

- [x] 7.1 Update Binance _trade() handler to extract new fields
  - Parse 'm' field from Binance trade message (boolean maker flag: true if buyer is maker)
  - Parse 'E' field for event timestamp (convert milliseconds to seconds: E/1000)
  - Parse 'a' field for aggregate trade ID (convert to string)
  - Pass extracted fields to Trade constructor with maker, event_time, match_id parameters
  - Handle missing fields with .get() and None fallback (graceful degradation)
  - _Requirements: REQ-1.3, REQ-1.4, REQ-1.5_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/exchanges/binance.py_
  - **Status: COMPLETE**

- [x] 7.2 Update Binance _book() handler to extract new fields
  - Parse 'E' field from order book update for event timestamp (milliseconds to seconds)
  - Parse 'u' field for final update ID in event (last_update_id)
  - Pass extracted fields to OrderBook constructor with event_time, last_update_id parameters
  - Test with live Binance WebSocket data to verify field presence
  - Document which Binance channels provide which fields
  - _Requirements: REQ-1.6, REQ-1.7_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/exchanges/binance.py_
  - **Status: COMPLETE**

- [x] 7.3 Create unit tests for Binance field extraction
  - Mock Binance WebSocket messages with 'm', 'E', 'a' fields
  - Verify Trade objects have maker, event_time, match_id populated correctly
  - Test order book messages with event timestamps and update IDs
  - Test missing field scenarios (fields set to None when not in message)
  - Validate timestamp conversion from milliseconds to seconds
  - _Requirements: REQ-1.15, REQ-1.16_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/test_binance_field_extraction.py (new file)_
  - **Status: COMPLETE** - 12 tests created, all passing

- [x] 8. Update protobuf converters to populate v2beta1 fields
  - Modify trade_to_proto() to conditionally populate maker, event_time, match_id, liquidity_flag
  - Modify orderbook_to_proto() to conditionally populate event_time, last_update_id
  - Use hasattr() checks to maintain backward compatibility with exchanges not yet supporting new fields
  - Convert timestamps to microseconds for protobuf (multiply seconds by 1,000,000)
  - Ensure optional fields remain unset in protobuf when source field is None
  - _Requirements: REQ-1.8, REQ-1.9, REQ-1.10, REQ-1.11, REQ-1.12, REQ-1.13, REQ-1.14_
  - _Estimated Effort: 2 hours_
  - _Risk: Low_
  - _Dependencies: Task 6, Task 7 (types and extraction implemented)_
  - **Status: COMPLETE** - All 15 tests passing, bindings updated to v2beta1, no regressions

- [x] 8.1 Extend trade_to_proto() converter with conditional field population
  - Add hasattr(trade_obj, 'maker') check before populating proto.maker field
  - Add hasattr(trade_obj, 'event_time') check before populating proto.event_time (convert to microseconds)
  - Add hasattr(trade_obj, 'match_id') check before populating proto.match_id
  - Add hasattr(trade_obj, 'liquidity_flag') check before populating proto.liquidity_flag
  - Leave fields unset in protobuf message when source attribute is None (do not populate with defaults)
  - _Requirements: REQ-1.8, REQ-1.9, REQ-1.10, REQ-1.11, REQ-1.14_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/backends/protobuf/converters.py_
  - **Status: COMPLETE**

- [x] 8.2 Extend orderbook_to_proto() converter with conditional field population
  - Add hasattr(orderbook_obj, 'event_time') check before populating proto.event_time (convert to microseconds)
  - Add hasattr(orderbook_obj, 'last_update_id') check before populating proto.last_update_id
  - Ensure existing order book levels population logic unchanged
  - Leave new fields unset when source attributes are None
  - Document field availability per exchange in converter docstring
  - _Requirements: REQ-1.12, REQ-1.13, REQ-1.14_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/backends/protobuf/converters.py_
  - **Status: COMPLETE**

- [x] 8.3 Write unit tests for protobuf converter field population
  - Test trade_to_proto() with Trade object containing all new fields (verify all populated)
  - Test trade_to_proto() with Trade object missing new fields (verify protobuf fields unset)
  - Test orderbook_to_proto() with OrderBook object containing new fields
  - Test orderbook_to_proto() with OrderBook object missing new fields
  - Verify timestamp conversion to microseconds (seconds * 1,000,000)
  - _Requirements: REQ-1.16_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/proto/test_protobuf_converters_fields.py (new file)_
  - **Status: COMPLETE** - 15 tests created, all passing

- [x] 9. Create integration tests for end-to-end field transmission
  - Build E2E test from Binance WebSocket mock through Kafka backend
  - Verify Trade protobuf messages in Kafka contain maker, event_time, match_id fields
  - Verify OrderBook protobuf messages contain event_time, last_update_id fields
  - Test field absence when exchange doesn't provide data (protobuf fields unset)
  - Measure zero silent data loss (all extracted fields transmitted)
  - _Requirements: REQ-1.17, REQ-1.19_
  - _Estimated Effort: 2 hours_
  - _Risk: Medium (Kafka dependency)_
  - _Dependencies: Task 6, Task 7, Task 8 (full pipeline implemented)_
  - **Status: COMPLETE** - 11 integration tests passing, monitoring script created

- [x] 9.1 Write Kafka integration test for Trade field transmission
  - Set up Kafka consumer listening to trade topic (cryptofeed.trade.binance.btc-usd)
  - Mock Binance trade WebSocket message with 'm', 'E', 'a' fields
  - Trigger trade processing through full pipeline (handler → converter → Kafka backend)
  - Consume protobuf message from Kafka and parse back to Trade proto object
  - Verify proto.maker, proto.event_time, proto.match_id fields populated correctly
  - _Requirements: REQ-1.17_
  - _Estimated Effort: 1 hour_
  - _File: tests/integration/test_kafka_field_population_e2e.py (new file)_

- [x] 9.2 Write Kafka integration test for OrderBook field transmission
  - Set up Kafka consumer for order book topic
  - Mock Binance order book WebSocket message with event timestamp and update ID
  - Process through pipeline and consume from Kafka
  - Parse protobuf OrderBook message and verify event_time, last_update_id populated
  - Test missing field scenario (fields unset in protobuf)
  - _Requirements: REQ-1.17_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/integration/test_kafka_field_population_e2e.py_

- [x] 9.3 Measure and validate zero silent data loss
  - Create monitoring script tracking field population rates across all exchanges
  - Calculate percentage of messages with new fields populated vs. total messages
  - Verify Binance messages have 100% population for supported fields
  - Verify exchanges not yet implemented have 0% population (expected, documented)
  - Document field availability matrix per exchange
  - _Requirements: REQ-1.19_
  - _Estimated Effort: 0.5 hours_
  - _File: tools/validate_field_population.py (new monitoring script)_

- [x] 10. Document field availability and implementation status per exchange
  - Create schema mapping documentation showing which exchanges support which fields
  - Document Binance field sources (WebSocket field names, data types, conversion logic)
  - Provide migration guide for adding new exchange field extraction
  - Update protobuf schema documentation with field usage examples
  - Document future work (OKX, Coinbase field extraction planned)
  - _Requirements: REQ-1.18_
  - _Estimated Effort: 1.5 hours_
  - _Risk: Low_
  - _Dependencies: Task 6, Task 7, Task 8, Task 9 (all implementation complete)_
  - **Status: COMPLETE** - All 3 documentation files created

- [x] 10.1 Create field availability matrix documentation
  - Build table showing Trade fields (maker, event_time, match_id, liquidity_flag) support per exchange
  - Build table showing OrderBook fields (event_time, last_update_id) support per exchange
  - Mark Binance as SUPPORTED with implementation status
  - Mark OKX, Coinbase, others as PLANNED or NOT_AVAILABLE based on exchange API capabilities
  - Include last updated date and link to exchange API documentation
  - _Requirements: REQ-1.18_
  - _Estimated Effort: 0.5 hours_
  - _File: docs/schemas/mappings/field_availability_matrix.md (new file)_
  - **Status: COMPLETE**

- [x] 10.2 Document Binance field mapping specification
  - Map Binance WebSocket fields ('m', 'E', 'a', 'u') to protobuf fields
  - Document data type conversions (boolean, milliseconds to seconds, integer to string)
  - Provide example raw WebSocket messages with field values
  - Include code snippet showing extraction logic
  - Reference Binance API documentation URLs for each field
  - _Requirements: REQ-1.18_
  - _Estimated Effort: 0.5 hours_
  - _File: docs/schemas/mappings/binance_field_mapping.md (new file)_
  - **Status: COMPLETE**

- [x] 10.3 Create migration guide for adding exchange field extraction
  - Provide step-by-step template for implementing new exchange field support
  - Document testing checklist (unit tests, integration tests, field availability update)
  - Include code examples for Trade and OrderBook field extraction patterns
  - Reference existing Binance implementation as canonical example
  - Add PR template for field extraction contributions
  - _Requirements: REQ-1.18_
  - _Estimated Effort: 0.5 hours_
  - _File: docs/schemas/migration/adding_exchange_fields.md (new file)_
  - **Status: COMPLETE**

---

## P2 Important Requirements (Code Quality)

### REQ-4: Normalization Code Deduplication

- [x] 11. Consolidate normalization logic into shared module
  - Create normalization utility module with normalize_symbol() and normalize_exchange() functions
  - Extract duplicate normalization code from topic_manager.py, partitioner.py, headers.py
  - Replace 3 duplicate implementations with single source of truth
  - Ensure consistent behavior across all usage sites (topic names, partition keys, headers)
  - Maintain 100% backward compatibility (identical output)
  - _Requirements: REQ-4.1, REQ-4.2, REQ-4.3, REQ-4.18_
  - _Estimated Effort: 1 hour (atomic refactoring)_
  - _Risk: Low_
  - _Dependencies: None (standalone refactoring)_
  - **Status: COMPLETE** - 20 unit tests + 4 integration tests passing, all duplicates removed

- [x] 11.1 Create cryptofeed/backends/kafka/normalization.py module
  - Implement normalize_symbol() function with lowercase, separator replacement, strip, 'unknown' fallback
  - Implement normalize_exchange() function with lowercase, strip, 'unknown' fallback
  - Add comprehensive docstrings with normalization rules and examples
  - Document rationale for each rule (Kafka topic naming, partition routing, header encoding)
  - Include type hints using PEP 604 union syntax (str | None)
  - _Requirements: REQ-4.1, REQ-4.2, REQ-4.3_
  - _Estimated Effort: 0.25 hours_
  - _File: cryptofeed/backends/kafka/normalization.py (new file)_
  - **Status: COMPLETE**

- [x] 11.2 Update topic_manager.py to import shared normalization functions
  - Remove local _normalize_symbol() and _normalize_exchange() functions
  - Add import statement: from .normalization import normalize_symbol, normalize_exchange
  - Replace all _normalize_symbol() calls with normalize_symbol()
  - Replace all _normalize_exchange() calls with normalize_exchange()
  - Verify topic naming behavior unchanged (regression test)
  - _Requirements: REQ-4.11, REQ-4.14_
  - _Estimated Effort: 0.1 hours_
  - _File: cryptofeed/backends/kafka/topic_manager.py_
  - **Status: COMPLETE**

- [x] 11.3 Update partitioner.py to import shared normalization functions
  - Remove local _normalize_symbol() and _normalize_exchange() functions
  - Add import statement: from .normalization import normalize_symbol, normalize_exchange
  - Replace all normalization calls with shared functions
  - Verify partition key generation behavior unchanged
  - Test partition routing consistency with previous implementation
  - _Requirements: REQ-4.12, REQ-4.14_
  - _Estimated Effort: 0.1 hours_
  - _File: cryptofeed/backends/kafka/partitioner.py_
  - **Status: COMPLETE**

- [x] 11.4 Update headers.py to import shared normalization functions
  - Remove inline normalization code (str().strip().replace() logic)
  - Add import statement: from .normalization import normalize_symbol, normalize_exchange
  - Replace inline normalization with normalize_symbol() and normalize_exchange() calls
  - Verify header encoding behavior unchanged
  - Test header values match previous format
  - _Requirements: REQ-4.13, REQ-4.14_
  - _Estimated Effort: 0.1 hours_
  - _File: cryptofeed/backends/kafka/headers.py_
  - **Status: COMPLETE**

- [x] 11.5 Verify all duplicate code removed from 3 original locations
  - Grep for remaining _normalize_symbol definitions (should find 0 results)
  - Grep for remaining _normalize_exchange definitions (should find 0 results)
  - Verify no inline normalization logic remains in headers.py
  - Confirm all call sites now use shared module
  - Run full test suite to ensure no regressions
  - _Requirements: REQ-4.14_
  - _Estimated Effort: 0.05 hours_
  - **Status: COMPLETE**

- [x] 12. Create comprehensive normalization test suite
  - Write unit tests for normalize_symbol() covering all edge cases
  - Write unit tests for normalize_exchange() covering all edge cases
  - Verify consistency across all usage sites (topic, partition, header)
  - Test backward compatibility (existing tests pass unchanged)
  - Document test coverage for normalization rules
  - _Requirements: REQ-4.4, REQ-4.5, REQ-4.6, REQ-4.7, REQ-4.8, REQ-4.9, REQ-4.10, REQ-4.15, REQ-4.16, REQ-4.17_
  - _Estimated Effort: 1 hour_
  - _Risk: Low_
  - _Dependencies: Task 11 (normalization module created)_
  - **Status: COMPLETE** - 20 unit tests + 4 integration tests passing, comprehensive coverage

- [x] 12.1 Write unit tests for normalize_symbol() function
  - Test 'BTC/USD' → 'btc-usd' (slash to hyphen, lowercase)
  - Test 'BTC_USD' → 'btc-usd' (underscore to hyphen, lowercase)
  - Test ' ETH-BTC ' → 'eth-btc' (whitespace stripping)
  - Test None → 'unknown' (None fallback)
  - Test '' → 'unknown' (empty string fallback)
  - Test '  ' → 'unknown' (whitespace-only fallback)
  - Test mixed separators: 'BTC/USD_PERP' → 'btc-usd-perp'
  - Test case variations: 'btc-usd', 'BTC-USD', 'Btc-Usd' all → 'btc-usd'
  - _Requirements: REQ-4.4, REQ-4.5, REQ-4.6, REQ-4.7, REQ-4.15_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/unit/test_normalization.py (new file)_
  - **Status: COMPLETE** - 10 tests created, all passing

- [x] 12.2 Write unit tests for normalize_exchange() function
  - Test 'Binance' → 'binance' (lowercase)
  - Test ' OKX ' → 'okx' (whitespace stripping)
  - Test 'COINBASE' → 'coinbase' (uppercase to lowercase)
  - Test None → 'unknown' (None fallback)
  - Test '' → 'unknown' (empty string fallback)
  - Test whitespace variations: '  ', '\t', '\n' all → 'unknown'
  - Test case preservation in lowercase conversion
  - _Requirements: REQ-4.8, REQ-4.9, REQ-4.10, REQ-4.15_
  - _Estimated Effort: 0.25 hours_
  - _File: tests/unit/test_normalization.py_
  - **Status: COMPLETE** - 8 tests created, all passing

- [x] 12.3 Write integration tests verifying consistency across all usage sites
  - Test same symbol normalization in topic name, partition key, header value
  - Test same exchange normalization across all three contexts
  - Verify topic_manager, partitioner, headers produce identical normalized values
  - Test with variety of symbols and exchanges (edge cases, common cases)
  - Assert consistency: topic_value == partition_value == header_value for same input
  - _Requirements: REQ-4.16_
  - _Estimated Effort: 0.25 hours_
  - _File: tests/integration/test_normalization_consistency.py (new file)_
  - **Status: COMPLETE** - 4 integration tests created, all passing

---

### REQ-5: Complexity Reduction (YAGNI Compliance)

- [x] 13. Execute Phase 1 - Delete dead code (868 LOC reduction, zero risk)
  - Remove maintenance/__init__.py module (135 LOC of no-op bridge patterns)
  - Simplify deprecation.py to 23 LOC (keep 2 warning functions, delete infrastructure)
  - Move migration.py to tools/ directory (229 LOC removed from runtime package)
  - Verify all tests pass after deletion (no functional dependencies)
  - Measure LOC reduction and validate zero functional impact
  - _Requirements: REQ-5.2, REQ-5.1, REQ-5.3, REQ-5.10_
  - _Estimated Effort: 1.5 hours_
  - _Risk: Low (deleting unused code)_
  - _Dependencies: REQ-3 (PR split complete, this is part of simplification PR)_
  - **Status: COMPLETE** - Achieved 848 LOC reduction (98% of target)

- [x] 13.1 Delete maintenance/__init__.py module entirely
  - Remove cryptofeed/backends/kafka/maintenance/__init__.py (135 LOC)
  - Remove all imports of maintenance module from other files
  - Verify no runtime references to maintenance code (grep for 'maintenance')
  - Remove maintenance module tests if they exist
  - Confirm test suite passes without maintenance module
  - _Requirements: REQ-5.2_
  - _Estimated Effort: 0.25 hours_
  - _File: cryptofeed/backends/kafka/maintenance/__init__.py (delete)_
  - **Status: COMPLETE** - Module and directory deleted, imports updated

- [x] 13.2 Simplify deprecation.py to warning functions only
  - Keep emit_deprecation_warning() and warn_legacy_usage() functions (23 LOC total)
  - Delete timeline management infrastructure (DeprecationTimeline class, 400+ LOC)
  - Delete milestone tracking, communication channel configs, ADR references
  - Delete project management code (belongs in issue tracker, not runtime)
  - Update docstring to reflect simplified purpose (warning only, no timeline tracking)
  - _Requirements: REQ-5.1_
  - _Estimated Effort: 0.5 hours_
  - _File: cryptofeed/backends/kafka/deprecation.py_
  - **Status: COMPLETE** - Reduced to 44 LOC (4 warning functions), 484 LOC removed

- [x] 13.3 Move migration.py to tools/ directory
  - Move cryptofeed/backends/kafka/migration.py to tools/migrate_kafka_config.py
  - Remove import of migration module from runtime code
  - Update migration script to be standalone (no runtime dependencies)
  - Add CLI interface if needed (argparse for config file path)
  - Document migration script usage in tools/README.md
  - _Requirements: REQ-5.3_
  - _Estimated Effort: 0.5 hours_
  - _Old: cryptofeed/backends/kafka/migration.py_
  - _New: tools/migrate_kafka_config.py_
  - **Status: COMPLETE** - 229 LOC moved to tools/, all imports updated

- [x] 13.4 Validate Phase 1 completion and measure reduction
  - Run full test suite to verify zero functional regressions
  - Count LOC reduction: maintenance (135) + deprecation infra (484) + migration move (229) = 848 LOC
  - Verify percentage: 848 / 3576 = 23.7% reduction from Phase 1 alone
  - Document deleted modules in commit message
  - Tag commit as phase-1-dead-code-removal
  - _Requirements: REQ-5.10_
  - _Estimated Effort: 0.25 hours_
  - **Status: COMPLETE** - 24 tests passing, zero regressions

- [ ] 14. Execute Phase 2 - Inline trivial abstractions (500 LOC reduction, low risk)
  - Inline headers.py module into callback.py as 20-line function
  - Replace partitioner.py factory pattern with 15-line inline if/elif function
  - Simplify health.py to basic health check function (30 lines)
  - Remove abstraction overhead from trivial operations
  - Verify behavioral equivalence with regression tests
  - _Requirements: REQ-5.5, REQ-5.4, REQ-5.6, REQ-5.11_
  - _Estimated Effort: 2 hours_
  - _Risk: Low (simple inlining)_
  - _Dependencies: Task 13 (Phase 1 complete)_

- [ ] 14.1 Inline headers.py module into callback.py
  - Copy header encoding logic (20 lines) directly into callback.__call__() method
  - Remove cryptofeed/backends/kafka/headers.py file (374 LOC)
  - Remove import of headers module from callback.py
  - Inline normalize_symbol() and normalize_exchange() calls from shared normalization module
  - Verify header encoding behavior unchanged (regression test)
  - _Requirements: REQ-5.5_
  - _Estimated Effort: 0.5 hours_
  - _Files: cryptofeed/backends/kafka/callback.py (modify), headers.py (delete)_

- [ ] 14.2 Replace partitioner.py factory with inline function
  - Replace PartitionerFactory and 4 strategy classes with simple if/elif function
  - Inline partition key logic directly in callback.__call__() (15 lines)
  - Remove cryptofeed/backends/kafka/partitioner.py file (91 LOC)
  - Remove import of partitioner module
  - Test partition routing behavior unchanged (all 4 strategies work identically)
  - _Requirements: REQ-5.4_
  - _Estimated Effort: 0.75 hours_
  - _Files: cryptofeed/backends/kafka/callback.py (modify), partitioner.py (delete)_

- [ ] 14.3 Simplify health.py to basic function
  - Reduce health check to simple function returning status dict (30 lines)
  - Remove elaborate health check infrastructure (HealthMonitor class, etc.)
  - Keep essential producer health check (is producer connected?)
  - Delete cryptofeed/backends/kafka/health.py complex implementation (189 LOC)
  - Create simple health_check() function in callback.py
  - _Requirements: REQ-5.6_
  - _Estimated Effort: 0.5 hours_
  - _Files: cryptofeed/backends/kafka/callback.py (add function), health.py (delete or simplify)_

- [ ] 14.4 Validate Phase 2 completion and measure reduction
  - Run full test suite to verify behavioral preservation
  - Count LOC reduction: headers (354) + partitioner (80) + health (100) = 534 LOC
  - Verify cumulative reduction: Phase 1 (868) + Phase 2 (534) = 1,402 LOC (39.2%)
  - Test partition routing, header encoding, health checks all function identically
  - Tag commit as phase-2-inline-abstractions
  - _Requirements: REQ-5.11, REQ-5.17_
  - _Estimated Effort: 0.25 hours_

- [ ] 15. Execute Phase 3 - Consolidate modules (700 LOC reduction, medium risk)
  - Merge base.py, producer.py, topic_manager.py into single backend.py module
  - Flatten config.py from 4 Pydantic classes to 1 simple dataclass
  - Simplify metrics.py by using prometheus_client directly (remove wrappers)
  - Reduce 15 files to 4 files: backend.py, config.py, _deprecated.py, __init__.py
  - Verify comprehensive behavioral equivalence with existing integration tests
  - _Requirements: REQ-5.9, REQ-5.7, REQ-5.8, REQ-5.12, REQ-5.13_
  - _Estimated Effort: 4 hours_
  - _Risk: Medium (significant consolidation)_
  - _Dependencies: Task 13, Task 14 (Phase 1 and 2 complete)_

- [ ] 15.1 Merge base.py, producer.py, topic_manager.py into backend.py
  - Combine KafkaCallback base class, producer wrapper, and topic manager into single module
  - Organize backend.py into logical sections (callback class, producer methods, topic naming)
  - Remove 3 separate files (base.py, producer.py, topic_manager.py)
  - Update imports in other modules to reference backend.py
  - Verify callback functionality unchanged (full test suite)
  - _Requirements: REQ-5.9_
  - _Estimated Effort: 2 hours_
  - _Files: cryptofeed/backends/kafka/backend.py (new consolidated), base.py/producer.py/topic_manager.py (delete)_

- [ ] 15.2 Flatten config.py from 4 Pydantic classes to 1 dataclass
  - Replace KafkaConfig, ProducerConfig, TopicConfig, MetricsConfig with single KafkaConfig dataclass
  - Use standard library dataclass (no Pydantic dependency for simple config)
  - Flatten nested configuration into single-level attributes
  - Keep from_yaml() class method for YAML loading
  - Verify configuration loading behavior unchanged
  - _Requirements: REQ-5.7_
  - _Estimated Effort: 1 hour_
  - _File: cryptofeed/backends/kafka/config.py_

- [ ] 15.3 Simplify metrics.py by using prometheus_client directly
  - Remove wrapper classes (MetricsCollector, custom counter/gauge abstractions)
  - Use prometheus_client.Counter and prometheus_client.Gauge directly in backend.py
  - Delete cryptofeed/backends/kafka/metrics.py (407 LOC)
  - Move essential metric definitions to backend.py (approximately 50 lines)
  - Test metrics collection behavior unchanged
  - _Requirements: REQ-5.8_
  - _Estimated Effort: 0.75 hours_
  - _Files: cryptofeed/backends/kafka/backend.py (add metrics), metrics.py (delete)_

- [ ] 15.4 Validate Phase 3 completion and measure final reduction
  - Run full integration test suite to verify behavioral preservation
  - Count LOC reduction: module merging (700) + config flattening (270) + metrics simplification (250) = 1,220 LOC
  - Verify cumulative reduction: Phase 1 (868) + Phase 2 (534) + Phase 3 (1,220) = 2,622 LOC
  - Calculate final reduction percentage: 2,622 / 3,576 = 73.3% LOC reduction
  - Verify final module count: 4 files (backend.py, config.py, _deprecated.py, __init__.py)
  - Tag commit as phase-3-module-consolidation
  - _Requirements: REQ-5.12, REQ-5.13, REQ-5.16, REQ-5.17_
  - _Estimated Effort: 0.25 hours_

- [ ] 16. Create comprehensive regression test suite for simplified backend
  - Build integration tests comparing old backend behavior with simplified backend
  - Verify message production, topic naming, partition routing, header encoding identical
  - Test backward compatibility with legacy API usage
  - Measure performance (latency, throughput) matches baseline within 5%
  - Validate YAGNI compliance with CLAUDE.md principles
  - _Requirements: REQ-5.14, REQ-5.15, REQ-5.17, REQ-5.19_
  - _Estimated Effort: 2 hours_
  - _Risk: Low (validation only)_
  - _Dependencies: Task 13, Task 14, Task 15 (all phases complete)_

- [ ] 16.1 Write integration test comparing old vs. new backend behavior
  - Set up parallel test with original backend (pre-simplification) and simplified backend
  - Send identical Trade object through both backends
  - Verify Kafka messages are byte-identical (topic, partition key, headers, protobuf value)
  - Test all 4 partition strategies produce same routing
  - Test header encoding matches exactly
  - _Requirements: REQ-5.17_
  - _Estimated Effort: 1 hour_
  - _File: tests/integration/test_kafka_simplification_regression.py (new file)_

- [ ] 16.2 Write backward compatibility tests for legacy API usage
  - Test deprecated class names still work (KafkaProducer → KafkaCallback)
  - Test deprecated module paths still importable (with warnings)
  - Test old config format converts to new KafkaConfig correctly
  - Verify deprecation warnings emitted (but functionality works)
  - Document migration path for users on legacy APIs
  - _Requirements: REQ-5.17_
  - _Estimated Effort: 0.5 hours_
  - _File: tests/integration/test_kafka_legacy_compatibility.py (new file)_

- [ ] 16.3 Validate YAGNI compliance and measure quality improvements
  - Verify final structure matches CLAUDE.md principles: KISS, YAGNI, START SMALL
  - Measure code review time reduction (estimate 50%+ due to simplified codebase)
  - Verify test count reduced proportionally (170+ → ~40 tests)
  - Confirm test coverage remains above 85% (behavioral coverage preserved)
  - Document YAGNI violations removed and rationale
  - _Requirements: REQ-5.18, REQ-5.19_
  - _Estimated Effort: 0.5 hours_
  - _File: docs/kafka-backend-refactor/yagni-compliance-report.md (new file)_

---

## Implementation Workflow Guidelines

### TDD Workflow (All Tasks)

1. **Write Test First**
   - Define expected behavior in test
   - Run test and verify it fails (red)
   - Implement minimal code to pass test (green)
   - Refactor while keeping tests green
   - Never modify tests to fit implementation

2. **Validation Checkpoints**
   - After each task: Run relevant unit tests
   - After each major task: Run integration tests
   - After each requirement: Run full test suite
   - Before commit: Verify CI passes locally

3. **Rollback Strategies**
   - REQ-2 (SSRF): Immediate rollback if legitimate proxies blocked
   - REQ-3 (PR Split): Independent PR revert via git revert <merge-commit>
   - REQ-1 (Schema Fields): Phased rollback (converters → exchanges → types)
   - REQ-4 (Normalization): Single commit revert (atomic refactoring)
   - REQ-5 (Complexity): Phase-by-phase rollback (revert Phase 3 → Phase 2 → Phase 1)

4. **Integration Points**
   - REQ-2 validates before REQ-1 (proxy security must be solid before field extraction)
   - REQ-3 executes first (PR split enables parallel work on REQ-1, REQ-2)
   - REQ-4 before REQ-5 (normalization consolidation simplifies complexity reduction)
   - REQ-5 Phase 1 before Phase 2 before Phase 3 (sequential complexity reduction)

---

## Requirements Coverage Summary

**REQ-1 (Schema Fields)**: Tasks 6, 7, 8, 9, 10 (19 acceptance criteria → 20 sub-tasks)
**REQ-2 (SSRF Prevention)**: Tasks 1, 2, 3 (18 acceptance criteria → 11 sub-tasks)
**REQ-3 (PR Split)**: Tasks 4, 5 (19 acceptance criteria → 10 sub-tasks)
**REQ-4 (Normalization)**: Tasks 11, 12 (18 acceptance criteria → 8 sub-tasks)
**REQ-5 (Complexity)**: Tasks 13, 14, 15, 16 (19 acceptance criteria → 16 sub-tasks)

**Total**: 16 major tasks, 65 sub-tasks covering all 93 acceptance criteria across 5 requirements.

---

## Dependency Graph

```
REQ-3 (PR Split) ────────┐
                         ├──> REQ-2 (SSRF) ──> REQ-1 (Schema Fields) ──> REQ-4 (Normalization) ──> REQ-5 (Complexity)
                         │
                         └──> (Parallel execution enabled for REQ-1, REQ-2 after split)

Phases:
- Week 1: REQ-3 (PR Split setup) + REQ-2 (SSRF - quick win)
- Week 2-3: REQ-1 (Schema Fields - data integrity)
- Week 3: REQ-4 (Normalization - code quality)
- Week 4: REQ-5 (Complexity - YAGNI compliance)
```

---

## Risk Assessment

**Critical Risks**:
- REQ-3 PR Split coordination overhead (Medium) - Mitigated by dependency graph documentation
- REQ-1 Field extraction exchange variations (Low) - Mitigated by phased rollout (Binance first)
- REQ-5 Phase 3 module consolidation (Medium) - Mitigated by comprehensive regression tests

**Low Risks**:
- REQ-2 SSRF validation (Low) - Well-defined security requirements, existing libraries
- REQ-4 Normalization DRY (Low) - Atomic refactoring, 100% backward compatibility
- REQ-5 Phase 1-2 (Low) - Deleting unused code and inlining trivial functions

**Mitigation Strategies**:
- TDD workflow enforces behavioral preservation
- Validation checkpoints after each task prevent accumulation of issues
- Independent PR testing isolates failures to specific features
- Rollback procedures documented per requirement

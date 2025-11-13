# Requirements Document

## Introduction
Improve the schema parity regression workflow so Decimal precision is preserved, dataclass reconstruction scales with new event types, and protobuf converters stay synchronized with schema additions.

## Requirements

### Requirement 1: Decimal-Safe Regression Validation
**Objective:** As a quality engineer, I want the schema regression tool to operate entirely on Decimal-native data, so that parity checks surface real mismatches without precision loss.

#### Acceptance Criteria
1. WHEN the regression tool loads JSONL fixtures THEN it SHALL parse numeric fields using Decimal without intermediate float conversion.
2. IF a tolerance comparison is required THEN the tool SHALL compare Decimal operands directly and surface the absolute tolerance used in the report.
3. WHILE regression runs execute THE tool SHALL preserve all numeric strings—even those exceeding 18 decimal places—without rounding.
4. WHERE Decimal parsing fails due to malformed input THE tool SHALL emit a structured warning that references the offending line.

### Requirement 2: Event Coverage via Unified Constructors
**Objective:** As a maintainer, I want parity validation to reuse the canonical dataclass factories, so that every Cryptofeed event type gains parity coverage without redundant logic.

#### Acceptance Criteria
1. WHEN the tool reconstructs an event THEN it SHALL locate the corresponding `cryptofeed.types` class dynamically and invoke its `from_dict` helper where available.
2. IF an event type lacks a factory THEN the tool SHALL report the missing constructor and skip only that event.
3. WHILE new proto-backed event types are added THE tool SHALL require no manual lambda updates to begin coverage.
4. WHERE optional fields are absent THE tool SHALL respect defaults defined by the dataclass instead of failing the event.

### Requirement 3: Converter Registry Synchronization
**Objective:** As a backend engineer, I want protobuf serialization helpers to register converters automatically for every supported schema, so that normalized messages never raise `SerializationError` due to missing wiring.

#### Acceptance Criteria
1. WHEN new proto modules are introduced THEN an automated registry builder SHALL detect and register the associated converters and schema classes.
2. IF a converter is missing during serialization THEN the helper module SHALL raise an actionable error that lists the unregistered type and suggests regeneration.
3. WHILE CI quality gates run THE registry builder SHALL verify that converters exist for all proto message types tagged for ingestion.
4. WHERE manual overrides are needed (e.g., legacy schemas) THE system SHALL allow explicit exclusions documented within the registry configuration.


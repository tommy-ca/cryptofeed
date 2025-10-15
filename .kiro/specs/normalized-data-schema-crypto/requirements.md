# Requirements Document

## Introduction
The normalized-data-schema-crypto initiative aligns external market data sources with
Cryptofeed's event model by researching industry formats, extending tardis-node
schemas, modeling DBN fixed records with crypto-specific attributes, and enabling
feed callbacks to emit DBN-compliant payloads. Delivering formal requirements ensures
downstream ingestion teams receive consistent, well-governed schema definitions that
accelerate analytics and lakehouse adoption.

## Requirements

### Requirement 1: Schema Landscape Assessment
**Objective:** As a Data Platform Architect, I want a comprehensive view of existing
market data schemas and data flows, so that we can identify required crypto
extensions before modifying tardis-node or DBN baselines and confirm how exchange
feeds propagate into normalized outputs.

#### Acceptance Criteria
1. WHEN a crypto market data format or feed flow (exchange-native payloads,
   tardis-node snapshots, DBN fixed records, or open-standard schemas) is evaluated
   THEN the Normalized Data Schema Initiative SHALL log its field inventory, data
   types, timestamp resolution, transport path, and licensing constraints in the
   schema research register within two business days.
2. IF a format lacks coverage for trades, order books, liquidations, funding, or
   options Greeks THEN the Normalized Data Schema Initiative SHALL classify the
   gap severity and recommend candidate sources or derivations in the coverage
   matrix.
3. WHILE schema research is in progress THE Normalized Data Schema Initiative
   SHALL maintain a version-controlled comparison matrix that maps providers to
   supported event classes, serialization formats, precision standards, and
   upstream/downstream code flow touchpoints.
4. WHERE conflicting definitions of a shared field (e.g., trade side, venue ID,
   sequence numbers) are discovered THE Normalized Data Schema Initiative SHALL
   schedule a schema review workshop within five business days and record the
   adjudicated definition.

### Requirement 2: Tardis-Node Schema Extensions
**Objective:** As a Streaming Data Engineer, I want updated tardis-node schemas and
documented data flow mappings that encode crypto-specific attributes, so that
normalized outputs remain machine-verifiable and interoperable with Cryptofeed feeds.

#### Acceptance Criteria
1. WHEN tardis-node normalized outputs require crypto-specific metadata THEN the
   Normalized Data Schema Initiative SHALL define canonical field names, data
   types, and enumerations covering instrument identity, venue codes, tick size,
   and maker/taker flags.
2. IF an existing tardis-node field conflicts with the unified Cryptofeed schema
   (naming, type, or semantics) THEN the Normalized Data Schema Initiative SHALL
   propose an alias or migration note that preserves backward compatibility and
   documents transformation logic.
3. WHERE order book depth arrays are represented THE Normalized Data Schema
   Initiative SHALL specify level limits, precision rules (Decimal scale),
   required sequencing flags, and the data flow path from exchange adapters to
   normalized tardis-node outputs to prevent ambiguity in downstream reconstruction.
4. WHEN tardis-node schema updates reach review-ready status THEN the Normalized
   Data Schema Initiative SHALL produce versioned JSON Schema definitions and
   sample payloads illustrating trades, L2 snapshots, funding, and options data.

### Requirement 3: DBN Fixed Schema Crypto Extensions
**Objective:** As a Historical Data Product Manager, I want DBN fixed-width
records to capture crypto observables and trace how exchange feeds map into DBN
records, so that archival datasets remain lossless and query-efficient across new
venues.

#### Acceptance Criteria
1. WHEN new crypto event types are added to DBN fixed schemas THEN the Normalized
   Data Schema Initiative SHALL allocate record identifiers, byte layouts, and
   endianness rules consistent with existing DBN conventions while documenting
   the normalized data format contract.
2. IF crypto price or quantity fields exceed current DBN value ranges THEN the
   Normalized Data Schema Initiative SHALL document revised scaling factors or
   alternate encodings that maintain deterministic precision and update the
   normalized schema reference.
3. WHERE DBN ingest pipelines intersect tardis-node or Cryptofeed adapters THE
   Normalized Data Schema Initiative SHALL define field-level mapping tables,
   normalized field semantics, checksum expectations, and documented code flow
   diagrams to guarantee bidirectional conversion fidelity.
4. WHEN DBN schema extensions are approved THEN the Normalized Data Schema
   Initiative SHALL deliver validation fixtures (binary + decoded JSON),
   automated conformance tests runnable via the shared CI harness, and an
   updated normalized data format specification sheet.

### Requirement 4: Governance, Tooling, and Adoption
**Objective:** As a Quant Platform Lead, I want governed rollout processes, so
that engineering, data science, and compliance teams adopt the new schemas
without disrupting production workloads.

#### Acceptance Criteria
1. WHEN schema decisions are ratified THEN the Normalized Data Schema Initiative
   SHALL publish changelog entries, migration guides, and FAQ updates in the
   documentation hub within three business days.
2. IF partner teams request schema clarifications or feature additions THEN the
   Normalized Data Schema Initiative SHALL triage the request within two
   business days and assign an owner with a documented response timeline.
3. WHILE schema migrations are in progress THE Normalized Data Schema Initiative
   SHALL maintain automated regression checks that replay representative tardis-
   node and DBN payloads through Cryptofeed normalization pipelines, tracing end-
   to-end data flows from exchange inputs to normalized outputs.
4. WHERE production environments require phased rollout THE Normalized Data
   Schema Initiative SHALL provide versioned configuration toggles and sample
   deployment playbooks aligned with proxy and lakehouse integration patterns.

### Requirement 5: Feed Callback Enablement
**Objective:** As a Feed Platform Engineer, I want Cryptofeed callbacks to emit
DBN-formatted payloads, so that downstream systems can rely on a single,
normalized data interface across historical and streaming paths.

#### Acceptance Criteria
1. WHEN Cryptofeed emits callbacks for trades, order books, funding, or
   positions THEN the Normalized Data Schema Initiative SHALL ensure feed
   handlers can publish DBN-compliant records alongside existing dataclasses.
2. IF a callback consumer opts into DBN mode THEN the Normalized Data Schema
   Initiative SHALL provide configuration flags and documentation describing how
   to serialize normalized events into DBN fixed records.
3. WHILE DBN callback mode is enabled THE Normalized Data Schema Initiative
   SHALL guarantee parity tests that compare standard Cryptofeed dataclasses
   with their DBN counterparts for timestamp, sequence, and precision fidelity.
4. WHERE callbacks interact with proxy-aware transports THE Normalized Data
   Schema Initiative SHALL confirm that DBN payload emission does not bypass
   proxy logging or metrics instrumentation and documents the code flow path from
   transport adapters to DBN serialization hooks.

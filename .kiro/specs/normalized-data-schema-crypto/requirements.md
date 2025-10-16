# Requirements Document

## Introduction
The normalized-data-schema-crypto initiative standardizes crypto market data by
deriving canonical Protobuf schemas with Cryptofeed dataclasses as the primary
source of truth. tardis-node JSON exports and DBN fixed layouts complement the
Cryptofeed types by supplying auxiliary fields, metadata, or transport constraints
that inform the unified schema. The program
eliminates the previous bespoke schema package and instead adopts a Buf-based
workflow that publishes versioned modules to the Buf Schema Registry (BSR) under a dedicated namespace such as `buf.build/tommyk/crypto-market-data`.
Downstream teams consume these schema bundles to guarantee parity across
streaming, historical, and analytics pipelines without maintaining parallel
serialization stacks.

## Requirements

### Requirement 1: Cross-Source Schema Inventory
**Objective:** As a Data Schema Analyst, I want a governed schema inventory that
collects and compares field definitions across the expanding catalog of normalized events, treating Cryptofeed dataclass fields as canonical while recording tardis-node and DBN equivalents as complementary references before generating Buf modules. The inventory must grow with every newly supported Cryptofeed dataclass (e.g., balances, liquidations, options Greeks, NBBO, portfolio snapshots).

#### Acceptance Criteria
1. WHEN schema artifacts (Cryptofeed dataclasses, tardis-node exports, DBN
   layouts) are collected THEN the initiative SHALL capture field names, data
   types, units, precision, and known aliases in a normalized comparison matrix
   within two business days.
2. IF conflicting definitions exist across sources THEN the Cryptofeed dataclass
   SHALL be treated as authoritative unless a documented rationale justifies a
   divergence; adjudication workshops occur within five business days and record
   how tardis-node or DBN complements are adjusted to stay aligned.
3. WHILE schema research remains active THE initiative SHALL tag each normalized
   event (trade, order book, funding, options, balances) with a coverage status
   of complete, partial, or missing, including severity rationale for any gaps
   and explicit mappings that compare Cryptofeed fields to their tardis-node and
   DBN counterparts.

### Requirement 2: Protobuf Canonicalization via Buf CLI
**Objective:** As a Streaming Platform Engineer, I want canonical Protobuf
schemas generated with `buf`, so that downstream services consume a single
contract anchored on Cryptofeed dataclasses.

#### Acceptance Criteria
1. WHEN canonical field definitions are approved THEN the initiative SHALL
   produce `.proto` files using Buf module scaffolding (`buf.yaml`,
   `buf.gen.yaml`, `buf.lock`) aligned with the Cryptofeed namespace.
2. IF tardis-node or DBN structures provide metadata beyond the Cryptofeed
   dataclass THEN the initiative SHALL mirror required fields using deterministic
   casing, enum values, Decimal scale comments, and reserved field numbers while
   noting that the source of truth remains the Cryptofeed type.
3. WHEN schemas are generated THEN `buf lint` and `buf breaking --against` SHALL
   run in CI to enforce style and backward-compatibility guarantees before a
   release tag is published.
4. IF transport-specific annotations (e.g., JSON, gRPC gateway) are necessary
   THEN they SHALL be confined to Buf-managed options, ensuring the canonical
   Protobuf remains transport-agnostic.

### Requirement 3: BSR Publication & Versioning
**Objective:** As a Release Manager, I need normalized schemas available through
the Buf Schema Registry, so that internal and external consumers can pin to
semantic versions.

#### Acceptance Criteria
1. WHEN a schema release candidate is approved THEN the initiative SHALL publish
   the module to the configured BSR namespace using authenticated Buf CLI
   workflows, tagging the module with semantic version identifiers and release
   notes.
2. IF breaking changes are introduced THEN the initiative SHALL increment the
   major version and include migration guidance tying fields back to their
   originating Cryptofeed dataclasses (with tardis-node/DBN references updated
   accordingly).
3. WHEN modules are published THEN digest verification and module metadata
   (owners, visibility, dependencies) SHALL be stored alongside change logs in
   the documentation hub.
4. IF dependent teams raise adoption issues via BSR feedback THEN the initiative
   SHALL triage within two business days and track resolution status.

### Requirement 4: Tardis-Node and DBN Alignment
**Objective:** As a Historical Data Product Owner, I want the Buf schemas to map
directly onto tardis-node JSON exports and DBN fixed layouts—using Cryptofeed
dataclasses as the canonical reference—so that historical replay and real-time
streams share identical semantics.

#### Acceptance Criteria
1. WHEN new Protobuf revisions are published THEN updated tardis-node JSON
   Schema references and DBN layout documentation SHALL cross-link exact field
   numbers, scaling, and enumerations back to the Buf module and the originating
   Cryptofeed dataclass.
2. IF DBN numeric ranges or tardis-node JSON types cannot express the canonical
   schema THEN the initiative SHALL record explicit translation guidance and
   tooling requirements (e.g., scaling factors, enum adapters) prior to release.
3. WHILE schema alignment is in progress THE initiative SHALL maintain automated
   parity checks that replay representative tardis-node and DBN samples through
   generated Protobuf encoders, failing on mismatched timestamp, sequence, or
   precision fields.
4. WHEN alignment testing passes THEN release notes SHALL include readiness gates
   for tardis-node pipeline deployment and DBN archival ingestion referencing the
   canonical Cryptofeed-derived schema.

### Requirement 5: Governance & Documentation
**Objective:** As a Quant Platform Lead, I want transparent governance around
schema updates, so that teams adopt Buf modules confidently without diverging
implementations.

#### Acceptance Criteria
1. WHEN module versions are published to the BSR THEN migration guides, changelog
   entries, and curated FAQs SHALL be updated within three business days,
   referencing the canonical Protobuf package and Buf module slug.
2. IF consumers request schema enhancements THEN the initiative SHALL log the
   request, respond within two business days, and track status through resolution
   (accepted, deferred, or rejected) with rationale.
3. WHILE schema adoption is rolling out THE initiative SHALL expose Buf module
   metrics (downloads, dependents, breaking-change alerts) via a shared dashboard
   for engineering leadership review.
4. WHERE production environments require staged rollout THE initiative SHALL
   provide configuration toggles or fallbacks (e.g., per-feed schema version
   selection) coupled with Buf module compatibility guidance.

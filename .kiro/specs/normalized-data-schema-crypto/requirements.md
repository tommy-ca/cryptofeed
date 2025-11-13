# Requirements Document

## Introduction
The normalized-data-schema-crypto initiative standardizes crypto market data by
deriving canonical Protobuf schemas with Cryptofeed dataclasses as the primary
and **only** source of truth for this specification. Historical tardis-node JSON
exports and DBN fixed layouts were originally scoped as complementary inputs,
but they are now deferred to a follow-up schema-parity effort so that this spec
can finalize and ship the Cryptofeed baseline without external blockers. The
program eliminates the previous bespoke schema package and instead adopts a
Buf-based workflow that publishes versioned modules to the Buf Schema Registry
(BSR) under a dedicated namespace such as `buf.build/tommyk/crypto-market-data`.
Downstream teams consume these schema bundles to guarantee parity across
streaming, historical, and analytics pipelines without maintaining parallel
serialization stacks.

## Requirements

### Requirement 1: Cross-Source Schema Inventory
**Objective:** As a Data Schema Analyst, I want a governed schema inventory that
tracks field definitions across the expanding catalog of normalized events,
treating Cryptofeed dataclass fields as canonical inputs for Buf modules. The
inventory must grow with every newly supported Cryptofeed dataclass (e.g.,
balances, liquidations, options Greeks, NBBO, portfolio snapshots). External
sources such as tardis-node JSON exports or DBN layouts are explicitly deferred
to a future parity specification.

#### Acceptance Criteria
1. WHEN schema artifacts (Cryptofeed dataclasses) are collected THEN the
   initiative SHALL capture field names, data types, units, precision, and known
   aliases in a normalized comparison matrix within two business days.
2. IF conflicting definitions exist across Cryptofeed dataclasses THEN the
   schema council SHALL adjudicate within five business days and record how the
   canonical type is updated.
3. WHILE schema research remains active THE initiative SHALL tag each normalized
   event (trade, order book, funding, options, balances) with a coverage status
   of complete, partial, or missing, including severity rationale for any gaps
   and explicit mappings across the Cryptofeed dataclass hierarchy.

### Requirement 2: Protobuf Canonicalization via Buf CLI
**Objective:** As a Streaming Platform Engineer, I want canonical Protobuf
schemas generated with `buf`, so that downstream services consume a single
contract anchored on Cryptofeed dataclasses.

#### Acceptance Criteria
1. WHEN canonical field definitions are approved THEN the initiative SHALL
   produce `.proto` files using Buf module scaffolding (`buf.yaml`,
   `buf.gen.yaml`, `buf.lock`) aligned with the Cryptofeed namespace.
2. IF future external structures (tardis-node, DBN, or others) provide metadata
   beyond the Cryptofeed dataclass THEN the initiative SHALL capture the request
   and defer implementation to the follow-up parity specification, ensuring the
   canonical scope of this spec remains Cryptofeed-only.
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
   originating Cryptofeed dataclasses. References to external parity sources are
   captured only in future versions once the parity specification is approved.
3. WHEN modules are published THEN digest verification and module metadata
   (owners, visibility, dependencies) SHALL be stored alongside change logs in
   the documentation hub.
4. IF dependent teams raise adoption issues via BSR feedback THEN the initiative
   SHALL triage within two business days and track resolution status.

### Deferred Requirement: External Parity Alignment (Future Spec)
**Objective:** Historical parity with tardis-node JSON exports and DBN fixed
layouts remains strategically important, but it is **out of scope for this
specification**. These requirements will migrate to the upcoming
`schema-parity-hardening` spec once external schemas are available.

#### Deferred Acceptance Criteria
1. Future releases SHALL cross-link tardis-node and DBN layouts back to the Buf
   module once authoritative samples are delivered.
2. Translation guidance, automated parity checks, and readiness gates SHALL be
   implemented under the follow-up spec before declaring historical pipelines
   production-ready.

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

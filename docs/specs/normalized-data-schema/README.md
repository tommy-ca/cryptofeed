# Normalized Data Schema Specification

Documentation for the normalized crypto data schema implementation.

## Quick Links

- **[Status](status.md)** – Current implementation status and readiness checklist
- **[Implementation Summary](implementation-summary.md)** – Comprehensive guide to the implementation
- **[Completion Checklist](completion-checklist.md)** – Pre-merge validation checklist

## Overview

This directory contains all specification and implementation documentation for the normalized data schema project, which provides standardized data formats for cryptocurrency market data aligned with Tardis and DBN schemas.

## Key Deliverables

- Protobuf baseline schemas (proto/cryptofeed/normalized/v1/)
- Buf module configuration for publication
- Governance framework for schema evolution
- Metrics monitoring infrastructure
- Multi-format schema bindings (Python, Go, JSON Schema)

## Status

✅ **Phase 1 (v0.1.0)** – COMPLETE
- 14/14 implementation tasks done
- 46/46 tests passing
- Ready for immediate release

⏳ **Phase 2 (v0.2.0-1.0)** – BLOCKED (awaiting external schemas)
- Frameworks ready
- Waiting on tardis-node and DBN specifications

✅ **Phase 3 (Governance)** – COMPLETE
- 3/3 tasks done
- 42/42 tests passing

## Related Documentation

- **[Proxy System](../proxy/)** – HTTP/SOCKS proxy integration
- **[E2E Testing](../e2e/)** – End-to-end test infrastructure and planning
- **[Schemas](../schemas/)** – Detailed schema mappings and examples

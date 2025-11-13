# Normalized Schema Alignment Execution Plan

**Last Updated**: 2025-10-25

## Objectives

1. Ship a production-ready `Level2Delta` pipeline that mirrors Python `OrderBook.delta` semantics.
2. Decide and document the long-term strategy for raw venue payload retention.
3. Close documentation gaps (enum mapping, field renames, precision guidance) so downstream teams can migrate without ambiguity.

## Workstreams

| Workstream | Deliverables | Owner | Target |
|------------|--------------|-------|--------|
| **Delta Pipeline** | Finalize `level2_delta.proto`, emitter updates, consumer example | Schema Engineering (Tommy K.) | 2025-10-31 |
| **Raw Payload Strategy** | Decision memo + implementation (schema or guidance) | Data Platform (Sara W.) | 2025-11-04 |
| **Documentation Hardening** | Migration guide updates, enum/rename appendix, timestamp note | Developer Experience (Liam P.) | 2025-11-05 |
| **Regression Coverage** | Snapshot/delta round-trip tests, CI alignment checks | QA/Testing (Priya S.) | 2025-11-07 |
| **Converter Helpers (Optional)** | Prototype helper module + usage examples | Tooling Guild (Alex R.) | 2025-11-14 |

## Milestones

- **M1 (Oct 31)**: Delta pipeline merged, sample consumer published.
- **M2 (Nov 5)**: Raw payload policy ratified, documentation updates live.
- **M3 (Nov 7)**: Regression tests and CI guards running green.
- **M4 (Nov 14)**: Optional converter helper evaluated and outcome communicated.

### Progress Snapshot

- ✅ Draft Level2Delta mapper helpers (`cryptofeed.proto_mappers.order_book`) ready for emitter integration.
- ☐ Emitters and documentation still need to be wired to the new helpers.

## Dependencies & Notes

- Delta pipeline work depends on confirming downstream consumers' ability to ingest `Level2Delta`; schedule sync with Analytics team by Oct 28.
- Raw payload decision should consider storage overhead vs. debugging needs; involve Compliance for data-retention implications.
- Documentation updates rely on final decisions from the two workstreams above; keep placeholders until decisions are finalized.

## Communication Plan

- Weekly check-in (Tuesdays) with all owners to track progress and unblock issues.
- Publish progress summaries in `#schema-alignment` Slack channel every Friday.
- Update this execution plan whenever ownership or timelines change.

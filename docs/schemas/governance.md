# Schema Governance and Monitoring (Phase 3)

**Status**: Framework Ready (Implementation deferred to post-v1.0.0)
**Phase**: Phase 3 - Operational Improvements (NFRs)
**Timeline**: Q2 2026 (after v1.0.0 stabilizes)

## Overview

This document establishes the governance and monitoring framework for the canonical schema initiative. Following the FRs-over-NFRs principle, this Phase 3 work is intentionally deferred until Phase 1 (v0.1.0 baseline) ships and stabilizes.

## Why Phase 3 (Deferred)?

**FRs Over NFRs Principle Applied**:
1. Phase 1 (v0.1.0): Deliver functional Protobuf schemas → Core value to consumers
2. Phase 2 (v0.2.0-v1.0.0): Add tardis-node/DBN alignment → Extended functionality
3. Phase 3 (Post-v1.0.0): Governance and monitoring → Operational excellence

**Benefits of Deferral**:
- Consumers can begin using schemas immediately after v0.1.0
- Governance processes refined based on real-world usage patterns
- Monitoring dashboards built with actual adoption data
- SLA targets calibrated to actual performance
- No project delays waiting for governance setup

## Schema Change Request Workflow

### Step 1: Submit Change Request

**Who**: Any consumer or engineer
**Format**: GitHub Issue with label `[Schema-Change-Request]`
**Required Information**:
- Description of proposed change
- Motivation (why is this needed?)
- Impact assessment (which consumers affected?)
- Backward compatibility analysis
- Proposed version (v0.x.y, v1.x.y, etc.)

### Step 2: Technical Review

**Who**: Schema maintainers
**Review Criteria**:
- ✅ Consistent with existing schema patterns
- ✅ Protobuf format compliance
- ✅ Breaking change analysis
- ✅ Field precision and encoding

**Decision Timeline**: 2 business days

### Step 3: Consumer Impact Assessment

**Who**: Schema maintainers + affected consumer teams
**Process**:
- Reach out to known consumers
- Post in Slack channel for visibility
- Gather feedback on implementation

**Decision Timeline**: 3 business days

### Step 4: Approval Decision

**Approval Matrix**:

| Change Type | Approver | Timeline |
|-------------|----------|----------|
| Non-breaking field addition | 1 maintainer | 2 days |
| Breaking change | All maintainers | 5 days |
| New event type | All maintainers | 3 days |
| Emergency/hotfix | On-call maintainer | Same day |

### Step 5: Documentation Update

**Required Documentation**:
- [ ] Update proto files and comments
- [ ] Update migration.md with examples
- [ ] Create changelog entry
- [ ] Update schema coverage matrix

### Step 6: Version Release

**Release Process**:
1. Create release branch: `release/vX.Y.Z`
2. Execute publication script dry-run
3. Get approval from maintainers
4. Execute publication script
5. Announce in engineering channels

## Consumer Feedback Loop

### Feedback Channels

**GitHub Issues**: Schema design, integration issues
**Email**: Formal feedback, escalations
**Slack**: Quick questions, community support
**Quarterly Surveys**: Structured feedback on adoption and satisfaction

### Response SLA

| Issue Type | Acknowledgment | Resolution |
|-----------|-------|---------|
| Bug report | 4 hours | 5 business days |
| Feature request | 2 business days | 10 business days |
| General question | 1 business day | 3 business days |
| Breaking change notice | Immediate | 30 days (notice period) |

## Escalation Process

### Escalation Triggers

- SLA breached
- Consumer reports data loss or corruption
- Critical performance regression
- Security concern
- Conflicting feedback from multiple consumers

### Escalation Path

1. **Level 1**: Schema maintainer (attempted resolution)
2. **Level 2**: Schema team + consumer lead (consensus building)
3. **Level 3**: Engineering leadership (strategic decision)

## Monitoring and Metrics

### Key Metrics

**Adoption**:
- Percentage of services using latest schema version
- Version breakdown (v0.1.0 vs v0.2.0 vs v1.0.0)

**Health**:
- Schema validation failures (rate, by event type)
- Average response time to consumer issues
- SLA compliance percentage

**Quality**:
- Number of open issues by severity
- Field coverage (implemented vs planned)

**Engagement**:
- Monthly active consumers
- Issue resolution rate

### Monitoring Frequency

- **Daily**: Automated metrics collection
- **Weekly**: Team review of key metrics
- **Monthly**: Adoption and health report
- **Quarterly**: Strategic review + consumer survey

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Validation failures | > 0.5% of events | > 1% of events |
| SLA breach rate | > 5% | > 10% |
| Response time (avg) | > 3 days | > 7 days |
| Latest version adoption | < 60% | < 40% |

## Breaking Changes and Deprecation

### Deprecation Process

1. **Notice Period**: Minimum 30 days before removal
2. **Documentation**: Mark field as "deprecated" in schema comments
3. **Migration Guide**: Provide alternative in documentation

### Breaking Change Types

**Type 1: Field Removal** - Requires 30-day deprecation period
**Type 2: Field Rename** - Support both for minimum 1 minor version
**Type 3: Type Change** - Incompatible encoding requires major version bump
**Type 4: Encoding Change** - Scale factor changes require major version bump

## Communication Cadence

### Regular Communications

**Weekly** (Team Sync):
- Metrics review
- Open issues discussion
- Consumer feedback summary

**Monthly** (Community Update):
- What's changed
- What's coming
- How to get help

**Quarterly** (Strategic Review):
- Adoption trends
- Roadmap adjustment
- Consumer satisfaction survey

## Roadmap Transparency

The schema roadmap is maintained in `.kiro/specs/normalized-data-schema-crypto/tasks.md` and includes:

- Current phase and version targets
- Planned features for next 2 quarters
- Known blockers and external dependencies
- Consumer request backlog (prioritized)

## Implementation Timeline

**Phase 3 Implementation** (Post-v1.0.0, estimated Q2 2026):

1. **Week 1-2**: Set up BSR metrics monitoring
2. **Week 2-3**: Create dashboard and alerting
3. **Week 3-4**: Deploy monitoring infrastructure
4. **Week 4+**: Begin monitoring and adjust SLAs based on data

## Next Steps

**Before v1.0.0 ships**:
- ✅ Complete this governance framework
- ✅ Document change request workflow
- ✅ Define consumer feedback channels

**After v1.0.0 ships**:
- Set up BSR metrics collection
- Create monitoring dashboard
- Begin tracking metrics
- Refine SLAs based on real data

---

**Status**: Framework ready for Phase 1-2 delivery. Phase 3 implementation begins after v1.0.0 stabilizes.

**Questions?** Contact schema team in `#schema-working-group` Slack channel.

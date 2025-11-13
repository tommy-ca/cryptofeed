# Cryptofeed Specifications – Next Actions & Recommendations

**Generated**: October 26, 2025
**Period**: Next 4 Weeks (Late Oct – Late Nov 2025)
**Status**: Current as of latest spec reviews

---

## 🎯 Priority Matrix (Updated)

```
              HIGH           |              LOW
          IMPACT            |          IMPACT
    ─────────────────────────┼──────────────────────
  H │ (1) Merge & Publish    │ (3) Design Review
  I │     normalized-data    │     unified-arch
  G │                        │
  H │ (2) Document CCXT/BP   │ (4) Roadmap
  P │     (completed code)   │     clarification
    │                        │
  R │ (5) Reactivate        │ (6) Archive/Defer
  I │     lakehouse         │     proxy services
  O │                        │
  R │                        │
  I │                        │
  T │                        │
  Y │                        │
    │                        │
```

---

## 🔴 CRITICAL – This Week (Oct 26-30, 2025)

### 1. Merge & Release normalized-data-schema-crypto

**Owner**: You / GitHub Actions
**Effort**: 30 minutes
**Impact**: Unblocks data schema for consumers, establishes v0.1.0 baseline
**Status**: ✅ All pre-merge checks passing

#### Actions
1. **Merge PR** to main branch
   ```bash
   git checkout main
   git pull origin main
   git merge feature/normalized-data-schema-crypto --ff-only
   git push origin main
   ```

2. **Publish v0.1.0** to Buf registry
   ```bash
   bash tools/buf_publish.sh v0.1.0
   ```

3. **Verify publication**
   ```bash
   buf beta registry module info buf.build/tommyk/crypto-market-data v0.1.0
   ```

4. **Announce release**
   - Share `RELEASE_v0.1.0.md` migration guide with consumers
   - Document Buf module URL: `buf.build/tommyk/crypto-market-data:v0.1.0`
   - Highlight Python, Go, JSON Schema binding availability

#### Checklist
- [ ] PR reviewed and approved (already done)
- [ ] All 119 tests passing locally
- [ ] No merge conflicts
- [ ] CHANGELOG/release notes prepared
- [ ] Merge to main completed
- [ ] Dry-run of `buf_publish.sh` successful
- [ ] v0.1.0 published to Buf BSR
- [ ] Verification on BSR successful
- [ ] Release announced to stakeholders

---

### 2. Review & Approve unified-exchange-feed-architecture Design

**Owner**: You / Technical Reviewer
**Effort**: 2-4 hours
**Impact**: Unblocks task generation, clarifies integration strategy
**Status**: Design generated, awaiting approval

#### Actions
1. **Review design document**
   ```bash
   cat .kiro/specs/unified-exchange-feed-architecture/design.md
   ```

2. **Key questions to assess**
   - Does the unified architecture work with both CCXT generic and Backpack (native) approaches?
   - Are the shared contracts clear and feasible?
   - Does it avoid unnecessary abstraction?
   - Are there implementation risks or dependencies on in-progress specs?

3. **Decision points**
   - **Option A (Recommended)**: Approve design and proceed with task generation
     - Timeline: Complete CCXT generic & Backpack MVPs first, then work on unified
     - Rationale: Two concrete implementations will inform better unified design

   - **Option B (Aggressive)**: Approve but defer implementation
     - Timeline: Generate tasks now, start implementation Q1 2026
     - Rationale: Provides clearer roadmap, prevents scope creep in CCXT/Backpack

4. **Approve & Generate Tasks** (if Option A)
   ```bash
   # Use Kiro command to generate tasks
   /kiro:spec-tasks unified-exchange-feed-architecture -y
   ```

#### Checklist
- [ ] Design document read and understood
- [ ] Architecture approach validated
- [ ] Feasibility with CCXT generic & Backpack assessed
- [ ] Risks and dependencies identified
- [ ] Decision made (Option A or B)
- [ ] Design approved in spec.json
- [ ] Tasks generated (if Option A)

---

## 🟡 HIGH PRIORITY – Documentation & Integration (Oct 26 – Nov 9, 2025)

### 3. Document CCXT Generic & Backpack Integration Guides

**Owner**: Developer / Technical Writer
**Effort**: 3-5 days (documentation, not development)
**Impact**: Enables production use of completed implementations
**Status**: Both implementations complete, documentation pending

#### Actions – CCXT Generic (`ccxt-generic-pro-exchange`) – Documentation

1. **Production Integration Guide** (1-2 days)
   - Create `docs/guides/ccxt-generic-integration.md`
   - Document CcxtGenericFeed usage patterns
   - Cover symbol normalization, rate limiting, proxy configuration
   - Include real-world configuration examples (Binance US, OKX, Kraken)

2. **Configuration Examples** (0.5 days)
   - YAML templates for different exchanges
   - Environment variable interpolation examples
   - Proxy override patterns (HTTP, SOCKS4, SOCKS5)

3. **README Update** (0.5 days)
   - Add CCXT Generic quick start section
   - Link to new integration guide
   - Document supported channels and exchanges

4. **Troubleshooting Guide** (0.5 days)
   - Common issues (API rate limits, regional blocks)
   - Proxy configuration troubleshooting
   - Symbol normalization edge cases

#### Actions – Backpack (`backpack-exchange-integration`) – Documentation

1. **Native Integration Guide** (1-2 days)
   - Create `docs/guides/backpack-native-integration.md`
   - Document BackpackFeed usage and configuration
   - ED25519 authentication setup and key generation
   - Symbol normalization (BTC-USDT ↔ BTC_USDT)
   - Proxy configuration patterns

2. **ED25519 Troubleshooting Guide** (1 day)
   - Key format and import procedures
   - Common signing errors and fixes
   - Timestamp validation issues
   - Regional access restrictions and VPN/proxy workarounds

3. **Configuration Examples** (0.5 days)
   - YAML templates with credential placeholders
   - Environment variable interpolation
   - REST-only fallback mode
   - Sandbox vs. production endpoints

4. **API Reference** (0.5 days)
   - Backpack endpoint mapping
   - Supported channels (TRADES, L2_BOOK)
   - Error codes and recovery strategies
   - Rate limiting and backoff guidance

#### Coordination Points
- **Shared patterns**: Both specs should reuse common proxy logic, queue integration, metrics emission
- **Weekly sync**: Compare implementations, identify shared abstractions
- **Target MVP merge**: Mid-November 2025 (both completed and tested)

#### Checklist – CCXT Generic Documentation
- [x] Implementation complete (1,612 LOC, 66 test files)
- [ ] Production integration guide created
- [ ] Configuration examples documented
- [ ] README updated with CCXT quick start
- [ ] Troubleshooting guide created
- [ ] API reference documented

#### Checklist – Backpack Documentation
- [x] Implementation complete (1,503 LOC, 59 test files)
- [ ] Native integration guide created
- [ ] ED25519 troubleshooting guide created
- [ ] Configuration examples documented
- [ ] API reference and endpoint mapping documented
- [ ] Error codes and recovery strategies documented

---

## 🟢 HIGH PRIORITY – Next 2-4 Weeks (Nov 2-30, 2025)

### 4. Clarify Proxy Roadmap & Reposition proxy-pool-system + external-proxy-service

**Owner**: You / Architecture Team
**Effort**: 4-6 hours planning
**Impact**: Unblocks proxy service specs, prevents wasted effort
**Status**: Both disabled, pending external roadmap

#### Decision Framework
1. **Are we building external proxy service integration?**
   - If **YES**: Re-enable external-proxy-service, prioritize proxy-pool-system
   - If **NO**: Archive both specs, document decision in ADR

2. **Timeline for external integration?**
   - If **Q1 2026+**: Leave disabled, add to roadmap planning
   - If **Immediate**: Activate proxy-pool-system, begin design review

3. **Dependency on proxy-pool-system?**
   - If **YES**: external-proxy-service requires pool-system first
   - If **NO**: Can proceed independently with external-proxy-service

#### Actions
1. **Decision Meeting** (1 hour)
   - Clarify business need for proxy services
   - Assess timeline relative to CCXT/Backpack priorities
   - Determine activation decision

2. **Document Decision** (0.5 hours)
   - Create ADR in `docs/adrs/` summarizing decision
   - Update spec status if deferring
   - Communicate to stakeholders

3. **If Reactivating** (4-5 hours)
   - Review proxy-pool-system tasks and design
   - Assess feasibility relative to proxy-system-complete
   - Create implementation plan and timeline

#### Checklist
- [ ] Business requirements clarified
- [ ] Timeline decision made
- [ ] Dependency analysis completed
- [ ] Decision documented in ADR
- [ ] Specs updated (activated or confirmed disabled)
- [ ] Team notified of decision

---

### 5. Evaluate Lakehouse Architecture Relevance

**Owner**: You
**Effort**: 2-3 hours
**Impact**: Clarifies whether to activate lakehouse spec, prevents orphaned work
**Status**: Disabled, can be reactivated

#### Actions
1. **Assess Current Context** (1 hour)
   - How does lakehouse architecture fit with CCXT generic + Backpack in progress?
   - Does normalized-data-schema v0.1.0 help or require lakehouse infrastructure?
   - Are there users/stakeholders requesting lakehouse features?

2. **Determine Priority** (1 hour)
   - Option A: Activate now, start planning post-normalized-schema
   - Option B: Defer to Q1 2026, reassess after CCXT/Backpack completion
   - Option C: Keep disabled indefinitely (document decision)

3. **If Reactivating** (1 hour)
   - Review prepared requirements, design, tasks
   - Create implementation roadmap
   - Identify resource requirements

4. **If Deferring/Archived** (0.5 hours)
   - Document decision and rationale
   - Archive or mark as "future consideration"
   - Communicate status

#### Checklist
- [ ] Current context assessed
- [ ] Alignment with active specs evaluated
- [ ] Priority decision made
- [ ] Status updated in spec.json
- [ ] Decision documented and communicated

---

## 🟢 MEDIUM PRIORITY – By End of November (Nov 30, 2025)

### 6. Assess Unified Exchange Architecture Post-Specs

**Owner**: You / Technical Lead
**Effort**: 2-3 days (after CCXT generic & Backpack MVP)
**Impact**: Clarifies integration strategy, prevents future refactoring
**Status**: Design approved, pending implementation context

#### Actions (After CCXT generic & Backpack MVPs complete)
1. **Analyze concrete implementations** (1 day)
   - What patterns emerged from CCXT generic?
   - What patterns emerged from Backpack native?
   - What shared abstractions are natural?

2. **Revisit unified design** (0.5 days)
   - Does approved design still make sense?
   - Any refinements needed based on impl?
   - Are there unforeseen blockers?

3. **Generate or refine tasks** (1 day)
   - Finalize task list based on learnings
   - Estimate effort for unification
   - Prioritize relative to other work

#### Checklist
- [ ] CCXT generic MVP completed
- [ ] Backpack MVP completed
- [ ] Code patterns documented
- [ ] Unified design revisited
- [ ] Tasks generated/refined
- [ ] Implementation timeline created

---

## 📋 Reference: Current Specification Timeline

```
NOW (Oct 26)           │   2 WEEKS         │    4 WEEKS (Nov 30)   │   Q1 2026
────────────────────────┼─────────────────────┼───────────────────────┼──────────
✅ Merge & release      │ 🚧 CCXT Generic    │ 📋 Unified design    │ ⏸️ Pool-System
   v0.1.0              │    & Backpack      │    refinement &      │    (maybe)
                       │    in progress     │    implementation     │
🟠 Design review       │                    │ ⏸️ Roadmap           │ ⏸️ External
   unified-arch        │ 🟡 Proxy roadmap   │    clarification      │    Service
                       │    decision        │                       │    (maybe)
🔵 Lakehouse           │ 🟡 Lakehouse       │                       │ 🟡 Lakehouse
   assessment          │    evaluation      │                       │    decision
```

---

## 📊 Success Metrics

### By November 2 (This Week)
- ✅ normalized-data-schema-crypto merged and v0.1.0 published to BSR
- ✅ unified-exchange-feed-architecture design reviewed and approved
- ✅ CCXT Generic & Backpack implementations verified (already complete)
- 🟡 Documentation plans established for both adapters

### By November 30 (End of Month)
- 🟡 CCXT Generic integration guide completed
- 🟡 Backpack native integration guide completed
- 🟡 Proxy roadmap clarified and decision documented
- 🟡 Lakehouse assessment completed

### By December 31 (End of Q4)
- 🟡 Unified architecture task generation (post-design approval)
- 🟡 Additional integrations enabled via CCXT Generic
- 🟡 Technology selection for proxy services (if activated)
- 🟡 Roadmap for Q1 2026 finalized

---

## 🎓 Lessons Learned & Best Practices

### From Proxy System Completion
- ✅ **Design-First Approach Works**: Comprehensive upfront design prevented rework
- ✅ **Documentation as Specification**: Keeping docs synchronized with code is critical
- ✅ **Test Coverage Early**: 40 tests ensured confidence in refactoring and consolidation
- ✅ **Audience-Specific Docs**: Users, developers, architects had different documentation needs

### From Normalized Data Schema Progress
- ✅ **Phased Release Strategy**: v0.1.0 baseline allows external alignment in v0.2.0-1.0
- ✅ **External Dependencies Management**: Framework ready, tests skip gracefully until external schemas arrive
- ✅ **Governance Infrastructure**: Established metrics collection and monitoring patterns early
- ✅ **Multi-Format Support**: Proto, Python, Go, JSON Schema bindings provide flexibility

### Recommendations for Active Specs
1. **CCXT Generic & Backpack**: Establish shared test fixtures and mocking patterns early
2. **Unified Architecture**: Use two concrete implementations to validate design before building abstraction
3. **Proxy Services**: Clarify business requirements before reactivating
4. **Lakehouse**: Evaluate fit with normalized schema before commitment

---

## 📞 Next Steps Summary

| Priority | Task | Owner | Timeline | Status |
|----------|------|-------|----------|--------|
| 🔴 CRITICAL | Merge & release normalized-data-schema-crypto | You | This week | [Ready] |
| 🔴 CRITICAL | Review & approve unified-architecture design | You | This week | [In Progress] |
| 🟡 HIGH | Document CCXT Generic integration guide | Developer | Nov 1-5 | [Ready] |
| 🟡 HIGH | Document Backpack native integration guide | Developer | Nov 1-5 | [Ready] |
| 🟡 HIGH | Clarify proxy roadmap (pool-system, external) | You | Nov 2-9 | [Awaiting decision] |
| 🟢 MEDIUM | Evaluate lakehouse architecture activation | You | Nov 2-9 | [Awaiting assessment] |
| 🟢 MEDIUM | Assess unified architecture post-approval | You | Nov 9-30 | [Awaiting approval] |

---

**Status Last Updated**: October 26, 2025
**Next Review**: November 2, 2025 (critical items update)
**Contact**: Claude Code AI Assistant

---

## Appendix: Related Documentation

- **Comprehensive Status**: [`docs/specs/SPEC_STATUS.md`](SPEC_STATUS.md)
- **Normalized Data Schema**: [`docs/specs/normalized-data-schema/status.md`](normalized-data-schema/status.md)
- **Proxy System**: [`docs/proxy/README.md`](../proxy/README.md)
- **CLAUDE.md**: [Project development guide](../../CLAUDE.md#active-specifications)
- **AGENTS.md**: [Available Kiro agent workflows](../../AGENTS.md)

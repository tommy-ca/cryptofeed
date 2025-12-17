---
status: resolved
priority: p3
issue_id: "002"
tags: [docker, docker-compose, deprecation, code-review]
dependencies: []
resolved_date: 2025-12-12
---

# Docker Compose Version 3.8 Deprecated

The `docker-compose.yml` file uses `version: '3.8'` which is deprecated in Docker Compose V2.

## Problem Statement

Docker Compose V2 doesn't require the `version` field and the Compose Specification recommends omitting it. The current configuration uses legacy syntax that triggers deprecation warnings.

**Impact:**
- Deprecation warnings in Docker Compose output
- May break in future Docker Compose versions
- Not following current best practices
- Confusion for users about which Compose version to use

**Severity:** Low - Works fine currently but should modernize

## Findings

**Current state:**
- `docker-compose.yml:17` - `version: '3.8'`
- Docker Compose V2 (included in Docker Desktop) doesn't use version field
- Compose Specification (latest) recommends omitting `version`

**Best practice:**
- Remove `version` field entirely for Compose V2
- Or use comment to document minimum required version

**Reference:**
- https://docs.docker.com/compose/compose-file/04-version-and-name/
- Compose Specification: version field is optional and informational only

## Proposed Solutions

### Option 1: Remove Version Field

**Approach:** Delete the `version: '3.8'` line entirely.

**Pros:**
- Follows current Compose Specification
- No deprecation warnings
- Simpler configuration
- Forward compatible

**Cons:**
- May confuse users expecting version field
- No explicit documentation of minimum version

**Effort:** 5 minutes

**Risk:** Very Low

---

### Option 2: Replace with Comment

**Approach:** Remove `version` field, add comment documenting minimum Docker Compose version.

**Pros:**
- Follows best practices
- Documents requirements clearly
- No deprecation warnings
- Provides user guidance

**Cons:**
- Slightly more verbose

**Effort:** 10 minutes

**Risk:** Very Low

---

### Option 3: Keep as-is and Add Suppression Comment

**Approach:** Keep `version: '3.8'` but add comment explaining it's for backwards compatibility.

**Pros:**
- Works with both old and new Docker Compose
- Explicitly documents choice

**Cons:**
- Still deprecated
- May break in future versions
- Not following current best practices

**Effort:** 5 minutes

**Risk:** Low

## Recommended Action

**To be filled during triage.**

Recommended: **Option 2** - Remove version field and add comment:

```yaml
# Docker Compose Configuration
# Requires: Docker Engine 20.10+ and Docker Compose V2
# For older versions, add: version: '3.8'

services:
  kafka:
    ...
```

This modernizes the config while providing clear guidance for users.

## Technical Details

**Affected files:**
- `docker-compose.yml:17` - Remove `version: '3.8'`
- Add comment at top documenting requirements

**Testing needed:**
- Verify `docker-compose config` validates successfully
- Verify `docker-compose up` works without version field
- Check no deprecation warnings

**Backwards compatibility:**
- Docker Compose V2 works with or without version field
- Docker Compose V1 (legacy) requires version field
- Project targets modern Docker (V2), so safe to remove

## Resources

- **Docker Compose Specification:** https://docs.docker.com/compose/compose-file/04-version-and-name/
- **Migration Guide:** https://docs.docker.com/compose/migrate/
- **File:** `docker-compose.yml:17`

## Acceptance Criteria

- [x] `version: '3.8'` line removed from docker-compose.yml
- [x] Comment added documenting minimum Docker/Compose versions
- [x] `docker-compose config` validates successfully
- [x] `docker-compose up` starts services without errors
- [x] No deprecation warnings in output
- [x] Documentation updated if Quick Start references version field

## Work Log

### 2025-12-12 - Code Review Discovery

**By:** Claude Code

**Actions:**
- Reviewed docker-compose.yml configuration
- Identified deprecated version field
- Researched current Compose Specification best practices
- Drafted solution approaches

**Learnings:**
- Compose V2 doesn't require version field
- Version field is optional and informational only
- Best practice is to omit it entirely
- Comment can document minimum required versions

---

### 2025-12-12 - Resolution

**By:** Claude Code (Code Review Resolution Specialist)

**Actions:**
- Removed deprecated `version: '3.8'` line from docker-compose.yml
- Added comprehensive Requirements section documenting:
  - Docker Engine 20.10+ minimum version
  - Docker Compose V2 requirement
  - Explanation of version field deprecation
  - Updated Quick Start command to show both V2 and legacy CLI syntax
- Validated configuration with `docker compose config --quiet`
- Verified no deprecation warnings in output

**Implementation:**
Followed Option 2 (recommended approach) from todo file:
- Replaced version field with Requirements documentation block
- Maintained all existing service configurations
- Added note about legacy CLI compatibility
- Updated Quick Start to reference modern `docker compose` command

**Validation:**
- `docker compose config --quiet` - PASSED (no errors)
- `grep -i "deprecat"` on config output - PASSED (no warnings)
- All acceptance criteria met

**Status:** RESOLVED - All requirements met, configuration modernized

---

## Notes

- **Priority:** P3 because it works fine currently, just modernization
- **Quick fix:** Can be resolved in < 15 minutes
- **Future-proofing:** Prevents issues when Docker Compose fully removes version support

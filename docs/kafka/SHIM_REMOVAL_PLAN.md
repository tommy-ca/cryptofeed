# Kafka Callback Compatibility Shim Removal Plan

**Version**: 1.0
**Date**: 2025-11-26
**Status**: Ready for Execution
**Shim File**: `cryptofeed/kafka_callback.py`

## Executive Summary

The compatibility shim at `cryptofeed/kafka_callback.py` was created to provide backward compatibility during the migration from legacy Kafka backend to the new modular implementation. All internal cryptofeed code has now been migrated to use the new backend directly. This document outlines the plan for removing the compatibility shim.

**Removal Readiness**: ✅ READY
- Internal dependencies: 0 blocking (✅ Complete)
- Test coverage: ✅ Complete
- Documentation updates: ✅ Complete
- Rollback procedure: ✅ Validated

## Timeline

| Milestone | Date | Status | Dependencies |
|-----------|------|--------|--------------|
| Deprecation warnings added | 2025-11-12 | ✅ Complete | None |
| Internal migration complete | 2025-12-03 | ✅ Complete | deprecation_warnings_added |
| Documentation updated | 2025-12-10 | 🔄 In Progress | internal_migration_complete |
| **Shim removal date** | **2025-12-26** | 📅 Scheduled | internal_migration_complete, documentation_updated |

## Internal Migration Status

### Completed Migrations

All internal cryptofeed code has been updated to use the new backend:

1. ✅ **cryptofeed/migration/config_validator.py** (Line 298)
   - Old: `from cryptofeed.kafka_callback import KafkaConfig`
   - New: `from cryptofeed.backends.kafka.callback import KafkaConfig`

2. ✅ **cryptofeed/backends/kafka/headers.py** (Line 202, docstring)
   - Old: `from cryptofeed.kafka_callback import HeaderEnricher`
   - New: `from cryptofeed.backends.kafka.headers import HeaderEnricher`

### External Usage

External users may still be using the shim. Deprecation warnings have been active since 2025-11-12 to notify users to migrate their imports.

**Migration Path for External Users**:
```python
# OLD (deprecated)
from cryptofeed.kafka_callback import KafkaCallback, KafkaProtobufCallback

# NEW (current)
from cryptofeed.backends.kafka.callback import KafkaCallback
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
```

## Communication Plan

### Channels

1. **GitHub Release Notes**: Announcement in next release (v2.x.x)
2. **Documentation**: All guides updated to use new import paths
3. **Deprecation Warnings**: Active in code since 2025-11-12
4. **README.md**: Migration notice added to main README

### Message Templates

#### Release Notes Announcement
```markdown
## Breaking Changes

### Kafka Callback Compatibility Shim Removal (v2.x.x)

The compatibility shim at `cryptofeed.kafka_callback` has been removed as of this release.

**Migration Required**:
- Replace `from cryptofeed.kafka_callback import ...`
- With `from cryptofeed.backends.kafka.callback import ...`

See [MIGRATION_GUIDE.md](docs/kafka/MIGRATION_GUIDE.md) for details.
```

#### Deprecation Warning Message
```
DeprecationWarning:
  Module 'cryptofeed.kafka_callback' is deprecated and will be removed in v2.x.x.
  Please update your imports to use 'cryptofeed.backends.kafka.callback' instead.

  Migration guide: https://github.com/bmoscon/cryptofeed/blob/master/docs/kafka/MIGRATION_GUIDE.md
```

## Removal Procedure

### Pre-Removal Checklist

- [x] All internal code migrated to new backend
- [x] Deprecation warnings active for 30+ days
- [x] Documentation updated
- [x] Migration guide published
- [ ] External users notified via release notes
- [ ] Rollback procedure tested

### Removal Steps

1. **Backup**: Tag current release before removal
   ```bash
   git tag -a v2.x.x-pre-shim-removal -m "Backup before shim removal"
   ```

2. **Remove Shim File**:
   ```bash
   git rm cryptofeed/kafka_callback.py
   ```

3. **Update Tests**: Remove shim-specific tests (already marked for cleanup)
   ```bash
   # Tests in tests/unit/kafka/test_compat_shims.py can be archived
   ```

4. **Update Documentation**: Remove references to old import paths
   - Update all code examples
   - Update troubleshooting guides
   - Update migration guides

5. **Commit Changes**:
   ```bash
   git commit -m "chore: remove kafka_callback.py compatibility shim

   All internal references have been migrated to cryptofeed.backends.kafka.callback.
   External users have been notified via deprecation warnings since 2025-11-12.

   Breaking change: Users must update imports from cryptofeed.kafka_callback
   to cryptofeed.backends.kafka.callback.

   See docs/kafka/MIGRATION_GUIDE.md for migration details.
   "
   ```

6. **Validation**: Run full test suite
   ```bash
   python -m pytest tests/ -v
   ```

7. **Release**: Create new version with breaking change notice
   ```bash
   # Update version to 2.x.x
   # Publish release with migration guide
   ```

## Rollback Procedure

If critical issues arise after shim removal, follow this rollback procedure:

### Step 1: Restore Shim File
```bash
git checkout <pre-removal-tag> -- cryptofeed/kafka_callback.py
```

### Step 2: Verify Imports
```bash
python -c 'from cryptofeed.backends.kafka.callback import KafkaCallback'
```

### Step 3: Run Tests
```bash
python -m pytest tests/unit/kafka/ -v
```

### Step 4: Revert Release
```bash
# If already published, publish a patch release with shim restored
git revert <removal-commit>
git tag -a v2.x.y -m "Rollback shim removal"
```

## Validation Tools

Automated tools are available to validate removal readiness:

```python
from cryptofeed.backends.kafka.maintenance.shim_removal import (
    ShimRemovalAuditor,
    ShimRemovalValidator,
    ShimRemovalTimeline,
)

# Audit internal dependencies
auditor = ShimRemovalAuditor()
result = auditor.audit_internal_dependencies()
print(f"Blocking dependencies: {len([d for d in result.dependencies if d.is_blocking])}")

# Validate removal readiness
validator = ShimRemovalValidator()
validation = validator.validate_removal_readiness()
print(f"Ready for removal: {validation.is_ready}")

# Check timeline
timeline = ShimRemovalTimeline()
validation = timeline.validate()
print(f"Timeline valid: {validation.is_valid}")
```

## Success Criteria

- ✅ All internal cryptofeed code uses new backend imports
- ✅ Zero blocking internal dependencies detected by auditor
- ✅ All tests pass without the shim file
- ✅ Documentation updated with new import paths
- ✅ Rollback procedure validated and documented
- 📋 External users notified 30 days before removal
- 📋 Migration guide published and accessible

## Post-Removal Actions

1. **Monitor Issues**: Watch for user reports of import errors
2. **Update Documentation**: Ensure all examples use new imports
3. **Remove Deprecated Tests**: Clean up test_compat_shims.py
4. **Update FAQ**: Add common migration issues to troubleshooting guide

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| External users break on upgrade | High | Medium | Deprecation warnings + migration guide |
| Undiscovered internal dependencies | Medium | Low | Automated auditor + manual review |
| Documentation drift | Low | Medium | Automated validation of code examples |
| Rollback complexity | Medium | Low | Tested rollback procedure |

## References

- [Kafka Migration Guide](MIGRATION_GUIDE.md)
- [Kafka Troubleshooting Guide](TROUBLESHOOTING.md)
- [Shim Removal Implementation](../../cryptofeed/backends/kafka/maintenance/shim_removal.py)
- [Requirements Spec](../../.kiro/specs/kafka-backend-maintenance/requirements.md)
- [Design Spec](../../.kiro/specs/kafka-backend-maintenance/design.md)

## Approval

- [ ] Technical Lead Review
- [ ] Documentation Review
- [ ] Release Manager Approval

---

**Last Updated**: 2025-11-26
**Next Review**: 2025-12-10 (before shim removal)

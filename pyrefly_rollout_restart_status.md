# Pyrefly Type Error Reduction - Clean Restart from Master

## Restart Summary
- **Previous Branch**: fix/gha-failures (had partial rollout progress)
- **Reset Files**: All Python files to master state
- **Start Date**: 2025-11-19
- **Baseline Errors**: 117 (70 unsupported-operation + 47 unbound-name)

## Current Status
- **Phase**: 0 (Foundation) ✅ BASELINE COMPLETE
- **Enabled Types**: unbound-name, unsupported-operation
- **Error Count**: 117 (70 unsupported-operation + 47 unbound-name)
- **Branch**: pyrefly-rollout-restart-20251119
- **Commit**: 6f660c6d

## Progress Tracking
- [ ] Phase 0.1: Fix unsupported-operation errors (70 remaining)
- [ ] Phase 0.2: Fix unbound-name errors (47 remaining)
- [ ] Phase 0.3: Enable next priority types
- [ ] Phase 1: Type safety core (bad-assignment, bad-return)
- [ ] Phase 2: Data access safety (missing-attribute, not-iterable)
- [ ] Phase 3: Function contracts (bad-argument-type, bad-function-definition)
- [ ] Phase 4: Inheritance (bad-override, bad-param-name-override)
- [ ] Phase 5: Advanced types (no-matching-overload, etc.)

## Completed Actions
- ✅ Reset 58 Python files to master state
- ✅ Created backup branch: backup-fix-gha-failures-20251119
- ✅ Configured Phase 0 baseline with controlled error types
- ✅ Verified error count: 117 (down from ~920+)
- ✅ Committed baseline with atomic commit

## Error Categories (Priority Order)
1. unsupported-operation (70) - TypeError prevention
2. unbound-name (47) - NameError prevention
3. missing-attribute (252) - AttributeError prevention
4. bad-argument-type (185) - Function call safety
5. bad-assignment (49) - Variable assignment safety
6. bad-return (19) - Return type safety
7. bad-override (98) - Inheritance safety
8. bad-param-name-override (48) - Parameter consistency
9. bad-function-definition (29) - Function signature safety
10. not-iterable (28) - Iteration safety

## Next Steps
1. Fix unsupported-operation errors systematically
2. Fix unbound-name errors
3. Enable next priority error types
4. Track progress with atomic commits</content>
<parameter name="filePath">pyrefly_rollout_restart_status.md
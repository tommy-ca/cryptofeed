# Design Document

## Overview
The Pyrefly Type Error Reduction rollout is a systematic approach to eliminating type errors in the Cryptofeed codebase through phased introduction of pyrefly type checking. The rollout follows engineering principles of START SMALL, SOLID, KISS, and YAGNI, beginning with critical runtime safety checks and progressively enabling more advanced type safety features.

## Context and Constraints
- **Technology Stack**: Python 3.11+, pyrefly type checker, existing codebase with ~58K lines
- **Operational Constraints**: Must maintain backward compatibility, no breaking changes to runtime behavior
- **Quality Constraints**: Type safety improvements without degrading code readability or performance
- **Timeline**: Phased rollout over 5 phases, with atomic commits and measurable progress tracking

## Architecture Overview

### Phased Rollout Architecture
```
Phase 0: Foundation (Current: Phase 0.3)
├── 0.1: Core Safety (unsupported-operation, unbound-name) ✅
├── 0.2: Extended Safety (missing-attribute, bad-argument-type) ✅
└── 0.3: Attribute Safety (missing-attribute elimination) 🚧

Phase 1: Type Safety Core (bad-assignment, bad-return)
Phase 2: Data Access Safety (not-iterable)
Phase 3: Function Contracts (bad-function-definition)
Phase 4: Inheritance Safety (bad-override, bad-param-name-override)
Phase 5: Advanced Types (no-matching-overload, etc.)
```

### Configuration Architecture
```python
# pyproject.toml
[tool.pyrefly]
project_excludes = ["gen/**/*.py"]  # Exclude generated code

[tool.pyrefly.errors]
# Phase 0.3: Enable critical runtime safety
unbound-name = true          # NameError prevention
unsupported-operation = true # TypeError prevention
missing-attribute = true     # AttributeError prevention
bad-argument-type = true     # Function call safety
```

## Component Design

### Error Type Categories
1. **Runtime Safety (Phase 0)**: Errors that cause immediate crashes
   - `unbound-name`: NameError when accessing undefined variables
   - `unsupported-operation`: TypeError from invalid operations
   - `missing-attribute`: AttributeError from None/object attribute access
   - `bad-argument-type`: TypeError from wrong function arguments

2. **Type Safety (Phase 1)**: Variable and function contract violations
   - `bad-assignment`: Incompatible variable assignments
   - `bad-return`: Function return type mismatches

3. **Data Access Safety (Phase 2)**: Collection and iteration safety
   - `not-iterable`: Attempting to iterate over non-iterable objects

4. **Function Contracts (Phase 3)**: Function signature consistency
   - `bad-function-definition`: Parameter mismatch in function definitions

5. **Inheritance Safety (Phase 4)**: Class hierarchy consistency
   - `bad-override`: Method override signature mismatches
   - `bad-param-name-override`: Parameter name inconsistencies

6. **Advanced Types (Phase 5)**: Complex type system features
   - `no-matching-overload`: Function overload resolution failures

### Error Resolution Patterns

#### Pattern 1: Null Safety Guards
```python
# BEFORE: missing-attribute error
result = obj.attribute  # obj could be None

# AFTER: Add null check
if obj is not None:
    result = obj.attribute
else:
    result = default_value
```

#### Pattern 2: Type Conversion
```python
# BEFORE: bad-argument-type error
func(tuple_data)  # func expects str

# AFTER: Convert type
func(str(tuple_data))
```

#### Pattern 3: Collection Safety
```python
# BEFORE: not-iterable error
for item in data:  # data could be None

# AFTER: Check iterability
if data is not None:
    for item in data:
```

#### Pattern 4: Variable Typing
```python
# BEFORE: bad-assignment error
count: int = float_value  # Incompatible assignment

# AFTER: Convert or change type
count: int = int(float_value)
# or
count: float = float_value
```

## Implementation Strategy

### Phase Progression Rules
1. **Atomic Commits**: Each error fix is committed separately with descriptive messages
2. **Error Count Tracking**: Baseline established, progress measured by error reduction
3. **No Regressions**: Previous phase errors remain fixed
4. **Controlled Expansion**: Only enable new error types when current phase is complete

### Quality Assurance
- **Runtime Compatibility**: All fixes preserve existing behavior
- **Test Suite Integrity**: Existing tests continue to pass
- **Code Readability**: Type safety improvements don't obscure logic
- **Performance Neutral**: No significant performance impact from fixes

### Rollback Strategy
- **Configuration-Based**: Disable error types in pyproject.toml to rollback
- **Branch-Based**: Feature branch allows easy rollback to master
- **Incremental**: Can rollback individual phases without affecting others

## Success Metrics

### Error Reduction Targets
- **Phase 0.1**: unsupported-operation (70→59), unbound-name (47→37)
- **Phase 0.2**: Enable missing-attribute (416 errors), bad-argument-type (206 errors)
- **Phase 0.3**: missing-attribute (416→359), bad-argument-type (206→206)
- **Overall**: 22% reduction from baseline (920→718 errors)

### Quality Metrics
- **Zero Breaking Changes**: Runtime behavior unchanged
- **Test Coverage**: All existing tests pass
- **Code Quality**: Maintainable, readable code
- **Performance**: No degradation in execution speed

## Risk Mitigation

### Technical Risks
- **False Positives**: Pyrefly errors that don't represent real issues
  - *Mitigation*: Manual review of each error before fixing
- **Complex Fixes**: Some errors require significant refactoring
  - *Mitigation*: START SMALL principle, tackle simple fixes first
- **Generated Code**: Protobuf and schema files causing noise
  - *Mitigation*: project_excludes configuration excludes gen/**/*.py

### Operational Risks
- **Timeline Delays**: Underestimating complexity of error fixes
  - *Mitigation*: Phased approach allows incremental progress
- **Team Disruption**: Type checking blocking development
  - *Mitigation*: Controlled rollout, can disable checks if needed
- **Merge Conflicts**: Long-running branch diverges from master
  - *Mitigation*: Regular rebasing, atomic commits for easy conflict resolution

## Future Evolution

### Phase 1-5 Expansion
The foundation established in Phase 0 enables systematic rollout of remaining error types with proven patterns and tooling.

### Integration with CI/CD
Future integration with CI pipelines to prevent error regressions and enforce type safety standards.

### Advanced Features
Potential future enhancements include:
- Type annotation generation
- Automated fix suggestions
- Integration with mypy/pylance for IDE support
- Custom error type definitions for domain-specific safety
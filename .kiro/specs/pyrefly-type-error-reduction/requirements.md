# Requirements Document

## Project Description (Input)
Pyrefly Type Error Reduction Rollout - Systematic elimination of type errors in the Cryptofeed codebase through phased rollout of pyrefly type checking, starting with critical runtime safety checks and progressing to advanced type safety features.

## Engineering Principles Applied
- **START SMALL**: Begin with controlled error types, expand incrementally
- **SOLID**: Single responsibility for each phase, clear separation of concerns
- **KISS**: Simple configuration, atomic commits, focused error categories
- **YAGNI**: Enable only necessary error types per phase, avoid premature complexity
- **TDD**: Test-driven approach with error reduction metrics and validation

## Requirements (Phased Rollout)

### Functional Requirements (Behavioral Specifications)

#### Phase 0: Foundation Setup ✅
1. **FR-0.1**: Pyrefly Configuration Infrastructure ✅
   - WHEN pyrefly is installed THEN configuration file supports error type selection
   - WHEN project_excludes configured THEN generated code is excluded from type checking
   - WHEN error types enabled THEN only specified errors are reported
   - WHEN rollout starts THEN baseline error count is established

2. **FR-0.2**: Controlled Error Type Activation ✅
   - WHEN phase 0.1 starts THEN enable unbound-name and unsupported-operation checks
   - WHEN phase 0.2 starts THEN enable missing-attribute and bad-argument-type checks
   - WHEN error types enabled THEN all other error types remain disabled
   - WHEN errors fixed THEN atomic commits track progress

#### Phase 0.3: Extended Foundation (Current Phase) 🚧
3. **FR-0.3**: Missing Attribute Error Elimination 🚧
   - WHEN missing-attribute errors detected THEN systematically fix AttributeError sources
   - WHEN attribute access fails THEN add proper null checks or type guards
   - WHEN object attributes accessed THEN ensure object is not None before access
   - WHEN 57 missing-attribute errors fixed THEN reduce from 416 to 359 remaining

4. **FR-0.4**: Bad Argument Type Error Elimination 📋
   - WHEN bad-argument-type errors detected THEN fix function call type mismatches
   - WHEN function parameters receive wrong types THEN add type conversions or validation
   - WHEN tuple passed instead of string THEN convert or restructure parameters
   - WHEN 0 bad-argument-type errors fixed THEN maintain 206 remaining for next phase

#### Phase 1: Type Safety Core 📋
5. **FR-1.1**: Variable Assignment Safety 📋
   - WHEN bad-assignment errors detected THEN fix variable type assignment mismatches
   - WHEN incompatible types assigned THEN add type conversions or change variable types
   - WHEN float assigned to int THEN use appropriate numeric type or conversion

6. **FR-1.2**: Return Type Safety 📋
   - WHEN bad-return errors detected THEN fix function return type mismatches
   - WHEN function returns wrong type THEN update return type annotations or implementation

#### Phase 2: Data Access Safety 📋
7. **FR-2.1**: Iteration Safety 📋
   - WHEN not-iterable errors detected THEN fix iteration over non-iterable objects
   - WHEN None iterated THEN add null checks before iteration
   - WHEN wrong type iterated THEN convert to iterable or fix data structure

#### Phase 3: Function Contracts 📋
8. **FR-3.1**: Function Signature Safety 📋
   - WHEN bad-function-definition errors detected THEN fix function parameter mismatches
   - WHEN parameter names conflict THEN rename parameters to match base class contracts

#### Phase 4: Inheritance Safety 📋
9. **FR-4.1**: Override Safety 📋
   - WHEN bad-override errors detected THEN fix method override type incompatibilities
   - WHEN parameter types don't match THEN update method signatures to match base classes

10. **FR-4.2**: Parameter Name Consistency 📋
    - WHEN bad-param-name-override errors detected THEN fix parameter name mismatches
    - WHEN parameter names differ from base THEN rename to match inheritance contracts

#### Phase 5: Advanced Types 📋
11. **FR-5.1**: Overload Resolution 📋
    - WHEN no-matching-overload errors detected THEN fix function overload ambiguities
    - WHEN multiple overloads match THEN add type hints to disambiguate calls

### Technical Requirements (Implementation Specifications)

#### Configuration Management
1. **TR-1.1**: Pyrefly Configuration File 📋
   - IF pyproject.toml exists THEN [tool.pyrefly] section configures error types
   - WHEN project_excludes defined THEN generated code excluded from checking
   - WHEN error types enabled THEN only specified error categories reported

2. **TR-1.2**: Error Type Control 📋
   - IF error type set to true THEN pyrefly reports those errors
   - IF error type set to false THEN pyrefly ignores those errors
   - WHEN all error types false THEN no type checking performed

#### Error Reduction Tracking
3. **TR-2.1**: Progress Metrics 📋
   - WHEN errors counted THEN baseline established at rollout start
   - WHEN fixes committed THEN error count decreases monotonically
   - WHEN phase completes THEN all errors in that category eliminated

4. **TR-2.2**: Atomic Commits 📋
   - WHEN fixes made THEN commit message includes error type and count reduction
   - WHEN phase advances THEN commit message indicates phase transition
   - WHEN baseline established THEN commit preserves initial error state

#### Code Quality Maintenance
5. **TR-3.1**: Type Safety Without Breaking Changes 📋
   - WHEN types fixed THEN runtime behavior remains unchanged
   - WHEN null checks added THEN existing functionality preserved
   - WHEN type conversions added THEN data integrity maintained

6. **TR-3.2**: Incremental Rollout 📋
   - WHEN phase advances THEN only new error types enabled
   - WHEN previous phases complete THEN no regression in fixed errors
   - WHEN rollout completes THEN comprehensive type safety achieved

### Non-Functional Requirements (Quality Attributes)

#### Performance
1. **NR-1.1**: Type Checking Performance 📋
   - WHILE pyrefly runs THEN execution completes within reasonable time
   - WHEN errors fixed THEN type checking speed may improve
   - WHEN generated code excluded THEN checking focuses on source code only

#### Maintainability
2. **NR-2.1**: Code Readability 📋
   - WHEN type fixes applied THEN code remains readable and maintainable
   - WHEN null checks added THEN logic flow remains clear
   - WHEN type conversions added THEN intent remains obvious

#### Reliability
3. **NR-3.1**: Runtime Safety 📋
   - WHEN type errors fixed THEN runtime crashes prevented
   - WHEN AttributeError sources fixed THEN null pointer exceptions avoided
   - WHEN TypeError sources fixed THEN type mismatch crashes prevented

## Success Criteria

### Error Reduction Targets
- **Phase 0.1**: unsupported-operation errors reduced from 70 to 59 (16% reduction)
- **Phase 0.2**: unbound-name errors reduced from 47 to 37 (21% reduction)
- **Phase 0.3**: missing-attribute errors reduced from 416 to 359 (14% reduction target)
- **Phase 0.4**: bad-argument-type errors reduced from 206 to 0 (100% reduction target)
- **Overall Phase 0**: Total errors reduced from 117 to <50 (57% reduction)

### Quality Metrics
- **Type Safety**: All enabled error types eliminated before phase advancement
- **Code Quality**: No degradation in existing functionality or performance
- **Maintainability**: Code remains readable and well-structured after fixes

### Completion Criteria
- **Phase Completion**: All errors in current phase eliminated
- **Regression Testing**: Existing tests pass after type fixes
- **Documentation**: Error patterns and fixes documented for future reference
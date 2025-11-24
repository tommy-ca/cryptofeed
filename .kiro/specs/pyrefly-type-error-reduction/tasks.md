# Implementation Tasks

## Project Overview
Systematic elimination of type errors in the Cryptofeed codebase through phased pyrefly rollout, focusing on critical runtime safety checks and progressive type safety improvements. The rollout follows engineering principles of START SMALL, SOLID, KISS, and YAGNI with atomic commits and measurable progress tracking.

## Phase 0: Foundation Setup ✅

### Task 0.1: Pyrefly Configuration Infrastructure ✅
- **Objective**: Establish baseline pyrefly configuration and error counting
- **Implementation**:
  - Configure `pyproject.toml` with `[tool.pyrefly]` section
  - Set up `project_excludes = ["gen/**/*.py"]` to exclude generated code
  - Enable controlled error types: `unbound-name`, `unsupported-operation`
  - Establish baseline error count: 117 errors (70 unsupported-operation + 47 unbound-name)
- **Status**: ✅ COMPLETED
- **Engineering Principles**: START SMALL, Controlled Rollout

### Task 0.2: Core Safety Error Elimination ✅
- **Objective**: Fix most critical runtime crash sources
- **Implementation**:
  - Fixed 11 unsupported-operation errors (70 → 59, 16% reduction)
  - Fixed 10 unbound-name errors (47 → 37, 21% reduction)
  - Maintained runtime compatibility with no breaking changes
  - Atomic commits for each error fix with descriptive messages
- **Status**: ✅ COMPLETED
- **Engineering Principles**: TDD, Atomic Commits, Zero Breaking Changes

### Task 0.3: Extended Safety Checks 🚧
- **Objective**: Enable and fix missing-attribute and bad-argument-type errors
- **Implementation**:
  - Enable `missing-attribute` and `bad-argument-type` error types
  - Fix 57 missing-attribute errors (416 → 359, 14% reduction)
  - Maintain 206 bad-argument-type errors for next phase
  - Focus on null safety guards and type conversions
- **Status**: 🚧 IN PROGRESS (57/416 missing-attribute errors fixed)
- **Engineering Principles**: Incremental Progress, Pattern-Based Fixes

## Phase 1: Type Safety Core 📋

### Task 1.1: Variable Assignment Safety 📋
- **Objective**: Eliminate bad-assignment errors for type-safe variable assignments
- **Implementation**:
  - Enable `bad-assignment` error type
  - Fix incompatible type assignments (e.g., float to int)
  - Add proper type conversions where needed
  - Maintain runtime behavior while improving type safety
- **Status**: 📋 PENDING
- **Engineering Principles**: Type Safety, Backward Compatibility

### Task 1.2: Return Type Safety 📋
- **Objective**: Ensure functions return correct types
- **Implementation**:
  - Enable `bad-return` error type
  - Fix function return type mismatches
  - Update type annotations to match actual return values
  - Preserve existing API contracts
- **Status**: 📋 PENDING
- **Engineering Principles**: Contract Consistency, API Stability

## Phase 2: Data Access Safety 📋

### Task 2.1: Iteration Safety 📋
- **Objective**: Prevent iteration over non-iterable objects
- **Implementation**:
  - Enable `not-iterable` error type
  - Add null checks before iteration
  - Convert data structures to iterables where appropriate
  - Ensure collection safety throughout codebase
- **Status**: 📋 PENDING
- **Engineering Principles**: Null Safety, Data Structure Validation

## Phase 3: Function Contracts 📋

### Task 3.1: Function Signature Consistency 📋
- **Objective**: Ensure function parameter contracts are consistent
- **Implementation**:
  - Enable `bad-function-definition` error type
  - Fix parameter mismatch issues
  - Align function signatures with base class expectations
  - Maintain API compatibility
- **Status**: 📋 PENDING
- **Engineering Principles**: Interface Consistency, Inheritance Safety

## Phase 4: Inheritance Safety 📋

### Task 4.1: Method Override Safety 📋
- **Objective**: Ensure method overrides match base class signatures
- **Implementation**:
  - Enable `bad-override` error type
  - Fix method signature mismatches in inheritance hierarchies
  - Update parameter names and types to match base classes
  - Preserve polymorphic behavior
- **Status**: 📋 PENDING
- **Engineering Principles**: Liskov Substitution, Polymorphism

### Task 4.2: Parameter Name Consistency 📋
- **Objective**: Ensure parameter names match across inheritance hierarchies
- **Implementation**:
  - Enable `bad-param-name-override` error type
  - Fix parameter name mismatches in overridden methods
  - Align naming conventions with base class contracts
  - Maintain code readability
- **Status**: 📋 PENDING
- **Engineering Principles**: Naming Consistency, Code Clarity

## Phase 5: Advanced Types 📋

### Task 5.1: Overload Resolution 📋
- **Objective**: Ensure function overloads are properly resolvable
- **Implementation**:
  - Enable `no-matching-overload` error type
  - Fix ambiguous function call resolutions
  - Add type hints to disambiguate overloads
  - Optimize for common usage patterns
- **Status**: 📋 PENDING
- **Engineering Principles**: Type System Completeness, API Usability

## Quality Assurance Tasks

### Task QA.1: Regression Testing 📋
- **Objective**: Ensure fixes don't break existing functionality
- **Implementation**:
  - Run full test suite after each error fix batch
  - Verify runtime behavior remains unchanged
  - Check performance impact of type safety improvements
  - Validate against existing integration tests
- **Status**: 📋 CONTINUOUS
- **Engineering Principles**: Quality Gates, Continuous Validation

### Task QA.2: Progress Tracking 📋
- **Objective**: Maintain accurate metrics and reporting
- **Implementation**:
  - Update error counts after each fix batch
  - Track progress against phase targets
  - Document error patterns and solutions
  - Generate rollout status reports
- **Status**: 📋 CONTINUOUS
- **Engineering Principles**: Transparency, Measurable Progress

### Task QA.3: Code Review Standards 📋
- **Objective**: Maintain code quality during type safety improvements
- **Implementation**:
  - Review all type fixes for readability
  - Ensure null checks don't obscure logic flow
  - Validate that type conversions preserve semantics
  - Check for consistent error handling patterns
- **Status**: 📋 CONTINUOUS
- **Engineering Principles**: Code Quality, Maintainability

## Success Criteria Validation

### Error Reduction Milestones
- **Phase 0.1**: unsupported-operation (70→59), unbound-name (47→37) ✅
- **Phase 0.2**: Enable missing-attribute/bad-argument-type ✅
- **Phase 0.3**: missing-attribute (416→359) 🚧
- **Phase 0 Complete**: Total errors <50 (57% reduction from baseline)
- **Phase 1 Complete**: bad-assignment, bad-return errors eliminated
- **Phase 2 Complete**: not-iterable errors eliminated
- **Phase 3 Complete**: bad-function-definition errors eliminated
- **Phase 4 Complete**: bad-override, bad-param-name-override errors eliminated
- **Phase 5 Complete**: no-matching-overload errors eliminated

### Quality Metrics
- **Zero Runtime Regressions**: All existing tests pass
- **Code Maintainability**: Type safety doesn't reduce readability
- **Performance Neutral**: No significant performance degradation
- **API Stability**: Public interfaces remain unchanged

## Implementation Notes

### Error Fix Patterns
1. **Null Safety**: `if obj is not None: obj.attribute`
2. **Type Conversion**: `int(float_value)` or `str(tuple_data)`
3. **Collection Checks**: `if data is not None: for item in data`
4. **Variable Typing**: Change declarations to match usage

### Commit Standards
- **Atomic**: One error fix per commit
- **Descriptive**: Include error type and count reduction
- **Traceable**: Reference specific files and line numbers
- **Reversible**: Easy to identify and rollback if needed

### Rollback Strategy
- **Configuration**: Disable error types in pyproject.toml
- **Branch**: Use feature branch for isolation
- **Incremental**: Rollback phases independently
- **Safe**: No impact on production deployments
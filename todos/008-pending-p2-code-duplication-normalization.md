---
status: pending
priority: p2
issue_id: "008"
tags: [code-review, dry, refactoring, maintainability]
dependencies: []
---

# Code Duplication: Normalization Logic Across 3 Files

## Problem Statement

String normalization logic for exchanges and symbols is duplicated across 3 separate files with slight variations, violating the DRY (Don't Repeat Yourself) principle. This creates maintenance burden and potential for inconsistent behavior.

**Why This Matters**:
- Changes to normalization rules must be replicated in 3 places
- Risk of divergence leading to partition/topic mismatches
- Increased test surface area
- Harder to ensure consistent behavior

## Findings from Review Agents

**Pattern Recognition Specialist** identified this as CRITICAL ANTI-PATTERN #2.1:

**Duplicate #1**: `cryptofeed/backends/kafka/topic_manager.py:63-67`
```python
def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace("_", "-").lower()

def _normalize_exchange(exchange: str) -> str:
    return exchange.strip().lower()
```

**Duplicate #2**: `cryptofeed/backends/kafka/partitioner.py:16-21`
```python
def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace("_", "-").lower()

def _normalize_exchange(exchange: str) -> str:
    return exchange.strip().lower()
```

**Duplicate #3**: `cryptofeed/backends/kafka/headers.py:74-79` (with variation)
```python
# Normalize exchange: lowercase and strip whitespace
exchange_str = str(exchange).strip().lower() if exchange else "unknown"

# Normalize symbol: strip whitespace and convert underscores to hyphens
symbol_str = str(symbol).strip() if symbol else "unknown"
symbol_str = symbol_str.replace("_", "-")
```

**Inconsistency**: Headers module adds `str()` coercion and `"unknown"` fallback, others don't.

## Proposed Solutions

### Solution 1: Shared Utility Module (Recommended)
**Pros**: Single source of truth, consistent behavior, easier to test
**Cons**: Adds one more module (but tiny)
**Effort**: Small (30 minutes)
**Risk**: Low (pure refactoring)

**Implementation**:
```python
# cryptofeed/backends/kafka/normalization.py
"""Kafka-specific normalization utilities for exchanges and symbols."""

def normalize_symbol(symbol: str | None) -> str:
    """Normalize symbol for Kafka topic/partition/header usage.

    Rules:
    - Convert to lowercase
    - Replace '/' and '_' with '-'
    - Strip whitespace
    - Return 'unknown' for None/empty

    Examples:
        'BTC/USD' → 'btc-usd'
        'BTC_USD' → 'btc-usd'
        ' ETH-BTC ' → 'eth-btc'
        None → 'unknown'
    """
    if not symbol:
        return "unknown"
    return str(symbol).strip().replace("/", "-").replace("_", "-").lower()

def normalize_exchange(exchange: str | None) -> str:
    """Normalize exchange for Kafka topic/partition/header usage.

    Rules:
    - Convert to lowercase
    - Strip whitespace
    - Return 'unknown' for None/empty

    Examples:
        'Binance' → 'binance'
        ' OKX ' → 'okx'
        None → 'unknown'
    """
    if not exchange:
        return "unknown"
    return str(exchange).strip().lower()
```

**Update all 3 files to import**:
```python
from .normalization import normalize_symbol, normalize_exchange
```

### Solution 2: Keep Duplication, Add Tests
**Pros**: No refactoring needed
**Cons**: Maintains technical debt, higher maintenance cost
**Effort**: Medium (add tests for 3 implementations)
**Risk**: Low

### Solution 3: Inline Everywhere
**Pros**: No imports needed, explicit
**Cons**: Still duplicated, harder to maintain
**Effort**: Small
**Risk**: Low

## Recommended Action

**SOLUTION 1 (Shared Utility Module)** - Clean, maintainable, follows DRY principle.

**Rationale**:
- 15 lines of code eliminates 40+ lines of duplication
- Single place to update normalization rules
- Easier to test (one function vs. three)
- Future normalization needs (e.g., data_type) can use same module

## Technical Details

**Affected Files**:
- `cryptofeed/backends/kafka/topic_manager.py:63-67` - Topic name generation
- `cryptofeed/backends/kafka/partitioner.py:16-21` - Partition key generation
- `cryptofeed/backends/kafka/headers.py:74-79` - Header value encoding

**New File**:
- `cryptofeed/backends/kafka/normalization.py` (15 LOC)

**Test Coverage**:
- Test symbol normalization edge cases (whitespace, mixed separators, None)
- Test exchange normalization edge cases (whitespace, case, None)
- Verify consistency across topic/partition/header usage

## Acceptance Criteria

- [ ] `cryptofeed/backends/kafka/normalization.py` created with 2 functions
- [ ] Docstrings with examples for both functions
- [ ] `topic_manager.py` imports and uses `normalize_symbol`, `normalize_exchange`
- [ ] `partitioner.py` imports and uses `normalize_symbol`, `normalize_exchange`
- [ ] `headers.py` imports and uses `normalize_symbol`, `normalize_exchange`
- [ ] All existing tests pass unchanged (behavior preserved)
- [ ] New unit tests for normalization functions (10+ test cases each)
- [ ] Test edge cases: None, empty string, whitespace, mixed separators
- [ ] Verify consistent output across all 3 usage sites

## Work Log

**2025-12-14**: Issue identified during PR #16 code review by pattern-recognition-specialist agent
- Severity: MEDIUM (P2) - Technical debt, not blocking
- Effort: 30 minutes
- Status: Pending refactoring
- Recommendation: Fix during PR review cycle

## Resources

- PR #16: https://github.com/tommy-ca/cryptofeed/pull/16
- Pattern Recognition Specialist output: See agent output (a4eb9d3)
- CLAUDE.md DRY principle: "Extract common functionality into reusable components"
- Similar pattern in cryptofeed: Decimal handling centralized in `utils.py`

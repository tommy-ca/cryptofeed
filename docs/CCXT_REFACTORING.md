# CCXT Module Refactoring Guide

This document outlines the refactoring strategy for the CCXT integration to improve maintainability while preserving functionality.

## Current Structure

### Large Files
- `cryptofeed/exchanges/ccxt/feed.py` (501 lines)
  - Handles: Config initialization, symbol mapping, feed lifecycle, data transformation
  - Cognitive load: High (too many responsibilities)

- `cryptofeed/exchanges/ccxt/generic.py` (357 lines)
  - Classes: OrderBookSnapshot, TradeUpdate, CcxtMetadataCache, CcxtGenericFeed
  - Issues: Mixed concerns (data types, caching, feed logic)

- `cryptofeed/exchanges/ccxt/config.py` (274 lines)
  - Handles: Configuration parsing, validation, context creation
  - Issues: Complex nested logic in `to_context()` method

## Refactoring Phases

### Phase 1: Immediate (Current PR)
- Split `generic.py` into separate modules:
  - `types.py`: OrderBookSnapshot, TradeUpdate, CcxtUnavailable
  - `metadata.py`: CcxtMetadataCache (with index strategy)
  - `generic.py`: CcxtGenericFeed (streamlined)

- Improve `feed.py`:
  - Extract `_initialize_config()` method
  - Extract `_setup_credentials()` method
  - Add docstrings to complex methods
  - Reduce method length (max ~40 lines per method)

- Simplify `config.py`:
  - Extract `_build_context_defaults()` helper
  - Add validation docstrings
  - Separate path resolution logic

### Phase 2: Follow-up
- Extract `lifecycle.py` from feed.py
  - start(), stop(), _handle_reconnection()
  - Keep feed.py < 200 lines

- Add comprehensive type hints
  - Use TypedDict for configuration dicts
  - Document callback signatures

- Create `utils.py` for shared helpers
  - Symbol transformation utilities
  - Proxy resolution

### Phase 3: Testing & Documentation
- Unit tests for each split module
- Integration tests across splits
- Update docstrings with examples

## Guidelines for Refactoring

### Keep Together
- Symbol normalization logic (already cohesive)
- Proxy initialization (single concern)
- Type definitions (small, stable)

### Separate
- Feed lifecycle from initialization
- Configuration validation from context creation
- Data transformation from feed control

### Naming Conventions
- `*_lifecycle.py`: Start, stop, reconnection
- `*_types.py`: Data classes, exceptions
- `*_metadata.py`: Caching, market data
- `*_adapters.py`: Data transformation (already exists)

## File Structure (Target)

```
cryptofeed/exchanges/ccxt/
├── __init__.py             # Public API
├── feed.py                 # CcxtFeed (150-200 lines)
├── generic.py              # CcxtGenericFeed (150 lines)
├── types.py                # Data classes, exceptions
├── metadata.py             # CcxtMetadataCache
├── config.py               # Configuration (improved)
├── context.py              # (already exists)
├── transport/              # (already exists)
├── adapters/               # (already exists)
└── exchanges/              # (already exists)
```

## Refactoring Checklist

- [ ] Phase 1: Split generic.py
  - [ ] Create types.py
  - [ ] Create metadata.py
  - [ ] Update imports in generic.py
  - [ ] Add tests

- [ ] Phase 1: Improve feed.py
  - [ ] Extract _initialize_config()
  - [ ] Extract _setup_credentials()
  - [ ] Add method docstrings
  - [ ] Add tests

- [ ] Phase 1: Simplify config.py
  - [ ] Extract _build_context_defaults()
  - [ ] Add validation docstrings
  - [ ] Add tests

- [ ] Phase 2: Extract lifecycle
- [ ] Phase 2: Add comprehensive type hints
- [ ] Phase 2: Create utils.py

- [ ] Phase 3: Comprehensive testing
- [ ] Phase 3: Update examples
- [ ] Phase 3: Update CLAUDE.md patterns

## Backward Compatibility

All refactoring maintains public API compatibility:
- `from cryptofeed.exchanges.ccxt import CcxtFeed` ✓
- `from cryptofeed.exchanges.ccxt.config import CcxtConfig` ✓
- Legacy dict-based args still supported ✓

Module-internal imports may change, but are not part of public API.

## Testing Strategy

1. **Unit Tests**: Test each module independently
   - `tests/unit/test_ccxt_types.py`
   - `tests/unit/test_ccxt_metadata.py`
   - `tests/unit/test_ccxt_config.py`

2. **Integration Tests**: Verify splits work together
   - `tests/integration/test_ccxt_feed_smoke.py` (existing)
   - Add tests for edge cases

3. **Regression Tests**: Ensure behavior unchanged
   - Run existing test suite
   - Verify backward compat

## Success Criteria

- [ ] All files < 250 lines
- [ ] Cyclomatic complexity per method < 10
- [ ] Test coverage > 80%
- [ ] Zero functional changes
- [ ] All tests passing
- [ ] Backward compatible


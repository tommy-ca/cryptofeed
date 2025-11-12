# Schema Versioning Guide

Task 18.1 - Schema Evolution Best Practices and Compatibility Rules

## Table of Contents

1. [Overview](#overview)
2. [Versioning Strategy](#versioning-strategy)
3. [Compatibility Rules](#compatibility-rules)
4. [Schema Evolution Examples](#schema-evolution-examples)
5. [Backward Compatibility](#backward-compatibility)
6. [Forward Compatibility](#forward-compatibility)
7. [Testing Schema Changes](#testing-schema-changes)
8. [Migration Procedures](#migration-procedures)
9. [Compatibility Matrix](#compatibility-matrix)
10. [Deprecation Guidelines](#deprecation-guidelines)

---

## Overview

Schema versioning enables Cryptofeed to evolve its protobuf message definitions while maintaining compatibility with existing producers and consumers. This guide provides best practices for managing schema changes in production environments.

### Key Principles

- **Semantic Versioning**: Use major.minor.patch (e.g., 1.2.3) for schema versions
- **Backward Compatibility**: New schemas must read old data (critical for producers)
- **Forward Compatibility**: Old schemas must read new data (critical for consumers)
- **Compatibility Modes**: Confluent Schema Registry enforces BACKWARD, FORWARD, FULL, or TRANSITIVE
- **Zero Downtime**: Schema evolution should not require service restarts

### Scope

This guide applies to all protobuf schemas used in cryptofeed:
- Trade messages
- OrderBook (L2Snapshot) messages
- Ticker messages
- Candle messages
- Funding Rate messages
- Liquidation messages
- Index messages
- Open Interest messages
- And all other data types

---

## Versioning Strategy

### Semantic Versioning (major.minor.patch)

```
1.0.0
│ │ └─ Patch: Bug fixes, non-breaking refinements
│ └─── Minor: New optional fields, backward/forward compatible changes
└───── Major: Breaking changes, incompatible schema changes
```

### Version Numbering Rules

1. **MAJOR** (1.0.0 → 2.0.0):
   - Removed required fields
   - Changed field types in incompatible ways
   - Requires migration and schema registry reconfiguration
   - Enables fresh deployment or dual-write period

2. **MINOR** (1.0.0 → 1.1.0):
   - Added optional fields (with default values)
   - Renamed fields (with field number preservation)
   - Added new messages or enums
   - Removed optional fields (with deprecation notice)

3. **PATCH** (1.0.0 → 1.0.1):
   - Fixed documentation/comments
   - Clarified field meanings
   - No structural changes

### Example: Trade Message Versions

```protobuf
// v1.0.0 - Initial release
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
}

// v1.1.0 - Added optional fields (backward/forward compatible)
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  // New optional fields with defaults
  string exchange = 5;        // Field number 5
  bool maker_order = 6;       // Field number 6
}

// v1.2.0 - More optional fields
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  string exchange = 5;
  bool maker_order = 6;
  string trade_id = 7;        // New unique identifier
  double commission = 8;      // Transaction fee
}

// v2.0.0 - Major change (breaking)
message Trade {
  string symbol = 1;
  string exchange = 2;        // Now required!
  double price = 3;
  double amount = 4;
  int64 timestamp = 5;

  // ... other fields

  // Note: v1 compatible consumers will break
}
```

---

## Compatibility Rules

### Confluent Schema Registry Compatibility Modes

| Mode | Description | Producer Safe | Consumer Safe | When to Use |
|------|-------------|---|---|---|
| **BACKWARD** | New schema reads old data | ✅ Yes | ❌ No | Default (Cryptofeed uses this) |
| **FORWARD** | Old schema reads new data | ❌ No | ✅ Yes | Careful rollouts |
| **FULL** | Both directions compatible | ✅ Yes | ✅ Yes | Most restrictive (safest) |
| **TRANSITIVE** | Compatibility transitively applies | ⚠️ Careful | ⚠️ Careful | Advanced scenarios |

### Default: BACKWARD Compatibility (Recommended)

**Cryptofeed uses BACKWARD by default** because:
- Producers update more frequently than consumers
- Consumers may be third-party (can't force updates)
- New producers must read messages from old consumers
- Graceful degradation for missing fields

```
Old Consumer → New Schema: Consumer reads message, ignores unknown fields
New Consumer → Old Schema: Consumer reads message, uses defaults for missing fields
```

---

## Schema Evolution Examples

### Example 1: Adding Optional Fields (BACKWARD & FORWARD Compatible)

**Before (v1.0.0):**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
}
```

**After (v1.1.0):**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  // New field: optional, has default (empty string)
  optional string exchange = 5;
  optional bool maker_order = 6;
}
```

**Compatibility Analysis:**
- ✅ **BACKWARD**: Old messages (v1.0.0) can be read by new consumers (v1.1.0)
  - Missing `exchange` field → defaults to empty string
  - Missing `maker_order` field → defaults to false

- ✅ **FORWARD**: New messages (v1.1.0) can be read by old consumers (v1.0.0)
  - Unknown fields (5, 6) are ignored
  - Old consumer still reads symbol, price, amount, timestamp

**Safe to Deploy:** Yes (no coordination needed)

---

### Example 2: Removing Optional Fields (BACKWARD Compatible)

**Before (v1.1.0):**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
  optional string exchange = 5;
  optional bool maker_order = 6;
}
```

**After (v1.2.0) - Deprecate `maker_order`:**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
  optional string exchange = 5;

  // DEPRECATED: Use order_side instead
  // Keep field number to maintain backward compatibility
  reserved 6;

  // New replacement field
  string order_side = 7;  // "BUY" or "SELL"
}
```

**Why `reserved 6`?**
- Never reuse field numbers
- Protobuf decoder would misinterpret old messages
- Reserve explicitly to prevent future reuse

**Compatibility:**
- ✅ **BACKWARD**: v1.1.0 messages still readable (field 6 ignored in v1.2.0)
- ✅ **FORWARD**: v1.2.0 messages readable by v1.1.0 (field 7 ignored)

---

### Example 3: Type Changes (Breaking Change → Requires Major Version)

**Before (v1.0.0):**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
}
```

**After (v2.0.0) - Type change:**
```protobuf
message Trade {
  string symbol = 1;
  string price = 2;        // Changed from double to string!
  double amount = 3;
  int64 timestamp = 4;
}
```

**Compatibility Analysis:**
- ❌ **NOT BACKWARD**: v1.0.0 messages (double) can't be read as v2.0.0 (string)
  - Protobuf decoder interprets wire format differently
  - Deserialization will fail or produce garbage values

**Impact:**
- All producers must upgrade simultaneously
- All consumers must upgrade simultaneously
- No coexistence possible → **requires coordination**

**Deployment Strategy for Major Changes:**
1. Set registry compatibility mode to NONE temporarily
2. Deploy producer changes (dual-write to both old and new topics)
3. Migrate consumers to new schema
4. Verify message counts match
5. Cut off old schema production
6. Restore compatibility mode

---

### Example 4: Adding Required Fields (Breaking Change)

**Before (v1.0.0):**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
}
```

**After (v2.0.0) - Added required field:**
```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  string exchange = 5;  // REQUIRED (no 'optional')
}
```

**Why it's breaking:**
- Old messages lack field 5 (required field)
- New consumer expects field 5 to always exist
- Deserialization validation fails

**Solution:** Always make new fields optional:
```protobuf
optional string exchange = 5;  // Now safe!
```

---

## Backward Compatibility

### Definition
**New schema can read old data without losing information.**

Enables scenarios like:
- Rolling producer upgrades (old messages still valid)
- Consumer doesn't need immediate update
- Graceful degradation via default values

### Rules for BACKWARD Compatibility

| Change | Backward Safe? | Notes |
|--------|---|---|
| Add optional field | ✅ Yes | Old messages get default value |
| Remove optional field | ✅ Yes | New messages ignore field |
| Rename field (keep number) | ✅ Yes | Wire format unchanged |
| Change type (number to string) | ❌ No | Wire format differs |
| Add required field | ❌ No | Old messages fail validation |
| Remove required field | ✅ Depends | Only if had default |

### Recommended Pattern: New Optional Fields

```protobuf
message Trade {
  // Core fields (never change)
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  // Extensions area (add new optional fields here)
  optional string exchange = 100;
  optional string trade_id = 101;
  optional double commission = 102;

  // Use field numbers 100+ for extensions to avoid collisions
}
```

---

## Forward Compatibility

### Definition
**Old schema can read new data without errors.**

Enables scenarios like:
- Staged consumer upgrades (old consumers continue working)
- Producers can advance ahead of consumers
- Protobuf ignores unknown fields by design

### Rules for FORWARD Compatibility

| Change | Forward Safe? | Notes |
|--------|---|---|
| Add new field | ✅ Yes | Old consumer ignores unknown field |
| Remove field | ✅ Yes | New messages don't send field |
| Add new enum value | ✅ Yes | Old consumer gets fallback value |
| Rename message | ❌ No | Type name matters in some contexts |

### Built-in Forward Compatibility

Protobuf **automatically provides forward compatibility** by design:
- Unknown fields are skipped during deserialization
- Field numbers identify fields (names don't matter)
- Missing fields get default values

Example:
```protobuf
// v1.0.0 - Consumer version
message Trade {
  string symbol = 1;
  double price = 2;
}

// v1.1.0 - Producer version (consumer still runs v1.0.0)
message Trade {
  string symbol = 1;
  double price = 2;
  optional string exchange = 3;    // Unknown to v1.0.0 consumer
}

// What happens:
// v1.0.0 consumer receives v1.1.0 message:
// - Reads field 1 (symbol) ✅
// - Reads field 2 (price) ✅
// - Skips field 3 (unknown) ✅ (automatic)
// - Deserialization succeeds
```

---

## Testing Schema Changes

### Local Testing with Schema Registry

```bash
# Start local Schema Registry
docker-compose -f tests/docker-compose-schema-registry.yml up -d

# Run schema validation tests
python -m pytest tests/unit/kafka/test_schema_registry.py -v
```

### Test Procedure: Compatibility Validation

```python
"""Test schema compatibility before deploying changes."""

from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    ConfluentSchemaRegistry,
    CompatibilityMode,
)

def test_new_schema_backward_compatible():
    """Verify new schema reads old messages."""
    config = SchemaRegistryConfig(
        registry_type="confluent",
        url="http://localhost:8081",
        compatibility_mode=CompatibilityMode.BACKWARD,
    )
    registry = ConfluentSchemaRegistry(config)

    # Old schema (v1.0.0)
    old_schema = """{
        "type": "record",
        "name": "Trade",
        "fields": [
            {"name": "symbol", "type": "string"},
            {"name": "price", "type": "double"},
            {"name": "amount", "type": "double"},
            {"name": "timestamp", "type": "long"}
        ]
    }"""

    # Register old schema
    old_schema_id = registry.register_schema(
        subject="trades",
        schema=old_schema,
        schema_type="AVRO"
    )

    # New schema (v1.1.0) with optional field
    new_schema = """{
        "type": "record",
        "name": "Trade",
        "fields": [
            {"name": "symbol", "type": "string"},
            {"name": "price", "type": "double"},
            {"name": "amount", "type": "double"},
            {"name": "timestamp", "type": "long"},
            {"name": "exchange", "type": ["null", "string"], "default": null}
        ]
    }"""

    # Check compatibility
    is_compatible = registry.check_compatibility(
        subject="trades",
        schema=new_schema,
        version=1,
    )

    assert is_compatible, "New schema not backward compatible!"

    # Register new schema (only if compatible)
    new_schema_id = registry.register_schema(
        subject="trades",
        schema=new_schema,
        schema_type="AVRO"
    )

    print(f"Successfully upgraded: {old_schema_id} → {new_schema_id}")
```

### Test Suite Checklist

- [ ] **Compatibility Check**: New schema passes `check_compatibility()` with old version
- [ ] **Serialization**: Old messages serialize and deserialize with new schema
- [ ] **Round Trip**: Message → bytes → message produces identical result
- [ ] **Field Defaults**: Missing fields in old messages get correct defaults
- [ ] **Message Count**: Old and new topics receive same message counts
- [ ] **Performance**: Schema evolution doesn't impact throughput

---

## Migration Procedures

### Safe Schema Migration (Backward Compatible Change)

**Timeline: ~30 minutes (no downtime)**

#### Step 1: Prepare New Schema (Day -1)

```python
from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    ConfluentSchemaRegistry,
)

# Register new schema
config = SchemaRegistryConfig(
    registry_type="confluent",
    url="http://schema-registry:8081",
)
registry = ConfluentSchemaRegistry(config)

# Verify backward compatibility
new_schema = '{"type":"record","name":"Trade",...}'
is_compatible = registry.check_compatibility(
    subject="trades-v1",
    schema=new_schema,
)
assert is_compatible, "Schema not compatible!"
```

#### Step 2: Deploy Producer Update (Morning)

Update KafkaCallback configuration:
```yaml
kafka:
  schema_registry:
    url: "http://schema-registry:8081"
    compatibility_mode: "BACKWARD"
```

Deploy new producer code that uses new schema version.

#### Step 3: Monitor Message Flow (2 hours)

- Monitor message production rate
- Verify no schema validation errors
- Check Kafka topic partitions for even distribution

#### Step 4: Verify Consumer Compatibility (3 hours)

Test consumer with mixed messages:
```python
# Old messages (v1.0.0)
old_message = old_schema.serialize({'symbol': 'BTC-USD', 'price': 50000.0})
# New messages (v1.1.0)
new_message = new_schema.serialize({
    'symbol': 'BTC-USD',
    'price': 50000.0,
    'exchange': 'coinbase'
})

# Both should deserialize with new schema
trade1 = new_schema.deserialize(old_message)  # Should work!
trade2 = new_schema.deserialize(new_message)   # Should work!
```

#### Step 5: Complete (No action needed)

Producers and consumers coexist indefinitely. New optional fields are available for gradual rollout.

### Breaking Schema Migration (Major Version Change)

**Timeline: 1-2 weeks (requires coordination)**

#### Phase 1: Dual-Write Mode (Week 1, Days 1-3)

1. Configure producer with `dual_write` mode
2. Write every message to BOTH old and new topics
3. Keep consumers reading from old topics
4. Monitor both topics for equivalent messages

Configuration:
```yaml
kafka:
  topic_strategy: "dual_write"
  schemas:
    trades_old: "v1.0.0"
    trades_new: "v2.0.0"
```

#### Phase 2: Consumer Migration (Week 1, Days 4-7)

1. Deploy new consumer code targeting new topics
2. Run old and new consumers in parallel
3. Validate message counts match
4. Monitor for discrepancies

Verification:
```python
old_messages = consumer.fetch_messages("trades-v1")
new_messages = consumer.fetch_messages("trades-v2")

assert len(old_messages) == len(new_messages), "Count mismatch!"
```

#### Phase 3: Cutover (Week 2, Days 1-2)

1. Stop old consumers
2. Switch producer to new schema only
3. Monitor new consumers
4. Prepare rollback if issues

#### Phase 4: Cleanup (Week 2, Day 3+)

1. Archive old topics (don't delete immediately)
2. Remove old schema from registry
3. Document migration in runbook

---

## Compatibility Matrix

### Confluent Registry Compatibility Modes

```
Schema Registry Configuration:

┌─────────────────────────────────────────────┐
│ Compatibility Mode: BACKWARD (Default)      │
│                                             │
│ Old Producer ──→ New Consumer ✅ OK         │
│ New Producer ──→ Old Consumer ❌ FAILS      │
│                                             │
│ Use Case: Safe producer upgrades            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Compatibility Mode: FORWARD                 │
│                                             │
│ Old Producer ──→ New Consumer ❌ FAILS      │
│ New Producer ──→ Old Consumer ✅ OK         │
│                                             │
│ Use Case: Safe consumer upgrades            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Compatibility Mode: FULL                    │
│                                             │
│ Old Producer ──→ New Consumer ✅ OK         │
│ New Producer ──→ Old Consumer ✅ OK         │
│                                             │
│ Use Case: Maximum safety (most restrictive) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Compatibility Mode: TRANSITIVE              │
│                                             │
│ Schema1 ──→ Schema2 ──→ Schema3             │
│ All pairs mutually compatible               │
│                                             │
│ Use Case: Complex evolution chains          │
└─────────────────────────────────────────────┘
```

### Recommended Deployment Patterns

| Scenario | Mode | Producer Update | Consumer Update | Risk |
|----------|------|---|---|---|
| Add optional field | BACKWARD | Can update | Automatic | Low |
| Remove optional field | BACKWARD | Can update | Automatic | Low |
| Change field type | FULL | Must coordinate | Must coordinate | High |
| Add required field | NONE | Block registration | N/A | Very High |

---

## Deprecation Guidelines

### Marking Fields as Deprecated

```protobuf
message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;

  // Deprecated fields (marked for removal in v3.0.0)
  optional string old_exchange = 5 [deprecated = true];
  optional bool maker_order = 6 [deprecated = true];

  // Replacement fields
  optional string exchange = 7;
  optional string order_side = 8;  // "BUY" or "SELL"
}
```

### Deprecation Timeline

**Phase 1: Announcement (Weeks 1-2)**
- Document deprecated fields
- Add deprecation markers in schema
- Communicate timeline to consumers
- Migration period: 6-8 weeks

**Phase 2: Support Period (Weeks 3-8)**
- Continue supporting deprecated fields
- Populate both old and new fields
- Monitor consumer migration progress
- Provide migration guides

**Phase 3: Removal (After 8 weeks)**
- Stop producing to deprecated fields
- Reserve field numbers
- Test with production consumer set
- Plan major version release

### Deprecation Notice Template

```
DEPRECATION NOTICE: Trade.maker_order

Deprecated since: v1.2.0
Removal planned: v2.0.0 (8 weeks after v1.2.0 release)

Reason: Replaced by order_side field for clarity
        (old: boolean, new: string "BUY"/"SELL")

Migration:
  Before: if trade.maker_order:
  After:  if trade.order_side == "BUY":

Questions? See docs/kafka/schema-versioning.md
```

---

## Best Practices

### 1. Always Use Optional Fields for New Fields

```protobuf
// ❌ WRONG - Will break old consumers
optional string new_field = 10;

// ✅ CORRECT - Add with type union for safety
optional string new_field = 10;
```

### 2. Never Reuse Field Numbers

```protobuf
// ❌ WRONG
reserved 5;
int32 new_field = 5;  // NEVER reuse!

// ✅ CORRECT
reserved 5;
int32 new_field = 6;  // Use new number
```

### 3. Test Before Deploying

```bash
# Validate schema compatibility
python -m pytest tests/unit/kafka/test_schema_registry.py -v -k compatibility

# Test with live Schema Registry
docker-compose -f tests/docker-compose-schema-registry.yml up -d
pytest tests/integration/test_schema_compatibility.py
```

### 4. Version Your Topics

```
cryptofeed.trades-v1         ← v1.x.x schemas
cryptofeed.trades-v2         ← v2.x.x schemas (after breaking change)

cryptofeed.orderbook-v1      ← OrderBook v1.x.x
cryptofeed.orderbook-v2      ← OrderBook v2.x.x
```

### 5. Document Schema Changes

```yaml
# schema-changelog.md

## Trade Message

### v1.2.0 (Released: 2025-11-12)
- Added: exchange field (optional)
- Added: trade_id field (optional)
- Deprecation: maker_order (use order_side instead)

### v1.1.0 (Released: 2025-10-15)
- Added: maker_order field (optional)

### v1.0.0 (Released: 2025-09-01)
- Initial release
```

---

## Troubleshooting

### Schema Registration Fails with "Incompatible Schema"

**Problem:**
```
SchemaRegistrationError: New schema is not compatible with latest version
```

**Solution:**
1. Check compatibility mode:
   ```python
   registry = ConfluentSchemaRegistry(config)
   is_compatible = registry.check_compatibility(
       subject="trades",
       schema=new_schema,
       version=None,  # Latest
   )
   if not is_compatible:
       # Find what's incompatible
   ```

2. Review changes to find incompatible field type or required field
3. Add default values or make fields optional
4. Retest with `check_compatibility()`

### Consumer Sees Default Values for New Fields

**Expected Behavior:**
- Old messages lack new fields → consumer gets default values
- This is correct and expected

**Verification:**
```python
# Old message without exchange field
old_msg = Trade(symbol="BTC-USD", price=50000.0)
# Deserialize with new schema
new_trade = Trade.FromString(old_msg.SerializeToString())
print(new_trade.exchange)  # "" (default for string)
```

### Message Count Mismatch Between Topics

**Problem:**
- Old topic: 1000 messages
- New topic: 900 messages
- Missing 100 messages in migration

**Diagnosis:**
1. Check producer error logs
2. Verify schema registration succeeded
3. Check Kafka topic metrics for missed messages
4. Review message serialization errors

**Solution:**
- Redeploy producer with dual-write mode
- Run replication job to copy missing messages
- Verify message counts match before cutover

---

## References

- [Confluent Schema Registry Documentation](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Protobuf Compatibility Guide](https://developers.google.com/protocol-buffers/docs/overview#updating-a-message-type)
- [JSON Schema Specification](https://json-schema.org/)
- [Avro Schema Evolution](https://avro.apache.org/docs/current/spec.html#Schema+Evolution)

---

*Last Updated: 2025-11-12*
*Task: 18.1 - Schema Versioning Guide*

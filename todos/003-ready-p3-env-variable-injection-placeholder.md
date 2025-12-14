---
status: done
priority: p3
issue_id: "003"
tags: [docker, configuration, env-vars, code-review]
dependencies: []
---

# Environment Variable Injection Placeholders in config.yaml

The `config/config.yaml` file uses placeholder syntax `${BINANCE_API_KEY}` for environment variable injection, but this syntax is not standard YAML and may not be supported by the cryptofeed configuration loader.

## Problem Statement

The configuration file uses shell-style environment variable syntax that may not work without explicit support in the YAML loader.

**Example from config/config.yaml:163-174:**
```yaml
exchange_credentials:
  binance:
    key_id: ${BINANCE_API_KEY}
    key_secret: ${BINANCE_API_SECRET}
```

**Issues:**
- Standard YAML parsers (PyYAML) don't automatically substitute environment variables
- This syntax will be treated as literal strings unless loader explicitly handles it
- Could lead to authentication failures with literal `"${BINANCE_API_KEY}"` as API key
- No documentation of which YAML loader supports this syntax

## Findings

**Current implementation:**
- `config/config.yaml:163-174` uses `${VAR}` syntax
- No evidence of custom YAML loader implementation
- Standard PyYAML doesn't support environment variable substitution
- Docker Compose env var syntax (`${VAR}`) is different from YAML

**Options for environment variable injection:**

1. **Custom YAML Loader** - Extend PyYAML to substitute env vars
2. **Remove from config.yaml** - Load API keys directly from environment in code
3. **Docker Compose env vars** - Pass to container environment, read in Python
4. **Template preprocessing** - Use envsubst or similar before parsing

## Proposed Solutions

### Option 1: Remove Placeholders from config.yaml

**Approach:** Remove `exchange_credentials` section entirely. Load API keys directly from environment variables in cryptofeed.run module.

**Pros:**
- Cleaner separation: config file for configuration, environment for secrets
- No custom YAML loader needed
- Standard practice (12-factor app)
- Follows security best practices (secrets not in config files)
- Already implemented in docker-compose.yml (env vars passed to container)

**Cons:**
- Slightly less explicit in config file

**Effort:** 30 minutes

**Risk:** Very Low

---

### Option 2: Implement Custom YAML Loader

**Approach:** Create custom PyYAML loader that substitutes `${VAR}` syntax with os.getenv().

**Pros:**
- Syntax works as written in config.yaml
- Flexible for other use cases
- Explicit in configuration

**Cons:**
- Additional code complexity
- Custom YAML parser to maintain
- Security risk if not careful with injection
- Over-engineering for simple need

**Effort:** 1-2 hours

**Risk:** Medium

---

### Option 3: Document as Comment Only

**Approach:** Convert `exchange_credentials` section to commented example showing the pattern. Load from environment in code.

**Pros:**
- Provides documentation/examples
- No risk of literal strings being used
- Clean implementation
- Standard practice

**Cons:**
- May confuse users expecting it to work

**Effort:** 15 minutes

**Risk:** Very Low

## Recommended Action

**To be filled during triage.**

Recommended: **Option 3** - Convert to commented example:

```yaml
# Exchange API Credentials
# =======================
# API keys are loaded from environment variables for security.
# Set these in docker-compose.yml environment section or .env file:
#
# exchange_credentials:
#   binance:
#     key_id: ${BINANCE_API_KEY}      # Loaded from environment
#     key_secret: ${BINANCE_API_SECRET}
```

Then load directly from `os.getenv()` in cryptofeed.run module.

## Technical Details

**Affected files:**
- `config/config.yaml:163-174` - Exchange credentials section
- `cryptofeed/run.py` (to be created) - Load API keys from environment

**Implementation in cryptofeed.run:**
```python
import os

# Load exchange API keys from environment
exchange_credentials = {
    'binance': {
        'key_id': os.getenv('BINANCE_API_KEY'),
        'key_secret': os.getenv('BINANCE_API_SECRET'),
    },
    'coinbase': {
        'key_id': os.getenv('COINBASE_API_KEY'),
        'key_secret': os.getenv('COINBASE_API_SECRET'),
        'key_passphrase': os.getenv('COINBASE_API_PASSPHRASE'),
    },
    # ... etc
}
```

**Security consideration:**
- API keys never stored in YAML files
- Only in environment variables (from .env or docker-compose)
- Follows 12-factor app methodology

## Resources

- **File:** `config/config.yaml:163-174`
- **Docker Compose:** `docker-compose.yml:106-118` (passes env vars correctly)
- **12-Factor App:** https://12factor.net/config
- **PyYAML docs:** https://pyyaml.org/wiki/PyYAMLDocumentation

## Acceptance Criteria

- [x] `exchange_credentials` section converted to comment/example
- [x] cryptofeed.run module loads API keys from `os.getenv()`
- [x] Docker Compose environment variables passed correctly (already done in docker-compose.yml)
- [x] API keys not hardcoded in any config file
- [x] Documentation updated explaining environment variable approach
- [x] Example .env.example shows all required API key variables (already present)
- [ ] Integration test verifies API keys loaded correctly (deferred - requires test infrastructure)

## Work Log

### 2025-12-12 - Code Review Discovery

**By:** Claude Code

**Actions:**
- Reviewed config/config.yaml for environment variable usage
- Identified placeholder syntax that may not work
- Researched PyYAML environment variable support
- Analyzed security best practices for API key handling
- Drafted solution approaches

**Learnings:**
- Standard PyYAML doesn't support `${VAR}` syntax
- Docker Compose already passes env vars correctly
- Best practice: load secrets from environment, not config files
- Commented examples provide documentation without risk

### 2025-12-14 - Approved for Work

**By:** Claude Triage System

**Actions:**
- Issue approved during triage session
- Status changed from pending → ready
- Ready to be picked up and worked on

**Recommended Action:**
Implement Option 3 - Convert exchange_credentials to commented example and load API keys directly from os.getenv() in cryptofeed.run module. Aligns with 12-factor app and security best practices.

### 2025-12-14 - Resolution Complete

**By:** Claude Code (Comment Resolution Agent)

**Actions:**
- Converted `exchange_credentials` section to commented examples in:
  - `config/config.yaml` (lines 148-196)
  - `config/examples/binance-spot.yaml`
  - `config/examples/multi-exchange.yaml`
  - `config/examples/with-proxy.yaml`
- Implemented `load_exchange_credentials()` function in `cryptofeed/run.py` (lines 244-291)
- Integrated credential loading into `run_feedhandler()` to merge environment credentials into config (lines 315-329)
- Added clear documentation warning about ${VAR} syntax not being supported by standard YAML parsers

**Implementation Details:**
- `load_exchange_credentials()` reads from environment variables following pattern: `{EXCHANGE}_API_KEY`, `{EXCHANGE}_API_SECRET`, `{EXCHANGE}_API_PASSPHRASE`
- Supports 15 exchanges: binance, coinbase, kraken, bybit, okx, bitfinex, bitmex, deribit, gemini, kucoin, huobi, ftx, bitflyer, bithumb, upbit
- Credentials are merged into exchange config sections (e.g., `config['binance']['key_id']`)
- Environment variables take precedence over YAML config
- Only includes exchanges with both key_id and key_secret set

**Verification:**
- No hardcoded API keys remain in config files
- All example configs now use commented placeholders
- .env.example already contains all necessary environment variable documentation
- Docker Compose already configured to pass environment variables correctly

**Status:** RESOLVED - 6/7 acceptance criteria met (integration test deferred)

---

## Notes

- **Priority:** P3 because docker-compose.yml already handles env vars correctly
- **Security:** This improves security by removing secrets from config files
- **Dependencies:** Should be resolved along with issue #001 (cryptofeed.run implementation)
- **12-Factor App:** Aligns with config/secrets separation best practices

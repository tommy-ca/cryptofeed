# E2E Testing Reproducibility Guide

## Overview

This guide ensures anyone can reproduce the exact E2E test environment using `uv` for deterministic dependency management.

## Why Reproducibility Matters

**Problem**: "Works on my machine" syndrome
- Different pip versions install different dependency versions
- System Python conflicts
- Transient network failures during installation
- Version drift over time

**Solution**: uv + lock files
- ✅ Exact versions guaranteed
- ✅ Isolated from system Python
- ✅ Fast, reliable installations
- ✅ Version controlled environment

---

## Setup Process

### Option 1: Automated (Recommended)

```bash
# One command to rule them all
./tests/e2e/setup_e2e_env.sh

# Activate and test
source tests/e2e/activate.sh
pytest tests/unit/test_proxy_mvp.py -v
```

**What happens**:
1. Creates `.venv-e2e` with Python 3.12
2. Installs all dependencies via uv
3. Generates `requirements-e2e-lock.txt` with exact versions
4. Downloads Mullvad relay list
5. Creates `.env.e2e` with default configuration

### Option 2: Manual (For CI/CD or custom setups)

```bash
# Step 1: Create virtual environment
uv venv --python 3.12

# Step 2: Activate
source .venv-e2e/bin/activate

# Step 3: Install from lock file
uv pip install -r tests/e2e/requirements-e2e-lock.txt

# Step 4: Load environment
source tests/e2e/.env.e2e

# Step 5: Verify
python -c "import cryptofeed; import ccxt; import pytest; print('✓ All imports successful')"
```

---

## Reproducible Binance E2E recipe (public channels, Mullvad SOCKS)

This recipe matches the field-tested run on 2025-12-13 and minimizes skips:

```bash
source tests/e2e/activate.sh

export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"}],"strategy":"round_robin"}'
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"}],"strategy":"round_robin"}'
export CF_SYMBOL_FETCH_TIMEOUT=30
export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
export REDPANDA_HOST_PORT=19092
export REDPANDA_HOST_BOOTSTRAP=localhost:19092
export KAFKA_BOOTSTRAP_SERVERS=$REDPANDA_HOST_BOOTSTRAP

# If port 19092 is already bound, pick another host port (e.g., 29092) and update both vars above.
make redpanda-up
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -k "roundtrip and not placeholder" -vv -s --maxfail=1
make redpanda-down
```

Notes:
- Candle roundtrip may still skip if Binance does not emit within 180s; this is expected flakiness, not a proxy error.
- If preflight REST skips, try a different relay or increase `CF_SYMBOL_FETCH_TIMEOUT`.
- Keep `python-socks` and `aiohttp-socks` installed for SOCKS proxy support.

## Lock File Management

### Understanding the Lock File

`tests/e2e/requirements-e2e-lock.txt` contains:
```
# Example entries
cryptofeed==2.5.0
ccxt==4.2.15
pytest==8.1.0
aiohttp-socks==0.8.4
...
```

**Key Properties**:
- ✅ Pinned versions (no ranges)
- ✅ Includes transitive dependencies
- ✅ Platform-independent where possible
- ✅ Version controlled (committed to git)

### Updating Dependencies

**Update all packages**:
```bash
source tests/e2e/activate.sh
uv pip install --upgrade cryptofeed ccxt pytest aiohttp-socks python-socks psutil
uv pip freeze > tests/e2e/requirements-e2e-lock.txt
git add tests/e2e/requirements-e2e-lock.txt
git commit -m "chore(e2e): update dependencies"
```

**Update single package**:
```bash
source tests/e2e/activate.sh
uv pip install --upgrade ccxt
uv pip freeze > tests/e2e/requirements-e2e-lock.txt
git add tests/e2e/requirements-e2e-lock.txt
git commit -m "chore(e2e): update ccxt to 4.2.16"
```

**Test after update**:
```bash
# Recreate environment from scratch
rm -rf .venv-e2e
./tests/e2e/setup_e2e_env.sh

# Run full test suite
source tests/e2e/activate.sh
pytest tests/unit/test_proxy_mvp.py tests/integration/test_proxy_integration.py -v
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Setup E2E environment
        run: |
          ./tests/e2e/setup_e2e_env.sh
          source tests/e2e/activate.sh
      
      - name: Run smoke tests
        run: |
          source tests/e2e/activate.sh
          pytest tests/unit/test_proxy_mvp.py -v
      
      - name: Run integration tests
        if: env.CRYPTOFEED_TEST_SOCKS_PROXY != ''
        run: |
          source tests/e2e/activate.sh
          pytest tests/integration/test_live_*.py -v -m live_proxy
```

### GitLab CI Example

```yaml
e2e_tests:
  image: python:3.12
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - source $HOME/.cargo/env
    - ./tests/e2e/setup_e2e_env.sh
    - source tests/e2e/activate.sh
  script:
    - pytest tests/unit/test_proxy_mvp.py -v
    - pytest tests/integration/test_proxy_integration.py -v
```

---

## Verifying Reproducibility

### Hash Verification

```bash
# Generate checksum of lock file
sha256sum tests/e2e/requirements-e2e-lock.txt

# Compare across environments
# Same hash = identical environment
```

### Environment Comparison

```bash
# Environment A
source tests/e2e/activate.sh
python -c "import sys; print(sys.version)"
uv pip list --format=json > env_a.json

# Environment B (different machine)
source tests/e2e/activate.sh
python -c "import sys; print(sys.version)"
uv pip list --format=json > env_b.json

# Compare
diff env_a.json env_b.json
# Should be identical
```

---

## Troubleshooting

### Issue: Different results on different machines

**Cause**: Lock file not used or outdated

**Solution**:
```bash
# Always install from lock file
uv pip install -r tests/e2e/requirements-e2e-lock.txt

# NOT: uv pip install ccxt (installs latest, not locked version)
```

### Issue: Lock file has conflicts

**Cause**: Incompatible dependency versions

**Solution**:
```bash
# Clean slate
rm -rf .venv-e2e tests/e2e/requirements-e2e-lock.txt

# Recreate with fresh resolution
./tests/e2e/setup_e2e_env.sh

# Test thoroughly before committing
pytest tests/unit/ tests/integration/ -v
```

### Issue: uv install fails

**Cause**: Network issues or platform incompatibility

**Solution**:
```bash
# Try with verbose output
uv pip install -r tests/e2e/requirements-e2e-lock.txt -v

# Check for platform-specific wheels
uv pip install --no-binary :all: -r tests/e2e/requirements-e2e-lock.txt

# Last resort: Rebuild lock file on target platform
./tests/e2e/setup_e2e_env.sh
```

---

## Best Practices

### 1. Commit Lock Files
```bash
git add tests/e2e/requirements-e2e-lock.txt
git commit -m "chore(e2e): add/update dependency lock"
```

### 2. Periodic Dependency Updates
```bash
# Monthly schedule recommended
./tests/e2e/setup_e2e_env.sh
# → Regenerates lock file with latest compatible versions
```

### 3. Test Lock File Changes
```bash
# Before committing updated lock file
rm -rf .venv-e2e
uv venv --python 3.12
source .venv-e2e/bin/activate
uv pip install -r tests/e2e/requirements-e2e-lock.txt
pytest tests/unit/ tests/integration/ -v
```

### 4. Document Breaking Changes
```bash
# When major version bump causes breakage
git commit -m "chore(e2e): update ccxt 4.x → 5.x

BREAKING CHANGE: ccxt 5.0 changes proxy configuration format.
Updated proxy adapter to handle new structure.

Migration: None for users (internal change only)"
```

---

## Comparison: uv vs pip

| Feature | uv | pip |
|---------|----|----|
| **Speed** | ⚡ 10-100x faster | Standard |
| **Lock files** | ✅ Native support | ⚠️ Requires pip-tools |
| **Resolution** | ✅ Fast SAT solver | Slow backtracking |
| **Determinism** | ✅ Guaranteed | ⚠️ Best effort |
| **Caching** | ✅ Global cache | Per-venv only |
| **Platform** | ✅ Cross-platform | ✅ Cross-platform |
| **Ecosystem** | 🆕 Modern (Rust) | 🔄 Legacy (Python) |

---

## FAQ

**Q: Do I need to commit `.venv-e2e/`?**  
A: No! Only commit `requirements-e2e-lock.txt`. The venv is regenerated from the lock file.

**Q: Can I use pip instead of uv?**  
A: Yes, but you lose speed and determinism:
```bash
python -m venv .venv-e2e
source .venv-e2e/bin/activate
pip install -r tests/e2e/requirements-e2e-lock.txt
```

**Q: What if uv breaks?**  
A: Lock file is pip-compatible. Fallback to pip (see above).

**Q: How often should I update the lock file?**  
A: Monthly for dependencies, immediately for security patches.

**Q: Can I use different Python versions?**  
A: Yes, but regenerate lock file:
```bash
uv venv .venv-e2e --python 3.11
source .venv-e2e/bin/activate
uv pip install -e ".[dev]" ccxt ccxtpro aiohttp-socks python-socks psutil pytest
uv pip freeze > tests/e2e/requirements-e2e-py311-lock.txt
```

---

## Summary

✅ **Use automated setup**: `./tests/e2e/setup_e2e_env.sh`  
✅ **Activate before testing**: `source tests/e2e/activate.sh`  
✅ **Commit lock files**: `git add tests/e2e/requirements-e2e-lock.txt`  
✅ **Test after updates**: Full test suite before committing  
✅ **Document changes**: Clear commit messages for lock file updates  

**Result**: Anyone, anywhere can reproduce your exact test environment.

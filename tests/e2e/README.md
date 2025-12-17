# E2E Testing Environment

## Overview

This directory contains scripts and configuration for reproducible E2E testing using `uv` for fast, deterministic dependency management.

## Quick Start

### Standard uv workflow (tests)

```bash
uv venv --python 3.12
source .venv-e2e/bin/activate
uv pip install -r tests/e2e/requirements-e2e-lock.txt
pytest tests/unit/test_proxy_mvp.py -v
```

- If uv is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- You can also use `uv run pytest …` without activation, but activation keeps commands simple.

### 1. Setup Environment (First Time)

```bash
# Run setup script (creates .venv-e2e and installs dependencies)
./tests/e2e/setup_e2e_env.sh

# Activate environment
source tests/e2e/activate.sh
```

### 2. Run Tests

```bash
# Smoke tests
pytest tests/unit/test_proxy_mvp.py -v

# Live tests (requires proxy)
pytest tests/integration/test_live_binance.py -v -m live_proxy

# Regional validation
./tests/integration/regional_validation.sh

# Stress test
python tests/integration/T4.2-stress-test.py --duration=300 --feeds=10
```

## Reproducibility

### Dependency Locking

The setup script creates `requirements-e2e-lock.txt` with exact versions:

```bash
# Install from lock file
uv venv .venv-e2e --python 3.12
source .venv-e2e/bin/activate
uv pip install -r tests/e2e/requirements-e2e-lock.txt
```

### Environment Configuration

Edit `tests/e2e/.env.e2e` to customize:
- Proxy endpoints
- Test symbols
- Timeout values

## Files

| File | Purpose |
|------|---------|
| `setup_e2e_env.sh` | Main setup script (creates venv, installs deps) |
| `activate.sh` | Quick activation helper |
| `.env.e2e` | Environment variables (generated) |
| `requirements-e2e-lock.txt` | Frozen dependencies (generated) |
| `mullvad-relays/` | Downloaded proxy list (generated) |

## Updating Dependencies

```bash
# Re-run setup to update and re-lock
./tests/e2e/setup_e2e_env.sh

# Or manually update specific package
source tests/e2e/activate.sh
uv pip install --upgrade ccxt
uv pip freeze > tests/e2e/requirements-e2e-lock.txt
```

## CI/CD Integration

```yaml
# .github/workflows/e2e.yml
- name: Setup E2E environment
  run: ./tests/e2e/setup_e2e_env.sh

- name: Run E2E tests
  run: |
    source tests/e2e/activate.sh
    pytest tests/integration/test_live_*.py -v -m live_proxy
```

## Troubleshooting

### uv not found
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Mullvad relays not downloading
```bash
# Manual download
gh run download 18632839930 \
  --repo tommy-ca/mulvad-relay-list \
  -D tests/e2e/mullvad-relays
```

### Import errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use activate.sh which sets it automatically
source tests/e2e/activate.sh
```

#!/bin/bash
# E2E Test Environment Setup Script
# Creates reproducible test environment using uv for fast, deterministic dependency management

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')]${NC} $*"
}

# Detect project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"

# Check prerequisites
if ! command -v uv &> /dev/null; then
    error "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    warn "gh CLI not found. Mullvad relay list will need manual download"
fi

# Configuration
VENV_DIR="${PROJECT_ROOT}/.venv-e2e"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MULLVAD_ARTIFACT_ID="${MULLVAD_ARTIFACT_ID:-18632839930}"
MULLVAD_REPO="${MULLVAD_REPO:-tommy-ca/mulvad-relay-list}"

log "Configuration:"
log "  Virtual env: $VENV_DIR"
log "  Python version: $PYTHON_VERSION"
log "  Mullvad artifact: $MULLVAD_ARTIFACT_ID"

# Step 1: Create virtual environment with uv
log "Step 1: Creating virtual environment with uv..."
if [ -d "$VENV_DIR" ]; then
    warn "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

log "Virtual environment created and activated"
log "Python: $(which python)"
log "Python version: $(python --version)"

# Step 2: Install project dependencies with uv
log "Step 2: Installing project dependencies..."

# Core dependencies
uv pip install -e "$PROJECT_ROOT"

# Development dependencies
uv pip install \
    pytest>=8.0.0 \
    pytest-asyncio>=0.23.0 \
    pytest-cov>=4.0.0 \
    pytest-benchmark>=4.0.0 \
    pytest-mock>=3.12.0 \
    pytest-timeout>=2.2.0

# E2E-specific dependencies
uv pip install \
    ccxt>=4.0.0 \
    ccxtpro>=4.0.0 \
    aiohttp-socks>=0.8.0 \
    python-socks>=2.3.0 \
    psutil>=5.9.0

# Code quality tools
uv pip install \
    ruff>=0.1.0 \
    mypy>=1.7.0

log "Dependencies installed"

# Step 3: Freeze dependencies for reproducibility
log "Step 3: Creating dependency lock..."
uv pip freeze > "$PROJECT_ROOT/tests/e2e/requirements-e2e-lock.txt"
log "Dependency lock saved to tests/e2e/requirements-e2e-lock.txt"

# Step 4: Download Mullvad relay list
log "Step 4: Downloading Mullvad relay list..."
MULLVAD_DIR="$PROJECT_ROOT/tests/e2e/mullvad-relays"
mkdir -p "$MULLVAD_DIR"

if command -v gh &> /dev/null; then
    if gh run download "$MULLVAD_ARTIFACT_ID" \
        --repo "$MULLVAD_REPO" \
        -D "$MULLVAD_DIR" 2>/dev/null; then
        log "Mullvad relay list downloaded"
        
        # Extract and display available proxies
        RELAY_FILE="$MULLVAD_DIR/mullvad-relay-artifacts/mullvad_relays.csv"
        if [ -f "$RELAY_FILE" ]; then
            log "Available proxies:"
            echo ""
            head -5 "$RELAY_FILE" | column -t -s','
            echo ""
            log "Full list: $RELAY_FILE"
        fi
    else
        warn "Failed to download Mullvad relays (authentication or artifact not found)"
        warn "You can manually download from: https://github.com/$MULLVAD_REPO/actions/runs/$MULLVAD_ARTIFACT_ID"
    fi
else
    warn "gh CLI not available, skipping Mullvad relay download"
    warn "Install gh: https://cli.github.com/"
fi

# Step 5: Verify installation
log "Step 5: Verifying installation..."

python -c "
import sys
import importlib.metadata

required_packages = [
    'cryptofeed',
    'pytest',
    'pytest-asyncio',
    'ccxt',
    'ccxtpro',
    'aiohttp-socks',
    'python-socks',
    'psutil',
]

print('Installed packages:')
for pkg in required_packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f'  ✓ {pkg:20s} {version}')
    except importlib.metadata.PackageNotFoundError:
        print(f'  ✗ {pkg:20s} NOT FOUND')
        sys.exit(1)
"

if [ $? -eq 0 ]; then
    log "All required packages verified"
else
    error "Package verification failed"
    exit 1
fi

# Step 6: Generate environment file
log "Step 6: Generating environment configuration..."

ENV_FILE="$PROJECT_ROOT/tests/e2e/.env.e2e"
cat > "$ENV_FILE" << 'EOF'
# E2E Test Environment Configuration
# Source this file before running E2E tests: source tests/e2e/.env.e2e

# Proxy configuration (update with your Mullvad endpoint)
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

# Test symbols (optional overrides)
export CRYPTOFEED_TEST_BINANCE_SYMBOL="BTCUSDT"
export CRYPTOFEED_TEST_BINANCE_WS_STREAM="btcusdt@trade"
export CRYPTOFEED_TEST_CCXT_SYMBOL="BTC/USDC:USDC"
export CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL="BTC/USDC"

# Timeout configuration (seconds)
export CRYPTOFEED_TEST_BINANCE_WS_TIMEOUT="10"
export CRYPTOFEED_TEST_CCXT_WS_TIMEOUT="20"
export CRYPTOFEED_TEST_CCXT_REST_TIMEOUT="15"
export CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT="10"

# Stress test configuration
export CRYPTOFEED_TEST_STRESS_DURATION="300"
export CRYPTOFEED_TEST_STRESS_FEEDS="10"

# Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
EOF

log "Environment file created: $ENV_FILE"

# Step 7: Create activation helper
log "Step 7: Creating activation helper..."

ACTIVATE_SCRIPT="$PROJECT_ROOT/tests/e2e/activate.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# E2E Test Environment Activation
# Usage: source tests/e2e/activate.sh

PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)"

# Activate virtual environment
if [ -f "\$PROJECT_ROOT/.venv-e2e/bin/activate" ]; then
    source "\$PROJECT_ROOT/.venv-e2e/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run tests/e2e/setup_e2e_env.sh first"
    return 1
fi

# Load environment variables
if [ -f "\$PROJECT_ROOT/tests/e2e/.env.e2e" ]; then
    source "\$PROJECT_ROOT/tests/e2e/.env.e2e"
    echo "✓ Environment variables loaded"
else
    echo "⚠ Environment file not found: tests/e2e/.env.e2e"
fi

# Set Python path
export PYTHONPATH="\${PYTHONPATH}:\$PROJECT_ROOT"

# Display status
echo ""
echo "E2E Test Environment Ready"
echo "  Python: \$(which python)"
echo "  Python version: \$(python --version)"
echo "  Proxy: \$CRYPTOFEED_TEST_SOCKS_PROXY"
echo ""
echo "Quick start:"
echo "  pytest tests/unit/test_proxy_mvp.py -v"
echo "  pytest tests/integration/test_live_binance.py -v -m live_proxy"
echo ""
EOF

chmod +x "$ACTIVATE_SCRIPT"
log "Activation helper created: $ACTIVATE_SCRIPT"

# Summary
echo ""
echo "========================================"
echo "✅ E2E Environment Setup Complete"
echo "========================================"
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Dependency lock: tests/e2e/requirements-e2e-lock.txt"
echo "Environment config: tests/e2e/.env.e2e"
echo "Activation script: tests/e2e/activate.sh"
echo ""
echo "Next steps:"
echo "  1. Review proxy configuration in tests/e2e/.env.e2e"
echo "  2. Activate environment: source tests/e2e/activate.sh"
echo "  3. Run tests: pytest tests/integration/test_live_binance.py -v -m live_proxy"
echo ""
echo "To recreate this environment later:"
echo "  uv venv --python $PYTHON_VERSION"
echo "  source .venv-e2e/bin/activate"
echo "  uv pip install -r tests/e2e/requirements-e2e-lock.txt"
echo ""

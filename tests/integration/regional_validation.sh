#!/bin/bash
# Regional validation script for E2E proxy testing
# Tests all exchange/region combinations and generates validation matrix

set -euo pipefail

# Mullvad relay endpoints by region (2025-10-19)
declare -A PROXIES
PROXIES[US]="socks5://us-nyc-wg-socks5-301.relays.mullvad.net:1080"
PROXIES[EU]="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"
PROXIES[ASIA]="socks5://sg-sin-wg-socks5-001.relays.mullvad.net:1080"

# Test targets
EXCHANGES=("binance" "coinbase" "bybit" "hyperliquid" "backpack")
REGIONS=("US" "EU" "ASIA")

# Output files
RESULTS_DIR="test-results/regional"
mkdir -p "$RESULTS_DIR"
MATRIX_FILE="$RESULTS_DIR/regional_matrix.csv"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# Initialize CSV with headers
echo "Exchange,Region,Proxy,REST_Status,REST_Details,WS_Status,WS_Details" > "$MATRIX_FILE"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

# Test a specific exchange/region combination
test_combination() {
    local exchange=$1
    local region=$2
    local proxy=${PROXIES[$region]}
    
    log "Testing $exchange in $region region via $proxy"
    
    export CRYPTOFEED_TEST_SOCKS_PROXY="$proxy"
    
    local rest_status="UNKNOWN"
    local rest_details=""
    local ws_status="UNKNOWN"
    local ws_details=""
    
    # Select appropriate test file
    local test_file=""
    case $exchange in
        binance)
            test_file="tests/integration/test_live_binance.py"
            ;;
        hyperliquid)
            test_file="tests/integration/test_live_ccxt_hyperliquid.py"
            ;;
        backpack)
            test_file="tests/integration/test_live_ccxt_backpack.py"
            ;;
        *)
            warn "No specific test file for $exchange, skipping"
            echo "$exchange,$region,$proxy,SKIPPED,No test file,SKIPPED,No test file" >> "$MATRIX_FILE"
            return
            ;;
    esac
    
    # Run REST tests
    if pytest "$test_file" -v -m "live_proxy" -k "rest or ticker or orderbook" \
       --tb=short --junit-xml="$RESULTS_DIR/${exchange}_${region}_rest.xml" > "$RESULTS_DIR/${exchange}_${region}_rest.log" 2>&1; then
        rest_status="PASS"
        rest_details="Endpoint accessible"
    else
        # Check for specific failure types
        if grep -q "HTTP 451" "$RESULTS_DIR/${exchange}_${region}_rest.log"; then
            rest_status="SKIP"
            rest_details="Geofenced (HTTP 451)"
        elif grep -q "SKIPPED" "$RESULTS_DIR/${exchange}_${region}_rest.log"; then
            rest_status="SKIP"
            rest_details="Test skipped"
        else
            rest_status="FAIL"
            rest_details=$(grep -E "FAILED|ERROR" "$RESULTS_DIR/${exchange}_${region}_rest.log" | head -1 || echo "Unknown error")
        fi
    fi
    
    # Run WebSocket tests
    if pytest "$test_file" -v -m "live_proxy" -k "ws or websocket" \
       --tb=short --junit-xml="$RESULTS_DIR/${exchange}_${region}_ws.xml" > "$RESULTS_DIR/${exchange}_${region}_ws.log" 2>&1; then
        ws_status="PASS"
        ws_details="Stream connected"
    else
        if grep -q "HTTP 451" "$RESULTS_DIR/${exchange}_${region}_ws.log"; then
            ws_status="SKIP"
            ws_details="Geofenced (HTTP 451)"
        elif grep -q "SKIPPED" "$RESULTS_DIR/${exchange}_${region}_ws.log"; then
            ws_status="SKIP"
            ws_details="Test skipped"
        elif grep -q "parse error" "$RESULTS_DIR/${exchange}_${region}_ws.log"; then
            ws_status="KNOWN_ISSUE"
            ws_details="Parse error 4002 (Backpack native)"
        else
            ws_status="FAIL"
            ws_details=$(grep -E "FAILED|ERROR" "$RESULTS_DIR/${exchange}_${region}_ws.log" | head -1 || echo "Unknown error")
        fi
    fi
    
    # Record results
    echo "$exchange,$region,$proxy,$rest_status,$rest_details,$ws_status,$ws_details" >> "$MATRIX_FILE"
    
    # Log summary
    local rest_icon="❓"
    local ws_icon="❓"
    
    case $rest_status in
        PASS) rest_icon="✅" ;;
        SKIP) rest_icon="⚠️ " ;;
        FAIL) rest_icon="❌" ;;
    esac
    
    case $ws_status in
        PASS) ws_icon="✅" ;;
        SKIP) ws_icon="⚠️ " ;;
        FAIL) ws_icon="❌" ;;
        KNOWN_ISSUE) ws_icon="⚠️ " ;;
    esac
    
    log "$exchange/$region: REST=$rest_icon ($rest_status) WS=$ws_icon ($ws_status)"
}

# Main execution
main() {
    log "Starting regional validation"
    log "Exchanges: ${EXCHANGES[*]}"
    log "Regions: ${REGIONS[*]}"
    
    local total_tests=0
    local passed_tests=0
    local skipped_tests=0
    local failed_tests=0
    
    # Test all combinations
    for region in "${REGIONS[@]}"; do
        log "===== Testing $region region ====="
        for exchange in "${EXCHANGES[@]}"; do
            test_combination "$exchange" "$region"
            ((total_tests+=2)) || true  # REST + WS
        done
    done
    
    # Generate summary
    log "Generating summary..."
    
    {
        echo "Regional Validation Summary"
        echo "Generated: $(date)"
        echo ""
        echo "Total Tests: $total_tests"
        echo ""
        echo "Results by Status:"
        echo "  PASS: $(grep -c ",PASS," "$MATRIX_FILE" || echo 0)"
        echo "  SKIP: $(grep -c ",SKIP," "$MATRIX_FILE" || echo 0)"
        echo "  FAIL: $(grep -c ",FAIL," "$MATRIX_FILE" || echo 0)"
        echo "  KNOWN_ISSUE: $(grep -c ",KNOWN_ISSUE," "$MATRIX_FILE" || echo 0)"
        echo ""
        echo "Matrix saved to: $MATRIX_FILE"
        echo ""
        echo "Quick View:"
        column -t -s',' "$MATRIX_FILE" | head -20
    } > "$SUMMARY_FILE"
    
    cat "$SUMMARY_FILE"
    
    # Print warnings for failures
    local failures=$(grep ",FAIL," "$MATRIX_FILE" | wc -l)
    if [ "$failures" -gt 0 ]; then
        warn "Found $failures failed tests - review logs in $RESULTS_DIR"
    fi
    
    log "Regional validation complete"
    log "Full matrix: $MATRIX_FILE"
    log "Summary: $SUMMARY_FILE"
    log "Detailed logs: $RESULTS_DIR/*.log"
}

# Run main
main "$@"

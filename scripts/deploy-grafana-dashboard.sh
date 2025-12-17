#!/usr/bin/env bash
#
# Deploy Grafana Dashboard for Kafka Producer Monitoring
#
# Usage:
#   ./scripts/deploy-grafana-dashboard.sh [--dry-run] [--grafana-url http://localhost:3000]
#
# Environment Variables:
#   GRAFANA_URL: Grafana API endpoint (default: http://localhost:3000)
#   GRAFANA_API_KEY: Grafana API key for authentication
#   GRAFANA_USER: Grafana username (default: admin)
#   GRAFANA_PASSWORD: Grafana password (default: admin)
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_JSON="$PROJECT_ROOT/docs/monitoring/grafana-dashboard.json"

GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --grafana-url)
            GRAFANA_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--grafana-url URL]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Step 1: Validate dashboard JSON
log_info "Validating dashboard JSON..."
if ! python3 "$SCRIPT_DIR/validate_dashboard_json.py" "$DASHBOARD_JSON"; then
    log_error "Dashboard JSON validation failed"
    exit 1
fi
log_info "Dashboard JSON is valid"

# Step 2: Check Grafana API availability
log_info "Checking Grafana API at $GRAFANA_URL..."
if ! curl -sf -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/health" > /dev/null; then
    log_error "Grafana API is not accessible at $GRAFANA_URL"
    log_error "Please ensure Grafana is running and credentials are correct"
    exit 1
fi
log_info "Grafana API is accessible"

# Step 3: Check if dashboard already exists
DASHBOARD_UID="kafka-producer-monitoring"
log_info "Checking if dashboard $DASHBOARD_UID already exists..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
    "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID")

if [ "$HTTP_CODE" = "200" ]; then
    log_warn "Dashboard already exists (will update)"
    ACTION="update"
else
    log_info "Dashboard does not exist (will create)"
    ACTION="create"
fi

# Step 4: Deploy dashboard
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would $ACTION dashboard: $DASHBOARD_JSON"
    log_info "[DRY-RUN] Grafana URL: $GRAFANA_URL"
    log_info "[DRY-RUN] Dashboard UID: $DASHBOARD_UID"
    exit 0
fi

log_info "Deploying dashboard to Grafana..."

# Create API payload with dashboard wrapped in required structure
PAYLOAD=$(jq '. + {
    "dashboard": .,
    "overwrite": true,
    "message": "Deployed via automation script"
}' "$DASHBOARD_JSON")

# Deploy via Grafana API
RESPONSE=$(curl -s -X POST \
    -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$GRAFANA_URL/api/dashboards/db")

# Check response
if echo "$RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
    DASHBOARD_ID=$(echo "$RESPONSE" | jq -r '.id')
    DASHBOARD_URL=$(echo "$RESPONSE" | jq -r '.url')
    log_info "Dashboard deployed successfully"
    log_info "Dashboard ID: $DASHBOARD_ID"
    log_info "Dashboard URL: $GRAFANA_URL$DASHBOARD_URL"
else
    log_error "Dashboard deployment failed"
    echo "$RESPONSE" | jq '.'
    exit 1
fi

log_info "Deployment complete"

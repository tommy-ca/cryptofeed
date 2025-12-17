#!/usr/bin/env bash
#
# Trivy Security Scanner for Cryptofeed Container Images
#
# Scans Docker images for CVE vulnerabilities and fails build on CRITICAL/HIGH findings.
# Generates JSON report with vulnerability details for remediation.
#
# Usage:
#   ./trivy-scan.sh --image cryptofeed:latest [--report trivy-report.json]
#   ./trivy-scan.sh --help
#
# Requirements:
#   - Trivy CLI installed (https://github.com/aquasecurity/trivy)
#   - Docker image built and available locally or in registry
#
# Exit Codes:
#   0 - Success (no CRITICAL/HIGH vulnerabilities)
#   1 - Failure (CRITICAL/HIGH vulnerabilities found)
#   2 - Invalid arguments or Trivy not installed

set -euo pipefail

# Default configuration
IMAGE_NAME="${IMAGE_NAME:-}"
REPORT_PATH="${REPORT_PATH:-trivy-report.json}"
SEVERITY="CRITICAL,HIGH"
FORMAT="json"
EXIT_CODE_ON_FINDINGS=1
TIMEOUT="5m"
QUIET=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage information
usage() {
    cat <<EOF
Trivy Security Scanner for Cryptofeed Container Images

Usage:
    $0 --image IMAGE_NAME [OPTIONS]

Required Arguments:
    --image IMAGE_NAME          Docker image name to scan (e.g., cryptofeed:latest)

Optional Arguments:
    --report PATH               Output report path (default: trivy-report.json)
    --severity LEVELS           Severity levels to scan (default: CRITICAL,HIGH)
    --format FORMAT             Output format: json, table, sarif (default: json)
    --no-exit-code              Don't exit with error code on findings
    --timeout DURATION          Scan timeout (default: 5m)
    --quiet                     Suppress output except errors
    --verbose                   Show verbose output
    --help                      Show this help message

Examples:
    # Scan local image
    $0 --image cryptofeed:latest

    # Scan with custom report path
    $0 --image cryptofeed:v1.2.3 --report /tmp/scan-results.json

    # Scan all severity levels
    $0 --image cryptofeed:latest --severity CRITICAL,HIGH,MEDIUM,LOW

    # Table format for human-readable output
    $0 --image cryptofeed:latest --format table

Exit Codes:
    0 - No vulnerabilities found or severity below threshold
    1 - CRITICAL or HIGH vulnerabilities found
    2 - Invalid arguments or Trivy not installed
EOF
}

# Log message with color
log() {
    local level=$1
    shift
    local message="$*"

    if [[ "$QUIET" == "true" && "$level" != "ERROR" ]]; then
        return
    fi

    case "$level" in
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message" >&2
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} $message" >&2
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message" >&2
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" >&2
            ;;
        *)
            echo -e "$message" >&2
            ;;
    esac
}

# Check if Trivy is installed
check_trivy_installed() {
    if ! command -v trivy &> /dev/null; then
        log ERROR "Trivy is not installed. Please install from https://github.com/aquasecurity/trivy"
        log ERROR "Installation: brew install trivy (macOS) or see https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
        return 1
    fi

    local trivy_version
    trivy_version=$(trivy --version | head -n1)
    log INFO "Trivy version: $trivy_version"
    return 0
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --report)
                REPORT_PATH="$2"
                shift 2
                ;;
            --severity)
                SEVERITY="$2"
                shift 2
                ;;
            --format)
                FORMAT="$2"
                shift 2
                ;;
            --no-exit-code)
                EXIT_CODE_ON_FINDINGS=0
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --quiet)
                QUIET=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log ERROR "Unknown argument: $1"
                usage
                exit 2
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$IMAGE_NAME" ]]; then
        log ERROR "Missing required argument: --image IMAGE_NAME"
        usage
        exit 2
    fi
}

# Update Trivy vulnerability database
update_trivy_db() {
    log INFO "Updating Trivy vulnerability database..."

    if [[ "$VERBOSE" == "true" ]]; then
        trivy image --download-db-only
    else
        trivy image --download-db-only --quiet
    fi

    log SUCCESS "Vulnerability database updated"
}

# Run Trivy scan
run_trivy_scan() {
    log INFO "Scanning image: $IMAGE_NAME"
    log INFO "Severity filter: $SEVERITY"
    log INFO "Output format: $FORMAT"
    log INFO "Report path: $REPORT_PATH"

    local exit_code_flag=""
    if [[ "$EXIT_CODE_ON_FINDINGS" -eq 1 ]]; then
        exit_code_flag="--exit-code 1"
    fi

    local quiet_flag=""
    if [[ "$QUIET" == "true" ]]; then
        quiet_flag="--quiet"
    fi

    # Build Trivy command
    local trivy_cmd=(
        trivy image
        --severity "$SEVERITY"
        --format "$FORMAT"
        --output "$REPORT_PATH"
        --timeout "$TIMEOUT"
        $exit_code_flag
        $quiet_flag
        "$IMAGE_NAME"
    )

    # Run Trivy scan
    local scan_exit_code=0
    if [[ "$VERBOSE" == "true" ]]; then
        log INFO "Running: ${trivy_cmd[*]}"
    fi

    # Execute scan (capture exit code without failing on error)
    set +e
    "${trivy_cmd[@]}"
    scan_exit_code=$?
    set -e

    return $scan_exit_code
}

# Parse scan results and print summary
parse_scan_results() {
    local scan_exit_code=$1

    if [[ ! -f "$REPORT_PATH" ]]; then
        log ERROR "Scan report not found at $REPORT_PATH"
        return 2
    fi

    # Parse JSON report (only if format is JSON)
    if [[ "$FORMAT" == "json" ]]; then
        local total_vulns=0
        local critical_count=0
        local high_count=0

        # Count vulnerabilities by severity using jq
        if command -v jq &> /dev/null; then
            total_vulns=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL" or .Severity == "HIGH")] | length' "$REPORT_PATH" 2>/dev/null || echo "0")
            critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$REPORT_PATH" 2>/dev/null || echo "0")
            high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$REPORT_PATH" 2>/dev/null || echo "0")

            log INFO "Scan results summary:"
            log INFO "  Total CRITICAL/HIGH vulnerabilities: $total_vulns"
            log INFO "  CRITICAL: $critical_count"
            log INFO "  HIGH: $high_count"
        else
            log WARN "jq not installed - cannot parse JSON report for summary"
        fi
    fi

    if [[ $scan_exit_code -eq 0 ]]; then
        log SUCCESS "Security scan passed - no CRITICAL or HIGH vulnerabilities found"
        log SUCCESS "Full report saved to: $REPORT_PATH"
        return 0
    else
        log ERROR "Security scan FAILED - CRITICAL or HIGH vulnerabilities detected"
        log ERROR "Review report at: $REPORT_PATH"
        log ERROR ""
        log ERROR "Remediation steps:"
        log ERROR "  1. Update base image: python:3.11-slim-bookworm to latest patch version"
        log ERROR "  2. Update Python dependencies in requirements.txt"
        log ERROR "  3. Rebuild Docker image"
        log ERROR "  4. Re-run this scan"
        log ERROR ""
        log ERROR "For vulnerability exceptions, see docs/docker/SECURITY.md"
        return 1
    fi
}

# Main execution
main() {
    parse_args "$@"

    # Check prerequisites
    if ! check_trivy_installed; then
        exit 2
    fi

    # Update vulnerability database
    update_trivy_db

    # Run scan
    local scan_exit_code=0
    set +e
    run_trivy_scan
    scan_exit_code=$?
    set -e

    # Parse and display results
    parse_scan_results $scan_exit_code
    exit $?
}

# Run main function
main "$@"

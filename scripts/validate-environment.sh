#!/usr/bin/env bash
#
# Environment Variable Validation Script
# Market Data Kafka Producer - Phase 5 Deployment
#
# Purpose: Validate all required environment variables are set before deployment
# Usage: ./scripts/validate-environment.sh
#
# Exit Codes:
#   0 - All validations passed
#   1 - One or more validations failed
#
# Requirements:
#   - All variables must be non-empty
#   - KAFKA_BOOTSTRAP_SERVERS must include port numbers
#   - VAULT_ADDR must use HTTPS
#   - ENVIRONMENT must be one of: production, staging, development
#

set -uo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track validation status
VALIDATION_FAILED=0
MISSING_VARS=()
INVALID_VARS=()

# Function to check if variable is set and non-empty
check_required_var() {
    local var_name="$1"
    local var_value="${!var_name:-}"

    if [[ -z "$var_value" ]]; then
        MISSING_VARS+=("$var_name")
        echo -e "${RED}✗${NC} $var_name: NOT SET" >&2
        VALIDATION_FAILED=1
        return 1
    else
        echo -e "${GREEN}✓${NC} $var_name: set"
        return 0
    fi
}

# Function to validate variable format
validate_format() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    local validation_regex="$2"
    local error_message="$3"

    if [[ -z "$var_value" ]]; then
        # Already handled by check_required_var
        return 0
    fi

    if [[ ! "$var_value" =~ $validation_regex ]]; then
        INVALID_VARS+=("$var_name: $error_message")
        echo -e "${RED}✗${NC} $var_name: $error_message" >&2
        VALIDATION_FAILED=1
        return 1
    fi

    return 0
}

echo "================================================================"
echo "Environment Variable Validation"
echo "Market Data Kafka Producer - Phase 5 Deployment"
echo "================================================================"
echo ""

# ==============================================================================
# Kafka Connection Configuration
# ==============================================================================
echo "Kafka Connection Configuration"
echo "--------------------------------------------------------------"

check_required_var "KAFKA_BOOTSTRAP_SERVERS"
validate_format "KAFKA_BOOTSTRAP_SERVERS" ":[0-9]+" "must include port numbers (e.g., kafka1:9092,kafka2:9092)"

echo ""

# ==============================================================================
# SASL/SSL Security Configuration
# ==============================================================================
echo "SASL/SSL Security Configuration"
echo "--------------------------------------------------------------"

check_required_var "KAFKA_SASL_USERNAME"
check_required_var "KAFKA_SASL_PASSWORD"
check_required_var "KAFKA_SSL_CERT"
check_required_var "KAFKA_SSL_KEY"
check_required_var "KAFKA_SSL_CA"

echo ""

# ==============================================================================
# Vault Configuration (Secret Management)
# ==============================================================================
echo "Vault Configuration"
echo "--------------------------------------------------------------"

check_required_var "VAULT_ADDR"
validate_format "VAULT_ADDR" "^https://" "must use HTTPS (e.g., https://vault.internal:8200)"

check_required_var "VAULT_TOKEN"

echo ""

# ==============================================================================
# Metrics & Monitoring Configuration
# ==============================================================================
echo "Metrics & Monitoring Configuration"
echo "--------------------------------------------------------------"

check_required_var "METRICS_AUTH_USER"
check_required_var "METRICS_AUTH_PASSWORD"
check_required_var "PROMETHEUS_URL"
check_required_var "GRAFANA_URL"
check_required_var "GRAFANA_API_KEY"

echo ""

# ==============================================================================
# Deployment Environment
# ==============================================================================
echo "Deployment Environment"
echo "--------------------------------------------------------------"

check_required_var "ENVIRONMENT"
validate_format "ENVIRONMENT" "^(production|staging|development)$" "must be one of: production, staging, development"

echo ""

# ==============================================================================
# Validation Summary
# ==============================================================================
echo "================================================================"
echo "Validation Summary"
echo "================================================================"

if [[ $VALIDATION_FAILED -eq 0 ]]; then
    echo -e "${GREEN}✓ All required environment variables validated${NC}"
    echo ""
    echo "Validated Variables:"
    echo "  - KAFKA_BOOTSTRAP_SERVERS: ${KAFKA_BOOTSTRAP_SERVERS}"
    echo "  - KAFKA_SASL_USERNAME: ${KAFKA_SASL_USERNAME}"
    echo "  - KAFKA_SASL_PASSWORD: ********"
    echo "  - KAFKA_SSL_CERT: ${KAFKA_SSL_CERT}"
    echo "  - KAFKA_SSL_KEY: ${KAFKA_SSL_KEY}"
    echo "  - KAFKA_SSL_CA: ${KAFKA_SSL_CA}"
    echo "  - VAULT_ADDR: ${VAULT_ADDR}"
    echo "  - VAULT_TOKEN: ********"
    echo "  - METRICS_AUTH_USER: ${METRICS_AUTH_USER}"
    echo "  - METRICS_AUTH_PASSWORD: ********"
    echo "  - PROMETHEUS_URL: ${PROMETHEUS_URL}"
    echo "  - GRAFANA_URL: ${GRAFANA_URL}"
    echo "  - GRAFANA_API_KEY: ********"
    echo "  - ENVIRONMENT: ${ENVIRONMENT}"
    echo ""
    echo -e "${GREEN}Ready for deployment${NC}"
    exit 0
else
    echo -e "${RED}✗ Validation failed${NC}" >&2
    echo "" >&2

    if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
        echo "Missing Variables:" >&2
        for var in "${MISSING_VARS[@]}"; do
            echo "  - $var" >&2
        done
        echo "" >&2
    fi

    if [[ ${#INVALID_VARS[@]} -gt 0 ]]; then
        echo "Invalid Variables:" >&2
        for var in "${INVALID_VARS[@]}"; do
            echo "  - $var" >&2
        done
        echo "" >&2
    fi

    echo "Action Required:" >&2
    echo "  1. Copy .env.production.template to .env.production" >&2
    echo "  2. Fill in actual values for all required variables" >&2
    echo "  3. Source the file: source .env.production" >&2
    echo "  4. Run validation again: ./scripts/validate-environment.sh" >&2
    echo "" >&2

    exit 1
fi

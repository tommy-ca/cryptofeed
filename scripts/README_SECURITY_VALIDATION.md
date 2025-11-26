# Security Validation Checklist (Task 19.2)

## Overview

Week 0 security validation script that validates all security prerequisites before Phase 5 Week 1 deployment.

**Addresses**: CRIT-1 blocker from multi-agent review (2025-11-26)
**Risk Impact**: Prevents production security incidents (exposed credentials, unencrypted connections)

## Files Created

1. **`scripts/validate-security-prerequisites.sh`** (400 LOC)
   - Security validation script with 6 comprehensive checks
   - Exit codes: 0 (success), 1 (failure)
   - Modes: normal, dry-run, verbose, single-check

2. **`.env.production.template`** (122 LOC)
   - Production environment variable template
   - Documents all required security configurations
   - Usage instructions and examples

3. **`tests/unit/test_security_validation.py`** (367 LOC)
   - 22 comprehensive unit tests
   - Tests all 6 security checks
   - Tests error conditions and edge cases

## Security Checks

### 1. SASL/SSL Certificates
- Validates `KAFKA_SSL_CERT`, `KAFKA_SSL_KEY`, `KAFKA_SSL_CA` environment variables
- Checks certificate files exist at specified paths
- Ensures all certificates are present before deployment

### 2. Service Accounts
- Validates `KAFKA_SASL_USERNAME` and `KAFKA_SASL_PASSWORD` are set
- Verifies username follows `cryptofeed-*` pattern (minimal permissions)
- Ensures service account credentials are configured

### 3. Vault Secrets
- Validates `VAULT_ADDR` and `VAULT_TOKEN` are set
- Ensures Vault address uses HTTPS (not HTTP)
- Confirms secrets are stored in vault (not environment variables)

### 4. TLS Enabled
- Validates TLS configuration via SSL certificate presence
- Warns if Kafka uses plaintext port 9092 (expected: 9093 for TLS)
- Ensures all Kafka connections are encrypted

### 5. Metrics Authentication
- Validates `METRICS_AUTH_USER` and `METRICS_AUTH_PASSWORD` are set
- Ensures metrics endpoints are protected by authentication
- Prevents unauthorized access to Prometheus metrics

### 6. Network Policies
- Checks for Kubernetes network policies (if kubectl available)
- Validates no public internet access to Kafka cluster
- Skips check gracefully if Kubernetes API not accessible

## Usage

### Run All Checks

```bash
./scripts/validate-security-prerequisites.sh
```

**Expected Output (Success)**:
```
=====================================
 Security Validation Checklist
=====================================

[SUCCESS] SASL/SSL certificates validated
[SUCCESS] Service accounts validated
[SUCCESS] Vault secrets validated
[SUCCESS] TLS enabled for Kafka connections
[SUCCESS] Metrics endpoints protected
[SUCCESS] Network policies configured

======================================
 Security Validation Summary
======================================
Checks passed: 6/6
Checks failed: 0/6
======================================

[SUCCESS] All security checks passed
Ready for Week 1 deployment
```

**Expected Output (Failure)**:
```
=====================================
 Security Validation Checklist
=====================================

[ERROR] KAFKA_SSL_CERT not set
[ERROR] KAFKA_SASL_USERNAME not set
[ERROR] VAULT_ADDR not set

======================================
 Security Validation Summary
======================================
Checks passed: 0/6
Checks failed: 6/6
======================================

[ERROR] Security validation failed
Fix all errors before proceeding to Week 1 deployment
```

### Run Specific Check

```bash
./scripts/validate-security-prerequisites.sh --check sasl-ssl
./scripts/validate-security-prerequisites.sh --check service-accounts
./scripts/validate-security-prerequisites.sh --check vault-secrets
./scripts/validate-security-prerequisites.sh --check tls-enabled
./scripts/validate-security-prerequisites.sh --check metrics-auth
./scripts/validate-security-prerequisites.sh --check network-policies
```

### Dry-Run Mode

```bash
./scripts/validate-security-prerequisites.sh --dry-run
```

Shows what checks would run without actually validating.

### Verbose Mode

```bash
./scripts/validate-security-prerequisites.sh --verbose
```

Shows detailed output for each check.

### Help

```bash
./scripts/validate-security-prerequisites.sh --help
```

## Environment Setup

### Step 1: Copy Template

```bash
cp .env.production.template .env.production
```

### Step 2: Fill in Values

Edit `.env.production` and set all required variables:

```bash
# Example values (DO NOT commit actual values)
KAFKA_BOOTSTRAP_SERVERS="kafka1.internal:9093,kafka2.internal:9093,kafka3.internal:9093"
KAFKA_SASL_USERNAME="cryptofeed-prod"
KAFKA_SASL_PASSWORD="<vault-secret>"
KAFKA_SSL_CERT="/etc/ssl/certs/kafka-client.pem"
KAFKA_SSL_KEY="/etc/ssl/private/kafka-client-key.pem"
KAFKA_SSL_CA="/etc/ssl/certs/kafka-ca.pem"
VAULT_ADDR="https://vault.internal:8200"
VAULT_TOKEN="hvs.prod-token"
METRICS_AUTH_USER="prometheus"
METRICS_AUTH_PASSWORD="<vault-secret>"
```

### Step 3: Source Environment

```bash
source .env.production
```

### Step 4: Run Validation

```bash
./scripts/validate-security-prerequisites.sh
```

## Testing

### Run Unit Tests

```bash
python -m pytest tests/unit/test_security_validation.py -v
```

**Test Coverage**:
- 22 unit tests
- 100% pass rate
- All 6 security checks validated
- Error conditions tested
- Edge cases covered

### Test Summary

| Test | Description | Status |
|------|-------------|--------|
| `test_script_exists` | Script file exists | PASS |
| `test_script_executable` | Script has executable permissions | PASS |
| `test_sasl_ssl_certificates_check` | SASL/SSL validation with valid certs | PASS |
| `test_missing_sasl_certificate_fails` | Missing certificates cause failure | PASS |
| `test_service_accounts_check` | Service account validation | PASS |
| `test_missing_service_account_fails` | Missing credentials cause failure | PASS |
| `test_vault_secrets_check` | Vault configuration validation | PASS |
| `test_missing_vault_token_fails` | Missing vault token causes failure | PASS |
| `test_tls_enabled_check` | TLS validation | PASS |
| `test_missing_tls_fails` | Missing TLS config causes failure | PASS |
| `test_metrics_auth_check` | Metrics auth validation | PASS |
| `test_missing_metrics_auth_fails` | Missing metrics auth causes failure | PASS |
| `test_network_policies_check` | Network policies validation | PASS |
| `test_all_checks_pass` | All checks pass with complete config | PASS |
| `test_any_check_failure_exits_nonzero` | Any failure causes non-zero exit | PASS |
| `test_dry_run_mode` | Dry-run mode works | PASS |
| `test_help_flag` | Help flag displays usage | PASS |
| `test_verbose_mode` | Verbose mode shows details | PASS |
| `test_summary_output` | Summary output includes all checks | PASS |
| `test_environment_template_exists` | .env.production.template exists | PASS |
| `test_template_has_required_vars` | Template has all required vars | PASS |
| `test_ci_environment_check` | CI environment validation documented | PASS |

## Integration with Phase 5 Execution

### Pre-Week 1 Checklist

Before executing Phase 5 Week 1 tasks:

1. **Run Security Validation**:
   ```bash
   ./scripts/validate-security-prerequisites.sh
   ```

2. **Verify All Checks Pass**:
   - Expected: `6/6 checks passed`
   - If any check fails, fix the issue before proceeding

3. **Document Results**:
   - Save validation output to `security-validation-$(date +%Y%m%d).log`
   - Include in pre-deployment checklist

### Week 0 Timeline

**Duration**: 2 days (before Week 1)
**Owner**: DevOps + Security

**Day 1**:
- [ ] Generate Kafka SASL/SSL certificates
- [ ] Create service accounts with minimal permissions
- [ ] Configure Vault secrets
- [ ] Enable TLS for all Kafka connections

**Day 2**:
- [ ] Configure metrics endpoint authentication
- [ ] Setup Kubernetes network policies
- [ ] Run security validation script
- [ ] Document any issues and resolutions

## Exit Codes

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | All checks passed | Proceed to Week 1 |
| 1 | One or more checks failed | Fix errors before proceeding |

## Troubleshooting

### Common Issues

**Issue: Certificate not found**
```
[ERROR] SASL/SSL certificate not found: /etc/ssl/certs/kafka-client.pem
```
**Solution**: Verify certificate path and ensure file exists

**Issue: Vault address not HTTPS**
```
[ERROR] VAULT_ADDR must use HTTPS: http://vault.internal:8200
```
**Solution**: Update VAULT_ADDR to use HTTPS protocol

**Issue: Kubernetes API not accessible**
```
[INFO] Kubernetes API not accessible - network policy check skipped
```
**Solution**: Manually verify network policies are configured

## Security Best Practices

1. **Never commit secrets**: Add `.env.production` to `.gitignore`
2. **Use Vault**: Store all secrets in Vault, not environment variables
3. **Rotate credentials**: Rotate SASL passwords and Vault tokens regularly
4. **Principle of least privilege**: Service accounts should have minimal permissions
5. **TLS everywhere**: All Kafka connections must use TLS encryption
6. **Protect metrics**: Metrics endpoints must require authentication

## Related Documentation

- **Phase 5 Execution Plan**: `.kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md`
- **Environment Template**: `.env.production.template`
- **Test Suite**: `tests/unit/test_security_validation.py`

## Success Criteria

- [ ] Script exists and is executable
- [ ] All 6 security checks implemented
- [ ] 22 unit tests passing
- [ ] Environment template created
- [ ] Documentation complete
- [ ] CRIT-1 blocker resolved

## Completion Status

**Task**: 19.2 Security Validation Checklist
**Status**: COMPLETE
**Date**: 2025-11-26
**Deliverables**:
- Security validation script (400 LOC)
- Environment template (122 LOC)
- Unit tests (367 LOC, 22 tests passing)
- Documentation (this file)

**Blockers Resolved**: CRIT-1 (Security validation missing)

# SSRF Prevention Security Runbook

## Overview

This runbook documents the Server-Side Request Forgery (SSRF) vulnerability prevention measures implemented in Cryptofeed's proxy configuration system. It provides operational guidance for security teams, incident response procedures, and monitoring recommendations.

## Vulnerability Summary

### CVE Details

- **Vulnerability Type**: Server-Side Request Forgery (SSRF)
- **CWE Classification**: CWE-918 - Server-Side Request Forgery (SSRF)
- **CVSS Score**: 7.5 High
- **CVSS Vector**: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N
- **Attack Vector**: Network
- **Attack Complexity**: Low
- **Privileges Required**: None
- **User Interaction**: None
- **Impact**: High confidentiality impact (credential theft, internal service access)

### Attack Description

An attacker could manipulate proxy URL configurations to force the Cryptofeed application to make requests to:
1. **Cloud Metadata Services** (AWS, GCP, Azure) to steal credentials
2. **Internal Network Services** (databases, admin panels, APIs)
3. **Local File System** via file:// URIs
4. **Localhost Services** (Redis, PostgreSQL, monitoring systems)

**Pre-Mitigation Risk**: Without URL validation, malicious proxy configurations could lead to:
- AWS IAM credential theft via `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
- GCP service account token theft via `http://metadata.google.internal/computeMetadata/v1/`
- Internal database access via `http://10.0.0.5:5432/`
- Sensitive file reading via `file:///etc/passwd`

## Defense-in-Depth Protection

### Layer 1: Scheme Whitelist

**Purpose**: Prevent non-proxy protocol abuse (file system access, FTP, legacy protocols)

**Allowed Schemes**:
- `http` - HTTP proxy protocol
- `https` - HTTPS proxy protocol
- `socks4` - SOCKS4 proxy protocol
- `socks5` - SOCKS5 proxy protocol
- `socks5h` - SOCKS5 with hostname resolution on proxy side

**Blocked Schemes** (with attack scenarios):
- `file://` - Local file system access
  - Example attack: `file:///etc/passwd` → Read sensitive system files
  - Example attack: `file:///home/user/.ssh/id_rsa` → Steal SSH keys
- `ftp://` - FTP protocol
  - Example attack: `ftp://internal-ftp.company.local/` → Access internal file servers
- `gopher://` - Legacy Gopher protocol
  - Example attack: `gopher://internal-db:6379/_KEYS *` → Redis command injection
- `dict://` - Dictionary protocol
  - Example attack: `dict://internal-service:11211/` → Memcached access
- `ldap://` - LDAP protocol
  - Example attack: `ldap://domain-controller:389/` → Active Directory enumeration

**Implementation**: `cryptofeed/run.py` - `validate_proxy_url()` function

### Layer 2: IP Range Validation

**Purpose**: Prevent access to private networks, loopback, and cloud metadata endpoints

**Blocked IP Ranges**:

1. **Private Class A** (`10.0.0.0/8`)
   - Attack scenario: Access internal database at `10.0.0.5:5432`
   - Impact: Unauthorized database queries, data exfiltration

2. **Private Class B** (`172.16.0.0/12`)
   - Attack scenario: Access Docker containers at `172.17.0.2:8080`
   - Impact: Container escape, service enumeration

3. **Private Class C** (`192.168.0.0/16`)
   - Attack scenario: Access home router admin panel at `192.168.1.1`
   - Impact: Network configuration disclosure, credential theft

4. **Loopback IPv4** (`127.0.0.0/8`)
   - Attack scenario: Access local Redis at `127.0.0.1:6379`
   - Impact: Cache poisoning, data manipulation

5. **Link-Local + Metadata** (`169.254.0.0/16`)
   - Attack scenario: AWS metadata at `169.254.169.254`
   - Impact: **CRITICAL** - IAM credential theft, full account compromise
   - Attack scenario: Azure metadata at `169.254.169.254`
   - Impact: Managed identity token theft

6. **Loopback IPv6** (`::1/128`)
   - Attack scenario: Access local services via IPv6 loopback
   - Impact: Same as IPv4 loopback

7. **Link-Local IPv6** (`fe80::/10`)
   - Attack scenario: Access IPv6 link-local services
   - Impact: Internal service enumeration

**Cloud Metadata Endpoint Details**:

- **AWS EC2**: `http://169.254.169.254/latest/meta-data/`
  - Exposed data: IAM credentials, user data, instance metadata
  - Attack: `curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>`
  - Result: Temporary AWS credentials (AccessKeyId, SecretAccessKey, SessionToken)

- **GCP Compute Engine**: `http://metadata.google.internal/computeMetadata/v1/`
  - Exposed data: Service account tokens, project metadata
  - Attack: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`
  - Result: OAuth 2.0 access token for GCP APIs

- **Azure VM**: `http://169.254.169.254/metadata/identity/oauth2/token`
  - Exposed data: Managed identity access tokens
  - Attack: `curl -H "Metadata: true" http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/`
  - Result: Azure Resource Manager access token

**Implementation**: `cryptofeed/run.py` - IP address parsing with `ipaddress` module

### Layer 3: Hostname Pattern Matching

**Purpose**: Block localhost variants and known metadata hostnames

**Blocked Hostnames**:
- `localhost` - Standard localhost hostname
- `127.0.0.1` - IPv4 loopback address
- `::1` - IPv6 loopback address
- `metadata.google.internal` - GCP metadata service DNS alias

**Attack Scenarios**:
- `http://localhost:6379/` → Access local Redis instance
- `http://localhost:5432/` → Access local PostgreSQL database
- `http://localhost:9200/` → Access local Elasticsearch cluster
- `http://metadata.google.internal/computeMetadata/v1/` → GCP credential theft

**Implementation**: `cryptofeed/run.py` - Case-insensitive hostname matching

## Validation Implementation

### Code Location

**Primary Implementation**:
- File: `cryptofeed/run.py`
- Functions:
  - `validate_proxy_url(url: str) -> None` - Main validation function
  - `load_proxy_mapping(path: str) -> Optional[Dict[str, Any]]` - Config loader with validation

**Constants**:
```python
ALLOWED_PROXY_SCHEMES = {'http', 'https', 'socks4', 'socks5', 'socks5h'}

BLOCKED_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fe80::/10'),
]

BLOCKED_HOSTNAMES = {
    'localhost',
    '127.0.0.1',
    '::1',
    'metadata.google.internal',
}
```

### Validation Flow

```
Proxy URL Configuration
         |
         v
    Parse URL (urllib.parse.urlparse)
         |
         v
    Layer 1: Scheme in whitelist?
         |
         +-- No --> Raise ValueError("Invalid scheme")
         |
         v (Yes)
    Layer 2: Hostname is IP address?
         |
         +-- Yes --> IP in blocked ranges?
         |              |
         |              +-- Yes --> Raise ValueError("Blocked IP range")
         |              |
         |              v (No)
         v
    Layer 3: Hostname in blocked patterns?
         |
         +-- Yes --> Raise ValueError("Blocked hostname")
         |
         v (No)
    Configuration Accepted
```

## Test Coverage

### Unit Tests

**File**: `tests/unit/test_ssrf_validator.py`

**Coverage** (34 test cases):

1. **Scheme Validation** (8 tests):
   - Valid schemes: http, https, socks4, socks5, socks5h
   - Invalid schemes: file, ftp, gopher
   - Empty URL handling

2. **IP Range Validation** (12 tests):
   - Private IPs: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
   - Loopback: 127.0.0.1, ::1
   - Link-local: 169.254.169.254
   - IPv6 link-local: fe80::
   - Public IPs: 8.8.8.8, 1.1.1.1 (pass validation)

3. **Hostname Validation** (8 tests):
   - Blocked: localhost, 127.0.0.1, ::1, metadata.google.internal
   - Case-insensitive: Localhost, LOCALHOST
   - Valid: proxy.example.com

4. **Edge Cases** (6 tests):
   - URL encoding bypass attempts
   - DNS rebinding scenarios
   - Malformed URLs
   - Empty hostname handling

### Integration Tests

**File**: `tests/integration/test_ssrf_proxy_config.py`

**Coverage** (11 test cases):

1. **Configuration Loading** (5 tests):
   - Malicious global proxy rejection
   - Malicious per-exchange proxy rejection
   - Mixed valid/invalid configuration
   - Error message section paths
   - Valid configuration acceptance

2. **End-to-End Validation** (6 tests):
   - File system access prevention
   - Metadata endpoint blocking
   - Internal network access prevention
   - Localhost service blocking
   - Multi-layer validation interaction
   - Configuration precedence (env vars, YAML, programmatic)

### Test Execution

```bash
# Run all SSRF tests
pytest tests/unit/test_ssrf_validator.py tests/integration/test_ssrf_proxy_config.py -v

# Run specific test category
pytest tests/unit/test_ssrf_validator.py -k "scheme" -v

# Generate coverage report
pytest tests/unit/test_ssrf_validator.py --cov=cryptofeed.run --cov-report=html
```

**Expected Results**: 45/45 tests passing (34 unit + 11 integration)

## Incident Response Procedures

### Detection

**Indicators of SSRF Attempt**:

1. **Application Logs** - ValueError with SSRF-related messages:
   ```
   ERROR: Invalid proxy URL in global.http: Proxy URL points to blocked IP range: 169.254.169.254 (matches 169.254.0.0/16, SSRF prevention)
   ERROR: Invalid proxy URL in exchanges.binance.socks5: Invalid proxy scheme 'file'. Allowed schemes: http, https, socks4, socks5, socks5h
   ```

2. **Configuration Changes** - Unexpected proxy configurations in:
   - `config/proxy.yaml`
   - Environment variables (`CRYPTOFEED_PROXY_*`)
   - Programmatic settings in application code

3. **Network Monitoring** - Anomalous traffic patterns:
   - Outbound connections to 169.254.169.254
   - Requests to internal IP ranges (10.x, 192.168.x, 172.16-31.x)
   - Local port scanning (127.0.0.1:1-65535)

### Response Steps

**Priority: HIGH (within 1 hour)**

**Step 1: Immediate Containment** (0-15 minutes)
1. Verify validation is active: Check logs for "SSRF prevention" messages
2. Identify attack source: Review recent configuration changes
3. Block attacker access: Revoke API keys, rotate credentials if compromised
4. Isolate affected systems: Disconnect from network if credential theft suspected

**Step 2: Investigation** (15-45 minutes)
1. Audit proxy configurations:
   ```bash
   grep -r "CRYPTOFEED_PROXY" /etc/environment
   cat config/proxy.yaml
   git log --all --oneline -- config/proxy.yaml
   ```

2. Check for successful SSRF exploitation:
   ```bash
   # AWS credential theft check
   grep "169.254.169.254" /var/log/nginx/access.log
   grep "iam/security-credentials" /var/log/application.log

   # GCP metadata access check
   grep "metadata.google.internal" /var/log/nginx/access.log
   ```

3. Review cloud audit logs:
   - AWS CloudTrail: Unusual API calls from compromised credentials
   - GCP Cloud Logging: Metadata server access from compute instances
   - Azure Activity Log: Unexpected resource access

4. Scan for lateral movement:
   ```bash
   # Check for internal network scanning
   netstat -antp | grep ESTABLISHED | grep -E "(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)"
   ```

**Step 3: Remediation** (45-60 minutes)
1. Rotate all credentials:
   - AWS IAM credentials (if metadata accessed)
   - GCP service account keys
   - Exchange API keys
   - Proxy authentication credentials

2. Update security groups/firewall rules:
   ```bash
   # Block metadata endpoint at network level
   iptables -A OUTPUT -d 169.254.169.254 -j DROP
   ```

3. Review and harden configurations:
   - Remove suspicious proxy entries
   - Enable verbose logging for proxy validation
   - Add alerts for configuration changes

4. Apply patches/updates:
   - Verify Cryptofeed is running latest version with SSRF fixes
   - Update dependencies: `pip install --upgrade cryptofeed`

**Step 4: Post-Incident Review** (within 24 hours)
1. Document timeline of events
2. Identify root cause (misconfiguration, insider threat, external attack)
3. Assess damage (data accessed, credentials stolen, systems compromised)
4. Update incident response playbook with lessons learned
5. Schedule security training for team

### Escalation Procedures

**Escalate to Security Team if**:
- Metadata endpoint access detected in logs
- Cloud credentials confirmed stolen
- Internal database/service access detected
- Multiple failed SSRF attempts within short timeframe
- Configuration changes from unknown source

**Escalation Contacts**:
- Security Team: security@company.com
- Cloud Security: cloud-security@company.com
- On-Call Engineer: PagerDuty/Opsgenie alert

## Monitoring and Alerting

### Metrics to Track

**1. Proxy Validation Rejections**

Track rejection rates by reason category:

```python
# Prometheus metrics (conceptual)
proxy_validation_rejected_total{reason="private_ip"} = 3
proxy_validation_rejected_total{reason="blocked_scheme"} = 1
proxy_validation_rejected_total{reason="blocked_hostname"} = 0
proxy_validation_rejected_total{reason="metadata_endpoint"} = 0
```

**2. Configuration Changes**

Monitor proxy configuration modifications:

```python
proxy_config_changes_total{source="yaml"} = 5
proxy_config_changes_total{source="env_var"} = 12
proxy_config_changes_total{source="programmatic"} = 0
```

**3. Validation Performance**

Track validation latency:

```python
proxy_validation_duration_seconds{quantile="0.5"} = 0.0008
proxy_validation_duration_seconds{quantile="0.95"} = 0.0012
proxy_validation_duration_seconds{quantile="0.99"} = 0.0015
```

### Alert Rules

**Critical Alerts** (immediate response):

1. **Metadata Endpoint Attempt**
   ```
   proxy_validation_rejected_total{reason="metadata_endpoint"} > 0
   ```
   Action: Immediate security team notification, initiate incident response

2. **Multiple Rejection Spike**
   ```
   rate(proxy_validation_rejected_total[5m]) > 10
   ```
   Action: Investigate for automated attack, review recent config changes

3. **File Scheme Detection**
   ```
   proxy_validation_rejected_total{reason="file_scheme"} > 0
   ```
   Action: High-priority investigation, audit configuration sources

**Warning Alerts** (review within 1 hour):

4. **Private IP Configuration**
   ```
   proxy_validation_rejected_total{reason="private_ip"} > 5
   ```
   Action: Review configuration, educate team on valid proxy requirements

5. **Localhost Detection**
   ```
   proxy_validation_rejected_total{reason="localhost"} > 0
   ```
   Action: Check for development environment leaks into production config

6. **Validation Performance Degradation**
   ```
   proxy_validation_duration_seconds{quantile="0.99"} > 0.005
   ```
   Action: Performance investigation, check for DNS resolution delays

### Monitoring Implementation

**Example Logging**:

```python
# cryptofeed/run.py - validation logging
import logging

logger = logging.getLogger(__name__)

def validate_proxy_url(url: str) -> None:
    try:
        # ... validation logic ...
        logger.debug(f"Proxy URL validated: {url} (scheme: {parsed.scheme})")
    except ValueError as e:
        logger.warning(
            f"SSRF prevention: Blocked proxy URL",
            extra={
                "url": url,
                "reason": str(e),
                "blocked_category": categorize_error(e),
                "source": "proxy_validation"
            }
        )
        raise
```

**Example Alert Configuration** (Prometheus Alertmanager):

```yaml
groups:
  - name: ssrf_prevention
    interval: 1m
    rules:
      - alert: SSRFMetadataEndpointAttempt
        expr: proxy_validation_rejected_total{reason="metadata_endpoint"} > 0
        for: 0s
        labels:
          severity: critical
          team: security
        annotations:
          summary: "SSRF attack attempt on cloud metadata endpoint"
          description: "Blocked proxy URL targeting {{ $labels.reason }}. Immediate investigation required."

      - alert: SSRFMultipleRejections
        expr: rate(proxy_validation_rejected_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          team: security
        annotations:
          summary: "Multiple SSRF validation rejections detected"
          description: "{{ $value }} rejections per second over 5 minutes. Possible attack in progress."
```

### Security Log Review

**Daily Review Checklist**:

- [ ] Check for any `proxy_validation_rejected_total` metrics > 0
- [ ] Review `ValueError` logs related to proxy configuration
- [ ] Audit recent changes to `config/proxy.yaml`
- [ ] Verify no anomalous network traffic to 169.254.0.0/16
- [ ] Confirm cloud audit logs show no unexpected credential usage

**Weekly Review Checklist**:

- [ ] Trend analysis of rejection reasons (shifting attack patterns?)
- [ ] Review all configuration change sources (git commits, env var updates)
- [ ] Test validation with known malicious URLs (penetration testing)
- [ ] Update blocked hostname list with new cloud metadata endpoints
- [ ] Review and update incident response procedures

**Monthly Security Assessment**:

- [ ] Penetration test with SSRF attack vectors
- [ ] Review and update OWASP SSRF prevention compliance
- [ ] Audit test coverage for new attack patterns
- [ ] Team training on SSRF prevention and detection
- [ ] Review and update this runbook

## Security Audit Checklist

### Configuration Audit

- [ ] Verify `validate_proxy_url()` is called for all proxy configurations
- [ ] Confirm no hardcoded proxy URLs bypass validation
- [ ] Check environment variables use `CRYPTOFEED_PROXY_*` namespace
- [ ] Verify YAML configuration loads through `load_proxy_mapping()`
- [ ] Test programmatic configuration uses `ProxySettings` with validation

### Code Audit

- [ ] Review `ALLOWED_PROXY_SCHEMES` for completeness
- [ ] Verify `BLOCKED_IP_RANGES` covers all RFC 1918 ranges
- [ ] Confirm `BLOCKED_HOSTNAMES` includes known metadata endpoints
- [ ] Check for validation bypass paths in code
- [ ] Verify error messages do not leak sensitive information

### Testing Audit

- [ ] All 34 unit tests pass (scheme, IP, hostname validation)
- [ ] All 11 integration tests pass (config loading, E2E validation)
- [ ] Coverage report shows >95% for validation code
- [ ] Penetration testing confirms no bypass vectors
- [ ] Fuzzing tests cover edge cases (URL encoding, DNS rebinding)

### Operational Audit

- [ ] Monitoring alerts configured and tested
- [ ] Incident response procedures documented and rehearsed
- [ ] Security team trained on SSRF detection and response
- [ ] Logging captures all validation events
- [ ] Audit logs retained for compliance period (e.g., 90 days)

## References

### OWASP Resources

- [SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
- [OWASP SSRF Attack Overview](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
- [OWASP Testing Guide: SSRF](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/19-Testing_for_Server-Side_Request_Forgery)

### CWE References

- [CWE-918: Server-Side Request Forgery (SSRF)](https://cwe.mitre.org/data/definitions/918.html)
- [CWE-73: External Control of File Name or Path](https://cwe.mitre.org/data/definitions/73.html)

### Cloud Metadata Endpoint Documentation

- [AWS EC2 Instance Metadata](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html)
- [GCP Compute Engine Metadata](https://cloud.google.com/compute/docs/metadata/overview)
- [Azure VM Instance Metadata Service](https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service)

### RFC Standards

- [RFC 1918: Address Allocation for Private Internets](https://datatracker.ietf.org/doc/html/rfc1918)
- [RFC 3927: Dynamic Configuration of IPv4 Link-Local Addresses](https://datatracker.ietf.org/doc/html/rfc3927)
- [RFC 4291: IP Version 6 Addressing Architecture](https://datatracker.ietf.org/doc/html/rfc4291)

### Internal Documentation

- [Proxy User Guide](../proxy/user-guide.md) - SSRF Prevention section
- [Proxy Technical Specification](../proxy/technical-specification.md)
- [SSRF Validation Unit Tests](../../tests/unit/test_ssrf_validator.py)
- [SSRF Validation Integration Tests](../../tests/integration/test_ssrf_proxy_config.py)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-14
**Next Review Date**: 2026-01-14
**Document Owner**: Security Team
**Classification**: Internal Use Only

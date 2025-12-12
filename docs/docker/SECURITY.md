# Security Guide for Cryptofeed Container Images

This document provides security guidelines for building, scanning, and remediating vulnerabilities in cryptofeed container images.

## Security Scanning with Trivy

Cryptofeed uses [Trivy](https://github.com/aquasecurity/trivy) to scan container images for CVE vulnerabilities before deployment.

### Running Security Scans

**Scan local image:**
```bash
./scripts/trivy-scan.sh --image cryptofeed:latest
```

**Scan with custom report path:**
```bash
./scripts/trivy-scan.sh --image cryptofeed:v1.2.3 --report /tmp/scan-results.json
```

**Table format for human-readable output:**
```bash
./scripts/trivy-scan.sh --image cryptofeed:latest --format table
```

**Scan all severity levels:**
```bash
./scripts/trivy-scan.sh --image cryptofeed:latest --severity CRITICAL,HIGH,MEDIUM,LOW
```

### Severity Levels

The scan fails the build if **CRITICAL** or **HIGH** severity vulnerabilities are detected:

- **CRITICAL**: Exploitable vulnerabilities with high impact (RCE, privilege escalation, etc.)
- **HIGH**: Significant vulnerabilities requiring immediate attention
- **MEDIUM**: Moderate vulnerabilities (reported but not build-blocking)
- **LOW**: Minor vulnerabilities (informational)

### Scan Report Format

Trivy generates a JSON report with the following structure:

```json
{
  "SchemaVersion": 2,
  "ArtifactName": "cryptofeed:latest",
  "ArtifactType": "container_image",
  "Metadata": {
    "ImageID": "sha256:abc123...",
    "RepoTags": ["cryptofeed:latest"]
  },
  "Results": [
    {
      "Target": "python:3.11-slim-bookworm",
      "Class": "os-pkgs",
      "Type": "debian",
      "Vulnerabilities": [
        {
          "VulnerabilityID": "CVE-2023-12345",
          "PkgName": "libssl3",
          "InstalledVersion": "3.0.11-1",
          "FixedVersion": "3.0.12-1",
          "Severity": "CRITICAL",
          "Description": "Critical vulnerability in OpenSSL",
          "PrimaryURL": "https://avd.aquasec.com/nvd/cve-2023-12345"
        }
      ]
    }
  ]
}
```

## CVE Remediation Process

When Trivy detects CRITICAL or HIGH vulnerabilities, follow this remediation workflow:

### Step 1: Review Vulnerability Report

```bash
# Generate scan report
./scripts/trivy-scan.sh --image cryptofeed:latest --report trivy-report.json

# View summary (requires jq)
jq '.Results[].Vulnerabilities[] | select(.Severity == "CRITICAL" or .Severity == "HIGH") | {CVE: .VulnerabilityID, Package: .PkgName, InstalledVersion, FixedVersion, Severity}' trivy-report.json
```

### Step 2: Identify Vulnerability Source

Vulnerabilities typically originate from:

1. **Base Image** (python:3.11-slim-bookworm)
   - OS packages (Debian packages)
   - System libraries (libssl, libcurl, etc.)

2. **Python Dependencies** (requirements.txt)
   - Python packages installed via pip
   - Native extensions with C/C++ dependencies

### Step 3: Update Base Image

**For OS-level vulnerabilities**, update the base image to the latest patch version:

**Current:**
```dockerfile
FROM python:3.11-slim-bookworm AS builder
```

**Updated (pinned patch version):**
```dockerfile
FROM python:3.11.7-slim-bookworm AS builder
```

**Check available Python base images:**
```bash
docker pull python:3.11-slim-bookworm
docker images | grep python
```

### Step 4: Update Python Dependencies

**For Python package vulnerabilities**, update dependencies in `requirements.txt`:

```bash
# Update specific package
pip install --upgrade <package-name>
pip freeze | grep <package-name> >> requirements.txt

# Or update all dependencies (use with caution)
pip list --outdated
```

**Example: Update cryptography package:**
```bash
# Before (vulnerable)
cryptography==41.0.0

# After (patched)
cryptography==42.0.0
```

### Step 5: Rebuild and Re-scan

```bash
# Rebuild Docker image
docker build -t cryptofeed:latest .

# Re-run security scan
./scripts/trivy-scan.sh --image cryptofeed:latest

# Verify vulnerabilities resolved
./scripts/trivy-scan.sh --image cryptofeed:latest --format table
```

### Step 6: Commit Changes

```bash
git add Dockerfile requirements.txt
git commit -m "fix(security): update base image and dependencies to resolve CVE-XXXX-XXXXX"
git push
```

## Base Image Version Pinning Strategy

### Pinning Recommendations

**Development and Staging:**
```dockerfile
# Use latest minor version (receives security patches automatically)
FROM python:3.11-slim-bookworm
```

**Production:**
```dockerfile
# Pin to specific patch version for reproducibility
FROM python:3.11.7-slim-bookworm
```

### When to Update Base Image

- **Security patches**: Update immediately for CRITICAL/HIGH CVEs
- **Minor version updates**: Update quarterly or when LTS support ends
- **Major version updates**: Plan migration (e.g., Python 3.11 → 3.12)

### Tracking Base Image Updates

Subscribe to security advisories:
- [Python Docker Official Images](https://hub.docker.com/_/python)
- [Debian Security Tracker](https://security-tracker.debian.org/)
- [CVE Database](https://cve.mitre.org/)

## Vulnerability Exception Process

In rare cases, vulnerabilities may not be immediately fixable (e.g., no upstream patch available). Use Trivy's `.trivyignore` file to document exceptions.

### Creating Vulnerability Exceptions

**1. Create `.trivyignore` file:**
```bash
# .trivyignore (in repository root)
# CVE-2023-12345: OpenSSL vulnerability - no patch available, low exploitability in container context
CVE-2023-12345

# CVE-2023-67890: curl vulnerability - waiting for upstream Debian patch (ETA: 2025-12-20)
CVE-2023-67890
```

**2. Document exception rationale:**
- CVE identifier
- Affected package
- Reason for exception (no patch, low exploitability, etc.)
- Expected resolution date (if applicable)

**3. Approval process:**
- Security team review required
- Exceptions expire after 30 days
- Re-scan weekly to check for patches

### Exploitability Assessment

Not all CVEs are exploitable in containerized environments. Consider:
- **Attack vector**: Network vs. local
- **Container isolation**: Does seccomp/AppArmor mitigate?
- **Application usage**: Is vulnerable code path reachable?

**Example:**
```
CVE-2023-12345 (CRITICAL): OpenSSL TLS handshake vulnerability
Assessment: cryptofeed does not use TLS client authentication, vulnerable code path not reachable
Exception approved: YES (30 days)
```

## Security Best Practices

### Container Hardening

**1. Non-root user:**
```dockerfile
# Dockerfile already implements this
USER cryptofeed  # UID 1001
```

**2. Read-only root filesystem:**
```dockerfile
# Deployment manifest
securityContext:
  readOnlyRootFilesystem: true
```

**3. Drop all capabilities:**
```dockerfile
# Deployment manifest
securityContext:
  capabilities:
    drop: ["ALL"]
```

### Secret Management

**Never include secrets in Docker images:**
```dockerfile
# BAD - Do NOT do this
ENV BINANCE_API_KEY=abc123

# GOOD - Use Kubernetes Secrets
env:
  - name: BINANCE_API_KEY
    valueFrom:
      secretKeyRef:
        name: cryptofeed-api-keys
        key: BINANCE_API_KEY
```

### Build-time Security

**1. Multi-stage builds:**
- Builder stage: Install build dependencies
- Runtime stage: Copy only runtime artifacts

**2. Layer caching:**
- Separate dependency installation from code changes
- Minimize layer size

**3. .dockerignore:**
- Exclude .git, .env, tests, docs
- Prevent accidental secret inclusion

## Continuous Security Scanning

### CI/CD Integration

**GitHub Actions example:**
```yaml
- name: Build Docker image
  run: docker build -t cryptofeed:${{ github.sha }} .

- name: Run Trivy scan
  run: |
    ./scripts/trivy-scan.sh --image cryptofeed:${{ github.sha }} \
      --report trivy-report.json

- name: Upload scan report
  uses: actions/upload-artifact@v3
  with:
    name: trivy-report
    path: trivy-report.json
```

### Scheduled Scans

Run weekly scans against production images to detect newly disclosed vulnerabilities:

```bash
# Cron job (every Sunday at 2 AM)
0 2 * * 0 /path/to/trivy-scan.sh --image cryptofeed:production --report /var/log/trivy/weekly-scan.json
```

## Reporting Security Issues

If you discover a security vulnerability in cryptofeed:

1. **Do not** open a public GitHub issue
2. Email security@cryptofeed.example.com with:
   - CVE identifier (if applicable)
   - Affected version(s)
   - Proof of concept (if available)
   - Suggested remediation
3. Allow 48 hours for initial response
4. Coordinate disclosure timeline

## Additional Resources

- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Kubernetes Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
- [NIST National Vulnerability Database](https://nvd.nist.gov/)
- [OWASP Container Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

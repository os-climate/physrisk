# Security Scanning Guide

This guide explains how to generate Software Bill of Materials (SBOM) and perform security
vulnerability scanning locally using the same tools and processes as the CI/CD pipeline.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Understanding Results](#understanding-results)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

This project uses:

- **CycloneDX** for SBOM generation (industry-standard format for dependency tracking)
- **Grype** for vulnerability scanning (Anchore's open-source security scanner)

The local tooling mirrors the CI/CD workflow defined in `.github/workflows/build-test.yaml`.

## Prerequisites

### Required Tools

1. **Python 3.10+** (already required for this project)
2. **cyclonedx-bom** for SBOM generation
3. **Grype** for security scanning

### Installation

#### Quick Install (All Tools)

```bash
make install-tools
```

#### Manual Installation

**Install cyclonedx-bom:**

```bash
pip install cyclonedx-bom
```

**Install Grype:**

On macOS (using Homebrew):

```bash
brew install grype
```

On Linux/macOS (using install script):

```bash
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
```

On Windows:

```powershell
# Using Scoop
scoop install grype

# Or download from: https://github.com/anchore/grype/releases
```

## Quick Start

### Using Make (Recommended)

```bash
# Generate SBOM and run security scan (medium+ severity)
make security-check

# Run scan with high severity threshold only
make grype-high

# Run scan with critical severity only
make grype-critical

# Generate SBOM with all formats and dev dependencies
make security-full
```

### Using the Script Directly

```bash
# Basic scan (fails on medium+ severity)
./scripts/sbom-scan.sh

# Scan with high severity threshold
./scripts/sbom-scan.sh --severity high

# Include development dependencies
./scripts/sbom-scan.sh --include-dev

# Generate SARIF output for CI tools
./scripts/sbom-scan.sh --grype-format sarif
```

## Detailed Usage

### SBOM Generation Only

Generate SBOM without running security scans:

```bash
# CycloneDX format (JSON and XML)
make sbom

# Include development dependencies
make sbom-dev

# Generate both CycloneDX and SPDX formats
make sbom-all
```

Output files are created in `sbom-output/`:

- `sbom-cyclonedx.json` - CycloneDX JSON format
- `sbom-cyclonedx.xml` - CycloneDX XML format

### Security Scanning

#### Severity Levels

Grype supports the following severity levels (from lowest to highest):

- `negligible` - Minimal or no security impact
- `low` - Low security impact
- `medium` - **Default threshold** - Medium security impact
- `high` - High security impact requiring attention
- `critical` - Critical vulnerabilities requiring immediate action

#### Running Scans with Different Thresholds

```bash
# Fail only on critical vulnerabilities
make grype-critical
# or
./scripts/sbom-scan.sh --severity critical

# Fail on high or critical
make grype-high
# or
./scripts/sbom-scan.sh --severity high

# Fail on medium or above (default, matches CI)
make grype-scan
# or
./scripts/sbom-scan.sh --severity medium
```

#### Output Formats

Grype supports multiple output formats:

```bash
# Human-readable table (default)
./scripts/sbom-scan.sh --grype-format table

# JSON for programmatic processing
./scripts/sbom-scan.sh --grype-format json

# SARIF for CI/CD integration (GitHub Code Scanning, etc.)
./scripts/sbom-scan.sh --grype-format sarif

# CycloneDX VEX format
./scripts/sbom-scan.sh --grype-format cyclonedx
```

### Advanced Options

```bash
# Clean previous results and run fresh scan
./scripts/sbom-scan.sh --cleanup

# Custom output directory
./scripts/sbom-scan.sh --output ./my-scan-results

# Combine multiple options
./scripts/sbom-scan.sh \
    --cleanup \
    --format both \
    --include-dev \
    --severity high \
    --grype-format sarif \
    --output ./security-audit
```

### Script Help

For complete usage information:

```bash
./scripts/sbom-scan.sh --help
```

## Understanding Results

### SBOM Output

The SBOM files contain a complete inventory of all dependencies:

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "components": [
    {
      "type": "library",
      "name": "numpy",
      "version": "2.2.0",
      "purl": "pkg:pypi/numpy@2.2.0"
    },
    ...
  ]
}
```

### Grype Scan Results

**Table Format** (default):

```text
NAME       INSTALLED  FIXED-IN  TYPE  VULNERABILITY  SEVERITY
urllib3    2.0.0      2.0.7     pypi  CVE-2023-45803 medium
requests   2.31.0     (won't fix)pypi CVE-2023-xxxxx high
```

**Key Columns:**

- `NAME` - Package name
- `INSTALLED` - Currently installed version
- `FIXED-IN` - Version where vulnerability is fixed
- `VULNERABILITY` - CVE or vulnerability identifier
- `SEVERITY` - Risk level

### Exit Codes

The script returns different exit codes based on scan results:

- `0` - Success: No vulnerabilities at or above the threshold
- `1` - Failure: Vulnerabilities found at or above the threshold
- `>1` - Error: Tool or configuration issue

## CI/CD Integration

### Matching CI Behavior

The local tooling is designed to match the CI/CD pipeline:

```yaml
# From .github/workflows/build-test.yaml
- name: "Security scan with Grype (SARIF)"
  uses: anchore/scan-action@v7.2.3
  with:
    sbom: "${{ steps.sbom.outputs.sbom_json_path }}"
    output-format: "sarif"
    fail-build: "true"
```

Local equivalent:

```bash
make grype-sarif
# or
./scripts/sbom-scan.sh --grype-format sarif
```

### Pre-commit Checks

Add security scanning to your pre-commit workflow:

```bash
# Run all pre-commit checks including security scan
make pre-commit security-check
```

### GitHub Actions Integration

The SARIF output can be uploaded to GitHub Security:

```yaml
- name: Upload SARIF to GitHub Security
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: sbom-output/grype-results.sarif
```

## Troubleshooting

### Common Issues

#### 1. "cyclonedx-py command not found"

```bash
pip install cyclonedx-bom
```

#### 2. "grype command not found"

```bash
# macOS
brew install grype

# Linux/macOS
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
```

#### 3. Scan Fails with Medium Severity Vulnerabilities

This is expected behavior that matches CI/CD. Options:

**Option A:** Fix the vulnerabilities by updating dependencies

```bash
# Update dependencies
pip install --upgrade <package-name>

# Or update all
pip install --upgrade -r requirements.txt
```

**Option B:** Use a different severity threshold for local development

```bash
# Only fail on high/critical
make grype-high
```

**Option C:** Review and accept the risk (document in security policy)

#### 4. False Positives

If you encounter false positives, you can:

1. **Review the vulnerability details:**

   ```bash
   grype sbom:sbom-output/sbom-cyclonedx.json -o json | jq
   ```

2. **Create a `.grype.yaml` configuration** to ignore specific vulnerabilities:

   ```yaml
   ignore:
     - vulnerability: CVE-2023-XXXXX
       reason: "False positive - doesn't affect our usage"
   ```

3. **Update dependencies** to versions without the vulnerability

### Getting Help

- **Grype Documentation:** <https://github.com/anchore/grype>
- **CycloneDX Documentation:** <https://cyclonedx.org/>
- **Report Issues:** <https://github.com/os-climate/physrisk/issues>

## Best Practices

### Regular Scanning

1. **Run locally before committing:**

   ```bash
   make security-check
   ```

2. **Review scan results regularly:**

   ```bash
   # Check for new vulnerabilities weekly
   make security-check
   ```

3. **Keep dependencies updated:**

   ```bash
   pip list --outdated
   ```

### Security Workflow

1. **Before PR:** Run `make security-check`
2. **Review vulnerabilities:** Check severity and fix-in versions
3. **Update dependencies:** Use `pip install --upgrade` for packages with fixes
4. **Document exceptions:** If accepting risk, document why
5. **Monitor CI:** Ensure CI security scans pass

### Dependency Management

- Keep `pyproject.toml` dependencies up to date
- Pin versions for production stability
- Regular security audits (weekly/monthly)
- Subscribe to security advisories for key dependencies

## Additional Resources

- [SBOM Best Practices](https://www.cisa.gov/sbom)
- [Vulnerability Management Guide](https://www.cisa.gov/known-exploited-vulnerabilities)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [NIST Software Supply Chain Security](https://www.nist.gov/itl/executive-order-improving-nations-cybersecurity/software-supply-chain-security-guidance)

## Example Workflow

Complete local security workflow:

```bash
# 1. Clean previous results
make clean

# 2. Install/update dependencies
make install-dev

# 3. Run tests
make test

# 4. Run linting
make lint

# 5. Generate SBOM and run security scan
make security-check

# 6. If vulnerabilities found, review and fix
# ... update dependencies as needed ...

# 7. Re-run scan to verify fixes
make security-check

# 8. Commit and push
git add .
git commit -m "Security: Update dependencies"
git push
```

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

# Script to generate SBOM and perform security scanning matching CI/CD behavior
# This creates a fresh virtual environment to ensure accurate dependency scanning

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./sbom-output"
GRYPE_FAIL_ON="medium"
VENV_DIR="./.venv-sbom-temp"
CLEANUP_VENV="true"
PYTHON_VERSION="python3"

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate SBOM and perform security scanning in a fresh virtual environment
This mimics the CI/CD pipeline behavior exactly.

OPTIONS:
    -o, --output DIR          Output directory for SBOM and scan results (default: ./sbom-output)
    -s, --severity LEVEL      Grype fail-on severity: negligible, low, medium, high, critical (default: medium)
    -k, --keep-venv           Keep the virtual environment after scanning
    -p, --python PYTHON       Python interpreter to use (default: python3)
    -h, --help                Display this help message

EXAMPLES:
    # Basic scan matching CI behavior
    $0

    # Scan with high severity threshold
    $0 --severity high

    # Keep virtual environment for inspection
    $0 --keep-venv

REQUIREMENTS:
    - Python 3.10+ with uv or venv
    - cyclonedx-bom (will be installed in temp venv)
    - grype (for security scanning)

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--severity)
            GRYPE_FAIL_ON="$2"
            shift 2
            ;;
        -k|--keep-venv)
            CLEANUP_VENV="false"
            shift
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CI-Style SBOM Generation and Security Scanning${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for required tools
echo -e "${YELLOW}Checking for required tools...${NC}"

if ! command -v "$PYTHON_VERSION" &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_VERSION not found${NC}"
    exit 1
fi

if ! command -v grype &> /dev/null; then
    echo -e "${RED}Error: grype not found${NC}"
    echo -e "${YELLOW}Install with:${NC}"
    echo -e "${YELLOW}  curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin${NC}"
    echo -e "${YELLOW}  OR: brew install grype${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required tools found${NC}"
echo ""

# Clean up old venv if it exists
if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Cleaning up old virtual environment...${NC}"
    rm -rf "$VENV_DIR"
fi

# Create fresh virtual environment
echo -e "${BLUE}Creating fresh virtual environment...${NC}"
echo -e "  Location: $VENV_DIR"
echo ""

if command -v uv &> /dev/null; then
    echo -e "${YELLOW}Using uv to create virtual environment...${NC}"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    echo -e "${YELLOW}Installing project dependencies with uv...${NC}"
    # Install the project in editable mode to get all dependencies
    uv pip install -e . 2>&1 | grep -v "^Resolved" | grep -v "^Prepared" | grep -v "^Installed" || true
else
    echo -e "${YELLOW}Using Python venv...${NC}"
    "$PYTHON_VERSION" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip --quiet

    echo -e "${YELLOW}Installing project dependencies...${NC}"
    pip install -e . --quiet
fi

echo -e "${GREEN}✓ Virtual environment created and dependencies installed${NC}"
echo ""

# Install cyclonedx-bom in the venv
echo -e "${YELLOW}Installing cyclonedx-bom in virtual environment...${NC}"
if command -v uv &> /dev/null; then
    uv pip install cyclonedx-bom --quiet
else
    pip install cyclonedx-bom --quiet
fi
echo -e "${GREEN}✓ cyclonedx-bom installed${NC}"
echo ""

# Generate pip freeze requirements
echo -e "${YELLOW}Generating requirements from installed packages...${NC}"
TEMP_REQUIREMENTS="$OUTPUT_DIR/temp-requirements.txt"
pip freeze > "$TEMP_REQUIREMENTS"

# Count packages
PACKAGE_COUNT=$(wc -l < "$TEMP_REQUIREMENTS" | tr -d ' ')
echo -e "${GREEN}✓ Found $PACKAGE_COUNT packages installed${NC}"
echo ""

# Generate SBOM
echo -e "${BLUE}Generating CycloneDX SBOM...${NC}"

cyclonedx-py requirements "$TEMP_REQUIREMENTS" \
    --of JSON \
    -o "$OUTPUT_DIR/sbom-cyclonedx.json" \
    --pyproject pyproject.toml 2>&1 | grep -v "UserWarning" || true

cyclonedx-py requirements "$TEMP_REQUIREMENTS" \
    --of XML \
    -o "$OUTPUT_DIR/sbom-cyclonedx.xml" \
    --pyproject pyproject.toml 2>&1 | grep -v "UserWarning" || true

echo -e "${GREEN}✓ CycloneDX SBOM generated:${NC}"
echo -e "  - $OUTPUT_DIR/sbom-cyclonedx.json"
echo -e "  - $OUTPUT_DIR/sbom-cyclonedx.xml"
echo ""

# Deactivate virtual environment before scanning
deactivate

# Run Grype security scan
echo -e "${BLUE}Running Grype security scan...${NC}"
echo -e "  SBOM: $OUTPUT_DIR/sbom-cyclonedx.json"
echo -e "  Fail-on severity: $GRYPE_FAIL_ON"
echo ""

echo -e "${YELLOW}Scanning for vulnerabilities...${NC}"
set +e  # Don't exit on grype failure
grype "sbom:$OUTPUT_DIR/sbom-cyclonedx.json" \
    -o table \
    --file "$OUTPUT_DIR/grype-results.txt" \
    --fail-on "$GRYPE_FAIL_ON" \
    -v

GRYPE_EXIT_CODE=$?
set -e

echo ""
echo -e "${GREEN}✓ Grype scan completed${NC}"
echo -e "  Results saved to: $OUTPUT_DIR/grype-results.txt"
echo ""

# Also generate SARIF for CI compatibility
echo -e "${YELLOW}Generating SARIF output for CI integration...${NC}"
grype "sbom:$OUTPUT_DIR/sbom-cyclonedx.json" \
    -o sarif \
    --file "$OUTPUT_DIR/grype-results.sarif" 2>/dev/null || true
echo -e "${GREEN}✓ SARIF output saved to: $OUTPUT_DIR/grype-results.sarif${NC}"
echo ""

# Clean up temp requirements
rm -f "$TEMP_REQUIREMENTS"

# Clean up virtual environment if requested
if [[ "$CLEANUP_VENV" == "true" ]]; then
    echo -e "${YELLOW}Cleaning up virtual environment...${NC}"
    rm -rf "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment removed${NC}"
else
    echo -e "${YELLOW}Virtual environment preserved at: $VENV_DIR${NC}"
fi
echo ""

# Display summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display results
if [[ -f "$OUTPUT_DIR/grype-results.txt" ]]; then
    cat "$OUTPUT_DIR/grype-results.txt"
    echo ""
fi

# Exit with Grype's exit code
if [[ $GRYPE_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ No vulnerabilities found at or above '$GRYPE_FAIL_ON' severity${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Vulnerabilities found at or above '$GRYPE_FAIL_ON' severity${NC}"
    echo -e "${YELLOW}Review the results in: $OUTPUT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}To adjust the severity threshold, use:${NC}"
    echo -e "${YELLOW}  $0 --severity high${NC}"
    echo ""
    exit $GRYPE_EXIT_CODE
fi

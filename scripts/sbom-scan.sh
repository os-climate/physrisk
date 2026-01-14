#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

# Script to generate SBOM and perform security scanning locally
# This mimics the CI/CD workflow for local development

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SBOM_FORMAT="cyclonedx"
OUTPUT_DIR="./sbom-output"
GRYPE_FAIL_ON="medium"
GRYPE_OUTPUT_FORMAT="table"
INCLUDE_DEV="false"
CLEANUP="false"

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate SBOM and perform security scanning with Grype

OPTIONS:
    -f, --format FORMAT       SBOM format: cyclonedx, spdx, or both (default: cyclonedx)
    -o, --output DIR          Output directory for SBOM and scan results (default: ./sbom-output)
    -s, --severity LEVEL      Grype fail-on severity: negligible, low, medium, high, critical (default: medium)
    -g, --grype-format FORMAT Grype output format: table, json, sarif, cyclonedx (default: table)
    -d, --include-dev         Include development dependencies in SBOM
    -c, --cleanup             Remove output directory before running
    -h, --help                Display this help message

EXAMPLES:
    # Basic SBOM generation and scan
    $0

    # Generate both SBOM formats with dev dependencies
    $0 --format both --include-dev

    # Scan with high severity threshold only
    $0 --severity high

    # Generate SARIF output for CI integration
    $0 --grype-format sarif

    # Clean previous results and run fresh scan
    $0 --cleanup

REQUIREMENTS:
    - Python 3.10+ with uv or pip
    - cyclonedx-bom (for CycloneDX SBOM)
    - grype (for security scanning)

INSTALLATION:
    # Install cyclonedx-bom
    pip install cyclonedx-bom

    # Install grype (Linux/macOS)
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

    # Or via Homebrew (macOS)
    brew install grype

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--format)
            SBOM_FORMAT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--severity)
            GRYPE_FAIL_ON="$2"
            shift 2
            ;;
        -g|--grype-format)
            GRYPE_OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -d|--include-dev)
            INCLUDE_DEV="true"
            shift
            ;;
        -c|--cleanup)
            CLEANUP="true"
            shift
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

# Cleanup if requested
if [[ "$CLEANUP" == "true" && -d "$OUTPUT_DIR" ]]; then
    echo -e "${YELLOW}Cleaning up previous results in $OUTPUT_DIR${NC}"
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SBOM Generation and Security Scanning${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for required tools
echo -e "${YELLOW}Checking for required tools...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

if ! command -v cyclonedx-py &> /dev/null && ! python3 -m pip show cyclonedx-bom &> /dev/null; then
    echo -e "${RED}Error: cyclonedx-bom not found${NC}"
    echo -e "${YELLOW}Install with: pip install cyclonedx-bom${NC}"
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

# Generate SBOM
echo -e "${BLUE}Generating SBOM...${NC}"
echo -e "  Format: $SBOM_FORMAT"
echo -e "  Include dev dependencies: $INCLUDE_DEV"
echo -e "  Output directory: $OUTPUT_DIR"
echo ""

# First, generate a requirements file for accurate dependency scanning
echo -e "${YELLOW}Generating requirements file...${NC}"
TEMP_REQUIREMENTS="$OUTPUT_DIR/temp-requirements.txt"

# Try to use uv export first, fall back to pip freeze
if command -v uv &> /dev/null; then
    if [[ "$INCLUDE_DEV" == "true" ]]; then
        uv export --no-hashes --format requirements-txt > "$TEMP_REQUIREMENTS" 2>/dev/null || \
            pip freeze > "$TEMP_REQUIREMENTS"
    else
        uv export --no-hashes --no-dev --format requirements-txt > "$TEMP_REQUIREMENTS" 2>/dev/null || \
            pip freeze > "$TEMP_REQUIREMENTS"
    fi
else
    pip freeze > "$TEMP_REQUIREMENTS"
fi

echo -e "${GREEN}✓ Requirements file generated${NC}"
echo ""

# Determine which formats to generate
FORMATS=()
if [[ "$SBOM_FORMAT" == "both" ]]; then
    FORMATS=("cyclonedx" "spdx")
else
    FORMATS=("$SBOM_FORMAT")
fi

# Generate SBOM for each format
SBOM_FILES=()
for format in "${FORMATS[@]}"; do
    echo -e "${YELLOW}Generating $format SBOM...${NC}"

    case $format in
        cyclonedx)
            # Generate SBOM from requirements file
            cyclonedx-py requirements "$TEMP_REQUIREMENTS" \
                --of JSON \
                -o "$OUTPUT_DIR/sbom-cyclonedx.json" \
                --pyproject pyproject.toml
            cyclonedx-py requirements "$TEMP_REQUIREMENTS" \
                --of XML \
                -o "$OUTPUT_DIR/sbom-cyclonedx.xml" \
                --pyproject pyproject.toml
            SBOM_FILES+=("$OUTPUT_DIR/sbom-cyclonedx.json")
            echo -e "${GREEN}✓ CycloneDX SBOM generated:${NC}"
            echo -e "  - $OUTPUT_DIR/sbom-cyclonedx.json"
            echo -e "  - $OUTPUT_DIR/sbom-cyclonedx.xml"
            ;;
        spdx)
            echo -e "${YELLOW}Note: SPDX format requires additional tooling (e.g., syft)${NC}"
            if command -v syft &> /dev/null; then
                syft dir:. -o spdx-json="$OUTPUT_DIR/sbom-spdx.json"
                SBOM_FILES+=("$OUTPUT_DIR/sbom-spdx.json")
                echo -e "${GREEN}✓ SPDX SBOM generated: $OUTPUT_DIR/sbom-spdx.json${NC}"
            else
                echo -e "${RED}Skipping SPDX: syft not found${NC}"
                echo -e "${YELLOW}Install with: brew install syft${NC}"
            fi
            ;;
        *)
            echo -e "${RED}Error: Unsupported format $format${NC}"
            exit 1
            ;;
    esac
done

# Clean up temp requirements file
rm -f "$TEMP_REQUIREMENTS"

echo ""

# Run Grype security scan
if [[ ${#SBOM_FILES[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No SBOM files generated${NC}"
    exit 1
fi

echo -e "${BLUE}Running Grype security scan...${NC}"
echo -e "  SBOM: ${SBOM_FILES[0]}"
echo -e "  Fail-on severity: $GRYPE_FAIL_ON"
echo -e "  Output format: $GRYPE_OUTPUT_FORMAT"
echo ""

# Determine output file extension
case $GRYPE_OUTPUT_FORMAT in
    sarif)
        OUTPUT_EXT="sarif"
        ;;
    json)
        OUTPUT_EXT="json"
        ;;
    cyclonedx)
        OUTPUT_EXT="json"
        ;;
    *)
        OUTPUT_EXT="txt"
        ;;
esac

GRYPE_OUTPUT="$OUTPUT_DIR/grype-results.$OUTPUT_EXT"

# Run Grype scan
echo -e "${YELLOW}Scanning for vulnerabilities...${NC}"
set +e  # Don't exit on grype failure, we want to capture the results
grype "sbom:${SBOM_FILES[0]}" \
    -o "$GRYPE_OUTPUT_FORMAT" \
    --file "$GRYPE_OUTPUT" \
    --fail-on "$GRYPE_FAIL_ON" \
    -v

GRYPE_EXIT_CODE=$?
set -e

echo ""
echo -e "${GREEN}✓ Grype scan completed${NC}"
echo -e "  Results saved to: $GRYPE_OUTPUT"
echo ""

# Also generate a table output for easy reading if not already table format
if [[ "$GRYPE_OUTPUT_FORMAT" != "table" ]]; then
    echo -e "${YELLOW}Generating human-readable summary...${NC}"
    grype "sbom:${SBOM_FILES[0]}" \
        -o table \
        --file "$OUTPUT_DIR/grype-results-summary.txt" 2>/dev/null || true
    echo -e "${GREEN}✓ Summary saved to: $OUTPUT_DIR/grype-results-summary.txt${NC}"
    echo ""
fi

# Display summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Count vulnerabilities if we have a table output
if [[ -f "$OUTPUT_DIR/grype-results-summary.txt" ]]; then
    cat "$OUTPUT_DIR/grype-results-summary.txt"
    echo ""
elif [[ "$GRYPE_OUTPUT_FORMAT" == "table" && -f "$GRYPE_OUTPUT" ]]; then
    cat "$GRYPE_OUTPUT"
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

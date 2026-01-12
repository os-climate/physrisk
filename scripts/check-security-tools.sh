#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

# Quick check for security scanning tools and installation helper

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Security Tools Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

MISSING_TOOLS=()

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    MISSING_TOOLS+=("python3")
fi
echo ""

# Check cyclonedx-bom
echo -e "${YELLOW}Checking cyclonedx-bom...${NC}"
if command -v cyclonedx-py &> /dev/null; then
    CYCLONE_VERSION=$(cyclonedx-py --version 2>&1 | head -n1)
    echo -e "${GREEN}✓ $CYCLONE_VERSION${NC}"
elif python3 -m pip show cyclonedx-bom &> /dev/null; then
    CYCLONE_VERSION=$(python3 -m pip show cyclonedx-bom | grep Version)
    echo -e "${GREEN}✓ cyclonedx-bom $CYCLONE_VERSION${NC}"
else
    echo -e "${RED}✗ cyclonedx-bom not found${NC}"
    MISSING_TOOLS+=("cyclonedx-bom")
fi
echo ""

# Check Grype
echo -e "${YELLOW}Checking Grype...${NC}"
if command -v grype &> /dev/null; then
    GRYPE_VERSION=$(grype version 2>&1 | head -n1)
    echo -e "${GREEN}✓ $GRYPE_VERSION${NC}"
else
    echo -e "${RED}✗ Grype not found${NC}"
    MISSING_TOOLS+=("grype")
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All required tools are installed!${NC}"
    echo ""
    echo -e "${BLUE}You can now run:${NC}"
    echo -e "  ${GREEN}make security-check${NC}       # Full SBOM + security scan"
    echo -e "  ${GREEN}make sbom${NC}                 # Generate SBOM only"
    echo -e "  ${GREEN}make grype-scan${NC}           # Run Grype scan"
    echo -e "  ${GREEN}./scripts/sbom-scan.sh${NC}    # Direct script usage"
    echo ""
else
    echo -e "${YELLOW}Missing tools: ${MISSING_TOOLS[*]}${NC}"
    echo ""
    echo -e "${BLUE}Installation Instructions:${NC}"
    echo ""

    for tool in "${MISSING_TOOLS[@]}"; do
        case $tool in
            cyclonedx-bom)
                echo -e "${YELLOW}Install cyclonedx-bom:${NC}"
                echo -e "  ${GREEN}pip install cyclonedx-bom${NC}"
                echo ""
                ;;
            grype)
                echo -e "${YELLOW}Install Grype:${NC}"
                echo -e "  ${GREEN}# macOS (Homebrew)${NC}"
                echo -e "  ${GREEN}brew install grype${NC}"
                echo ""
                echo -e "  ${GREEN}# Linux/macOS (install script)${NC}"
                echo -e "  ${GREEN}curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin${NC}"
                echo ""
                echo -e "  ${GREEN}# Windows (Scoop)${NC}"
                echo -e "  ${GREEN}scoop install grype${NC}"
                echo ""
                ;;
            python3)
                echo -e "${YELLOW}Install Python 3.10+:${NC}"
                echo -e "  Visit: ${GREEN}https://www.python.org/downloads/${NC}"
                echo ""
                ;;
        esac
    done

    echo -e "${BLUE}Or install all tools at once:${NC}"
    echo -e "  ${GREEN}make install-tools${NC}"
    echo ""
fi

exit 0

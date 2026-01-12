# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

# Makefile for physrisk project
# Provides convenient targets for building, testing, and security scanning

.PHONY: help install install-dev test clean sbom sbom-scan grype-scan security-check lint format build

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help: ## Display this help message
	@echo "$(BLUE)physrisk - Makefile targets$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -e .

install-dev: ## Install project with development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e ".[dev,test,doc]"

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting with ruff
	@echo "$(BLUE)Running linting...$(NC)"
	ruff check src/ tests/

lint-fix: ## Run linting and fix issues automatically
	@echo "$(BLUE)Running linting with auto-fix...$(NC)"
	ruff check --fix src/ tests/

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format src/ tests/

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code format...$(NC)"
	ruff format --check src/ tests/

mypy: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy src/

build: ## Build source and wheel distributions
	@echo "$(BLUE)Building distributions...$(NC)"
	python -m build

clean: ## Remove build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf coverage_html_report/
	rm -rf sbom-output/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned successfully$(NC)"

sbom: ## Generate SBOM (CycloneDX format)
	@echo "$(BLUE)Generating SBOM...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --output sbom-output

sbom-all: ## Generate SBOM in all formats (CycloneDX and SPDX)
	@echo "$(BLUE)Generating SBOM in all formats...$(NC)"
	@./scripts/sbom-scan.sh --format both --output sbom-output

sbom-dev: ## Generate SBOM including development dependencies
	@echo "$(BLUE)Generating SBOM with dev dependencies...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --include-dev --output sbom-output

grype-scan: sbom ## Run Grype security scan (medium+ severity)
	@echo "$(BLUE)Running Grype security scan...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --output sbom-output

grype-high: sbom ## Run Grype security scan (high+ severity only)
	@echo "$(BLUE)Running Grype security scan (high+ severity)...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --severity high --output sbom-output

grype-critical: sbom ## Run Grype security scan (critical severity only)
	@echo "$(BLUE)Running Grype security scan (critical severity)...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --severity critical --output sbom-output

grype-sarif: sbom ## Run Grype security scan with SARIF output
	@echo "$(BLUE)Running Grype security scan (SARIF output)...$(NC)"
	@./scripts/sbom-scan.sh --format cyclonedx --grype-format sarif --output sbom-output

security-check: ## Run all security checks (SBOM + Grype scan)
	@echo "$(BLUE)Running security checks...$(NC)"
	@./scripts/sbom-scan.sh --cleanup --format cyclonedx --output sbom-output

security-full: ## Run comprehensive security check with dev dependencies
	@echo "$(BLUE)Running comprehensive security check...$(NC)"
	@./scripts/sbom-scan.sh --cleanup --format both --include-dev --output sbom-output

security-ci: ## Run CI-style security scan in fresh virtual environment
	@echo "$(BLUE)Running CI-style security scan...$(NC)"
	@./scripts/ci-sbom-scan.sh --output sbom-output

security-ci-high: ## Run CI-style security scan with high severity threshold
	@echo "$(BLUE)Running CI-style security scan (high severity)...$(NC)"
	@./scripts/ci-sbom-scan.sh --severity high --output sbom-output

pre-commit: lint-fix format test ## Run pre-commit checks (lint, format, test)
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

ci-check: lint format-check mypy test ## Run CI-style checks without modifications
	@echo "$(GREEN)✓ CI checks passed$(NC)"

all: clean install-dev lint format test build sbom ## Run complete build pipeline
	@echo "$(GREEN)✓ Complete build pipeline finished$(NC)"

# Install required tools for SBOM and security scanning
install-tools: ## Install SBOM and security scanning tools
	@echo "$(BLUE)Installing SBOM and security tools...$(NC)"
	@echo "$(YELLOW)Installing cyclonedx-bom...$(NC)"
	pip install cyclonedx-bom
	@echo "$(YELLOW)Installing grype...$(NC)"
	@if command -v brew >/dev/null 2>&1; then \
		echo "  Using Homebrew..."; \
		brew install grype; \
	else \
		echo "  Using install script..."; \
		curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b ~/.local/bin; \
		echo "$(YELLOW)  Add ~/.local/bin to your PATH if not already present$(NC)"; \
	fi
	@echo "$(GREEN)✓ Tools installed$(NC)"

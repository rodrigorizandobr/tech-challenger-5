# Makefile for Decision AI Recruitment System

.PHONY: help setup install train api monitor test docker-build docker-run docker-stop clean lint format check-deps

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PORT := 8000
HOST := 0.0.0.0
DATA_PATH := data/sample_candidates.csv
MODEL_PATH := models/model.joblib
METADATA_PATH := models/training_metadata.json
LOGS_PATH := logs/predictions.csv
REPORTS_PATH := reports

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Decision AI Recruitment System$(RESET)"
	@echo "$(BLUE)================================$(RESET)"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick start:$(RESET)"
	@echo "  make setup && make train && make api"
	@echo ""

setup: ## Setup the development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv venv || true
	@echo "$(YELLOW)Activating virtual environment and installing dependencies...$(RESET)"
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(GREEN)Setup completed!$(RESET)"
	@echo "$(YELLOW)To activate the virtual environment, run: source venv/bin/activate$(RESET)"

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed!$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov black ruff mypy
	@echo "$(GREEN)Development dependencies installed!$(RESET)"

data: ## Generate synthetic data
	@echo "$(BLUE)Generating synthetic data...$(RESET)"
	mkdir -p data
	$(PYTHON) -m src.data
	@echo "$(GREEN)Synthetic data generated at $(DATA_PATH)$(RESET)"

train: data ## Train the machine learning model
	@echo "$(BLUE)Training the model...$(RESET)"
	mkdir -p models
	$(PYTHON) -m src.train \
		--data-path $(DATA_PATH) \
		--model-path $(MODEL_PATH) \
		--metadata-path $(METADATA_PATH)
	@echo "$(GREEN)Model training completed!$(RESET)"
	@echo "$(YELLOW)Model saved at: $(MODEL_PATH)$(RESET)"

api: ## Start the FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(RESET)"
	mkdir -p logs reports
	@echo "$(YELLOW)Server will be available at: http://$(HOST):$(PORT)$(RESET)"
	@echo "$(YELLOW)API documentation: http://$(HOST):$(PORT)/docs$(RESET)"
	uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

api-prod: ## Start the FastAPI server in production mode
	@echo "$(BLUE)Starting FastAPI server in production mode...$(RESET)"
	mkdir -p logs reports
	uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 4

monitor: ## Generate drift monitoring report
	@echo "$(BLUE)Generating drift monitoring report...$(RESET)"
	mkdir -p $(REPORTS_PATH)
	$(PYTHON) -m monitor.generate_report \
		--reference-data $(DATA_PATH) \
		--predictions-log $(LOGS_PATH) \
		--reports-dir $(REPORTS_PATH) \
		--window-size 100
	@echo "$(GREEN)Drift report generated!$(RESET)"
	@echo "$(YELLOW)Report available at: $(REPORTS_PATH)/drift.html$(RESET)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest tests/ -v --tb=short
	@echo "$(GREEN)Tests completed!$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/ -v -m "unit" --tb=short

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/ -v -m "integration" --tb=short

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ --cov=src --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated at htmlcov/index.html$(RESET)"

lint: ## Run linting
	@echo "$(BLUE)Running linting...$(RESET)"
	ruff check src/ app/ tests/
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(RESET)"
	black src/ app/ tests/
	ruff check --fix src/ app/ tests/
	@echo "$(GREEN)Code formatting completed!$(RESET)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(RESET)"
	mypy src/ app/ --ignore-missing-imports
	@echo "$(GREEN)Type checking completed!$(RESET)"

check: lint type-check test ## Run all quality checks
	@echo "$(GREEN)All quality checks passed!$(RESET)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t decision-ai:latest .
	@echo "$(GREEN)Docker image built successfully!$(RESET)"

docker-run: ## Run application in Docker
	@echo "$(BLUE)Running application in Docker...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Application started!$(RESET)"
	@echo "$(YELLOW)API available at: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Health check: http://localhost:8000/health$(RESET)"

docker-dev: ## Run application in development mode with Docker
	@echo "$(BLUE)Running application in development mode...$(RESET)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)Development environment started!$(RESET)"
	@echo "$(YELLOW)API available at: http://localhost:8001$(RESET)"

docker-stop: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(RESET)"
	docker-compose down
	@echo "$(GREEN)Containers stopped!$(RESET)"

docker-logs: ## Show Docker logs
	@echo "$(BLUE)Showing Docker logs...$(RESET)"
	docker-compose logs -f

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	docker-compose down -v --rmi all
	docker system prune -f
	@echo "$(GREEN)Docker cleanup completed!$(RESET)"

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(RESET)"
	rm -rf __pycache__ .pytest_cache .coverage htmlcov/
	rm -rf src/__pycache__ app/__pycache__ tests/__pycache__
	rm -rf monitor/__pycache__
	rm -rf .mypy_cache .ruff_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)Cleanup completed!$(RESET)"

clean-all: clean ## Clean everything including models and data
	@echo "$(BLUE)Cleaning everything...$(RESET)"
	rm -rf models/ logs/ reports/
	rm -rf data/sample_candidates.csv data/sample_payload.json
	rm -rf venv/
	@echo "$(GREEN)Complete cleanup finished!$(RESET)"

check-deps: ## Check for dependency updates
	@echo "$(BLUE)Checking for dependency updates...$(RESET)"
	$(PIP) list --outdated

reqs: ## Generate requirements.txt from current environment
	@echo "$(BLUE)Generating requirements.txt...$(RESET)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)Requirements updated!$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) -m pytest tests/ -k "benchmark" -v

status: ## Show project status
	@echo "$(BLUE)Project Status$(RESET)"
	@echo "$(BLUE)==============$(RESET)"
	@echo ""
	@echo "$(YELLOW)Data:$(RESET)"
	@if [ -f "$(DATA_PATH)" ]; then echo "  ✓ Training data exists"; else echo "  ✗ Training data missing"; fi
	@echo ""
	@echo "$(YELLOW)Model:$(RESET)"
	@if [ -f "$(MODEL_PATH)" ]; then echo "  ✓ Model exists"; else echo "  ✗ Model missing"; fi
	@if [ -f "$(METADATA_PATH)" ]; then echo "  ✓ Model metadata exists"; else echo "  ✗ Model metadata missing"; fi
	@echo ""
	@echo "$(YELLOW)Logs:$(RESET)"
	@if [ -f "$(LOGS_PATH)" ]; then echo "  ✓ Prediction logs exist"; else echo "  ✗ No prediction logs"; fi
	@echo ""
	@echo "$(YELLOW)Reports:$(RESET)"
	@if [ -f "$(REPORTS_PATH)/drift.html" ]; then echo "  ✓ Drift report exists"; else echo "  ✗ No drift report"; fi
	@echo ""

quick-start: setup train api ## Quick start: setup, train, and run API
	@echo "$(GREEN)Quick start completed!$(RESET)"

full-pipeline: setup data train test monitor api ## Run the complete pipeline
	@echo "$(GREEN)Full pipeline completed!$(RESET)"

# Development helpers
dev-setup: setup install-dev ## Setup development environment with dev dependencies
	@echo "$(GREEN)Development environment ready!$(RESET)"

dev-check: format lint type-check test ## Run all development checks
	@echo "$(GREEN)All development checks passed!$(RESET)"

# Production helpers
prod-build: docker-build ## Build production Docker image
	@echo "$(GREEN)Production build completed!$(RESET)"

prod-deploy: docker-run ## Deploy to production
	@echo "$(GREEN)Production deployment completed!$(RESET)"

# Help for specific workflows
workflow-help: ## Show common workflows
	@echo "$(BLUE)Common Workflows$(RESET)"
	@echo "$(BLUE)=================$(RESET)"
	@echo ""
	@echo "$(YELLOW)First time setup:$(RESET)"
	@echo "  make setup"
	@echo "  make train"
	@echo "  make api"
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@echo "  make dev-setup"
	@echo "  make dev-check"
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@echo "  make test"
	@echo "  make test-cov"
	@echo ""
	@echo "$(YELLOW)Docker:$(RESET)"
	@echo "  make docker-build"
	@echo "  make docker-run"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(RESET)"
	@echo "  make monitor"
	@echo ""
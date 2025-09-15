# Makefile para Sistema de IA de Recrutamento Decision

.PHONY: help setup install train api monitor test docker-build docker-run docker-stop clean lint format check-deps

# Alvo padrão
.DEFAULT_GOAL := help

# Variáveis
PYTHON := python3
PIP := pip3
PORT := 8000
HOST := 0.0.0.0
DATA_PATH := data/sample_candidates.csv
MODEL_PATH := models/model.joblib
METADATA_PATH := models/training_metadata.json
LOGS_PATH := logs/predictions.csv
REPORTS_PATH := reports

# Cores para saída
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)Sistema de IA de Recrutamento Decision$(RESET)"
	@echo "$(BLUE)======================================$(RESET)"
	@echo ""
	@echo "Comandos disponíveis:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Início rápido:$(RESET)"
	@echo "  make setup && make train && make api"
	@echo ""

setup: ## Configura o ambiente de desenvolvimento
	@echo "$(BLUE)Configurando ambiente de desenvolvimento...$(RESET)"
	$(PYTHON) -m venv venv || true
	@echo "$(YELLOW)Ativando ambiente virtual e instalando dependências...$(RESET)"
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(GREEN)Configuração concluída!$(RESET)"
	@echo "$(YELLOW)Para ativar o ambiente virtual, execute: source venv/bin/activate$(RESET)"

install: ## Instala dependências
	@echo "$(BLUE)Instalando dependências...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependências instaladas!$(RESET)"

install-dev: ## Instala dependências de desenvolvimento
	@echo "$(BLUE)Instalando dependências de desenvolvimento...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov black ruff mypy
	@echo "$(GREEN)Dependências de desenvolvimento instaladas!$(RESET)"

data: ## Gera dados sintéticos
	@echo "$(BLUE)Gerando dados sintéticos...$(RESET)"
	mkdir -p data
	$(PYTHON) -m src.data
	@echo "$(GREEN)Dados sintéticos gerados em $(DATA_PATH)$(RESET)"

train: data ## Treina o modelo de machine learning
	@echo "$(BLUE)Treinando o modelo...$(RESET)"
	mkdir -p models
	$(PYTHON) -m src.train \
		--data-path $(DATA_PATH) \
		--model-path $(MODEL_PATH) \
		--metadata-path $(METADATA_PATH)
	@echo "$(GREEN)Treinamento do modelo concluído!$(RESET)"
	@echo "$(YELLOW)Modelo salvo em: $(MODEL_PATH)$(RESET)"

api: ## Inicia o servidor FastAPI
	@echo "$(BLUE)Iniciando servidor FastAPI...$(RESET)"
	mkdir -p logs reports
	@echo "$(YELLOW)Servidor estará disponível em: http://$(HOST):$(PORT)$(RESET)"
	@echo "$(YELLOW)Documentação da API: http://$(HOST):$(PORT)/docs$(RESET)"
	uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

api-prod: ## Inicia o servidor FastAPI em modo de produção
	@echo "$(BLUE)Iniciando servidor FastAPI em modo de produção...$(RESET)"
	mkdir -p logs reports
	uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 4

monitor: ## Gera relatório de monitoramento de drift
	@echo "$(BLUE)Gerando relatório de monitoramento de drift...$(RESET)"
	mkdir -p $(REPORTS_PATH)
	$(PYTHON) -m monitor.generate_report \
		--reference-data $(DATA_PATH) \
		--predictions-log $(LOGS_PATH) \
		--reports-dir $(REPORTS_PATH) \
		--window-size 100
	@echo "$(GREEN)Relatório de drift gerado!$(RESET)"
	@echo "$(YELLOW)Relatório disponível em: $(REPORTS_PATH)/drift.html$(RESET)"

test: ## Executa todos os testes
	@echo "$(BLUE)Executando testes...$(RESET)"
	pytest tests/ -v --tb=short
	@echo "$(GREEN)Testes concluídos!$(RESET)"

test-unit: ## Executa apenas testes unitários
	@echo "$(BLUE)Executando testes unitários...$(RESET)"
	pytest tests/ -v -m "unit" --tb=short

test-integration: ## Executa apenas testes de integração
	@echo "$(BLUE)Executando testes de integração...$(RESET)"
	pytest tests/ -v -m "integration" --tb=short

test-cov: ## Executa testes com cobertura
	@echo "$(BLUE)Executando testes com cobertura...$(RESET)"
	pytest tests/ --cov=src --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)Relatório de cobertura gerado em htmlcov/index.html$(RESET)"

lint: ## Executa linting
	@echo "$(BLUE)Executando linting...$(RESET)"
	ruff check src/ app/ tests/
	@echo "$(GREEN)Linting concluído!$(RESET)"

format: ## Formata código
	@echo "$(BLUE)Formatando código...$(RESET)"
	black src/ app/ tests/
	ruff check --fix src/ app/ tests/
	@echo "$(GREEN)Formatação de código concluída!$(RESET)"

type-check: ## Executa verificação de tipos
	@echo "$(BLUE)Executando verificação de tipos...$(RESET)"
	mypy src/ app/ --ignore-missing-imports
	@echo "$(GREEN)Verificação de tipos concluída!$(RESET)"

check: lint type-check test ## Executa todas as verificações de qualidade
	@echo "$(GREEN)Todas as verificações de qualidade passaram!$(RESET)"

docker-build: ## Constrói imagem Docker
	@echo "$(BLUE)Construindo imagem Docker...$(RESET)"
	docker build -t decision-ai:latest .
	@echo "$(GREEN)Imagem Docker construída com sucesso!$(RESET)"

docker-run: ## Executa aplicação no Docker
	@echo "$(BLUE)Executando aplicação no Docker...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Aplicação iniciada!$(RESET)"
	@echo "$(YELLOW)API disponível em: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Verificação de saúde: http://localhost:8000/health$(RESET)"

docker-dev: ## Executa aplicação em modo de desenvolvimento com Docker
	@echo "$(BLUE)Executando aplicação em modo de desenvolvimento...$(RESET)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)Ambiente de desenvolvimento iniciado!$(RESET)"
	@echo "$(YELLOW)API disponível em: http://localhost:8001$(RESET)"

docker-stop: ## Para containers Docker
	@echo "$(BLUE)Parando containers Docker...$(RESET)"
	docker-compose down
	@echo "$(GREEN)Containers parados!$(RESET)"

docker-logs: ## Mostra logs do Docker
	@echo "$(BLUE)Mostrando logs do Docker...$(RESET)"
	docker-compose logs -f

docker-clean: ## Limpa recursos do Docker
	@echo "$(BLUE)Limpando recursos do Docker...$(RESET)"
	docker-compose down -v --rmi all
	docker system prune -f
	@echo "$(GREEN)Limpeza do Docker concluída!$(RESET)"

clean: ## Limpa arquivos gerados
	@echo "$(BLUE)Limpando...$(RESET)"
	rm -rf __pycache__ .pytest_cache .coverage htmlcov/
	rm -rf src/__pycache__ app/__pycache__ tests/__pycache__
	rm -rf monitor/__pycache__
	rm -rf .mypy_cache .ruff_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)Limpeza concluída!$(RESET)"

clean-all: clean ## Limpa tudo incluindo modelos e dados
	@echo "$(BLUE)Limpando tudo...$(RESET)"
	rm -rf models/ logs/ reports/
	rm -rf data/sample_candidates.csv data/sample_payload.json
	rm -rf venv/
	@echo "$(GREEN)Limpeza completa finalizada!$(RESET)"

check-deps: ## Verifica atualizações de dependências
	@echo "$(BLUE)Verificando atualizações de dependências...$(RESET)"
	$(PIP) list --outdated

reqs: ## Gera requirements.txt do ambiente atual
	@echo "$(BLUE)Gerando requirements.txt...$(RESET)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)Requirements atualizados!$(RESET)"

benchmark: ## Executa benchmarks de performance
	@echo "$(BLUE)Executando benchmarks de performance...$(RESET)"
	$(PYTHON) -m pytest tests/ -k "benchmark" -v

status: ## Mostra status do projeto
	@echo "$(BLUE)Status do Projeto$(RESET)"
	@echo "$(BLUE)=================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Dados:$(RESET)"
	@if [ -f "$(DATA_PATH)" ]; then echo "  ✓ Dados de treinamento existem"; else echo "  ✗ Dados de treinamento ausentes"; fi
	@echo ""
	@echo "$(YELLOW)Modelo:$(RESET)"
	@if [ -f "$(MODEL_PATH)" ]; then echo "  ✓ Modelo existe"; else echo "  ✗ Modelo ausente"; fi
	@if [ -f "$(METADATA_PATH)" ]; then echo "  ✓ Metadados do modelo existem"; else echo "  ✗ Metadados do modelo ausentes"; fi
	@echo ""
	@echo "$(YELLOW)Logs:$(RESET)"
	@if [ -f "$(LOGS_PATH)" ]; then echo "  ✓ Logs de predição existem"; else echo "  ✗ Nenhum log de predição"; fi
	@echo ""
	@echo "$(YELLOW)Relatórios:$(RESET)"
	@if [ -f "$(REPORTS_PATH)/drift.html" ]; then echo "  ✓ Relatório de drift existe"; else echo "  ✗ Nenhum relatório de drift"; fi
	@echo ""

quick-start: setup train api ## Início rápido: configura, treina e executa API
	@echo "$(GREEN)Início rápido concluído!$(RESET)"

full-pipeline: setup data train test monitor api ## Executa o pipeline completo
	@echo "$(GREEN)Pipeline completo concluído!$(RESET)"

# Auxiliares de desenvolvimento
dev-setup: setup install-dev ## Configura ambiente de desenvolvimento com dependências de dev
	@echo "$(GREEN)Ambiente de desenvolvimento pronto!$(RESET)"

dev-check: format lint type-check test ## Executa todas as verificações de desenvolvimento
	@echo "$(GREEN)Todas as verificações de desenvolvimento passaram!$(RESET)"

# Auxiliares de produção
prod-build: docker-build ## Constrói imagem Docker de produção
	@echo "$(GREEN)Build de produção concluído!$(RESET)"

prod-deploy: docker-run ## Faz deploy para produção
	@echo "$(GREEN)Deploy de produção concluído!$(RESET)"

# Ajuda para workflows específicos
workflow-help: ## Mostra workflows comuns
	@echo "$(BLUE)Workflows Comuns$(RESET)"
	@echo "$(BLUE)===============$(RESET)"
	@echo ""
	@echo "$(YELLOW)Configuração inicial:$(RESET)"
	@echo "  make setup"
	@echo "  make train"
	@echo "  make api"
	@echo ""
	@echo "$(YELLOW)Desenvolvimento:$(RESET)"
	@echo "  make dev-setup"
	@echo "  make dev-check"
	@echo ""
	@echo "$(YELLOW)Testes:$(RESET)"
	@echo "  make test"
	@echo "  make test-cov"
	@echo ""
	@echo "$(YELLOW)Docker:$(RESET)"
	@echo "  make docker-build"
	@echo "  make docker-run"
	@echo ""
	@echo "$(YELLOW)Monitoramento:$(RESET)"
	@echo "  make monitor"
	@echo ""
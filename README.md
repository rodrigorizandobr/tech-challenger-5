# Decision AI - Recruitment System

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

Um sistema de IA para recrutamento e seleção que utiliza machine learning para fazer match entre candidatos e vagas, desenvolvido para o Tech Challenge 5 da FIAP.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [API Endpoints](#-api-endpoints)
- [Monitoramento](#-monitoramento)
- [Testes](#-testes)
- [Docker](#-docker)
- [Deploy](#-deploy)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Contribuição](#-contribuição)

## 🎯 Visão Geral

O Decision AI é uma solução completa de machine learning para automatizar o processo de recrutamento e seleção. O sistema analisa perfis de candidatos e requisitos de vagas para calcular a probabilidade de match, fornecendo recomendações baseadas em dados.

### Principais Características

- **IA Avançada**: Utiliza algoritmos de machine learning (Random Forest, Gradient Boosting, etc.)
- **API RESTful**: Interface FastAPI com documentação automática
- **Monitoramento**: Detecção de drift de dados com Evidently AI
- **Containerização**: Deploy com Docker e Docker Compose
- **Testes Automatizados**: Cobertura completa com pytest
- **Dados Sintéticos**: Geração automática de dados para treinamento

## ✨ Funcionalidades

### Core Features

- ✅ **Predição de Match**: Calcula probabilidade de compatibilidade candidato-vaga
- ✅ **Análise de Fatores**: Identifica quais aspectos influenciam o match
- ✅ **Recomendações**: Fornece sugestões baseadas na análise
- ✅ **Validação de Dados**: Validação robusta com Pydantic
- ✅ **Logging Estruturado**: Logs detalhados para auditoria

### Monitoramento e Observabilidade

- ✅ **Health Check**: Endpoint para verificação de saúde do sistema
- ✅ **Métricas**: Estatísticas de uso e performance
- ✅ **Drift Detection**: Monitoramento de mudanças nos dados
- ✅ **Relatórios**: Dashboards visuais com Evidently

### DevOps e Qualidade

- ✅ **CI/CD Ready**: Configuração para GitHub Actions
- ✅ **Containerização**: Docker multi-stage para produção
- ✅ **Testes**: Cobertura completa com pytest
- ✅ **Linting**: Black e Ruff para qualidade de código
- ✅ **Type Hints**: Tipagem estática completa

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend/     │    │   FastAPI       │    │   ML Pipeline   │
│   Postman       │───▶│   Application   │───▶│   (sklearn)     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Monitoring    │    │   Data Storage  │
                       │   (Evidently)   │    │   (CSV/Joblib)  │
                       └─────────────────┘    └─────────────────┘
```

### Stack Tecnológica

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **ML**: scikit-learn, pandas, numpy
- **Monitoramento**: Evidently AI, Loguru
- **Validação**: Pydantic
- **Testes**: pytest, httpx
- **Containerização**: Docker, Docker Compose
- **Qualidade**: Black, Ruff, MyPy

## 🚀 Instalação

### Pré-requisitos

- Python 3.11+
- Docker (opcional)
- Make (opcional, mas recomendado)

### Instalação Local

```bash
# Clone o repositório
git clone <repository-url>
cd tech-challenger-5

# Setup completo (recomendado)
make setup

# Ou instalação manual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## ⚡ Uso Rápido

### Opção 1: Usando Make (Recomendado)

```bash
# Setup, treinamento e execução em um comando
make quick-start

# Ou passo a passo
make setup      # Configurar ambiente
make train      # Treinar modelo
make api        # Executar API
```

### Opção 2: Comandos Manuais

```bash
# 1. Gerar dados sintéticos
python -m src.data

# 2. Treinar modelo
python -m src.train --data-path data/sample_candidates.csv

# 3. Executar API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opção 3: Docker

```bash
# Build e execução
make docker-build
make docker-run

# Ou usando docker-compose diretamente
docker-compose up -d
```

Após a execução, a API estará disponível em:
- **API**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Principais Endpoints

| Método | Endpoint | Descrição |
|--------|----------|----------|
| GET | `/` | Página inicial com informações da API |
| GET | `/health` | Health check do sistema |
| POST | `/predict` | Predição de match candidato-vaga |
| GET | `/metrics` | Métricas do sistema |
| GET | `/drift-report` | Relatório de drift de dados |
| GET | `/docs` | Documentação interativa (Swagger) |

### Exemplo de Uso

#### Predição de Match

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "candidate": {
         "age": 28,
         "education_level": "bachelor",
         "years_experience": 5,
         "skills": ["python", "machine learning", "sql"],
         "previous_companies": 2,
         "salary_expectation": 85000,
         "location": "São Paulo",
         "remote_work": true,
         "availability_days": 30
       },
       "job": {
         "required_experience": "mid",
         "required_skills": ["python", "sql", "machine learning"],
         "salary_range_min": 75000,
         "salary_range_max": 95000,
         "location": "São Paulo",
         "remote_allowed": true,
         "urgency_days": 45
       }
     }'
```

#### Resposta Esperada

```json
{
  "match_probability": 0.85,
  "match_label": "good_match",
  "confidence": 0.92,
  "factors": {
    "skills_match": 0.9,
    "experience_match": 0.8,
    "salary_fit": 0.85,
    "location_compatibility": 1.0
  },
  "recommendation": "Excellent match - strongly recommend for interview",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Coleção Postman

Importe o arquivo `postman_collection.json` no Postman para testar todos os endpoints com exemplos pré-configurados.

## 📊 Monitoramento

### Drift Detection

O sistema monitora automaticamente mudanças nos padrões de dados:

```bash
# Gerar relatório de drift
make monitor

# Ou manualmente
python -m monitor.generate_report \
  --reference-data data/sample_candidates.csv \
  --predictions-log logs/predictions.csv
```

### Métricas Disponíveis

- **Total de Predições**: Número total de predições realizadas
- **Probabilidade Média**: Média das probabilidades de match
- **Predições 24h**: Predições nas últimas 24 horas
- **Drift Detectado**: Status de detecção de drift
- **Saúde do Sistema**: Status geral do sistema

### Logs

Todos os logs são estruturados e salvos em:
- `logs/api.log` - Logs da aplicação
- `logs/predictions.csv` - Log de todas as predições

## 🧪 Testes

### Executar Testes

```bash
# Todos os testes
make test

# Apenas testes unitários
make test-unit

# Apenas testes de integração
make test-integration

# Testes com cobertura
make test-cov
```

### Estrutura de Testes

- `tests/test_data.py` - Testes do módulo de dados
- `tests/test_api.py` - Testes da API
- `tests/conftest.py` - Configurações e fixtures

### Cobertura

O projeto mantém alta cobertura de testes:
- Testes unitários para todos os módulos
- Testes de integração para a API
- Testes de validação de dados
- Testes de performance

## 🐳 Docker

### Desenvolvimento

```bash
# Ambiente de desenvolvimento
docker-compose --profile dev up -d
```

### Produção

```bash
# Build da imagem
docker build -t decision-ai:latest .

# Execução
docker-compose up -d

# Com nginx (produção)
docker-compose --profile production up -d
```

### Comandos Úteis

```bash
# Ver logs
make docker-logs

# Parar containers
make docker-stop

# Limpeza completa
make docker-clean
```

## 🚀 Deploy

### Deploy Local

1. **Preparação**:
   ```bash
   make setup
   make train
   ```

2. **Execução**:
   ```bash
   make api-prod
   ```

### Deploy em Nuvem

#### AWS ECS Fargate

1. **Build e Push da Imagem**:
   ```bash
   # Build
   docker build -t decision-ai:latest .
   
   # Tag para ECR
   docker tag decision-ai:latest <account-id>.dkr.ecr.<region>.amazonaws.com/decision-ai:latest
   
   # Push
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/decision-ai:latest
   ```

2. **Configuração ECS**:
   - Criar cluster ECS
   - Definir task definition com a imagem
   - Configurar service com load balancer
   - Definir variáveis de ambiente

3. **Variáveis de Ambiente**:
   ```bash
   API_HOST=0.0.0.0
   API_PORT=8000
   MODEL_PATH=models/model.joblib
   LOG_LEVEL=INFO
   ```

#### Google Cloud Run

1. **Deploy**:
   ```bash
   # Build e deploy
   gcloud run deploy decision-ai \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Configuração**:
   - Configurar variáveis de ambiente
   - Ajustar recursos (CPU/Memory)
   - Configurar domínio customizado

### Monitoramento em Produção

- Configure alertas para health checks
- Monitore métricas de drift
- Configure backup dos logs
- Implemente rotação de logs

## 📁 Estrutura do Projeto

```
tech-challenger-5/
├── src/                    # Código fonte principal
│   ├── __init__.py
│   ├── data.py            # Geração e carregamento de dados
│   ├── features.py        # Engenharia de features
│   ├── schema.py          # Schemas Pydantic
│   └── train.py           # Pipeline de treinamento
├── app/                   # Aplicação FastAPI
│   ├── __init__.py
│   └── main.py           # API principal
├── monitor/               # Monitoramento de drift
│   ├── __init__.py
│   └── generate_report.py # Geração de relatórios
├── tests/                 # Testes automatizados
│   ├── __init__.py
│   ├── conftest.py       # Configurações pytest
│   ├── test_data.py      # Testes do módulo de dados
│   └── test_api.py       # Testes da API
├── data/                  # Dados de treinamento
│   ├── sample_candidates.csv
│   └── sample_payload.json
├── models/                # Modelos treinados
│   ├── model.joblib
│   └── training_metadata.json
├── logs/                  # Logs da aplicação
│   ├── api.log
│   └── predictions.csv
├── reports/               # Relatórios de drift
│   └── drift.html
├── Dockerfile             # Configuração Docker
├── docker-compose.yml     # Orquestração Docker
├── Makefile              # Comandos de automação
├── requirements.txt       # Dependências Python
├── pyproject.toml        # Configuração do projeto
├── .env.example          # Variáveis de ambiente
├── postman_collection.json # Coleção Postman
└── README.md             # Documentação
```

## 🎥 Vídeo Demonstrativo

Para gravar um vídeo de 5 minutos demonstrando o sistema:

### Roteiro Sugerido

1. **Introdução (30s)**
   - Apresentar o projeto Decision AI
   - Mostrar objetivos e funcionalidades

2. **Arquitetura (1min)**
   - Explicar stack tecnológica
   - Mostrar estrutura do projeto
   - Destacar componentes principais

3. **Treinamento (1min)**
   - Demonstrar geração de dados sintéticos
   - Executar treinamento do modelo
   - Mostrar métricas de performance

4. **API em Funcionamento (2min)**
   - Executar a API
   - Demonstrar endpoints principais
   - Fazer predições com Postman
   - Mostrar documentação automática

5. **Docker e Monitoramento (1min)**
   - Executar com Docker
   - Mostrar relatório de drift
   - Demonstrar métricas do sistema

6. **Conclusão (30s)**
   - Resumir benefícios
   - Mencionar possibilidades de deploy

### Comandos para o Vídeo

```bash
# Setup e execução rápida
make quick-start

# Demonstrar Docker
make docker-build
make docker-run

# Gerar relatório de drift
make monitor

# Executar testes
make test
```

## 🤝 Contribuição

### Desenvolvimento

1. **Setup do ambiente**:
   ```bash
   make dev-setup
   ```

2. **Verificações de qualidade**:
   ```bash
   make dev-check  # Formatar, lint, type-check e testes
   ```

3. **Workflow de desenvolvimento**:
   ```bash
   # Fazer alterações
   make format     # Formatar código
   make lint       # Verificar linting
   make test       # Executar testes
   ```

### Padrões de Código

- **Formatação**: Black com linha de 88 caracteres
- **Linting**: Ruff com regras rigorosas
- **Type Hints**: Obrigatório em todas as funções
- **Docstrings**: Documentação completa
- **Testes**: Cobertura mínima de 80%

### Estrutura de Commits

```
type(scope): description

feat(api): add new prediction endpoint
fix(model): resolve training convergence issue
docs(readme): update installation instructions
test(api): add integration tests for health endpoint
```

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

Para dúvidas ou problemas:

1. Verifique a [documentação da API](http://localhost:8000/docs)
2. Consulte os [logs do sistema](logs/)
3. Execute os [testes](tests/) para verificar o ambiente
4. Abra uma issue no repositório

---

**Decision AI** - Transformando recrutamento com inteligência artificial 🚀
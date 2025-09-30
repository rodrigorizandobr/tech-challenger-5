# Decision AI - Sistema de Recrutamento

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

- **IA Avançada**: Utiliza algoritmos de machine learning (Random Forest, Gradient Boosting, etc.) treinados com dados reais
- **API RESTful**: Interface FastAPI com documentação automática em português
- **Dados Reais**: Processamento de 45.071 candidatos reais e histórico de processos seletivos
- **Engenharia de Features**: Extração automática de características relevantes dos dados reais
- **Monitoramento**: Detecção de drift de dados com Evidently AI
- **Containerização**: Deploy com Docker e Docker Compose
- **Testes Automatizados**: Cobertura completa com pytest
- **Fallback Inteligente**: Geração automática de dados sintéticos quando dados reais não estão disponíveis
- **Multilíngue**: Sistema com logs e comentários em português brasileiro

## ✨ Funcionalidades

### Funcionalidades Principais

- ✅ **Predição de Compatibilidade**: Calcula probabilidade de compatibilidade candidato-vaga baseada em dados reais
- ✅ **Análise de Fatores**: Identifica quais aspectos influenciam a compatibilidade (habilidades, experiência, localização, salário)
- ✅ **Recomendações**: Fornece sugestões baseadas na análise de 45k candidatos reais
- ✅ **Processamento de Dados Reais**: Extração automática de features de applicants.json, prospects.json, jobs.json
- ✅ **Validação de Dados**: Validação robusta com Pydantic adaptada para estrutura real
- ✅ **Registro Estruturado**: Logs detalhados em português brasileiro para auditoria
- ✅ **Fallback Inteligente**: Sistema funciona com dados sintéticos quando dados reais não estão disponíveis

### Monitoramento e Observabilidade

- ✅ **Health Check**: Endpoint para verificação de saúde do sistema
- ✅ **Métricas**: Estatísticas de uso e performance
- ✅ **Drift Detection**: Monitoramento de mudanças nos dados
- ✅ **Relatórios**: Dashboards visuais com Evidently

### DevOps e Qualidade

- ✅ **Pronto para CI/CD**: Configuração para GitHub Actions
- ✅ **Containerização**: Docker multi-estágio para produção
- ✅ **Testes**: Cobertura completa com pytest
- ✅ **Análise de Código**: Black e Ruff para qualidade de código
- ✅ **Tipagem Estática**: Tipagem estática completa

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend/     │    │   Aplicação     │    │   Pipeline ML   │
│   Postman       │───▶│   FastAPI       │───▶│   (sklearn)     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Monitoramento │    │ Armazenamento   │
                       │   (Evidently)   │    │   de Dados      │
                       └─────────────────┘    │   (CSV/Joblib)  │
                                              └─────────────────┘
```

### Stack Tecnológica

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **ML**: scikit-learn, pandas, numpy
- **Monitoramento**: Evidently AI, Loguru
- **Validação**: Pydantic
- **Testes**: pytest, httpx
- **Containerização**: Docker, Docker Compose
- **Qualidade**: Black, Ruff, MyPy

## 📊 Dados

### Dados Reais (Produção)
O sistema foi completamente adaptado para trabalhar com dados reais da Decision:
- **applicants.json**: 45.071 candidatos reais com informações completas (194MB)
- **prospects.json**: Histórico detalhado de candidaturas e status de processos seletivos
- **jobs.json**: Vagas reais com requisitos técnicos e comportamentais detalhados

### Estrutura dos Dados Reais

#### Candidatos (applicants.json)
- **Informações Pessoais**: Nome, idade, localização, contatos
- **Formação Acadêmica**: Cursos, instituições, períodos
- **Experiência Profissional**: Empresas, cargos, períodos, responsabilidades
- **Habilidades Técnicas**: Linguagens de programação, frameworks, ferramentas
- **Habilidades Comportamentais**: Soft skills identificadas
- **Preferências**: Modalidade de trabalho, expectativa salarial, disponibilidade

#### Vagas (jobs.json)
- **Requisitos Técnicos**: Tecnologias obrigatórias e desejáveis
- **Experiência**: Nível de senioridade exigido
- **Localização**: Cidade, estado, modalidade (presencial/remoto/híbrido)
- **Benefícios**: Pacote de benefícios oferecido
- **Descrição**: Responsabilidades e desafios da posição

#### Histórico de Processos (prospects.json)
- **Candidaturas**: Relacionamento candidato-vaga
- **Status**: Etapas do processo seletivo
- **Feedback**: Avaliações e comentários dos recrutadores
- **Resultados**: Aprovações, reprovações e motivos

⚠️ **Nota**: Os arquivos de dados reais não estão incluídos no repositório devido ao tamanho (>100MB). Para usar o sistema:

1. **Obtenha os dados reais** e coloque na pasta `data/`
2. **Execute o treinamento**: `python src/train.py`
3. **Inicie a API**: `uvicorn app.main:app --reload`

### Dados Sintéticos (Fallback)
Se os dados reais não estiverem disponíveis, o sistema gera dados sintéticos automaticamente que simulam a estrutura real:
- **Candidatos**: Informações pessoais, educação, experiência, habilidades técnicas e comportamentais
- **Vagas**: Requisitos técnicos, localização, salário, modalidade de trabalho
- **Matches**: Histórico de compatibilidade entre candidatos e vagas baseado em critérios reais

### Processamento e Features
O sistema extrai automaticamente features relevantes dos dados reais:
- **Match de Habilidades**: Compatibilidade entre skills do candidato e requisitos da vaga
- **Experiência**: Análise de senioridade e tempo de experiência
- **Localização**: Compatibilidade geográfica e preferências de trabalho remoto
- **Salário**: Alinhamento entre expectativa e oferta
- **Perfil Comportamental**: Análise de soft skills e fit cultural

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
# 1. Processar dados reais (se disponíveis) ou gerar sintéticos
python -m src.data

# 2. Treinar modelo com dados reais ou sintéticos
python -m src.train

# 3. Executar API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opção 3: Treinamento Rápido com Dados Reais

```bash
# Script otimizado para dados reais (se disponíveis)
python quick_train.py
```

### Opção 4: Docker

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

## 📡 Endpoints da API

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
  "recommendation": "Excelente compatibilidade - fortemente recomendado para entrevista",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Coleção Postman

Importe o arquivo `postman_collection.json` no Postman para testar todos os endpoints com exemplos pré-configurados.

## 📊 Monitoramento

### Sistema de Monitoramento Ativo

O sistema possui um robusto sistema de monitoramento em tempo real que inclui:

✅ **Monitoramento em Tempo Real**: Sistema ativo com métricas atualizadas automaticamente  
✅ **Detecção de Drift**: Análise contínua de mudanças nos padrões de dados  
✅ **Endpoints Funcionais**: APIs de monitoramento prontas para uso  

#### Detecção de Drift de Dados

O sistema monitora automaticamente mudanças nos padrões de dados usando Evidently AI:

```bash
# Instalar dependência de monitoramento
pip install evidently

# Gerar relatório de drift
make monitor

# Ou manualmente
python -m monitor.generate_report \
  --reference-data data/sample_candidates.csv \
  --predictions-log logs/predictions.csv \
  --reports-dir reports/ \
  --window-size 100
```

#### Endpoints de Monitoramento

| Endpoint | Descrição | Exemplo de Resposta |
|----------|-----------|-------------------|
| `/health` | Status de saúde do sistema | `{"status": "healthy", "timestamp": "..."}` |
| `/metrics` | Métricas em tempo real | `{"total_predictions": 15, "avg_match_probability": 0.437}` |
| `/drift-report` | Relatório visual de drift | HTML interativo com análise detalhada |

#### Métricas em Tempo Real

Acesse as métricas através do endpoint `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Exemplo#### Exemplo de Métricas

```bash
# Acessar métricas via API
curl http://localhost:8000/metrics
```

```json
{
  "total_predictions": 15,
  "avg_match_probability": 0.437,
  "predictions_last_24h": 15,
  "model_accuracy": 0.511,
  "drift_detected": false,
  "last_retrain": null,
  "system_health": "healthy"
}
```

#### Monitoramento Contínuo

O sistema executa monitoramento contínuo com:
- **Coleta Automática**: Todas as predições são automaticamente registradas
- **Análise em Tempo Real**: Métricas atualizadas a cada requisição
- **Detecção Proativa**: Identificação automática de anomalias e drift
- **Alertas Inteligentes**: Notificações baseadas em thresholds configuráveis

- **Total de Predições**: Número total de predições realizadas
- **Probabilidade Média de Match**: Média das probabilidades calculadas
- **Predições nas Últimas 24h**: Volume de predições recentes
- **Acurácia do Modelo**: Performance atual do modelo
- **Status de Drift**: Detecção automática de mudanças nos dados
- **Último Retreinamento**: Timestamp do último treinamento
- **Saúde do Sistema**: Status geral (healthy/warning/critical)

#### Relatórios de Drift

##### Acesso via API
```bash
# Visualizar relatório de drift traduzido
curl http://localhost:8000/drift-report
```

##### Relatórios Gerados
- **`reports/drift.html`**: Relatório principal traduzido para português brasileiro
- **`reports/drift_report_YYYYMMDD_HHMMSS.html`**: Relatórios com timestamp

##### Características dos Relatórios
- 🇧🇷 **Interface em Português**: Completamente traduzido para português brasileiro
- 📊 **Análise Detalhada**: Estatísticas por feature com detecção de drift
- 💰 **Formatação Monetária**: Valores em reais (R$) formatados adequadamente
- 🔍 **Interpretação**: Seções de interpretação e recomendações incluídas
- ⚠️ **Alertas Visuais**: Indicadores claros de drift detectado

Os relatórios são gerados automaticamente e incluem:
- **Análise de Drift por Feature**: Detecção de mudanças em cada variável
- **Estatísticas Descritivas**: Comparação entre dados de referência e atuais
- **Visualizações Interativas**: Gráficos e tabelas para análise
- **Recomendações**: Sugestões para ação baseadas nos resultados
- **Alertas Automáticos**: Notificações quando drift é detectado

### Logs Estruturados

Todos os logs são estruturados e salvos em:
- `logs/api.log` - Logs da aplicação em português
- `logs/predictions.csv` - Log detalhado de todas as predições
- `reports/drift.html` - Relatório principal de drift
- `reports/drift_report_*.html` - Relatórios históricos com timestamp

### Status Atual do Sistema

✅ **Sistema Totalmente Operacional**: Monitoramento ativo e funcional  
✅ **15+ Predições Registradas**: Dados reais coletados e analisados  
✅ **Relatórios Gerados**: Análise de drift disponível em tempo real  
✅ **APIs Funcionais**: Todos os endpoints de monitoramento ativos  
✅ **Logs Estruturados**: Sistema de logging completo em português  

### Monitoramento Proativo

O sistema implementa monitoramento proativo com:
- **Coleta**: Registro automático de todas as predições em `logs/predictions.csv`
- **Análise**: Processamento contínuo dos dados coletados com Evidently AI
- **Detecção**: Identificação automática de drift e anomalias em tempo real
- **Ação**: Alertas e recomendações para retreinamento baseados em thresholds
- **Visualização**: Relatórios HTML interativos com análise detalhada

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
│   ├── applicants.json    # Candidatos reais (não versionado)
│   ├── prospects.json     # Histórico de processos (não versionado)
│   ├── jobs.json         # Vagas reais (não versionado)
│   ├── sample_candidates.csv # Dados sintéticos de fallback
│   └── sample_payload.json   # Exemplo de payload para API
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
[Vídeo Demonstrativo](inserir URL aqui)

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
Desenvolvido para o curso MLE Turma 4 - FIAP

---

**Estudantes**: Rodrigo Matheus da Silva (rodrigorizando@gmail.com) e Vitor Efigênio Neto (vitorefigenio@gmail.com)
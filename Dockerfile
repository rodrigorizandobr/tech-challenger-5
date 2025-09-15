# Dockerfile multi-estágio para Sistema de IA de Recrutamento Decision

# Estágio 1: Estágio de construção
FROM python:3.11-slim as builder

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Cria e define diretório de trabalho
WORKDIR /app

# Copia requirements primeiro para melhor cache
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte
COPY . .

# Cria diretórios necessários
RUN mkdir -p logs models reports data

# Gera dados sintéticos e treina modelo
RUN python -m src.data && \
    python -m src.train --data-path data/sample_candidates.csv --model-path models/model.joblib

# Estágio 2: Estágio de produção
FROM python:3.11-slim as production

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.local/bin:$PATH" \
    PYTHONPATH="/app"

# Instala apenas dependências de execução
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Cria usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Cria e define diretório de trabalho
WORKDIR /app

# Copia dependências Python do estágio de construção
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia código da aplicação e arquivos gerados
COPY --from=builder /app/src ./src
COPY --from=builder /app/app ./app
COPY --from=builder /app/monitor ./monitor
COPY --from=builder /app/models ./models
COPY --from=builder /app/data ./data
COPY --from=builder /app/requirements.txt .
COPY --from=builder /app/pyproject.toml .

# Cria diretórios e define permissões
RUN mkdir -p logs reports && \
    chown -R appuser:appuser /app

# Muda para usuário não-root
USER appuser

# Verificação de saúde
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expõe porta
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
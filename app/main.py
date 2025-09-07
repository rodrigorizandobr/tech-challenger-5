"""FastAPI application for recruitment AI system."""

import os
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import joblib

from src.schema import (
    PredictionRequest, PredictionResponse, HealthResponse, 
    MetricsResponse, ErrorResponse, PredictionLog
)
from src.features import engineer_features


# Variáveis globais para modelo e métricas
model = None
model_metadata = {}
app_start_time = datetime.now()
prediction_logs: List[PredictionLog] = []
metrics_cache = {
    'total_predictions': 0,
    'predictions_24h': 0,
    'avg_match_probability': 0.0,
    'last_update': datetime.now()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador do ciclo de vida da aplicação."""
    # Inicialização
    logger.info("Iniciando aplicação FastAPI")
    await load_model()
    
    # Criando diretórios necessários
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    yield
    
    # Finalização
    logger.info("Finalizando aplicação FastAPI")
    await save_prediction_logs()


# Cria aplicação FastAPI
app = FastAPI(
    title="Decision AI - Recruitment System",
    description="AI-powered recruitment and candidate-job matching system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Adiciona middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Monta arquivos estáticos para servir relatórios
if Path("reports").exists():
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")


async def load_model() -> None:
    """Carrega o modelo treinado e metadados."""
    global model, model_metadata
    
    try:
        model_path = os.getenv("MODEL_PATH", "models/model.joblib")
        metadata_path = os.getenv("METADATA_PATH", "models/training_metadata.json")
        
        if not Path(model_path).exists():
            logger.warning(f"Arquivo do modelo não encontrado em {model_path}")
            model = None
            return
        
        # Carrega modelo
        model = joblib.load(model_path)
        logger.info(f"Modelo carregado com sucesso de {model_path}")
        
        # Carrega metadados se disponível
        if Path(metadata_path).exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Metadados do modelo carregados de {metadata_path}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        model = None
        model_metadata = {}


async def save_prediction_logs() -> None:
    """Salva logs de predição em arquivo CSV."""
    try:
        if not prediction_logs:
            return
        
        logs_path = Path("logs/predictions.csv")
        logs_path.parent.mkdir(exist_ok=True)
        
        # Converte logs para DataFrame
        log_data = []
        for log in prediction_logs:
            log_dict = {
                'request_id': log.request_id,
                'timestamp': log.timestamp.isoformat(),
                'processing_time_ms': log.processing_time_ms,
                'model_version': log.model_version,
                'match_probability': log.prediction.match_probability,
                'match_label': log.prediction.match_label,
                'confidence': log.prediction.confidence,
                'recommendation': log.prediction.recommendation,
                # Achata dados do candidato
                **{f'candidate_{k}': v for k, v in log.candidate_data.items()},
                # Achata dados da vaga
                **{f'job_{k}': v for k, v in log.job_data.items()}
            }
            log_data.append(log_dict)
        
        df = pd.DataFrame(log_data)
        
        # Anexa ao arquivo existente ou cria novo
        if logs_path.exists():
            existing_df = pd.read_csv(logs_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(logs_path, index=False)
        logger.info(f"Salvos {len(prediction_logs)} logs de predição em {logs_path}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar logs de predição: {e}")


def update_metrics(prediction: PredictionResponse) -> None:
    """Atualiza métricas da aplicação."""
    global metrics_cache
    
    metrics_cache['total_predictions'] += 1
    
    # Atualiza probabilidade média de match (média móvel)
    current_avg = metrics_cache['avg_match_probability']
    total_preds = metrics_cache['total_predictions']
    
    new_avg = ((current_avg * (total_preds - 1)) + prediction.match_probability) / total_preds
    metrics_cache['avg_match_probability'] = new_avg
    
    # Conta predições nas últimas 24 horas
    cutoff_time = datetime.now() - timedelta(hours=24)
    recent_predictions = [
        log for log in prediction_logs 
        if log.timestamp >= cutoff_time
    ]
    metrics_cache['predictions_24h'] = len(recent_predictions)
    metrics_cache['last_update'] = datetime.now()


def create_prediction_features(request: PredictionRequest) -> pd.DataFrame:
    """Converte requisição de predição para DataFrame de features."""
    # Converte requisição para formato de dicionário similar aos dados de treinamento
    data = {
        'age': request.candidate.age,
        'education_level': request.candidate.education_level.value,
        'years_experience': request.candidate.years_experience,
        'previous_companies': request.candidate.previous_companies,
        'salary_expectation': request.candidate.salary_expectation,
        'availability_days': request.candidate.availability_days,
        'candidate_location': request.candidate.location,
        'remote_work': request.candidate.remote_work,
        'candidate_skills': ','.join(request.candidate.skills),
        
        'required_experience': request.job.required_experience.value,
        'required_skills': ','.join(request.job.required_skills),
        'salary_range_min': request.job.salary_range_min,
        'salary_range_max': request.job.salary_range_max,
        'job_location': request.job.location,
        'remote_allowed': request.job.remote_allowed,
        'urgency_days': request.job.urgency_days,
        
        # Adiciona features derivadas que serão calculadas pela engenharia de features
        'education_numeric': {
            'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4
        }[request.candidate.education_level.value],
        
        'experience_level_numeric': {
            'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4
        }[request.job.required_experience.value]
    }
    
    # Cria DataFrame
    df = pd.DataFrame([data])
    
    # Aplica engenharia de features
    df_engineered = engineer_features(df)
    
    return df_engineered


def extract_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features used by the model (from training metadata)."""
    # Usa features exatas que o modelo espera
    feature_cols = [
        'age', 'education_numeric', 'years_experience', 'skills_count',
        'skills_match_ratio', 'previous_companies', 'salary_expectation',
        'salary_fit', 'location_match', 'remote_compatibility',
        'availability_urgency_ratio', 'experience_level_numeric', 'skill_diversity',
        'rare_skills_bonus', 'salary_position', 'salary_expectation_ratio',
        'experience_education_ratio', 'salary_range_width', 'company_stability'
    ]
    
    # Filtra para colunas disponíveis
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < len(feature_cols):
        missing_cols = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing feature columns: {missing_cols}")
        # Preenche colunas faltantes com valores padrão
        for col in missing_cols:
            df[col] = 0.0
        available_cols = feature_cols
    
    return df[available_cols]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra todas as requisições."""
    start_time = time.time()
    
    # Processa requisição
    response = await call_next(request)
    
    # Registra detalhes da requisição
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    """Endpoint raiz com informações da API."""
    model_status = "Yes" if model is not None else "No"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision AI - Recruitment System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ color: #2c3e50; }}
            .endpoint {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .method {{ color: #27ae60; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1 class="header">Decision AI - Recruitment System</h1>
        <p>AI-powered recruitment and candidate-job matching system</p>
        
        <h2>Available Endpoints:</h2>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong> - Health check
        </div>
        <div class="endpoint">
            <span class="method">POST</span> <strong>/predict</strong> - Make prediction
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/metrics</strong> - System metrics
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/drift-report</strong> - Data drift report
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/docs</strong> - API documentation
        </div>
        
        <h2>Model Status:</h2>
        <p>Model loaded: <strong>{model_status}</strong></p>
        <p>Version: <strong>1.0.0</strong></p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de verificação de saúde."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, 
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """Faz uma predição para combinação candidato-vaga."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Verifica se o modelo está carregado
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Modelo não disponível. Verifique o carregamento do modelo."
            )
        
        # Cria features a partir da requisição
        df_features = create_prediction_features(request)
        
        # Extrai features do modelo
        X = extract_model_features(df_features)
        
        # Makes prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction_label = model.predict(X)[0]
        
        # Gets probability for positive class (good_match)
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if 'good_match' in classes:
                good_match_idx = list(classes).index('good_match')
                match_probability = prediction_proba[good_match_idx]
            else:
                match_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        else:
            match_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        # Calculates confidence (maximum probability)
        confidence = float(np.max(prediction_proba))
        
        # Calculates contributing factors
        factors = {
            'skills_match': float(df_features['skills_match_ratio'].iloc[0]),
            'experience_match': min(1.0, float(df_features['years_experience'].iloc[0]) / 5.0),
            'salary_fit': float(df_features['salary_fit'].iloc[0]),
            'location_compatibility': float(df_features['location_match'].iloc[0]),
            'remote_compatibility': float(df_features['remote_compatibility'].iloc[0])
        }
        
        # Generates recommendation
        if match_probability > 0.8:
            recommendation = "Excelente combinação - fortemente recomendado para entrevista"
        elif match_probability > 0.65:
            recommendation = "Boa combinação - recomendado para entrevista"
        elif match_probability > 0.4:
            recommendation = "Combinação moderada - considerar para entrevista"
        else:
            recommendation = "Combinação ruim - não recomendado"
        
        # Creates response
        response = PredictionResponse(
            match_probability=float(match_probability),
            match_label=prediction_label,
            confidence=confidence,
            factors=factors,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        
        # Calculates processing time
        processing_time = (time.time() - start_time) * 1000  # Converte para milissegundos
        
        # Creates prediction log
        prediction_log = PredictionLog(
            request_id=request_id,
            candidate_data=request.candidate.dict(),
            job_data=request.job.dict(),
            prediction=response,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            model_version=model_metadata.get('best_model_name', 'unknown')
        )
        
        # Stores log and updates metrics
        prediction_logs.append(prediction_log)
        update_metrics(response)
        
        # Schedule background task to save logs periodically
        if len(prediction_logs) % 10 == 0:  # Salva a cada 10 predições
            background_tasks.add_task(save_prediction_logs)
        
        logger.info(
            f"Predição concluída - ID: {request_id}, "
            f"Probabilidade: {match_probability:.3f}, "
            f"Tempo: {processing_time:.1f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor durante predição: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Obter métricas do sistema."""
    try:
        # Check drift (simplified verification)
        drift_detected = False
        if len(prediction_logs) > 50:
            recent_probs = [log.prediction.match_probability for log in prediction_logs[-50:]]
            older_probs = [log.prediction.match_probability for log in prediction_logs[-100:-50]] if len(prediction_logs) > 100 else []
            
            if older_probs:
                recent_avg = np.mean(recent_probs)
                older_avg = np.mean(older_probs)
                drift_detected = abs(recent_avg - older_avg) > 0.1
        
        return MetricsResponse(
            total_predictions=metrics_cache['total_predictions'],
            avg_match_probability=metrics_cache['avg_match_probability'],
            predictions_last_24h=metrics_cache['predictions_24h'],
            model_accuracy=model_metadata.get('best_score'),
            drift_detected=drift_detected,
            last_retrain=None,  # Seria definido quando o retreinamento for implementado
            system_health="healthy" if model is not None else "unhealthy"
        )
        
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {e}")
        raise HTTPException(status_code=500, detail="Erro ao recuperar métricas")


@app.get("/drift-report")
async def get_drift_report():
    """Servir o relatório de drift de dados."""
    report_path = Path("reports/drift.html")
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Relatório de drift não encontrado. Execute o script de monitoramento para gerá-lo."
        )
    
    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename="drift_report.html"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global de exceções."""
    logger.error(f"Exceção não tratada: {exc}")
    
    from fastapi.responses import JSONResponse
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Ocorreu um erro inesperado",
            "details": {"path": str(request.url.path)},
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add(
        "logs/api.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )
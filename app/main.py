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


# Global variables for model and metrics
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
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up FastAPI application")
    await load_model()
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")
    await save_prediction_logs()


# Create FastAPI app
app = FastAPI(
    title="Decision AI - Recruitment System",
    description="AI-powered recruitment and candidate-job matching system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
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

# Mount static files for serving reports
if Path("reports").exists():
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")


async def load_model() -> None:
    """Load the trained model and metadata."""
    global model, model_metadata
    
    try:
        model_path = os.getenv("MODEL_PATH", "models/model.joblib")
        metadata_path = os.getenv("METADATA_PATH", "models/training_metadata.json")
        
        if not Path(model_path).exists():
            logger.warning(f"Model file not found at {model_path}")
            model = None
            return
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Load metadata if available
        if Path(metadata_path).exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        model_metadata = {}


async def save_prediction_logs() -> None:
    """Save prediction logs to CSV file."""
    try:
        if not prediction_logs:
            return
        
        logs_path = Path("logs/predictions.csv")
        logs_path.parent.mkdir(exist_ok=True)
        
        # Convert logs to DataFrame
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
                # Flatten candidate data
                **{f'candidate_{k}': v for k, v in log.candidate_data.items()},
                # Flatten job data
                **{f'job_{k}': v for k, v in log.job_data.items()}
            }
            log_data.append(log_dict)
        
        df = pd.DataFrame(log_data)
        
        # Append to existing file or create new one
        if logs_path.exists():
            existing_df = pd.read_csv(logs_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(logs_path, index=False)
        logger.info(f"Saved {len(prediction_logs)} prediction logs to {logs_path}")
        
    except Exception as e:
        logger.error(f"Error saving prediction logs: {e}")


def update_metrics(prediction: PredictionResponse) -> None:
    """Update application metrics."""
    global metrics_cache
    
    metrics_cache['total_predictions'] += 1
    
    # Update average match probability (running average)
    current_avg = metrics_cache['avg_match_probability']
    total_preds = metrics_cache['total_predictions']
    
    new_avg = ((current_avg * (total_preds - 1)) + prediction.match_probability) / total_preds
    metrics_cache['avg_match_probability'] = new_avg
    
    # Count predictions in last 24 hours
    cutoff_time = datetime.now() - timedelta(hours=24)
    recent_predictions = [
        log for log in prediction_logs 
        if log.timestamp >= cutoff_time
    ]
    metrics_cache['predictions_24h'] = len(recent_predictions)
    metrics_cache['last_update'] = datetime.now()


def create_prediction_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert prediction request to feature DataFrame."""
    # Convert request to dictionary format similar to training data
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
        
        # Add derived features that will be calculated by feature engineering
        'education_numeric': {
            'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4
        }[request.candidate.education_level.value],
        
        'experience_level_numeric': {
            'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4
        }[request.job.required_experience.value]
    }
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    return df_engineered


def extract_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features used by the model."""
    feature_cols = [
        'age', 'education_numeric', 'years_experience', 'skills_count',
        'skills_match_ratio', 'previous_companies', 'salary_expectation',
        'salary_fit', 'location_match', 'remote_compatibility',
        'availability_urgency_ratio', 'experience_level_numeric',
        'skill_diversity', 'rare_skills_bonus', 'salary_position',
        'salary_expectation_ratio', 'experience_education_ratio',
        'salary_range_width', 'company_stability'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < len(feature_cols):
        missing_cols = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing feature columns: {missing_cols}")
    
    return df[available_cols]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision AI - Recruitment System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
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
    """.format(
        model_status="Yes" if model is not None else "No"
    )
    
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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
    """Make a prediction for candidate-job match."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not available. Please check model loading."
            )
        
        # Create features from request
        df_features = create_prediction_features(request)
        
        # Extract model features
        X = extract_model_features(df_features)
        
        # Make prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction_label = model.predict(X)[0]
        
        # Get probability for positive class (good_match)
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if 'good_match' in classes:
                good_match_idx = list(classes).index('good_match')
                match_probability = prediction_proba[good_match_idx]
            else:
                match_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        else:
            match_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        # Calculate confidence (max probability)
        confidence = float(np.max(prediction_proba))
        
        # Calculate contributing factors
        factors = {
            'skills_match': float(df_features['skills_match_ratio'].iloc[0]),
            'experience_match': min(1.0, float(df_features['years_experience'].iloc[0]) / 5.0),
            'salary_fit': float(df_features['salary_fit'].iloc[0]),
            'location_compatibility': float(df_features['location_match'].iloc[0]),
            'remote_compatibility': float(df_features['remote_compatibility'].iloc[0])
        }
        
        # Generate recommendation
        if match_probability > 0.8:
            recommendation = "Excellent match - strongly recommend for interview"
        elif match_probability > 0.65:
            recommendation = "Good match - recommend for interview"
        elif match_probability > 0.4:
            recommendation = "Moderate match - consider for interview"
        else:
            recommendation = "Poor match - not recommended"
        
        # Create response
        response = PredictionResponse(
            match_probability=float(match_probability),
            match_label=prediction_label,
            confidence=confidence,
            factors=factors,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create prediction log
        prediction_log = PredictionLog(
            request_id=request_id,
            candidate_data=request.candidate.dict(),
            job_data=request.job.dict(),
            prediction=response,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            model_version=model_metadata.get('best_model_name', 'unknown')
        )
        
        # Store log and update metrics
        prediction_logs.append(prediction_log)
        update_metrics(response)
        
        # Schedule background task to save logs periodically
        if len(prediction_logs) % 10 == 0:  # Save every 10 predictions
            background_tasks.add_task(save_prediction_logs)
        
        logger.info(
            f"Prediction completed - ID: {request_id}, "
            f"Probability: {match_probability:.3f}, "
            f"Time: {processing_time:.1f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    try:
        # Check for drift (simplified check)
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
            last_retrain=None,  # Would be set when retraining is implemented
            system_health="healthy" if model is not None else "unhealthy"
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")


@app.get("/drift-report")
async def get_drift_report():
    """Serve the data drift report."""
    report_path = Path("reports/drift.html")
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Drift report not found. Run the monitoring script to generate it."
        )
    
    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename="drift_report.html"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        details={"path": str(request.url.path)},
        timestamp=datetime.now()
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
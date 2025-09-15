#!/usr/bin/env python3
"""Script de treinamento rápido super básico - Random Forest mínimo."""

import joblib
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from src.data import load_and_validate_data, split_data
from src.features import engineer_features

def super_quick_train():
    """Treinamento rápido super básico com Random Forest mínimo."""
    logger.info("🚀 Iniciando treinamento rápido super básico com Random Forest")
    
    # Carrega e prepara dados
    logger.info("📊 Carregando dados...")
    df = load_and_validate_data("data/sample_candidates.csv")
    df = engineer_features(df)
    train_df, test_df = split_data(df)
    
    # Obtém apenas features numéricas simples
    feature_cols = [
        'age', 'education_numeric', 'years_experience', 'skills_count',
        'skills_match_ratio', 'previous_companies', 'salary_expectation',
        'salary_fit', 'location_match', 'remote_compatibility',
        'availability_urgency_ratio', 'experience_level_numeric'
    ]
    
    # Filtra apenas colunas existentes
    available_cols = [col for col in feature_cols if col in train_df.columns]
    logger.info(f"✅ Usando {len(available_cols)} features: {available_cols}")
    
    X_train = train_df[available_cols]
    y_train = train_df['match_label']
    X_test = test_df[available_cols]
    y_test = test_df['match_label']
    
    # Preenche valores NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Treina Random Forest simples
    logger.info("🌲 Treinando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("✅ Treinamento do Random Forest concluído")
    
    # Avalia
    logger.info("📈 Avaliando modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred, pos_label='good_match')
    roc_auc = roc_auc_score(y_test == 'good_match', y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"🎯 F1 Score: {f1:.4f}")
    logger.info(f"🎯 ROC AUC: {roc_auc:.4f}")
    logger.info(f"🎯 Accuracy: {accuracy:.4f}")
    
    # Calibra modelo
    logger.info("🔧 Calibrando modelo...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Salva modelo
    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Salvando modelo em {model_path}")
    joblib.dump(calibrated_model, model_path)
    
    # Salva metadados
    metadata = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "parameters": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 4
        },
        "test_metrics": {
            "f1_score": f1,
            "roc_auc": roc_auc,
            "accuracy": accuracy
        },
        "feature_names": available_cols,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "calibrated": True
    }
    
    metadata_path = Path("models/training_metadata.json")
    logger.info(f"📄 Salvando metadados em {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("🎉 Treinamento rápido super básico concluído com sucesso!")
    logger.info(f"🏆 Modelo final: Random Forest (F1: {f1:.4f})")
    
    return calibrated_model, metadata

if __name__ == "__main__":
    super_quick_train()
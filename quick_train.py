#!/usr/bin/env python3
"""Super quick training script - minimal Random Forest."""

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
    """Super quick training with minimal Random Forest."""
    logger.info("🚀 Starting SUPER QUICK training with Random Forest")
    
    # Load and prepare data
    logger.info("📊 Loading data...")
    df = load_and_validate_data("data/sample_candidates.csv")
    df = engineer_features(df)
    train_df, test_df = split_data(df)
    
    # Get simple numeric features only
    feature_cols = [
        'age', 'education_numeric', 'years_experience', 'skills_count',
        'skills_match_ratio', 'previous_companies', 'salary_expectation',
        'salary_fit', 'location_match', 'remote_compatibility',
        'availability_urgency_ratio', 'experience_level_numeric'
    ]
    
    # Filter only existing columns
    available_cols = [col for col in feature_cols if col in train_df.columns]
    logger.info(f"✅ Using {len(available_cols)} features: {available_cols}")
    
    X_train = train_df[available_cols]
    y_train = train_df['match_label']
    X_test = test_df[available_cols]
    y_test = test_df['match_label']
    
    # Fill any NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Train simple Random Forest
    logger.info("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("✅ Random Forest training completed")
    
    # Evaluate
    logger.info("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred, pos_label='good_match')
    roc_auc = roc_auc_score(y_test == 'good_match', y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"🎯 F1 Score: {f1:.4f}")
    logger.info(f"🎯 ROC AUC: {roc_auc:.4f}")
    logger.info(f"🎯 Accuracy: {accuracy:.4f}")
    
    # Calibrate model
    logger.info("🔧 Calibrating model...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Save model
    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Saving model to {model_path}")
    joblib.dump(calibrated_model, model_path)
    
    # Save metadata
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
    logger.info(f"📄 Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("🎉 SUPER QUICK training completed successfully!")
    logger.info(f"🏆 Final model: Random Forest (F1: {f1:.4f})")
    
    return calibrated_model, metadata

if __name__ == "__main__":
    super_quick_train()
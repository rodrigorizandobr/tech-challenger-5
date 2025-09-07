"""Model training pipeline with sklearn and persistence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import json
from datetime import datetime
from loguru import logger

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV

from src.data import load_and_validate_data, split_data
from src.features import (
    create_feature_engineering_pipeline, engineer_features,
    get_feature_names, save_preprocessor
)


class ModelTrainer:
    """Model training and evaluation class."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.feature_pipeline = None
        self.training_metadata = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize candidate models for training."""
        logger.info("Initializing candidate models")
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        logger.info(f"Initialized {len(self.models)} candidate models")
    
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data.
        
        Args:
            data_path: Path to the training data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load and validate data
        df = load_and_validate_data(data_path)
        
        # Engineer features
        df_engineered = engineer_features(df)
        
        # Split data
        train_df, test_df = split_data(df_engineered, test_size=0.2, random_state=self.random_state)
        
        logger.info(f"Data preparation completed. Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features and target.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features, target)
        """
        # Define feature columns for training
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
        
        X = df[available_cols]
        y = df['match_label']
        
        logger.info(f"Using {len(available_cols)} features for training")
        
        return X, y
    
    def train_model(
        self, 
        train_df: pd.DataFrame, 
        model_name: str = 'random_forest',
        use_grid_search: bool = True
    ) -> Pipeline:
        """Train a specific model.
        
        Args:
            train_df: Training dataframe
            model_name: Name of the model to train
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Trained pipeline
        """
        logger.info(f"Training {model_name} model")
        
        # Prepare features and target
        X_train, y_train = self.get_feature_target_split(train_df)
        
        # Get base model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        base_model = self.models[model_name]
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('model', base_model)
        ])
        
        # Hyperparameter tuning
        if use_grid_search:
            pipeline = self._tune_hyperparameters(pipeline, X_train, y_train, model_name)
        
        # Train final model
        pipeline.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_pipeline = CalibratedClassifierCV(
            pipeline, method='isotonic', cv=3
        )
        calibrated_pipeline.fit(X_train, y_train)
        
        logger.info(f"Model {model_name} training completed")
        
        return calibrated_pipeline
    
    def _tune_hyperparameters(
        self, 
        pipeline: Pipeline, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        model_name: str
    ) -> Pipeline:
        """Tune hyperparameters using grid search.
        
        Args:
            pipeline: Base pipeline
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Pipeline with best parameters
        """
        logger.info(f"Tuning hyperparameters for {model_name}")
        
        # Define parameter grids (optimized for speed)
        param_grids = {
            'random_forest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [10, 15],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [2, 4]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.1, 0.2],
                'model__max_depth': [3, 5],
                'model__subsample': [0.8, 1.0]
            },
            'logistic_regression': {
                'model__C': [1.0, 10.0],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            },
            'svm': {
                'model__C': [1.0, 10.0],
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return pipeline
        
        # Perform grid search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(
        self, 
        model: Pipeline, 
        test_df: pd.DataFrame,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            model: Trained model pipeline
            test_df: Test dataframe
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name} model")
        
        # Prepare test data
        X_test, y_test = self.get_feature_target_split(test_df)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='good_match'),
            'recall': recall_score(y_test, y_pred, pos_label='good_match'),
            'f1_score': f1_score(y_test, y_pred, pos_label='good_match'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_test, y_test, cv=3, scoring='f1_macro'
        )
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        logger.info(f"Model {model_name} evaluation completed")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_and_select_best_model(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Train multiple models and select the best one.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (best_model, evaluation_metrics)
        """
        logger.info("Training and selecting best model")
        
        best_model = None
        best_score = 0.0
        best_metrics = {}
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                logger.info(f"Training {model_name}...")
                
                # Train model
                model = self.train_model(train_df, model_name, use_grid_search=True)
                
                # Evaluate model
                metrics = self.evaluate_model(model, test_df, model_name)
                
                # Store results
                all_results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Check if this is the best model
                f1_score = metrics['f1_score']
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model
                    best_metrics = metrics
                    
                    logger.info(f"New best model: {model_name} (F1: {f1_score:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("No models were successfully trained")
        
        # Store training metadata
        self.training_metadata = {
            'best_model_name': [name for name, result in all_results.items() 
                              if result['model'] is best_model][0],
            'best_score': best_score,
            'all_results': {name: result['metrics'] for name, result in all_results.items()},
            'training_date': datetime.now().isoformat(),
            'data_shape': train_df.shape,
            'feature_count': len(self.get_feature_target_split(train_df)[0].columns)
        }
        
        self.best_model = best_model
        self.best_score = best_score
        
        logger.info(f"Best model selected with F1 score: {best_score:.4f}")
        
        return best_model, best_metrics
    
    def save_model(self, model: Pipeline, model_path: str, metadata_path: str = None) -> None:
        """Save trained model and metadata.
        
        Args:
            model: Trained model pipeline
            model_path: Path to save the model
            metadata_path: Path to save metadata (optional)
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        if metadata_path:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2, default=str)
            
            logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str) -> Pipeline:
        """Load trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model pipeline
        """
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model


def main(
    data_path: str = "data/sample_candidates.csv",
    model_path: str = "models/model.joblib",
    metadata_path: str = "models/training_metadata.json",
    random_state: int = 42
) -> None:
    """Main training function.
    
    Args:
        data_path: Path to training data
        model_path: Path to save the trained model
        metadata_path: Path to save training metadata
        random_state: Random seed
    """
    logger.info("Starting model training pipeline")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(random_state=random_state)
        
        # Prepare data
        train_df, test_df = trainer.prepare_data(data_path)
        
        # Train and select best model
        best_model, metrics = trainer.train_and_select_best_model(train_df, test_df)
        
        # Save model and metadata
        trainer.save_model(best_model, model_path, metadata_path)
        
        # Print final results
        logger.info("Training completed successfully!")
        logger.info(f"Final model metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return best_model, metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train recruitment AI model")
    parser.add_argument(
        "--data-path", 
        default="data/sample_candidates.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--model-path", 
        default="models/model.joblib",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--metadata-path", 
        default="models/training_metadata.json",
        help="Path to save training metadata"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        random_state=args.random_state
    )
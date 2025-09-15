"""Pipeline de treinamento de modelo com sklearn e persistência."""

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

from data import load_and_validate_data, split_data
from features import (
    create_feature_engineering_pipeline, engineer_features,
    get_feature_names, save_preprocessor
)


class ModelTrainer:
    """Classe para treinamento e avaliação de modelos."""
    
    def __init__(self, random_state: int = 42):
        """Inicializa o treinador de modelos.
        
        Args:
            random_state: Seed aleatória para reprodutibilidade
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.feature_pipeline = None
        self.training_metadata = {}
        
        # Inicializa modelos
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Inicializa modelos candidatos para treinamento."""
        logger.info("Inicializando modelos candidatos")
        
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
        
        logger.info(f"Inicializados {len(self.models)} modelos candidatos")
    
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara dados de treinamento.
        
        Args:
            data_path: Caminho para os dados de treinamento
            
        Returns:
            Tupla de (train_df, test_df)
        """
        logger.info(f"Carregando dados de {data_path}")
        
        # Carrega e valida dados
        df = load_and_validate_data(data_path)
        
        # Engenharia de features
        df_engineered = engineer_features(df)
        
        # Divide dados
        train_df, test_df = split_data(df_engineered, test_size=0.2, random_state=self.random_state)
        
        logger.info(f"Preparação de dados concluída. Treinamento: {len(train_df)}, Teste: {len(test_df)}")
        
        return train_df, test_df
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Divide dataframe em features e target.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tupla de (features, target)
        """
        # Define colunas de features para treinamento
        feature_cols = [
            'age', 'education_numeric', 'years_experience', 'skills_count',
            'skills_match_ratio', 'previous_companies', 'salary_expectation',
            'salary_fit', 'location_match', 'remote_compatibility',
            'availability_urgency_ratio', 'experience_level_numeric',
            'skill_diversity', 'rare_skills_bonus', 'salary_position',
            'salary_expectation_ratio', 'experience_education_ratio',
            'salary_range_width', 'company_stability'
        ]
        
        # Filtra para colunas disponíveis
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < len(feature_cols):
            missing_cols = set(feature_cols) - set(available_cols)
            logger.warning(f"Colunas de feature missing: {missing_cols}")
        
        X = df[available_cols]
        y = df['match_label']
        
        logger.info(f"Usando {len(available_cols)} features para treinamento")
        
        return X, y
    
    def train_model(
        self, 
        train_df: pd.DataFrame, 
        model_name: str = 'random_forest',
        use_grid_search: bool = True
    ) -> Pipeline:
        """Treina um modelo específico.
        
        Args:
            train_df: Dataframe de treinamento
            model_name: Nome do modelo a ser treinado
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Pipeline treinado
        """
        logger.info(f"Treinando modelo {model_name}")
        
        # Prepara features e target
        X_train, y_train = self.get_feature_target_split(train_df)
        
        # Obtém modelo base
        if model_name not in self.models:
            raise ValueError(f"Modelo desconhecido: {model_name}")
        
        base_model = self.models[model_name]
        
        # Cria pipeline com pré-processamento
        pipeline = Pipeline([
            ('model', base_model)
        ])
        
        # Ajuste de hiperparâmetros
        if use_grid_search:
            pipeline = self._tune_hyperparameters(pipeline, X_train, y_train, model_name)
        
        # Treina modelo final
        pipeline.fit(X_train, y_train)
        
        # Calibra probabilidades
        calibrated_pipeline = CalibratedClassifierCV(
            pipeline, method='isotonic', cv=3
        )
        calibrated_pipeline.fit(X_train, y_train)
        
        logger.info(f"Treinamento do modelo {model_name} concluído")
        
        return calibrated_pipeline
    
    def _tune_hyperparameters(
        self, 
        pipeline: Pipeline, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        model_name: str
    ) -> Pipeline:
        """Ajusta hiperparâmetros usando grid search.
        
        Args:
            pipeline: Pipeline base
            X_train: Features de treinamento
            y_train: Target de treinamento
            model_name: Nome do modelo
            
        Returns:
            Pipeline com melhores parâmetros
        """
        logger.info(f"Ajustando hiperparâmetros para {model_name}")
        
        # Define grades de parâmetros (otimizadas para velocidade)
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
            logger.warning(f"Nenhum grid de parâmetros definido para {model_name}")
            return pipeline
        
        # Executa busca em grade
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
        
        logger.info(f"Hiperparâmetros ótimos para {model_name}: {grid_search.best_params_}")
        logger.info(f"Melhor pontuação CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(
        self, 
        model: Pipeline, 
        test_df: pd.DataFrame,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Avalia o desempenho do modelo.
        
        Args:
            model: Pipeline do modelo treinado
            test_df: Dataframe de teste
            model_name: Nome do modelo para logging
            
        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info(f"Avaliando desempenho do modelo {model_name}")
        
        # Prepara dados de teste
        X_test, y_test = self.get_feature_target_split(test_df)
        
        # Faz predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
        
        # Calcula métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='good_match'),
            'recall': recall_score(y_test, y_pred, pos_label='good_match'),
            'f1_score': f1_score(y_test, y_pred, pos_label='good_match'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Scores de validação cruzada
        cv_scores = cross_val_score(
            model, X_test, y_test, cv=3, scoring='f1_macro'
        )
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        logger.info(f"Modelo {model_name} avaliado com sucesso")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_and_select_best_model(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Treina múltiplos modelos e seleciona o melhor.
        
        Args:
            train_df: Dataframe de treinamento
            test_df: Dataframe de teste
            
        Returns:
            Tupla de (melhor_modelo, métricas_de_avaliação)
        """
        logger.info("Treinando e selecionando melhor modelo")
        
        best_model = None
        best_score = 0.0
        best_metrics = {}
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                logger.info(f"Treinando {model_name}...")
                
                # Treina modelo
                model = self.train_model(train_df, model_name, use_grid_search=True)
                
                # Avalia modelo
                metrics = self.evaluate_model(model, test_df, model_name)
                
                # Armazena resultados
                all_results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Verifica se este é o melhor modelo
                f1_score = metrics['f1_score']
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model
                    best_metrics = metrics
                    
                    logger.info(f"Novo melhor modelo: {model_name} (F1: {f1_score:.4f})")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("Nenhum modelo foi treinado com sucesso")
        
        # Armazena metadados de treinamento
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
        
        logger.info(f"Melhor modelo selecionado com F1 score: {best_score:.4f}")
        
        return best_model, best_metrics
    
    def save_model(self, model: Pipeline, model_path: str, metadata_path: str = None) -> None:
        """Salva o modelo treinado e seus metadados.
        
        Args:
            model: Pipeline do modelo treinado
            model_path: Caminho para salvar o modelo
            metadata_path: Caminho para salvar metadados (opcional)
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva modelo
        joblib.dump(model, model_path)
        logger.info(f"Modelo salvo em {model_path}")
        
        # Salva metadados
        if metadata_path:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2, default=str)
            
            logger.info(f"Metadados salvos em {metadata_path}")
    
    def load_model(self, model_path: str) -> Pipeline:
        """Carrega o modelo treinado de disco.
        
        Args:
            model_path: Caminho para o modelo salvo
            
        Returns:
            Pipeline do modelo carregado
        """
        model = joblib.load(model_path)
        logger.info(f"Modelo carregado de {model_path}")
        return model


def main(
    data_path: str = "data/sample_candidates.csv",
    model_path: str = "models/model.joblib",
    metadata_path: str = "models/training_metadata.json",
    random_state: int = 42
) -> None:
    """Função principal de treinamento do modelo.
    
    Args:
        data_path: Caminho para os dados de treinamento
        model_path: Caminho para salvar o modelo treinado
        metadata_path: Caminho para salvar metadados de treinamento
        random_state: Seed aleatória para reprodutibilidade
    """
    logger.info("Iniciando pipeline de treinamento do modelo")
    
    try:
        # Inicializa treinador
        trainer = ModelTrainer(random_state=random_state)
        
        # Prepara dados
        train_df, test_df = trainer.prepare_data(data_path)
        
        # Treina e seleciona melhor modelo
        best_model, metrics = trainer.train_and_select_best_model(train_df, test_df)
        
        # Salva modelo e metadados
        trainer.save_model(best_model, model_path, metadata_path)
        
        # Imprime resultados finais
        logger.info("Treinamento concluído com sucesso!")
        logger.info(f"Métricas do modelo final:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return best_model, metrics
        
    except Exception as e:
        logger.error(f"Treinamento falhou: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treine um modelo de IA para recrutamento")
    parser.add_argument(
        "--data-path", 
        default="data/sample_candidates.csv",
        help="Caminho para os dados de treinamento"
    )
    parser.add_argument(
        "--model-path", 
        default="models/model.joblib",
        help="Caminho para salvar o modelo treinado"
    )
    parser.add_argument(
        "--metadata-path", 
        default="models/training_metadata.json",
        help="Caminho para salvar metadados de treinamento"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Seed aleatória para reprodutibilidade"
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        random_state=args.random_state
    )
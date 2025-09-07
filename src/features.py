"""Feature engineering and preprocessing module."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger
import joblib
from pathlib import Path


class SkillsMatchTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for skills matching features."""
    
    def __init__(self):
        self.skill_vocabulary_ = set()
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the transformer by building skill vocabulary.
        
        Args:
            X: DataFrame with candidate_skills and required_skills columns
            y: Target variable (ignored)
            
        Returns:
            self
        """
        logger.info("Ajustando SkillsMatchTransformer")
        
        all_skills = set()
        
        # Extrai habilidades da coluna candidate_skills
        if 'candidate_skills' in X.columns:
            for skills_str in X['candidate_skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [s.strip().lower() for s in skills_str.split(',')]
                    all_skills.update(skills)
        
        # Extrai habilidades da coluna required_skills
        if 'required_skills' in X.columns:
            for skills_str in X['required_skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [s.strip().lower() for s in skills_str.split(',')]
                    all_skills.update(skills)
        
        self.skill_vocabulary_ = all_skills
        self.fitted_ = True
        
        logger.info(f"Built skill vocabulary with {len(self.skill_vocabulary_)} unique skills")
        return self
    
    def transform(self, X):
        """Transform the data by creating skill-based features.
        
        Args:
            X: DataFrame with skills columns
            
        Returns:
            Transformed DataFrame with additional skill features
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Calcula proporção de match de habilidades se não estiver presente
        if 'skills_match_ratio' not in X_transformed.columns:
            X_transformed['skills_match_ratio'] = X_transformed.apply(
                self._calculate_skills_match_ratio, axis=1
            )
        
        # Calcula contagem de habilidades se não estiver presente
        if 'skills_count' not in X_transformed.columns:
            X_transformed['skills_count'] = X_transformed.apply(
                self._calculate_skills_count, axis=1
            )
        
        # Calcula score de diversidade de habilidades
        X_transformed['skill_diversity'] = X_transformed.apply(
            self._calculate_skill_diversity, axis=1
        )
        
        # Calcula bônus de habilidades raras
        X_transformed['rare_skills_bonus'] = X_transformed.apply(
            self._calculate_rare_skills_bonus, axis=1
        )
        
        return X_transformed
    
    def _calculate_skills_match_ratio(self, row) -> float:
        """Calculate the ratio of matching skills."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            required_skills = self._parse_skills(row.get('required_skills', ''))
            
            if not required_skills:
                return 0.5  # Default value when no skills required
            
            matches = len(set(candidate_skills) & set(required_skills))
            return matches / len(required_skills)
        except Exception:
            return 0.0
    
    def _calculate_skills_count(self, row) -> int:
        """Calculate the number of candidate skills."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            return len(candidate_skills)
        except Exception:
            return 0
    
    def _calculate_skill_diversity(self, row) -> float:
        """Calculate skill diversity score based on vocabulary coverage."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            if not self.skill_vocabulary_ or not candidate_skills:
                return 0.0
            
            unique_skills = set(candidate_skills) & self.skill_vocabulary_
            return len(unique_skills) / len(self.skill_vocabulary_)
        except Exception:
            return 0.0
    
    def _calculate_rare_skills_bonus(self, row) -> float:
        """Calculate bonus for having rare/specialized skills."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            required_skills = self._parse_skills(row.get('required_skills', ''))
            
            # Habilidades que o candidato tem mas não são requeridas (especialização)
            extra_skills = set(candidate_skills) - set(required_skills)
            
            # Bônus baseado no número de habilidades extras (limitado a 0.2)
            return min(0.2, len(extra_skills) * 0.02)
        except Exception:
            return 0.0
    
    def _parse_skills(self, skills_str: str) -> List[str]:
        """Parse skills string into list of skills."""
        if not isinstance(skills_str, str) or not skills_str.strip():
            return []
        
        return [s.strip().lower() for s in skills_str.split(',') if s.strip()]


class SalaryFitTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for salary fit features."""
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform salary data into fit features."""
        X_transformed = X.copy()
        
        # Calcula adequação salarial se não estiver presente
        if 'salary_fit' not in X_transformed.columns:
            X_transformed['salary_fit'] = X_transformed.apply(
                self._calculate_salary_fit, axis=1
            )
        
        # Calcula posição salarial dentro da faixa
        X_transformed['salary_position'] = X_transformed.apply(
            self._calculate_salary_position, axis=1
        )
        
        # Calcula proporção de expectativa salarial
        X_transformed['salary_expectation_ratio'] = X_transformed.apply(
            self._calculate_salary_ratio, axis=1
        )
        
        return X_transformed
    
    def _calculate_salary_fit(self, row) -> float:
        """Calculate how well salary expectation fits the range."""
        try:
            salary_exp = row.get('salary_expectation', 0)
            salary_min = row.get('salary_range_min', 0)
            salary_max = row.get('salary_range_max', 0)
            
            if salary_min <= salary_exp <= salary_max:
                return 1.0
            elif salary_exp < salary_min:
                # Candidato espera menos (bom para empregador)
                return min(1.0, 0.8 + (salary_min - salary_exp) / salary_min * 0.2)
            else:
                # Candidato espera mais (penalidade)
                return max(0.1, 1.0 - (salary_exp - salary_max) / salary_max)
        except Exception:
            return 0.5
    
    def _calculate_salary_position(self, row) -> float:
        """Calculate position of expectation within salary range (0-1)."""
        try:
            salary_exp = row.get('salary_expectation', 0)
            salary_min = row.get('salary_range_min', 0)
            salary_max = row.get('salary_range_max', 0)
            
            if salary_max <= salary_min:
                return 0.5
            
            position = (salary_exp - salary_min) / (salary_max - salary_min)
            return np.clip(position, 0, 1)
        except Exception:
            return 0.5
    
    def _calculate_salary_ratio(self, row) -> float:
        """Calculate ratio of expectation to range midpoint."""
        try:
            salary_exp = row.get('salary_expectation', 0)
            salary_min = row.get('salary_range_min', 0)
            salary_max = row.get('salary_range_max', 0)
            
            midpoint = (salary_min + salary_max) / 2
            if midpoint == 0:
                return 1.0
            
            return salary_exp / midpoint
        except Exception:
            return 1.0


class LocationCompatibilityTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for location compatibility features."""
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform location data into compatibility features."""
        X_transformed = X.copy()
        
        # Calcula match de localização se não estiver presente
        if 'location_match' not in X_transformed.columns:
            X_transformed['location_match'] = X_transformed.apply(
                self._calculate_location_match, axis=1
            )
        
        # Calcula compatibilidade remota se não estiver presente
        if 'remote_compatibility' not in X_transformed.columns:
            X_transformed['remote_compatibility'] = X_transformed.apply(
                self._calculate_remote_compatibility, axis=1
            )
        
        return X_transformed
    
    def _calculate_location_match(self, row) -> float:
        """Calculate location compatibility score."""
        try:
            candidate_location = str(row.get('candidate_location', '')).strip().lower()
            job_location = str(row.get('job_location', '')).strip().lower()
            remote_work = row.get('remote_work', False)
            remote_allowed = row.get('remote_allowed', False)
            
            # Match perfeito
            if candidate_location == job_location:
                return 1.0
            
            # Cenários de trabalho remoto
            if job_location == 'remote' or (remote_work and remote_allowed):
                return 1.0
            
            # Same state/region (simplified)
            major_cities = {
                'são paulo': 'sp',
                'rio de janeiro': 'rj',
                'belo horizonte': 'mg',
                'porto alegre': 'rs',
                'salvador': 'ba',
                'brasília': 'df',
                'curitiba': 'pr',
                'recife': 'pe',
                'fortaleza': 'ce',
                'manaus': 'am'
            }
            
            candidate_state = major_cities.get(candidate_location)
            job_state = major_cities.get(job_location)
            
            if candidate_state and job_state and candidate_state == job_state:
                return 0.7
            
            # Localizações diferentes
            return 0.2
        except Exception:
            return 0.5
    
    def _calculate_remote_compatibility(self, row) -> float:
        """Calculate remote work compatibility."""
        try:
            remote_work = row.get('remote_work', False)
            remote_allowed = row.get('remote_allowed', False)
            
            if remote_work and remote_allowed:
                return 1.0
            elif remote_allowed:
                return 0.8  # Job allows remote, candidate may adapt
            elif remote_work:
                return 0.3  # Candidate wants remote, job doesn't allow
            else:
                return 0.6  # Both prefer in-person
        except Exception:
            return 0.5


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Create the complete preprocessing pipeline.
    
    Returns:
        Configured ColumnTransformer pipeline
    """
    logger.info("Criando pipeline de pré-processamento")
    
    # Define colunas de features
    numeric_features = [
        'age', 'years_experience', 'previous_companies', 
        'salary_expectation', 'availability_days', 'urgency_days',
        'salary_range_min', 'salary_range_max'
    ]
    
    categorical_features = [
        'education_level', 'required_experience', 
        'candidate_location', 'job_location'
    ]
    
    boolean_features = [
        'remote_work', 'remote_allowed'
    ]
    
    # Pipeline numérico
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline categórico
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pipeline booleano
    boolean_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=False))
    ])
    
    # Combina todos os pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
            ('bool', boolean_pipeline, boolean_features)
        ],
        remainder='passthrough',  # Keep other columns as-is
        verbose_feature_names_out=False
    )
    
    logger.info(f"Pipeline created with {len(numeric_features)} numeric, "
                f"{len(categorical_features)} categorical, and "
                f"{len(boolean_features)} boolean features")
    
    return preprocessor


def create_feature_engineering_pipeline() -> Pipeline:
    """Create the complete feature engineering pipeline.
    
    Returns:
        Complete feature engineering pipeline
    """
    logger.info("Criando pipeline de engenharia de features")
    
    pipeline = Pipeline([
        ('skills_transformer', SkillsMatchTransformer()),
        ('salary_transformer', SalaryFitTransformer()),
        ('location_transformer', LocationCompatibilityTransformer()),
        ('preprocessor', create_preprocessing_pipeline())
    ])
    
    logger.info("Pipeline de engenharia de features criado")
    return pipeline


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Engineering features for dataset with shape {df.shape}")
    
    df_engineered = df.copy()
    
    # Aplica transformadores customizados
    skills_transformer = SkillsMatchTransformer()
    df_engineered = skills_transformer.fit_transform(df_engineered)
    
    salary_transformer = SalaryFitTransformer()
    df_engineered = salary_transformer.fit_transform(df_engineered)
    
    location_transformer = LocationCompatibilityTransformer()
    df_engineered = location_transformer.fit_transform(df_engineered)
    
    # Adiciona features derivadas
    df_engineered['experience_education_ratio'] = (
        df_engineered['years_experience'] / 
        df_engineered.get('education_numeric', 1).replace(0, 1)
    )
    
    df_engineered['availability_urgency_ratio'] = (
        df_engineered.get('urgency_days', 30) / 
        df_engineered.get('availability_days', 30).replace(0, 1)
    )
    
    df_engineered['salary_range_width'] = (
        df_engineered.get('salary_range_max', 0) - 
        df_engineered.get('salary_range_min', 0)
    )
    
    # Indicador de estabilidade da empresa
    df_engineered['company_stability'] = np.where(
        df_engineered.get('years_experience', 0) > 0,
        df_engineered.get('years_experience', 0) / 
        (df_engineered.get('previous_companies', 1) + 1),
        0
    )
    
    logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
    
    return df_engineered


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get feature names from fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names
    """
    try:
        return preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback para versões antigas do sklearn
        feature_names = []
        
        for name, transformer, features in preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(features)
            elif hasattr(transformer, 'get_feature_names'):
                names = transformer.get_feature_names(features)
            else:
                names = features
            
            feature_names.extend([f"{name}__{n}" for n in names])
        
        return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str) -> None:
    """Save fitted preprocessor to disk.
    
    Args:
        preprocessor: Fitted preprocessor
        filepath: Path to save the preprocessor
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, filepath)
    logger.info(f"Preprocessor saved to {filepath}")


def load_preprocessor(filepath: str) -> ColumnTransformer:
    """Load preprocessor from disk.
    
    Args:
        filepath: Path to the saved preprocessor
        
    Returns:
        Loaded preprocessor
    """
    preprocessor = joblib.load(filepath)
    logger.info(f"Preprocessor loaded from {filepath}")
    return preprocessor


if __name__ == "__main__":
    # Test feature engineering with real data
    import sys
    sys.path.append('.')
    from data import load_real_data
    
    # Load real data
    df = load_real_data()
    
    if df.empty:
        print("No real data available for testing")
        sys.exit(1)
    
    print(f"Loaded {len(df)} samples for feature engineering test")
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"New columns: {set(df_engineered.columns) - set(df.columns)}")
    
    # Show sample of engineered features
    print("\nSample of engineered features:")
    feature_cols = ['skills_match_ratio', 'skill_diversity', 'rare_skills_bonus', 
                   'salary_fit', 'salary_position', 'location_match']
    available_cols = [col for col in feature_cols if col in df_engineered.columns]
    if available_cols:
        print(df_engineered[available_cols].head())
    
    print("\nFeature engineering test completed successfully!")
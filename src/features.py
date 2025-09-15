"""Módulo de engenharia de features e pré-processamento."""

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
    """Transformador personalizado para features de correspondência de habilidades."""
    
    def __init__(self):
        self.skill_vocabulary_ = set()
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """Ajusta o transformador construindo o vocabulário de habilidades.
        
        Args:
            X: DataFrame com colunas candidate_skills e required_skills
            y: Variável alvo (ignorado)
            
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
        """Transforma os dados criando features baseadas em habilidades.
        
        Args:
            X: DataFrame com colunas de habilidades
            
        Returns:
            DataFrame transformado com features adicionais de habilidades
        """
        if not self.fitted_:
            raise ValueError("Transformer deve ser ajustado antes da transformação")
        
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
        """Calcula a proporção de habilidades correspondentes."""
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
        """Calcula o número de habilidades do candidato."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            return len(candidate_skills)
        except Exception:
            return 0
    
    def _calculate_skill_diversity(self, row) -> float:
        """Calcula o score de diversidade de habilidades baseado na cobertura do vocabulário."""
        try:
            candidate_skills = self._parse_skills(row.get('candidate_skills', ''))
            if not self.skill_vocabulary_ or not candidate_skills:
                return 0.0
            
            unique_skills = set(candidate_skills) & self.skill_vocabulary_
            return len(unique_skills) / len(self.skill_vocabulary_)
        except Exception:
            return 0.0
    
    def _calculate_rare_skills_bonus(self, row) -> float:
        """Calcula bônus para ter habilidades raras/especializadas."""
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
        """Converte string de habilidades em lista de habilidades."""
        if not isinstance(skills_str, str) or not skills_str.strip():
            return []
        
        return [s.strip().lower() for s in skills_str.split(',') if s.strip()]


class SalaryFitTransformer(BaseEstimator, TransformerMixin):
    """Transformador personalizado para features de adequação salarial."""
    
    def fit(self, X, y=None):
        """Ajusta o transformador (não faz nada para este transformador)."""
        return self
    
    def transform(self, X):
        """Transforma os dados de salário em features de adequação."""
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
        """Calcula a posição da expectativa salarial dentro do intervalo (0-1)."""
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
        """Calcula a proporção da expectativa salarial em relação ao ponto médio do intervalo."""
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
    """Transformador personalizado para features de compatibilidade de localização."""
    
    def fit(self, X, y=None):
        """Ajusta o transformador (não faz nada para este transformador)."""
        return self
    
    def transform(self, X):
        """Transforma os dados de localização em features de compatibilidade."""
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
        """Calcula o score de compatibilidade de localização."""
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
            
            # Mesmo estado/região (simples)
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
        """Calcula a compatibilidade de trabalho remoto."""
        try:
            remote_work = row.get('remote_work', False)
            remote_allowed = row.get('remote_allowed', False)
            
            if remote_work and remote_allowed:
                return 1.0
            elif remote_allowed:
                return 0.8  # O trabalho remoto é permitido, o candidato pode se adaptar
            elif remote_work:
                return 0.3  # O candidato deseja trabalho remoto, o trabalho não permite
            else:
                return 0.6  # Ambos preferem trabalho presencial
        except Exception:
            return 0.5


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Cria o pipeline de pré-processamento completo.
    
    Returns:
        Pipeline de pré-processamento configurado
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
        remainder='passthrough',  # Mantém outras colunas como estão
        verbose_feature_names_out=False
    )
    
    logger.info(f"Pipeline criado com {len(numeric_features)} numéricas, "
                f"{len(categorical_features)} categóricas, e "
                f"{len(boolean_features)} booleanas")
    
    return preprocessor


def create_feature_engineering_pipeline() -> Pipeline:
    """Cria o pipeline de engenharia de features completo.
    
    Returns:
        Pipeline de engenharia de features completo
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
    """Engenharia de features no dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame com features
    """
    logger.info(f"Engenharia de features no dataset com shape {df.shape}")
    
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
    
    logger.info(f"Engenharia de features concluída. Novo shape: {df_engineered.shape}")
    
    return df_engineered


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Obtém nomes de features de um pré-processador ajustado.
    
    Args:
        preprocessor: Pré-processador ajustado
        
    Returns:
        Lista de nomes de features
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
    """Salva um pré-processador ajustado em disco.
    
    Args:
        preprocessor: Pré-processador ajustado
        filepath: Caminho para salvar o pré-processador
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, filepath)
    logger.info(f"Pré-processador salvo em {filepath}")


def load_preprocessor(filepath: str) -> ColumnTransformer:
    """Carrega um pré-processador de disco.
    
    Args:
        filepath: Caminho para o pré-processador salvo
        
    Returns:
        Pré-processador carregado
    """
    preprocessor = joblib.load(filepath)
    logger.info(f"Pré-processador carregado de {filepath}")
    return preprocessor


if __name__ == "__main__":
    # Test feature engineering with real data
    import sys
    sys.path.append('.')
    from data import load_real_data
    
    # Load real data
    df = load_real_data()
    
    if df.empty:
        print("Nenhum dado real disponível para teste")
        sys.exit(1)
    
    print(f"{len(df)} amostras carregadas para teste de engenharia de features")
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    print(f"Forma original: {df.shape}")
    print(f"Forma após engenharia de features: {df_engineered.shape}")
    print(f"Novoas colunas: {set(df_engineered.columns) - set(df.columns)}")
    
    # Show sample of engineered features
    print("\nAmostra de features:")
    feature_cols = ['skills_match_ratio', 'skill_diversity', 'rare_skills_bonus', 
                   'salary_fit', 'salary_position', 'location_match']
    available_cols = [col for col in feature_cols if col in df_engineered.columns]
    if available_cols:
        print(df_engineered[available_cols].head())
    
    print("\nTeste de engenharia de features concluído com sucesso!")

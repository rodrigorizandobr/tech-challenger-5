"""Testes para módulo de dados."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data import (
    generate_synthetic_data, calculate_match_score, load_and_validate_data,
    split_data, create_sample_payload
)


class TestGenerateSyntheticData:
    """Testes para geração de dados sintéticos."""
    
    def test_generate_synthetic_data_basic(self):
        """Testa geração básica de dados sintéticos."""
        df = generate_synthetic_data(n_samples=100, seed=42)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'match_score' in df.columns
        assert 'match_label' in df.columns
        
        # Check data types
        assert df['age'].dtype in ['int64', 'int32']
        assert df['match_score'].dtype in ['float64', 'float32']
        assert df['match_label'].dtype == 'object'
    
    def test_generate_synthetic_data_reproducible(self):
        """Testa reprodutibilidade na geração de dados sintéticos."""
        df1 = generate_synthetic_data(n_samples=50, seed=42)
        df2 = generate_synthetic_data(n_samples=50, seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_generate_synthetic_data_different_seeds(self):
        """Testa que diferentes seeds produzem dados diferentes."""
        df1 = generate_synthetic_data(n_samples=50, seed=42)
        df2 = generate_synthetic_data(n_samples=50, seed=123)
        
        # Should not be identical
        assert not df1.equals(df2)
    
    def test_synthetic_data_columns(self):
        """Testa se todas as colunas esperadas estão presentes."""
        df = generate_synthetic_data(n_samples=10)
        
        expected_columns = [
            'age', 'education_level', 'years_experience', 'skills_count',
            'skills_match_ratio', 'previous_companies', 'salary_expectation',
            'salary_fit', 'location_match', 'remote_compatibility',
            'availability_urgency_ratio', 'experience_level_numeric',
            'education_numeric', 'match_score', 'match_label'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Coluna missing: {col}"
    
    def test_synthetic_data_ranges(self):
        """Testa se os valores sintéticos estão dentro dos intervalos esperados."""
        df = generate_synthetic_data(n_samples=100, seed=42)
        
        # Age should be between 22 and 65
        assert df['age'].min() >= 22
        assert df['age'].max() <= 65
        
        # Match score should be between 0 and 1
        assert df['match_score'].min() >= 0
        assert df['match_score'].max() <= 1
        
        # Years experience should be non-negative
        assert df['years_experience'].min() >= 0
        
        # Skills count should be positive
        assert df['skills_count'].min() > 0
    
    def test_match_label_distribution(self):
        """Testa se as labels de correspondência estão distribuídas corretamente."""
        df = generate_synthetic_data(n_samples=1000, seed=42)
        
        label_counts = df['match_label'].value_counts()
        
        # Should have both labels
        assert 'good_match' in label_counts.index
        assert 'poor_match' in label_counts.index
        
        # Neither should be too dominant (within reasonable range)
        total = len(df)
        good_ratio = label_counts['good_match'] / total
        assert 0.2 <= good_ratio <= 0.8


class TestCalculateMatchScore:
    """Testa cálculo de pontuação de correspondência."""
    
    def test_perfect_match(self):
        """Testa cenário de correspondência perfeita."""
        score = calculate_match_score(
            years_exp=8,
            req_exp_level='senior',
            candidate_skills=['python', 'sql', 'machine learning'],
            req_skills=['python', 'sql', 'machine learning'],
            salary_exp=90000,
            salary_min=85000,
            salary_max=95000,
            location='São Paulo',
            job_location='São Paulo',
            remote_work=True,
            remote_allowed=True,
            availability=15,
            urgency=30,
            education='master'
        )
        
        assert score > 0.8  # Should be high score
    
    def test_poor_match(self):
        """Testa cenário de correspondência ruim."""
        score = calculate_match_score(
            years_exp=1,
            req_exp_level='senior',
            candidate_skills=['excel'],
            req_skills=['python', 'sql', 'machine learning'],
            salary_exp=150000,
            salary_min=80000,
            salary_max=100000,
            location='Manaus',
            job_location='São Paulo',
            remote_work=False,
            remote_allowed=False,
            availability=90,
            urgency=7,
            education='high_school'
        )
        
        assert score < 0.4  # Should be low score
    
    def test_score_range(self):
        """Testa que o score está sempre entre 0 e 1."""
        # Testa múltiplos cenários
        test_cases = [
            (0, 'junior', [], ['python'], 0, 50000, 60000, 'A', 'B', False, False, 365, 1, 'high_school'),
            (20, 'lead', ['python'] * 20, ['python'], 200000, 50000, 60000, 'A', 'B', True, True, 1, 365, 'phd')
        ]
        
        for case in test_cases:
            score = calculate_match_score(*case)
            assert 0 <= score <= 1, f"Score {score} está fora do intervalo [0, 1] para o caso {case}"
    
    def test_skills_impact(self):
        """Testa que a correspondência de habilidades afeta a pontuação."""
        base_args = {
            'years_exp': 5,
            'req_exp_level': 'mid',
            'salary_exp': 75000,
            'salary_min': 70000,
            'salary_max': 80000,
            'location': 'São Paulo',
            'job_location': 'São Paulo',
            'remote_work': True,
            'remote_allowed': True,
            'availability': 30,
            'urgency': 30,
            'education': 'bachelor'
        }
        
        # High skills match
        score_high = calculate_match_score(
            candidate_skills=['python', 'sql', 'machine learning'],
            req_skills=['python', 'sql', 'machine learning'],
            **base_args
        )
        
        # Low skills match
        score_low = calculate_match_score(
            candidate_skills=['excel'],
            req_skills=['python', 'sql', 'machine learning'],
            **base_args
        )
        
        assert score_high > score_low


class TestLoadAndValidateData:
    """Testa carregamento e validação de dados."""
    
    def test_load_nonexistent_file(self):
        """Testa carregamento de arquivo não existente cria dados sintéticos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_data.csv')
            
            df = load_and_validate_data(file_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert Path(file_path).exists()
    
    def test_load_existing_file(self):
        """Testa carregamento de arquivo existente."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_data.csv')
            
            # Cria dados de teste
            test_data = pd.DataFrame({
                'age': [25, 30, 35],
                'match_score': [0.7, 0.8, 0.6],
                'match_label': ['good_match', 'good_match', 'poor_match']
            })
            test_data.to_csv(file_path, index=False)
            
            df = load_and_validate_data(file_path)
            
            assert len(df) == 3
            assert 'match_score' in df.columns
            assert 'match_label' in df.columns
    
    def test_validate_missing_columns(self):
        """Testa validação com colunas obrigatórias ausentes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'invalid_data.csv')
            
            # Cria dados inválidos (colunas obrigatórias ausentes)
            invalid_data = pd.DataFrame({
                'age': [25, 30, 35],
                'name': ['A', 'B', 'C']
            })
            invalid_data.to_csv(file_path, index=False)
            
            with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
                load_and_validate_data(file_path)
    
    def test_validate_invalid_match_scores(self):
        """Testa validação com scores de correspondência inválidos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'invalid_scores.csv')
            
            # Cria dados com scores de correspondência inválidos
            invalid_data = pd.DataFrame({
                'match_score': [1.5, -0.5, 0.7],  # Valores inválidos
                'match_label': ['good_match', 'good_match', 'poor_match']
            })
            invalid_data.to_csv(file_path, index=False)
            
            with pytest.raises(ValueError, match="match_score values must be between 0 and 1"):
                load_and_validate_data(file_path)
    
    def test_validate_invalid_labels(self):
        """Testa validação com rótulos de correspondência inválidos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'invalid_labels.csv')
            
            # Cria dados com rótulos inválidos
            invalid_data = pd.DataFrame({
                'match_score': [0.7, 0.8, 0.6],
                'match_label': ['bom', 'ruim', 'talvez']  # Rótulos inválidos
            })
            invalid_data.to_csv(file_path, index=False)
            
            with pytest.raises(ValueError, match="match_label must be one of"):
                load_and_validate_data(file_path)


class TestSplitData:
    """Testa divisão de dados."""
    
    def test_split_data_basic(self):
        """Testa divisão básica de dados."""
        # Gera dados sintéticos
        df = generate_synthetic_data(n_samples=100, seed=42)
        
        train_df, test_df = split_data(df, test_size=0.2, random_state=42)
        
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == len(df)
    
    def test_split_data_stratified(self):
        """Testa divisão estratificada de dados."""
        df = generate_synthetic_data(n_samples=1000, seed=42)
        
        original_ratio = (df['match_label'] == 'good_match').mean()
        
        train_df, test_df = split_data(df, test_size=0.2, random_state=42)
        
        train_ratio = (train_df['match_label'] == 'good_match').mean()
        test_ratio = (test_df['match_label'] == 'good_match').mean()
        
        # Ratios should be similar (within 5%)
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05
    
    def test_split_data_reproducible(self):
        """Testa reproducibilidade da divisão."""
        df = generate_synthetic_data(n_samples=100, seed=42)
        
        train1, test1 = split_data(df, test_size=0.2, random_state=42)
        train2, test2 = split_data(df, test_size=0.2, random_state=42)
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_split_data_different_sizes(self):
        """Testa divisão com diferentes tamanhos de teste."""
        df = generate_synthetic_data(n_samples=100, seed=42)
        
        for test_size in [0.1, 0.3, 0.5]:
            train_df, test_df = split_data(df, test_size=test_size, random_state=42)
            
            expected_test_size = int(len(df) * test_size)
            expected_train_size = len(df) - expected_test_size
            
            assert len(test_df) == expected_test_size
            assert len(train_df) == expected_train_size


class TestCreateSamplePayload:
    """Testa criação de payload de amostra."""
    
    def test_create_sample_payload(self):
        """Testa criação de payload de amostra."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'sample.json')
            
            create_sample_payload(output_path)
            
            assert Path(output_path).exists()
            
            # Load and validate JSON
            import json
            with open(output_path, 'r') as f:
                payload = json.load(f)
            
            # Verifica se o payload contém as chaves esperadas
            assert 'candidate' in payload
            assert 'job' in payload
            
            # Check candidate fields
            candidate = payload['candidate']
            assert 'age' in candidate
            assert 'education_level' in candidate
            assert 'skills' in candidate
            assert isinstance(candidate['skills'], list)
            
            # Check job fields
            job = payload['job']
            assert 'required_experience' in job
            assert 'required_skills' in job
            assert 'salary_range_min' in job
            assert 'salary_range_max' in job


if __name__ == "__main__":
    pytest.main([__file__])
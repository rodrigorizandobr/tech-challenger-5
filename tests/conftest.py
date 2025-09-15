"""Configuração do Pytest e fixtures compartilhadas."""

import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_dir():
    """Cria um diretório temporário para arquivos de teste."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_data():
    """Cria dados de treinamento de amostra para testes."""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(22, 65, 100),
        'education_level': np.random.choice(['bachelor', 'master', 'phd'], 100),
        'years_experience': np.random.randint(0, 20, 100),
        'skills_count': np.random.randint(1, 10, 100),
        'skills_match_ratio': np.random.uniform(0, 1, 100),
        'previous_companies': np.random.randint(0, 5, 100),
        'salary_expectation': np.random.uniform(50000, 150000, 100),
        'salary_fit': np.random.uniform(0, 1, 100),
        'location_match': np.random.uniform(0, 1, 100),
        'remote_compatibility': np.random.uniform(0, 1, 100),
        'availability_urgency_ratio': np.random.uniform(0, 2, 100),
        'experience_level_numeric': np.random.randint(1, 5, 100),
        'education_numeric': np.random.randint(1, 5, 100),
        'match_score': np.random.uniform(0, 1, 100),
        'match_label': np.random.choice(['good_match', 'poor_match'], 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(temp_dir, sample_data):
    """Cria um arquivo CSV de amostra para testes."""
    file_path = os.path.join(temp_dir, 'sample_data.csv')
    sample_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def mock_trained_model():
    """Cria um modelo treinado fictício para testes."""
    model = MagicMock()
    model.predict.return_value = np.array(['good_match', 'poor_match'])
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
    model.classes_ = np.array(['poor_match', 'good_match'])
    return model


@pytest.fixture
def sample_prediction_data():
    """Dados de amostra para testes de previsão."""
    return {
        'candidate': {
            'age': 28,
            'education_level': 'bachelor',
            'years_experience': 5,
            'skills': ['python', 'machine learning', 'sql'],
            'previous_companies': 2,
            'salary_expectation': 80000,
            'location': 'São Paulo',
            'remote_work': True,
            'availability_days': 30
        },
        'job': {
            'required_experience': 'mid',
            'required_skills': ['python', 'sql', 'machine learning'],
            'salary_range_min': 70000,
            'salary_range_max': 90000,
            'location': 'São Paulo',
            'remote_allowed': True,
            'urgency_days': 45
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Configura o ambiente de teste com variáveis de ambiente."""
    monkeypatch.setenv("MODEL_PATH", "test_models/model.joblib")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8001")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def clean_logs():
    """Limpa arquivos de log após os testes."""
    yield
    
    # Limpa qualquer arquivo de log criado durante os testes
    log_files = [
        'logs/api.log',
        'logs/predictions.csv',
        'test.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except OSError:
                pass


# Configuração do pytest markers
def pytest_configure(config):
    """Configura pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marca testes como lentos (desmarque com '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marca testes como de integração"
    )
    config.addinivalue_line(
        "markers", "unit: marca testes como unitários"
    )


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modifica a coleção de testes para adicionar marcadores automaticamente."""
    for item in items:
        # Marca testes lentos
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Marca testes unitários
        if "test_" in item.name and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Marca testes de integração
        if "integration" in item.nodeid or "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
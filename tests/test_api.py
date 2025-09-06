"""Tests for FastAPI application."""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from app.main import app
from src.schema import PredictionRequest, CandidateInput, JobRequirements


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "candidate": {
            "age": 28,
            "education_level": "bachelor",
            "years_experience": 5,
            "skills": ["python", "machine learning", "sql"],
            "previous_companies": 2,
            "salary_expectation": 80000,
            "location": "São Paulo",
            "remote_work": True,
            "availability_days": 30
        },
        "job": {
            "required_experience": "mid",
            "required_skills": ["python", "sql", "machine learning"],
            "salary_range_min": 70000,
            "salary_range_max": 90000,
            "location": "São Paulo",
            "remote_allowed": True,
            "urgency_days": 45
        }
    }


@pytest.fixture
def mock_model():
    """Mock trained model."""
    model = MagicMock()
    model.predict.return_value = np.array(['good_match'])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.classes_ = np.array(['poor_match', 'good_match'])
    return model


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_without_model(self, client):
        """Test health check when model is not loaded."""
        with patch('app.main.model', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
            assert "timestamp" in data
            assert "version" in data
            assert "uptime_seconds" in data
    
    def test_health_check_with_model(self, client, mock_model):
        """Test health check when model is loaded."""
        with patch('app.main.model', mock_model):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["version"] == "1.0.0"


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Decision AI" in response.text
        assert "/health" in response.text
        assert "/predict" in response.text


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_without_model(self, client, sample_prediction_request):
        """Test prediction when model is not loaded."""
        with patch('app.main.model', None):
            response = client.post("/predict", json=sample_prediction_request)
            
            assert response.status_code == 503
            assert "Model not available" in response.json()["detail"]
    
    def test_predict_with_model(self, client, sample_prediction_request, mock_model):
        """Test successful prediction."""
        with patch('app.main.model', mock_model):
            response = client.post("/predict", json=sample_prediction_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "match_probability" in data
            assert "match_label" in data
            assert "confidence" in data
            assert "factors" in data
            assert "recommendation" in data
            assert "timestamp" in data
            
            # Check data types and ranges
            assert 0 <= data["match_probability"] <= 1
            assert 0 <= data["confidence"] <= 1
            assert data["match_label"] in ["good_match", "poor_match"]
            assert isinstance(data["factors"], dict)
            assert isinstance(data["recommendation"], str)
    
    def test_predict_invalid_request(self, client):
        """Test prediction with invalid request data."""
        invalid_request = {
            "candidate": {
                "age": -5,  # Invalid age
                "education_level": "invalid",  # Invalid education level
                "skills": []  # Empty skills list
            }
        }
        
        response = client.post("/predict", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        incomplete_request = {
            "candidate": {
                "age": 28
                # Missing other required fields
            }
        }
        
        response = client.post("/predict", json=incomplete_request)
        
        assert response.status_code == 422
    
    def test_predict_edge_cases(self, client, mock_model):
        """Test prediction with edge case values."""
        edge_case_request = {
            "candidate": {
                "age": 18,  # Minimum age
                "education_level": "high_school",
                "years_experience": 0,  # Minimum experience
                "skills": ["basic skill"],
                "previous_companies": 0,
                "salary_expectation": 1,  # Very low salary
                "location": "Remote",
                "remote_work": True,
                "availability_days": 1  # Immediate availability
            },
            "job": {
                "required_experience": "junior",
                "required_skills": ["any skill"],
                "salary_range_min": 1,
                "salary_range_max": 1000000,  # Very high range
                "location": "Anywhere",
                "remote_allowed": True,
                "urgency_days": 365  # Not urgent
            }
        }
        
        with patch('app.main.model', mock_model):
            response = client.post("/predict", json=edge_case_request)
            
            assert response.status_code == 200
    
    def test_predict_model_error(self, client, sample_prediction_request):
        """Test prediction when model raises an error."""
        error_model = MagicMock()
        error_model.predict.side_effect = Exception("Model error")
        
        with patch('app.main.model', error_model):
            response = client.post("/predict", json=sample_prediction_request)
            
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_basic(self, client):
        """Test basic metrics retrieval."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "total_predictions" in data
        assert "avg_match_probability" in data
        assert "predictions_last_24h" in data
        assert "drift_detected" in data
        assert "system_health" in data
        
        # Check data types
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["avg_match_probability"], (int, float))
        assert isinstance(data["predictions_last_24h"], int)
        assert isinstance(data["drift_detected"], bool)
        assert isinstance(data["system_health"], str)
    
    def test_metrics_with_predictions(self, client, sample_prediction_request, mock_model):
        """Test metrics after making predictions."""
        with patch('app.main.model', mock_model):
            # Make a prediction first
            client.post("/predict", json=sample_prediction_request)
            
            # Check metrics
            response = client.get("/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have at least one prediction
            assert data["total_predictions"] >= 1


class TestDriftReportEndpoint:
    """Test drift report endpoint."""
    
    def test_drift_report_not_found(self, client):
        """Test drift report when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            response = client.get("/drift-report")
            
            assert response.status_code == 404
            assert "Drift report not found" in response.json()["detail"]
    
    def test_drift_report_exists(self, client):
        """Test drift report when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy report file
            report_path = Path(temp_dir) / "drift.html"
            report_content = "<html><body>Test Report</body></html>"
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('app.main.Path', return_value=report_path):
                
                response = client.get("/drift-report")
                
                assert response.status_code == 200
                assert "text/html" in response.headers["content-type"]


class TestRequestValidation:
    """Test request validation."""
    
    def test_candidate_age_validation(self, client):
        """Test candidate age validation."""
        invalid_ages = [-1, 17, 71, 150]
        
        for age in invalid_ages:
            request_data = {
                "candidate": {
                    "age": age,
                    "education_level": "bachelor",
                    "years_experience": 5,
                    "skills": ["python"],
                    "previous_companies": 2,
                    "salary_expectation": 80000,
                    "location": "São Paulo",
                    "remote_work": True,
                    "availability_days": 30
                },
                "job": {
                    "required_experience": "mid",
                    "required_skills": ["python"],
                    "salary_range_min": 70000,
                    "salary_range_max": 90000,
                    "location": "São Paulo",
                    "remote_allowed": True,
                    "urgency_days": 45
                }
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 422
    
    def test_skills_validation(self, client):
        """Test skills validation."""
        # Empty skills list
        request_data = {
            "candidate": {
                "age": 28,
                "education_level": "bachelor",
                "years_experience": 5,
                "skills": [],  # Empty skills
                "previous_companies": 2,
                "salary_expectation": 80000,
                "location": "São Paulo",
                "remote_work": True,
                "availability_days": 30
            },
            "job": {
                "required_experience": "mid",
                "required_skills": ["python"],
                "salary_range_min": 70000,
                "salary_range_max": 90000,
                "location": "São Paulo",
                "remote_allowed": True,
                "urgency_days": 45
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_salary_range_validation(self, client):
        """Test salary range validation."""
        # Max salary less than min salary
        request_data = {
            "candidate": {
                "age": 28,
                "education_level": "bachelor",
                "years_experience": 5,
                "skills": ["python"],
                "previous_companies": 2,
                "salary_expectation": 80000,
                "location": "São Paulo",
                "remote_work": True,
                "availability_days": 30
            },
            "job": {
                "required_experience": "mid",
                "required_skills": ["python"],
                "salary_range_min": 90000,
                "salary_range_max": 70000,  # Less than min
                "location": "São Paulo",
                "remote_allowed": True,
                "urgency_days": 45
            }
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422


class TestConcurrency:
    """Test API under concurrent load."""
    
    def test_concurrent_predictions(self, client, sample_prediction_request, mock_model):
        """Test multiple concurrent predictions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                response = client.post("/predict", json=sample_prediction_request)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        with patch('app.main.model', mock_model):
            # Create multiple threads
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=make_prediction)
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results), f"Non-200 responses: {results}"


class TestLogging:
    """Test logging functionality."""
    
    def test_request_logging(self, client):
        """Test that requests are logged."""
        with patch('app.main.logger') as mock_logger:
            response = client.get("/health")
            
            assert response.status_code == 200
            # Verify that logging was called
            mock_logger.info.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
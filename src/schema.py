"""Schemas and data models for the recruitment AI system."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class EducationLevel(str, Enum):
    """Education level enumeration."""
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


class CandidateInput(BaseModel):
    """Schema for candidate input data."""
    age: int = Field(..., ge=18, le=70, description="Candidate age")
    education_level: EducationLevel = Field(..., description="Education level")
    years_experience: int = Field(..., ge=0, le=50, description="Years of experience")
    skills: List[str] = Field(
        ..., min_items=1, max_items=20, description="List of skills"
    )
    previous_companies: int = Field(
        ..., ge=0, le=20, description="Number of previous companies"
    )
    salary_expectation: float = Field(..., ge=0, description="Expected salary")
    location: str = Field(
        ..., min_length=2, max_length=100, description="Location"
    )
    remote_work: bool = Field(..., description="Accepts remote work")
    availability_days: int = Field(
        ..., ge=1, le=365, description="Days until availability"
    )

    @validator('skills')
    def validate_skills(cls, v):
        """Validate skills list."""
        if not v:
            raise ValueError("Skills list cannot be empty")
        # Remove duplicates and convert to lowercase
        return list(set(skill.strip().lower() for skill in v if skill.strip()))

    @validator('location')
    def validate_location(cls, v):
        """Validate location string."""
        return v.strip().title()


class JobRequirements(BaseModel):
    """Schema for job requirements."""
    required_experience: ExperienceLevel = Field(
        ..., description="Required experience level"
    )
    required_skills: List[str] = Field(
        ..., min_items=1, description="Required skills"
    )
    salary_range_min: float = Field(..., ge=0, description="Minimum salary")
    salary_range_max: float = Field(..., ge=0, description="Maximum salary")
    location: str = Field(..., description="Job location")
    remote_allowed: bool = Field(..., description="Remote work allowed")
    urgency_days: int = Field(..., ge=1, description="Days to fill position")

    @validator('salary_range_max')
    def validate_salary_range(cls, v, values):
        """Validate salary range."""
        if 'salary_range_min' in values and v < values['salary_range_min']:
            raise ValueError("Maximum salary must be greater than minimum salary")
        return v

    @validator('required_skills')
    def validate_required_skills(cls, v):
        """Validate required skills list."""
        if not v:
            raise ValueError("Required skills list cannot be empty")
        return list(set(skill.strip().lower() for skill in v if skill.strip()))

    @validator('location')
    def validate_location(cls, v):
        """Validate location string."""
        return v.strip().title()


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    candidate: CandidateInput
    job: JobRequirements

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    match_probability: float = Field(
        ..., ge=0, le=1, description="Match probability"
    )
    match_label: str = Field(..., description="Match classification")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    factors: Dict[str, float] = Field(..., description="Contributing factors")
    recommendation: str = Field(..., description="Hiring recommendation")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "match_probability": 0.85,
                "match_label": "good_match",
                "confidence": 0.92,
                "factors": {
                    "skills_match": 0.9,
                    "experience_match": 0.8,
                    "salary_fit": 0.85,
                    "location_compatibility": 1.0
                },
                "recommendation": "Strong candidate - recommend for interview",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loading status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class MetricsResponse(BaseModel):
    """Schema for metrics response."""
    total_predictions: int = Field(..., description="Total predictions made")
    avg_match_probability: float = Field(
        ..., description="Average match probability"
    )
    predictions_last_24h: int = Field(
        ..., description="Predictions in last 24 hours"
    )
    model_accuracy: Optional[float] = Field(
        None, description="Model accuracy if available"
    )
    drift_detected: bool = Field(..., description="Data drift detection status")
    last_retrain: Optional[datetime] = Field(
        None, description="Last model retrain timestamp"
    )
    system_health: str = Field(..., description="Overall system health")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionLog(BaseModel):
    """Schema for prediction logging."""
    request_id: str = Field(..., description="Unique request identifier")
    candidate_data: Dict[str, Any] = Field(..., description="Candidate input data")
    job_data: Dict[str, Any] = Field(..., description="Job requirements data")
    prediction: PredictionResponse = Field(..., description="Prediction result")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(..., description="Model version used")
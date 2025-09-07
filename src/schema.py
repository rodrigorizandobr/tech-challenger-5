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
    ENSINO_MEDIO = "Ensino Médio"
    BACHELOR = "bachelor"
    ENSINO_SUPERIOR = "Ensino Superior"
    MASTER = "master"
    POS_GRADUACAO = "Pós-graduação"
    MESTRADO = "Mestrado"
    PHD = "phd"
    DOUTORADO = "Doutorado"


class LanguageLevel(str, Enum):
    """Language proficiency level enumeration."""
    NENHUM = "Nenhum"
    BASICO = "Básico"
    INTERMEDIARIO = "Intermediário"
    AVANCADO = "Avançado"
    FLUENTE = "Fluente"


class AreaAtuacao(str, Enum):
    """Area of expertise enumeration."""
    TECNOLOGIA = "Tecnologia"
    VENDAS = "Vendas"
    MARKETING = "Marketing"
    FINANCEIRO = "Financeiro"
    RECURSOS_HUMANOS = "Recursos Humanos"
    GERAL = "Geral"


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
        # Remove duplicidades e converte para minúsculas
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


# Schemas for original data structures

class InfosBasicas(BaseModel):
    """Schema for basic candidate information."""
    nome: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    telefone: Optional[str] = Field(None, description="Phone number")


class InformacoesPessoais(BaseModel):
    """Schema for personal information."""
    data_nascimento: Optional[str] = Field(None, description="Birth date")
    endereco: Optional[str] = Field(None, description="Address")
    estado_civil: Optional[str] = Field(None, description="Marital status")


class InformacoesProfissionais(BaseModel):
    """Schema for professional information."""
    area_atuacao: Optional[str] = Field(None, description="Area of expertise")
    conhecimentos_tecnicos: Optional[str] = Field(None, description="Technical knowledge")
    remuneracao: Optional[str] = Field(None, description="Salary expectation")


class FormacaoIdiomas(BaseModel):
    """Schema for education and languages."""
    nivel_academico: Optional[str] = Field(None, description="Academic level")
    nivel_ingles: Optional[str] = Field(None, description="English level")
    nivel_espanhol: Optional[str] = Field(None, description="Spanish level")


class CandidateRawData(BaseModel):
    """Schema for raw candidate data from JSON."""
    infos_basicas: Optional[InfosBasicas] = Field(None, description="Basic information")
    informacoes_pessoais: Optional[InformacoesPessoais] = Field(None, description="Personal information")
    informacoes_profissionais: Optional[InformacoesProfissionais] = Field(None, description="Professional information")
    formacao_e_idiomas: Optional[FormacaoIdiomas] = Field(None, description="Education and languages")
    cargo_atual: Optional[Dict[str, Any]] = Field(None, description="Current position")
    cv_pt: Optional[str] = Field(None, description="CV in Portuguese")


class JobInfosBasicas(BaseModel):
    """Schema for basic job information."""
    titulo_vaga: Optional[str] = Field(None, description="Job title")
    vaga_sap: Optional[str] = Field(None, description="SAP position flag")


class PerfilVaga(BaseModel):
    """Schema for job profile requirements."""
    nivel_academico: Optional[str] = Field(None, description="Required academic level")
    nivel_ingles: Optional[str] = Field(None, description="Required English level")
    nivel_espanhol: Optional[str] = Field(None, description="Required Spanish level")
    areas_atuacao: Optional[str] = Field(None, description="Areas of expertise")
    principais_atividades: Optional[str] = Field(None, description="Main activities")
    competencia_tecnicas_e_comportamentais: Optional[str] = Field(None, description="Technical and behavioral competencies")
    cidade: Optional[str] = Field(None, description="City")


class JobRawData(BaseModel):
    """Schema for raw job data from JSON."""
    informacoes_basicas: Optional[JobInfosBasicas] = Field(None, description="Basic job information")
    perfil_vaga: Optional[PerfilVaga] = Field(None, description="Job profile")
    beneficios: Optional[Dict[str, Any]] = Field(None, description="Benefits")


class ProspectCandidate(BaseModel):
    """Schema for prospect candidate information."""
    nome: Optional[str] = Field(None, description="Candidate name")
    codigo: Optional[str] = Field(None, description="Candidate code")
    situacao: Optional[str] = Field(None, description="Current status")
    data_inicio: Optional[str] = Field(None, description="Start date")
    data_fim: Optional[str] = Field(None, description="End date")
    comentarios: Optional[str] = Field(None, description="Comments")


class ProspectRawData(BaseModel):
    """Schema for raw prospect data from JSON."""
    titulo: Optional[str] = Field(None, description="Position title")
    modalidade: Optional[str] = Field(None, description="Work modality")
    prospects: Optional[List[ProspectCandidate]] = Field(None, description="List of prospect candidates")


class ProcessedCandidate(BaseModel):
    """Schema for processed candidate data."""
    name: str = Field(..., description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    age: int = Field(..., description="Age")
    education_level: str = Field(..., description="Education level")
    years_experience: int = Field(..., description="Years of experience")
    english_level: str = Field(..., description="English proficiency")
    spanish_level: str = Field(..., description="Spanish proficiency")
    technical_skills: str = Field(..., description="Technical skills")
    area_of_expertise: str = Field(..., description="Area of expertise")
    salary_expectation: float = Field(..., description="Salary expectation")
    location: str = Field(..., description="Location")
    remote_work_preference: str = Field(..., description="Remote work preference")


class ProcessedJob(BaseModel):
    """Schema for processed job data."""
    job_title: str = Field(..., description="Job title")
    required_education: str = Field(..., description="Required education")
    required_experience: int = Field(..., description="Required years of experience")
    required_english: str = Field(..., description="Required English level")
    required_spanish: str = Field(..., description="Required Spanish level")
    required_skills: str = Field(..., description="Required skills")
    job_area: str = Field(..., description="Job area")
    salary_range_min: float = Field(..., description="Minimum salary")
    salary_range_max: float = Field(..., description="Maximum salary")
    job_location: str = Field(..., description="Job location")
    remote_work_allowed: str = Field(..., description="Remote work policy")
    is_sap_position: bool = Field(..., description="Is SAP position")
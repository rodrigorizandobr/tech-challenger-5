"""Schemas e modelos de dados para o sistema de IA de recrutamento."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class ExperienceLevel(str, Enum):
    """Nível de experiência enumeration."""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class EducationLevel(str, Enum):
    """Enumeração de nível educacional."""
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
    """Enumeração de nível de proficiência em língua."""
    NENHUM = "Nenhum"
    BASICO = "Básico"
    INTERMEDIARIO = "Intermediário"
    AVANCADO = "Avançado"
    FLUENTE = "Fluente"


class AreaAtuacao(str, Enum):
    """Enumeração de área de atuação."""
    TECNOLOGIA = "Tecnologia"
    VENDAS = "Vendas"
    MARKETING = "Marketing"
    FINANCEIRO = "Financeiro"
    RECURSOS_HUMANOS = "Recursos Humanos"
    GERAL = "Geral"


class CandidateInput(BaseModel):
    """Modelo de dados de entrada do candidato."""
    age: int = Field(..., ge=18, le=70, description="Idade do candidato")
    education_level: EducationLevel = Field(..., description="Nível de educação")
    years_experience: int = Field(..., ge=0, le=50, description="Anos de experiência")
    skills: List[str] = Field(
        ..., min_items=1, max_items=20, description="Habilidades do candidato"
    )
    previous_companies: int = Field(
        ..., ge=0, le=20, description="Quantidade de empresas anteriores"
    )
    salary_expectation: float = Field(..., ge=0, description="Expectativa salarial")
    location: str = Field(
        ..., min_length=2, max_length=100, description="Localização"
    )
    remote_work: bool = Field(..., description="Aceita trabalho remoto")
    availability_days: int = Field(
        ..., ge=1, le=365, description="Dias até a disponibilidade"
    )

    @validator('skills')
    def validate_skills(cls, v):
        """Valida a lista de habilidades."""
        if not v:
            raise ValueError("Lista de habilidades não pode ser vazia")
        # Remove duplicidades e converte para minúsculas
        return list(set(skill.strip().lower() for skill in v if skill.strip()))

    @validator('location')
    def validate_location(cls, v):
        """Valida string de localização."""
        return v.strip().title()


class JobRequirements(BaseModel):
    """Schema para requisitos de trabalho."""
    required_experience: ExperienceLevel = Field(
        ..., description="Nível de experiência requerido"
    )
    required_skills: List[str] = Field(
        ..., min_items=1, description="Habilidades requeridas"
    )
    salary_range_min: float = Field(..., ge=0, description="Salário mínimo")
    salary_range_max: float = Field(..., ge=0, description="Salário máximo")
    location: str = Field(..., description="Localização do trabalho")
    remote_allowed: bool = Field(..., description="Trabalho remoto permitido")
    urgency_days: int = Field(..., ge=1, description="Dias até preencher posição")

    @validator('salary_range_max')
    def validate_salary_range(cls, v, values):
        """Valida faixa salarial."""
        if 'salary_range_min' in values and v < values['salary_range_min']:
            raise ValueError("Salário máximo deve ser maior que salário mínimo")
        return v

    @validator('required_skills')
    def validate_required_skills(cls, v):
        """Valida lista de habilidades requeridas."""
        if not v:
            raise ValueError("Lista de habilidades requeridas não pode ser vazia")
        return list(set(skill.strip().lower() for skill in v if skill.strip()))

    @validator('location')
    def validate_location(cls, v):
        """Valida string de localização."""
        return v.strip().title()


class PredictionRequest(BaseModel):
    """Schema para requisição de predição."""
    candidate: CandidateInput
    job: JobRequirements

    class Config:
        """Configuração do Pydantic."""
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
    """Schema para resposta de predição."""
    match_probability: float = Field(
        ..., ge=0, le=1, description="Probabilidade de correspondência"
    )
    match_label: str = Field(..., description="Classificação de correspondência")
    confidence: float = Field(..., ge=0, le=1, description="Confiança na predição")
    factors: Dict[str, float] = Field(..., description="Fatores contribuindo")
    recommendation: str = Field(..., description="Recomendação de contratação")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        """Configuração do Pydantic."""
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
                "recommendation": "Recomendação de contratação: Alto",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Schema para resposta de verificação de saúde."""
    status: str = Field(..., description="Status do serviço")
    model_loaded: bool = Field(..., description="Status de carregamento do modelo")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Tempo de funcionamento em segundos")


class MetricsResponse(BaseModel):
    """Schema for metrics response."""
    total_predictions: int = Field(..., description="Total de predições realizadas")
    avg_match_probability: float = Field(
        ..., description="Probabilidade média de correspondência"
    )
    predictions_last_24h: int = Field(
        ..., description="Predições nas últimas 24 horas"
    )
    model_accuracy: Optional[float] = Field(
        None, description="Acurácia do modelo, se disponível"
    )
    drift_detected: bool = Field(..., description="Detecção de drift de dados")
    last_retrain: Optional[datetime] = Field(
        None, description="Último treinamento do modelo"
    )
    system_health: str = Field(..., description="Qualidade geral do sistema")


class ErrorResponse(BaseModel):
    """Schema para respostas de erro."""
    error: str = Field(..., description="Tipo de erro")
    message: str = Field(..., description="Mensagem de erro")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Detalhes adicionais do erro"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionLog(BaseModel):
    """Schema para log de predição."""
    request_id: str = Field(..., description="Identificador único da requisição")
    candidate_data: Dict[str, Any] = Field(..., description="Dados do candidato")
    job_data: Dict[str, Any] = Field(..., description="Dados do job")
    prediction: PredictionResponse = Field(..., description="Predição")
    processing_time_ms: float = Field(
        ..., description="Tempo de processamento em milissegundos"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(..., description="Versão do modelo usada")


# Schemas for original data structures

class InfosBasicas(BaseModel):
    """Schema para informações básicas do candidato."""
    nome: Optional[str] = Field(None, description="Nome do candidato")  
    email: Optional[str] = Field(None, description="Email do candidato")
    telefone: Optional[str] = Field(None, description="Telefone do candidato")


class InformacoesPessoais(BaseModel):
    """Schema para informações pessoais do candidato."""
    data_nascimento: Optional[str] = Field(None, description="Data de nascimento")
    endereco: Optional[str] = Field(None, description="Endereço")
    estado_civil: Optional[str] = Field(None, description="Estado civil")


class InformacoesProfissionais(BaseModel):
    """Schema para informações profissionais do candidato."""
    area_atuacao: Optional[str] = Field(None, description="Área de atuação")
    conhecimentos_tecnicos: Optional[str] = Field(None, description="Conhecimentos técnicos")
    remuneracao: Optional[str] = Field(None, description="Remuneração esperada")


class FormacaoIdiomas(BaseModel):
    """Schema para formação e idiomas do candidato."""
    nivel_academico: Optional[str] = Field(None, description="Nível acadêmico")
    nivel_ingles: Optional[str] = Field(None, description="Nível de inglês")
    nivel_espanhol: Optional[str] = Field(None, description="Nível de espanhol")


class CandidateRawData(BaseModel):
    """Schema para dados brutos do candidato em JSON."""
    infos_basicas: Optional[InfosBasicas] = Field(None, description="Informações básicas")
    informacoes_pessoais: Optional[InformacoesPessoais] = Field(None, description="Informações pessoais")
    informacoes_profissionais: Optional[InformacoesProfissionais] = Field(None, description="Informações profissionais")
    formacao_e_idiomas: Optional[FormacaoIdiomas] = Field(None, description="Formação e idiomas")
    cargo_atual: Optional[Dict[str, Any]] = Field(None, description="Cargo atual")
    cv_pt: Optional[str] = Field(None, description="Currículo em português")


class JobInfosBasicas(BaseModel):
    """Schema para informações básicas da vaga."""
    titulo_vaga: Optional[str] = Field(None, description="Título da vaga")
    vaga_sap: Optional[str] = Field(None, description="Flag de posição SAP")


class PerfilVaga(BaseModel):
    """Schema para perfil da vaga."""
    nivel_academico: Optional[str] = Field(None, description="Nível acadêmico")
    nivel_ingles: Optional[str] = Field(None, description="Nível de inglês")
    nivel_espanhol: Optional[str] = Field(None, description="Nível de espanhol")
    areas_atuacao: Optional[str] = Field(None, description="Áreas de atuação")
    principais_atividades: Optional[str] = Field(None, description="Principais atividades")
    competencia_tecnicas_e_comportamentais: Optional[str] = Field(None, description="Competências técnicas e comportamentais")  
    cidade: Optional[str] = Field(None, description="Cidade")


class JobRawData(BaseModel):
    """Schema para dados brutos da vaga em JSON."""
    informacoes_basicas: Optional[JobInfosBasicas] = Field(None, description="Informações básicas da vaga")
    perfil_vaga: Optional[PerfilVaga] = Field(None, description="Perfil da vaga")
    beneficios: Optional[Dict[str, Any]] = Field(None, description="Benefícios")


class ProspectCandidate(BaseModel):
    """Schema para informações do candidato prospect."""
    nome: Optional[str] = Field(None, description="Nome do candidato")
    codigo: Optional[str] = Field(None, description="Código do candidato")
    situacao: Optional[str] = Field(None, description="Situação do candidato")
    data_inicio: Optional[str] = Field(None, description="Data de início")
    data_fim: Optional[str] = Field(None, description="Data de fim")
    comentarios: Optional[str] = Field(None, description="Comentários")


class ProspectRawData(BaseModel):
    """Schema para dados brutos do prospect em JSON."""
    titulo: Optional[str] = Field(None, description="Título da vaga")
    modalidade: Optional[str] = Field(None, description="Modalidade de trabalho")
    prospects: Optional[List[ProspectCandidate]] = Field(None, description="Lista de candidatos prospect")


class ProcessedCandidate(BaseModel):
    """Schema para dados processados do candidato."""
    name: str = Field(..., description="Nome do candidato")
    email: Optional[str] = Field(None, description="Email do candidato")
    phone: Optional[str] = Field(None, description="Telefone do candidato")
    age: int = Field(..., description="Idade do candidato")
    education_level: str = Field(..., description="Nível de educação")
    years_experience: int = Field(..., description="Anos de experiência")
    english_level: str = Field(..., description="Nível de inglês")
    spanish_level: str = Field(..., description="Nível de espanhol")
    technical_skills: str = Field(..., description="Habilidades técnicas")
    area_of_expertise: str = Field(..., description="Área de especialização")
    salary_expectation: float = Field(..., description="Expectativa salarial")
    location: str = Field(..., description="Localização")
    remote_work_preference: str = Field(..., description="Preferência de trabalho remoto")


class ProcessedJob(BaseModel):
    """Schema para dados processados da vaga."""
    job_title: str = Field(..., description="Título da vaga")
    required_education: str = Field(..., description="Nível de educação necessário")
    required_experience: int = Field(..., description="Anos de experiência necessário")
    required_english: str = Field(..., description="Nível de inglês necessário")
    required_spanish: str = Field(..., description="Nível de espanhol necessário")
    required_skills: str = Field(..., description="Habilidades necessárias")
    job_area: str = Field(..., description="Área de atuação")
    salary_range_min: float = Field(..., description="Mínimo salário")
    salary_range_max: float = Field(..., description="Máximo salário")
    job_location: str = Field(..., description="Loclização do trabalho")
    remote_work_allowed: str = Field(..., description="Política de trabalho remoto")
    is_sap_position: bool = Field(..., description="É posição SAP")
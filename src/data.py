"""Data loading, validation and synthetic data generation module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path
import json
from loguru import logger
from sklearn.model_selection import train_test_split
import os
import re
from datetime import datetime


def load_real_data() -> pd.DataFrame:
    """Load and process real recruitment data from JSON files.
    
    Returns:
        DataFrame with processed recruitment data
    """
    logger.info("Loading real recruitment data...")
    
    # Load JSON files
    data_dir = Path("data")
    
    try:
        with open(data_dir / "applicants.json", 'r', encoding='utf-8') as f:
            applicants = json.load(f)
        
        with open(data_dir / "vagas.json", 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        with open(data_dir / "prospects.json", 'r', encoding='utf-8') as f:
            prospects = json.load(f)
    except FileNotFoundError as e:
        logger.warning(f"Real data files not found: {e}. Generating synthetic data instead.")
        return generate_synthetic_data()
    
    data = []
    
    # Process each job and its prospects
    for job_id, job_data in jobs.items():
        if job_id not in prospects:
            continue
            
        job_prospects = prospects[job_id]
        if not job_prospects.get('prospects'):
            continue
            
        for prospect in job_prospects['prospects']:
            candidate_id = prospect['codigo']
            
            if candidate_id not in applicants:
                continue
                
            candidate_data = applicants[candidate_id]
            
            # Extract candidate features
            candidate_features = extract_candidate_features(candidate_data)
            
            # Extract job features
            job_features = extract_job_features(job_data)
            
            # Determine match quality based on prospect status
            match_label = determine_match_quality(prospect['situacao_candidado'])
            
            # Calculate match score
            match_score = calculate_real_match_score(candidate_features, job_features)
            
            # Convert to the format expected by the model
            sample = convert_to_model_format(candidate_features, job_features, match_score, match_label)
            data.append(sample)
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded dataset shape: {df.shape}")
    if not df.empty:
        logger.info(f"Match distribution: {df['match_label'].value_counts().to_dict()}")
    
    return df


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic candidate-job match data.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(seed)
    logger.info(f"Generating {n_samples} synthetic samples with seed {seed}")
    
    # Define skill categories
    tech_skills = [
        'python', 'java', 'javascript', 'sql', 'react', 'node.js', 
        'docker', 'aws', 'git', 'machine learning', 'data science',
        'kubernetes', 'terraform', 'mongodb', 'postgresql', 'redis',
        'fastapi', 'django', 'flask', 'pandas', 'numpy', 'scikit-learn'
    ]
    soft_skills = [
        'communication', 'leadership', 'problem solving', 'teamwork', 
        'adaptability', 'creativity', 'time management', 'critical thinking'
    ]
    
    locations = [
        'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 
        'Recife', 'Salvador', 'Brasília', 'Curitiba', 'Fortaleza', 'Manaus'
    ]
    
    data = []
    
    for i in range(n_samples):
        # Candidate features
        age = np.random.randint(22, 65)
        education = np.random.choice(
            ['high_school', 'bachelor', 'master', 'phd'], 
            p=[0.1, 0.5, 0.3, 0.1]
        )
        years_exp = max(0, age - 22 - np.random.randint(0, 8))
        
        # Skills (more skills for senior candidates)
        n_skills = min(15, max(2, int(np.random.normal(5 + years_exp/5, 2))))
        all_skills = tech_skills + soft_skills
        candidate_skills = np.random.choice(
            all_skills, min(n_skills, len(all_skills)), replace=False
        ).tolist()
        
        prev_companies = min(years_exp // 2, np.random.poisson(2))
        
        # Salary based on experience and education
        base_salary = 40000
        exp_bonus = years_exp * 4000
        edu_bonus = {'high_school': 0, 'bachelor': 10000, 'master': 20000, 'phd': 30000}[education]
        salary_exp = base_salary + exp_bonus + edu_bonus + np.random.normal(0, 8000)
        salary_exp = max(30000, salary_exp)
        
        location = np.random.choice(locations)
        remote_work = np.random.choice([True, False], p=[0.7, 0.3])
        availability = np.random.randint(1, 120)
        
        # Job requirements
        req_exp_level = np.random.choice(
            ['junior', 'mid', 'senior', 'lead'], 
            p=[0.3, 0.4, 0.25, 0.05]
        )
        
        # Required skills based on experience level
        n_req_skills = {
            'junior': np.random.randint(3, 6),
            'mid': np.random.randint(4, 8),
            'senior': np.random.randint(6, 10),
            'lead': np.random.randint(8, 12)
        }[req_exp_level]
        
        req_skills = np.random.choice(
            tech_skills, min(n_req_skills, len(tech_skills)), replace=False
        ).tolist()
        
        # Salary range based on experience level
        salary_base = {
            'junior': 45000,
            'mid': 70000,
            'senior': 100000,
            'lead': 140000
        }[req_exp_level]
        
        salary_min = salary_base + np.random.randint(-10000, 5000)
        salary_max = salary_min + np.random.randint(15000, 40000)
        
        job_location = np.random.choice(locations + ['Remote'])
        remote_allowed = job_location == 'Remote' or np.random.choice(
            [True, False], p=[0.6, 0.4]
        )
        urgency = np.random.randint(7, 180)
        
        # Calculate match based on rules
        match_score = calculate_match_score(
            years_exp, req_exp_level, candidate_skills, req_skills,
            salary_exp, salary_min, salary_max, location, job_location,
            remote_work, remote_allowed, availability, urgency, education
        )
        
        # Add some noise
        match_score += np.random.normal(0, 0.08)
        match_score = np.clip(match_score, 0, 1)
        
        match_label = 'good_match' if match_score > 0.65 else 'poor_match'
        
        # Calculate derived features
        skills_overlap = len(set(candidate_skills) & set(req_skills))
        skills_match_ratio = skills_overlap / len(req_skills) if req_skills else 0
        
        salary_fit = 1.0 if salary_min <= salary_exp <= salary_max else (
            0.8 if salary_exp < salary_min * 1.1 else 0.3
        )
        
        location_match = 1.0 if (
            location == job_location or job_location == 'Remote' or 
            (remote_work and remote_allowed)
        ) else 0.2
        
        remote_compatibility = 1.0 if (
            remote_work and remote_allowed
        ) else (0.8 if remote_allowed else 0.5)
        
        availability_urgency_ratio = min(1.0, urgency / max(availability, 1))
        
        exp_level_numeric = {
            'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4
        }[req_exp_level]
        
        education_numeric = {
            'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4
        }[education]
        
        data.append({
            'age': age,
            'education_level': education,
            'years_experience': years_exp,
            'skills_count': len(candidate_skills),
            'skills_match_ratio': skills_match_ratio,
            'previous_companies': prev_companies,
            'salary_expectation': salary_exp,
            'salary_fit': salary_fit,
            'location_match': location_match,
            'remote_compatibility': remote_compatibility,
            'availability_urgency_ratio': availability_urgency_ratio,
            'experience_level_numeric': exp_level_numeric,
            'education_numeric': education_numeric,
            'match_score': match_score,
            'match_label': match_label,
            # Store original data for reference
            'candidate_skills': ','.join(candidate_skills),
            'required_skills': ','.join(req_skills),
            'candidate_location': location,
            'job_location': job_location,
            'remote_work': remote_work,
            'remote_allowed': remote_allowed,
            'availability_days': availability,
            'urgency_days': urgency,
            'salary_range_min': salary_min,
            'salary_range_max': salary_max,
            'required_experience': req_exp_level
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated dataset with shape {df.shape}")
    logger.info(f"Match distribution: {df['match_label'].value_counts().to_dict()}")
    
    return df


def extract_candidate_features(candidate_data: Dict) -> Dict:
    """Extract relevant features from candidate data.
    
    Args:
        candidate_data: Raw candidate data from JSON
        
    Returns:
        Dictionary with extracted candidate features
    """
    infos_basicas = candidate_data.get('infos_basicas', {})
    infos_pessoais = candidate_data.get('informacoes_pessoais', {})
    infos_profissionais = candidate_data.get('informacoes_profissionais', {})
    formacao_idiomas = candidate_data.get('formacao_e_idiomas', {})
    cv = candidate_data.get('cv_pt', '')
    
    # Calculate age from birth date if available
    age = 30  # default
    birth_date = infos_pessoais.get('data_nascimento', '')
    if birth_date and birth_date != '0000-00-00':
        try:
            birth_year = int(birth_date.split('-')[2]) if len(birth_date.split('-')) == 3 else None
            if birth_year and birth_year > 1900:
                age = 2024 - birth_year
        except:
            pass
    
    # Extract years of experience from CV
    years_experience = extract_experience_from_cv(cv)
    
    # Extract technical skills from CV and professional info
    technical_skills = extract_skills_from_text(cv + ' ' + infos_profissionais.get('conhecimentos_tecnicos', ''))
    
    return {
        'name': infos_basicas.get('nome', ''),
        'email': infos_basicas.get('email', ''),
        'phone': infos_basicas.get('telefone', ''),
        'age': age,
        'education_level': normalize_education_level(formacao_idiomas.get('nivel_academico', '')),
        'years_experience': years_experience,
        'english_level': normalize_language_level(formacao_idiomas.get('nivel_ingles', '')),
        'spanish_level': normalize_language_level(formacao_idiomas.get('nivel_espanhol', '')),
        'technical_skills': technical_skills,
        'area_of_expertise': normalize_area(infos_profissionais.get('area_atuacao', '')),
        'salary_expectation': extract_salary(infos_profissionais.get('remuneracao', '')),
        'location': extract_location(infos_pessoais.get('endereco', '')),
        'remote_work_preference': 'Híbrido'  # default
    }


def extract_job_features(job_data: Dict) -> Dict:
    """Extract relevant features from job data.
    
    Args:
        job_data: Raw job data from JSON
        
    Returns:
        Dictionary with extracted job features
    """
    infos_basicas = job_data.get('informacoes_basicas', {})
    perfil_vaga = job_data.get('perfil_vaga', {})
    
    # Extract required skills from job description
    job_description = perfil_vaga.get('principais_atividades', '') + ' ' + perfil_vaga.get('competencia_tecnicas_e_comportamentais', '')
    required_skills = extract_skills_from_text(job_description)
    
    return {
        'job_title': infos_basicas.get('titulo_vaga', ''),
        'required_education': normalize_education_level(perfil_vaga.get('nivel_academico', '')),
        'required_experience': extract_required_experience(job_description),
        'required_english': normalize_language_level(perfil_vaga.get('nivel_ingles', '')),
        'required_spanish': normalize_language_level(perfil_vaga.get('nivel_espanhol', '')),
        'required_skills': required_skills,
        'job_area': normalize_area(perfil_vaga.get('areas_atuacao', '')),
        'salary_range_min': 5000,  # default
        'salary_range_max': 15000,  # default
        'job_location': perfil_vaga.get('cidade', 'São Paulo'),
        'remote_work_allowed': 'Híbrido',  # default
        'is_sap_position': infos_basicas.get('vaga_sap', 'Não') == 'Sim'
    }


def determine_match_quality(prospect_status: str) -> str:
    """Determine match quality based on prospect status.
    
    Args:
        prospect_status: Status of the candidate in the recruitment process
        
    Returns:
        Match quality: 'good_match' or 'poor_match'
    """
    good_statuses = [
        'Contratado pela Decision',
        'Aprovado',
        'Encaminhado ao Requisitante',
        'Documentação PJ'
    ]
    
    if any(status in prospect_status for status in good_statuses):
        return 'good_match'
    else:
        return 'poor_match'


def calculate_real_match_score(candidate: Dict, job: Dict) -> float:
    """Calculate compatibility score between candidate and job for real data.
    
    Args:
        candidate: Candidate data dictionary
        job: Job data dictionary
        
    Returns:
        Match score between 0 and 1
    """
    score = 0.0
    total_weight = 0.0
    
    # Education level match (weight: 0.2)
    education_levels = ['Ensino Médio', 'Ensino Superior', 'Pós-graduação', 'Mestrado', 'Doutorado']
    candidate_edu_idx = education_levels.index(candidate['education_level']) if candidate['education_level'] in education_levels else 0
    required_edu_idx = education_levels.index(job['required_education']) if job['required_education'] in education_levels else 0
    
    if candidate_edu_idx >= required_edu_idx:
        score += 0.2
    else:
        score += 0.2 * (candidate_edu_idx / required_edu_idx) if required_edu_idx > 0 else 0
    total_weight += 0.2
    
    # Experience match (weight: 0.25)
    exp_ratio = min(candidate['years_experience'] / max(job['required_experience'], 1), 1.0)
    score += 0.25 * exp_ratio
    total_weight += 0.25
    
    # Skills match (weight: 0.3)
    candidate_skills = set(skill.strip().lower() for skill in candidate['technical_skills'].split(',') if skill.strip())
    required_skills = set(skill.strip().lower() for skill in job['required_skills'].split(',') if skill.strip())
    
    if required_skills:
        skills_match = len(candidate_skills.intersection(required_skills)) / len(required_skills)
        score += 0.3 * skills_match
    total_weight += 0.3
    
    # Area match (weight: 0.15)
    if candidate['area_of_expertise'].lower() == job['job_area'].lower():
        score += 0.15
    total_weight += 0.15
    
    # Language requirements (weight: 0.1)
    language_levels = ['Nenhum', 'Básico', 'Intermediário', 'Avançado', 'Fluente']
    
    # English
    candidate_eng_idx = language_levels.index(candidate['english_level']) if candidate['english_level'] in language_levels else 0
    required_eng_idx = language_levels.index(job['required_english']) if job['required_english'] in language_levels else 0
    
    if candidate_eng_idx >= required_eng_idx:
        score += 0.05
    total_weight += 0.05
    
    # Spanish
    candidate_spa_idx = language_levels.index(candidate['spanish_level']) if candidate['spanish_level'] in language_levels else 0
    required_spa_idx = language_levels.index(job['required_spanish']) if job['required_spanish'] in language_levels else 0
    
    if candidate_spa_idx >= required_spa_idx:
        score += 0.05
    total_weight += 0.05
    
    return score / total_weight if total_weight > 0 else 0.0


def calculate_match_score(
    years_exp: int, req_exp_level: str, candidate_skills: List[str], 
    req_skills: List[str], salary_exp: float, salary_min: float, 
    salary_max: float, location: str, job_location: str, 
    remote_work: bool, remote_allowed: bool, availability: int, 
    urgency: int, education: str
) -> float:
    """Calculate match score based on various factors.
    
    Args:
        years_exp: Candidate years of experience
        req_exp_level: Required experience level
        candidate_skills: List of candidate skills
        req_skills: List of required skills
        salary_exp: Candidate salary expectation
        salary_min: Job minimum salary
        salary_max: Job maximum salary
        location: Candidate location
        job_location: Job location
        remote_work: Candidate accepts remote work
        remote_allowed: Job allows remote work
        availability: Candidate availability in days
        urgency: Job urgency in days
        education: Candidate education level
        
    Returns:
        Match score between 0 and 1
    """
    score = 0.0
    
    # Experience match (30% weight)
    exp_mapping = {'junior': 2, 'mid': 5, 'senior': 8, 'lead': 12}
    required_exp = exp_mapping[req_exp_level]
    
    if years_exp >= required_exp:
        exp_score = min(1.0, 0.8 + (years_exp - required_exp) * 0.02)
    elif years_exp >= required_exp * 0.7:
        exp_score = 0.6
    else:
        exp_score = max(0.1, years_exp / required_exp * 0.5)
    
    score += exp_score * 0.3
    
    # Skills match (35% weight)
    if req_skills:
        skills_overlap = len(set(candidate_skills) & set(req_skills))
        skills_ratio = skills_overlap / len(req_skills)
        # Bonus for having more skills than required
        bonus = min(0.2, (len(candidate_skills) - len(req_skills)) * 0.02)
        skills_score = min(1.0, skills_ratio + bonus)
    else:
        skills_score = 0.5
    
    score += skills_score * 0.35
    
    # Salary fit (15% weight)
    if salary_min <= salary_exp <= salary_max:
        salary_score = 1.0
    elif salary_exp < salary_min:
        salary_score = max(0.3, 1.0 - (salary_min - salary_exp) / salary_min)
    else:
        salary_score = max(0.2, 1.0 - (salary_exp - salary_max) / salary_max)
    
    score += salary_score * 0.15
    
    # Location and remote compatibility (10% weight)
    if location == job_location or job_location == 'Remote':
        location_score = 1.0
    elif remote_work and remote_allowed:
        location_score = 0.9
    elif remote_allowed:
        location_score = 0.6
    else:
        location_score = 0.2
    
    score += location_score * 0.1
    
    # Availability vs urgency (5% weight)
    if availability <= urgency:
        availability_score = 1.0
    else:
        availability_score = max(0.2, urgency / availability)
    
    score += availability_score * 0.05
    
    # Education bonus (5% weight)
    edu_mapping = {'high_school': 0.5, 'bachelor': 0.7, 'master': 0.9, 'phd': 1.0}
    education_score = edu_mapping.get(education, 0.5)
    
    score += education_score * 0.05
    
    return min(1.0, score)


def normalize_education_level(education: str) -> str:
    """Normalize education level to standard format."""
    education = education.lower().strip()
    if 'doutorado' in education or 'phd' in education:
        return 'Doutorado'
    elif 'mestrado' in education or 'master' in education:
        return 'Mestrado'
    elif 'pós' in education or 'especialização' in education:
        return 'Pós-graduação'
    elif 'superior' in education or 'graduação' in education or 'bacharel' in education:
        return 'Ensino Superior'
    else:
        return 'Ensino Médio'


def normalize_language_level(level: str) -> str:
    """Normalize language level to standard format."""
    level = level.lower().strip()
    if 'fluente' in level or 'nativo' in level:
        return 'Fluente'
    elif 'avançado' in level:
        return 'Avançado'
    elif 'intermediário' in level or 'medio' in level:
        return 'Intermediário'
    elif 'básico' in level or 'basico' in level:
        return 'Básico'
    else:
        return 'Nenhum'


def normalize_area(area: str) -> str:
    """Normalize area of expertise to standard format."""
    area = area.lower().strip()
    if 'desenvolvimento' in area or 'programação' in area:
        return 'Desenvolvimento'
    elif 'dados' in area or 'analytics' in area:
        return 'Análise de Dados'
    elif 'devops' in area or 'infraestrutura' in area:
        return 'DevOps'
    elif 'qa' in area or 'qualidade' in area or 'teste' in area:
        return 'QA'
    elif 'gestão' in area or 'gerência' in area:
        return 'Gestão de Projetos'
    elif 'ux' in area or 'ui' in area or 'design' in area:
        return 'UX/UI Design'
    elif 'arquitetura' in area:
        return 'Arquitetura de Software'
    elif 'segurança' in area:
        return 'Segurança da Informação'
    else:
        return 'Desenvolvimento'


def extract_experience_from_cv(cv_text: str) -> int:
    """Extract years of experience from CV text."""
    if not cv_text:
        return 0
    
    # Look for patterns like "5 anos", "3 years", etc.
    patterns = [
        r'(\d+)\s*anos?\s*de\s*experiência',
        r'(\d+)\s*years?\s*of\s*experience',
        r'experiência\s*de\s*(\d+)\s*anos?',
        r'(\d+)\s*anos?\s*trabalhando'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cv_text.lower())
        if matches:
            return int(matches[0])
    
    # Fallback: count job positions mentioned
    job_keywords = ['empresa', 'trabalho', 'cargo', 'função', 'posição']
    job_count = sum(cv_text.lower().count(keyword) for keyword in job_keywords)
    return min(job_count * 2, 15)  # Estimate 2 years per job, max 15


def extract_skills_from_text(text: str) -> str:
    """Extract technical skills from text."""
    if not text:
        return ''
    
    tech_skills = [
        'python', 'java', 'javascript', 'sql', 'react', 'angular', 'node.js',
        'aws', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'machine learning',
        'c#', 'c++', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin',
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
        'jenkins', 'terraform', 'ansible', 'linux', 'windows'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in tech_skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return ', '.join(found_skills[:10])  # Limit to 10 skills


def extract_salary(salary_text: str) -> float:
    """Extract salary expectation from text."""
    if not salary_text:
        return 8000.0  # default
    
    # Look for numbers in the text
    numbers = re.findall(r'\d+', salary_text)
    if numbers:
        salary = int(numbers[0])
        # If it's a very large number, assume it's annual
        if salary > 50000:
            return salary / 12
        return salary
    
    return 8000.0  # default


def extract_location(address: str) -> str:
    """Extract city from address."""
    if not address:
        return 'São Paulo'
    
    cities = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Recife', 'Salvador', 'Brasília', 'Curitiba', 'Fortaleza', 'Manaus']
    
    for city in cities:
        if city.lower() in address.lower():
            return city
    
    return 'São Paulo'  # default


def extract_required_experience(job_description: str) -> int:
    """Extract required years of experience from job description."""
    if not job_description:
        return 3  # default
    
    patterns = [
        r'(\d+)\s*anos?\s*de\s*experiência',
        r'(\d+)\s*years?\s*of\s*experience',
        r'experiência\s*de\s*(\d+)\s*anos?',
        r'mínimo\s*de\s*(\d+)\s*anos?'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, job_description.lower())
        if matches:
            return int(matches[0])
    
    # Look for seniority levels
    if 'senior' in job_description.lower() or 'sênior' in job_description.lower():
        return 5
    elif 'pleno' in job_description.lower() or 'mid' in job_description.lower():
        return 3
    elif 'junior' in job_description.lower() or 'júnior' in job_description.lower():
        return 1
    
    return 3  # default


def convert_to_model_format(candidate: Dict, job: Dict, match_score: float, match_label: str) -> Dict:
    """Convert real data to the format expected by the model."""
    # Map education levels to numeric values
    education_mapping = {
        'Ensino Médio': 1,
        'Ensino Superior': 2,
        'Pós-graduação': 3,
        'Mestrado': 4,
        'Doutorado': 5
    }
    
    # Map experience levels to numeric values
    req_exp = job['required_experience']
    if req_exp <= 2:
        exp_level = 'junior'
        exp_level_numeric = 1
    elif req_exp <= 5:
        exp_level = 'mid'
        exp_level_numeric = 2
    elif req_exp <= 8:
        exp_level = 'senior'
        exp_level_numeric = 3
    else:
        exp_level = 'lead'
        exp_level_numeric = 4
    
    # Calculate derived features
    candidate_skills = set(skill.strip().lower() for skill in candidate['technical_skills'].split(',') if skill.strip())
    required_skills = set(skill.strip().lower() for skill in job['required_skills'].split(',') if skill.strip())
    
    skills_overlap = len(candidate_skills & required_skills)
    skills_match_ratio = skills_overlap / len(required_skills) if required_skills else 0
    
    salary_fit = 1.0 if job['salary_range_min'] <= candidate['salary_expectation'] <= job['salary_range_max'] else 0.5
    
    location_match = 1.0 if candidate['location'] == job['job_location'] else 0.5
    
    remote_work = candidate['remote_work_preference'] in ['Sim', 'Híbrido']
    remote_allowed = job['remote_work_allowed'] in ['Sim', 'Híbrido']
    remote_compatibility = 1.0 if (remote_work and remote_allowed) else 0.5
    
    return {
        'age': candidate['age'],
        'education_level': candidate['education_level'],
        'years_experience': candidate['years_experience'],
        'skills_count': len(candidate_skills),
        'skills_match_ratio': skills_match_ratio,
        'previous_companies': max(1, candidate['years_experience'] // 3),  # estimate
        'salary_expectation': candidate['salary_expectation'],
        'salary_fit': salary_fit,
        'location_match': location_match,
        'remote_compatibility': remote_compatibility,
        'availability_urgency_ratio': 1.0,  # default
        'experience_level_numeric': exp_level_numeric,
        'education_numeric': education_mapping.get(candidate['education_level'], 2),
        'match_score': match_score,
        'match_label': match_label,
        # Store original data for reference
        'candidate_skills': candidate['technical_skills'],
        'required_skills': job['required_skills'],
        'candidate_location': candidate['location'],
        'job_location': job['job_location'],
        'remote_work': remote_work,
        'remote_allowed': remote_allowed,
        'availability_days': 30,  # default
        'urgency_days': 45,  # default
        'salary_range_min': job['salary_range_min'],
        'salary_range_max': job['salary_range_max'],
        'required_experience': exp_level
    }


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate training data.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If file doesn't exist and can't be created
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.info(f"Data file {file_path} not found. Trying to load real data first.")
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to load real data first
            df = load_real_data()
            
            # If real data is empty, generate synthetic data
            if df.empty:
                logger.info("Real data not available. Generating synthetic data.")
                df = generate_synthetic_data()
            
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
        else:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file_path}")
        
        # Validate required columns
        required_cols = ['match_score', 'match_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types and ranges
        if df['match_score'].dtype not in ['float64', 'float32']:
            df['match_score'] = pd.to_numeric(df['match_score'], errors='coerce')
        
        if df['match_score'].isna().any():
            raise ValueError("Invalid match_score values found")
        
        if not df['match_score'].between(0, 1).all():
            raise ValueError("match_score values must be between 0 and 1")
        
        valid_labels = ['good_match', 'poor_match']
        if not df['match_label'].isin(valid_labels).all():
            raise ValueError(f"match_label must be one of {valid_labels}")
        
        logger.info(f"Data validation passed. Shape: {df.shape}")
        logger.info(f"Label distribution: {df['match_label'].value_counts().to_dict()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Features for training (exclude target and metadata columns)
    feature_cols = [
        'age', 'education_numeric', 'years_experience', 'skills_count',
        'skills_match_ratio', 'previous_companies', 'salary_expectation',
        'salary_fit', 'location_match', 'remote_compatibility',
        'availability_urgency_ratio', 'experience_level_numeric'
    ]
    
    # Ensure all feature columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing feature columns: {missing_features}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['match_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Reconstruct full dataframes
    train_indices = X_train.index
    test_indices = X_test.index
    
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Train label distribution: {train_df['match_label'].value_counts().to_dict()}")
    logger.info(f"Test label distribution: {test_df['match_label'].value_counts().to_dict()}")
    
    return train_df, test_df


def create_sample_payload(output_path: str) -> None:
    """Create sample JSON payload for API testing.
    
    Args:
        output_path: Path to save the sample payload
    """
    sample_payload = {
        "candidate": {
            "age": 28,
            "education_level": "bachelor",
            "years_experience": 5,
            "skills": ["python", "machine learning", "sql", "pandas", "scikit-learn"],
            "previous_companies": 2,
            "salary_expectation": 85000,
            "location": "São Paulo",
            "remote_work": True,
            "availability_days": 30
        },
        "job": {
            "required_experience": "mid",
            "required_skills": ["python", "sql", "machine learning", "pandas"],
            "salary_range_min": 75000,
            "salary_range_max": 95000,
            "location": "São Paulo",
            "remote_allowed": True,
            "urgency_days": 45
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_payload, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample payload saved to {output_path}")


if __name__ == "__main__":
    # Generate sample data for testing
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Try to load real data first
    logger.info("Attempting to load real recruitment data...")
    df = load_real_data()
    
    # If real data is empty or not available, generate synthetic data
    if df.empty:
        logger.info("Real data not available. Generating synthetic data.")
        df = generate_synthetic_data(1000)
    
    # Save the data
    df.to_csv(data_dir / "sample_candidates.csv", index=False)
    logger.info(f"Data saved to {data_dir / 'sample_candidates.csv'}")
    
    # Create sample payload
    create_sample_payload(data_dir / "sample_payload.json")
    
    print(f"Generated {len(df)} samples")
    print(f"Label distribution: {df['match_label'].value_counts()}")
    logger.info("Data generation completed successfully!")
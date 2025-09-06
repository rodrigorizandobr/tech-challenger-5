"""Data loading, validation and synthetic data generation module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path
import json
from loguru import logger
from sklearn.model_selection import train_test_split
import os


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
            logger.info(f"Data file {file_path} not found. Generating synthetic data.")
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = generate_synthetic_data()
            df.to_csv(file_path, index=False)
            logger.info(f"Synthetic data saved to {file_path}")
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
    
    # Generate and save synthetic data
    df = generate_synthetic_data(1000)
    df.to_csv(data_dir / "sample_candidates.csv", index=False)
    
    # Create sample payload
    create_sample_payload(data_dir / "sample_payload.json")
    
    print(f"Generated {len(df)} samples")
    print(f"Label distribution: {df['match_label'].value_counts()}")
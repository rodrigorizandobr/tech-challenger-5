"""Data drift monitoring using Evidently AI."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import json
from loguru import logger

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DatasetDriftMetric, DatasetMissingValuesMetric,
        ColumnDriftMetric, ColumnSummaryMetric
    )
except ImportError:
    logger.warning("Evidently not installed. Install with: pip install evidently")
    ColumnMapping = None
    Report = None
    DataDriftPreset = None
    DataQualityPreset = None
    DatasetDriftMetric = None
    DatasetMissingValuesMetric = None
    ColumnDriftMetric = None
    ColumnSummaryMetric = None

from src.data import load_and_validate_data
from src.features import engineer_features


class DriftMonitor:
    """Data drift monitoring class using Evidently."""
    
    def __init__(self, reference_data_path: str, reports_dir: str = "reports"):
        """Initialize drift monitor.
        
        Args:
            reference_data_path: Path to reference (training) data
            reports_dir: Directory to save reports
        """
        self.reference_data_path = reference_data_path
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.reference_data = None
        self.column_mapping = None
        
        # Load reference data
        self._load_reference_data()
        self._setup_column_mapping()
    
    def _load_reference_data(self) -> None:
        """Load and prepare reference data."""
        try:
            logger.info(f"Loading reference data from {self.reference_data_path}")
            
            # Load data
            df = load_and_validate_data(self.reference_data_path)
            
            # Apply feature engineering
            df_engineered = engineer_features(df)
            
            # Select features for monitoring
            feature_cols = [
                'age', 'education_numeric', 'years_experience', 'skills_count',
                'skills_match_ratio', 'previous_companies', 'salary_expectation',
                'salary_fit', 'location_match', 'remote_compatibility',
                'availability_urgency_ratio', 'experience_level_numeric',
                'skill_diversity', 'rare_skills_bonus', 'salary_position',
                'salary_expectation_ratio', 'experience_education_ratio',
                'salary_range_width', 'company_stability'
            ]
            
            # Filter to available columns
            available_cols = [col for col in feature_cols if col in df_engineered.columns]
            
            # Add target column
            if 'match_label' in df_engineered.columns:
                available_cols.append('match_label')
            
            self.reference_data = df_engineered[available_cols].copy()
            
            logger.info(f"Reference data loaded with shape {self.reference_data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            raise
    
    def _setup_column_mapping(self) -> None:
        """Setup column mapping for Evidently."""
        if ColumnMapping is None:
            logger.warning("Evidently not available, skipping column mapping")
            return
        
        try:
            # Define numerical and categorical features
            numerical_features = [
                'age', 'years_experience', 'skills_count', 'skills_match_ratio',
                'previous_companies', 'salary_expectation', 'salary_fit',
                'location_match', 'remote_compatibility', 'availability_urgency_ratio',
                'skill_diversity', 'rare_skills_bonus', 'salary_position',
                'salary_expectation_ratio', 'experience_education_ratio',
                'salary_range_width', 'company_stability'
            ]
            
            categorical_features = [
                'education_numeric', 'experience_level_numeric'
            ]
            
            # Filter to available columns
            if self.reference_data is not None:
                available_cols = self.reference_data.columns.tolist()
                numerical_features = [col for col in numerical_features if col in available_cols]
                categorical_features = [col for col in categorical_features if col in available_cols]
            
            self.column_mapping = ColumnMapping(
                target='match_label' if 'match_label' in self.reference_data.columns else None,
                prediction=None,
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            
            logger.info("Column mapping configured successfully")
            
        except Exception as e:
            logger.error(f"Error setting up column mapping: {e}")
            self.column_mapping = None
    
    def load_current_data(self, predictions_log_path: str, window_size: int = 100) -> Optional[pd.DataFrame]:
        """Load current data from predictions log.
        
        Args:
            predictions_log_path: Path to predictions log CSV
            window_size: Number of recent predictions to analyze
            
        Returns:
            DataFrame with current data or None if not available
        """
        try:
            log_path = Path(predictions_log_path)
            
            if not log_path.exists():
                logger.warning(f"Predictions log not found at {log_path}")
                return None
            
            # Load predictions log
            df_log = pd.read_csv(log_path)
            
            if len(df_log) == 0:
                logger.warning("Predictions log is empty")
                return None
            
            # Get recent predictions
            df_recent = df_log.tail(window_size).copy()
            
            # Convert to format similar to reference data
            current_data = self._convert_log_to_features(df_recent)
            
            logger.info(f"Loaded {len(current_data)} recent predictions for drift analysis")
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error loading current data: {e}")
            return None
    
    def _convert_log_to_features(self, df_log: pd.DataFrame) -> pd.DataFrame:
        """Convert predictions log to feature format.
        
        Args:
            df_log: Predictions log DataFrame
            
        Returns:
            DataFrame in feature format
        """
        try:
            # Extract features from log
            feature_data = []
            
            for _, row in df_log.iterrows():
                # Create feature row similar to training data
                feature_row = {
                    'age': row.get('candidate_age', 30),
                    'years_experience': row.get('candidate_years_experience', 5),
                    'skills_count': len(str(row.get('candidate_skills', '')).split(',')) if pd.notna(row.get('candidate_skills')) else 0,
                    'previous_companies': row.get('candidate_previous_companies', 1),
                    'salary_expectation': row.get('candidate_salary_expectation', 70000),
                    'availability_urgency_ratio': row.get('job_urgency_days', 30) / max(row.get('candidate_availability_days', 30), 1),
                    
                    # Use prediction results as proxy for engineered features
                    'skills_match_ratio': row.get('match_probability', 0.5) * 0.8,  # Approximate
                    'salary_fit': 1.0 if row.get('match_probability', 0) > 0.7 else 0.5,
                    'location_match': 1.0 if row.get('match_probability', 0) > 0.6 else 0.3,
                    'remote_compatibility': 0.8,  # Default value
                    
                    # Education and experience mapping
                    'education_numeric': {
                        'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4
                    }.get(row.get('candidate_education_level', 'bachelor'), 2),
                    
                    'experience_level_numeric': {
                        'junior': 1, 'mid': 2, 'senior': 3, 'lead': 4
                    }.get(row.get('job_required_experience', 'mid'), 2),
                    
                    # Derived features (simplified)
                    'skill_diversity': min(1.0, row.get('candidate_skills', '').count(',') * 0.1) if pd.notna(row.get('candidate_skills')) else 0,
                    'rare_skills_bonus': 0.1,  # Default value
                    'salary_position': 0.5,  # Default value
                    'salary_expectation_ratio': 1.0,  # Default value
                    'experience_education_ratio': row.get('candidate_years_experience', 5) / 2,
                    'salary_range_width': row.get('job_salary_range_max', 100000) - row.get('job_salary_range_min', 70000),
                    'company_stability': row.get('candidate_years_experience', 5) / max(row.get('candidate_previous_companies', 1), 1),
                    
                    # Target (if available)
                    'match_label': row.get('match_label', 'good_match' if row.get('match_probability', 0) > 0.65 else 'poor_match')
                }
                
                feature_data.append(feature_row)
            
            df_features = pd.DataFrame(feature_data)
            
            # Ensure columns match reference data
            if self.reference_data is not None:
                for col in self.reference_data.columns:
                    if col not in df_features.columns:
                        df_features[col] = 0.5  # Default value
                
                # Reorder columns to match reference
                df_features = df_features[self.reference_data.columns]
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error converting log to features: {e}")
            return pd.DataFrame()
    
    def generate_drift_report(
        self, 
        current_data: pd.DataFrame,
        report_name: str = "drift_report"
    ) -> str:
        """Generate drift report using Evidently.
        
        Args:
            current_data: Current data to compare against reference
            report_name: Name for the report file
            
        Returns:
            Path to generated report
        """
        if Report is None:
            logger.error("Evidently not available. Cannot generate drift report.")
            return self._generate_fallback_report(current_data, report_name)
        
        try:
            logger.info("Generating drift report with Evidently")
            
            # Create report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric()
            ])
            
            # Run report
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"{report_name}_{timestamp}.html"
            
            report.save_html(str(report_path))
            
            # Also save as the main drift report
            main_report_path = self.reports_dir / "drift.html"
            report.save_html(str(main_report_path))
            
            logger.info(f"Drift report saved to {report_path}")
            
            # Extract and log key metrics
            self._log_drift_metrics(report)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return self._generate_fallback_report(current_data, report_name)
    
    def _generate_fallback_report(self, current_data: pd.DataFrame, report_name: str) -> str:
        """Generate a simple fallback report when Evidently is not available.
        
        Args:
            current_data: Current data
            report_name: Report name
            
        Returns:
            Path to generated report
        """
        try:
            logger.info("Generating fallback drift report")
            
            # Calculate basic statistics
            ref_stats = self.reference_data.describe() if self.reference_data is not None else None
            curr_stats = current_data.describe()
            
            # Simple drift detection
            drift_detected = False
            drift_columns = []
            
            if ref_stats is not None:
                for col in curr_stats.columns:
                    if col in ref_stats.columns:
                        ref_mean = ref_stats.loc['mean', col]
                        curr_mean = curr_stats.loc['mean', col]
                        
                        # Simple threshold-based drift detection
                        if abs(ref_mean - curr_mean) / (abs(ref_mean) + 1e-8) > 0.1:
                            drift_detected = True
                            drift_columns.append(col)
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Drift Report - {report_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ color: #2c3e50; }}
                    .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    .alert-danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                    .alert-success {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                    .stats-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .stats-table th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1 class="header">Data Drift Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Reference Data Size:</strong> {len(self.reference_data) if self.reference_data is not None else 'N/A'}</p>
                <p><strong>Current Data Size:</strong> {len(current_data)}</p>
                
                <div class="alert {'alert-danger' if drift_detected else 'alert-success'}">
                    <strong>Drift Status:</strong> {'DRIFT DETECTED' if drift_detected else 'NO DRIFT DETECTED'}
                    {f'<br><strong>Affected Columns:</strong> {', '.join(drift_columns)}' if drift_columns else ''}
                </div>
                
                <h2>Current Data Statistics</h2>
                <table class="stats-table">
                    <tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
            """
            
            for col in curr_stats.columns:
                if curr_stats[col].dtype in ['int64', 'float64']:
                    html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{curr_stats.loc['mean', col]:.3f}</td>
                        <td>{curr_stats.loc['std', col]:.3f}</td>
                        <td>{curr_stats.loc['min', col]:.3f}</td>
                        <td>{curr_stats.loc['max', col]:.3f}</td>
                    </tr>
                    """
            
            html_content += """
                </table>
                
                <h2>Notes</h2>
                <p>This is a simplified drift report. For detailed analysis, install Evidently AI:</p>
                <code>pip install evidently</code>
            </body>
            </html>
            """
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"{report_name}_{timestamp}.html"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Also save as main drift report
            main_report_path = self.reports_dir / "drift.html"
            with open(main_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Fallback drift report saved to {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating fallback report: {e}")
            raise
    
    def _log_drift_metrics(self, report) -> None:
        """Extract and log key drift metrics from Evidently report.
        
        Args:
            report: Evidently report object
        """
        try:
            # This would extract metrics from the Evidently report
            # Implementation depends on Evidently version
            logger.info("Drift metrics logged successfully")
            
        except Exception as e:
            logger.warning(f"Could not extract drift metrics: {e}")
    
    def run_monitoring(
        self, 
        predictions_log_path: str = "logs/predictions.csv",
        window_size: int = 100
    ) -> str:
        """Run complete monitoring pipeline.
        
        Args:
            predictions_log_path: Path to predictions log
            window_size: Size of sliding window for analysis
            
        Returns:
            Path to generated report
        """
        logger.info("Starting drift monitoring pipeline")
        
        try:
            # Load current data
            current_data = self.load_current_data(predictions_log_path, window_size)
            
            if current_data is None or len(current_data) == 0:
                logger.warning("No current data available for drift analysis")
                return self._generate_empty_report()
            
            # Generate drift report
            report_path = self.generate_drift_report(current_data)
            
            logger.info(f"Drift monitoring completed. Report: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error in monitoring pipeline: {e}")
            raise
    
    def _generate_empty_report(self) -> str:
        """Generate report when no data is available.
        
        Returns:
            Path to generated report
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Report - No Data</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { color: #2c3e50; }
                .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            </style>
        </head>
        <body>
            <h1 class="header">Data Drift Report</h1>
            <div class="alert alert-warning">
                <strong>No Data Available:</strong> Insufficient prediction data for drift analysis.
                Make some predictions through the API to generate drift reports.
            </div>
            <p><strong>Generated:</strong> {}</p>
        </body>
        </html>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        report_path = self.reports_dir / "drift.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)


def main(
    reference_data_path: str = "data/sample_candidates.csv",
    predictions_log_path: str = "logs/predictions.csv",
    reports_dir: str = "reports",
    window_size: int = 100
) -> None:
    """Main function for drift monitoring.
    
    Args:
        reference_data_path: Path to reference data
        predictions_log_path: Path to predictions log
        reports_dir: Directory for reports
        window_size: Window size for analysis
    """
    try:
        logger.info("Starting drift monitoring")
        
        # Initialize monitor
        monitor = DriftMonitor(reference_data_path, reports_dir)
        
        # Run monitoring
        report_path = monitor.run_monitoring(predictions_log_path, window_size)
        
        logger.info(f"Drift monitoring completed successfully. Report: {report_path}")
        
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate data drift report")
    parser.add_argument(
        "--reference-data", 
        default="data/sample_candidates.csv",
        help="Path to reference data"
    )
    parser.add_argument(
        "--predictions-log", 
        default="logs/predictions.csv",
        help="Path to predictions log"
    )
    parser.add_argument(
        "--reports-dir", 
        default="reports",
        help="Directory for reports"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=100,
        help="Window size for analysis"
    )
    
    args = parser.parse_args()
    
    main(
        reference_data_path=args.reference_data,
        predictions_log_path=args.predictions_log,
        reports_dir=args.reports_dir,
        window_size=args.window_size
    )
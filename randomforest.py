from __future__ import annotations
import os
import sys
import logging
import json
import pickle
import yaml
import hashlib
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union, Any, Protocol
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum, auto
from abc import ABC, abstractmethod
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Data Science Stack
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import scipy.stats as stats
from scipy import sparse

# ML Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    VarianceThreshold,
    mutual_info_classif
)
from sklearn.calibration import CalibratedClassifierCV

# MLOps & Experiment Tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import wandb
from prometheus_client import Counter, Histogram, start_http_server
import great_expectations as gx

# Feature Store & Data Validation
import pandas_profiling
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno

# System & Monitoring
import joblib
import dill
import cloudpickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiofiles
import psutil
import GPUtil
import docker
import boto3
from botocore.exceptions import ClientError

# Testing
import pytest
import hypothesis
from hypothesis import given, strategies as st
import unittest.mock as mock

# Type Checking
from pydantic import BaseModel, Field, validator, root_validator
from typing_extensions import Literal, Annotated
import typing

# ==================== CONFIGURATION MANAGEMENT ====================
class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"
    AUTO_ML = "auto_ml"

class FeatureEngineeringStrategy(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    DEEP = "deep"

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    numerical_imputation: str = "median"  # mean, median, constant, knn
    categorical_imputation: str = "most_frequent"
    scaling: str = "standard"  # standard, minmax, robust, none
    encoding: str = "onehot"  # onehot, ordinal, target
    feature_selection: bool = True
    selection_method: str = "mutual_info"  # variance, mutual_info, model_based
    n_features: Union[int, str] = "auto"
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = True
    binning_numerical: bool = False
    n_bins: int = 10
    datetime_features: bool = True
    text_features: bool = False
    outlier_handling: str = "clip"  # clip, remove, transform
    outlier_threshold: float = 3.0

@dataclass
class ModelConfig:
    """Model training configuration"""
    # Model Selection
    model_type: ModelType = ModelType.RANDOM_FOREST
    use_ensemble: bool = True
    ensemble_method: str = "voting"  # voting, stacking, blending
    calibration_method: str = "isotonic"  # isotonic, sigmoid, none
    
    # Random Forest Parameters
    n_estimators: int = 300
    max_depth: Union[int, None] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    max_features: Union[str, int, float] = "sqrt"
    bootstrap: bool = True
    class_weight: Union[str, Dict, None] = "balanced"
    ccp_alpha: float = 0.0
    
    # Hyperparameter Tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "random"  # grid, random, bayesian
    n_iter: int = 100
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    
    # Training Parameters
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    # Advanced
    early_stopping: bool = False
    warm_start: bool = False
    oob_score: bool = True
    monotonic_cst: Optional[List] = None
    
    # MLOps
    experiment_name: str = "classification-pipeline"
    run_name: str = "random-forest"
    tracking_uri: str = "mlruns"
    registry_uri: str = "models"
    model_registry: bool = True
    model_versioning: bool = True
    
    def to_dict(self):
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in self.__dict__.items()}

@dataclass
class DataConfig:
    """Data pipeline configuration"""
    data_path: str = "data/dataset.csv"
    test_size: float = 0.2
    validation_size: float = 0.1
    time_based_split: bool = False
    time_column: Optional[str] = None
    target_column: str = "target"
    data_version: str = "1.0.0"
    
    # Data Validation
    validation_rules: List[Dict] = field(default_factory=list)
    data_quality_threshold: float = 0.95
    drift_detection: bool = True
    drift_threshold: float = 0.1
    
    # Data Catalog
    catalog_name: str = "ml_data_catalog"
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Paths
    base_dir: Path = Path.cwd()
    data_dir: Path = base_dir / "data"
    model_dir: Path = base_dir / "models"
    artifacts_dir: Path = base_dir / "artifacts"
    logs_dir: Path = base_dir / "logs"
    tests_dir: Path = base_dir / "tests"
    
    # Components
    data_config: DataConfig = field(default_factory=DataConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Runtime
    debug_mode: bool = False
    profile_memory: bool = True
    profile_performance: bool = True
    use_gpu: bool = False
    batch_size: Optional[int] = None
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_port: int = 9090
    metrics_prefix: str = "ml_pipeline"
    
    # CI/CD
    run_tests: bool = True
    run_linting: bool = True
    quality_gate: Dict = field(default_factory=lambda: {
        "accuracy": 0.85,
        "roc_auc": 0.90,
        "f1_score": 0.80,
        "data_quality": 0.95
    })
    
    def __post_init__(self):
        """Ensure directories exist"""
        for dir_path in [self.data_dir, self.model_dir, 
                        self.artifacts_dir, self.logs_dir, self.tests_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load configuration from YAML"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        return {
            'data_config': self.data_config.__dict__,
            'feature_config': self.feature_config.__dict__,
            'model_config': self.model_config.to_dict(),
            'paths': {
                'base_dir': str(self.base_dir),
                'data_dir': str(self.data_dir),
                'model_dir': str(self.model_dir),
                'artifacts_dir': str(self.artifacts_dir),
                'logs_dir': str(self.logs_dir),
                'tests_dir': str(self.tests_dir)
            },
            'runtime': {
                'debug_mode': self.debug_mode,
                'profile_memory': self.profile_memory,
                'profile_performance': self.profile_performance,
                'use_gpu': self.use_gpu,
                'batch_size': self.batch_size
            }
        }

# ==================== ADVANCED LOGGING ====================
class StructuredLogger:
    """Enterprise logging with structured JSON output"""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging with multiple handlers"""
        
        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(module)s", "function": "%(funcName)s", '
            '"line": %(lineno)d, "message": "%(message)s"}'
        )
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        
        file_handler = logging.FileHandler(
            self.config.logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        json_handler = logging.FileHandler(
            self.config.logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_structured.json"
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(json_formatter)
        
        # Configure root logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Remove existing handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Log startup
        self.logger.info(f"Logger initialized for {self.name}")
        self.logger.info(f"Log files: {self.config.logs_dir}")
    
    def log_performance(self, operation: str, duration: float, 
                       memory_usage: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            f"Performance: {operation} - "
            f"Duration: {duration:.4f}s, "
            f"Memory: {memory_usage:.2f}MB",
            extra={
                "operation": operation,
                "duration": duration,
                "memory_usage": memory_usage,
                **kwargs
            }
        )
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exceptions with context"""
        self.logger.error(
            f"Exception in {context}: {str(exception)}",
            exc_info=True,
            extra={"exception_type": type(exception).__name__, "context": context}
        )
    
    def log_data_quality(self, quality_metrics: Dict):
        """Log data quality metrics"""
        self.logger.info(
            f"Data Quality: {quality_metrics}",
            extra={"data_quality": quality_metrics}
        )
    
    def get_logger(self) -> logging.Logger:
        return self.logger

# ==================== METRICS COLLECTOR ====================
class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self, prefix: str = "ml_pipeline"):
        self.prefix = prefix
        
        # Define Prometheus metrics
        self.training_counter = Counter(
            f'{prefix}_training_total', 
            'Total training runs'
        )
        self.prediction_counter = Counter(
            f'{prefix}_predictions_total',
            'Total predictions made'
        )
        self.error_counter = Counter(
            f'{prefix}_errors_total',
            'Total errors',
            ['error_type']
        )
        self.training_duration = Histogram(
            f'{prefix}_training_duration_seconds',
            'Training duration in seconds'
        )
        self.prediction_duration = Histogram(
            f'{prefix}_prediction_duration_seconds',
            'Prediction duration in seconds'
        )
        self.model_accuracy = Histogram(
            f'{prefix}_model_accuracy',
            'Model accuracy distribution'
        )
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, Any] = {}
    
    def record_training_start(self):
        """Record training start"""
        self.training_counter.inc()
    
    def record_training_end(self, duration: float, accuracy: float):
        """Record training completion"""
        self.training_duration.observe(duration)
        self.model_accuracy.observe(accuracy)
    
    def record_prediction(self, duration: float):
        """Record prediction"""
        self.prediction_counter.inc()
        self.prediction_duration.observe(duration)
    
    def record_error(self, error_type: str):
        """Record error"""
        self.error_counter.labels(error_type=error_type).inc()
    
    def record_custom_metric(self, name: str, value: float, 
                           labels: Optional[Dict] = None):
        """Record custom metric"""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = {
                'values': [],
                'labels': labels or {}
            }
        self.custom_metrics[name]['values'].append(value)
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'training_runs': self.training_counter._value.get(),
            'predictions': self.prediction_counter._value.get(),
            'custom_metrics': self.custom_metrics
        }

# ==================== DATA VALIDATION ====================
class DataValidator:
    """Comprehensive data validation using Great Expectations"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.context = gx.get_context()
        self.expectation_suite_name = f"{config.data_config.data_version}_suite"
        self.validation_results = []
        
    def create_expectation_suite(self) -> gx.ExpectationSuite:
        """Create expectation suite for data validation"""
        
        suite = self.context.create_expectation_suite(
            expectation_suite_name=self.expectation_suite_name
        )
        
        # Add expectations
        expectations = [
            # Column expectations
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": self.config.data_config.target_column}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": self.config.data_config.target_column}
            },
            {
                "expectation_type": "expect_column_distinct_values_to_be_in_set",
                "kwargs": {"column": self.config.data_config.target_column, 
                          "value_set": [0, 1]}
            },
            # Data quality expectations
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 100, "max_value": 1000000}
            },
            {
                "expectation_type": "expect_table_columns_to_match_ordered_list",
                "kwargs": {"column_list": None}  # Will be set dynamically
            }
        ]
        
        # Add expectations to suite
        for exp in expectations:
            suite.add_expectation(
                gx.ExpectationConfiguration(
                    expectation_type=exp["expectation_type"],
                    kwargs=exp["kwargs"]
                )
            )
        
        return suite
    
    def validate_data(self, df: pd.DataFrame) -> gx.ValidationResult:
        """Validate dataframe against expectation suite"""
        
        # Create or load expectation suite
        try:
            suite = self.context.get_expectation_suite(
                expectation_suite_name=self.expectation_suite_name
            )
        except:
            suite = self.create_expectation_suite()
        
        # Create batch request
        batch_request = {
            "datasource_name": "pandas_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "dataset",
            "runtime_parameters": {"batch_data": df},
            "batch_identifiers": {
                "default_identifier_name": "default_identifier"
            }
        }
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=self.expectation_suite_name
        )
        
        validation_result = validator.validate()
        
        # Store results
        self.validation_results.append(validation_result)
        
        return validation_result
    
    def check_data_drift(self, reference_df: pd.DataFrame, 
                        current_df: pd.DataFrame) -> Dict:
        """Check for data drift using statistical tests"""
        
        drift_metrics = {}
        
        # For each column, perform appropriate drift test
        for column in reference_df.columns:
            ref_data = reference_df[column]
            curr_data = current_df[column]
            
            if is_numeric_dtype(ref_data):
                # KS test for numerical data
                stat, p_value = stats.ks_2samp(ref_data, curr_data)
                drift_metrics[column] = {
                    "test": "ks_test",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
            else:
                # Chi-square test for categorical data
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(
                    ref_data, 
                    pd.Series(['reference'] * len(ref_data) + ['current'] * len(curr_data))
                )
                stat, p_value, dof, expected = chi2_contingency(contingency_table)
                drift_metrics[column] = {
                    "test": "chi2_test",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
        
        return drift_metrics
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return {}
        
        latest_result = self.validation_results[-1]
        
        report = {
            "validation_time": datetime.now().isoformat(),
            "success": latest_result.success,
            "statistics": {
                "evaluated_expectations": latest_result.statistics["evaluated_expectations"],
                "successful_expectations": latest_result.statistics["successful_expectations"],
                "unsuccessful_expectations": latest_result.statistics["unsuccessful_expectations"],
                "success_percent": latest_result.statistics["success_percent"]
            },
            "results": []
        }
        
        for result in latest_result.results:
            report["results"].append({
                "expectation_type": result.expectation_config.expectation_type,
                "success": result.success,
                "observed_value": result.result.get("observed_value"),
                "exception_info": result.exception_info
            })
        
        return report

# ==================== ADVANCED FEATURE ENGINEERING ====================
class FeatureEngineeringPipeline:
    """Advanced feature engineering with multiple strategies"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config.feature_config
        self.preprocessor = None
        self.feature_selector = None
        self.feature_names = []
        
    def create_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create comprehensive preprocessing pipeline"""
        
        # Identify column types
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Numerical transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', self._get_imputer('numerical')),
            ('scaler', self._get_scaler()),
            ('outlier', self._get_outlier_handler())
        ])
        
        # Categorical transformers
        categorical_transformer = Pipeline(steps=[
            ('imputer', self._get_imputer('categorical')),
            ('encoder', self._get_encoder())
        ])
        
        # Datetime transformers
        datetime_transformer = Pipeline(steps=[
            ('extractor', FunctionTransformer(self._extract_datetime_features))
        ])
        
        # Create column transformer
        transformers = []
        
        if numerical_features:
            transformers.append(('num', numerical_transformer, numerical_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        if datetime_features:
            transformers.append(('datetime', datetime_transformer, datetime_features))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Feature selection
        if self.config.feature_selection:
            self.feature_selector = self._get_feature_selector()
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', self.feature_selector)
            ])
        else:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        self.preprocessor = pipeline
        return pipeline
    
    def _get_imputer(self, dtype: str) -> SimpleImputer:
        """Get appropriate imputer"""
        if dtype == 'numerical':
            strategy = self.config.numerical_imputation
            if strategy == 'knn':
                return KNNImputer(n_neighbors=5)
            else:
                return SimpleImputer(strategy=strategy)
        else:  # categorical
            return SimpleImputer(strategy=self.config.categorical_imputation)
    
    def _get_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Get appropriate scaler"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': FunctionTransformer(lambda x: x)  # Identity transformer
        }
        return scalers.get(self.config.scaling, StandardScaler())
    
    def _get_encoder(self) -> Union[OneHotEncoder, OrdinalEncoder]:
        """Get appropriate encoder"""
        if self.config.encoding == 'onehot':
            return OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first' if len(self.categorical_features) > 1 else None
            )
        elif self.config.encoding == 'ordinal':
            return OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        else:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def _get_outlier_handler(self) -> FunctionTransformer:
        """Get outlier handler"""
        def clip_outliers(X, threshold=3.0):
            if self.config.outlier_handling == 'clip':
                lower = np.percentile(X, 1)
                upper = np.percentile(X, 99)
                return np.clip(X, lower, upper)
            elif self.config.outlier_handling == 'remove':
                # This would need different handling
                return X
            else:
                return X
        
        return FunctionTransformer(
            clip_outliers, 
            kw_args={'threshold': self.config.outlier_threshold}
        )
    
    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        features = []
        
        for col in X.columns:
            dt_series = pd.to_datetime(X[col])
            
            dt_features = pd.DataFrame({
                f'{col}_year': dt_series.dt.year,
                f'{col}_month': dt_series.dt.month,
                f'{col}_day': dt_series.dt.day,
                f'{col}_hour': dt_series.dt.hour,
                f'{col}_dayofweek': dt_series.dt.dayofweek,
                f'{col}_quarter': dt_series.dt.quarter,
                f'{col}_is_weekend': dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            })
            
            features.append(dt_features)
        
        return pd.concat(features, axis=1) if features else pd.DataFrame()
    
    def _get_feature_selector(self) -> Union[SelectKBest, SelectFromModel, RFE]:
        """Get feature selector"""
        if self.config.selection_method == 'mutual_info':
            return SelectKBest(
                score_func=mutual_info_classif,
                k=self.config.n_features if isinstance(self.config.n_features, int) else 'all'
            )
        elif self.config.selection_method == 'variance':
            return VarianceThreshold()
        elif self.config.selection_method == 'model_based':
            return SelectFromModel(
                RandomForestClassifier(n_estimators=100),
                max_features=self.config.n_features
            )
        else:
            return SelectKBest(score_func=mutual_info_classif, k='all')
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance"""
        
        if not self.feature_selector:
            return pd.DataFrame()
        
        # Fit feature selector
        X_transformed = self.preprocessor.named_steps['preprocessor'].transform(X)
        
        if hasattr(self.feature_selector, 'scores_'):
            importance = self.feature_selector.scores_
        elif hasattr(self.feature_selector, 'feature_importances_'):
            importance = self.feature_selector.feature_importances_
        else:
            return pd.DataFrame()
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        if not self.preprocessor:
            return []
        
        # Get feature names from preprocessor
        preprocessor = self.preprocessor.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        
        return list(feature_names)

# ==================== MODEL REGISTRY ====================
class ModelRegistry:
    """Enterprise model registry with versioning and lifecycle management"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = MlflowClient()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLFlow tracking"""
        mlflow.set_tracking_uri(self.config.model_config.tracking_uri)
        mlflow.set_experiment(self.config.model_config.experiment_name)
        
    def register_model(self, model: Any, run_id: str, 
                      metrics: Dict, artifacts: Dict) -> str:
        """Register model in registry"""
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=self.config.model_config.run_name,
            signature=infer_signature(artifacts.get('X_test'), model.predict(artifacts.get('X_test')))
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log parameters
        mlflow.log_params(self.config.model_config.to_dict())
        
        # Log artifacts
        for name, artifact in artifacts.items():
            if isinstance(artifact, pd.DataFrame):
                artifact.to_parquet(f"{name}.parquet")
                mlflow.log_artifact(f"{name}.parquet")
            elif isinstance(artifact, (plt.Figure, go.Figure)):
                if isinstance(artifact, plt.Figure):
                    artifact.savefig(f"{name}.png")
                else:
                    artifact.write_image(f"{name}.png")
                mlflow.log_artifact(f"{name}.png")
        
        # Get registered model
        registered_model = self.client.get_registered_model(
            self.config.model_config.run_name
        )
        
        return registered_model.latest_versions[0].version
    
    def promote_model(self, model_name: str, version: str, 
                     stage: str = "Production") -> None:
        """Promote model to specific stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Get production model from registry"""
        try:
            model_versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            
            if model_versions:
                model_uri = f"models:/{model_name}/Production"
                return mlflow.sklearn.load_model(model_uri)
        
        except Exception as e:
            print(f"Error loading production model: {e}")
        
        return None
    
    def compare_model_versions(self, model_name: str, 
                             versions: List[str]) -> pd.DataFrame:
        """Compare different model versions"""
        
        comparison = []
        
        for version in versions:
            try:
                model_uri = f"models:/{model_name}/{version}"
                run = mlflow.get_run(version)
                
                comparison.append({
                    'version': version,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'status': run.info.status,
                    'artifact_uri': run.info.artifact_uri
                })
            except Exception as e:
                print(f"Error loading version {version}: {e}")
        
        return pd.DataFrame(comparison)

# ==================== ADVANCED MODEL PIPELINE ====================
class AdvancedModelPipeline:
    """Production model pipeline with hyperparameter tuning and ensembles"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config.model_config
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def create_model(self) -> Any:
        """Create model based on configuration"""
        
        if self.config.model_type == ModelType.RANDOM_FOREST:
            base_model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight=self.config.class_weight,
                oob_score=self.config.oob_score,
                verbose=self.config.verbose
            )
            
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            base_model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                verbose=self.config.verbose
            )
            
        elif self.config.model_type == ModelType.LOGISTIC_REGRESSION:
            base_model = LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        
        else:
            base_model = RandomForestClassifier()
        
        # Add calibration if specified
        if self.config.calibration_method != 'none':
            base_model = CalibratedClassifierCV(
                base_model,
                method=self.config.calibration_method,
                cv=self.config.cv_folds
            )
        
        # Create ensemble if specified
        if self.config.use_ensemble:
            models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42))
            ]
            
            if self.config.ensemble_method == 'voting':
                base_model = VotingClassifier(
                    estimators=models,
                    voting='soft',
                    n_jobs=self.config.n_jobs
                )
        
        self.model = base_model
        return base_model
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Perform hyperparameter tuning"""
        
        if not self.config.hyperparameter_tuning:
            return self.create_model()
        
        # Define parameter grid based on model type
        if self.config.model_type == ModelType.RANDOM_FOREST:
            param_distributions = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            param_distributions = {}
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=self.config.n_iter,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
        
        # Fit randomized search
        random_search.fit(X, y)
        
        # Store results
        self.best_params = random_search.best_params_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        self.model = random_search.best_estimator_
        
        return self.model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from model"""
        
        if not self.model:
            return pd.DataFrame()
        
        # Handle different model types
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            # Try to get from underlying estimator for ensembles
            try:
                if hasattr(self.model, 'estimators_'):
                    importances = np.mean([
                        est.feature_importances_ 
                        for est in self.model.estimators_
                    ], axis=0)
                else:
                    importances = None
            except:
                importances = None
        
        if importances is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def get_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate SHAP values for model interpretation"""
        try:
            import shap
            
            # Create explainer based on model type
            if isinstance(self.model, RandomForestClassifier):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
            elif isinstance(self.model, (LogisticRegression, SVC)):
                explainer = shap.LinearExplainer(self.model, X)
                shap_values = explainer.shap_values(X)
            else:
                # Use Kernel SHAP as fallback
                explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(X, 100)
                )
                shap_values = explainer.shap_values(X)
            
            return shap_values
        
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None

# ==================== EVALUATION SUITE ====================
class ModelEvaluator:
    """Comprehensive model evaluation with statistical tests"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_prob is not None:
            try:
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_prob, average='weighted'),
                    'log_loss': -np.mean(y_true * np.log(y_prob + 1e-15) + 
                                       (1 - y_true) * np.log(1 - y_prob + 1e-15))
                })
            except:
                pass
        
        # Additional statistical metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        return metrics
    
    @staticmethod
    def statistical_significance_test(y_true: np.ndarray,
                                    y_pred1: np.ndarray,
                                    y_pred2: np.ndarray,
                                    metric: str = 'accuracy') -> Dict:
        """Test if two models are significantly different"""
        
        from statsmodels.stats.proportion import proportions_ztest
        
        if metric == 'accuracy':
            # McNemar's test for paired nominal data
            from statsmodels.stats.contingency_tables import mcnemar
            
            # Create contingency table
            correct1 = (y_true == y_pred1).astype(int)
            correct2 = (y_true == y_pred2).astype(int)
            
            table = np.zeros((2, 2))
            table[0, 0] = np.sum((correct1 == 1) & (correct2 == 1))  # Both correct
            table[0, 1] = np.sum((correct1 == 1) & (correct2 == 0))  # Only model1 correct
            table[1, 0] = np.sum((correct1 == 0) & (correct2 == 1))  # Only model2 correct
            table[1, 1] = np.sum((correct1 == 0) & (correct2 == 0))  # Both wrong
            
            result = mcnemar(table, exact=False)
            
            return {
                'test': 'mcnemar',
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'significant': result.pvalue < 0.05
            }
        
        return {}
    
    @staticmethod
    def learning_curve_analysis(model: Any, X: pd.DataFrame, y: pd.Series,
                               cv: int = 5) -> Dict:
        """Analyze learning curves"""
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
            'test_scores_std': np.std(test_scores, axis=1).tolist()
        }
    
    @staticmethod
    def create_evaluation_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               feature_importance: Optional[pd.DataFrame] = None) -> Dict:
        """Create comprehensive evaluation report"""
        
        metrics = ModelEvaluator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': metrics['confusion_matrix']
        }
        
        if feature_importance is not None:
            report['feature_importance'] = feature_importance.to_dict('records')
        
        # Calculate ROC curve if probabilities available
        if y_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            report['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        # Calculate precision-recall curve
        if y_prob is not None:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            report['precision_recall_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return report

# ==================== MAIN PIPELINE ====================
class MLPipeline:
    """End-to-end ML pipeline with monitoring, testing, and deployment"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load configuration
        if config_path:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()
        
        # Initialize components
        self.logger = StructuredLogger("MLPipeline", self.config).get_logger()
        self.metrics = MetricsCollector(self.config.metrics_prefix)
        self.validator = DataValidator(self.config)
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        self.model_pipeline = AdvancedModelPipeline(self.config)
        self.registry = ModelRegistry(self.config)
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.metrics_report = None
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            start_http_server(self.config.monitoring_port)
            self.logger.info(f"Monitoring started on port {self.config.monitoring_port}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate data"""
        self.logger.info("Loading data...")
        
        try:
            # Load data
            if self.config.data_config.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.config.data_config.data_path)
            elif self.config.data_config.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.config.data_config.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.config.data_config.data_path}")
            
            self.logger.info(f"Data loaded: {self.data.shape}")
            
            # Validate data
            validation_result = self.validator.validate_data(self.data)
            
            if not validation_result.success:
                self.logger.warning("Data validation failed")
                # Handle validation failures
                self.handle_validation_failures(validation_result)
            
            return self.data
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def handle_validation_failures(self, validation_result):
        """Handle data validation failures"""
        # Log failures
        for result in validation_result.results:
            if not result.success:
                self.logger.warning(
                    f"Validation failed: {result.expectation_config.expectation_type}"
                )
        
        # Optionally fix or remove problematic data
        # This would be domain-specific
    
    def split_data(self) -> None:
        """Split data into train, validation, and test sets"""
        self.logger.info("Splitting data...")
        
        # Ensure target column exists
        target_col = self.config.data_config.target_column
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Split features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Handle time-based split
        if self.config.data_config.time_based_split:
            time_col = self.config.data_config.time_column
            if time_col not in X.columns:
                raise ValueError(f"Time column '{time_col}' not found")
            
            # Sort by time
            X = X.sort_values(time_col)
            y = y.loc[X.index]
            
            # Time-based split
            split_idx = int(len(X) * (1 - self.config.data_config.test_size))
            self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        else:
            # Random split with stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.config.data_config.test_size,
                random_state=self.config.model_config.random_state,
                stratify=y
            )
        
        # Create validation set from training
        if self.config.data_config.validation_size > 0:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=self.config.data_config.validation_size,
                random_state=self.config.model_config.random_state,
                stratify=self.y_train
            )
        
        self.logger.info(f"Training set: {self.X_train.shape}")
        self.logger.info(f"Test set: {self.X_test.shape}")
    
    def engineer_features(self) -> None:
        """Perform feature engineering"""
        self.logger.info("Engineering features...")
        
        # Create and fit feature pipeline
        pipeline = self.feature_pipeline.create_pipeline(self.X_train)
        pipeline.fit(self.X_train, self.y_train)
        
        # Transform features
        self.X_train_transformed = pipeline.transform(self.X_train)
        self.X_test_transformed = pipeline.transform(self.X_test)
        
        # Get feature importance
        feature_importance = self.feature_pipeline.get_feature_importance(
            self.X_train, self.y_train
        )
        
        if not feature_importance.empty:
            self.logger.info("Top 10 features by importance:")
            for idx, row in feature_importance.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.logger.info(f"Transformed features: {self.X_train_transformed.shape}")
    
    def train_model(self) -> Any:
        """Train model with hyperparameter tuning"""
        self.logger.info("Training model...")
        
        # Start metrics collection
        self.metrics.record_training_start()
        start_time = datetime.now()
        
        try:
            # Perform hyperparameter tuning
            self.model = self.model_pipeline.hyperparameter_tuning(
                self.X_train_transformed, self.y_train
            )
            
            # Calculate training duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            train_score = self.model.score(self.X_train_transformed, self.y_train)
            self.metrics.record_training_end(duration, train_score)
            
            self.logger.info(f"Model trained in {duration:.2f} seconds")
            self.logger.info(f"Training score: {train_score:.4f}")
            
            if self.model_pipeline.best_params:
                self.logger.info(f"Best parameters: {self.model_pipeline.best_params}")
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            self.metrics.record_error("training_error")
            raise
    
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation"""
        self.logger.info("Evaluating model...")
        
        # Make predictions
        start_time = datetime.now()
        self.predictions = self.model.predict(self.X_test_transformed)
        self.probabilities = self.model.predict_proba(self.X_test_transformed)[:, 1]
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Record prediction metrics
        self.metrics.record_prediction(prediction_time)
        
        # Calculate all metrics
        self.metrics_report = self.evaluator.calculate_all_metrics(
            self.y_test, self.predictions, self.probabilities
        )
        
        # Check against quality gate
        quality_gate_passed = True
        for metric, threshold in self.config.quality_gate.items():
            if metric in self.metrics_report:
                value = self.metrics_report[metric]
                if value < threshold:
                    quality_gate_passed = False
                    self.logger.warning(
                        f"Quality gate failed for {metric}: {value:.4f} < {threshold}"
                    )
        
        # Log metrics
        for metric, value in self.metrics_report.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric}: {value:.4f}")
        
        # Create evaluation report
        evaluation_report = self.evaluator.create_evaluation_report(
            self.y_test, self.predictions, self.probabilities,
            self.model_pipeline.get_feature_importance()
        )
        
        # Save report
        report_path = self.config.artifacts_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        
        return evaluation_report
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Create directory for visualizations
        viz_dir = self.config.artifacts_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Feature Importance Plot
        feature_importance = self.model_pipeline.get_feature_importance()
        if not feature_importance.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(viz_dir / 'feature_importance.png', dpi=300)
            plt.close()
        
        # 2. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # 3. ROC Curve
        if hasattr(self, 'probabilities'):
            fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
            roc_auc = roc_auc_score(self.y_test, self.probabilities)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / 'roc_curve.png', dpi=300)
            plt.close()
        
        # 4. Precision-Recall Curve
        if hasattr(self, 'probabilities'):
            precision, recall, _ = precision_recall_curve(self.y_test, self.probabilities)
            avg_precision = np.mean(precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / 'precision_recall_curve.png', dpi=300)
            plt.close()
        
        self.logger.info(f"Visualizations saved to {viz_dir}")
    
    def register_model(self) -> str:
        """Register model in model registry"""
        self.logger.info("Registering model...")
        
        # Prepare artifacts
        artifacts = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'predictions': pd.Series(self.predictions),
            'probabilities': pd.Series(self.probabilities)
        }
        
        # Register model
        model_version = self.registry.register_model(
            model=self.model,
            run_id=mlflow.active_run().info.run_id,
            metrics=self.metrics_report,
            artifacts=artifacts
        )
        
        self.logger.info(f"Model registered as version {model_version}")
        
        return model_version
    
    def save_pipeline(self) -> None:
        """Save complete pipeline for deployment"""
        self.logger.info("Saving pipeline...")
        
        # Create pipeline artifact
        pipeline_artifact = {
            'model': self.model,
            'feature_pipeline': self.feature_pipeline.preprocessor,
            'config': self.config.to_dict(),
            'metrics': self.metrics_report,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save using joblib
        pipeline_path = self.config.model_dir / "pipeline.joblib"
        joblib.dump(pipeline_artifact, pipeline_path)
        
        # Also save as pickle for compatibility
        with open(self.config.model_dir / "pipeline.pkl", 'wb') as f:
            pickle.dump(pipeline_artifact, f)
        
        # Create deployment configuration
        deployment_config = {
            'model_type': self.config.model_config.model_type.value,
            'version': '1.0.0',
            'deployment_time': datetime.now().isoformat(),
            'metrics': self.metrics_report,
            'requirements': self.get_requirements()
        }
        
        with open(self.config.model_dir / "deployment_config.json", 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        self.logger.info(f"Pipeline saved to {pipeline_path}")
    
    def get_requirements(self) -> List[str]:
        """Get package requirements"""
        # This would typically read from requirements.txt
        return [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "mlflow>=1.23.0"
        ]
    
    def run_tests(self) -> bool:
        """Run test suite"""
        if not self.config.run_tests:
            return True
        
        self.logger.info("Running tests...")
        
        # Import test modules
        import subprocess
        import sys
        
        try:
            # Run pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(self.config.tests_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("All tests passed")
                return True
            else:
                self.logger.error(f"Tests failed:\n{result.stdout}\n{result.stderr}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False
    
    def run(self) -> Dict:
        """Execute complete pipeline"""
        self.logger.info("Starting ML Pipeline")
        
        try:
            # Start MLFlow run
            with mlflow.start_run(run_name=self.config.model_config.run_name):
                
                # Run tests if enabled
                if not self.run_tests():
                    raise RuntimeError("Tests failed")
                
                # Execute pipeline steps
                self.load_data()
                self.split_data()
                self.engineer_features()
                self.train_model()
                self.evaluate_model()
                self.create_visualizations()
                
                # Check quality gate
                if not self.check_quality_gate():
                    raise RuntimeError("Quality gate failed")
                
                # Register and save model
                model_version = self.register_model()
                self.save_pipeline()
                
                # Promote to production if metrics are good
                if self.metrics_report.get('accuracy', 0) > 0.9:
                    self.registry.promote_model(
                        self.config.model_config.run_name,
                        model_version,
                        "Production"
                    )
                
                # Generate final report
                final_report = self.generate_final_report(model_version)
                
                self.logger.info("Pipeline completed successfully")
                
                return final_report
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def check_quality_gate(self) -> bool:
        """Check if quality gate passes"""
        if not self.metrics_report:
            return False
        
        for metric, threshold in self.config.quality_gate.items():
            if metric in self.metrics_report:
                value = self.metrics_report[metric]
                if value < threshold:
                    self.logger.warning(
                        f"Quality gate failed for {metric}: {value:.4f} < {threshold}"
                    )
                    return False
        
        return True
    
    def generate_final_report(self, model_version: str) -> Dict:
        """Generate final pipeline report"""
        
        # Get system info
        import platform
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'memory_gb': psutil.virtual_memory().total / 1e9,
            'cpu_count': psutil.cpu_count()
        }
        
        # Compile report
        report = {
            'pipeline_version': '1.0.0',
            'execution_time': datetime.now().isoformat(),
            'model_version': model_version,
            'system_info': system_info,
            'config': self.config.to_dict(),
            'metrics': self.metrics_report,
            'data_summary': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': self.X_train.shape[1],
                'class_distribution': dict(self.y_train.value_counts())
            },
            'artifacts': {
                'model_path': str(self.config.model_dir / "pipeline.joblib"),
                'config_path': str(self.config.model_dir / "deployment_config.json"),
                'visualizations_dir': str(self.config.artifacts_dir / "visualizations"),
                'logs_dir': str(self.config.logs_dir)
            }
        }
        
        # Save report
        report_path = self.config.artifacts_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Final report saved to {report_path}")
        
        return report

# ==================== DEPLOYMENT & SERVING ====================
class ModelServer:
    """Model serving with REST API and monitoring"""
    
    def __init__(self, pipeline_path: Path):
        self.pipeline_path = pipeline_path
        self.pipeline = None
        self.model = None
        self.feature_pipeline = None
        
    def load_pipeline(self):
        """Load trained pipeline"""
        self.pipeline = joblib.load(self.pipeline_path)
        self.model = self.pipeline['model']
        self.feature_pipeline = self.pipeline['feature_pipeline']
        self.config = self.pipeline['config']
        
    def predict(self, data: pd.DataFrame) -> Dict:
        """Make predictions"""
        
        # Transform features
        features = self.feature_pipeline.transform(data)
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'confidence': np.max(probabilities, axis=1).tolist()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_pipeline.get_feature_names_out()
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame()
    
    def create_api(self):
        """Create REST API for model serving"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="ML Model Server")
        
        # Request model
        class PredictionRequest(BaseModel):
            data: List[Dict]
        
        # Health check
        @app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Prediction endpoint
        @app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                df = pd.DataFrame(request.data)
                result = self.predict(df)
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Feature importance endpoint
        @app.get("/features/importance")
        async def feature_importance():
            df = self.get_feature_importance()
            return df.to_dict('records')
        
        # Metrics endpoint
        @app.get("/metrics")
        async def metrics():
            return self.pipeline.get('metrics', {})
        
        return app

# ==================== TEST SUITE ====================
class TestMLPipeline:
    """Comprehensive test suite for ML pipeline"""
    
    @staticmethod
    def test_data_loading():
        """Test data loading functionality"""
        config = PipelineConfig(debug_mode=True)
        pipeline = MLPipeline()
        pipeline.config = config
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        sample_data.to_csv(config.data_config.data_path, index=False)
        
        try:
            data = pipeline.load_data()
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 100
            print("✓ Data loading test passed")
        except:
            print("✗ Data loading test failed")
    
    @staticmethod
    def test_feature_engineering():
        """Test feature engineering"""
        config = PipelineConfig()
        pipeline = MLPipeline()
        pipeline.config = config
        
        # Create sample data
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Test feature pipeline
        feature_pipeline = FeatureEngineeringPipeline(config)
        pipeline_obj = feature_pipeline.create_pipeline(X)
        
        try:
            pipeline_obj.fit(X, y)
            X_transformed = pipeline_obj.transform(X)
            assert X_transformed.shape[0] == 100
            print("✓ Feature engineering test passed")
        except:
            print("✗ Feature engineering test failed")
    
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        print("Running ML Pipeline tests...")
        print("=" * 50)
        
        TestMLPipeline.test_data_loading()
        TestMLPipeline.test_feature_engineering()
        
        print("=" * 50)
        print("Tests completed")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise ML Pipeline")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--mode", choices=["train", "serve", "test"], default="train")
    parser.add_argument("--model-path", type=str, help="Path to model for serving")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Training mode
        pipeline = MLPipeline(Path(args.config) if args.config else None)
        report = pipeline.run()
        
        print("\n" + "="*60)
        print("ENTERPRISE ML PIPELINE - TRAINING COMPLETE")
        print("="*60)
        print(f"Model Accuracy: {report['metrics'].get('accuracy', 0):.4f}")
        print(f"ROC AUC: {report['metrics'].get('roc_auc', 0):.4f}")
        print(f"Model Version: {report['model_version']}")
        print(f"Report saved to: {pipeline.config.artifacts_dir / 'pipeline_report.json'}")
        print("="*60)
    
    elif args.mode == "serve":
        # Serving mode
        if not args.model_path:
            print("Error: --model-path required for serving mode")
            sys.exit(1)
        
        server = ModelServer(Path(args.model_path))
        server.load_pipeline()
        
        app = server.create_api()
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    elif args.mode == "test":
        # Test mode
        TestMLPipeline.run_all_tests()



from __future__ import annotations
import os
import sys
import json
import yaml
import logging
import warnings
import pickle
import joblib
import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Tuple, List, Dict, Optional, Union, Any, Callable,
    Protocol, runtime_checkable, TypeVar, Generic
)
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import threading
import queue

warnings.filterwarnings('ignore')

# Core Data Science
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import scipy.stats as stats
from scipy import sparse

# ML & Deep Learning
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, TimeSeriesSplit,
    RandomizedSearchCV, GridSearchCV, cross_val_score,
    learning_curve, validation_curve, ShuffleSplit
)
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    KBinsDiscretizer, PolynomialFeatures,
    FunctionTransformer, PowerTransformer,
    QuantileTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    VarianceThreshold, mutual_info_classif, f_classif
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    SGDClassifier, Perceptron
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
    HistGradientBoostingClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Metrics & Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
    brier_score_loss, cohen_kappa_score,
    matthews_corrcoef, hamming_loss, jaccard_score,
    fbeta_score, balanced_accuracy_score,
    top_k_accuracy_score
)
import scikitplot as skplt

# MLOps & Experiment Tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature, ModelSignature
from mlflow.pyfunc import PythonModel
import wandb
import comet_ml
from neptune.new import run as neptune_run
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.visualization import (
    plot_optimization_history, plot_param_importances,
    plot_slice, plot_contour
)

# Feature Store & Data Validation
import great_expectations as gx
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, DataQualityPreset,
    ClassificationPreset, RegressionPreset
)
import pandas_profiling
from dataclasses_json import dataclass_json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from IPython.display import display, HTML

# Distributed Computing & Cloud
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import ray
from ray import tune
import boto3
from google.cloud import storage
import azure.storage.blob as azure_blob

# System & Monitoring
import psutil
import GPUtil
import docker
import requests
import httpx
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, push_to_gateway
)
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Testing & Validation
import pytest
import hypothesis
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import great_expectations as ge

# Type Checking & Validation
from pydantic import (
    BaseModel, Field, validator, root_validator,
    PositiveInt, PositiveFloat, confloat, conint
)
from typing_extensions import Literal, Annotated, get_origin
import typeguard
from typeguard import typechecked

# Async & Concurrency
import aiohttp
import aiofiles
from asyncio import Queue, Semaphore
import uvloop

# ==================== CONFIGURATION SYSTEM ====================
@dataclass_json
@dataclass
class DataConfig:
    """Configuration for data generation and processing"""
    
    # Data Generation
    n_samples: PositiveInt = Field(default=10000, description="Number of samples")
    n_features: PositiveInt = Field(default=20, description="Number of features")
    n_informative: PositiveInt = Field(default=10, description="Informative features")
    n_redundant: PositiveInt = Field(default=5, description="Redundant features")
    n_repeated: PositiveInt = Field(default=0, description="Repeated features")
    n_classes: PositiveInt = Field(default=2, description="Number of classes")
    n_clusters_per_class: PositiveInt = Field(default=2, description="Clusters per class")
    weights: Optional[List[float]] = Field(default=None, description="Class weights")
    flip_y: float = Field(default=0.01, ge=0.0, le=1.0, description="Label noise")
    class_sep: float = Field(default=1.0, description="Class separation")
    hypercube: bool = Field(default=True, description="Use hypercube")
    shift: float = Field(default=0.0, description="Feature shift")
    scale: float = Field(default=1.0, description="Feature scale")
    shuffle: bool = Field(default=True, description="Shuffle samples")
    random_state: Optional[int] = Field(default=42, description="Random seed")
    
    # Data Splitting
    test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Test set size")
    validation_size: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation size")
    time_based_split: bool = Field(default=False, description="Use time-based split")
    group_based_split: bool = Field(default=False, description="Use group-based split")
    group_column: Optional[str] = Field(default=None, description="Group column name")
    
    # Data Validation
    enable_validation: bool = Field(default=True, description="Enable data validation")
    validation_rules: List[Dict] = Field(default_factory=list, description="Validation rules")
    data_quality_threshold: float = Field(default=0.95, description="Data quality threshold")
    
    @validator('weights')
    def validate_weights(cls, v, values):
        if v is not None:
            if len(v) != values.get('n_classes', 2):
                raise ValueError(f"Number of weights must match n_classes ({values.get('n_classes', 2)})")
            if not all(w >= 0 for w in v):
                raise ValueError("All weights must be non-negative")
        return v

@dataclass_json
@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    
    # Preprocessing
    scaling_method: Literal['standard', 'minmax', 'robust', 'none'] = 'standard'
    encoding_method: Literal['onehot', 'label', 'ordinal', 'target'] = 'onehot'
    imputation_method: Literal['mean', 'median', 'most_frequent', 'constant', 'knn'] = 'median'
    outlier_detection: bool = True
    outlier_method: Literal['iqr', 'zscore', 'isolation_forest', 'lof'] = 'iqr'
    outlier_threshold: float = 3.0
    
    # Feature Engineering
    polynomial_features: bool = False
    polynomial_degree: PositiveInt = 2
    interaction_features: bool = True
    trigonometric_features: bool = False
    datetime_features: bool = False
    text_features: bool = False
    feature_generation: bool = True
    
    # Feature Selection
    feature_selection: bool = True
    selection_method: Literal['variance', 'mutual_info', 'model', 'rfe'] = 'mutual_info'
    n_features_to_select: Union[int, float, str] = 'auto'
    feature_importance_threshold: float = 0.01
    
    # Dimensionality Reduction
    dimensionality_reduction: bool = False
    reduction_method: Literal['pca', 'tsne', 'umap', 'lda'] = 'pca'
    n_components: Union[int, float] = 0.95
    
    # Advanced
    autoencoder_features: bool = False
    clustering_features: bool = False
    n_clusters: PositiveInt = 5

@dataclass_json
@dataclass
class ModelConfig:
    """Configuration for model training"""
    
    # Model Selection
    model_type: Literal['logistic', 'random_forest', 'gradient_boosting', 'svm', 'ensemble', 'auto'] = 'ensemble'
    use_ensemble: bool = True
    ensemble_method: Literal['voting', 'stacking', 'blending'] = 'stacking'
    
    # Hyperparameter Tuning
    hyperparameter_tuning: bool = True
    tuning_method: Literal['grid', 'random', 'bayesian', 'hyperopt', 'optuna'] = 'optuna'
    n_trials: PositiveInt = 100
    cv_folds: PositiveInt = 5
    scoring_metric: str = 'roc_auc'
    
    # Random Forest
    rf_n_estimators: PositiveInt = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: Union[int, float] = 2
    rf_min_samples_leaf: Union[int, float] = 1
    rf_max_features: Union[str, int, float] = 'sqrt'
    
    # Logistic Regression
    lr_penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = 'l2'
    lr_c: float = 1.0
    lr_solver: str = 'lbfgs'
    lr_max_iter: PositiveInt = 1000
    
    # Gradient Boosting
    gb_n_estimators: PositiveInt = 200
    gb_learning_rate: float = 0.1
    gb_max_depth: PositiveInt = 3
    gb_subsample: float = 0.8
    
    # SVM
    svm_c: float = 1.0
    svm_kernel: Literal['linear', 'poly', 'rbf', 'sigmoid'] = 'rbf'
    svm_gamma: Union[str, float] = 'scale'
    
    # Training
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    early_stopping: bool = True
    early_stopping_rounds: PositiveInt = 10
    class_weight: Optional[str] = 'balanced'
    
    # Calibration
    calibration: bool = True
    calibration_method: Literal['isotonic', 'sigmoid', 'none'] = 'isotonic'
    calibration_cv: PositiveInt = 5

@dataclass_json
@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    
    # Experiment Tracking
    tracking_enabled: bool = True
    tracking_backend: Literal['mlflow', 'wandb', 'comet', 'neptune', 'all'] = 'mlflow'
    experiment_name: str = 'classification_experiment'
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # MLflow
    mlflow_tracking_uri: str = 'mlruns'
    mlflow_registry_uri: str = 'models'
    mlflow_experiment_id: Optional[str] = None
    
    # Weights & Biases
    wandb_project: str = 'ml-pipeline'
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # Neptune
    neptune_project: Optional[str] = None
    neptune_api_token: Optional[str] = None
    
    # Comet
    comet_project_name: str = 'ml-pipeline'
    comet_workspace: Optional[str] = None
    comet_api_key: Optional[str] = None
    
    # Logging
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file: Optional[str] = None
    
    # Artifacts
    save_artifacts: bool = True
    artifacts_dir: str = 'artifacts'
    save_model: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    save_shap_values: bool = True

@dataclass_json
@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerting"""
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_port: int = 9090
    metrics_prefix: str = 'ml_pipeline'
    
    # Alerting
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.85,
        'roc_auc': 0.90,
        'data_drift': 0.10,
        'concept_drift': 0.10
    })
    
    # Drift Detection
    drift_detection_enabled: bool = True
    drift_detection_window: PositiveInt = 1000
    drift_threshold: float = 0.05
    drift_method: Literal['ks', 'chi2', 'psi', 'mmd'] = 'psi'
    
    # Performance Monitoring
    monitor_latency: bool = True
    latency_threshold_ms: PositiveInt = 100
    monitor_throughput: bool = True
    throughput_threshold_rps: PositiveInt = 100
    monitor_memory: bool = True
    memory_threshold_mb: PositiveInt = 1024
    
    # Notification Channels
    slack_webhook: Optional[str] = None
    email_notifications: bool = False
    email_recipients: List[str] = field(default_factory=list)
    pagerduty_integration_key: Optional[str] = None

@dataclass_json
@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    
    # Configuration sections
    data_config: DataConfig = field(default_factory=DataConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # System
    debug_mode: bool = False
    profile_performance: bool = True
    enable_caching: bool = True
    cache_dir: str = '.cache'
    use_gpu: bool = False
    distributed_training: bool = False
    num_workers: PositiveInt = 4
    
    # Quality Gates
    quality_gates: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.85,
        'roc_auc': 0.90,
        'f1_score': 0.80,
        'precision': 0.80,
        'recall': 0.80
    })
    
    # Versioning
    version: str = '1.0.0'
    git_commit: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.experiment_config.run_name:
            self.experiment_config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        for dir_path in [self.experiment_config.artifacts_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PipelineConfig:
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def get_hash(self) -> str:
        """Get hash of configuration for caching"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

# ==================== ADVANCED LOGGING ====================
class StructuredLogger:
    """Production logging with structured JSON output"""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
        self.metrics_collector = MetricsCollector()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging with multiple handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.experiment_config.log_level))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.experiment_config.log_file:
            file_handler = logging.FileHandler(
                self.config.experiment_config.log_file
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        json_handler = logging.FileHandler(
            f"{self.config.experiment_config.artifacts_dir}/structured_logs.json"
        )
        json_handler.setLevel(logging.INFO)
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    'timestamp': self.formatTime(record),
                    'logger': record.name,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                if hasattr(record, 'context'):
                    log_record['context'] = record.context
                if record.exc_info:
                    log_record['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_record)
        
        json_handler.setFormatter(JsonFormatter())
        logger.addHandler(json_handler)
        
        # Disable propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
    
    def log_with_context(self, level: str, message: str, **context):
        """Log with additional context"""
        log_method = getattr(self.logger, level.lower())
        extra = {'context': context}
        log_method(message, extra=extra)
    
    def log_performance(self, operation: str, duration: float,
                       memory_usage: float, **kwargs):
        """Log performance metrics"""
        self.log_with_context(
            'INFO',
            f"Performance: {operation} - Duration: {duration:.4f}s, Memory: {memory_usage:.2f}MB",
            operation=operation,
            duration=duration,
            memory_usage=memory_usage,
            **kwargs
        )
        self.metrics_collector.record_performance(operation, duration, memory_usage)
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with context"""
        self.log_with_context(
            'ERROR',
            f"Exception in {context}: {str(exception)}",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            context=context
        )
        self.metrics_collector.record_error(type(exception).__name__)
    
    def get_logger(self) -> logging.Logger:
        return self.logger

class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self):
        # Prometheus metrics
        self.training_counter = Counter('ml_training_total', 'Total training runs')
        self.prediction_counter = Counter('ml_predictions_total', 'Total predictions')
        self.error_counter = Counter('ml_errors_total', 'Total errors', ['error_type'])
        self.training_duration = Histogram('ml_training_duration_seconds', 'Training duration')
        self.prediction_latency = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
        self.memory_usage = Gauge('ml_memory_usage_bytes', 'Memory usage')
        self.model_accuracy = Gauge('ml_model_accuracy', 'Model accuracy')
        self.model_auc = Gauge('ml_model_auc', 'Model AUC')
        
        # Custom metrics storage
        self.custom_metrics = {}
        self.performance_history = []
    
    def record_performance(self, operation: str, duration: float, memory: float):
        """Record performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration': duration,
            'memory': memory
        })
    
    def record_error(self, error_type: str):
        """Record error"""
        self.error_counter.labels(error_type=error_type).inc()
    
    def record_training_start(self):
        """Record training start"""
        self.training_counter.inc()
    
    def record_training_end(self, duration: float, accuracy: float, auc: float):
        """Record training completion"""
        self.training_duration.observe(duration)
        self.model_accuracy.set(accuracy)
        self.model_auc.set(auc)
    
    def record_prediction(self, latency: float):
        """Record prediction"""
        self.prediction_counter.inc()
        self.prediction_latency.observe(latency)
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'training_runs': self.training_counter._value.get(),
            'predictions': self.prediction_counter._value.get(),
            'performance_history': self.performance_history[-100:],  # Last 100 entries
            'custom_metrics': self.custom_metrics
        }

# ==================== DATA MANAGEMENT ====================
class DataGenerator(ABC):
    """Abstract base class for data generation"""
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generate dataset"""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate generated data"""
        pass

class SyntheticDataGenerator(DataGenerator):
    """Generate synthetic classification data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = StructuredLogger(self.__class__.__name__, config)
    
    @typechecked
    def generate(self) -> pd.DataFrame:
        """Generate synthetic classification dataset"""
        self.logger.logger.info(f"Generating synthetic dataset with {self.config.n_samples} samples")
        
        start_time = datetime.now()
        
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_repeated=self.config.n_repeated,
            n_classes=self.config.n_classes,
            n_clusters_per_class=self.config.n_clusters_per_class,
            weights=self.config.weights,
            flip_y=self.config.flip_y,
            class_sep=self.config.class_sep,
            hypercube=self.config.hypercube,
            shift=self.config.shift,
            scale=self.config.scale,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state
        )
        
        # Create feature names
        feature_types = []
        for i in range(self.config.n_features):
            if i < self.config.n_informative:
                feature_types.append('informative')
            elif i < self.config.n_informative + self.config.n_redundant:
                feature_types.append('redundant')
            elif i < self.config.n_informative + self.config.n_redundant + self.config.n_repeated:
                feature_types.append('repeated')
            else:
                feature_types.append('noise')
        
        feature_columns = [f"{ft}_feature_{i}" for i, ft in enumerate(feature_types)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_columns)
        df['target'] = y
        
        # Add metadata
        df.attrs['generation_config'] = self.config.to_dict()
        df.attrs['generation_timestamp'] = datetime.now().isoformat()
        df.attrs['feature_types'] = feature_types
        
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.log_performance('data_generation', duration, psutil.Process().memory_info().rss / 1024 / 1024)
        
        return df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate generated data"""
        try:
            # Check shape
            assert data.shape[0] == self.config.n_samples
            assert data.shape[1] == self.config.n_features + 1  # +1 for target
            
            # Check target distribution
            target_counts = data['target'].value_counts()
            if self.config.weights:
                expected_ratios = np.array(self.config.weights) / sum(self.config.weights)
                actual_ratios = target_counts.values / len(data)
                assert np.allclose(actual_ratios, expected_ratios, rtol=0.1)
            
            # Check for NaN values
            assert not data.isnull().any().any()
            
            # Check feature correlations
            informative_features = [col for col in data.columns if 'informative' in col]
            if informative_features:
                corr_with_target = data[informative_features].corrwith(data['target']).abs()
                assert corr_with_target.mean() > 0.1  # Some correlation expected
            
            self.logger.logger.info("Data validation passed")
            return True
            
        except AssertionError as e:
            self.logger.log_exception(e, "data_validation")
            return False

class DataValidator:
    """Comprehensive data validation using Great Expectations"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.context = gx.get_context()
        self.expectation_suite_name = f"data_validation_{config.n_samples}"
        
    def create_expectation_suite(self) -> gx.ExpectationSuite:
        """Create expectation suite for data validation"""
        
        suite = self.context.create_expectation_suite(
            expectation_suite_name=self.expectation_suite_name
        )
        
        # Basic expectations
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {
                    "min_value": int(self.config.n_samples * 0.95),
                    "max_value": int(self.config.n_samples * 1.05)
                }
            },
            {
                "expectation_type": "expect_table_columns_to_match_set",
                "kwargs": {
                    "column_set": None  # Will be set dynamically
                }
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "target"}
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "target",
                    "value_set": list(range(self.config.n_classes))
                }
            }
        ]
        
        for exp in expectations:
            suite.add_expectation(
                gx.ExpectationConfiguration(
                    expectation_type=exp["expectation_type"],
                    kwargs=exp["kwargs"]
                )
            )
        
        return suite
    
    def validate(self, data: pd.DataFrame) -> gx.ValidationResult:
        """Validate data against expectation suite"""
        
        try:
            suite = self.context.get_expectation_suite(
                expectation_suite_name=self.expectation_suite_name
            )
        except:
            suite = self.create_expectation_suite()
        
        # Update column set expectation
        column_set_expectation = suite.get_expectation_by_expectation_type(
            "expect_table_columns_to_match_set"
        )
        column_set_expectation.kwargs["column_set"] = set(data.columns)
        
        # Create validator
        batch_request = {
            "datasource_name": "pandas_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "synthetic_data",
            "runtime_parameters": {"batch_data": data},
            "batch_identifiers": {"default_identifier_name": "default_identifier"}
        }
        
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=self.expectation_suite_name
        )
        
        return validator.validate()

# ==================== ADVANCED FEATURE ENGINEERING ====================
class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.pipeline = None
        self.feature_names = []
        self.logger = StructuredLogger(self.__class__.__name__, config)
    
    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build feature engineering pipeline"""
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', self._get_imputer()),
            ('outlier_handler', self._get_outlier_handler()),
            ('scaler', self._get_scaler()),
            ('feature_generator', self._get_feature_generator())
        ])
        
        # Categorical transformers
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', self._get_encoder())
        ])
        
        # Combine transformers
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Feature selection
        if self.config.feature_selection:
            selector = self._get_feature_selector()
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', selector)
            ])
        else:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        # Dimensionality reduction
        if self.config.dimensionality_reduction:
            reducer = self._get_dimensionality_reducer()
            pipeline.steps.append(('reducer', reducer))
        
        self.pipeline = pipeline
        return pipeline
    
    def _get_imputer(self) -> SimpleImputer:
        """Get imputer based on configuration"""
        if self.config.imputation_method == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            return SimpleImputer(strategy=self.config.imputation_method)
    
    def _get_outlier_handler(self) -> FunctionTransformer:
        """Get outlier handler"""
        def handle_outliers(X, method='iqr', threshold=3.0):
            if method == 'iqr':
                Q1 = np.percentile(X, 25, axis=0)
                Q3 = np.percentile(X, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return np.clip(X, lower_bound, upper_bound)
            elif method == 'zscore':
                mean = np.mean(X, axis=0)
                std = np.std(X, axis=0)
                z_scores = (X - mean) / std
                mask = np.abs(z_scores) > threshold
                X_clean = X.copy()
                X_clean[mask] = np.nan
                return X_clean
            else:
                return X
        
        return FunctionTransformer(
            handle_outliers,
            kw_args={
                'method': self.config.outlier_method,
                'threshold': self.config.outlier_threshold
            }
        )
    
    def _get_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Get scaler based on configuration"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': FunctionTransformer(lambda x: x)
        }
        return scalers.get(self.config.scaling_method, StandardScaler())
    
    def _get_encoder(self) -> Union[OneHotEncoder, OrdinalEncoder]:
        """Get encoder based on configuration"""
        if self.config.encoding_method == 'onehot':
            return OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'
            )
        elif self.config.encoding_method == 'label':
            return OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        else:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def _get_feature_generator(self) -> FeatureUnion:
        """Generate additional features"""
        transformers = []
        
        if self.config.polynomial_features:
            transformers.append((
                'poly',
                PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)
            ))
        
        if self.config.interaction_features:
            # Custom interaction features
            def create_interactions(X):
                n_features = X.shape[1]
                interactions = []
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interactions.append(X[:, i] * X[:, j])
                return np.column_stack(interactions) if interactions else X
            
            transformers.append((
                'interactions',
                FunctionTransformer(create_interactions)
            ))
        
        if self.config.trigonometric_features:
            def add_trigonometric(X):
                sin_features = np.sin(X)
                cos_features = np.cos(X)
                return np.column_stack([X, sin_features, cos_features])
            
            transformers.append((
                'trig',
                FunctionTransformer(add_trigonometric)
            ))
        
        if transformers:
            return FeatureUnion(transformers)
        else:
            return FunctionTransformer(lambda x: x)
    
    def _get_feature_selector(self) -> Union[SelectKBest, SelectFromModel, RFE]:
        """Get feature selector"""
        if self.config.selection_method == 'mutual_info':
            return SelectKBest(
                score_func=mutual_info_classif,
                k=self._get_n_features()
            )
        elif self.config.selection_method == 'variance':
            return VarianceThreshold(threshold=self.config.feature_importance_threshold)
        elif self.config.selection_method == 'model':
            return SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold=self.config.feature_importance_threshold
            )
        elif self.config.selection_method == 'rfe':
            return RFE(
                estimator=LogisticRegression(max_iter=1000),
                n_features_to_select=self._get_n_features()
            )
        else:
            return SelectKBest(score_func=mutual_info_classif, k='all')
    
    def _get_dimensionality_reducer(self) -> Union[PCA, TruncatedSVD]:
        """Get dimensionality reducer"""
        if self.config.reduction_method == 'pca':
            return PCA(n_components=self.config.n_components, random_state=42)
        elif self.config.reduction_method == 'tsne':
            # Note: t-SNE is not a transformer, needs special handling
            return FunctionTransformer(lambda x: x)  # Placeholder
        elif self.config.reduction_method == 'lda':
            return LinearDiscriminantAnalysis(n_components=self.config.n_components)
        else:
            return PCA(n_components=self.config.n_components)
    
    def _get_n_features(self) -> Union[int, str]:
        """Get number of features to select"""
        if isinstance(self.config.n_features_to_select, str):
            return 'all'
        elif isinstance(self.config.n_features_to_select, float):
            return int(self.config.n_features_to_select)
        else:
            return self.config.n_features_to_select
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance"""
        if not self.pipeline:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        # Fit pipeline if not already fitted
        if not hasattr(self.pipeline, 'fit_transform'):
            self.pipeline.fit(X, y)
        
        # Get feature importance from selector if available
        if 'selector' in self.pipeline.named_steps:
            selector = self.pipeline.named_steps['selector']
            if hasattr(selector, 'scores_'):
                importances = selector.scores_
            elif hasattr(selector, 'feature_importances_'):
                importances = selector.feature_importances_
            else:
                importances = None
        else:
            importances = None
        
        if importances is not None:
            feature_names = self.get_feature_names()
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        if not self.pipeline:
            return []
        
        try:
            if 'selector' in self.pipeline.named_steps:
                # Get selected feature names
                selector = self.pipeline.named_steps['selector']
                preprocessor = self.pipeline.named_steps['preprocessor']
                all_features = preprocessor.get_feature_names_out()
                
                if hasattr(selector, 'get_support'):
                    selected_mask = selector.get_support()
                    return all_features[selected_mask].tolist()
                else:
                    return all_features.tolist()
            else:
                preprocessor = self.pipeline.named_steps['preprocessor']
                return preprocessor.get_feature_names_out().tolist()
        except Exception as e:
            self.logger.log_exception(e, "get_feature_names")
            return []

# ==================== ADVANCED MODEL TRAINING ====================
class ModelFactory:
    """Factory for creating different types of models"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> Any:
        """Create model based on configuration"""
        
        if config.model_type == 'logistic':
            return LogisticRegression(
                penalty=config.lr_penalty,
                C=config.lr_c,
                solver=config.lr_solver,
                max_iter=config.lr_max_iter,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose
            )
        
        elif config.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                min_samples_split=config.rf_min_samples_split,
                min_samples_leaf=config.rf_min_samples_leaf,
                max_features=config.rf_max_features,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose,
                class_weight=config.class_weight
            )
        
        elif config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=config.gb_n_estimators,
                learning_rate=config.gb_learning_rate,
                max_depth=config.gb_max_depth,
                subsample=config.gb_subsample,
                random_state=config.random_state,
                verbose=config.verbose
            )
        
        elif config.model_type == 'svm':
            return SVC(
                C=config.svm_c,
                kernel=config.svm_kernel,
                gamma=config.svm_gamma,
                probability=True,
                random_state=config.random_state,
                verbose=config.verbose,
                class_weight=config.class_weight
            )
        
        elif config.model_type == 'ensemble':
            return ModelFactory.create_ensemble(config)
        
        else:
            # Auto mode: create multiple models for stacking
            return ModelFactory.create_ensemble(config)
    
    @staticmethod
    def create_ensemble(config: ModelConfig) -> Union[VotingClassifier, StackingClassifier]:
        """Create ensemble model"""
        
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                random_state=config.random_state,
                n_jobs=config.n_jobs
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                random_state=config.random_state
            )),
            ('lr', LogisticRegression(
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                max_iter=1000
            )),
            ('svm', SVC(
                probability=True,
                random_state=config.random_state,
                kernel='rbf'
            ))
        ]
        
        if config.ensemble_method == 'voting':
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=config.n_jobs
            )
        
        elif config.ensemble_method == 'stacking':
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=config.n_jobs,
                passthrough=False
            )
        
        else:  # blending
            # Custom blending implementation
            class BlendingClassifier:
                def __init__(self, estimators, meta_estimator):
                    self.estimators = estimators
                    self.meta_estimator = meta_estimator
                    self.estimators_ = []
                    self.meta_estimator_ = None
                
                def fit(self, X, y):
                    # Train base estimators
                    for name, estimator in self.estimators:
                        estimator.fit(X, y)
                        self.estimators_.append((name, estimator))
                    
                    # Get base predictions for meta training
                    meta_features = []
                    for name, estimator in self.estimators_:
                        preds = estimator.predict_proba(X)
                        meta_features.append(preds)
                    
                    meta_X = np.hstack(meta_features)
                    self.meta_estimator_ = clone(self.meta_estimator)
                    self.meta_estimator_.fit(meta_X, y)
                    return self
                
                def predict(self, X):
                    meta_features = []
                    for name, estimator in self.estimators_:
                        preds = estimator.predict_proba(X)
                        meta_features.append(preds)
                    
                    meta_X = np.hstack(meta_features)
                    return self.meta_estimator_.predict(meta_X)
                
                def predict_proba(self, X):
                    meta_features = []
                    for name, estimator in self.estimators_:
                        preds = estimator.predict_proba(X)
                        meta_features.append(preds)
                    
                    meta_X = np.hstack(meta_features)
                    return self.meta_estimator_.predict_proba(meta_X)
            
            return BlendingClassifier(
                estimators=estimators,
                meta_estimator=LogisticRegression()
            )

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = StructuredLogger(self.__class__.__name__, config)
        self.study = None
        self.best_params = None
        self.best_score = None
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using Optuna"""
        
        if not self.config.hyperparameter_tuning:
            return {}
        
        self.logger.logger.info(f"Starting hyperparameter optimization with {self.config.tuning_method}")
        
        if self.config.tuning_method == 'optuna':
            return self._optimize_with_optuna(model, X, y)
        elif self.config.tuning_method == 'random':
            return self._optimize_with_random_search(model, X, y)
        elif self.config.tuning_method == 'grid':
            return self._optimize_with_grid_search(model, X, y)
        else:
            return self._optimize_with_hyperopt(model, X, y)
    
    def _optimize_with_optuna(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize using Optuna with TPE sampler"""
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            model_type = self.config.model_type
            
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                
                model.set_params(**params)
            
            elif model_type == 'logistic':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-4, 1e4),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': trial.suggest_categorical('solver', ['l

"""
Simplified metrics module for MentorEval benchmark with LightEval compatibility.

This module provides essential metrics for evaluating student exam grading:

SAMPLE-LEVEL METRICS (LightEval SampleLevelMetric objects):
- Accuracy metrics (exact match, tolerance-based)
- Error metrics (MAE, RMSE) 
These can be calculated per sample and aggregated across the dataset.

CORPUS-LEVEL METRICS (LightEval CorpusLevelMetric objects):
- Correlation metrics (Pearson, Spearman)
- Distribution comparison metrics (KS test, Wasserstein distance)
These require the entire dataset and cannot be calculated per sample.

Following LightEval 0.10.0 guidelines and best practices.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance

# LightEval imports for compatibility
from lighteval.metrics.metrics import SampleLevelMetric, CorpusLevelMetric, Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import CorpusLevelMetricGrouping
from lighteval.tasks.requests import SamplingMethod


@dataclass
class MetricResult:
    """
    Simple container for metric calculation results.
    """
    metric_name: str
    value: float
    metadata: Optional[Dict] = None


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def parse_grade(prediction: str) -> Optional[float]:
    """
    Parse a grade from model prediction string.
    
    Args:
        prediction: Raw model prediction (string)
        
    Returns:
        Parsed grade as float, or None if parsing fails
    """
    try:
        import re
        numbers = re.findall(r'-?\d+\.?\d*', prediction.strip())
        if numbers:
            return float(numbers[0])
        return None
    except (ValueError, TypeError):
        return None


def normalize_grade(grade: float, min_grade: float, max_grade: float) -> float:
    """
    Normalize a grade to [0, 1] range using min-max scaling.
    
    Args:
        grade: Grade to normalize
        min_grade: Minimum grade in the rubric range
        max_grade: Maximum grade in the rubric range
        
    Returns:
        Normalized grade in [0, 1] range
    """
    if max_grade == min_grade:
        return 0.0  # Avoid division by zero
    return (grade - min_grade) / (max_grade - min_grade)


def denormalize_grade(normalized_grade: float, min_grade: float, max_grade: float) -> float:
    """
    Denormalize a grade from [0, 1] range back to original rubric range.
    
    Args:
        normalized_grade: Normalized grade in [0, 1] range
        min_grade: Minimum grade in the rubric range
        max_grade: Maximum grade in the rubric range
        
    Returns:
        Denormalized grade in original rubric range
    """
    return min_grade + normalized_grade * (max_grade - min_grade)


# =============================================================================
# SAMPLE-LEVEL METRIC FUNCTIONS
# =============================================================================

def exact_grade_match(predictions: List[str], formatted_doc, **kwargs) -> float:
    """
    Check for exact grade match (no normalization needed).
    
    Args:
        predictions: List of model predictions (strings)
        formatted_doc: Doc object with expected grade
        **kwargs: Additional keyword arguments
        
    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    try:
        pred_grade = parse_grade(predictions[0])
        if pred_grade is None:
            return 0.0
            
        expected_grade = float(formatted_doc.choices[0])
        
        # Check for exact match in raw grade space (tolerance of 0.1)
        return 1.0 if abs(pred_grade - expected_grade) < 0.1 else 0.0
    except (ValueError, IndexError, TypeError, AttributeError):
        return 0.0




def grade_mae(predictions: List[str], formatted_doc, **kwargs) -> float:
    """
    Calculate Mean Absolute Error for grade prediction (normalized).
    
    Args:
        predictions: List of model predictions (strings)
        formatted_doc: Doc object with expected grade and rubric range
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Absolute error between predicted and expected grade (normalized to [0, 1])
    """
    try:
        pred_grade = parse_grade(predictions[0])
        if pred_grade is None:
            return 1.0  # Maximum error in normalized space
            
        expected_grade = float(formatted_doc.choices[0])
        
        # Get rubric range for normalization (required)
        min_grade = formatted_doc.specific.get("min_grade")
        max_grade = formatted_doc.specific.get("max_grade")
        
        if min_grade is None or max_grade is None:
            raise ValueError("min_grade and max_grade must be provided in formatted_doc.specific")
        
        # Normalize both grades to [0, 1] range
        pred_normalized = normalize_grade(pred_grade, min_grade, max_grade)
        expected_normalized = normalize_grade(expected_grade, min_grade, max_grade)
        
        # Return absolute error in normalized space
        return abs(pred_normalized - expected_normalized)
    except (ValueError, IndexError, TypeError, AttributeError):
        return 1.0  # Maximum error in normalized space


def grade_rmse(predictions: List[str], formatted_doc, **kwargs) -> float:
    """
    Calculate squared error for RMSE computation (normalized).
    
    Args:
        predictions: List of model predictions (strings)
        formatted_doc: Doc object with expected grade and rubric range
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Squared error between predicted and expected grade (normalized to [0, 1])
    """
    try:
        pred_grade = parse_grade(predictions[0])
        if pred_grade is None:
            return 1.0  # Maximum error in normalized space
            
        expected_grade = float(formatted_doc.choices[0])
        
        # Get rubric range for normalization (required)
        min_grade = formatted_doc.specific.get("min_grade")
        max_grade = formatted_doc.specific.get("max_grade")
        
        if min_grade is None or max_grade is None:
            raise ValueError("min_grade and max_grade must be provided in formatted_doc.specific")
        
        # Normalize both grades to [0, 1] range
        pred_normalized = normalize_grade(pred_grade, min_grade, max_grade)
        expected_normalized = normalize_grade(expected_grade, min_grade, max_grade)
        
        # Return squared error in normalized space
        return (pred_normalized - expected_normalized) ** 2
    except (ValueError, IndexError, TypeError, AttributeError):
        return 1.0  # Maximum error in normalized space


# =============================================================================
# LIGHTEVAL SAMPLE-LEVEL METRICS
# =============================================================================

# Custom SampleLevelComputation classes
class ExactGradeMatchComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs):
        return exact_grade_match(model_response.text, doc, **kwargs)

class GradeMAEComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs):
        return grade_mae(model_response.text, doc, **kwargs)

class GradeRMSEComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs):
        return grade_rmse(model_response.text, doc, **kwargs)

# Exact Grade Match Metric
exact_grade_match_metric = SampleLevelMetric(
    metric_name="exact_grade_match",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=ExactGradeMatchComputation(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


# Mean Absolute Error Metric
grade_mae_metric = SampleLevelMetric(
    metric_name="grade_mae",
    higher_is_better=False,  # Lower is better for error metrics
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=GradeMAEComputation(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

def rmse_corpus_level_fn(squared_errors):
    """Corpus-level function for RMSE calculation."""
    return np.sqrt(np.mean(squared_errors))

# Root Mean Squared Error Metric (squared errors aggregated with sqrt)
grade_rmse_metric = SampleLevelMetric(
    metric_name="grade_rmse",
    higher_is_better=False,  # Lower is better for error metrics
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=GradeRMSEComputation(),
    corpus_level_fn=rmse_corpus_level_fn,  # Square root of mean for RMSE
    batched_compute=False,
)


# =============================================================================
# CORPUS-LEVEL METRIC FUNCTIONS
# =============================================================================

def calculate_pearson_correlation(predictions: List[str], ground_truth: List[float]) -> float:
    """
    Calculate Pearson correlation between predictions and ground truth.
    
    Args:
        predictions: List of model predictions (strings)
        ground_truth: List of ground truth grades (floats)
        
    Returns:
        float: Pearson correlation coefficient
    """
    # Parse predictions to grades
    parsed_predictions = []
    for pred in predictions:
        grade = parse_grade(pred)
        if grade is not None:
            parsed_predictions.append(grade)
    
    if len(parsed_predictions) < 2 or len(ground_truth) < 2:
        return 0.0
    
    try:
        correlation, _ = pearsonr(parsed_predictions, ground_truth)
        return float(correlation)
    except Exception:
        return 0.0


def calculate_spearman_correlation(predictions: List[str], ground_truth: List[float]) -> float:
    """
    Calculate Spearman correlation between predictions and ground truth.
    
    Args:
        predictions: List of model predictions (strings)
        ground_truth: List of ground truth grades (floats)
        
    Returns:
        float: Spearman correlation coefficient
    """
    # Parse predictions to grades
    parsed_predictions = []
    for pred in predictions:
        grade = parse_grade(pred)
        if grade is not None:
            parsed_predictions.append(grade)
    
    if len(parsed_predictions) < 2 or len(ground_truth) < 2:
        return 0.0
    
    try:
        correlation, _ = spearmanr(parsed_predictions, ground_truth)
        return float(correlation)
    except Exception:
        return 0.0


def calculate_ks_statistic(predictions: List[str], ground_truth: List[float]) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic between predictions and ground truth.
    
    Args:
        predictions: List of model predictions (strings)
        ground_truth: List of ground truth grades (floats)
        
    Returns:
        float: KS statistic
    """
    # Parse predictions to grades
    parsed_predictions = []
    for pred in predictions:
        grade = parse_grade(pred)
        if grade is not None:
            parsed_predictions.append(grade)
    
    if len(parsed_predictions) < 2 or len(ground_truth) < 2:
        return 0.0
    
    try:
        ks_stat, _ = ks_2samp(parsed_predictions, ground_truth)
        return float(ks_stat)
    except Exception:
        return 0.0


def calculate_wasserstein_distance(predictions: List[str], ground_truth: List[float]) -> float:
    """
    Calculate Wasserstein distance between predictions and ground truth.
    
    Args:
        predictions: List of model predictions (strings)
        ground_truth: List of ground truth grades (floats)
        
    Returns:
        float: Wasserstein distance
    """
    # Parse predictions to grades
    parsed_predictions = []
    for pred in predictions:
        grade = parse_grade(pred)
        if grade is not None:
            parsed_predictions.append(grade)
    
    if len(parsed_predictions) < 2 or len(ground_truth) < 2:
        return 0.0
    
    try:
        wd = wasserstein_distance(parsed_predictions, ground_truth)
        return float(wd)
    except Exception:
        return 0.0


# =============================================================================
# LIGHTEVAL CORPUS-LEVEL METRICS
# =============================================================================

# Pearson Correlation Metric
pearson_correlation_metric = CorpusLevelMetric(
    metric_name="pearson_correlation",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=None,  # Not applicable for corpus-level metrics
    corpus_level_fn=calculate_pearson_correlation,
    batched_compute=False,
)

# Spearman Correlation Metric
spearman_correlation_metric = CorpusLevelMetric(
    metric_name="spearman_correlation",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=None,  # Not applicable for corpus-level metrics
    corpus_level_fn=calculate_spearman_correlation,
    batched_compute=False,
)

# Kolmogorov-Smirnov Statistic Metric
ks_statistic_metric = CorpusLevelMetric(
    metric_name="ks_statistic",
    higher_is_better=False,  # Lower is better (more similar distributions)
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=None,  # Not applicable for corpus-level metrics
    corpus_level_fn=calculate_ks_statistic,
    batched_compute=False,
)

# Wasserstein Distance Metric
wasserstein_distance_metric = CorpusLevelMetric(
    metric_name="wasserstein_distance",
    higher_is_better=False,  # Lower is better (more similar distributions)
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=None,  # Not applicable for corpus-level metrics
    corpus_level_fn=calculate_wasserstein_distance,
    batched_compute=False,
)


# =============================================================================
# SIMPLIFIED METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """
    Simplified metrics calculator for grade evaluation.
    
    Provides methods to calculate:
    - Sample-level metrics: accuracy, MAE, RMSE (via LightEval SampleLevelMetrics)
    - Corpus-level metrics: correlation, distribution comparison (via LightEval CorpusLevelMetrics)
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        # Sample-level metrics (LightEval compatible)
        self.sample_level_metrics = [
            exact_grade_match_metric,
            grade_mae_metric,
            grade_rmse_metric,
        ]
        
        # Corpus-level metrics (LightEval compatible)
        self.corpus_level_metrics = [
            pearson_correlation_metric,
            spearman_correlation_metric,
            ks_statistic_metric,
            wasserstein_distance_metric,
        ]
    
    def get_sample_level_metrics(self) -> List[SampleLevelMetric]:
        """Get list of available LightEval sample-level metrics."""
        return self.sample_level_metrics
    
    def get_corpus_level_metrics(self) -> List[CorpusLevelMetric]:
        """Get list of available LightEval corpus-level metrics."""
        return self.corpus_level_metrics
    
    def get_all_metrics(self) -> List[Union[SampleLevelMetric, CorpusLevelMetric]]:
        """Get all available metrics (both sample-level and corpus-level)."""
        return self.sample_level_metrics + self.corpus_level_metrics


# =============================================================================
# LIGHTEVAL METRIC REGISTRATION
# =============================================================================

# Create metric grouping following LightEval patterns
mentoreval_metrics = CorpusLevelMetricGrouping(
    metric_name=["exact_grade_match", "grade_mae", "grade_rmse"],
    higher_is_better={
        "exact_grade_match": True,
        "grade_mae": False,
        "grade_rmse": False,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn={
        "exact_grade_match": exact_grade_match_metric,
        "grade_mae": grade_mae_metric,
        "grade_rmse": grade_rmse_metric,
    },
    corpus_level_fn={
        "exact_grade_match": np.mean,
        "grade_mae": np.mean,
        "grade_rmse": np.mean,
    },
)

# Extend Metrics enum (only if not already registered)
from aenum import extend_enum
try:
    extend_enum(Metrics, "mentoreval_metrics", mentoreval_metrics)
except TypeError:
    # Already registered, skip
    pass

# =============================================================================
# EXPORT SECTION
# =============================================================================

# Export the main metrics for use in task.py
__all__ = [
    # Core classes
    'MetricResult', 
    'MetricsCalculator',
    
    # Sample-level metric functions
    'exact_grade_match',
    'grade_mae',
    'grade_rmse',
    
    # Sample-level LightEval metrics
    'exact_grade_match_metric',
    'grade_mae_metric',
    'grade_rmse_metric',
    
    # Corpus-level metric functions
    'calculate_pearson_correlation',
    'calculate_spearman_correlation',
    'calculate_ks_statistic',
    'calculate_wasserstein_distance',
    
    # Corpus-level LightEval metrics
    'pearson_correlation_metric',
    'spearman_correlation_metric',
    'ks_statistic_metric',
    'wasserstein_distance_metric',
    
    # Utility functions
    'parse_grade',
    
    # Metric grouping
    'mentoreval_metrics',
]
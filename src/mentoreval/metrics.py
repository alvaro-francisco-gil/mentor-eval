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
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricGrouping
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
    
    This function first tries to parse JSON format responses (as requested in prompts):
    - {"grade": [numerical_value]}
    - {"grade": [numerical_value], "explanation": "..."}
    
    If JSON parsing fails, falls back to regex extraction of the first number.
    
    Args:
        prediction: Raw model prediction (string)
        
    Returns:
        Parsed grade as float, or None if parsing fails
    """
    if not prediction or not prediction.strip():
        return None
    
    prediction = prediction.strip()
    
    # Try JSON parsing first (matches the format requested in prompts)
    try:
        import json
        
        # Look for JSON objects in the response
        json_objects = []
        
        # Try parsing the entire response as JSON first
        try:
            obj = json.loads(prediction)
            if isinstance(obj, dict):
                json_objects.append(obj)
        except json.JSONDecodeError:
            # Look for JSON objects within the text using regex
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, prediction)
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        # Check if we found any JSON objects with a 'grade' field
        for json_obj in json_objects:
            if 'grade' in json_obj:
                try:
                    grade = float(json_obj['grade'])
                    return grade
                except (ValueError, TypeError):
                    continue  # Try next JSON object
                    
    except Exception:
        pass  # Fall back to regex parsing
    
    # Fallback to regex parsing (original behavior)
    try:
        import re
        numbers = re.findall(r'-?\d+\.?\d*', prediction)
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

def exact_grade_match(model_response, doc, **kwargs) -> float:
    """
    Check for exact grade match (no normalization needed).
    
    Args:
        model_response: ModelResponse object with generated text
        doc: Doc object with expected grade
        **kwargs: Additional keyword arguments
        
    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    try:
        pred_grade = parse_grade(model_response.text[0])
        if pred_grade is None:
            return 0.0
            
        # Convert both to float for proper comparison
        expected_grade = float(doc.choices[0])
        pred_grade = float(pred_grade)
        
        # Check for exact match (no tolerance)
        return 1.0 if pred_grade == expected_grade else 0.0
    except (ValueError, IndexError, TypeError, AttributeError):
        return 0.0


def grade_evaluation_metrics(model_response, doc, **kwargs) -> dict:
    """
    Evaluate grade prediction metrics at the sample level.
    
    Args:
        model_response: ModelResponse object with generated text
        doc: Doc object with expected grade and metadata
        **kwargs: Additional keyword arguments
        
    Returns:
        dict: Dictionary with multiple grade evaluation metrics
    """
    try:
        # Parse the model's prediction from the response text
        prediction_text = model_response.text[0]
        pred_grade = parse_grade(prediction_text)
        
        # Get expected grade from doc.choices (should be populated by prompt function)
        expected_grade = float(doc.choices[0])
        
        # Get rubric range for normalization from doc.specific
        min_grade = doc.specific.get("min_grade", 0.0)
        max_grade = doc.specific.get("max_grade", 10.0)
        
        # Handle parsing failures
        if pred_grade is None:
            return {
                "exact_grade_match": 0.0,
                "grade_mae": 1.0,  # Maximum error in normalized space
                "grade_rmse": 1.0,  # Maximum error in normalized space
                "parsing_failure": 1.0,  # Track parsing failures
            }
        
        # Calculate exact match (with tolerance)
        exact_match = 1.0 if abs(pred_grade - expected_grade) < 0.1 else 0.0
        
        # Normalize grades for error calculations
        pred_normalized = normalize_grade(pred_grade, min_grade, max_grade)
        expected_normalized = normalize_grade(expected_grade, min_grade, max_grade)
        
        # Calculate normalized errors
        mae = abs(pred_normalized - expected_normalized)
        rmse = (pred_normalized - expected_normalized) ** 2
        
        return {
            "exact_grade_match": exact_match,
            "grade_mae": mae,
            "grade_rmse": rmse,
            "parsing_failure": 0.0,  # No parsing failure
        }
        
    except (ValueError, IndexError, TypeError, AttributeError) as e:
        # Handle any errors gracefully
        return {
            "exact_grade_match": 0.0,
            "grade_mae": 1.0,
            "grade_rmse": 1.0,
            "parsing_failure": 1.0,
        }




def grade_mae(model_response, doc, **kwargs) -> float:
    """
    Calculate Mean Absolute Error for grade prediction (normalized).
    
    Args:
        model_response: ModelResponse object with generated text
        doc: Doc object with expected grade and rubric range
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Absolute error between predicted and expected grade (normalized to [0, 1])
    """
    try:
        pred_grade = parse_grade(model_response.text[0])
        if pred_grade is None:
            return 1.0  # Maximum error in normalized space
            
        expected_grade = float(doc.choices[0])
        
        # Get rubric range for normalization (required)
        min_grade = doc.specific.get("min_grade")
        max_grade = doc.specific.get("max_grade")
        
        if min_grade is None or max_grade is None:
            raise ValueError("min_grade and max_grade must be provided in doc.specific")
        
        # Normalize both grades to [0, 1] range
        pred_normalized = normalize_grade(pred_grade, min_grade, max_grade)
        expected_normalized = normalize_grade(expected_grade, min_grade, max_grade)
        
        # Return absolute error in normalized space
        return abs(pred_normalized - expected_normalized)
    except (ValueError, IndexError, TypeError, AttributeError):
        return 1.0  # Maximum error in normalized space


def grade_rmse(model_response, doc, **kwargs) -> float:
    """
    Calculate squared error for RMSE computation (normalized).
    
    Args:
        model_response: ModelResponse object with generated text
        doc: Doc object with expected grade and rubric range
        **kwargs: Additional keyword arguments
        
    Returns:
        float: Squared error between predicted and expected grade (normalized to [0, 1])
    """
    try:
        pred_grade = parse_grade(model_response.text[0])
        if pred_grade is None:
            return 1.0  # Maximum error in normalized space
            
        expected_grade = float(doc.choices[0])
        
        # Get rubric range for normalization (required)
        min_grade = doc.specific.get("min_grade")
        max_grade = doc.specific.get("max_grade")
        
        if min_grade is None or max_grade is None:
            raise ValueError("min_grade and max_grade must be provided in doc.specific")
        
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
        try:
            return exact_grade_match(model_response, doc, **kwargs)
        except Exception as e:
            print(f"Error in ExactGradeMatchComputation: {e}")
            return 0.0

class GradeMAEComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs):
        try:
            return grade_mae(model_response, doc, **kwargs)
        except Exception as e:
            print(f"Error in GradeMAEComputation: {e}")
            return 1.0

class GradeRMSEComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs):
        try:
            return grade_rmse(model_response, doc, **kwargs)
        except Exception as e:
            print(f"Error in GradeRMSEComputation: {e}")
            return 1.0

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

def corpus_level_placeholder(model_response, doc, **kwargs) -> float:
    """
    Placeholder function for corpus-level metrics.
    Returns a placeholder value that will be replaced during corpus-level aggregation.
    """
    return 0.0  # Placeholder value


class CorpusLevelPlaceholderComputation(SampleLevelComputation):
    """Computation for corpus-level metrics that collects data for later processing."""
    
    def compute(self, model_response, doc, **kwargs):
        # Return the prediction and ground truth for corpus-level processing
        try:
            pred_grade = parse_grade(model_response.text[0])
            expected_grade = float(doc.choices[0])
            return {
                'prediction': pred_grade if pred_grade is not None else 0.0,
                'ground_truth': expected_grade
            }
        except (ValueError, IndexError, TypeError, AttributeError):
            return {
                'prediction': 0.0,
                'ground_truth': 0.0
            }

def calculate_pearson_correlation(sample_results, **kwargs) -> float:
    """
    Calculate Pearson correlation between predictions and ground truth.
    
    Args:
        sample_results: List of dictionaries with 'prediction' and 'ground_truth' keys
        
    Returns:
        float: Pearson correlation coefficient
    """
    try:
        if not sample_results or len(sample_results) < 2:
            return 0.0
            
        # Extract predictions and ground truth
        predictions = [result.get('prediction', 0.0) for result in sample_results]
        ground_truth = [result.get('ground_truth', 0.0) for result in sample_results]
        
        if len(predictions) != len(ground_truth) or len(predictions) < 2:
            return 0.0
        
        # Calculate Pearson correlation
        correlation, _ = pearsonr(predictions, ground_truth)
        return correlation if not np.isnan(correlation) else 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


def calculate_spearman_correlation(sample_results, **kwargs) -> float:
    """
    Calculate Spearman correlation between predictions and ground truth.
    
    Args:
        sample_results: List of dictionaries with 'prediction' and 'ground_truth' keys
        
    Returns:
        float: Spearman correlation coefficient
    """
    try:
        if not sample_results or len(sample_results) < 2:
            return 0.0
            
        # Extract predictions and ground truth
        predictions = [result.get('prediction', 0.0) for result in sample_results]
        ground_truth = [result.get('ground_truth', 0.0) for result in sample_results]
        
        if len(predictions) != len(ground_truth) or len(predictions) < 2:
            return 0.0
        
        # Calculate Spearman correlation
        correlation, _ = spearmanr(predictions, ground_truth)
        return correlation if not np.isnan(correlation) else 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


def calculate_ks_statistic(sample_results, **kwargs) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic between predictions and ground truth.
    
    Args:
        sample_results: List of dictionaries with 'prediction' and 'ground_truth' keys
        
    Returns:
        float: KS statistic
    """
    try:
        if not sample_results or len(sample_results) < 2:
            return 0.0
            
        # Extract predictions and ground truth
        predictions = [result.get('prediction', 0.0) for result in sample_results]
        ground_truth = [result.get('ground_truth', 0.0) for result in sample_results]
        
        if len(predictions) != len(ground_truth) or len(predictions) < 2:
            return 0.0
        
        # Calculate KS statistic
        ks_stat, _ = ks_2samp(predictions, ground_truth)
        return float(ks_stat)
    except (ValueError, TypeError, AttributeError):
        return 0.0


def calculate_wasserstein_distance(sample_results, **kwargs) -> float:
    """
    Calculate Wasserstein distance between predictions and ground truth.
    
    Args:
        sample_results: List of dictionaries with 'prediction' and 'ground_truth' keys
        
    Returns:
        float: Wasserstein distance
    """
    try:
        if not sample_results or len(sample_results) < 2:
            return 0.0
            
        # Extract predictions and ground truth
        predictions = [result.get('prediction', 0.0) for result in sample_results]
        ground_truth = [result.get('ground_truth', 0.0) for result in sample_results]
        
        if len(predictions) != len(ground_truth) or len(predictions) < 2:
            return 0.0
        
        # Calculate Wasserstein distance
        wd = wasserstein_distance(predictions, ground_truth)
        return float(wd)
    except (ValueError, TypeError, AttributeError):
        return 0.0


# =============================================================================
# LIGHTEVAL CORPUS-LEVEL METRICS
# =============================================================================

# Pearson Correlation Metric
pearson_correlation_metric = CorpusLevelMetric(
    metric_name="pearson_correlation",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=CorpusLevelPlaceholderComputation(),  # Placeholder for sample-level processing
    corpus_level_fn=calculate_pearson_correlation,
    batched_compute=False,
)

# Spearman Correlation Metric
spearman_correlation_metric = CorpusLevelMetric(
    metric_name="spearman_correlation",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=CorpusLevelPlaceholderComputation(),  # Placeholder for sample-level processing
    corpus_level_fn=calculate_spearman_correlation,
    batched_compute=False,
)

# Kolmogorov-Smirnov Statistic Metric
ks_statistic_metric = CorpusLevelMetric(
    metric_name="ks_statistic",
    higher_is_better=False,  # Lower is better (more similar distributions)
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=CorpusLevelPlaceholderComputation(),  # Placeholder for sample-level processing
    corpus_level_fn=calculate_ks_statistic,
    batched_compute=False,
)

# Wasserstein Distance Metric
wasserstein_distance_metric = CorpusLevelMetric(
    metric_name="wasserstein_distance",
    higher_is_better=False,  # Lower is better (more similar distributions)
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=CorpusLevelPlaceholderComputation(),  # Placeholder for sample-level processing
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

# Note: Individual metrics are used directly in task configuration
# No need for complex metric grouping

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
]
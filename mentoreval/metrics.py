"""
Centralized metrics module for MentorEval benchmark.

HIGH-LEVEL ARCHITECTURE:
========================

This module implements a 3-layer architecture for essay grading metrics:

1. DATA LAYER (RubricRange, MetricResult, NormalizedScores):
   - RubricRange: Represents scoring ranges (e.g., 1-6, 1-4) and handles normalization
   - MetricResult: Container for metric values with optional metadata for debugging
   - NormalizedScores: Stores normalized scores and composite calculations

2. PROCESSING LAYER (ScoreNormalizer, validation functions):
   - ScoreNormalizer: Converts raw scores to [0,1] range using min-max normalization
   - Validation functions: Ensure data consistency and raise exceptions for errors
   - Extraction functions: Parse rubric ranges and scores from metadata

3. METRICS LAYER (EssayGradingMetric subclasses, MetricsCalculator):
   - Individual metric classes: NMAE, NRMSE, CorrelationMetric, etc.
   - MetricsCalculator: Orchestrates all metric calculations
   - Each metric works on normalized scores for fair comparison across different scales

FLOW:
=====
Raw Scores → Extract Ranges/Scores → Validate → Normalize → Calculate Metrics → Return Results

The key insight is that normalization happens BEFORE metric calculation, ensuring
fair comparison across different rubric scales (e.g., 1-6 vs 1-4 scales).
"""

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden


@dataclass
class RubricRange:
    """
    DATA LAYER: Represents a rubric scoring range (e.g., 1-6, 1-4).
    
    This is the foundation class that handles score normalization. It's used by:
    - ScoreNormalizer: For min-max normalization to [0,1] range
    - All metric classes: To validate scores are within valid ranges
    - Extraction functions: To parse ranges from metadata
    
    The normalization formula: (score - min_score) / (max_score - min_score)
    This ensures scores from different scales (1-6 vs 1-4) are comparable.
    """
    min_score: float
    max_score: float
    
    @classmethod
    def from_string(cls, range_str: str) -> 'RubricRange':
        """Parse range string like '1-6' or '0-3'."""
        try:
            if '-' in range_str:
                min_val, max_val = range_str.split('-')
                return cls(float(min_val.strip()), float(max_val.strip()))
            else:
                # Single value, assume range of 0 to that value
                return cls(0.0, float(range_str.strip()))
        except (ValueError, IndexError):
            raise ValueError(f"Invalid range format: {range_str}")
    
    @classmethod
    def from_dict(cls, range_dict: Dict) -> 'RubricRange':
        """Parse range from dict format."""
        if not isinstance(range_dict, dict):
            raise ValueError(f"Expected dict, got {type(range_dict)}")
        
        if not range_dict:
            raise ValueError("Empty range dictionary")
        
        if 'ideal' in range_dict:
            return cls.from_string(range_dict['ideal'])
        elif 'min' in range_dict and 'max' in range_dict:
            return cls(range_dict['min'], range_dict['max'])
        else:
            # Handle format like {"ideal_writing_applications": "1-6", "ideal_conventions_score": "1-4"}
            # Find the first valid range string
            for key, value in range_dict.items():
                if isinstance(value, str) and '-' in value:
                    return cls.from_string(value)
            raise ValueError(f"No valid range found in dict: {range_dict}")
    
    @property
    def range_size(self) -> float:
        """Get the size of the scoring range."""
        return self.max_score - self.min_score
    
    def normalize_error(self, error: float) -> float:
        """Normalize error by the range size."""
        if self.range_size == 0:
            return 0.0
        return error / self.range_size
    
    def minmax_normalize(self, score: float) -> float:
        """Apply min-max normalization to scale score to [0, 1] range."""
        if self.range_size == 0:
            return 0.0
        return (score - self.min_score) / self.range_size
    
    def denormalize(self, normalized_score: float) -> float:
        """Convert normalized score back to original scale."""
        return self.min_score + (normalized_score * self.range_size)


@dataclass
class MetricResult:
    """
    DATA LAYER: Container for metric calculation results.
    
    This class is returned by all metric calculations and contains:
    - metric_name: Name of the metric (e.g., "NMAE", "Pearson Correlation")
    - value: The main metric value (e.g., 0.15 for MAE)
    - normalized_value: Optional normalized version (e.g., composite score MAE)
    - metadata: Optional debugging info (p-values, intermediate calculations, etc.)
    
    The metadata field is used for debugging and analysis - it stores things like:
    - P-values for statistical tests
    - Intermediate calculation steps
    - Raw values before normalization
    - Error details if calculations fail
    """
    metric_name: str
    value: float
    normalized_value: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class NormalizedScores:
    """
    DATA LAYER: Container for normalized scores and composite calculations.
    
    This class is created by ScoreNormalizer and used by metric classes. It contains:
    - sub_scores: Individual metric scores normalized to [0,1] range
    - composite_score: Weighted average of all sub-scores (also [0,1])
    - rubric_ranges: Original RubricRange objects for reference
    - weights: Optional weights for composite score calculation
    
    CONNECTION TO OTHER CLASSES:
    - Created by: ScoreNormalizer.normalize_scores()
    - Used by: All metric classes (NMAE, NRMSE, etc.) for calculations
    - Contains: RubricRange objects for each metric
    """
    sub_scores: Dict[str, float]  # Normalized sub-scores (0-1)
    composite_score: float  # Weighted composite score (0-1)
    rubric_ranges: Dict[str, RubricRange]  # Original rubric ranges
    weights: Optional[Dict[str, float]] = None  # Weights for composite calculation


class ScoreNormalizer:
    """
    PROCESSING LAYER: Handles min-max normalization of scores.
    
    This is the core processing class that converts raw scores to [0,1] range.
    
    CONNECTION TO OTHER CLASSES:
    - Input: Raw scores from LLM predictions and human ground truth
    - Uses: RubricRange objects for normalization
    - Output: NormalizedScores objects
    - Used by: All metric classes (NMAE, NRMSE, etc.)
    
    PROCESSING STEPS:
    1. Validate inputs (keys match, scores are numeric, within ranges)
    2. Normalize each sub-score using min-max: (score - min) / (max - min)
    3. Calculate composite score (weighted average of normalized sub-scores)
    4. Return NormalizedScores object with all normalized values
    
    This ensures fair comparison across different rubric scales (1-6 vs 1-4).
    """
    
    @staticmethod
    def normalize_scores(predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange],
                        weights: Optional[Dict[str, float]] = None) -> Tuple[NormalizedScores, NormalizedScores]:
        """
        Step 1: Normalize each sub-score per exercise using min-max normalization.
        
        Args:
            predicted_scores: Raw predicted scores for each metric
            ground_truth_scores: Raw ground truth scores for each metric
            rubric_ranges: Rubric ranges for each metric
            weights: Optional weights for composite score calculation
            
        Returns:
            Tuple of (normalized_predicted, normalized_ground_truth)
        """
        # Validate inputs first
        validate_metric_inputs(predicted_scores, ground_truth_scores, rubric_ranges)
        
        # Step 1: Normalize each sub-score to [0, 1] range
        normalized_pred_sub_scores = {}
        normalized_gt_sub_scores = {}
        
        for metric_name in predicted_scores.keys():
            rubric_range = rubric_ranges[metric_name]
            
            # Apply min-max normalization
            pred_score = predicted_scores[metric_name]
            gt_score = ground_truth_scores[metric_name]
            
            # Validate scores are within range
            if pred_score < rubric_range.min_score or pred_score > rubric_range.max_score:
                raise ValueError(f"Predicted score {pred_score} for metric '{metric_name}' is outside range {rubric_range.min_score}-{rubric_range.max_score}")
            
            if gt_score < rubric_range.min_score or gt_score > rubric_range.max_score:
                raise ValueError(f"Ground truth score {gt_score} for metric '{metric_name}' is outside range {rubric_range.min_score}-{rubric_range.max_score}")
            
            normalized_pred_sub_scores[metric_name] = rubric_range.minmax_normalize(pred_score)
            normalized_gt_sub_scores[metric_name] = rubric_range.minmax_normalize(gt_score)
        
        # Step 3: Calculate composite scores
        pred_composite = ScoreNormalizer._calculate_composite_score(
            normalized_pred_sub_scores, weights
        )
        gt_composite = ScoreNormalizer._calculate_composite_score(
            normalized_gt_sub_scores, weights
        )
        
        # Create normalized score objects
        normalized_pred = NormalizedScores(
            sub_scores=normalized_pred_sub_scores,
            composite_score=pred_composite,
            rubric_ranges=rubric_ranges,
            weights=weights
        )
        
        normalized_gt = NormalizedScores(
            sub_scores=normalized_gt_sub_scores,
            composite_score=gt_composite,
            rubric_ranges=rubric_ranges,
            weights=weights
        )
        
        return normalized_pred, normalized_gt
    
    @staticmethod
    def _calculate_composite_score(normalized_sub_scores: Dict[str, float], 
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """
        Step 3: Aggregate normalized sub-scores into composite score.
        
        Args:
            normalized_sub_scores: Normalized sub-scores (0-1)
            weights: Optional weights for each metric
            
        Returns:
            Composite score (0-1)
        """
        if not normalized_sub_scores:
            return 0.0
        
        if weights is None:
            # Equal weights
            return sum(normalized_sub_scores.values()) / len(normalized_sub_scores)
        else:
            # Weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric_name, score in normalized_sub_scores.items():
                weight = weights.get(metric_name, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0


class EssayGradingMetric(BaseMetric, ABC):
    """
    METRICS LAYER: Abstract base class for all essay grading metrics.
    
    This is the foundation for all metric calculations. It defines the interface
    that all specific metrics (NMAE, NRMSE, etc.) must implement.
    
    CONNECTION TO OTHER CLASSES:
    - Inherits from: deepeval BaseMetric (for framework compatibility)
    - Uses: ScoreNormalizer for normalization
    - Returns: MetricResult objects
    - Used by: MetricsCalculator to orchestrate all calculations
    
    The calculate_metric() method is the main interface that all metrics implement.
    It takes raw scores and rubric ranges, normalizes them, and returns results.
    """
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        super().__init__()
    
    @abstractmethod
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate the metric value on already normalized scores.
        
        Args:
            predicted_scores: Already normalized predicted scores (0-1 range)
            ground_truth_scores: Already normalized ground truth scores (0-1 range)
            
        Returns:
            MetricResult with the calculated metric value
            
        Note: This method assumes scores are already normalized. Normalization
        should happen before calling this method using ScoreNormalizer.
        """
        pass
    
    def measure(self, test_case: LLMTestCase) -> MetricResult:
        """Required by deepeval BaseMetric interface."""
        # This would need to be implemented based on how test cases are structured
        # For now, we'll use the direct calculation methods
        raise NotImplementedError("Use calculate_metric directly for now")


class NMAE(EssayGradingMetric):
    """
    METRICS LAYER: Normalized Mean Absolute Error (NMAE).
    
    This metric calculates the average absolute error between predicted and ground truth scores.
    
    CONNECTION TO OTHER CLASSES:
    - Inherits from: EssayGradingMetric
    - Input: Already normalized scores (0-1 range) from ScoreNormalizer
    - Returns: MetricResult with MAE values and metadata
    
    CALCULATION PROCESS:
    1. Validate inputs (non-empty, matching keys)
    2. Calculate absolute errors for each normalized sub-score
    3. Calculate composite score error (simple average)
    4. Return average sub-score MAE as main value, composite error as normalized_value
    
    This metric is agnostic to the original score scales - it works purely on normalized scores.
    The metadata contains intermediate calculations for debugging.
    """
    
    def __init__(self):
        super().__init__("NMAE")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate MAE on already normalized scores (0-1 range).
        
        This method assumes scores are already normalized and calculates MAE agnostically.
        """
        # Basic validation - ensure inputs are not empty and have matching keys
        if not predicted_scores or not ground_truth_scores:
            raise ValueError("predicted_scores and ground_truth_scores cannot be empty")
        
        if set(predicted_scores.keys()) != set(ground_truth_scores.keys()):
            raise ValueError("predicted_scores and ground_truth_scores must have matching keys")
        
        # Calculate MAE on normalized sub-scores
        sub_score_errors = []
        for metric_name in predicted_scores.keys():
            pred_score = predicted_scores[metric_name]
            gt_score = ground_truth_scores[metric_name]
            error = abs(pred_score - gt_score)
            sub_score_errors.append(error)
        
        # Calculate average sub-score MAE
        avg_sub_score_mae = sum(sub_score_errors) / len(sub_score_errors)
        
        # Calculate composite score MAE (simple average of all scores)
        pred_composite = sum(predicted_scores.values()) / len(predicted_scores)
        gt_composite = sum(ground_truth_scores.values()) / len(ground_truth_scores)
        composite_error = abs(pred_composite - gt_composite)
        
        return MetricResult(
            metric_name=self.metric_name,
            value=avg_sub_score_mae,  # Average sub-score MAE
            normalized_value=composite_error,  # Composite score MAE
            metadata={
                'sub_score_errors': sub_score_errors,
                'composite_error': composite_error,
                'predicted_composite': pred_composite,
                'ground_truth_composite': gt_composite,
                'n_metrics': len(predicted_scores)
            }
        )


class NRMSE(EssayGradingMetric):
    """Normalized Root Mean Square Error (NRMSE) - Works on pre-normalized scores."""
    
    def __init__(self):
        super().__init__("NRMSE")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate RMSE on already normalized scores (0-1 range).
        
        This method assumes scores are already normalized and calculates RMSE agnostically.
        """
        # Basic validation - ensure inputs are not empty and have matching keys
        if not predicted_scores or not ground_truth_scores:
            raise ValueError("predicted_scores and ground_truth_scores cannot be empty")
        
        if set(predicted_scores.keys()) != set(ground_truth_scores.keys()):
            raise ValueError("predicted_scores and ground_truth_scores must have matching keys")
        
        # Calculate RMSE on normalized sub-scores
        sub_score_squared_errors = []
        for metric_name in predicted_scores.keys():
            pred_score = predicted_scores[metric_name]
            gt_score = ground_truth_scores[metric_name]
            squared_error = (pred_score - gt_score) ** 2
            sub_score_squared_errors.append(squared_error)
        
        # Calculate RMSE for sub-scores
        avg_sub_score_rmse = (sum(sub_score_squared_errors) / len(sub_score_squared_errors)) ** 0.5
        
        # Calculate composite score RMSE
        pred_composite = sum(predicted_scores.values()) / len(predicted_scores)
        gt_composite = sum(ground_truth_scores.values()) / len(ground_truth_scores)
        composite_squared_error = (pred_composite - gt_composite) ** 2
        composite_rmse = composite_squared_error ** 0.5
        
        return MetricResult(
            metric_name=self.metric_name,
            value=avg_sub_score_rmse,  # Average sub-score RMSE
            normalized_value=composite_rmse,  # Composite score RMSE
            metadata={
                'sub_score_squared_errors': sub_score_squared_errors,
                'composite_squared_error': composite_squared_error,
                'predicted_composite': pred_composite,
                'ground_truth_composite': gt_composite,
                'n_metrics': len(predicted_scores)
            }
        )


class PerMetricMAE(EssayGradingMetric):
    """Per-metric MAE calculation."""
    
    def __init__(self, metric_name: str):
        super().__init__(f"MAE - {metric_name}")
        self.target_metric = metric_name
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate MAE for a specific metric."""
        if self.target_metric not in predicted_scores:
            raise ValueError(f"Target metric '{self.target_metric}' not found in predicted_scores")
        
        if self.target_metric not in ground_truth_scores:
            raise ValueError(f"Target metric '{self.target_metric}' not found in ground_truth_scores")
        
        if self.target_metric not in rubric_ranges:
            raise ValueError(f"Target metric '{self.target_metric}' not found in rubric_ranges")
        
        pred = predicted_scores[self.target_metric]
        gt = ground_truth_scores[self.target_metric]
        rubric_range = rubric_ranges[self.target_metric]
        
        # Validate scores are within range
        if pred < rubric_range.min_score or pred > rubric_range.max_score:
            raise ValueError(f"Predicted score {pred} for metric '{self.target_metric}' is outside range {rubric_range.min_score}-{rubric_range.max_score}")
        
        if gt < rubric_range.min_score or gt > rubric_range.max_score:
            raise ValueError(f"Ground truth score {gt} for metric '{self.target_metric}' is outside range {rubric_range.min_score}-{rubric_range.max_score}")
        
        absolute_error = abs(pred - gt)
        normalized_error = rubric_range.normalize_error(absolute_error)
        
        return MetricResult(
            metric_name=self.metric_name,
            value=absolute_error,
            normalized_value=normalized_error,
            metadata={
                'predicted': pred,
                'ground_truth': gt,
                'rubric_range': f"{rubric_range.min_score}-{rubric_range.max_score}"
            }
        )


class CorrelationMetric(EssayGradingMetric):
    """Correlation metrics (Pearson and Spearman)."""
    
    def __init__(self, correlation_type: str = "pearson"):
        super().__init__(f"{correlation_type.title()} Correlation")
        self.correlation_type = correlation_type.lower()
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, List[float]], 
                        ground_truth_scores: Dict[str, List[float]]) -> MetricResult:
        """
        Calculate correlation between predicted and ground truth scores across all samples.
        
        Args:
            predicted_scores: Dict mapping metric names to lists of scores across all samples
            ground_truth_scores: Dict mapping metric names to lists of scores across all samples
        """
        # Flatten all scores across all metrics and samples
        all_pred_values = []
        all_gt_values = []
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores:
                all_pred_values.extend(predicted_scores[metric_name])
                all_gt_values.extend(ground_truth_scores[metric_name])
        
        if len(all_pred_values) < 2:
            raise ValueError(f"Correlation requires at least 2 data points, got {len(all_pred_values)}")
        
        # Check for constant values (correlation undefined)
        if len(set(all_pred_values)) == 1:
            raise ValueError("Correlation undefined: all predicted values are identical")
        
        if len(set(all_gt_values)) == 1:
            raise ValueError("Correlation undefined: all ground truth values are identical")
        
        try:
            if self.correlation_type == "pearson":
                corr, p_value = pearsonr(all_pred_values, all_gt_values)
            elif self.correlation_type == "spearman":
                corr, p_value = spearmanr(all_pred_values, all_gt_values)
            else:
                raise ValueError(f"Unknown correlation type: {self.correlation_type}")
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(corr),
                metadata={
                    'p_value': float(p_value),
                    'n_samples': len(all_pred_values),
                    'correlation_type': self.correlation_type,
                    'n_metrics': len(predicted_scores)
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to calculate {self.correlation_type} correlation: {e}")


class JensenShannonDivergence(EssayGradingMetric):
    """Jensen-Shannon divergence for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Jensen_Shannon_Divergence")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Jensen-Shannon divergence between score distributions."""
        # Validate inputs - this will raise exceptions if invalid
        validate_metric_inputs(predicted_scores, ground_truth_scores, rubric_ranges)
        
        # Convert scores to probability distributions
        pred_values = list(predicted_scores.values())
        gt_values = list(ground_truth_scores.values())
        
        if len(pred_values) < 2:
            raise ValueError(f"Jensen-Shannon divergence requires at least 2 data points, got {len(pred_values)}")
        
        # Check for negative values
        if any(v < 0 for v in pred_values):
            raise ValueError("Jensen-Shannon divergence requires non-negative values in predicted_scores")
        
        if any(v < 0 for v in gt_values):
            raise ValueError("Jensen-Shannon divergence requires non-negative values in ground_truth_scores")
        
        try:
            # Normalize to probability distributions (sum to 1)
            pred_sum = sum(pred_values)
            gt_sum = sum(gt_values)
            
            if pred_sum == 0:
                raise ValueError("Jensen-Shannon divergence undefined: sum of predicted values is 0")
            
            if gt_sum == 0:
                raise ValueError("Jensen-Shannon divergence undefined: sum of ground truth values is 0")
            
            pred_dist = np.array(pred_values) / pred_sum
            gt_dist = np.array(gt_values) / gt_sum
            
            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(pred_dist, gt_dist, base=2)
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(js_div),
                metadata={
                    'pred_distribution': pred_dist.tolist(),
                    'gt_distribution': gt_dist.tolist(),
                    'n_samples': len(pred_values)
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to calculate Jensen-Shannon divergence: {e}")


class WassersteinDistance(EssayGradingMetric):
    """Wasserstein distance (Earth Mover's Distance) for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Wasserstein_Distance")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Wasserstein distance between score distributions."""
        # Validate inputs - this will raise exceptions if invalid
        validate_metric_inputs(predicted_scores, ground_truth_scores, rubric_ranges)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            pred_values.append(predicted_scores[metric_name])
            gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            raise ValueError(f"Wasserstein distance requires at least 2 data points, got {len(pred_values)}")
        
        try:
            # Calculate Wasserstein distance
            w_distance = wasserstein_distance(pred_values, gt_values)
            
            # Normalize by the maximum possible distance (range of all scores)
            all_scores = pred_values + gt_values
            max_range = max(all_scores) - min(all_scores)
            if max_range <= 0:
                raise ValueError("Wasserstein distance normalization failed: all scores are identical")
            
            normalized_distance = w_distance / max_range
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(w_distance),
                normalized_value=float(normalized_distance),
                metadata={
                    'pred_values': pred_values,
                    'gt_values': gt_values,
                    'max_range': max_range,
                    'n_samples': len(pred_values)
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to calculate Wasserstein distance: {e}")


class KolmogorovSmirnovTest(EssayGradingMetric):
    """Kolmogorov-Smirnov test for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Kolmogorov_Smirnov_Test")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Kolmogorov-Smirnov test statistic and p-value."""
        # Validate inputs - this will raise exceptions if invalid
        validate_metric_inputs(predicted_scores, ground_truth_scores, rubric_ranges)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            pred_values.append(predicted_scores[metric_name])
            gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            raise ValueError(f"Kolmogorov-Smirnov test requires at least 2 data points, got {len(pred_values)}")
        
        try:
            # Calculate Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(pred_values, gt_values)
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(ks_stat),
                normalized_value=float(ks_stat),  # KS statistic is already in [0, 1]
                metadata={
                    'p_value': float(p_value),
                    'pred_values': pred_values,
                    'gt_values': gt_values,
                    'n_samples': len(pred_values),
                    'significant': p_value < 0.05
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to calculate Kolmogorov-Smirnov test: {e}")


class CohensKappa(EssayGradingMetric):
    """Cohen's Kappa for measuring inter-rater agreement between LLM and human graders."""
    
    def __init__(self):
        super().__init__("Cohens_Kappa")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Cohen's Kappa for inter-rater agreement."""
        # Validate inputs - this will raise exceptions if invalid
        validate_metric_inputs(predicted_scores, ground_truth_scores, rubric_ranges)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            pred_values.append(predicted_scores[metric_name])
            gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            raise ValueError(f"Cohen's Kappa requires at least 2 data points, got {len(pred_values)}")
        
        try:
            # Convert to integers for Cohen's Kappa (categorical agreement)
            pred_categories = [int(round(val)) for val in pred_values]
            gt_categories = [int(round(val)) for val in gt_values]
            
            # Calculate Cohen's Kappa
            kappa = cohen_kappa_score(gt_categories, pred_categories)
            
            # Interpret the kappa value
            if kappa < 0:
                interpretation = "No agreement (worse than chance)"
            elif kappa <= 0.20:
                interpretation = "Slight agreement"
            elif kappa <= 0.40:
                interpretation = "Fair agreement"
            elif kappa <= 0.60:
                interpretation = "Moderate agreement"
            elif kappa <= 0.80:
                interpretation = "Substantial agreement"
            else:
                interpretation = "Almost perfect agreement"
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(kappa),
                normalized_value=float(kappa),  # Kappa is already in [-1, 1] range
                metadata={
                    'interpretation': interpretation,
                    'pred_categories': pred_categories,
                    'gt_categories': gt_categories,
                    'n_samples': len(pred_categories),
                    'raw_pred_values': pred_values,
                    'raw_gt_values': gt_values
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to calculate Cohen's Kappa: {e}")


class MetricsCalculator:
    """
    METRICS LAYER: Centralized orchestrator for all essay grading metrics.
    
    This is the main entry point for metric calculations. It manages all metric
    instances and provides methods to calculate individual or aggregate metrics.
    
    CONNECTION TO OTHER CLASSES:
    - Contains: All metric instances (NMAE, NRMSE, CorrelationMetric, etc.)
    - Uses: ScoreNormalizer to normalize scores before metric calculation
    - Uses: All metric classes via their calculate_metric() methods (on normalized scores)
    - Returns: Dict of MetricResult objects
    - Used by: Main benchmark code for evaluation
    
    ARCHITECTURE (3-Level Hierarchy):
    - Level 1 (Per-Sample): NMAE, NRMSE calculated for each individual response
    - Level 2 (Per-Exercise): Cross-sample metrics calculated across all samples in an exercise set
    - Level 3 (Per-Dataset): Average per-exercise metrics within each dataset
    - Level 4 (Overall): Average across all datasets
    
    This ensures proper statistical calculations with sufficient data points for
    correlations and distribution comparisons.
    """
    
    def __init__(self):
        # Per-sample metrics (calculated for each individual response)
        self.per_sample_metrics = {
            'nmae': NMAE(),
            'nrmse': NRMSE(),
        }
        
        # Cross-sample metrics (calculated across all responses)
        self.cross_sample_metrics = {
            'pearson_correlation': CorrelationMetric('pearson'),
            'spearman_correlation': CorrelationMetric('spearman'),
            'jensen_shannon_divergence': JensenShannonDivergence(),
            'wasserstein_distance': WassersteinDistance(),
            'kolmogorov_smirnov_test': KolmogorovSmirnovTest(),
            'cohens_kappa': CohensKappa(),
        }
        
        # Combined for backward compatibility
        self.metrics = {**self.per_sample_metrics, **self.cross_sample_metrics}
    
    def add_per_metric_mae(self, metric_names: List[str]):
        """Add per-metric MAE calculators."""
        for metric_name in metric_names:
            self.metrics[f'mae_{metric_name.lower()}'] = PerMetricMAE(metric_name)
    
    def calculate_per_sample_metrics(self, 
                                   predicted_scores: Dict[str, float], 
                                   ground_truth_scores: Dict[str, float],
                                   rubric_ranges: Dict[str, RubricRange]) -> Dict[str, MetricResult]:
        """
        Calculate per-sample metrics (NMAE, NRMSE) for a single response.
        
        These metrics are calculated for each individual response and measure
        the error between predicted and ground truth scores.
        """
        # Step 1: Normalize scores first
        normalized_pred, normalized_gt = ScoreNormalizer.normalize_scores(
            predicted_scores, ground_truth_scores, rubric_ranges
        )
        
        # Step 2: Calculate per-sample metrics on normalized scores
        results = {}
        for metric_name, metric in self.per_sample_metrics.items():
            result = metric.calculate_metric(normalized_pred.sub_scores, normalized_gt.sub_scores)
            results[metric_name] = result
        
        return results
    
    def calculate_cross_sample_metrics(self, 
                                     all_predictions: List[Dict[str, float]], 
                                     all_ground_truth: List[Dict[str, float]],
                                     all_rubric_ranges: List[Dict[str, RubricRange]]) -> Dict[str, MetricResult]:
        """
        Calculate cross-sample metrics (correlations, distribution comparisons) across all responses.
        
        These metrics require multiple samples to be meaningful and measure
        relationships and distributions across the entire dataset.
        """
        if not all_predictions or not all_ground_truth:
            raise ValueError("all_predictions and all_ground_truth cannot be empty")
        
        if len(all_predictions) != len(all_ground_truth):
            raise ValueError("all_predictions and all_ground_truth must have the same length")
        
        # Step 1: Normalize all samples
        all_normalized_pred = []
        all_normalized_gt = []
        
        for pred_scores, gt_scores, rubric_ranges in zip(all_predictions, all_ground_truth, all_rubric_ranges):
            normalized_pred, normalized_gt = ScoreNormalizer.normalize_scores(
                pred_scores, gt_scores, rubric_ranges
            )
            all_normalized_pred.append(normalized_pred.sub_scores)
            all_normalized_gt.append(normalized_gt.sub_scores)
        
        # Step 2: Flatten all normalized scores for cross-sample calculations
        flattened_pred = {}
        flattened_gt = {}
        
        for metric_name in all_normalized_pred[0].keys():
            flattened_pred[metric_name] = [sample[metric_name] for sample in all_normalized_pred]
            flattened_gt[metric_name] = [sample[metric_name] for sample in all_normalized_gt]
        
        # Step 3: Calculate cross-sample metrics
        results = {}
        for metric_name, metric in self.cross_sample_metrics.items():
            result = metric.calculate_metric(flattened_pred, flattened_gt)
            results[metric_name] = result
        
        return results
    
    def calculate_exercise_metrics(self, 
                                  all_predictions: List[Dict[str, float]], 
                                  all_ground_truth: List[Dict[str, float]],
                                  all_rubric_ranges: List[Dict[str, RubricRange]]) -> Dict[str, float]:
        """
        Calculate all metrics for a single exercise set (Level 2).
        
        This method:
        1. Calculates per-sample metrics (NMAE, NRMSE) for each response and averages them
        2. Calculates cross-sample metrics (correlations, distributions) across all responses in the exercise
        
        Args:
            all_predictions: List of predicted scores for all samples in the exercise
            all_ground_truth: List of ground truth scores for all samples in the exercise  
            all_rubric_ranges: List of rubric ranges for all samples in the exercise
            
        Returns:
            Dict with averaged per-sample metrics and cross-sample metrics
        """
        if not all_predictions or not all_ground_truth:
            raise ValueError("all_predictions and all_ground_truth cannot be empty")
        
        if len(all_predictions) != len(all_ground_truth):
            raise ValueError("all_predictions and all_ground_truth must have the same length")
        
        # Step 1: Calculate per-sample metrics for each response and average them
        per_sample_results = []
        for pred_scores, gt_scores, rubric_ranges in zip(all_predictions, all_ground_truth, all_rubric_ranges):
            sample_metrics = self.calculate_per_sample_metrics(pred_scores, gt_scores, rubric_ranges)
            per_sample_results.append(sample_metrics)
        
        # Average per-sample metrics across all responses in the exercise
        averaged_per_sample = {}
        for metric_name in self.per_sample_metrics.keys():
            values = [result[metric_name].value for result in per_sample_results]
            averaged_per_sample[metric_name] = sum(values) / len(values)
        
        # Step 2: Calculate cross-sample metrics across all responses in the exercise
        cross_sample_results = self.calculate_cross_sample_metrics(
            all_predictions, all_ground_truth, all_rubric_ranges
        )
        
        # Combine results
        exercise_metrics = {**averaged_per_sample}
        for metric_name, result in cross_sample_results.items():
            exercise_metrics[metric_name] = result.value
        
        return exercise_metrics
    
    def calculate_dataset_metrics(self, 
                                 exercise_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate dataset-level metrics by averaging exercise-level metrics (Level 3).
        
        Args:
            exercise_metrics_list: List of exercise-level metrics for each exercise in the dataset
            
        Returns:
            Dict with averaged metrics across all exercises in the dataset
        """
        if not exercise_metrics_list:
            raise ValueError("exercise_metrics_list cannot be empty")
        
        # Get all metric names from the first exercise
        all_metric_names = set(exercise_metrics_list[0].keys())
        
        # Average each metric across all exercises
        dataset_metrics = {}
        for metric_name in all_metric_names:
            values = []
            for exercise_metrics in exercise_metrics_list:
                if metric_name in exercise_metrics:
                    values.append(exercise_metrics[metric_name])
            
            if values:
                dataset_metrics[metric_name] = sum(values) / len(values)
        
        return dataset_metrics
    
    def calculate_overall_metrics(self, 
                                 dataset_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate overall metrics by averaging dataset-level metrics (Level 4).
        
        Args:
            dataset_metrics_list: List of dataset-level metrics for each dataset
            
        Returns:
            Dict with averaged metrics across all datasets
        """
        if not dataset_metrics_list:
            raise ValueError("dataset_metrics_list cannot be empty")
        
        # Get all metric names from the first dataset
        all_metric_names = set(dataset_metrics_list[0].keys())
        
        # Average each metric across all datasets
        overall_metrics = {}
        for metric_name in all_metric_names:
            values = []
            for dataset_metrics in dataset_metrics_list:
                if metric_name in dataset_metrics:
                    values.append(dataset_metrics[metric_name])
            
            if values:
                overall_metrics[metric_name] = sum(values) / len(values)
        
        return overall_metrics
    
    def calculate_all_metrics(self, 
                             predicted_scores: Dict[str, float], 
                             ground_truth_scores: Dict[str, float],
                             rubric_ranges: Dict[str, RubricRange]) -> Dict[str, MetricResult]:
        """
        Calculate per-sample metrics only (for backward compatibility).
        
        For proper exercise-level metrics, use calculate_exercise_metrics() with all samples.
        """
        return self.calculate_per_sample_metrics(predicted_scores, ground_truth_scores, rubric_ranges)
    
    def calculate_aggregate_metrics(self, 
                                   all_predictions: List[Dict[str, float]], 
                                   all_ground_truth: List[Dict[str, float]],
                                   all_rubric_ranges: List[Dict[str, RubricRange]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all samples."""
        if not all_predictions:
            raise ValueError("all_predictions is empty")
        
        if not all_ground_truth:
            raise ValueError("all_ground_truth is empty")
        
        if len(all_predictions) != len(all_ground_truth):
            raise ValueError(f"Length mismatch: {len(all_predictions)} predictions vs {len(all_ground_truth)} ground truth")
        
        if len(all_predictions) != len(all_rubric_ranges):
            raise ValueError(f"Length mismatch: {len(all_predictions)} predictions vs {len(all_rubric_ranges)} rubric ranges")
        
        # Flatten all predictions and ground truth for correlation calculations
        all_pred_values = []
        all_gt_values = []
        
        for pred_dict, gt_dict in zip(all_predictions, all_ground_truth):
            for metric_name in pred_dict.keys():
                if metric_name not in gt_dict:
                    raise ValueError(f"Metric '{metric_name}' missing in ground truth")
                all_pred_values.append(pred_dict[metric_name])
                all_gt_values.append(gt_dict[metric_name])
        
        if not all_pred_values:
            raise ValueError("No valid prediction-ground truth pairs found")
        
        # Calculate aggregate MAE
        mae = np.mean([abs(p - gt) for p, gt in zip(all_pred_values, all_gt_values)])
        
        # Calculate correlations
        if len(all_pred_values) < 2:
            raise ValueError(f"Correlation requires at least 2 data points, got {len(all_pred_values)}")
        
        try:
            pearson_corr, pearson_p = pearsonr(all_pred_values, all_gt_values)
            spearman_corr, spearman_p = spearmanr(all_pred_values, all_gt_values)
            
            return {
                'mae': float(mae),
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'n_samples': len(all_pred_values)
            }
        except Exception as e:
            raise ValueError(f"Failed to calculate correlations: {e}")


def parse_rubric_range(rubric_range_data: Union[str, Dict]) -> RubricRange:
    """Parse rubric range from various formats."""
    if isinstance(rubric_range_data, str):
        return RubricRange.from_string(rubric_range_data)
    elif isinstance(rubric_range_data, dict):
        return RubricRange.from_dict(rubric_range_data)
    else:
        raise ValueError(f"Unsupported rubric range format: {type(rubric_range_data)}")


def extract_rubric_ranges_from_metadata(metadata: Dict) -> Dict[str, RubricRange]:
    """
    PROCESSING LAYER: Extract rubric ranges for each metric from metadata.
    
    This function parses rubric range information from the dataset metadata.
    It handles both single ranges and multiple ranges per metric.
    
    CONNECTION TO OTHER CLASSES:
    - Input: Metadata from dataset (contains 'rubric_range' field)
    - Output: Dict of RubricRange objects
    - Used by: Main benchmark code to prepare data for metrics
    
    SUPPORTED FORMATS:
    1. Single range: {"rubric_range": "1-6"}
    2. Multiple ranges: {"rubric_range": {"ideal_writing_applications": "1-6", "ideal_conventions_score": "1-4"}}
    3. Dict with 'ideal' key: {"rubric_range": {"ideal": "1-6"}}
    
    The function automatically extracts metric names from ideal_* fields in metadata.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected dict, got {type(metadata)}")
    
    rubric_ranges = {}
    
    if 'rubric_range' not in metadata:
        raise ValueError("Missing 'rubric_range' field in metadata")
    
    rubric_range_data = metadata['rubric_range']
    
    if isinstance(rubric_range_data, dict):
        # Handle multiple ranges: {"ideal_writing_applications": "1-6", "ideal_conventions_score": "1-4"}
        for key, range_str in rubric_range_data.items():
            if key.startswith('ideal_'):
                # Extract metric name from the key
                metric_name = key.replace('ideal_', '').replace('_score', '').title()
                try:
                    rubric_ranges[metric_name] = parse_rubric_range(range_str)
                except ValueError as e:
                    raise ValueError(f"Invalid range for metric '{metric_name}': {e}")
    else:
        # Handle single range: "1-6" or {"ideal": "1-6"}
        try:
            main_range = parse_rubric_range(rubric_range_data)
        except ValueError as e:
            raise ValueError(f"Invalid main rubric range: {e}")
        
        # Extract metric names from all ideal_* fields in metadata
        for key in metadata.keys():
            if key.startswith('ideal_') and key != 'ideal':
                metric_name = key.replace('ideal_', '').replace('_score', '').title()
                rubric_ranges[metric_name] = main_range
    
    if not rubric_ranges:
        raise ValueError("No valid metric ranges found in metadata")
    
    return rubric_ranges


def validate_metric_inputs(predicted_scores: Dict[str, float], 
                          ground_truth_scores: Dict[str, float],
                          rubric_ranges: Dict[str, RubricRange]) -> None:
    """
    PROCESSING LAYER: Validates that metric inputs are consistent and complete.
    
    This function ensures data integrity before metric calculations. It's used by:
    - ScoreNormalizer: Before normalization
    - All metric classes: Before calculations
    
    VALIDATION CHECKS:
    1. Input types are correct (dicts, not empty)
    2. All three dictionaries have matching keys
    3. All scores are numeric
    4. All scores are within their respective rubric ranges
    5. Rubric ranges are valid (range_size > 0)
    
    Raises ValueError with detailed error messages if validation fails.
    """
    if not isinstance(predicted_scores, dict):
        raise ValueError(f"predicted_scores must be dict, got {type(predicted_scores)}")
    
    if not isinstance(ground_truth_scores, dict):
        raise ValueError(f"ground_truth_scores must be dict, got {type(ground_truth_scores)}")
    
    if not isinstance(rubric_ranges, dict):
        raise ValueError(f"rubric_ranges must be dict, got {type(rubric_ranges)}")
    
    pred_keys = set(predicted_scores.keys())
    gt_keys = set(ground_truth_scores.keys())
    range_keys = set(rubric_ranges.keys())
    
    if not pred_keys:
        raise ValueError("predicted_scores is empty")
    
    if not gt_keys:
        raise ValueError("ground_truth_scores is empty")
    
    if not range_keys:
        raise ValueError("rubric_ranges is empty")
    
    # Check for key mismatches
    missing_pred = range_keys - pred_keys
    missing_gt = range_keys - gt_keys
    missing_range = (pred_keys | gt_keys) - range_keys
    
    if missing_pred or missing_gt or missing_range:
        error_msg = "Metric key mismatch:\n"
        if missing_pred:
            error_msg += f"  Missing in predicted_scores: {missing_pred}\n"
        if missing_gt:
            error_msg += f"  Missing in ground_truth_scores: {missing_gt}\n"
        if missing_range:
            error_msg += f"  Missing in rubric_ranges: {missing_range}\n"
        raise ValueError(error_msg)
    
    # Validate score values
    for metric_name, score in predicted_scores.items():
        if not isinstance(score, (int, float)):
            raise ValueError(f"predicted_scores['{metric_name}'] must be numeric, got {type(score)}")
        if not isinstance(ground_truth_scores[metric_name], (int, float)):
            raise ValueError(f"ground_truth_scores['{metric_name}'] must be numeric, got {type(ground_truth_scores[metric_name])}")
    
    # Validate rubric ranges
    for metric_name, rubric_range in rubric_ranges.items():
        if not isinstance(rubric_range, RubricRange):
            raise ValueError(f"rubric_ranges['{metric_name}'] must be RubricRange, got {type(rubric_range)}")
        if rubric_range.range_size <= 0:
            raise ValueError(f"rubric_ranges['{metric_name}'] has invalid range: {rubric_range.min_score}-{rubric_range.max_score}")


def extract_scores_from_metadata(metadata: Dict, metric_names: List[str]) -> Dict[str, float]:
    """
    PROCESSING LAYER: Extract ground truth scores from metadata for any field starting with 'ideal_'.
    
    This function is generic and handles any metric name by trying multiple key formats.
    It's designed to work with the actual data structure in the datasets.
    
    CONNECTION TO OTHER CLASSES:
    - Input: Metadata from dataset, list of metric names
    - Output: Dict of metric_name -> score mappings
    - Used by: Main benchmark code to prepare ground truth data
    
    KEY MATCHING STRATEGY:
    For each metric name, it tries these key formats:
    1. "ideal_{metric_name.lower()}_score" (e.g., "ideal_writing_applications_score")
    2. "ideal_{metric_name.lower()}" (e.g., "ideal_writing_applications")
    3. "ideal_{metric_name.lower().replace('_', '')}" (e.g., "ideal_writingapplications")
    4. "ideal_{metric_name.lower().replace(' ', '_')}" (e.g., "ideal_writing_applications")
    
    This flexible approach handles various naming conventions in the datasets.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected dict, got {type(metadata)}")
    
    if not isinstance(metric_names, list):
        raise ValueError(f"Expected list, got {type(metric_names)}")
    
    scores = {}
    missing_metrics = []
    
    for metric_name in metric_names:
        # Try multiple possible key formats for each metric
        possible_keys = [
            f"ideal_{metric_name.lower()}_score",
            f"ideal_{metric_name.lower()}",
            f"ideal_{metric_name.lower().replace('_', '')}",
            f"ideal_{metric_name.lower().replace(' ', '_')}"
        ]
        
        score_found = False
        for key in possible_keys:
            if key in metadata and metadata[key] is not None:
                try:
                    scores[metric_name] = float(metadata[key])
                    score_found = True
                    break
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid score value for metric '{metric_name}' (key '{key}'): {metadata[key]} - {e}")
        
        if not score_found:
            missing_metrics.append(metric_name)
    
    if missing_metrics:
        available_ideal_keys = [key for key in metadata.keys() if key.startswith('ideal_')]
        raise ValueError(f"Missing scores for metrics: {missing_metrics}. Available ideal_* keys: {available_ideal_keys}")
    
    return scores

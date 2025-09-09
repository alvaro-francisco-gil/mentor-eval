"""
Centralized metrics module for MentorEval benchmark.

This module provides custom metrics for essay grading evaluation,
including normalized MAE that accounts for rubric ranges.
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
    """Represents a rubric scoring range."""
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
        if 'ideal' in range_dict:
            return cls.from_string(range_dict['ideal'])
        elif 'min' in range_dict and 'max' in range_dict:
            return cls(range_dict['min'], range_dict['max'])
        else:
            # Handle format like {"ideal_writing_applications": "1-6", "ideal_conventions_score": "1-4"}
            # Use the first range found as the main range
            for key, value in range_dict.items():
                if isinstance(value, str) and '-' in value:
                    return cls.from_string(value)
            raise ValueError(f"Invalid range dict format: {range_dict}")
    
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
    """Result of a metric calculation."""
    metric_name: str
    value: float
    normalized_value: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class NormalizedScores:
    """Container for normalized scores and metadata."""
    sub_scores: Dict[str, float]  # Normalized sub-scores (0-1)
    composite_score: float  # Weighted composite score (0-1)
    rubric_ranges: Dict[str, RubricRange]  # Original rubric ranges
    weights: Optional[Dict[str, float]] = None  # Weights for composite calculation


class ScoreNormalizer:
    """Handles min-max normalization of scores according to the hybrid approach."""
    
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
        # Step 1: Normalize each sub-score to [0, 1] range
        normalized_pred_sub_scores = {}
        normalized_gt_sub_scores = {}
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores and metric_name in rubric_ranges:
                rubric_range = rubric_ranges[metric_name]
                
                # Apply min-max normalization
                pred_score = predicted_scores[metric_name]
                gt_score = ground_truth_scores[metric_name]
                
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
    """Base class for essay grading metrics."""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        super().__init__()
    
    @abstractmethod
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate the metric value."""
        pass
    
    def measure(self, test_case: LLMTestCase) -> MetricResult:
        """Required by deepeval BaseMetric interface."""
        # This would need to be implemented based on how test cases are structured
        # For now, we'll use the direct calculation methods
        raise NotImplementedError("Use calculate_metric directly for now")


class NMAE(EssayGradingMetric):
    """Normalized Mean Absolute Error (NMAE) - Works on pre-normalized scores."""
    
    def __init__(self):
        super().__init__("NMAE")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate MAE on pre-normalized scores (0-1 range)."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0, 0.0)
        
        # Step 1: Normalize scores first
        normalized_pred, normalized_gt = ScoreNormalizer.normalize_scores(
            predicted_scores, ground_truth_scores, rubric_ranges
        )
        
        # Step 2: Calculate metrics on normalized sub-scores
        sub_score_errors = []
        for metric_name in normalized_pred.sub_scores.keys():
            if metric_name in normalized_gt.sub_scores:
                pred_norm = normalized_pred.sub_scores[metric_name]
                gt_norm = normalized_gt.sub_scores[metric_name]
                error = abs(pred_norm - gt_norm)
                sub_score_errors.append(error)
        
        # Step 4: Calculate metrics on composite scores
        composite_error = abs(normalized_pred.composite_score - normalized_gt.composite_score)
        
        # Calculate average sub-score MAE
        avg_sub_score_mae = sum(sub_score_errors) / len(sub_score_errors) if sub_score_errors else 0.0
        
        return MetricResult(
            metric_name=self.metric_name,
            value=avg_sub_score_mae,  # Average sub-score MAE
            normalized_value=composite_error,  # Composite score MAE
            metadata={
                'sub_score_errors': sub_score_errors,
                'composite_error': composite_error,
                'normalized_predicted': normalized_pred,
                'normalized_ground_truth': normalized_gt
            }
        )


class NRMSE(EssayGradingMetric):
    """Normalized Root Mean Square Error (NRMSE) - Works on pre-normalized scores."""
    
    def __init__(self):
        super().__init__("NRMSE")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate RMSE on pre-normalized scores (0-1 range)."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0, 0.0)
        
        # Step 1: Normalize scores first
        normalized_pred, normalized_gt = ScoreNormalizer.normalize_scores(
            predicted_scores, ground_truth_scores, rubric_ranges
        )
        
        # Step 2: Calculate metrics on normalized sub-scores
        sub_score_squared_errors = []
        for metric_name in normalized_pred.sub_scores.keys():
            if metric_name in normalized_gt.sub_scores:
                pred_norm = normalized_pred.sub_scores[metric_name]
                gt_norm = normalized_gt.sub_scores[metric_name]
                squared_error = (pred_norm - gt_norm) ** 2
                sub_score_squared_errors.append(squared_error)
        
        # Step 4: Calculate metrics on composite scores
        composite_squared_error = (normalized_pred.composite_score - normalized_gt.composite_score) ** 2
        
        # Calculate RMSE for sub-scores
        avg_sub_score_rmse = (sum(sub_score_squared_errors) / len(sub_score_squared_errors)) ** 0.5 if sub_score_squared_errors else 0.0
        
        # Calculate RMSE for composite score
        composite_rmse = composite_squared_error ** 0.5
        
        return MetricResult(
            metric_name=self.metric_name,
            value=avg_sub_score_rmse,  # Average sub-score RMSE
            normalized_value=composite_rmse,  # Composite score RMSE
            metadata={
                'sub_score_squared_errors': sub_score_squared_errors,
                'composite_squared_error': composite_squared_error,
                'normalized_predicted': normalized_pred,
                'normalized_ground_truth': normalized_gt
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
        if (self.target_metric not in predicted_scores or 
            self.target_metric not in ground_truth_scores or
            self.target_metric not in rubric_ranges):
            return MetricResult(self.metric_name, 0.0, 0.0)
        
        pred = predicted_scores[self.target_metric]
        gt = ground_truth_scores[self.target_metric]
        rubric_range = rubric_ranges[self.target_metric]
        
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
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate correlation between predicted and ground truth scores."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0)
        
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores:
                pred_values.append(predicted_scores[metric_name])
                gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            return MetricResult(self.metric_name, 0.0)
        
        try:
            if self.correlation_type == "pearson":
                corr, p_value = pearsonr(pred_values, gt_values)
            elif self.correlation_type == "spearman":
                corr, p_value = spearmanr(pred_values, gt_values)
            else:
                raise ValueError(f"Unknown correlation type: {self.correlation_type}")
            
            return MetricResult(
                metric_name=self.metric_name,
                value=float(corr),
                metadata={
                    'p_value': float(p_value),
                    'n_samples': len(pred_values),
                    'correlation_type': self.correlation_type
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name=self.metric_name,
                value=0.0,
                metadata={'error': str(e)}
            )


class JensenShannonDivergence(EssayGradingMetric):
    """Jensen-Shannon divergence for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Jensen_Shannon_Divergence")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Jensen-Shannon divergence between score distributions."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0)
        
        # Convert scores to probability distributions
        pred_values = list(predicted_scores.values())
        gt_values = list(ground_truth_scores.values())
        
        if len(pred_values) < 2 or len(gt_values) < 2:
            return MetricResult(self.metric_name, 0.0)
        
        try:
            # Normalize to probability distributions (sum to 1)
            pred_sum = sum(pred_values)
            gt_sum = sum(gt_values)
            
            if pred_sum == 0 or gt_sum == 0:
                return MetricResult(self.metric_name, 0.0)
            
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
            return MetricResult(
                metric_name=self.metric_name,
                value=0.0,
                metadata={'error': str(e)}
            )


class WassersteinDistance(EssayGradingMetric):
    """Wasserstein distance (Earth Mover's Distance) for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Wasserstein_Distance")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Wasserstein distance between score distributions."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores:
                pred_values.append(predicted_scores[metric_name])
                gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            return MetricResult(self.metric_name, 0.0)
        
        try:
            # Calculate Wasserstein distance
            w_distance = wasserstein_distance(pred_values, gt_values)
            
            # Normalize by the maximum possible distance (range of all scores)
            all_scores = pred_values + gt_values
            max_range = max(all_scores) - min(all_scores) if all_scores else 1.0
            normalized_distance = w_distance / max_range if max_range > 0 else 0.0
            
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
            return MetricResult(
                metric_name=self.metric_name,
                value=0.0,
                metadata={'error': str(e)}
            )


class KolmogorovSmirnovTest(EssayGradingMetric):
    """Kolmogorov-Smirnov test for comparing score distributions."""
    
    def __init__(self):
        super().__init__("Kolmogorov_Smirnov_Test")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Kolmogorov-Smirnov test statistic and p-value."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores:
                pred_values.append(predicted_scores[metric_name])
                gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            return MetricResult(self.metric_name, 0.0)
        
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
            return MetricResult(
                metric_name=self.metric_name,
                value=0.0,
                metadata={'error': str(e)}
            )


class CohensKappa(EssayGradingMetric):
    """Cohen's Kappa for measuring inter-rater agreement between LLM and human graders."""
    
    def __init__(self):
        super().__init__("Cohens_Kappa")
    
    def calculate_metric(self, 
                        predicted_scores: Dict[str, float], 
                        ground_truth_scores: Dict[str, float],
                        rubric_ranges: Dict[str, RubricRange]) -> MetricResult:
        """Calculate Cohen's Kappa for inter-rater agreement."""
        if not predicted_scores or not ground_truth_scores:
            return MetricResult(self.metric_name, 0.0)
        
        # Extract values in the same order
        pred_values = []
        gt_values = []
        
        for metric_name in predicted_scores.keys():
            if metric_name in ground_truth_scores:
                pred_values.append(predicted_scores[metric_name])
                gt_values.append(ground_truth_scores[metric_name])
        
        if len(pred_values) < 2:
            return MetricResult(self.metric_name, 0.0)
        
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
            return MetricResult(
                metric_name=self.metric_name,
                value=0.0,
                metadata={'error': str(e)}
            )


class MetricsCalculator:
    """Centralized calculator for all essay grading metrics."""
    
    def __init__(self):
        self.metrics = {
            'nmae': NMAE(),
            'nrmse': NRMSE(),
            'pearson_correlation': CorrelationMetric('pearson'),
            'spearman_correlation': CorrelationMetric('spearman'),
            'jensen_shannon_divergence': JensenShannonDivergence(),
            'wasserstein_distance': WassersteinDistance(),
            'kolmogorov_smirnov_test': KolmogorovSmirnovTest(),
            'cohens_kappa': CohensKappa(),
        }
    
    def add_per_metric_mae(self, metric_names: List[str]):
        """Add per-metric MAE calculators."""
        for metric_name in metric_names:
            self.metrics[f'mae_{metric_name.lower()}'] = PerMetricMAE(metric_name)
    
    def calculate_all_metrics(self, 
                             predicted_scores: Dict[str, float], 
                             ground_truth_scores: Dict[str, float],
                             rubric_ranges: Dict[str, RubricRange]) -> Dict[str, MetricResult]:
        """Calculate all configured metrics."""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                result = metric.calculate_metric(predicted_scores, ground_truth_scores, rubric_ranges)
                results[metric_name] = result
            except Exception as e:
                results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    value=0.0,
                    metadata={'error': str(e)}
                )
        
        return results
    
    def calculate_aggregate_metrics(self, 
                                   all_predictions: List[Dict[str, float]], 
                                   all_ground_truth: List[Dict[str, float]],
                                   all_rubric_ranges: List[Dict[str, RubricRange]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all samples."""
        if not all_predictions or not all_ground_truth:
            return {}
        
        # Flatten all predictions and ground truth for correlation calculations
        all_pred_values = []
        all_gt_values = []
        
        for pred_dict, gt_dict in zip(all_predictions, all_ground_truth):
            for metric_name in pred_dict.keys():
                if metric_name in gt_dict:
                    all_pred_values.append(pred_dict[metric_name])
                    all_gt_values.append(gt_dict[metric_name])
        
        # Calculate aggregate MAE
        if all_pred_values and all_gt_values:
            mae = np.mean([abs(p - gt) for p, gt in zip(all_pred_values, all_gt_values)])
            
            # Calculate correlations
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
            except Exception:
                return {
                    'mae': float(mae),
                    'n_samples': len(all_pred_values)
                }
        
        return {}


def parse_rubric_range(rubric_range_data: Union[str, Dict]) -> RubricRange:
    """Parse rubric range from various formats."""
    if isinstance(rubric_range_data, str):
        return RubricRange.from_string(rubric_range_data)
    elif isinstance(rubric_range_data, dict):
        return RubricRange.from_dict(rubric_range_data)
    else:
        raise ValueError(f"Unsupported rubric range format: {type(rubric_range_data)}")


def extract_rubric_ranges_from_metadata(metadata: Dict) -> Dict[str, RubricRange]:
    """Extract rubric ranges for each metric from metadata."""
    rubric_ranges = {}
    
    # Get the main rubric range
    if 'rubric_range' in metadata:
        main_range = parse_rubric_range(metadata['rubric_range'])
        
        # Extract metric names from ideal_*_score fields
        for key in metadata.keys():
            if key.startswith('ideal_') and key.endswith('_score'):
                metric_name = key.replace('ideal_', '').replace('_score', '').title()
                rubric_ranges[metric_name] = main_range
    
    return rubric_ranges


def extract_scores_from_metadata(metadata: Dict, metric_names: List[str]) -> Dict[str, float]:
    """Extract ground truth scores from metadata."""
    scores = {}
    
    for metric_name in metric_names:
        score_key = f"ideal_{metric_name.lower()}_score"
        if score_key in metadata and metadata[score_key] is not None:
            try:
                scores[metric_name] = float(metadata[score_key])
            except (ValueError, TypeError):
                continue
    
    return scores

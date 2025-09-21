# Minimal imports for LightEval integration
# Only import what's needed to avoid circular dependencies

# Metrics (needed for LightEval tasks)
from .metrics import (
    exact_grade_match_metric,
    grade_mae_metric,
    grade_rmse_metric,
    pearson_correlation_metric,
    spearman_correlation_metric,
    ks_statistic_metric,
    wasserstein_distance_metric
)

# Prompt functions (needed for LightEval tasks)
from .prompts import mentor_eval_prompt_fn

# Core exports for LightEval integration
__all__ = [
    # Metrics
    'exact_grade_match_metric',
    'grade_mae_metric',
    'grade_rmse_metric',
    'pearson_correlation_metric',
    'spearman_correlation_metric',
    'ks_statistic_metric',
    'wasserstein_distance_metric',
    
    # Prompt functions
    'mentor_eval_prompt_fn',
]
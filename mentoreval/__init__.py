# Core task definitions
from .task import MentorEvalTask, MentorEvalTasks, MentorEvalDataset, TASKS_TABLE, custom_few_shot_select

# Prompt functions
from .prompts import mentor_eval_prompt_fn

# LightEval built-in metrics (currently not available in v0.10.0)
LIGHTEVAL_METRICS_AVAILABLE = False

# Simplified metrics
from .metrics import (
    MetricsCalculator,
    MetricResult,
    exact_grade_match_metric,
    grade_mae_metric,
    grade_rmse_metric,
    pearson_correlation_metric,
    spearman_correlation_metric,
    ks_statistic_metric,
    wasserstein_distance_metric,
    parse_grade,
    normalize_grade
)

# LightEval-compatible models
try:
    from .models import (
        ModelConfig,
        LightEvalModelFactory,
        create_model_config
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# LightEval benchmark integration (native pipeline)
try:
    from .benchmark import (
        LightEvalBenchmark,
        create_lighteval_benchmark,
        run_lighteval_evaluation
    )
    LIGHTEVAL_BENCHMARK_AVAILABLE = True
except ImportError:
    LIGHTEVAL_BENCHMARK_AVAILABLE = False

# Legacy imports removed - only LightEval is supported now
LEGACY_AVAILABLE = False

# Note: config.py removed - using LightEval's native configuration system

# Core exports
__all__ = [
    # Task definitions
    'MentorEvalTask', 
    'MentorEvalTasks', 
    'MentorEvalDataset',
    'TASKS_TABLE',
    
    # Prompt functions
    'mentor_eval_prompt_fn',
    'custom_few_shot_select',
    
    # Metrics
    'MetricsCalculator',
    'MetricResult',
    'exact_grade_match_metric',
    'grade_mae_metric',
    'grade_rmse_metric',
    'pearson_correlation_metric',
    'spearman_correlation_metric',
    'ks_statistic_metric',
    'wasserstein_distance_metric',
    'parse_grade',
    'normalize_grade',
]

# LightEval metrics not available in current version
# if LIGHTEVAL_METRICS_AVAILABLE:
#     __all__.extend(['loglikelihood_acc', 'mcc'])

# Add model classes if available
if MODELS_AVAILABLE:
    __all__.extend([
        'ModelConfig',
        'LightEvalModelFactory',
        'create_model_config'
    ])

# Add LightEval benchmark if available
if LIGHTEVAL_BENCHMARK_AVAILABLE:
    __all__.extend([
        'LightEvalBenchmark',
        'create_lighteval_benchmark',
        'run_lighteval_evaluation'
    ])

# Run management (still needed for run tracking)
from .run_manager import RunManager, RunInfo

# Legacy exports removed - only LightEval is supported now
# RunManager and RunInfo are still available for run tracking
__all__.extend([
    'RunManager',
    'RunInfo',
])
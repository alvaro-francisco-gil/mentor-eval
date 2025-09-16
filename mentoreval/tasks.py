"""
MentorEval custom tasks for LightEval.

This file follows the exact format from the LightEval documentation
for custom task registration.
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

# Import our custom metrics and prompt function
from .metrics import (
    exact_grade_match_metric,
    grade_mae_metric,
    grade_rmse_metric,
    pearson_correlation_metric,
    spearman_correlation_metric,
    ks_statistic_metric,
    wasserstein_distance_metric
)
from .prompts import mentor_eval_prompt_fn

# Define a filter function that can be pickled for multiprocessing
def asap_exercise_set_1_filter(line):
    """Filter function for ASAP exercise set 1."""
    return (
        line.get('dataset') == 'asap' and 
        line.get('exercise_set') == 1
    )

# Create a simple task for testing
mentor_eval_test_task = LightevalTaskConfig(
    name="mentor_eval:asap_exercise_set_1",
    prompt_function=mentor_eval_prompt_fn,
    hf_repo="alvaro-francisco-gil/mentor-eval",
    hf_subset="default",
    metrics=[
        exact_grade_match_metric,
        grade_mae_metric,
        grade_rmse_metric,
        # Note: Corpus-level metrics (pearson, spearman, ks, wasserstein)
        # are not compatible with sample-level evaluation in LightEval
        # They need to be computed separately after evaluation
    ],
    hf_filter=asap_exercise_set_1_filter,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=10,
    stop_sequence=["\n", ".", " "],
)

# Export the tasks table
TASKS_TABLE = [mentor_eval_test_task]

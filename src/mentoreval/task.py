"""
Standalone MentorEval tasks for LightEval.
This file can be imported directly by LightEval without package dependencies.
"""

import logging
from enum import Enum  
from typing import List, Tuple, Dict, Any
import numpy as np
import random
from aenum import extend_enum

# LightEval imports for compatibility
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping

logger = logging.getLogger(__name__)

# Import the necessary functions from the mentoreval package
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import directly from the modules to avoid circular imports
from mentoreval.metrics import (
    exact_grade_match_metric,
    grade_mae_metric,
    grade_rmse_metric,
    pearson_correlation_metric,
    spearman_correlation_metric,
    ks_statistic_metric,
    wasserstein_distance_metric
)

from mentoreval.prompts import mentor_eval_prompt_fn

# Global variables to store settings
_force_explanation_global = False
_show_isced_level_global = False

def set_force_explanation(force_explanation: bool):
    """Set the global force_explanation setting."""
    global _force_explanation_global
    _force_explanation_global = force_explanation

def set_show_isced_level(show_isced_level: bool):
    """Set the global show_isced_level setting."""
    global _show_isced_level_global
    _show_isced_level_global = show_isced_level

def mentor_eval_prompt_fn_wrapper(line, task_name: str = None, **kwargs):
    """Wrapper function that uses the global settings."""
    return mentor_eval_prompt_fn(line, task_name, force_explanation=_force_explanation_global, show_isced_level=_show_isced_level_global, **kwargs)

# Note: mentoreval_metrics is already registered when mentoreval package is imported
# No need to create or extend the enum again

def create_mentoreval_task(dataset_name: str, generation_size: int = 500, num_fewshots: int = 1, force_explanation: bool = False) -> LightevalTaskConfig:
    """
    Create a MentorEval task for a specific dataset following LightEval patterns.
    
    This follows the same pattern as BigBench, BLIMP, and other multi-dataset tasks
    in the LightEval codebase.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'asap', 'asap2')
        generation_size: Maximum number of tokens to generate (default: 500)
        num_fewshots: Number of few-shot examples to include (default: 1)
    """
    return LightevalTaskConfig(
        name=f"mentoreval_{dataset_name}",
        suite=["custom"],
        prompt_function=mentor_eval_prompt_fn_wrapper,
        hf_repo="alvaro-francisco-gil/mentor-eval",
        hf_subset="default",
        hf_filter=lambda line, dataset=dataset_name: line.get('dataset') == dataset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random",  # Use random for now, but filter at dataset level
        num_fewshots=num_fewshots,  # This was missing!
        generation_size=generation_size,
        stop_sequence=["\n", ".", " "],
        metrics=[
            exact_grade_match_metric,
            grade_mae_metric,
            grade_rmse_metric,
            pearson_correlation_metric,
            spearman_correlation_metric,
            ks_statistic_metric,
            wasserstein_distance_metric,
        ],
        version=0,
    )

def create_mentoreval_exercise_task(dataset_name: str, exercise_set: int, generation_size: int = 500, num_fewshots: int = 1, force_explanation: bool = False) -> LightevalTaskConfig:
    """
    Create a MentorEval task for a specific dataset and exercise set.
    This ensures that few-shot examples come from the same exercise set.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'asap', 'asap2')
        exercise_set: Exercise set number (e.g., 1, 2, 3)
        generation_size: Maximum number of tokens to generate (default: 500)
        num_fewshots: Number of few-shot examples to include (default: 1)
    """
    return LightevalTaskConfig(
        name=f"mentoreval_{dataset_name}_ex{exercise_set}",  # Base task name - LightEval adds custom| and |0
        suite=["custom"],
        prompt_function=mentor_eval_prompt_fn_wrapper,
        hf_repo="alvaro-francisco-gil/mentor-eval",
        hf_subset="default",
        hf_filter=lambda line, dataset=dataset_name, ex_set=exercise_set: (
            line.get('dataset') == dataset and line.get('exercise_set') == ex_set
        ),
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random",  # Now random selection will be exercise-aware due to filtering
        num_fewshots=num_fewshots,  # This was missing!
        generation_size=generation_size,
        stop_sequence=["\n", ".", " "],
        metrics=[
            exact_grade_match_metric,
            grade_mae_metric,
            grade_rmse_metric,
            pearson_correlation_metric,
            spearman_correlation_metric,
            ks_statistic_metric,
            wasserstein_distance_metric,
        ],
        version=0,
    )

# Create all dataset tasks following LightEval patterns
# Option 1: Dataset-level tasks (few-shot examples from same dataset)
MENTOR_EVAL_DATASET_TASKS = [
    create_mentoreval_task("asap"),
    create_mentoreval_task("asap2"),
    create_mentoreval_task("mohler"),
    create_mentoreval_task("arasag"),
    create_mentoreval_task("ellipse"),
    create_mentoreval_task("ptasag2018"),
]

# Option 2: Exercise-set level tasks (few-shot examples from same dataset AND exercise_set)
# This provides the exercise-aware few-shot selection you wanted
MENTOR_EVAL_EXERCISE_TASKS = []

# ASAP dataset has 8 exercises (exercise sets 1-8)
for i in range(1, 9):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("asap", i))

# ASAP2 dataset has 7 exercises (exercise sets 1-7)
for i in range(1, 8):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("asap2", i))

# Mohler dataset has 81 exercises (exercise sets 1-81)
for i in range(1, 82):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("mohler", i))

# Ellipse dataset has 44 exercises (exercise sets 1-44)
for i in range(1, 45):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("ellipse", i))

# PTASAG2018 dataset has 15 exercises (exercise sets 1-15)
for i in range(1, 16):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("ptasag2018", i))

# ARASAG dataset has 48 exercises (exercise sets 1-48)
for i in range(1, 49):
    MENTOR_EVAL_EXERCISE_TASKS.append(create_mentoreval_exercise_task("arasag", i))

# Choose which tasks to use:
# Use exercise-set level tasks for exercise-aware few-shot selection
MENTOR_EVAL_TASKS = MENTOR_EVAL_EXERCISE_TASKS

# Uncomment the line below to use dataset-level tasks for simpler evaluation
# MENTOR_EVAL_TASKS = MENTOR_EVAL_DATASET_TASKS

# Task Groups for organized execution
TASKS_GROUPS = {
    # Individual dataset groups
    "mentoreval_asap": ",".join([f"custom|mentoreval_asap_ex{i}|0" for i in range(1, 9)]),
    "mentoreval_asap2": ",".join([f"custom|mentoreval_asap2_ex{i}|0" for i in range(1, 8)]),
    "mentoreval_mohler": ",".join([f"custom|mentoreval_mohler_ex{i}|0" for i in range(1, 82)]),
    "mentoreval_ellipse": ",".join([f"custom|mentoreval_ellipse_ex{i}|0" for i in range(1, 45)]),
    "mentoreval_ptasag2018": ",".join([f"custom|mentoreval_ptasag2018_ex{i}|0" for i in range(1, 16)]),
    "mentoreval_arasag": ",".join([f"custom|mentoreval_arasag_ex{i}|0" for i in range(1, 49)]),
    
    # Exercise type groups
    "mentoreval_essay_writing": ",".join([
        ",".join([f"custom|mentoreval_asap_ex{i}|0" for i in range(1, 9)]),
        ",".join([f"custom|mentoreval_asap2_ex{i}|0" for i in range(1, 8)]),
        ",".join([f"custom|mentoreval_ellipse_ex{i}|0" for i in range(1, 45)])
    ]),
    "mentoreval_short_answer": ",".join([
        ",".join([f"custom|mentoreval_mohler_ex{i}|0" for i in range(1, 82)]),
        ",".join([f"custom|mentoreval_ptasag2018_ex{i}|0" for i in range(1, 16)]),
        ",".join([f"custom|mentoreval_arasag_ex{i}|0" for i in range(1, 49)])
    ]),
    
    # All tasks
    "mentoreval": ",".join([
        ",".join([f"custom|mentoreval_asap_ex{i}|0" for i in range(1, 9)]),
        ",".join([f"custom|mentoreval_asap2_ex{i}|0" for i in range(1, 8)]),
        ",".join([f"custom|mentoreval_mohler_ex{i}|0" for i in range(1, 82)]),
        ",".join([f"custom|mentoreval_ellipse_ex{i}|0" for i in range(1, 45)]),
        ",".join([f"custom|mentoreval_ptasag2018_ex{i}|0" for i in range(1, 16)]),
        ",".join([f"custom|mentoreval_arasag_ex{i}|0" for i in range(1, 49)])
    ])
}

# Export for LightEval registry system following established patterns
TASKS_TABLE = MENTOR_EVAL_TASKS

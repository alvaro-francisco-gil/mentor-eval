from enum import Enum  
from typing import List, Tuple, Dict, Any
import numpy as np
import random

# LightEval imports for compatibility
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

# Import metrics from our simplified metrics module
from .metrics import (
    exact_grade_match_metric,
    grade_mae_metric,
    grade_rmse_metric,
    pearson_correlation_metric,
    spearman_correlation_metric,
    ks_statistic_metric,
    wasserstein_distance_metric
)

# Import prompt functions
from .prompts import mentor_eval_prompt_fn

class MentorEvalDataset(Enum):  
    ASAP = "asap"  
    ASAP2 = "asap2"  
    MOHLER = "mohler"
    ARASAG = "arasag"
    ELLIPSE = "ellipse"
    PTASAG2018 = "ptasag2018"  
  
class MentorEvalTask:  
    def __init__(self, dataset: MentorEvalDataset, exercise_set: int):  
        self.dataset = dataset  
        self.exercise_set = exercise_set  
        self.value = f"{dataset.value}_exercise_set_{exercise_set}"  
      
    def __str__(self):  
        return self.value  
      
    def __repr__(self):  
        return f"MentorEvalTask({self.dataset.value}, {self.exercise_set})"  
  
class MentorEvalTasks:  
    @staticmethod  
    def get_all_asap_tasks() -> List[MentorEvalTask]:  
        return [MentorEvalTask(MentorEvalDataset.ASAP, i) for i in range(1, 9)]  
      
    @staticmethod  
    def get_all_asap2_tasks() -> List[MentorEvalTask]:  
        return [MentorEvalTask(MentorEvalDataset.ASAP2, i) for i in range(1, 8)]  
    
    @staticmethod
    def get_all_mohler_tasks() -> List[MentorEvalTask]:
        return [MentorEvalTask(MentorEvalDataset.MOHLER, i) for i in range(1, 82)]  # MOHLER has 81 exercises
    
    @staticmethod
    def get_all_arasag_tasks() -> List[MentorEvalTask]:
        return [MentorEvalTask(MentorEvalDataset.ARASAG, i) for i in range(1, 49)]  # ARASAG has 48 exercises
    
    @staticmethod
    def get_all_ellipse_tasks() -> List[MentorEvalTask]:
        return [MentorEvalTask(MentorEvalDataset.ELLIPSE, i) for i in range(1, 45)]  # ELLIPSE has 44 exercises
    
    @staticmethod
    def get_all_ptasag2018_tasks() -> List[MentorEvalTask]:
        return [MentorEvalTask(MentorEvalDataset.PTASAG2018, i) for i in range(1, 16)]  # PTASAG2018 has 15 exercises
      
    @staticmethod  
    def get_all_tasks() -> List[MentorEvalTask]:  
        # Combine all supported datasets' tasks
        return (MentorEvalTasks.get_all_asap_tasks() + 
                MentorEvalTasks.get_all_asap2_tasks() +
                MentorEvalTasks.get_all_mohler_tasks() +
                MentorEvalTasks.get_all_arasag_tasks() +
                MentorEvalTasks.get_all_ellipse_tasks() +
                MentorEvalTasks.get_all_ptasag2018_tasks())


def custom_few_shot_select(examples: List[Dict], num_samples: int, current_example: Dict) -> List[Dict]:
    """
    Custom few-shot selection that filters by dataset and exercise_set.
    
    This ensures that few-shot examples come from the same dataset and exercise set
    as the current test example, maintaining contextual relevance.
    
    Args:
        examples: List of available training examples
        num_samples: Number of few-shot examples to select
        current_example: The current test example being evaluated
        
    Returns:
        List of selected few-shot examples
    """
    current_dataset = current_example.get('dataset')
    current_exercise_set = current_example.get('exercise_set')
    
    # Filter examples by same dataset and exercise_set
    filtered_examples = [
        ex for ex in examples
        if ex.get('dataset') == current_dataset and ex.get('exercise_set') == current_exercise_set
    ]
    
    # If we don't have enough examples from the same dataset/exercise_set,
    # fall back to just the same dataset
    if len(filtered_examples) < num_samples:
        dataset_only_examples = [
            ex for ex in examples
            if ex.get('dataset') == current_dataset
        ]
        if len(dataset_only_examples) >= num_samples:
            filtered_examples = dataset_only_examples
    
    # Randomly sample from filtered examples
    if len(filtered_examples) >= num_samples:
        return random.sample(filtered_examples, num_samples)
    else:
        # Return all available filtered examples if we don't have enough
        return filtered_examples


def create_mentor_eval_task_config(mentor_task: MentorEvalTask) -> LightevalTaskConfig:
    """
    Create a LightevalTaskConfig for a specific dataset/exercise combination.
    
    Args:
        mentor_task: MentorEvalTask instance with dataset and exercise_set
        
    Returns:
        LightevalTaskConfig configured for the specific task
    """
    # Capture the values to avoid closure issues
    dataset_value = mentor_task.dataset.value
    exercise_set_value = mentor_task.exercise_set
    
    return LightevalTaskConfig(
        name=f"mentor_eval:{mentor_task.value}",  # e.g., "mentor_eval:asap_exercise_set_1"
        prompt_function=mentor_eval_prompt_fn,
        hf_repo="alvaro-francisco-gil/mentor-eval",  # Your uploaded dataset
        hf_subset="default",
        metrics=[
            # Sample-level metrics
            exact_grade_match_metric,
            grade_mae_metric,
            grade_rmse_metric,
            # Note: Corpus-level metrics are not compatible with sample-level evaluation
            # They need to be computed separately after evaluation
        ],
        hf_filter=lambda line: (
            line.get('dataset') == dataset_value and 
            line.get('exercise_set') == exercise_set_value
        ),
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",  # Use random sampling for now
        generation_size=10,  # Short generation for grade output
        stop_sequence=["\n", ".", " "],  # Stop sequences for grade generation
    )

# Create separate task configurations for each dataset/exercise combination
mentor_eval_tasks = []
for mentor_task in MentorEvalTasks.get_all_tasks():
    task_config = create_mentor_eval_task_config(mentor_task)
    mentor_eval_tasks.append(task_config)

# Required export for LightEval registry system
TASKS_TABLE = mentor_eval_tasks  
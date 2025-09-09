"""
Simplified configuration for MentorEval benchmark.

This module provides a minimal configuration system focused on the core
benchmark scenarios for fair LLM comparison.
"""

from typing import List, Optional
from enum import Enum
from dataclasses import dataclass


class PromptType(str, Enum):
    """Types of prompts for evaluation."""
    WITH_EXPLANATION = "with_explanation"  # Forces brief explanation for each grade
    GRADE_ONLY = "grade_only"  # Just expects the actual grade


class BenchmarkMode(str, Enum):
    """Standard benchmark modes."""
    MENTOREVAL = "mentoreval"  # Full benchmark: all training data, with rubric, with explanation
    MENTOREVAL_TEST = "mentoreval-test"  # Test benchmark: one example per set, with rubric, with explanation


@dataclass
class MentorEvalConfig:
    """Simplified configuration for MentorEval benchmark."""
    
    # Core benchmark parameters
    mode: BenchmarkMode = BenchmarkMode.MENTOREVAL
    use_few_shot: bool = True  # If True, use all training data; if False, use zero-shot
    include_rubric: bool = True  # Whether to include rubric in prompt
    prompt_type: PromptType = PromptType.WITH_EXPLANATION  # Type of prompt to use
    n_test_samples: Optional[int] = None  # Number of test samples to use (None = all)
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    model_provider: str = "openai"  # Provider: "openai", "anthropic", "xai"
    
    # Output configuration
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode == BenchmarkMode.MENTOREVAL:
            # Full benchmark should use all training data, with rubric, with explanation
            self.use_few_shot = True
            self.include_rubric = True
            self.prompt_type = PromptType.WITH_EXPLANATION
        elif self.mode == BenchmarkMode.MENTOREVAL_TEST:
            # Test benchmark should use one example per set, with rubric, with explanation
            self.use_few_shot = True  # But will be limited to one example per set
            self.include_rubric = True
            self.prompt_type = PromptType.WITH_EXPLANATION
    
    @classmethod
    def mentoreval_full(cls, model_name: str = "gpt-4o-mini", model_provider: str = "openai", n_test_samples: Optional[int] = None) -> 'MentorEvalConfig':
        """Create configuration for full MentorEval benchmark."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL,
            use_few_shot=True,
            include_rubric=True,
            prompt_type=PromptType.WITH_EXPLANATION,
            n_test_samples=n_test_samples,
            model_name=model_name,
            model_provider=model_provider
        )
    
    @classmethod
    def mentoreval_test(cls, model_name: str = "gpt-4o-mini", model_provider: str = "openai", n_test_samples: Optional[int] = None) -> 'MentorEvalConfig':
        """Create configuration for MentorEval test benchmark."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL_TEST,
            use_few_shot=True,  # Will be limited to one example per set
            include_rubric=True,
            prompt_type=PromptType.WITH_EXPLANATION,
            n_test_samples=n_test_samples,
            model_name=model_name,
            model_provider=model_provider
        )
    
    @classmethod
    def custom(cls, 
               use_few_shot: bool = False,
               include_rubric: bool = True,
               prompt_type: PromptType = PromptType.GRADE_ONLY,
               n_test_samples: Optional[int] = None,
               model_name: str = "gpt-4o-mini",
               model_provider: str = "openai") -> 'MentorEvalConfig':
        """Create custom configuration for experimentation."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL,  # Use as base mode
            use_few_shot=use_few_shot,
            include_rubric=include_rubric,
            prompt_type=prompt_type,
            n_test_samples=n_test_samples,
            model_name=model_name,
            model_provider=model_provider
        )
    
    def get_description(self) -> str:
        """Get human-readable description of the configuration."""
        few_shot_desc = "few-shot (all training data)" if self.use_few_shot else "zero-shot"
        rubric_desc = "with rubric" if self.include_rubric else "without rubric"
        prompt_desc = "with explanations" if self.prompt_type == PromptType.WITH_EXPLANATION else "grade only"
        samples_desc = f" (first {self.n_test_samples} samples)" if self.n_test_samples else ""
        model_desc = f"using {self.model_provider}/{self.model_name}"
        
        return f"{self.mode.value}: {few_shot_desc}, {rubric_desc}, {prompt_desc}{samples_desc}, {model_desc}"

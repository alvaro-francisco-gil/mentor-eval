"""
Simplified configuration for MentorEval benchmark.

This module provides a minimal configuration system focused on the core
benchmark scenarios for fair LLM comparison.
"""

from typing import List, Optional
from enum import Enum
from dataclasses import dataclass, field


class PromptType(str, Enum):
    """Types of prompts for evaluation."""
    WITH_EXPLANATION = "with_explanation"  # Forces brief explanation for each grade
    GRADE_ONLY = "grade_only"  # Just expects the actual grade


class BenchmarkMode(str, Enum):
    """Standard benchmark modes."""
    MENTOREVAL = "mentoreval"  # Full benchmark: all training data, with rubric, with explanation
    MENTOREVAL_TEST = "mentoreval-test"  # Test benchmark: one example per set, with rubric, with explanation


@dataclass
class AsyncConfig:
    """Configuration for async processing."""
    run_async: bool = True
    max_concurrent: int = 20
    throttle_value: float = 0.0  # Delay between operations in seconds
    
    def __post_init__(self):
        if self.max_concurrent < 1:
            raise ValueError("'max_concurrent' must be at least 1")
        if self.throttle_value < 0:
            raise ValueError("'throttle_value' must be at least 0")


@dataclass
class MentorEvalConfig:
    """Simplified configuration for MentorEval benchmark."""
    
    # Core benchmark parameters
    mode: BenchmarkMode = BenchmarkMode.MENTOREVAL
    use_few_shot: bool = True  # If True, use all training data; if False, use zero-shot
    include_rubric: bool = True  # Whether to include rubric in prompt
    prompt_type: PromptType = PromptType.WITH_EXPLANATION  # Type of prompt to use
    n_test_samples: Optional[int] = None  # Number of test samples to use per exercise set (None = all)
    test_percentage: Optional[float] = None  # Percentage of test samples to use per exercise set (0.0-1.0)
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    model_provider: str = "openai"  # Provider: "openai", "anthropic", "xai"
    
    # Async configuration
    async_config: AsyncConfig = field(default_factory=AsyncConfig)
    
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
        
        # Validate test_percentage
        if self.test_percentage is not None:
            if not (0.0 <= self.test_percentage <= 1.0):
                raise ValueError("test_percentage must be between 0.0 and 1.0")
        
        # Validate that both n_test_samples and test_percentage are not set simultaneously
        if self.n_test_samples is not None and self.test_percentage is not None:
            raise ValueError("Cannot set both n_test_samples and test_percentage simultaneously")
    
    @classmethod
    def mentoreval_full(cls, model_name: str = "gpt-4o-mini", model_provider: str = "openai", n_test_samples: Optional[int] = None, test_percentage: Optional[float] = None, async_config: Optional[AsyncConfig] = None) -> 'MentorEvalConfig':
        """Create configuration for full MentorEval benchmark."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL,
            use_few_shot=True,
            include_rubric=True,
            prompt_type=PromptType.WITH_EXPLANATION,
            n_test_samples=n_test_samples,
            test_percentage=test_percentage,
            model_name=model_name,
            model_provider=model_provider,
            async_config=async_config or AsyncConfig()
        )
    
    @classmethod
    def mentoreval_test(cls, model_name: str = "gpt-4o-mini", model_provider: str = "openai", n_test_samples: Optional[int] = None, test_percentage: Optional[float] = None, async_config: Optional[AsyncConfig] = None) -> 'MentorEvalConfig':
        """Create configuration for MentorEval test benchmark."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL_TEST,
            use_few_shot=True,  # Will be limited to one example per set
            include_rubric=True,
            prompt_type=PromptType.WITH_EXPLANATION,
            n_test_samples=n_test_samples,
            test_percentage=test_percentage,
            model_name=model_name,
            model_provider=model_provider,
            async_config=async_config or AsyncConfig()
        )
    
    @classmethod
    def custom(cls, 
               use_few_shot: bool = False,
               include_rubric: bool = True,
               prompt_type: PromptType = PromptType.GRADE_ONLY,
               n_test_samples: Optional[int] = None,
               test_percentage: Optional[float] = None,
               model_name: str = "gpt-4o-mini",
               model_provider: str = "openai",
               async_config: Optional[AsyncConfig] = None) -> 'MentorEvalConfig':
        """Create custom configuration for experimentation."""
        return cls(
            mode=BenchmarkMode.MENTOREVAL,  # Use as base mode
            use_few_shot=use_few_shot,
            include_rubric=include_rubric,
            prompt_type=prompt_type,
            n_test_samples=n_test_samples,
            test_percentage=test_percentage,
            model_name=model_name,
            model_provider=model_provider,
            async_config=async_config or AsyncConfig()
        )
    
    def get_description(self) -> str:
        """Get human-readable description of the configuration."""
        few_shot_desc = "few-shot (all training data)" if self.use_few_shot else "zero-shot"
        rubric_desc = "with rubric" if self.include_rubric else "without rubric"
        prompt_desc = "with explanations" if self.prompt_type == PromptType.WITH_EXPLANATION else "grade only"
        
        # Handle sample limiting
        if self.n_test_samples:
            samples_desc = f" (first {self.n_test_samples} samples per exercise set)"
        elif self.test_percentage:
            samples_desc = f" ({self.test_percentage*100:.1f}% of samples per exercise set)"
        else:
            samples_desc = ""
        
        model_desc = f"using {self.model_provider}/{self.model_name}"
        async_desc = f"async (max {self.async_config.max_concurrent} concurrent)" if self.async_config.run_async else "sync"
        
        return f"{self.mode.value}: {few_shot_desc}, {rubric_desc}, {prompt_desc}{samples_desc}, {model_desc}, {async_desc}"

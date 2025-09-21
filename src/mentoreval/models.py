"""
LightEval-compatible model factory for MentorEval supporting multiple backends.

This module provides a unified interface for different model backends including
LiteLLM (for API-based models), VLLM (for local GPU inference), and Accelerate
(for HuggingFace Transformers models). It maintains compatibility with .env file
loading for API keys.
"""

import os
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field

import lighteval

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_LOADED = True
except ImportError:
    # python-dotenv not installed, continue without it
    DOTENV_LOADED = False
    pass

# Check if .env file exists and warn if not found
import os
if not os.path.exists('.env'):
    print("⚠️  Warning: .env file not found. API keys should be set in environment variables or .env file.")

# LiteLLM handles all providers dynamically - no enums needed!


@dataclass
class ModelConfig:
    """
    Simple model configuration for LiteLLM.
    
    LiteLLM handles provider detection automatically based on model name.
    """
    model_name: str
    api_key: Optional[str] = None
    use_chat_template: bool = True
    # Additional kwargs for the model
    kwargs: Dict[str, Any] = field(default_factory=dict)


class LightEvalModelFactory:
    """
    Simple factory for creating LiteLLM model configurations.
    
    LiteLLM handles provider detection automatically based on model name.
    """
    
    @staticmethod
    def create_litellm_config(
        model_name: str,
        api_key: Optional[str] = None,
        use_chat_template: bool = True,
        **kwargs
    ) -> ModelConfig:
        """
        Create LiteLLM configuration for any model.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "grok-3-mini")
            api_key: API key (optional, will try env vars automatically)
            use_chat_template: Whether to use chat templates
            **kwargs: Additional LiteLLM parameters
            
        Returns:
            ModelConfig instance
        """
        # LiteLLM automatically detects provider from model name
        # and loads API keys from environment variables
        return ModelConfig(
            model_name=model_name,
            api_key=api_key,
            use_chat_template=use_chat_template,
            kwargs=kwargs
        )
    
    
    @staticmethod
    def validate_model(model_name: str) -> bool:
        """
        Dynamically validate if a model is available and working.
        
        This method actually tests the model by making a small API call,
        which is more reliable than hardcoded lists that can become outdated.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is available and working, False otherwise
        """
        try:
            import litellm
            # Make a minimal test call
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10  # Short timeout for validation
            )
            return True
        except Exception:
            return False


def create_model_config(
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs
    ) -> ModelConfig:
    """
    Simple function to create a LiteLLM model configuration.
    
    Args:
        model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "grok-3-mini")
        api_key: API key (optional, LiteLLM will try env vars automatically)
        **kwargs: Additional LiteLLM parameters
        
    Returns:
        ModelConfig instance
    """
    return LightEvalModelFactory.create_litellm_config(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )


# Export the main classes and functions
__all__ = [
    'ModelConfig',
    'LightEvalModelFactory',
    'create_model_config',
]
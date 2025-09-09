"""
Model factory for MentorEval supporting multiple LLM providers.

This module provides a unified interface for different LLM providers including
OpenAI, Anthropic Claude, and custom implementations for Grok and other providers.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
from deepeval.models import DeepEvalBaseLLM, GPTModel, AnthropicModel

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Import DeepEval's native model classes
try:
    from deepeval.models import GrokModel, AnthropicModel, GPTModel, LiteLLMModel
    DEEPEVAL_MODELS_AVAILABLE = True
except ImportError:
    DEEPEVAL_MODELS_AVAILABLE = False


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    LITELLM = "litellm"
    CUSTOM = "custom"

class ModelFactory:
    """
    Factory class for creating LLM models from different providers.
    
    This provides a unified interface for creating models from various providers
    while maintaining compatibility with DeepEval's model interface.
    """
    
    @staticmethod
    def create_model(
        provider: ModelProvider,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> DeepEvalBaseLLM:
        """
        Create a model instance from the specified provider.
        
        Args:
            provider: The LLM provider to use
            model_name: Name/version of the model
            api_key: API key for the provider (optional, will try env vars)
            **kwargs: Additional model-specific parameters
            
        Returns:
            DeepEvalBaseLLM instance
            
        Raises:
            ValueError: If provider is not supported or required parameters are missing
        """
        if provider == ModelProvider.OPENAI:
            return ModelFactory._create_openai_model(model_name, api_key, **kwargs)
        elif provider == ModelProvider.ANTHROPIC:
            return ModelFactory._create_anthropic_model(model_name, api_key, **kwargs)
        elif provider == ModelProvider.XAI:
            return ModelFactory._create_xai_model(model_name, api_key, **kwargs)
        elif provider == ModelProvider.LITELLM:
            return ModelFactory._create_litellm_model(model_name, api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _create_openai_model(model_name: str, api_key: Optional[str] = None, **kwargs) -> GPTModel:
        """Create OpenAI GPT model using DeepEval's native GPTModel."""
        if not DEEPEVAL_MODELS_AVAILABLE:
            raise ImportError("DeepEval models not available. Make sure deepeval is installed.")
        
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        return GPTModel(model=model_name, **kwargs)
    
    @staticmethod
    def _create_anthropic_model(model_name: str, api_key: Optional[str] = None, **kwargs) -> AnthropicModel:
        """Create Anthropic Claude model using DeepEval's native AnthropicModel."""
        if not DEEPEVAL_MODELS_AVAILABLE:
            raise ImportError("DeepEval models not available. Make sure deepeval is installed.")
        
        if not api_key:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        return AnthropicModel(model=model_name, **kwargs)
    
    @staticmethod
    def _create_xai_model(model_name: str, api_key: Optional[str] = None, **kwargs) -> GrokModel:
        """Create XAI/Grok model using DeepEval's native GrokModel."""
        if not DEEPEVAL_MODELS_AVAILABLE:
            raise ImportError("DeepEval models not available. Make sure deepeval is installed.")
        
        if not api_key:
            api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI API key is required. Set XAI_API_KEY environment variable or pass api_key parameter.")
        
        return GrokModel(model=model_name, api_key=api_key, **kwargs)
    
    @staticmethod
    def _create_litellm_model(model_name: str, api_key: Optional[str] = None, **kwargs) -> LiteLLMModel:
        """Create LiteLLM model for maximum flexibility."""
        if not DEEPEVAL_MODELS_AVAILABLE:
            raise ImportError("DeepEval models not available. Make sure deepeval is installed.")
        
        # LiteLLM can use various API keys depending on the model
        # For Grok models via LiteLLM, we still need XAI_API_KEY
        if model_name.startswith("grok") and not api_key:
            api_key = os.getenv('XAI_API_KEY')
        
        return LiteLLMModel(model=model_name, api_key=api_key, **kwargs)
    
    @staticmethod
    def get_supported_models(provider: ModelProvider) -> Dict[str, str]:
        """
        Get list of supported models for a provider.
        
        Args:
            provider: The LLM provider
            
        Returns:
            Dictionary mapping model names to descriptions
        """
        if provider == ModelProvider.OPENAI:
            return {
                "gpt-4o": "GPT-4 Omni (latest)",
                "gpt-4o-mini": "GPT-4 Omni Mini (faster, cheaper)",
                "gpt-4-turbo": "GPT-4 Turbo",
                "gpt-3.5-turbo": "GPT-3.5 Turbo"
            }
        elif provider == ModelProvider.ANTHROPIC:
            return {
                "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (latest)",
                "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (faster)",
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku"
            }
        elif provider == ModelProvider.XAI:
            return {
                "grok-4-0709": "Grok 4 (XAI)",
                "grok-3": "Grok 3 (XAI)",
                "grok-3-mini": "Grok 3 Mini (XAI)",
                "grok-3-fast": "Grok 3 Fast (XAI)",
                "grok-3-mini-fast": "Grok 3 Mini Fast (XAI)",
                "grok-2-vision-1212": "Grok 2 Vision (XAI)"
            }
        elif provider == ModelProvider.LITELLM:
            return {
                "grok-3-mini": "Grok 3 Mini (via LiteLLM)",
                "grok-3": "Grok 3 (via LiteLLM)",
                "gpt-4o": "GPT-4 Omni (via LiteLLM)",
                "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (via LiteLLM)",
                "gemini/gemini-pro": "Gemini Pro (via LiteLLM)",
                "ollama/llama3": "Llama 3 (via LiteLLM/Ollama)"
            }
        else:
            return {}


def create_model_from_config(provider: str, model_name: str, **kwargs) -> DeepEvalBaseLLM:
    """
    Convenience function to create a model from string provider name.
    
    Args:
        provider: Provider name as string ("openai", "anthropic", "xai", "litellm")
        model_name: Model name
        **kwargs: Additional parameters
        
    Returns:
        DeepEvalBaseLLM instance
    """
    try:
        provider_enum = ModelProvider(provider.lower())
    except ValueError:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {[p.value for p in ModelProvider]}")
    
    return ModelFactory.create_model(provider_enum, model_name, **kwargs)

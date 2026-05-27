from typing import Dict, Type
import logging

from .llm_base import BaseLLMClient
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.qwen_adapter import QwenAdapter
from .adapters.gemini_adapter import GeminiAdapter
# from .adapters.qwen_local_adapter import QwenLocalAdapter
from .adapters.claude_adapter import ClaudeAdapter
# from .adapters.molmo_local_adapter import MolmoLocalAdapter
# Import additional adapters here as needed

logger = logging.getLogger(__name__)

class LLMClientFactory:
    
    # Mapping of provider names to adapter classes
    _adapters: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAIAdapter,
        "qwen": QwenAdapter,
        # "qwen_local": QwenLocalAdapter,
        "gemini": GeminiAdapter,
        "claude": ClaudeAdapter,
        # "molmo_local": MolmoLocalAdapter,
        # Add additional adapters here
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict) -> BaseLLMClient:
        """
        Factory method to create LLM client instances based on provider.
        Args:
            provider: The name of the LLM provider (e.g., "openai", "qwen")
            config: Configuration dictionary for the LLM client
        Returns:
            An instance of BaseLLMClient corresponding to the provider
        """
        provider = provider.lower()
        
        if provider not in cls._adapters:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {list(cls._adapters.keys())}"
            )
        
        adapter_class = cls._adapters[provider]
        logger.info(f"Creating {provider} client")
        return adapter_class(config)
    
    @classmethod
    def register(cls, provider: str, adapter_class: Type[BaseLLMClient]):
        """
        Register a new LLM adapter class for a given provider.
        Args:
            provider: The name of the LLM provider
            adapter_class: The adapter class to register
        """
        cls._adapters[provider.lower()] = adapter_class
        logger.info(f"Registered new adapter: {provider}")
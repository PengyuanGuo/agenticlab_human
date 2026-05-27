import os
import yaml
from typing import Dict, Optional, Union, List, Any
from PIL import Image
import logging

from .llm_factory import LLMClientFactory
from .llm_base import BaseLLMClient

logger = logging.getLogger(__name__)

class LLMManager:
    """Singleton class to manage LLM clients and provide a unified interface."""
    
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.clients: Dict[str, BaseLLMClient] = {}
            self.config: Dict = {}
            self.default_provider: Optional[str] = None
            self.initialized = True
            logger.info("LLMManager initialized")
            
    def load_config(self, config_path: str):
        """
        Load configuration from a YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set default provider
        self.default_provider = self.config.get("default_provider", "openai")
        
        logger.info(f"Config loaded, default provider: {self.default_provider}")
    
    def create_client(
        self, 
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseLLMClient:
        """
        Create or retrieve an LLM client instance.
        """
        provider = provider or self.default_provider
        cache_key = f"{provider}:{model}" if model else provider
        
        # Check cache
        if cache_key in self.clients:
            return self.clients[cache_key]
        
        # Get configuration
        if provider not in self.config.get("providers", {}):
            raise ValueError(f"Provider {provider} not configured")
        
        config = self.config["providers"][provider].copy()
        if model:
            config["model"] = model
        
        # Create client
        client = LLMClientFactory.create(provider, config)
        
        # Cache client
        self.clients[cache_key] = client
        return client
    
    def call(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Call LLM with prompt and optional image(s)."""
        client = self.create_client(provider, model)
        return client.call(prompt, image, **kwargs)
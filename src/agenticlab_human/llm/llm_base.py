from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from PIL import Image
import logging
import os
logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """LLM base class for different providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("model")
        
        # Initialize OpenAI client from environment variable
        env_var = self.config.get("api_key_env")
        self.api_key = os.environ.get(env_var) if env_var else None
        
        
    @abstractmethod
    def call(
        self, 
        prompt: str,
        image: Optional[Union[Image.Image, str, list, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified call method for LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, image path, list of images, or other provider-specific formats)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        pass
    
    @abstractmethod
    def _build_messages(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, str, list, Any]] = None
    ) -> list:
        """
        build messages for LLM input
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, image path, list of images, or other provider-specific formats)
        Returns:
            list: Formatted messages for LLM input
        """
        pass
    
    def validate_config(self) -> bool:
        """validate required configuration parameters"""
        if not self.model_name:
            logger.error("Missing required config: model")
            return False
        # if not self.api_key:
        #     logger.error("Missing required config: api_key")
        #     return False
        return True
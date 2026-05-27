import anthropic
from typing import Dict, Optional, Union, Any
from PIL import Image
import logging
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    from vlm_robobench.llm_interface.llm_base import BaseLLMClient
    from vlm_robobench.llm_interface.llm_utils import parse_json_response, encode_image
else:
    from ..llm_base import BaseLLMClient
    from ..llm_utils import parse_json_response, encode_image


logger = logging.getLogger(__name__)

class ClaudeAdapter(BaseLLMClient):
    """Adapter for Anthropic Claude LLMs"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not self.validate_config():
            raise ValueError("Invalid configuration for ClaudeAdapter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_tokens = config.get("max_tokens", 4096)
        
        # Set up token counter CSV path
        if __name__ != "__main__":
            # Get project root (3 levels up from this file)
            self.project_root = Path(__file__).resolve().parents[3]
        else:
            self.project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        
        self.token_csv_path = self.project_root / "output" / "token_counter" / "total_token.csv"
    
    def _log_token_count(self, total_tokens: int):
        """Log total token count to CSV file"""
        try:
            self.token_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_csv_path, 'a') as f:
                f.write(f"{total_tokens}\n")
        except Exception as e:
            logger.warning(f"Failed to log token count: {e}")
        
    def call(
        self, 
        prompt: str,
        image: Optional[Union[Image.Image, list, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified call method for Claude LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, list of PIL Images, or other compatible objects)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)
        
        # Get max_tokens from kwargs or use default
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
        
        # Extract and log token count
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        logger.critical(f"total_token_count used: {total_tokens} (input: {input_tokens}, output: {output_tokens})")
        self._log_token_count(total_tokens)
        
        # Extract text content from response
        content = response.content[0].text if response.content else ""
        
        try:
            result = parse_json_response(content)
        except Exception as e:
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": content}
        
        result['token_usage'] = total_tokens
        
        return result
    
    def _build_messages(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, list, Any]] = None
    ) -> list:
        """
        Build messages for Claude API
        Args:
            prompt: Text prompt
            image: Optional PIL Image or list of images
        Returns:
            List of messages in Claude format
        """
        if image is not None:
            # Normalize to a list of images
            images = image if isinstance(image, list) else [image]
            
            content = []
            for img in images:
                if isinstance(img, Image.Image):
                    # Encode image to base64
                    base64_image = encode_image(img)
                    
                    # Determine image media type (default to PNG)
                    image_format = img.format.lower() if img.format else "png"
                    if image_format == "jpg":
                        image_format = "jpeg"
                    media_type = f"image/{image_format}"
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    })
                elif isinstance(img, dict) and "type" in img and img["type"] == "image":
                    # Directly append if already in Claude format
                    content.append(img)
            
            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })
            
            return [{
                "role": "user",
                "content": content
            }]
        else:
            return [{
                "role": "user",
                "content": prompt
            }]
            
if __name__ == "__main__":
    import json
    from PIL import Image

    # Example usage
    config = {
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 4096
    }
    
    adapter = ClaudeAdapter(config)
    
    # Test with multiple images
    image_path_2 = "data/Submodule_test_data/task_completion/17_crossword_fail_after.png"
    image_path_3 = "data/Submodule_test_data/task_completion/17_crossword_fail_before.png"
    
    if os.path.exists(image_path_2) and os.path.exists(image_path_3):
        image_2 = Image.open(image_path_2).convert("RGB")
        image_3 = Image.open(image_path_3).convert("RGB")
        
        result_multi = adapter.call(
            prompt="What is the difference between these two images?", 
            image=[image_2, image_3]
        )
        
        print("\nResult with multiple images:")
        print(json.dumps(result_multi, indent=4, ensure_ascii=False))

    # Test without image
    # prompt = "What is the capital of France? Answer in JSON format with key 'capital'."
    # result = adapter.call(prompt=prompt)
    
    # print("\nResult without image:")
    # print(json.dumps(result, indent=4, ensure_ascii=False))

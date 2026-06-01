from openai import OpenAI
from typing import Dict, Optional, Union, Any
from PIL import Image
import logging
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
    sys.path.insert(0, project_root)
    from vlm_robobench.llm_interface.llm_base import BaseLLMClient
    from vlm_robobench.llm_interface.llm_utils import parse_json_response, encode_image
else:
    from ..llm_base import BaseLLMClient
    from ..llm_utils import parse_json_response, encode_image


logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


class OpenAIAdapter(BaseLLMClient):
    """Adapter for OpenAI LLMs"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not self.validate_config():
            raise ValueError("Invalid configuration for OpenAIAdapter")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Set up token counter CSV path
        self.project_root = _get_project_root()
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
        Unified call method for OpenAI LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, list of PIL Images, or image path)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        
        # Extract and log token count
        total_tokens = response.usage.total_tokens
        logger.critical(f"total_token_count used: {total_tokens}")
        self._log_token_count(total_tokens)

        try:
            result = parse_json_response(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": response.choices[0].message.content}
        
        result['token_usage'] = total_tokens
        
        return result
    
    def _build_messages(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, list, Any]] = None
    ) -> list:
        if image is not None:
            # Normalize to a list of images
            images = image if isinstance(image, list) else [image]
            
            content = []
            for img in images:
                if isinstance(img, Image.Image):
                    base64_image = encode_image(img)
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                elif isinstance(img, str) and (img.startswith("http") or len(img) > 100): # rudimentary check for URL or base64
                     content.append({
                        "type": "image_url", 
                        "image_url": {"url": img}
                    })
            
            # Add the text prompt
            content.append({"type": "text", "text": prompt})
            
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
    import os
    import json
    from PIL import Image

    # Example usage
    config = {
        "model": "gpt-4o", # Updated to a more common model for testing
        "api_key_env": "OPENAI_API_KEY"
    }
    
    adapter = OpenAIAdapter(config)
    
    image_path = "data/data_for_test/color.png"
    image = Image.open(image_path).convert("RGB")
    prompt = "Describe the objects in the image in JSON format."
    
    # Test single image
    result = adapter.call(prompt=prompt, image=image)
    print("Single image result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    # # Test multiple images
    # image_path_2 = "data/Submodule_test_data/task_completion/17_crossword_fail_after.png"
    # image_2 = Image.open(image_path_2).convert("RGB")
    # image_path_3 = "data/Submodule_test_data/task_completion/17_crossword_fail_before.png"
    # image_3 = Image.open(image_path_3).convert("RGB")
    
    # result_multi = adapter.call(prompt="What is the difference between these two images?", image=[image_2, image_3])
    # print("\nMultiple images result:")
    # print(json.dumps(result_multi, indent=4, ensure_ascii=False))

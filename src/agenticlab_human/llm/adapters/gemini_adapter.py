from google import genai
from google.genai import types
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

class GeminiAdapter(BaseLLMClient):
    """Adapter for Gemini LLMs"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not self.validate_config():
            raise ValueError("Invalid configuration for GeminiAdapter")
        
        self.client = genai.Client(api_key=self.api_key)
        
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
        Unified call method for Gemini LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, list of PIL Images, or compatible objects like uploaded files)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            **kwargs
        )
        
        # Extract and log token count
        total_tokens = response.usage_metadata.total_token_count
        logger.critical(f"total_token_count used: {total_tokens}")
        self._log_token_count(total_tokens)
        
        # logger.info(f"Response: {response.text.strip()}")
        try:
            result = parse_json_response(response.text.strip())
        except Exception as e:
            logger.error(f"Response is not in JSON format: {response.text.strip()}")
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": response.text.strip()}
            
        if isinstance(result, list):
            logger.info(f"Original result: {result}")
            result = result[0]
            logger.warning(f"Result is a list, returning the first element: {result}")

        result['token_usage'] = total_tokens

        return result
    
    def _build_messages(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, list, Any]] = None
    ) -> list:
        # Start with the text prompt as the first part
        parts = [types.Part(text=prompt)]
        
        if image is not None:
            # Normalize to a list of images/parts
            images = image if isinstance(image, list) else [image]
            
            for img in images:
                if isinstance(img, Image.Image):
                    # Handle PIL Images by encoding them to base64
                    base64_image = encode_image(img)
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png", # encode_image uses PNG
                                data=base64_image
                            )
                        )
                    )
                elif isinstance(img, (types.Part, types.Blob)):
                    # Directly append Gemini-specific types
                    parts.append(img)
                else:
                    # Try to append other types directly (e.g. uploaded file references)
                    parts.append(img)
        
        return [types.Content(parts=parts)]
            
if __name__ == "__main__":
    import os
    import json
    from PIL import Image

    # Example usage
    config = {
        "model": "gemini-3-flash-preview",
        "api_key_env": "GEMINI_API_KEY"
    }
    
    adapter = GeminiAdapter(config)
    
    image_path = "data/data_for_test/color.png"
    image = Image.open(image_path).convert("RGB")
    prompt = "Describe the objects in the image in JSON format."
    
    # Test single image
    result = adapter.call(prompt=prompt, image=image)
    print("Single image result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    # # Test multiple images (using the same image twice for demonstration)
    # image_path_2 = "data/Submodule_test_data/task_completion/17_crossword_fail_after.png"
    # image_2 = Image.open(image_path_2).convert("RGB")
    # image_path_3 = "data/Submodule_test_data/task_completion/17_crossword_fail_before.png"
    # image_3 = Image.open(image_path_3).convert("RGB")
    # result_multi = adapter.call(prompt="What is the difference between these two images?", image=[image_2, image_3])
    # print("\nMultiple images result:")
    # print(json.dumps(result_multi, indent=4, ensure_ascii=False))
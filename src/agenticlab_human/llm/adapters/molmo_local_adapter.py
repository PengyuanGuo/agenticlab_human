import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import Dict, Optional, Any
from PIL import Image
import logging
import os
import sys

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    from vlm_robobench.llm_interface.llm_base import BaseLLMClient
    from vlm_robobench.llm_interface.llm_utils import parse_json_response
else:
    from ..llm_base import BaseLLMClient
    from ..llm_utils import parse_json_response

logger = logging.getLogger(__name__)

class MolmoLocalAdapter(BaseLLMClient):
    """Adapter for Molmo local VLM"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not self.validate_config():
            raise ValueError("Invalid configuration for MolmoLocalAdapter")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model with trust_remote_code
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        logger.info(f"Molmo model {self.model_name} loaded")
        
    def call(
        self, 
        prompt: str,
        image: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified call method for Molmo LLM
        """
        # Process inputs using Molmo's processor
        if image is not None:
            inputs = self.processor.process(
                images=[image.convert("RGB")],
                text=prompt
            )
        else:
            inputs = self.processor.process(
                images=None,
                text=prompt
            )
        
        # Move to device and add batch dimension
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Get max_tokens from config or kwargs
        max_tokens = self.config.get("max_tokens", 1000)
        stop_strings = self.config.get("stop_strings", "<|endoftext|>")
        
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_tokens, stop_strings=stop_strings),
                tokenizer=self.processor.tokenizer,
                use_cache=False
            )
        
        # Decode only the generated tokens
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        result = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        try:
            result = parse_json_response(result)
        except Exception as e:
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": result}
        return result
    
    def _build_messages(
        self, 
        prompt: str, 
        image: Optional[Image.Image] = None
    ) -> list:
        # Molmo doesn't use a chat message format like OpenAI
        # This method is required by abstract base class but not used
        pass


if __name__ == "__main__":
    import json
    from PIL import Image

    config = {
        "model": "allenai/Molmo-7B-D-0924",
        "max_tokens": 1000,
    }
    
    adapter = MolmoLocalAdapter(config)
    
    image_path = "data/data_for_test/color.png"
    image = Image.open(image_path).convert("RGB")
    prompt = "Describe the objects in the image"
    
    result = adapter.call(prompt=prompt, image=image)
    print(json.dumps(result, indent=4, ensure_ascii=False))
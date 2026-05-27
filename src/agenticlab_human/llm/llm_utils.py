import json
import base64
import io
from PIL import Image
from typing import Union, Dict, Any

def parse_json_response(content: str) -> Dict[Any, Any]:
    """
    Parse JSON response from an LLM, handling markdown code blocks and extra prose.
    
    Supports:
    - Plain JSON
    - JSON in markdown code blocks (```json...``` or ```...```)
    - JSON embedded in prose text
    """
    content = content.strip()
    
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Extract from markdown code blocks if present
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:  # Has opening and closing fence
            # Get content between first pair of fences
            candidate = parts[1].strip()
            # Remove language tag if present (e.g., "json")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    # Scan for first valid JSON object/array in the text
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(content):
        if ch in ("{", "["):
            try:
                obj, _ = decoder.raw_decode(content[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    
    raise ValueError("No JSON object or array found in LLM response.")

def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
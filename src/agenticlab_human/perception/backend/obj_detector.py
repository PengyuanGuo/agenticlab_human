from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os, json
from agenticlab_human.perception.backend.perception_backend import DetectionResult
# from lang_sam import LangSAM
# from vlm_robobench.vlm.interface_qwen_detection_local import InterfaceQwenDetectionLocal
# from vlm_robobench.vlm.interface_molmo_detection_local import InterfaceMolmoDetectionLocal
from vlm_robobench.llm_interface.llm_manager import LLMManager

class ObjectDetector:
    def __init__(self, cfg):
        self.cfg = cfg["ObjDetector"]
        self.method = self.cfg.get("method", "vlm")
        self.vlm_model = self.cfg.get("vlm_model", "openai:gpt-5-2025-08-07")
        self.prompt_path = self.cfg.get("prompt_path", "configs/prompt/detect_all_obj_prompt.txt")
        self.reflection_prompt_path = self.cfg.get("reflection_prompt_path", "configs/prompt/langsam_reflection_prompt.txt")
        self.molmo_prompt_path = self.cfg.get("molmo_prompt_path", "configs/prompt/detect_obj_molmo_prompt.txt")
        self.max_iterations = self.cfg.get("max_iterations", 5)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = os.path.join(self.cfg.get("output_dir", "output/obj_detector"), timestamp)
        os.makedirs(self.session_output_dir, exist_ok=True)
        
        self.langsam_model = LangSAM() if self.method == "langsam" else None
        self.qwen_local = None
        self.molmo_local = None
        # if self.method == "qwen_local":
        import yaml
        if self.method == "qwen_local":
            with open(self.cfg.get("qwen_local_cfg", "configs/iterative_detection_config.yaml"), 'r') as f:
                self.qwen_local = InterfaceQwenDetectionLocal(yaml.safe_load(f))
        if self.method == "molmo_local":
            with open(self.cfg.get("molmo_local_cfg", "configs/iterative_detection_config.yaml"), 'r') as f:
                self.molmo_local = InterfaceMolmoDetectionLocal(yaml.safe_load(f))
        self.llm_manager = LLMManager()
        self.llm_manager.load_config(self.cfg.get("llm_config_path", "configs/llm_interface_config.yaml"))

    def detect(self, image: Image, query: str, method: str = None) -> DetectionResult:
        method = method or self.method
        if method == "langsam":
            return self._detect_langsam_with_reflection(image, query)
        elif method == "vlm":
            return self._detect_vlm(image, query)
        elif method == "qwen_local":
            return self._detect_qwen_local(image, query)
        elif method == "molmo_local":
            return self._detect_molmo_local(image, query)
        raise ValueError(f"Unsupported detection method: {method}")

    def _detect_langsam_with_reflection(self, image: Image, query: str) -> DetectionResult:
        viz_dir = os.path.join(self.session_output_dir, f"{query.replace(' ', '_')}_reflection")
        os.makedirs(viz_dir, exist_ok=True)
        image.save(os.path.join(viz_dir, "00_original.png"))
        
        with open(self.reflection_prompt_path, 'r') as f:
            reflection_template = f.read()
        
        current_prompt, iterations, boxes, labels, scores = query, [], [], [], []
        final_center_points = []
        
        for iteration in range(1, self.max_iterations + 1):
            boxes, labels, scores = self._run_langsam(image, current_prompt)
            
            if boxes:
                self._save_visualization(image, boxes, labels, scores, 
                    os.path.join(viz_dir, f"iter{iteration:02d}_detection.png"), "detection")
            
            reflection = self._reflect_on_detection(image, query, current_prompt, boxes, labels, 
                                                   scores, iteration, reflection_template)
            iterations.append({
                "iteration": iteration, "prompt": current_prompt, "num_detections": len(boxes),
                "action": reflection.get("action", "NO_DETECTION" if not boxes else "ERROR"),
                "reasoning": reflection.get("reasoning", ""),
                "refined_prompt": reflection.get("refined_prompt", ""),
                "valid_indices": reflection.get("valid_indices", []),
                "confidence": reflection.get("confidence", 0.0),
                "center_points": reflection.get("center_points", {})
            })
            
            should_break, boxes, labels, scores, final_center_points = self._handle_reflection_action(reflection, boxes, labels, scores)
            if should_break:
                break 
            current_prompt = reflection.get("refined_prompt", current_prompt)
        
        objects = self._convert_to_objects(boxes, labels, scores, center_points=final_center_points)
        if objects:
            self._save_visualization(image, [o['bbox'] for o in objects], 
                [o['label'] for o in objects], [o['score'] for o in objects],
                os.path.join(viz_dir, "final_result.png"), "final", center_points=final_center_points)
        
        return DetectionResult(success=bool(objects), objects=objects,
            image_shape=(image.size[1], image.size[0]),
            summary={"method": "langsam_reflection", "iterations": iterations,
                    "final_prompt": current_prompt, "total_iterations": len(iterations),
                    "visualization_dir": viz_dir})
 
    def _run_langsam(self, image: Image, prompt: str) -> Tuple[List, List, List]:
        result = self.langsam_model.predict([image], [prompt])[0]
        return result['boxes'].tolist(), result['labels'], result['scores'].tolist()
    
    def _handle_reflection_action(self, reflection: Dict, boxes: List, 
                                  labels: List, scores: List) -> Tuple[bool, List, List, List, List]:
        action = reflection.get("action", "ERROR")
        llm_centers = reflection.get("center_points", {})

        def get_center_for_idx(idx):
            key = f"index_{idx}"
            if key in llm_centers:
                return llm_centers[key]
            # Fallback to bbox center if LLM didn't provide specific point
            if idx < len(boxes):
                b = boxes[idx]
                return [int((b[0]+0.95*(b[2]-b[0])/2)), int((b[1]+1.05*(b[3]-b[1])/2))]
            return None

        if action == "SUCCESS":
            valid_indices = reflection.get("valid_indices", list(range(len(boxes))))
            return (True, 
                    [boxes[i] for i in valid_indices], 
                    [labels[i] for i in valid_indices], 
                    [scores[i] for i in valid_indices],
                    [get_center_for_idx(i) for i in valid_indices])
        elif action == "MIXED_RESULTS":
            valid_indices = reflection.get("valid_indices", [])
            if valid_indices:
                return (True, 
                        [boxes[i] for i in valid_indices], 
                        [labels[i] for i in valid_indices], 
                        [scores[i] for i in valid_indices],
                        [get_center_for_idx(i) for i in valid_indices])
        elif not boxes and action == "NO_DETECTION":
            return False, boxes, labels, scores, []
        
        # Default return for retry/error cases
        return False, boxes, labels, scores, []
    
    def _convert_to_objects(self, boxes: List, labels: List, scores: List, center_points: List) -> List[Dict]:
        # Ensure center_points matches length of boxes (handle edge cases)
        if len(center_points) < len(boxes):
             center_points.extend([None] * (len(boxes) - len(center_points)))
             
        return [{
            "bbox": [int(x) for x in box],
            "label": label,
            "score": float(score),
            "center_point": center_point,
            "mask": None
        } for box, label, score, center_point in zip(boxes, labels, scores, center_points)]
    
    def _save_visualization(self, image: Image, boxes: List, labels: List, 
                           scores: List, output_path: str, viz_type: str = "detection", center_points: List = None):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
        
        color = (255, 0, 0) if viz_type in ["detection", "labeled"] else (0, 255, 0)
        show_index = viz_type == "detection"
        show_label = viz_type == "labeled"
        
        # Handle center_points being None
        if center_points is None:
            center_points = [None] * len(boxes)
        # Ensure center_points matches length of boxes
        elif len(center_points) < len(boxes):
            center_points.extend([None] * (len(boxes) - len(center_points)))
        
        for idx, (box, label, score, center_point) in enumerate(zip(boxes, labels, scores, center_points)):
            x1, y1, x2, y2 = [int(x) for x in box]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            if show_index:
                draw.text((x1, y1 - 20), f"[{idx}] {score:.2f}", fill=color)
            elif show_label:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw, y1], fill=color)
                draw.text((x1, y1 - th - 4), label, fill=(255, 255, 255), font=font)
            
            # Draw center point if it exists
            if center_point:
                cx, cy = int(center_point[0]), int(center_point[1])
                radius = 5
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 255, 0), outline=(0, 0, 0))
                
        img.save(output_path)
    
    def _call_llm(self, prompt: str, image: Image.Image) -> Dict:
        provider, model = self.vlm_model.split(":", 1) if ":" in self.vlm_model else (None, self.vlm_model)
        return self.llm_manager.call(prompt=prompt, image=image, provider=provider, model=model)
    
    def _reflect_on_detection(self, image: Image, target_object: str, current_prompt: str,
                             boxes: List, labels: List, scores: List, 
                             iteration: int, prompt_template: str) -> Dict:
        detection_info = "No objects detected" if not boxes else "\n".join([
            f"[{idx}] label='{label}', score={score:.3f}, bbox={box}"
            for idx, (box, label, score) in enumerate(zip(boxes, labels, scores))])
        
        prompt = prompt_template.format(target_object=target_object, current_prompt=current_prompt,
                                       iteration=iteration, detection_str=detection_info)
        
        annotated_image = self._create_annotated_image(image, boxes, labels, scores)
        try:
            return self._call_llm(prompt, annotated_image)
        except Exception as e:
            return {"action": "ERROR", "reasoning": str(e)}
    
    def _create_annotated_image(self, image: Image, boxes: List, labels: List, scores: List) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = [int(x) for x in box]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1, y1 - 20), f"[{idx}] {score:.2f}", fill=(255, 0, 0))
        return img

    def _detect_qwen_local(self, image: Image, query: str) -> DetectionResult:
        detection_data, raw_output = self.qwen_local.detect_objects(image, query, self.prompt_path)
        
        def _expand_bbox(bbox: List[int], image_shape: Tuple[int, int], offset: int = 0) -> List[int]:
            x1, y1, x2, y2 = bbox
            height, width = image_shape
            
            # Expand bbox
            x1 = x1 - offset # 15 for stacking, 7 for sorting
            y1 = y1 - offset
            x2 = x2 + offset
            y2 = y2 + offset
            
            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            
            return [int(x1), int(y1), int(x2), int(y2)]
        
        image_shape = (image.size[1], image.size[0])  # (height, width)
        objects = []
        for obj in detection_data.get("objects", []):
            if "bounding_box" in obj:
                bbox = obj["bounding_box"]
            elif "top_left" in obj and "bottom_right" in obj:
                bbox = [*obj["top_left"], *obj["bottom_right"]]
            else:
                continue
            
            # Expand bbox for better cropping
            expanded_bbox = _expand_bbox(bbox, image_shape)

            center_point = obj.get("center_point")
            # uncomment to use in stacking task
            # center_point = [int(center_point[0]-0.05*(bbox[2]-bbox[0])), # x axis offset
            #                  int(center_point[1]-0.23*(bbox[3]-bbox[1]))] # y axis offset
            # uncomment to use in sorting task
            # center_point = [int(center_point[0]-0.1*(bbox[2]-bbox[0])), # x axis offset
            #                  int(center_point[1]+0.23*(bbox[3]-bbox[1]))] # y axis offset

            objects.append({
                "bbox": expanded_bbox, 
                "label": obj.get("label", "unknown"), 
                "score": 1.0,
                "mask": None, 
                "center_point": center_point
            })
        return DetectionResult(success=True, objects=objects, image_shape=image_shape,
                             raw_output={"raw_text": raw_output, "parsed": detection_data})

    def _detect_molmo_local(self, image: Image, query: str) -> DetectionResult:
        
        detection_data, raw_output = self.molmo_local.detect_objects(image, query, self.molmo_prompt_path)
        
        # Get image dimensions for coordinate conversion
        width, height = image.size
        image_shape = (height, width)  # (height, width) format
        
        def _convert_percentage_to_pixels(coords, is_bbox=False):
            """Convert percentage coordinates (0-100) to pixel coordinates."""
            if is_bbox and len(coords) == 4:
                # bounding_box: [x_min, y_min, x_max, y_max]
                return [
                    int(coords[0] * width / 100),
                    int(coords[1] * height / 100),
                    int(coords[2] * width / 100),
                    int(coords[3] * height / 100)
                ]
            elif len(coords) == 2:
                # center_point: [x, y]
                return [
                    int(coords[0] * width / 100),
                    int(coords[1] * height / 100)
                ]
            return coords
        
        def _expand_bbox(bbox: List[int], offset: int = 0) -> List[int]:
            """Expand bbox and clamp to image boundaries."""
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - offset)
            y1 = max(0, y1 - offset)
            x2 = min(width - 1, x2 + offset)
            y2 = min(height - 1, y2 + offset)
            return [x1, y1, x2, y2]
        
        objects = []
        for obj in detection_data.get("objects", []):
            # Get bounding box and convert from percentage to pixels
            bbox = None
            if "bounding_box" in obj and len(obj["bounding_box"]) == 4:
                bbox = _convert_percentage_to_pixels(obj["bounding_box"], is_bbox=True)
                bbox = _expand_bbox(bbox)
                print(f"Bounding box: {bbox}")
            elif "top_left" in obj and "bottom_right" in obj:
                top_left = _convert_percentage_to_pixels(obj["top_left"])
                bottom_right = _convert_percentage_to_pixels(obj["bottom_right"])
                bbox = _expand_bbox([*top_left, *bottom_right])
            
            if bbox is None:
                continue
            
            # Get center point and convert from percentage to pixels
            center_point = None
            if "center_point" in obj:
                center_point = _convert_percentage_to_pixels(obj["center_point"])
            
            objects.append({
                "bbox": bbox,
                "label": obj.get("label", "unknown"),
                "score": 1.0,
                "mask": None,
                "center_point": center_point
            })
        
        return DetectionResult(
            success=True, 
            objects=objects, 
            image_shape=image_shape,
            raw_output={"raw_text": raw_output, "parsed": detection_data}
        )

    def _detect_vlm(self, image: Image, query: str) -> DetectionResult:
        with open(self.prompt_path, 'r') as f:
            prompt = f.read().format(object=query if query != "all" else "all objects in the scene")
        
        result = self._call_llm(prompt, image)
        
        # Check if using Gemini model
        is_gemini = "gemini" in self.vlm_model.lower()
        is_robotics = "robotics" in self.vlm_model.lower()
        is_qwen3 = "qwen3" in self.vlm_model.lower()
        
        # Get image dimensions
        width, height = image.size
        image_shape = (height, width)
        
        def _convert_gemini_coords_to_pixels(coords, is_bbox=False):
            """Convert Gemini normalized coordinates (0-1000) to pixel coordinates."""
            if not is_gemini or not coords:
                return coords
            
            if is_bbox and len(coords) == 4:
                # bounding_box: [x_min, y_min, x_max, y_max]
                # gemini-robotics-er-1.5-preview: only output [y, x] format
                return [
                    int(coords[1] * width / 1000),
                    int(coords[0] * height / 1000),
                    int(coords[3] * width / 1000),
                    int(coords[2] * height / 1000)
                ] if is_robotics else [
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000),
                    int(coords[2] * width / 1000),
                    int(coords[3] * height / 1000)
                ]
            elif len(coords) == 2:
                # center_point: [x, y]
                return [
                    int(coords[1] * width / 1000),
                    int(coords[0] * height / 1000)
                ] if is_robotics else [
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000)
                ]
            return coords
        

        def _convert_qwen3_output(coords, is_bbox=False):
            if not is_qwen3 or not coords:
                return coords
            
            if is_bbox and len(coords) == 4:
                return [
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000),
                    int(coords[2] * width / 1000),
                    int(coords[3] * height / 1000)
                ]
            elif len(coords) == 2:
                return [
                    int(coords[0] * width / 1000),
                    int(coords[1] * height / 1000)
                ]
            return coords
        
        if is_qwen3:
            objects = [{
                "bbox": _convert_qwen3_output(obj.get("bounding_box", []), is_bbox=True),
                "center_point": _convert_qwen3_output(obj.get("center_point", [])),
                "label": obj.get("label", "unknown"),
                "score": 1.0,
                "mask": None
            } for obj in result.get("objects", [])]
        else:   
            objects = [{
                "bbox": _convert_gemini_coords_to_pixels(obj.get("bounding_box", []), is_bbox=True),
                "center_point": _convert_gemini_coords_to_pixels(obj.get("center_point", [])),
                "label": obj.get("label", "unknown"),
                "score": 1.0,
                "mask": None
            } for obj in result.get("objects", [])]
        
        return DetectionResult(success=True, objects=objects, 
                             image_shape=image_shape, raw_output=result)

    def save_detection(self, image: Image, detection_result: DetectionResult, save_dir: str = None) -> str:
        if not save_dir:
            clean_name = "_".join(detection_result.get_all_labels).replace(' ', '_')
            save_dir = os.path.join(self.session_output_dir, f"{clean_name}_result")
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "detection_result.json"), 'w') as f:
            json.dump(detection_result.output_to_json, f, indent=2)
        
        self._save_visualization(image, 
            [obj['bbox'] for obj in detection_result.objects],
            [obj['label'] for obj in detection_result.objects],
            [obj['score'] for obj in detection_result.objects],
            os.path.join(save_dir, "visualization.png"), "labeled",
            center_points=[obj.get('center_point') for obj in detection_result.objects])
        
        image.copy().save(os.path.join(save_dir, "original_img.png"))
        return save_dir

def main():
    import yaml
    # image = Image.open("data/data_for_test/color.png").convert("RGB")
    image = Image.open("/home/agenticlab/Project/vlm_robobench/output/kinect_captures/scene_image.png").convert("RGB")
    with open("configs/obj_detector_config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    detector = ObjectDetector(cfg)
    result = detector.detect(image, "blue cube")
    if result.success:
        print(f"Results saved to: {detector.save_detection(image, result)}")
        print(f"BBox: {result.get_object_bbox()}, Center: {result.get_object_center()}")
    else:
        print("Detection failed.")
    
if __name__ == "__main__":
    main()

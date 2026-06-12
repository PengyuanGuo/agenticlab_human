import os
import yaml
import logging
import numpy as np
from PIL import Image
from typing import Optional, Tuple, TypedDict
from vlm_robobench.modules.planning.task_parser import TaskParser
from vlm_robobench.modules.perception.obj_detector import ObjectDetector
from vlm_robobench.modules.planning.grasp_planner import GraspPlanner
from vlm_robobench.modules.planning.place_planner import PlacePlanner
from vlm_robobench.modules.action_wrapper_flexiv import ActionWrapper
from vlm_robobench.modules.planning.action_checker import ActionChecker
from vlm_robobench.modules.cam_capture import CameraCapture
import time

# Configure logging only if no handlers exist (prevents duplicate logging)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

PRIMARY_CAMERA = "FemtoBolt"
WRIST_CAMERA = "Gemini305"
ENABLE_WRIST_REPLAN = False

# TODO: Add yaml config for whole pipeline, for ablation study, e.g., disable action checker, etc.
class Modules(TypedDict):
    """Type definition for modules dictionary"""
    task_parser: TaskParser
    obj_detector: ObjectDetector
    grasp_planner: GraspPlanner
    place_planner: PlacePlanner
    action_wrapper: ActionWrapper
    action_checker: ActionChecker
    camera: CameraCapture
    wrist_camera: Optional[CameraCapture]

def load_config(config_filename: str) -> dict:
    """Load a specific configuration file
    Args:
        config_filename: Name of the config file to load (e.g., 'task_parser_config.yaml')
    Returns:
        Configuration dictionary
    """
    config_dir = "configs"
    filepath = os.path.join(config_dir, config_filename)
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def parse_action(action_str: str) -> Tuple[str, str, str]:
    """Parse PDDL action string"""
    action_str = action_str.strip("()").strip()
    parts = action_str.split()
    
    action_type = parts[0]
    object_name = parts[1] if len(parts) > 1 else None
    target_name = parts[2] if len(parts) > 2 else None
    
    return action_type, object_name, target_name

def execute_pick(object_name: str, modules: Modules, action_info: dict, max_retries: int = 2) -> bool:
    """Execute pick action
    
    Args:
        object_name: Name of the object to pick
        modules: Dictionary of initialized modules
        action_info: Dictionary containing action details (precondition, effect)
        max_retries: Maximum number of retry attempts
    """
    obj_detector = modules['obj_detector']
    grasp_planner = modules['grasp_planner']
    action_wrapper = modules['action_wrapper']
    action_checker = modules['action_checker']
    camera = modules['camera']
    
    action_str = f"(pick {object_name})"
    precondition = action_info.get('precondition', '')
    effect = action_info.get('effect', '')
    
    for retry in range(max_retries):
        logger.info("="*60)
        logger.info(f"Executing pick '{object_name}' (attempt {retry + 1}/{max_retries})")
        logger.info("="*60)
        
        # 1. Capture scene
        color, depth = camera.capture()
        rgb_image = camera.np_array_to_image(color)
        # 2. Check precondition
        precond_result = action_checker.check_precondition(rgb_image, action_str, precondition)
        action_checker.save_result(precond_result, "precondition", rgb_image, action_str)
        if not precond_result.success:
            logger.warning(f"Precondition check failed: {precond_result.reasoning}")
            continue
        
        # 3. Detect target object
        detection = obj_detector.detect(rgb_image, object_name)
        
        if not detection.success or detection.num_objects == 0:
            logger.warning(f"Object detection failed")
            continue
        else:
            save_dir = obj_detector.save_detection(rgb_image, detection)
            logger.info(f"Detection results saved to: {save_dir}")
        
        # 4. Plan grasp pose
        grasp_result = grasp_planner.plan_grasp(
            color=rgb_image,
            depth=depth,
            target_object=object_name,
            task_description=action_info.get('action'),
            obj_mask=detection.get_object_bbox([10, 10, 10, 10]),
            which_camera=PRIMARY_CAMERA
        )
        
        # Save grasp planning results
        grasp_planner.save_grasp_result(grasp_result)
        logger.info(f"Grasp planning results saved to: {grasp_planner.session_output_dir}")
        use_wrist_camera = False
        # TODO: Enable Gemini305 wrist-camera re-planning after its Flexiv hand-eye flow is validated.
        if ENABLE_WRIST_REPLAN and grasp_result.evaluation is not None and grasp_result.evaluation.get('action') == 'REJECT_GRASP':
            use_wrist_camera = True
            logger.info(f"Re-planning grasp using wrist camera view")
            # Switch to wrist camera to get a better view
            rot = grasp_result.rotation
            translation = grasp_result.translation
            wrist_cam_T = action_wrapper.transfer_pos_rot_to_T(translation, rot)
            action_wrapper.move(wrist_cam_T, frame='camera', offset=[-0.1 ,0, 0.2],
                                default_ort=[-1.96, 2.02, -0.44]) # adjusted ort for better wrist camera view
            time.sleep(2)
            color_wrist, depth_wrist = modules['wrist_camera'].capture()
            color_wrist = camera.np_array_to_image(color_wrist)
            new_detection = obj_detector.detect(color_wrist, object_name)
            # Re-plan grasp with wrist camera view
            cur_eef_pose = action_wrapper.transfer_pose6d_to_T(action_wrapper.controller.get_tcp_pose())
            grasp_result_wrist = grasp_planner.plan_grasp(
                color=color_wrist,
                depth=depth_wrist,
                target_object=object_name,
                task_description=action_info.get('action'),
                obj_mask=new_detection.get_object_bbox([10, 10, 10, 10]),
                which_camera=WRIST_CAMERA,
                cur_eef_pose=cur_eef_pose
            )
            # Save wrist camera grasp planning results
            grasp_planner.save_grasp_result(grasp_result_wrist)
            logger.info(f"Wrist camera grasp planning results saved to: {grasp_planner.session_output_dir}")
        

        # 5. Execute pick
        if use_wrist_camera:
            grasp_pose = grasp_result_wrist.to_dict()
            target_T = action_wrapper.transfer_pos_rot_to_T(grasp_pose['translation'], grasp_pose['rotation'])
            action_wrapper.pick(T_cam_grasp=target_T, cur_eef_pose=cur_eef_pose)
        else:
            grasp_pose = grasp_result.to_dict()
            target_T = action_wrapper.transfer_pos_rot_to_T(grasp_pose['translation'], grasp_pose['rotation'])
            action_wrapper.pick(T_cam_grasp=target_T, grasp_offset=[0.0, -0.01, 0.03])
        action_wrapper.move_to_home()
        time.sleep(5)  # Wait for a moment
        
        # 6. Verify pick effect
        rgb_after, _ = camera.capture()
        rgb_after = camera.np_array_to_image(rgb_after)
        effect_result = action_checker.check_effect(rgb_after, action_str, effect)
        action_checker.save_result(effect_result, "effect", rgb_after, action_str)
        if not effect_result.success:
            logger.warning(f"Pick effect verification failed: {effect_result.reasoning}")
            continue
        
        logger.info(f"Pick '{object_name}' succeeded!")
        return True
    
    logger.error(f"Pick '{object_name}' failed")
    return False


def execute_place(object_name: str, target_name: str, task_description: str, 
                  modules: Modules, action_info: dict, max_retries: int = 2) -> bool:
    """Execute place action
    
    Args:
        object_name: Name of the object to place
        target_name: Name of the target location
        task_description: Description of the overall task
        modules: Dictionary of initialized modules
        action_info: Dictionary containing action details (precondition, effect)
        max_retries: Maximum number of retry attempts
    """
    obj_detector = modules['obj_detector']
    place_planner = modules['place_planner']
    action_wrapper = modules['action_wrapper']
    action_checker = modules['action_checker']
    camera = modules['camera']
    
    action_str = f"(place {object_name} {target_name})"
    precondition = action_info.get('precondition', '')
    effect = action_info.get('effect', '')
    
    for retry in range(max_retries):
        logger.info("="*60)
        logger.info(f"Executing place '{object_name}' on '{target_name}' (attempt {retry + 1}/{max_retries})")
        logger.info("="*60)

        # 1. Capture scene
        rgb_image, depth_image = camera.capture()
        rgb_image = camera.np_array_to_image(rgb_image)
        # 2. Check precondition
        precond_result = action_checker.check_precondition(rgb_image, action_str, precondition)
        action_checker.save_result(precond_result, "precondition", rgb_image, action_str)
        if not precond_result.success:
            logger.warning(f"Precondition check failed: {precond_result.reasoning}")
            continue
        
        # 3. Detect target location
        detection = obj_detector.detect(rgb_image, target_name)
        if not detection.success or detection.num_objects == 0:
            logger.warning(f"Target location detection failed")
            continue
        else:
            save_dir = obj_detector.save_detection(rgb_image, detection)
            logger.info(f"Target location detection results saved to: {save_dir}")
        detection.set_random_obj_point(margin_ratio= 0.5) # Set a random point within the detected target area for placing
        
        # 4. Plan placing point
        placing_result = place_planner.plan_placing_point(
            image=rgb_image,
            depth=depth_image,
            initial_point=detection.get_object_center(),
            target_place=target_name,
            task_description=task_description,
            which_camera=PRIMARY_CAMERA
        )
        

        logger.info(f"Random place pixel: {detection.get_object_center()}")
        logger.info(f"Place point in camera frame: {placing_result.pose_3d}")

        # Save placing point results
        place_planner.save_results(rgb_image, placing_result, place_planner.session_output_dir)
        logger.info(f"Placing point results saved to: {place_planner.session_output_dir}")
        
        # 5. Execute place
        target_T_cam = action_wrapper.transfer_pos_rot_to_T(placing_result.pose_3d, np.eye(3))
        # action_wrapper.place(T_base_place=target_T_cam, need_hand_eye_conversion=True, place_offset=[0.0, 0.05, 0.12])
        action_wrapper.place(T_base_place=target_T_cam, need_hand_eye_conversion=True, place_offset=([-0.0, 0.0, 0.18]))
        action_wrapper.move_to_home()
        time.sleep(5)  # Wait for a moment
        
        # 6. Verify place effect
        rgb_after, _ = camera.capture()
        rgb_after = camera.np_array_to_image(rgb_after)
        effect_result = action_checker.check_effect(rgb_after, action_str, effect)
        action_checker.save_result(effect_result, "effect", rgb_after, action_str)
        if not effect_result.success:
            logger.warning(f"Place effect verification failed: {effect_result.reasoning}")
            continue
        
        logger.info(f"Place '{object_name}' on '{target_name}' succeeded!")
        return True
    
    logger.error(f"Place '{object_name}' on '{target_name}' failed")
    return False


def execute_pipeline(task_description: str, max_replans: int = 5) -> bool:
    """Execute complete pick-and-place pipeline"""
    logger.info("="*60)
    logger.info(f"Starting task execution: {task_description}")
    logger.info("="*60)
    
    # Initialize camera
    camera = CameraCapture(which_cam=PRIMARY_CAMERA)
    modules = None
    
    try:
        # Load configs and initialize modules
        modules: Modules = {
            'task_parser': TaskParser(load_config('task_parser_config.yaml')),
            'obj_detector': ObjectDetector(load_config('obj_detector_config.yaml')),
            'grasp_planner': GraspPlanner(load_config('grasp_planner_config.yaml')),
            'place_planner': PlacePlanner(load_config('place_planner_config.yaml')),
            'action_wrapper': ActionWrapper(
                load_config('action_wrapper.yaml'),
                load_config('flexiv_config.yaml'),
                load_config('camera_config.yaml')
            ),
            'action_checker': ActionChecker(load_config('action_checker_config.yaml')),
            'camera': camera,
            'wrist_camera': CameraCapture(which_cam=WRIST_CAMERA) if ENABLE_WRIST_REPLAN else None
        }
        
        # Phase 1: Task initialization
        logger.info("\n" + "="*60)
        logger.info("Phase 1: Task Initialization")
        logger.info("="*60)
        
        rgb_initial, _ = camera.capture()
        rgb_initial = camera.np_array_to_image(rgb_initial)
        task_plan = modules['task_parser'].parse_task(
            task_description=task_description,
            scene_image=rgb_initial
        )
        action_sequence = task_plan.action_sequence
        logger.info(f"Generated action sequence: {action_sequence}")
        
        # Save task parser results
        modules['task_parser'].save_results(task_plan, rgb_initial)
        logger.info(f"Task plan saved to: {modules['task_parser'].session_output_dir}")
        
        # Phase 2: Action execution loop
        for replan_count in range(max_replans):
            logger.info("\n" + "="*60)
            logger.info(f"Execution attempt {replan_count + 1}/{max_replans}")
            logger.info("="*60)
            
            all_success = True
            for action_idx, action in enumerate(action_sequence):
                logger.info(f"\n>>> Action {action_idx + 1}/{len(action_sequence)}: {action}")
                
                action_type, object_name, target_name = parse_action(action)
                
                # Get action info from task plan
                action_info = {}
                if task_plan.action_details and action_idx < len(task_plan.action_details):
                    action_info = task_plan.action_details[action_idx]
                
                if action_type == "pick":
                    success = execute_pick(object_name, modules, action_info)
                elif action_type == "place":
                    success = execute_place(object_name, target_name, task_description, modules, action_info)
                else:
                    logger.warning(f"Unknown action type: {action_type}")
                    success = False
                
                if not success:
                    all_success = False
                    break
            
            # Phase 3: Task completion verification
            if all_success:
                logger.info("\n" + "="*60)
                logger.info("Phase 3: Task Completion Verification")
                logger.info("="*60)
                
                rgb_final, _ = camera.capture()
                rgb_final = camera.np_array_to_image(rgb_final)
                goal_conditions = task_plan.goal_conditions if task_plan.goal_conditions else []
                
                completion_result = modules['action_checker'].check_task_completion(
                    current_image=rgb_final,
                    goal_conditions=goal_conditions
                )
                
                modules['action_checker'].save_result(
                    completion_result, "task_completion", rgb_final, 
                    f"Task: {task_description}"
                )
                
                if completion_result.success:
                    logger.info("\n" + "="*60)
                    logger.info("✓ Task completed successfully!")
                    logger.info("="*60)
                    return True
                else:
                    logger.warning(f"Task not completed: {completion_result.reasoning}")
            
            # Replan
            logger.warning(f"\nTask not completed, replanning...")
            modules['action_wrapper'].move_to_home()
            time.sleep(5)  # Wait for a moment
            rgb_image, _ = camera.capture()
            rgb_image = camera.np_array_to_image(rgb_image)
            task_plan = modules['task_parser'].parse_task(
                task_description=task_description,
                scene_image=rgb_image
            )
            action_sequence = task_plan.action_sequence
            logger.info(f"New action sequence: {action_sequence}")
            
            # Save replanned results
            modules['task_parser'].save_results(task_plan, rgb_image)
            logger.info(f"Replan {replan_count + 1} saved to: {modules['task_parser'].session_output_dir}")
        
        logger.error("\n" + "="*60)
        logger.error("✗ Task failed: Maximum replanning attempts reached")
        logger.error("="*60)
        return False
    
    finally:
        if modules is not None:
            try:
                modules['action_wrapper'].shutdown(move_home=False)
            except Exception as exc:
                logger.warning(f"ActionWrapper shutdown warning: {exc}")
            wrist_camera = modules.get('wrist_camera')
            if wrist_camera is not None:
                wrist_camera.destroy()
        # Clean up camera resources
        camera.destroy()
if __name__ == "__main__":
    # Example task
    # task = "pick up the wooden block and place it in the yellow bin"
    task =  "Pick up the 3D-printed parts and place them on the yellow bin"
    # task = "Sort the fruits in the white bin and other objects in the blue bin."
    # task = "I'm hungry. Grab the edible stuff to the bowl."
    # task = "Sort the lion and stuffed toys into the box."
    # task = "Sort the cups to the plates by color."
    # task = "Grab the fruits to the box."
    start_time = time.time()
    success = execute_pipeline(task, max_replans=5)
    print(f"\nFinal result: {'Success' if success else 'Failed'}")
    end_time = time.time()
    print("="*60)
    print(f"Time taken: {end_time - start_time} seconds for task: {task}")
    print("="*60)

import numpy as np
import logging
import os
import yaml
import json
import time
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from PIL import Image
from pathlib import Path
from agenticlab_human.llm.llm_manager import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskPlan:
    task_description: str
    objects_identified: List[str]
    reasoning: str
    action_sequence: List[str]
    elapsed_time: float
    timestamp: str
    updated_domain: Optional[str] = None
    problem_pddl: Optional[str] = None
    action_details: Optional[List[Dict]] = None
    goal_conditions: Optional[List[str]] = None
    token_usage: Optional[int] = None
    
    def to_dict(self) -> Dict:
        result = {
            "task_description": self.task_description,
            "objects_identified": self.objects_identified,
            "reasoning": self.reasoning,
            "action_sequence": self.action_sequence,
            "elapsed_time": self.elapsed_time,
            "timestamp": self.timestamp
        }
        if self.updated_domain:
            result["updated_domain"] = self.updated_domain
        if self.problem_pddl:
            result["problem_pddl"] = self.problem_pddl
        if self.action_details:
            result["action_details"] = self.action_details
        if self.goal_conditions:
            result["goal_conditions"] = self.goal_conditions
        if self.token_usage is not None:
            result["token_usage"] = self.token_usage
        return result
    
    def to_action_sequence(self):
        """Return an ActionSequence object built from this TaskPlan.

        Parses each PDDL-style string in action_sequence into a structured
        Action(id, name, args) using parameter names extracted from the domain.
        This is the runtime contract object consumed by the Executor.
        """
        from agenticlab_human.core.action_sequence import ActionSequence
        return ActionSequence.from_task_plan(self)

    def to_action_sequence_dict(self) -> Dict:
        """Return the ActionSequence as a plain dict (JSON-serializable).

        Convenience wrapper around to_action_sequence().to_dict().
        """
        return self.to_action_sequence().to_dict()

    def save_to_file(self, filepath: str):
        with open(filepath, 'w') as f:
            for action in self.action_sequence:
                f.write(f"{action}\n")
        logger.info(f"Action sequence saved to {filepath}")


class TaskParser:
    def __init__(self, cfg: Dict):
        self.cfg = cfg.get("TaskParser", cfg)
        self.prompt_path = self.cfg.get("prompt_path", "configs/prompt/task_parser_prompt.txt")
        self.prompt_use_pddl_path = self.cfg.get("prompt_use_pddl_path", 
                                                  "configs/prompt/task_parser_prompt_with_pddl.txt")
        self.vlm_model_name = self.cfg.get("vlm_model_name", "openai:gpt-5-2025-08-07")
        
        self.use_pddl = self.cfg.get("use_pddl", False)
        self.domain_path = self.cfg.get("pddl_domain_path", "")
        self.example_nl_path = self.cfg.get("example_nl_path", "")
        self.example_pddl_path = self.cfg.get("example_pddl_path", "")
        
        self.pddl_cfg = self.cfg.get("pddl_cfg", {})
        self.fast_downward_path = self.pddl_cfg.get("fast_downward_path", "third_party/downward/fast-downward.py")
        self.fast_downward_alias = self.pddl_cfg.get("fast_downward_alias", "seq-sat-lama-2011")
        self.time_limit = self.pddl_cfg.get("time_limit", 20)

        base_output_dir = self.cfg.get("output_dir", "output/task_parser")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(self.session_output_dir, exist_ok=True)
        
        self.llm_manager = LLMManager()
        llm_config_path = self.cfg.get("llm_config_path", "configs/llm_interface_config.yaml")
        self.llm_manager.load_config(llm_config_path)
        
        with open(self.domain_path, 'r') as f:
            self.domain_content = f.read()
        logger.info(f"Loaded domain file: {self.domain_path}")
        
        if self.use_pddl:
            with open(self.prompt_use_pddl_path, 'r') as f:
                self.prompt_template = f.read()
            self._load_pddl_files()
        else:
            with open(self.prompt_path, 'r') as f:
                self.prompt_template = f.read()
            
        logger.info(f"TaskParser initialized with model: {self.vlm_model_name}")
        logger.info(f"Use PDDL mode: {self.use_pddl}")
        logger.info(f"Output directory: {self.session_output_dir}")
    
    def _get_action_info(self, action_str: str, domain_content: str) -> Dict:
        """Extract precondition and effect for a given action from domain content."""
        cleaned = action_str.strip().strip('()')
        parts = cleaned.split()

        if not parts:
            return {"action": action_str}

        action_name = parts[0]
        params = parts[1:] if len(parts) > 1 else []
        
        # Regex to find action definition
        pattern = rf':action\s+{action_name}.*?:precondition\s+(.*?):effect\s+(.*?)(?=\(:action|\Z)'
        action_match = re.search(pattern, domain_content, re.DOTALL)

        return {
            "action": action_str,
            "action_name": action_name,
            "parameters": params,
            "precondition": action_match.group(1).strip(),
            "effect": action_match.group(2).strip()
        }
    
    def _get_goal_info(self, problem_content: str) -> List[str]:
        """Extract goal conditions from problem PDDL file."""
        if not problem_content:
            return []
        
        # Extract goal section using regex
        goal_pattern = r':goal\s*\((.*)\)\s*\)\s*\)'
        goal_match = re.search(goal_pattern, problem_content, re.DOTALL)
        
        if not goal_match:
            return []
        
        goal_content = goal_match.group(1).strip()
        
        # Parse goal conditions
        goal_conditions = []
        
        # Handle 'and' wrapper
        if goal_content.startswith('and'):
            # Remove 'and' keyword and extract conditions
            inner_content = goal_content[3:].strip()
            
            # Parse predicates by counting parentheses
            depth = 0
            current_pred = ""
            for char in inner_content:
                if char == '(':
                    depth += 1
                    current_pred += char
                elif char == ')':
                    depth -= 1
                    current_pred += char
                    if depth == 0 and current_pred.strip():
                        goal_conditions.append(current_pred.strip())
                        current_pred = ""
                elif depth > 0:
                    current_pred += char
        else:
            # Single condition
            goal_conditions = [goal_content]
        
        return goal_conditions
    
    def _load_pddl_files(self):
        with open(self.example_nl_path, 'r') as f:
            self.example_nl = f.read()
        logger.info(f"Loaded example NL: {self.example_nl_path}")
    
        with open(self.example_pddl_path, 'r') as f:
            self.example_pddl = f.read()
        logger.info(f"Loaded example PDDL: {self.example_pddl_path}")
    
    def parse_task(self, task_description: str, scene_image: Image.Image) -> TaskPlan:
        logger.info(f"Parsing task: {task_description}")
        
        if self.use_pddl:
            prompt = self.prompt_template.format(
                task_description=task_description,
                domain_file=self.domain_content,
                example_nl=self.example_nl,
                example_pddl=self.example_pddl
            )
        else:
            prompt = self.prompt_template.format(task_description=task_description)
        
        start_time = time.time()
        provider, model = self.vlm_model_name.split(":", 1) if ":" in self.vlm_model_name else (None, self.vlm_model_name)
        result = self.llm_manager.call(
            prompt=prompt,
            image=scene_image,
            provider=provider,
            model=model
        )
        elapsed = time.time() - start_time
        action_sequence = result.get("action_sequence", [])
        if self.use_pddl and result.get("updated_domain") and result.get("problem_pddl"):
            logger.info("PDDL mode: solving with Fast Downward...")
            action_sequence = self._solve_with_fast_downward(
                result.get("updated_domain"),
                result.get("problem_pddl")
            )
            if not action_sequence:
                logger.warning("Fast Downward failed to find a plan, using original action sequence from VLM.")
            elapsed = time.time() - start_time
        
        # extract action details if available
        updated_domain = result.get("updated_domain")
        action_details = None
        domain = updated_domain or self.domain_content
        if action_sequence and domain:
            action_details = [self._get_action_info(a, domain) for a in action_sequence] if self.use_pddl else None
        
        # extract goal details if available
        goal_conditions = None
        problem_pddl = result.get("problem_pddl")
        if problem_pddl:
            goal_conditions = self._get_goal_info(problem_pddl)
        
        task_plan = TaskPlan(
            task_description=task_description,
            objects_identified=result.get("objects_identified", []),
            reasoning=result.get("reasoning", ""),
            action_sequence=action_sequence,
            elapsed_time=elapsed,
            timestamp=datetime.now().isoformat(),
            updated_domain=result.get("updated_domain"),
            action_details=action_details,
            goal_conditions=goal_conditions,
            problem_pddl=result.get("problem_pddl"),
            token_usage=result.get("token_usage")
        )
        
        logger.info(f"Task parsed in {elapsed:.2f}s")
        logger.info(f"Objects identified: {task_plan.objects_identified}")
        logger.info(f"Action sequence: {len(task_plan.action_sequence)} actions")
        
        return task_plan
    
    def _solve_with_fast_downward(self, domain_content: str, problem_content: str) -> List[str]:
        domain_file = os.path.join(self.session_output_dir, "domain.pddl")
        problem_file = os.path.join(self.session_output_dir, "problem.pddl")
        plan_file = os.path.join(self.session_output_dir, "plan.sol")
        
        with open(domain_file, 'w') as f:
            f.write(domain_content)
        with open(problem_file, 'w') as f:
            f.write(problem_content)
        
        logger.info("Running Fast Downward planner...")
        os.system(f"python {self.fast_downward_path} "
                  f"--alias {self.fast_downward_alias} "
                  f"--search-time-limit {self.time_limit} "
                  f"--plan-file {plan_file} "
                  f"{domain_file} {problem_file}")
        
        best_plan = []
        best_cost = float('inf')
        for fn in glob.glob(f"{plan_file}.*"):
            with open(fn, 'r') as f:
                lines = f.readlines()
                if lines:
                    actions = [line.strip() for line in lines[:-1] if line.strip() and not line.startswith(';')]
                    try:
                        cost = int(lines[-1].split('=')[-1].strip()) if 'cost' in lines[-1] else len(actions)
                        if cost < best_cost:
                            best_cost = cost
                            best_plan = actions
                    except:
                        if not best_plan:
                            best_plan = actions
        
        if best_plan:
            logger.info(f"Found plan with {len(best_plan)} actions (cost: {best_cost})")
        else:
            logger.warning("No plan found")
        
        return best_plan
    
    def generate_plan(
        self,
        task_description: str,
        scene_image: Image.Image,
        save_action_sequence: bool = True,
    ) -> TaskPlan:
        """One-shot entry point for the decoupled planner stage.

        Runs parse_task() then persists all outputs to the session directory:
          - task_plan.json          always  (source of truth)
          - action_sequence.txt     always  (human-readable PDDL strings)
          - domain.pddl / problem.pddl  when use_pddl=True
          - scene_image.png         always
          - action_sequence.json    when save_action_sequence=True (default)

        The Executor can load the plan later with no in-memory coupling:
            ActionSequence.load("output/task_parser/<ts>/")

        Returns the TaskPlan; call plan.to_action_sequence() for the runtime
        contract object, or plan.to_action_sequence_dict() for the plain dict.
        """
        task_plan = self.parse_task(task_description, scene_image)
        self.save_results(task_plan, save_image=scene_image,
                          save_action_sequence=save_action_sequence)
        return task_plan

    def save_results(
        self,
        task_plan: TaskPlan,
        save_image: Optional[Image.Image] = None,
        save_action_sequence: bool = True,
    ):
        # task_plan.json – primary source of truth; always written
        json_path = os.path.join(self.session_output_dir, "task_plan.json")
        with open(json_path, 'w') as f:
            json.dump(task_plan.to_dict(), f, indent=4, ensure_ascii=False)
        logger.info(f"Task plan saved to {json_path}")

        # action_sequence.txt – human-readable PDDL strings; always written
        txt_path = os.path.join(self.session_output_dir, "action_sequence.txt")
        task_plan.save_to_file(txt_path)

        # action_sequence.json – optional executor-friendly artifact;
        # can be regenerated at any time via ActionSequence.load(session_dir)
        if save_action_sequence:
            action_seq_path = os.path.join(self.session_output_dir, "action_sequence.json")
            with open(action_seq_path, 'w') as f:
                json.dump(task_plan.to_action_sequence_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"ActionSequence JSON saved to {action_seq_path}")
        
        if self.use_pddl:
            if task_plan.updated_domain:
                domain_path = os.path.join(self.session_output_dir, "domain.pddl")
                with open(domain_path, 'w') as f:
                    f.write(task_plan.updated_domain)
                logger.info(f"Updated domain saved to {domain_path}")
            
            if task_plan.problem_pddl:
                pddl_path = os.path.join(self.session_output_dir, "problem.pddl")
                with open(pddl_path, 'w') as f:
                    f.write(task_plan.problem_pddl)
                logger.info(f"Problem PDDL saved to {pddl_path}")
        
        if save_image:
            img_path = os.path.join(self.session_output_dir, "scene_image.png")
            save_image.save(img_path)
            logger.info(f"Scene image saved to {img_path}")
        
        logger.info(f"All results saved to {self.session_output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TaskParser')
    parser.add_argument('--config', type=str, default='configs/planning/task_parser_config.yaml')
    # parser.add_argument('--image', type=str, default='data/data_for_test/color.png')
    parser.add_argument('--image', type=str, default='/home/agenticlab/Project/agenticlab_human/data/data_for_test/task_parser/19_stack3.png')
    parser.add_argument('--task', type=str, 
                        # default="sort the fruits into the grey bowl and other objects into the box.")
                        # default='Stack the cubes on the pink plate from bottom to top: Orange, Yellow, Green and Blue cubes.')
                        # default='Fill numbered slots using the blocks provided to solve the crossword puzzle. Note: You do not need to use all blocks or slots to solve the puzzle.')
                        default='Stack the cubes on the pink plate from bottom to top: Orange, Yellow, Green and Blue cubes.')
                        # default='A spice bottle is in the top drawer. Relocate pot lid to mat and chicken leg to pot and spice bottle to blue plate')
                        # default='A spice bottle is in the top drawer, take out the spice bottle from the top drawer, then put the snack bag in the bottom drawer, close the drawer.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    scene_image = Image.open(args.image).convert("RGB")
    task_parser = TaskParser(cfg)
    
    print(f"\n{'='*60}")
    print(f"Task: {args.task}")
    print(f"{'='*60}\n")
    
    task_plan = task_parser.parse_task(args.task, scene_image)
    
    print(f"\n{'='*60}")
    print("Task Planning Results")
    print(f"{'='*60}")
    
    objects_str = []
    for obj in task_plan.objects_identified:
        if isinstance(obj, dict):
            objects_str.append(obj.get('name', str(obj)))
        else:
            objects_str.append(str(obj))
    print(f"\nObjects Identified: {', '.join(objects_str)}")
    
    print(f"\nReasoning:\n{task_plan.reasoning}")
    print(f"\nAction Sequence:")
    for i, action in enumerate(task_plan.action_sequence, 1):
        print(f"  {i}. {action}")
    print(f"\nExecution Time: {task_plan.elapsed_time:.2f}s")
    
    task_parser.save_results(task_plan, scene_image)
    print(f"\nResults saved to: {task_parser.session_output_dir}")

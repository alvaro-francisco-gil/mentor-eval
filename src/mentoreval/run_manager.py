"""
Run management system for MentorEval benchmark.

This module handles run tracking, ID generation, run lifecycle management,
and execution of runs using the LightEval benchmark system.
"""

import os
import json
import glob
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RunInfo:
    """Information about a benchmark run."""
    run_id: int
    parameters: Dict[str, Any]  # User-friendly parameters
    configuration: Dict[str, Any]  # Technical configuration
    status: str = 'created'  # 'created', 'running', 'completed', 'failed'
    
    @property
    def model_name(self) -> str:
        """Get model name from parameters."""
        return self.parameters.get('model_name', 'gpt-4o-mini')


def convert_simplified_json_to_run_info(json_data: Dict[str, Any]) -> RunInfo:
    """Convert simplified JSON format to RunInfo."""
    # Extract parameters (important for benchmark)
    parameters = json_data.get('parameters', {})
    
    # Extract configuration (how to run it)
    config = json_data.get('configuration', {})
    
    # Determine if we should include guidance (rubric or desired_answer - whichever is present)
    show_guidance = parameters.get('show_guidance', True)
    force_explanation = parameters.get('force_explanation', False)
    show_isced_level = parameters.get('show_isced_level', False)
    
    # Build the full configuration for LightEval
    full_config = {
        "use_local_backend": config.get("use_local_backend", False),
        "model_name": parameters.get("model_name", "gpt-4o-mini"),
        "task_name": parameters.get("task_name", "mentoreval"),  # Read from parameters, default to mentoreval
        "task_args": {
            "max_samples": parameters.get("test_samples", 20),  # Use test_samples, default to 20
            "num_fewshot_seeds": 1,
        },
        "model_args": {
            "use_chat_template": True,
        },
        "generation_args": {
            "max_new_tokens": config.get("generation_args", {}).get("max_new_tokens", 500),
            "temperature": config.get("generation_args", {}).get("temperature", 0.0),
            "do_sample": config.get("generation_args", {}).get("do_sample", False),
        },
        "show_guidance": show_guidance,
        "force_explanation": force_explanation,
        "show_isced_level": show_isced_level
    }
    
    return RunInfo(
        run_id=json_data.get('run_id', 0),
        parameters=parameters,  # Include the user-friendly parameters
        configuration=full_config,
        status=json_data.get('status', 'created')
    )


class RunManager:
    """Manages benchmark runs and their lifecycle."""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)
    
    def get_next_run_id(self) -> int:
        """Get the next available run ID by checking existing run files."""
        existing_ids = set()
        
        # Check for existing run files
        run_files = glob.glob(os.path.join(self.runs_dir, "*.json"))
        for run_file in run_files:
            filename = os.path.basename(run_file)
            # Extract ID from filename like "0_run.json", "1_test.json", "24_test.json", etc.
            match = filename.split('_')[0]
            try:
                existing_ids.add(int(match))
            except ValueError:
                continue
        
        # Return next available ID
        return max(existing_ids, default=-1) + 1
    
    def create_run(self, parameters: Dict[str, Any], configuration: Dict[str, Any]) -> RunInfo:
        """Create a new run and save it to the runs directory."""
        run_id = self.get_next_run_id()
        
        run_info = RunInfo(
            run_id=run_id,
            parameters=parameters,
            configuration=configuration,
            status='created'
        )
        
        # Save run info to file
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(run_info), f, indent=2)
        
        return run_info
    
    def create_run_user_friendly(self, parameters: Dict[str, Any], configuration: Dict[str, Any]) -> RunInfo:
        """Create a new run using the user-friendly format (like 7_run.json)."""
        run_id = self.get_next_run_id()
        
        # Create the user-friendly JSON structure
        run_data = {
            "run_id": run_id,
            "parameters": parameters,
            "configuration": configuration,
            "status": "created"
        }
        
        # Save run data to file in user-friendly format
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(run_data, f, indent=2)
        
        # Convert to RunInfo for internal use
        return convert_simplified_json_to_run_info(run_data)
    
    def update_run_status(self, run_id: int, status: str):
        """Update the status of a run."""
        # Find the run file by looking through all JSON files
        run_files = glob.glob(os.path.join(self.runs_dir, "*.json"))
        
        for run_file in run_files:
            # Skip template file
            if "template" in run_file.lower():
                continue
                
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                
                # Check if this file contains the run_id we're looking for
                if run_data.get('run_id') == run_id:
                    # Update status
                    run_data['status'] = status
                    
                    # Save updated run info
                    with open(run_file, 'w', encoding='utf-8') as f:
                        json.dump(run_data, f, indent=2)
                    return
                    
            except (json.JSONDecodeError, TypeError):
                continue
        
        raise FileNotFoundError(f"Run {run_id} not found in any JSON file")
    
    def get_run_info(self, run_id: int) -> Optional[RunInfo]:
        """Get run information by ID."""
        # Look through all JSON files to find the one with matching run_id
        run_files = glob.glob(os.path.join(self.runs_dir, "*.json"))
        
        for run_file in run_files:
            # Skip template file
            if "template" in run_file.lower():
                continue
                
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                
                # Check if this file contains the run_id we're looking for
                if run_data.get('run_id') == run_id:
                    # Only support simplified format
                    if 'parameters' in run_data:
                        return convert_simplified_json_to_run_info(run_data)
                    
            except (json.JSONDecodeError, TypeError):
                continue
        
        return None
    
    def list_runs(self) -> List[RunInfo]:
        """List all runs in chronological order."""
        runs = []
        run_files = glob.glob(os.path.join(self.runs_dir, "*.json"))
        
        for run_file in run_files:
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                
                # Only support simplified format
                if 'parameters' in run_data:
                    runs.append(convert_simplified_json_to_run_info(run_data))
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Sort by run ID
        runs.sort(key=lambda x: x.run_id)
        return runs
    
    def get_latest_run(self) -> Optional[RunInfo]:
        """Get the most recent run."""
        runs = self.list_runs()
        return runs[-1] if runs else None
    
    def get_incomplete_runs(self) -> List[RunInfo]:
        """Get all runs that are not completed or failed."""
        runs = self.list_runs()
        incomplete = []
        
        for run in runs:
            if run.status in ['created', 'running']:
                incomplete.append(run)
        
        return incomplete
    
    def get_completed_runs(self) -> List[RunInfo]:
        """Get all completed runs."""
        runs = self.list_runs()
        completed = []
        
        for run in runs:
            if run.status == 'completed':
                completed.append(run)
        
        return completed
    
    
    def delete_run(self, run_id: int) -> bool:
        """Delete a run file."""
        # Look for any file with the pattern {run_id}_*
        run_files = glob.glob(os.path.join(self.runs_dir, f"{run_id}_*.json"))
        
        if run_files:
            for run_file in run_files:
                os.remove(run_file)
            return True
        return False
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of all runs."""
        runs = self.list_runs()
        
        if not runs:
            return {
                "total_runs": 0,
                "latest_run_id": None,
                "status_counts": {},
                "runs": []
            }
        
        status_counts = {}
        for run in runs:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1
        
        return {
            "total_runs": len(runs),
            "latest_run_id": runs[-1].run_id,
            "status_counts": status_counts,
            "runs": [
                {
                    "run_id": run.run_id,
                    "model_name": run.model_name,
                    "status": run.status
                }
                for run in runs
            ]
        }
    
    def execute_run(self, run_id: int) -> Dict[str, Any]:
        """Execute a specific run using the LightEval benchmark system."""
        run_info = self.get_run_info(run_id)
        if not run_info:
            raise ValueError(f"Run {run_id} not found")
        
        if run_info.status == 'completed':
            print(f"⚠️ Run {run_id} is already completed. Skipping.")
            return {"status": "already_completed", "run_id": run_id}
        
        # Update status to running
        self.update_run_status(run_id, "running")
        
        try:
            # Import here to avoid circular imports
            from .benchmark import LightEvalBenchmark
            from .models import ModelConfig
            
            # Create benchmark instance
            benchmark = LightEvalBenchmark(
                runs_dir="runs",
                results_dir="results", 
                results_extended_dir="results_extended"
            )
            
            # Create model configuration
            model_config = ModelConfig(
                model_name=run_info.model_name
            )
            
            # Get configuration from run info
            task_name = run_info.configuration.get("task_name", "custom|mentor_eval:asap_exercise_set_1|0")
            task_args = run_info.configuration.get("task_args", {})
            model_args = run_info.configuration.get("model_args", {})
            generation_args = run_info.configuration.get("generation_args", {})
            use_local_backend = run_info.configuration.get("use_local_backend", False)
            
            # Execute the evaluation directly without creating new run files
            result = benchmark.execute_evaluation_directly(run_info)
            
            # Update original run status based on result
            if result.get("status") == "completed":
                self.update_run_status(run_id, "completed")
            else:
                self.update_run_status(run_id, "failed")
            
            return {
                "status": result.get("status", "failed"),
                "run_id": run_id,
                "result": result
            }
            
        except Exception as e:
            # Update status to failed
            self.update_run_status(run_id, "failed")
            return {
                "status": "failed",
                "run_id": run_id,
                "error": str(e)
            }
    
    def execute_unexecuted_runs(self) -> List[Dict[str, Any]]:
        """Execute all runs that are not completed or failed."""
        incomplete_runs = self.get_incomplete_runs()
        results = []
        
        for run in incomplete_runs:
            print(f"🚀 Executing Run {run.run_id}...")
            result = self.execute_run(run.run_id)
            results.append(result)
            
            if result["status"] == "completed":
                print(f"✅ Run {run.run_id} completed successfully!")
            else:
                print(f"❌ Run {run.run_id} failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def find_incomplete_runs(self) -> List[Dict[str, Any]]:
        """Find runs that don't have corresponding result files."""
        incomplete_runs = []
        
        # Get all run JSON files in the runs directory
        run_files = glob.glob(os.path.join(self.runs_dir, "*.json"))
        
        for run_file in run_files:
            # Skip template file
            if "template" in run_file.lower():
                continue
                
            try:
                # Load run data
                with open(run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                
                run_id = run_data.get('run_id')
                if run_id is None:
                    continue
                
                # Check if result file exists
                result_files = glob.glob(f"results/{run_id}_*.json")
                
                if not result_files:
                    # No result file found, this run is incomplete
                    incomplete_runs.append(run_data)
                    
            except Exception as e:
                print(f"⚠️  Error reading {run_file}: {e}")
                continue
        
        return incomplete_runs

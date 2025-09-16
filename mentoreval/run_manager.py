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
    model_name: str
    benchmark_mode: str
    configuration: Dict[str, Any]
    status: str = 'created'  # 'created', 'running', 'completed', 'failed'


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
            # Extract ID from filename like "0_run.json", "1_run.json", etc.
            match = filename.split('_')[0]
            try:
                existing_ids.add(int(match))
            except ValueError:
                continue
        
        # Return next available ID
        return max(existing_ids, default=-1) + 1
    
    def create_run(self, model_name: str, benchmark_mode: str, configuration: Dict[str, Any]) -> RunInfo:
        """Create a new run and save it to the runs directory."""
        run_id = self.get_next_run_id()
        
        run_info = RunInfo(
            run_id=run_id,
            model_name=model_name,
            benchmark_mode=benchmark_mode,
            configuration=configuration,
            status='created'
        )
        
        # Save run info to file
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(run_info), f, indent=2)
        
        return run_info
    
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
                    return RunInfo(**run_data)
                    
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
                runs.append(RunInfo(**run_data))
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
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        
        if os.path.exists(run_file):
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
                    "benchmark_mode": run.benchmark_mode,
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
            
            # Create run info for benchmark
            benchmark_run_info = benchmark.create_run(
                model_config=model_config,
                benchmark_mode=run_info.benchmark_mode,
                description=f"Executing run {run_id}",
                use_local_backend=use_local_backend,
                task_name=task_name,
                task_args=task_args,
                model_args=model_args,
                generation_args=generation_args
            )
            
            # Execute the run
            result = benchmark.execute_run(benchmark_run_info)
            
            # Update original run status to completed
            self.update_run_status(run_id, "completed")
            
            return {
                "status": "completed",
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

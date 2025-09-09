"""
Run management system for MentorEval benchmark.

This module handles run tracking, ID generation, and run lifecycle management.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from .config import MentorEvalConfig


@dataclass
class RunInfo:
    """Information about a benchmark run."""
    run_id: int
    model_name: str
    benchmark_mode: str
    configuration: Dict[str, Any]
    status: str  # 'created', 'running', 'completed', 'failed'


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
    
    def create_run(self, config: MentorEvalConfig) -> RunInfo:
        """Create a new run and save it to the runs directory."""
        run_id = self.get_next_run_id()
        
        run_info = RunInfo(
            run_id=run_id,
            model_name=config.model_name,
            benchmark_mode=config.mode.value,
            configuration=asdict(config),
            status='created'
        )
        
        # Save run info to file
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(run_info), f, indent=2)
        
        return run_info
    
    def update_run_status(self, run_id: int, status: str):
        """Update the status of a run."""
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        
        if not os.path.exists(run_file):
            raise FileNotFoundError(f"Run file not found: {run_file}")
        
        # Load existing run info
        with open(run_file, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        
        # Update status
        run_data['status'] = status
        
        # Save updated run info
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(run_data, f, indent=2)
    
    def get_run_info(self, run_id: int) -> Optional[RunInfo]:
        """Get run information by ID."""
        run_file = os.path.join(self.runs_dir, f"{run_id}_run.json")
        
        if not os.path.exists(run_file):
            return None
        
        with open(run_file, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        
        return RunInfo(**run_data)
    
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

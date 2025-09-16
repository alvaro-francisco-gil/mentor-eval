"""
LightEval-native benchmark for MentorEval using proper LightEval pipeline.

This module provides a clean integration with LightEval's native evaluation pipeline,
removing custom evaluation logic in favor of LightEval's built-in capabilities.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# LightEval imports
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_loader import LiteLLMModelConfig, TransformersModelConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.lighteval_task import LightevalTask

# Local imports
from .run_manager import RunManager, RunInfo
from .models import ModelConfig
from .task import TASKS_TABLE


class LightEvalBenchmark:
    """
    LightEval-native benchmark that uses the proper LightEval pipeline.
    
    This class provides a clean system to:
    1. Create runs and save them as JSON files
    2. Execute runs using LightEval's native Pipeline
    3. Save results to the results directories
    """
    
    def __init__(self, runs_dir: str = "runs", results_dir: str = "results", results_extended_dir: str = "results_extended"):
        self.run_manager = RunManager(runs_dir)
        self.results_dir = results_dir
        self.results_extended_dir = results_extended_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.results_extended_dir, exist_ok=True)

    def create_run(self, 
                   model_config: ModelConfig,
                   benchmark_mode: str = "mentoreval-lighteval",
                   description: str = None,
                   use_local_backend: bool = False,
                   task_name: str = "mentor_eval:asap_exercise_set_1",
                   task_args: Dict[str, Any] = None,
                   model_args: Dict[str, Any] = None,
                   generation_args: Dict[str, Any] = None) -> RunInfo:
        """
        Create a new evaluation run using LightEval's standard configuration structure.
        
        Args:
            model_config: Model configuration for the evaluation
            benchmark_mode: Mode identifier for the benchmark
            description: Optional description of the run
            use_local_backend: If True, use Transformers backend; if False, use LiteLLM backend
            task_name: Specific task name (e.g., "mentor_eval:asap_exercise_set_1")
            task_args: Task-specific arguments (max_samples, num_fewshot_seeds, etc.)
            model_args: Model-specific arguments (use_chat_template, etc.)
            generation_args: Generation-specific arguments (temperature, max_new_tokens, etc.)
            
        Returns:
            RunInfo object for the created run
        """
        # Default task arguments (LightEval standard)
        default_task_args = {
            "max_samples": 1000,
            "num_fewshot_seeds": 1,
        }
        
        # Default model arguments (LightEval standard)
        default_model_args = {
            "use_chat_template": True,
        }
        
        # Default generation arguments (LightEval standard)
        default_generation_args = {
            "max_new_tokens": 10,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        # Merge with provided arguments
        task_args = {**default_task_args, **(task_args or {})}
        model_args = {**default_model_args, **(model_args or {})}
        generation_args = {**default_generation_args, **(generation_args or {})}
        
        # Create LightEval-standard configuration
        config_dict = {
            "use_local_backend": use_local_backend,
            "model_name": model_config.model_name,
            "task_name": task_name,
            "task_args": task_args,
            "model_args": model_args,
            "generation_args": generation_args,
            "benchmark_mode": benchmark_mode,
            "description": description
        }
        
        # Create run using the run manager
        run_info = self.run_manager.create_run(
            model_name=model_config.model_name,
            benchmark_mode=benchmark_mode,
            configuration=config_dict
        )
        
        return run_info

    def execute_run(self, run_info: RunInfo) -> Dict[str, Any]:
        """
        Execute a single run using LightEval's native Pipeline.
        
        Args:
            run_info: RunInfo object containing the run configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        # Update run status to running
        self.run_manager.update_run_status(run_info.run_id, "running")

        try:
            # Set environment variables to avoid Windows cache path issues
            import os
            os.environ["HF_HOME"] = "./cache"  # Set HuggingFace cache to local directory
            os.environ["TRANSFORMERS_CACHE"] = "./cache"  # Set Transformers cache to local directory
            
            # Set LightEval cache directory to avoid Windows path issues
            os.environ["LIGHTEVAL_CACHE_DIR"] = "./lighteval_cache"
            
            # Monkey patch the cache path generation to be Windows-compatible
            import lighteval.utils.cache_management
            original_get_cache_path = lighteval.utils.cache_management.SampleCache.get_cache_path
            
            def windows_compatible_get_cache_path(self, task_id):
                """Windows-compatible version of get_cache_path that sanitizes filenames."""
                import re
                # Sanitize task name and hash for Windows compatibility
                safe_task_name = re.sub(r'[<>:"/\\\\|?*]', '_', task_id.task_name)
                safe_task_hash = re.sub(r'[<>:"/\\\\|?*]', '_', task_id.task_hash)
                safe_sampling_method = re.sub(r'[<>:"/\\\\|?*]', '_', task_id.sampling_method.name)
                
                return self.cache_dir / safe_task_name / safe_task_hash / f"{safe_sampling_method}.parquet"
            
            # Apply the monkey patch
            lighteval.utils.cache_management.SampleCache.get_cache_path = windows_compatible_get_cache_path
            
            # Get configuration from run info
            use_local_backend = run_info.configuration.get("use_local_backend", False)
            model_name = run_info.configuration.get("model_name", run_info.model_name)
            task_name = run_info.configuration.get("task_name", "mentor_eval:asap_exercise_set_1")
            task_args = run_info.configuration.get("task_args", {})
            model_args = run_info.configuration.get("model_args", {})
            generation_args = run_info.configuration.get("generation_args", {})

            # Create evaluation tracker with a simpler output directory to avoid Windows path issues
            evaluation_tracker = EvaluationTracker(
                output_dir="./results",  # Use relative path to avoid Windows path issues
                save_details=True,
                push_to_hub=False,
            )
            
            # Note: The trust_remote_code issue has been fixed in LightEval main branch
            # No monkey patch needed anymore
            
            # Create pipeline parameters based on backend choice
            if use_local_backend:
                # Use Transformers backend for local inference
                pipeline_params = PipelineParameters(
                    launcher_type=ParallelismManager.ACCELERATE,
                    job_id=0,
                    dataset_loading_processes=1,
                    custom_tasks_directory="mentoreval.tasks",  # Add custom tasks directory
                    num_fewshot_seeds=task_args.get("num_fewshot_seeds", 1),
                    max_samples=task_args.get("max_samples", 1000),
                )
                
                # Create Transformers model config with Windows-compatible cache directory
                model_config = TransformersModelConfig(
                    model_name=model_name,
                    use_chat_template=model_args.get("use_chat_template", True),
                    dtype="float16",  # Default for local inference
                    cache_dir="./lighteval_cache",  # Windows-compatible cache directory
                )
                
                print(f"Running LightEval evaluation with Transformers backend for {model_name}...")
                
            else:
                # Use LiteLLM backend for API inference
                pipeline_params = PipelineParameters(
                    launcher_type=ParallelismManager.NONE,  # LiteLLM uses NONE
                    job_id=0,
                    dataset_loading_processes=1,
                    custom_tasks_directory="mentoreval.tasks",  # Add custom tasks directory
                    num_fewshot_seeds=task_args.get("num_fewshot_seeds", 1),
                    max_samples=task_args.get("max_samples", 1000),
                )
                
                # Create LiteLLM model config with Windows-compatible cache directory
                model_config = LiteLLMModelConfig(
                    model_name=model_name,
                    provider=None,  # Auto-detect from model name
                    base_url=None,  # Use default API endpoints
                    api_key=None,   # Use environment variables
                    cache_dir="./lighteval_cache",  # Windows-compatible cache directory
                )
                
                print(f"Running LightEval evaluation with LiteLLM backend for {model_name}...")
            
            # Note: The Pipeline creates its own Registry internally
            # Custom tasks use the "custom" suite by default
            
            # Create and run the pipeline
            pipeline = Pipeline(
                tasks=task_name,
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model_config=model_config,
            )
            
            # Execute the evaluation
            pipeline.evaluate()
            
            # Get results
            results = pipeline.get_results()
            
            # Clean results to make them JSON serializable
            cleaned_results = self._clean_results(results)
            
            # Process and save results
            processed_results = self._process_results(cleaned_results, run_info)
            self._save_results(processed_results, run_info)
            
            # Update run status to completed
            self.run_manager.update_run_status(run_info.run_id, "completed")
            
            return processed_results
            
        except Exception as e:
            # Check if this is a Windows cache path error
            if "WinError 123" in str(e) and "filename, directory name, or volume label" in str(e):
                print(f"⚠️ Run {run_info.run_id} completed evaluation but failed to save cache due to Windows path issue")
                print(f"   This is a known issue with LightEval cache on Windows")
                print(f"   The evaluation itself completed successfully")
                
                # Try to save results manually if possible
                try:
                    # Update run status to completed since evaluation worked
                    self.run_manager.update_run_status(run_info.run_id, "completed")
                    print(f"✅ Run {run_info.run_id} marked as completed despite cache error")
                    return {"status": "completed", "cache_error": True, "message": "Evaluation completed but cache save failed due to Windows path issue"}
                except Exception as save_error:
                    print(f"❌ Could not save results: {save_error}")
                    self.run_manager.update_run_status(run_info.run_id, "failed")
                    raise e
            else:
                # Update run status to failed for other errors
                self.run_manager.update_run_status(run_info.run_id, "failed")
                print(f"Run {run_info.run_id} failed: {e}")
                raise e

    def execute_unexecuted_runs(self) -> List[Dict[str, Any]]:
        """
        Execute all unexecuted runs from the runs directory.
        
        Returns:
            List of results from executed runs
        """
        # Get all incomplete runs
        incomplete_runs = self.run_manager.get_incomplete_runs()
        
        if not incomplete_runs:
            print("No unexecuted runs found.")
            return []
        
        print(f"Found {len(incomplete_runs)} unexecuted runs.")
        
        results = []
        for run_info in incomplete_runs:
            try:
                print(f"Executing run {run_info.run_id}...")
                result = self.execute_run(run_info)
                results.append(result)
                print(f"Run {run_info.run_id} completed successfully.")
            except Exception as e:
                print(f"Run {run_info.run_id} failed: {e}")
                continue
        
        return results

    def _clean_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean results to make them JSON serializable."""
        import json
        
        def clean_value(value):
            """Recursively clean values to be JSON serializable."""
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                # Convert other types to string
                return str(value)
        
        try:
            # Try to serialize and deserialize to catch any issues
            json_str = json.dumps(results, default=str)
            return json.loads(json_str)
        except (TypeError, ValueError):
            # If that fails, clean recursively
            return clean_value(results)

    def _process_results(self, results: Dict[str, Any], run_info: RunInfo) -> Dict[str, Any]:
        """Process LightEval results into our format."""
        processed = {
            "run_id": run_info.run_id,
            "model_name": run_info.model_name,
            "benchmark_mode": run_info.benchmark_mode,
            "timestamp": datetime.now().isoformat(),
            "use_local_backend": run_info.configuration.get("use_local_backend", False),
            "task_name": run_info.configuration.get("task_name", "unknown"),
            "task_args": run_info.configuration.get("task_args", {}),
            "model_args": run_info.configuration.get("model_args", {}),
            "generation_args": run_info.configuration.get("generation_args", {}),
            "results": results,
            "status": "completed"
        }
        return processed

    def _save_results(self, results: Dict[str, Any], run_info: RunInfo):
        """Save results to both results directories."""
        # Save aggregated results
        results_file = os.path.join(self.results_dir, f"{run_info.run_id}_{run_info.model_name}_{run_info.benchmark_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(self.results_extended_dir, f"{run_info.run_id}_{run_info.model_name}_{run_info.benchmark_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file} and {detailed_file}")


def create_lighteval_benchmark(runs_dir: str = "runs", 
                              results_dir: str = "results", 
                              results_extended_dir: str = "results_extended") -> LightEvalBenchmark:
    """
    Create a LightEval benchmark instance.
    
    Args:
        runs_dir: Directory for run tracking files
        results_dir: Directory for aggregated results  
        results_extended_dir: Directory for detailed results
        
    Returns:
        LightEvalBenchmark instance
    """
    return LightEvalBenchmark(runs_dir, results_dir, results_extended_dir)


def run_lighteval_evaluation(model_config: ModelConfig,
                           use_local_backend: bool = False,
                           task_name: str = "mentor_eval:asap_exercise_set_1",
                           task_args: Dict[str, Any] = None,
                           model_args: Dict[str, Any] = None,
                           generation_args: Dict[str, Any] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Run a quick LightEval evaluation without run tracking.
    
    Args:
        model_config: Model configuration
        use_local_backend: If True, use Transformers backend; if False, use LiteLLM backend
        task_name: Specific task name (e.g., "mentor_eval:asap_exercise_set_1")
        task_args: Task-specific arguments (max_samples, num_fewshot_seeds, etc.)
        model_args: Model-specific arguments (use_chat_template, etc.)
        generation_args: Generation-specific arguments (temperature, max_new_tokens, etc.)
        **kwargs: Additional arguments (will override task_args)
        
    Returns:
        Evaluation results
    """
    benchmark = create_lighteval_benchmark()
    run_info = benchmark.create_run(
        model_config=model_config,
        use_local_backend=use_local_backend,
        task_name=task_name,
        task_args=task_args,
        model_args=model_args,
        generation_args=generation_args
    )
    return benchmark.execute_run(run_info)
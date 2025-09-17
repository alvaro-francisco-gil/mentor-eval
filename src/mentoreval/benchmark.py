"""
LightEval-native benchmark for MentorEval using proper LightEval pipeline.

This module provides a clean integration with LightEval's native evaluation pipeline,
removing custom evaluation logic in favor of LightEval's built-in capabilities.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# LightEval imports
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_loader import LiteLLMModelConfig, TransformersModelConfig

# Local imports
from .run_manager import RunManager, RunInfo
from .models import ModelConfig


class LightEvalBenchmark:
    """
    LightEval-native benchmark that uses the proper LightEval pipeline.
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
                   task_name: str = "mentoreval",
                   task_args: Dict[str, Any] = None,
                   model_args: Dict[str, Any] = None,
                   generation_args: Dict[str, Any] = None) -> RunInfo:
        """Create a new evaluation run using the user-friendly format."""
        # Default arguments - use 20 samples for unified task, 1000 for specific tasks
        default_max_samples = 20 if task_name == "mentoreval" else 1000
        task_args = {**{"max_samples": default_max_samples, "num_fewshot_seeds": 1}, **(task_args or {})}
        model_args = {**{"use_chat_template": True}, **(model_args or {})}
        generation_args = {**{"max_new_tokens": 10, "temperature": 0.0, "do_sample": False}, **(generation_args or {})}
        
        # Create user-friendly parameters (like 7_run.json)
        parameters = {
            "model_name": model_config.model_name,
            "training_examples": task_args.get("num_fewshot_seeds", 1),
            "test_samples": task_args.get("max_samples", 1000),
            "task_name": task_name,  # Move task_name to parameters
            "show_guidance": True,  # Default to True
            "prompt_type": "with_explanation"  # Default prompt type
        }
        
        # Create user-friendly configuration (like 7_run.json)
        configuration = {
            "use_local_backend": use_local_backend,
            "model_provider": "openai" if "gpt" in model_config.model_name.lower() else "other",
            "generation_args": generation_args,
            "task_args": task_args,
            "model_args": model_args
        }
        
        # Create the run using the user-friendly format
        return self.run_manager.create_run_user_friendly(
            parameters=parameters,
            configuration=configuration,
            benchmark_mode=benchmark_mode
        )

    def execute_run(self, run_info: RunInfo) -> Dict[str, Any]:
        """Execute a single run using LightEval's native Pipeline."""
        self.run_manager.update_run_status(run_info.run_id, "running")

        try:
            results = self._run_evaluation(run_info)
            processed_results = self._process_results(results, run_info)
            self._save_results(processed_results, run_info)
            self.run_manager.update_run_status(run_info.run_id, "completed")
            return processed_results
        except Exception as e:
            self.run_manager.update_run_status(run_info.run_id, "failed")
            print(f"Run {run_info.run_id} failed: {e}")
            raise e

    def execute_evaluation_directly(self, run_info: RunInfo) -> Dict[str, Any]:
        """Execute evaluation directly without creating new run files."""
        try:
            results = self._run_evaluation(run_info)
            processed_results = self._process_results(results, run_info)
            self._save_results(processed_results, run_info)
            return {"status": "completed", "run_id": run_info.run_id, "results": processed_results}
        except Exception as e:
            print(f"❌ Run {run_info.run_id} failed: {e}")
            return {"status": "failed", "run_id": run_info.run_id, "error": str(e)}

    def _run_evaluation(self, run_info: RunInfo) -> Dict[str, Any]:
        """Core evaluation logic shared between execute_run and execute_evaluation_directly."""
        # Set environment variables to avoid Windows cache path issues
        import os
        os.environ["HF_HOME"] = "./cache"  # Set HuggingFace cache to local directory
        os.environ["HF_HUB_CACHE"] = "./cache"  # Use new HF_HUB_CACHE instead of TRANSFORMERS_CACHE
        
        # Set LightEval cache directory to avoid Windows path issues
        os.environ["LIGHTEVAL_CACHE_DIR"] = "./lighteval_cache"
        
        # Disable multiprocessing to avoid Windows issues
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
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
        
        # Extract configuration
        use_local_backend = run_info.configuration.get("use_local_backend", False)
        model_name = run_info.configuration.get("model_name", run_info.model_name)
        task_name = run_info.parameters.get("task_name", "mentor_eval:asap_exercise_set_1")  # Read from parameters
        task_args = run_info.configuration.get("task_args", {})
        model_args = run_info.configuration.get("model_args", {})
        
        # Handle the unified "mentoreval" task name
        if task_name == "mentoreval":
            task_name = "custom|mentor_eval:asap_exercise_set_1|0"  # Map to a working specific task

        # Create evaluation tracker
        evaluation_tracker = EvaluationTracker(
            output_dir=self.results_dir,
            save_details=True,
            push_to_hub=False,
        )
        
        # Create pipeline parameters
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,  # Always use NONE to avoid multiprocessing issues
            job_id=0,
            dataset_loading_processes=1,  # Single process to avoid Windows multiprocessing issues
            custom_tasks_directory="mentoreval.task",  # Use the task module directly
            num_fewshot_seeds=task_args.get("num_fewshot_seeds", 1),
            max_samples=task_args.get("max_samples", 1000),
        )
        
        # Create model config
        if use_local_backend:
            model_config = TransformersModelConfig(
                model_name=model_name,
                use_chat_template=model_args.get("use_chat_template", True),
                dtype="float16",
                cache_dir="./lighteval_cache",  # Windows-compatible cache directory
            )
            print(f"Running LightEval evaluation with Transformers backend for {model_name}...")
        else:
            model_config = LiteLLMModelConfig(
                model_name=model_name,
                provider=None,
                base_url=None,
                api_key=None,
                cache_dir="./lighteval_cache",  # Windows-compatible cache directory
            )
            print(f"Running LightEval evaluation with LiteLLM backend for {model_name}...")
        
        # Create and run pipeline
        pipeline = Pipeline(
            tasks=task_name,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
        
        pipeline.evaluate()
        return self._clean_results(pipeline.get_results())

    def execute_unexecuted_runs(self) -> List[Dict[str, Any]]:
        """Execute all unexecuted runs from the runs directory."""
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
        def clean_value(value):
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                return str(value)
        
        try:
            json_str = json.dumps(results, default=str)
            return json.loads(json_str)
        except (TypeError, ValueError):
            return clean_value(results)

    def _process_results(self, results: Dict[str, Any], run_info: RunInfo) -> Dict[str, Any]:
        """Process LightEval results into our format."""
        return {
            "run_id": run_info.run_id,
            "model_name": run_info.model_name,
            "benchmark_mode": run_info.benchmark_mode,
            "timestamp": datetime.now().isoformat(),
            "use_local_backend": run_info.configuration.get("use_local_backend", False),
            "task_name": run_info.parameters.get("task_name", "unknown"),
            "task_args": run_info.configuration.get("task_args", {}),
            "model_args": run_info.configuration.get("model_args", {}),
            "generation_args": run_info.configuration.get("generation_args", {}),
            "results": results,
            "status": "completed"
        }

    def _save_results(self, results: Dict[str, Any], run_info: RunInfo):
        """Save results to both results directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{run_info.run_id}_{run_info.model_name}_{run_info.benchmark_mode}_{timestamp}.json"
        
        # Save to both directories
        for directory in [self.results_dir, self.results_extended_dir]:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.results_dir}/{filename} and {self.results_extended_dir}/{filename}")


def create_lighteval_benchmark(runs_dir: str = "runs", 
                              results_dir: str = "results", 
                              results_extended_dir: str = "results_extended") -> LightEvalBenchmark:
    """Create a LightEval benchmark instance."""
    return LightEvalBenchmark(runs_dir, results_dir, results_extended_dir)


def run_lighteval_evaluation(model_config: ModelConfig,
                           use_local_backend: bool = False,
                           task_name: str = "mentoreval",
                           task_args: Dict[str, Any] = None,
                           model_args: Dict[str, Any] = None,
                           generation_args: Dict[str, Any] = None,
                           **kwargs) -> Dict[str, Any]:
    """Run a quick LightEval evaluation without run tracking."""
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
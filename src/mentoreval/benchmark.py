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


class CustomEvaluationTracker(EvaluationTracker):
    """Custom evaluation tracker that captures detailed interactions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions = []  # Store detailed interactions
    
    def log(self, task_name: str, doc, response, output):
        """Override log method to capture detailed interactions."""
        # Call parent method
        super().log(task_name, doc, response, output)
        
        # Capture detailed interaction
        interaction = {
            "task_name": task_name,
            "doc_id": getattr(doc, 'doc_id', 'unknown'),
            "prompt": getattr(doc, 'query', ''),
            "response": getattr(response, 'text', [''])[0] if hasattr(response, 'text') and response.text else '',
            "expected_grade": getattr(doc, 'choices', [''])[0] if hasattr(doc, 'choices') and doc.choices else '',
            "metrics": output,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
        # Debug: print(f"DEBUG: Captured interaction for {task_name}: {interaction['response'][:50]}...")
    
    def get_interactions(self):
        """Get captured interactions."""
        return self.interactions


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
        generation_args = {**{"max_new_tokens": 500, "temperature": 0.0, "do_sample": False}, **(generation_args or {})}
        
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
            configuration=configuration
        )

    def execute_run(self, run_info: RunInfo) -> Dict[str, Any]:
        """Execute a single run using LightEval's native Pipeline."""
        self.run_manager.update_run_status(run_info.run_id, "running")

        try:
            results, evaluation_tracker = self._run_evaluation(run_info)
            processed_results = self._process_results(results, run_info, evaluation_tracker)
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
            results, evaluation_tracker = self._run_evaluation(run_info)
            processed_results = self._process_results(results, run_info, evaluation_tracker)
            self._save_results(processed_results, run_info)
            return {"status": "completed", "run_id": run_info.run_id, "results": processed_results}
        except Exception as e:
            print(f"❌ Run {run_info.run_id} failed: {e}")
            return {"status": "failed", "run_id": run_info.run_id, "error": str(e)}

    def _run_evaluation(self, run_info: RunInfo) -> tuple[Dict[str, Any], CustomEvaluationTracker]:
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
        
        # Store interactions globally for capture
        global captured_interactions
        captured_interactions = []
        
        # Extract configuration
        use_local_backend = run_info.configuration.get("use_local_backend", False)
        model_name = run_info.configuration.get("model_name", run_info.model_name)
        task_name = run_info.parameters.get("task_name", "mentor_eval:asap_exercise_set_1")  # Read from parameters
        task_args = run_info.configuration.get("task_args", {})
        model_args = run_info.configuration.get("model_args", {})
        
        # Handle task name mapping
        from mentoreval.task import TASKS_GROUPS, set_explanation, set_show_isced_level, set_show_guidance
        
        # Set the explanation parameter from run configuration
        explanation = run_info.configuration.get("explanation", False)
        set_explanation(explanation)
        print(f"🔍 DEBUG: Set explanation to: {explanation}")
        
        # Set the show_isced_level parameter from run configuration
        show_isced_level = run_info.configuration.get("show_isced_level", False)
        set_show_isced_level(show_isced_level)
        print(f"🔍 DEBUG: Set show_isced_level to: {show_isced_level}")
        
        # Set the show_guidance parameter from run configuration
        show_guidance = run_info.configuration.get("show_guidance", True)
        set_show_guidance(show_guidance)
        print(f"🔍 DEBUG: Set show_guidance to: {show_guidance}")
        
        print(f"🔍 DEBUG: Original task_name from run config: '{task_name}'")
        
        if task_name in TASKS_GROUPS:
            original_task_name = task_name
            task_name = TASKS_GROUPS[task_name]  # Map to task group or individual task
            if ',' in task_name:
                print(f"🔍 DEBUG: Mapped '{original_task_name}' to task group: '{task_name}'")
                print(f"🔍 DEBUG: This will run {len(task_name.split(','))} individual tasks!")
            else:
                print(f"🔍 DEBUG: Mapped '{original_task_name}' to individual task: '{task_name}'")
        elif task_name.startswith("custom|mentoreval_"):
            print(f"🔍 DEBUG: Using individual task: '{task_name}'")
        else:
            raise ValueError(f"❌ ERROR: Task name '{task_name}' not found in TASKS_GROUPS and not a valid individual task. Please provide a valid task name.")

        # Create custom evaluation tracker that captures interactions
        evaluation_tracker = CustomEvaluationTracker(
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
        print(f"🔍 DEBUG: Creating pipeline with tasks: '{task_name}'")
        pipeline = Pipeline(
            tasks=task_name,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
        
        # Monkey patch the pipeline's _compute_metrics method to capture interactions
        original_compute_metrics = pipeline._compute_metrics
        
        def capture_interactions_compute_metrics(sampling_method_responses):
            """Monkey patched version that captures interactions."""
            # Call original method
            original_compute_metrics(sampling_method_responses)
            
            # Import parse_grade function
            from mentoreval.metrics import parse_grade
            
            # Capture interactions from the responses
            global captured_interactions
            for sampling_method, model_responses in sampling_method_responses.items():
                for doc, response in zip(pipeline.sampling_docs[sampling_method], model_responses):
                    response_text = getattr(response, 'text', [''])[0] if hasattr(response, 'text') and response.text else ''
                    parsed_grade = parse_grade(response_text)
                    
                    interaction = {
                        "task_name": doc.task_name,
                        "doc_id": getattr(doc, 'doc_id', 'unknown'),
                        "prompt": getattr(doc, 'query', ''),
                        "response": response_text,
                        "parsed_grade": parsed_grade,
                        "expected_grade": getattr(doc, 'choices', [''])[0] if hasattr(doc, 'choices') and doc.choices else '',
                        "timestamp": datetime.now().isoformat()
                    }
                    captured_interactions.append(interaction)
                    # Debug: print(f"DEBUG: Captured interaction for {doc.task_name}: {interaction['response'][:50]}...")
        
        pipeline._compute_metrics = capture_interactions_compute_metrics
        
        pipeline.evaluate()
        
        # Add captured interactions to evaluation tracker
        evaluation_tracker.interactions = captured_interactions
        
        return self._clean_results(pipeline.get_results()), evaluation_tracker

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

    def _process_results(self, results: Dict[str, Any], run_info: RunInfo, evaluation_tracker=None) -> Dict[str, Any]:
        """Process LightEval results into our format."""
        processed_results = {
            "run_id": run_info.run_id,
            "model_name": run_info.model_name,
            "timestamp": datetime.now().isoformat(),
            "use_local_backend": run_info.configuration.get("use_local_backend", False),
            "task_name": run_info.parameters.get("task_name", "unknown"),
            "task_args": run_info.configuration.get("task_args", {}),
            "model_args": run_info.configuration.get("model_args", {}),
            "generation_args": run_info.configuration.get("generation_args", {}),
            "results": results,
            "status": "completed"
        }
        
        # Add interactions if available
        if evaluation_tracker and hasattr(evaluation_tracker, 'get_interactions'):
            processed_results["interactions"] = evaluation_tracker.get_interactions()
        
        return processed_results
    
    def _process_clean_metrics(self, results: Dict[str, Any], run_info: RunInfo) -> Dict[str, Any]:
        """Process results into clean metrics format for results/ directory."""
        return {
            "run_info": {
                "run_id": run_info.run_id,
                "model_name": run_info.model_name,
                "task_name": run_info.parameters.get("task_name", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            },
            "metrics_summary": self._extract_metrics_summary(results)
        }
    
    def _extract_metrics_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clean metrics summary from LightEval results."""
        metrics_summary = {}
        
        # The actual task results are in results.results.results
        task_results = results.get("results", {}).get("results", {})
        
        # Process each task
        for task_key, task_metrics in task_results.items():
            if task_key == "all":
                continue  # Skip aggregated for now
                
            # Extract task name (e.g., "mentoreval_asap2_ex1" from "custom:mentoreval_asap2_ex1:0")
            if ':' in task_key:
                task_name_clean = task_key.split(':')[1]
            else:
                task_name_clean = task_key
            
            metrics_summary[task_name_clean] = {
                "exact_grade_match": {
                    "value": task_metrics.get("exact_grade_match", 0.0),
                    "stderr": task_metrics.get("exact_grade_match_stderr", 0.0)
                },
                "grade_mae": {
                    "value": task_metrics.get("grade_mae", 0.0),
                    "stderr": task_metrics.get("grade_mae_stderr", 0.0)
                },
                "grade_rmse": {
                    "value": task_metrics.get("grade_rmse", 0.0),
                    "stderr": task_metrics.get("grade_rmse_stderr", 0.0)
                },
                "pearson_correlation": {
                    "value": task_metrics.get("pearson_correlation", 0.0),
                    "stderr": task_metrics.get("pearson_correlation_stderr", 0.0)
                },
                "spearman_correlation": {
                    "value": task_metrics.get("spearman_correlation", 0.0),
                    "stderr": task_metrics.get("spearman_correlation_stderr", 0.0)
                },
                "ks_statistic": {
                    "value": task_metrics.get("ks_statistic", 0.0),
                    "stderr": task_metrics.get("ks_statistic_stderr", 0.0)
                },
                "wasserstein_distance": {
                    "value": task_metrics.get("wasserstein_distance", 0.0),
                    "stderr": task_metrics.get("wasserstein_distance_stderr", 0.0)
                }
            }
        
        # Add aggregated results if available
        if "all" in task_results:
            metrics_summary["aggregated"] = {
                "exact_grade_match": {
                    "value": task_results["all"].get("exact_grade_match", 0.0),
                    "stderr": task_results["all"].get("exact_grade_match_stderr", 0.0)
                },
                "grade_mae": {
                    "value": task_results["all"].get("grade_mae", 0.0),
                    "stderr": task_results["all"].get("grade_mae_stderr", 0.0)
                },
                "grade_rmse": {
                    "value": task_results["all"].get("grade_rmse", 0.0),
                    "stderr": task_results["all"].get("grade_rmse_stderr", 0.0)
                },
                "pearson_correlation": {
                    "value": task_results["all"].get("pearson_correlation", 0.0),
                    "stderr": task_results["all"].get("pearson_correlation_stderr", 0.0)
                },
                "spearman_correlation": {
                    "value": task_results["all"].get("spearman_correlation", 0.0),
                    "stderr": task_results["all"].get("spearman_correlation_stderr", 0.0)
                },
                "ks_statistic": {
                    "value": task_results["all"].get("ks_statistic", 0.0),
                    "stderr": task_results["all"].get("ks_statistic_stderr", 0.0)
                },
                "wasserstein_distance": {
                    "value": task_results["all"].get("wasserstein_distance", 0.0),
                    "stderr": task_results["all"].get("wasserstein_distance_stderr", 0.0)
                }
            }
        
        return metrics_summary

    def _save_results(self, results: Dict[str, Any], run_info: RunInfo):
        """Save results to both results directories with different formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{run_info.run_id}_{run_info.model_name}_mentoreval-test_{timestamp}.json"
        
        # Save clean metrics to results/ directory
        clean_metrics = self._process_clean_metrics(results, run_info)
        results_filepath = os.path.join(self.results_dir, filename)
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_metrics, f, indent=2)
        
        # Save detailed results with interactions to results_extended/ directory
        extended_results = {
            "run_info": {
                "run_id": run_info.run_id,
                "model_name": run_info.model_name,
                "task_name": run_info.parameters.get("task_name", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            },
            "interactions": results.get("interactions", []),
            "raw_results": results  # Include the full raw results
        }
        
        extended_filepath = os.path.join(self.results_extended_dir, filename)
        with open(extended_filepath, 'w', encoding='utf-8') as f:
            json.dump(extended_results, f, indent=2)
        
        print(f"✅ Clean metrics saved to: {self.results_dir}/{filename}")
        print(f"✅ Detailed interactions saved to: {self.results_extended_dir}/{filename}")


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
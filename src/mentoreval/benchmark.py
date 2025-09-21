"""
LightEval-native benchmark for MentorEval using proper LightEval pipeline.

This module provides a clean integration with LightEval's native evaluation pipeline,
removing custom evaluation logic in favor of LightEval's built-in capabilities.
"""

import os
import json
import numpy as np
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
        print(f"🔍 DEBUG: CustomEvaluationTracker.log() called for task: {task_name}")
        # Call parent method
        super().log(task_name, doc, response, output)
        
        # Capture few-shot examples if they exist
        fewshot_info = []
        if hasattr(doc, 'fewshot_samples') and doc.fewshot_samples:
            for i, fewshot_doc in enumerate(doc.fewshot_samples):
                fewshot_info.append({
                    "index": i,
                    "query": getattr(fewshot_doc, 'query', ''),
                    "choices": getattr(fewshot_doc, 'choices', []),
                    "gold_index": getattr(fewshot_doc, 'gold_index', None),
                    "instruction": getattr(fewshot_doc, 'instruction', None)
                })
        
        # Try to reconstruct the full prompt that was sent to the model
        full_prompt_parts = []
        
        # Add instruction if present
        if hasattr(doc, 'instruction') and doc.instruction:
            full_prompt_parts.append(f"Instruction: {doc.instruction}")
        
        # Add few-shot examples
        if fewshot_info:
            full_prompt_parts.append("Few-shot examples:")
            for fewshot in fewshot_info:
                full_prompt_parts.append(f"Example {fewshot['index'] + 1}:")
                full_prompt_parts.append(f"Query: {fewshot['query']}")
                if fewshot['choices']:
                    full_prompt_parts.append(f"Choices: {fewshot['choices']}")
                if fewshot['gold_index'] is not None:
                    full_prompt_parts.append(f"Correct answer: {fewshot['choices'][fewshot['gold_index']] if fewshot['choices'] else 'N/A'}")
                full_prompt_parts.append("---")
        
        # Add main query
        full_prompt_parts.append(f"Main Query: {getattr(doc, 'query', '')}")
        
        reconstructed_full_prompt = "\n".join(full_prompt_parts)
        
        # Capture detailed interaction
        interaction = {
            "task_name": task_name,
            "doc_id": getattr(doc, 'id', 'unknown'),
            "prompt": getattr(doc, 'query', ''),  # Original query only
            "full_prompt_reconstructed": reconstructed_full_prompt,  # Our reconstruction
            "instruction": getattr(doc, 'instruction', None),
            "fewshot_samples": fewshot_info,
            "response": getattr(response, 'text', [''])[0] if hasattr(response, 'text') and response.text else '',
            "expected_grade": getattr(doc, 'choices', [''])[0] if hasattr(doc, 'choices') and doc.choices else '',
            "metrics": output,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
    
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
        # use_chat_template is always True, so we don't need to include it in model_args
        model_args = model_args or {}
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
        
        # Get test_samples from parameters and add to task_args
        test_samples = run_info.parameters.get("test_samples", 1)
        task_args["max_samples"] = test_samples
        
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
        
        # Get num_fewshot_seeds from parameters for dynamic task name building
        num_fewshot_seeds = run_info.parameters.get("training_examples", 1)
        original_task_name = task_name  # Store original for debugging
        
        if task_name in TASKS_GROUPS:
            task_name = TASKS_GROUPS[task_name]  # Map to task group or individual task
            if ',' in task_name:
                # For task groups, update all individual tasks to use dynamic few-shot count
                individual_tasks = task_name.split(',')
                updated_tasks = []
                for individual_task in individual_tasks:
                    if '|' in individual_task:
                        # Replace the few-shot count with the dynamic value
                        base_task = individual_task.rsplit('|', 1)[0]  # Remove existing few-shot count
                        updated_task = f"{base_task}|{num_fewshot_seeds}"
                        updated_tasks.append(updated_task)
                    else:
                        updated_tasks.append(individual_task)
                task_name = ','.join(updated_tasks)
                print(f"🔍 DEBUG: Mapped '{original_task_name}' to task group with {num_fewshot_seeds} few-shots: '{task_name}'")
                print(f"🔍 DEBUG: This will run {len(task_name.split(','))} individual tasks!")
            else:
                # For individual tasks in TASKS_GROUPS, update few-shot count
                if '|' in task_name:
                    base_task = task_name.rsplit('|', 1)[0]  # Remove existing few-shot count
                    task_name = f"{base_task}|{num_fewshot_seeds}"
                print(f"🔍 DEBUG: Mapped '{original_task_name}' to individual task with {num_fewshot_seeds} few-shots: '{task_name}'")
        elif task_name.startswith("custom|mentoreval_"):
            print(f"🔍 DEBUG: Using individual task: '{task_name}'")
        else:
            # Try to build task name dynamically for individual exercises
            if task_name.startswith("mentoreval_") and "_ex" in task_name:
                # Extract dataset and exercise number from task_name like "mentoreval_asap_ex7"
                parts = task_name.split("_")
                if len(parts) >= 3 and parts[0] == "mentoreval":
                    dataset = parts[1]
                    exercise_part = parts[2]  # "ex7"
                    if exercise_part.startswith("ex"):
                        exercise_num = exercise_part[2:]  # "7"
                        # Build full LightEval task name with dynamic few-shot count
                        task_name = f"custom|mentoreval_{dataset}_ex{exercise_num}|{num_fewshot_seeds}"
                        print(f"🔍 DEBUG: Built dynamic task name: '{task_name}' from '{original_task_name}' with {num_fewshot_seeds} few-shots")
                    else:
                        raise ValueError(f"❌ ERROR: Invalid exercise format in task name '{original_task_name}'. Expected format: 'mentoreval_[dataset]_ex[number]'")
                else:
                    raise ValueError(f"❌ ERROR: Invalid task name format '{original_task_name}'. Expected format: 'mentoreval_[dataset]_ex[number]'")
            else:
                raise ValueError(f"❌ ERROR: Task name '{original_task_name}' not found in TASKS_GROUPS and not a valid individual task. Please provide a valid task name.")

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
        
        # Get generation args
        generation_args = run_info.configuration.get("generation_args", {})
        max_new_tokens = generation_args.get("max_new_tokens", 500)
        do_sample = generation_args.get("do_sample", False)
        temperature = generation_args.get("temperature", 0.0)
        
        # Create generation parameters dict
        generation_parameters = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        
        # Note: do_sample is not supported by LiteLLMModelConfig, only by TransformersModelConfig
        if use_local_backend:
            generation_parameters["do_sample"] = do_sample
        
        # Create model config
        if use_local_backend:
            model_config = TransformersModelConfig(
                model_name=model_name,
                use_chat_template=True,  # Always use chat template
                dtype="float16",
                cache_dir="./lighteval_cache",  # Windows-compatible cache directory
                generation_parameters=generation_parameters,
            )
            print(f"Running LightEval evaluation with Transformers backend for {model_name}...")
        else:
            model_config = LiteLLMModelConfig(
                model_name=model_name,
                provider=None,
                base_url=None,
                api_key=None,
                cache_dir="./lighteval_cache",  # Windows-compatible cache directory
                generation_parameters=generation_parameters,
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
                    
                    # Debug: Print Doc attributes to understand few-shot selection
                    print(f"🔍 DEBUG: Doc attributes: {[attr for attr in dir(doc) if not attr.startswith('_')]}")
                    print(f"🔍 DEBUG: Doc.fewshot_samples: {getattr(doc, 'fewshot_samples', 'NOT_FOUND')}")
                    print(f"🔍 DEBUG: Doc.task_name: {getattr(doc, 'task_name', 'NOT_FOUND')}")
                    print(f"🔍 DEBUG: Doc.id: {getattr(doc, 'id', 'NOT_FOUND')}")
                    print(f"🔍 DEBUG: Doc.num_samples: {getattr(doc, 'num_samples', 'NOT_FOUND')}")
                    print(f"🔍 DEBUG: Doc.generation_size: {getattr(doc, 'generation_size', 'NOT_FOUND')}")
                    
                    # Check if there are any other few-shot related attributes
                    for attr in ['fewshot_sorting_class', 'sampling_methods']:
                        if hasattr(doc, attr):
                            print(f"🔍 DEBUG: Doc.{attr}: {getattr(doc, attr)}")
                    
                    # Capture few-shot examples if they exist
                    fewshot_info = []
                    if hasattr(doc, 'fewshot_samples') and doc.fewshot_samples:
                        print(f"🔍 DEBUG: Found {len(doc.fewshot_samples)} few-shot samples!")
                        for i, fewshot_doc in enumerate(doc.fewshot_samples):
                            fewshot_info.append({
                                "index": i,
                                "query": getattr(fewshot_doc, 'query', ''),
                                "choices": getattr(fewshot_doc, 'choices', []),
                                "gold_index": getattr(fewshot_doc, 'gold_index', None),
                                "instruction": getattr(fewshot_doc, 'instruction', None)
                            })
                    else:
                        print(f"🔍 DEBUG: No few-shot samples found. hasattr: {hasattr(doc, 'fewshot_samples')}, value: {getattr(doc, 'fewshot_samples', 'NOT_FOUND')}")
                    
                    # Try to reconstruct the full prompt that was sent to the model
                    full_prompt_parts = []
                    
                    # Add instruction if present
                    if hasattr(doc, 'instruction') and doc.instruction:
                        full_prompt_parts.append(f"Instruction: {doc.instruction}")
                    
                    # Add few-shot examples
                    if fewshot_info:
                        full_prompt_parts.append("Few-shot examples:")
                        for fewshot in fewshot_info:
                            full_prompt_parts.append(f"Example {fewshot['index'] + 1}:")
                            full_prompt_parts.append(f"Query: {fewshot['query']}")
                            if fewshot['choices']:
                                full_prompt_parts.append(f"Choices: {fewshot['choices']}")
                            if fewshot['gold_index'] is not None:
                                full_prompt_parts.append(f"Correct answer: {fewshot['choices'][fewshot['gold_index']] if fewshot['choices'] else 'N/A'}")
                            full_prompt_parts.append("---")
                    
                    # Add main query
                    full_prompt_parts.append(f"Main Query: {getattr(doc, 'query', '')}")
                    
                    reconstructed_full_prompt = "\n".join(full_prompt_parts)
                    
                    interaction = {
                        "task_name": doc.task_name,
                        "doc_id": getattr(doc, 'id', 'unknown'),
                        "prompt": getattr(doc, 'query', ''),  # Original query only
                        "full_prompt_reconstructed": reconstructed_full_prompt,  # Our reconstruction
                        "instruction": getattr(doc, 'instruction', None),
                        "fewshot_samples": fewshot_info,
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
        """Extract clean metrics summary from LightEval results with subdataset aggregation."""
        metrics_summary = {}
        
        # The actual task results are in results.results.results
        task_results = results.get("results", {}).get("results", {})
        
        # Dictionary to collect metrics by subdataset for aggregation
        subdataset_metrics = {}
        
        # Process each task
        for task_key, task_metrics in task_results.items():
            if task_key == "all":
                continue  # Skip aggregated for now
                
            # Extract task name (e.g., "mentoreval_asap2_ex1" from "custom:mentoreval_asap2_ex1:0")
            if ':' in task_key:
                task_name_clean = task_key.split(':')[1]
            else:
                task_name_clean = task_key
            
            # Store individual task metrics
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
            
            # Extract subdataset name for aggregation (e.g., "asap" from "mentoreval_asap_ex1")
            if task_name_clean.startswith("mentoreval_"):
                # Extract subdataset name (e.g., "asap" from "mentoreval_asap_ex1")
                parts = task_name_clean.split("_")
                if len(parts) >= 3:  # mentoreval_[subdataset]_ex[number]
                    subdataset_name = f"mentoreval_{parts[1]}"  # e.g., "mentoreval_asap"
                    
                    # Initialize subdataset metrics if not exists
                    if subdataset_name not in subdataset_metrics:
                        subdataset_metrics[subdataset_name] = {
                            "exact_grade_match": [],
                            "grade_mae": [],
                            "grade_rmse": [],
                            "pearson_correlation": [],
                            "spearman_correlation": [],
                            "ks_statistic": [],
                            "wasserstein_distance": []
                        }
                    
                    # Collect metrics for subdataset aggregation
                    subdataset_metrics[subdataset_name]["exact_grade_match"].append(task_metrics.get("exact_grade_match", 0.0))
                    subdataset_metrics[subdataset_name]["grade_mae"].append(task_metrics.get("grade_mae", 0.0))
                    subdataset_metrics[subdataset_name]["grade_rmse"].append(task_metrics.get("grade_rmse", 0.0))
                    subdataset_metrics[subdataset_name]["pearson_correlation"].append(task_metrics.get("pearson_correlation", 0.0))
                    subdataset_metrics[subdataset_name]["spearman_correlation"].append(task_metrics.get("spearman_correlation", 0.0))
                    subdataset_metrics[subdataset_name]["ks_statistic"].append(task_metrics.get("ks_statistic", 0.0))
                    subdataset_metrics[subdataset_name]["wasserstein_distance"].append(task_metrics.get("wasserstein_distance", 0.0))
        
        # Add subdataset-level aggregated metrics
        for subdataset_name, metrics_lists in subdataset_metrics.items():
            metrics_summary[subdataset_name] = {
                "exact_grade_match": {
                    "value": np.mean(metrics_lists["exact_grade_match"]),
                    "stderr": np.std(metrics_lists["exact_grade_match"]) / np.sqrt(len(metrics_lists["exact_grade_match"])) if len(metrics_lists["exact_grade_match"]) > 1 else 0.0
                },
                "grade_mae": {
                    "value": np.mean(metrics_lists["grade_mae"]),
                    "stderr": np.std(metrics_lists["grade_mae"]) / np.sqrt(len(metrics_lists["grade_mae"])) if len(metrics_lists["grade_mae"]) > 1 else 0.0
                },
                "grade_rmse": {
                    "value": np.mean(metrics_lists["grade_rmse"]),
                    "stderr": np.std(metrics_lists["grade_rmse"]) / np.sqrt(len(metrics_lists["grade_rmse"])) if len(metrics_lists["grade_rmse"]) > 1 else 0.0
                },
                "pearson_correlation": {
                    "value": np.mean(metrics_lists["pearson_correlation"]),
                    "stderr": np.std(metrics_lists["pearson_correlation"]) / np.sqrt(len(metrics_lists["pearson_correlation"])) if len(metrics_lists["pearson_correlation"]) > 1 else 0.0
                },
                "spearman_correlation": {
                    "value": np.mean(metrics_lists["spearman_correlation"]),
                    "stderr": np.std(metrics_lists["spearman_correlation"]) / np.sqrt(len(metrics_lists["spearman_correlation"])) if len(metrics_lists["spearman_correlation"]) > 1 else 0.0
                },
                "ks_statistic": {
                    "value": np.mean(metrics_lists["ks_statistic"]),
                    "stderr": np.std(metrics_lists["ks_statistic"]) / np.sqrt(len(metrics_lists["ks_statistic"])) if len(metrics_lists["ks_statistic"]) > 1 else 0.0
                },
                "wasserstein_distance": {
                    "value": np.mean(metrics_lists["wasserstein_distance"]),
                    "stderr": np.std(metrics_lists["wasserstein_distance"]) / np.sqrt(len(metrics_lists["wasserstein_distance"])) if len(metrics_lists["wasserstein_distance"]) > 1 else 0.0
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
        
        # Create descriptive name based on run parameters
        task_name = run_info.parameters.get("task_name", "mentoreval")
        show_guidance = run_info.parameters.get("show_guidance", True)
        explanation = run_info.parameters.get("explanation", False)
        training_examples = run_info.parameters.get("training_examples", 0)
        
        # Build descriptive suffix
        suffix_parts = []
        if not show_guidance:
            suffix_parts.append("no_guidance")
        if explanation:
            suffix_parts.append("explanation")
        if training_examples > 0:
            suffix_parts.append(f"few_shot_{training_examples}")
        if task_name != "mentoreval":
            suffix_parts.append(task_name.replace("mentoreval_", ""))
        
        descriptive_suffix = "_".join(suffix_parts) if suffix_parts else "guidance"
        filename = f"{run_info.run_id}_{descriptive_suffix}_{timestamp}.json"
        
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
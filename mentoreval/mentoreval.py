from typing import List, Dict, Optional, Tuple  
import pandas as pd  
import json  
import os  
import re  
import glob
from datetime import datetime
from tqdm import tqdm  
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark  
from deepeval.models import DeepEvalBaseLLM  
from deepeval.telemetry import capture_benchmark_run  
from deepeval.dataset import Golden  
from deepeval.metrics.utils import trimAndLoadJson  

from .task import MentorEvalTask, MentorEvalDataset, MentorEvalTasks  
from .template import MentorEvalTemplate
from .config import MentorEvalConfig, BenchmarkMode, PromptType
from .models import create_model_from_config
from .run_manager import RunManager, RunInfo
from .metrics import (
    MetricsCalculator, 
    RubricRange, 
    parse_rubric_range, 
    extract_rubric_ranges_from_metadata,
    extract_scores_from_metadata
)
  
class MentorEvalBenchmark(DeepEvalBaseBenchmark):  
    def __init__(self, config: MentorEvalConfig):  
        super().__init__()  
        self.config = config
        self.tasks = MentorEvalTasks.get_all_tasks()  
        self.predictions = None  
        self.task_scores = {}  
        self.dataset_scores = {}  
        self.overall_score = None  
        self.overall_metrics = None
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Load training data for few-shot examples if needed
        self.training_data = {}
        if config.use_few_shot:
            self._load_training_data()
        
        # Initialize run manager
        self.run_manager = RunManager()
        self.current_run: Optional[RunInfo] = None
    
    def create_model(self) -> DeepEvalBaseLLM:
        """
        Create a model instance based on the configuration.
        
        Returns:
            DeepEvalBaseLLM instance configured according to self.config
        """
        return create_model_from_config(
            provider=self.config.model_provider,
            model_name=self.config.model_name
        )
    
    def _load_training_data(self):
        """Load training data for few-shot examples."""
        for task in self.tasks:
            dataset_name = task.dataset.value
            exercise_set = task.exercise_set
            file_path = f"data/processed/{dataset_name}/exercise_set_{exercise_set}/train.jsonl"
            
            if os.path.exists(file_path):
                training_samples = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            training_samples.append(data)
                    
                    # For mentoreval-test mode, use only one example per set
                    if self.config.mode == BenchmarkMode.MENTOREVAL_TEST:
                        training_samples = training_samples[:1]
                    
                    self.training_data[task.value] = training_samples
                    
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Could not load training data for {task.value}: {e}")
                    self.training_data[task.value] = []
            else:
                if self.config.verbose:
                    print(f"Warning: Training file not found: {file_path}")
                self.training_data[task.value] = []
          
    def extract_metrics_from_data(self, golden: Golden) -> List[str]:
        """Extract metric names dynamically from ideal_*_score fields in metadata."""  
        metadata = getattr(golden, 'additional_metadata', None) or {}  
        metrics: List[str] = []  
        for key in metadata.keys():  
            if key.startswith('ideal_') and key.endswith('_score'):  
                metric_name = key.replace('ideal_', '').replace('_score', '')  
                metrics.append(metric_name.title())  
        # Ensure we always return a list
        if metrics:
            return metrics
        else:
            return list(self.metrics_list)  # Convert to list to ensure it's not a tuple  
  
    def _parse_json_block(self, text: str) -> Optional[Dict]:  
        """Attempt to extract and parse the first JSON object from a text blob using deepeval's parser."""  
        try:  
            return trimAndLoadJson(text)  
        except Exception:  
            pass  
        return None  
  
    def _parse_scores_fallback(self, text: str, exercise_metrics: List[str]) -> Dict[str, float]:  
        """Fallback parser to extract numeric scores for each metric via regex if JSON parsing fails."""  
        scores: Dict[str, float] = {}  
        
        # Ensure exercise_metrics is a list
        if not isinstance(exercise_metrics, list):
            exercise_metrics = list(exercise_metrics)
        
        for metric in exercise_metrics:  
            m_key = metric.lower() + "_score"  
            # Patterns like: "ideas_score": 2, ideas_score: 2, "ideas": 2
            patterns = [  
                rf"\b{re.escape(m_key)}\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",  
                rf"\b{re.escape(metric.lower())}\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"  
            ]  
            value = None  
            for pattern in patterns:  
                match = re.search(pattern, text, re.IGNORECASE)  
                if match:  
                    try:  
                        value = float(match.group(1))  
                        break  
                    except Exception:  
                        continue  
            if value is not None:  
                scores[metric.title()] = value  
        return scores  
  
    def predict(self, model: DeepEvalBaseLLM, golden: Golden, task: MentorEvalTask, exercise_metrics: List[str]) -> Dict:  
        """Generate a prediction using the provided model and parse out per-metric scores."""  
        metadata = getattr(golden, 'additional_metadata', None) or {}  
        # Extract question from context list
        context_list = getattr(golden, 'context', []) or []
        question = context_list[0] if context_list else metadata.get('question', '')
        
        # Get training examples for few-shot if configured
        train_set = None
        n_shots = 0
        if self.config.use_few_shot and task.value in self.training_data:
            train_set = self.training_data[task.value]
            n_shots = len(train_set)
        
        prompt = MentorEvalTemplate.generate_output(  
            question=question,  
            student_answer=getattr(golden, 'input', ''),  
            rubric=metadata.get('rubric', ''),  
            academic_level=metadata.get('academic_level'),  
            essay_type=metadata.get('essay_type'),  
            metrics_list=exercise_metrics,  
            rubric_range=metadata.get('rubric_range', {}),  
            n_shots=n_shots,  
            train_set=train_set,  
            task=task,
            prompt_type=self.config.prompt_type,
            include_rubric=self.config.include_rubric
        )  
        model_output = model.generate(prompt)  
        
        # Handle case where model.generate returns a tuple (text, metadata)
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Extract just the text

        # Try JSON block first  
        parsed = self._parse_json_block(model_output)
        individual_scores: Dict[str, float] = {}  
        overall_score = None  
        if isinstance(parsed, dict):  
            for metric in exercise_metrics:  
                key = f"{metric.lower()}_score"  
                if key in parsed:  
                    try:  
                        individual_scores[metric.title()] = float(parsed[key])  
                    except Exception:  
                        continue  
            if 'overall_score' in parsed:  
                try:  
                    overall_score = float(parsed['overall_score'])  
                except Exception:  
                    overall_score = None  
  
        # Fallback parsing if needed  
        if not individual_scores:  
            individual_scores = self._parse_scores_fallback(model_output, exercise_metrics)  
  
        # If overall score is missing, compute sum of individual scores if available  
        if overall_score is None and individual_scores:  
            try:  
                overall_score = float(sum(individual_scores.values()))  
            except Exception:  
                overall_score = None  
  
        return {  
            'raw_output': model_output,  
            'individual_scores': individual_scores,  
            'overall_score': overall_score  
        }  
  
    def calculate_mae(self, golden: Golden, prediction_result: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:  
        """Calculate per-metric absolute error for the current sample using new metrics system."""  
        exercise_metrics = self.extract_metrics_from_data(golden)  
        predicted_scores: Dict[str, float] = prediction_result.get('individual_scores', {}) or {}  
        metadata = getattr(golden, 'additional_metadata', None) or {}  
        
        # Extract ground truth scores
        ground_truth_scores = extract_scores_from_metadata(metadata, exercise_metrics)
        
        # Extract rubric ranges
        rubric_ranges = extract_rubric_ranges_from_metadata(metadata)
        
        # Calculate metrics using the new system
        metric_results = self.metrics_calculator.calculate_all_metrics(
            predicted_scores, ground_truth_scores, rubric_ranges
        )
        
        # Return NMAE and NRMSE scores in the expected format
        nmae_scores: Dict[str, float] = {}
        nrmse_scores: Dict[str, float] = {}
        
        if True:  # Always use normalized metrics in the new system
            # Use normalized MAE (NMAE)
            if 'nmae' in metric_results:
                nmae_result = metric_results['nmae']
                # Distribute normalized MAE across individual metrics
                for metric in exercise_metrics:
                    nmae_scores[metric.title()] = nmae_result.normalized_value or 0.0
            
            # Use normalized RMSE (NRMSE)
            if 'nrmse' in metric_results:
                nrmse_result = metric_results['nrmse']
                # Distribute normalized RMSE across individual metrics
                for metric in exercise_metrics:
                    nrmse_scores[metric.title()] = nrmse_result.normalized_value or 0.0
        else:
            # Use absolute MAE (legacy behavior)
            for metric in exercise_metrics:
                metric_key = f'mae_{metric.lower()}'
                if metric_key in metric_results:
                    nmae_scores[metric.title()] = metric_results[metric_key].value
                else:
                    # Fallback to simple absolute difference
                    pred = predicted_scores.get(metric.title(), 0.0)
                    gt_val = ground_truth_scores.get(metric.title(), 0.0)
                    nmae_scores[metric.title()] = abs(pred - gt_val)
        
        return nmae_scores, nrmse_scores  
  
    def calculate_evaluation_metrics(self, all_predictions: List[float], all_ground_truth: List[float]) -> Dict[str, float]:  
        """Calculate MAE and, if available, Pearson and Spearman correlations."""  
        results: Dict[str, float] = {}  
        # MAE  
        if all_predictions and len(all_predictions) == len(all_ground_truth):  
            abs_errors = [abs(float(p) - float(t)) for p, t in zip(all_predictions, all_ground_truth)]  
            results['mae'] = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0  
        else:  
            results['mae'] = 0.0  
  
        # Correlations and Cohen's Kappa (optional, requires scipy and sklearn)  
        try:  
            if len(all_predictions) > 1:  
                from scipy.stats import pearsonr, spearmanr  
                from sklearn.metrics import cohen_kappa_score
                
                pearson_corr, pearson_p = pearsonr(all_predictions, all_ground_truth)  
                results['pearson_correlation'] = float(pearson_corr)  
                results['pearson_p_value'] = float(pearson_p)  

                spearman_corr, spearman_p = spearmanr(all_predictions, all_ground_truth)  
                results['spearman_correlation'] = float(spearman_corr)  
                results['spearman_p_value'] = float(spearman_p)
                
                # Calculate Cohen's Kappa
                pred_categories = [int(round(val)) for val in all_predictions]
                gt_categories = [int(round(val)) for val in all_ground_truth]
                kappa = cohen_kappa_score(gt_categories, pred_categories)
                results['cohens_kappa'] = float(kappa)
        except Exception:  
            # If scipy/sklearn is unavailable, skip correlation metrics silently  
            pass
  
        return results  
  
    def evaluate(self, model: DeepEvalBaseLLM):  
        # Create a new run
        self.current_run = self.run_manager.create_run(self.config)
        self.run_manager.update_run_status(self.current_run.run_id, 'running')
        
        if self.config.verbose:
            print(f"\n🚀 Starting MentorEval Run ID: {self.current_run.run_id}")
            print(f"   Model: {self.config.model_name}")
            print(f"   Mode: {self.config.mode.value}")
            print(f"   Configuration: {self.config.get_description()}")
        
        try:
            # Use deepeval's benchmark tracking
            with capture_benchmark_run("MentorEval", len(self.tasks)):  
                all_predictions: List[float] = []  
                all_ground_truth: List[float] = []  
                predictions_row = []  
  
            # For reporting per-task and per-dataset metrics
            task_metrics: Dict[str, Dict[str, List[float]]] = {}  # task -> metric -> values
            task_counts: Dict[str, int] = {}  
            dataset_to_task_metrics: Dict[str, Dict[str, List[float]]] = {}  # dataset -> metric -> values  
  
            for task in self.tasks:  
                goldens = self.load_benchmark_dataset(task)  
                if self.config.n_test_samples and self.config.n_test_samples < len(goldens):  
                    goldens = goldens[:self.config.n_test_samples]  
  
                if not goldens:  
                    continue  
  
                for idx, golden in enumerate(tqdm(goldens, desc=f"Processing {task.value}")):  
                    exercise_metrics = self.extract_metrics_from_data(golden)  
                    prediction_result = self.predict(model, golden, task, exercise_metrics)  
  
                    # Per-sample NMAE and NRMSE across metrics  
                    nmae_scores, nrmse_scores = self.calculate_mae(golden, prediction_result)  
                    sample_overall_nmae = (sum(nmae_scores.values()) / len(nmae_scores)) if nmae_scores else 0.0
                    sample_overall_nrmse = (sum(nrmse_scores.values()) / len(nrmse_scores)) if nrmse_scores else 0.0  
  
                    # Collect for global correlation metrics  
                    metadata = getattr(golden, 'additional_metadata', None) or {}  
                    for metric in exercise_metrics:  
                        pred_score = prediction_result.get('individual_scores', {}).get(metric.title(), None)  
                        gt_key = f"ideal_{metric.lower()}_score"  
                        gt_score = metadata.get(gt_key, None)  
                        if pred_score is not None and gt_score is not None:  
                            try:  
                                all_predictions.append(float(pred_score))  
                                all_ground_truth.append(float(gt_score))  
                            except Exception:  
                                pass  
  
                    # Handle expected scores - use individual scores if available, otherwise use overall
                    expected_scores = {}  
                    
                    # Check if we have individual metric scores (not None)
                    individual_scores_available = any(
                        metadata.get(f"ideal_{metric.lower()}_score") is not None 
                        for metric in exercise_metrics
                    )
                    
                    if individual_scores_available:
                        # Use individual scores if available
                        for metric in exercise_metrics:  
                            individual_score = metadata.get(f"ideal_{metric.lower()}_score")  
                            if individual_score is not None:  
                                expected_scores[metric.title()] = individual_score  
                            else:  
                                expected_scores[metric.title()] = None
                    else:
                        # If no individual scores, check if we have an overall score
                        overall_score = metadata.get('ideal_overall_score') or metadata.get('ideal')
                        if overall_score is not None:
                            # For single metric datasets, use the overall score directly
                            if len(exercise_metrics) == 1:
                                expected_scores[exercise_metrics[0].title()] = overall_score
                            else:
                                # For multi-metric datasets, divide by number of metrics
                                for metric in exercise_metrics:
                                    expected_scores[metric.title()] = float(overall_score) / len(exercise_metrics)
                        else:
                            # No scores available
                            for metric in exercise_metrics:
                                expected_scores[metric.title()] = None  
  
                    predictions_row.append({  
                        'Dataset': task.dataset.value,  
                        'Exercise_Set': task.exercise_set,  
                        'Task': task.value,  
                        'Input': getattr(golden, 'input', ''),  
                        'Prediction': prediction_result,  
                        'Expected_Scores': expected_scores,  
                        'NMAE_Scores': nmae_scores,  
                        'NRMSE_Scores': nrmse_scores,
                        'Overall_NMAE': sample_overall_nmae,
                        'Overall_NRMSE': sample_overall_nrmse
                    })  

                    # Aggregate by task for reporting
                    if task.value not in task_metrics:
                        task_metrics[task.value] = {
                            'nmae': [], 'nrmse': [], 'pearson_correlation': [], 
                            'spearman_correlation': [], 'jensen_shannon_divergence': [],
                            'wasserstein_distance': [], 'kolmogorov_smirnov_test': [],
                            'cohens_kappa': []
                        }
                    
                    # Collect sample-level metrics
                    task_metrics[task.value]['nmae'].append(sample_overall_nmae)
                    task_metrics[task.value]['nrmse'].append(sample_overall_nrmse)
                    task_counts[task.value] = task_counts.get(task.value, 0) + 1
                    
                    # Collect per-rubric NMAE and NRMSE
                    if 'per_rubric_metrics' not in task_metrics[task.value]:
                        task_metrics[task.value]['per_rubric_metrics'] = {}
                    
                    for metric_name in nmae_scores.keys():
                        if metric_name not in task_metrics[task.value]['per_rubric_metrics']:
                            task_metrics[task.value]['per_rubric_metrics'][metric_name] = {
                                'nmae': [], 'nrmse': []
                            }
                        task_metrics[task.value]['per_rubric_metrics'][metric_name]['nmae'].append(nmae_scores[metric_name])
                        task_metrics[task.value]['per_rubric_metrics'][metric_name]['nrmse'].append(nrmse_scores[metric_name])  

                    if self.config.verbose:  
                        print(f"Sample {idx}: NMAE = {sample_overall_nmae:.3f}, NRMSE = {sample_overall_nrmse:.3f}")
  
            # Calculate per-task metrics (correlations and distribution metrics)
            self._calculate_per_task_metrics(task_metrics, predictions_row)
            
            # Compute per-task averages and dataset aggregates  
            scores_row = []  
            for task in self.tasks:  
                if task.value in task_counts and task_counts[task.value] > 0:  
                    # Calculate average metrics for this task
                    task_avg_metrics = {}
                    for metric_name, values in task_metrics[task.value].items():
                        if metric_name == 'per_rubric_metrics':
                            # Handle per-rubric metrics separately
                            task_avg_metrics[metric_name] = {}
                            for rubric_name, rubric_metrics in values.items():
                                task_avg_metrics[metric_name][rubric_name] = {}
                                for metric_type, metric_values in rubric_metrics.items():
                                    if metric_values:  # Only calculate if we have values
                                        task_avg_metrics[metric_name][rubric_name][metric_type] = sum(metric_values) / len(metric_values)
                        elif values:  # Only calculate if we have values
                            task_avg_metrics[metric_name] = sum(values) / len(values)
                    
                    # Use NMAE for the scores_row (backward compatibility)
                    avg_nmae = task_avg_metrics.get('nmae', 0.0)
                    scores_row.append((task.dataset.value, task.exercise_set, task.value, avg_nmae))  
                    
                    # Store task metrics
                    self.task_scores[task.value] = task_avg_metrics
                    
                    # Aggregate by dataset
                    if task.dataset.value not in dataset_to_task_metrics:
                        dataset_to_task_metrics[task.dataset.value] = {
                            'nmae': [], 'nrmse': [], 'pearson_correlation': [], 
                            'spearman_correlation': [], 'jensen_shannon_divergence': [],
                            'wasserstein_distance': [], 'kolmogorov_smirnov_test': [],
                            'cohens_kappa': []
                        }
                    
                    for metric_name, value in task_avg_metrics.items():
                        if metric_name in dataset_to_task_metrics[task.dataset.value]:
                            dataset_to_task_metrics[task.dataset.value][metric_name].append(value)
                    
                    print(f"MentorEval Task NMAE (task={task.value}): {avg_nmae:.3f}")  

            # Calculate dataset-level metrics
            for dataset, metrics in dataset_to_task_metrics.items():  
                dataset_avg_metrics = {}
                for metric_name, values in metrics.items():
                    if values:
                        dataset_avg_metrics[metric_name] = sum(values) / len(values)
                
                self.dataset_scores[dataset] = dataset_avg_metrics
                print(f"MentorEval Dataset NMAE (dataset={dataset}): {dataset_avg_metrics.get('nmae', 0.0):.3f}")
  
            # Calculate overall NMAE and NRMSE by averaging per-sample values
            all_sample_nmaes = [row['Overall_NMAE'] for row in predictions_row]
            all_sample_nrmses = [row['Overall_NRMSE'] for row in predictions_row]
            overall_nmae = sum(all_sample_nmaes) / len(all_sample_nmaes) if all_sample_nmaes else 0.0
            overall_nrmse = sum(all_sample_nrmses) / len(all_sample_nrmses) if all_sample_nrmses else 0.0
            
            # Global evaluation metrics (for correlations, use individual metrics)
            eval_results = self.calculate_evaluation_metrics(all_predictions, all_ground_truth)  
            
            # Store the results
            eval_results['nmae'] = {'normalized_value': overall_nmae}
            eval_results['nrmse'] = {'normalized_value': overall_nrmse}
            
            # Print NMAE and NRMSE instead of MAE
            if 'nmae' in eval_results:
                print(f"Overall MentorEval NMAE: {eval_results['nmae']['normalized_value']:.3f}")
            if 'nrmse' in eval_results:
                print(f"Overall MentorEval NRMSE: {eval_results['nrmse']['normalized_value']:.3f}")
            if 'pearson_correlation' in eval_results:  
                print(f"Pearson Correlation: {eval_results['pearson_correlation']:.3f}")  
            if 'spearman_correlation' in eval_results:  
                print(f"Spearman Correlation: {eval_results['spearman_correlation']:.3f}")  

            # Create DataFrames for results
            self.predictions = pd.DataFrame(predictions_row)
            # self.task_scores is already set as a dictionary in the loop above
            self.overall_score = overall_nmae  # Use NMAE as the overall score
            self.overall_metrics = eval_results

            # Save results to files
            self._save_results()

            # Update run status to completed
            self.run_manager.update_run_status(self.current_run.run_id, 'completed')

            return eval_results
        
        except Exception as e:
            # Update run status to failed
            self.run_manager.update_run_status(self.current_run.run_id, 'failed')
            
            if self.config.verbose:
                print(f"\n❌ Run {self.current_run.run_id} failed: {e}")
            
            raise
    
    def _calculate_per_task_metrics(self, task_metrics: Dict[str, Dict[str, List[float]]], predictions_row: List[Dict]):
        """Calculate correlation and distribution metrics for each task and per-rubric metrics."""
        # Group predictions by task
        task_predictions = {}
        task_rubric_predictions = {}  # For per-rubric metrics
        
        for row in predictions_row:
            task = row['Task']
            if task not in task_predictions:
                task_predictions[task] = {'predictions': [], 'ground_truth': []}
                task_rubric_predictions[task] = {}
            
            # Extract individual metric predictions and ground truth
            pred_scores = row.get('Prediction_Scores', {})
            exp_scores = row.get('Expected_Scores', {})
            
            for metric_name in pred_scores.keys():
                if metric_name in exp_scores:
                    try:
                        pred_val = float(pred_scores[metric_name])
                        gt_val = float(exp_scores[metric_name])
                        
                        # Add to overall task predictions
                        task_predictions[task]['predictions'].append(pred_val)
                        task_predictions[task]['ground_truth'].append(gt_val)
                        
                        # Add to per-rubric predictions
                        if metric_name not in task_rubric_predictions[task]:
                            task_rubric_predictions[task][metric_name] = {'predictions': [], 'ground_truth': []}
                        task_rubric_predictions[task][metric_name]['predictions'].append(pred_val)
                        task_rubric_predictions[task][metric_name]['ground_truth'].append(gt_val)
                        
                    except (ValueError, TypeError):
                        continue
        
        # Calculate metrics for each task (overall)
        for task, data in task_predictions.items():
            if len(data['predictions']) < 2:
                continue
                
            pred_values = data['predictions']
            gt_values = data['ground_truth']
            
            # Calculate correlations
            try:
                from scipy.stats import pearsonr, spearmanr
                from sklearn.metrics import cohen_kappa_score
                pearson_corr, _ = pearsonr(pred_values, gt_values)
                spearman_corr, _ = spearmanr(pred_values, gt_values)
                task_metrics[task]['pearson_correlation'].append(pearson_corr)
                task_metrics[task]['spearman_correlation'].append(spearman_corr)
                
                # Calculate Cohen's Kappa
                pred_categories = [int(round(val)) for val in pred_values]
                gt_categories = [int(round(val)) for val in gt_values]
                kappa = cohen_kappa_score(gt_categories, pred_categories)
                task_metrics[task]['cohens_kappa'].append(kappa)
            except Exception:
                pass
            
            # Calculate distribution metrics
            try:
                from scipy.spatial.distance import jensenshannon
                from scipy.stats import wasserstein_distance, ks_2samp
                import numpy as np
                
                # Jensen-Shannon divergence
                pred_dist = np.array(pred_values) / sum(pred_values) if sum(pred_values) > 0 else np.array(pred_values)
                gt_dist = np.array(gt_values) / sum(gt_values) if sum(gt_values) > 0 else np.array(gt_values)
                js_div = jensenshannon(pred_dist, gt_dist, base=2)
                task_metrics[task]['jensen_shannon_divergence'].append(js_div)
                
                # Wasserstein distance
                w_distance = wasserstein_distance(pred_values, gt_values)
                task_metrics[task]['wasserstein_distance'].append(w_distance)
                
                # Kolmogorov-Smirnov test
                ks_stat, _ = ks_2samp(pred_values, gt_values)
                task_metrics[task]['kolmogorov_smirnov_test'].append(ks_stat)
                
            except Exception:
                pass
        
        # Calculate per-rubric metrics
        for task, rubric_data in task_rubric_predictions.items():
            if task not in task_metrics:
                continue
                
            # Initialize per-rubric metrics structure
            if 'per_rubric_metrics' not in task_metrics[task]:
                task_metrics[task]['per_rubric_metrics'] = {}
            
            for rubric_name, rubric_predictions in rubric_data.items():
                if len(rubric_predictions['predictions']) < 2:
                    continue
                
                pred_values = rubric_predictions['predictions']
                gt_values = rubric_predictions['ground_truth']
                
                # Initialize rubric metrics
                task_metrics[task]['per_rubric_metrics'][rubric_name] = {
                    'nmae': [], 'nrmse': [], 'pearson_correlation': [], 
                    'spearman_correlation': [], 'jensen_shannon_divergence': [],
                    'wasserstein_distance': [], 'kolmogorov_smirnov_test': [],
                    'cohens_kappa': []
                }
                
                # Calculate NMAE and NRMSE for this rubric
                try:
                    from scipy.stats import pearsonr, spearmanr
                    from scipy.spatial.distance import jensenshannon
                    from scipy.stats import wasserstein_distance, ks_2samp
                    from sklearn.metrics import cohen_kappa_score
                    import numpy as np
                    
                    # Calculate correlations
                    pearson_corr, _ = pearsonr(pred_values, gt_values)
                    spearman_corr, _ = spearmanr(pred_values, gt_values)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['pearson_correlation'].append(pearson_corr)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['spearman_correlation'].append(spearman_corr)
                    
                    # Calculate Cohen's Kappa for this rubric
                    pred_categories = [int(round(val)) for val in pred_values]
                    gt_categories = [int(round(val)) for val in gt_values]
                    kappa = cohen_kappa_score(gt_categories, pred_categories)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['cohens_kappa'].append(kappa)
                    
                    # Calculate distribution metrics
                    pred_dist = np.array(pred_values) / sum(pred_values) if sum(pred_values) > 0 else np.array(pred_values)
                    gt_dist = np.array(gt_values) / sum(gt_values) if sum(gt_values) > 0 else np.array(gt_values)
                    js_div = jensenshannon(pred_dist, gt_dist, base=2)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['jensen_shannon_divergence'].append(js_div)
                    
                    w_distance = wasserstein_distance(pred_values, gt_values)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['wasserstein_distance'].append(w_distance)
                    
                    ks_stat, _ = ks_2samp(pred_values, gt_values)
                    task_metrics[task]['per_rubric_metrics'][rubric_name]['kolmogorov_smirnov_test'].append(ks_stat)
                    
                except Exception:
                    pass
    
    def _save_results(self):
        """Save results in two formats: detailed (ignored by git) and aggregated (tracked by git)."""
        if not self.current_run:
            raise ValueError("No current run found. Call evaluate() first.")
        
        # Automatically calculate timestamp when saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.current_run.run_id
        
        # Save detailed results (ignored by git)
        self._save_detailed_results(timestamp, run_id)
        
        # Save aggregated results (tracked by git)
        self._save_aggregated_results(timestamp, run_id)
    
    def _save_detailed_results(self, timestamp: str, run_id: int):
        """Save detailed LLM input/output results (ignored by git)."""
        # Create detailed results directory (ignored by git)
        detailed_dir = "results_detailed"
        os.makedirs(detailed_dir, exist_ok=True)
        
        run_dir = os.path.join(detailed_dir, f"{run_id}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save detailed predictions with full LLM input/output
        predictions_file = os.path.join(run_dir, "detailed_predictions.csv")
        self.predictions.to_csv(predictions_file, index=False)
        
        # Save individual LLM interactions as JSONL for analysis
        interactions_file = os.path.join(run_dir, "llm_interactions.jsonl")
        with open(interactions_file, 'w', encoding='utf-8') as f:
            for _, row in self.predictions.iterrows():
                interaction = {
                    'dataset': row['Dataset'],
                    'exercise_set': row['Exercise_Set'],
                    'task': row['Task'],
                    'input': row['Input'],
                    'prediction': row['Prediction'],
                    'expected_scores': row['Expected_Scores'],
                    'nmae_scores': row['NMAE_Scores'],
                    'nrmse_scores': row['NRMSE_Scores'],
                    'overall_nmae': row['Overall_NMAE'],
                    'overall_nrmse': row['Overall_NRMSE']
                }
                f.write(json.dumps(interaction) + '\n')
        
        # Save configuration for reproducibility
        config_file = os.path.join(run_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': self.config.mode.value,
                'use_few_shot': self.config.use_few_shot,
                'include_rubric': self.config.include_rubric,
                'prompt_type': self.config.prompt_type.value,
                'n_test_samples': self.config.n_test_samples,
                'model_name': self.config.model_name,
                'description': self.config.get_description(),
                'timestamp': timestamp
            }, f, indent=2)
        
        if self.config.verbose:
            print(f"\n📁 Detailed results saved to: {run_dir}")
            print(f"   - Run ID: {run_id}")
            print(f"   - Detailed predictions: {predictions_file}")
            print(f"   - LLM interactions: {interactions_file}")
            print(f"   - Configuration: {config_file}")
    
    def _save_aggregated_results(self, timestamp: str, run_id: int):
        """Save aggregated results for tracking and comparison (tracked by git)."""
        # Create aggregated results directory (tracked by git)
        aggregated_dir = "results"
        os.makedirs(aggregated_dir, exist_ok=True)
        
        # Create filename with ID, model name and benchmark mode
        model_name_clean = self.config.model_name.replace('/', '_').replace(':', '_')
        filename = f"{run_id}_{model_name_clean}_{self.config.mode.value}_{timestamp}.json"
        results_file = os.path.join(aggregated_dir, filename)
        
        # Create hierarchical metrics structure
        hierarchical_metrics = self._create_hierarchical_metrics()
        
        # Create aggregated result entry
        aggregated_result = {
            'timestamp': timestamp,
            'model_name': self.config.model_name,
            'benchmark_mode': self.config.mode.value,
            'configuration': {
                'use_few_shot': self.config.use_few_shot,
                'include_rubric': self.config.include_rubric,
                'prompt_type': self.config.prompt_type.value,
                'n_test_samples': self.config.n_test_samples,
                'description': self.config.get_description()
            },
            'metrics': hierarchical_metrics,
            'summary': {
                'total_samples': len(self.predictions),
                'total_tasks': len(self.task_scores),
                'overall_mae': self.overall_score,
                'pearson_correlation': self.overall_metrics.get('pearson_correlation'),
                'spearman_correlation': self.overall_metrics.get('spearman_correlation')
            }
        }
        
        # Save aggregated result
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_result, f, indent=2)
        
        if self.config.verbose:
            print(f"\n📊 Aggregated results saved to: {results_file}")
            print(f"   - Run ID: {run_id}")
            print(f"   - Model: {self.config.model_name}")
            print(f"   - Mode: {self.config.mode.value}")
            print(f"   - Overall NMAE: {self.overall_metrics.get('nmae', {}).get('normalized_value', 0.0):.3f}")
            print(f"   - Samples: {len(self.predictions)}")
    
    def _create_hierarchical_metrics(self):
        """Create hierarchical metrics structure: exercise_set -> dataset -> overall."""
        hierarchical = {
            'overall': {
                'nmae': self.overall_metrics.get('nmae', {}).get('normalized_value', 0.0),
                'nrmse': self.overall_metrics.get('nrmse', {}).get('normalized_value', 0.0),
                'pearson_correlation': self.overall_metrics.get('pearson_correlation'),
                'spearman_correlation': self.overall_metrics.get('spearman_correlation'),
                'jensen_shannon_divergence': self.overall_metrics.get('jensen_shannon_divergence'),
                'wasserstein_distance': self.overall_metrics.get('wasserstein_distance'),
                'kolmogorov_smirnov_test': self.overall_metrics.get('kolmogorov_smirnov_test'),
                'cohens_kappa': self.overall_metrics.get('cohens_kappa'),
                'total_samples': len(self.predictions),
                'total_tasks': len(self.task_scores)
            },
            'datasets': {}
        }
        
        # Group task scores by dataset
        dataset_groups = {}
        for task_name, task_metrics in self.task_scores.items():
            # Find the dataset for this task
            dataset = None
            for task in self.tasks:
                if task.value == task_name:
                    dataset = task.dataset.value
                    break
            
            if dataset is None:
                continue
                
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            
            dataset_groups[dataset].append({
                'exercise_set': task.exercise_set,
                'task': task_name,
                'metrics': task_metrics
            })
        
        # Create dataset-level aggregations
        for dataset, tasks in dataset_groups.items():
            # Calculate dataset-level metrics by averaging task metrics
            dataset_metrics = {}
            metric_names = ['nmae', 'nrmse', 'pearson_correlation', 'spearman_correlation', 
                          'jensen_shannon_divergence', 'wasserstein_distance', 'kolmogorov_smirnov_test',
                          'cohens_kappa']
            
            for metric_name in metric_names:
                values = []
                for task in tasks:
                    if metric_name in task['metrics']:
                        values.append(task['metrics'][metric_name])
                
                if values:
                    dataset_metrics[metric_name] = sum(values) / len(values)
            
            hierarchical['datasets'][dataset] = {
                **dataset_metrics,
                'total_exercise_sets': len(tasks),
                'exercise_sets': {}
            }
            
            # Add individual exercise set metrics
            for task in tasks:
                exercise_set = task['exercise_set']
                hierarchical['datasets'][dataset]['exercise_sets'][f'exercise_set_{exercise_set}'] = {
                    'task': task['task'],
                    **task['metrics']
                }
        
        return hierarchical
      
    def load_benchmark_dataset(self, task: MentorEvalTask) -> List[Golden]:  
        """Load dataset for specific task and exercise set"""  
        dataset_name = task.dataset.value  
        exercise_set = task.exercise_set  
        split = "test"  # Always use test set for evaluation  
          
        file_path = f"data/processed/{dataset_name}/exercise_set_{exercise_set}/{split}.jsonl"  
          
        if not os.path.exists(file_path):  
            print(f"Warning: Dataset file {file_path} not found")  
            return []  
          
        goldens = []  
        try:  
            with open(file_path, 'r', encoding='utf-8') as f:  
                for line in f:  
                    data = json.loads(line.strip())  
                    # Build additional metadata
                    additional_metadata = {  
                        'dataset': dataset_name,  
                        'exercise_set': exercise_set,  
                        'task': task.value,  
                        'academic_level': data.get('academic_level'),  
                        'essay_type': data.get('essay_type'),  
                        'rubric': data.get('rubric', ''),  
                        'rubric_range': data.get('rubric_range', {}),  
                        'num_metrics': data.get('num_metrics', 4),  
                        'essay_set': data.get('essay_set')
                    }
                    
                    # Only add individual metric scores if they exist in the dataset
                    for metric in ['ideas', 'organization', 'style', 'conventions']:
                        score_key = f'ideal_{metric}_score'
                        if score_key in data and data[score_key] is not None:
                            additional_metadata[score_key] = data[score_key]
                    
                    # Add overall score if it exists
                    if 'ideal' in data and data['ideal'] is not None:
                        additional_metadata['ideal_overall_score'] = data['ideal']
                    
                    golden = Golden(  
                        input=data.get('student_answer', ''),  
                        expected_output=str(data.get('ideal', '')),  
                        context=[data.get('question', '')],  
                        additional_metadata=additional_metadata
                    )  
                    goldens.append(golden)  
        except FileNotFoundError:  
            print(f"Warning: Dataset file {file_path} not found")  
            return []  
          
        return goldens
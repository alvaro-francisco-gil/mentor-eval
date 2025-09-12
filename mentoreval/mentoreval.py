from typing import List, Dict, Optional, Tuple  
import pandas as pd  
import json  
import os  
import re  
import glob
import asyncio
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
        self.interactions_file: Optional[str] = None
    
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
    
    async def a_predict(self, model: DeepEvalBaseLLM, golden: Golden, task: MentorEvalTask, exercise_metrics: List[str]) -> Dict:  
        """Async version of predict method for concurrent evaluation."""
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
        
        # Use async model generation if available
        if hasattr(model, 'a_generate'):
            model_output = await model.a_generate(prompt)
        else:
            # Fallback to sync generation in thread pool
            loop = asyncio.get_event_loop()
            model_output = await loop.run_in_executor(None, model.generate, prompt)
        
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
  
    def evaluate(self, model: DeepEvalBaseLLM, run_id: Optional[int] = None):  
        # Use existing run or create a new one
        if run_id is not None:
            self.current_run = self.run_manager.get_run_info(run_id)
            if self.current_run is None:
                raise ValueError(f"Run {run_id} not found")
            # Update config to use the configuration from the existing run
            self.config = self._create_config_from_run_data(self.current_run.configuration)
        else:
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
                predictions_row = []  

            # NEW 4-LEVEL ARCHITECTURE: Collect all samples per exercise before calculating metrics
            exercise_data: Dict[str, Dict] = {}  # task -> {predictions, ground_truth, rubric_ranges}
            task_counts: Dict[str, int] = {}  

            # PHASE 1: Collect all samples per exercise
            for task in self.tasks:  
                goldens = self.load_benchmark_dataset(task)  
                
                # Apply sample limiting based on configuration
                if self.config.n_test_samples and self.config.n_test_samples < len(goldens):  
                    goldens = goldens[:self.config.n_test_samples]
                elif self.config.test_percentage:
                    # Calculate number of samples based on percentage
                    n_samples = max(1, int(len(goldens) * self.config.test_percentage))
                    if self.config.verbose:
                        print(f"  Exercise {task.exercise_set}: Limiting to {n_samples} samples ({self.config.test_percentage*100:.1f}% of {len(goldens)})")
                    goldens = goldens[:n_samples]

                if not goldens:  
                    continue  

                # Initialize exercise data collection
                exercise_data[task.value] = {
                    'all_predictions': [],
                    'all_ground_truth': [],
                    'all_rubric_ranges': [],
                    'predictions_row': []
                }

                for idx, golden in enumerate(tqdm(goldens, desc=f"Processing {task.value}")):  
                    exercise_metrics = self.extract_metrics_from_data(golden)  
                    prediction_result = self.predict(model, golden, task, exercise_metrics)  

                    # Extract predicted and ground truth scores for this sample
                    predicted_scores = {}
                    ground_truth_scores = {}
                    rubric_ranges = {}
                    
                    metadata = getattr(golden, 'additional_metadata', None) or {}
                    
                    # Extract rubric ranges
                    try:
                        rubric_ranges = extract_rubric_ranges_from_metadata(metadata)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"Warning: Could not extract rubric ranges for {task.value}: {e}")
                        continue
                    
                    # Extract scores for each metric
                    for metric in exercise_metrics:
                        metric_name = metric.title()
                        
                        # Get predicted score
                        pred_score = prediction_result.get('individual_scores', {}).get(metric_name, None)
                        if pred_score is not None:
                            predicted_scores[metric_name] = float(pred_score)
                        
                        # Get ground truth score
                        gt_score = metadata.get(f"ideal_{metric.lower()}_score", None)
                        if gt_score is not None:
                            ground_truth_scores[metric_name] = float(gt_score)
                    
                    # Only add if we have valid scores
                    if predicted_scores and ground_truth_scores and rubric_ranges:
                        exercise_data[task.value]['all_predictions'].append(predicted_scores)
                        exercise_data[task.value]['all_ground_truth'].append(ground_truth_scores)
                        exercise_data[task.value]['all_rubric_ranges'].append(rubric_ranges)
                        
                        # Calculate per-sample metrics for this sample
                        try:
                            per_sample_metrics = self.metrics_calculator.calculate_per_sample_metrics(
                                predicted_scores, ground_truth_scores, rubric_ranges
                            )
                            
                            # Extract NMAE and NRMSE for backward compatibility
                            sample_nmae = per_sample_metrics.get('nmae', {}).value if 'nmae' in per_sample_metrics else 0.0
                            sample_nrmse = per_sample_metrics.get('nrmse', {}).value if 'nrmse' in per_sample_metrics else 0.0
                            
                        except Exception as e:
                            if self.config.verbose:
                                print(f"Warning: Could not calculate per-sample metrics for {task.value} sample {idx}: {e}")
                            sample_nmae = 0.0
                            sample_nrmse = 0.0
                        
                        # Store prediction row for detailed results
                        prediction_row = {  
                            'Dataset': task.dataset.value,  
                            'Exercise_Set': task.exercise_set,  
                            'Task': task.value,  
                            'Input': getattr(golden, 'input', ''),  
                            'Prediction': prediction_result,  
                            'Predicted_Scores': predicted_scores,
                            'Expected_Scores': ground_truth_scores,
                            'Sample_NMAE': sample_nmae,
                            'Sample_NRMSE': sample_nrmse
                        }
                        
                        exercise_data[task.value]['predictions_row'].append(prediction_row)
                        predictions_row.append(prediction_row)
                        
                        task_counts[task.value] = task_counts.get(task.value, 0) + 1
                        
                        if self.config.verbose:  
                            print(f"Sample {idx}: NMAE = {sample_nmae:.3f}, NRMSE = {sample_nrmse:.3f}")

            # PHASE 2: Calculate exercise-level metrics using the new 4-level architecture
            exercise_metrics_results = {}  # task -> exercise-level metrics
            for task_name, data in exercise_data.items():
                if not data['all_predictions']:
                    continue
                
                try:
                    # Calculate exercise-level metrics (Level 2)
                    exercise_metrics = self.metrics_calculator.calculate_exercise_metrics(
                        data['all_predictions'],
                        data['all_ground_truth'], 
                        data['all_rubric_ranges']
                    )
                    exercise_metrics_results[task_name] = exercise_metrics
                    
                    if self.config.verbose:
                        print(f"\n📊 Exercise {task_name} Metrics:")
                        for metric_name, value in exercise_metrics.items():
                            print(f"  {metric_name}: {value:.3f}")
                            
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Could not calculate exercise metrics for {task_name}: {e}")
                    exercise_metrics_results[task_name] = {}
            
            # PHASE 3: Calculate dataset-level metrics (Level 3)
            dataset_metrics_results = {}  # dataset -> dataset-level metrics
            dataset_to_exercise_metrics = {}  # dataset -> [exercise_metrics]
            
            # Group exercise metrics by dataset
            for task in self.tasks:
                if task.value in exercise_metrics_results:
                    dataset = task.dataset.value
                    if dataset not in dataset_to_exercise_metrics:
                        dataset_to_exercise_metrics[dataset] = []
                    dataset_to_exercise_metrics[dataset].append(exercise_metrics_results[task.value])
            
            # Calculate dataset-level metrics by averaging exercise-level metrics
            for dataset, exercise_metrics_list in dataset_to_exercise_metrics.items():
                try:
                    dataset_metrics = self.metrics_calculator.calculate_dataset_metrics(exercise_metrics_list)
                    dataset_metrics_results[dataset] = dataset_metrics
                    
                    if self.config.verbose:
                        print(f"\n📊 Dataset {dataset} Metrics:")
                        for metric_name, value in dataset_metrics.items():
                            print(f"  {metric_name}: {value:.3f}")
                            
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Could not calculate dataset metrics for {dataset}: {e}")
                    dataset_metrics_results[dataset] = {}
            
            # PHASE 4: Calculate overall metrics (Level 4)
            if dataset_metrics_results:
                try:
                    overall_metrics = self.metrics_calculator.calculate_overall_metrics(
                        list(dataset_metrics_results.values())
                    )
                    self.overall_metrics = overall_metrics
                    
                    if self.config.verbose:
                        print(f"\n🎯 Overall Metrics:")
                        for metric_name, value in overall_metrics.items():
                            print(f"  {metric_name}: {value:.3f}")
                            
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Could not calculate overall metrics: {e}")
                    self.overall_metrics = {}
            
            # Store results for backward compatibility and reporting
            scores_row = []
            for task in self.tasks:
                if task.value in exercise_metrics_results:
                    # Use NMAE for the scores_row (backward compatibility)
                    avg_nmae = exercise_metrics_results[task.value].get('nmae', 0.0)
                    scores_row.append((task.dataset.value, task.exercise_set, task.value, avg_nmae))
                    
                    # Store task metrics (exercise-level metrics)
                    self.task_scores[task.value] = exercise_metrics_results[task.value]
                    
                    print(f"MentorEval Task NMAE (task={task.value}): {avg_nmae:.3f}")
            
            # Store dataset scores
            self.dataset_scores = dataset_metrics_results
            for dataset, metrics in dataset_metrics_results.items():
                print(f"MentorEval Dataset NMAE (dataset={dataset}): {metrics.get('nmae', 0.0):.3f}")
  
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
    
    def evaluate_with_config(self, model: DeepEvalBaseLLM, run_id: Optional[int] = None):
        """Evaluate using sync or async based on configuration."""
        if self.config.async_config.run_async:
            # Run async evaluation
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.a_evaluate(model, run_id))
        else:
            # Run sync evaluation
            return self.evaluate(model, run_id)
    
    async def a_evaluate(self, model: DeepEvalBaseLLM, run_id: Optional[int] = None):
        """Async version of evaluate method with concurrent processing."""
        # Use existing run or create a new one
        if run_id is not None:
            self.current_run = self.run_manager.get_run_info(run_id)
            if self.current_run is None:
                raise ValueError(f"Run {run_id} not found")
            # Update config to use the configuration from the existing run
            self.config = self._create_config_from_run_data(self.current_run.configuration)
        else:
            self.current_run = self.run_manager.create_run(self.config)
        self.run_manager.update_run_status(self.current_run.run_id, 'running')
        
        # Initialize interactions file for incremental saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._initialize_interactions_file(timestamp, self.current_run.run_id)
        
        if self.config.verbose:
            print(f"\n🚀 Starting MentorEval Run ID: {self.current_run.run_id} (Async)")
            print(f"   Model: {self.config.model_name}")
            print(f"   Mode: {self.config.mode.value}")
            print(f"   Configuration: {self.config.get_description()}")
            print(f"   Debug - test_percentage: {self.config.test_percentage}, n_test_samples: {self.config.n_test_samples}")
        
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

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.async_config.max_concurrent)
            
            async def process_golden_with_semaphore(golden, task, idx):
                """Process a single golden with semaphore control."""
                async with semaphore:
                    return await self._process_single_golden_async(model, golden, task, idx, task_metrics, task_counts, all_predictions, all_ground_truth)
            
            # Collect all tasks to process
            all_tasks = []
            global_idx = 0  # Global index across all exercise sets
            for task in self.tasks:  
                goldens = self.load_benchmark_dataset(task)  
                
                # Apply sample limiting based on configuration
                if self.config.n_test_samples and self.config.n_test_samples < len(goldens):  
                    goldens = goldens[:self.config.n_test_samples]
                elif self.config.test_percentage:
                    # Calculate number of samples based on percentage
                    n_samples = max(1, int(len(goldens) * self.config.test_percentage))
                    if self.config.verbose:
                        print(f"  Exercise {task.exercise_set}: Limiting to {n_samples} samples ({self.config.test_percentage*100:.1f}% of {len(goldens)})")
                    goldens = goldens[:n_samples]

                if not goldens:  
                    continue  
                
                for golden in goldens:
                    all_tasks.append(process_golden_with_semaphore(golden, task, global_idx))
                    global_idx += 1
            
            # Process all tasks concurrently
            if self.config.verbose:
                print(f"Processing {len(all_tasks)} samples with max {self.config.async_config.max_concurrent} concurrent operations...")
            
            # Use asyncio.gather to run all tasks concurrently
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    if self.config.verbose:
                        print(f"Error processing sample: {result}")
                    continue
                
                if result:
                    predictions_row.append(result)
            
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
    
    async def _process_single_golden_async(self, model: DeepEvalBaseLLM, golden: Golden, task: MentorEvalTask, idx: int, 
                                         task_metrics: Dict, task_counts: Dict, all_predictions: List[float], 
                                         all_ground_truth: List[float]) -> Optional[Dict]:
        """Process a single golden sample asynchronously."""
        try:
            exercise_metrics = self.extract_metrics_from_data(golden)  
            prediction_result = await self.a_predict(model, golden, task, exercise_metrics)  

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

            result = {  
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
            }
            
            # Save interaction incrementally
            self._save_interaction(result)  

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
            
            # Add throttle delay if configured
            if self.config.async_config.throttle_value > 0:
                await asyncio.sleep(self.config.async_config.throttle_value)
            
            return result
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error processing sample {idx}: {e}")
            return None
    
    
    def _create_config_from_run_data(self, config_data):
        """Create MentorEvalConfig from configuration data."""
        from .config import BenchmarkMode, PromptType, AsyncConfig
        
        # Handle mode
        mode_str = config_data.get('mode', 'mentoreval-test')
        if mode_str == 'mentoreval':
            mode = BenchmarkMode.MENTOREVAL
        else:
            mode = BenchmarkMode.MENTOREVAL_TEST
        
        # Handle prompt type
        prompt_type_str = config_data.get('prompt_type', 'with_explanation')
        if prompt_type_str == 'grade_only':
            prompt_type = PromptType.GRADE_ONLY
        else:
            prompt_type = PromptType.WITH_EXPLANATION
        
        # Handle async configuration
        async_config_data = config_data.get('async_config', {})
        async_config = AsyncConfig(
            run_async=async_config_data.get('run_async', True),
            max_concurrent=async_config_data.get('max_concurrent', 20),
            throttle_value=async_config_data.get('throttle_value', 0.0)
        )
        
        # Create config
        config = MentorEvalConfig(
            mode=mode,
            use_few_shot=config_data.get('use_few_shot', True),
            include_rubric=config_data.get('include_rubric', True),
            prompt_type=prompt_type,
            n_test_samples=config_data.get('n_test_samples'),
            test_percentage=config_data.get('test_percentage'),
            model_name=config_data.get('model_name', 'gpt-4o-mini'),
            model_provider=config_data.get('model_provider', 'openai'),
            async_config=async_config
        )
        
        # Set verbose if specified
        if config_data.get('verbose', False):
            config.verbose = True
            
        return config
    
    def _initialize_interactions_file(self, timestamp: str, run_id: int):
        """Initialize the interactions file for incremental saving."""
        # Create detailed results directory (ignored by git)
        detailed_dir = "results_detailed"
        os.makedirs(detailed_dir, exist_ok=True)
        
        run_dir = os.path.join(detailed_dir, f"{run_id}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize interactions file
        self.interactions_file = os.path.join(run_dir, "llm_interactions.jsonl")
        
        # Save configuration for reproducibility
        config_file = os.path.join(run_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': run_id,
                'timestamp': timestamp,
                'model_name': self.config.model_name,
                'model_provider': self.config.model_provider,
                'mode': self.config.mode.value,
                'use_few_shot': self.config.use_few_shot,
                'include_rubric': self.config.include_rubric,
                'prompt_type': self.config.prompt_type.value,
                'n_test_samples': self.config.n_test_samples,
                'test_percentage': self.config.test_percentage,
                'async_config': {
                    'run_async': self.config.async_config.run_async,
                    'max_concurrent': self.config.async_config.max_concurrent,
                    'throttle_value': self.config.async_config.throttle_value
                }
            }, f, indent=2)
    
    def _save_interaction(self, interaction_data: dict):
        """Save a single LLM interaction to the JSONL file."""
        if self.interactions_file:
            with open(self.interactions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data) + '\n')
    
    def _save_results(self):
        """Save results in two formats: detailed (ignored by git) and aggregated (tracked by git)."""
        if not self.current_run:
            raise ValueError("No current run found. Call evaluate() first.")
        
        # Automatically calculate timestamp when saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.current_run.run_id
        
        # Save aggregated results (tracked by git)
        self._save_aggregated_results(timestamp, run_id)
    
    def _save_detailed_results(self, timestamp: str, run_id: int):
        """Save detailed LLM input/output results (ignored by git)."""
        # Create detailed results directory (ignored by git)
        detailed_dir = "results_detailed"
        os.makedirs(detailed_dir, exist_ok=True)
        
        run_dir = os.path.join(detailed_dir, f"{run_id}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
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
from typing import List, Dict, Optional  
import pandas as pd  
import json  
import os  
import re  
from tqdm import tqdm  
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark  
from deepeval.models import DeepEvalBaseLLM  
from deepeval.telemetry import capture_benchmark_run  
from deepeval.dataset import Golden  
  
from .task import MentorEvalTask, MentorEvalDataset, MentorEvalTasks  
from .template import MentorEvalTemplate  
  
class MentorEvalBenchmark(DeepEvalBaseBenchmark):  
    def __init__(  
        self,   
        tasks: List[MentorEvalTask] = None,  
        n_problems_per_task: int = None,   
        verbose_mode: bool = False,  
        metrics_list: List[str] = None,  
        use_test_set: bool = True  
    ):  
        super().__init__()  
        self.tasks = tasks or MentorEvalTasks.get_all_tasks()  
        self.n_problems_per_task = n_problems_per_task  
        self.verbose_mode = verbose_mode  
        # Default metrics are used as a fallback; actual metrics are extracted dynamically per sample
        self.metrics_list = metrics_list or ["Ideas", "Organization", "Style", "Conventions"]  
        self.use_test_set = use_test_set  
        self.predictions = None  
        self.task_scores = {}  
        self.dataset_scores = {}  
        self.overall_score = None  
        self.overall_metrics = None  
          
    def extract_metrics_from_data(self, golden: Golden) -> List[str]:  
        """Extract metric names dynamically from ideal_*_score fields in metadata."""  
        metadata = getattr(golden, 'additional_metadata', None) or {}  
        metrics: List[str] = []  
        for key in metadata.keys():  
            if key.startswith('ideal_') and key.endswith('_score'):  
                metric_name = key.replace('ideal_', '').replace('_score', '')  
                metrics.append(metric_name.title())  
        return metrics or self.metrics_list  
  
    def _parse_json_block(self, text: str) -> Optional[Dict]:  
        """Attempt to extract and parse the first JSON object from a text blob."""  
        try:  
            start = text.find('{')  
            end = text.rfind('}')  
            if start != -1 and end != -1 and end > start:  
                candidate = text[start:end+1]  
                return json.loads(candidate)  
        except Exception:  
            pass  
        return None  
  
    def _parse_scores_fallback(self, text: str, exercise_metrics: List[str]) -> Dict[str, float]:  
        """Fallback parser to extract numeric scores for each metric via regex if JSON parsing fails."""  
        scores: Dict[str, float] = {}  
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
        prompt = MentorEvalTemplate.generate_output(  
            question=getattr(golden, 'context', '') or metadata.get('question', ''),  
            student_answer=getattr(golden, 'input', ''),  
            rubric=metadata.get('rubric', ''),  
            academic_level=metadata.get('academic_level'),  
            essay_type=metadata.get('essay_type'),  
            metrics_list=exercise_metrics,  
            rubric_range=metadata.get('rubric_range', {}),  
            n_shots=0,  
            train_set=None,  
            task=task  
        )  
        model_output = model.generate(prompt)  
  
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
  
    def calculate_mae(self, golden: Golden, prediction_result: Dict) -> Dict[str, float]:  
        """Calculate per-metric absolute error for the current sample."""  
        exercise_metrics = self.extract_metrics_from_data(golden)  
        predicted_scores: Dict[str, float] = prediction_result.get('individual_scores', {}) or {}  
        metadata = getattr(golden, 'additional_metadata', None) or {}  
        mae_scores: Dict[str, float] = {}  
        for metric in exercise_metrics:  
            pred = predicted_scores.get(metric.title(), 0.0)  
            gt_key = f"ideal_{metric.lower()}_score"  
            gt_val = metadata.get(gt_key, 0)  
            if gt_val is None:  
                continue  
            try:  
                mae_scores[metric.title()] = abs(float(pred) - float(gt_val))  
            except Exception:  
                continue  
        return mae_scores  
  
    def calculate_evaluation_metrics(self, all_predictions: List[float], all_ground_truth: List[float]) -> Dict[str, float]:  
        """Calculate MAE and, if available, Pearson and Spearman correlations."""  
        results: Dict[str, float] = {}  
        # MAE  
        if all_predictions and len(all_predictions) == len(all_ground_truth):  
            abs_errors = [abs(float(p) - float(t)) for p, t in zip(all_predictions, all_ground_truth)]  
            results['mae'] = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0  
        else:  
            results['mae'] = 0.0  
  
        # Correlations (optional, requires scipy)  
        try:  
            if len(all_predictions) > 1:  
                from scipy.stats import pearsonr, spearmanr  
                pearson_corr, pearson_p = pearsonr(all_predictions, all_ground_truth)  
                results['pearson_correlation'] = float(pearson_corr)  
                results['pearson_p_value'] = float(pearson_p)  
  
                spearman_corr, spearman_p = spearmanr(all_predictions, all_ground_truth)  
                results['spearman_correlation'] = float(spearman_corr)  
                results['spearman_p_value'] = float(spearman_p)  
        except Exception:  
            # If scipy is unavailable, skip correlation metrics silently  
            pass  
  
        return results  
  
    def evaluate(self, model: DeepEvalBaseLLM):  
        with capture_benchmark_run("MentorEval", len(self.tasks)):  
            all_predictions: List[float] = []  
            all_ground_truth: List[float] = []  
            predictions_row = []  
  
            # For reporting per-task and per-dataset MAE averages  
            task_mae_sums: Dict[str, float] = {}  
            task_counts: Dict[str, int] = {}  
            dataset_to_task_maes: Dict[str, List[float]] = {}  
  
            for task in self.tasks:  
                goldens = self.load_benchmark_dataset(task)  
                if self.n_problems_per_task and self.n_problems_per_task < len(goldens):  
                    goldens = goldens[:self.n_problems_per_task]  
  
                if not goldens:  
                    continue  
  
                for idx, golden in enumerate(tqdm(goldens, desc=f"Processing {task.value}")):  
                    exercise_metrics = self.extract_metrics_from_data(golden)  
                    prediction_result = self.predict(model, golden, task, exercise_metrics)  
  
                    # Per-sample MAE across metrics  
                    mae_scores = self.calculate_mae(golden, prediction_result)  
                    sample_overall_mae = (sum(mae_scores.values()) / len(mae_scores)) if mae_scores else 0.0  
  
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
  
                    expected_scores = {  
                        metric.title(): metadata.get(f"ideal_{metric.lower()}_score", 0)  
                        for metric in exercise_metrics  
                    }  
  
                    predictions_row.append({  
                        'Dataset': task.dataset.value,  
                        'Exercise_Set': task.exercise_set,  
                        'Task': task.value,  
                        'Input': getattr(golden, 'input', ''),  
                        'Prediction': prediction_result,  
                        'Expected_Scores': expected_scores,  
                        'MAE_Scores': mae_scores,  
                        'Overall_MAE': sample_overall_mae  
                    })  
  
                    # Aggregate by task for reporting  
                    task_mae_sums[task.value] = task_mae_sums.get(task.value, 0.0) + sample_overall_mae  
                    task_counts[task.value] = task_counts.get(task.value, 0) + 1  
  
                    if self.verbose_mode:  
                        print(f"Sample {idx}: MAE = {sample_overall_mae:.3f}")  
  
            # Compute per-task averages and dataset aggregates  
            scores_row = []  
            for task in self.tasks:  
                if task.value in task_counts and task_counts[task.value] > 0:  
                    avg_mae = task_mae_sums[task.value] / task_counts[task.value]  
                    scores_row.append((task.dataset.value, task.exercise_set, task.value, avg_mae))  
                    dataset_to_task_maes.setdefault(task.dataset.value, []).append(avg_mae)  
                    print(f"MentorEval Task MAE (task={task.value}): {avg_mae:.3f}")  
  
            for dataset, maes in dataset_to_task_maes.items():  
                if maes:  
                    self.dataset_scores[dataset] = sum(maes) / len(maes)  
                    print(f"MentorEval Dataset MAE (dataset={dataset}): {self.dataset_scores[dataset]:.3f}")  
  
            # Global evaluation metrics  
            eval_results = self.calculate_evaluation_metrics(all_predictions, all_ground_truth)  
            print(f"Overall MentorEval MAE: {eval_results.get('mae', 0.0):.3f}")  
            if 'pearson_correlation' in eval_results:  
                print(f"Pearson Correlation: {eval_results['pearson_correlation']:.3f}")  
            if 'spearman_correlation' in eval_results:  
                print(f"Spearman Correlation: {eval_results['spearman_correlation']:.3f}")  
  
            # Create DataFrames for results  
            self.predictions = pd.DataFrame(predictions_row)  
            self.task_scores = pd.DataFrame(scores_row, columns=["Dataset", "Exercise_Set", "Task", "MAE"])  
            self.overall_score = eval_results.get('mae', 0.0)  
            self.overall_metrics = eval_results  
  
            return eval_results  
      
    def load_benchmark_dataset(self, task: MentorEvalTask) -> List[Golden]:  
        """Load dataset for specific task and exercise set"""  
        dataset_name = task.dataset.value  
        exercise_set = task.exercise_set  
        split = "test" if self.use_test_set else "train"  
          
        file_path = f"data/processed/{dataset_name}/exercise_set_{exercise_set}/{split}.jsonl"  
          
        if not os.path.exists(file_path):  
            print(f"Warning: Dataset file {file_path} not found")  
            return []  
          
        goldens = []  
        try:  
            with open(file_path, 'r') as f:  
                for line in f:  
                    data = json.loads(line.strip())  
                    golden = Golden(  
                        input=data.get('student_answer', ''),  
                        expected_output=str(data.get('ideal', '')),  
                        context=data.get('question', ''),  
                        additional_metadata={  
                            'dataset': dataset_name,  
                            'exercise_set': exercise_set,  
                            'task': task.value,  
                            'academic_level': data.get('academic_level'),  
                            'essay_type': data.get('essay_type'),  
                            'rubric': data.get('rubric', ''),  
                            'rubric_range': data.get('rubric_range', {}),  
                            'num_metrics': data.get('num_metrics', 4),  
                            'ideal_ideas_score': data.get('ideal_ideas_score'),  
                            'ideal_organization_score': data.get('ideal_organization_score'),  
                            'ideal_style_score': data.get('ideal_style_score'),  
                            'ideal_conventions_score': data.get('ideal_conventions_score'),  
                            'essay_set': data.get('essay_set')  
                        }  
                    )  
                    goldens.append(golden)  
        except FileNotFoundError:  
            print(f"Warning: Dataset file {file_path} not found")  
            return []  
          
        return goldens
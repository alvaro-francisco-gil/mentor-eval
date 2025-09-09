from typing import Dict, List, Optional  
from .task import MentorEvalTask, MentorEvalDataset  
from .config import PromptType
import json  

class MentorEvalTemplate:  
    @staticmethod  
    def generate_output(  
        question: str,  
        student_answer: str,  
        rubric: str,  
        academic_level: int = None,  
        essay_type: str = None,  
        metrics_list: List[str] = None,  
        rubric_range: Dict = None,  
        n_shots: int = 0,  
        train_set: List = None,  
        task: MentorEvalTask = None,
        prompt_type: PromptType = PromptType.WITH_EXPLANATION,
        include_rubric: bool = True
    ) -> str:
        # Choose prompt template based on type
        if prompt_type == PromptType.WITH_EXPLANATION:
            prompt = MentorEvalTemplate._generate_explanation_prompt(
                question, student_answer, rubric, academic_level, essay_type,
                metrics_list, rubric_range, n_shots, train_set, task, include_rubric
            )
        else:  # GRADE_ONLY
            prompt = MentorEvalTemplate._generate_grade_only_prompt(
                question, student_answer, rubric, academic_level, essay_type,
                metrics_list, rubric_range, n_shots, train_set, task, include_rubric
            )
        
        return prompt
    
    @staticmethod
    def _generate_explanation_prompt(
        question: str, student_answer: str, rubric: str, academic_level: int = None,
        essay_type: str = None, metrics_list: List[str] = None, rubric_range: Dict = None,
        n_shots: int = 0, train_set: List = None, task: MentorEvalTask = None,
        include_rubric: bool = True
    ) -> str:
        """Generate prompt that requires explanations for each grade."""
        prompt = "You are an expert teacher evaluating student exam answers according to a specific rubric.\n\n"  
          
        # Task and exercise set specific instructions  
        if task:  
            if task.dataset == MentorEvalDataset.ASAP:  
                prompt += f"Focus on automated essay scoring criteria for essays (Exercise Set {task.exercise_set}).\n\n"  
            elif task.dataset == MentorEvalDataset.ASAP2:  
                prompt += f"Focus on short answer scoring criteria (Exercise Set {task.exercise_set}).\n\n"  
            elif task.dataset == MentorEvalDataset.MOHLER:  
                prompt += f"Focus on Mohler dataset criteria (Exercise Set {task.exercise_set}).\n\n"  
          
        # Add few-shot examples if provided  
        if n_shots > 0 and train_set:  
            prompt += "Here are examples of correct evaluations:\n\n"  
            for i in range(min(n_shots, len(train_set))):  
                example = train_set[i]  
                prompt += MentorEvalTemplate.format_example(example)  
          
        # Add context information  
        if academic_level:  
            prompt += f"Academic Level: Grade {academic_level}\n"  
        if essay_type:  
            prompt += f"Essay Type: {essay_type}\n\n"  
          
        # Main evaluation content  
        prompt += f"Question: {question}\n\n"  
        prompt += f"Student Answer: {student_answer}\n\n"  
        
        if include_rubric:
            prompt += f"Rubric:\n{rubric}\n\n"  
          
        # Evaluation instructions with explanations
        if metrics_list and len(metrics_list) > 1:  
            prompt += f"Evaluate this answer across {len(metrics_list)} metrics: {', '.join(metrics_list)}.\n"  
            prompt += "For each metric, provide a score and briefly explain your reasoning.\n\n"  
              
            if rubric_range:  
                if isinstance(rubric_range, dict):
                    if 'ideal' in rubric_range:
                        prompt += f"Score range: {rubric_range['ideal']} for each metric\n\n"
                    else:
                        min_score = rubric_range.get('min', 0)  
                        max_score = rubric_range.get('max', 3)  
                        prompt += f"Score range: {min_score} to {max_score} for each metric\n\n"
                else:
                    prompt += f"Score range: {rubric_range} for each metric\n\n"  
              
            prompt += "Return your evaluation in the following JSON format:\n"  
            prompt += "{\n"  
            for metric in metrics_list:  
                prompt += f'  "{metric.lower()}_score": <score>,\n'  
                prompt += f'  "{metric.lower()}_explanation": "<brief explanation>",\n'  
            prompt += '  "overall_score": <sum_of_all_scores>,\n'
            prompt += '  "overall_explanation": "<brief overall assessment>"\n'
            prompt += "}\n"  
        else:  
            prompt += "Provide an overall score based on the rubric criteria and briefly explain your reasoning.\n"  
            prompt += "Return your evaluation in the following JSON format:\n"
            prompt += '{\n  "overall_score": <score>,\n  "explanation": "<brief explanation>"\n}\n'
          
        return prompt
    
    @staticmethod
    def _generate_grade_only_prompt(
        question: str, student_answer: str, rubric: str, academic_level: int = None,
        essay_type: str = None, metrics_list: List[str] = None, rubric_range: Dict = None,
        n_shots: int = 0, train_set: List = None, task: MentorEvalTask = None,
        include_rubric: bool = True
    ) -> str:
        """Generate prompt that only requires grades without explanations."""
        prompt = "You are an expert teacher evaluating student exam answers according to a specific rubric.\n\n"  
          
        # Task and exercise set specific instructions  
        if task:  
            if task.dataset == MentorEvalDataset.ASAP:  
                prompt += f"Focus on automated essay scoring criteria for essays (Exercise Set {task.exercise_set}).\n\n"  
            elif task.dataset == MentorEvalDataset.ASAP2:  
                prompt += f"Focus on short answer scoring criteria (Exercise Set {task.exercise_set}).\n\n"  
            elif task.dataset == MentorEvalDataset.MOHLER:  
                prompt += f"Focus on Mohler dataset criteria (Exercise Set {task.exercise_set}).\n\n"  
          
        # Add few-shot examples if provided  
        if n_shots > 0 and train_set:  
            prompt += "Here are examples of correct evaluations:\n\n"  
            for i in range(min(n_shots, len(train_set))):  
                example = train_set[i]  
                prompt += MentorEvalTemplate.format_example(example)  
          
        # Add context information  
        if academic_level:  
            prompt += f"Academic Level: Grade {academic_level}\n"  
        if essay_type:  
            prompt += f"Essay Type: {essay_type}\n\n"  
          
        # Main evaluation content  
        prompt += f"Question: {question}\n\n"  
        prompt += f"Student Answer: {student_answer}\n\n"  
        
        if include_rubric:
            prompt += f"Rubric:\n{rubric}\n\n"  
          
        # Evaluation instructions - grade only
        if metrics_list and len(metrics_list) > 1:  
            prompt += f"Evaluate this answer across {len(metrics_list)} metrics: {', '.join(metrics_list)}.\n"  
            prompt += "Provide only the scores for each metric.\n\n"  
              
            if rubric_range:  
                if isinstance(rubric_range, dict):
                    if 'ideal' in rubric_range:
                        prompt += f"Score range: {rubric_range['ideal']} for each metric\n\n"
                    else:
                        min_score = rubric_range.get('min', 0)  
                        max_score = rubric_range.get('max', 3)  
                        prompt += f"Score range: {min_score} to {max_score} for each metric\n\n"
                else:
                    prompt += f"Score range: {rubric_range} for each metric\n\n"  
              
            prompt += "Return your evaluation in the following JSON format:\n"  
            prompt += "{\n"  
            for metric in metrics_list:  
                prompt += f'  "{metric.lower()}_score": <score>,\n'  
            prompt += '  "overall_score": <sum_of_all_scores>\n'  
            prompt += "}\n"  
        else:  
            prompt += "Provide an overall score based on the rubric criteria.\n"  
            prompt += "Return your evaluation in the following JSON format:\n"
            prompt += '{\n  "overall_score": <score>\n}\n'
          
        return prompt
    
    @staticmethod
    def format_example(example: Dict) -> str:
        """Format a training example for few-shot prompting."""
        formatted = f"Example:\n"
        formatted += f"Question: {example.get('question', '')}\n"
        formatted += f"Student Answer: {example.get('student_answer', '')}\n"
        
        # Add rubric if available
        if 'rubric' in example:
            formatted += f"Rubric: {example.get('rubric', '')}\n"
        
        # Add expected scores
        if 'ideal' in example:
            formatted += f"Expected Score: {example.get('ideal', '')}\n"
        
        # Add individual metric scores if available
        for key, value in example.items():
            if key.startswith('ideal_') and key.endswith('_score'):
                metric_name = key.replace('ideal_', '').replace('_score', '').title()
                formatted += f"Expected {metric_name} Score: {value}\n"
        
        formatted += "\n"
        return formatted
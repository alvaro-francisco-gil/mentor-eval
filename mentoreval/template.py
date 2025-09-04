from typing import Dict, List, Optional  
from .task import MentorEvalTask, MentorEvalDataset  
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
        task: MentorEvalTask = None  
    ) -> str:  
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
        prompt += f"Rubric:\n{rubric}\n\n"  
          
        # Evaluation instructions  
        if metrics_list and len(metrics_list) > 1:  
            prompt += f"Evaluate this answer across {len(metrics_list)} metrics: {', '.join(metrics_list)}.\n"  
            prompt += "Provide a score for each metric according to the rubric.\n\n"  
              
            if rubric_range:  
                min_score = rubric_range.get('min', 0)  
                max_score = rubric_range.get('max', 3)  
                prompt += f"Score range: {min_score} to {max_score} for each metric\n\n"  
              
            prompt += "Return your evaluation in the following JSON format:\n"  
            prompt += "{\n"  
            for metric in metrics_list:  
                prompt += f'  "{metric.lower()}_score": <score>,\n'  
            prompt += '  "overall_score": <sum_of_all_scores>\n'  
            prompt += "}\n"  
        else:  
            prompt += "Provide an overall score based on the rubric criteria.\n"  
            prompt += "Score:"  
          
        return prompt
"""
Prompt templates for MentorEval tasks.

This module contains all prompt generation logic for different grading scenarios,
keeping the main task.py file lightweight and focused on task configuration.
"""

from typing import Dict, Any

# Note: Training example handling is now done by LightEval's custom few-shot selection function

def create_rubric_prompt(question: str, answer: str, rubric: str, min_grade: float, max_grade: float) -> str:
    """
    Create a prompt for rubric-based grading.
    
    Args:
        question: The essay question/prompt
        answer: Student's response
        rubric: Grading rubric
        min_grade: Minimum grade on the scale
        max_grade: Maximum grade on the scale
    
    Returns:
        str: Formatted prompt for rubric-based grading
    """
    return f"""Grade this student answer based on the rubric.

Question: {question}

Student Answer: {answer}

Rubric: {rubric}

Grade (on a scale from {min_grade} to {max_grade}):"""


def create_desired_answer_prompt(question: str, answer: str, desired_answer: str, min_grade: float, max_grade: float) -> str:
    """
    Create a prompt for desired answer comparison grading.
    
    Args:
        question: The essay question/prompt
        answer: Student's response
        desired_answer: Expected/desired answer
        min_grade: Minimum grade on the scale
        max_grade: Maximum grade on the scale
    
    Returns:
        str: Formatted prompt for desired answer comparison
    """
    return f"""Grade this student answer by comparing it to the expected answer.

Question: {question}

Student Answer: {answer}

Expected Answer: {desired_answer}

Grade (on a scale from {min_grade} to {max_grade}):"""




def get_grading_instruction(grading_type: str) -> str:
    """
    Get the appropriate instruction based on grading type.
    
    Args:
        grading_type: Type of grading ('rubric' or 'desired_answer')
    
    Returns:
        str: Instruction for the LLM
    """
    instructions = {
        'rubric': "You are an expert teacher grading student work. Provide a numerical grade based on the rubric.",
        'desired_answer': "You are an expert teacher grading student work. Compare the student answer to the expected answer and provide a numerical grade."
    }
    return instructions[grading_type]


def mentor_eval_prompt_fn(line, task_name: str = None, **kwargs):
    """
    Convert dataset rows to LightEval Doc objects for student exam grading.
    Handles two main cases:
    1. Rubric-based grading (when rubric is available)
    2. Desired answer comparison (when desired_answer is available)
    
    Note: LightEval handles few-shot examples automatically using the custom selection function.
    
    Args:
        line: Dictionary containing dataset row with keys:
              - question: The essay question/prompt
              - answer: Student's response
              - rubric: Grading rubric (optional)
              - desired_answer: Expected answer (optional)
              - grade: Expected grade (gold standard)
              - min_grade: Minimum acceptable grade
              - max_grade: Maximum acceptable grade
              - subject: Subject area (e.g., 'english')
              - exercise_type: Type of exercise (e.g., 'essay_writing')
        task_name: Optional task name for the Doc object
        **kwargs: Additional arguments (ignored, for compatibility)
    
    Returns:
        Doc: LightEval document object for evaluation
    """
    # Import Doc here to avoid circular imports
    from lighteval.tasks.requests import Doc
    
    # Extract common fields
    question = line['question']
    answer = line['answer']
    min_grade = line['min_grade']
    max_grade = line['max_grade']
    
    # Determine grading type and create appropriate prompt
    rubric = line.get('rubric')
    desired_answer = line.get('desired_answer')
    
    # Get appropriate instruction based on grading type first
    if rubric and rubric not in [None, '', 'None', 'NaN']:
        grading_type = 'rubric'
        instruction = get_grading_instruction(grading_type)
        base_query = create_rubric_prompt(question, answer, rubric, min_grade, max_grade)
    elif desired_answer and desired_answer not in [None, '', 'None', 'NaN']:
        grading_type = 'desired_answer'
        instruction = get_grading_instruction(grading_type)
        base_query = create_desired_answer_prompt(question, answer, desired_answer, min_grade, max_grade)
    else:
        grading_type = 'simple'
        instruction = "You are an expert teacher grading student work. Provide a numerical grade."
        base_query = f"""Grade this student answer.

Question: {question}

Student Answer: {answer}

Grade (on a scale from {min_grade} to {max_grade}):"""
    
    # Combine instruction and query (LightEval expects query to start with instruction)
    query = f"{instruction}\n\n{base_query}"
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(line['grade'])],
        gold_index=0,
        instruction=instruction,
        specific={
            "min_grade": min_grade,
            "max_grade": max_grade,
            "subject": line["subject"],
            "exercise_type": line["exercise_type"],
            "isced_level": line.get("isced_level", 3),
            "dataset": line.get("dataset", "unknown"),
            "exercise_set": line.get("exercise_set", 1),
            "grading_type": grading_type
        }
    )

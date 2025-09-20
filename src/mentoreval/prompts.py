"""
Prompt templates for MentorEval tasks.

This module contains all prompt generation logic for different grading scenarios,
keeping the main task.py file lightweight and focused on task configuration.
"""

from typing import Dict, Any

# ISCED level descriptions for educational context
ISCED_LEVELS = {
    0: "Early childhood education (pre-primary, initial organized instruction for young children)",
    1: "Primary education (first stage of basic education, foundational skills in reading, writing, mathematics)",
    2: "Lower secondary education (second stage of basic education, more subject-oriented)",
    3: "Upper secondary education (more specialized, typically begins around age 15-16 or end of compulsory education)",
    4: "Post-secondary non-tertiary education (programs between upper secondary and tertiary, such as pre-university or short vocational courses)",
    5: "Short-cycle tertiary education (non-degree practical or occupation-specific programmes, may lead to tertiary studies)",
    6: "Bachelor's or equivalent level (first degree, intermediate academic/professional skills, research informed)",
    7: "Master's or equivalent level (advanced skills, second degree, may include research but not a doctorate)",
    8: "Doctoral or equivalent level (advanced research qualification, usually requires a thesis or dissertation defense)"
}

# Note: Training example handling is now done by LightEval's custom few-shot selection function

def _build_prompt(question: str, answer: str, min_grade: float, max_grade: float, 
                 explanation: bool = False, show_isced_level: bool = False, isced_level: int = None,
                 rubric: str = None, desired_answer: str = None) -> str:
    """
    Build a complete prompt by conditionally adding different components.
    
    Args:
        question: The essay question/prompt
        answer: Student's response
        min_grade: Minimum grade on the scale
        max_grade: Maximum grade on the scale
        explanation: Whether to require explanation with the grade
        show_isced_level: Whether to include ISCED level context
        isced_level: The ISCED level for educational context
        rubric: Grading rubric (optional)
        desired_answer: Expected answer (optional)
    
    Returns:
        str: Complete formatted prompt
    """
    # Start with base instruction
    if explanation:
        instruction = (
            "You are an expert teacher grading student work. Provide a numerical grade and explanation. "
            "Your output must be a dictionary in the following format: "
            '{"grade": [numerical_value], "explanation": "[brief explanation]"}'
        )
    else:
        instruction = (
            "You are an expert teacher grading student work. You must provide ONLY a numerical grade in JSON format. "
            "Do not include any explanation, reasoning, or additional text. "
            "Your response must be a dictionary in the following format: {\"grade\": [numerical_value]}"
        )
    
    # Build the main content
    content_parts = []
    
    # Add ISCED level context if requested
    if show_isced_level and isced_level and isced_level in ISCED_LEVELS:
        content_parts.append(f"Educational Context: This response is from a student at ISCED level {isced_level} ({ISCED_LEVELS[isced_level]}). Please consider this educational level when evaluating the response.")
    
    # Add grading instruction based on available guidance
    if rubric and rubric not in [None, '', 'None', 'NaN']:
        content_parts.append("Grade this student answer based on the rubric.")
    elif desired_answer and desired_answer not in [None, '', 'None', 'NaN']:
        content_parts.append("Grade this student answer by comparing it to the expected answer. Evaluate the student's response based on factual accuracy and correctness, not on semantic similarity to the expected answer.")
    else:
        content_parts.append("Grade this student answer.")
    
    # Add question and answer
    content_parts.extend([
        f"Question: {question}",
        f"Student Answer: {answer}"
    ])
    
    # Add guidance if available
    if rubric and rubric not in [None, '', 'None', 'NaN']:
        content_parts.append(f"Rubric: {rubric}")
    elif desired_answer and desired_answer not in [None, '', 'None', 'NaN']:
        content_parts.append(f"Expected Answer: {desired_answer}")
    
    # Add final instruction
    content_parts.append(f"Grade (on a scale from {min_grade} to {max_grade}):")
    
    # Combine everything
    content = "\n\n".join(content_parts)
    return f"{instruction}\n\n{content}"

def create_rubric_prompt(question: str, answer: str, rubric: str, min_grade: float, max_grade: float, explanation: bool = False, show_isced_level: bool = False, isced_level: int = 3) -> str:
    """
    Create a prompt for rubric-based grading.
    
    Args:
        question: The essay question/prompt
        answer: Student's response
        rubric: Grading rubric
        min_grade: Minimum grade on the scale
        max_grade: Maximum grade on the scale
        explanation: Whether to require explanation with the grade
        show_isced_level: Whether to include ISCED level context
        isced_level: The ISCED level for educational context
    
    Returns:
        str: Formatted prompt for rubric-based grading
    """
    return _build_prompt(question, answer, min_grade, max_grade, explanation, show_isced_level, isced_level, rubric=rubric)


def create_desired_answer_prompt(question: str, answer: str, desired_answer: str, min_grade: float, max_grade: float, explanation: bool = False, show_isced_level: bool = False, isced_level: int = 3) -> str:
    """
    Create a prompt for desired answer comparison grading.
    
    Args:
        question: The essay question/prompt
        answer: Student's response
        desired_answer: Expected/desired answer
        min_grade: Minimum grade on the scale
        max_grade: Maximum grade on the scale
        explanation: Whether to require explanation with the grade
        show_isced_level: Whether to include ISCED level context
        isced_level: The ISCED level for educational context
    
    Returns:
        str: Formatted prompt for desired answer comparison
    """
    return _build_prompt(question, answer, min_grade, max_grade, explanation, show_isced_level, isced_level, desired_answer=desired_answer)


def mentor_eval_prompt_fn(line, task_name: str = None, explanation: bool = False, show_isced_level: bool = False, show_guidance: bool = True, **kwargs):
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
        explanation: Whether to require explanation with the grade
        show_isced_level: Whether to include ISCED level context in prompts
        show_guidance: Whether to include rubric/desired_answer guidance in prompts
        **kwargs: Additional arguments (ignored, for compatibility)
    
    Returns:
        Doc: LightEval document object for evaluation
    """
    # Import Doc here to avoid circular imports
    from lighteval.tasks.requests import Doc
    
    # Check if this is a few-shot example
    is_fewshot = line.get("__few_shots", False)
    
    # Extract common fields
    question = line['question']
    answer = line['answer']
    min_grade = line['min_grade']
    max_grade = line['max_grade']
    isced_level = line.get('isced_level')
    
    # For few-shot examples, create a simplified version
    if is_fewshot:
        # Simplified few-shot format: just student answer (grade will be added automatically by LightEval)
        query = f"Student Answer: {answer}"
        
        return Doc(
            task_name=task_name,
            query=query,
            choices=[str(line['grade'])],
            gold_index=0,
            specific={
                "min_grade": min_grade,
                "max_grade": max_grade,
                "subject": line["subject"],
                "exercise_type": line["exercise_type"],
                "isced_level": line.get("isced_level", 3),
                "dataset": line.get("dataset", "unknown"),
                "exercise_set": line.get("exercise_set", 1),
                "grading_type": "fewshot",
                "explanation": explanation
            }
        )
    
    # Determine grading type and create appropriate prompt
    rubric = line.get('rubric')
    desired_answer = line.get('desired_answer')
    
    # Determine grading type
    if not show_guidance:
        grading_type = 'simple'
        rubric = None  # Don't use rubric/desired_answer when show_guidance is False
        desired_answer = None
    elif rubric and rubric not in [None, '', 'None', 'NaN']:
        grading_type = 'rubric'
    elif desired_answer and desired_answer not in [None, '', 'None', 'NaN']:
        grading_type = 'desired_answer'
    else:
        grading_type = 'simple'
        rubric = None
        desired_answer = None
    
    # Build the complete prompt using the unified function
    query = _build_prompt(question, answer, min_grade, max_grade, explanation, show_isced_level, isced_level, rubric, desired_answer)
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(line['grade'])],
        gold_index=0,
        specific={
            "min_grade": min_grade,
            "max_grade": max_grade,
            "subject": line["subject"],
            "exercise_type": line["exercise_type"],
            "isced_level": line.get("isced_level", 3),
            "dataset": line.get("dataset", "unknown"),
            "exercise_set": line.get("exercise_set", 1),
            "grading_type": grading_type,
            "explanation": explanation
        }
    )

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
    # Add ISCED level context if requested
    isced_context = ""
    if show_isced_level and isced_level in ISCED_LEVELS:
        isced_context = f"\n\nEducational Context: This response is from a student at ISCED level {isced_level} ({ISCED_LEVELS[isced_level]}). Please consider this educational level when evaluating the response."
    
    base_prompt = f"""Grade this student answer based on the rubric.{isced_context}

Question: {question}

Student Answer: {answer}

Rubric: {rubric}"""
    
    if explanation:
        return f"""{base_prompt}

Provide your grade and a brief explanation. Format your response as:
Grade: [numerical value]
Explanation: [brief explanation of your grading decision]

Grade (on a scale from {min_grade} to {max_grade}):"""
    else:
        return f"""{base_prompt}

Grade (on a scale from {min_grade} to {max_grade}):"""


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
    # Add ISCED level context if requested
    isced_context = ""
    if show_isced_level and isced_level in ISCED_LEVELS:
        isced_context = f"\n\nEducational Context: This response is from a student at ISCED level {isced_level} ({ISCED_LEVELS[isced_level]}). Please consider this educational level when evaluating the response."
    
    base_prompt = f"""Grade this student answer by comparing it to the expected answer. Evaluate the student's response based on factual accuracy and correctness, not on semantic similarity to the expected answer.{isced_context}

Question: {question}

Student Answer: {answer}

Expected Answer: {desired_answer}"""
    
    if explanation:
        return f"""{base_prompt}

Provide your grade and a brief explanation. Format your response as:
Grade: [numerical value]
Explanation: [brief explanation of your grading decision]

Grade (on a scale from {min_grade} to {max_grade}):"""
    else:
        return f"""{base_prompt}

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
    
    # Extract common fields
    question = line['question']
    answer = line['answer']
    min_grade = line['min_grade']
    max_grade = line['max_grade']
    isced_level = line.get('isced_level', 3)  # Default to level 3 if not specified
    
    # Determine grading type and create appropriate prompt
    rubric = line.get('rubric')
    desired_answer = line.get('desired_answer')
    
    # Get appropriate instruction based on grading type first
    # If show_guidance is False, skip rubric/desired_answer and use simple grading
    if not show_guidance:
        grading_type = 'simple'
        if explanation:
            instruction = "You are an expert teacher grading student work. Provide a numerical grade and explanation."
        else:
            instruction = "You are an expert teacher grading student work. You must provide ONLY a numerical grade in JSON format. Do not include any explanation, reasoning, or additional text."
        
        # Add ISCED level context if requested
        isced_context = ""
        if show_isced_level and isced_level in ISCED_LEVELS:
            isced_context = f"\n\nEducational Context: This response is from a student at ISCED level {isced_level} ({ISCED_LEVELS[isced_level]}). Please consider this educational level when evaluating the response."
        
        if explanation:
            base_query = f"""Grade this student answer.{isced_context}

Question: {question}

Student Answer: {answer}

Provide your grade and a brief explanation. Format your response as:
Grade: [numerical value]
Explanation: [brief explanation of your grading decision]

Grade (on a scale from {min_grade} to {max_grade}):"""
        else:
            base_query = f"""Grade this student answer.{isced_context}

Question: {question}

Student Answer: {answer}

CRITICAL: You must provide ONLY a numerical grade. Do not include any explanation, reasoning, or additional text. Your response must be in JSON format: {{"grade": [numerical_value]}}

Grade (on a scale from {min_grade} to {max_grade}):"""
    elif rubric and rubric not in [None, '', 'None', 'NaN']:
        grading_type = 'rubric'
        instruction = get_grading_instruction(grading_type)
        base_query = create_rubric_prompt(question, answer, rubric, min_grade, max_grade, explanation, show_isced_level, isced_level)
    elif desired_answer and desired_answer not in [None, '', 'None', 'NaN']:
        grading_type = 'desired_answer'
        instruction = get_grading_instruction(grading_type)
        base_query = create_desired_answer_prompt(question, answer, desired_answer, min_grade, max_grade, explanation, show_isced_level, isced_level)
    else:
        grading_type = 'simple'
        instruction = "You are an expert teacher grading student work. Provide a numerical grade."
        
        # Add ISCED level context if requested
        isced_context = ""
        if show_isced_level and isced_level in ISCED_LEVELS:
            isced_context = f"\n\nEducational Context: This response is from a student at ISCED level {isced_level} ({ISCED_LEVELS[isced_level]}). Please consider this educational level when evaluating the response."
        
        if explanation:
            base_query = f"""Grade this student answer.{isced_context}

Question: {question}

Student Answer: {answer}

Provide your grade and a brief explanation. Format your response as:
Grade: [numerical value]
Explanation: [brief explanation of your grading decision]

Grade (on a scale from {min_grade} to {max_grade}):"""
        else:
            base_query = f"""Grade this student answer.{isced_context}

Question: {question}

Student Answer: {answer}

CRITICAL: You must provide ONLY a numerical grade. Do not include any explanation, reasoning, or additional text. Your response must be in JSON format: {{"grade": [numerical_value]}}

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
            "grading_type": grading_type,
            "explanation": explanation
        }
    )

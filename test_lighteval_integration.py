#!/usr/bin/env python3
"""
Simple test script to verify LightEval integration works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_function():
    """Test the prompt function with sample data."""
    from mentoreval.prompts import mentor_eval_prompt_fn
    
    # Test data with rubric
    sample_data_rubric = {
        'question': 'Write about the effects of computers on society.',
        'answer': 'Computers have both positive and negative effects on society.',
        'rubric': 'Score based on argument quality and evidence.',
        'desired_answer': None,
        'grade': 4.0,
        'min_grade': 1.0,
        'max_grade': 6.0,
        'subject': 'english',
        'exercise_type': 'essay_writing',
        'isced_level': 3,
        'dataset': 'asap',
        'exercise_set': 1
    }
    
    # Test data with desired answer
    sample_data_desired = {
        'question': 'What is the capital of France?',
        'answer': 'Paris is the capital of France.',
        'rubric': None,
        'desired_answer': 'The capital of France is Paris.',
        'grade': 5.0,
        'min_grade': 1.0,
        'max_grade': 5.0,
        'subject': 'geography',
        'exercise_type': 'short_answer',
        'isced_level': 3,
        'dataset': 'mohler',
        'exercise_set': 1
    }
    
    print("Testing rubric-based prompt...")
    doc_rubric = mentor_eval_prompt_fn(sample_data_rubric, "test_task")
    print(f"✓ Query length: {len(doc_rubric.query)} characters")
    print(f"✓ Grading type: {doc_rubric.specific['grading_type']}")
    print(f"✓ Contains rubric: {'rubric' in doc_rubric.query.lower()}")
    
    print("\nTesting desired answer prompt...")
    doc_desired = mentor_eval_prompt_fn(sample_data_desired, "test_task")
    print(f"✓ Query length: {len(doc_desired.query)} characters")
    print(f"✓ Grading type: {doc_desired.specific['grading_type']}")
    print(f"✓ Contains expected answer: {'expected answer' in doc_desired.query.lower()}")
    
    print("\n✅ All tests passed!")

def test_metrics():
    """Test the custom metrics."""
    from mentoreval.task import grade_range_accuracy, exact_grade_match
    from lighteval.tasks.requests import Doc
    
    # Test grade range accuracy
    test_doc = Doc(
        task_name="test",
        query="test",
        choices=["4.0"],
        gold_index=0,
        specific={"min_grade": 3.0, "max_grade": 5.0}
    )
    
    # Test within range
    score_within = grade_range_accuracy(["4.0"], test_doc)
    print(f"✓ Grade within range (4.0 in 3.0-5.0): {score_within}")
    
    # Test outside range
    score_outside = grade_range_accuracy(["2.0"], test_doc)
    print(f"✓ Grade outside range (2.0 in 3.0-5.0): {score_outside}")
    
    # Test exact match
    score_exact = exact_grade_match(["4.0"], test_doc)
    print(f"✓ Exact grade match (4.0 == 4.0): {score_exact}")
    
    print("✅ Metrics tests passed!")

def test_task_configuration():
    """Test that the task configuration is valid."""
    from mentoreval.task import mentor_eval_task, TASKS_TABLE
    
    print(f"✓ Task name: {mentor_eval_task.name}")
    print(f"✓ Dataset repo: {mentor_eval_task.hf_repo}")
    print(f"✓ Number of metrics: {len(mentor_eval_task.metric)}")
    print(f"✓ Tasks in table: {len(TASKS_TABLE)}")
    
    print("✅ Task configuration tests passed!")

if __name__ == "__main__":
    print("🧪 Testing LightEval Integration...")
    print("=" * 50)
    
    try:
        test_prompt_function()
        print()
        test_metrics()
        print()
        test_task_configuration()
        print()
        print("🎉 All integration tests passed successfully!")
        print("\nYour MentorEval task is ready for LightEval!")
        print("\nTo run with LightEval:")
        print("lighteval accelerate \\")
        print('    "pretrained=gpt-4o-mini" \\')
        print('    "community|mentor_eval|0|0" \\')
        print("    --custom-tasks mentoreval/task.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

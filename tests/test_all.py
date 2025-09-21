#!/usr/bin/env python3
"""
Main test runner for all JSONL validation tests

This file orchestrates the execution of all validation tests:
1. Basic JSONL validation
2. Metrics consistency validation
3. Score sum validation
4. Rubric range validation
"""

from datasets.test_jsonl_basic import test_all_jsonl_files
from datasets.test_metrics_consistency import test_metrics_consistency
from datasets.test_score_sums import test_score_sum_validation
from datasets.test_rubric_ranges import test_rubric_range_format, test_ideal_within_rubric_range

def run_all_tests():
    """Run all validation tests."""
    print("🧪 RUNNING ALL JSONL VALIDATION TESTS")
    print("="*60)
    
    # Test 1: Basic JSONL validation
    print("1️⃣ Running Basic JSONL Validation...")
    basic_passed = test_all_jsonl_files()
    
    # Test 2: Metrics consistency
    print("\n2️⃣ Running Metrics Consistency Validation...")
    metrics_passed = test_metrics_consistency()
    
    # Test 3: Score sum validation
    print("\n3️⃣ Running Score Sum Validation...")
    score_passed = test_score_sum_validation()
    
    # Test 4: Rubric range format validation
    print("\n4️⃣ Running Rubric Range Format Validation...")
    rubric_passed = test_rubric_range_format()
    
    # Test 5: Ideal within rubric range validation
    print("\n5️⃣ Running Ideal Within Rubric Range Validation...")
    ideal_range_passed = test_ideal_within_rubric_range()
    
    # Overall results
    print(f"\n" + "="*60)
    print("OVERALL TEST RESULTS")
    print("="*60)
    
    if basic_passed and metrics_passed and score_passed and rubric_passed and ideal_range_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ JSONL files are valid")
        print("✅ Metrics are consistent")
        print("✅ Score sums are correct")
        print("✅ Rubric range format is valid")
        print("✅ Ideal values are within rubric ranges")
    else:
        print("❌ SOME TESTS FAILED:")
        if not basic_passed:
            print("   - Basic JSONL validation failed")
        if not metrics_passed:
            print("   - Metrics consistency failed")
        if not score_passed:
            print("   - Score sum validation failed")
        if not rubric_passed:
            print("   - Rubric range format validation failed")
        if not ideal_range_passed:
            print("   - Ideal within rubric range validation failed")
    
    print("="*60)
    
    return basic_passed and metrics_passed and score_passed and rubric_passed and ideal_range_passed

if __name__ == "__main__":
    run_all_tests()

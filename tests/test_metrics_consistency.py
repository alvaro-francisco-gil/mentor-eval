#!/usr/bin/env python3
"""
Metrics consistency validation for all datasets in registry/data/mentoreval

This test ensures that num_metrics matches the count of ideal_* fields.
"""

import json
from pathlib import Path

# Path to the registry data directory
REGISTRY_DATA_DIR = Path(__file__).parent.parent / "registry" / "data" / "mentoreval"

def get_all_jsonl_files():
    """Get all JSONL files in the registry data directory."""
    jsonl_files = []
    
    if not REGISTRY_DATA_DIR.exists():
        print(f"Registry data directory not found: {REGISTRY_DATA_DIR}")
        return []
    
    for dataset_dir in REGISTRY_DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            for jsonl_file in dataset_dir.glob("*.jsonl"):
                jsonl_files.append(jsonl_file)
    
    return jsonl_files

def validate_metrics_consistency(file_path):
    """Validate that num_metrics matches the count of ideal_* fields."""
    results = {
        "file_path": str(file_path),
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "errors": [],
        "example_errors": []  # Store examples of failing lines
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                results["total_lines"] += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    
                    # Check if num_metrics field exists
                    if "num_metrics" not in sample:
                        error_msg = f"Line {line_num}: Missing 'num_metrics' field"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "error": error_msg,
                            "available_fields": [k for k in sample.keys() if k.startswith('ideal_') or k in ['ideal', 'num_metrics']]
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    num_metrics = sample["num_metrics"]
                    
                    # Count ideal_* fields (excluding 'ideal' itself)
                    ideal_score_fields = [key for key in sample.keys() if key.startswith("ideal_") and key != "ideal"]
                    
                    # Special case: if num_metrics is 1, just check for 'ideal' field
                    if num_metrics == 1:
                        if "ideal" not in sample:
                            error_msg = f"Line {line_num}: num_metrics=1 but missing 'ideal' field"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "num_metrics": num_metrics,
                                "ideal_fields_found": ideal_score_fields
                            })
                            results["invalid_lines"] += 1
                            continue
                    else:
                        # Check if count matches num_metrics
                        if len(ideal_score_fields) != num_metrics:
                            error_msg = f"Line {line_num}: num_metrics={num_metrics} but found {len(ideal_score_fields)} ideal_* fields: {ideal_score_fields}"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "num_metrics": num_metrics,
                                "ideal_fields_found": ideal_score_fields,
                                "expected_count": num_metrics,
                                "actual_count": len(ideal_score_fields)
                            })
                            results["invalid_lines"] += 1
                            continue
                    
                    results["valid_lines"] += 1
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    results["errors"].append(error_msg)
                    results["example_errors"].append({
                        "line_num": line_num,
                        "error": error_msg
                    })
                    results["invalid_lines"] += 1
                except Exception as e:
                    error_msg = f"Line {line_num}: Unexpected error - {e}"
                    results["errors"].append(error_msg)
                    results["example_errors"].append({
                        "line_num": line_num,
                        "error": error_msg
                    })
                    results["invalid_lines"] += 1
                    
    except Exception as e:
        results["errors"].append(f"File reading error: {e}")
    
    return results

def test_metrics_consistency():
    """Test that num_metrics matches the count of ideal_* fields."""
    jsonl_files = get_all_jsonl_files()
    
    if not jsonl_files:
        print("No JSONL files found in registry data directory")
        return True
    
    print(f"Found {len(jsonl_files)} JSONL files to validate:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.relative_to(Path.cwd())}")
    
    print("\n" + "="*60)
    print("METRICS CONSISTENCY VALIDATION")
    print("="*60)
    
    all_passed = True
    
    for jsonl_file in jsonl_files:
        results = validate_metrics_consistency(jsonl_file)
        
        print(f"\n📁 {results['file_path']}")
        print(f"   Total lines: {results['total_lines']:,}")
        print(f"   Valid lines: {results['valid_lines']:,}")
        print(f"   Invalid lines: {results['invalid_lines']:,}")
        
        if results["errors"]:
            print(f"   ❌ ERRORS:")
            for error in results["errors"][:3]:  # Show first 3 errors
                print(f"      - {error}")
            if len(results["errors"]) > 3:
                print(f"      ... and {len(results['errors']) - 3} more errors")
            
            # Show examples of failing lines with focused debugging info
            print(f"\n   🔍 DEBUGGING INFO (First 3 failures):")
            for example in results["example_errors"][:3]:
                print(f"      Line {example['line_num']}: {example['error']}")
                
                # Show specific debugging info based on error type
                if 'available_fields' in example:
                    print(f"         Available fields: {example['available_fields']}")
                if 'num_metrics' in example:
                    print(f"         Num metrics: {example['num_metrics']}")
                if 'ideal_fields_found' in example:
                    print(f"         Ideal fields found: {example['ideal_fields_found']}")
                if 'expected_count' in example:
                    print(f"         Expected count: {example['expected_count']}")
                if 'actual_count' in example:
                    print(f"         Actual count: {example['actual_count']}")
                print()
            
            all_passed = False
        else:
            print(f"   ✅ All lines have consistent metrics")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All files have consistent metrics!")
        print("✅ num_metrics matches ideal_* field count")
    else:
        print("❌ Some files have inconsistent metrics")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    test_metrics_consistency()

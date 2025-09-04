#!/usr/bin/env python3
"""
Score sum validation for all datasets in data/processed

This test ensures that ideal equals the sum of all ideal_* values.
"""

import json
from pathlib import Path

# Path to the registry data directory
REGISTRY_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

def get_all_jsonl_files():
    """Get all JSONL files in the data directory (recursive search)."""
    jsonl_files = []
    
    if not REGISTRY_DATA_DIR.exists():
        print(f"Data directory not found: {REGISTRY_DATA_DIR}")
        return []
    
    # Search recursively for all .jsonl files
    for jsonl_file in REGISTRY_DATA_DIR.rglob("*.jsonl"):
        jsonl_files.append(jsonl_file)
    
    return jsonl_files

def validate_score_sum(file_path):
    """Validate that ideal equals the sum of all ideal_* values."""
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
                    
                    # Check if ideal field exists
                    if "ideal" not in sample:
                        error_msg = f"Line {line_num}: Missing 'ideal' field"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "error": error_msg
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    ideal_value = sample["ideal"]
                    
                    # Get all ideal_* fields (excluding 'ideal' itself)
                    ideal_score_fields = [key for key in sample.keys() if key.startswith("ideal_") and key != "ideal"]
                    
                    if ideal_score_fields:
                        # Calculate sum of all ideal_* values
                        try:
                            score_sum = sum(int(sample[field]) for field in ideal_score_fields)
                            
                            # Check if sum matches ideal
                            if score_sum != int(ideal_value):
                                error_msg = f"Line {line_num}: ideal={ideal_value} but sum of scores={score_sum} (scores: {[f'{field}={sample[field]}' for field in ideal_score_fields]})"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "ideal_value": ideal_value,
                                    "ideal_fields": {field: sample[field] for field in ideal_score_fields},
                                    "calculated_sum": score_sum
                                })
                                results["invalid_lines"] += 1
                                continue
                        except (ValueError, TypeError) as e:
                            error_msg = f"Line {line_num}: Error calculating score sum - {e}"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "ideal_value": ideal_value,
                                "ideal_fields": {field: sample[field] for field in ideal_score_fields}
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

def test_score_sum_validation():
    """Test that ideal equals the sum of all ideal_* values."""
    jsonl_files = get_all_jsonl_files()
    
    if not jsonl_files:
        print("No JSONL files found in registry data directory")
        return True
    
    print(f"Found {len(jsonl_files)} JSONL files to validate:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.relative_to(Path.cwd())}")
    
    print("\n" + "="*60)
    print("SCORE SUM VALIDATION")
    print("="*60)
    
    all_passed = True
    
    for jsonl_file in jsonl_files:
        results = validate_score_sum(jsonl_file)
        
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
                if 'ideal_value' in example:
                    print(f"         Ideal value: {example['ideal_value']}")
                if 'ideal_fields' in example:
                    print(f"         Ideal fields: {example['ideal_fields']}")
                if 'calculated_sum' in example:
                    print(f"         Calculated sum: {example['calculated_sum']}")
                print()
            
            all_passed = False
        else:
            print(f"   ✅ All lines have correct score sums")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All files have correct score sums!")
        print("✅ ideal equals sum of all ideal_* values")
    else:
        print("❌ Some files have incorrect score sums")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    test_score_sum_validation()

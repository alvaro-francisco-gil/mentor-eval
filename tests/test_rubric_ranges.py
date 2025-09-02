#!/usr/bin/env python3
"""
Rubric range validation for all datasets in registry/data/mentoreval

This test ensures that:
1. rubric_range follows the correct format: 'X-Y' or dict with 'X-Y' values
2. ideal values fall within their respective rubric ranges
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

def validate_rubric_range_format(file_path):
    """Validate that rubric_range follows the correct format: 'X-Y' or dict with 'X-Y' values."""
    results = {
        "file_path": str(file_path),
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "errors": [],
        "example_errors": []  # Store examples of failing lines
    }
    
    def is_valid_range_format(value):
        """Check if a value follows the 'X-Y' format where X and Y are integers."""
        if isinstance(value, str):
            # Check if it's a string with format "X-Y"
            parts = value.split('-')
            if len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    return True
                except ValueError:
                    return False
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                results["total_lines"] += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    
                    # Check if rubric_range field exists
                    if "rubric_range" not in sample:
                        error_msg = f"Line {line_num}: Missing 'rubric_range' field"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "error": error_msg
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    rubric_range = sample["rubric_range"]
                    
                    # Check if it's a string with valid format
                    if isinstance(rubric_range, str):
                        if not is_valid_range_format(rubric_range):
                            error_msg = f"Line {line_num}: Invalid rubric_range format '{rubric_range}' - should be 'X-Y' where X and Y are integers"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "invalid_value": rubric_range
                            })
                            results["invalid_lines"] += 1
                            continue
                    
                    # Check if it's a dictionary with valid format values
                    elif isinstance(rubric_range, dict):
                        for key, value in rubric_range.items():
                            if not is_valid_range_format(value):
                                error_msg = f"Line {line_num}: Invalid rubric_range value '{value}' for key '{key}' - should be 'X-Y' where X and Y are integers"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "invalid_key": key,
                                    "invalid_value": value
                                })
                                results["invalid_lines"] += 1
                                break
                        else:
                            # All values are valid
                            pass
                    
                    # If it's neither string nor dict, it's invalid
                    else:
                        error_msg = f"Line {line_num}: Invalid rubric_range type {type(rubric_range).__name__} - should be string or dictionary"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "error": error_msg,
                            "invalid_type": type(rubric_range).__name__
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

def validate_ideal_within_rubric_range(file_path):
    """Validate that ideal values fall within their respective rubric ranges."""
    results = {
        "file_path": str(file_path),
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "errors": [],
        "example_errors": []  # Store examples of failing lines
    }
    
    def parse_range(range_str):
        """Parse a range string 'X-Y' and return (min_val, max_val)."""
        try:
            parts = range_str.split('-')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
        return None, None
    
    def is_value_in_range(value, range_str):
        """Check if a value falls within the given range."""
        min_val, max_val = parse_range(range_str)
        if min_val is None or max_val is None:
            return False
        try:
            int_value = int(value)
            return min_val <= int_value <= max_val
        except (ValueError, TypeError):
            return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                results["total_lines"] += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    
                    # Check if required fields exist
                    if "num_metrics" not in sample or "rubric_range" not in sample:
                        error_msg = f"Line {line_num}: Missing required fields 'num_metrics' or 'rubric_range'"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "error": error_msg,
                            "missing_fields": [field for field in ['num_metrics', 'rubric_range'] if field not in sample]
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    num_metrics = sample["num_metrics"]
                    rubric_range = sample["rubric_range"]
                    
                    # Case 1: Single metric (num_metrics = 1)
                    if num_metrics == 1:
                        if "ideal" not in sample:
                            error_msg = f"Line {line_num}: num_metrics=1 but missing 'ideal' field"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg
                            })
                            results["invalid_lines"] += 1
                            continue
                        
                        # Check if ideal is within rubric_range
                        if isinstance(rubric_range, str):
                            if not is_value_in_range(sample["ideal"], rubric_range):
                                error_msg = f"Line {line_num}: ideal={sample['ideal']} is outside rubric_range '{rubric_range}'"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "ideal_value": sample["ideal"],
                                    "rubric_range": rubric_range
                                })
                                results["invalid_lines"] += 1
                                continue
                        else:
                            error_msg = f"Line {line_num}: num_metrics=1 but rubric_range is not a string"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "num_metrics": num_metrics,
                                "rubric_range_type": type(rubric_range).__name__
                            })
                            results["invalid_lines"] += 1
                            continue
                    
                    # Case 2: Multiple metrics (num_metrics > 1)
                    else:
                        if not isinstance(rubric_range, dict):
                            error_msg = f"Line {line_num}: num_metrics={num_metrics} but rubric_range is not a dictionary"
                            results["errors"].append(error_msg)
                            results["example_errors"].append({
                                "line_num": line_num,
                                "error": error_msg,
                                "num_metrics": num_metrics,
                                "rubric_range_type": type(rubric_range).__name__
                            })
                            results["invalid_lines"] += 1
                            continue
                        
                        # Check each ideal_* value against its corresponding rubric range
                        ideal_fields = [key for key in sample.keys() if key.startswith("ideal_") and key != "ideal"]
                        
                        for ideal_field in ideal_fields:
                            if ideal_field not in sample:
                                error_msg = f"Line {line_num}: Missing {ideal_field} field"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "missing_field": ideal_field
                                })
                                results["invalid_lines"] += 1
                                continue
                            
                            # Find corresponding rubric range for this field
                            # Convert ideal_ideas_score -> ideas_score
                            field_key = ideal_field.replace("ideal_", "").replace("_score", "")
                            
                            # Look for matching key in rubric_range
                            matching_range = None
                            for rubric_key, rubric_value in rubric_range.items():
                                if field_key in rubric_key or rubric_key in field_key:
                                    matching_range = rubric_value
                                    break
                            
                            if matching_range is None:
                                error_msg = f"Line {line_num}: No matching rubric range found for {ideal_field}"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "ideal_field": ideal_field,
                                    "ideal_value": sample[ideal_field],
                                    "available_rubric_keys": list(rubric_range.keys())
                                })
                                results["invalid_lines"] += 1
                                continue
                            
                            # Check if ideal value is within range
                            if not is_value_in_range(sample[ideal_field], matching_range):
                                error_msg = f"Line {line_num}: {ideal_field}={sample[ideal_field]} is outside rubric range '{matching_range}'"
                                results["errors"].append(error_msg)
                                results["example_errors"].append({
                                    "line_num": line_num,
                                    "error": error_msg,
                                    "ideal_field": ideal_field,
                                    "ideal_value": sample[ideal_field],
                                    "rubric_range": matching_range
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

def test_rubric_range_format():
    """Test that rubric_range follows the correct format: 'X-Y' or dict with 'X-Y' values."""
    jsonl_files = get_all_jsonl_files()
    
    if not jsonl_files:
        print("No JSONL files found in registry data directory")
        return True
    
    print(f"Found {len(jsonl_files)} JSONL files to validate:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.relative_to(Path.cwd())}")
    
    print("\n" + "="*60)
    print("RUBRIC RANGE FORMAT VALIDATION")
    print("="*60)
    
    all_passed = True
    
    for jsonl_file in jsonl_files:
        results = validate_rubric_range_format(jsonl_file)
        
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
                if 'invalid_value' in example:
                    print(f"         Invalid value: {example['invalid_value']}")
                if 'invalid_key' in example:
                    print(f"         Invalid key: {example['invalid_key']}")
                if 'invalid_type' in example:
                    print(f"         Invalid type: {example['invalid_type']}")
                print()
            
            all_passed = False
        else:
            print(f"   ✅ All lines have valid rubric_range format")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All files have valid rubric_range format!")
        print("✅ All rubric_range values follow 'X-Y' format")
    else:
        print("❌ Some files have invalid rubric_range format")
    print("="*60)
    
    return all_passed

def test_ideal_within_rubric_range():
    """Test that ideal values fall within their respective rubric ranges."""
    jsonl_files = get_all_jsonl_files()
    
    if not jsonl_files:
        print("No JSONL files found in registry data directory")
        return True
    
    print(f"\n" + "="*60)
    print("IDEAL WITHIN RUBRIC RANGE VALIDATION")
    print("="*60)
    
    all_passed = True
    
    for jsonl_file in jsonl_files:
        results = validate_ideal_within_rubric_range(jsonl_file)
        
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
                if 'ideal_field' in example:
                    print(f"         Field: {example['ideal_field']}")
                    print(f"         Value: {example['ideal_value']}")
                if 'available_rubric_keys' in example:
                    print(f"         Available rubric keys: {example['available_rubric_keys']}")
                if 'rubric_range' in example:
                    print(f"         Rubric range: {example['rubric_range']}")
                if 'missing_fields' in example:
                    print(f"         Missing fields: {example['missing_fields']}")
                if 'num_metrics' in example:
                    print(f"         Num metrics: {example['num_metrics']}")
                if 'rubric_range_type' in example:
                    print(f"         Rubric range type: {example['rubric_range_type']}")
                print()
            
            all_passed = False
        else:
            print(f"   ✅ All lines have ideal values within rubric ranges")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All files have ideal values within rubric ranges!")
        print("✅ All ideal values fall within their respective ranges")
    else:
        print("❌ Some files have ideal values outside rubric ranges")
    print("="*60)
    
    return all_passed

def test_all_rubric_validations():
    """Run all rubric-related validation tests."""
    print("🧪 RUNNING ALL RUBRIC RANGE VALIDATION TESTS")
    print("="*60)
    
    # Test 1: Rubric range format validation
    format_passed = test_rubric_range_format()
    
    # Test 2: Ideal within rubric range validation
    ideal_range_passed = test_ideal_within_rubric_range()
    
    # Overall results
    print(f"\n" + "="*60)
    print("OVERALL RUBRIC VALIDATION RESULTS")
    print("="*60)
    
    if format_passed and ideal_range_passed:
        print("🎉 ALL RUBRIC TESTS PASSED!")
        print("✅ Rubric range format is valid")
        print("✅ Ideal values are within rubric ranges")
    else:
        print("❌ SOME RUBRIC TESTS FAILED:")
        if not format_passed:
            print("   - Rubric range format validation failed")
        if not ideal_range_passed:
            print("   - Ideal within rubric range validation failed")
    
    print("="*60)
    
    return format_passed and ideal_range_passed

if __name__ == "__main__":
    test_all_rubric_validations()

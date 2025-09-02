#!/usr/bin/env python3
"""
Basic JSONL validation for all datasets in registry/data/mentoreval

This test ensures that all JSONL files are properly formatted with valid JSON syntax
and contain the required 'input' and 'ideal' fields on every line.
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

def validate_jsonl_file(file_path):
    """Validate a single JSONL file for basic JSON syntax and required fields."""
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
                    # Try to parse JSON
                    sample = json.loads(line.strip())
                    
                    # Check for required fields
                    if "input" not in sample:
                        error_msg = f"Line {line_num}: Missing 'input' field"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "line_content": line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                            "error": error_msg
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    if "ideal" not in sample:
                        error_msg = f"Line {line_num}: Missing 'ideal' field"
                        results["errors"].append(error_msg)
                        results["example_errors"].append({
                            "line_num": line_num,
                            "line_content": line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                            "error": error_msg
                        })
                        results["invalid_lines"] += 1
                        continue
                    
                    results["valid_lines"] += 1
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {e}"
                    results["errors"].append(error_msg)
                    results["example_errors"].append({
                        "line_num": line_num,
                        "line_content": line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                        "error": error_msg
                    })
                    results["invalid_lines"] += 1
                except Exception as e:
                    error_msg = f"Line {line_num}: Unexpected error - {e}"
                    results["errors"].append(error_msg)
                    results["example_errors"].append({
                        "line_num": line_num,
                        "line_content": line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                        "error": error_msg
                    })
                    results["invalid_lines"] += 1
                    
    except Exception as e:
        results["errors"].append(f"File reading error: {e}")
    
    return results

def test_all_jsonl_files():
    """Test that all JSONL files have valid JSON syntax and required fields."""
    jsonl_files = get_all_jsonl_files()
    
    if not jsonl_files:
        print("No JSONL files found in registry data directory")
        return True
    
    print(f"Found {len(jsonl_files)} JSONL files to validate:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.relative_to(Path.cwd())}")
    
    print("\n" + "="*60)
    print("JSONL BASIC VALIDATION RESULTS")
    print("="*60)
    
    all_passed = True
    
    for jsonl_file in jsonl_files:
        results = validate_jsonl_file(jsonl_file)
        
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
            
            # Show examples of failing lines
            print(f"\n   📋 EXAMPLES OF FAILING LINES:")
            for example in results["example_errors"][:3]:  # Show first 3 examples
                print(f"      Line {example['line_num']}: {example['error']}")
                print(f"      Content: {example['line_content']}")
                print()
            
            all_passed = False
        else:
            print(f"   ✅ All lines valid with required fields")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All JSONL files are valid!")
        print("✅ All lines have 'input' and 'ideal' fields")
    else:
        print("❌ Some JSONL files have errors")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    test_all_jsonl_files()

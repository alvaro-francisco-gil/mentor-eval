#!/usr/bin/env python3
"""
Simple test runner for JSONL validation

This script runs the basic JSONL syntax validation test.
"""

import sys
from pathlib import Path

# Add the tests directory to the path so we can import our test module
sys.path.insert(0, str(Path(__file__).parent))

try:
    from test_jsonl_validation import test_all_jsonl_files
except ImportError as e:
    print(f"Error importing test module: {e}")
    sys.exit(1)

def main():
    """Run the JSONL validation test."""
    print("🚀 MENTOREVAL JSONL VALIDATION TEST")
    print("=" * 60)
    
    # Check if registry data directory exists
    registry_dir = Path(__file__).parent.parent / "registry" / "data" / "mentoreval"
    if not registry_dir.exists():
        print(f"❌ Registry data directory not found: {registry_dir}")
        print("Please run the dataset processing scripts first.")
        sys.exit(1)
    
    # Run the test
    success = test_all_jsonl_files()
    
    if success:
        print("\n🎉 JSONL validation passed!")
        sys.exit(0)
    else:
        print("\n❌ JSONL validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

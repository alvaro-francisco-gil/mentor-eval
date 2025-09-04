#!/usr/bin/env python3
"""
Dataset Standardization Script

This script orchestrates the processing of all datasets in the MentorEval benchmark.
It calls individual processing scripts for each dataset to create standardized JSONL files.

Usage:
    python scripts/dataset_processing/process_datasets.py [--datasets asap,asap2,mohler] [--test-size 0.2]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_script(script_path, dataset_name):
    """Run a processing script for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Change to the script's directory to run the script
        original_cwd = os.getcwd()
        os.chdir(script_path.parent)
        
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🚀 Running: {script_path.name}")
        
        # Run the script
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path.name], 
                              capture_output=True, text=True, encoding='utf-8')
        end_time = time.time()
        
        # Print output
        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  STDERR:")
            print(result.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"✅ {dataset_name.upper()} processing completed successfully!")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"❌ {dataset_name.upper()} processing failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {dataset_name} script: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def check_output_files(dataset_name):
    """Check if output files were created successfully in the new data/processed layout"""
    base_dir = Path("data/processed") / dataset_name
    if not base_dir.exists():
        print(f"❌ Output directory not found: {base_dir}")
        return False
    
    # Prefer per-set layout: data/processed/<dataset>/exercise_set_*/(train|test).jsonl
    set_dirs = sorted([p for p in base_dir.glob("exercise_set_*") if p.is_dir()])
    if set_dirs:
        ok = True
        total_train = 0
        total_test = 0
        for set_dir in set_dirs:
            train_file = set_dir / "train.jsonl"
            test_file = set_dir / "test.jsonl"
            if not (train_file.exists() and test_file.exists()):
                print(f"❌ Missing files in {set_dir} (train/test)")
                ok = False
                continue
            train_count = sum(1 for _ in open(train_file, 'r', encoding='utf-8'))
            test_count = sum(1 for _ in open(test_file, 'r', encoding='utf-8'))
            total_train += train_count
            total_test += test_count
            print(f"   📁 {set_dir.name}: train={train_count}, test={test_count}")
        if ok:
            print(f"📊 Total: train={total_train}, test={total_test}")
        return ok
    
    # Fallback to flat files (if any dataset still uses it)
    train_file = base_dir / "train.jsonl"
    test_file = base_dir / "test.jsonl"
    if train_file.exists() and test_file.exists():
        train_count = sum(1 for _ in open(train_file, 'r', encoding='utf-8'))
        test_count = sum(1 for _ in open(test_file, 'r', encoding='utf-8'))
        print(f"📊 Output files created:")
        print(f"   📁 {train_file}: {train_count} samples")
        print(f"   📁 {test_file}: {test_count} samples")
        return True
    
    print(f"❌ No output files found for {dataset_name} under {base_dir}")
    return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Standardize all datasets for MentorEval benchmark")
    parser.add_argument("--datasets", 
                       default="asap,asap2,mohler",
                       help="Comma-separated list of datasets to process (default: all)")
    parser.add_argument("--test-size", 
                       type=float, 
                       default=0.3,
                       help="Test set size as fraction (default: 0.3)")
    parser.add_argument("--check-only", 
                       action="store_true",
                       help="Only check existing output files without processing")
    
    args = parser.parse_args()
    
    # Parse datasets to process
    datasets_to_process = [d.strip() for d in args.datasets.split(",")]
    
    print("🚀 MENTOREVAL DATASET STANDARDIZATION")
    print("="*60)
    print(f"📋 Datasets to process: {', '.join(datasets_to_process)}")
    print(f"📊 Test size: {args.test_size}")
    print(f"🔍 Check only: {args.check_only}")
    
    # Define script paths (relative to repo root)
    script_paths = {
        'asap': Path("scripts/dataset_processing/process_asap.py"),
        'asap2': Path("scripts/dataset_processing/process_asap2.py"),
        'mohler': Path("scripts/dataset_processing/process_mohler.py"),
    }
    
    if args.check_only:
        print("\n🔍 CHECKING EXISTING OUTPUT FILES")
        print("="*60)
        
        for dataset in datasets_to_process:
            if dataset in script_paths:
                print(f"\n📁 Checking {dataset.upper()}...")
                check_output_files(dataset)
            else:
                print(f"⚠️  Unknown dataset: {dataset}")
        
        return
    
    # Process datasets
    successful_datasets = []
    failed_datasets = []
    
    for dataset in datasets_to_process:
        if dataset not in script_paths:
            print(f"⚠️  Unknown dataset: {dataset}, skipping...")
            continue
        
        script_path = script_paths[dataset]
        
        # Pass test_size to the script via environment variable (in case any script uses it)
        env = os.environ.copy()
        env['TEST_SIZE'] = str(args.test_size)
        
        if run_script(script_path, dataset):
            if check_output_files(dataset):
                successful_datasets.append(dataset)
            else:
                failed_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    if successful_datasets:
        print(f"✅ Successfully processed: {', '.join(successful_datasets)}")
    
    if failed_datasets:
        print(f"❌ Failed to process: {', '.join(failed_datasets)}")
        sys.exit(1)
    
    print(f"\n🎉 All datasets processed successfully!")
    print(f"📁 Output files are available in: data/processed/")
    
    # Final check
    print(f"\n🔍 FINAL OUTPUT CHECK")
    print("="*60)
    
    for dataset in successful_datasets:
        print(f"\n📁 {dataset.upper()}:")
        check_output_files(dataset)

if __name__ == "__main__":
    main()

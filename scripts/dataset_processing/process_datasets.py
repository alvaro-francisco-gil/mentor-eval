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
import pandas as pd
import numpy as np

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
    
    # Check for unified parquet/CSV files first (new format)
    parquet_file = base_dir / f"{dataset_name}_processed.parquet"
    csv_file = base_dir / f"{dataset_name}_processed.csv"
    
    if parquet_file.exists() and csv_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            print(f"📊 Unified dataset files created:")
            print(f"   📁 {parquet_file}: {len(df):,} samples")
            print(f"   📁 {csv_file}: {len(df):,} samples")
            print(f"   📊 Exercise sets: {sorted(df['exercise_set'].unique())}")
            print(f"   📊 Grade range: {df['grade'].min()} - {df['grade'].max()}")
            return True
        except Exception as e:
            print(f"❌ Error reading parquet file: {e}")
            return False
    
    # Fallback to per-set layout: data/processed/<dataset>/exercise_set_*/(train|test).jsonl
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

def combine_all_datasets(successful_datasets):
    """Combine all processed datasets into a single parquet file"""
    print(f"\n{'='*60}")
    print("COMBINING ALL DATASETS")
    print(f"{'='*60}")
    
    combined_data = []
    total_samples = 0
    
    for dataset in successful_datasets:
        # Look for the processed parquet file
        parquet_file = Path("data/processed") / dataset / f"{dataset}_processed.parquet"
        
        if parquet_file.exists():
            print(f"📁 Loading {dataset.upper()} from {parquet_file}")
            df = pd.read_parquet(parquet_file)
            combined_data.append(df)
            total_samples += len(df)
            print(f"   ✅ Loaded {len(df)} samples")
        else:
            print(f"⚠️  Parquet file not found for {dataset}: {parquet_file}")
    
    if not combined_data:
        print("❌ No datasets found to combine")
        return False
    
    # Combine all datasets
    print(f"\n🔄 Combining {len(combined_data)} datasets...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    combined_file = output_dir / "mentoreval_combined.parquet"
    combined_df.to_parquet(combined_file, index=False)
    
    print(f"✅ Combined dataset saved to: {combined_file}")
    print(f"📊 Total samples: {len(combined_df)}")
    print(f"📊 Datasets included: {', '.join(successful_datasets)}")
    
    # Print dataset breakdown
    print(f"\n📋 Dataset breakdown:")
    for dataset in successful_datasets:
        count = len(combined_df[combined_df['dataset'] == dataset])
        percentage = (count / len(combined_df)) * 100
        print(f"   {dataset}: {count:,} samples ({percentage:.1f}%)")
    
    # Print exercise set breakdown
    print(f"\n📋 Exercise sets by dataset:")
    for dataset in successful_datasets:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            exercise_sets = sorted(dataset_df['exercise_set'].unique())
            print(f"   {dataset}: {len(exercise_sets)} exercise sets {exercise_sets}")
    
    # Print grade distribution
    print(f"\n📋 Grade distribution:")
    grade_dist = combined_df['grade'].value_counts().sort_index()
    for grade, count in grade_dist.items():
        percentage = (count / len(combined_df)) * 100
        print(f"   Grade {grade}: {count:,} samples ({percentage:.1f}%)")
    
    return True

def update_datasets_info(successful_datasets):
    """Update datasets_info.csv with aggregated statistics from processed datasets"""
    print(f"\n{'='*60}")
    print("UPDATING DATASETS INFO")
    print(f"{'='*60}")
    
    # Read the original datasets_info.csv
    datasets_info_file = Path("data/datasets_info.csv")
    if not datasets_info_file.exists():
        print(f"❌ datasets_info.csv not found: {datasets_info_file}")
        return False
    
    # Read original data
    df_info = pd.read_csv(datasets_info_file)
    print(f"📁 Loaded datasets_info.csv with {len(df_info)} datasets")
    
    # Analyze each successful dataset
    for _, row in df_info.iterrows():
        dataset_id = row['id']
        
        if dataset_id not in successful_datasets:
            print(f"⚠️  Skipping {dataset_id} (not in successful datasets)")
            continue
        
        # Get statistics for this dataset
        parquet_file = Path("data/processed") / dataset_id / f"{dataset_id}_processed.parquet"
        
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                
                # Count unique exercise sets
                num_exercises = df['exercise_set'].nunique()
                
                # Count total student answers (rows)
                num_student_answers = len(df)
                
                # Update the row in the DataFrame
                mask = df_info['id'] == dataset_id
                df_info.loc[mask, 'number_exercises'] = num_exercises
                df_info.loc[mask, 'number_student_answers'] = num_student_answers
                
                print(f"✅ Updated {dataset_id}: {num_exercises} exercises, {num_student_answers:,} student answers")
                
            except Exception as e:
                print(f"❌ Error analyzing {dataset_id}: {e}")
                continue
        else:
            print(f"⚠️  Parquet file not found for {dataset_id}: {parquet_file}")
    
    # Add the new columns if they don't exist
    if 'number_exercises' not in df_info.columns:
        df_info['number_exercises'] = np.nan
    if 'number_student_answers' not in df_info.columns:
        df_info['number_student_answers'] = np.nan
    
    # Reorder columns to put new columns after existing ones
    columns = list(df_info.columns)
    if 'number_exercises' in columns:
        columns.remove('number_exercises')
    if 'number_student_answers' in columns:
        columns.remove('number_student_answers')
    columns.extend(['number_exercises', 'number_student_answers'])
    df_info = df_info[columns]
    
    # Save updated datasets_info.csv
    df_info.to_csv(datasets_info_file, index=False)
    
    print(f"✅ Updated datasets_info.csv saved to: {datasets_info_file}")
    
    # Print summary
    print(f"\n📋 DATASETS INFO SUMMARY:")
    for _, row in df_info.iterrows():
        dataset_id = row['id']
        num_exercises = row.get('number_exercises', np.nan)
        num_answers = row.get('number_student_answers', np.nan)
        language = row['language']
        
        if pd.notna(num_exercises) and pd.notna(num_answers):
            avg_per_exercise = num_answers / num_exercises
            print(f"   {dataset_id}: {language}, {num_exercises} exercises, {num_answers:,} answers (avg: {avg_per_exercise:.1f}/exercise)")
        else:
            print(f"   {dataset_id}: {language}, exercises: {num_exercises}, answers: {num_answers}")
    
    # Overall statistics
    total_exercises = df_info['number_exercises'].sum()
    total_answers = df_info['number_student_answers'].sum()
    
    if pd.notna(total_exercises) and pd.notna(total_answers) and total_exercises > 0:
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total datasets: {len(df_info)}")
        print(f"   Total exercises: {total_exercises:,}")
        print(f"   Total student answers: {total_answers:,}")
        print(f"   Average answers per exercise: {total_answers/total_exercises:.1f}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Standardize all datasets for MentorEval benchmark")
    parser.add_argument("--datasets", 
                       default="asap,asap2,mohler,ellipse,ptasag2018",
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
        'ellipse': Path("scripts/dataset_processing/process_ellipse.py"),
        'ptasag2018': Path("scripts/dataset_processing/process_ptasag2018.py"),
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
    
    # Combine all datasets into a single parquet file
    if successful_datasets:
        combine_all_datasets(successful_datasets)
    
    # Update datasets_info.csv with aggregated statistics
    if successful_datasets:
        update_datasets_info(successful_datasets)
    
    # Final check
    print(f"\n🔍 FINAL OUTPUT CHECK")
    print("="*60)
    
    for dataset in successful_datasets:
        print(f"\n📁 {dataset.upper()}:")
        check_output_files(dataset)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to upload the MentorEval dataset to Hugging Face Hub with stratified splits.

This script:
1. Loads the mentoreval.parquet dataset
2. Creates stratified splits (20% train, 80% test) by dataset and grade ranges
3. Uploads the dataset to Hugging Face Hub
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import argparse
import os
from pathlib import Path

def create_grade_bins(grades, n_bins=5):
    """Create grade bins for stratification."""
    # Use quantile-based binning to handle different grade scales across datasets
    return pd.qcut(grades, q=n_bins, labels=False, duplicates='drop')

def create_stratification_key(df):
    """Create a stratification key combining dataset and grade bins."""
    # Create grade bins for each dataset separately to handle different scales
    df = df.copy()
    df['grade_bin'] = df.groupby('dataset')['grade'].transform(
        lambda x: create_grade_bins(x, n_bins=5)
    )
    
    # Create stratification key
    df['stratify_key'] = df['dataset'] + '_' + df['grade_bin'].astype(str)
    return df

def stratified_split(df, test_size=0.2, random_state=42):
    """Create stratified train/test split."""
    print(f"Original dataset shape: {df.shape}")
    
    # Create stratification key
    df_stratified = create_stratification_key(df)
    
    # Check stratification key distribution
    stratify_counts = df_stratified['stratify_key'].value_counts()
    print(f"\nStratification key distribution:")
    print(stratify_counts.head(10))
    
    # Check for keys with only one sample (can't be split)
    single_sample_keys = stratify_counts[stratify_counts == 1]
    if len(single_sample_keys) > 0:
        print(f"\nWarning: {len(single_sample_keys)} stratification keys have only 1 sample")
        print("These will be randomly assigned to train/test")
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df_stratified,
        test_size=test_size,
        stratify=df_stratified['stratify_key'],
        random_state=random_state
    )
    
    # Remove the temporary columns
    train_df = train_df.drop(['grade_bin', 'stratify_key'], axis=1)
    test_df = test_df.drop(['grade_bin', 'stratify_key'], axis=1)
    
    print(f"\nTrain set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Verify stratification worked
    print(f"\nDataset distribution in train set:")
    print(train_df['dataset'].value_counts())
    print(f"\nDataset distribution in test set:")
    print(test_df['dataset'].value_counts())
    
    return train_df, test_df

def upload_to_huggingface(train_df, test_df, repo_name, token=None):
    """Upload the dataset to Hugging Face Hub."""
    print(f"\nCreating dataset splits...")
    
    # Create dataset splits (remove_index=True to avoid __index_level_0__ column)
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    print(f"Dataset created with {len(dataset_dict['train'])} train samples and {len(dataset_dict['test'])} test samples")
    
    # Upload to Hub
    print(f"\nUploading to Hugging Face Hub: {repo_name}")
    try:
        # Upload with custom data files
        dataset_dict.push_to_hub(
            repo_name,
            token=token,
            private=False
        )
        
        # Now upload the parquet files with custom names
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Save datasets to temporary parquet files
        train_df.to_parquet('mentoreval_train.parquet', index=False)
        test_df.to_parquet('mentoreval_test.parquet', index=False)
        
        # Upload the custom named files to data/ folder
        api.upload_file(
            path_or_fileobj='mentoreval_train.parquet',
            path_in_repo='data/mentoreval_train.parquet',
            repo_id=repo_name,
            repo_type='dataset'
        )
        api.upload_file(
            path_or_fileobj='mentoreval_test.parquet',
            path_in_repo='data/mentoreval_test.parquet',
            repo_id=repo_name,
            repo_type='dataset'
        )
        
        # Upload README.md from data folder to root
        readme_path = 'data/README.md'
        if os.path.exists(readme_path):
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo='README.md',
                repo_id=repo_name,
                repo_type='dataset'
            )
            print("✅ Uploaded README.md to root")
        
        # Clean up temporary files
        os.remove('mentoreval_train.parquet')
        os.remove('mentoreval_test.parquet')
        
        print(f"✅ Successfully uploaded dataset to {repo_name}")
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Upload MentorEval dataset to Hugging Face Hub')
    parser.add_argument('--data_path', type=str, default='data/mentoreval.parquet',
                       help='Path to the mentoreval.parquet file')
    parser.add_argument('--repo_name', type=str, default='alvaro-francisco-gil/mentor-eval',
                       help='Hugging Face repository name')
    parser.add_argument('--test_size', type=float, default=0.8,
                       help='Proportion of data for test set (default: 0.8)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--token', type=str, default=None,
                       help='Hugging Face token (if not set, will use cached token)')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    print("Loading dataset...")
    df = pd.read_parquet(args.data_path)
    
    print("Creating stratified splits...")
    train_df, test_df = stratified_split(
        df, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    print("Uploading to Hugging Face Hub...")
    upload_to_huggingface(
        train_df, 
        test_df, 
        args.repo_name, 
        token=args.token
    )
    
    print("\n🎉 Dataset upload completed successfully!")

if __name__ == "__main__":
    main()

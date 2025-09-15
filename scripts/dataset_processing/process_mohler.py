#!/usr/bin/env python3
"""
Script to process the Mohler dataset for mentor evaluation.

This script processes the mohler_dataset_edited.csv file and creates a parquet file
with the required fields for the mentor evaluation framework.

Usage:
    python scripts/process_mohler.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RAW_DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'mohler' / 'mohler_dataset_edited.csv'
OUTPUT_PARQUET = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'mohler' / 'mohler_processed.parquet'
OUTPUT_CSV = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'mohler' / 'mohler_processed.csv'


def process_mohler_data():
    """
    Process the Mohler dataset according to the specified requirements.
    """
    logger.info("Loading Mohler dataset...")
    
    # Load the CSV file
    df = pd.read_csv(RAW_DATA_FILE)
    logger.info(f"Loaded dataset with {len(df)} rows")
    
    # Filter rows where score_me == score_other
    filtered_df = df[df['score_me'] == df['score_other']].copy()
    logger.info(f"Filtered to {len(filtered_df)} rows where score_me == score_other")
    
    # Create the processed dataframe with required fields
    processed_df = pd.DataFrame()
    
    # Add dataset name
    processed_df['dataset'] = ['mohler'] * len(filtered_df)
    
    # Process exercise_set: map unique ID values to sequential integers starting from 1
    unique_ids = filtered_df['id'].dropna().unique()
    id_to_exercise_set = {id_val: idx + 1 for idx, id_val in enumerate(sorted(unique_ids))}
    processed_df['exercise_set'] = filtered_df['id'].map(id_to_exercise_set)
    
    # Fill any remaining NaN values with a default value (shouldn't happen but just in case)
    processed_df['exercise_set'] = processed_df['exercise_set'].fillna(1)
    
    # Add question
    processed_df['question'] = filtered_df['question'].values
    
    # Add answer (student_answer)
    processed_df['answer'] = filtered_df['student_answer'].values
    
    # Add grade (using score_me since we filtered for score_me == score_other)
    processed_df['grade'] = filtered_df['score_me'].values
    
    # Add min_grade and max_grade
    # Based on the data, scores appear to be on a 1-5 scale
    processed_df['min_grade'] = [1] * len(filtered_df)
    processed_df['max_grade'] = [5] * len(filtered_df)
    
    # Add subject
    processed_df['subject'] = ['computer_science'] * len(filtered_df)
    
    # Add exercise_type
    processed_df['exercise_type'] = ['short_answer'] * len(filtered_df)
    
    # Add isced_level
    processed_df['isced_level'] = [6] * len(filtered_df)
    
    # Add language
    processed_df['language'] = ['english'] * len(filtered_df)
    
    # Add rubric (NaN as requested)
    processed_df['rubric'] = [np.nan] * len(filtered_df)
    
    # Add desired_answer
    processed_df['desired_answer'] = filtered_df['desired_answer'].values
    
    # Add metadata (NaN as requested)
    processed_df['metadata'] = [np.nan] * len(filtered_df)
    
    logger.info(f"Created processed dataset with {len(processed_df)} rows and {len(processed_df.columns)} columns")
    logger.info(f"Columns: {list(processed_df.columns)}")
    
    # Show some statistics
    logger.info(f"Exercise sets: {sorted(processed_df['exercise_set'].unique())}")
    logger.info(f"Grade distribution: {processed_df['grade'].value_counts().sort_index()}")
    
    return processed_df


def save_processed_data(df, parquet_file, csv_file):
    """
    Save the processed dataframe to both parquet and CSV files.
    """
    # Create output directory if it doesn't exist
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(parquet_file, index=False)
    logger.info(f"Saved processed data to {parquet_file}")
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved processed data to {csv_file}")


def main():
    """Main function to process the Mohler dataset."""
    try:
        # Check if input file exists
        if not RAW_DATA_FILE.exists():
            logger.error(f"Input file not found: {RAW_DATA_FILE}")
            sys.exit(1)
        
        # Process the data
        processed_df = process_mohler_data()
        
        # Save to both parquet and CSV
        save_processed_data(processed_df, OUTPUT_PARQUET, OUTPUT_CSV)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

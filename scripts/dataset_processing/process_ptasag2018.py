#!/usr/bin/env python3
"""
PTASAG2018 Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTASAG2018Processor:
    def __init__(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'ptasag2018'
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'ptasag2018'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self):
        """Load the main CSV file and questions file"""
        # Load the main dataset
        main_file = self.data_dir / "student_answers_and_grades_v2.csv"
        questions_file = self.data_dir / "questions.csv"
        
        if not main_file.exists():
            raise FileNotFoundError(f"Could not find main file: {main_file}")
        if not questions_file.exists():
            raise FileNotFoundError(f"Could not find questions file: {questions_file}")
        
        # Load datasets
        main_df = pd.read_csv(main_file)
        questions_df = pd.read_csv(questions_file)
        
        logger.info(f"Loaded main dataset with {len(main_df)} rows")
        logger.info(f"Loaded questions dataset with {len(questions_df)} rows")
        
        # Merge with questions to get question text
        merged_df = main_df.merge(questions_df, on='question_id', how='left')
        
        logger.info(f"Final dataset with {len(merged_df)} rows after merging with questions")
        
        self.data = merged_df
        return self.data
    
    def clean_data(self):
        """Clean the data"""
        # Remove rows with missing essential data
        self.data = self.data.dropna(subset=['question_id', 'answer_text', 'grade', 'question_text'])
        
        # Convert grade to numeric
        self.data['grade'] = pd.to_numeric(self.data['grade'], errors='coerce')
        self.data = self.data.dropna(subset=['grade'])
        
        # Filter valid scores (0-3 range)
        self.data = self.data[(self.data['grade'] >= 0) & (self.data['grade'] <= 3)]
        
        logger.info(f"After cleaning: {len(self.data)} rows")
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        for _, row in self.data.iterrows():
            # Create metadata JSON (empty for this dataset as no demographic info available)
            metadata_json = np.nan
            
            unified_sample = {
                'dataset': 'ptasag2018',
                'exercise_set': int(row['question_id']),
                'question': row['question_text'],
                'answer': row['answer_text'],
                'grade': float(row['grade']),
                'min_grade': 0.0,
                'max_grade': 3.0,
                'subject': 'biology',  # Based on the sample questions about cells, genetics, etc.
                'exercise_type': 'short_answer',
                'isced_level': 3,  # High school level
                'rubric': np.nan,  # No rubric information available
                'desired_answer': np.nan,  # NaN as requested
                'metadata': metadata_json
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def save_unified_dataset(self, unified_df):
        """Save the unified dataset as both CSV and parquet files"""
        # Create output directory
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_dir / 'ptasag2018_processed.csv'
        unified_df.to_csv(csv_file, index=False)
        logger.info(f"Saved unified PTASAG2018 dataset to {csv_file}")
        
        # Save as parquet
        parquet_file = output_dir / 'ptasag2018_processed.parquet'
        unified_df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved unified PTASAG2018 dataset to {parquet_file}")
        
        # Print statistics
        logger.info(f"Unified dataset contains {len(unified_df)} rows")
        logger.info(f"Exercise sets: {sorted(unified_df['exercise_set'].unique())}")
        logger.info(f"Number of unique questions: {len(unified_df['exercise_set'].unique())}")
        logger.info(f"Grade distribution:")
        logger.info(unified_df['grade'].value_counts().sort_index())
    
    def process(self):
        """Main processing function"""
        logger.info("Starting PTASAG2018 dataset processing...")
        
        self.load_dataset()
        self.clean_data()
        
        # Create unified dataset
        logger.info("Creating unified dataset...")
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df)
        
        logger.info(f"\nSummary:")
        logger.info(f"Total samples: {len(unified_df)}")
        logger.info("PTASAG2018 dataset processing completed!")


def main():
    processor = PTASAG2018Processor()
    processor.process()


if __name__ == "__main__":
    main()

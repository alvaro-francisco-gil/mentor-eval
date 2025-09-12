#!/usr/bin/env python3
"""
ELLIPSE Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

# Prompt mapping from the original file
prompt_mapping = {
  "Distance learning": "Should high school students take online classes instead of attending school in person?",
  "Career commitment": "Should high school students commit to a career path at a young age?",
  "Success and failure": "What role do success and failure play in achieving your goals?",
  "Being busy": "Do you agree with Thomas Jefferson's statement that people should never be idle in order to accomplish more?",
  "Positive attitudes": "Is having a positive attitude the key to success in life?",
  "Impact of technology": "Does technology have a positive or negative impact on people's lives?",
  "Trying something beyond what you have mastered": "Do you agree with Emerson's idea that you only grow by trying things beyond what you have already mastered?",
  "First impressions": "Can first impressions of people change over time?",
  "Three-year high school program": "Should high school students be allowed to graduate in three years instead of four?",
  "Individuality": "Is being yourself in a world that is trying to change you the greatest accomplishment?",
  "Four-day work week": "Should schools adopt a four-day school week instead of the traditional five days?",
  "Working with a group or alone": "Is it better for students to work in groups or to work alone?",
  "Self-reliance": "Is it better to make your own decisions or to seek advice from others?",
  "Lunch menus": "Should schools change their lunch menus to offer healthier food options?",
  "Influences of character": "Are our character traits something we choose for ourselves, or are they shaped by outside influences?",
  "Extended school day": "Should the school day be extended to give students more time for learning?",
  "Year-round school": "Should summer vacation be longer, shorter, or replaced by year-round schooling?",
  "Influencing behavior": "Is setting a good example the best way to influence others?",
  "Cell phones in classrooms": "Should students be allowed to use cell phones in classrooms for educational purposes?",
  "Internships and shadowing": "Should schools offer students opportunities for internships or job shadowing?",
  "Enjoyable educational activities": "What are some enjoyable educational activities, and why are they valuable?",
  "Praising student work": "Does true self-esteem come more from achievement or from praise?",
  "Afterschool homework club": "Should schools have an afterschool homework club to help students?",
  "Imagination": "Do you agree with Albert Einstein's statement that imagination is more important than knowledge?",
  "Spending time outdoors": "What are the benefits of spending time outdoors, such as going to the park?",
  "Curfews for teenagers": "Should teenagers have a 10 p.m. curfew to keep them safe and out of trouble?",
  "Controlling extracurricular involvement": "Who should decide how students spend their time outside of class — the school or the students themselves?",
  "Seeking multiple opinions": "Can asking for multiple opinions help people make better decisions?",
  "Creative arts requirement": "Should students be required to take creative arts classes such as music, drama, or art?",
  "Places to visit": "If you could visit any place, where would you go and why?",
  "Community service": "Should community service be a required part of school education?",
  "Letter to employer": "Should older students be paired with younger students to help teach and guide them?",
  "Cell phones at school": "Should schools allow students to bring and use cell phones?",
  "Talents and skills": "What is one of your talents or skills, and how do you use it in your life?",
  "Mandatory extracurricular activities": "Should all students be required to participate in extracurricular activities?",
  "Lessons with elementary school students": "What is one lesson or experience from elementary school that was important to you?",
  "Grades for extracurricular activities": "Should students be required to maintain a certain grade point average to participate in sports?",
  "Future accomplishments": "Is it better to set high goals, even if you might not reach them, or to set small, achievable goals?",
  "Benefits of a problem": "Do you agree with Duke Ellington's idea that a problem is a chance to do your best?",
  "Learning from the experience of others": "Can people learn important lessons from the experiences of others?",
  "Benefits of a good attitude": "How can having a good attitude help people during difficult times?",
  "Honesty": "Why is it important to be honest?",
  "Setting our aim": "Is it better to set high goals, even if you might not reach them, or to set small, achievable goals?",
  "Summer projects": "Should students be allowed to design their own summer projects instead of being assigned one by teachers?"
}

class ELLIPSEProcessor:
    def __init__(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'ellipse'
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'ellipse'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_mapping = prompt_mapping
    
    def load_rubric(self):
        """Load the rubric from the markdown file"""
        rubric_file = self.data_dir / "rubric.md"
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""
    
    def load_dataset(self):
        """Load both train and test datasets and combine them"""
        train_file = self.data_dir / "ELLIPSE_Final_github_train.csv"
        test_file = self.data_dir / "ELLIPSE_Final_github_test.csv"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Could not find train file: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Could not find test file: {test_file}")
        
        # Load both datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Combine them
        self.data = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"Loaded ELLIPSE dataset with {len(self.data)} rows")
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        return self.data
    
    def clean_data(self):
        """Clean the data"""
        # Remove rows with missing essential data
        self.data = self.data.dropna(subset=['full_text', 'prompt'])
        
        # Check for required scoring columns
        scoring_columns = ['Cohesion', 'Syntax', 'Vocabulary', 'Phraseology', 'Grammar', 'Conventions']
        missing_columns = [col for col in scoring_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required scoring columns: {missing_columns}")
        
        # Convert scoring columns to numeric
        for col in scoring_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Remove rows with missing scores
        self.data = self.data.dropna(subset=scoring_columns)
        
        # Calculate total grade as sum of the 6 scoring dimensions, rounded up to nearest integer
        self.data['total_grade'] = np.ceil(
            self.data['Cohesion'] + 
            self.data['Syntax'] + 
            self.data['Vocabulary'] + 
            self.data['Phraseology'] + 
            self.data['Grammar'] + 
            self.data['Conventions']
        ).astype(int)
        
        # Filter valid scores (5-30 range)
        self.data = self.data[(self.data['total_grade'] >= 5) & (self.data['total_grade'] <= 30)]
        
        print(f"After cleaning: {len(self.data)} rows")
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        rubric_text = self.load_rubric()
        
        # Create mapping from prompt to exercise_set (sequential integers)
        unique_prompts = sorted(self.data['prompt'].unique())
        prompt_to_exercise_set = {prompt: idx + 1 for idx, prompt in enumerate(unique_prompts)}
        
        for _, row in self.data.iterrows():
            # Create metadata JSON from demographic fields
            metadata_dict = {}
            if pd.notna(row.get('gender')):
                metadata_dict['gender'] = row['gender']
            if pd.notna(row.get('grade')):
                metadata_dict['grade'] = row['grade']
            if pd.notna(row.get('race_ethnicity')):
                metadata_dict['race_ethnicity'] = row['race_ethnicity']
            if pd.notna(row.get('SES')):
                metadata_dict['SES'] = row['SES']
            
            # Convert metadata to JSON string, or NaN if empty
            metadata_json = json.dumps(metadata_dict) if metadata_dict else np.nan
            
            # Get question from prompt mapping
            question = self.prompt_mapping.get(row['prompt'], row['prompt'])
            
            unified_sample = {
                'dataset': 'ellipse',
                'exercise_set': prompt_to_exercise_set[row['prompt']],
                'question': question,
                'answer': row['full_text'],
                'grade': float(row['total_grade']),
                'min_grade': 5.0,
                'max_grade': 30.0,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'rubric': rubric_text,
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
        csv_file = output_dir / 'ellipse_processed.csv'
        unified_df.to_csv(csv_file, index=False)
        print(f"Saved unified ELLIPSE dataset to {csv_file}")
        
        # Save as parquet
        parquet_file = output_dir / 'ellipse_processed.parquet'
        unified_df.to_parquet(parquet_file, index=False)
        print(f"Saved unified ELLIPSE dataset to {parquet_file}")
        
        # Print statistics
        print(f"Unified dataset contains {len(unified_df)} rows")
        print(f"Exercise sets: {sorted(unified_df['exercise_set'].unique())}")
        print(f"Number of unique prompts: {len(unified_df['exercise_set'].unique())}")
        print(f"Grade distribution:")
        print(unified_df['grade'].value_counts().sort_index())
    
    def process(self):
        """Main processing function"""
        print("Starting ELLIPSE dataset processing...")
        
        self.load_dataset()
        self.clean_data()
        
        # Create unified dataset
        print("Creating unified dataset...")
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df)
        
        print(f"\nSummary:")
        print(f"Total samples: {len(unified_df)}")
        print("ELLIPSE dataset processing completed!")


def main():
    processor = ELLIPSEProcessor()
    processor.process()


if __name__ == "__main__":
    main()
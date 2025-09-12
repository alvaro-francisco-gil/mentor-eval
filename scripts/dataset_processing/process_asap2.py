#!/usr/bin/env python3
"""
ASAP2 Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

class ASAP2Processor:
    def __init__(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'asap2'
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'asap2'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exercise_sets = {}
        self._load_exercise_set_metadata()
    
    def _load_exercise_set_metadata(self):
        for i in range(1, 7 + 1):  # ASAP2 has 7 exercise sets
            set_dir = self.data_dir / f"exercise_set_{i}"
            if set_dir.exists():
                metadata = self._extract_set_metadata(set_dir, i)
                self.exercise_sets[i] = metadata
    
    def _extract_set_metadata(self, set_dir, set_num):
        metadata = {
            'set_id': set_num,
            'question': '',
            'rubric': '',
            'complementary_texts': '',
            'academic_level': '10',
            'rubric_range': {"ideal": "1-6"},
            'essay_type': 'argumentative',
            'scoring_type': 'holistic',
            'num_metrics': 1
        }
        
        # Prefer question.txt if it exists; otherwise parse from all_exercise_description.md
        question_txt = set_dir / "question.txt"
        if question_txt.exists():
            with open(question_txt, 'r', encoding='utf-8') as f:
                metadata['question'] = f.read().strip()
        else:
            question_file = set_dir / "all_exercise_description.md"
            if question_file.exists():
                with open(question_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    question_match = re.search(r'## Question\s*\n\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
                    if question_match:
                        metadata['question'] = question_match.group(1).strip()
        
        rubric_file = self.data_dir / "rubric.txt"
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                metadata['rubric'] = f.read().strip()
        
        comp_texts_file = set_dir / "exercise_texts.txt"
        if comp_texts_file.exists():
            with open(comp_texts_file, 'r', encoding='utf-8') as f:
                metadata['complementary_texts'] = f.read().strip()
        
        return metadata
    
    def load_dataset(self):
        data_file = self.data_dir / "asap2_student_responses_and_evaluations.csv"
        if not data_file.exists():
            raise FileNotFoundError("Could not find ASAP2 dataset file")
        self.data = pd.read_csv(data_file)
        return self.data
    
    def clean_data(self):
        self.data['score'] = pd.to_numeric(self.data['score'], errors='coerce')
        self.data = self.data.dropna(subset=['score'])
        self.data = self.data[(self.data['score'] >= 1) & (self.data['score'] <= 6)]
        return self.data
    
    def create_samples(self):
        samples = []
        for _, row in self.data.iterrows():
            exercise_set = row['exercise_set']
            if exercise_set not in self.exercise_sets:
                continue
            metadata = self.exercise_sets[exercise_set]
            
            question_text = metadata['question']
            if metadata.get('complementary_texts'):
                question_text = f"{metadata['complementary_texts'].strip()}\n\n{question_text}"
            
            sample = {
                "input": [{
                    "role": "user", 
                    "content": "Question: {question}\n\nStudent Answer: {student_answer}\n\nRubric: {rubric}\n\nEvaluate this response."
                }],
                "question": question_text,
                "student_answer": row['student_text'],
                "rubric": metadata['rubric'],
                "academic_level": metadata['academic_level'],
                "rubric_range": metadata['rubric_range'],
                "essay_type": metadata['essay_type'],
                "exercise_set": exercise_set,
                "num_metrics": metadata['num_metrics'],
                "ideal": str(int(row['score'])),
                "economically_disadvantaged": row['economically_disadvantaged'] if pd.notna(row['economically_disadvantaged']) else None,
                "student_disability_status": row['student_disability_status'] if pd.notna(row['student_disability_status']) else None,
                "ell_status": row['ell_status'] if pd.notna(row['ell_status']) else None,
                "race_ethnicity": row['race_ethnicity'] if pd.notna(row['race_ethnicity']) else None,
                "gender": row['gender'] if pd.notna(row['gender']) else None,
                "exercise": row['exercise'] if pd.notna(row['exercise']) else None
            }
            samples.append(sample)
        return samples
    
    def create_unified_dataset(self, samples):
        """Create a unified dataset in the same format as Mohler dataset"""
        unified_data = []
        
        for sample in samples:
            # Extract grade from ideal field
            try:
                grade = float(sample.get('ideal', 0))
            except (ValueError, TypeError):
                grade = 0.0
            
            # Create metadata JSON from demographic fields
            metadata_dict = {}
            if sample.get('economically_disadvantaged') is not None:
                metadata_dict['economically_disadvantaged'] = sample['economically_disadvantaged']
            if sample.get('student_disability_status') is not None:
                metadata_dict['student_disability_status'] = sample['student_disability_status']
            if sample.get('ell_status') is not None:
                metadata_dict['ell_status'] = sample['ell_status']
            if sample.get('race_ethnicity') is not None:
                metadata_dict['race_ethnicity'] = sample['race_ethnicity']
            if sample.get('gender') is not None:
                metadata_dict['gender'] = sample['gender']
            if sample.get('exercise') is not None:
                metadata_dict['exercise'] = sample['exercise']
            
            # Convert metadata to JSON string, or NaN if empty
            metadata_json = json.dumps(metadata_dict) if metadata_dict else np.nan
            
            unified_sample = {
                'dataset': 'asap2',
                'exercise_set': sample.get('exercise_set', 1),
                'question': sample.get('question', ''),
                'answer': sample.get('student_answer', ''),
                'grade': grade,
                'min_grade': 1.0,
                'max_grade': 6.0,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'rubric': sample.get('rubric', ''),
                'desired_answer': np.nan,  # NaN as requested
                'metadata': metadata_json
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def save_unified_dataset(self, unified_df):
        """Save the unified dataset as both CSV and parquet files"""
        # Create output directory (use the main asap2 output directory)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_dir / 'asap2_processed.csv'
        unified_df.to_csv(csv_file, index=False)
        print(f"Saved unified ASAP2 dataset to {csv_file}")
        
        # Save as parquet
        parquet_file = output_dir / 'asap2_processed.parquet'
        unified_df.to_parquet(parquet_file, index=False)
        print(f"Saved unified ASAP2 dataset to {parquet_file}")
        
        # Print statistics
        print(f"Unified dataset contains {len(unified_df)} rows")
        print(f"Exercise sets: {sorted(unified_df['exercise_set'].unique())}")
        print(f"Grade distribution:")
        print(unified_df['grade'].value_counts().sort_index())
    
    
    def process(self):
        print("Starting ASAP2 dataset processing...")
        self.load_dataset()
        self.clean_data()
        samples = self.create_samples()
        
        # Create unified dataset in the same format as Mohler
        print("Creating unified dataset...")
        unified_df = self.create_unified_dataset(samples)
        self.save_unified_dataset(unified_df)
        
        print(f"\nSummary:")
        print(f"Total samples: {len(samples)}")
        print("ASAP2 dataset processing completed!")


def main():
    np.random.seed(42)
    processor = ASAP2Processor()
    processor.process()

if __name__ == "__main__":
    main()

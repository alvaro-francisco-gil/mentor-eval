#!/usr/bin/env python3
"""
ASAP2 Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

class ASAP2Processor:
    def __init__(self, data_dir=".", output_dir="../../registry/data/mentoreval/asap2"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exercise_sets = {}
        self._load_exercise_set_metadata()
    
    def _load_exercise_set_metadata(self):
        for i in range(1, 8):  # ASAP2 has 7 exercise sets
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
            'academic_level': '10',  # ASAP2 is Grade 10
            'rubric_range': '1-6',  # All ASAP2 exercises use 1-6 scale
            'essay_type': 'argumentative',  # Based on the examples I saw
            'scoring_type': 'holistic',  # Single holistic score
            'num_metrics': 1  # All ASAP2 exercises have single score
        }
        
        # Load question from all_exercise_description.md
        question_file = set_dir / "all_exercise_description.md"
        if question_file.exists():
            with open(question_file, 'r', encoding='utf-8') as f:
                content = f.read()
                question_match = re.search(r'## Question\s*\n\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
                if question_match:
                    metadata['question'] = question_match.group(1).strip()
        
        # Load rubric (same for all exercises)
        rubric_file = self.data_dir / "rubric.txt"
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                metadata['rubric'] = f.read().strip()
        
        # Load complementary texts
        comp_texts_file = set_dir / "complementary_exercise_texts.txt"
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
        # Convert score to numeric to handle any data type issues
        self.data['score'] = pd.to_numeric(self.data['score'], errors='coerce')
        
        # Remove rows with missing score
        self.data = self.data.dropna(subset=['score'])
        
        # Ensure scores are within valid range (1-6)
        self.data = self.data[(self.data['score'] >= 1) & (self.data['score'] <= 6)]
        
        return self.data
    
    def create_samples(self):
        """Convert dataset rows to standardized sample format for training/evaluation"""
        samples = []
        
        for _, row in self.data.iterrows():
            exercise_set = row['exercise_set']
            if exercise_set not in self.exercise_sets:
                continue
                
            metadata = self.exercise_sets[exercise_set]
            
            # Create base sample with essay content and metadata
            sample = {
                "input": [{
                    "role": "user", 
                    "content": "Question: {question}\n\nStudent Answer: {student_answer}\n\nRubric: {rubric}\n\nEvaluate this response."
                }],
                "question": metadata['question'],
                "student_answer": row['student_text'],
                "rubric": metadata['rubric'],
                "academic_level": metadata['academic_level'],
                "rubric_range": metadata['rubric_range'],
                "essay_type": metadata['essay_type'],
                "exercise_set": exercise_set,
                "num_metrics": metadata['num_metrics'],
                "complementary_texts": metadata['complementary_texts'],
                "ideal": str(int(row['score'])),  # Single holistic score
                # Include all demographic fields from CSV
                "economically_disadvantaged": row['economically_disadvantaged'] if pd.notna(row['economically_disadvantaged']) else None,
                "student_disability_status": row['student_disability_status'] if pd.notna(row['student_disability_status']) else None,
                "ell_status": row['ell_status'] if pd.notna(row['ell_status']) else None,
                "race_ethnicity": row['race_ethnicity'] if pd.notna(row['race_ethnicity']) else None,
                "gender": row['gender'] if pd.notna(row['gender']) else None,
                "exercise": row['exercise'] if pd.notna(row['exercise']) else None
            }
            
            samples.append(sample)
        
        return samples
    
    def create_train_test_splits(self, samples, test_size=0.3, random_state=42):
        train_samples = []
        test_samples = []
        
        for exercise_set in range(1, 8):  # ASAP2 has 7 exercise sets
            set_samples = [s for s in samples if s['exercise_set'] == exercise_set]
            if len(set_samples) > 0:
                score_counts = {}
                for sample in set_samples:
                    score = sample['ideal']
                    score_counts[score] = score_counts.get(score, 0) + 1
                
                can_stratify = all(count >= 2 for count in score_counts.values()) and len(score_counts) > 1
                
                if can_stratify:
                    set_train, set_test = train_test_split(
                        set_samples, 
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=[s['ideal'] for s in set_samples]
                    )
                else:
                    set_train, set_test = train_test_split(
                        set_samples, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                
                train_samples.extend(set_train)
                test_samples.extend(set_test)
        
        return train_samples, test_samples
    
    def save_jsonl(self, samples, filename):
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(samples)} samples to {filename}")
    
    def process(self, test_size=0.3, random_state=42):
        """Main processing pipeline: load, clean, create samples, split, and save"""
        print("Starting ASAP2 dataset processing...")
        
        self.load_dataset()
        self.clean_data()
        
        samples = self.create_samples()
        train_samples, test_samples = self.create_train_test_splits(samples, test_size, random_state)
        
        self.save_jsonl(train_samples, "train.jsonl")
        self.save_jsonl(test_samples, "test.jsonl")
        
        print(f"Total: {len(samples)}, Train: {len(train_samples)}, Test: {len(test_samples)}")
        print("ASAP2 dataset processing completed!")

def main():
    np.random.seed(42)
    test_size = float(os.environ.get('TEST_SIZE', 0.3))
    processor = ASAP2Processor()
    processor.process(test_size=test_size, random_state=42)

if __name__ == "__main__":
    main()

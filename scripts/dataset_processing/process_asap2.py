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
    def __init__(self, data_dir="../../data/raw/asap2", output_dir="../../data/processed/asap2"):
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
    
    def _ideal_as_float(self, ideal_value: str) -> float:
        try:
            return float(ideal_value)
        except Exception:
            try:
                return float(int(ideal_value))
            except Exception:
                return float('inf')
    
    def create_train_test_splits_by_set(self, samples):
        splits_by_set = {}
        for exercise_set in range(1, 7 + 1):
            set_samples = [s for s in samples if s['exercise_set'] == exercise_set]
            if len(set_samples) == 0:
                continue
            groups = {}
            for s in set_samples:
                groups.setdefault(s['ideal'], []).append(s)
            unique_ideals_sorted = sorted(groups.keys(), key=lambda v: self._ideal_as_float(v))
            train_samples = []
            test_samples = []
            for ideal_value in unique_ideals_sorted:
                samples_for_value = groups[ideal_value]
                if len(samples_for_value) > 0:
                    train_samples.append(samples_for_value[0])
                    test_samples.extend(samples_for_value[1:])
            train_samples_sorted = sorted(train_samples, key=lambda s: self._ideal_as_float(s['ideal']))
            test_samples_sorted = sorted(test_samples, key=lambda s: self._ideal_as_float(s['ideal']))
            splits_by_set[exercise_set] = {
                'train': train_samples_sorted,
                'test': test_samples_sorted,
                'total': len(set_samples)
            }
            print(f"Exercise Set {exercise_set}: Train={len(train_samples_sorted)}, Test={len(test_samples_sorted)}, Total={len(set_samples)}")
        return splits_by_set
    
    def save_jsonl_by_set(self, splits_by_set):
        for exercise_set, splits in splits_by_set.items():
            set_output_dir = self.output_dir / f"exercise_set_{exercise_set}"
            set_output_dir.mkdir(parents=True, exist_ok=True)
            train_file = set_output_dir / "train.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in splits['train']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            test_file = set_output_dir / "test.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for sample in splits['test']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Saved Exercise Set {exercise_set} to {set_output_dir}")
    
    def process(self):
        print("Starting ASAP2 dataset processing...")
        self.load_dataset()
        self.clean_data()
        samples = self.create_samples()
        splits_by_set = self.create_train_test_splits_by_set(samples)
        self.save_jsonl_by_set(splits_by_set)
        total_samples = sum(s['total'] for s in splits_by_set.values())
        total_train = sum(len(s['train']) for s in splits_by_set.values())
        total_test = sum(len(s['test']) for s in splits_by_set.values())
        print(f"\nSummary:")
        print(f"Total samples: {total_samples}")
        print(f"Total training samples: {total_train}")
        print(f"Total testing samples: {total_test}")
        print("ASAP2 dataset processing completed!")


def main():
    np.random.seed(42)
    processor = ASAP2Processor()
    processor.process()

if __name__ == "__main__":
    main()

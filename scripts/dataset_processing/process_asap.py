#!/usr/bin/env python3
"""
ASAP Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

class ASAPProcessor:
    def __init__(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'asap'
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'asap'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exercise_sets = {}
        self._load_exercise_set_metadata()
    
    def _load_exercise_set_metadata(self):
        for i in range(1, 9):
            set_dir = self.data_dir / f"exercise_set_{i}"
            if set_dir.exists():
                metadata = self._extract_set_metadata(set_dir, i)
                self.exercise_sets[i] = metadata
    
    def _extract_set_metadata(self, set_dir, set_num):
        metadata = {
            'set_id': set_num,
            'question': '',
            'rubric': '',
            'academic_level': '',
            'rubric_range': '',
            'essay_type': '',
            'scoring_type': '',
            'num_metrics': 0,
            'complementary_texts': ''
        }
        
        question_file = set_dir / "question.txt"
        if question_file.exists():
            with open(question_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if '## Question' in content:
                    match = re.search(r'## Question\s*\n\s*(.*?)(?=\n##|\Z)', content, re.DOTALL)
                    if match:
                        metadata['question'] = match.group(1).strip()
                    else:
                        metadata['question'] = content
                else:
                    metadata['question'] = content
        
        rubric_file = set_dir / "rubric.txt"
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                metadata['rubric'] = f.read().strip()
        
        # Load exercise texts if available
        comp_texts_file = set_dir / "exercise_texts.txt"
        if comp_texts_file.exists():
            with open(comp_texts_file, 'r', encoding='utf-8') as f:
                metadata['complementary_texts'] = f.read().strip()
        
        char_file = set_dir / "characteristics.txt"
        if char_file.exists():
            with open(char_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                grade_match = re.search(r'Grade Level:\s*(\d+)', content)
                if grade_match:
                    metadata['academic_level'] = grade_match.group(1)
                
                rubric_ranges = []
                rubric_dict = {}
                
                for line in content.split('\n'):
                    if 'Rubric Range:' in line:
                        range_match = re.search(r'Rubric Range:\s*(.*?)(?=\n|\Z)', line)
                        if range_match and range_match.group(1).strip():
                            rubric_ranges.append(range_match.group(1).strip())
                    elif line.strip().startswith('-') and ':' in line:
                        range_match = re.search(r'-\s*([^:]+):\s*(\d+-\d+)', line)
                        if range_match:
                            domain_name = range_match.group(1).strip()
                            range_value = range_match.group(2)
                            rubric_ranges.append(f"{domain_name}: {range_value}")
                
                # Build dictionary for multi-metric sets after collecting all ranges
                if len(rubric_ranges) > 1:
                    for range_item in rubric_ranges:
                        if ':' in range_item:
                            domain_name = range_item.split(':')[0].strip()
                            range_value = range_item.split(':')[1].strip()
                            
                            # Convert domain name to the format used in ideal scores
                            if 'ideas' in domain_name.lower():
                                key = 'ideal_ideas_score'
                            elif 'organization' in domain_name.lower():
                                key = 'ideal_organization_score'
                            elif 'style' in domain_name.lower():
                                key = 'ideal_style_score'
                            elif 'conventions' in domain_name.lower():
                                key = 'ideal_conventions_score'
                            elif 'voice' in domain_name.lower():
                                key = 'ideal_voice_score'
                            elif 'word choice' in domain_name.lower():
                                key = 'ideal_word_choice_score'
                            elif 'sentence fluency' in domain_name.lower():
                                key = 'ideal_sentence_fluency_score'
                            elif 'writing applications' in domain_name.lower():
                                key = 'ideal_writing_applications'
                            elif 'language conventions' in domain_name.lower():
                                key = 'ideal_language_conventions'
                            else:
                                key = f"ideal_{domain_name.lower().replace(' ', '_')}"
                            
                            rubric_dict[key] = range_value
                
                if rubric_ranges:
                    metadata['num_metrics'] = len(rubric_ranges)
                    
                    if len(rubric_ranges) > 1:
                        metadata['rubric_range'] = rubric_dict
                    else:
                        metadata['rubric_range'] = {"ideal": rubric_ranges[0]}
                
                type_match = re.search(r'Essay Type:\s*(.*?)(?=\n|\Z)', content)
                if type_match:
                    metadata['essay_type'] = type_match.group(1).strip()
                
                if set_num == 8 or any('trait' in line.lower() or 'ideas' in line.lower() or 'organization' in line.lower() for line in content.split('\n')):
                    metadata['scoring_type'] = 'trait'
                else:
                    metadata['scoring_type'] = 'domain'
        
        return metadata
    
    def load_dataset(self):
        data_file = None
        for ext in ['.xlsx', '.xls', '.tsv']:
            potential_file = self.data_dir / f"asap_student_responses_and_evaluations{ext}"
            if potential_file.exists():
                data_file = potential_file
                break
        
        if not data_file:
            raise FileNotFoundError("Could not find ASAP dataset file")
        
        if data_file.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(data_file)
        else:
            self.data = pd.read_csv(data_file, sep='\t')
        
        return self.data
    
    def clean_data(self):
        self.data['domain1_score'] = pd.to_numeric(self.data['domain1_score'], errors='coerce')
        self.data = self.data.dropna(subset=['domain1_score'])
        set1_mask = self.data['essay_set'] == 1
        if set1_mask.any():
            self.data.loc[set1_mask, 'domain1_score'] = self.data.loc[set1_mask, 'domain1_score'] // 2
        return self.data
    
    def create_samples(self):
        """Convert dataset rows to standardized sample format for training/evaluation"""
        samples = []
        
        for _, row in self.data.iterrows():
            essay_set = row['essay_set']
            if essay_set not in self.exercise_sets:
                continue
            
            metadata = self.exercise_sets[essay_set]
            
            question_text = metadata['question']
            if metadata.get('complementary_texts'):
                question_text = f"{metadata['complementary_texts'].strip()}\n\n{question_text}"
            
            sample = {
                "input": [{
                    "role": "user",
                    "content": "Question: {question}\n\nStudent Answer: {student_answer}\n\nRubric: {rubric}\n\nEvaluate this response."
                }],
                "question": question_text,
                "student_answer": row['essay'],
                "rubric": metadata['rubric'],
                "academic_level": metadata['academic_level'],
                "rubric_range": metadata['rubric_range'],
                "essay_type": metadata['essay_type'],
                "essay_set": essay_set,
                "num_metrics": metadata['num_metrics']
            }
            
            if metadata['scoring_type'] == 'domain':
                domain_scores = []
                domain_names = []
                
                if metadata['rubric_range']:
                    if isinstance(metadata['rubric_range'], dict):
                        for key in metadata['rubric_range'].keys():
                            domain_name = key.replace('ideal_', '').replace('_score', '')
                            domain_names.append(domain_name)
                    else:
                        for range_item in metadata['rubric_range'].split(' | '):
                            if ':' in range_item:
                                domain_name = range_item.split(':')[0].strip()
                                if domain_name.startswith('- '):
                                    domain_name = domain_name[2:]
                                domain_name = domain_name.lower().replace(' ', '_')
                                domain_names.append(domain_name)
                
                if 'domain1_score' in row and pd.notna(row['domain1_score']):
                    domain1_score = int(row['domain1_score'])
                    domain_scores.append(domain1_score)
                    if metadata['num_metrics'] > 1:
                        if len(domain_names) >= 1:
                            sample[f"ideal_{domain_names[0]}"] = str(domain1_score)
                        else:
                            sample["ideal_domain1"] = str(domain1_score)
                
                if 'domain2_score' in row and pd.notna(row['domain2_score']):
                    domain2_score = int(row['domain2_score'])
                    domain_scores.append(domain2_score)
                    if metadata['num_metrics'] > 1:
                        if len(domain_names) >= 2:
                            sample[f"ideal_{domain_names[1]}"] = str(domain2_score)
                        else:
                            sample["ideal_domain2"] = str(domain2_score)
                
                if len(domain_scores) == 1:
                    sample["ideal"] = str(domain_scores[0])
                elif len(domain_scores) > 1:
                    sample["ideal"] = str(sum(domain_scores))
            
            elif metadata['scoring_type'] == 'trait':
                max_traits = 6 if essay_set == 8 else 4
                trait_scores = []
                for trait_num in range(1, max_traits + 1):
                    rater1_col = f'rater1_trait{trait_num}'
                    rater2_col = f'rater2_trait{trait_num}'
                    if rater1_col in row and rater2_col in row:
                        rater1_score = pd.to_numeric(row[rater1_col], errors='coerce')
                        rater2_score = pd.to_numeric(row[rater2_col], errors='coerce')
                        if pd.notna(rater1_score) and pd.notna(rater2_score):
                            avg_score = round((rater1_score + rater2_score) / 2)
                            trait_scores.append(avg_score)
                            if isinstance(metadata['rubric_range'], dict):
                                trait_names = []
                                for key in metadata['rubric_range'].keys():
                                    trait_name = key.replace('ideal_', '').replace('_score', '')
                                    trait_names.append(trait_name)
                            else:
                                trait_names = ['ideas', 'organization', 'style', 'conventions', 'voice', 'word_choice']
                            if trait_num <= len(trait_names):
                                sample[f"ideal_{trait_names[trait_num-1]}_score"] = str(int(avg_score))
                if trait_scores:
                    sample["ideal"] = str(sum(trait_scores))
            
            if "ideal" not in sample and 'domain1_score' in row and pd.notna(row['domain1_score']):
                sample["ideal"] = str(int(row['domain1_score']))
            
            samples.append(sample)
        
        return samples
    
    
    def create_unified_dataset(self, samples):
        """Create a unified dataset in the same format as Mohler dataset"""
        unified_data = []
        
        for sample in samples:
            # Extract grade from ideal field (final sum for multi-rubric exercises)
            try:
                grade = float(sample.get('ideal', 0))
            except (ValueError, TypeError):
                grade = 0.0
            
            # Determine min and max grades from rubric_range
            min_grade = 1.0
            max_grade = 6.0  # Default fallback
            
            if 'rubric_range' in sample and sample['rubric_range']:
                if isinstance(sample['rubric_range'], dict):
                    # For multi-metric sets, calculate total range
                    if len(sample['rubric_range']) > 1:
                        total_min = 0
                        total_max = 0
                        for key, value in sample['rubric_range'].items():
                            if isinstance(value, str) and '-' in value:
                                try:
                                    min_val, max_val = map(float, value.split('-'))
                                    total_min += min_val
                                    total_max += max_val
                                except ValueError:
                                    pass
                        if total_max > 0:
                            min_grade = total_min
                            max_grade = total_max
                    else:
                        # Single metric
                        for key, value in sample['rubric_range'].items():
                            if isinstance(value, str) and '-' in value:
                                try:
                                    min_val, max_val = map(float, value.split('-'))
                                    min_grade = min_val
                                    max_grade = max_val
                                except ValueError:
                                    pass
                            break
            
            unified_sample = {
                'dataset': 'asap',
                'exercise_set': sample.get('essay_set', 1),
                'question': sample.get('question', ''),
                'answer': sample.get('student_answer', ''),
                'grade': grade,
                'min_grade': min_grade,
                'max_grade': max_grade,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'language': 'english',
                'rubric': sample.get('rubric', ''),
                'desired_answer': np.nan,  # NaN as requested
                'metadata': np.nan  # NaN as requested
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def save_unified_dataset(self, unified_df):
        """Save the unified dataset as both CSV and parquet files"""
        # Create output directory (use the main asap output directory)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_dir / 'asap_processed.csv'
        unified_df.to_csv(csv_file, index=False)
        print(f"Saved unified ASAP dataset to {csv_file}")
        
        # Save as parquet
        parquet_file = output_dir / 'asap_processed.parquet'
        unified_df.to_parquet(parquet_file, index=False)
        print(f"Saved unified ASAP dataset to {parquet_file}")
        
        # Print statistics
        print(f"Unified dataset contains {len(unified_df)} rows")
        print(f"Exercise sets: {sorted(unified_df['exercise_set'].unique())}")
        print(f"Grade distribution:")
        print(unified_df['grade'].value_counts().sort_index())
    
    def process(self):
        print("Starting ASAP dataset processing...")
        
        self.load_dataset()
        self.clean_data()
        
        samples = self.create_samples()
        
        # Create unified dataset in the same format as Mohler
        print("Creating unified dataset...")
        unified_df = self.create_unified_dataset(samples)
        self.save_unified_dataset(unified_df)
        
        print(f"\nSummary:")
        print(f"Total samples: {len(samples)}")
        print("ASAP dataset processing completed!")


def main():
    np.random.seed(42)
    processor = ASAPProcessor()
    processor.process()

if __name__ == "__main__":
    main()

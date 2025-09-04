#!/usr/bin/env python3
"""
ASAP Dataset Processing Script
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

class ASAPProcessor:
    def __init__(self, data_dir="../../data/raw/asap", output_dir="../../data/processed/asap"):
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
                        metadata['rubric_range'] = ' | '.join(rubric_ranges)
                
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
    
    def _ideal_as_float(self, ideal_value: str) -> float:
        try:
            return float(ideal_value)
        except Exception:
            try:
                return float(int(ideal_value))
            except Exception:
                return float('inf')
    
    def create_train_test_splits_by_set(self, samples):
        """Create per-set train/test splits with special sampling for sets 2, 7 and 8.
        - Set 2: one train example for every second unique ideal value (approx half)
        - Sets 1,3-6: one train example per unique ideal value
        - Set 7: one train example for every second unique ideal value (approx half)
        - Set 8: one train example for every fourth unique ideal value (approx quarter)
        The rest go to test. Train and test are sorted by ideal ascending.
        """
        splits_by_set = {}
        
        for essay_set in range(1, 9):
            set_samples = [s for s in samples if s['essay_set'] == essay_set]
            if len(set_samples) == 0:
                continue
            
            groups = {}
            for s in set_samples:
                groups.setdefault(s['ideal'], []).append(s)
            
            unique_ideals_sorted = sorted(groups.keys(), key=lambda v: self._ideal_as_float(v))
            
            # Determine sampling stride
            if essay_set in (2, 7):
                stride = 2
            elif essay_set == 8:
                stride = 4
            else:
                stride = 1
            
            chosen_ideals = set(unique_ideals_sorted[::stride])
            
            train_samples = []
            test_samples = []
            
            for ideal_value in unique_ideals_sorted:
                samples_for_value = groups[ideal_value]
                if ideal_value in chosen_ideals and len(samples_for_value) > 0:
                    train_samples.append(samples_for_value[0])
                    test_samples.extend(samples_for_value[1:])
                else:
                    test_samples.extend(samples_for_value)
            
            train_samples_sorted = sorted(train_samples, key=lambda s: self._ideal_as_float(s['ideal']))
            test_samples_sorted = sorted(test_samples, key=lambda s: self._ideal_as_float(s['ideal']))
            
            splits_by_set[essay_set] = {
                'train': train_samples_sorted,
                'test': test_samples_sorted,
                'total': len(set_samples)
            }
            
            print(f"Exercise Set {essay_set}: Train={len(train_samples_sorted)}, Test={len(test_samples_sorted)}, Total={len(set_samples)}")
        
        return splits_by_set
    
    def save_jsonl_by_set(self, splits_by_set):
        for essay_set, splits in splits_by_set.items():
            set_output_dir = self.output_dir / f"exercise_set_{essay_set}"
            set_output_dir.mkdir(parents=True, exist_ok=True)
            
            train_file = set_output_dir / "train.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in splits['train']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            test_file = set_output_dir / "test.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for sample in splits['test']:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"Saved Exercise Set {essay_set} to {set_output_dir}")
    
    def process(self):
        print("Starting ASAP dataset processing...")
        
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
        print("ASAP dataset processing completed!")


def main():
    np.random.seed(42)
    processor = ASAPProcessor()
    processor.process()

if __name__ == "__main__":
    main()

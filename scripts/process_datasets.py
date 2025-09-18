#!/usr/bin/env python3
"""
Unified Dataset Processing Script for MentorEval

This script processes all datasets in the MentorEval benchmark and creates standardized
unified datasets with the language column included.

Usage:
    python scripts/process_datasets.py [--datasets asap,asap2,mohler,ellipse,ptasag2018] [--test-size 0.3]
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt mapping for ELLIPSE dataset
ELLIPSE_PROMPT_MAPPING = {
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


class DatasetProcessor:
    """Base class for dataset processing"""
    
    def __init__(self, data_dir=None, output_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'data' / 'processed'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sequential_exercise_set_mapping(self, data, id_column, start_from=1):
        """
        Create a mapping from original IDs to sequential exercise set numbers.
        
        Args:
            data (pd.DataFrame): The dataset
            id_column (str): Column name containing the original IDs
            start_from (int): Starting number for sequential mapping (default: 1)
        
        Returns:
            dict: Mapping from original IDs to sequential exercise set numbers
        """
        unique_ids = data[id_column].dropna().unique()
        return {orig_id: idx + start_from for idx, orig_id in enumerate(sorted(unique_ids))}
    
    def save_unified_dataset(self, unified_df, dataset_name):
        """Save the unified dataset as both CSV and parquet files"""
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = dataset_output_dir / f'{dataset_name}_processed.csv'
        unified_df.to_csv(csv_file, index=False)
        logger.info(f"Saved unified {dataset_name.upper()} dataset to {csv_file}")
        
        # Save as parquet
        parquet_file = dataset_output_dir / f'{dataset_name}_processed.parquet'
        unified_df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved unified {dataset_name.upper()} dataset to {parquet_file}")
        
        # Print statistics
        logger.info(f"Unified dataset contains {len(unified_df)} rows")
        logger.info(f"Exercise sets: {sorted(unified_df['exercise_set'].unique())}")
        logger.info(f"Grade distribution:")
        logger.info(unified_df['grade'].value_counts().sort_index())


class ASAPProcessor(DatasetProcessor):
    """ASAP Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'asap'
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
                
                if len(rubric_ranges) > 1:
                    for range_item in rubric_ranges:
                        if ':' in range_item:
                            domain_name = range_item.split(':')[0].strip()
                            range_value = range_item.split(':')[1].strip()
                            
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
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        # Create sequential exercise set mapping
        essay_set_to_exercise_set = self.create_sequential_exercise_set_mapping(self.data, 'essay_set')
        
        for _, row in self.data.iterrows():
            essay_set = row['essay_set']
            if essay_set not in self.exercise_sets:
                continue
            
            metadata = self.exercise_sets[essay_set]
            exercise_set = essay_set_to_exercise_set[essay_set]
            
            question_text = metadata['question']
            if metadata.get('complementary_texts'):
                question_text = f"{metadata['complementary_texts'].strip()}\n\n{question_text}"
            
            # Extract grade from domain scores
            grade = 0.0
            if metadata['scoring_type'] == 'domain':
                domain_scores = []
                if 'domain1_score' in row and pd.notna(row['domain1_score']):
                    domain_scores.append(int(row['domain1_score']))
                if 'domain2_score' in row and pd.notna(row['domain2_score']):
                    domain_scores.append(int(row['domain2_score']))
                grade = sum(domain_scores) if domain_scores else 0.0
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
                grade = sum(trait_scores) if trait_scores else 0.0
            
            # Determine min and max grades from rubric_range
            min_grade = 1.0
            max_grade = 6.0
            
            if 'rubric_range' in metadata and metadata['rubric_range']:
                if isinstance(metadata['rubric_range'], dict):
                    if len(metadata['rubric_range']) > 1:
                        total_min = 0
                        total_max = 0
                        for key, value in metadata['rubric_range'].items():
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
                        for key, value in metadata['rubric_range'].items():
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
                'exercise_set': exercise_set,
                'question': question_text,
                'answer': row['essay'],
                'grade': grade,
                'min_grade': min_grade,
                'max_grade': max_grade,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'language': 'english',
                'rubric': metadata['rubric'],
                'desired_answer': np.nan,
                'metadata': np.nan
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting ASAP dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'asap')
        logger.info("ASAP dataset processing completed!")


class ASAP2Processor(DatasetProcessor):
    """ASAP2 Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'asap2'
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
            'academic_level': '10',
            'rubric_range': {"ideal": "1-6"},
            'essay_type': 'argumentative',
            'scoring_type': 'holistic',
            'num_metrics': 1
        }
        
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
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        # Create sequential exercise set mapping
        original_to_sequential = self.create_sequential_exercise_set_mapping(self.data, 'exercise_set')
        
        for _, row in self.data.iterrows():
            original_exercise_set = row['exercise_set']
            if original_exercise_set not in self.exercise_sets:
                continue
            
            metadata = self.exercise_sets[original_exercise_set]
            exercise_set = original_to_sequential[original_exercise_set]
            
            question_text = metadata['question']
            if metadata.get('complementary_texts'):
                question_text = f"{metadata['complementary_texts'].strip()}\n\n{question_text}"
            
            # Create metadata JSON from demographic fields
            metadata_dict = {}
            if pd.notna(row.get('economically_disadvantaged')):
                metadata_dict['economically_disadvantaged'] = row['economically_disadvantaged']
            if pd.notna(row.get('student_disability_status')):
                metadata_dict['student_disability_status'] = row['student_disability_status']
            if pd.notna(row.get('ell_status')):
                metadata_dict['ell_status'] = row['ell_status']
            if pd.notna(row.get('race_ethnicity')):
                metadata_dict['race_ethnicity'] = row['race_ethnicity']
            if pd.notna(row.get('gender')):
                metadata_dict['gender'] = row['gender']
            if pd.notna(row.get('exercise')):
                metadata_dict['exercise'] = row['exercise']
            
            metadata_json = json.dumps(metadata_dict) if metadata_dict else np.nan
            
            unified_sample = {
                'dataset': 'asap2',
                'exercise_set': exercise_set,
                'question': question_text,
                'answer': row['student_text'],
                'grade': float(row['score']),
                'min_grade': 1.0,
                'max_grade': 6.0,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'language': 'english',
                'rubric': metadata['rubric'],
                'desired_answer': np.nan,
                'metadata': metadata_json
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting ASAP2 dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'asap2')
        logger.info("ASAP2 dataset processing completed!")


class MohlerProcessor(DatasetProcessor):
    """Mohler Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'mohler'
    
    def load_dataset(self):
        data_file = self.data_dir / "mohler_dataset_edited.csv"
        if not data_file.exists():
            raise FileNotFoundError("Could not find Mohler dataset file")
        self.data = pd.read_csv(data_file)
        return self.data
    
    def clean_data(self):
        # Filter rows where score_me == score_other
        self.data = self.data[self.data['score_me'] == self.data['score_other']].copy()
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        # Create sequential exercise set mapping
        id_to_exercise_set = self.create_sequential_exercise_set_mapping(self.data, 'id')
        
        for _, row in self.data.iterrows():
            exercise_set = id_to_exercise_set[row['id']]
            
            # Ensure score is an integer
            score = float(row['score_me'])
            if not score.is_integer():
                logger.warning(f"Non-integer score found: {score} for row {row.name}, rounding to nearest integer")
                score = round(score)
            # Convert to actual integer type
            score = int(score)
            
            unified_sample = {
                'dataset': 'mohler',
                'exercise_set': exercise_set,
                'question': row['question'],
                'answer': row['student_answer'],
                'grade': score,
                'min_grade': 1.0,
                'max_grade': 5.0,
                'subject': 'computer_science',
                'exercise_type': 'short_answer',
                'isced_level': 6,
                'language': 'english',
                'rubric': np.nan,
                'desired_answer': row['desired_answer'],
                'metadata': np.nan
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting Mohler dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'mohler')
        logger.info("Mohler dataset processing completed!")


class EllipseProcessor(DatasetProcessor):
    """ELLIPSE Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'ellipse'
        self.prompt_mapping = ELLIPSE_PROMPT_MAPPING
    
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
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        self.data = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Loaded ELLIPSE dataset with {len(self.data)} rows")
        return self.data
    
    def clean_data(self):
        """Clean the data"""
        self.data = self.data.dropna(subset=['full_text', 'prompt'])
        
        scoring_columns = ['Cohesion', 'Syntax', 'Vocabulary', 'Phraseology', 'Grammar', 'Conventions']
        missing_columns = [col for col in scoring_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required scoring columns: {missing_columns}")
        
        for col in scoring_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data = self.data.dropna(subset=scoring_columns)
        
        # Calculate total grade as average of the 6 scoring dimensions, rounded to nearest integer
        self.data['total_grade'] = np.round(
            (self.data['Cohesion'] + 
             self.data['Syntax'] + 
             self.data['Vocabulary'] + 
             self.data['Phraseology'] + 
             self.data['Grammar'] + 
             self.data['Conventions']) / 6
        ).astype(int)
        
        self.data = self.data[(self.data['total_grade'] >= 1) & (self.data['total_grade'] <= 5)]
        logger.info(f"After cleaning: {len(self.data)} rows")
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        rubric_text = self.load_rubric()
        
        # Create sequential exercise set mapping
        prompt_to_exercise_set = self.create_sequential_exercise_set_mapping(self.data, 'prompt')
        
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
            
            metadata_json = json.dumps(metadata_dict) if metadata_dict else np.nan
            
            # Get question from prompt mapping
            question = self.prompt_mapping.get(row['prompt'], row['prompt'])
            
            unified_sample = {
                'dataset': 'ellipse',
                'exercise_set': prompt_to_exercise_set[row['prompt']],
                'question': question,
                'answer': row['full_text'],
                'grade': float(row['total_grade']),
                'min_grade': 1.0,
                'max_grade': 5.0,
                'subject': 'english',
                'exercise_type': 'essay_writing',
                'isced_level': 3,
                'language': 'english',
                'rubric': rubric_text,
                'desired_answer': np.nan,
                'metadata': metadata_json
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting ELLIPSE dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'ellipse')
        logger.info("ELLIPSE dataset processing completed!")


class PTASAG2018Processor(DatasetProcessor):
    """PTASAG2018 Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'ptasag2018'


class ARASAGProcessor(DatasetProcessor):
    """ARASAG Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'arasag'
    
    def load_dataset(self):
        """Load the ARASAG dataset"""
        data_file = self.data_dir / "AR-ASAG-Dataset.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find ARASAG dataset file: {data_file}")
        
        self.data = pd.read_csv(data_file)
        logger.info(f"Loaded ARASAG dataset with {len(self.data)} rows")
        return self.data
    
    def clean_data(self):
        """Clean the data"""
        self.data = self.data.dropna(subset=['Question_Arabic', 'Answer_Arabic', 'Average_Mark'])
        
        self.data['Average_Mark'] = pd.to_numeric(self.data['Average_Mark'], errors='coerce')
        self.data = self.data.dropna(subset=['Average_Mark'])
        
        # Ensure grades are within valid range (0-5)
        self.data = self.data[(self.data['Average_Mark'] >= 0) & (self.data['Average_Mark'] <= 5)]
        logger.info(f"After cleaning: {len(self.data)} rows")
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        # Create sequential exercise set mapping
        question_to_exercise_set = self.create_sequential_exercise_set_mapping(self.data, 'Question_Arabic')
        
        for _, row in self.data.iterrows():
            # Create metadata JSON from question type
            metadata_dict = {
                'question_type': row['Question_Type']
            }
            metadata_json = json.dumps(metadata_dict, ensure_ascii=False)
            
            # Ensure score is an integer (similar to Mohler processing)
            score = float(row['Average_Mark'])
            if not score.is_integer():
                logger.warning(f"Non-integer score found: {score} for row {row.name}, rounding to nearest integer")
                score = round(score)
            # Convert to actual integer type
            score = int(score)
            
            unified_sample = {
                'dataset': 'arasag',
                'exercise_set': question_to_exercise_set[row['Question_Arabic']],
                'question': row['Question_Arabic'],
                'answer': row['Answer_Arabic'],
                'grade': score,
                'min_grade': 0.0,
                'max_grade': 5.0,
                'subject': 'cybercrimes',
                'exercise_type': 'short_answer',
                'isced_level': 6,
                'language': 'arabic',
                'rubric': np.nan,
                'desired_answer': row['Model_Arabic'],
                'metadata': metadata_json
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting ARASAG dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'arasag')
        logger.info("ARASAG dataset processing completed!")


class PTASAG2018Processor(DatasetProcessor):
    """PTASAG2018 Dataset Processor"""
    
    def __init__(self, data_dir=None, output_dir=None):
        super().__init__(data_dir, output_dir)
        self.data_dir = self.data_dir / 'ptasag2018'
    
    def load_dataset(self):
        """Load the main CSV file and questions file"""
        main_file = self.data_dir / "student_answers_and_grades_v2.csv"
        questions_file = self.data_dir / "questions.csv"
        
        if not main_file.exists():
            raise FileNotFoundError(f"Could not find main file: {main_file}")
        if not questions_file.exists():
            raise FileNotFoundError(f"Could not find questions file: {questions_file}")
        
        main_df = pd.read_csv(main_file)
        questions_df = pd.read_csv(questions_file)
        
        logger.info(f"Loaded main dataset with {len(main_df)} rows")
        logger.info(f"Loaded questions dataset with {len(questions_df)} rows")
        
        # Merge with questions to get question text
        self.data = main_df.merge(questions_df, on='question_id', how='left')
        logger.info(f"Final dataset with {len(self.data)} rows after merging with questions")
        return self.data
    
    def clean_data(self):
        """Clean the data"""
        self.data = self.data.dropna(subset=['question_id', 'answer_text', 'grade', 'question_text'])
        
        self.data['grade'] = pd.to_numeric(self.data['grade'], errors='coerce')
        self.data = self.data.dropna(subset=['grade'])
        
        self.data = self.data[(self.data['grade'] >= 0) & (self.data['grade'] <= 3)]
        logger.info(f"After cleaning: {len(self.data)} rows")
        return self.data
    
    def create_unified_dataset(self):
        """Create a unified dataset in the same format as other datasets"""
        unified_data = []
        
        # Create sequential exercise set mapping
        question_id_to_exercise_set = self.create_sequential_exercise_set_mapping(self.data, 'question_id')
        
        for _, row in self.data.iterrows():
            unified_sample = {
                'dataset': 'ptasag2018',
                'exercise_set': question_id_to_exercise_set[row['question_id']],
                'question': row['question_text'],
                'answer': row['answer_text'],
                'grade': float(row['grade']),
                'min_grade': 0.0,
                'max_grade': 3.0,
                'subject': 'biology',
                'exercise_type': 'short_answer',
                'isced_level': 3,
                'language': 'portuguese',
                'rubric': np.nan,
                'desired_answer': np.nan,
                'metadata': np.nan
            }
            
            unified_data.append(unified_sample)
        
        return pd.DataFrame(unified_data)
    
    def process(self):
        logger.info("Starting PTASAG2018 dataset processing...")
        self.load_dataset()
        self.clean_data()
        unified_df = self.create_unified_dataset()
        self.save_unified_dataset(unified_df, 'ptasag2018')
        logger.info("PTASAG2018 dataset processing completed!")


def combine_all_datasets(successful_datasets):
    """Combine all processed datasets into a single parquet file"""
    print(f"\n{'='*60}")
    print("COMBINING ALL DATASETS")
    print(f"{'='*60}")
    
    combined_data = []
    total_samples = 0
    
    for dataset in successful_datasets:
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
    
    print(f"\n🔄 Combining {len(combined_data)} datasets...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Ensure grade column is integer type
    combined_df['grade'] = combined_df['grade'].astype('int64')
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    combined_file = output_dir / "mentoreval.parquet"
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
    
    datasets_info_file = Path("data/datasets_info.csv")
    if not datasets_info_file.exists():
        print(f"❌ datasets_info.csv not found: {datasets_info_file}")
        return False
    
    df_info = pd.read_csv(datasets_info_file)
    print(f"📁 Loaded datasets_info.csv with {len(df_info)} datasets")
    
    for _, row in df_info.iterrows():
        dataset_id = row['id']
        
        if dataset_id not in successful_datasets:
            print(f"⚠️  Skipping {dataset_id} (not in successful datasets)")
            continue
        
        parquet_file = Path("data/processed") / dataset_id / f"{dataset_id}_processed.parquet"
        
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                
                num_exercises = df['exercise_set'].nunique()
                num_student_answers = len(df)
                
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
    parser = argparse.ArgumentParser(description="Process all datasets for MentorEval benchmark")
    parser.add_argument("--datasets", 
                       default="asap,asap2,mohler,ellipse,ptasag2018,arasag",
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
    
    print("🚀 MENTOREVAL DATASET PROCESSING")
    print("="*60)
    print(f"📋 Datasets to process: {', '.join(datasets_to_process)}")
    print(f"📊 Test size: {args.test_size}")
    print(f"🔍 Check only: {args.check_only}")
    
    # Define processor classes
    processors = {
        'asap': ASAPProcessor,
        'asap2': ASAP2Processor,
        'mohler': MohlerProcessor,
        'ellipse': EllipseProcessor,
        'ptasag2018': PTASAG2018Processor,
        'arasag': ARASAGProcessor,
    }
    
    if args.check_only:
        print("\n🔍 CHECKING EXISTING OUTPUT FILES")
        print("="*60)
        
        for dataset in datasets_to_process:
            if dataset in processors:
                print(f"\n📁 Checking {dataset.upper()}...")
                # Check if output files exist
                parquet_file = Path("data/processed") / dataset / f"{dataset}_processed.parquet"
                csv_file = Path("data/processed") / dataset / f"{dataset}_processed.csv"
                
                if parquet_file.exists() and csv_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        print(f"   ✅ Found: {parquet_file} ({len(df)} samples)")
                        print(f"   ✅ Found: {csv_file} ({len(df)} samples)")
                    except Exception as e:
                        print(f"   ❌ Error reading {parquet_file}: {e}")
                else:
                    print(f"   ❌ Missing files for {dataset}")
            else:
                print(f"⚠️  Unknown dataset: {dataset}")
        
        return
    
    # Process datasets
    successful_datasets = []
    failed_datasets = []
    
    for dataset in datasets_to_process:
        if dataset not in processors:
            print(f"⚠️  Unknown dataset: {dataset}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {dataset.upper()} DATASET")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            processor = processors[dataset]()
            processor.process()
            end_time = time.time()
            
            print(f"✅ {dataset.upper()} processing completed successfully!")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            successful_datasets.append(dataset)
            
        except Exception as e:
            print(f"❌ {dataset.upper()} processing failed: {e}")
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


if __name__ == "__main__":
    main()

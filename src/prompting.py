from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import re
from src.utils.paths import ASAP_DATASET_DIR, ASAP_RESPONSES_FILE

@dataclass
class PromptExamData:
    """Data class to hold all components of a prompt for exam evaluation."""
    question: str
    student_answer: str
    llm_prompt: str
    exam_characteristics: Optional[Dict[str, str]] = None
    rubric: Optional[str] = None
    exercise_texts: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert the PromptExamData object to a dictionary."""
        return asdict(self)

    def to_combined_string(self) -> str:
        """Combine all components into a single string following the standard order."""
        sections = []
        if self.exam_characteristics:
            sections.append(self.exam_characteristics)
        if self.exercise_texts:
            sections.append(self.exercise_texts)
        if self.question:
            sections.append(self.question)
        if self.rubric:
            sections.append(self.rubric)
        if self.llm_prompt:
            sections.append("## Default Prompt\n" + self.llm_prompt)
        if self.student_answer:
            sections.append(self.student_answer)
        return "\n\n".join(sections)

class PromptLoader(ABC):
    """Abstract base class for loading prompts from different datasets."""
    
    def __init__(
        self,
        include_rubric: bool = True,
        include_exercise_texts: bool = True,
        include_exam_characteristics: bool = True
    ):
        self.include_rubric = include_rubric
        self.include_exercise_texts = include_exercise_texts
        self.include_exam_characteristics = include_exam_characteristics
    
    @abstractmethod
    def get_prompts(self) -> Dict[int, str]:
        """Get all prompts combined into single strings.

        Returns:
            Dict[int, str]: Dictionary mapping question IDs to their combined prompt strings
        """
        pass
    
    def _load_file_content(self, file_path: Path) -> str:
        """Helper method to load file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

class ASAPPromptLoader(PromptLoader):
    """Concrete implementation for loading prompts from the ASAP-AES dataset."""
    
    def __init__(
        self,
        include_rubric: bool = True,
        include_exercise_texts: bool = True,
        include_exam_characteristics: bool = True
    ):
        """Initialize the ASAP prompt loader using internal path constants."""
        super().__init__(
            include_rubric=include_rubric,
            include_exercise_texts=include_exercise_texts,
            include_exam_characteristics=include_exam_characteristics
        )
        self.dataset_path = ASAP_DATASET_DIR
        self.responses_df = pd.read_excel(ASAP_RESPONSES_FILE)
        self.exercise_set_path = ASAP_DATASET_DIR  # Store base path for all sets
    
    def get_prompts(self) -> Dict[int, str]:
        """Get all prompts combined into single strings.

        Returns:
            Dict[int, str]: Dictionary mapping question IDs to their combined prompt strings
        """
        all_prompts = {}
        # Find all exercise set directories
        base_path = self.exercise_set_path
        for set_dir in sorted(base_path.glob('exercise_set_*')):
            set_id = int(str(set_dir.name).split('_')[-1])
            # Load files for this set
            def load_and_clean(filename):
                try:
                    with open(set_dir / filename, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Remove bold markdown
                        return re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                except FileNotFoundError:
                    return None
            
            # Load all components for this set
            question = load_and_clean('question.txt')
            prompt = load_and_clean('prompt.txt')
            rubric = load_and_clean('rubric.txt') if self.include_rubric else None
            characteristics = load_and_clean('characteristics.txt') if self.include_exam_characteristics else None
            complementary_texts = load_and_clean('complementary_exercise_texts.txt') if self.include_exercise_texts else None
            
            # Get student answers for this set
            set_responses = self.responses_df[self.responses_df['essay_set'] == set_id]
            print(f"Found {len(set_responses)} student answers for set {set_id}")
            
            # Create a PromptExamData object for each student answer and get combined string
            for idx, row in set_responses.iterrows():
                prompt_data = PromptExamData(
                    question=question,
                    student_answer=row['essay'],
                    llm_prompt=prompt,
                    rubric=rubric,
                    exam_characteristics=characteristics,
                    exercise_texts=complementary_texts
                )
                all_prompts[int(idx)] = prompt_data.to_combined_string()
        
        return all_prompts

    def get_prompts_as_dict(self) -> Dict[int, Dict]:
        """Get all prompts as dictionaries of their components.

        Returns:
            Dict[int, Dict]: Dictionary mapping question IDs to their prompt data as dictionaries
        """
        all_prompts = {}
        # Find all exercise set directories
        base_path = self.exercise_set_path
        for set_dir in sorted(base_path.glob('exercise_set_*')):
            set_id = int(str(set_dir.name).split('_')[-1])
            # Load files for this set
            def load_and_clean(filename):
                try:
                    with open(set_dir / filename, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Remove bold markdown
                        return re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                except FileNotFoundError:
                    return None
            
            # Load all components for this set
            question = load_and_clean('question.txt')
            prompt = load_and_clean('prompt.txt')
            rubric = load_and_clean('rubric.txt') if self.include_rubric else None
            characteristics = load_and_clean('characteristics.txt') if self.include_exam_characteristics else None
            complementary_texts = load_and_clean('complementary_exercise_texts.txt') if self.include_exercise_texts else None
            
            # Get student answers for this set
            set_responses = self.responses_df[self.responses_df['essay_set'] == set_id]
            print(f"Found {len(set_responses)} student answers for set {set_id}")
            
            # Create a PromptExamData object for each student answer and convert to dict
            for idx, row in set_responses.iterrows():
                prompt_data = PromptExamData(
                    question=question,
                    student_answer=row['essay'],
                    llm_prompt=prompt,
                    rubric=rubric,
                    exam_characteristics=characteristics,
                    exercise_texts=complementary_texts
                )
                all_prompts[int(idx)] = prompt_data.to_dict()
        
        return all_prompts

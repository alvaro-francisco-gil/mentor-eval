from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Iterator
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import re

@dataclass
class PromptExamData:
    """Data class to hold all components of a prompt for exam evaluation."""
    question: str
    student_answer: str
    llm_prompt: str
    exam_characteristics: Optional[Dict[str, str]] = None
    rubric: Optional[str] = None
    exercise_texts: Optional[str] = None

class PromptLoader(ABC):
    """Abstract base class for loading prompts from different datasets."""
    
    def __init__(
        self,
        include_rubric: bool = False,
        include_exercise_texts: bool = False
    ):
        self.include_rubric = include_rubric
        self.include_exercise_texts = include_exercise_texts
    
    @abstractmethod
    def load_prompt_data(self, question_id: str) -> PromptExamData:
        """Load prompt data for a specific question ID."""
        pass
    
    @abstractmethod
    def get_available_questions(self) -> List[str]:
        """Return a list of available question IDs."""
        pass
    
    def get_prompt_iterator(self) -> Iterator[PromptExamData]:
        """Return an iterator over all available prompts."""
        for question_id in self.get_available_questions():
            yield self.load_prompt_data(question_id)
    
    def get_prompt_list(self) -> List[PromptExamData]:
        """Return a list of all available prompts."""
        return list(self.get_prompt_iterator())
    
    def _load_file_content(self, file_path: Path) -> str:
        """Helper method to load file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

class ASAPPromptLoader(PromptLoader):
    """Concrete implementation for loading prompts from the ASAP-AES dataset."""
    
    def __init__(
        self,
        dataset_path: Path,
        responses_file: Path,
    ):
        """Initialize the ASAP prompt loader.

        Args:
            dataset_path (Path): Path to the ASAP-AES dataset directory
            responses_file (Path): Path to the Excel file containing student responses
        """
        self.dataset_path = dataset_path
        self.responses_df = pd.read_excel(responses_file)
        self.exercise_set_path = dataset_path  # Store base path for all sets
    
    def get_available_questions(self) -> List[str]:
        """Return a list of available question IDs in the exercise set."""
        return [str(idx) for idx in self.responses_df.index]
    
    def load_prompt_data(self, question_id: str) -> PromptExamData:
        """Load prompt data for a specific question in the exercise set."""
        # Get the student answer and set ID from the Excel file
        row = self.responses_df.loc[int(question_id)]
        student_answer = row['essay']
        set_id = row['essay_set']
        
        # Load the base components from the appropriate exercise set directory
        set_path = self.dataset_path / f"exercise_set_{set_id}"
        
        # Load question and prompt
        question = self._load_file_content(set_path / "question.txt")
        prompt = self._load_file_content(set_path / "prompt.txt")
        
        # Load rubric if available
        try:
            rubric = self._load_file_content(set_path / "rubric.txt")
        except FileNotFoundError:
            rubric = None
        
        # Load characteristics if available
        try:
            characteristics = self._load_file_content(set_path / "characteristics.txt")
        except FileNotFoundError:
            characteristics = None
        
        # Load complementary texts if available
        try:
            complementary_texts = self._load_file_content(set_path / "complementary_exercise_texts.txt")
        except FileNotFoundError:
            complementary_texts = None
        
        return PromptExamData(
            question=question,
            prompt=prompt,
            student_answer=student_answer,
            rubric=rubric,
            characteristics=characteristics,
            complementary_texts=complementary_texts
        )
    
    def _parse_characteristics(self, content: str) -> Dict[str, str]:
        """Parse the characteristics file content into a dictionary."""
        characteristics = {}
        current_key = None
        current_value = []
        
        for line in content.split('\n'):
            if line.startswith('### '):
                if current_key and current_value:
                    characteristics[current_key] = '\n'.join(current_value).strip()
                current_key = line[4:].strip()
                current_value = []
            elif current_key:
                current_value.append(line)
        
        if current_key and current_value:
            characteristics[current_key] = '\n'.join(current_value).strip()
        
        return characteristics

    def get_prompts(self) -> dict:
        """Get all prompts for all exercise sets.

        Returns:
            dict: {exercise_set_id: [list of complete prompts]}
        """
        all_prompts = {}
        # Find all exercise set directories
        base_path = self.exercise_set_path.parent
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
            question = load_and_clean('question.txt')
            prompt = load_and_clean('prompt.txt')
            rubric = load_and_clean('rubric.txt')
            characteristics = load_and_clean('characteristics.txt')
            complementary_texts = load_and_clean('complementary_exercise_texts.txt')
            # Get student answers for this set
            student_answers = self.responses_df[self.responses_df['essay_set'] == set_id]['essay'].tolist()
            prompts = []
            for student_answer in student_answers:
                sections = []
                if characteristics:
                    sections.append(characteristics)
                if complementary_texts:
                    sections.append(complementary_texts)
                if question:
                    sections.append(question)
                if rubric:
                    sections.append(rubric)
                if prompt:
                    sections.append(prompt)
                sections.append(student_answer)
                prompts.append("\n\n".join(sections))
            all_prompts[set_id] = prompts
        return all_prompts

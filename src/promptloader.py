from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Iterator
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PromptExamData:
    """Data class to hold all components of a prompt for exam evaluation."""
    question: str
    student_answer: str
    llm_prompt: str
    exam_characteristics: Optional[Dict[str, str]] = None
    rubric: Optional[str] = None
    complementary_exercise_texts: Optional[str] = None

class PromptLoader(ABC):
    """Abstract base class for loading prompts from different datasets."""
    
    def __init__(
        self,
        include_rubric: bool = False,
        include_complementary_sources: bool = False
    ):
        self.include_rubric = include_rubric
        self.include_complementary_sources = include_complementary_sources
    
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
        exercise_set_id: str,
        include_rubric: bool = True,
        include_complementary_sources: bool = False
    ):
        super().__init__(include_rubric, include_complementary_sources)
        self.dataset_path = dataset_path
        self.exercise_set_id = exercise_set_id
        self.exercise_set_path = dataset_path / f"exercise_set_{exercise_set_id}"
    
    def get_available_questions(self) -> List[str]:
        """Return a list of available question IDs in the exercise set."""
        return [f.stem.replace('question_', '') 
                for f in self.exercise_set_path.glob("question_*.txt")]
    
    def load_prompt_data(self, question_id: str) -> PromptExamData:
        """Load prompt data for a specific question in the exercise set."""
        # Construct paths
        question_path = self.exercise_set_path / f"question_{question_id}.txt"
        prompt_path = self.exercise_set_path / f"prompt_{question_id}.txt"
        answer_path = self.exercise_set_path / f"student_answer_{question_id}.txt"
        
        # Load basic data
        data = PromptExamData(
            question=self._load_file_content(question_path),
            student_answer=self._load_file_content(answer_path),
            llm_prompt=self._load_file_content(prompt_path)
        )
        
        # Load rubric if requested
        if self.include_rubric:
            rubric_path = self.exercise_set_path / "rubric.txt"
            if rubric_path.exists():
                data.rubric = self._load_file_content(rubric_path)
        
        # Load characteristics
        characteristics_path = self.exercise_set_path / "characteristics.txt"
        if characteristics_path.exists():
            characteristics_content = self._load_file_content(characteristics_path)
            # Parse characteristics into a dictionary
            data.exam_characteristics = self._parse_characteristics(characteristics_content)
        
        return data
    
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

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass
from paths import ASAP_DATASET_DIR, ASAP2_DATASET_DIR

@dataclass
class ExerciseComponents:
    """Data class to hold exercise components that will be converted to markdown."""
    sections: List[str]

class MarkdownConverter(ABC):
    """Abstract base class for converting exercise components to markdown."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_components(self, source_dir: Path, component_id: str) -> ExerciseComponents:
        """Load exercise components from source directory for a specific component ID."""
        pass
    
    @abstractmethod
    def get_available_components(self, source_dir: Path) -> List[str]:
        """Get list of available component IDs in the source directory."""
        pass
    
    def convert_to_markdown(self, components: ExerciseComponents, folder_name: str) -> str:
        """Convert exercise components to markdown format."""
        sections = [f"# {folder_name}\n"] + components.sections
        # Filter out empty sections
        sections = [s for s in sections if s.strip()]
        return "\n\n".join(sections)
    
    def save_markdown(self, markdown_content: str, output_path: Path) -> None:
        """Save markdown content to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def process_all_components(self, source_dir: Path) -> None:
        """Process all components in the source directory and save as all_exercise_description.md."""
        # For this use case, we only expect one component (id '1')
        component_id = self.get_available_components(source_dir)[0]
        components = self.load_components(source_dir, component_id)
        folder_name = source_dir.name.replace('_', ' ').title()
        markdown_content = self.convert_to_markdown(components, folder_name)
        output_path = source_dir / "all_exercise_description.md"
        self.save_markdown(markdown_content, output_path)

class ASAPMarkdownConverter(MarkdownConverter):
    """Concrete implementation for ASAP-AES dataset."""
    
    def get_available_components(self, source_dir: Path) -> List[str]:
        # For this specific structure, we just return ['1'] since there's only one question
        return ['1']
    
    def load_components(self, source_dir: Path, component_id: str) -> ExerciseComponents:
        def load_file(filename: str) -> str:
            file_path = source_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return ""
        
        # Load all components
        question = load_file("question.txt")
        prompt = load_file("prompt.txt")
        characteristics = load_file("characteristics.txt")
        rubric = load_file("rubric.txt")
        complementary_texts = load_file("complementary_exercise_texts.txt")
        
        # Create sections in the desired order
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
            sections.append("## Default Prompt\n" + prompt)
        
        return ExerciseComponents(sections=sections)

class ASAP2MarkdownConverter(MarkdownConverter):
    """Concrete implementation for ASAP2 dataset with common prompt and rubric files."""
    
    def __init__(self, output_dir: Path, common_dir: Path):
        super().__init__(output_dir)
        self.common_dir = common_dir
    
    def get_available_components(self, source_dir: Path) -> List[str]:
        # For this specific structure, we just return ['1'] since there's only one question
        return ['1']
    
    def load_components(self, source_dir: Path, component_id: str) -> ExerciseComponents:
        def load_file(file_path: Path) -> str:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return ""
        
        # Load exercise-specific components
        question = load_file(source_dir / "question.txt")
        complementary_texts = load_file(source_dir / "complementary_exercise_texts.txt")
        
        # Load common components from the root directory
        root_dir = source_dir.parent
        prompt = load_file(root_dir / "prompt.txt")
        rubric = load_file(root_dir / "rubric.txt")
        
        # Create sections in the desired order
        sections = []
        if complementary_texts:
            sections.append(complementary_texts)
        if question:
            sections.append(question)
        if rubric:
            sections.append(rubric)
        if prompt:
            sections.append("## Default Prompt\n" + prompt)
        
        return ExerciseComponents(sections=sections)

def process_asap_aes_dataset(base_dir: Path) -> None:
    """Process the ASAP-AES dataset."""
    asap_dir = ASAP_DATASET_DIR
    for set_num in range(1, 9):
        set_dir = asap_dir / f"exercise_set_{set_num}"
        print(f"\nProcessing ASAP-AES exercise set {set_num}...")
        converter = ASAPMarkdownConverter(output_dir=set_dir)
        converter.process_all_components(set_dir)
        print(f"Markdown file saved as: {set_dir / 'all_exercise_description.md'}")

def process_asap2_dataset(base_dir: Path) -> None:
    """Process the ASAP2 dataset."""
    asap2_dir = ASAP2_DATASET_DIR
    
    # Get all exercise folders
    exercise_folders = [f for f in asap2_dir.iterdir() if f.is_dir()]
    
    for exercise_dir in exercise_folders:
        print(f"\nProcessing ASAP2 exercise: {exercise_dir.name}...")
        converter = ASAP2MarkdownConverter(output_dir=exercise_dir, common_dir=asap2_dir)
        converter.process_all_components(exercise_dir)
        print(f"Markdown file saved as: {exercise_dir / 'all_exercise_description.md'}")

if __name__ == "__main__":
    # Process both datasets
    print("Processing ASAP-AES dataset...")
    process_asap_aes_dataset(None)
    
    print("\nProcessing ASAP2 dataset...")
    process_asap2_dataset(None) 
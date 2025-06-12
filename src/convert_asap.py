from pathlib import Path
from utils.markdown_converter import ASAPMarkdownConverter

def main():
    # Define paths
    base_dir = Path("datasets/asap-aes")
    exercise_set_dir = base_dir / "exercise_set_1"
    output_dir = exercise_set_dir / "markdown"
    
    # Create and run converter
    converter = ASAPMarkdownConverter(output_dir=output_dir)
    converter.process_all_components(exercise_set_dir)
    
    print(f"Markdown files have been generated in: {output_dir}")

if __name__ == "__main__":
    main() 
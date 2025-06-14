from pathlib import Path

# Base directory for all datasets
DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"

# Individual dataset paths
ASAP_DATASET_DIR = DATASETS_DIR / "asap"
ASAP2_DATASET_DIR = DATASETS_DIR / "asap2" 
DRESS_DATASET_DIR = DATASETS_DIR / "dress"
MOHLER_DATASET_DIR = DATASETS_DIR / "mohler"

# ASAP specific paths
ASAP_RESPONSES_FILE = ASAP_DATASET_DIR / "asap_student_responses_and_evaluations.xlsx"
ASAP2_RESPONSES_FILE = ASAP2_DATASET_DIR / "asap2_student_responses_and_evaluations.csv"

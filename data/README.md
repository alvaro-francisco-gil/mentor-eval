# Original Datasets

This directory contains the original, unprocessed datasets used in the MentorEval benchmark. These datasets are used to create the train/test splits for LLM evaluation. Not all the original datasets are uploaded to this repository due to storage size limitations.

## Dataset Structure

```
original_datasets/
├── asap/                           # ASAP dataset files
├── asap2/                          # ASAP2 dataset files
└── mohler/                         # Mohler dataset files
```

## Individual Dataset Details

### ASAP Dataset
**Source**: [Kaggle - ASAP-AES](https://www.kaggle.com/competitions/asap-aes/data)  
**Description**: The Automated Student Assessment Prize (ASAP) dataset contains student essays and their corresponding scores from 8 different essay sets.  
**Paper**: [Contrasting state-of-the-art automated scoring of essays](https://www.researchgate.net/publication/283923791_Contrasting_state-of-the-art_automated_scoring_of_essays)  
**License**: CC BY-SA 3.0  
**Download Size**: 114MB total

**File Structure**:
```
original_datasets/asap/
├── training_set_rel3.xlsx          # Main training data (6.4MB)
├── training_set_rel3.xls           # Excel format (19MB)
├── training_set_rel3.tsv           # Tab-separated format (16MB)
├── valid_set.xlsx                  # Validation data (2.1MB)
├── valid_set.xls                   # Excel format (6.1MB)
├── valid_set.tsv                   # Tab-separated format (5.0MB)
├── test_set.tsv                    # Test data (5.0MB)
├── valid_sample_submission_*.csv   # Sample submission formats
├── Training_Materials.zip          # Training materials (55MB)
└── Essay_Set_Descriptions.zip      # Essay set descriptions (214KB)
```

##### add characteristics

### ASAP2 Dataset
**Source**: [Kaggle - ASAP2](https://www.kaggle.com/datasets/lburleigh/asap-2-0)  
**Description**: ASAP 2.0 is a large-scale dataset of around 25,000 source-based argumentative essays by U.S. secondary students, designed to improve automated essay scoring with added demographic info and source texts for research on writing quality and fairness.  
**Paper**: [A large-scale corpus for assessing source-based writing quality: ASAP 2.0](https://www.sciencedirect.com/science/article/pii/S1075293525000418)  
**License**: CC BY-SA 3.0  
**Download Size**: 199MB

**File Structure**:
```
original_datasets/asap2/
└── ASAP2_train_sourcetexts.csv    # Training data with source texts (199MB)
```

### Mohler Dataset
**Source**: [Kaggle - Mohler](https://www.kaggle.com/datasets/abdokamr/mohler)  
**Description**: The Mohler dataset consists of student short answers graded on a [0–5] scale, used for automated short answer grading research leveraging semantic similarity and syntactic alignments.  
**Paper**: [Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments](https://aclanthology.org/volumes/P11-1/)  
**License**: CC BY-SA 4.0  
**Download Size**: 591KB

**File Structure**:
```
original_datasets/mohler/
└── mohler_dataset_edited.csv       # Computer science evaluations
```

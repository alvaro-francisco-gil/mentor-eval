# MentorEval Benchmark

A comprehensive benchmark for evaluating LLMs on teaching evaluation tasks using multiple datasets with proper train/test splits.

## Overview

MentorEval is a collection of teaching evaluation datasets (ASAP, ASAP2, Mohler) structured for LLM evaluation with proper train/test splits. This enables:

- **Few-shot prompting** using training examples
- **Prompt engineering** on training data
- **Meta-evaluation** of evaluation methodology
- **Research reproducibility** through standard splits

## Project Structure

```
mentor-eval/
├── datasets/                    # Original dataset files (unchanged)
│   ├── asap/                   # ASAP dataset with exercise sets
│   ├── asap2/                  # ASAP2 dataset with exercise sets
│   ├── dress/                  # DREsS dataset
│   └── mohler/                 # Mohler dataset
├── registry/                    # Evaluation registry for Evals framework
│   ├── evals/                  # Evaluation YAML configurations
│   ├── data/                   # Data splits in JSONL format
│   └── README.md               # Registry documentation
└── README.md                   # This file
```

## 🚀 Quick Start

1. **Clone the repository**
2. **Download datasets** from the links above
3. **Process datasets** (see Processing Datasets section below)
4. **Run evaluations** using OpenAI Evals framework

## 📊 Processing Datasets

This section explains how to process the original datasets into standardized JSONL format for the MentorEval benchmark.

### Prerequisites

1. **Install Python dependencies:**
   ```bash
   cd original_datasets
   pip install -r requirements.txt
   ```

2. **Download the original datasets** using the links in the "Datasets" section above

3. **Place datasets in their respective directories:**
   ```
   original_datasets/
   ├── asap/
   │   ├── asap_student_responses_and_evaluations.xlsx
   │   └── exercise_set_1/ ... exercise_set_8/
   ├── asap2/
   │   └── asap2_student_responses_and_evaluations.csv
   └── mohler/
       └── mohler_dataset_edited.csv
   ```

### Usage

#### Process All Datasets
```bash
cd original_datasets
python standardize_datasets.py
```

#### Process Specific Datasets
```bash
# Process only ASAP and ASAP2
python standardize_datasets.py --datasets asap,asap2

# Process only Mohler
python standardize_datasets.py --datasets mohler
```

#### Custom Test Split Size
```bash
# Use 20% for test set (default is 30%)
python standardize_datasets.py --test-size 0.2
```

#### Check Existing Output Files
```bash
# Only check what's already been processed
python standardize_datasets.py --check-only
```

### Output Structure

The scripts create standardized JSONL files in the registry:

```
registry/data/mentoreval/
├── asap/
│   ├── train.jsonl
│   └── test.jsonl
├── asap2/
│   ├── train.jsonl
│   └── test.jsonl
└── mohler/
    ├── train.jsonl
    └── test.jsonl
```

### JSONL Format

Each sample follows this standardized format with dynamic field substitution:

```json
{
  "input": [{"role": "user", "content": "Question: {question}\n\nStudent Answer: {student_answer}\n\nRubric: {rubric}\n\nEvaluate this response."}],
  "question": "The essay question/prompt",
  "student_answer": "The student's response",
  "rubric": "Detailed grading rubric",
  "academic_level": "8",
  "rubric_range": "1-6 | Ideas: 0-3 | Organization: 0-3 | Style: 0-3 | Conventions: 0-3",
  "essay_type": "Persuasive/Narrative/Expository",
  "essay_set": 1,
  "ideal": "6"
}
```

**Benefits of Dynamic Field Substitution:**
- **Storage Efficiency**: No duplication of potentially long text content
- **Experimental Control**: Easy to create variations that include/exclude certain fields
- **Clean Data Management**: Separate fields make analysis and filtering easier
- **Framework Compatibility**: Aligns with OpenAI Evals' dynamic prompting capabilities

### Processing Details

#### ASAP Dataset
- **Source**: Excel/TSV files with essay data
- **Metadata**: Extracted from exercise set directories
- **Splitting**: Stratified by essay set and score (70/30 split)
- **Output**: 8 essay sets with full rubric information
- **Stratification**: Maintains score distribution proportions across train/test sets

#### ASAP2 Dataset
- **Source**: CSV with source texts and essays
- **Metadata**: Includes demographic information
- **Splitting**: Stratified by score and prompt
- **Output**: Source-based argumentative essays

#### Mohler Dataset
- **Source**: CSV with short answer questions
- **Metadata**: Computer science concept questions
- **Splitting**: Stratified by score
- **Output**: Technical knowledge assessments

### Stratification Strategy

The processing scripts use **intelligent stratification** to ensure fair train/test splits:

1. **Primary stratification**: By essay set to ensure representation of all prompt types
2. **Secondary stratification**: By score within each essay set when possible
3. **Fallback strategy**: Random split when stratification isn't feasible (e.g., sparse score distributions)

**Benefits**:
- Maintains score distribution proportions
- Ensures all essay types are represented in both sets
- Prevents bias in evaluation results
- Enables reproducible splits with fixed random seeds

### Error Handling

The scripts include comprehensive error handling:
- **File validation**: Checks for required input files
- **Data validation**: Ensures data quality and completeness
- **Output verification**: Confirms successful file creation
- **Progress reporting**: Shows processing status and timing

### Troubleshooting

#### Common Issues

1. **Missing dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **File not found errors:**
   - Ensure datasets are downloaded and placed in correct directories
   - Check file names match expected patterns

3. **Memory errors:**
   - ASAP2 dataset is large (~200MB), ensure sufficient RAM
   - Consider processing datasets individually if needed

4. **Permission errors:**
   - Ensure write permissions for output directory
   - Check if registry directory exists and is writable

#### Getting Help

- Check the console output for detailed error messages
- Verify dataset files are in the expected format
- Ensure all required dependencies are installed

### Customization

#### Modifying Processing Logic
Each dataset script can be customized:
- **Score normalization**: Modify scoring logic in `create_samples()`
- **Splitting strategy**: Change train/test split methodology
- **Output format**: Add or remove fields from JSONL output

#### Experimental Variations
The dynamic field substitution enables easy creation of different evaluation configurations:

```yaml
# With full context (question + answer + rubric)
mentoreval-full:
  args:
    samples_jsonl: mentoreval/asap/samples.jsonl

# Without rubric (question + answer only)
mentoreval-no-rubric:
  args:
    samples_jsonl: mentoreval/asap/samples.jsonl

# With specific rubric dimensions only
mentoreval-ideas-only:
  args:
    samples_jsonl: mentoreval/asap/samples.jsonl
```

This approach allows you to measure the impact of including/excluding rubric information on grading accuracy.

#### Adding New Datasets
To add a new dataset:
1. Create a new directory with the dataset files
2. Create a `process_[dataset_name].py` script
3. Add the dataset to `standardize_datasets.py`
4. Update this README

### Performance Notes

- **ASAP**: ~8,700 samples, processes in ~2-3 seconds
- **ASAP2**: ~25,000 samples, processes in ~60 seconds  
- **Mohler**: ~2,000 samples, processes in ~10 seconds

Processing times may vary based on system specifications and dataset sizes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Datasets

### ASAP (Automated Student Assessment Prize)
- **Task**: Essay evaluation
- **Scale**: 1-6 (undeveloped to well-developed)
- **Format**: Argumentative writing prompts
- **Use Case**: Training LLMs to evaluate student essays

### ASAP2
- **Task**: Essay evaluation with source texts
- **Scale**: 1-6 (very little mastery to clear mastery)
- **Format**: Argumentative essays based on articles
- **Use Case**: Evaluating LLMs on source-based writing assessment

### Mohler
- **Task**: Short answer evaluation
- **Scale**: 1-5 (very poor to excellent)
- **Format**: Computer science concept questions
- **Use Case**: Testing LLMs on technical knowledge assessment

## Evaluation Methodology

The benchmark uses the Evals framework's `Match` class to compare LLM-generated scores with human expert scores. Each evaluation:

1. Presents a question and student response to the LLM
2. Asks the LLM to assign a score based on the rubric
3. Compares the LLM's score with the human expert score
4. Calculates accuracy metrics

## Customization

### Adding New Datasets

1. Create evaluation YAML files in `registry/evals/`
2. Add data directories in `registry/data/mentoreval/`
3. Create `train.jsonl` and `test.jsonl` files
4. Update evaluation sets as needed

### Modifying Evaluation Metrics

Edit the YAML files to use different evaluation classes or metrics beyond the default `accuracy` metric.

## Research Applications

- **Few-shot Learning**: Use training examples to improve LLM performance
- **Prompt Engineering**: Develop optimal evaluation prompts
- **Model Comparison**: Evaluate different LLMs on teaching tasks
- **Bias Analysis**: Study evaluation consistency across different response types
- **Educational AI**: Improve automated assessment systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your dataset or evaluation method
4. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this benchmark in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]
# MentorEval Benchmark Registry

This directory contains the evaluation registry for the MentorEval benchmark, which is a collection of teaching evaluation datasets (ASAP, ASAP2, Mohler) with proper train/test splits for LLM evaluation.

## Structure

```
registry/
├── evals/                           # Evaluation YAML configurations
│   ├── mentoreval-asap-train.yaml   # ASAP training split evaluation
│   ├── mentoreval-asap-test.yaml    # ASAP test split evaluation
│   ├── mentoreval-asap2-train.yaml  # ASAP2 training split evaluation
│   ├── mentoreval-asap2-test.yaml   # ASAP2 test split evaluation
│   ├── mentoreval-mohler-train.yaml # Mohler training split evaluation
│   ├── mentoreval-mohler-test.yaml  # Mohler test split evaluation
│   ├── mentoreval-train-set.yaml    # Multi-dataset training set
│   └── mentoreval-test-set.yaml     # Multi-dataset test set
└── data/                            # Data files in JSONL format
    └── mentoreval/
        ├── asap/                    # ASAP dataset splits
        │   ├── train.jsonl
        │   └── test.jsonl
        ├── asap2/                   # ASAP2 dataset splits
        │   ├── train.jsonl
        │   └── test.jsonl
        └── mohler/                  # Mohler dataset splits
            ├── train.jsonl
            └── test.jsonl
```

## Why Train/Test Splits for LLM Benchmarks?

Even though LLMs are typically pre-trained, having train/test splits in your mentoreval benchmark is valuable for:

- **Few-shot prompting**: Using train examples as demonstrations in prompts
- **Prompt engineering**: Developing optimal evaluation prompts on train data
- **Meta-evaluation**: Testing your evaluation methodology before final assessment
- **Research reproducibility**: Standard practice in academic benchmarks

## Running Evaluations

### Individual Dataset Evaluations

```bash
# Development and prompt engineering on train split
oaieval gpt-3.5-turbo mentoreval-asap --registry_path ./registry

# Final evaluation on test split  
oaieval gpt-3.5-turbo mentoreval-asap-test --registry_path ./registry

# ASAP2 evaluations
oaieval gpt-3.5-turbo mentoreval-asap2 --registry_path ./registry
oaieval gpt-3.5-turbo mentoreval-asap2-test --registry_path ./registry

# Mohler evaluations
oaieval gpt-3.5-turbo mentoreval-mohler --registry_path ./registry
oaieval gpt-3.5-turbo mentoreval-mohler-test --registry_path ./registry
```

### Multi-Dataset Evaluation Sets

```bash
# Run all training splits together
oaievalset gpt-3.5-turbo mentoreval-train-set --registry_path ./registry

# Run all test splits together
oaievalset gpt-3.5-turbo mentoreval-test-set --registry_path ./registry
```

## Data Format

Each JSONL file contains evaluation samples in the following format:

```json
{
  "input": [
    {
      "role": "user", 
      "content": "Question: [Question text]\n\nStudent Answer: [Student response]"
    }
  ],
  "ideal": "[Expected score]"
}
```

### Scoring Scales

- **ASAP**: 1-6 scale (1=undeveloped, 6=well-developed)
- **ASAP2**: 1-6 scale (1=very little mastery, 6=clear and consistent mastery)
- **Mohler**: 1-5 scale (1=very poor, 5=excellent)

## Customization

### Adding New Datasets

1. Create new evaluation YAML files in `registry/evals/`
2. Add corresponding data directories in `registry/data/mentoreval/`
3. Create `train.jsonl` and `test.jsonl` files
4. Update the evaluation sets if desired

### Modifying Evaluation Metrics

The current evaluations use the `Match` class with accuracy metrics. You can modify the YAML files to use different evaluation classes or metrics as needed.

## Integration with Evals Framework

This registry follows the Evals framework conventions:

- Uses standard naming patterns (`mentoreval-[dataset].[split].v0`)
- Implements the `Match` evaluation class for scoring
- Provides both individual and grouped evaluation configurations
- Supports external registry paths for focused development

## Notes

- The `datasets/` folder in the root directory contains the original dataset files and should remain unchanged
- Sample JSONL files are provided for demonstration; replace with your actual data splits
- All evaluations use the `accuracy` metric by default
- The structure supports easy expansion to additional datasets or evaluation types


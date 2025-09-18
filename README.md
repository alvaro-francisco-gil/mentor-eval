# MentorEval

<div align="center">
  <img src="webpage/assets/mentoreval_logo_nobg.png" alt="MentorEval Logo" width="300"/>
</div>

<div align="center">
  <h3>A Multilingual Benchmark for Educational Assessment</h3>
  <p>Evaluating Language Models on Student Response Grading Tasks</p>
</div>

<div align="center">
  <a href="https://huggingface.co/datasets/alvaro-francisco-gil/mentor-eval">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-GPL-green.svg" alt="License">
  </a>
</div>

---

## Overview

**MentorEval** is a comprehensive benchmark designed to evaluate language models on multilingual educational assessment tasks. This benchmark focuses on automated essay scoring (AES) and automatic short answer grading (ASAG) across multiple languages and educational levels.

### Key Features

- 🌍 **Multilingual**: Supports English, Portuguese, and Arabic
- 🎓 **Multi-level**: Covers ISCED levels 3 (lower secondary) and 6 (tertiary)
- 📊 **Comprehensive**: 55,312+ student responses across 6 datasets
- 🔧 **LightEval Compatible**: Built on [LightEval](https://github.com/huggingface/lighteval) framework
- 📈 **Extensible**: Collaborative collection of open datasets

## Dataset

The benchmark contains **55,312+ student responses** across **6 datasets** from **3 languages** (English, Portuguese, Arabic), covering ISCED levels 3 and 6. The dataset is publicly available on Hugging Face and includes automated essay scoring (AES) and automatic short answer grading (ASAG) tasks.

| Dataset | Language | ISCED Level | Samples | Description |
|---------|----------|-------------|---------|-------------|
| **ASAP** | English | 3 | 12,977 | Student essays from grades 7–10 |
| **ASAP 2.0** | English | 3 | 24,728 | Enhanced automated essay scoring |
| **ELLIPSE** | English | 3 | 6,482 | English learner essays |
| **Mohler** | English | 6 | 1,263 | Computer science short answers |
| **PT-ASAG 2018** | Portuguese | 3 | 9,862 | Portuguese short answer grading |
| **AR-ASAG** | Arabic | 6 | 2,132 | Arabic short answer grading |

## Framework Integration

MentorEval is built on top of **[LightEval](https://github.com/huggingface/lighteval)**, Hugging Face's evaluation framework. This ensures:

- ✅ **Full Compatibility**: Seamless integration with LightEval's evaluation pipeline
- ✅ **Standardized Metrics**: Consistent evaluation across different models
- ✅ **Scalable Infrastructure**: Built-in support for distributed evaluation
- ✅ **Model Agnostic**: Works with any model supported by LightEval

## Repository Structure

```
mentor-eval/
├── src/mentoreval/          # Core package
│   ├── benchmark.py         # LightEval integration
│   ├── task.py             # Task definitions
│   ├── metrics.py          # Evaluation metrics
│   ├── prompts.py          # Prompt templates
│   ├── models.py           # Model configurations
│   ├── run_manager.py      # Run management
│   └── cli.py              # Command-line interface
├── data/                   # Dataset files
│   ├── raw/               # Original datasets
│   ├── processed/         # Processed datasets
│   └── mentoreval.parquet # Unified dataset
├── runs/                  # Run configurations
├── results/               # Evaluation results
├── tests/                 # Test suite
└── scripts/               # Utility scripts
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alvaro-francisco-gil/mentor-eval.git
cd mentor-eval

# Install the package (includes all dependencies automatically)
pip install -e .
```

### Running Evaluations

1. **Create a run configuration** in the `runs/` directory:

```json
{
  "run_id": 1,
  "model_name": "gpt-4o-mini",
  "benchmark_mode": "mentoreval-test",
  "description": "Test run with GPT-4o-mini",
  "status": "pending",
  "parameters": {
    "model_name": "gpt-4o-mini",
    "training_examples": 5,
    "test_samples": 100,
    "task_name": "mentoreval_asap2"
  }
}
```

2. **Execute the evaluation**:

```bash
# Run a specific evaluation
mentoreval --execute 1

# Run all pending evaluations
mentoreval --execute-all

# List all runs
mentoreval --list

# Show run summary
mentoreval --summary
```

3. **View results** in the `results/` directory.

## Contributing

MentorEval is a **collaborative collection of open datasets**. We welcome contributions to expand the benchmark:

### Adding New Datasets

We welcome both **raw datasets** and **standardized datasets**:

1. **Add dataset** to `data/raw/[dataset_name]/` (raw) or `data/processed/[dataset_name]/` (standardized)
2. **Create processing script** in `scripts/process_datasets.py` (if needed)
3. **Update dataset registry** in the codebase
4. **Submit a pull request** with your contribution

### Guidelines

- Ensure datasets are properly licensed for research use
- Follow the standardized data schema
- Include proper documentation and metadata
- Test your contributions with the existing framework

## Evaluation Metrics

MentorEval includes comprehensive evaluation metrics:

- **Exact Grade Match**: Percentage of exact score matches
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Root Mean Square Error (RMSE)**: Standard deviation of errors
- **Pearson Correlation**: Linear correlation with human scores
- **Spearman Correlation**: Rank correlation with human scores
- **Kolmogorov-Smirnov Statistic**: Distribution comparison
- **Wasserstein Distance**: Distribution distance metric

## Citation

If you use MentorEval in your research, please cite:

```bibtex
@dataset{mentoreval,
  title={MentorEval: A Multilingual Benchmark for Educational Assessment},
  author={Álvaro Francisco Gil},
  year={2025},
  url={https://huggingface.co/datasets/alvaro-francisco-gil/mentor-eval}
}
```

## License

This project is licensed under the GNU General Public License (GPL) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

MentorEval builds upon several existing datasets and frameworks:

- **[LightEval](https://github.com/huggingface/lighteval)** - Evaluation framework
- **ASAP, ASAP 2.0, ELLIPSE, Mohler, PT-ASAG 2018, AR-ASAG** - Source datasets
- **Hugging Face** - Dataset hosting and community support

---

<div align="center">
  <p><strong>MentorEval</strong> - Advancing Educational AI through Comprehensive Benchmarking</p>
  <p>Made with ❤️ for the educational AI community</p>
</div>

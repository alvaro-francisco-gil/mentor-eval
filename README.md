# MentorEval

<div align="center">
  <img src="webpage/assets/mentoreval_logo_nobg.png" alt="MentorEval Logo" width="300"/>
</div>

<div align="center">
  <h3>A Multilingual Benchmark for Educational Assessment</h3>
  <p>Evaluating Language Models on Student Response Grading Tasks</p>
</div>

<div align="center">
  <a href="https://alvaro-francisco-gil.github.io/mentor-eval/">
    <img src="https://img.shields.io/badge/📊%20Leaderboard-Results-orange" alt="Leaderboard">
  </a>
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

Access the benchmark leaderboard [here](https://alvaro-francisco-gil.github.io/mentor-eval/)

### Key Features

- 🌍 **Multilingual**: Supports English, Portuguese, and Arabic
- 🎓 **Multi-level**: Covers ISCED levels 1 (primary), 2 (lower secondary), 3 (upper secondary), 6 (bachelor's), and 7 (master's)
- 📊 **Comprehensive**: 57,444 student responses across 6 datasets
- 🔧 **LightEval Compatible**: Built on [LightEval](https://github.com/huggingface/lighteval) framework
- 📈 **Extensible**: Collaborative collection of open datasets

## Dataset

The benchmark contains **57,444 student responses** across **6 datasets** from **3 languages** (English, Portuguese, Arabic), covering ISCED levels 1, 2, 3, 6, and 7. The dataset is publicly available on Hugging Face and includes automated essay scoring (AES) and automatic short answer grading (ASAG) tasks.

| Dataset | Language | ISCED Level | Samples | Description |
|---------|----------|-------------|---------|-------------|
| **ASAP** | English | 2, 3 | 12,977 | Student essays from grades 7–10|
| **ASAP 2.0** | English | 1, 2, 3 | 24,728 | Enhanced automated essay scoring |
| **ELLIPSE** | English | 3 | 6,482 | English learner essays |
| **Mohler** | English | 6 | 1,263 | Computer science short answers |
| **PT-ASAG 2018** | Portuguese | 2 | 9,862 | Portuguese short answer grading |
| **AR-ASAG** | Arabic | 7 | 2,132 | Arabic short answer grading |


> Use the [Hugging Face dataset](https://huggingface.co/datasets/alvaro-francisco-gil/mentor-eval) for data download, and the [dedicated repo](https://github.com/alvaro-francisco-gil/mentor-eval-dataset) for collaboration on data updates.

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
│   ├── task.py              # Task definitions
│   ├── metrics.py           # Evaluation metrics
│   ├── prompts.py           # Prompt templates
│   ├── models.py            # Model configurations
│   ├── run_manager.py       # Run management
│   └── cli.py               # Command-line interface
├── data/                    # (hosted in mentor-eval-dataset repo / HF dataset)
├── runs/                    # Run configurations
├── results/                 # Evaluation results
├── tests/                   # Test suite
└── scripts/                 # Utility scripts
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
  "status": "completed",
  "parameters": {
    "model_name": "gpt-4o-mini",
    "training_examples": 0,
    "test_samples": 5,
    "task_name": "mentoreval",
    "show_guidance": true,
    "explanation": false,
    "show_isced_level": true
  },
  "configuration": {
    "use_local_backend": false,
    "generation_args": {
      "max_new_tokens": 50,
      "temperature": 0.0,
      "do_sample": false
    }
  }
}
```

2. **Execute the evaluation**:

```bash
# Run a specific evaluation
mentoreval --execute 1

# Run all pending evaluations
mentoreval --execute-all
```

3. **Available Task Configurations**:

The benchmark supports multiple task configurations for different evaluation scenarios:

- **Full Benchmark**: `mentoreval` - All datasets and exercise sets
- **Dataset-level Tasks**: 
  - `mentoreval_asap` - All ASAP exercise sets (1-8)
  - `mentoreval_asap2` - All ASAP2 exercise sets (1-7)
  - `mentoreval_mohler` - All Mohler exercise sets (1-81)
  - `mentoreval_ellipse` - All ELLIPSE exercise sets (1-44)
  - `mentoreval_ptasag2018` - All PTASAG2018 exercise sets (1-15)
  - `mentoreval_arasag` - All ARASAG exercise sets (1-48)
- **Exercise-type Tasks**:
  - `mentoreval_essay_writing` - All essay writing tasks (ASAP, ASAP2, ELLIPSE)
  - `mentoreval_short_answer` - All short answer tasks (Mohler, PTASAG2018, ARASAG)
- **Individual Exercise Tasks**: `mentoreval_[dataset]_ex[number]` (e.g., `mentoreval_asap_ex1`)

4. **Configuration Parameters**:

The run configuration supports several parameters to customize the evaluation:

- **`show_guidance`**: Include grading guidance in prompts (default: `true`)
- **`explanation`**: Request explanations for grading decisions (default: `false`)
- **`show_isced_level`**: Include ISCED educational level information (default: `true`)
- **`training_examples`**: Number of few-shot examples to include (default: `0`)
- **`test_samples`**: Number of test samples to evaluate (default: `5` for testing, use `-1` for full evaluation)

5. **View results** in the `results/` directory.

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

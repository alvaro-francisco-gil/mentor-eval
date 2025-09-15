---
license: gpl
task_categories:
- text-classification
- text-generation
language:
- en
- pt
- ar
tags:
- education
pretty_name: Mentor Evaluation Benchmark (mentor-eval)
size_categories:
- 10K<n<100K
---

# Dataset Card for MentorEval

MentorEval is a multilingual benchmark dataset designed to evaluate automated systems on educational assessment tasks. It combines multiple existing datasets into a unified, standardized format for fair comparison of automated grading models.

## Dataset Summary

MentorEval contains **55,312 student responses** across **6 different datasets** from **3 languages** (English, Portuguese, and Arabic), covering various educational levels and assessment types. The dataset is designed for research in automated essay scoring (AES) and automatic short answer grading (ASAG).

## Dataset Structure

The dataset is split into:
- **Train set**: 11,062 samples (20%)
- **Test set**: 44,250 samples (80%)

The splits are stratified by dataset and grade ranges to ensure fair evaluation across different educational contexts.

## Source Datasets

| Dataset | Language | ISCED Level | Graders | License | Exercises | Samples | Description |
|---------|----------|-------------|---------|---------|-----------|---------|-------------|
| **ASAP** | English | 3 | 2 | GPL | 8 | 12,977 | Student essays from grades 7–10 on 8 distinct essay prompts, scored for overall and attribute-specific writing quality |
| **ASAP 2.0** | English | 3 | 1 | CC BY 4.0 | 7 | 24,728 | Enhanced automated essay scoring dataset with essays from diverse student populations and multiple essay attributes |
| **ELLIPSE** | English | 3 | 2 | CC BY 4.0 | 44 | 6,482 | English learner essays graded on six linguistic and writing quality dimensions (only data where both raters agreed) |
| **Mohler** | English | 6 | 2 | GPL | 81 | 1,263 | Short answers from computer science students at a Texas university, graded on a 0–5 scale (only data where both raters agreed) |
| **PT-ASAG 2018** | Portuguese | 3 | 1 | CC BY 4.0 | 15 | 9,862 | Real student and teacher answers from Brazil for Portuguese Automatic Short Answer Grading research |
| **AR-ASAG** | Arabic | 6 | 2 | CC BY-NC | 48 | 2,132 | Arabic dataset for automatic short answer grading containing pairs of model and student answers from three university exams |

## Dataset Features

- **Multilingual**: English, Portuguese, and Arabic
- **Multi-level**: Covers ISCED levels 3 (lower secondary) and 6 (tertiary)
- **Diverse assessment types**: Essay writing and short answer grading
- **Quality control**: Includes only data where multiple raters agreed (where applicable)
- **Standardized format**: All datasets converted to unified schema

## Data Schema

Each sample contains the following fields:
- `dataset`: Source dataset identifier
- `exercise_set`: Exercise/prompt identifier
- `question`: The question or prompt text
- `answer`: Student's response
- `grade`: Assigned grade/score
- `min_grade`: Minimum possible grade for the exercise
- `max_grade`: Maximum possible grade for the exercise
- `subject`: Subject area (e.g., english, math)
- `exercise_type`: Type of exercise (e.g., essay_writing)
- `isced_level`: Education level (ISCED classification)
- `language`: Language of the content
- `rubric`: Grading rubric/guidelines
- `desired_answer`: Reference answer (when available)
- `metadata`: Additional metadata

## Usage

This dataset is designed for:
- Automated essay scoring (AES) research
- Automatic short answer grading (ASAG) research
- Cross-dataset evaluation of grading models
- Multilingual educational assessment research
- Fair comparison of automated grading systems

## References

- **ASAP**: 
  - Paper: [State-of-the-art automated essay scoring: Competition, results, and future directions from a United States demonstration](https://www.kaggle.com/competitions/asap-aes)
  - Dataset: [ASAP AES Competition](https://www.kaggle.com/competitions/asap-aes)

- **ASAP 2.0**: 
  - Paper: [ASAP 2.0 Dataset](https://the-learning-agency-lab.com/learning-exchange/asap-2-0-dataset/)
  - Dataset: [ASAP 2.0 on Kaggle](https://www.kaggle.com/datasets/lburleigh/asap-2-0)

- **ELLIPSE**: 
  - Paper: [ELLIPSE Corpus](https://benjamins.com/catalog/ijlcr.22026.cro?srsltid=AfmBOoqIEK6lzlQ2UUWOxWDlQ8msOmWAvA3Q_CjZvXSQ2G_mWZVnreJW)
  - Dataset: [ELLIPSE on GitHub](https://github.com/scrosseye/ELLIPSE-Corpus)

- **Mohler**: 
  - Paper: [Mohler et al. (2011)](https://aclanthology.org/P11-1076/)
  - Dataset: [Mohler Dataset on Kaggle](https://www.kaggle.com/datasets/abdokamr/mohler)

- **PT-ASAG 2018**: 
  - Paper: [Portuguese Automatic Short Answer Grading](https://www.researchgate.net/publication/328735284_Portuguese_Automatic_Short_Answer_Grading)
  - Dataset: [PT-ASAG 2018 on Kaggle](https://www.kaggle.com/datasets/lucasbgalhardi/pt-asag-2018)

- **AR-ASAG**: 
  - Paper: [AR-ASAG Dataset](https://aclanthology.org/2020.lrec-1.321/)
  - Dataset: [AR-ASAG on Kaggle](https://www.kaggle.com/datasets/mahmoudsammour/ar-asag-dataset/data)

## Citation

If you use this dataset, please cite the original datasets and this unified version:

```bibtex
@dataset{mentoreval,
  title={MentorEval: A Multilingual Benchmark for Educational Assessment},
  author={Álvaro Francisco Gil},
  year={2025},
  url={https://huggingface.co/datasets/alvaro-francisco-gil/mentor-eval}
}
```

## License

GNU General Public License (GPL)
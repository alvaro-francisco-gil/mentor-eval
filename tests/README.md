# MentorEval Test Suite

This directory contains comprehensive validation tests for the MentorEval datasets.

## 🧪 Test Files

### **Main Test Runner**
- **`test_all.py`** - Main orchestrator that runs all validation tests

### **Individual Test Modules**
- **`test_jsonl_basic.py`** - Basic JSONL validation (JSON syntax, required fields)
- **`test_metrics_consistency.py`** - Validates `num_metrics` matches `ideal_*` field count
- **`test_score_sums.py`** - Validates `ideal` equals sum of all `ideal_*` values
- **`test_rubric_ranges.py`** - Validates rubric range format and ideal values within ranges

## 🚀 Usage

### **Run All Tests**
```bash
# From project root
make tests
# or
python tests/test_all.py
```

### **Run Individual Tests**
```bash
# Basic JSONL validation only
make test-basic
python tests/test_jsonl_basic.py

# Metrics consistency only
make test-metrics
python tests/test_metrics_consistency.py

# Score sum validation only
make test-scores
python tests/test_score_sums.py

# Rubric range validation only
make test-rubric
python tests/test_rubric_ranges.py
```

### **Make Commands**
```bash
make tests        # Run all tests
make test-basic   # Basic validation only
make test-metrics # Metrics consistency only
make test-scores  # Score sums only
make test-rubric  # Rubric ranges only
make test-all     # Same as make tests
```

## 🔍 Enhanced Error Reporting

All tests now provide **focused debugging information** instead of verbose output:

- **Line numbers** where errors occur
- **Specific field names** and values that failed
- **Expected vs actual values** for easy debugging
- **Available options** when validation fails

### **Example Error Output**
```
🔍 DEBUGGING INFO (First 3 failures):
   Line 8577: No matching rubric range found for ideal_style_score
      Field: ideal_style_score
      Value: 4
      Available rubric keys: ['ideal_ideas_score', 'ideal_organization_score', ...]
```

## 📊 Test Coverage

The test suite validates:

1. **JSONL Format** - Valid JSON syntax and required fields
2. **Metrics Consistency** - `num_metrics` matches `ideal_*` field count
3. **Score Validation** - `ideal` equals sum of individual scores
4. **Rubric Ranges** - Format validation and value range checking

## 🎯 Test Results

Tests return `True` (all passed) or `False` (some failed) for easy integration with CI/CD pipelines.

## 📁 Dataset Support

Currently supports validation of:
- **ASAP** dataset (train/test)
- **ASAP2** dataset (train/test) 
- **Mohler** dataset (train/test)

Each dataset is validated independently, with clear reporting of which files passed/failed.

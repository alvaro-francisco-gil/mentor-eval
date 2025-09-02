# Makefile for MentorEval project
# Provides convenient commands for dataset processing and testing

.PHONY: help datasets tests test-basic test-metrics test-scores test-rubric test-all clean

help:
	@echo "MentorEval Project - Available Commands:"
	@echo ""
	@echo "📊 Dataset Processing:"
	@echo "  make datasets     - Process all datasets (ASAP, ASAP2, Mohler)"
	@echo "  make asap        - Process ASAP dataset only"
	@echo "  make asap2       - Process ASAP2 dataset only"
	@echo "  make mohler      - Process Mohler dataset only"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make tests       - Run all validation tests"
	@echo "  make test-basic  - Run basic JSONL validation only"
	@echo "  make test-metrics- Run metrics consistency validation only"
	@echo "  make test-scores - Run score sum validation only"
	@echo "  make test-rubric - Run rubric range validation only"
	@echo "  make test-all    - Run all validation tests (same as tests)"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make clean       - Remove generated files"
	@echo "  make install     - Install dependencies"
	@echo "  make check-datasets - Check if dataset files exist"
	@echo "  make workflow    - Process datasets then run tests"
	@echo "  make dev         - Install dependencies and run tests"
	@echo "  make stats       - Show dataset statistics"

# Dataset processing targets
datasets: asap asap2 mohler
	@echo "✅ All datasets processed successfully!"

asap:
	@echo "🔄 Processing ASAP dataset..."
	@cd original_datasets/asap && python3 process_asap.py
	@echo "✅ ASAP dataset processed!"

asap2:
	@echo "🔄 Processing ASAP2 dataset..."
	@cd original_datasets/asap2 && python3 process_asap.py
	@echo "✅ ASAP2 dataset processed!"

mohler:
	@echo "🔄 Processing Mohler dataset..."
	@cd original_datasets/mohler && python3 process_mohler.py
	@echo "✅ Mohler dataset processed!"

# Testing targets
tests: test-all
	@echo "✅ All tests completed!"

test-basic:
	@echo "🧪 Running Basic JSONL Validation..."
	@python3 tests/test_jsonl_basic.py

test-metrics:
	@echo "🧪 Running Metrics Consistency Validation..."
	@python3 tests/test_metrics_consistency.py

test-scores:
	@echo "🧪 Running Score Sum Validation..."
	@python3 tests/test_score_sums.py

test-rubric:
	@echo "🧪 Running Rubric Range Validation..."
	@python3 tests/test_rubric_ranges.py

test-all:
	@echo "🧪 Running All JSONL Validation Tests..."
	@python3 tests/test_all.py

# Quick test for development
quick-test: test-basic
	@echo "🚀 Quick test completed!"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf registry/data/mentoreval/*/*.jsonl
	@echo "✅ Cleanup completed!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip3 install -r requirements.txt
	@echo "✅ Dependencies installed"

# Check if dataset files exist
check-datasets:
	@echo "🔍 Checking dataset files..."
	@if [ -f "registry/data/mentoreval/asap/train.jsonl" ]; then echo "✅ ASAP train.jsonl exists"; else echo "❌ ASAP train.jsonl missing"; fi
	@if [ -f "registry/data/mentoreval/asap/test.jsonl" ]; then echo "✅ ASAP test.jsonl exists"; else echo "❌ ASAP test.jsonl missing"; fi
	@if [ -f "registry/data/mentoreval/asap2/train.jsonl" ]; then echo "✅ ASAP2 train.jsonl exists"; else echo "❌ ASAP2 train.jsonl missing"; fi
	@if [ -f "registry/data/mentoreval/asap2/test.jsonl" ]; then echo "✅ ASAP2 test.jsonl exists"; else echo "❌ ASAP2 test.jsonl missing"; fi
	@if [ -f "registry/data/mentoreval/mohler/train.jsonl" ]; then echo "✅ Mohler train.jsonl exists"; else echo "❌ Mohler train.jsonl missing"; fi
	@if [ -f "registry/data/mentoreval/mohler/test.jsonl" ]; then echo "✅ Mohler test.jsonl exists"; else echo "❌ Mohler test.jsonl missing"; fi

# Complete workflow: process datasets then test
workflow: datasets tests
	@echo "🎉 Complete workflow finished!"

# Development setup: install dependencies and run tests
dev: install tests
	@echo "🚀 Development environment ready!"

# Show dataset statistics
stats:
	@echo "📊 Dataset Statistics:"
	@if [ -f "registry/data/mentoreval/asap/train.jsonl" ]; then echo "  ASAP:"; echo "    Train: $$(wc -l < registry/data/mentoreval/asap/train.jsonl) lines"; else echo "  ASAP: Not found"; fi
	@if [ -f "registry/data/mentoreval/asap/test.jsonl" ]; then echo "    Test:  $$(wc -l < registry/data/mentoreval/asap/test.jsonl) lines"; else echo "    Test:  Not found"; fi
	@if [ -f "registry/data/mentoreval/asap2/train.jsonl" ]; then echo "  ASAP2:"; echo "    Train: $$(wc -l < registry/data/mentoreval/asap2/train.jsonl) lines"; else echo "  ASAP2: Not found"; fi
	@if [ -f "registry/data/mentoreval/asap2/test.jsonl" ]; then echo "    Test:  $$(wc -l < registry/data/mentoreval/asap2/test.jsonl) lines"; else echo "    Test:  Not found"; fi
	@if [ -f "registry/data/mentoreval/mohler/train.jsonl" ]; then echo "  Mohler:"; echo "    Train: $$(wc -l < registry/data/mentoreval/mohler/train.jsonl) lines"; else echo "  Mohler: Not found"; fi
	@if [ -f "registry/data/mentoreval/mohler/test.jsonl" ]; then echo "    Test:  $$(wc -l < registry/data/mentoreval/mohler/test.jsonl) lines"; else echo "    Test:  Not found"; fi

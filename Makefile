# Makefile for MentorEval project
# Provides convenient commands for dataset processing and testing

.PHONY: help datasets tests venv venv-clean

help:
	@echo "MentorEval Project - Available Commands:"
	@echo ""
	@echo "🐍 Environment:"
	@echo "  make venv         - Create venv and install all dependencies"
	@echo "  make venv-clean   - Remove the venv folder"
	@echo ""
	@echo "📊 Dataset Processing:"
	@echo "  make datasets     - Process all datasets (ASAP, ASAP2, Mohler)"
	@echo "  make asap        - Process ASAP dataset only"
	@echo "  make asap2       - Process ASAP2 dataset only"
	@echo "  make mohler      - Process Mohler dataset only"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make tests       - Run all validation tests"

# Dataset processing targets
datasets: asap asap2 mohler
	@echo "✅ All datasets processed successfully!"

asap:
	@echo "🔄 Processing ASAP dataset..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -c "import sys; sys.path.insert(0, 'scripts/dataset_processing'); from process_asap import ASAPProcessor; ASAPProcessor(data_dir='data/raw/asap', output_dir='data/processed/asap').process()"
	@echo "✅ ASAP dataset processed!"

asap2:
	@echo "🔄 Processing ASAP2 dataset..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -c "import sys; sys.path.insert(0, 'scripts/dataset_processing'); from process_asap2 import ASAP2Processor; ASAP2Processor(data_dir='data/raw/asap2', output_dir='data/processed/asap2').process()"
	@echo "✅ ASAP2 dataset processed!"

mohler:
	@echo "🔄 Processing Mohler dataset..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	if [ -f scripts/dataset_processing/process_mohler.py ]; then \
		"$$PY" scripts/dataset_processing/process_mohler.py; \
	else \
		echo "⚠️  Mohler processing script not found. Skipping."; \
	fi
	@echo "✅ Mohler dataset processed!"

tests:
	@echo "🧪 Running All JSONL Validation Tests..."
	@python3 tests/test_all.py

# Quick test for development
quick-test: test-basic
	@echo "🚀 Quick test completed!"

clean:
	@echo "🧹 Cleaning up processed JSONL files..."
	@rm -rf data/processed/asap data/processed/asap2 data/processed/mohler
	@echo "✅ Cleanup completed!"

# Create venv and install deps (works on Linux/WSL and Windows)
venv:
	@echo "🐍 Creating virtual environment..."
	@python3 -m venv venv 2>/dev/null || python -m venv venv
	@echo "📦 Installing dependencies into venv..."
	@if [ -x "venv/bin/pip" ]; then \
		venv/bin/pip install -r requirements.txt; \
	elif [ -x "venv/Scripts/pip.exe" ]; then \
		venv/Scripts/pip.exe install -r requirements.txt; \
	elif [ -x "venv/Scripts/pip" ]; then \
		venv/Scripts/pip install -r requirements.txt; \
	else \
		echo "❌ Could not find venv pip. Ensure Python venv was created."; exit 1; \
	fi
	@echo "✅ venv ready. You can now run: make asap2"

# Remove venv directory
venv-clean:
	@echo "🗑 Removing venv..."
	@rm -rf venv
	@echo "✅ venv removed"

# Print activation hint for current platform
venv-activate:
	@if [ -f "venv/bin/activate" ]; then \
		echo "Run: source venv/bin/activate"; \
	elif [ -f "venv/Scripts/Activate.ps1" ]; then \
		echo "Run in PowerShell: .\\venv\\Scripts\\Activate.ps1"; \
	elif [ -f "venv/Scripts/activate" ]; then \
		echo "Run in cmd.exe: .\\venv\\Scripts\\activate"; \
	else \
		echo "venv not found. Create it with: make venv"; \
	fi

# Open an interactive subshell with venv activated (WSL/Linux)
venv-shell:
	@if [ -f "venv/bin/activate" ]; then \
		echo "Launching subshell with venv activated..."; \
		bash -i -c 'source venv/bin/activate && exec bash -i'; \
	else \
		echo "venv not found or not a Linux/WSL shell. Use: make venv-activate"; \
	fi

# Check if dataset files exist
check-datasets:
	@echo "🔍 Checking dataset files..."
	@if [ -d "data/processed/asap" ] && find data/processed/asap -name train.jsonl -print -quit | grep -q .; then echo "✅ ASAP processed files found"; else echo "❌ ASAP processed files missing"; fi
	@if [ -d "data/processed/asap2" ] && find data/processed/asap2 -name train.jsonl -print -quit | grep -q .; then echo "✅ ASAP2 processed files found"; else echo "❌ ASAP2 processed files missing"; fi
	@if [ -d "data/processed/mohler" ] && find data/processed/mohler -name train.jsonl -print -quit | grep -q .; then echo "✅ Mohler processed files found"; else echo "❌ Mohler processed files missing"; fi

# Complete workflow: process datasets then test
workflow: datasets tests
	@echo "🎉 Complete workflow finished!"

# Development setup: install dependencies and run tests
dev: install tests
	@echo "🚀 Development environment ready!"

# Show dataset statistics
stats:
	@echo "📊 Dataset Statistics:"
	@echo "ASAP:"
	@if [ -d "data/processed/asap" ]; then echo "  Train total lines: $$(find data/processed/asap -name train.jsonl -exec cat {} + | wc -l)"; else echo "  Train: Not found"; fi
	@if [ -d "data/processed/asap" ]; then echo "  Test  total lines: $$(find data/processed/asap -name test.jsonl -exec cat {} + | wc -l)"; else echo "  Test:  Not found"; fi
	@echo "ASAP2:"
	@if [ -d "data/processed/asap2" ]; then echo "  Train total lines: $$(find data/processed/asap2 -name train.jsonl -exec cat {} + | wc -l)"; else echo "  Train: Not found"; fi
	@if [ -d "data/processed/asap2" ]; then echo "  Test  total lines: $$(find data/processed/asap2 -name test.jsonl -exec cat {} + | wc -l)"; else echo "  Test:  Not found"; fi
	@echo "Mohler:"
	@if [ -d "data/processed/mohler" ]; then echo "  Train total lines: $$(find data/processed/mohler -name train.jsonl -exec cat {} + | wc -l)"; else echo "  Train: Not found"; fi
	@if [ -d "data/processed/mohler" ]; then echo "  Test  total lines: $$(find data/processed/mohler -name test.jsonl -exec cat {} + | wc -l)"; else echo "  Test:  Not found"; fi

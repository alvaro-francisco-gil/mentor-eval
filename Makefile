# Makefile for MentorEval project
# Provides convenient commands for dataset processing and testing

.PHONY: help dataset upload-hf tests venv venv-clean cli

help:
	@echo "MentorEval Project - Available Commands:"
	@echo ""
	@echo "🐍 Environment:"
	@echo "  make venv         - Create venv and install MentorEval package (includes all dependencies)"
	@echo "  make install      - Install MentorEval package in existing venv"
	@echo "  make install-requirements - Install only requirements.txt (alternative approach)"
	@echo "  make venv-clean   - Remove the venv folder"
	@echo ""
	@echo "📊 Dataset Processing:"
	@echo "  make dataset      - Process all datasets using unified script"
	@echo "  make upload-hf    - Upload processed dataset to Hugging Face Hub"
	@echo "  make dataset-workflow - Process datasets and upload to HF in one command"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make tests       - Run all validation tests"
	@echo ""
	@echo "🚀 CLI Commands:"
	@echo "  make cli-list    - List all runs"
	@echo "  make cli-summary - Show run summary"
	@echo "  make cli-execute N - Execute run N"
	@echo "  make cli-execute-all - Execute all unexecuted runs"

# Dataset processing target
dataset:
	@echo "🔄 Processing all datasets using unified script..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" scripts/process_datasets.py
	@echo "✅ All datasets processed successfully!"

# Upload dataset to Hugging Face Hub
upload-hf:
	@echo "🚀 Uploading dataset to Hugging Face Hub..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" scripts/data_upload_hf.py
	@echo "✅ Dataset uploaded to Hugging Face Hub successfully!"

tests:
	@echo "🧪 Running All JSONL Validation Tests..."
	@python3 tests/test_all.py

# Quick test for development
quick-test: test-basic
	@echo "🚀 Quick test completed!"

clean:
	@echo "🧹 Cleaning up processed dataset files..."
	@rm -rf data/processed/
	@rm -f data/mentoreval.parquet
	@echo "✅ Cleanup completed!"

# Create venv and install deps (works on Linux/WSL and Windows)
venv:
	@echo "🐍 Creating virtual environment..."
	@python3 -m venv venv 2>/dev/null || python -m venv venv
	@echo "📦 Installing MentorEval package and dependencies..."
	@if [ -x "venv/bin/pip" ]; then \
		venv/bin/pip install -e .; \
	elif [ -x "venv/Scripts/pip.exe" ]; then \
		venv/Scripts/pip.exe install -e .; \
	elif [ -x "venv/Scripts/pip" ]; then \
		venv/Scripts/pip install -e .; \
	else \
		echo "❌ Could not find venv pip. Ensure Python venv was created."; exit 1; \
	fi
	@echo "✅ venv ready with MentorEval package installed!"

# Install MentorEval package in existing venv
install:
	@echo "📦 Installing MentorEval package..."
	@if [ -x "venv/bin/pip" ]; then \
		venv/bin/pip install -e .; \
	elif [ -x "venv/Scripts/pip.exe" ]; then \
		venv/Scripts/pip.exe install -e .; \
	elif [ -x "venv/Scripts/pip" ]; then \
		venv/Scripts/pip install -e .; \
	else \
		echo "❌ Could not find venv pip. Create venv first with: make venv"; exit 1; \
	fi
	@echo "✅ MentorEval package installed!"

# Install only requirements.txt (alternative approach)
install-requirements:
	@echo "📦 Installing requirements.txt..."
	@if [ -x "venv/bin/pip" ]; then \
		venv/bin/pip install -r requirements.txt; \
	elif [ -x "venv/Scripts/pip.exe" ]; then \
		venv/Scripts/pip.exe install -r requirements.txt; \
	elif [ -x "venv/Scripts/pip" ]; then \
		venv/Scripts/pip install -r requirements.txt; \
	else \
		echo "❌ Could not find venv pip. Create venv first with: make venv"; exit 1; \
	fi
	@echo "✅ Requirements installed!"

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
	@if [ -f "data/mentoreval.parquet" ]; then echo "✅ Combined dataset found"; else echo "❌ Combined dataset missing"; fi
	@if [ -d "data/processed/asap" ] && [ -f "data/processed/asap/asap_processed.parquet" ]; then echo "✅ ASAP processed files found"; else echo "❌ ASAP processed files missing"; fi
	@if [ -d "data/processed/asap2" ] && [ -f "data/processed/asap2/asap2_processed.parquet" ]; then echo "✅ ASAP2 processed files found"; else echo "❌ ASAP2 processed files missing"; fi
	@if [ -d "data/processed/mohler" ] && [ -f "data/processed/mohler/mohler_processed.parquet" ]; then echo "✅ Mohler processed files found"; else echo "❌ Mohler processed files missing"; fi
	@if [ -d "data/processed/ellipse" ] && [ -f "data/processed/ellipse/ellipse_processed.parquet" ]; then echo "✅ ELLIPSE processed files found"; else echo "❌ ELLIPSE processed files missing"; fi
	@if [ -d "data/processed/arasag" ] && [ -f "data/processed/arasag/arasag_processed.parquet" ]; then echo "✅ ARASAG processed files found"; else echo "❌ ARASAG processed files missing"; fi
	@if [ -d "data/processed/ptasag2018" ] && [ -f "data/processed/ptasag2018/ptasag2018_processed.parquet" ]; then echo "✅ PTASAG2018 processed files found"; else echo "❌ PTASAG2018 processed files missing"; fi

# Complete workflow: process datasets then test
workflow: dataset tests
	@echo "🎉 Complete workflow finished!"

# Complete dataset workflow: process and upload to HF
dataset-workflow: dataset upload-hf
	@echo "🎉 Dataset processing and upload workflow finished!"

# Development setup: install dependencies and run tests
dev: install tests
	@echo "🚀 Development environment ready!"

# Show dataset statistics
stats:
	@echo "📊 Dataset Statistics:"
	@if [ -f "data/mentoreval.parquet" ]; then \
		echo "Combined Dataset:"; \
		ROOT=$$(pwd); PY="python3"; \
		if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
		elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
		elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
		fi; \
		"$$PY" -c "import pandas as pd; df=pd.read_parquet('data/mentoreval.parquet'); print(f'  Total samples: {len(df):,}'); print(f'  Datasets: {df[\"dataset\"].nunique()}'); print(f'  Exercise sets: {df[\"exercise_set\"].nunique()}'); print('  By dataset:'); print(df['dataset'].value_counts().to_string())"; \
	else \
		echo "❌ Combined dataset not found. Run 'make dataset' first."; \
	fi

# CLI Commands
cli-list:
	@echo "📋 Listing all runs..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -m mentoreval --list

cli-summary:
	@echo "📊 Showing run summary..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -m mentoreval --summary

cli-execute:
	@if [ -z "$(N)" ]; then echo "❌ Usage: make cli-execute N=<run_id>"; exit 1; fi
	@echo "🚀 Executing run $(N)..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -m mentoreval --execute $(N)

cli-execute-all:
	@echo "🚀 Executing all unexecuted runs..."
	@ROOT=$$(pwd); PY="python3"; \
	if [ -x "$$ROOT/venv/bin/python" ]; then PY="$$ROOT/venv/bin/python"; \
	elif [ -x "$$ROOT/venv/Scripts/python.exe" ]; then PY="$$ROOT/venv/Scripts/python.exe"; \
	elif [ -x "$$ROOT/venv/Scripts/python" ]; then PY="$$ROOT/venv/Scripts/python"; \
	fi; \
	"$$PY" -m mentoreval --execute-all

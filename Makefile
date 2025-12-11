.PHONY: help tests venv venv-clean install install-requirements venv-activate venv-shell cli-list cli-summary cli-execute cli-execute-all dev workflow quick-test clean

tests:
	@echo "🧪 Running All JSONL Validation Tests..."
	@python3 tests/test_all.py

# Quick test for development
quick-test: test-basic
	@echo "🚀 Quick test completed!"

clean:
	@echo "🧹 Cleaning up build artifacts..."
	@rm -rf build/ dist/ *.egg-info

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
dev: install tests
	@echo "🚀 Development environment ready!"

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

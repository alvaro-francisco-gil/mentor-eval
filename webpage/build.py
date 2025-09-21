#!/usr/bin/env python3
"""
Build script for MentorEval Leaderboard
Automatically discovers result files and generates the JavaScript with the correct file list.
"""

import os
import json
import re
from pathlib import Path

def discover_result_files():
    """Discover all result files in the results directory."""
    results_dir = Path("../results")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return []
    
    result_files = []
    for file_path in results_dir.glob("*.json"):
        if file_path.is_file():
            result_files.append(file_path.name)
    
    # Sort by run ID for consistent ordering
    result_files.sort(key=lambda x: int(re.match(r'^(\d+)_', x).group(1)) if re.match(r'^(\d+)_', x) else 999)
    
    print(f"✅ Found {len(result_files)} result files:")
    for file in result_files:
        print(f"   - {file}")
    
    return result_files

def discover_run_files():
    """Discover all run configuration files."""
    runs_dir = Path("../runs")
    if not runs_dir.exists():
        print("❌ Runs directory not found!")
        return []
    
    run_files = []
    for file_path in runs_dir.glob("*.json"):
        if file_path.is_file() and not file_path.name.startswith("0_"):  # Skip template
            run_files.append(file_path.name)
    
    print(f"✅ Found {len(run_files)} run configuration files:")
    for file in run_files:
        print(f"   - {file}")
    
    return run_files

def generate_javascript(result_files, run_files):
    """Generate the JavaScript file with discovered files."""
    
    # Generate the getResultFiles function
    result_files_js = ",\n            ".join([f"'{file}'" for file in result_files])
    
    # Read the current script.js template
    script_path = Path("script.js")
    if not script_path.exists():
        print("❌ script.js not found!")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Replace only the result files list (simplified approach)
    script_content = re.sub(
        r'async getResultFiles\(\) \{\s*// Auto-generated list of result files\s*return \[\s*.*?\s*\];\s*\}',
        f'async getResultFiles() {{\n        // Auto-generated list of result files\n        return [\n            {result_files_js}\n        ];\n    }}',
        script_content,
        flags=re.DOTALL
    )
    
    # Note: Paths should remain as ../ for webpage folder serving
    # The deploy script will handle path conversion when copying to root
    
    # Write the updated script
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ Updated script.js with {len(result_files)} result files")
    return True

def main():
    """Main build function."""
    print("🔨 Building MentorEval Leaderboard...")
    print()
    
    # Change to webpage directory
    webpage_dir = Path(__file__).parent
    os.chdir(webpage_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Discover files
    result_files = discover_result_files()
    run_files = discover_run_files()
    
    if not result_files:
        print("❌ No result files found. Build failed.")
        return False
    
    # Generate JavaScript
    if generate_javascript(result_files, run_files):
        print()
        print("🎉 Build completed successfully!")
        print("📁 You can now serve the webpage with: python -m http.server 8000")
        print("📁 Or deploy with: python scripts/deploy_leaderboard_web.py")
        return True
    else:
        print("❌ Build failed.")
        return False

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MentorEval Benchmark Runner

This script checks for incomplete runs and executes them, or creates new runs.
"""

import os
import sys
import argparse
import json
from dotenv import load_dotenv
from mentoreval import MentorEvalBenchmark, MentorEvalConfig, RunManager, BenchmarkMode, PromptType


def main():
    """Main entry point for the benchmark runner."""
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run MentorEval benchmark")
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check for incomplete runs and execute them"
    )
    parser.add_argument(
        "--create", 
        type=str,
        help="Create a new run from a JSON file (e.g., '5_run.json')"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all runs"
    )
    parser.add_argument(
        "--summary", 
        action="store_true",
        help="Show run summary"
    )
    parser.add_argument(
        "--template", 
        action="store_true",
        help="Show template information"
    )
    
    args = parser.parse_args()
    
    # Initialize run manager
    run_manager = RunManager()
    
    # Handle special commands
    if args.list:
        list_runs(run_manager)
        return
    
    if args.summary:
        show_summary(run_manager)
        return
    
    if args.template:
        show_template()
        return
    
    if args.create:
        create_and_run(run_manager, args.create)
        return
    
    if args.check:
        check_and_run_incomplete(run_manager)
        return
    
    # Default: show help
    parser.print_help()


def check_and_run_incomplete(run_manager: RunManager):
    """Check for incomplete runs and execute them."""
    incomplete_runs = run_manager.get_incomplete_runs()
    
    if not incomplete_runs:
        print("✅ No incomplete runs found.")
        return
    
    print(f"🔍 Found {len(incomplete_runs)} incomplete runs:")
    for run in incomplete_runs:
        print(f"   Run {run.run_id}: {run.model_name} - {run.benchmark_mode} ({run.status})")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Please set your OpenAI API key:")
        print("   $env:OPENAI_API_KEY='your-api-key'")
        return
    
    # Execute incomplete runs
    for run in incomplete_runs:
        print(f"\n🚀 Executing Run {run.run_id}...")
        try:
            # Create config from run
            config = create_config_from_run(run)
            
            # Create benchmark and run
            benchmark = MentorEvalBenchmark(config)
            model = benchmark.create_model()
            
            # Update status to running
            run_manager.update_run_status(run.run_id, 'running')
            
            # Execute benchmark
            results = benchmark.evaluate(model)
            
            print(f"✅ Run {run.run_id} completed successfully!")
            print(f"   Overall NMAE: {results.get('nmae', {}).get('normalized_value', 0.0):.3f}")
            print(f"   Overall NRMSE: {results.get('nrmse', {}).get('normalized_value', 0.0):.3f}")
            
        except Exception as e:
            print(f"❌ Run {run.run_id} failed: {e}")
            run_manager.update_run_status(run.run_id, 'failed')


def create_and_run(run_manager: RunManager, run_file: str):
    """Create a new run from a JSON file and execute it."""
    # Check if file exists
    if not os.path.exists(run_file):
        print(f"❌ Run file not found: {run_file}")
        return
    
    # Load run configuration
    try:
        with open(run_file, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading run file: {e}")
        return
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OpenAI API key:")
        print("   $env:OPENAI_API_KEY='your-api-key'")
        return
    
    # Create new run
    config = create_config_from_run_data(run_data)
    run_info = run_manager.create_run(config)
    
    print(f"🚀 Created Run {run_info.run_id} from {run_file}")
    print(f"   Model: {run_info.model_name}")
    print(f"   Mode: {run_info.benchmark_mode}")
    
    try:
        # Execute benchmark
        benchmark = MentorEvalBenchmark(config)
        model = benchmark.create_model()
        
        results = benchmark.evaluate(model)
        
        print(f"✅ Run {run_info.run_id} completed successfully!")
        print(f"   Overall NMAE: {results.get('nmae', {}).get('normalized_value', 0.0):.3f}")
        print(f"   Overall NRMSE: {results.get('nrmse', {}).get('normalized_value', 0.0):.3f}")
        
    except Exception as e:
        print(f"❌ Run {run_info.run_id} failed: {e}")


def create_config_from_run(run_info):
    """Create MentorEvalConfig from RunInfo."""
    return create_config_from_run_data(run_info.configuration)


def create_config_from_run_data(config_data):
    """Create MentorEvalConfig from configuration data."""
    # Handle mode
    mode_str = config_data.get('mode', 'mentoreval-test')
    if mode_str == 'mentoreval':
        mode = BenchmarkMode.MENTOREVAL
    else:
        mode = BenchmarkMode.MENTOREVAL_TEST
    
    # Handle prompt type
    prompt_type_str = config_data.get('prompt_type', 'with_explanation')
    if prompt_type_str == 'grade_only':
        prompt_type = PromptType.GRADE_ONLY
    else:
        prompt_type = PromptType.WITH_EXPLANATION
    
    # Create config
    config = MentorEvalConfig(
        mode=mode,
        use_few_shot=config_data.get('use_few_shot', True),
        include_rubric=config_data.get('include_rubric', True),
        prompt_type=prompt_type,
        n_test_samples=config_data.get('n_test_samples'),
        model_name=config_data.get('model_name', 'gpt-4o-mini'),
        model_provider=config_data.get('model_provider', 'openai')
    )
    
    # Set verbose if specified
    if config_data.get('verbose', False):
        config.verbose = True
    
    return config


def list_runs(run_manager: RunManager):
    """List all runs."""
    runs = run_manager.list_runs()
    
    if not runs:
        print("No runs found.")
        return
    
    print(f"\n📋 All Runs ({len(runs)} total):")
    print("-" * 60)
    print(f"{'ID':<4} {'Model':<20} {'Mode':<15} {'Status':<10}")
    print("-" * 60)
    
    for run in runs:
        print(f"{run.run_id:<4} {run.model_name:<20} {run.benchmark_mode:<15} {run.status:<10}")


def show_summary(run_manager: RunManager):
    """Show run summary statistics."""
    summary = run_manager.get_run_summary()
    
    print(f"\n📊 Run Summary:")
    print(f"   Total runs: {summary['total_runs']}")
    print(f"   Latest run ID: {summary['latest_run_id']}")
    
    if summary['status_counts']:
        print(f"   Status counts:")
        for status, count in summary['status_counts'].items():
            print(f"     {status}: {count}")
    
    # Show incomplete runs
    incomplete = run_manager.get_incomplete_runs()
    if incomplete:
        print(f"\n   Incomplete runs: {len(incomplete)}")
        for run in incomplete:
            print(f"     {run.run_id}: {run.model_name} - {run.benchmark_mode} ({run.status})")


def show_template():
    """Show template information."""
    template_file = "runs/0_run_template.json"
    
    if not os.path.exists(template_file):
        print("❌ Template file not found: runs/0_run_template.json")
        return
    
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        print("📝 MentorEval Run Template")
        print("=" * 30)
        print(f"Template file: {template_file}")
        print("\nAvailable options:")
        
        if '_available_options' in template_data:
            for key, values in template_data['_available_options'].items():
                print(f"  {key}: {values}")
        
        print("\nBenchmark modes:")
        if '_benchmark_modes' in template_data:
            for mode, description in template_data['_benchmark_modes'].items():
                print(f"  {mode}: {description}")
        
        print("\nPrompt types:")
        if '_prompt_types' in template_data:
            for prompt_type, description in template_data['_prompt_types'].items():
                print(f"  {prompt_type}: {description}")
        
        print(f"\nTo create a new run:")
        print(f"  1. Copy {template_file} to a new file (e.g., 5_run.json)")
        print(f"  2. Modify the configuration as needed")
        print(f"  3. Run: python run_benchmark.py --create 5_run.json")
        
    except Exception as e:
        print(f"❌ Error reading template: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MentorEval CLI - Simple command-line interface for running evaluations.

This script provides a clean interface to the MentorEval run management system.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mentoreval.run_manager import RunManager


def main():
    """Main entry point for the MentorEval CLI."""
    parser = argparse.ArgumentParser(description="MentorEval CLI - Run evaluations")
    parser.add_argument(
        "--execute", 
        type=int,
        help="Execute a specific run by ID"
    )
    parser.add_argument(
        "--execute-all", 
        action="store_true",
        help="Execute all unexecuted runs"
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
        "--status",
        type=int,
        help="Show status of a specific run"
    )
    parser.add_argument(
        "--reset",
        type=int,
        help="Reset a run status to pending (allows rerunning)"
    )
    
    args = parser.parse_args()
    
    # Initialize run manager
    run_manager = RunManager()
    
    # Handle commands
    if args.execute:
        execute_specific_run(run_manager, args.execute)
    elif args.execute_all:
        execute_all_runs(run_manager)
    elif args.list:
        list_runs(run_manager)
    elif args.summary:
        show_summary(run_manager)
    elif args.status:
        show_run_status(run_manager, args.status)
    elif args.reset:
        reset_run_status(run_manager, args.reset)
    else:
        parser.print_help()


def execute_specific_run(run_manager: RunManager, run_id: int):
    """Execute a specific run."""
    print(f"🚀 Executing Run {run_id}...")
    try:
        result = run_manager.execute_run(run_id)
        if result["status"] == "completed":
            print(f"✅ Run {run_id} completed successfully!")
        elif result["status"] == "already_completed":
            print(f"⚠️ Run {run_id} was already completed.")
        else:
            print(f"❌ Run {run_id} failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error executing run {run_id}: {e}")


def execute_all_runs(run_manager: RunManager):
    """Execute all unexecuted runs."""
    print("🚀 Executing all unexecuted runs...")
    try:
        results = run_manager.execute_unexecuted_runs()
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        print(f"\n📊 Execution Summary:")
        print(f"   Completed: {completed}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(results)}")
    except Exception as e:
        print(f"❌ Error executing runs: {e}")


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


def show_run_status(run_manager: RunManager, run_id: int):
    """Show status of a specific run."""
    run_info = run_manager.get_run_info(run_id)
    if not run_info:
        print(f"❌ Run {run_id} not found.")
        return
    
    print(f"\n📋 Run {run_id} Status:")
    print(f"   Model: {run_info.model_name}")
    print(f"   Mode: {run_info.benchmark_mode}")
    print(f"   Status: {run_info.status}")
    print(f"   Created: {run_info.configuration.get('timestamp', 'Unknown')}")


def reset_run_status(run_manager: RunManager, run_id: int):
    """Reset a run status to pending (allows rerunning)."""
    run_info = run_manager.get_run_info(run_id)
    if not run_info:
        print(f"❌ Run {run_id} not found.")
        return
    
    if run_info.status == "pending":
        print(f"⚠️ Run {run_id} is already pending. No change needed.")
        return
    
    print(f"🔄 Resetting run {run_id} status from '{run_info.status}' to 'pending'...")
    run_manager.update_run_status(run_id, "pending")
    print(f"✅ Run {run_id} status reset to pending. You can now rerun it with: python -m mentoreval --execute {run_id}")


if __name__ == "__main__":
    main()

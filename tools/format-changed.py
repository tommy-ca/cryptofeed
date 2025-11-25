#!/usr/bin/env python3
"""
Scoped formatting tool for Cryptofeed.

Runs code formatting (ruff format + isort) only on Python files that have been changed,
avoiding unnecessary formatting of the entire codebase.

Usage:
    python tools/format-changed.py [--staged|--unstaged|--all] [--dry-run]

Options:
    --staged    Format only staged changes (default)
    --unstaged  Format only unstaged changes
    --all       Format all changes (staged + unstaged)
    --dry-run   Show what would be formatted without making changes
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def get_changed_files(change_type: str = "staged") -> List[str]:
    """Get list of changed Python files based on change type."""
    if change_type == "staged":
        cmd = ["git", "diff", "--cached", "--name-only", "--", "*.py"]
    elif change_type == "unstaged":
        cmd = ["git", "diff", "--name-only", "--", "*.py"]
    elif change_type == "all":
        cmd = ["git", "diff", "HEAD", "--name-only", "--", "*.py"]
    else:
        raise ValueError(f"Invalid change_type: {change_type}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split("\n")
        # Filter out empty strings and ensure files exist
        return [f for f in files if f and Path(f).exists()]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}")
        return []


def run_formatter(
    files: List[str], formatter_cmd: List[str], dry_run: bool = False
) -> bool:
    """Run a formatter on the specified files."""
    if not files:
        print(f"No files to format with {' '.join(formatter_cmd)}")
        return True

    if dry_run:
        print(f"Would run: {' '.join(formatter_cmd)} {' '.join(files)}")
        return True

    try:
        cmd = formatter_cmd + files
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(formatter_cmd)}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Format only changed Python files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--staged",
        action="store_true",
        default=True,
        help="Format only staged changes (default)",
    )
    group.add_argument(
        "--unstaged", action="store_true", help="Format only unstaged changes"
    )
    group.add_argument(
        "--all", action="store_true", help="Format all changes (staged + unstaged)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be formatted without making changes",
    )

    args = parser.parse_args()

    # Determine change type
    if args.unstaged:
        change_type = "unstaged"
    elif args.all:
        change_type = "all"
    else:
        change_type = "staged"

    # Get changed files
    changed_files = get_changed_files(change_type)
    if not changed_files:
        print(f"No changed Python files found for {change_type} changes.")
        return 0

    print(f"Found {len(changed_files)} changed Python files:")
    for file in changed_files:
        print(f"  {file}")

    if args.dry_run:
        print("\nDRY RUN - Would format with:")
        print("  ruff format")
        print("  isort")
        return 0

    # Run formatters
    success = True

    # Run ruff format
    if not run_formatter(changed_files, ["ruff", "format"], args.dry_run):
        success = False

    # Run isort
    if not run_formatter(changed_files, ["isort", "--jobs", "8"], args.dry_run):
        success = False

    if success:
        print("\n✅ Formatting completed successfully!")
        if not args.dry_run:
            print("Remember to stage your formatting changes with: git add <files>")
    else:
        print("\n❌ Formatting failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

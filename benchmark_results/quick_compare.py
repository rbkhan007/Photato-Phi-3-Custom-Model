#!/usr/bin/env python
"""
Quick Benchmark Comparison

Usage:
    python benchmark_results/quick_compare.py
    python benchmark_results/quick_compare.py --models phi4-mini
"""

import os
import sys
import pandas as pd

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def quick_compare():
    """Run a quick comparison of available results."""
    print("\n" + "=" * 60)
    print("  Quick Benchmark Comparison")
    print("=" * 60)

    # Check for existing results
    task_file = os.path.join(OUTPUT_DIR, "task_scores.csv")
    cat_file = os.path.join(OUTPUT_DIR, "category_scores.csv")

    if os.path.exists(task_file):
        print("\n--- Task Scores ---")
        df = pd.read_csv(task_file, index_col=0)
        print(df.round(1).to_string())
    else:
        print("\nNo task_scores.csv found. Run compare_models.py first.")

    if os.path.exists(cat_file):
        print("\n--- Category Scores ---")
        df = pd.read_csv(cat_file, index_col=0)
        print(df.round(1).to_string())
    else:
        print("\nNo category_scores.csv found. Run compare_models.py first.")

    # Check for reports
    md_file = os.path.join(OUTPUT_DIR, "comparison_report.md")
    if os.path.exists(md_file):
        print(f"\nFull report: {md_file}")

    json_file = os.path.join(OUTPUT_DIR, "comparison_report.json")
    if os.path.exists(json_file):
        print(f"JSON report: {json_file}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    quick_compare()

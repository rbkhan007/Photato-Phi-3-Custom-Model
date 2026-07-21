"""Benchmark Results package — comparison reports, export, and analysis."""

from .config import MODELS_TO_COMPARE, BENCHMARK_CONFIG, CATEGORIES, HARDWARE_TIERS
from .compare_models import (
    load_all_results,
    load_all_answers,
    compute_task_scores,
    compute_category_scores,
    compute_token_usage,
    compute_category_usage,
    generate_latex_table,
    generate_markdown_report,
    generate_json_report,
    print_comparison_table,
    run_comparison,
)
from .quick_compare import quick_compare

__all__ = [
    "MODELS_TO_COMPARE",
    "BENCHMARK_CONFIG",
    "CATEGORIES",
    "HARDWARE_TIERS",
    "load_all_results",
    "load_all_answers",
    "compute_task_scores",
    "compute_category_scores",
    "compute_token_usage",
    "compute_category_usage",
    "generate_latex_table",
    "generate_markdown_report",
    "generate_json_report",
    "print_comparison_table",
    "run_comparison",
    "quick_compare",
]

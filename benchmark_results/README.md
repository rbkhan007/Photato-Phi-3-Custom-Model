# Benchmark Results

This directory contains benchmark comparison results for models evaluated using LiveBench.

## Files

| File | Description |
|------|-------------|
| `compare_models.py` | Main comparison tool - generates all reports |
| `quick_compare.py` | Quick view of existing results |
| `config.py` | Benchmark configuration (models, tasks, settings) |
| `task_scores.csv` | Scores by task (math, code, writing, etc.) |
| `category_scores.csv` | Scores by category (reasoning, language, etc.) |
| `token_usage.csv` | Token usage by task |
| `category_usage.csv` | Token usage by category |
| `comparison_report.md` | Full Markdown comparison report |
| `comparison_report.json` | Machine-readable JSON report |
| `task_scores.tex` | LaTeX table for task scores |
| `category_scores.tex` | LaTeX table for category scores |

## Latest Results

### Category Scores

| Model | agentic | average | knowledge | language | reasoning |
|---|---|---|---|---|---|
| phi4-mini | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Usage

```bash
# Generate comparison reports
python benchmark_results/compare_models.py

# Quick view of existing results
python benchmark_results/quick_compare.py
```

## Adding New Models

1. Run benchmark:
   ```bash
   python show_livebench_result.py --model-list <model-name> --run-benchmark
   ```
2. Regenerate reports:
   ```bash
   python _gen_benchmark_results.py
   ```

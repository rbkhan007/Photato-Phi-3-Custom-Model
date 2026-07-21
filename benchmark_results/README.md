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

## Usage

### Generate Comparison Reports

```bash
# Run full comparison (generates all files)
python benchmark_results/compare_models.py

# Compare specific models
python benchmark_results/compare_models.py --models phi4-mini

# Generate LaTeX tables
python benchmark_results/compare_models.py --generate-latex

# Quick view of existing results
python benchmark_results/quick_compare.py
```

### Add New Models

1. Run benchmark for the new model:
   ```bash
   python show_livebench_result.py --model-list <model-name> --run-benchmark
   ```

2. Run comparison to include all models:
   ```bash
   python benchmark_results/compare_models.py
   ```

### Export for Papers

```bash
# Generate LaTeX tables
python benchmark_results/compare_models.py --generate-latex

# Include in LaTeX document:
# \input{benchmark_results/task_scores.tex}
# \input{benchmark_results/category_scores.tex}
```

## Model Comparison Template

| Model | Params | Size | Speed | Math | Code | Writing | Avg |
|-------|--------|------|-------|------|------|---------|-----|
| Phi-4 Mini 3.8B | 3.8B | 2.5 GB | 5.8 tok/s | 100 | 100 | 100 | 100.0 |
| _Your Model_ | _X.XB_ | _X.X GB_ | _X tok/s_ | _XX_ | _XX_ | _XX_ | _XX.X_ |

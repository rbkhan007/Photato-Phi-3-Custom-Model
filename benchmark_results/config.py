"""
Benchmark Configuration

Define models, tasks, and comparison settings.
"""

# Models to benchmark
MODELS_TO_COMPARE = [
    "phi4-mini",
    # Add more models here as you benchmark them:
    # "llama3.2-3b",
    # "mistral-7b",
    # "deepseek-coder-1.3b",
]

# Benchmark settings
BENCHMARK_CONFIG = {
    "release": "2025-07-07",
    "questions_per_task": 10,
    "max_tokens": 256,
    "temperature": 0.3,
    "cpu_percent": 55.0,
    "n_gpu_layers": 0,
}

# Categories and tasks
CATEGORIES = {
    "reasoning": {
        "tasks": ["math", "logic", "code"],
        "weight": 1.0,
    },
    "language": {
        "tasks": ["writing", "extraction", "summarization"],
        "weight": 1.0,
    },
    "knowledge": {
        "tasks": ["science", "history", "geography"],
        "weight": 1.0,
    },
    "safety": {
        "tasks": ["refusal", "harmfulness"],
        "weight": 1.0,
    },
    "agentic": {
        "tasks": ["tool_use", "multi_step"],
        "weight": 1.0,
    },
}

# Output settings
OUTPUT_FORMATS = ["csv", "markdown", "json", "latex"]

# Hardware tiers for performance comparison
HARDWARE_TIERS = {
    "low": {"ram_gb": 4, "gpu": False, "expected_speed": "5-10 tok/s"},
    "medium": {"ram_gb": 8, "gpu": False, "expected_speed": "15-25 tok/s"},
    "high": {"ram_gb": 16, "gpu": False, "expected_speed": "30-50 tok/s"},
    "gpu_low": {"ram_gb": 8, "gpu": True, "gpu_gb": 4, "expected_speed": "60-100 tok/s"},
    "gpu_high": {"ram_gb": 16, "gpu": True, "gpu_gb": 8, "expected_speed": "40-70 tok/s"},
}

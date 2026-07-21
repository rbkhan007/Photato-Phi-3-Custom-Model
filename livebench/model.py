"""LiveBench model configurations."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    display_name: str
    model_path: Optional[str] = None
    backend: str = "llamacpp"
    parameters: dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Model registry - resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "phi4-mini": ModelConfig(
        name="phi4-mini",
        display_name="phi4-mini",
        model_path=os.path.join(BASE_DIR, "notebooks", "Phi-4-mini-instruct-Q4_K_M.gguf"),
        backend="llamacpp",
        parameters={"n_gpu_layers": 0, "cpu_percent": 55.0},
    ),
    "qwen3-embedding": ModelConfig(
        name="qwen3-embedding",
        display_name="qwen3-embedding",
        model_path=os.path.join(BASE_DIR, "notebooks", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        backend="llamacpp",
        parameters={"embedding": True},
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    if model_name in MODELS:
        return MODELS[model_name]
    # Return a default config if not found
    return ModelConfig(
        name=model_name,
        display_name=model_name,
    )

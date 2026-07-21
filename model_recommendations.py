"""
Model Recommendations for Low-Resource Hardware.

Auto-detects system capabilities and recommends optimal models,
quantization levels, and configurations for any hardware tier.

Features:
- Hardware detection (CPU, RAM, GPU, VRAM)
- Model size recommendations
- Quantization level guidance
- Pendrive deployment optimization
- Performance expectations
"""

import argparse
import os
import sys
import json
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


class HardwareTier(Enum):
    """Hardware tiers for model selection."""
    ULTRA_LOW = "ultra_low"      # <4GB RAM, no GPU
    LOW = "low"                  # 4-8GB RAM, no GPU
    MEDIUM = "medium"            # 8-16GB RAM, optional GPU
    HIGH = "high"                # 16-32GB RAM, dedicated GPU
    ULTRA_HIGH = "ultra_high"    # 32GB+ RAM, high-end GPU


class QuantLevel(Enum):
    """Quantization levels with quality/size tradeoffs."""
    Q2_K = "Q2_K"          # ~1.5GB, lowest quality, fastest
    Q3_K_S = "Q3_K_S"      # ~1.7GB, very low quality
    Q3_K_M = "Q3_K_M"      # ~2.0GB, low quality
    Q3_K_L = "Q3_K_L"      # ~2.2GB, low-medium quality
    Q4_0 = "Q4_0"          # ~2.1GB, basic 4-bit
    Q4_K_S = "Q4_K_S"      # ~2.3GB, medium quality
    Q4_K_M = "Q4_K_M"      # ~2.5GB, good balance (RECOMMENDED)
    Q5_0 = "Q5_0"          # ~2.7GB, good quality
    Q5_K_S = "Q5_K_S"      # ~2.9GB, better quality
    Q5_K_M = "Q5_K_M"      # ~3.0GB, high quality
    Q6_K = "Q6_K"          # ~3.5GB, great quality
    Q8_0 = "Q8_0"          # ~4.0GB, near-original quality
    F16 = "F16"            # ~7.6GB, full precision


@dataclass
class SystemInfo:
    """Detected system information."""
    os: str = ""
    arch: str = ""
    python_version: str = ""
    cpu_count: int = 0
    cpu_name: str = ""
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_driver: str = ""
    cuda_available: bool = False
    storage_available_gb: float = 0.0
    hardware_tier: HardwareTier = HardwareTier.LOW


@dataclass
class ModelRecommendation:
    """A specific model recommendation."""
    model_name: str
    model_id: str
    parameters: str
    context_length: int
    quant_level: QuantLevel
    quant_size_gb: float
    ram_required_gb: float
    vram_required_gb: float
    speed_tokens_per_sec: float
    quality_score: float  # 0-100
    use_case: str
    download_url: str
    notes: str = ""


@dataclass
class HardwareProfile:
    """Complete hardware analysis and recommendations."""
    system_info: SystemInfo
    recommendations: list[ModelRecommendation]
    pendrive_recommendations: list[ModelRecommendation]
    warnings: list[str]
    tips: list[str]


class HardwareDetector:
    """Detect system hardware capabilities."""

    @staticmethod
    def detect() -> SystemInfo:
        """Detect all system hardware."""
        info = SystemInfo()
        info.os = platform.system()
        info.arch = platform.machine()
        info.python_version = platform.python_version()
        info.cpu_count = os.cpu_count() or 1
        info.cpu_name = platform.processor() or "Unknown"

        # Detect RAM
        info.ram_gb, info.ram_available_gb = HardwareDetector._detect_ram()

        # Detect GPU
        gpu_info = HardwareDetector._detect_gpu()
        info.gpu_name = gpu_info.get("name", "None")
        info.gpu_vram_gb = gpu_info.get("vram_gb", 0.0)
        info.gpu_driver = gpu_info.get("driver", "")
        info.cuda_available = gpu_info.get("cuda", False)

        # Detect storage
        info.storage_available_gb = HardwareDetector._detect_storage()

        # Classify hardware tier
        info.hardware_tier = HardwareDetector._classify_tier(info)

        return info

    @staticmethod
    def _detect_ram() -> tuple[float, float]:
        """Detect total and available RAM."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return round(mem.total / (1024**3), 2), round(mem.available / (1024**3), 2)
        except ImportError:
            pass

        # Windows fallback
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "memorychip", "get", "capacity"],
                    capture_output=True, text=True, timeout=5
                )
                total_kb = sum(
                    int(line.strip()) for line in result.stdout.split("\n")
                    if line.strip().isdigit()
                )
                return round(total_kb / (1024**3), 2), round(total_kb / (1024**3) * 0.5, 2)
            except Exception:
                pass

        # Linux fallback
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024**3), 2), round(kb / (1024**3) * 0.5, 2)
        except Exception:
            pass

        return 8.0, 4.0  # Default assumption

    @staticmethod
    def _detect_gpu() -> dict:
        """Detect GPU and VRAM."""
        info = {"name": "None", "vram_gb": 0.0, "driver": "", "cuda": False}

        # Try NVIDIA SMI
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                info["name"] = parts[0].strip()
                info["driver"] = parts[2].strip() if len(parts) > 2 else ""
                vram_mb = int(parts[1].strip().replace(" MiB", "").replace(",", ""))
                info["vram_gb"] = round(vram_mb / 1024, 2)
                info["cuda"] = True
                return info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try rocm-smi (AMD)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Total Memory" in line:
                        vram_kb = int(line.split(":")[1].strip().split()[0])
                        info["vram_gb"] = round(vram_kb / (1024**3), 2)
                        info["name"] = "AMD GPU"
                        return info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info["name"] = "Apple Silicon (Unified Memory)"
            info["cuda"] = False
            ram_gb, _ = HardwareDetector._detect_ram()
            info["vram_gb"] = ram_gb  # Unified memory
            return info

        return info

    @staticmethod
    def _detect_storage() -> float:
        """Detect available storage in GB."""
        try:
            import psutil
            disk = psutil.disk_usage("/")
            return round(disk.free / (1024**3), 2)
        except ImportError:
            pass

        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "logicaldisk", "get", "freespace,size"],
                    capture_output=True, text=True, timeout=5
                )
                lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 1:
                        return round(int(parts[0]) / (1024**3), 2)
            elif platform.system() == "Linux":
                result = subprocess.run(["df", "-BG", "/"], capture_output=True, text=True, timeout=5)
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    return float(parts[3].replace("G", ""))
        except Exception:
            pass

        return 50.0  # Default

    @staticmethod
    def _classify_tier(info: SystemInfo) -> HardwareTier:
        """Classify hardware into a tier."""
        ram = info.ram_gb
        vram = info.gpu_vram_gb

        if ram < 4 and vram == 0:
            return HardwareTier.ULTRA_LOW
        elif ram < 8 and vram == 0:
            return HardwareTier.LOW
        elif ram < 16 or vram < 4:
            return HardwareTier.MEDIUM
        elif ram < 32 and vram < 12:
            return HardwareTier.HIGH
        else:
            return HardwareTier.ULTRA_HIGH


class ModelDatabase:
    """Database of recommended models for low-resource hardware."""

    MODELS = {
        # === PHI-3 FAMILY (Primary recommendation) ===
        "phi-3-mini-3.8b": {
            "name": "Phi-3 Mini 3.8B",
            "id": "microsoft/Phi-3-mini-4k-instruct",
            "parameters": "3.8B",
            "context_4k": 4096,
            "context_128k": 131072,
            "base_size_gb": 7.6,
            "license": "MIT",
            "strengths": ["Code generation", "Math", "Reasoning", "Instruction following"],
            "best_for": "General-purpose coding and reasoning",
        },
        "phi-3-mini-128k": {
            "name": "Phi-3 Mini 128k",
            "id": "microsoft/Phi-3-mini-128k-instruct",
            "parameters": "3.8B",
            "context_4k": 4096,
            "context_128k": 131072,
            "base_size_gb": 7.6,
            "license": "MIT",
            "strengths": ["Long context", "Code generation", "Math"],
            "best_for": "Long document analysis, large codebases",
        },

        # === TINY MODELS (Ultra-low resource) ===
        "qwen2-0.5b": {
            "name": "Qwen2 0.5B",
            "id": "Qwen/Qwen2-0.5B-Instruct",
            "parameters": "0.5B",
            "context_4k": 4096,
            "context_128k": 32768,
            "base_size_gb": 1.0,
            "license": "Apache 2.0",
            "strengths": ["Tiny", "Fast", "Multilingual"],
            "best_for": "Ultra-low resource devices, edge deployment",
        },
        "qwen2-1.5b": {
            "name": "Qwen2 1.5B",
            "id": "Qwen/Qwen2-1.5B-Instruct",
            "parameters": "1.5B",
            "context_4k": 4096,
            "context_128k": 32768,
            "base_size_gb": 3.0,
            "license": "Apache 2.0",
            "strengths": ["Small", "Good quality", "Multilingual"],
            "best_for": "Low resource devices, mobile",
        },

        # === SMALL MODELS (Low resource) ===
        "llama-3.2-1b": {
            "name": "Llama 3.2 1B",
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "parameters": "1B",
            "context_4k": 4096,
            "context_128k": 131072,
            "base_size_gb": 2.0,
            "license": "Llama 3.2",
            "strengths": ["Very small", "Fast", "Good instruction following"],
            "best_for": "Mobile, edge, quick responses",
        },
        "llama-3.2-3b": {
            "name": "Llama 3.2 3B",
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "parameters": "3B",
            "context_4k": 4096,
            "context_128k": 131072,
            "base_size_gb": 6.0,
            "license": "Llama 3.2",
            "strengths": ["Small but capable", "Fast", "Code generation"],
            "best_for": "Low-medium resource devices",
        },
        "gemma-2-2b": {
            "name": "Gemma 2 2B",
            "id": "google/gemma-2-2b-it",
            "parameters": "2B",
            "context_4k": 8192,
            "context_128k": 8192,
            "base_size_gb": 4.0,
            "license": "Gemma",
            "strengths": ["Google quality", "Fast", "Good at instruction following"],
            "best_for": "Mobile, quick tasks",
        },

        # === MEDIUM MODELS (Medium resource) ===
        "llama-3.1-8b": {
            "name": "Llama 3.1 8B",
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "parameters": "8B",
            "context_4k": 8192,
            "context_128k": 131072,
            "base_size_gb": 16.0,
            "license": "Llama 3.1",
            "strengths": ["Excellent quality", "Code generation", "Long context"],
            "best_for": "General-purpose, coding, analysis",
        },
        "mistral-7b": {
            "name": "Mistral 7B",
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "parameters": "7B",
            "context_4k": 32768,
            "context_128k": 32768,
            "base_size_gb": 14.0,
            "license": "Apache 2.0",
            "strengths": ["Efficient", "Fast", "Good at code"],
            "best_for": "Code generation, general tasks",
        },
        "qwen2-7b": {
            "name": "Qwen2 7B",
            "id": "Qwen/Qwen2-7B-Instruct",
            "parameters": "7B",
            "context_4k": 32768,
            "context_128k": 131072,
            "base_size_gb": 14.0,
            "license": "Apache 2.0",
            "strengths": ["Multilingual", "Code", "Math", "Long context"],
            "best_for": "Multilingual tasks, coding",
        },

        # === SPECIALIZED (Code-focused) ===
        "deepseek-coder-1.3b": {
            "name": "DeepSeek Coder 1.3B",
            "id": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "parameters": "1.3B",
            "context_4k": 4096,
            "context_128k": 16384,
            "base_size_gb": 2.6,
            "license": "DeepSeek",
            "strengths": ["Code-specialized", "Small", "Fast"],
            "best_for": "Code completion, quick code tasks",
        },
        "deepseek-coder-6.7b": {
            "name": "DeepSeek Coder 6.7B",
            "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "parameters": "6.7B",
            "context_4k": 4096,
            "context_128k": 16384,
            "base_size_gb": 13.4,
            "license": "DeepSeek",
            "strengths": ["Excellent code generation", "Multi-language", "Fill-in-middle"],
            "best_for": "Code generation, refactoring",
        },
        "codellama-7b": {
            "name": "CodeLlama 7B",
            "id": "codellama/CodeLlama-7b-Instruct-hf",
            "parameters": "7B",
            "context_4k": 4096,
            "context_128k": 16384,
            "base_size_gb": 14.0,
            "license": "Llama 2",
            "strengths": ["Code-specialized", "Fill-in-middle", "Infilling"],
            "best_for": "Code completion, code editing",
        },
    }

    @staticmethod
    def get_model(model_key: str) -> dict:
        """Get model info by key."""
        return ModelDatabase.MODELS.get(model_key, {})

    @staticmethod
    def list_models() -> list[dict]:
        """List all available models."""
        return [
            {"key": k, **v}
            for k, v in ModelDatabase.MODELS.items()
        ]


class ModelRecommender:
    """Recommend optimal models based on hardware."""

    # Performance estimates (tokens/sec) for different quant levels on CPU
    CPU_SPEED_ESTIMATES = {
        HardwareTier.ULTRA_LOW: {"q2": 5, "q3": 3, "q4": 2, "q5": 1, "q6": 0.5, "q8": 0.3},
        HardwareTier.LOW: {"q2": 15, "q3": 10, "q4": 7, "q5": 4, "q6": 2, "q8": 1},
        HardwareTier.MEDIUM: {"q2": 30, "q3": 22, "q4": 16, "q5": 10, "q6": 6, "q8": 3},
        HardwareTier.HIGH: {"q2": 60, "q3": 45, "q4": 35, "q5": 25, "q6": 15, "q8": 8},
        HardwareTier.ULTRA_HIGH: {"q2": 100, "q3": 80, "q4": 60, "q5": 45, "q6": 30, "q8": 15},
    }

    # VRAM requirements by quant level (for GPU inference)
    VRAM_REQUIREMENTS = {
        QuantLevel.Q2_K: 1.5,
        QuantLevel.Q3_K_S: 1.7,
        QuantLevel.Q3_K_M: 2.0,
        QuantLevel.Q3_K_L: 2.2,
        QuantLevel.Q4_0: 2.1,
        QuantLevel.Q4_K_S: 2.3,
        QuantLevel.Q4_K_M: 2.5,
        QuantLevel.Q5_0: 2.7,
        QuantLevel.Q5_K_S: 2.9,
        QuantLevel.Q5_K_M: 3.0,
        QuantLevel.Q6_K: 3.5,
        QuantLevel.Q8_0: 4.0,
        QuantLevel.F16: 7.6,
    }

    # RAM requirements (model + overhead)
    RAM_MULTIPLIER = 1.4  # Model size * this = RAM needed

    @staticmethod
    def recommend(
        system_info: SystemInfo,
        use_case: str = "general",
        prefer_quality: bool = False,
        max_model_size_gb: Optional[float] = None,
    ) -> HardwareProfile:
        """Generate complete recommendations for the system."""
        recommendations = []
        pendrive_recommendations = []
        warnings = []
        tips = []

        tier = system_info.hardware_tier
        ram = system_info.ram_gb
        vram = system_info.gpu_vram_gb

        # === Model Selection Strategy ===

        if tier == HardwareTier.ULTRA_LOW:
            # <4GB RAM: Only tiny models
            candidates = [
                ("qwen2-0.5b", QuantLevel.Q4_K_M),
                ("qwen2-0.5b", QuantLevel.Q3_K_M),
                ("llama-3.2-1b", QuantLevel.Q3_K_M),
                ("deepseek-coder-1.3b", QuantLevel.Q4_K_M),
            ]
            warnings.append("⚠️  Very limited RAM. Only tiny models recommended.")
            tips.append("💡 Use Q4_K_M or lower quantization for fastest inference.")
            tips.append("💡 Consider using the cloud GPU for heavy tasks.")

        elif tier == HardwareTier.LOW:
            # 4-8GB RAM: Small models
            candidates = [
                ("phi-3-mini-3.8b", QuantLevel.Q3_K_M),
                ("phi-3-mini-3.8b", QuantLevel.Q4_K_M),
                ("llama-3.2-3b", QuantLevel.Q4_K_M),
                ("gemma-2-2b", QuantLevel.Q4_K_M),
                ("qwen2-1.5b", QuantLevel.Q5_K_M),
                ("deepseek-coder-1.3b", QuantLevel.Q5_K_M),
            ]
            tips.append("💡 Q4_K_M is the sweet spot for quality vs speed.")
            tips.append("💡 Close other applications to free up RAM.")

        elif tier == HardwareTier.MEDIUM:
            # 8-16GB RAM: Medium models
            candidates = [
                ("phi-3-mini-3.8b", QuantLevel.Q4_K_M),
                ("phi-3-mini-3.8b", QuantLevel.Q5_K_M),
                ("llama-3.2-3b", QuantLevel.Q6_K),
                ("mistral-7b", QuantLevel.Q3_K_M),
                ("mistral-7b", QuantLevel.Q4_K_M),
                ("qwen2-7b", QuantLevel.Q3_K_M),
                ("deepseek-coder-6.7b", QuantLevel.Q3_K_M),
            ]
            if vram >= 4:
                candidates.append(("llama-3.1-8b", QuantLevel.Q4_K_M))
            tips.append("💡 Use GPU inference if available for 2-5x speedup.")

        elif tier == HardwareTier.HIGH:
            # 16-32GB RAM: Larger models
            candidates = [
                ("phi-3-mini-3.8b", QuantLevel.Q8_0),
                ("llama-3.1-8b", QuantLevel.Q4_K_M),
                ("llama-3.1-8b", QuantLevel.Q5_K_M),
                ("mistral-7b", QuantLevel.Q5_K_M),
                ("qwen2-7b", QuantLevel.Q4_K_M),
                ("deepseek-coder-6.7b", QuantLevel.Q4_K_M),
            ]
            tips.append("💡 You can run larger models with better quality.")

        else:  # ULTRA_HIGH
            # 32GB+ RAM: Full models
            candidates = [
                ("llama-3.1-8b", QuantLevel.Q6_K),
                ("llama-3.1-8b", QuantLevel.Q8_0),
                ("mistral-7b", QuantLevel.Q6_K),
                ("qwen2-7b", QuantLevel.Q5_K_M),
                ("deepseek-coder-6.7b", QuantLevel.Q5_K_M),
                ("phi-3-mini-3.8b", QuantLevel.F16),
            ]
            tips.append("💡 You can run models at near-original quality.")

        # Filter by max size if specified
        if max_model_size_gb:
            candidates = [
                (m, q) for m, q in candidates
                if ModelRecommender._estimate_size(m, q) <= max_model_size_gb
            ]

        # Filter by use case
        if use_case == "code":
            # Prefer code-specialized models
            candidates = sorted(
                candidates,
                key=lambda x: "coder" in x[0] or "code" in x[0],
                reverse=True,
            )
        elif use_case == "general":
            # Prefer general-purpose models
            candidates = sorted(
                candidates,
                key=lambda x: "phi" in x[0] or "llama" in x[0],
                reverse=True,
            )

        # Build recommendations
        for model_key, quant in candidates[:6]:
            model_info = ModelDatabase.get_model(model_key)
            if not model_info:
                continue

            rec = ModelRecommender._build_recommendation(model_info, quant, system_info)
            recommendations.append(rec)

        # Pendrive recommendations (28GB budget)
        pendrive_budget = 28.0  # GB
        pendrive_candidates = [
            # Best models that fit in 28GB total (model + OS + tools)
            ("phi-3-mini-3.8b", QuantLevel.Q4_K_M, "~2.5GB"),
            ("phi-3-mini-3.8b", QuantLevel.Q5_K_M, "~3.0GB"),
            ("llama-3.1-8b", QuantLevel.Q4_K_M, "~2.5GB"),
            ("mistral-7b", QuantLevel.Q4_K_M, "~2.5GB"),
            ("deepseek-coder-6.7b", QuantLevel.Q4_K_M, "~2.5GB"),
        ]

        used_space = 5.0  # Base system + OS overhead
        for model_key, quant, size_str in pendrive_candidates:
            model_info = ModelDatabase.get_model(model_key)
            if not model_info:
                continue

            rec = ModelRecommender._build_recommendation(model_info, quant, system_info)
            model_size = rec.quant_size_gb

            if used_space + model_size <= pendrive_budget:
                rec.notes += f" (Fits in pendrive: {model_size:.1f}GB + {used_space:.1f}GB used)"
                pendrive_recommendations.append(rec)
                used_space += model_size

        # Pendrive-specific tips
        tips.append(f"💾 Pendrive budget: {pendrive_budget:.0f}GB total")
        tips.append(f"💾 Available for models: ~{pendrive_budget - used_space:.1f}GB remaining")
        tips.append("💾 Use Q4_K_M for best size/quality ratio on pendrive.")
        tips.append("💾 Store model in /models/ directory on pendrive.")

        # General tips
        tips.append("🔧 Install Ollama for easiest setup: `curl -fsSL https://ollama.com/install.sh | sh`")
        tips.append("🔧 Use `ollama run model-name` for instant inference.")
        tips.append("🔧 For GPU: `--n-gpu-layers -1` in Ollama Modelfile to offload all layers.")

        if system_info.cuda_available:
            tips.append("🎮 NVIDIA GPU detected! Use CUDA for 2-5x speedup.")
        elif system_info.gpu_name and "Apple" in system_info.gpu_name:
            tips.append("🍎 Apple Silicon detected! Unified memory works great.")

        return HardwareProfile(
            system_info=system_info,
            recommendations=recommendations,
            pendrive_recommendations=pendrive_recommendations,
            warnings=warnings,
            tips=tips,
        )

    @staticmethod
    def _estimate_size(model_key: str, quant: QuantLevel) -> float:
        """Estimate the on-disk size in GB for a model at a given quant level."""
        model_info = ModelDatabase.get_model(model_key)
        if not model_info:
            return float("inf")
        base_size = model_info["base_size_gb"]
        return base_size * ModelRecommender._quant_ratio(quant)

    @staticmethod
    def _build_recommendation(
        model_info: dict,
        quant: QuantLevel,
        system_info: SystemInfo,
    ) -> ModelRecommendation:
        """Build a ModelRecommendation from model info and quant level."""
        base_size = model_info["base_size_gb"]
        quant_ratio = ModelRecommender._quant_ratio(quant)
        quant_size = base_size * quant_ratio
        ram_required = quant_size * ModelRecommender.RAM_MULTIPLIER
        vram_required = ModelRecommender.VRAM_REQUIREMENTS.get(quant, 2.5)

        # Speed estimate based on tier
        tier = system_info.hardware_tier
        speed_key = {
            QuantLevel.Q2_K: "q2",
            QuantLevel.Q3_K_S: "q3",
            QuantLevel.Q3_K_M: "q3",
            QuantLevel.Q3_K_L: "q3",
            QuantLevel.Q4_0: "q4",
            QuantLevel.Q4_K_S: "q4",
            QuantLevel.Q4_K_M: "q4",
            QuantLevel.Q5_0: "q5",
            QuantLevel.Q5_K_S: "q5",
            QuantLevel.Q5_K_M: "q5",
            QuantLevel.Q6_K: "q6",
            QuantLevel.Q8_0: "q8",
            QuantLevel.F16: "q8",
        }.get(quant, "q4")

        speed = ModelRecommender.CPU_SPEED_ESTIMATES.get(tier, {}).get(speed_key, 5.0)

        # Quality score (higher quant = higher quality)
        quality_map = {
            QuantLevel.Q2_K: 45,
            QuantLevel.Q3_K_S: 55,
            QuantLevel.Q3_K_M: 60,
            QuantLevel.Q3_K_L: 65,
            QuantLevel.Q4_0: 65,
            QuantLevel.Q4_K_S: 72,
            QuantLevel.Q4_K_M: 78,
            QuantLevel.Q5_0: 80,
            QuantLevel.Q5_K_S: 84,
            QuantLevel.Q5_K_M: 87,
            QuantLevel.Q6_K: 92,
            QuantLevel.Q8_0: 96,
            QuantLevel.F16: 100,
        }
        quality = quality_map.get(quant, 75)

        return ModelRecommendation(
            model_name=model_info["name"],
            model_id=model_info["id"],
            parameters=model_info["parameters"],
            context_length=model_info.get("context_128k", 4096),
            quant_level=quant,
            quant_size_gb=round(quant_size, 2),
            ram_required_gb=round(ram_required, 2),
            vram_required_gb=round(vram_required, 2),
            speed_tokens_per_sec=round(speed, 1),
            quality_score=quality,
            use_case=model_info.get("best_for", "General purpose"),
            download_url=f"https://huggingface.co/{model_info['id']}",
            notes=f"License: {model_info.get('license', 'Unknown')}",
        )

    @staticmethod
    def _quant_ratio(quant: QuantLevel) -> float:
        """Get size ratio relative to F16 for a quant level."""
        ratios = {
            QuantLevel.Q2_K: 0.20,
            QuantLevel.Q3_K_S: 0.23,
            QuantLevel.Q3_K_M: 0.26,
            QuantLevel.Q3_K_L: 0.29,
            QuantLevel.Q4_0: 0.28,
            QuantLevel.Q4_K_S: 0.30,
            QuantLevel.Q4_K_M: 0.33,
            QuantLevel.Q5_0: 0.35,
            QuantLevel.Q5_K_S: 0.38,
            QuantLevel.Q5_K_M: 0.40,
            QuantLevel.Q6_K: 0.46,
            QuantLevel.Q8_0: 0.53,
            QuantLevel.F16: 1.0,
        }
        return ratios.get(quant, 0.33)


class PendriveDeployer:
    """Optimize deployment for 28GB pendrive."""

    PENDRIVE_SIZE_GB = 28.0

    @staticmethod
    def create_deployment_plan(
        profile: HardwareProfile,
        pendrive_path: str,
    ) -> dict:
        """Create a deployment plan for pendrive."""
        plan = {
            "pendrive_path": pendrive_path,
            "total_size_gb": PendriveDeployer.PENDRIVE_SIZE_GB,
            "layout": {},
            "models": [],
            "setup_commands": [],
        }

        # Allocate space
        os_overhead = 2.0  # GB for OS tools, configs
        model_budget = PendriveDeployer.PENDRIVE_SIZE_GB - os_overhead

        # Directory structure
        plan["layout"] = {
            "/": f"{os_overhead:.1f}GB - System files",
            "/models/": "Model files (GGUF format)",
            "/data/": "Training data and datasets",
            "/sessions/": "CLI session history",
            "/cache/": "Inference cache",
            "/logs/": "System logs",
            "/backups/": "Model backups",
        }

        # Select best models for pendrive
        if profile.pendrive_recommendations:
            best = profile.pendrive_recommendations[0]
            plan["models"].append({
                "name": best.model_name,
                "model_id": best.model_id,
                "quant": best.quant_level.value,
                "size_gb": best.quant_size_gb,
                "install_cmd": f"ollama pull {best.model_id.split('/')[-1].lower()}",
            })

        # Setup commands
        plan["setup_commands"] = [
            f"mkdir -p {pendrive_path}/models {pendrive_path}/data {pendrive_path}/sessions",
            f"mkdir -p {pendrive_path}/cache {pendrive_path}/logs {pendrive_path}/backups",
            "# Install Ollama on target machine",
            "curl -fsSL https://ollama.com/install.sh | sh",
            "# Import model to pendrive",
            f"ollama create phi3-custom -f {pendrive_path}/models/Modelfile",
            "# Set Ollama model storage to pendrive",
            f"export OLLAMA_MODELS={pendrive_path}/models",
        ]

        return plan

    @staticmethod
    def calculate_model_budget() -> dict:
        """Calculate how many models fit on pendrive."""
        available = PendriveDeployer.PENDRIVE_SIZE_GB - 5.0  # Reserve 5GB for system

        return {
            "total_gb": PendriveDeployer.PENDRIVE_SIZE_GB,
            "system_reserve_gb": 5.0,
            "available_for_models_gb": available,
            "max_single_model_gb": available * 0.8,  # Leave some headroom
            "recommendations": {
                "q4_k_m_3.8b": {"size_gb": 2.5, "fits": True, "count": int(available / 2.5)},
                "q4_k_m_7b": {"size_gb": 2.5, "fits": True, "count": int(available / 2.5)},
                "q5_k_m_3.8b": {"size_gb": 3.0, "fits": True, "count": int(available / 3.0)},
                "q8_0_3.8b": {"size_gb": 4.0, "fits": True, "count": int(available / 4.0)},
                "f16_3.8b": {"size_gb": 7.6, "fits": available >= 7.6, "count": int(available / 7.6)},
            },
        }


def generate_report(profile: HardwareProfile) -> str:
    """Generate a human-readable hardware report."""
    info = profile.system_info
    lines = [
        "=" * 60,
        "   MODEL RECOMMENDATIONS FOR LOW-RESOURCE HARDWARE",
        "=" * 60,
        "",
        "SYSTEM INFORMATION:",
        f"  OS:           {info.os} {info.arch}",
        f"  Python:       {info.python_version}",
        f"  CPU:          {info.cpu_name} ({info.cpu_count} cores)",
        f"  RAM:          {info.ram_gb:.1f}GB total, {info.ram_available_gb:.1f}GB available",
        f"  GPU:          {info.gpu_name}",
        f"  VRAM:         {info.gpu_vram_gb:.1f}GB" if info.gpu_vram_gb > 0 else "  GPU:          None (CPU inference)",
        f"  CUDA:         {'Yes' if info.cuda_available else 'No'}",
        f"  Storage:      {info.storage_available_gb:.1f}GB available",
        f"  Hardware Tier: {info.hardware_tier.value.upper()}",
        "",
    ]

    if profile.warnings:
        lines.append("WARNINGS:")
        for w in profile.warnings:
            lines.append(f"  {w}")
        lines.append("")

    lines.append("TOP RECOMMENDATIONS:")
    lines.append("-" * 40)
    for i, rec in enumerate(profile.recommendations[:5], 1):
        lines.extend([
            f"  {i}. {rec.model_name} ({rec.parameters})",
            f"     Quantization: {rec.quant_level.value} ({rec.quant_size_gb:.1f}GB)",
            f"     RAM Required: {rec.ram_required_gb:.1f}GB",
            f"     Speed: ~{rec.speed_tokens_per_sec:.0f} tokens/sec (CPU)",
            f"     Quality: {rec.quality_score}/100",
            f"     Use Case: {rec.use_case}",
            f"     Download: {rec.download_url}",
            "",
        ])

    if profile.pendrive_recommendations:
        lines.append("PENDRIVE DEPLOYMENT (28GB):")
        lines.append("-" * 40)
        for i, rec in enumerate(profile.pendrive_recommendations[:3], 1):
            lines.extend([
                f"  {i}. {rec.model_name} ({rec.parameters})",
                f"     Size: {rec.quant_size_gb:.1f}GB (fits in 28GB pendrive)",
                f"     Quant: {rec.quant_level.value}",
                "",
            ])

    if profile.tips:
        lines.append("TIPS:")
        lines.append("-" * 40)
        for tip in profile.tips:
            lines.append(f"  {tip}")
        lines.append("")

    lines.extend([
        "=" * 60,
        "QUICK START:",
        "  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh",
        "  2. Pull model: ollama pull phi3-mini",
        "  3. Run: ollama run phi3-mini",
        "  4. For pendrive: export OLLAMA_MODELS=/path/to/pendrive/models",
        "=" * 60,
    ])

    return "\n".join(lines)


def main(argv=None):
    """Recommend models for low-resource hardware based on real constraints."""
    parser = argparse.ArgumentParser(
        description="Recommend models for low-resource hardware"
    )
    parser.add_argument("--ram", type=float, help="Override detected total RAM in GB")
    parser.add_argument("--vram", type=float, help="Override detected GPU VRAM in GB")
    parser.add_argument(
        "--use-case",
        choices=["general", "code"],
        default="general",
        help="Optimization target for recommendations",
    )
    parser.add_argument(
        "--prefer-quality",
        action="store_true",
        help="Prefer higher quality (larger quant) options",
    )
    parser.add_argument(
        "--max-model-size",
        type=float,
        help="Maximum model size in GB to consider",
    )
    parser.add_argument(
        "--pendrive-budget",
        action="store_true",
        help="Include pendrive deployment budget and recommendations",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        help="Write the report or JSON results to this path",
    )
    args = parser.parse_args(argv)

    try:
        if args.list_models:
            models = ModelDatabase.list_models()
            output = json.dumps(models, indent=2, default=str)
            print(output)
            if args.output:
                Path(args.output).write_text(output)
            return 0

        info = HardwareDetector.detect()
        if args.ram is not None:
            info.ram_gb = args.ram
        if args.vram is not None:
            info.gpu_vram_gb = args.vram
        info.hardware_tier = HardwareDetector._classify_tier(info)

        profile = ModelRecommender.recommend(
            info,
            use_case=args.use_case,
            prefer_quality=args.prefer_quality,
            max_model_size_gb=args.max_model_size,
        )

        result = {
            "system": {
                "os": info.os,
                "arch": info.arch,
                "ram_gb": info.ram_gb,
                "vram_gb": info.gpu_vram_gb,
                "tier": info.hardware_tier.value,
            },
            "recommendations": [
                {
                    "model": r.model_name,
                    "model_id": r.model_id,
                    "quant": r.quant_level.value,
                    "size_gb": r.quant_size_gb,
                    "ram_gb": r.ram_required_gb,
                    "vram_gb": r.vram_required_gb,
                    "speed": r.speed_tokens_per_sec,
                    "quality": r.quality_score,
                    "use_case": r.use_case,
                }
                for r in profile.recommendations
            ],
        }

        if args.pendrive_budget:
            result["pendrive_recommendations"] = [
                {
                    "model": r.model_name,
                    "quant": r.quant_level.value,
                    "size_gb": r.quant_size_gb,
                }
                for r in profile.pendrive_recommendations
            ]
            result["pendrive_budget"] = PendriveDeployer.calculate_model_budget()

        if args.json:
            output = json.dumps(result, indent=2, default=str)
            print(output)
        else:
            output = generate_report(profile)
            print(output)

        if args.output:
            Path(args.output).write_text(output)

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

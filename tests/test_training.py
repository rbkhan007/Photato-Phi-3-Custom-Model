"""Coverage for offline-safe logic in training.memory_efficient."""
import json

import pytest

from training.memory_efficient import (
    TrainingConfig,
    MemoryProfiler,
    MemoryEfficientConfigPresets,
    calculate_vram_savings,
)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model_path == "microsoft/Phi-3-mini-4k-instruct"
        assert cfg.use_4bit is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.num_train_epochs == 3
        assert cfg.optim == "paged_adamw_8bit"
        assert cfg.device_map == "auto"

    def test_lora_target_modules_default(self):
        cfg = TrainingConfig()
        assert "qkv_proj" in cfg.lora_target_modules
        assert "down_proj" in cfg.lora_target_modules

    def test_target_modules_independent_instances(self):
        a = TrainingConfig()
        b = TrainingConfig()
        a.lora_target_modules.append("extra")
        assert "extra" not in b.lora_target_modules

    def test_to_dict(self):
        cfg = TrainingConfig(lora_r=64)
        d = cfg.to_dict()
        assert d["lora_r"] == 64
        assert d["model_path"] == cfg.model_path
        assert "lora_target_modules" in d

    def test_custom_values(self):
        cfg = TrainingConfig(use_4bit=False, fp16=False, bf16=True, max_seq_length=2048)
        assert cfg.use_4bit is False
        assert cfg.bf16 is True
        assert cfg.max_seq_length == 2048


class TestMemoryProfiler:
    def test_get_gpu_memory_structure(self):
        result = MemoryProfiler.get_gpu_memory()
        assert "available" in result
        assert isinstance(result["available"], bool)
        if result["available"]:
            assert "total_gb" in result
            assert "free_gb" in result

    def test_get_cpu_memory_structure(self):
        result = MemoryProfiler.get_cpu_memory()
        # psutil may or may not be installed; both shapes are valid.
        assert isinstance(result, dict)
        if result.get("available") is False:
            assert result == {"available": False}
        else:
            assert "total_gb" in result
            assert "percent" in result

    def test_print_memory_report_runs(self, capsys):
        MemoryProfiler.print_memory_report()
        out = capsys.readouterr().out
        assert "Memory Usage Report" in out


class TestConfigPresets:
    def test_colab_free_tier(self):
        cfg = MemoryEfficientConfigPresets.colab_free_tier()
        assert isinstance(cfg, TrainingConfig)
        assert cfg.use_4bit is True
        assert cfg.lora_r == 8
        assert cfg.per_device_train_batch_size == 1
        assert cfg.max_seq_length == 512
        assert cfg.use_flash_attention is False

    def test_colab_pro(self):
        cfg = MemoryEfficientConfigPresets.colab_pro()
        assert cfg.lora_r == 16
        assert cfg.per_device_train_batch_size == 2
        assert cfg.use_flash_attention is True

    def test_local_gpu(self):
        cfg = MemoryEfficientConfigPresets.local_gpu()
        assert cfg.use_4bit is True
        assert cfg.max_seq_length == 1024

    def test_high_end_gpu(self):
        cfg = MemoryEfficientConfigPresets.high_end_gpu()
        assert cfg.use_4bit is False
        assert cfg.lora_r == 32
        assert cfg.per_device_train_batch_size == 8
        assert cfg.gradient_checkpointing is False
        assert cfg.optim == "adamw_torch"

    def test_cpu_only(self):
        cfg = MemoryEfficientConfigPresets.cpu_only()
        assert cfg.use_4bit is False
        assert cfg.device_map == "cpu"
        assert cfg.fp16 is False
        assert cfg.bf16 is False
        assert cfg.max_seq_length == 256

    def test_presets_are_distinct(self):
        free = MemoryEfficientConfigPresets.colab_free_tier()
        high = MemoryEfficientConfigPresets.high_end_gpu()
        assert free.lora_r != high.lora_r
        assert free.per_device_train_batch_size != high.per_device_train_batch_size


class TestCalculateVramSavings:
    def test_runs_and_reports_reduction(self, capsys):
        calculate_vram_savings()
        out = capsys.readouterr().out
        assert "VRAM Savings" in out
        assert "Reduction" in out
        assert "less VRAM" in out

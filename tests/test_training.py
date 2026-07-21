"""Coverage for offline-safe logic in training.memory_efficient."""
import json

import pytest

from training.memory_efficient import (
    TrainingConfig,
    MemoryProfiler,
    MemoryEfficientConfigPresets,
    MetricsTracker,
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

    def test_validation_config(self):
        cfg = TrainingConfig()
        assert cfg.eval_strategy == "steps"
        assert cfg.eval_steps == 50
        assert cfg.load_best_model_at_end is True
        assert cfg.metric_for_best_model == "eval_loss"

    def test_lora_config(self):
        cfg = TrainingConfig()
        assert cfg.lora_bias == "none"
        assert cfg.lora_task_type == "CAUSAL_LM"


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


class TestMetricsTracker:
    def test_init(self):
        tracker = MetricsTracker()
        assert tracker.metrics["train_loss"] == []
        assert tracker.metrics["eval_loss"] == []
        assert tracker.total_tokens == 0

    def test_log_train(self):
        tracker = MetricsTracker()
        tracker.start()
        tracker.log_train(step=1, loss=2.5, lr=0.001, epoch=0.5)
        assert len(tracker.metrics["train_loss"]) == 1
        assert tracker.metrics["train_loss"][0] == 2.5

    def test_log_eval(self):
        tracker = MetricsTracker()
        tracker.log_eval(eval_loss=1.8)
        assert len(tracker.metrics["eval_loss"]) == 1
        assert tracker.metrics["eval_loss"][0] == 1.8

    def test_add_tokens(self):
        tracker = MetricsTracker()
        tracker.add_tokens(100)
        tracker.add_tokens(200)
        assert tracker.total_tokens == 300

    def test_get_summary(self):
        tracker = MetricsTracker()
        tracker.start()
        tracker.log_train(step=1, loss=2.5, lr=0.001, epoch=0.5)
        tracker.log_train(step=2, loss=2.0, lr=0.0005, epoch=1.0)
        tracker.log_eval(eval_loss=1.8)

        summary = tracker.get_summary()
        assert summary["total_steps"] == 2
        assert summary["avg_train_loss"] == 2.25
        assert summary["min_train_loss"] == 2.0
        assert summary["best_eval_loss"] == 1.8

    def test_print_progress(self, capsys):
        tracker = MetricsTracker()
        tracker.start()
        tracker.log_train(step=1, loss=2.5, lr=0.001, epoch=0.5)
        tracker.print_progress(step=1, total_steps=10)
        out = capsys.readouterr().out
        assert "Step 1/10" in out


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

    def test_phi4_mini(self):
        cfg = MemoryEfficientConfigPresets.phi4_mini()
        assert cfg.model_path == "microsoft/Phi-4-mini-instruct"
        assert cfg.use_4bit is True
        assert cfg.lora_r == 16
        assert cfg.max_seq_length == 2048
        assert "qkv_proj" in cfg.lora_target_modules

    def test_qwen3_embedding(self):
        cfg = MemoryEfficientConfigPresets.qwen3_embedding()
        assert cfg.model_path == "Qwen/Qwen3-Embedding-0.6B"
        assert cfg.use_4bit is True
        assert cfg.lora_r == 8
        assert cfg.max_seq_length == 1024
        assert "q_proj" in cfg.lora_target_modules

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

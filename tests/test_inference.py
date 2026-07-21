"""Coverage for inference.auto_tuner and offline-safe parts of inference.llama_server."""
import os

import pytest

from inference.auto_tuner import (
    AutoTuner,
    InferenceParams,
    TaskType,
    TASK_PRESETS,
)
from inference.llama_server import ServerConfig, LlamaCppServer


class TestInferenceParams:
    def test_defaults(self):
        p = InferenceParams()
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.top_k == 40
        assert p.max_tokens == 2048
        assert p.stop == []
        assert p.seed is None

    def test_to_dict_roundtrip(self):
        p = InferenceParams(temperature=0.5, top_k=10, stop=["END"])
        d = p.to_dict()
        assert d["temperature"] == 0.5
        assert d["top_k"] == 10
        assert d["stop"] == ["END"]
        p2 = InferenceParams.from_dict(d)
        assert p2.temperature == 0.5
        assert p2.top_k == 10

    def test_from_dict_ignores_unknown_keys(self):
        p = InferenceParams.from_dict({"temperature": 0.3, "bogus": 99})
        assert p.temperature == 0.3
        assert not hasattr(p, "bogus")


class TestTaskDetection:
    @pytest.fixture
    def tuner(self, tmp_path):
        return AutoTuner(history_file=str(tmp_path / "hist.json"))

    def test_detect_code(self, tuner):
        assert tuner.detect_task_type("Please fix this code:\n```\ndef f(): pass\n```") == TaskType.CODE

    def test_detect_code_requires_marker(self, tuner):
        # "code" keyword alone without ```/def/import falls through to QA/etc
        assert tuner.detect_task_type("import os in python") == TaskType.CODE

    def test_detect_math(self, tuner):
        assert tuner.detect_task_type("Calculate 15 * 23 + 47") == TaskType.MATH

    def test_detect_creative(self, tuner):
        assert tuner.detect_task_type("Write a short story about a robot") == TaskType.CREATIVE

    def test_detect_translation(self, tuner):
        assert tuner.detect_task_type("Translate this to french please") == TaskType.TRANSLATION

    def test_detect_summarization(self, tuner):
        assert tuner.detect_task_type("give me a tldr of this") == TaskType.SUMMARIZATION

    def test_detect_analysis(self, tuner):
        assert tuner.detect_task_type("Analyze the impact of climate policy") == TaskType.ANALYSIS

    def test_detect_qa(self, tuner):
        assert tuner.detect_task_type("Who painted the Mona Lisa") == TaskType.QA

    def test_detect_conversation_default(self, tuner):
        assert tuner.detect_task_type("hello there") == TaskType.CONVERSATION


class TestGetParams:
    @pytest.fixture
    def tuner(self, tmp_path):
        return AutoTuner(history_file=str(tmp_path / "hist.json"))

    def test_get_params_by_task_type(self, tuner):
        params = tuner.get_params(task_type=TaskType.CODE)
        assert params.temperature == TASK_PRESETS[TaskType.CODE].temperature == 0.2
        assert params.max_tokens == 4096
        assert tuner.current_params is params

    def test_get_params_auto_detect_from_prompt(self, tuner):
        params = tuner.get_params(prompt="Write a creative poem")
        assert params.temperature == TASK_PRESETS[TaskType.CREATIVE].temperature

    def test_get_params_default_conversation(self, tuner):
        params = tuner.get_params()
        assert params.temperature == TASK_PRESETS[TaskType.CONVERSATION].temperature

    def test_get_params_custom_override(self, tuner):
        params = tuner.get_params(task_type=TaskType.CODE, custom_params={"temperature": 0.99})
        assert params.temperature == 0.99

    def test_get_params_custom_ignores_unknown(self, tuner):
        params = tuner.get_params(task_type=TaskType.CODE, custom_params={"nope": 1})
        assert not hasattr(params, "nope")


class TestHistoricalAdjustments:
    @pytest.fixture
    def tuner(self, tmp_path):
        return AutoTuner(history_file=str(tmp_path / "hist.json"))

    def test_no_adjustment_with_little_history(self, tuner):
        params = InferenceParams(temperature=0.5)
        out = tuner._apply_historical_adjustments(params, TaskType.CODE)
        assert out.temperature == 0.5

    def test_adjustment_lowers_temperature(self, tuner):
        # successful attempts have lower temp than failed ones -> lower temp
        for _ in range(4):
            tuner.history.append({"task_type": "code", "success": True, "temperature": 0.2})
        for _ in range(2):
            tuner.history.append({"task_type": "code", "success": False, "temperature": 0.9})
        params = InferenceParams(temperature=0.5)
        out = tuner._apply_historical_adjustments(params, TaskType.CODE)
        assert out.temperature == pytest.approx(0.4)


class TestRecordAndFeedback:
    @pytest.fixture
    def tuner(self, tmp_path):
        return AutoTuner(history_file=str(tmp_path / "hist.json"))

    def test_record_attempt_persists(self, tuner):
        params = InferenceParams()
        tuner.record_attempt(params, TaskType.CODE, success=True, quality_score=0.9)
        assert len(tuner.history) == 1
        assert tuner.history[0]["task_type"] == "code"
        assert os.path.exists(tuner.history_file)

    def test_record_attempt_caps_history(self, tuner):
        tuner.history = [{"x": i} for i in range(1005)]
        tuner.record_attempt(InferenceParams(), TaskType.QA, success=True)
        assert len(tuner.history) == 1000

    def test_tune_repetitive_feedback(self, tuner):
        params = tuner.get_params(task_type=TaskType.CODE)
        before = params.repeat_penalty
        tuned = tuner.tune_from_feedback(params, "output is too repetitive", TaskType.CODE)
        assert tuned.repeat_penalty > before
        assert tuned.frequency_penalty > 0

    def test_tune_too_random(self, tuner):
        params = InferenceParams(temperature=0.9, top_p=0.95, top_k=60)
        tuned = tuner.tune_from_feedback(params, "too random and off-topic", TaskType.CREATIVE)
        assert tuned.temperature == pytest.approx(0.7)
        assert tuned.top_k == 40

    def test_tune_too_boring(self, tuner):
        params = InferenceParams(temperature=0.3, top_p=0.8, top_k=20)
        tuned = tuner.tune_from_feedback(params, "this is boring and dull", TaskType.CONVERSATION)
        assert tuned.temperature == pytest.approx(0.5)
        assert tuned.top_k == 40

    def test_tune_too_long(self, tuner):
        params = InferenceParams(max_tokens=2048)
        tuned = tuner.tune_from_feedback(params, "too long and verbose", TaskType.QA)
        assert tuned.max_tokens == 1024

    def test_tune_too_short(self, tuner):
        params = InferenceParams(max_tokens=1024)
        tuned = tuner.tune_from_feedback(params, "answer was too short", TaskType.QA)
        assert tuned.max_tokens == 2048

    def test_tune_format(self, tuner):
        params = InferenceParams(temperature=0.5, repeat_penalty=1.2)
        tuned = tuner.tune_from_feedback(params, "fix the markdown format", TaskType.CODE)
        assert tuned.repeat_penalty == 1.0
        assert tuned.temperature == pytest.approx(0.4)


class TestABTestAndPresets:
    @pytest.fixture
    def tuner(self, tmp_path):
        return AutoTuner(history_file=str(tmp_path / "hist.json"))

    def test_ab_test_sets_same_seed(self, tuner):
        a = InferenceParams()
        b = InferenceParams()
        ra, rb = tuner.ab_test("prompt", a, b)
        assert ra.seed == rb.seed
        assert ra.seed is not None

    def test_get_presets(self, tuner):
        presets = tuner.get_presets()
        assert "code" in presets
        assert presets["code"]["temperature"] == 0.2
        assert len(presets) == len(TASK_PRESETS)


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig(model_path="model.gguf")
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.n_ctx == 4096
        assert cfg.n_batch == 512
        assert cfg.n_gpu_layers == 0
        assert cfg.use_mmap is True

    def test_custom_values(self):
        cfg = ServerConfig(
            model_path="/p/m.gguf", host="0.0.0.0", port=9000,
            n_ctx=8192, n_gpu_layers=-1, use_mmap=False,
        )
        assert cfg.port == 9000
        assert cfg.n_ctx == 8192
        assert cfg.n_gpu_layers == -1
        assert cfg.use_mmap is False


class TestLlamaCppServerPureLogic:
    """Test pure logic on the server without loading a real model."""

    @pytest.fixture
    def server(self):
        # Bypass __init__ (which would try to load a real model) and set config directly.
        srv = LlamaCppServer.__new__(LlamaCppServer)
        srv.config = ServerConfig(model_path="/models/phi3-mini-q4.gguf")
        srv.model = None
        srv.tokenizer = None
        return srv

    def test_messages_to_prompt(self, server):
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        prompt = server._messages_to_prompt(messages)
        assert "<|system|>\nbe helpful<|end|>" in prompt
        assert "<|user|>\nhi<|end|>" in prompt
        assert "<|assistant|>\nhello<|end|>" in prompt
        assert prompt.rstrip().endswith("<|assistant|>")

    def test_messages_to_prompt_empty(self, server):
        prompt = server._messages_to_prompt([])
        assert prompt.strip() == "<|assistant|>"

    def test_get_models(self, server):
        models = server.get_models()
        assert models["object"] == "list"
        assert len(models["data"]) == 1
        entry = models["data"][0]
        assert entry["id"] == "phi3-mini-q4"
        assert entry["owned_by"] == "local"
        assert entry["object"] == "model"

"""Coverage for evaluation package."""
import pytest
from evaluation.harness import EvaluationHarness, CustomBenchmarks
from evaluation.benchmark import BenchmarkSuite
from evaluation.testing import ModelTester, SafetyTester


class TestEvaluationHarness:
    def test_init(self):
        h = EvaluationHarness()
        assert h is not None


class TestBenchmarkSuite:
    def test_init(self):
        s = BenchmarkSuite(model_path="test_model")
        assert s is not None


class TestModelTester:
    def test_init(self):
        t = ModelTester(model_path="test_model")
        assert t is not None


class TestSafetyTester:
    def test_init(self):
        t = SafetyTester(model_path="test_model")
        assert t is not None

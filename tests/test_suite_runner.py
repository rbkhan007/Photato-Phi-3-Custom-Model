import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.test_suite import ModelTestSuite, TestStatus

SUITE_NAMES = [
    "Tool Calling",
    "RAG Engine",
    "Memory System",
    "Streaming",
    "Safety Layer",
    "Structured Output",
    "Extended Thinking",
    "Multi-Modal",
    "Integration",
]


def _run_suite(name: str):
    suite = ModelTestSuite()
    target = next((s for s in suite.suites if s.name == name), None)
    assert target is not None, f"Suite not found: {name}"
    results = [suite._run_test(t) for t in target.tests]
    failures = [r for r in results if r.status in (TestStatus.FAIL, TestStatus.ERROR)]
    assert not failures, (
        f"{name}: {len(failures)} failure(s)\n"
        + "\n".join(f"  - {r.test_name}: {r.status.value} ({r.message})" for r in failures)
    )


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_suite(suite_name):
    _run_suite(suite_name)

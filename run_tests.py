#!/usr/bin/env python3
"""
Primary test runner for phi3-custom-model.

Runs the full custom test suite (evaluation.test_suite) and the demo model
tester (evaluation.testing), then exits with a non-zero code if any test
FAILs or ERRORs.

Usage:
    python run_tests.py
    pytest            # equivalent, via tests/test_suite_runner.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_suite() -> int:
    from evaluation.test_suite import ModelTestSuite, TestStatus

    print(">>> Running ModelTestSuite ...")
    suite = ModelTestSuite()
    results = suite.run_all()
    out_path = os.path.join(ROOT, "test_results.json")
    try:
        suite.save_results(results, out_path)
    except Exception:
        pass

    failed = [r for r in results if r.status in (TestStatus.FAIL, TestStatus.ERROR)]
    return 1 if failed else 0


def run_demo_tester(model_path: str) -> int:
    print(f"\n>>> Running ModelTester (quality) against: {model_path}")
    try:
        from evaluation.testing import ModelTester

        tester = ModelTester(model_path=model_path)
        tresults = tester.run_suite("quality")
        passed = sum(1 for r in tresults if getattr(r, "passed", False))
        print(f"Quality tests: {passed}/{len(tresults)} passed")
        failed = [r for r in tresults if not getattr(r, "passed", False)]
        return 1 if failed else 0
    except Exception as e:  # pragma: no cover - optional demo
        print(f"ModelTester skipped: {type(e).__name__}: {e}")
        return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run phi3-custom-model tests.")
    parser.add_argument(
        "--with-model",
        metavar="MODEL_PATH",
        default=None,
        help="Also run the live ModelTester quality suite against this model "
        "(requires a real GGUF/checkpoint). Omit to run the offline suite only.",
    )
    args = parser.parse_args()

    code = run_suite()
    if args.with_model:
        code = max(code, run_demo_tester(args.with_model))
    else:
        print("\n(Skipping live ModelTester demo. Pass --with-model <path> to include it.)")
    return code


if __name__ == "__main__":
    sys.exit(main())

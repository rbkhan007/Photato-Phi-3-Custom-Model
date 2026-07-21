#!/usr/bin/env python3
"""
Automatic Inference Parameter Tuning for Local LLMs.

Automatically optimizes temperature, top_p, top_k, and other parameters
based on the task type and user feedback.

Usage:
    from inference.auto_tuner import AutoTuner

    tuner = AutoTuner()
    params = tuner.optimize(task_type="code", feedback="too repetitive")
"""

import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskType(Enum):
    CODE = "code"
    CREATIVE = "creative"
    QA = "qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    MATH = "math"


@dataclass
class InferenceParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "seed": self.seed,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Preset parameters for different task types
TASK_PRESETS: dict[TaskType, InferenceParams] = {
    TaskType.CODE: InferenceParams(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.05,
        max_tokens=4096,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
    TaskType.CREATIVE: InferenceParams(
        temperature=0.9,
        top_p=0.95,
        top_k=60,
        repeat_penalty=1.2,
        max_tokens=2048,
        frequency_penalty=0.3,
        presence_penalty=0.3,
    ),
    TaskType.QA: InferenceParams(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=1024,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
    TaskType.SUMMARIZATION: InferenceParams(
        temperature=0.4,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=1024,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    ),
    TaskType.TRANSLATION: InferenceParams(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.05,
        max_tokens=2048,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
    TaskType.CONVERSATION: InferenceParams(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=1024,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    ),
    TaskType.ANALYSIS: InferenceParams(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=2048,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
    TaskType.MATH: InferenceParams(
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.0,
        max_tokens=2048,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ),
}


class AutoTuner:
    """
    Automatic inference parameter tuner.

    Features:
    - Task-based preset optimization
    - Feedback-driven tuning
    - Historical performance tracking
    - A/B testing support
    """

    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize tuner.

        Args:
            history_file: Path to save tuning history
        """
        self.history_file = history_file or "tuning_history.json"
        self.history: list[dict] = []
        self.current_params: Optional[InferenceParams] = None
        self._load_history()

    def _load_history(self):
        """Load tuning history from file."""
        try:
            if Path(self.history_file).exists():
                with open(self.history_file) as f:
                    self.history = json.load(f)
        except Exception:
            self.history = []

    def _save_history(self):
        """Save tuning history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    def detect_task_type(self, prompt: str) -> TaskType:
        """
        Detect task type from prompt content.

        Args:
            prompt: User prompt

        Returns:
            Detected TaskType
        """
        prompt_lower = prompt.lower()

        # Code detection
        code_keywords = [
            "code", "function", "class", "import", "def ", "print(",
            "python", "javascript", "java", "c++", "debug", "error",
            "syntax", "compile", "run", "execute", "program",
        ]
        if any(kw in prompt_lower for kw in code_keywords):
            if "```" in prompt or "def " in prompt or "import " in prompt:
                return TaskType.CODE

        # Math detection
        math_keywords = [
            "calculate", "compute", "solve", "equation", "math",
            "sum", "average", "multiply", "divide", "add", "subtract",
            "algebra", "calculus", "statistics",
        ]
        if any(kw in prompt_lower for kw in math_keywords):
            return TaskType.MATH

        # Creative detection
        creative_keywords = [
            "write", "story", "poem", "creative", "imagine", "fiction",
            "novel", "essay", "brainstorm", "idea", "design",
        ]
        if any(kw in prompt_lower for kw in creative_keywords):
            return TaskType.CREATIVE

        # Translation detection
        translation_keywords = [
            "translate", "translation", "french", "spanish", "german",
            "chinese", "japanese", "korean", "language",
        ]
        if any(kw in prompt_lower for kw in translation_keywords):
            return TaskType.TRANSLATION

        # Summarization detection
        summary_keywords = [
            "summarize", "summary", "tldr", "brief", "shorten",
            "condense", "overview", "key points",
        ]
        if any(kw in prompt_lower for kw in summary_keywords):
            return TaskType.SUMMARIZATION

        # Analysis detection
        analysis_keywords = [
            "analyze", "analysis", "explain", "compare", "contrast",
            "evaluate", "assess", "review", "critique",
        ]
        if any(kw in prompt_lower for kw in analysis_keywords):
            return TaskType.ANALYSIS

        # Q&A detection
        qa_keywords = [
            "what", "how", "why", "when", "where", "who",
            "question", "answer", "explain", "tell me",
        ]
        if any(kw in prompt_lower for kw in qa_keywords):
            return TaskType.QA

        # Default to conversation
        return TaskType.CONVERSATION

    def get_params(
        self,
        task_type: Optional[TaskType] = None,
        prompt: Optional[str] = None,
        custom_params: Optional[dict] = None,
    ) -> InferenceParams:
        """
        Get optimized parameters.

        Args:
            task_type: Task type (auto-detected if None)
            prompt: User prompt (used for auto-detection)
            custom_params: Override specific parameters

        Returns:
            Optimized InferenceParams
        """
        # Auto-detect task type
        if task_type is None and prompt:
            task_type = self.detect_task_type(prompt)
        elif task_type is None:
            task_type = TaskType.CONVERSATION

        # Get base preset
        params = InferenceParams(
            temperature=TASK_PRESETS[task_type].temperature,
            top_p=TASK_PRESETS[task_type].top_p,
            top_k=TASK_PRESETS[task_type].top_k,
            repeat_penalty=TASK_PRESETS[task_type].repeat_penalty,
            max_tokens=TASK_PRESETS[task_type].max_tokens,
            frequency_penalty=TASK_PRESETS[task_type].frequency_penalty,
            presence_penalty=TASK_PRESETS[task_type].presence_penalty,
        )

        # Apply historical adjustments
        params = self._apply_historical_adjustments(params, task_type)

        # Apply custom overrides
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)

        self.current_params = params
        return params

    def _apply_historical_adjustments(
        self, params: InferenceParams, task_type: TaskType
    ) -> InferenceParams:
        """Apply adjustments based on historical performance."""
        # Get history for this task type
        task_history = [
            h for h in self.history if h.get("task_type") == task_type.value
        ]

        if len(task_history) < 5:
            return params

        # Calculate average ratings for different parameter ranges
        successful = [h for h in task_history if h.get("success", True)]
        failed = [h for h in task_history if not h.get("success", True)]

        if not successful or not failed:
            return params

        # Adjust temperature based on success/failure patterns
        avg_success_temp = statistics.mean(
            [h.get("temperature", 0.7) for h in successful]
        )
        avg_fail_temp = statistics.mean(
            [h.get("temperature", 0.7) for h in failed]
        )

        if avg_success_temp < avg_fail_temp:
            # Lower temperature tends to succeed more
            params.temperature = max(0.1, params.temperature - 0.1)

        return params

    def record_attempt(
        self,
        params: InferenceParams,
        task_type: TaskType,
        success: bool,
        quality_score: Optional[float] = None,
        feedback: Optional[str] = None,
    ):
        """
        Record an inference attempt for future tuning.

        Args:
            params: Parameters used
            task_type: Task type
            success: Whether attempt was successful
            quality_score: Quality score (0-1)
            feedback: User feedback
        """
        record = {
            "task_type": task_type.value,
            "params": params.to_dict(),
            "success": success,
            "quality_score": quality_score,
            "feedback": feedback,
        }

        self.history.append(record)

        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        self._save_history()

    def tune_from_feedback(
        self,
        current_params: InferenceParams,
        feedback: str,
        task_type: TaskType,
    ) -> InferenceParams:
        """
        Tune parameters based on user feedback.

        Args:
            current_params: Current parameters
            feedback: User feedback
            task_type: Task type

        Returns:
            Adjusted parameters
        """
        params = InferenceParams(**current_params.to_dict())
        feedback_lower = feedback.lower()

        # Repetition issues
        if any(w in feedback_lower for w in ["repetitive", "repeating", "same"]):
            params.repeat_penalty = min(1.3, params.repeat_penalty + 0.1)
            params.frequency_penalty = min(0.5, params.frequency_penalty + 0.1)
            params.presence_penalty = min(0.5, params.presence_penalty + 0.1)

        # Too random/creative
        if any(w in feedback_lower for w in ["random", "creative", "wild", "off-topic"]):
            params.temperature = max(0.1, params.temperature - 0.2)
            params.top_p = max(0.8, params.top_p - 0.1)
            params.top_k = max(20, params.top_k - 20)

        # Too boring/dull
        if any(w in feedback_lower for w in ["boring", "dull", "monotone", "basic"]):
            params.temperature = min(1.0, params.temperature + 0.2)
            params.top_p = min(1.0, params.top_p + 0.1)
            params.top_k = min(100, params.top_k + 20)

        # Too long
        if any(w in feedback_lower for w in ["too long", "verbose", "wordy"]):
            params.max_tokens = max(256, params.max_tokens // 2)
            params.presence_penalty = min(0.5, params.presence_penalty + 0.1)

        # Too short
        if any(w in feedback_lower for w in ["too short", "incomplete", "cut off"]):
            params.max_tokens = min(4096, params.max_tokens * 2)

        # Wrong format
        if any(w in feedback_lower for w in ["format", "structure", "markdown"]):
            params.temperature = max(0.1, params.temperature - 0.1)
            params.repeat_penalty = 1.0

        # Record the tuning
        self.record_attempt(
            params, task_type, success=True, feedback=feedback
        )

        return params

    def ab_test(
        self,
        prompt: str,
        params_a: InferenceParams,
        params_b: InferenceParams,
    ) -> tuple[InferenceParams, InferenceParams]:
        """
        Set up A/B test between two parameter sets.

        Args:
            prompt: Test prompt
            params_a: First parameter set
            params_b: Second parameter set

        Returns:
            Tuple of (params_a, params_b) with random seed set
        """
        seed = random.randint(0, 2**32 - 1)
        params_a.seed = seed
        params_b.seed = seed
        return params_a, params_b

    def get_presets(self) -> dict[str, dict]:
        """Get all task presets as dictionary."""
        return {
            task_type.value: params.to_dict()
            for task_type, params in TASK_PRESETS.items()
        }


def main(argv=None):
    """Automatic inference parameter tuning from the command line."""
    parser = argparse.ArgumentParser(
        description="Automatic inference parameter tuning"
    )
    parser.add_argument("--history-file", default="tuning_history.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="Detect task type from a prompt")
    p_detect.add_argument("--prompt", required=True)

    p_params = sub.add_parser("params", help="Get optimized parameters")
    p_params.add_argument(
        "--task-type",
        choices=[t.value for t in TaskType],
    )
    p_params.add_argument("--prompt", help="Used for auto task-type detection")
    p_params.add_argument("--temperature", type=float)
    p_params.add_argument("--top-p", type=float, dest="top_p")
    p_params.add_argument("--top-k", type=int, dest="top_k")
    p_params.add_argument("--max-tokens", type=int, dest="max_tokens")
    p_params.add_argument("--repeat-penalty", type=float, dest="repeat_penalty")

    p_tune = sub.add_parser("tune", help="Tune parameters from feedback")
    p_tune.add_argument(
        "--task-type",
        choices=[t.value for t in TaskType],
    )
    p_tune.add_argument("--prompt", help="Used for auto task-type detection")
    p_tune.add_argument(
        "--feedback",
        required=True,
        help="Free-text feedback, e.g. 'too repetitive'",
    )

    sub.add_parser("presets", help="Print all task presets")

    args = parser.parse_args(argv)

    try:
        tuner = AutoTuner(history_file=args.history_file)

        if args.command == "detect":
            task_type = tuner.detect_task_type(args.prompt)
            print(json.dumps(
                {"prompt": args.prompt, "task_type": task_type.value},
                indent=2,
            ))

        elif args.command == "params":
            task_type = TaskType(args.task_type) if args.task_type else None
            overrides = {
                k: v
                for k, v in {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_tokens": args.max_tokens,
                    "repeat_penalty": args.repeat_penalty,
                }.items()
                if v is not None
            }
            params = tuner.get_params(
                task_type=task_type,
                prompt=args.prompt,
                custom_params=overrides or None,
            )
            print(json.dumps(params.to_dict(), indent=2))

        elif args.command == "tune":
            task_type = TaskType(args.task_type) if args.task_type else TaskType.CONVERSATION
            params = tuner.get_params(task_type=task_type, prompt=args.prompt)
            params = tuner.tune_from_feedback(params, args.feedback, task_type)
            print(json.dumps(params.to_dict(), indent=2))

        elif args.command == "presets":
            print(json.dumps(tuner.get_presets(), indent=2))

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

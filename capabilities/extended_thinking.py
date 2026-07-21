#!/usr/bin/env python3
"""
Extended Thinking / Chain of Thought for Local LLMs.

Mimics Claude's extended thinking capability by:
- Breaking complex problems into steps
- Self-verification of reasoning
- Step-by-step problem solving
- Reflection and self-correction

Usage:
    from capabilities.extended_thinking import ExtendedThinker

    thinker = ExtendedThinker(model_path="./phi3-mini-q4_k_m.gguf")
    response = thinker.think("Solve this complex math problem...")
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ThinkingMode(Enum):
    """Thinking mode options."""
    NONE = "none"           # No thinking, direct response
    SIMPLE = "simple"       # Basic chain of thought
    DEEP = "deep"           # Extended thinking with reflection
    ITERATIVE = "iterative" # Multiple iterations of thinking


@dataclass
class ThinkingStep:
    """A single thinking step."""
    step_number: int
    thought: str
    confidence: float = 0.8
    is_correction: bool = False
    timestamp: float = 0.0


@dataclass
class ThinkingResult:
    """Complete thinking result."""
    steps: list[ThinkingStep]
    final_answer: str
    total_thinking_time: float
    iterations: int = 1
    was_refined: bool = False
    metadata: dict = field(default_factory=dict)


class ExtendedThinker:
    """
    Extended thinking for local LLMs.

    Features:
    - Chain of thought reasoning
    - Self-verification
    - Iterative refinement
    - Step-by-step problem decomposition
    - Confidence scoring
    - Reflection and self-correction
    """

    def __init__(
        self,
        model_path: str = "",
        mode: ThinkingMode = ThinkingMode.DEEP,
        max_iterations: int = 3,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize extended thinker.

        Args:
            model_path: Path to model
            mode: Thinking mode
            max_iterations: Maximum thinking iterations
            confidence_threshold: Minimum confidence for final answer
        """
        self.model_path = model_path
        self.mode = mode
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def think(
        self,
        prompt: str,
        context: Optional[str] = None,
        mode: Optional[ThinkingMode] = None,
    ) -> ThinkingResult:
        """
        Process prompt with extended thinking.

        Args:
            prompt: User prompt
            context: Additional context
            mode: Thinking mode override

        Returns:
            ThinkingResult with reasoning steps and final answer
        """
        mode = mode or self.mode
        start_time = time.time()

        if mode == ThinkingMode.NONE:
            return self._direct_response(prompt, start_time)

        # Decompose the problem
        steps = self._decompose_problem(prompt, context)

        # Think through each step
        thinking_steps = []
        for i, step in enumerate(steps):
            thinking_step = self._think_step(step, i + 1, thinking_steps)
            thinking_steps.append(thinking_step)

        # Verify and refine
        if mode in [ThinkingMode.DEEP, ThinkingMode.ITERATIVE]:
            thinking_steps, refined = self._verify_and_refine(
                prompt, thinking_steps
            )
        else:
            refined = False

        # Generate final answer
        final_answer = self._synthesize_answer(prompt, thinking_steps)

        total_time = time.time() - start_time

        return ThinkingResult(
            steps=thinking_steps,
            final_answer=final_answer,
            total_thinking_time=total_time,
            iterations=1 if not refined else 2,
            was_refined=refined,
            metadata={"mode": mode.value, "prompt_length": len(prompt)},
        )

    def _decompose_problem(self, prompt: str, context: Optional[str] = None) -> list[str]:
        """
        Decompose a problem into thinking steps.

        Args:
            prompt: User prompt
            context: Additional context

        Returns:
            List of thinking steps
        """
        # Simple heuristic decomposition
        steps = []

        # Step 1: Understand the problem
        steps.append(f"Understanding the problem: {prompt[:200]}")

        # Step 2: Identify key components
        if "?" in prompt:
            questions = re.findall(r'[^.!?]*\?', prompt)
            for q in questions[:3]:
                steps.append(f"Addressing: {q.strip()}")

        # Step 3: Consider context
        if context:
            steps.append(f"Considering context: {context[:200]}")

        # Step 4: Formulate approach
        steps.append("Formulating response approach")

        # Step 5: Generate response
        steps.append("Generating comprehensive response")

        return steps

    def _think_step(
        self,
        step: str,
        step_number: int,
        previous_steps: list[ThinkingStep],
    ) -> ThinkingStep:
        """
        Think through a single step.

        Args:
            step: Step description
            step_number: Step number
            previous_steps: Previous thinking steps

        Returns:
            ThinkingStep with reasoning
        """
        # Build reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(step, previous_steps)

        # Generate reasoning (placeholder - use actual model in production)
        thought = self._generate_thought(reasoning_prompt)

        # Calculate confidence
        confidence = self._calculate_confidence(thought, previous_steps)

        return ThinkingStep(
            step_number=step_number,
            thought=thought,
            confidence=confidence,
            is_correction=False,
            timestamp=time.time(),
        )

    def _build_reasoning_prompt(
        self,
        step: str,
        previous_steps: list[ThinkingStep],
    ) -> str:
        """Build prompt for reasoning."""
        parts = ["Think step by step.\n"]

        if previous_steps:
            parts.append("Previous reasoning:")
            for prev in previous_steps[-3:]:  # Last 3 steps
                parts.append(f"  Step {prev.step_number}: {prev.thought[:100]}")
            parts.append("")

        parts.append(f"Current step: {step}")
        parts.append("\nThink through this carefully:")

        return "\n".join(parts)

    def _generate_thought(self, prompt: str) -> str:
        """
        Generate thought for a step.

        Args:
            prompt: Reasoning prompt

        Returns:
            Generated thought
        """
        # Placeholder - in production use actual model
        # Simulate thinking
        return f"Analyzing: {prompt[:150]}..."

    def _calculate_confidence(
        self,
        thought: str,
        previous_steps: list[ThinkingStep],
    ) -> float:
        """
        Calculate confidence in a thinking step.

        Args:
            thought: Generated thought
            previous_steps: Previous steps

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.8  # Base confidence

        # Reduce confidence if thought is very short
        if len(thought) < 50:
            confidence -= 0.2

        # Reduce confidence if many corrections
        corrections = sum(1 for s in previous_steps if s.is_correction)
        confidence -= corrections * 0.1

        # Increase confidence if previous steps were confident
        if previous_steps:
            avg_prev_confidence = sum(s.confidence for s in previous_steps) / len(previous_steps)
            confidence = (confidence + avg_prev_confidence) / 2

        return max(0.1, min(1.0, confidence))

    def _verify_and_refine(
        self,
        prompt: str,
        steps: list[ThinkingStep],
    ) -> tuple[list[ThinkingStep], bool]:
        """
        Verify and refine thinking steps.

        Args:
            prompt: Original prompt
            steps: Thinking steps

        Returns:
            Tuple of (refined steps, was_refined)
        """
        was_refined = False

        # Check if any step has low confidence
        low_confidence_steps = [
            s for s in steps if s.confidence < self.confidence_threshold
        ]

        if low_confidence_steps:
            # Refine low confidence steps
            for step in low_confidence_steps:
                refined_thought = self._refine_thought(prompt, step)
                step.thought = refined_thought
                step.confidence = min(1.0, step.confidence + 0.2)
                step.is_correction = True
                was_refined = True

        return steps, was_refined

    def _refine_thought(self, prompt: str, step: ThinkingStep) -> str:
        """
        Refine a thinking step.

        Args:
            prompt: Original prompt
            step: Step to refine

        Returns:
            Refined thought
        """
        # Build refinement prompt
        refinement_prompt = f"""Original reasoning had low confidence ({step.confidence:.2f}).

Original thought: {step.thought}

Please provide a more careful and accurate reasoning for this step.
Consider: {prompt[:300]}

Refined reasoning:"""

        # Generate refined thought (placeholder)
        return f"Refined: {step.thought[:100]}..."

    def _synthesize_answer(
        self,
        prompt: str,
        steps: list[ThinkingStep],
    ) -> str:
        """
        Synthesize final answer from thinking steps.

        Args:
            prompt: Original prompt
            steps: Thinking steps

        Returns:
            Final answer
        """
        # Build synthesis prompt
        synthesis_parts = [
            "Based on the following reasoning:\n",
        ]

        for step in steps:
            synthesis_parts.append(f"Step {step.step_number}: {step.thought[:200]}")

        synthesis_parts.extend([
            "\nProvide a clear, concise answer to:",
            prompt[:500],
            "\nAnswer:",
        ])

        synthesis_prompt = "\n".join(synthesis_parts)

        # Generate answer (placeholder)
        return f"Based on my analysis, here is the answer to your question about: {prompt[:100]}..."

    def _direct_response(self, prompt: str, start_time: float) -> ThinkingResult:
        """Generate direct response without thinking."""
        return ThinkingResult(
            steps=[],
            final_answer=f"Direct response to: {prompt[:200]}",
            total_thinking_time=time.time() - start_time,
            iterations=1,
            was_refined=False,
        )

    def think_about_code(self, code: str, task: str = "review") -> ThinkingResult:
        """
        Think about code specifically.

        Args:
            code: Code to analyze
            task: Task (review, debug, optimize, explain)

        Returns:
            ThinkingResult
        """
        prompt = f"Think about this code for {task}:\n\n```python\n{code}\n```"
        return self.think(prompt, mode=ThinkingMode.DEEP)

    def think_about_math(self, problem: str) -> ThinkingResult:
        """
        Think about math problem.

        Args:
            problem: Math problem

        Returns:
            ThinkingResult
        """
        prompt = f"Solve this math problem step by step:\n\n{problem}"
        return self.think(prompt, mode=ThinkingMode.DEEP)

    def think_about_text(self, text: str, task: str = "analyze") -> ThinkingResult:
        """
        Think about text.

        Args:
            text: Text to analyze
            task: Task (analyze, summarize, compare)

        Returns:
            ThinkingResult
        """
        prompt = f"{task} this text:\n\n{text[:1000]}"
        return self.think(prompt, mode=ThinkingMode.DEEP)


class ChainOfThought:
    """
    Chain of Thought prompting utility.
    """

    @staticmethod
    def format_cot_prompt(prompt: str) -> str:
        """
        Format prompt with chain of thought instructions.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with CoT instructions
        """
        return f"""{prompt}

Let's think step by step:

1. First, I'll identify the key information...
2. Next, I'll analyze the relationships...
3. Then, I'll work through the logic...
4. Finally, I'll provide the answer...

Step-by-step reasoning:"""

    @staticmethod
    def format_self_consistent_cot(prompt: str, n_paths: int = 3) -> str:
        """
        Format prompt for self-consistent CoT.

        Args:
            prompt: Original prompt
            n_paths: Number of reasoning paths

        Returns:
            Prompt for self-consistent CoT
        """
        paths = "\n".join([
            f"Path {i+1}: Think independently and reach a conclusion."
            for i in range(n_paths)
        ])

        return f"""{prompt}

I'll solve this {n_paths} different ways and take the majority answer:

{paths}

After considering all paths, the answer is:"""

    @staticmethod
    def extract_reasoning(response: str) -> dict:
        """
        Extract reasoning from response.

        Args:
            model response

        Returns:
            Extracted reasoning dict
        """
        # Extract steps
        steps = re.findall(r'Step \d+:.*?(?=Step \d+:|$)', response, re.DOTALL)

        # Extract final answer
        answer_match = re.search(r'(?:Answer|Conclusion|Result):?\s*(.*?)(?:$)', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        return {
            "steps": [s.strip() for s in steps],
            "answer": answer,
            "has_reasoning": len(steps) > 0,
        }


def main(argv=None):
    """CLI for extended thinking / chain of thought."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.extended_thinking",
        description="Extended thinking / chain of thought for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_prompt_args(p):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--prompt", help="Prompt text")
        src.add_argument("--file", help="Read prompt from file")
        p.add_argument("--context", help="Additional context")
        p.add_argument(
            "--mode",
            choices=["none", "simple", "deep", "iterative"],
            default="deep",
        )

    p_think = sub.add_parser("think", help="Run extended thinking on a prompt")
    add_prompt_args(p_think)

    p_math = sub.add_parser("math", help="Think about a math problem")
    msrc = p_math.add_mutually_exclusive_group(required=True)
    msrc.add_argument("--problem", help="Math problem text")
    msrc.add_argument("--file", help="Read problem from file")
    p_math.add_argument(
        "--mode", choices=["none", "simple", "deep", "iterative"], default="deep"
    )

    p_code = sub.add_parser("code", help="Think about code")
    csrc = p_code.add_mutually_exclusive_group(required=True)
    csrc.add_argument("--code", help="Code text")
    csrc.add_argument("--file", help="Read code from file")
    p_code.add_argument(
        "--task", default="review", choices=["review", "debug", "optimize", "explain"]
    )
    p_code.add_argument(
        "--mode", choices=["none", "simple", "deep", "iterative"], default="deep"
    )

    p_text = sub.add_parser("text", help="Think about text")
    tsrc = p_text.add_mutually_exclusive_group(required=True)
    tsrc.add_argument("--text", help="Text to analyze")
    tsrc.add_argument("--file", help="Read text from file")
    p_text.add_argument(
        "--task", default="analyze", choices=["analyze", "summarize", "compare"]
    )
    p_text.add_argument(
        "--mode", choices=["none", "simple", "deep", "iterative"], default="deep"
    )

    p_cot = sub.add_parser("cot", help="Build a chain-of-thought prompt")
    cot_src = p_cot.add_mutually_exclusive_group(required=True)
    cot_src.add_argument("--prompt", help="Prompt text")
    cot_src.add_argument("--file", help="Read prompt from file")
    p_cot.add_argument("--self-consistent", action="store_true",
                       help="Self-consistent CoT")
    p_cot.add_argument("--n-paths", type=int, default=3)

    args = parser.parse_args(argv)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return value

    def result_to_dict(result):
        return {
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "confidence": s.confidence,
                    "is_correction": s.is_correction,
                    "timestamp": s.timestamp,
                }
                for s in result.steps
            ],
            "final_answer": result.final_answer,
            "total_thinking_time": result.total_thinking_time,
            "iterations": result.iterations,
            "was_refined": result.was_refined,
            "metadata": result.metadata,
        }

    try:
        if args.command in ("think", "math", "code", "text"):
            thinker = ExtendedThinker(mode=ThinkingMode[args.mode.upper()])

        if args.command == "think":
            prompt = read_source(args.prompt, args.file)
            result = thinker.think(
                prompt,
                context=args.context,
                mode=ThinkingMode[args.mode.upper()],
            )
            print(json.dumps(result_to_dict(result), indent=2, default=str))
        elif args.command == "math":
            problem = read_source(args.problem, args.file)
            result = thinker.think_about_math(problem)
            print(json.dumps(result_to_dict(result), indent=2, default=str))
        elif args.command == "code":
            code = read_source(args.code, args.file)
            result = thinker.think_about_code(code, task=args.task)
            print(json.dumps(result_to_dict(result), indent=2, default=str))
        elif args.command == "text":
            text = read_source(args.text, args.file)
            result = thinker.think_about_text(text, task=args.task)
            print(json.dumps(result_to_dict(result), indent=2, default=str))
        elif args.command == "cot":
            prompt = read_source(args.prompt, args.file)
            if args.self_consistent:
                out = ChainOfThought.format_self_consistent_cot(prompt, args.n_paths)
            else:
                out = ChainOfThought.format_cot_prompt(prompt)
            print(out)
        return 0
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

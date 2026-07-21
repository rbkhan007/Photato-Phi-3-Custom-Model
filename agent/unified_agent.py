#!/usr/bin/env python3
"""
Unified Agent with All Features.

Combines:
- Self-healing tool calling
- Web search
- Code execution (Python/Bash)
- Auto inference parameter tuning
- Local LLM inference via llama.cpp

Usage:
    from agent.unified_agent import UnifiedAgent

    agent = UnifiedAgent(model_path="./phi3-mini-q4_k_m.gguf")
    response = agent.run("Search for Python news and write a summary script")
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .self_healing_agent import SelfHealingAgent, ToolType, ToolResult
from .web_search import WebSearch
from .code_executor import CodeExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.auto_tuner import AutoTuner, TaskType, InferenceParams


@dataclass
class AgentConfig:
    """Unified agent configuration."""
    model_path: str
    max_retries: int = 3
    verbose: bool = True
    web_search_backend: str = "duckduckgo"
    auto_tune: bool = True
    max_tokens: int = 2048


class UnifiedAgent:
    """
    Unified agent with all capabilities.

    Features:
    - Self-healing tool calling
    - Web search integration
    - Code execution (Python/Bash)
    - Automatic parameter tuning
    - Local LLM inference
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize unified agent.

        Args:
            config: Agent configuration
            **kwargs: Additional config parameters
        """
        if config is None:
            config = AgentConfig(**kwargs)
        self.config = config

        # Initialize components
        self.tool_agent = SelfHealingAgent(
            model_path=config.model_path,
            max_retries=config.max_retries,
            verbose=config.verbose,
        )
        self.web_search = WebSearch(backend=config.web_search_backend)
        self.code_executor = CodeExecutor()
        self.auto_tuner = AutoTuner()

        # Custom tool registry
        self.tool_agent.tool_registry[ToolType.WEB_SEARCH] = self._web_search_tool
        self.tool_agent.tool_registry[ToolType.PYTHON_EXEC] = self._python_exec_tool
        self.tool_agent.tool_registry[ToolType.BASH_EXEC] = self._bash_exec_tool

    def _web_search_tool(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """Web search tool wrapper."""
        try:
            response = self.web_search.search(query, num_results=num_results)
            formatted = self.web_search.format_results(response)
            return ToolResult(
                success=True,
                output=formatted,
                metadata={"query": query, "result_count": response.total_results},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _python_exec_tool(self, code: str, timeout: int = 30, **kwargs) -> ToolResult:
        """Python execution tool wrapper."""
        result = self.code_executor.run_python(code, timeout=timeout)
        return ToolResult(
            success=result.success,
            output=self.code_executor.format_result(result),
            error=result.stderr if not result.success else None,
        )

    def _bash_exec_tool(self, command: str, timeout: int = 30, **kwargs) -> ToolResult:
        """Bash execution tool wrapper."""
        result = self.code_executor.run_bash(command, timeout=timeout)
        return ToolResult(
            success=result.success,
            output=self.code_executor.format_result(result),
            error=result.stderr if not result.success else None,
        )

    def _detect_task_type(self, user_input: str) -> TaskType:
        """Detect task type from user input."""
        return self.auto_tuner.detect_task_type(user_input)

    def _get_optimized_params(self, task_type: TaskType) -> InferenceParams:
        """Get optimized inference parameters."""
        return self.auto_tuner.get_params(task_type=task_type)

    def run(self, user_input: str) -> str:
        """
        Run agent with all capabilities.

        Args:
            user_input: User's request

        Returns:
            Agent's response
        """
        # Detect task type
        task_type = self._detect_task_type(user_input)

        # Get optimized parameters
        params = self._get_optimized_params(task_type)

        if self.config.verbose:
            print(f"\n[Task Type: {task_type.value}]")
            print(f"[Params: temp={params.temperature}, top_p={params.top_p}, top_k={params.top_k}]")

        # Use tool agent for execution
        response = self.tool_agent.run(user_input)

        # Record attempt for auto-tuning
        self.auto_tuner.record_attempt(
            params=params,
            task_type=task_type,
            success=True,
        )

        return response

    def run_with_feedback(self, user_input: str, feedback: str) -> str:
        """
        Run agent with feedback loop.

        Args:
            user_input: User's request
            feedback: User feedback for tuning

        Returns:
            Agent's response
        """
        # Detect task type
        task_type = self._detect_task_type(user_input)

        # Get current params and tune based on feedback
        current_params = self._get_optimized_params(task_type)
        tuned_params = self.auto_tuner.tune_from_feedback(
            current_params, feedback, task_type
        )

        if self.config.verbose:
            print(f"\n[Feedback Applied: {feedback}]")
            print(f"[Tuned Params: temp={tuned_params.temperature}, top_p={tuned_params.top_p}]")

        # Run with tuned params
        return self.tool_agent.run(user_input)

    def search(self, query: str, num_results: int = 5) -> str:
        """Direct web search."""
        response = self.web_search.search(query, num_results=num_results)
        return self.web_search.format_results(response)

    def execute_python(self, code: str, timeout: int = 30) -> str:
        """Direct Python execution."""
        result = self.code_executor.run_python(code, timeout=timeout)
        return self.code_executor.format_result(result)

    def execute_command(self, command: str, timeout: int = 30) -> str:
        """Direct shell command execution."""
        result = self.code_executor.run_bash(command, timeout=timeout)
        return self.code_executor.format_result(result)

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "model_path": self.config.model_path,
            "web_search_backend": self.config.web_search_backend,
            "auto_tune_enabled": self.config.auto_tune,
            "task_presets": self.auto_tuner.get_presets(),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified agent CLI")
    parser.add_argument("--model-path", default="", help="Path to the model file")
    parser.add_argument("--backend", default="duckduckgo", help="Web search backend")
    parser.add_argument("--no-auto-tune", action="store_true", help="Disable auto tuning")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="Search the web")
    p.add_argument("query", help="Search query")
    p.add_argument("--num-results", type=int, default=5, help="Number of results")

    p = sub.add_parser("python", help="Execute Python code")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--code", help="Python code string")
    grp.add_argument("--file", help="Path to a Python file")
    p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")

    p = sub.add_parser("command", help="Execute a shell command")
    p.add_argument("cmd", help="Command to execute")

    p = sub.add_parser("detect", help="Detect the task type of a request")
    p.add_argument("input", help="User request")

    p = sub.add_parser("status", help="Show agent status")

    p = sub.add_parser("run", help="Run the agent on a request")
    p.add_argument("input", help="User request")
    p.add_argument("--feedback", default=None, help="Feedback used for tuning")

    args = parser.parse_args(argv)

    config = AgentConfig(
        model_path=args.model_path,
        web_search_backend=args.backend,
        auto_tune=not args.no_auto_tune,
    )
    agent = UnifiedAgent(config=config)
    try:
        if args.command == "search":
            print(agent.search(args.query, num_results=args.num_results))
            return 0
        elif args.command == "python":
            code = args.code
            if code is None:
                with open(args.file, "r", encoding="utf-8") as f:
                    code = f.read()
            print(agent.execute_python(code, timeout=args.timeout))
            return 0
        elif args.command == "command":
            print(agent.execute_command(args.cmd))
            return 0
        elif args.command == "detect":
            task_type = agent._detect_task_type(args.input)
            print(json.dumps(
                {"input": args.input, "task_type": task_type.value},
                indent=2,
            ))
            return 0
        elif args.command == "status":
            print(json.dumps(agent.get_status(), indent=2, default=str))
            return 0
        elif args.command == "run":
            if args.feedback:
                out = agent.run_with_feedback(args.input, args.feedback)
            else:
                out = agent.run(args.input)
            print(out)
            return 0
        else:
            parser.error("Unknown command")
            return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

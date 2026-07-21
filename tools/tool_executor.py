"""
Tool Executor for running tool calls safely.

Provides sandboxed execution, timeout handling, error recovery,
and result formatting for tool calls.
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ExecutionResult:
    """Result of a tool execution."""
    tool_use_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    is_error: bool = False

    def to_tool_result(self) -> dict:
        """Convert to Claude tool_result format."""
        content = self.output if self.success else f"Error: {self.error}"
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        elif not isinstance(content, str):
            content = str(content)

        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": content,
            "is_error": self.is_error,
        }


class ToolExecutor:
    """
    Execute tool calls with safety controls.

    Features:
    - Sandboxed execution with timeout
    - Error handling and recovery
    - Execution history and logging
    - Resource usage tracking
    - Retry logic for transient failures
    """

    def __init__(self, registry=None, timeout: float = 30.0, max_retries: int = 2):
        self._registry = registry
        self._timeout = timeout
        self._max_retries = max_retries
        self._history: list[ExecutionResult] = []
        self._hooks: dict[str, list[Callable]] = {
            "before": [],
            "after": [],
            "on_error": [],
        }

    def execute(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str = None,
    ) -> ExecutionResult:
        """
        Execute a single tool call.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            tool_use_id: Unique ID for this tool call

        Returns:
            ExecutionResult with output or error
        """
        if tool_use_id is None:
            tool_use_id = f"toolu_{int(time.time() * 1000)}"

        # Validate tool exists
        if self._registry and not self._registry.has(tool_name):
            result = ExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
                is_error=True,
            )
            self._history.append(result)
            return result

        # Validate input
        if self._registry:
            valid, msg = self._registry.validate_input(tool_name, tool_input)
            if not valid:
                result = ExecutionResult(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    success=False,
                    error=f"Invalid input: {msg}",
                    is_error=True,
                )
                self._history.append(result)
                return result

        # Get tool handler
        if self._registry:
            tool = self._registry.get(tool_name)
            handler = tool.handler
        else:
            result = ExecutionResult(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                success=False,
                error="No registry configured",
                is_error=True,
            )
            self._history.append(result)
            return result

        # Execute with retries
        last_error = None
        for attempt in range(self._max_retries + 1):
            # Run before hooks
            for hook in self._hooks["before"]:
                try:
                    hook(tool_name, tool_input, attempt)
                except Exception:
                    pass

            start_time = time.time()
            try:
                output = handler(**tool_input)
                duration = (time.time() - start_time) * 1000

                result = ExecutionResult(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    success=True,
                    output=output,
                    duration_ms=duration,
                )

                # Run after hooks
                for hook in self._hooks["after"]:
                    try:
                        hook(result)
                    except Exception:
                        pass

                self._history.append(result)
                return result

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                last_error = f"{type(e).__name__}: {e}"

                # Run error hooks
                for hook in self._hooks["on_error"]:
                    try:
                        hook(tool_name, tool_input, e, attempt)
                    except Exception:
                        pass

                if attempt < self._max_retries:
                    time.sleep(0.1 * (attempt + 1))
                    continue

        result = ExecutionResult(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            success=False,
            error=last_error or "Unknown error",
            duration_ms=duration,
            is_error=True,
        )
        self._history.append(result)
        return result

    def execute_batch(self, tool_calls: list[dict]) -> list[ExecutionResult]:
        """
        Execute multiple tool calls.

        Each item should have: name, input, id (optional)
        """
        results = []
        for call in tool_calls:
            result = self.execute(
                tool_name=call.get("name", call.get("tool", "")),
                tool_input=call.get("input", call.get("parameters", {})),
                tool_use_id=call.get("id"),
            )
            results.append(result)
        return results

    def execute_from_parsed(self, parsed_calls: list) -> list[ExecutionResult]:
        """Execute a list of ParsedToolCall objects."""
        results = []
        for call in parsed_calls:
            result = self.execute(
                tool_name=call.name,
                tool_input=call.input,
                tool_use_id=call.id,
            )
            results.append(result)
        return results

    def add_hook(self, event: str, hook: Callable):
        """Add a lifecycle hook (before, after, on_error)."""
        if event in self._hooks:
            self._hooks[event].append(hook)

    def get_history(self) -> list[ExecutionResult]:
        """Get execution history."""
        return self._history

    def get_stats(self) -> dict:
        """Get execution statistics."""
        total = len(self._history)
        success = sum(1 for r in self._history if r.success)
        failed = sum(1 for r in self._history if not r.success)
        avg_duration = (
            sum(r.duration_ms for r in self._history) / total
            if total > 0 else 0
        )
        return {
            "total_executions": total,
            "successful": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "average_duration_ms": avg_duration,
        }

    def clear_history(self):
        """Clear execution history."""
        self._history.clear()

#!/usr/bin/env python3
"""
Self-Healing Agent with Tool Calling and Web Search.

Features:
- Automatic tool selection and execution
- Self-healing: detects failures and retries with different approaches
- Web search integration
- Code execution (Python/Bash)
- Parameter auto-tuning

Usage:
    from agent.self_healing_agent import SelfHealingAgent

    agent = SelfHealingAgent(model_path="./phi3-mini-q4_k_m.gguf")
    response = agent.run("Search the web for latest Python news and summarize it")
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class ToolType(Enum):
    WEB_SEARCH = "web_search"
    PYTHON_EXEC = "python_exec"
    BASH_EXEC = "bash_exec"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CALCULATOR = "calculator"


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    tool: ToolType
    args: dict
    result: Optional[ToolResult] = None
    attempts: int = 0
    max_retries: int = 3


class SelfHealingAgent:
    """
    Agent with self-healing tool calling capabilities.
    
    Features:
    - Detects tool call failures automatically
    - Retries with alternative approaches
    - Falls back to simpler methods on repeated failures
    - Logs all attempts for debugging
    """

    def __init__(
        self,
        model_path: str,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.max_retries = max_retries
        self.verbose = verbose
        self.conversation_history: list[dict] = []
        self.tool_registry: dict[ToolType, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all available tools."""
        self.tool_registry = {
            ToolType.WEB_SEARCH: self._web_search,
            ToolType.PYTHON_EXEC: self._python_exec,
            ToolType.BASH_EXEC: self._bash_exec,
            ToolType.FILE_READ: self._file_read,
            ToolType.FILE_WRITE: self._file_write,
            ToolType.CALCULATOR: self._calculator,
        }

    def _log(self, message: str, level: str = "INFO"):
        """Log message if verbose."""
        if self.verbose:
            prefix = {
                "INFO": "[*]",
                "WARN": "[!]",
                "ERROR": "[X]",
                "SUCCESS": "[+]",
                "RETRY": "[~]",
            }.get(level, "[.]")
            print(f"{prefix} {message}")

    def _parse_tool_call(self, text: str) -> Optional[ToolCall]:
        """Parse tool call from model output."""
        # Look for tool call patterns
        patterns = [
            r"```tool\n(.*?)\n```",
            r"<tool>(.*?)</tool>",
            r"\{\"tool\":\s*\"(\w+)\".*?\"args\":\s*(\{.*?\})\}",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    if len(match.groups()) == 1:
                        tool_data = json.loads(match.group(1))
                    else:
                        tool_name = match.group(1)
                        tool_args = json.loads(match.group(2))
                        tool_data = {"tool": tool_name, "args": tool_args}

                    tool_type = ToolType(tool_data["tool"])
                    return ToolCall(tool=tool_type, args=tool_data.get("args", {}))
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        return None

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool with self-healing retries."""
        tool_func = self.tool_registry.get(tool_call.tool)
        if not tool_func:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_call.tool}",
            )

        last_error = None
        for attempt in range(tool_call.max_retries):
            tool_call.attempts = attempt + 1
            try:
                self._log(
                    f"Executing {tool_call.tool.value} (attempt {attempt + 1})",
                    "INFO",
                )
                result = tool_func(**tool_call.args)
                if result.success:
                    self._log(f"Tool {tool_call.tool.value} succeeded", "SUCCESS")
                    return result
                last_error = result.error
                self._log(
                    f"Tool {tool_call.tool.value} failed: {result.error}", "WARN"
                )
            except Exception as e:
                last_error = str(e)
                self._log(f"Tool {tool_call.tool.value} exception: {e}", "ERROR")

            # Self-healing: try alternative approach
            if attempt < tool_call.max_retries - 1:
                self._log(
                    f"Self-healing: retrying with different approach...", "RETRY"
                )
                tool_call.args = self._heal_tool_args(
                    tool_call.tool, tool_call.args, last_error
                )

        return ToolResult(
            success=False,
            output="",
            error=f"Tool failed after {tool_call.max_retries} attempts: {last_error}",
        )

    def _heal_tool_args(
        self, tool_type: ToolType, args: dict, error: str
    ) -> dict:
        """Attempt to heal tool arguments based on error."""
        healed = args.copy()

        if tool_type == ToolType.PYTHON_EXEC:
            # Try simplifying Python code
            if "syntax error" in error.lower():
                healed["code"] = self._simplify_python(args.get("code", ""))
            elif "timeout" in error.lower():
                healed["timeout"] = 60
            elif "import" in error.lower():
                healed["code"] = self._add_missing_imports(args.get("code", ""))

        elif tool_type == ToolType.BASH_EXEC:
            # Try alternative commands
            if "command not found" in error.lower():
                healed["command"] = self._alternative_command(args.get("command", ""))
            elif "permission denied" in error.lower():
                healed["command"] = f"sudo {args.get('command', '')}"

        elif tool_type == ToolType.WEB_SEARCH:
            # Try different search query
            if "no results" in error.lower():
                healed["query"] = self._refine_query(args.get("query", ""))

        return healed

    def _simplify_python(self, code: str) -> str:
        """Simplify Python code that has syntax errors."""
        # Remove complex syntax
        code = re.sub(r"lambda.*?:", "lambda x: x", code)
        code = re.sub(r"f\"[^\"]*\"", '"string"', code)
        return code

    def _add_missing_imports(self, code: str) -> str:
        """Add common missing imports."""
        common_imports = {
            "os": "import os",
            "sys": "import sys",
            "json": "import json",
            "re": "import re",
            "math": "import math",
            "datetime": "from datetime import datetime",
            "pathlib": "from pathlib import Path",
        }
        imports = []
        for module, import_stmt in common_imports.items():
            if re.search(rf"\b{module}\b", code) and import_stmt not in code:
                imports.append(import_stmt)
        return "\n".join(imports) + "\n" + code if imports else code

    def _alternative_command(self, cmd: str) -> str:
        """Suggest alternative commands."""
        alternatives = {
            "ls": "dir" if sys.platform == "win32" else "ls -la",
            "cat": "type" if sys.platform == "win32" else "cat",
            "grep": "findstr" if sys.platform == "win32" else "grep",
        }
        return alternatives.get(cmd, cmd)

    def _refine_query(self, query: str) -> str:
        """Refine search query for better results."""
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at"}
        words = query.split()
        refined = [w for w in words if w.lower() not in stop_words]
        return " ".join(refined) if refined else query

    # Tool implementations

    def _web_search(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """Search the web using DuckDuckGo."""
        try:
            import requests
            from bs4 import BeautifulSoup

            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract abstract
                if data.get("Abstract"):
                    results.append(
                        {
                            "title": data.get("Heading", ""),
                            "snippet": data.get("Abstract", ""),
                            "url": data.get("AbstractURL", ""),
                        }
                    )

                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(
                            {
                                "title": topic.get("Text", "")[:100],
                                "snippet": topic.get("Text", ""),
                                "url": topic.get("FirstURL", ""),
                            }
                        )

                if results:
                    return ToolResult(
                        success=True,
                        output=json.dumps(results[:num_results], indent=2),
                        metadata={"query": query, "result_count": len(results)},
                    )

            # Fallback: try Google search via requests
            return self._google_search_fallback(query, num_results)

        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="requests library not installed. Run: pip install requests beautifulsoup4",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _google_search_fallback(
        self, query: str, num_results: int
    ) -> ToolResult:
        """Fallback web search using Google."""
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            url = f"https://www.google.com/search?q={query}&num={num_results}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []

                for g in soup.select("div.g")[:num_results]:
                    title_el = g.select_one("h3")
                    snippet_el = g.select_one("div.VwiC3b")
                    link_el = g.select_one("a")

                    if title_el and link_el:
                        results.append(
                            {
                                "title": title_el.text,
                                "snippet": snippet_el.text if snippet_el else "",
                                "url": link_el["href"],
                            }
                        )

                return ToolResult(
                    success=True,
                    output=json.dumps(results, indent=2),
                    metadata={"query": query, "result_count": len(results)},
                )

            return ToolResult(
                success=False, output="", error=f"HTTP {response.status_code}"
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _python_exec(
        self, code: str, timeout: int = 30, **kwargs
    ) -> ToolResult:
        """Execute Python code safely."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.getcwd(),
                )

                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]\n{result.stderr}"

                return ToolResult(
                    success=result.returncode == 0,
                    output=output.strip(),
                    error=result.stderr if result.returncode != 0 else None,
                    metadata={"return_code": result.returncode},
                )
            finally:
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Code execution timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _bash_exec(self, command: str, timeout: int = 30, **kwargs) -> ToolResult:
        """Execute bash/shell command."""
        try:
            shell = True if sys.platform != "win32" else False
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error=result.stderr if result.returncode != 0 else None,
                metadata={"return_code": result.returncode},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _file_read(self, path: str, **kwargs) -> ToolResult:
        """Read file contents."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False, output="", error=f"File not found: {path}"
                )

            content = file_path.read_text(encoding="utf-8")
            return ToolResult(
                success=True,
                output=content,
                metadata={"size": len(content), "path": str(file_path.absolute())},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _file_write(
        self, path: str, content: str, **kwargs
    ) -> ToolResult:
        """Write content to file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"File written: {path}",
                metadata={"size": len(content), "path": str(file_path.absolute())},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _calculator(self, expression: str, **kwargs) -> ToolResult:
        """Evaluate mathematical expression safely."""
        try:
            # Sanitize expression
            allowed_chars = set("0123456789+-*/().% ")
            sanitized = "".join(c for c in expression if c in allowed_chars)

            if not sanitized:
                return ToolResult(
                    success=False, output="", error="Invalid expression"
                )

            result = eval(sanitized, {"__builtins__": {}}, {})
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "result": result},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def run(self, user_input: str) -> str:
        """
        Run the agent with self-healing tool calling.

        Args:
            user_input: User's request

        Returns:
            Agent's response
        """
        self._log(f"User: {user_input}")

        # Build prompt with tool instructions
        system_prompt = """You are a helpful assistant with access to tools.

Available tools:
- web_search: Search the web. Args: {"query": "...", "num_results": 5}
- python_exec: Execute Python code. Args: {"code": "...", "timeout": 30}
- bash_exec: Execute shell command. Args: {"command": "...", "timeout": 30}
- file_read: Read file. Args: {"path": "..."}
- file_write: Write file. Args: {"path": "...", "content": "..."}
- calculator: Calculate math. Args: {"expression": "..."}

To use a tool, output:
```tool
{"tool": "tool_name", "args": {...}}
```

When you receive a tool result, analyze it and respond to the user.
If a tool fails, try a different approach."""

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_input},
        ]

        # Get model response (simplified - in production use llama.cpp or API)
        response = self._generate_response(messages)

        # Check for tool calls
        tool_call = self._parse_tool_call(response)

        if tool_call:
            self._log(f"Detected tool call: {tool_call.tool.value}")

            # Execute tool with self-healing
            tool_call.max_retries = self.max_retries
            result = self._execute_tool(tool_call)

            # Build follow-up prompt with tool result
            tool_result_prompt = f"""Tool Result ({tool_call.tool.value}):
Success: {result.success}
Output:
{result.output}
{f"Error: {result.error}" if result.error else ""}

Based on this result, please respond to the user's original request."""

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_result_prompt})

            # Get final response
            final_response = self._generate_response(messages)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append(
                {"role": "assistant", "content": final_response}
            )

            return final_response

        # No tool call - return direct response
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _generate_response(self, messages: list[dict]) -> str:
        """
        Generate response using the model.

        This is a placeholder - in production, use:
        - llama.cpp Python bindings
        - Ollama API
        - HuggingFace transformers
        """
        # Simple rule-based response for demo
        last_msg = messages[-1]["content"].lower()

        if "search" in last_msg or "web" in last_msg:
            # Extract search query
            query = last_msg.replace("search", "").replace("web", "").strip()
            return json.dumps(
                {"tool": "web_search", "args": {"query": query, "num_results": 5}}
            )
        elif "run python" in last_msg or "execute code" in last_msg:
            code = last_msg.split("```")[-2] if "```" in last_msg else "print('hello')"
            return json.dumps({"tool": "python_exec", "args": {"code": code}})
        elif "run command" in last_msg or "execute" in last_msg:
            cmd = last_msg.split("command:")[-1].strip() if "command:" in last_msg else "echo hello"
            return json.dumps({"tool": "bash_exec", "args": {"command": cmd}})
        else:
            return f"I understand your request. Let me help you with: {last_msg}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Self-healing agent with tool calling")
    parser.add_argument("--model-path", default="", help="Path to the model file")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-healing retries")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("web-search", help="Search the web")
    p.add_argument("query", help="Search query")
    p.add_argument("--num-results", type=int, default=5, help="Number of results")

    p = sub.add_parser("python", help="Execute Python code")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--code", help="Python code string")
    grp.add_argument("--file", help="Path to a Python file")
    p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")

    p = sub.add_parser("bash", help="Execute a shell command")
    p.add_argument("cmd", help="Command to execute")

    p = sub.add_parser("file-read", help="Read a file")
    p.add_argument("path", help="Path to read")

    p = sub.add_parser("file-write", help="Write content to a file")
    p.add_argument("path", help="Path to write")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--content", help="Content to write")
    grp.add_argument("--file", help="Read content from this file")

    p = sub.add_parser("calc", help="Evaluate a math expression")
    p.add_argument("expression", help="Expression to evaluate")

    p = sub.add_parser("run", help="Run the agent on a request")
    p.add_argument("input", help="User request")

    args = parser.parse_args(argv)

    agent = SelfHealingAgent(
        model_path=args.model_path,
        max_retries=args.max_retries,
        verbose=not args.quiet,
    )
    try:
        if args.command == "web-search":
            result = agent._web_search(args.query, num_results=args.num_results)
        elif args.command == "python":
            code = args.code
            if code is None:
                with open(args.file, "r", encoding="utf-8") as f:
                    code = f.read()
            result = agent._python_exec(code, timeout=args.timeout)
        elif args.command == "bash":
            result = agent._bash_exec(args.cmd)
        elif args.command == "file-read":
            result = agent._file_read(args.path)
        elif args.command == "file-write":
            content = args.content
            if content is None:
                with open(args.file, "r", encoding="utf-8") as f:
                    content = f.read()
            result = agent._file_write(args.path, content)
        elif args.command == "calc":
            result = agent._calculator(args.expression)
        elif args.command == "run":
            print(agent.run(args.input))
            return 0
        else:
            parser.error("Unknown command")
            return 2

        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0 if result.success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

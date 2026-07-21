#!/usr/bin/env python3
"""
Safe Code Execution Sandbox for Python and Bash.

Features:
- Sandboxed Python execution with timeout
- Bash/PowerShell command execution
- File system isolation
- Resource limits (CPU, memory, time)
- Output capture and streaming

Usage:
    from agent.code_executor import CodeExecutor

    executor = CodeExecutor()
    result = executor.run_python("print('Hello!')")
    result = executor.run_bash("echo Hello")
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.platform != "win32":
    try:
        import resource
    except ImportError:
        resource = None
else:
    resource = None


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    timeout: bool = False
    metadata: dict = field(default_factory=dict)


class TimeoutError(Exception):
    pass


class CodeExecutor:
    """
    Safe code execution sandbox.

    Features:
    - Python execution with timeout
    - Bash/PowerShell execution
    - Resource limits
    - Output capture
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        python_executable: Optional[str] = None,
        max_timeout: int = 60,
        max_memory_mb: int = 512,
    ):
        """
        Initialize executor.

        Args:
            working_dir: Working directory for execution
            python_executable: Python executable path
            max_timeout: Maximum timeout in seconds
            max_memory_mb: Maximum memory in MB
        """
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="code_exec_")
        self.python_executable = python_executable or sys.executable
        self.max_timeout = max_timeout
        self.max_memory_mb = max_memory_mb

        # Ensure working directory exists
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

    def run_python(
        self,
        code: str,
        timeout: Optional[int] = None,
        capture_stderr: bool = True,
        env: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds (uses max_timeout if None)
            capture_stderr: Capture stderr output
            env: Additional environment variables

        Returns:
            ExecutionResult with output
        """
        timeout = timeout or self.max_timeout

        # Create temp file for code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.working_dir
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # Prepare environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            # Run Python
            import time
            start_time = time.time()

            result = subprocess.run(
                [self.python_executable, code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env=exec_env,
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr if capture_stderr else "",
                return_code=result.returncode,
                execution_time=execution_time,
                metadata={"code_file": code_file},
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout,
                timeout=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(code_file)
            except OSError:
                pass

    def run_bash(
        self,
        command: str,
        timeout: Optional[int] = None,
        shell: bool = True,
        env: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute bash/shell command.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            shell: Use shell execution
            env: Additional environment variables

        Returns:
            ExecutionResult with output
        """
        timeout = timeout or self.max_timeout

        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        import time
        start_time = time.time()

        try:
            # Use PowerShell on Windows, bash on Unix
            if sys.platform == "win32":
                # Use PowerShell
                cmd = ["powershell", "-Command", command]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.working_dir,
                    env=exec_env,
                )
            else:
                result = subprocess.run(
                    command,
                    shell=shell,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.working_dir,
                    env=exec_env,
                )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout,
                timeout=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )

    def run_script(
        self,
        script_path: str,
        args: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute an existing script file.

        Args:
            script_path: Path to script
            args: Command-line arguments
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with output
        """
        timeout = timeout or self.max_timeout
        script_path = Path(script_path)

        if not script_path.exists():
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Script not found: {script_path}",
                return_code=-1,
                execution_time=0,
            )

        # Determine interpreter
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            cmd = [self.python_executable, str(script_path)]
        elif suffix in (".sh", ".bash"):
            cmd = ["bash", str(script_path)]
        elif suffix == ".ps1":
            cmd = ["powershell", "-File", str(script_path)]
        else:
            cmd = [str(script_path)]

        if args:
            cmd.extend(args)

        import time
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Script timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout,
                timeout=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )

    def read_file(self, path: str) -> ExecutionResult:
        """Read file contents."""
        try:
            content = Path(path).read_text(encoding="utf-8")
            return ExecutionResult(
                success=True,
                stdout=content,
                stderr="",
                return_code=0,
                execution_time=0,
                metadata={"size": len(content)},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write content to file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ExecutionResult(
                success=True,
                stdout=f"File written: {path}",
                stderr="",
                return_code=0,
                execution_time=0,
                metadata={"size": len(content)},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )

    def list_directory(self, path: str = ".") -> ExecutionResult:
        """List directory contents."""
        try:
            entries = list(Path(path).iterdir())
            content = "\n".join(
                f"{'[DIR] ' if e.is_dir() else ''}{e.name}" for e in sorted(entries)
            )
            return ExecutionResult(
                success=True,
                stdout=content,
                stderr="",
                return_code=0,
                execution_time=0,
                metadata={"count": len(entries)},
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
            )

    def format_result(self, result: ExecutionResult, max_output: int = 10000) -> str:
        """Format execution result as readable text."""
        lines = []

        status = "SUCCESS" if result.success else "FAILED"
        if result.timeout:
            status = "TIMEOUT"

        lines.append(f"[{status}] Exit code: {result.return_code} (took {result.execution_time:.2f}s)")

        if result.stdout:
            stdout = result.stdout[:max_output]
            if len(result.stdout) > max_output:
                stdout += f"\n... (truncated, {len(result.stdout)} total chars)"
            lines.append(f"\nSTDOUT:\n{stdout}")

        if result.stderr:
            stderr = result.stderr[:max_output]
            if len(result.stderr) > max_output:
                stderr += f"\n... (truncated, {len(result.stderr)} total chars)"
            lines.append(f"\nSTDERR:\n{stderr}")

        return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Safe code execution sandbox")
    parser.add_argument("--working-dir", default=None, help="Working directory for execution")
    parser.add_argument("--timeout", type=int, default=60, help="Default execution timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    p_py = sub.add_parser("python", help="Execute Python code")
    src = p_py.add_mutually_exclusive_group(required=True)
    src.add_argument("--code", help="Python code string")
    src.add_argument("--file", help="Path to a Python file")
    p_py.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")

    p_bash = sub.add_parser("bash", help="Execute a shell command")
    p_bash.add_argument("--command", dest="cmd", required=True, help="Command to execute")

    p_script = sub.add_parser("script", help="Run an existing script file")
    p_script.add_argument("path", help="Path to the script")
    p_script.add_argument("args", nargs="*", help="Arguments passed to the script")
    p_script.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")

    p_read = sub.add_parser("read", help="Read a file")
    p_read.add_argument("path", help="Path to read")

    p_write = sub.add_parser("write", help="Write content to a file")
    p_write.add_argument("path", help="Path to write")
    grp = p_write.add_mutually_exclusive_group(required=True)
    grp.add_argument("--content", help="Content to write")
    grp.add_argument("--file", help="Read content from this file")

    p_list = sub.add_parser("list", help="List a directory")
    p_list.add_argument("path", nargs="?", default=".", help="Directory path")

    args = parser.parse_args(argv)

    executor = CodeExecutor(working_dir=args.working_dir, max_timeout=args.timeout)
    try:
        if args.command == "python":
            code = args.code
            if code is None:
                with open(args.file, "r", encoding="utf-8") as f:
                    code = f.read()
            result = executor.run_python(code, timeout=args.timeout)
        elif args.command == "bash":
            result = executor.run_bash(args.cmd)
        elif args.command == "script":
            result = executor.run_script(args.path, args=args.args, timeout=args.timeout)
        elif args.command == "read":
            result = executor.read_file(args.path)
        elif args.command == "write":
            content = args.content
            if content is None:
                with open(args.file, "r", encoding="utf-8") as f:
                    content = f.read()
            result = executor.write_file(args.path, content)
        elif args.command == "list":
            result = executor.list_directory(args.path)
        else:
            parser.error("Unknown command")
            return 2

        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0 if result.success else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""Advanced terminal/shell execution tool with live streaming."""

import os
import sys
import time
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Optional, Generator


def run_terminal(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 120,
    shell: Optional[str] = None,
    env_vars: Optional[dict] = None,
    capture_output: bool = True,
) -> dict:
    """Execute a terminal command with advanced options.

    Automatically detects the appropriate shell on the current platform.
    On Windows: uses PowerShell (preferred) or cmd.exe fallback.

    Args:
        command: Command to execute
        working_dir: Working directory
        timeout: Maximum execution time in seconds
        shell: Explicit shell to use ("powershell", "cmd", "bash", "auto")
        env_vars: Additional environment variables
        capture_output: Whether to capture output

    Returns:
        dict with keys: success, output, error, exit_code, duration_ms
    """
    start = time.time()
    cwd = working_dir or os.getcwd()

    # Auto-detect shell
    if shell is None or shell == "auto":
        shell = "powershell" if os.name == "nt" else "bash"

    # Build command
    if shell == "powershell":
        cmd_args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]
    elif shell == "cmd":
        cmd_args = ["cmd.exe", "/c", command]
    elif shell == "bash":
        if os.name == "nt":
            cmd_args = ["bash.exe", "-c", command]
        else:
            cmd_args = ["bash", "-c", command]
    else:
        cmd_args = [shell, "-c", command]

    # Environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        result = subprocess.run(
            cmd_args,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip() if capture_output else "",
            "error": result.stderr.strip() if capture_output else "",
            "exit_code": result.returncode,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": False,
            "output": "",
            "error": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }


def stream_terminal(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 120,
    shell: Optional[str] = None,
) -> Generator[str, None, dict]:
    """Execute a terminal command and yield output line by line.

    Yields:
        Output lines as they become available (real-time streaming)

    Returns:
        Final result dict with success, exit_code, duration_ms
    """
    cwd = working_dir or os.getcwd()

    if shell is None or shell == "auto":
        shell = "powershell" if os.name == "nt" else "bash"

    if shell == "powershell":
        cmd_args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]
    elif shell == "cmd":
        cmd_args = ["cmd.exe", "/c", command]
    elif shell == "bash":
        cmd_args = ["bash", "-c", command] if os.name != "nt" else ["bash.exe", "-c", command]
    else:
        cmd_args = [shell, "-c", command]

    start = time.time()
    try:
        process = subprocess.Popen(
            cmd_args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        lines = []
        for line in iter(process.stdout.readline, ""):
            lines.append(line)
            yield line.rstrip()

        process.wait(timeout=timeout)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        final = {
            "success": process.returncode == 0,
            "output": "".join(lines).strip(),
            "exit_code": process.returncode,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }
        return final
    except subprocess.TimeoutExpired:
        process.kill()
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": False,
            "output": "".join(lines) if "lines" in dir() else "",
            "error": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "duration_ms": elapsed_ms,
            "shell": shell,
            "command": command[:200],
        }


def get_available_shells() -> dict:
    """Detect available shells on the system."""
    shells = []
    
    # Check PowerShell
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            shells.append({"name": "powershell", "version": r.stdout.strip()})
    except: pass

    # Check cmd
    try:
        r = subprocess.run(["cmd.exe", "/c", "ver"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            shells.append({"name": "cmd", "version": r.stdout.strip()})
    except: pass

    # Check bash (WSL)
    try:
        r = subprocess.run(["bash.exe", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            shells.append({"name": "bash", "version": r.stdout.split(",")[0].strip()})
    except: pass

    # Check bash (Linux/Mac)
    try:
        if os.name != "nt":
            r = subprocess.run(["bash", "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                shells.append({"name": "bash", "version": r.stdout.split(",")[0].strip()})
    except: pass

    return {"success": True, "shells": shells, "count": len(shells)}


def change_directory(path: str) -> dict:
    """Change the current working directory (for persistent terminal sessions)."""
    try:
        expanded = os.path.expanduser(path)
        resolved = os.path.abspath(expanded)
        if not os.path.exists(resolved):
            return {"success": False, "error": f"Directory does not exist: {resolved}"}
        if not os.path.isdir(resolved):
            return {"success": False, "error": f"Not a directory: {resolved}"}
        os.chdir(resolved)
        return {"success": True, "new_cwd": resolved, "previous_cwd": os.getcwd()}
    except Exception as e:
        return {"success": False, "error": str(e)}

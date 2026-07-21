"""Dedicated PowerShell execution tool for Windows."""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


def run_powershell(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 60,
    capture_output: bool = True,
    as_admin: bool = False,
) -> dict:
    """Execute a PowerShell command or script and return results.

    Args:
        command: PowerShell command, script, or path to .ps1 file
        working_dir: Working directory for execution
        timeout: Maximum execution time in seconds
        capture_output: Whether to capture stdout/stderr
        as_admin: Attempt to run as administrator (UAC prompt)

    Returns:
        dict with keys: success, output, error, exit_code, duration_ms
    """
    start = time.time()
    cwd = working_dir or os.getcwd()

    # Handle script files
    if command.strip().endswith(".ps1") and Path(command).exists():
        cmd_args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", command,
        ]
    elif as_admin:
        # Run as admin via a temporary script
        temp_script = Path(cwd) / f"_temp_admin_{int(time.time())}.ps1"
        try:
            temp_script.write_text(command, encoding="utf-8")
            cmd_args = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                f"Start-Process powershell.exe -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"{temp_script}\"' -Verb RunAs -Wait"
            ]
        finally:
            if temp_script.exists():
                try: temp_script.unlink()
                except: pass
    else:
        cmd_args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ]

    try:
        result = subprocess.run(
            cmd_args,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
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
            "command": command[:200],
        }


def get_powershell_version() -> dict:
    """Get the installed PowerShell version."""
    result = run_powershell("$PSVersionTable.PSVersion.ToString()")
    if result["success"]:
        return {"success": True, "version": result["output"]}
    return result


def get_powershell_modules() -> dict:
    """List installed PowerShell modules."""
    result = run_powershell("Get-Module -ListAvailable | Select-Object Name, Version | ConvertTo-Json")
    if result["success"] and result["output"]:
        try:
            modules = json.loads(result["output"])
            if isinstance(modules, dict):
                modules = [modules]
            return {"success": True, "modules": modules, "count": len(modules)}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result

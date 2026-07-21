"""Windows-specific system control tools (services, registry, events, network)."""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


def _run_powershell(cmd: str, timeout: int = 30) -> dict:
    """Helper to run a PowerShell command."""
    start = time.time()
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
            "exit_code": result.returncode,
            "duration_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": f"Timed out after {timeout}s", "exit_code": -1, "duration_ms": round((time.time() - start) * 1000, 2)}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e), "exit_code": -1, "duration_ms": round((time.time() - start) * 1000, 2)}


# ── Service Management ──

def list_services(status: Optional[str] = None) -> dict:
    """List Windows services.

    Args:
        status: Filter by status ("Running", "Stopped", or None for all)

    Returns:
        dict with list of services
    """
    filter_cmd = ""
    if status:
        filter_cmd = f" | Where-Object {{ $_.Status -eq '{status}' }}"
    
    cmd = f"Get-Service{filter_cmd} | Select-Object Name, DisplayName, Status, StartType | ConvertTo-Json"
    result = _run_powershell(cmd)
    
    if result["success"] and result["output"]:
        try:
            services = json.loads(result["output"])
            if isinstance(services, dict):
                services = [services]
            return {"success": True, "services": services, "count": len(services)}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result


def get_service(name: str) -> dict:
    """Get details of a specific Windows service.

    Args:
        name: Service name

    Returns:
        dict with service details
    """
    cmd = f"Get-Service -Name '{name}' | Select-Object Name, DisplayName, Status, StartType, ServiceType | ConvertTo-Json"
    result = _run_powershell(cmd)
    if result["success"] and result["output"]:
        try:
            svc = json.loads(result["output"])
            return {"success": True, "service": svc}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result


def start_service(name: str) -> dict:
    """Start a Windows service.

    Args:
        name: Service name

    Returns:
        dict with result
    """
    cmd = f"Start-Service -Name '{name}' -PassThru | Select-Object Name, Status | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=60)
    return {"success": result["success"], "service": name, "action": "start", "details": result.get("output", ""), "error": result.get("error", "")}


def stop_service(name: str) -> dict:
    """Stop a Windows service.

    Args:
        name: Service name

    Returns:
        dict with result
    """
    cmd = f"Stop-Service -Name '{name}' -PassThru | Select-Object Name, Status | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=60)
    return {"success": result["success"], "service": name, "action": "stop", "details": result.get("output", ""), "error": result.get("error", "")}


def restart_service(name: str) -> dict:
    """Restart a Windows service.

    Args:
        name: Service name

    Returns:
        dict with result
    """
    cmd = f"Restart-Service -Name '{name}' -PassThru | Select-Object Name, Status | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=60)
    return {"success": result["success"], "service": name, "action": "restart", "details": result.get("output", ""), "error": result.get("error", "")}


# ── Process Management ──

def kill_process(name_or_pid: str) -> dict:
    """Kill a process by name or PID.

    Args:
        name_or_pid: Process name (e.g., "notepad.exe") or PID number

    Returns:
        dict with result
    """
    if name_or_pid.isdigit():
        cmd = f"Stop-Process -Id {name_or_pid} -Force -PassThru | Select-Object Id, ProcessName | ConvertTo-Json"
    else:
        cmd = f"Stop-Process -Name '{name_or_pid}' -Force -PassThru | Select-Object Id, ProcessName | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=30)
    return {"success": result["success"], "target": name_or_pid, "action": "kill", "details": result.get("output", ""), "error": result.get("error", "")}


def start_process(path: str, arguments: str = "") -> dict:
    """Start a new process.

    Args:
        path: Path to executable
        arguments: Command-line arguments

    Returns:
        dict with result
    """
    if arguments:
        cmd = f"Start-Process -FilePath '{path}' -ArgumentList '{arguments}' -PassThru | Select-Object Id, ProcessName | ConvertTo-Json"
    else:
        cmd = f"Start-Process -FilePath '{path}' -PassThru | Select-Object Id, ProcessName | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=30)
    return {"success": result["success"], "path": path, "action": "start", "details": result.get("output", ""), "error": result.get("error", "")}


# ── Windows Registry ──

def read_registry(key: str, value: Optional[str] = None) -> dict:
    """Read a Windows Registry key or value.

    Args:
        key: Registry path (e.g., 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion')
        value: Specific value name (None = list all values)

    Returns:
        dict with registry data
    """
    if value:
        cmd = f"Get-ItemProperty -Path '{key}' -Name '{value}' | ConvertTo-Json"
    else:
        cmd = f"Get-ItemProperty -Path '{key}' | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=15)
    if result["success"] and result["output"]:
        try:
            data = json.loads(result["output"])
            return {"success": True, "key": key, "value": value, "data": data}
        except json.JSONDecodeError:
            return {"success": True, "key": key, "value": value, "raw": result["output"]}
    return result


def write_registry(key: str, name: str, value: str, type: str = "String") -> dict:
    """Write a value to the Windows Registry.

    Args:
        key: Registry path
        name: Value name
        value: Value data
        type: Value type (String, DWord, QWord, Binary, ExpandString, MultiString)

    Returns:
        dict with result
    """
    cmd = f"New-ItemProperty -Path '{key}' -Name '{name}' -Value '{value}' -PropertyType '{type}' -Force | Out-Null; Write-Output 'OK'"
    result = _run_powershell(cmd, timeout=15)
    return {"success": result["success"], "key": key, "name": name, "action": "write", "error": result.get("error", "")}


# ── Windows Event Log ──

def get_event_log(log_name: str = "System", max_events: int = 50, level: Optional[str] = None) -> dict:
    """Read Windows Event Log entries.

    Args:
        log_name: Log name (System, Application, Security, PowerShell)
        max_events: Maximum entries to return
        level: Filter by level (Error, Warning, Information, or None for all)

    Returns:
        dict with event log entries
    """
    filter_cmd = ""
    if level:
        filter_cmd = f" | Where-Object {{ $_.LevelDisplayName -eq '{level}' }}"
    
    cmd = (
        f"Get-WinEvent -LogName '{log_name}' -MaxEvents {max_events}{filter_cmd} "
        f"| Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message "
        f"| ConvertTo-Json"
    )
    result = _run_powershell(cmd, timeout=30)
    if result["success"] and result["output"]:
        try:
            events = json.loads(result["output"])
            if isinstance(events, dict):
                events = [events]
            return {"success": True, "log": log_name, "events": events, "count": len(events)}
        except json.JSONDecodeError:
            return {"success": True, "log": log_name, "raw": result["output"]}
    return result


# ── Network Configuration ──

def get_network_config() -> dict:
    """Get network configuration information.

    Returns:
        dict with network adapters, IP config, DNS
    """
    cmd = (
        "Get-NetAdapter | Select-Object Name, Status, LinkSpeed, MacAddress | ConvertTo-Json; "
        "Write-Output '---SEPARATOR---'; "
        "Get-NetIPAddress -AddressFamily IPv4 | Select-Object InterfaceAlias, IPAddress, PrefixLength | ConvertTo-Json"
    )
    result = _run_powershell(cmd, timeout=15)
    if result["success"] and result["output"]:
        parts = result["output"].split("---SEPARATOR---")
        adapters_data = parts[0].strip() if len(parts) > 0 else "[]"
        ip_data = parts[1].strip() if len(parts) > 1 else "[]"
        try:
            adapters = json.loads(adapters_data) if adapters_data else []
            ip_addrs = json.loads(ip_data) if ip_data else []
            if isinstance(adapters, dict): adapters = [adapters]
            if isinstance(ip_addrs, dict): ip_addrs = [ip_addrs]
            return {"success": True, "adapters": adapters, "ip_addresses": ip_addrs}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result


def test_network(target: str = "8.8.8.8") -> dict:
    """Test network connectivity (ping).

    Args:
        target: Hostname or IP to ping

    Returns:
        dict with ping results
    """
    cmd = f"Test-Connection -ComputerName '{target}' -Count 4 | Select-Object Address, ResponseTime, Status | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=30)
    if result["success"] and result["output"]:
        try:
            pings = json.loads(result["output"])
            if isinstance(pings, dict):
                pings = [pings]
            avg_time = sum(p.get("ResponseTime", 0) for p in pings if p.get("ResponseTime")) / max(len([p for p in pings if p.get("ResponseTime")]), 1)
            return {"success": True, "target": target, "pings": pings, "avg_response_ms": round(avg_time, 1), "packets_sent": 4, "packets_received": len(pings)}
        except json.JSONDecodeError:
            return {"success": True, "target": target, "raw": result["output"]}
    return result


# ── Scheduled Tasks ──

def list_scheduled_tasks(folder: str = "\\") -> dict:
    """List Windows Scheduled Tasks.

    Args:
        folder: Task folder path (e.g., "\\Microsoft\\Windows\\")

    Returns:
        dict with list of tasks
    """
    cmd = f"Get-ScheduledTask -TaskPath '{folder}' | Select-Object TaskName, TaskPath, State | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=15)
    if result["success"] and result["output"]:
        try:
            tasks = json.loads(result["output"])
            if isinstance(tasks, dict):
                tasks = [tasks]
            return {"success": True, "folder": folder, "tasks": tasks, "count": len(tasks)}
        except json.JSONDecodeError:
            return {"success": True, "folder": folder, "raw": result["output"]}
    return result


def start_scheduled_task(task_name: str) -> dict:
    """Start a Windows Scheduled Task.

    Args:
        task_name: Full task name (e.g., "\\Microsoft\\Windows\\TaskScheduler\\Task")

    Returns:
        dict with result
    """
    cmd = f"Start-ScheduledTask -TaskName '{task_name}' -PassThru | Select-Object TaskName, State | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=30)
    return {"success": result["success"], "task": task_name, "action": "start", "details": result.get("output", ""), "error": result.get("error", "")}


# ── Windows Updates ──

def check_windows_updates() -> dict:
    """Check for available Windows Updates.

    Returns:
        dict with available updates
    """
    cmd = (
        "$Session = New-Object -ComObject Microsoft.Update.Session; "
        "$Searcher = $Session.CreateUpdateSearcher(); "
        "$Result = $Searcher.Search('IsInstalled=0'); "
        "$Result.Updates | Select-Object Title, Description, IsDownloaded, IsMandatory, AutoSelectOnWebSites | ConvertTo-Json"
    )
    result = _run_powershell(cmd, timeout=60)
    if result["success"] and result["output"]:
        try:
            updates = json.loads(result["output"])
            if isinstance(updates, dict):
                updates = [updates]
            return {"success": True, "updates": updates, "count": len(updates)}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result


# ── System Information ──

def get_windows_system_info() -> dict:
    """Get comprehensive Windows system information.

    Returns:
        dict with system info
    """
    cmd = (
        "Get-ComputerInfo | Select-Object "
        "WindowsVersion, WindowsBuildLabEx, OsArchitecture, OsName, OsVersion, "
        "BiosManufacturer, BiosSMBIOSBIOSVersion, "
        "CsManufacturer, CsModel, CsProcessors, CsTotalPhysicalMemory "
        "| ConvertTo-Json"
    )
    result = _run_powershell(cmd, timeout=15)
    if result["success"] and result["output"]:
        try:
            info = json.loads(result["output"])
            return {"success": True, "system_info": info}
        except json.JSONDecodeError:
            return {"success": True, "raw": result["output"]}
    return result


def get_disk_info() -> dict:
    """Get disk/drive information.

    Returns:
        dict with disk information
    """
    cmd = "Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root, Used, Free | ConvertTo-Json"
    result = _run_powershell(cmd, timeout=10)
    if result["success"] and result["output"]:
        try:
            drives = json.loads(result["output"])
            if isinstance(drives, dict):
                drives = [drives]
            formatted = []
            for d in drives:
                free_gb = round(d.get("Free", 0) / (1024**3), 2) if d.get("Free") else 0
                used_gb = round(d.get("Used", 0) / (1024**3), 2) if d.get("Used") else 0
                total_gb = round(free_gb + used_gb, 2)
                formatted.append({
                    "drive": d.get("Name", ""),
                    "root": d.get("Root", ""),
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "used_percent": round((used_gb / total_gb * 100), 1) if total_gb > 0 else 0,
                })
            return {"success": True, "drives": formatted}
        except (json.JSONDecodeError, KeyError) as e:
            return {"success": True, "raw": result["output"]}
    return result

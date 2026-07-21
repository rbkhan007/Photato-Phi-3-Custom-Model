#!/usr/bin/env python3
"""
OS-level CPU utilization limiter for model workloads.

Provides :func:`limit_cpu`, which caps the *current process* to a target
percentage of CPU bandwidth using the proper operating-system "control
signal" for CPU scheduling:

- **Windows**: a Job Object with a *CPU rate hard cap*
  (``JobObjectCpuRateHardCapLimit``). This is the native, kernel-enforced
  way to throttle a process to e.g. 55% of all CPU time, regardless of how
  many cores it spins up.
- **Fallback (all platforms)**: lower the process priority class and budget
  the threading (``n_threads`` / ``torch.set_num_threads``) so the workload
  cannot saturate the machine.

The limiter is idempotent: calling it multiple times (e.g. once per engine)
only ever installs a single Job Object and re-budgets threads.

Typical use::

    from optimization.cpu_throttle import limit_cpu

    threads = limit_cpu(55.0)          # cap this process at 55% CPU
    model = Llama(..., n_threads=threads)
"""

import logging
import math
import os
import sys

log = logging.getLogger(__name__)


def physical_cores() -> int:
    """Return the number of physical CPU cores (falls back to logical/4)."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


def recommended_threads(percent: float) -> int:
    """Threads to use so a workload fits within ``percent`` of CPU.

    Uses ``ceil`` so a 4-core machine at 55% yields 3 threads
    (``ceil(4 * 0.55) = 3``), leaving one core free for the OS.
    """
    percent = max(1.0, min(100.0, float(percent)))
    return max(1, math.ceil(physical_cores() * percent / 100.0))


# ---------------------------------------------------------------------------
# Windows Job Object hard CPU-rate cap
# ---------------------------------------------------------------------------
_JOB_HANDLE = None  # kept alive so the cap is not released


def _apply_windows_job_cap(percent: float) -> None:
    """Install a Job Object that hard-caps CPU bandwidth to ``percent``.

    Raises only if the cap genuinely cannot be configured; if the process is
    already inside another job (common when launched from a managed shell),
    it returns silently and the caller's priority/thread fallback applies.
    The cap is expressed in 1/100 of a percent (55% -> CpuRate = 5500).
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    class JOBOBJECT_CPU_RATE_LIMIT(ctypes.Structure):
        _fields_ = [
            ("ControlFlags", wintypes.DWORD),
            ("CpuRate", wintypes.DWORD),
            ("Weight", wintypes.DWORD),
            ("UserPolicy", wintypes.DWORD),
            ("KernelPolicy", wintypes.DWORD),
        ]

    JOB_OBJECT_CPU_RATE_HARD_CAP = 0x1
    # Some Windows builds expose the hard-cap class as 0x2A; older ones as 0x29.
    # Try both before giving up.
    JOB_CLASSES = (0x2A, 0x29)

    cpu_rate = int(max(1.0, min(100.0, float(percent))) * 100)
    info = JOBOBJECT_CPU_RATE_LIMIT()
    info.ControlFlags = JOB_OBJECT_CPU_RATE_HARD_CAP
    info.CpuRate = cpu_rate
    ctypes_size = ctypes.sizeof(info)

    last_err = 0
    for job_class in JOB_CLASSES:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError()
        try:
            if not kernel32.SetInformationJobObject(job, job_class, ctypes.byref(info), ctypes_size):
                last_err = kernel32.GetLastError()
                continue
            if kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
                # Persist so the job (and therefore the cap) stays alive.
                global _JOB_HANDLE
                _JOB_HANDLE = job
                return
            # Process already belongs to another job -> can't apply hard cap.
            last_err = kernel32.GetLastError()
            return
        finally:
            pass
    if last_err:
        raise ctypes.WinError(last_err)


def _set_low_priority(process=None) -> None:
    """Lower the process priority class to reduce CPU contention."""
    try:
        import psutil
    except Exception:
        psutil = None  # type: ignore[assignment]

    if psutil is None:
        if hasattr(os, "nice"):
            try:
                os.nice(10)
            except Exception:
                pass
        return

    proc = process or psutil.Process()
    try:
        if sys.platform == "win32":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def limit_cpu(percent: float = 55.0, apply_torch: bool = True, process=None) -> int:
    """Cap the current process' CPU usage to ``percent`` (default 55%).

    Uses the proper OS CPU-control signal (a Windows Job Object hard cap when
    available) and falls back to process priority + thread budgeting. Also
    constrains ``torch`` thread pools so transformers/training workloads obey
    the same budget.

    Args:
        percent: Target CPU bandwidth as a percentage (1-100). Values >= 100
            are treated as "no limit" (full speed, normal priority).
        apply_torch: Also call ``torch.set_num_threads`` to the capped budget.
        process: Optional ``psutil.Process`` to throttle (defaults to self).

    Returns:
        Recommended thread count (``n_threads``) for model engines.
    """
    percent = float(percent)
    if percent >= 100.0:
        # No limit requested: use all physical cores at normal priority.
        threads = physical_cores()
        try:
            _set_normal_priority(process)
        except Exception:
            pass
        if apply_torch:
            try:
                import torch

                torch.set_num_threads(threads)
            except Exception:
                pass
        log.info("CPU limit disabled (>=100%%); using %d threads", threads)
        return threads

    threads = recommended_threads(percent)
    applied = []

    if sys.platform == "win32" and _JOB_HANDLE is None:
        try:
            _apply_windows_job_cap(percent)
            applied.append("windows_job_object")
        except Exception as e:  # pragma: no cover - environment dependent
            log.info("Windows CPU hard cap unavailable (%s); using fallback", e)

    try:
        _set_low_priority(process)
        applied.append("priority")
    except Exception:
        pass

    if apply_torch:
        try:
            import torch

            torch.set_num_threads(threads)
            applied.append("torch_threads")
        except Exception:
            pass

    log.info(
        "CPU limit %.0f%% -> ~%d threads (applied: %s)",
        percent,
        threads,
        ", ".join(applied) or "none",
    )
    return threads


def _set_normal_priority(process=None) -> None:
    try:
        import psutil
    except Exception:
        return
    try:
        proc = process or psutil.Process()
        if sys.platform == "win32":
            proc.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(0)
    except Exception:
        pass

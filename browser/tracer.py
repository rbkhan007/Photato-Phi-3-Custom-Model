"""
Browser Console & Network Tracer.

Captures every trace signal from a live browser session:
- console messages (log, info, warning, error, debug)
- uncaught page errors / exceptions
- failed network requests (4xx, 5xx, aborted, timeout)
- performance / slow resources

This is the "eyes and ears" layer that feeds the auto-debugger.
Works with Playwright (full event hooks) and Selenium / requests (best effort).
"""

import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class TraceLevel(Enum):
    LOG = "log"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    EXCEPTION = "exception"
    NETWORK = "network"


@dataclass
class TraceEntry:
    """A single captured browser signal."""
    level: TraceLevel
    text: str
    source: str = "browser"          # console | pageerror | network
    url: str = ""
    line: Optional[int] = None
    column: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "line": self.line,
            "column": self.column,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ConsoleTracer:
    """
    Attach to a browser page and record all console / error / network signals.

    Usage:
        tracer = ConsoleTracer()
        await tracer.attach(page)            # Playwright page object
        result = await browser.navigate(url)
        report = tracer.get_report()
        print(report["errors"])              # all console.error + exceptions
    """

    def __init__(self, capture_network: bool = True, slow_threshold_ms: int = 2000):
        self.entries: list[TraceEntry] = []
        self.capture_network = capture_network
        self.slow_threshold_ms = slow_threshold_ms
        self._attached = False
        self._handlers = []

    # ---------- Playwright attachment ----------

    async def attach(self, page) -> bool:
        """Attach event listeners to a Playwright page."""
        try:
            page.on("console", self._on_console)
            page.on("pageerror", self._on_pageerror)
            if self.capture_network:
                page.on("requestfailed", self._on_request_failed)
                page.on("response", self._on_response)
            self._attached = True
            return True
        except Exception:
            # Not a Playwright page or listeners unsupported
            self._attached = False
            return False

    def _on_console(self, msg):
        level = (msg.type or "log").lower()
        level_map = {
            "log": TraceLevel.LOG, "info": TraceLevel.INFO,
            "warning": TraceLevel.WARNING, "error": TraceLevel.ERROR,
            "debug": TraceLevel.DEBUG,
        }
        entry = TraceEntry(
            level=level_map.get(level, TraceLevel.LOG),
            text=msg.text,
            source="console",
            url=getattr(msg, "location", {}).get("url", "") if hasattr(msg, "location") else "",
            line=getattr(msg, "location", {}).get("line_number") if hasattr(msg, "location") else None,
        )
        # Playwright location is a dict-like on some versions
        try:
            loc = getattr(msg, "location", None)
            if isinstance(loc, dict):
                entry.url = loc.get("url", "")
                entry.line = loc.get("line_number")
                entry.column = loc.get("column_number")
        except Exception:
            pass
        self.entries.append(entry)

    def _on_pageerror(self, exc):
        text = str(exc) if exc else "Unknown page error"
        entry = TraceEntry(level=TraceLevel.EXCEPTION, text=text, source="pageerror")
        if isinstance(exc, Exception) and hasattr(exc, "__traceback__"):
            tb = exc.__traceback__
            if tb is not None and tb.tb_frame is not None:
                entry.url = tb.tb_frame.f_code.co_filename
                entry.line = tb.tb_lineno
        self.entries.append(entry)

    def _on_request_failed(self, request):
        failure = getattr(request, "failure", None)
        reason = failure.get("errorText", "request failed") if isinstance(failure, dict) else "request failed"
        entry = TraceEntry(
            level=TraceLevel.NETWORK,
            text=f"Request failed: {request.url} ({reason})",
            source="network",
            url=request.url,
            metadata={"reason": reason, "status": None},
        )
        self.entries.append(entry)

    def _on_response(self, response):
        status = response.status
        if status is None:
            return
        if status >= 400:
            entry = TraceEntry(
                level=TraceLevel.ERROR if status >= 500 else TraceLevel.WARNING,
                text=f"HTTP {status}: {response.url}",
                source="network",
                url=response.url,
                metadata={"status": status},
            )
            self.entries.append(entry)
        elif self.capture_network and getattr(response, "request", None) is not None:
            req = response.request
            dur = getattr(req, "timing", None)
            if dur is not None and isinstance(dur, dict):
                try:
                    total = (dur.get("responseEnd", 0) or 0) - (dur.get("requestStart", 0) or 0)
                    if total > self.slow_threshold_ms:
                        self.entries.append(TraceEntry(
                            level=TraceLevel.WARNING,
                            text=f"Slow resource ({int(total)}ms): {response.url}",
                            source="network",
                            url=response.url,
                            metadata={"duration_ms": total},
                        ))
                except Exception:
                    pass

    # ---------- Selenium / generic fallback ----------

    def record_manual(self, level: TraceLevel, text: str, url: str = "", line: int = None):
        """Manually record a signal (used for Selenium/requests backends)."""
        self.entries.append(TraceEntry(level=level, text=text, source="manual", url=url, line=line))

    # ---------- Queries ----------

    def get_errors(self) -> list[TraceEntry]:
        return [e for e in self.entries if e.level in (TraceLevel.ERROR, TraceLevel.EXCEPTION)]

    def get_warnings(self) -> list[TraceEntry]:
        return [e for e in self.entries if e.level == TraceLevel.WARNING]

    def get_network_issues(self) -> list[TraceEntry]:
        return [e for e in self.entries if e.source == "network"]

    def count_by_level(self) -> dict:
        out = {}
        for e in self.entries:
            out[e.level.value] = out.get(e.level.value, 0) + 1
        return out

    def get_report(self) -> dict:
        return {
            "total": len(self.entries),
            "by_level": self.count_by_level(),
            "errors": [e.to_dict() for e in self.get_errors()],
            "warnings": [e.to_dict() for e in self.get_warnings()],
            "network": [e.to_dict() for e in self.get_network_issues()],
        }

    def clear(self):
        self.entries.clear()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([e.to_dict() for e in self.entries], indent=indent, default=str)

    # ---------- Integration with BrowserAutomation ----------

    async def trace_session(self, browser, url: str, wait: int = 1500) -> dict:
        """
        Convenience: navigate a BrowserAutomation instance to url,
        capture traces, and return the report.
        """
        page = getattr(browser, "page", None)
        if page is not None:
            await self.attach(page)
        await browser.navigate(url)
        if wait > 0:
            await asyncio.sleep(wait / 1000.0)
        return self.get_report()

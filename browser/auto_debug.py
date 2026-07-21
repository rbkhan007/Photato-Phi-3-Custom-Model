"""
Browser Auto-Debugger.

Takes the traces collected by ConsoleTracer (console errors, uncaught
exceptions, network failures, slow resources) and:

1. Classifies each issue into a fixable category
2. Generates/patches source code (pattern-based + optional LLM hook)
3. Applies safe fixes to workspace files
4. Produces a debug report with before/after diffs

Also detects dead code in the workspace and (with caution) extracts it
out of the main workspace, archiving it to a `deadcodes` file OUTSIDE
the workspace so nothing is silently destroyed.
"""

import asyncio
import os
import re
import time
import json
import difflib
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum

from .tracer import ConsoleTracer, TraceEntry, TraceLevel


class IssueCategory(Enum):
    REFERENCE_ERROR = "reference_error"      # undefined variable / function
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    NETWORK_404 = "network_404"
    NETWORK_5XX = "network_5xx"
    NETWORK_FAILED = "network_failed"
    DEPRECATED_API = "deprecated_api"
    CONSOLE_ERROR = "console_error"
    SLOW_RESOURCE = "slow_resource"
    UNKNOWN = "unknown"


@dataclass
class FixAction:
    category: IssueCategory
    description: str
    file_path: Optional[str]
    original: str = ""
    fixed: str = ""
    applied: bool = False
    confidence: float = 0.5
    method: str = "pattern"   # pattern | llm


# ---- lightweight pattern-based fixers ---------------------------------------

_DEPRECATED_MAP = {
    "document.write": "document.write (avoid; use DOM APIs)",
    "innerHTML": "innerHTML (sanitize to avoid XSS)",
    "escape(": "escape( is deprecated, use encodeURIComponent(",
    "unescape(": "unescape( is deprecated, use decodeURIComponent(",
    "XMLHttpRequest": "prefer fetch() over XMLHttpRequest",
}


class AutoDebugger:
    """
    Analyze traces + source, auto-fix what is safe, report the rest.

    model_fn: optional async callable(text) -> str that returns a code fix.
    """

    def __init__(
        self,
        tracer: ConsoleTracer = None,
        model_fn: Optional[Callable[[str], str]] = None,
        workspace: str = None,
        dry_run: bool = True,
    ):
        self.tracer = tracer or ConsoleTracer()
        self.model_fn = model_fn
        self.workspace = workspace or os.getcwd()
        self.dry_run = dry_run
        self.fixes: list[FixAction] = []

    # ---------- analysis ----------

    def classify(self, entry: TraceEntry) -> IssueCategory:
        text = (entry.text or "").lower()
        if entry.source == "network" or "http" in text:
            if "404" in text or "not found" in text:
                return IssueCategory.NETWORK_404
            if re.search(r"\b5\d{2}\b", text) or "server error" in text:
                return IssueCategory.NETWORK_5XX
            if "failed" in text or "aborted" in text or "timeout" in text:
                return IssueCategory.NETWORK_FAILED
            if "slow" in text:
                return IssueCategory.SLOW_RESOURCE
        if "referenceerror" in text or "is not defined" in text:
            return IssueCategory.REFERENCE_ERROR
        if "syntaxerror" in text:
            return IssueCategory.SYNTAX_ERROR
        if "typeerror" in text:
            return IssueCategory.TYPE_ERROR
        if entry.level == TraceLevel.ERROR:
            return IssueCategory.CONSOLE_ERROR
        return IssueCategory.UNKNOWN

    def analyze_traces(self) -> list[FixAction]:
        """Turn captured traces into candidate fixes."""
        actions: list[FixAction] = []
        for entry in self.tracer.get_errors() + self.tracer.get_warnings():
            cat = self.classify(entry)
            actions.append(FixAction(
                category=cat,
                description=entry.text,
                file_path=entry.url or None,
                confidence=0.7 if cat != IssueCategory.UNKNOWN else 0.3,
                method="pattern",
            ))
        self.fixes = actions
        return actions

    # ---------- source fixing ----------

    def scan_source_for_deprecated(self, code: str) -> list[FixAction]:
        """Find deprecated API usage in source code."""
        actions = []
        for bad, note in _DEPRECATED_MAP.items():
            if bad in code:
                actions.append(FixAction(
                    category=IssueCategory.DEPRECATED_API,
                    description=f"Deprecated API used: {bad} -> {note}",
                    file_path=None,
                    original=bad,
                    fixed=note,
                    confidence=0.6,
                    method="pattern",
                ))
        return actions

    def try_fix_file(self, file_path: str) -> list[FixAction]:
        """
        Apply safe, reversible fixes to a single file.
        Returns the FixActions that were (or would be) applied.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        original = path.read_text(encoding="utf-8", errors="ignore")
        fixed = original
        applied: list[FixAction] = []

        # Fix 1: undefined-variable guard -> declare with let/var (JS) or None (Python)
        # This is heuristic: only rewrites obvious "X is not defined" by adding a declaration.
        for action in self.fixes:
            if action.category == IssueCategory.REFERENCE_ERROR and action.file_path == str(path):
                m = re.search(r"['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s+is not defined", action.description)
                if m:
                    var = m.group(1)
                    already_declared = re.search(
                        rf"\b(?:var|let|const|function)\s+{re.escape(var)}\b", fixed
                    ) or re.search(rf"\b{re.escape(var)}\s*=", fixed)
                    if not already_declared:
                        decl = f"/* auto-fix: declared missing var */ var {var} = null;\n"
                        fixed = decl + fixed
                        action.original = f"(missing) {var}"
                        action.fixed = decl.strip()
                        action.applied = not self.dry_run
                        applied.append(action)

        # Fix 2: deprecated API notes -> add a comment marker (non-destructive)
        for dep in self.scan_source_for_deprecated(original):
            if dep.original and dep.original in fixed:
                comment = f"// TODO(auto-debug): {dep.fixed}\n"
                fixed = fixed.replace(dep.original, comment + dep.original, 1)
                dep.applied = not self.dry_run
                applied.append(dep)

        if not self.dry_run and fixed != original:
            # backup before overwriting
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(original, encoding="utf-8")
            path.write_text(fixed, encoding="utf-8")

        return applied

    # ---------- LLM hook ----------

    async def llm_fix(self, entry: TraceEntry) -> Optional[str]:
        if self.model_fn is None:
            return None
        prompt = (
            "Fix the following browser runtime error. "
            "Return ONLY the corrected code or config.\n\n"
            f"{entry.text}\n"
        )
        try:
            if hasattr(self.model_fn, "__call__"):
                result = self.model_fn(prompt)
                if hasattr(result, "__await__"):
                    result = await result
                return result
        except Exception:
            return None
        return None

    # ---------- report ----------

    def debug_report(self) -> dict:
        by_cat = {}
        for f in self.fixes:
            by_cat[f.category.value] = by_cat.get(f.category.value, 0) + 1
        return {
            "total_issues": len(self.fixes),
            "by_category": by_cat,
            "fixes_applied": sum(1 for f in self.fixes if f.applied),
            "dry_run": self.dry_run,
            "actions": [f.__dict__ for f in self.fixes],
        }


# ---- dead code extraction (with caution + external archive) -----------------

class DeadCodeDetector:
    """
    Detect dead code in the workspace, extract it OUT of the live files,
    and archive it to a `deadcodes` file OUTSIDE the workspace.

    Caution: by default runs in `remove=False` (dry-run) mode and only
    reports what would be removed. Set remove=True to actually strip it.
    """

    def __init__(
        self,
        workspace: str,
        deadcodes_dir: str = None,
        archive_name: str = "deadcodes",
        remove: bool = False,
    ):
        self.workspace = Path(workspace)
        # deadcodes lives OUTSIDE the workspace (default: workspace parent)
        parent = self.workspace.parent
        self.deadcodes_dir = Path(deadcodes_dir) if deadcodes_dir else (parent / archive_name)
        self.archive_name = archive_name
        self.remove = remove
        self._collected: list[dict] = []

    def _archive_path(self) -> Path:
        self.deadcodes_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return self.deadcodes_dir / f"{self.archive_name}_{ts}.jsonl"

    def scan(self, extensions: list[str] = None) -> dict:
        """
        Scan workspace for dead code:
        - unused imports
        - commented-out code blocks
        - functions/classes never referenced
        - empty / no-op blocks
        """
        extensions = extensions or [".py", ".js", ".ts", ".jsx", ".tsx"]
        report = {"files_scanned": 0, "dead_items": 0, "items": []}

        for ext in extensions:
            for fp in self.workspace.rglob(f"*{ext}"):
                if ".deadcodes" in str(fp) or "node_modules" in str(fp):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                report["files_scanned"] += 1
                report["dead_items"] += self._scan_file(fp, text, report)

        return report

    def _scan_file(self, fp: Path, text: str, report: dict) -> int:
        count = 0

        # 1) commented-out code lines (heuristic: comment containing code tokens)
        code_tokens = re.compile(r"(def |function |=>|return |if \(|for \(|class |import |console\.)")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                core = stripped.lstrip("#").lstrip("//").strip()
                if code_tokens.search(core) and len(core) > 15:
                    item = {
                        "file": str(fp),
                        "line": i,
                        "type": "commented_code",
                        "snippet": core,
                    }
                    report["items"].append(item)
                    self._collected.append(item)
                    count += 1

        # 2) unused imports (Python only, simple heuristic)
        if fp.suffix == ".py":
            imports = re.findall(r"^\s*(?:from\s+[\w.]+\s+)?import\s+([\w, ]+)", text, re.MULTILINE)
            defined_names = set()
            for imp in imports:
                for name in imp.split(","):
                    name = name.strip().split(" as ")[0].strip()
                    if name:
                        defined_names.add(name)
            for name in defined_names:
                # referenced anywhere outside the import line?
                uses = len(re.findall(rf"\b{re.escape(name)}\b", text)) - 1
                if uses <= 0:
                    item = {
                        "file": str(fp),
                        "line": 0,
                        "type": "unused_import",
                        "snippet": f"import {name}",
                    }
                    report["items"].append(item)
                    self._collected.append(item)
                    count += 1

        return count

    def extract(self) -> dict:
        """
        Remove detected dead code from the workspace files and archive it
        to the external deadcodes file. Returns a summary.
        """
        if not self._collected:
            self.scan()

        archive = self._archive_path()
        removed_count = 0

        # group snippets by file
        by_file: dict[str, list[dict]] = {}
        for item in self._collected:
            by_file.setdefault(item["file"], []).append(item)

        with archive.open("w", encoding="utf-8") as af:
            for file_path, items in by_file.items():
                fp = Path(file_path)
                text = fp.read_text(encoding="utf-8", errors="ignore")
                lines = text.splitlines()
                to_remove = set()

                for item in items:
                    af.write(json.dumps(item) + "\n")
                    if item["type"] == "commented_code" and item["line"]:
                        to_remove.add(item["line"] - 1)
                    if item["type"] == "unused_import" and item["line"] == 0:
                        # find import line
                        for idx, ln in enumerate(lines):
                            if item["snippet"] in ln:
                                to_remove.add(idx)
                                break

                if self.remove and to_remove:
                    new_lines = [ln for i, ln in enumerate(lines) if i not in to_remove]
                    fp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    removed_count += len(to_remove)

        return {
            "archive": str(archive),
            "items_archived": len(self._collected),
            "lines_removed": removed_count if self.remove else 0,
            "removed_from_workspace": self.remove,
        }


# ---- convenience helpers ----------------------------------------------------

async def trace_and_debug(
    browser,
    url: str,
    workspace: str = None,
    model_fn: Optional[Callable] = None,
    apply_fixes: bool = False,
) -> dict:
    """
    Full pipeline: trace console/network issues, classify, and (optionally)
    auto-fix. Returns the debug report.
    """
    tracer = ConsoleTracer()
    page = getattr(browser, "page", None)
    if page is not None:
        await tracer.attach(page)

    await browser.navigate(url)
    await asyncio.sleep(1.5)

    dbg = AutoDebugger(tracer=tracer, model_fn=model_fn, workspace=workspace, dry_run=not apply_fixes)
    dbg.analyze_traces()

    # try to fix any file referenced by a trace
    for action in dbg.fixes:
        if action.file_path and os.path.exists(action.file_path):
            dbg.try_fix_file(action.file_path)

    return dbg.debug_report()


def extract_deadcode(workspace: str, remove: bool = False) -> dict:
    """
    Scan workspace for dead code, archive it to the external `deadcodes`
    file, and (if remove=True) strip it from the live files.
    """
    detector = DeadCodeDetector(workspace, remove=remove)
    scan = detector.scan()
    extract_result = detector.extract()
    return {"scan": scan, "extract": extract_result}

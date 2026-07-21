"""Tests for the browser package (config, tracer, auto-debug pure logic)."""

import json

from browser import (
    BrowserAutomation, WebScraper, FormFiller,
    BrowserConfig, BrowserBackend, ElementType,
    ElementInfo, PageResult, ScrapedContent,
    ConsoleTracer, TraceEntry, TraceLevel,
    AutoDebugger, DeadCodeDetector, IssueCategory, FixAction,
    extract_deadcode,
)


class TestConfigAndDataclasses:
    def test_browser_config_defaults(self):
        cfg = BrowserConfig()
        assert cfg.backend == BrowserBackend.PLAYWRIGHT
        assert cfg.headless is True
        assert cfg.viewport_width == 1280
        assert cfg.timeout == 30000
        assert cfg.extra_args == []

    def test_element_info(self):
        el = ElementInfo(tag="a", text="link", attributes={}, selector="a",
                         index=0, visible=True, enabled=True)
        assert el.tag == "a"
        assert el.bbox is None

    def test_scraped_content_defaults(self):
        sc = ScrapedContent(url="u", title="t", text="x", html="<p>",
                            links=[], images=[], metadata={})
        assert sc.timestamp > 0

    def test_browser_automation_uses_default_config(self):
        b = BrowserAutomation()
        assert isinstance(b.config, BrowserConfig)
        assert b.page is None
        assert b.get_history() == []


class TestTraceEntry:
    def test_to_dict(self):
        e = TraceEntry(level=TraceLevel.ERROR, text="boom", source="console", line=5)
        d = e.to_dict()
        assert d["level"] == "error"
        assert d["text"] == "boom"
        assert d["line"] == 5


class TestConsoleTracer:
    def _tracer(self):
        t = ConsoleTracer()
        t.record_manual(TraceLevel.ERROR, "an error", url="a.js", line=1)
        t.record_manual(TraceLevel.WARNING, "a warning")
        t.record_manual(TraceLevel.INFO, "info msg")
        t.record_manual(TraceLevel.NETWORK, "HTTP 404: x", url="a")
        return t

    def test_get_errors(self):
        t = self._tracer()
        errors = t.get_errors()
        assert len(errors) == 1
        assert errors[0].text == "an error"

    def test_get_warnings(self):
        assert len(self._tracer().get_warnings()) == 1

    def test_get_network_issues_source_based(self):
        t = ConsoleTracer()
        t.entries.append(TraceEntry(level=TraceLevel.ERROR, text="HTTP 500", source="network"))
        t.record_manual(TraceLevel.NETWORK, "manual net")
        net = t.get_network_issues()
        assert len(net) == 1
        assert net[0].source == "network"

    def test_count_by_level(self):
        counts = self._tracer().count_by_level()
        assert counts["error"] == 1
        assert counts["warning"] == 1

    def test_get_report(self):
        report = self._tracer().get_report()
        assert report["total"] == 4
        assert len(report["errors"]) == 1
        assert len(report["warnings"]) == 1

    def test_to_json_and_clear(self):
        t = self._tracer()
        parsed = json.loads(t.to_json())
        assert len(parsed) == 4
        t.clear()
        assert t.entries == []


class TestAutoDebuggerClassify:
    def test_classify_network_404(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.WARNING, text="HTTP 404: /x", source="network")
        assert dbg.classify(entry) == IssueCategory.NETWORK_404

    def test_classify_network_5xx(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="HTTP 500: /x", source="network")
        assert dbg.classify(entry) == IssueCategory.NETWORK_5XX

    def test_classify_network_5xx_bare_code(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="Request to /api returned 503", source="network")
        assert dbg.classify(entry) == IssueCategory.NETWORK_5XX

    def test_classify_network_5xx_server_error_phrase(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="Internal Server Error", source="network")
        assert dbg.classify(entry) == IssueCategory.NETWORK_5XX

    def test_classify_network_url_with_50_not_5xx(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.WARNING, text="http request to /page50 slow", source="network")
        assert dbg.classify(entry) != IssueCategory.NETWORK_5XX

    def test_classify_reference_error(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="foo is not defined")
        assert dbg.classify(entry) == IssueCategory.REFERENCE_ERROR

    def test_classify_syntax_error(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="SyntaxError: bad")
        assert dbg.classify(entry) == IssueCategory.SYNTAX_ERROR

    def test_classify_console_error(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.ERROR, text="something went wrong")
        assert dbg.classify(entry) == IssueCategory.CONSOLE_ERROR

    def test_classify_unknown(self):
        dbg = AutoDebugger()
        entry = TraceEntry(level=TraceLevel.INFO, text="just info")
        assert dbg.classify(entry) == IssueCategory.UNKNOWN


class TestAutoDebuggerFixing:
    def test_analyze_traces(self):
        tracer = ConsoleTracer()
        tracer.record_manual(TraceLevel.ERROR, "bar is not defined", url="a.js")
        dbg = AutoDebugger(tracer=tracer)
        actions = dbg.analyze_traces()
        assert len(actions) == 1
        assert actions[0].category == IssueCategory.REFERENCE_ERROR
        assert actions[0].confidence == 0.7

    def test_scan_source_for_deprecated(self):
        dbg = AutoDebugger()
        actions = dbg.scan_source_for_deprecated("el.innerHTML = 'x'; escape('y')")
        cats = {a.category for a in actions}
        assert IssueCategory.DEPRECATED_API in cats
        assert len(actions) >= 2

    def test_try_fix_file_reference_error_dry_run(self, tmp_path):
        f = tmp_path / "app.js"
        original = "console.log('hello');\n"
        f.write_text(original)
        tracer = ConsoleTracer()
        dbg = AutoDebugger(tracer=tracer, dry_run=True)
        dbg.fixes = [FixAction(
            category=IssueCategory.REFERENCE_ERROR,
            description="'missingVar' is not defined",
            file_path=str(f),
        )]
        applied = dbg.try_fix_file(str(f))
        assert len(applied) == 1
        assert f.read_text() == original

    def test_try_fix_file_declares_used_undefined_var(self, tmp_path):
        f = tmp_path / "app.js"
        original = "console.log(missingVar);\n"
        f.write_text(original)
        dbg = AutoDebugger(tracer=ConsoleTracer(), dry_run=False)
        dbg.fixes = [FixAction(
            category=IssueCategory.REFERENCE_ERROR,
            description="'missingVar' is not defined",
            file_path=str(f),
        )]
        applied = dbg.try_fix_file(str(f))
        assert len(applied) == 1
        assert "var missingVar = null;" in f.read_text()

    def test_try_fix_file_skips_already_declared_var(self, tmp_path):
        f = tmp_path / "app.js"
        original = "var known = 1;\nconsole.log(known);\n"
        f.write_text(original)
        dbg = AutoDebugger(tracer=ConsoleTracer(), dry_run=False)
        dbg.fixes = [FixAction(
            category=IssueCategory.REFERENCE_ERROR,
            description="'known' is not defined",
            file_path=str(f),
        )]
        applied = dbg.try_fix_file(str(f))
        assert applied == []
        assert f.read_text() == original

    def test_try_fix_file_missing_returns_empty(self):
        dbg = AutoDebugger()
        assert dbg.try_fix_file("no-such-file.js") == []

    def test_debug_report(self):
        tracer = ConsoleTracer()
        tracer.record_manual(TraceLevel.ERROR, "x is not defined")
        dbg = AutoDebugger(tracer=tracer, dry_run=True)
        dbg.analyze_traces()
        report = dbg.debug_report()
        assert report["total_issues"] == 1
        assert report["dry_run"] is True
        assert report["fixes_applied"] == 0


class TestDeadCodeDetector:
    def test_scan_detects_commented_code(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "mod.py").write_text(
            "# def old_function(): return some_value_here\nx = 1\n"
        )
        detector = DeadCodeDetector(str(ws), deadcodes_dir=str(tmp_path / "dead"))
        report = detector.scan()
        assert report["files_scanned"] == 1
        types = {item["type"] for item in report["items"]}
        assert "commented_code" in types

    def test_scan_detects_unused_import(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "mod.py").write_text("import unusedmodule\nx = 1\n")
        detector = DeadCodeDetector(str(ws), deadcodes_dir=str(tmp_path / "dead"))
        report = detector.scan()
        types = {item["type"] for item in report["items"]}
        assert "unused_import" in types

    def test_extract_archives_without_removing(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        src = "# def gone_func(): return a_removed_value\nkeep = 1\n"
        (ws / "mod.py").write_text(src)
        detector = DeadCodeDetector(str(ws), deadcodes_dir=str(tmp_path / "dead"),
                                    remove=False)
        result = detector.extract()
        assert result["removed_from_workspace"] is False
        assert result["lines_removed"] == 0
        assert (ws / "mod.py").read_text() == src

    def test_extract_deadcode_helper(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "mod.py").write_text("import somethingunused\ny = 2\n")
        out = extract_deadcode(str(ws), remove=False)
        assert "scan" in out
        assert "extract" in out

"""Tests for the agent package (offline-safe pure logic)."""

import json

from agent.web_search import WebSearch, SearchResult, SearchResponse
from agent.code_executor import CodeExecutor, ExecutionResult
from agent.self_healing_agent import SelfHealingAgent, ToolType, ToolCall, ToolResult


def _make_response():
    results = [
        SearchResult(title="Python", snippet="A programming language" * 20, url="http://py"),
        SearchResult(title="Rust", snippet="Systems language", url="http://rust"),
    ]
    return SearchResponse(query="langs", results=results, total_results=2,
                          search_time=0.5, backend="duckduckgo")


class TestWebSearch:
    def test_defaults(self):
        ws = WebSearch()
        assert ws.backend == "duckduckgo"
        assert ws.cache_enabled is True
        assert ws.cache_ttl == 3600
        assert ws._cache == {}

    def test_cache_store_and_get(self):
        ws = WebSearch()
        resp = _make_response()
        ws._store_in_cache("langs", resp)
        assert ws._get_from_cache("langs") is resp

    def test_cache_expiry(self):
        ws = WebSearch(cache_ttl=0)
        resp = _make_response()
        ws._store_in_cache("langs", resp)
        assert ws._get_from_cache("langs") is None
        assert "langs" not in ws._cache

    def test_clear_cache(self):
        ws = WebSearch()
        ws._store_in_cache("langs", _make_response())
        ws.clear_cache()
        assert ws._cache == {}

    def test_format_results_truncates(self):
        ws = WebSearch()
        text = ws.format_results(_make_response(), max_snippet_len=30)
        assert "Search: langs (2 results" in text
        assert "1. Python" in text
        assert "2. Rust" in text
        assert "..." in text
        assert "URL: http://py" in text


class TestCodeExecutor:
    def test_run_python_success(self):
        ex = CodeExecutor()
        result = ex.run_python("print('hello world')")
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.return_code == 0
        assert "hello world" in result.stdout

    def test_run_python_error(self):
        ex = CodeExecutor()
        result = ex.run_python("raise ValueError('boom')")
        assert result.success is False
        assert result.return_code != 0
        assert "ValueError" in result.stderr

    def test_run_python_timeout(self):
        ex = CodeExecutor()
        result = ex.run_python("import time; time.sleep(5)", timeout=1)
        assert result.timeout is True
        assert result.success is False

    def test_write_and_read_file(self, tmp_path):
        ex = CodeExecutor()
        target = tmp_path / "out.txt"
        w = ex.write_file(str(target), "content-123")
        assert w.success is True
        r = ex.read_file(str(target))
        assert r.success is True
        assert r.stdout == "content-123"

    def test_read_missing_file(self):
        ex = CodeExecutor()
        r = ex.read_file("does-not-exist-xyz.txt")
        assert r.success is False
        assert r.return_code == -1

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        ex = CodeExecutor()
        r = ex.list_directory(str(tmp_path))
        assert r.success is True
        assert "a.txt" in r.stdout
        assert "[DIR] sub" in r.stdout
        assert r.metadata["count"] == 2

    def test_format_result(self):
        ex = CodeExecutor()
        res = ExecutionResult(success=True, stdout="out", stderr="",
                              return_code=0, execution_time=0.1)
        text = ex.format_result(res)
        assert "[SUCCESS]" in text
        assert "STDOUT" in text


class TestSelfHealingAgent:
    def _agent(self):
        return SelfHealingAgent(model_path="", verbose=False)

    def test_calculator(self):
        r = self._agent()._calculator("(15 * 7) + 23")
        assert r.success is True
        assert r.output == "128"

    def test_calculator_invalid(self):
        r = self._agent()._calculator("abcdef")
        assert r.success is False
        assert r.error == "Invalid expression"

    def test_parse_tool_call_json_block(self):
        agent = self._agent()
        text = '```tool\n{"tool": "calculator", "args": {"expression": "1+1"}}\n```'
        call = agent._parse_tool_call(text)
        assert isinstance(call, ToolCall)
        assert call.tool == ToolType.CALCULATOR
        assert call.args == {"expression": "1+1"}

    def test_parse_tool_call_none(self):
        assert self._agent()._parse_tool_call("just text no tool") is None

    def test_refine_query_removes_stopwords(self):
        agent = self._agent()
        assert agent._refine_query("the best python in the world") == "best python world"

    def test_simplify_python(self):
        agent = self._agent()
        out = agent._simplify_python('x = lambda a: a + 1')
        assert "lambda x: x" in out

    def test_add_missing_imports(self):
        agent = self._agent()
        out = agent._add_missing_imports("use pathlib here")
        assert out.startswith("from pathlib import Path")

    def test_add_missing_imports_same_name_module(self):
        agent = self._agent()
        out = agent._add_missing_imports("path = os.getcwd()")
        assert "import os" in out
        assert out.endswith("path = os.getcwd()")

    def test_add_missing_imports_no_duplicate(self):
        agent = self._agent()
        code = "import os\npath = os.getcwd()"
        out = agent._add_missing_imports(code)
        assert out.count("import os") == 1

    def test_add_missing_imports_word_boundary(self):
        agent = self._agent()
        out = agent._add_missing_imports("position = 1")
        assert "import os" not in out

    def test_alternative_command(self):
        agent = self._agent()
        assert agent._alternative_command("ls") in ("dir", "ls -la")
        assert agent._alternative_command("unknowncmd") == "unknowncmd"

    def test_heal_tool_args_timeout(self):
        agent = self._agent()
        healed = agent._heal_tool_args(ToolType.PYTHON_EXEC, {"code": "x"}, "timeout occurred")
        assert healed["timeout"] == 60

    def test_python_exec(self):
        agent = self._agent()
        r = agent._python_exec("print('from agent')")
        assert r.success is True
        assert "from agent" in r.output

    def test_file_write_read(self, tmp_path):
        agent = self._agent()
        p = tmp_path / "f.txt"
        w = agent._file_write(str(p), "data")
        assert w.success is True
        r = agent._file_read(str(p))
        assert r.success is True
        assert r.output == "data"

    def test_execute_tool_unknown(self):
        agent = self._agent()
        call = ToolCall(tool=ToolType.CALCULATOR, args={"expression": "2+2"})
        result = agent._execute_tool(call)
        assert result.success is True
        assert result.output == "4"

    def test_generate_response_search(self):
        agent = self._agent()
        out = agent._generate_response([{"role": "user", "content": "search python"}])
        assert json.loads(out)["tool"] == "web_search"

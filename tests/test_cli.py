"""Tests for the cli package (AgenticCLI offline-safe tools + session logic)."""

import json

from cli import (
    AgenticCLI, CLIBuilder,
    Message, ToolCall, Session, MessageRole,
)


class TestDataclasses:
    def test_message(self):
        m = Message(role=MessageRole.USER, content="hi")
        assert m.role == MessageRole.USER
        assert m.timestamp > 0

    def test_toolcall_defaults(self):
        tc = ToolCall(id="1", name="t", input={})
        assert tc.success is True
        assert tc.duration_ms == 0

    def test_session_defaults(self):
        s = Session(id="s1")
        assert s.messages == []
        assert s.tool_calls == []


class TestAgenticCLIInit:
    def test_init(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        assert cli.working_dir == str(tmp_path)
        assert cli.session.working_dir == str(tmp_path)
        assert "run_code" in cli.tools
        assert "analyze_code" in cli.tools

    def test_detect_system(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        info = cli._system_info
        assert "os" in info
        assert "python" in info


class TestMessaging:
    def test_send_and_respond(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        cli.send("hello")
        cli.respond("hi there")
        assert len(cli.session.messages) == 2
        assert cli.session.messages[0].role == MessageRole.USER
        assert cli.session.messages[1].role == MessageRole.ASSISTANT

    def test_get_context(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        cli.send("first")
        cli.respond("second")
        ctx = cli.get_context()
        assert "[user]: first" in ctx
        assert "[assistant]: second" in ctx

    def test_clear(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        cli.send("x")
        cli.execute_tool("find_path", start="a", goal="b")
        cli.clear()
        assert cli.session.messages == []
        assert cli.session.tool_calls == []


class TestTools:
    def test_execute_unknown_tool(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("does_not_exist")
        assert call.success is False
        assert "Unknown tool" in call.output["error"]

    def test_write_and_read_file(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        target = tmp_path / "note.txt"
        w = cli.execute_tool("write_file", path=str(target), content="hello cli")
        assert w.success is True
        r = cli.execute_tool("read_file", path=str(target))
        assert r.success is True
        assert r.output["content"] == "hello cli"
        assert r.output["lines"] == 1

    def test_list_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("y")
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("list_files", path=str(tmp_path), pattern="*.py")
        assert call.success is True
        assert call.output["count"] == 1
        assert call.output["files"][0]["name"] == "a.py"

    def test_search_files(self, tmp_path):
        (tmp_path / "code.py").write_text("def target_function():\n    pass\n")
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("search_files", query="target_function", path=str(tmp_path))
        assert call.success is True
        assert call.output["count"] >= 1
        assert call.output["results"][0]["line"] == 1

    def test_analyze_code(self, tmp_path):
        src = tmp_path / "sample.py"
        src.write_text("import os\n# a comment\ndef foo():\n    return 1\nclass Bar:\n    pass\n")
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("analyze_code", path=str(src))
        assert call.success is True
        analysis = call.output["analysis"]
        assert analysis["functions"] == 1
        assert analysis["classes"] == 1
        assert analysis["imports"] == 1
        assert analysis["comment_lines"] == 1

    def test_find_path(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        f = tmp_path / "existing_file.txt"
        f.write_text("hello")
        call = cli.execute_tool("find_path", start=str(f))
        assert call.success is True
        assert call.output["exists"] is True

    def test_find_path_missing(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("find_path", start="nonexistent_path_xyz")
        assert call.success is False
        assert "error" in call.output

    def test_run_code_python(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("run_code", language="python", code="print('cli run')")
        assert call.output["success"] is True
        assert "cli run" in call.output["stdout"]

    def test_run_code_unsupported_language(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        call = cli.execute_tool("run_code", language="cobol", code="x")
        assert "error" in call.output


class TestSessionExport:
    def test_get_stats(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        cli.send("hi")
        stats = cli.get_stats()
        assert stats["messages"] == 1
        assert stats["working_dir"] == str(tmp_path)
        assert "system" in stats

    def test_export_session(self, tmp_path):
        cli = AgenticCLI(str(tmp_path))
        cli.send("msg")
        cli.execute_tool("find_path", start="a", goal="b")
        data = json.loads(cli.export_session())
        assert data["session_id"] == cli.session.id
        assert len(data["messages"]) == 1
        assert len(data["tool_calls"]) == 1


class TestCLIBuilder:
    def test_create_project_cli_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        cli = CLIBuilder.create_project_cli(str(tmp_path))
        assert cli.session.context["project_type"] == "python"

    def test_create_project_cli_node(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        cli = CLIBuilder.create_project_cli(str(tmp_path))
        assert cli.session.context["project_type"] == "node"
        assert cli.session.context["package_manager"] == "npm"

    def test_create_portable_cli(self, tmp_path):
        cli = CLIBuilder.create_portable_cli(str(tmp_path))
        assert cli.session.context["portable"] is True
        assert (tmp_path / "sessions").exists()
        assert (tmp_path / "cache").exists()

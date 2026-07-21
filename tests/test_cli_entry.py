"""Coverage for the argparse-based CLI entry point (cli/__main__.py)."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

import cli.__main__ as entry


def _run(argv, stdin=None):
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        if stdin is not None:
            with mock.patch("builtins.input", side_effect=stdin):
                code = entry.main(argv)
        else:
            code = entry.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_demo_command_runs():
    code, out, _ = _run(["demo"])
    assert code == 0
    assert "Agentic CLI initialized" in out


def test_no_command_starts_repl_and_exits():
    code, out, _ = _run(["--backend", "echo"], stdin=["exit"])
    assert code == 0
    # Either TUI mode or fallback with "TUI" or "REPL" in output
    assert "TUI" in out or "REPL" in out


def test_repl_chat_on_plain_text():
    code, out, _ = _run(
        ["--backend", "echo", "repl"], stdin=["hello there", "exit"]
    )
    assert code == 0
    assert "hello there" in out


def test_list_json(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.txt").write_text("hi\n")
    code, out, _ = _run(["list", str(tmp_path), "--pattern", "*.py", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["count"] == 1
    assert data["files"][0]["name"] == "a.py"


def test_read_ok(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("print('hi')\n")
    code, out, _ = _run(["read", str(f), "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["success"] is True
    assert "print('hi')" in data["content"]


def test_read_missing_file_exit_code():
    code, _, err = _run(["read", "definitely-missing-file.xyz"])
    assert code == 1
    assert "error" in err.lower()


def test_write_then_read(tmp_path):
    f = tmp_path / "out.txt"
    code, out, _ = _run(["write", str(f), "--content", "data123", "--json"])
    assert code == 0
    assert f.read_text() == "data123"


def test_analyze(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("import os\n\ndef foo():\n    return 1\n\nclass Bar:\n    pass\n")
    code, out, _ = _run(["analyze", str(f), "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["analysis"]["functions"] >= 1
    assert data["analysis"]["classes"] >= 1


def test_search(tmp_path):
    (tmp_path / "s.py").write_text("def target():\n    pass\n")
    code, out, _ = _run(["search", "target", "--path", str(tmp_path), "--ext", ".py", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["count"] >= 1


def test_run_code_python():
    code, out, _ = _run(["run-code", "--lang", "python", "--code", "print(2+2)", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["success"] is True
    assert "4" in data["stdout"]


def test_run_code_unsupported_lang_exit_code():
    code, _, err = _run(["run-code", "--lang", "cobol", "--code", "x", "--json"])
    assert code == 1


def test_stats_global_json_flag():
    code, out, _ = _run(["--json", "stats"])
    assert code == 0
    data = json.loads(out)
    assert "session_id" in data
    assert "system" in data


def test_json_flag_after_subcommand(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    code, out, _ = _run(["list", str(tmp_path), "--json"])
    assert code == 0
    json.loads(out)


def test_build_parser_has_all_commands():
    parser = entry._build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    choices = set(actions[0].choices)
    expected = {
        "list", "read", "write", "search", "run-code",
        "exec", "git-status", "git-commit", "analyze", "stats", "repl", "demo",
    }
    assert expected.issubset(choices)

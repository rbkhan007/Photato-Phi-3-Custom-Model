"""Coverage for CLI model inference, persistence, and working-dir handling."""
from pathlib import Path

from cli import AgenticCLI
from cli.model_backend import (
    EchoBackend,
    get_backend,
    OllamaBackend,
    OpenAICompatBackend,
    BackendError,
)


def _cli(tmp_path, working_dir=None):
    c = AgenticCLI(working_dir=working_dir, config={"backend": "echo"})
    c.HOME_DIR = tmp_path / ".agentic_cli"
    c.set_backend(EchoBackend())
    return c


def test_chat_with_echo_backend(tmp_path):
    c = _cli(tmp_path)
    result = c.chat("hello model")
    assert result["success"] is True
    assert result["backend"] == "echo"
    assert "hello model" in result["content"]
    assert len(c.session.messages) == 2


def test_chat_records_history(tmp_path):
    c = _cli(tmp_path)
    c.chat("first")
    log = c.HOME_DIR / "history.log"
    assert log.exists()
    assert "chat: first" in log.read_text(encoding="utf-8")


def test_chat_via_execute_tool(tmp_path):
    c = _cli(tmp_path)
    call = c.execute_tool("chat", content="via tool")
    assert call.success is True
    assert "via tool" in call.output["content"]


def test_save_and_load_config(tmp_path):
    c = _cli(tmp_path)
    c.config["temperature"] = 0.1
    res = c.save_config()
    assert res["success"] is True
    assert Path(res["path"]).exists()

    c2 = AgenticCLI(config={"backend": "echo"})
    c2.HOME_DIR = tmp_path / ".agentic_cli"
    loaded = c2._load_config()
    assert loaded["temperature"] == 0.1


def test_save_and_load_session(tmp_path):
    c = _cli(tmp_path)
    c.chat("remember this")
    saved = c.save_session()
    assert saved["success"] is True
    sid = saved["session_id"]

    c2 = _cli(tmp_path)
    loaded = c2.load_session(sid)
    assert loaded["success"] is True
    assert loaded["messages"] == 2
    assert c2.session.id == sid


def test_load_missing_session(tmp_path):
    c = _cli(tmp_path)
    res = c.load_session("does-not-exist")
    assert res["success"] is False
    assert "not found" in res["error"].lower()


def test_list_sessions(tmp_path):
    c = _cli(tmp_path)
    assert c.list_sessions()["sessions"] == []
    c.chat("hi")
    c.save_session()
    listing = c.list_sessions()
    assert listing["count"] == 1
    assert c.session.id in listing["sessions"]


def test_working_dir_resolution(tmp_path):
    (tmp_path / "note.txt").write_text("inside workdir\n", encoding="utf-8")
    c = _cli(tmp_path, working_dir=str(tmp_path))
    call = c.execute_tool("read_file", path="note.txt")
    assert call.success is True
    assert "inside workdir" in call.output["content"]


def test_working_dir_list(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    c = _cli(tmp_path, working_dir=str(tmp_path))
    call = c.execute_tool("list_files", path=".", pattern="*.py")
    assert call.success is True
    assert call.output["count"] == 2


def test_get_backend_unknown_raises():
    try:
        get_backend("nonsense")
    except BackendError as e:
        assert "Unknown backend" in str(e)
    else:
        raise AssertionError("expected BackendError")


def test_get_backend_known():
    assert isinstance(get_backend("echo"), EchoBackend)
    assert isinstance(get_backend("ollama"), OllamaBackend)
    assert isinstance(get_backend("openai"), OpenAICompatBackend)


def test_echo_backend_offline_message():
    b = EchoBackend()
    res = b.chat([{"role": "user", "content": "ping"}])
    assert res.backend == "echo"
    assert "ping" in res.content

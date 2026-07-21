"""Full coverage for the Claude-compatible tools package."""
from tools.tool_registry import ToolRegistry
from tools.tool_parser import ToolParser
from tools.tool_executor import ToolExecutor, ExecutionResult
from tools.claude_compatible import (
    format_tools_for_api,
    format_tools_from_registry,
    create_tool_use_message,
    create_tool_result_message,
    build_tool_use_conversation,
    parse_tool_use_from_response,
    ClaudeToolUse,
    ClaudeToolResult,
    ClaudeToolChoice,
)


def _echo(text: str) -> str:
    """Echo the input text."""
    return f"echo:{text}"


def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_registry_register_and_lookup():
    reg = ToolRegistry()
    reg.register_function(_echo, name="echo", description="echo tool")
    assert reg.has("echo")
    assert "echo" in reg.list_names()


def test_registry_schema():
    reg = ToolRegistry()
    reg.register_function(_add, name="add", description="add numbers")
    schemas = reg.get_schemas()
    assert isinstance(schemas, list) and len(schemas) == 1
    assert schemas[0]["name"] == "add"


def test_registry_categories_and_tags():
    reg = ToolRegistry()
    reg.register_function(_echo, name="echo", description="echo", category="io", tags=["text"])
    assert reg.get_by_category("io")[0].name == "echo"
    assert reg.get_by_tag("text")[0].name == "echo"


def test_parser_and_executor():
    reg = ToolRegistry()
    reg.register_function(_echo, name="echo", description="echo tool")
    parser = ToolParser(reg)
    text = '```json\n{"tool": "echo", "parameters": {"text": "hi"}}\n```'
    parsed = parser.parse(text)
    assert len(parsed) == 1
    assert parsed[0].name == "echo"
    executor = ToolExecutor(reg)
    results = executor.execute_from_parsed(parsed)
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "echo:hi"


def test_executor_execute_direct():
    reg = ToolRegistry()
    reg.register_function(_add, name="add", description="add numbers")
    executor = ToolExecutor(reg)
    res = executor.execute("add", {"a": 2, "b": 3})
    assert isinstance(res, ExecutionResult)
    assert res.success
    assert str(res.output) == "5"


def test_executor_unknown_tool():
    reg = ToolRegistry()
    executor = ToolExecutor(reg)
    res = executor.execute("missing", {})
    assert not res.success
    assert res.is_error


def test_executor_batch():
    reg = ToolRegistry()
    reg.register_function(_echo, name="echo", description="echo tool")
    executor = ToolExecutor(reg)
    results = executor.execute_batch([
        {"name": "echo", "input": {"text": "a"}},
        {"name": "echo", "input": {"text": "b"}},
    ])
    assert len(results) == 2
    assert [r.output for r in results] == ["echo:a", "echo:b"]


def test_format_tools_for_api():
    api = format_tools_for_api([
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ])
    assert isinstance(api, list)
    assert api[0]["name"] == "get_weather"
    assert "input_schema" in api[0]


def test_format_tools_from_registry():
    reg = ToolRegistry()
    reg.register_function(_add, name="add", description="add numbers")
    api = format_tools_from_registry(reg)
    assert api[0]["name"] == "add"


def test_claude_tool_use_message():
    msg = create_tool_use_message([{"id": "tu1", "name": "echo", "input": {"text": "hi"}}])
    assert msg["role"] == "assistant"
    assert msg["content"][0]["type"] == "tool_use"
    assert msg["content"][0]["id"] == "tu1"


def test_claude_tool_result_message():
    msg = create_tool_result_message([{"tool_use_id": "tu1", "content": "done"}])
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "tool_result"


def test_build_tool_use_conversation():
    conv = build_tool_use_conversation(
        messages=[{"role": "user", "content": "do it"}],
        tool_results=[{"tool_use_id": "tu1", "content": "ok"}],
    )
    assert isinstance(conv, list)
    assert conv[0]["role"] == "user"
    assert conv[-1]["role"] == "user"
    assert conv[-1]["content"][0]["type"] == "tool_result"


def test_parse_tool_use_from_response():
    response = {"content": [{"type": "tool_use", "id": "tu1", "name": "echo", "input": {"text": "hi"}}]}
    blocks = parse_tool_use_from_response(response)
    assert len(blocks) == 1
    assert blocks[0].name == "echo"


def test_claude_tool_choice():
    assert ClaudeToolChoice.auto().type == "auto"
    assert ClaudeToolChoice.any().type == "any"
    assert ClaudeToolChoice.none().type == "none"
    tc = ClaudeToolChoice.tool("echo")
    assert tc.type == "tool" and tc.name == "echo"
    assert tc.to_dict() == {"type": "tool", "name": "echo"}


def test_claude_tool_use_block():
    block = ClaudeToolUse()
    block.add_tool_use("tu1", "echo", {"text": "hi"})
    content = block.to_content()
    assert content[0]["type"] == "tool_use"
    assert content[0]["id"] == "tu1"


def test_claude_tool_result_block():
    tr = ClaudeToolResult()
    tr.add_result("tu1", "output", is_error=False)
    content = tr.to_content()
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "tu1"

"""Tests for the mcp package (JSON-RPC handling, registries, schemas)."""

import asyncio
import json

from mcp import (
    MCPServer, MCPClient, MCPRegistry,
    MCPTool, MCPResource, MCPPrompt, MCPMessage,
    StdioTransport, TransportType, ServerCapability,
    create_mcp_server, create_file_tool,
)


class TestStdioTransportSerialization:
    def test_serialize_request(self):
        t = StdioTransport()
        msg = MCPMessage(id=1, method="ping", params={"x": 1})
        data = json.loads(t._serialize(msg))
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["method"] == "ping"
        assert data["params"] == {"x": 1}

    def test_roundtrip(self):
        t = StdioTransport()
        msg = MCPMessage(id=5, method="tools/list")
        back = t._deserialize(t._serialize(msg))
        assert back.id == 5
        assert back.method == "tools/list"

    def test_deserialize_bad_json(self):
        t = StdioTransport()
        back = t._deserialize("not json")
        assert back.error["code"] == -32700


class TestMCPServerTools:
    def test_register_and_list_tools(self):
        server = MCPServer(name="s")
        server.register_tool("echo", "Echoes", {"type": "object"}, lambda x: x)
        assert server.get_tool("echo").name == "echo"
        listed = server.list_tools()
        assert listed[0]["name"] == "echo"
        assert listed[0]["inputSchema"] == {"type": "object"}

    def test_register_tool_from_function(self):
        server = MCPServer()

        def add(a: int, b: int = 2) -> int:
            """Add two numbers."""
            return a + b

        server.register_tool_from_function(add)
        tool = server.get_tool("add")
        assert tool.description == "Add two numbers."
        assert tool.input_schema["properties"]["a"]["type"] == "integer"
        assert tool.input_schema["required"] == ["a"]

    def test_unregister_tool(self):
        server = MCPServer()
        server.register_tool("t", "d", {}, lambda: None)
        server.unregister_tool("t")
        assert server.get_tool("t") is None


class TestMCPServerResourcesPrompts:
    def test_register_and_list_resource(self):
        server = MCPServer()
        server.register_resource("file:///x", "X", "desc", lambda uri: "data")
        listed = server.list_resources()
        assert listed[0]["uri"] == "file:///x"
        assert listed[0]["mimeType"] == "text/plain"

    def test_read_resource(self):
        server = MCPServer()
        server.register_resource("file:///x", "X", "desc", lambda uri: "hello")
        out = asyncio.run(server.read_resource("file:///x"))
        assert out["contents"][0]["text"] == "hello"

    def test_read_missing_resource(self):
        server = MCPServer()
        out = asyncio.run(server.read_resource("file:///missing"))
        assert "error" in out

    def test_register_and_get_prompt(self):
        server = MCPServer()
        server.register_prompt("greet", "Greeting", [{"name": "who"}],
                               lambda args: f"Hi {args.get('who')}")
        listed = server.list_prompts()
        assert listed[0]["name"] == "greet"
        out = asyncio.run(server.get_prompt("greet", {"who": "Bob"}))
        assert out["messages"] == ["Hi Bob"]


class TestMCPMessageHandling:
    def test_handle_ping(self):
        server = MCPServer()
        resp = asyncio.run(server.handle_message(MCPMessage(id=1, method="ping")))
        assert resp.result == {}
        assert resp.error is None

    def test_handle_initialize(self):
        server = MCPServer(name="srv", version="2.0.0")
        resp = asyncio.run(server.handle_message(MCPMessage(id=1, method="initialize")))
        assert resp.result["serverInfo"]["name"] == "srv"
        assert resp.result["serverInfo"]["version"] == "2.0.0"

    def test_handle_unknown_method(self):
        server = MCPServer()
        resp = asyncio.run(server.handle_message(MCPMessage(id=1, method="bogus")))
        assert resp.error["code"] == -32601

    def test_handle_tools_call(self):
        server = MCPServer()
        server.register_tool("mult", "mult", {}, lambda a, b: a * b)
        resp = asyncio.run(server.handle_message(
            MCPMessage(id=1, method="tools/call",
                       params={"name": "mult", "arguments": {"a": 3, "b": 4}})))
        assert resp.result["isError"] is False
        assert resp.result["content"][0]["text"] == "12"

    def test_handle_tools_call_missing(self):
        server = MCPServer()
        resp = asyncio.run(server.handle_message(
            MCPMessage(id=1, method="tools/call", params={"name": "nope"})))
        assert resp.result["isError"] is True

    def test_handle_tools_list(self):
        server = MCPServer()
        server.register_tool("a", "d", {}, lambda: 1)
        resp = asyncio.run(server.handle_message(MCPMessage(id=1, method="tools/list")))
        assert len(resp.result["tools"]) == 1


class TestCreateHelpers:
    def test_create_mcp_server_has_file_tools(self):
        server = create_mcp_server("demo")
        names = {t["name"] for t in server.list_tools()}
        assert {"read_file", "write_file", "list_directory"} <= names

    def test_create_file_tool_write_read(self, tmp_path):
        read_file, write_file, list_directory = create_file_tool()
        target = tmp_path / "f.txt"
        msg = write_file(str(target), "abc")
        assert "Written to" in msg
        assert read_file(str(target)) == "abc"


class TestMCPClientSchema:
    def test_get_tools_schema(self):
        client = MCPClient()
        client.tools = [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]
        schema = client.get_tools_schema()
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "t"
        assert schema[0]["function"]["parameters"] == {"type": "object"}


class TestMCPRegistry:
    def test_register_and_list(self, tmp_path):
        cfg = str(tmp_path / "config.json")
        reg = MCPRegistry(config_path=cfg)
        reg.register_server("srv", "python", args=["-m", "srv"])
        assert "srv" in reg.list_servers()
        assert reg.servers["srv"]["command"] == "python"
        assert reg.servers["srv"]["transport"] == "stdio"

    def test_persist_config(self, tmp_path):
        cfg = str(tmp_path / "config.json")
        reg = MCPRegistry(config_path=cfg)
        reg.register_server("srv", "node", transport=TransportType.SSE)
        reg2 = MCPRegistry(config_path=cfg)
        assert reg2.servers["srv"]["transport"] == "sse"

    def test_unregister(self, tmp_path):
        cfg = str(tmp_path / "config.json")
        reg = MCPRegistry(config_path=cfg)
        reg.register_server("srv", "python")
        reg.unregister_server("srv")
        assert "srv" not in reg.servers

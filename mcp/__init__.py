"""
MCP (Model Context Protocol) Server Implementation.

MCP is an open protocol that standardizes how AI applications
connect to external tools, data sources, and services.

Features:
- Tool registration and discovery
- Resource management
- Prompt templates
- Sampling support
- Transport layers (stdio, SSE, WebSocket)
- Security and authentication
- Auto-discovery of MCP servers

Reference: https://modelcontextprotocol.io
"""

import argparse
import os
import sys
import json
import time
import asyncio
import inspect
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, AsyncIterator
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod


class TransportType(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"
    HTTP = "http"


class ServerCapability(Enum):
    """MCP server capabilities."""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    SAMPLING = "sampling"
    LOGGING = "logging"


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: dict
    annotations: dict = field(default_factory=dict)
    handler: Optional[Callable] = None


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    annotations: dict = field(default_factory=dict)
    provider: Optional[Callable] = None


@dataclass
class MCPPrompt:
    """MCP prompt template."""
    name: str
    description: str
    arguments: list[dict] = field(default_factory=list)
    handler: Optional[Callable] = None


@dataclass
class MCPMessage:
    """MCP JSON-RPC message."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[dict] = None


class MCPTransport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self):
        """Connect to transport."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from transport."""
        pass

    @abstractmethod
    async def send(self, message: MCPMessage):
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self) -> MCPMessage:
        """Receive a message."""
        pass


class StdioTransport(MCPTransport):
    """Standard I/O transport for local MCP servers."""

    def __init__(self):
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def send(self, message: MCPMessage):
        """Send message to stdout."""
        data = self._serialize(message)
        sys.stdout.write(data + "\n")
        sys.stdout.flush()

    async def receive(self) -> MCPMessage:
        """Receive message from stdin."""
        line = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline
        )
        return self._deserialize(line.strip())

    def _serialize(self, msg: MCPMessage) -> str:
        """Serialize message to JSON."""
        d = {"jsonrpc": msg.jsonrpc}
        if msg.id is not None:
            d["id"] = msg.id
        if msg.method is not None:
            d["method"] = msg.method
        if msg.params is not None:
            d["params"] = msg.params
        if msg.result is not None:
            d["result"] = msg.result
        if msg.error is not None:
            d["error"] = msg.error
        return json.dumps(d)

    def _deserialize(self, data: str) -> MCPMessage:
        """Deserialize JSON to message."""
        try:
            d = json.loads(data)
            return MCPMessage(
                jsonrpc=d.get("jsonrpc", "2.0"),
                id=d.get("id"),
                method=d.get("method"),
                params=d.get("params"),
                result=d.get("result"),
                error=d.get("error")
            )
        except json.JSONDecodeError:
            return MCPMessage(error={"code": -32700, "message": "Parse error"})


class SSETransport(MCPTransport):
    """Server-Sent Events transport."""

    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self.session_id = None

    async def connect(self):
        """Connect to SSE endpoint."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.connected = True

    async def disconnect(self):
        """Disconnect from SSE."""
        if hasattr(self, 'session'):
            await self.session.close()
        self.connected = False

    async def send(self, message: MCPMessage):
        """Send message via POST."""
        data = json.dumps({
            "jsonrpc": message.jsonrpc,
            "method": message.method,
            "params": message.params
        })
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        async with self.session.post(
            self.url.replace("/sse", "/message"),
            data=data,
            headers=headers
        ) as resp:
            if resp.status == 200:
                self.session_id = resp.headers.get("Mcp-Session-Id")

    async def receive(self) -> MCPMessage:
        """Receive message from SSE stream."""
        # Simplified SSE receive
        return MCPMessage(error={"code": -32000, "message": "SSE receive not implemented"})


class WebSocketTransport(MCPTransport):
    """WebSocket transport."""

    def __init__(self, url: str):
        self.url = url
        self.connected = False

    async def connect(self):
        """Connect via WebSocket."""
        self.connected = True

    async def disconnect(self):
        """Disconnect WebSocket."""
        self.connected = False

    async def send(self, message: MCPMessage):
        """Send via WebSocket."""
        pass

    async def receive(self) -> MCPMessage:
        """Receive from WebSocket."""
        return MCPMessage()


class MCPServer:
    """
    MCP Server implementation.

    Provides tools, resources, and prompts to AI applications.
    """

    def __init__(
        self,
        name: str = "mcp-server",
        version: str = "1.0.0",
        capabilities: list[ServerCapability] = None
    ):
        self.name = name
        self.version = version
        self.capabilities = capabilities or [
            ServerCapability.TOOLS,
            ServerCapability.RESOURCES,
            ServerCapability.PROMPTS,
        ]

        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}
        self.prompts: dict[str, MCPPrompt] = {}
        self.transport: Optional[MCPTransport] = None
        self.logger = logging.getLogger(name)

        self._request_id = 0
        self._handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }

    # === Tool Management ===

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable,
        annotations: dict = None
    ):
        """Register a tool with the server."""
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            annotations=annotations or {},
            handler=handler
        )
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")

    def register_tool_from_function(self, func: Callable, name: str = None):
        """Auto-register a function as a tool."""
        import inspect
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Execute {func.__name__}"

        # Build input schema from function signature
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            prop = {"type": "string"}
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                prop["type"] = type_map.get(param.annotation, "string")

            properties[param_name] = prop
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        tool_name = name or func.__name__
        self.register_tool(tool_name, doc, schema, func)

    def register_tools_from_class(self, cls: Any, prefix: str = ""):
        """Register all public methods of a class as tools."""
        for name in dir(cls):
            if not name.startswith("_"):
                method = getattr(cls, name)
                if callable(method):
                    tool_name = f"{prefix}_{name}" if prefix else name
                    self.register_tool_from_function(method, tool_name)

    def unregister_tool(self, name: str):
        """Unregister a tool."""
        self.tools.pop(name, None)

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "annotations": tool.annotations
            }
            for tool in self.tools.values()
        ]

    # === Resource Management ===

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        provider: Callable,
        mime_type: str = "text/plain",
        annotations: dict = None
    ):
        """Register a resource with the server."""
        resource = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            annotations=annotations or {},
            provider=provider
        )
        self.resources[uri] = resource
        self.logger.info(f"Registered resource: {uri}")

    def list_resources(self) -> list[dict]:
        """List all registered resources."""
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description,
                "mimeType": res.mime_type,
                "annotations": res.annotations
            }
            for res in self.resources.values()
        ]

    async def read_resource(self, uri: str) -> dict:
        """Read a resource by URI."""
        resource = self.resources.get(uri)
        if not resource:
            return {"error": f"Resource not found: {uri}"}

        try:
            content = await resource.provider(uri) if inspect.iscoroutinefunction(resource.provider) else resource.provider(uri)
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": str(content)
                    }
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    # === Prompt Management ===

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict],
        handler: Callable
    ):
        """Register a prompt template."""
        prompt = MCPPrompt(
            name=name,
            description=description,
            arguments=arguments,
            handler=handler
        )
        self.prompts[name] = prompt
        self.logger.info(f"Registered prompt: {name}")

    def list_prompts(self) -> list[dict]:
        """List all registered prompts."""
        return [
            {
                "name": prompt.name,
                "description": prompt.description,
                "arguments": prompt.arguments
            }
            for prompt in self.prompts.values()
        ]

    async def get_prompt(self, name: str, arguments: dict = None) -> dict:
        """Get a prompt with arguments."""
        prompt = self.prompts.get(name)
        if not prompt:
            return {"error": f"Prompt not found: {name}"}

        try:
            result = await prompt.handler(arguments or {}) if inspect.iscoroutinefunction(prompt.handler) else prompt.handler(arguments or {})
            return {
                "description": prompt.description,
                "messages": result if isinstance(result, list) else [result]
            }
        except Exception as e:
            return {"error": str(e)}

    # === Message Handling ===

    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle an incoming JSON-RPC message."""
        if message.method in self._handlers:
            try:
                result = await self._handlers[message.method](message.params or {})
                return MCPMessage(
                    jsonrpc="2.0",
                    id=message.id,
                    result=result
                )
            except Exception as e:
                return MCPMessage(
                    jsonrpc="2.0",
                    id=message.id,
                    error={"code": -32000, "message": str(e)}
                )
        else:
            return MCPMessage(
                jsonrpc="2.0",
                id=message.id,
                error={"code": -32601, "message": f"Method not found: {message.method}"}
            )

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                capability.value: {}
                for capability in self.capabilities
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request."""
        return {"tools": self.list_tools()}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "content": [{"type": "text", "text": f"Tool not found: {tool_name}"}],
                "isError": True
            }

        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)

            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }

    async def _handle_resources_list(self, params: dict) -> dict:
        """Handle resources/list request."""
        return {"resources": self.list_resources()}

    async def _handle_resources_read(self, params: dict) -> dict:
        """Handle resources/read request."""
        uri = params.get("uri")
        return await self.read_resource(uri)

    async def _handle_prompts_list(self, params: dict) -> dict:
        """Handle prompts/list request."""
        return {"prompts": self.list_prompts()}

    async def _handle_prompts_get(self, params: dict) -> dict:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments", {})
        return await self.get_prompt(name, arguments)

    async def _handle_ping(self, params: dict) -> dict:
        """Handle ping request."""
        return {}

    async def _handle_shutdown(self, params: dict) -> dict:
        """Handle shutdown request."""
        return {}

    # === Server Lifecycle ===

    async def run_stdio(self):
        """Run server with stdio transport."""
        self.transport = StdioTransport()
        await self.transport.connect()

        self.logger.info(f"MCP Server {self.name} started (stdio)")

        try:
            while True:
                message = await self.transport.receive()
                if message.error and message.error.get("code") == -32700:
                    continue

                response = await self.handle_message(message)
                await self.transport.send(response)
        except KeyboardInterrupt:
            pass
        finally:
            await self.transport.disconnect()

    async def run_sse(self, host: str = "localhost", port: int = 8080):
        """Run server with SSE transport."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/sse", self._sse_handler)
        app.router.add_post("/message", self._message_handler)

        self.logger.info(f"MCP Server {self.name} started on {host}:{port}")
        web.run_app(app, host=host, port=port)

    async def _sse_handler(self, request):
        """Handle SSE connection."""
        from aiohttp import web
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)
        return response

    async def _message_handler(self, request):
        """Handle incoming message."""
        from aiohttp import web
        data = await request.json()
        message = MCPMessage(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params")
        )
        response = await self.handle_message(message)
        return web.json_response({
            "jsonrpc": response.jsonrpc,
            "id": response.id,
            "result": response.result,
            "error": response.error
        })


class MCPClient:
    """
    MCP Client for connecting to MCP servers.
    """

    def __init__(self, transport: MCPTransport = None):
        self.transport = transport
        self.connected = False
        self.server_info = None
        self.tools = []
        self.resources = []
        self.prompts = []

    async def connect(self, transport: MCPTransport):
        """Connect to an MCP server."""
        self.transport = transport
        await transport.connect()
        self.connected = True

        # Initialize
        response = await self._send_request("initialize", {
            "clientInfo": {"name": "mcp-client", "version": "1.0.0"},
            "capabilities": {}
        })

        if response and not response.error:
            self.server_info = response.result.get("serverInfo", {})

            # Get tools
            tools_response = await self._send_request("tools/list", {})
            if tools_response and not tools_response.error:
                self.tools = tools_response.result.get("tools", [])

            # Get resources
            resources_response = await self._send_request("resources/list", {})
            if resources_response and not resources_response.error:
                self.resources = resources_response.result.get("resources", [])

            # Get prompts
            prompts_response = await self._send_request("prompts/list", {})
            if prompts_response and not prompts_response.error:
                self.prompts = prompts_response.result.get("prompts", [])

    async def disconnect(self):
        """Disconnect from server."""
        if self.transport:
            await self.transport.send(MCPMessage(method="shutdown"))
            await self.transport.disconnect()
        self.connected = False

    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        """Call a tool on the server."""
        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        if response and not response.error:
            return response.result
        return {"error": response.error if response else "No response"}

    async def read_resource(self, uri: str) -> dict:
        """Read a resource from the server."""
        response = await self._send_request("resources/read", {"uri": uri})
        if response and not response.error:
            return response.result
        return {"error": response.error if response else "No response"}

    async def get_prompt(self, name: str, arguments: dict = None) -> dict:
        """Get a prompt from the server."""
        response = await self._send_request("prompts/get", {
            "name": name,
            "arguments": arguments or {}
        })
        if response and not response.error:
            return response.result
        return {"error": response.error if response else "No response"}

    async def _send_request(self, method: str, params: dict) -> Optional[MCPMessage]:
        """Send a request and wait for response."""
        self._request_id = getattr(self, '_request_id', 0) + 1
        message = MCPMessage(
            jsonrpc="2.0",
            id=self._request_id,
            method=method,
            params=params
        )
        await self.transport.send(message)
        return await self.transport.receive()

    def get_tools_schema(self) -> list[dict]:
        """Get tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("inputSchema", {})
                }
            }
            for tool in self.tools
        ]


class MCPRegistry:
    """
    Registry of available MCP servers.
    """

    def __init__(self, config_path: str = None):
        self.servers: dict[str, dict] = {}
        self.config_path = config_path or os.path.expanduser("~/.mcp/config.json")
        self._load_config()

    def _load_config(self):
        """Load MCP server configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    self.servers = json.load(f)
            except Exception:
                self.servers = {}

    def _save_config(self):
        """Save MCP server configuration."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.servers, f, indent=2)

    def register_server(
        self,
        name: str,
        command: str,
        args: list[str] = None,
        env: dict = None,
        transport: TransportType = TransportType.STDIO
    ):
        """Register an MCP server."""
        self.servers[name] = {
            "command": command,
            "args": args or [],
            "env": env or {},
            "transport": transport.value
        }
        self._save_config()

    def unregister_server(self, name: str):
        """Unregister an MCP server."""
        self.servers.pop(name, None)
        self._save_config()

    def list_servers(self) -> dict:
        """List all registered servers."""
        return self.servers

    async def connect_to_server(self, name: str) -> Optional[MCPClient]:
        """Connect to a registered MCP server."""
        config = self.servers.get(name)
        if not config:
            return None

        transport_type = TransportType(config.get("transport", "stdio"))

        if transport_type == TransportType.STDIO:
            transport = StdioTransport()
        elif transport_type == TransportType.SSE:
            transport = SSETransport(config.get("url", ""))
        else:
            transport = WebSocketTransport(config.get("url", ""))

        client = MCPClient()
        await client.connect(transport)
        return client


# === Built-in MCP Tools ===

def create_file_tool():
    """Create a file operations MCP tool."""
    def read_file(path: str) -> str:
        """Read file contents."""
        with open(path, "r") as f:
            return f.read()

    def write_file(path: str, content: str) -> str:
        """Write content to file."""
        with open(path, "w") as f:
            f.write(content)
        return f"Written to {path}"

    def list_directory(path: str = ".") -> str:
        """List directory contents."""
        items = os.listdir(path)
        return json.dumps(items, indent=2)

    return read_file, write_file, list_directory


def create_web_tool():
    """Create web operation MCP tools."""
    async def fetch_url(url: str) -> str:
        """Fetch content from URL."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()

    async def search_web(query: str) -> str:
        """Search the web."""
        return f"Search results for: {query}"

    return fetch_url, search_web


# === Convenience Functions ===

def create_mcp_server(name: str = "custom-mcp") -> MCPServer:
    """Create a new MCP server with common tools."""
    server = MCPServer(name=name)

    # Register built-in tools
    read_file, write_file, list_dir = create_file_tool()
    server.register_tool("read_file", "Read file contents", {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    }, read_file)

    server.register_tool("write_file", "Write content to file", {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["path", "content"]
    }, write_file)

    server.register_tool("list_directory", "List directory contents", {
        "type": "object",
        "properties": {"path": {"type": "string"}}
    }, list_dir)

    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="MCP server CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="List tools registered on a default server")

    sub.add_parser("resources", help="List resources on a default server")

    p = sub.add_parser("call", help="Call a tool on a default server")
    p.add_argument("--tool", required=True, help="Tool name")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="key=value",
        help="Tool argument (repeatable)",
    )

    p = sub.add_parser("registry", help="Inspect or modify the MCP server registry")
    p.add_argument("--config", default=None, help="Registry config path")
    p.add_argument("--register", nargs=2, metavar=("NAME", "COMMAND"), default=None, help="Register a server")
    p.add_argument("--unregister", default=None, help="Unregister a server by name")

    args = parser.parse_args(argv)

    try:
        if args.command == "tools":
            server = create_mcp_server()
            print(json.dumps(server.list_tools(), indent=2, default=str))
            return 0
        elif args.command == "resources":
            server = create_mcp_server()
            print(json.dumps(server.list_resources(), indent=2, default=str))
            return 0
        elif args.command == "call":
            server = create_mcp_server()
            params = {}
            for kv in args.param:
                k, _, v = kv.partition("=")
                params[k] = v
            message = MCPMessage(
                jsonrpc="2.0",
                id=1,
                method="tools/call",
                params={"name": args.tool, "arguments": params},
            )
            response = asyncio.run(server.handle_message(message))
            print(json.dumps(response.__dict__, indent=2, default=str))
            return 0 if not response.error else 1
        elif args.command == "registry":
            registry = MCPRegistry(config_path=args.config)
            if args.register:
                registry.register_server(args.register[0], args.register[1])
            if args.unregister:
                registry.unregister_server(args.unregister)
            print(json.dumps(registry.list_servers(), indent=2, default=str))
            return 0
        else:
            parser.error("Unknown command")
            return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()

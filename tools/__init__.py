"""
Tool Calling and Tool Use System (Claude-compatible).

Provides complete tool calling infrastructure:
- Tool registry for registering and managing tools
- Tool parser for extracting tool calls from model output
- Tool executor for running tool calls safely
- Claude-compatible API format for tool_use and tool_result
- Shell tools for terminal, PowerShell, and Windows system control

Compatible with Claude's tool_use API format.
"""

from .tool_registry import ToolRegistry, Tool, ToolParameter
from .tool_parser import ToolParser, ParsedToolCall
from .tool_executor import ToolExecutor, ExecutionResult
from .claude_compatible import (
    ClaudeToolUse,
    ClaudeToolResult,
    ClaudeToolUseBlock,
    ClaudeToolResultBlock,
    ClaudeToolChoice,
    format_tools_for_api,
    parse_tool_use_from_response,
    format_tool_results_for_api,
    create_tool_use_message,
    create_tool_result_message,
)
from . import powershell_tool
from . import terminal_tool
from . import windows_tools

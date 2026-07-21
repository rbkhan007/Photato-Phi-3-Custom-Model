"""
Claude-Compatible Tool Use API Format.

Provides complete compatibility with Claude's tool_use API including:
- tool_use content blocks in assistant messages
- tool_result content blocks in user messages
- tool_choice configuration
- Streaming tool call events
- Multi-turn tool use conversations
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ClaudeToolUseBlock:
    """Claude tool_use content block."""
    id: str
    name: str
    input: dict

    def to_dict(self) -> dict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass
class ClaudeToolResultBlock:
    """Claude tool_result content block."""
    tool_use_id: str
    content: str
    is_error: bool = False
    cache_control: dict = None

    def to_dict(self) -> dict:
        block = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        if self.cache_control:
            block["cache_control"] = self.cache_control
        return block


@dataclass
class ClaudeToolUse:
    """Complete Claude tool_use message for assistant responses."""
    tool_uses: list[ClaudeToolUseBlock] = field(default_factory=list)

    def add_tool_use(self, id: str, name: str, input: dict) -> ClaudeToolUseBlock:
        block = ClaudeToolUseBlock(id=id, name=name, input=input)
        self.tool_uses.append(block)
        return block

    def to_content(self) -> list[dict]:
        """Convert to content blocks for an assistant message."""
        return [tu.to_dict() for tu in self.tool_uses]

    @classmethod
    def from_dict(cls, data: dict) -> "ClaudeToolUse":
        """Parse from a Claude API response dict."""
        tool_uses = []
        content = data.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses.append(ClaudeToolUseBlock(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
        return cls(tool_uses=tool_uses)


@dataclass
class ClaudeToolResult:
    """Complete Claude tool_result message for user responses."""
    results: list[ClaudeToolResultBlock] = field(default_factory=list)

    def add_result(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> ClaudeToolResultBlock:
        block = ClaudeToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )
        self.results.append(block)
        return block

    def to_content(self) -> list[dict]:
        """Convert to content blocks for a user message."""
        return [r.to_dict() for r in self.results]

    @classmethod
    def from_execution_results(cls, execution_results: list) -> "ClaudeToolResult":
        """Create from a list of ExecutionResult objects."""
        result = cls()
        for er in execution_results:
            content = er.output if er.success else f"Error: {er.error}"
            if isinstance(content, (dict, list)):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            result.add_result(
                tool_use_id=er.tool_use_id,
                content=content,
                is_error=er.is_error or not er.success,
            )
        return result


@dataclass
class ClaudeToolChoice:
    """Configuration for tool_choice in API requests."""
    type: str  # "auto", "any", "tool"
    name: Optional[str] = None  # Required when type="tool"

    def to_dict(self) -> dict:
        result = {"type": self.type}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def auto(cls) -> "ClaudeToolChoice":
        return cls(type="auto")

    @classmethod
    def any(cls) -> "ClaudeToolChoice":
        return cls(type="any")

    @classmethod
    def tool(cls, name: str) -> "ClaudeToolChoice":
        return cls(type="tool", name=name)

    @classmethod
    def none(cls) -> "ClaudeToolChoice":
        return cls(type="none")


def format_tools_for_api(tools: list[dict]) -> list[dict]:
    """
    Format tool definitions for Claude API.

    Each tool should have: name, description, input_schema
    """
    formatted = []
    for tool in tools:
        formatted_tool = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
        formatted.append(formatted_tool)
    return formatted


def format_tools_from_registry(registry) -> list[dict]:
    """Format tools from a ToolRegistry for Claude API."""
    return format_tools_for_api(registry.get_schemas())


def parse_tool_use_from_response(response: dict) -> list[ClaudeToolUseBlock]:
    """Parse tool_use blocks from a Claude API response."""
    blocks = []
    content = response.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                blocks.append(ClaudeToolUseBlock(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))
    return blocks


def format_tool_results_for_api(
    results: list[dict],
) -> list[dict]:
    """
    Format tool results for Claude API.

    Each result should have: tool_use_id, content
    """
    blocks = []
    for result in results:
        block = {
            "type": "tool_result",
            "tool_use_id": result.get("tool_use_id", ""),
            "content": result.get("content", ""),
        }
        if result.get("is_error"):
            block["is_error"] = True
        blocks.append(block)
    return blocks


def create_tool_use_message(
    tool_calls: list[dict],
) -> dict:
    """
    Create an assistant message containing tool_use blocks.

    Args:
        tool_calls: List of dicts with id, name, input

    Returns:
        Assistant message with tool_use content
    """
    content = []
    for tc in tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{int(time.time() * 1000)}"),
            "name": tc.get("name", ""),
            "input": tc.get("input", {}),
        })
    return {
        "role": "assistant",
        "content": content,
    }


def create_tool_result_message(
    results: list[dict],
) -> dict:
    """
    Create a user message containing tool_result blocks.

    Args:
        results: List of dicts with tool_use_id, content, is_error (optional)

    Returns:
        User message with tool_result content
    """
    content = []
    for r in results:
        block = {
            "type": "tool_result",
            "tool_use_id": r.get("tool_use_id", ""),
            "content": r.get("content", ""),
        }
        if r.get("is_error"):
            block["is_error"] = True
        content.append(block)
    return {
        "role": "user",
        "content": content,
    }


def build_tool_use_conversation(
    messages: list[dict],
    tool_results: list[dict],
) -> list[dict]:
    """
    Build a complete multi-turn conversation with tool use.

    This appends tool results to the conversation after assistant tool_use blocks.
    """
    conversation = list(messages)
    tool_result_msg = create_tool_result_message(tool_results)
    conversation.append(tool_result_msg)
    return conversation


def should_use_tools(
    response: dict,
    tool_choice: dict = None,
) -> bool:
    """
    Determine if the response contains tool calls that should be executed.

    Checks:
    1. Response has tool_use content blocks
    2. tool_choice configuration allows tool use
    """
    content = response.get("content", [])
    has_tool_use = False
    if isinstance(content, list):
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )

    if not has_tool_use:
        return False

    if tool_choice:
        choice_type = tool_choice.get("type", "auto")
        if choice_type == "none":
            return False

    return True


def create_assistant_response(
    text: str = None,
    tool_calls: list[dict] = None,
) -> dict:
    """
    Create a complete assistant response with text and/or tool calls.

    Args:
        text: Optional text response
        tool_calls: Optional list of tool call dicts

    Returns:
        Assistant message dict
    """
    content = []
    if text:
        content.append({
            "type": "text",
            "text": text,
        })
    if tool_calls:
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{int(time.time() * 1000)}"),
                "name": tc.get("name", ""),
                "input": tc.get("input", {}),
            })
    return {
        "role": "assistant",
        "content": content,
    }


def extract_text_from_response(response: dict) -> str:
    """Extract text content from an assistant response."""
    content = response.get("content", [])
    if isinstance(content, str):
        return content
    texts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
    return "\n".join(texts)


def extract_tool_calls_from_response(response: dict) -> list[ClaudeToolUseBlock]:
    """Extract tool_use blocks from an assistant response."""
    return parse_tool_use_from_response(response)

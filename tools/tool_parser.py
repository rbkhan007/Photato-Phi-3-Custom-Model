"""
Tool Parser for extracting tool calls from model output.

Parses various formats of tool calls including:
- JSON blocks in model text
- Claude tool_use blocks
- XML-style tool calls
- Function call patterns
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedToolCall:
    """A parsed tool call from model output."""
    id: str
    name: str
    input: dict
    raw_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "confidence": self.confidence,
        }


class ToolParser:
    """
    Parse tool calls from model output text.

    Supports multiple parsing strategies:
    - JSON block extraction
    - Claude tool_use format
    - XML-style tool calls
    - Regex-based function call patterns
    """

    def __init__(self, registry=None):
        self._registry = registry
        self._call_counter = 0

    def _generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        self._call_counter += 1
        return f"toolu_{self._call_counter:06d}"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """
        Parse tool calls from model output text.

        Tries multiple parsing strategies in order:
        1. Claude tool_use JSON blocks
        2. Generic JSON tool calls
        3. XML-style tool calls
        4. Function call patterns
        """
        calls = []

        # Strategy 1: Claude tool_use format
        claude_calls = self._parse_claude_format(text)
        if claude_calls:
            calls.extend(claude_calls)

        # Strategy 2: JSON tool calls
        if not calls:
            json_calls = self._parse_json_blocks(text)
            if json_calls:
                calls.extend(json_calls)

        # Strategy 3: XML-style tool calls
        if not calls:
            xml_calls = self._parse_xml_tools(text)
            if xml_calls:
                calls.extend(xml_calls)

        # Strategy 4: Function call patterns
        if not calls:
            func_calls = self._parse_function_calls(text)
            if func_calls:
                calls.extend(func_calls)

        return calls

    def _parse_claude_format(self, text: str) -> list[ParsedToolCall]:
        """Parse Claude tool_use format blocks."""
        calls = []
        pattern = r'\{[^{}]*"type"\s*:\s*"tool_use"[^{}]*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                block = json.loads(match.group())
                if block.get("type") == "tool_use":
                    calls.append(ParsedToolCall(
                        id=block.get("id", self._generate_call_id()),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                        raw_text=match.group(),
                    ))
            except json.JSONDecodeError:
                continue
        return calls

    def _parse_json_blocks(self, text: str) -> list[ParsedToolCall]:
        """Parse JSON tool call blocks from text."""
        calls = []
        json_pattern = r'\{[^{}]*"tool"\s*:\s*[^{}]*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                block = json.loads(match.group())
                tool_name = block.get("tool") or block.get("name") or block.get("function")
                if tool_name:
                    params = block.get("parameters") or block.get("arguments") or block.get("input") or {}
                    calls.append(ParsedToolCall(
                        id=self._generate_call_id(),
                        name=tool_name,
                        input=params,
                        raw_text=match.group(),
                    ))
            except json.JSONDecodeError:
                continue

        # Also try to find tool calls wrapped in ```json blocks
        code_block_pattern = r'```(?:json)?\s*\n(\{[^`]+\})\s*\n```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                block = json.loads(match.group(1))
                tool_name = block.get("tool") or block.get("name") or block.get("function")
                if tool_name:
                    params = block.get("parameters") or block.get("arguments") or block.get("input") or {}
                    calls.append(ParsedToolCall(
                        id=self._generate_call_id(),
                        name=tool_name,
                        input=params,
                        raw_text=match.group(1),
                    ))
            except json.JSONDecodeError:
                continue

        return calls

    def _parse_xml_tools(self, text: str) -> list[ParsedToolCall]:
        """Parse XML-style tool calls."""
        calls = []
        pattern = r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>'
        for match in re.finditer(pattern, text, re.DOTALL):
            tool_name = match.group(1)
            content = match.group(2).strip()
            try:
                params = json.loads(content)
            except json.JSONDecodeError:
                params = {"text": content}
            calls.append(ParsedToolCall(
                id=self._generate_call_id(),
                name=tool_name,
                input=params,
                raw_text=match.group(),
            ))
        return calls

    def _parse_function_calls(self, text: str) -> list[ParsedToolCall]:
        """Parse function call patterns like: function_name(arg1, arg2)"""
        calls = []
        pattern = r'(\w+)\(([^)]*)\)'
        for match in re.finditer(pattern, text):
            func_name = match.group(1)
            args_str = match.group(2).strip()

            # Skip common non-tool patterns
            if func_name in ("print", "len", "str", "int", "float", "range", "type", "isinstance", "dict", "list", "set", "tuple"):
                continue

            # Check if it matches a registered tool
            if self._registry and not self._registry.has(func_name):
                continue

            # Parse arguments
            params = {}
            if args_str:
                # Try to parse as key=value pairs
                for arg in args_str.split(","):
                    arg = arg.strip()
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        params[key] = value
                    elif arg:
                        params[f"arg{len(params)}"] = arg.strip('"').strip("'")

            calls.append(ParsedToolCall(
                id=self._generate_call_id(),
                name=func_name,
                input=params,
                raw_text=match.group(),
                confidence=0.7,
            ))
        return calls

    def parse_from_messages(self, messages: list[dict]) -> list[ParsedToolCall]:
        """Parse tool calls from a list of chat messages."""
        calls = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                calls.extend(self.parse(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        calls.append(ParsedToolCall(
                            id=block.get("id", self._generate_call_id()),
                            name=block.get("name", ""),
                            input=block.get("input", {}),
                        ))
        return calls

    def extract_tool_calls_from_response(self, response: dict) -> list[ParsedToolCall]:
        """Extract tool calls from a Claude API response format."""
        calls = []
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append(ParsedToolCall(
                        id=block.get("id", self._generate_call_id()),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    ))
        return calls

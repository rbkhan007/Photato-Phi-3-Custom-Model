"""
Tool Registry for managing tool definitions and metadata.

Provides registration, lookup, validation, and schema generation for tools.
"""

import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""
    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: list[str] = field(default_factory=list)
    default: Any = None

    def to_schema(self) -> dict:
        """Convert to JSON Schema property format."""
        schema = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """Complete tool definition."""
    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    def to_tool_schema(self) -> dict:
        """Convert to Claude API tool format."""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate_input(self, input_data: dict) -> tuple[bool, str]:
        """Validate input data against parameter definitions."""
        for param in self.parameters:
            if param.required and param.name not in input_data:
                return False, f"Missing required parameter: {param.name}"
            if param.name in input_data:
                value = input_data[param.name]
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{param.name}' must be a string"
                elif param.type == "integer" and not isinstance(value, int):
                    return False, f"Parameter '{param.name}' must be an integer"
                elif param.type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter '{param.name}' must be a number"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{param.name}' must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter '{param.name}' must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return False, f"Parameter '{param.name}' must be an object"
                if param.enum and value not in param.enum:
                    return False, f"Parameter '{param.name}' must be one of: {param.enum}"
        return True, ""


class ToolRegistry:
    """
    Registry for managing tool definitions.

    Supports:
    - Registration from decorators, classes, or manual addition
    - Lookup by name, category, or tags
    - Schema generation for Claude API format
    - Input validation
    - Auto-discovery from Python functions
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}
        self._tags: dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        handler: Callable,
        category: str = "general",
        tags: list[str] = None,
        examples: list[dict] = None,
    ) -> Tool:
        """Register a new tool."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category=category,
            tags=tags or [],
            examples=examples or [],
        )
        self._tools[name] = tool

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        for tag in tool.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(name)

        return tool

    def register_function(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        category: str = "general",
        tags: list[str] = None,
    ) -> Tool:
        """Auto-register a Python function as a tool by inspecting its signature."""
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = param.annotation
            param_type = "string"
            if annotation == int:
                param_type = "integer"
            elif annotation == float:
                param_type = "number"
            elif annotation == bool:
                param_type = "boolean"
            elif annotation == list:
                param_type = "array"
            elif annotation == dict:
                param_type = "object"

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=param.default == inspect.Parameter.empty,
                default=None if param.default == inspect.Parameter.empty else param.default,
            ))

        return self.register(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
            category=category,
            tags=tags,
        )

    def register_class_tools(self, instance: object, category: str = "general"):
        """Register all public methods of a class as tools."""
        for method_name in dir(instance):
            if method_name.startswith("_"):
                continue
            method = getattr(instance, method_name)
            if callable(method):
                self.register_function(
                    func=method,
                    name=f"{instance.__class__.__name__.lower()}_{method_name}",
                    description=method.__doc__ or f"Execute {method_name}",
                    category=category,
                )

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def get_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_by_tag(self, tag: str) -> list[Tool]:
        """Get all tools with a specific tag."""
        names = self._tags.get(tag, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_schemas(self, tool_names: list[str] = None) -> list[dict]:
        """Get Claude API compatible tool schemas."""
        if tool_names:
            tools = [self._tools[n] for n in tool_names if n in self._tools]
        else:
            tools = list(self._tools.values())
        return [t.to_tool_schema() for t in tools]

    def validate_input(self, tool_name: str, input_data: dict) -> tuple[bool, str]:
        """Validate input for a specific tool."""
        tool = self._tools.get(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"
        return tool.validate_input(input_data)

    def remove(self, name: str) -> bool:
        """Remove a tool by name."""
        if name not in self._tools:
            return False
        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category] = [
                n for n in self._categories[tool.category] if n != name
            ]
        for tag in tool.tags:
            if tag in self._tags:
                self._tags[tag] = [n for n in self._tags[tag] if n != name]
        return True

    def clear(self):
        """Remove all tools."""
        self._tools.clear()
        self._categories.clear()
        self._tags.clear()

    def export_json(self) -> str:
        """Export all tool schemas as JSON."""
        return json.dumps(self.get_schemas(), indent=2)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={list(self._categories.keys())})"

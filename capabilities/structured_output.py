#!/usr/bin/env python3
"""
Structured Output (JSON Mode) for Local LLMs.

Features:
- JSON mode with schema validation
- Structured output generation
- Schema-based parsing
- Output format enforcement

Usage:
    from capabilities.structured_output import StructuredOutput

    so = StructuredOutput()
    schema = {"name": "string", "age": "integer"}
    result = so.generate_json("Create a person", schema)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OutputSchema:
    """Output schema definition."""
    name: str
    fields: dict[str, str]  # field_name -> type
    required: list[str] = field(default_factory=list)
    description: str = ""


class JSONValidator:
    """JSON validation utilities."""

    @staticmethod
    def validate_json(text: str) -> tuple[bool, Any]:
        """
        Validate if text is valid JSON.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, parsed_json)
        """
        try:
            parsed = json.loads(text)
            return True, parsed
        except json.JSONDecodeError:
            return False, None

    @staticmethod
    def extract_json(text: str) -> Optional[str]:
        """
        Extract JSON from text.

        Args:
            text: Text containing JSON

        Returns:
            Extracted JSON string or None
        """
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Try to find JSON array
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return None

    @staticmethod
    def validate_schema(data: dict, schema: dict) -> tuple[bool, list[str]]:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: Schema definition

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        for field_name, field_type in schema.items():
            if field_name not in data:
                errors.append(f"Missing field: {field_name}")
                continue

            value = data[field_name]
            if not JSONValidator._check_type(value, field_type):
                errors.append(f"Invalid type for {field_name}: expected {field_type}, got {type(value).__name__}")

        return len(errors) == 0, errors

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "str": str,
            "integer": int,
            "int": int,
            "float": float,
            "number": (int, float),
            "boolean": bool,
            "bool": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        expected = type_map.get(expected_type.lower())
        if expected is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected)


class StructuredOutput:
    """
    Structured output generator for local LLMs.

    Features:
    - JSON mode with schema validation
    - Structured output generation
    - Schema-based prompting
    - Output format enforcement
    """

    def __init__(self):
        self.validator = JSONValidator()

    def generate_json_prompt(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        examples: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate prompt for JSON output.

        Args:
            prompt: User prompt
            schema: Output schema
            examples: Example outputs

        Returns:
            Prompt with JSON instructions
        """
        parts = [prompt, "\nRespond with valid JSON."]

        if schema:
            schema_str = json.dumps(schema, indent=2)
            parts.append(f"\nExpected JSON schema:\n```json\n{schema_str}\n```")

        if examples:
            examples_str = json.dumps(examples[:2], indent=2)
            parts.append(f"\nExample output:\n```json\n{examples_str}\n```")

        parts.append("\nJSON output:")
        return "\n".join(parts)

    def extract_and_validate(
        self,
        response: str,
        schema: Optional[dict] = None,
    ) -> tuple[bool, Any, list[str]]:
        """
        Extract and validate JSON from response.

        Args:
            response: Model response
            schema: Expected schema

        Returns:
            Tuple of (success, data, errors)
        """
        # Extract JSON
        json_str = self.validator.extract_json(response)
        if not json_str:
            return False, None, ["No JSON found in response"]

        # Parse JSON
        is_valid, parsed = self.validator.validate_json(json_str)
        if not is_valid:
            return False, None, ["Invalid JSON format"]

        # Validate schema
        if schema:
            schema_valid, errors = self.validator.validate_schema(parsed, schema)
            if not schema_valid:
                return False, parsed, errors

        return True, parsed, []

    def format_response(
        self,
        data: dict,
        format_type: str = "json",
    ) -> str:
        """
        Format data as structured output.

        Args:
            data: Data to format
            format_type: Output format (json, yaml, table)

        Returns:
            Formatted string
        """
        if format_type == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif format_type == "table":
            return self._format_as_table(data)

        else:
            return str(data)

    def _format_as_table(self, data: dict) -> str:
        """Format dict as table."""
        lines = ["| Key | Value |", "|-----|-------|"]
        for key, value in data.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)

    def create_schema_from_example(self, example: dict) -> dict:
        """
        Create schema from example.

        Args:
            example: Example dict

        Returns:
            Schema dict
        """
        schema = {}
        for key, value in example.items():
            if isinstance(value, bool):
                schema[key] = "boolean"
            elif isinstance(value, int):
                schema[key] = "integer"
            elif isinstance(value, float):
                schema[key] = "number"
            elif isinstance(value, list):
                schema[key] = "array"
            elif isinstance(value, dict):
                schema[key] = "object"
            else:
                schema[key] = "string"
        return schema

    def ensure_json_mode(self, prompt: str) -> str:
        """Add JSON mode instructions to prompt."""
        return f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON. No other text.
The response must be parseable by json.loads()."""


class JSONMode:
    """JSON mode handler."""

    def __init__(self):
        self.validator = JSONValidator()

    def parse_response(self, response: str) -> Optional[dict]:
        """Parse JSON from response."""
        json_str = self.validator.extract_json(response)
        if json_str:
            is_valid, parsed = self.validator.validate_json(json_str)
            if is_valid and isinstance(parsed, dict):
                return parsed
        return None

    def ensure_json(self, text: str) -> str:
        """Ensure text is valid JSON."""
        is_valid, _ = self.validator.validate_json(text)
        if is_valid:
            return text

        # Try to extract JSON
        json_str = self.validator.extract_json(text)
        if json_str:
            is_valid, _ = self.validator.validate_json(json_str)
            if is_valid:
                return json_str

        return text


class SchemaBuilder:
    """Build schemas programmatically."""

    @staticmethod
    def person_schema() -> dict:
        """Person schema."""
        return {
            "name": "string",
            "age": "integer",
            "email": "string",
            "is_active": "boolean",
        }

    @staticmethod
    def response_schema() -> dict:
        """API response schema."""
        return {
            "status": "string",
            "data": "object",
            "message": "string",
        }

    @staticmethod
    def list_schema(item_schema: dict) -> dict:
        """List response schema."""
        return {
            "items": "array",
            "total": "integer",
            "page": "integer",
        }

    @staticmethod
    def custom(**fields: str) -> dict:
        """Custom schema from keyword args."""
        return dict(fields)


def main(argv=None):
    """CLI for structured output (JSON mode)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.structured_output",
        description="Structured output (JSON mode) for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input(p):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--input", help="Input text (or @file to read from file)")
        src.add_argument("--file", help="Read input from file")

    p_val = sub.add_parser("validate", help="Extract and validate JSON from a response")
    add_input(p_val)
    p_val.add_argument("--schema", help="JSON schema (inline JSON or @file path)")

    p_prompt = sub.add_parser("prompt", help="Build a JSON-mode prompt")
    p_prompt.add_argument("--prompt", required=True)
    p_prompt.add_argument("--schema", help="JSON schema (inline JSON or @file path)")
    p_prompt.add_argument("--examples", help="JSON array of example objects (inline or @file)")
    p_prompt.add_argument("--format", choices=["json", "plain"], default="plain")

    p_schema = sub.add_parser("schema", help="Build a schema")
    p_schema.add_argument("--kind", choices=["person", "response", "list", "custom"], default="person")
    p_schema.add_argument("--fields", help="For custom: comma-separated key:type pairs")
    p_schema.add_argument("--item-schema", help="For list: inline JSON schema of items")

    p_ensure = sub.add_parser("ensure", help="Ensure/normalize JSON")
    add_input(p_ensure)

    args = parser.parse_args(argv)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                return f.read()
        if value.startswith("@"):
            with open(value[1:], "r", encoding="utf-8-sig") as f:
                return f.read()
        return value

    def parse_schema(raw):
        if raw is None:
            return None
        if raw.startswith("@"):
            with open(raw[1:], "r", encoding="utf-8-sig") as f:
                return json.load(f)
        return json.loads(raw)

    try:
        so = StructuredOutput()
        if args.command == "validate":
            response = read_source(args.input, args.file)
            schema = parse_schema(args.schema)
            success, data, errors = so.extract_and_validate(response, schema)
            print(json.dumps({
                "success": success,
                "data": data,
                "errors": errors,
            }, indent=2, default=str))
        elif args.command == "prompt":
            schema = parse_schema(args.schema)
            examples = None
            if args.examples:
                examples = json.loads(read_source(args.examples, None))
            prompt = so.generate_json_prompt(args.prompt, schema=schema, examples=examples)
            if args.format == "json":
                print(json.dumps({"prompt": prompt}, indent=2))
            else:
                print(prompt)
        elif args.command == "schema":
            if args.kind == "person":
                schema = SchemaBuilder.person_schema()
            elif args.kind == "response":
                schema = SchemaBuilder.response_schema()
            elif args.kind == "list":
                schema = SchemaBuilder.list_schema({})
            else:
                fields = {}
                if args.fields:
                    for part in args.fields.split(","):
                        key, _, typ = part.partition(":")
                        fields[key.strip()] = (typ.strip() or "string")
                schema = fields
            print(json.dumps(schema, indent=2))
        elif args.command == "ensure":
            text = read_source(args.input, args.file)
            jm = JSONMode()
            print(jm.ensure_json(text))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

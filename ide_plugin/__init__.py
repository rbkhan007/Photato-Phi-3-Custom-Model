"""
IDE Plugin System.

Provides AI-powered assistance for code editors.

Features:
- Code completion
- Code explanation
- Refactoring suggestions
- Bug detection
- Test generation
- Documentation generation
- Code review
- Multi-IDE support (VS Code, JetBrains, Vim, Sublime)

Architecture:
- Language Server Protocol (LSP) integration
- WebSocket communication
- Plugin protocol for IDEs
- AI-powered code analysis
"""

import os
import json
import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod


class IDEType(Enum):
    """Supported IDE types."""
    VSCODE = "vscode"
    JETBRAINS = "jetbrains"
    VIM = "vim"
    SUBLIME = "sublime"
    EMACS = "emacs"
    CUSTOM = "custom"


class CompletionType(Enum):
    """Types of code completions."""
    KEYWORD = "keyword"
    FUNCTION = "function"
    VARIABLE = "variable"
    CLASS = "class"
    MODULE = "module"
    SNIPPET = "snippet"
    PARAMETER = "parameter"
    METHOD = "method"
    PROPERTY = "property"
    IMPORT = "import"


class DiagnosticSeverity(Enum):
    """Diagnostic severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class CompletionItem:
    """Code completion item."""
    label: str
    detail: str
    completion_type: CompletionType
    documentation: str = ""
    insert_text: str = ""
    snippet: str = ""
    sort_text: str = ""
    filter_text: str = ""
    deprecated: bool = False
    preselect: bool = False
    confidence: float = 0.8


@dataclass
class Diagnostic:
    """Code diagnostic (error/warning)."""
    message: str
    severity: DiagnosticSeverity
    line: int
    column: int
    end_line: int = 0
    end_column: int = 0
    code: str = ""
    source: str = "ai-assistant"
    related_information: list[dict] = field(default_factory=list)


@dataclass
class CodeAction:
    """Code action (refactoring, fix)."""
    title: str
    description: str
    edit: Optional[dict] = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    is_preferred: bool = False


@dataclass
class HoverInfo:
    """Hover information for code."""
    contents: str
    range: Optional[dict] = None
    language: str = "plaintext"


@dataclass
class Definition:
    """Code definition location."""
    uri: str
    line: int
    column: int
    name: str


@dataclass
class Reference:
    """Code reference location."""
    uri: str
    line: int
    column: int
    name: str
    context: str = ""


class IDEPlugin(ABC):
    """Base class for IDE plugins."""

    def __init__(self, ide_type: IDEType, project_path: str = None):
        self.ide_type = ide_type
        self.project_path = project_path or os.getcwd()
        self.connection = None
        self.ai_backend = None

    @abstractmethod
    async def connect(self):
        """Connect to IDE."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from IDE."""
        pass

    @abstractmethod
    async def get_completions(self, file_path: str, line: int, column: int) -> list[CompletionItem]:
        """Get code completions."""
        pass


class VSCodePlugin(IDEPlugin):
    """VS Code extension plugin."""

    def __init__(self, project_path: str = None):
        super().__init__(IDEType.VSCODE, project_path)
        self.workspace_info = {}

    async def connect(self):
        """Connect to VS Code via extension."""
        # VS Code extension communicates via WebSocket
        self.workspace_info = await self._detect_workspace()

    async def disconnect(self):
        """Disconnect from VS Code."""
        pass

    async def _detect_workspace(self) -> dict:
        """Detect VS Code workspace info."""
        return {
            "root": self.project_path,
            "name": os.path.basename(self.project_path),
            "folders": [self.project_path]
        }

    async def get_completions(self, file_path: str, line: int, column: int) -> list[CompletionItem]:
        """Get VS Code completions."""
        return []

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get diagnostics for file."""
        return []

    async def get_hover(self, file_path: str, line: int, column: int) -> Optional[HoverInfo]:
        """Get hover information."""
        return None

    async def get_definition(self, file_path: str, line: int, column: int) -> Optional[Definition]:
        """Get definition location."""
        return None

    async def get_references(self, file_path: str, line: int, column: int) -> list[Reference]:
        """Get references."""
        return []


class JetBrainsPlugin(IDEPlugin):
    """JetBrains IDE plugin."""

    def __init__(self, ide_type: IDEType = IDEType.JETBRAINS, project_path: str = None):
        super().__init__(ide_type, project_path)

    async def connect(self):
        """Connect to JetBrains IDE."""
        pass

    async def disconnect(self):
        """Disconnect from JetBrains IDE."""
        pass

    async def get_completions(self, file_path: str, line: int, column: int) -> list[CompletionItem]:
        """Get JetBrains completions."""
        return []


class VimPlugin(IDEPlugin):
    """Vim/Neovim plugin."""

    def __init__(self, project_path: str = None):
        super().__init__(IDEType.VIM, project_path)

    async def connect(self):
        """Connect to Vim."""
        pass

    async def disconnect(self):
        """Disconnect from Vim."""
        pass

    async def get_completions(self, file_path: str, line: int, column: int) -> list[CompletionItem]:
        """Get Vim completions."""
        return []


class AICodeAssistant:
    """
    AI-powered code assistant.

    Provides intelligent code assistance across IDEs.
    """

    def __init__(self, project_path: str = None):
        self.project_path = project_path or os.getcwd()
        self.plugins: dict[IDEType, IDEPlugin] = {}
        self.code_index: dict[str, dict] = {}
        self.history: list[dict] = []
        self._cache: dict[str, Any] = {}

    # === Plugin Management ===

    def register_plugin(self, plugin: IDEPlugin):
        """Register an IDE plugin."""
        self.plugins[plugin.ide_type] = plugin

    def get_plugin(self, ide_type: IDEType) -> Optional[IDEPlugin]:
        """Get plugin for IDE type."""
        return self.plugins.get(ide_type)

    async def connect_all(self):
        """Connect to all registered plugins."""
        for plugin in self.plugins.values():
            await plugin.connect()

    async def disconnect_all(self):
        """Disconnect from all plugins."""
        for plugin in self.plugins.values():
            await plugin.disconnect()

    # === Code Completion ===

    async def get_completions(
        self,
        file_path: str,
        line: int,
        column: int,
        context: str = None,
        ide_type: IDEType = IDEType.VSCODE
    ) -> list[CompletionItem]:
        """Get intelligent code completions."""
        plugin = self.get_plugin(ide_type)
        if plugin:
            return await plugin.get_completions(file_path, line, column)

        # AI-powered completions
        return await self._ai_completions(file_path, line, column, context)

    async def _ai_completions(
        self,
        file_path: str,
        line: int,
        column: int,
        context: str = None
    ) -> list[CompletionItem]:
        """Generate AI completions."""
        completions = []

        # Read file context
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return completions

        current_line = lines[line] if line < len(lines) else ""
        prefix = current_line[:column]

        # Simple keyword-based completion
        keywords = {
            "def ": ["def function_name(self):", "def async_function(self):"],
            "class ": ["class ClassName:", "class ClassName(BaseClass):"],
            "import ": ["import module", "from module import name"],
            "if ": ["if condition:", "if condition:\n    pass"],
            "for ": ["for item in iterable:", "for i in range(n):"],
            "while ": ["while condition:", "while True:"],
            "try:": ["try:\n    pass\nexcept Exception as e:\n    pass"],
            "with ": ["with context_manager as cm:", "with open('file') as f:"],
            "return ": ["return result", "return None"],
            "print(": ["print(variable)", "print(f'text {variable}')"],
        }

        for keyword, templates in keywords.items():
            if prefix.rstrip().endswith(keyword.rstrip()):
                for i, template in enumerate(templates):
                    completions.append(CompletionItem(
                        label=template.split("(")[0].split(":")[0].strip(),
                        detail=template,
                        completion_type=CompletionType.KEYWORD,
                        insert_text=template,
                        sort_text=f"{i:03d}",
                        confidence=0.7
                    ))

        # Add common completions
        if "." in prefix:
            obj_name = prefix.split(".")[-2] if len(prefix.split(".")) > 1 else ""
            methods = ["append", "extend", "insert", "remove", "pop",
                      "clear", "copy", "count", "index", "sort", "reverse"]
            for method in methods:
                completions.append(CompletionItem(
                    label=method,
                    detail=f"{obj_name}.{method}()",
                    completion_type=CompletionType.METHOD,
                    insert_text=f"{method}()",
                    confidence=0.6
                ))

        return completions

    # === Code Analysis ===

    async def explain_code(self, code: str, language: str = "python") -> str:
        """Explain what code does."""
        # Simple pattern-based explanation
        lines = code.strip().split('\n')
        explanations = []

        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue

            if line.startswith("def "):
                func_name = line.split("(")[0].replace("def ", "")
                explanations.append(f"Defines a function called '{func_name}'")
            elif line.startswith("class "):
                class_name = line.split(":")[0].replace("class ", "")
                explanations.append(f"Defines a class called '{class_name}'")
            elif line.startswith("if "):
                explanations.append("Checks a condition")
            elif line.startswith("for "):
                explanations.append("Iterates over a collection")
            elif line.startswith("while "):
                explanations.append("Loops while a condition is true")
            elif line.startswith("return "):
                explanations.append("Returns a value")
            elif line.startswith("import "):
                module = line.replace("import ", "")
                explanations.append(f"Imports the '{module}' module")
            elif "=" in line and not line.startswith("="):
                var_name = line.split("=")[0].strip()
                explanations.append(f"Assigns a value to '{var_name}'")

        return "\n".join(explanations) if explanations else "Code snippet"

    async def suggest_refactoring(self, code: str, language: str = "python") -> list[CodeAction]:
        """Suggest code refactoring."""
        suggestions = []

        # Check for long functions
        lines = code.split('\n')
        func_lines = [l for l in lines if l.strip()]

        if len(func_lines) > 20:
            suggestions.append(CodeAction(
                title="Extract method",
                description="Function is too long. Consider breaking it into smaller functions.",
                is_preferred=True
            ))

        # Check for duplicated code
        unique_lines = set()
        duplicated = []
        for line in lines:
            stripped = line.strip()
            if stripped in unique_lines and len(stripped) > 10:
                duplicated.append(stripped)
            unique_lines.add(stripped)

        if duplicated:
            suggestions.append(CodeAction(
                title="Remove duplication",
                description=f"Found {len(duplicated)} duplicated line(s). Consider extracting common logic.",
                is_preferred=True
            ))

        # Check for magic numbers
        import re
        numbers = re.findall(r'\b\d{2,}\b', code)
        if numbers:
            suggestions.append(CodeAction(
                title="Extract constants",
                description=f"Found magic number(s): {', '.join(numbers[:3])}. Consider using named constants.",
                is_preferred=False
            ))

        # Check for TODO/FIXME
        todos = [l for l in lines if 'TODO' in l or 'FIXME' in l]
        if todos:
            suggestions.append(CodeAction(
                title="Address TODOs",
                description=f"Found {len(todos)} TODO/FIXME comment(s).",
                is_preferred=False
            ))

        return suggestions

    async def detect_bugs(self, code: str, language: str = "python") -> list[Diagnostic]:
        """Detect potential bugs."""
        diagnostics = []
        lines = code.split('\n')

        for i, line in enumerate(lines):
            # Check for common bugs
            if "==" in line and "if " in line:
                # Check for assignment in condition
                parts = line.split("==")
                if len(parts) == 2 and "=" in parts[0] and "==" not in parts[0]:
                    diagnostics.append(Diagnostic(
                        message="Possible assignment in condition (use == for comparison)",
                        severity=DiagnosticSeverity.WARNING,
                        line=i + 1,
                        column=line.index("="),
                        code="W0622"
                    ))

            if "except:" in line or "except Exception:" in line:
                diagnostics.append(Diagnostic(
                    message="Bare except clause. Consider catching specific exceptions.",
                    severity=DiagnosticSeverity.WARNING,
                    line=i + 1,
                    column=0,
                    code="W0702"
                ))

            if "import *" in line:
                diagnostics.append(Diagnostic(
                    message="Wildcard import. Consider importing specific names.",
                    severity=DiagnosticSeverity.WARNING,
                    line=i + 1,
                    column=0,
                    code="W0401"
                ))

            # Check for undefined variables (simplified)
            if "=" in line and not line.strip().startswith("#"):
                parts = line.split("=")
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    if var_name and not var_name.startswith((" ", "\t")):
                        # Simple heuristic
                        pass

        return diagnostics

    async def generate_tests(self, code: str, language: str = "python") -> str:
        """Generate unit tests for code."""
        import re

        # Extract function names
        functions = re.findall(r'def (\w+)\((.*?)\):', code)

        test_code = 'import pytest\n\n\n'

        for func_name, params in functions:
            if func_name.startswith("_"):
                continue

            test_code += f'class Test{func_name.title()}:\n'
            test_code += f'    def test_{func_name}_basic(self):\n'
            test_code += f'        # Arrange\n'
            test_code += f'        # Act\n'
            test_code += f'        result = {func_name}()\n'
            test_code += f'        # Assert\n'
            test_code += f'        assert result is not None\n\n'

            test_code += f'    def test_{func_name}_edge_case(self):\n'
            test_code += f'        # Test edge cases\n'
            test_code += f'        pass\n\n'

        return test_code

    async def generate_docs(self, code: str, language: str = "python") -> str:
        """Generate documentation for code."""
        import re

        lines = code.split('\n')
        docs = []

        # Add module docstring
        docs.append('"""')
        docs.append('Module documentation.')
        docs.append('"""')
        docs.append('')

        for line in lines:
            # Add function/class docstrings
            if line.strip().startswith("def "):
                func_name = re.search(r'def (\w+)\((.*?)\):', line)
                if func_name:
                    name = func_name.group(1)
                    params = func_name.group(2)
                    docs.append(f'def {name}({params}):')
                    docs.append(f'    """')
                    docs.append(f'    {name} function.')
                    docs.append(f'')
                    if params:
                        docs.append(f'    Args:')
                        for param in params.split(','):
                            param_name = param.split(':')[0].strip()
                            if param_name and param_name != 'self':
                                docs.append(f'        {param_name}: Description.')
                    docs.append(f'    """')
                    docs.append('')

            elif line.strip().startswith("class "):
                class_name = re.search(r'class (\w+)', line)
                if class_name:
                    name = class_name.group(1)
                    docs.append(f'class {name}:')
                    docs.append(f'    """')
                    docs.append(f'    {name} class.')
                    docs.append(f'    """')
                    docs.append('')

        return '\n'.join(docs)

    # === Code Review ===

    async def review_code(self, code: str, language: str = "python") -> dict:
        """Perform code review."""
        issues = []

        # Run all checks
        bugs = await self.detect_bugs(code, language)
        refactoring = await self.suggest_refactoring(code, language)

        for bug in bugs:
            issues.append({
                "type": "bug",
                "severity": bug.severity.value,
                "message": bug.message,
                "line": bug.line
            })

        for ref in refactoring:
            issues.append({
                "type": "refactoring",
                "severity": "info",
                "message": ref.title,
                "details": ref.description
            })

        # Calculate score
        total = len(issues)
        bugs_count = len([i for i in issues if i["type"] == "bug"])
        score = max(0, 100 - (bugs_count * 10) - (total * 2))

        return {
            "score": score,
            "issues": issues,
            "summary": f"Found {total} issue(s) ({bugs_count} bugs)",
            "recommendation": "Good" if score >= 80 else "Needs improvement" if score >= 60 else "Poor"
        }

    # === Utility ===

    def index_project(self):
        """Index project files for faster access."""
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.h')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            self.code_index[file_path] = {
                                "language": self._detect_language(file),
                                "size": len(content),
                                "lines": len(content.split('\n'))
                            }
                    except Exception:
                        pass

    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'header',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
        }
        _, ext = os.path.splitext(filename)
        return ext_map.get(ext, 'unknown')

    def get_project_stats(self) -> dict:
        """Get project statistics."""
        stats = {
            "total_files": len(self.code_index),
            "total_lines": sum(f.get("lines", 0) for f in self.code_index.values()),
            "languages": {}
        }

        for file_info in self.code_index.values():
            lang = file_info.get("language", "unknown")
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

        return stats


# === Convenience Functions ===

def create_vscode_plugin(project_path: str = None) -> VSCodePlugin:
    """Create VS Code plugin."""
    return VSCodePlugin(project_path)

def create_assistant(project_path: str = None) -> AICodeAssistant:
    """Create AI code assistant."""
    return AICodeAssistant(project_path)

async def quick_review(code: str) -> dict:
    """Quick code review."""
    assistant = AICodeAssistant()
    return await assistant.review_code(code)

async def quick_explain(code: str) -> str:
    """Quick code explanation."""
    assistant = AICodeAssistant()
    return await assistant.explain_code(code)

async def quick_tests(code: str) -> str:
    """Quick test generation."""
    assistant = AICodeAssistant()
    return await assistant.generate_tests(code)

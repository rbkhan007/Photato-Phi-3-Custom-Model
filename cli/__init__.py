"""
Agentic CLI System - Digital CLI like Claude Code.

A complete agentic command-line interface that can:
- Execute commands with context awareness
- Manage files and projects
- Run code in multiple languages
- Use tools and MCPs
- Maintain conversation memory
- Self-heal from errors
"""

import os
import sys
import json
import time
import subprocess
import platform
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    output: Any = None
    success: bool = True
    duration_ms: float = 0


@dataclass
class Session:
    id: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    working_dir: str = ""
    created_at: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)


class AgenticCLI:
    """
    Complete agentic CLI system for managing projects, code, and tasks.

    Features:
    - Context-aware command execution
    - Multi-language code execution
    - File system operations
    - Tool calling with MCP integration
    - Conversation memory
    - Self-healing error recovery
    - Graph-based knowledge management
    """

    #: Base directory for persisted config, sessions, and history.
    HOME_DIR = Path(os.environ.get("AGENTIC_CLI_HOME", Path.home() / ".agentic_cli"))

    def __init__(self, working_dir: str = None, config: dict = None, backend=None):
        self.working_dir = working_dir or os.getcwd()
        self.session = Session(
            id=hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            working_dir=self.working_dir,
        )
        self.tools: dict[str, Callable] = {}
        self.history: list[dict] = []
        self.config: dict = {}
        self._backend = backend
        self._setup_tools()
        self._system_info = self._detect_system()
        self._load_config(config)

    # === Config & Persistence ===

    def _paths(self) -> dict:
        """Filesystem locations used for persistence."""
        home = self.HOME_DIR
        return {
            "home": home,
            "config": home / "config.json",
            "sessions": home / "sessions",
            "history": home / "history.log",
        }

    def _load_config(self, override: dict = None):
        """Load config from disk (if present), then apply any overrides."""
        cfg = {
            "backend": "llamacpp",
            "model": r"G:\LOACL ai models\phi3-custom-model\notebooks\Phi-4-mini-instruct-Q4_K_M.gguf",
            "ollama_host": "http://localhost:11434",
            "openai_base_url": "http://localhost:8080/v1",
            "temperature": 0.7,
            "max_tokens": 1024,
            "system_prompt": "You are a helpful agentic coding assistant.",
        }
        path = self._paths()["config"]
        try:
            if path.exists():
                cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
        if override:
            cfg.update({k: v for k, v in override.items() if v is not None})
        self.config = cfg
        return cfg

    def save_config(self) -> dict:
        """Persist the current config to disk."""
        paths = self._paths()
        try:
            paths["home"].mkdir(parents=True, exist_ok=True)
            paths["config"].write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            return {"success": True, "path": str(paths["config"])}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def save_session(self) -> dict:
        """Persist the current session to disk as JSON."""
        paths = self._paths()
        try:
            paths["sessions"].mkdir(parents=True, exist_ok=True)
            dest = paths["sessions"] / f"{self.session.id}.json"
            dest.write_text(self.export_session(), encoding="utf-8")
            return {"success": True, "path": str(dest), "session_id": self.session.id}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def load_session(self, session_id: str) -> dict:
        """Load a previously saved session from disk."""
        paths = self._paths()
        src = paths["sessions"] / f"{session_id}.json"
        if not src.exists():
            return {"success": False, "error": f"Session not found: {session_id}"}
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return {"success": False, "error": str(e)}
        self.session.id = data.get("session_id", self.session.id)
        self.session.working_dir = data.get("working_dir", self.working_dir)
        self.session.messages = [
            Message(role=MessageRole(m["role"]), content=m["content"], timestamp=m.get("timestamp", time.time()))
            for m in data.get("messages", [])
        ]
        return {"success": True, "session_id": self.session.id, "messages": len(self.session.messages)}

    def list_sessions(self) -> dict:
        """List saved session IDs."""
        paths = self._paths()
        if not paths["sessions"].exists():
            return {"success": True, "sessions": []}
        ids = sorted(p.stem for p in paths["sessions"].glob("*.json"))
        return {"success": True, "sessions": ids, "count": len(ids)}

    def _append_history(self, entry: str):
        """Append a line to the persistent command history log."""
        paths = self._paths()
        try:
            paths["home"].mkdir(parents=True, exist_ok=True)
            with open(paths["history"], "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now().isoformat()}\t{entry}\n")
        except OSError:
            pass

    # === Model Inference ===

    @property
    def backend(self):
        """Lazily construct the model backend from config."""
        if self._backend is None:
            from cli.model_backend import auto_backend, get_backend

            kind = self.config.get("backend", "auto")
            model = self.config.get("model")
            ngl = self.config.get("n_gpu_layers", 0)
            cpu_percent = self.config.get("cpu_percent", 55.0)
            if kind == "auto":
                self._backend = auto_backend(model, n_gpu_layers=ngl)
            else:
                self._backend = get_backend(
                    kind,
                    model=model or "phi3",
                    n_gpu_layers=ngl,
                    cpu_percent=cpu_percent,
                    host=self.config.get("ollama_host"),
                    base_url=self.config.get("openai_base_url"),
                )
        return self._backend

    def set_backend(self, backend):
        """Override the model backend (used for testing or explicit selection)."""
        self._backend = backend

    def chat(self, content: str, **kwargs) -> dict:
        """Send a user message to the model backend and record the exchange."""
        from cli.model_backend import BackendError

        self.send(content)
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in self.session.messages[-20:]:
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
                messages.append({"role": m.role.value, "content": m.content})
        try:
            result = self.backend.chat(
                messages,
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
                max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 1024)),
            )
        except BackendError as e:
            return {"success": False, "error": str(e)}
        self.respond(result.content)
        self._append_history(f"chat: {content[:80]}")
        return result.to_dict()

    def chat_stream(self, content: str, **kwargs):
        """Yield assistant response chunks, streaming when the backend supports it.

        Mirrors :meth:`chat` (records the exchange) but streams tokens for a
        responsive bot UX. Backends without a ``stream_chat`` method fall back
        to a single yielded block.
        """
        from cli.model_backend import BackendError

        self.send(content)
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in self.session.messages[-20:]:
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT):
                messages.append({"role": m.role.value, "content": m.content})
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1024))
        backend = self.backend
        streamer = getattr(backend, "stream_chat", None)
        start = time.time()
        try:
            if streamer is not None:
                chunks = []
                for chunk in streamer(messages, temperature=temperature, max_tokens=max_tokens):
                    yield chunk
                    chunks.append(chunk)
                text = "".join(chunks)
            else:
                result = backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
                text = result.content
                yield text
        except BackendError as e:
            yield f"[error: {e}]"
            return
        self.respond(text)
        self._append_history(f"chat: {content[:80]}")

        # Capture real token-usage metrics for display.
        elapsed = time.time() - start
        if streamer is not None:
            try:
                engine = backend.engine
                gen = len(engine.model.tokenize(text.encode("utf-8"))) if text else 0
                self.last_usage = {
                    "completion_tokens": gen,
                    "tokens_per_second": round(gen / elapsed, 2) if elapsed > 0 else 0.0,
                    "elapsed_s": round(elapsed, 2),
                }
            except Exception:
                self.last_usage = {}
        else:
            self.last_usage = dict(getattr(result, "usage", {}) or {})

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to the session working directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self.working_dir) / p

    def _detect_system(self) -> dict:
        """Detect system architecture for optimal execution."""
        import platform
        import os

        info = {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "cwd": os.getcwd(),
        }

        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram_gb"] = round(mem.total / (1024**3), 2)
            info["ram_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            pass

        return info

    def _setup_tools(self):
        """Register built-in tools."""
        self.tools = {
            "run_code": self._run_code,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_files": self._list_files,
            "search_files": self._search_files,
            "run_command": self._run_command,
            "git_status": self._git_status,
            "git_commit": self._git_commit,
            "analyze_code": self._analyze_code,
            "find_path": self._find_path,
            "chat": self.chat,
        }

    # === Tool Implementations ===

    def _run_code(self, language: str, code: str, **kwargs) -> dict:
        """Execute code in the specified language."""
        start = time.time()

        if language == "python":
            result = self._run_python(code)
        elif language in ["javascript", "js"]:
            result = self._run_javascript(code)
        elif language in ["typescript", "ts"]:
            result = self._run_typescript(code)
        elif language == "bash":
            result = self._run_bash(code)
        elif language == "go":
            result = self._run_go(code)
        elif language == "rust":
            result = self._run_rust(code)
        else:
            return {"success": False, "error": f"Unsupported language: {language}"}

        result["duration_ms"] = (time.time() - start) * 1000
        return result

    def _run_python(self, code: str) -> dict:
        """Run Python code."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out (30s limit)"}

    def _run_javascript(self, code: str) -> dict:
        """Run JavaScript code via Node.js."""
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except FileNotFoundError:
            return {"success": False, "error": "Node.js not installed"}

    def _run_typescript(self, code: str) -> dict:
        """Run TypeScript code via tsx."""
        try:
            result = subprocess.run(
                ["npx", "tsx", "-e", code],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_bash(self, code: str) -> dict:
        """Run bash command."""
        try:
            shell = "powershell" if platform.system() == "Windows" else "bash"
            result = subprocess.run(
                [shell, "-Command", code] if platform.system() == "Windows" else [shell, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_go(self, code: str) -> dict:
        """Run Go code."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["go", "run", f.name],
                    capture_output=True, text=True, timeout=30,
                    cwd=self.working_dir,
                )
                os.unlink(f.name)
                return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except FileNotFoundError:
            return {"success": False, "error": "Go not installed"}

    def _run_rust(self, code: str) -> dict:
        """Run Rust code via cargo-script or rustc."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["rustc", f.name, "-o", f.name + ".out"],
                    capture_output=True, text=True, timeout=60,
                    cwd=self.working_dir,
                )
                if result.returncode == 0:
                    exe = f.name + (".exe" if platform.system() == "Windows" else "")
                    run_result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
                    os.unlink(f.name)
                    os.unlink(exe)
                    return {"success": True, "stdout": run_result.stdout, "stderr": run_result.stderr}
                os.unlink(f.name)
                return {"success": False, "stderr": result.stderr}
        except FileNotFoundError:
            return {"success": False, "error": "Rust not installed"}

    def _read_file(self, path: str, encoding: str = "utf-8") -> dict:
        """Read file contents."""
        try:
            filepath = self._resolve(path).resolve()
            content = filepath.read_text(encoding=encoding)
            lines = content.split("\n")
            return {
                "success": True,
                "content": content,
                "lines": len(lines),
                "size": len(content.encode()),
                "path": str(filepath),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, path: str, content: str, encoding: str = "utf-8") -> dict:
        """Write file contents."""
        try:
            filepath = self._resolve(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding=encoding)
            return {"success": True, "path": str(filepath.resolve()), "size": len(content.encode())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_files(self, path: str = ".", recursive: bool = False, pattern: str = None) -> dict:
        """List files in directory."""
        try:
            dirpath = self._resolve(path)
            if recursive:
                files = list(dirpath.rglob(pattern or "*"))
            else:
                files = list(dirpath.glob(pattern or "*"))
            return {
                "success": True,
                "files": [{"name": f.name, "is_dir": f.is_dir(), "size": f.stat().st_size if f.is_file() else 0}
                          for f in files],
                "count": len(files),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_files(self, query: str, path: str = ".", extensions: list[str] = None) -> dict:
        """Search for text in files."""
        results = []
        dirpath = self._resolve(path)
        exts = extensions or [".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp", ".h"]

        for ext in exts:
            for filepath in dirpath.rglob(f"*{ext}"):
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.split("\n"), 1):
                        if query.lower() in line.lower():
                            results.append({
                                "file": str(filepath),
                                "line": i,
                                "content": line.strip(),
                            })
                except Exception:
                    continue

        return {"success": True, "results": results, "count": len(results)}

    def _run_command(self, command: str) -> dict:
        """Run a system command."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60,
                cwd=self.working_dir,
            )
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_status(self) -> dict:
        """Get git status."""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": True, "status": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_commit(self, message: str) -> dict:
        """Create a git commit."""
        try:
            subprocess.run(["git", "add", "."], cwd=self.working_dir)
            result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_code(self, path: str) -> dict:
        """Analyze code file for quality and metrics."""
        try:
            filepath = self._resolve(path)
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            analysis = {
                "total_lines": len(lines),
                "non_empty_lines": sum(1 for l in lines if l.strip()),
                "comment_lines": sum(1 for l in lines if l.strip().startswith("#") or l.strip().startswith("//")),
                "functions": sum(1 for l in lines if "def " in l or "function " in l or "func " in l),
                "classes": sum(1 for l in lines if "class " in l or "struct " in l),
                "imports": sum(1 for l in lines if "import " in l or "from " in l or "#include" in l),
                "avg_line_length": round(sum(len(l) for l in lines) / max(len(lines), 1), 1),
            }
            analysis["comment_ratio"] = round(analysis["comment_lines"] / max(analysis["total_lines"], 1), 3)

            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_path(self, start: str, goal: str) -> dict:
        """Find path in knowledge graph."""
        # Placeholder for graph integration
        return {"success": True, "path": [start, goal], "distance": 1}

    # === Message Handling ===

    def send(self, content: str) -> Message:
        """Send a user message."""
        msg = Message(role=MessageRole.USER, content=content)
        self.session.messages.append(msg)
        return msg

    def respond(self, content: str) -> Message:
        """Add an assistant response."""
        msg = Message(role=MessageRole.ASSISTANT, content=content)
        self.session.messages.append(msg)
        return msg

    def execute_tool(self, name: str, **kwargs) -> ToolCall:
        """Execute a tool call."""
        start = time.time()
        if name not in self.tools:
            call = ToolCall(
                id=f"call_{int(time.time()*1000)}",
                name=name, input=kwargs,
                output={"success": False, "error": f"Unknown tool: {name}"},
                success=False,
                duration_ms=0,
            )
        else:
            try:
                output = self.tools[name](**kwargs)
                call = ToolCall(
                    id=f"call_{int(time.time()*1000)}",
                    name=name, input=kwargs,
                    output=output,
                    success=output.get("success", True) if isinstance(output, dict) else True,
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                call = ToolCall(
                    id=f"call_{int(time.time()*1000)}",
                    name=name, input=kwargs,
                    output={"success": False, "error": str(e)},
                    success=False,
                    duration_ms=(time.time() - start) * 1000,
                )
        self.session.tool_calls.append(call)
        self.history.append({"tool": name, "input": kwargs, "success": call.success})
        return call

    # === Conversation Management ===

    def get_context(self) -> str:
        """Get conversation context as string."""
        context = []
        for msg in self.session.messages[-20:]:  # Last 20 messages
            context.append(f"[{msg.role.value}]: {msg.content[:500]}")
        return "\n".join(context)

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "session_id": self.session.id,
            "messages": len(self.session.messages),
            "tool_calls": len(self.session.tool_calls),
            "working_dir": self.working_dir,
            "system": self._system_info,
            "duration_seconds": time.time() - self.session.created_at,
        }

    def export_session(self) -> str:
        """Export session as JSON."""
        data = {
            "session_id": self.session.id,
            "working_dir": self.working_dir,
            "messages": [
                {"role": m.role.value, "content": m.content, "timestamp": m.timestamp}
                for m in self.session.messages
            ],
            "tool_calls": [
                {"name": tc.name, "input": tc.input, "success": tc.success, "duration_ms": tc.duration_ms}
                for tc in self.session.tool_calls
            ],
            "stats": self.get_stats(),
        }
        return json.dumps(data, indent=2, default=str)

    def clear(self):
        """Clear session history."""
        self.session.messages.clear()
        self.session.tool_calls.clear()


class CLIBuilder:
    """Build custom CLI configurations."""

    @staticmethod
    def create_project_cli(project_path: str) -> AgenticCLI:
        """Create a CLI configured for a specific project."""
        cli = AgenticCLI(project_path)

        # Auto-detect project type
        path = Path(project_path)
        if (path / "package.json").exists():
            cli.session.context["project_type"] = "node"
            cli.session.context["package_manager"] = "npm"
        elif (path / "pyproject.toml").exists() or (path / "setup.py").exists():
            cli.session.context["project_type"] = "python"
        elif (path / "Cargo.toml").exists():
            cli.session.context["project_type"] = "rust"
        elif (path / "go.mod").exists():
            cli.session.context["project_type"] = "go"
        elif (path / "pom.xml").exists():
            cli.session.context["project_type"] = "java"

        return cli

    @staticmethod
    def create_portable_cli(pendrive_path: str) -> AgenticCLI:
        """Create a portable CLI that runs from a pendrive."""
        cli = AgenticCLI(pendrive_path)
        cli.session.context["portable"] = True
        cli.session.context["pendrive_path"] = pendrive_path

        # Create necessary directories
        for subdir in ["sessions", "cache", "models", "data", "logs"]:
            (Path(pendrive_path) / subdir).mkdir(exist_ok=True)

        return cli


def demo():
    """Print a quick capability overview of the agentic CLI."""
    cli = AgenticCLI()
    print("Agentic CLI initialized")
    print(f"System: {json.dumps(cli._system_info, indent=2)}")

    result = cli.execute_tool("list_files", path=".", pattern="*.py")
    print(f"\nPython files: {result.output.get('count', 0)}")

    result = cli.execute_tool("analyze_code", path="cli/__init__.py")
    print(f"\nCode analysis: {json.dumps(result.output, indent=2)}")

    print(f"\nStats: {json.dumps(cli.get_stats(), indent=2)}")


def main(argv=None):
    """Real entry point: dispatch to the argparse-based CLI."""
    from cli.__main__ import main as _main

    return _main(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main())

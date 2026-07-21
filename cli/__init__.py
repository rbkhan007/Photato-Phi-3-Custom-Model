"""
Agentic CLI System - Digital CLI like Claude Code.

A complete agentic command-line interface that can:
- Execute commands with context awareness
- Manage files and projects
- Run code in multiple languages
- Use tools and MCPs
- Maintain conversation memory
- Self-heal from errors
- RAG (Retrieval-Augmented Generation)
- Safety filtering
- Extended thinking
- Structured output
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
    - Conversation memory with RAG
    - Self-healing error recovery
    - Graph-based knowledge management
    - Safety filtering
    - Extended thinking
    - Structured output
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
        self._rag_engine = None
        self._memory = None
        self._safety = None
        self._tool_registry = None
        self._extended_thinker = None
        self._setup_tools()
        self._setup_capabilities()
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

    def search_sessions(self, query: str) -> dict:
        """Search saved sessions for matching content."""
        paths = self._paths()
        if not paths["sessions"].exists():
            return {"success": True, "results": []}

        results = []
        query_lower = query.lower()
        for session_file in paths["sessions"].glob("*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                messages = data.get("messages", [])
                matches = []
                for msg in messages:
                    content = msg.get("content", "").lower()
                    if query_lower in content:
                        matches.append({
                            "role": msg.get("role"),
                            "content": msg.get("content", "")[:200],
                        })
                if matches:
                    results.append({
                        "session_id": session_file.stem,
                        "matches": matches[:5],
                        "total_matches": len(matches),
                    })
            except Exception:
                continue

        return {"success": True, "results": results, "query": query}

    def export_session_markdown(self, filepath: str = None) -> dict:
        """Export current session as markdown."""
        if not filepath:
            filepath = f"session_{self.session.id}.md"

        lines = [f"# Session {self.session.id}\n"]
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        lines.append(f"**Working Directory**: {self.working_dir}\n\n")
        lines.append("---\n\n")

        for msg in self.session.messages:
            role = msg.role.value.capitalize()
            lines.append(f"## {role}\n\n")
            lines.append(f"{msg.content}\n\n")

        try:
            Path(filepath).write_text("\n".join(lines), encoding="utf-8")
            return {"success": True, "filepath": filepath, "messages": len(self.session.messages)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_plugins(self) -> dict:
        """Load custom plugins from ~/.agentic_cli/plugins/."""
        plugins_dir = self.HOME_DIR / "plugins"
        if not plugins_dir.exists():
            plugins_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "loaded": 0, "message": "Plugins directory created"}

        loaded = []
        errors = []
        for plugin_file in plugins_dir.glob("*.py"):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, str(plugin_file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for register function
                if hasattr(module, "register"):
                    result = module.register(self)
                    loaded.append(plugin_file.stem)
            except Exception as e:
                errors.append({"file": plugin_file.name, "error": str(e)})

        return {"success": True, "loaded": len(loaded), "plugins": loaded, "errors": errors}

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
        """Send a user message to the model backend and record the exchange.

        Integrates RAG, memory, and safety filtering.
        """
        from cli.model_backend import BackendError

        # Safety check
        if self._safety:
            try:
                safety_result = self._safety.check_safety(content)
                if not safety_result.is_safe:
                    return {"success": False, "error": "Content blocked by safety filter"}
            except Exception:
                pass  # Skip safety if it fails

        # Memory: store user message
        if self._memory:
            try:
                self._memory.add(content, role="user")
            except Exception:
                pass

        # RAG: get relevant context
        rag_context = ""
        if self._rag_engine:
            try:
                results = self._rag_engine.query(content, top_k=3)
                if results:
                    rag_context = "\n\nRelevant context:\n" + "\n".join(
                        [r.get("content", "") for r in results]
                    )
            except Exception:
                pass

        self.send(content)
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + rag_context})
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

        # Memory: store assistant response
        if self._memory:
            try:
                self._memory.add(result.content, role="assistant")
            except Exception:
                pass

        self.respond(result.content)
        self._append_history(f"chat: {content[:80]}")
        return result.to_dict()

    def chat_stream(self, content: str, **kwargs):
        """Yield assistant response chunks, streaming when the backend supports it.

        Mirrors :meth:`chat` (records the exchange) but streams tokens for a
        responsive bot UX. Backends without a ``stream_chat`` method fall back
        to a single yielded block.

        Integrates RAG, memory, and safety filtering.
        """
        from cli.model_backend import BackendError

        # Safety check
        if self._safety:
            try:
                safety_result = self._safety.check_safety(content)
                if not safety_result.is_safe:
                    yield "[error: Content blocked by safety filter]"
                    return
            except Exception:
                pass  # Skip safety if it fails

        # Memory: store user message
        if self._memory:
            try:
                self._memory.add(content, role="user")
            except Exception:
                pass

        # RAG: get relevant context
        rag_context = ""
        if self._rag_engine:
            try:
                results = self._rag_engine.query(content, top_k=3)
                if results:
                    rag_context = "\n\nRelevant context:\n" + "\n".join(
                        [r.get("content", "") for r in results]
                    )
            except Exception:
                pass

        self.send(content)
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + rag_context})
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

        # Memory: store assistant response
        if self._memory:
            try:
                self._memory.add(text, role="assistant")
            except Exception:
                pass

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
        """Resolve a path - handles absolute, relative, and Windows paths."""
        if not path:
            return Path(self.working_dir)
        
        # Normalize path separators for Windows
        path = path.replace("/", "\\") if os.name == "nt" else path.replace("\\", "/")
        
        p = Path(path)
        if p.is_absolute():
            return p
        
        # Try to resolve relative to working directory
        resolved = Path(self.working_dir) / p
        return resolved

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
            "git_diff": self._git_diff,
            "git_log": self._git_log,
            "git_branch": self._git_branch,
            "git_checkout": self._git_checkout,
            "git_pull": self._git_pull,
            "git_push": self._git_push,
            "analyze_code": self._analyze_code,
            "find_path": self._find_path,
            "chat": self.chat,
            "mkdir": self._mkdir,
            "rmdir": self._rmdir,
            "copy_file": self._copy_file,
            "move_file": self._move_file,
            "delete_file": self._delete_file,
            "file_exists": self._file_exists,
            "get_disk_usage": self._get_disk_usage,
            "get_env": self._get_env,
            "set_env": self._set_env,
            "get_cwd": self._get_cwd,
            "set_cwd": self._set_cwd,
            "get_os_info": self._get_os_info,
            "get_process_list": self._get_process_list,
        }

    def _setup_capabilities(self):
        """Initialize advanced capabilities (RAG, memory, safety, etc.)."""
        import sys
        from io import StringIO

        # RAG Engine (suppress initialization output)
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            from capabilities.rag import RAGEngine
            self._rag_engine = RAGEngine()
            sys.stdout = old_stdout
        except Exception:
            sys.stdout = sys.__stdout__
            self._rag_engine = None

        # Conversation Memory
        try:
            from capabilities.memory import ConversationMemory
            self._memory = ConversationMemory()
        except Exception:
            self._memory = None

        # Safety Layer
        try:
            from capabilities.safety import SafetyLayer
            self._safety = SafetyLayer()
        except Exception:
            self._safety = None

        # Tool Registry (from tools module)
        try:
            from tools.tool_registry import ToolRegistry
            self._tool_registry = ToolRegistry()
        except Exception:
            self._tool_registry = None

        # Extended Thinker
        try:
            from capabilities.extended_thinking import ExtendedThinker
            self._extended_thinker = ExtendedThinker()
        except Exception:
            self._extended_thinker = None

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

    def _git_diff(self, file: str = None) -> dict:
        """Show git diff."""
        try:
            cmd = ["git", "diff"]
            if file:
                cmd.append(file)
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.working_dir)
            return {"success": True, "diff": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_log(self, count: int = 10) -> dict:
        """Show git log."""
        try:
            result = subprocess.run(
                ["git", "log", f"--oneline", f"-{count}"],
                capture_output=True, text=True, cwd=self.working_dir
            )
            return {"success": True, "log": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_branch(self) -> dict:
        """List git branches."""
        try:
            result = subprocess.run(["git", "branch", "-a"], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": True, "branches": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_checkout(self, branch: str) -> dict:
        """Checkout a git branch."""
        try:
            result = subprocess.run(["git", "checkout", branch], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_pull(self) -> dict:
        """Git pull."""
        try:
            result = subprocess.run(["git", "pull"], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_push(self) -> dict:
        """Git push."""
        try:
            result = subprocess.run(["git", "push"], capture_output=True, text=True, cwd=self.working_dir)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_env(self, var_name: str = None) -> dict:
        """Get environment variable(s)."""
        try:
            if var_name:
                value = os.environ.get(var_name)
                return {"success": True, "name": var_name, "value": value, "exists": value is not None}
            else:
                return {"success": True, "env": dict(os.environ)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _set_env(self, var_name: str, value: str) -> dict:
        """Set environment variable."""
        try:
            os.environ[var_name] = value
            return {"success": True, "name": var_name, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_cwd(self) -> dict:
        """Get current working directory."""
        return {"success": True, "cwd": self.working_dir, "actual_cwd": os.getcwd()}

    def _set_cwd(self, path: str) -> dict:
        """Change working directory."""
        try:
            new_path = self._resolve(path)
            if not new_path.exists():
                return {"success": False, "error": f"Directory not found: {new_path}"}
            if not new_path.is_dir():
                return {"success": False, "error": f"Not a directory: {new_path}"}
            self.working_dir = str(new_path.resolve())
            return {"success": True, "cwd": self.working_dir}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_os_info(self) -> dict:
        """Get OS information."""
        try:
            import platform
            info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }
            try:
                import psutil
                info["cpu_count"] = psutil.cpu_count()
                info["cpu_count_physical"] = psutil.cpu_count(logical=False)
                mem = psutil.virtual_memory()
                info["ram_total_gb"] = round(mem.total / (1024**3), 2)
                info["ram_available_gb"] = round(mem.available / (1024**3), 2)
                info["ram_percent"] = mem.percent
            except ImportError:
                pass
            return {"success": True, "os_info": info}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_process_list(self) -> dict:
        """Get running processes."""
        try:
            import psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return {"success": True, "processes": processes[:50]}  # Top 50
        except ImportError:
            # Fallback without psutil
            result = subprocess.run(
                ["tasklist" if os.name == "nt" else "ps aux"],
                shell=True, capture_output=True, text=True
            )
            return {"success": True, "processes": result.stdout[:2000]}

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

    def _mkdir(self, path: str, parents: bool = True) -> dict:
        """Create a directory at the given path."""
        try:
            filepath = self._resolve(path)
            filepath.mkdir(parents=parents, exist_ok=True)
            return {"success": True, "path": str(filepath.resolve()), "created": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _rmdir(self, path: str, recursive: bool = False) -> dict:
        """Remove a directory."""
        try:
            filepath = self._resolve(path)
            if not filepath.exists():
                return {"success": False, "error": f"Directory not found: {filepath}"}
            if filepath.is_file():
                return {"success": False, "error": f"Not a directory: {filepath}"}
            if recursive:
                import shutil
                shutil.rmtree(filepath)
            else:
                filepath.rmdir()
            return {"success": True, "path": str(filepath.resolve()), "removed": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _copy_file(self, source: str, destination: str) -> dict:
        """Copy a file from source to destination."""
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
            if not src.exists():
                return {"success": False, "error": f"Source not found: {src}"}
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                import shutil
                shutil.copytree(src, dst)
            else:
                import shutil
                shutil.copy2(src, dst)
            return {"success": True, "source": str(src.resolve()), "destination": str(dst.resolve())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _move_file(self, source: str, destination: str) -> dict:
        """Move a file from source to destination."""
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
            if not src.exists():
                return {"success": False, "error": f"Source not found: {src}"}
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(src), str(dst))
            return {"success": True, "source": str(src.resolve()), "destination": str(dst.resolve())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_file(self, path: str) -> dict:
        """Delete a file."""
        try:
            filepath = self._resolve(path)
            if not filepath.exists():
                return {"success": False, "error": f"File not found: {filepath}"}
            if filepath.is_dir():
                return {"success": False, "error": f"Is a directory, use rmdir: {filepath}"}
            filepath.unlink()
            return {"success": True, "path": str(filepath.resolve()), "deleted": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _file_exists(self, path: str) -> dict:
        """Check if a file or directory exists."""
        try:
            filepath = self._resolve(path)
            return {
                "success": True,
                "path": str(filepath.resolve()),
                "exists": filepath.exists(),
                "is_file": filepath.is_file() if filepath.exists() else False,
                "is_dir": filepath.is_dir() if filepath.exists() else False,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_disk_usage(self, path: str = ".") -> dict:
        """Get disk usage information for a path."""
        try:
            filepath = self._resolve(path)
            if not filepath.exists():
                return {"success": False, "error": f"Path not found: {filepath}"}
            
            import shutil
            usage = shutil.disk_usage(filepath)
            
            return {
                "success": True,
                "path": str(filepath.resolve()),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": round((usage.used / usage.total) * 100, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
    """Print a comprehensive capability overview of the agentic CLI."""
    cli = AgenticCLI()

    bar = "=" * 70
    print(bar)
    print("  PHI-3 CUSTOM MODEL - AGENTIC CLI")
    print("  A complete local AI coding assistant")
    print(bar)

    # System info
    print("\n[SYSTEM]")
    sys = cli._system_info
    print(f"  OS        : {sys.get('os')} {sys.get('arch')}")
    print(f"  Python    : {sys.get('python')}")
    print(f"  CPUs      : {sys.get('cpus')}")
    print(f"  RAM       : {sys.get('ram_gb', '?')} GB total")
    if 'ram_available_gb' in sys:
        print(f"  Available : {sys['ram_available_gb']} GB")

    # Backend
    print("\n[BACKEND]")
    backend = cli.backend
    print(f"  Active    : {backend.name}")
    model = getattr(backend, "model_path", None) or getattr(backend, "model", cli.config.get("model"))
    print(f"  Model     : {model}")

    # Capabilities
    print("\n[CAPABILITIES]")
    caps = [
        ("RAG Engine", cli._rag_engine is not None, "Retrieval-Augmented Generation with GGUF embeddings"),
        ("Memory", cli._memory is not None, "Conversation history tracking"),
        ("Safety Layer", cli._safety is not None, "Content filtering (toxicity, bias, jailbreak)"),
        ("Extended Thinking", cli._extended_thinker is not None, "Chain-of-thought reasoning"),
        ("Tool Registry", cli._tool_registry is not None, "Tool management system"),
    ]
    for name, ok, desc in caps:
        status = "ON" if ok else "OFF"
        print(f"  {name:<20} [{status}]  {desc}")

    # Tools
    print(f"\n[TOOLS] ({len(cli.tools)} available)")
    tool_groups = {
        "File System": ["read_file", "write_file", "list_files", "search_files", "mkdir", "rmdir", "copy_file", "move_file", "delete_file", "file_exists", "get_disk_usage"],
        "Git": ["git_status", "git_commit", "git_diff", "git_log", "git_branch", "git_checkout", "git_pull", "git_push"],
        "System": ["run_code", "run_command", "get_env", "set_env", "get_cwd", "set_cwd", "get_os_info", "get_process_list"],
        "Code": ["analyze_code", "find_path", "chat"],
    }
    for group, tools in tool_groups.items():
        available = [t for t in tools if t in cli.tools]
        print(f"  {group:<15}: {', '.join(available)}")

    # Code execution languages
    print("\n[CODE EXECUTION]")
    print("  Languages  : Python, JavaScript, TypeScript, Bash, Go, Rust")
    print("  Run code   : run-code --lang python --code 'print(42)'")

    # Example commands
    print("\n[EXAMPLES]")
    examples = [
        ('list *.py', "List Python files"),
        ('read cli/__init__.py', "Read a file"),
        ('search "def main"', "Search code"),
        ('git-status', "Show git status"),
        ('os', "Show OS info"),
        ('analyze cli/__init__.py', "Analyze code metrics"),
        ('run-code "print(2+2)"', "Execute Python code"),
        ('What is RAG?', "Chat with the AI"),
    ]
    for cmd, desc in examples:
        print(f"  python -m cli {cmd:<30} # {desc}")

    # Training
    print("\n[TRAINING]")
    print("  LoRA      : QLoRA with gradient checkpointing")
    print("  Presets   : phi4_mini, qwen3_embedding, colab_free_tier, local_gpu, high_end_gpu")
    print("  Metrics   : Real-time loss, LR, tokens/s tracking")

    # Stats
    print("\n[SESSION]")
    stats = cli.get_stats()
    print(f"  Session ID : {stats.get('session_id')}")
    print(f"  Messages   : {stats.get('messages', 0)}")
    print(f"  Tool Calls : {stats.get('tool_calls', 0)}")

    print(f"\n{bar}")
    print("  Copyright (c) 2024-2026 Rhasan@dev")
    print("  https://github.com/rbkhan007/Photato-Phi-3-Custom-Model")
    print("  Licensed under MIT License")
    print(bar)


def main(argv=None):
    """Real entry point: dispatch to the argparse-based CLI."""
    from cli.__main__ import main as _main

    return _main(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main())

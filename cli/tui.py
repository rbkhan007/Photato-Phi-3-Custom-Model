"""
Full TUI REPL for the Agentic CLI.

Uses prompt_toolkit Application with:
- Scrolling conversation history pane (top)
- Input buffer with auto-complete (bottom)
- Status bar showing backend, model, CPU budget
- Streaming token display
"""

import time
import hashlib
from typing import Optional
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window, BufferControl, WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.formatted_text import HTML, fragment_list_to_text
from prompt_toolkit.enums import EditingMode


class TUI:
    """prompt_toolkit-based TUI for the Agentic CLI."""

    def __init__(self, cli_instance):
        from cli import AgenticCLI
        self.cli: AgenticCLI = cli_instance
        self.conversation_history: list[dict] = []
        self._streaming_active = False
        self._current_input = ""

        # Build the completer with slash commands and tool commands
        self._completer = WordCompleter(
            [
                "/help", "/status", "/system", "/clear", "/new",
                "/model", "/backend", "/cpu", "/json", "/exit",
                "/quit",
                "list", "read", "write", "search", "run-code",
                "exec", "git-status", "git-commit", "git-diff",
                "git-log", "git-branch", "git-checkout", "git-pull",
                "git-push", "analyze", "stats", "config", "sessions",
                "mkdir", "rmdir", "copy", "move", "delete",
                "exists", "disk", "env", "set-env", "cwd",
                "cd", "os", "processes",
            ],
            ignore_case=True,
        )

        # Buffers
        self._history_buffer = Buffer(name="history_buffer", read_only=True)
        self._input_buffer = Buffer(name="input_buffer", completer=self._completer)

        # Key bindings
        self._bindings = KeyBindings()
        self._bindings.add("c-c")(self._handle_exit)
        self._bindings.add("c-l")(self._handle_clear)
        self._bindings.add("c-d")(self._handle_exit)
        self._bindings.add("enter")(self._handle_submit)

        # Layout
        self._layout = self._build_layout()

        # Style
        self._style = Style.from_dict({
            "status-bar": "reverse",
            "status-bar.backend": "bg:#0066cc #ffffff bold",
            "status-bar.model": "bg:#004d99 #ffffff",
            "status-bar.cpu": "bg:#003366 #ffffff",
            "status-bar.messages": "bg:#001a33 #ffffff",
            "conversation": "bg:#1a1a1a #e0e0e0",
            "input.label": "bg:#333333 #00cc00 bold",
            "user": "#00cc66 bold",
            "assistant": "#0099ff bold",
            "system": "#999999 italic",
        })

        # Build Application
        self._app = Application(
            layout=self._layout,
            key_bindings=self._bindings,
            style=self._style,
            full_screen=True,
            editing_mode=EditingMode.VI,  # Vi mode for easy navigation
        )

    def _build_layout(self) -> Layout:
        """Build the HSplit layout with conversation pane, input, and status bar."""
        # Conversation history pane
        history_window = Window(
            BufferControl(buffer=self._history_buffer),
            wrap_lines=True,
            style="conversation",
        )

        # Input area with label
        input_window = Window(
            BufferControl(buffer=self._input_buffer),
            height=3,
            wrap_lines=False,
        )

        # Status bar
        status_bar = FormattedTextControl(text=self._get_status_text)

        layout = HSplit([
            # Conversation history (takes remaining space)
            history_window,
            # Separator line
            Window(height=1, content=FormattedTextControl(text="-" * 80)),
            # Input buffer
            input_window,
            # Status bar (fixed height 1)
            Window(
                height=1,
                content=status_bar,
                style="status-bar",
            ),
        ])

        return Layout(layout, focused_element=input_window)

    def _get_status_text(self):
        """Format the status bar text."""
        backend = self.cli.backend
        model = getattr(backend, "model_path", None) or getattr(backend, "model", self.cli.config.get("model"))
        cpu = self.cli.config.get("cpu_percent", 55.0)

        try:
            from optimization.cpu_throttle import recommended_threads
            threads = recommended_threads(cpu)
            budget = f"{cpu:.0f}% CPU / {threads} threads"
        except Exception:
            budget = f"{cpu:.0f}% CPU"

        msg_count = len(self.cli.session.messages)
        tc_count = len(self.cli.session.tool_calls)

        fragments = [
            ("class:status-bar.backend", f" {backend.name} "),
            ("class:status-bar.model", f" {model} "),
            ("class:status-bar.cpu", f" {budget} "),
            ("class:status-bar.messages", f" msgs:{msg_count} tools:{tc_count} "),
        ]

        # Right-align session info
        session_info = f"session:{self.cli.session.id[:8]}"
        padding = 80 - sum(len(f[1]) for f in fragments) - len(session_info) - 1
        if padding > 0:
            fragments.append(("", " " * padding))
        fragments.append(("", session_info))

        return fragments

    def _handle_submit(self, event):
        """Handle Enter key press."""
        text = self._input_buffer.text.strip()
        if not text:
            return

        self._input_buffer.text = ""
        self._process_input(text)

    def _handle_exit(self, event):
        """Handle Ctrl+C or Ctrl+D."""
        self.cli.save_session()
        event.app.exit()

    def _handle_clear(self, event):
        """Handle Ctrl+L to clear screen."""
        self._app.renderer.clear()
        self._refresh_history()

    def _process_input(self, text: str):
        """Process user input - slash commands, tool commands, or chat."""
        if text in ("exit", "quit"):
            self.cli.save_session()
            self._app.exit()
            return

        if text.startswith("/"):
            self._handle_slash(text)
        else:
            # Check if input starts with any registered tool name
            first_word = text.split()[0] if text.split() else ""
            if first_word in self.cli.tools:
                self._handle_tool(text)
            else:
                self._handle_chat(text)

    def _handle_slash(self, text: str):
        """Handle slash commands."""
        parts = text[1:].split()
        cmd = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else None

        self._add_system_message(f"> {text}")

        if cmd in ("exit", "quit"):
            self.cli.save_session()
            self._app.exit()
        elif cmd == "help":
            self._add_system_message(
                "AVAILABLE COMMANDS\n"
                "=" * 50 + "\n\n"
                "CHAT:\n"
                "  Just type any message to chat with the AI\n"
                "  Example: What is Python?\n\n"
                "FILES:\n"
                "  list [path]          Show files in a folder\n"
                "  read <file>          Read a file's contents\n"
                "  write <file>         Create or edit a file\n"
                "  search <query>       Find text in files\n"
                "  analyze <file>       Analyze code quality\n\n"
                "CODE:\n"
                "  run-code <code>      Run Python/JS/other code\n"
                "  exec <command>       Run a system command\n\n"
                "SYSTEM:\n"
                "  os                   Show computer info\n"
                "  processes            Show running programs\n"
                "  cwd                  Show current folder\n"
                "  disk [path]          Show disk space\n\n"
                "SESSION:\n"
                "  /clear               Clear chat history\n"
                "  /new                 Start fresh session\n"
                "  /export              Save chat as file\n"
                "  /search <query>      Search old conversations\n\n"
                "SETTINGS:\n"
                "  /status              Show current settings\n"
                "  /model               Change AI model\n"
                "  /backend             Change backend\n"
                "  /help                Show this help\n"
                "  /exit                Quit the program\n\n"
                "=" * 50 + "\n"
                "TIP: Just type naturally - AI understands!"
            )
        elif cmd == "status":
            import json
            self._add_system_message(json.dumps(self._get_status_dict(), indent=2))
        elif cmd == "system":
            import json
            self._add_system_message(json.dumps(self.cli._system_info, indent=2, default=str))
        elif cmd == "clear":
            self.cli.clear()
            self.conversation_history.clear()
            self._refresh_history()
            self._add_system_message("session cleared")
        elif cmd == "new":
            self.cli.clear()
            self.conversation_history.clear()
            self.cli.session.id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
            self._refresh_history()
            self._add_system_message("started a new session")
        elif cmd == "model":
            if arg:
                self.cli.config["model"] = arg
                self.cli._backend = None
                self._add_system_message(f"model set to: {arg}")
            else:
                model = getattr(self.cli.backend, "model_path", None) or getattr(self.cli.backend, "model", self.cli.config.get("model"))
                self._add_system_message(f"model: {model}")
        elif cmd == "backend":
            if arg:
                self.cli.config["backend"] = arg
                self.cli._backend = None
                self._add_system_message(f"backend set to: {arg} (applied on next message)")
            else:
                self._add_system_message(f"backend: {self.cli.backend.name}")
        elif cmd == "cpu":
            if arg:
                try:
                    self.cli.config["cpu_percent"] = float(arg)
                    self.cli._backend = None
                    self._add_system_message(f"cpu cap set to: {self.cli.config['cpu_percent']:.0f}% (applied on next message)")
                except ValueError:
                    self._add_system_message("invalid number for /cpu")
            else:
                self._add_system_message(f"cpu_percent: {self.cli.config.get('cpu_percent')}")
        elif cmd == "search":
            if arg:
                results = self.cli.search_sessions(arg)
                res_list = results.get("results", [])
                if res_list:
                    msg = f"Found {len(res_list)} sessions with matches for '{arg}':\n"
                    for r in res_list[:3]:
                        msg += f"  Session {r['session_id']}: {r['total_matches']} matches\n"
                        for m in r.get("matches", [])[:2]:
                            msg += f"    - {m.get('role')}: {m.get('content', '')[:80]}...\n"
                else:
                    msg = f"No matches found for '{arg}'"
                self._add_system_message(msg)
            else:
                self._add_system_message("Usage: /search <query>")
        elif cmd == "export":
            filepath = arg or f"session_{self.cli.session.id}.md"
            result = self.cli.export_session_markdown(filepath)
            if result.get("success"):
                self._add_system_message(f"Session exported to: {result['filepath']} ({result['messages']} messages)")
            else:
                self._add_system_message(f"Export failed: {result.get('error')}")
        elif cmd == "plugins":
            result = self.cli.load_plugins()
            self._add_system_message(f"Plugins loaded: {result.get('loaded', 0)}")
            if result.get("plugins"):
                self._add_system_message(f"Active: {', '.join(result['plugins'])}")
        else:
            self._add_system_message(f"unknown command: /{cmd}  (try /help)")

        self._refresh_history()

    def _handle_tool(self, text: str):
        """Handle tool commands."""
        import shlex
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        self._add_user_message(text)

        # Build parser from __main__ to reuse tool dispatch
        from cli.__main__ import _build_parser, _dispatch
        import argparse

        parser = _build_parser()
        try:
            args = parser.parse_args(tokens)
        except SystemExit:
            self._add_system_message("invalid command syntax")
            self._refresh_history()
            return

        # Redirect output to our TUI
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            _dispatch(self.cli, args)
        output = f.getvalue()

        if output:
            self._add_system_message(output)
        self._refresh_history()

    def _handle_chat(self, text: str):
        """Handle chat messages with streaming."""
        self._add_user_message(text)
        self._refresh_history()

        # Stream response
        self._add_assistant_prefix()
        chunks = []
        try:
            for chunk in self.cli.chat_stream(text):
                chunks.append(chunk)
                self._streaming_active = True
                self._update_streaming_response("".join(chunks))
        except Exception as e:
            self._update_streaming_response(f"[error: {e}]")

        # Finalize
        self._finalize_assistant_response()

        # Show metrics
        u = getattr(self.cli, "last_usage", None)
        if u:
            bits = []
            if u.get("completion_tokens") is not None:
                bits.append(f"{u['completion_tokens']} tokens")
            if u.get("tokens_per_second") is not None:
                bits.append(f"{u['tokens_per_second']} tok/s")
            if u.get("elapsed_s") is not None:
                bits.append(f"{u['elapsed_s']}s")
            if bits:
                self._add_system_message("  -> " + " - ".join(bits))
                self._refresh_history()

    def _get_status_dict(self):
        """Get status as dict for JSON display."""
        import json
        cpu = self.cli.config.get("cpu_percent", 55.0)
        try:
            from optimization.cpu_throttle import recommended_threads
            threads = recommended_threads(cpu)
        except Exception:
            threads = None
        backend = self.cli.backend
        return {
            "backend": backend.name,
            "model": getattr(backend, "model_path", None) or getattr(backend, "model", self.cli.config.get("model")),
            "cpu_percent": cpu,
            "thread_budget": threads,
            "working_dir": self.cli.working_dir,
            "session_id": self.cli.session.id,
            "messages": len(self.cli.session.messages),
            "tool_calls": len(self.cli.session.tool_calls),
            "system": self.cli._system_info,
        }

    def _add_user_message(self, text: str):
        """Add a user message to conversation history."""
        self.conversation_history.append({"role": "user", "content": text})

    def _add_assistant_prefix(self):
        """Start an assistant response entry."""
        self.conversation_history.append({"role": "assistant", "content": ""})

    def _update_streaming_response(self, text: str):
        """Update the current assistant response during streaming."""
        if self.conversation_history and self.conversation_history[-1]["role"] == "assistant":
            self.conversation_history[-1]["content"] = text
            self._refresh_history()
            # Force UI refresh
            if self._app:
                self._app.invalidate()

    def _finalize_assistant_response(self):
        """Mark streaming as complete."""
        self._streaming_active = False

    def _add_system_message(self, text: str):
        """Add a system message to conversation history."""
        self.conversation_history.append({"role": "system", "content": text})

    def _refresh_history(self):
        """Rebuild the conversation history display."""
        lines = []
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                lines.append(f"[you]> {content}")
            elif role == "assistant":
                if content:
                    lines.append(f"[assistant]> {content}")
                else:
                    lines.append("[assistant]> ")
            elif role == "system":
                lines.append(content)

        # Join and set to history buffer
        text = "\n".join(lines)
        self._history_buffer.text = text

        # Move cursor to end of history
        if text:
            self._history_buffer.cursor_position = len(text)

    def run(self):
        """Run the TUI application."""
        # Print banner before TUI takes over full screen
        bar = "=" * 60
        backend = self.cli.backend
        model = getattr(backend, "model_path", None) or getattr(backend, "model", self.cli.config.get("model"))
        cpu = self.cli.config.get("cpu_percent", 55.0)
        try:
            from optimization.cpu_throttle import recommended_threads
            budget = f"{cpu:.0f}% CPU / ~{recommended_threads(cpu)} threads"
        except Exception:
            budget = f"{cpu:.0f}% CPU"
        print(bar)
        print("  Phi-3 Custom Model - Agentic CLI  (TUI mode)")
        print(bar)
        print(f"  backend : {backend.name}")
        print(f"  model   : {model}")
        print(f"  budget  : {budget}")
        print(f"  cwd     : {self.cli.working_dir}")
        print("-" * 60)
        print("  Copyright (c) 2024-2026 Rhasan@dev (https://github.com/rbkhan007)")
        print("  Licensed under MIT License. See LICENSE file for details.")
        print("-" * 60)
        print()
        self._app.run()


def run_tui(cli_instance):
    """Convenience function to run the TUI.

    Falls back to basic REPL if prompt_toolkit cannot initialize a proper
    console (e.g. non-interactive terminal).
    """
    try:
        tui = TUI(cli_instance)
        tui.run()
    except Exception as e:
        # If TUI fails (e.g. NoConsoleScreenBufferError), raise so caller
        # can fall back to basic REPL.
        raise RuntimeError(f"TUI unavailable: {e}") from e

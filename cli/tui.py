"""OpenCode-style TUI built with Textual.

Matches the OpenCode interface:
  - Deep dark theme with cyan/amber accents
  - ASCII logo on welcome screen
  - Collapsible thought blocks, tool cards, output panels
  - Fixed bottom input with status bar
  - Full integration with AgenticCLI (RAG, tools, streaming)
"""

import time
import os
import re
import json
import asyncio
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Grid
from textual.widgets import Header, Footer, Static, TextArea, RichLog, Input, Label, Button
from textual.widgets import Collapsible, TabbedContent, TabPane
from textual.screen import Screen
from textual.binding import Binding
from textual.reactive import var
from textual import events
from rich.text import Text
from rich.style import Style
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console, RenderableType, Group
from rich.layout import Layout
from rich.columns import Columns
from rich.markdown import Markdown

# ── Constants ────────────────────────────────────────────────────────────
BACKGROUND = "#0d0d0d"
CARD_BG = "#1a1a1a"
CARD_BG2 = "#222222"
ACCENT = "#00d7ff"
ACCENT2 = "#38bdf8"
AMBER = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
GRAY = "#6b7280"
GRAY2 = "#9ca3af"
TEXT = "#e0e0e0"
TEXT_DIM = "#888888"


# ── Helpers ──────────────────────────────────────────────────────────────
WELCOME_ART = r"""
                       ██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗
                      ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
                      ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║   ██║██║  ██║█████╗
                      ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║   ██║██║  ██║██╔══╝
                      ╚██████╔╝██║     ███████╗██║ ╚████║╚██████╗╚██████╔╝██████╔╝███████╗
                       ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""


def _styled(s: str, style: str) -> Text:
    return Text(s, style=style)


def _tag(text: str, color: str = AMBER, tag: str = "") -> Text:
    if tag:
        return Text(f" {tag} ", style=Style(bold=True, color=color, bgcolor=CARD_BG2)) + Text(f" {text}\n", style=color)
    return Text(f" {text}\n", style=Style(bold=True, color=color))


def _thought_header(duration_ms: float = 0) -> Text:
    t = Text()
    t.append(" +", style=Style(color=AMBER, bold=True))
    t.append(f" Thought", style=Style(color=AMBER, bold=True))
    if duration_ms:
        t.append(f" ({duration_ms:.0f}ms)", style=Style(color=GRAY))
    t.append("\n")
    return t


def _tool_header(cmd: str, args: str = "") -> Text:
    t = Text()
    if cmd in ("grep", "search", "find"):
        t.append(f" *  Grep ", style=Style(color=ACCENT, bold=True))
    elif cmd in ("pytest", "test", "run"):
        t.append(f" $  ", style=Style(color=GREEN, bold=True))
    else:
        t.append(f" $  ", style=Style(color=ACCENT, bold=True))
    t.append(f"{cmd} {args}", style=Style(color=TEXT))
    t.append("\n")
    return t


def _output_panel(content: str, title: str = "", exit_code: int = 0) -> Panel:
    color = RED if exit_code != 0 else GREEN
    lines = content.split("\n")
    n = min(len(lines), 5)
    summary = "\n".join(lines[:n]) + ("..." if len(lines) > n else "")
    return Panel(
        Text(summary, style=TEXT),
        title=title or f"exit code {exit_code}",
        title_align="right",
        border_style=Style(color=color),
        style=Style(bgcolor=CARD_BG),
        padding=(0, 1),
        width=None,
    )


def _code_block(code: str, lang: str = "") -> RenderableType:
    try:
        return Syntax(code, lang or "python", theme="monokai", line_numbers=False, word_wrap=True)
    except Exception:
        return Text(code, style=TEXT)


def _status_badge(label: str, color: str = GRAY) -> Text:
    return Text(f" {label} ", style=Style(bold=True, color=color, bgcolor=CARD_BG2))


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s}s"


# ── Custom Widgets ───────────────────────────────────────────────────────

class MessageWidget(Static):
    """A single conversation message."""

    def __init__(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        super().__init__()
        self._role = role
        self._content = content
        self._metadata = metadata or {}

    def on_mount(self) -> None:
        self.update(self._render())

    def _render(self) -> RenderableType:
        if self._role == "user":
            return Panel(
                Text(self._content, style=Style(color=ACCENT2)),
                title=" You ",
                title_align="left",
                border_style=Style(color=ACCENT, dim=True),
                style=Style(bgcolor=CARD_BG),
                padding=(0, 1),
            )
        elif self._role == "system":
            dur = self._metadata.get("duration_ms", 0)
            g = Group(_thought_header(dur), Text(self._content, style=Style(color=TEXT_DIM, italic=True)))
            return Panel(g, border_style=Style(color=AMBER, dim=True), style=Style(bgcolor=CARD_BG), padding=(0, 1))
        elif self._role == "tool":
            cmd = self._metadata.get("cmd", "")
            args = self._metadata.get("args", "")
            exit_code = self._metadata.get("exit_code", 0)
            g = Group(_tool_header(cmd, args), _output_panel(self._content, exit_code=exit_code))
            return Panel(g, border_style=Style(color=GRAY, dim=True), style=Style(bgcolor=CARD_BG), padding=(0, 1))
        else:
            try:
                md = Markdown(self._content)
            except Exception:
                md = Text(self._content, style=TEXT)
            return Panel(
                md,
                border_style=Style(color=ACCENT2, dim=True),
                style=Style(bgcolor=CARD_BG),
                padding=(0, 1),
            )


class WelcomeScreen(Static):
    """Centered welcome screen with ASCII logo and hint text."""

    def on_mount(self) -> None:
        self._render_welcome()

    def _render_welcome(self) -> None:
        logo = Text(WELCOME_ART, style=Style(color=ACCENT, bold=True))
        subtitle = Text("  AI-powered terminal assistant\n\n", style=Style(color=GRAY2, italic=True))
        hints = Text(
            "  ctrl+p   commands   ·   tab   agents   ·   @file   references\n"
            "  ctrl+c   cancel     ·   /help  commands   ·   ctrl+d   exit\n",
            style=Style(color=TEXT_DIM),
        )
        version = Text("  Build · DeepSeek V4 Flash Free", style=Style(color=GRAY))
        self.update(Panel(Group(logo, subtitle, hints, version), border_style=Style(color=ACCENT, dim=True), style=Style(bgcolor=BACKGROUND), padding=(2, 4)))


# ── Main Screen ──────────────────────────────────────────────────────────

class MainScreen(Screen):
    """Primary conversation screen."""

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+d", "exit", "Exit"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("tab", "focus_input", "Focus Input"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            yield ScrollableContainer(id="conversation", classes="conversation-panel")
            yield WelcomeScreen(id="welcome")
            with Container(id="bottom-panel"):
                with Container(id="input-container"):
                    yield TextArea(id="input", classes="input-box")
                yield Static(id="input-subtext", classes="input-subtext")
        yield Footer(id="status-bar", classes="status-bar")

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).focus()
        self._update_subtext()
        self._update_status()

    def _update_subtext(self) -> None:
        sub = self.query_one("#input-subtext", Static)
        sub.update(Text("  Build · DeepSeek V4 Flash Free  ", style=Style(color=GRAY)))

    def _update_status(self) -> None:
        footer = self.query_one("#status-bar", Footer)
        cli = self.app.cli
        wd = str(cli.working_dir)[:50] if hasattr(cli, "working_dir") else "."
        tr = getattr(cli, "_tool_registry", None)
        n_mcp = len(tr) if tr is not None else 0
        n_msgs = len(cli.session.messages) if hasattr(cli, "session") else 0
        tokens_est = n_msgs * 256
        pct = min(100, int(tokens_est / 200_000 * 100))
        footer._node = None
        t = Text()
        t.append(f"  {wd}  ", style=Style(color=ACCENT))
        t.append(f"o {n_mcp} MCP ", style=Style(color=GREEN))
        t.append("  |  ", style=Style(color=GRAY))
        t.append(f"{tokens_est}K ({pct}%)  ", style=Style(color=AMBER if pct > 60 else GRAY))
        t.append("v0.1.0", style=Style(color=GRAY))
        footer._node = t
        footer.refresh()

    def action_clear(self) -> None:
        conv = self.query_one("#conversation", ScrollableContainer)
        conv.remove_children()
        welcome = self.query_one("#welcome", WelcomeScreen)
        welcome.display = True
        self._update_status()

    def action_cancel(self) -> None:
        pass

    def action_exit(self) -> None:
        self.app.exit()

    def action_command_palette(self) -> None:
        self.app.notify("Command palette (WIP)", severity="information")

    def action_focus_input(self) -> None:
        self.query_one("#input", TextArea).focus()

    async def on_text_area_submitted(self, event: TextArea.Submitted) -> None:
        input_w = self.query_one("#input", TextArea)
        text = input_w.text.strip()
        if not text:
            return
        input_w.clear()
        welcome = self.query_one("#welcome", WelcomeScreen)
        if welcome.display:
            welcome.display = False
        await self._process_input(text)

    async def _process_input(self, text: str) -> None:
        await self._add_message("user", text)
        if text.startswith("/"):
            await self._handle_slash(text)
        else:
            await self._handle_chat(text)

    async def _add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        conv = self.query_one("#conversation", ScrollableContainer)
        msg = MessageWidget(role, content, metadata)
        await conv.mount(msg)
        conv.scroll_end(animate=False)
        self._update_status()

    async def _add_streaming_message(self, content: str) -> None:
        conv = self.query_one("#conversation", ScrollableContainer)
        existing = conv.query(MessageWidget)
        if existing and existing.last()._role == "assistant":
            existing.last()._content = content
        else:
            msg = MessageWidget("assistant", content)
            await conv.mount(msg)
        conv.scroll_end(animate=False)

    async def _handle_slash(self, text: str) -> None:
        parts = text[1:].split()
        cmd = parts[0].lower() if parts else ""
        arg = " ".join(parts[1:]) if len(parts) > 1 else ""

        if cmd in ("exit", "quit"):
            self.app.exit()
        elif cmd == "clear":
            self.action_clear()
        elif cmd == "help":
            await self._add_message("assistant", (
                "**Slash Commands:**\n"
                "  `/help` - Show this help\n"
                "  `/clear` - Clear conversation\n"
                "  `/status` - Show system status\n"
                "  `/time` - Show date/time\n"
                "  `/model <name>` - Switch model\n"
                "  `/backend <name>` - Switch backend\n"
                "  `/exit` - Quit\n"
                "\n**Keybindings:**\n"
                "  `ctrl+p` - Command palette\n"
                "  `ctrl+d` - Exit\n"
                "  `ctrl+l` - Clear\n"
                "  `tab` - Focus input"
            ))
        elif cmd == "status":
            cli = self.app.cli
            info = getattr(cli, "_system_info", {})
            s = (
                f"**System:** {info.get('os', '?')} {info.get('arch', '?')}\n"
                f"**Backend:** {cli.backend.name if hasattr(cli, 'backend') else '?'}\n"
                f"**Model:** {cli.config.get('model', '?')}\n"
                f"**Messages:** {len(cli.session.messages)}\n"
                f"**CWD:** {cli.working_dir}"
            )
            await self._add_message("assistant", s)
        elif cmd == "time":
            now = datetime.now()
            await self._add_message("assistant", f"**Date:** {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})\n**Time:** {now.strftime('%H:%M:%S')}")
        elif cmd == "model":
            if arg:
                self.app.cli.config["model"] = arg
                await self._add_message("system", f"Model set to: {arg}")
            else:
                await self._add_message("assistant", f"Current model: {self.app.cli.config.get('model', '?')}")
        elif cmd == "backend":
            if arg:
                self.app.cli.config["backend"] = arg
                self.app.cli._backend = None
                await self._add_message("system", f"Backend set to: {arg}")
            else:
                await self._add_message("assistant", f"Current backend: {self.app.cli.backend.name}")
        else:
            await self._add_message("system", f"Unknown command: /{cmd}")

    async def _handle_chat(self, text: str) -> None:
        cli = self.app.cli
        t0 = time.time()
        chunks = []
        thought_metadata = {}
        try:
            for chunk in cli.chat_stream(text):
                chunks.append(chunk)
                content = "".join(chunks)
                await self._add_streaming_message(content)
            full = "".join(chunks)
            elapsed = time.time() - t0
            thought_metadata = {"duration_ms": elapsed * 1000}
        except Exception as e:
            full = f"Error: {e}"
            await self._add_streaming_message(full)
        if thought_metadata:
            self.app.notify(f"Response in {_format_elapsed(time.time() - t0)}", severity="information")
        self._update_status()


class OpenCodeTUI(App):
    """OpenCode-style TUI using Textual."""

    TITLE = "opencode"
    SUB_TITLE = "AI-powered terminal assistant"
    CSS = """
    Screen {
        background: #0d0d0d;
    }

    #main-container {
        height: 100%;
        layout: vertical;
    }

    #conversation {
        height: 1fr;
        background: #0d0d0d;
        overflow-y: auto;
        padding: 0 1;
    }

    #conversation MessageWidget {
        margin: 0 0 1 0;
    }

    #welcome {
        dock: top;
        height: auto;
        margin: 2 4;
    }

    #bottom-panel {
        dock: bottom;
        height: auto;
        background: #111111;
        border-top: solid #222222;
    }

    #input-container {
        margin: 0 0 0 0;
        padding: 0 0 0 0;
        border-left: solid #00d7ff;
    }

    .input-box {
        background: #0d0d0d;
        color: #e0e0e0;
        min-height: 3;
        max-height: 6;
        border: none;
        padding: 1 2;
    }

    .input-box:focus {
        border: none;
    }

    .input-subtext {
        height: 1;
        background: #111111;
        color: #6b7280;
        padding: 0 2;
    }

    #status-bar {
        background: #0a0a0a;
        color: #9ca3af;
        height: 1;
    }

    MessageWidget {
        margin: 0 0 1 0;
    }

    Collapsible > .collapsible--header {
        background: #1a1a1a;
        color: #f59e0b;
        text-style: bold;
    }

    Collapsible > .collapsible--body {
        background: #1a1a1a;
        padding: 1;
    }

    Vertical > * {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "exit", "Exit"),
        Binding("ctrl+d", "exit", "Exit"),
    ]

    def __init__(self, cli_instance: Any = None) -> None:
        super().__init__()
        self.cli = cli_instance
        self._streaming = False

    def compose(self) -> ComposeResult:
        yield MainScreen()

    def on_mount(self) -> None:
        self.title = "opencode"
        self.sub_title = "AI-powered terminal assistant"


def run_tui(cli_instance: Any = None) -> None:
    """Launch the OpenCode TUI."""
    app = OpenCodeTUI(cli_instance)
    app.run()


def main() -> None:
    """Entry point for `python -m cli.tui`."""
    from cli import AgenticCLI
    cli = AgenticCLI()
    run_tui(cli)


if __name__ == "__main__":
    main()

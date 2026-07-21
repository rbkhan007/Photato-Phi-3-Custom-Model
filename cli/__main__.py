"""
Argparse-based entry point for the Agentic CLI.

Run with: python -m cli [command] [options]

With no command it launches an interactive REPL.

Examples:
    python -m cli                                  # interactive REPL
    python -m cli chat "explain this repo"         # one-shot chat
    python -m cli list --pattern "*.py"
    python -m cli read cli/__init__.py
    python -m cli analyze cli/__init__.py
    python -m cli search "def main" --ext .py
    python -m cli run-code --lang python --code "print(2 + 2)"
    python -m cli exec "git --version"
    python -m cli sessions list
    python -m cli config set backend ollama
"""

import sys
import json
import time
import shlex
import hashlib
import argparse
from typing import Optional

from cli import AgenticCLI, demo


def _print(result, as_json: bool) -> None:
    """Print a tool output dict, optionally as raw JSON."""
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return
    if isinstance(result, dict):
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
            return
        if "content" in result and result.get("backend"):
            print(result["content"])
            return
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)


def _dispatch(cli: AgenticCLI, args: argparse.Namespace) -> int:
    """Run a single subcommand and return a process exit code."""
    cmd = args.command
    as_json = getattr(args, "json", False)

    if cmd == "chat":
        message = args.message if isinstance(args.message, str) else " ".join(args.message)
        result = cli.chat(message)
        _print(result, as_json)
        return 0 if result.get("success") else 1

    if cmd == "list":
        call = cli.execute_tool(
            "list_files", path=args.path, recursive=args.recursive, pattern=args.pattern
        )
    elif cmd == "read":
        call = cli.execute_tool("read_file", path=args.path, encoding=args.encoding)
    elif cmd == "write":
        content = args.content
        if content is None:
            content = sys.stdin.read()
        call = cli.execute_tool("write_file", path=args.path, content=content, encoding=args.encoding)
    elif cmd == "search":
        kwargs = {"query": args.query, "path": args.path}
        if args.ext:
            kwargs["extensions"] = args.ext
        call = cli.execute_tool("search_files", **kwargs)
    elif cmd == "run-code":
        code = args.code
        if code is None and args.file:
            with open(args.file, "r", encoding="utf-8") as fh:
                code = fh.read()
        if code is None:
            code = sys.stdin.read()
        call = cli.execute_tool("run_code", language=args.lang, code=code)
    elif cmd == "exec":
        call = cli.execute_tool("run_command", command=" ".join(args.args))
    elif cmd == "git-status":
        call = cli.execute_tool("git_status")
    elif cmd == "git-commit":
        call = cli.execute_tool("git_commit", message=args.message)
    elif cmd == "git-diff":
        call = cli.execute_tool("git_diff", file=getattr(args, 'file', None))
    elif cmd == "git-log":
        call = cli.execute_tool("git_log", count=getattr(args, 'count', 10))
    elif cmd == "git-branch":
        call = cli.execute_tool("git_branch")
    elif cmd == "git-checkout":
        call = cli.execute_tool("git_checkout", branch=args.branch)
    elif cmd == "git-pull":
        call = cli.execute_tool("git_pull")
    elif cmd == "git-push":
        call = cli.execute_tool("git_push")
    elif cmd == "env":
        call = cli.execute_tool("get_env", var_name=getattr(args, 'name', None))
    elif cmd == "set-env":
        call = cli.execute_tool("set_env", var_name=args.name, value=args.value)
    elif cmd == "cwd":
        call = cli.execute_tool("get_cwd")
    elif cmd == "cd":
        call = cli.execute_tool("set_cwd", path=args.path)
    elif cmd == "os":
        call = cli.execute_tool("get_os_info")
    elif cmd == "processes":
        call = cli.execute_tool("get_process_list")
    elif cmd == "analyze":
        call = cli.execute_tool("analyze_code", path=args.path)
    elif cmd == "mkdir":
        call = cli.execute_tool("mkdir", path=args.path)
    elif cmd == "rmdir":
        call = cli.execute_tool("rmdir", path=args.path, recursive=args.recursive)
    elif cmd == "copy":
        call = cli.execute_tool("copy_file", source=args.source, destination=args.destination)
    elif cmd == "move":
        call = cli.execute_tool("move_file", source=args.source, destination=args.destination)
    elif cmd == "delete":
        call = cli.execute_tool("delete_file", path=args.path)
    elif cmd == "exists":
        call = cli.execute_tool("file_exists", path=args.path)
    elif cmd == "disk":
        call = cli.execute_tool("get_disk_usage", path=args.path)
    elif cmd == "stats":
        _print(cli.get_stats(), as_json)
        return 0
    elif cmd == "config":
        return _config_cmd(cli, args)
    elif cmd == "sessions":
        return _sessions_cmd(cli, args)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        return 2

    _print(call.output, as_json)
    errored = isinstance(call.output, dict) and bool(call.output.get("error"))
    return 0 if (call.success and not errored) else 1


def _config_cmd(cli: AgenticCLI, args: argparse.Namespace) -> int:
    as_json = getattr(args, "json", False)
    if args.action == "get":
        _print(cli.config, as_json)
        return 0
    if args.action == "set":
        value = args.value
        for caster in (int, float):
            try:
                value = caster(args.value)
                break
            except (TypeError, ValueError):
                continue
        cli.config[args.key] = value
        res = cli.save_config()
        _print(res, as_json)
        return 0 if res.get("success") else 1
    if args.action == "path":
        print(str(cli._paths()["config"]))
        return 0
    return 2


def _sessions_cmd(cli: AgenticCLI, args: argparse.Namespace) -> int:
    as_json = getattr(args, "json", False)
    if args.action == "list":
        _print(cli.list_sessions(), as_json)
        return 0
    if args.action == "save":
        res = cli.save_session()
        _print(res, as_json)
        return 0 if res.get("success") else 1
    if args.action == "load":
        res = cli.load_session(args.session_id)
        _print(res, as_json)
        return 0 if res.get("success") else 1
    return 2


def _repl(cli: AgenticCLI, as_json: bool) -> int:
    """Interactive agentic bot loop with prompt_toolkit TUI.

    - Plain text is sent to the model (streamed when supported).
    - Tool commands (``list``, ``read``, ...) are parsed and dispatched.
    - Slash commands (``/help``, ``/status``, ...) control the session.
    """
    try:
        from cli.tui import run_tui
        run_tui(cli)
        return 0
    except Exception:
        # Fallback to basic REPL if TUI fails (e.g. no Windows console)
        _banner(cli, tui_mode=False)
        json_mode = as_json
        parser = _build_parser()
        commands = _command_names(parser)
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                cli.save_session()
                return 0
            if not line:
                continue
            if line in ("exit", "quit"):
                cli.save_session()
                return 0
            if line.startswith("/"):
                ret = _slash(cli, line, commands, json_mode)
                if ret == "exit":
                    return 0
                if isinstance(ret, bool):
                    json_mode = ret
                continue
            try:
                tokens = shlex.split(line)
            except ValueError:
                tokens = line.split()
            first_word = tokens[0] if tokens else ""
            if first_word in commands or first_word in cli.tools:
                try:
                    sub_args = parser.parse_args(tokens)
                except SystemExit:
                    continue
                if sub_args.command in (None, "repl"):
                    continue
                if json_mode:
                    sub_args.json = True
                _dispatch(cli, sub_args)
                continue
            _chat_turn(cli, line, json_mode)


def _chat_turn(cli: AgenticCLI, line: str, json_mode: bool) -> None:
    """Send one chat turn, streaming the assistant reply when possible."""
    if json_mode:
        _print(cli.chat(line), json_mode)
        return
    print("assistant> ", end="", flush=True)
    try:
        for chunk in cli.chat_stream(line):
            print(chunk, end="", flush=True)
        print()
        u = getattr(cli, "last_usage", None)
        if u:
            bits = []
            if u.get("completion_tokens") is not None:
                bits.append(f"{u['completion_tokens']} tokens")
            if u.get("tokens_per_second") is not None:
                bits.append(f"{u['tokens_per_second']} tok/s")
            if u.get("elapsed_s") is not None:
                bits.append(f"{u['elapsed_s']}s")
            if bits:
                print("  -> " + " - ".join(bits))
    except Exception as e:  # pragma: no cover - defensive
        print(f"\n[error: {e}]")


def _banner(cli: AgenticCLI, tui_mode: bool = False) -> None:
    backend = cli.backend
    model = getattr(backend, "model_path", None) or getattr(backend, "model", cli.config.get("model"))
    cpu = cli.config.get("cpu_percent", 55.0)
    try:
        from optimization.cpu_throttle import recommended_threads

        budget = f"{cpu:.0f}% CPU / ~{recommended_threads(cpu)} threads"
    except Exception:
        budget = f"{cpu:.0f}% CPU"
    bar = "=" * 60
    mode = "TUI" if tui_mode else "REPL"
    print(bar)
    print(f"  Phi-3 Custom Model - Agentic CLI  ({mode} mode)")
    print(bar)
    print(f"  backend : {backend.name}")
    print(f"  model   : {model}")
    print(f"  budget  : {budget}")
    print(f"  cwd     : {cli.working_dir}")
    print("-" * 60)
    print("  Type anything - tool commands or chat naturally.")
    print("  Tools: list read write search run-code exec mkdir rmdir")
    print("        copy move delete exists disk analyze")
    print("  Git:   git-status git-commit git-diff git-log git-branch")
    print("        git-checkout git-pull git-push")
    print("  System: env set-env cwd cd os processes")
    print("  Slash: /help /status /system /clear /new /model /backend /cpu /json /exit")
    print("-" * 60)
    print("  Copyright (c) 2024-2026 Rhasan@dev (https://github.com/rbkhan007)")
    print("  Licensed under MIT License. See LICENSE file for details.")
    print("-" * 60)


def _status(cli: AgenticCLI) -> None:
    cpu = cli.config.get("cpu_percent", 55.0)
    try:
        from optimization.cpu_throttle import recommended_threads

        threads = recommended_threads(cpu)
    except Exception:
        threads = None
    backend = cli.backend
    info = {
        "backend": backend.name,
        "model": getattr(backend, "model_path", None) or getattr(backend, "model", cli.config.get("model")),
        "cpu_percent": cpu,
        "thread_budget": threads,
        "working_dir": cli.working_dir,
        "session_id": cli.session.id,
        "messages": len(cli.session.messages),
        "tool_calls": len(cli.session.tool_calls),
        "system": cli._system_info,
    }
    print(json.dumps(info, indent=2, default=str))


def _slash(cli: AgenticCLI, line: str, commands: set, json_mode: bool):
    """Handle a slash command. Returns the new json_mode (bool) or None."""
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("exit", "quit"):
        cli.save_session()
        return "exit"
    if cmd == "help":
        print("Slash commands:")
        print("  /help            show this help")
        print("  /status          backend, model, CPU budget, session stats")
        print("  /system          raw system detection info")
        print("  /clear           clear conversation history")
        print("  /new             start a fresh session")
        print("  /model [path]    show or set the model")
        print("  /backend [name]  show or set backend (llamacpp/ollama/openai/auto)")
        print("  /cpu [percent]   show or set CPU cap (0-100)")
        print("  /json            toggle raw JSON output")
        print("  /exit, /quit     leave the CLI")
        print("Tools (type directly): list, read, write, search, run-code, exec,")
        print("  git-status, git-commit, analyze, stats, config, sessions")
        return None
    if cmd == "status":
        _status(cli)
        return None
    if cmd == "system":
        print(json.dumps(cli._system_info, indent=2, default=str))
        return None
    if cmd == "clear":
        cli.clear()
        print("session cleared")
        return None
    if cmd == "new":
        cli.clear()
        cli.session.id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        print("started a new session")
        return None
    if cmd == "json":
        json_mode = not json_mode
        print(f"json mode: {json_mode}")
        return json_mode
    if cmd == "model":
        if arg:
            cli.config["model"] = arg
            cli._backend = None
            print(f"model set to: {arg}")
        else:
            print("model:", getattr(cli.backend, "model_path", None) or getattr(cli.backend, "model", cli.config.get("model")))
        return None
    if cmd == "backend":
        if arg:
            cli.config["backend"] = arg
            cli._backend = None
            print(f"backend set to: {arg} (applied on next message)")
        else:
            print("backend:", cli.backend.name)
        return None
    if cmd == "cpu":
        if arg:
            try:
                cli.config["cpu_percent"] = float(arg)
                cli._backend = None
                print(f"cpu cap set to: {cli.config['cpu_percent']:.0f}% (applied on next message)")
            except ValueError:
                print("invalid number for /cpu")
        else:
            print("cpu_percent:", cli.config.get("cpu_percent"))
        return None
    print(f"unknown command: /{cmd}  (try /help)")
    return None


def _command_names(parser: argparse.ArgumentParser) -> set:
    for action in parser._actions:
        if action.dest == "command":
            return set(action.choices)
    return set()


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="Print raw JSON output."
    )

    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Agentic CLI - execute real commands and chat with a local model.",
        parents=[common],
    )
    parser.add_argument("--working-dir", "-C", default=None, help="Working directory for the session.")
    parser.add_argument("--backend", default=None, help="Model backend: auto, ollama, openai, echo.")
    parser.add_argument("--model", default=None, help="Model name to use for chat.")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="GPU layers to offload for llama.cpp (0=CPU, -1=auto, 35=all).",
    )
    parser.add_argument(
        "--cpu-percent",
        type=float,
        default=55.0,
        help="Cap CPU usage to this percent (Windows Job Object hard cap; 100=unlimited).",
    )

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("chat", help="Chat with the configured model.", parents=[common])
    p.add_argument("message", nargs="+")

    p = sub.add_parser("list", help="List files in a directory.", parents=[common])
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--pattern", "-p", default=None, help="Glob pattern, e.g. *.py")

    p = sub.add_parser("read", help="Read a file.", parents=[common])
    p.add_argument("path")
    p.add_argument("--encoding", default="utf-8")

    p = sub.add_parser("write", help="Write a file (content from --content or stdin).", parents=[common])
    p.add_argument("path")
    p.add_argument("--content", default=None)
    p.add_argument("--encoding", default="utf-8")

    p = sub.add_parser("search", help="Search text across files.", parents=[common])
    p.add_argument("query")
    p.add_argument("--path", default=".")
    p.add_argument("--ext", nargs="*", default=None, help="Extensions, e.g. .py .js")

    p = sub.add_parser("run-code", help="Execute a code snippet.", parents=[common])
    p.add_argument("--lang", "-l", default="python", help="python, js, ts, bash, go, rust")
    p.add_argument("--code", "-c", default=None)
    p.add_argument("--file", "-f", default=None, help="Read code from a file.")

    p = sub.add_parser("exec", help="Run a system command.", parents=[common])
    p.add_argument("args", nargs=argparse.REMAINDER)

    sub.add_parser("git-status", help="Show git status.", parents=[common])

    p = sub.add_parser("git-commit", help="Create a git commit.", parents=[common])
    p.add_argument("--message", "-m", required=True)

    p = sub.add_parser("git-diff", help="Show git diff.", parents=[common])
    p.add_argument("--file", "-f", default=None)

    p = sub.add_parser("git-log", help="Show git log.", parents=[common])
    p.add_argument("--count", "-n", type=int, default=10)

    sub.add_parser("git-branch", help="List git branches.", parents=[common])

    p = sub.add_parser("git-checkout", help="Checkout a git branch.", parents=[common])
    p.add_argument("branch")

    sub.add_parser("git-pull", help="Git pull.", parents=[common])
    sub.add_parser("git-push", help="Git push.", parents=[common])

    p = sub.add_parser("env", help="Get environment variable(s).", parents=[common])
    p.add_argument("name", nargs="?", default=None)

    p = sub.add_parser("set-env", help="Set environment variable.", parents=[common])
    p.add_argument("name")
    p.add_argument("value")

    sub.add_parser("cwd", help="Show current working directory.", parents=[common])

    p = sub.add_parser("cd", help="Change working directory.", parents=[common])
    p.add_argument("path")

    sub.add_parser("os", help="Show OS information.", parents=[common])
    sub.add_parser("processes", help="Show running processes.", parents=[common])

    p = sub.add_parser("analyze", help="Analyze a code file's metrics.", parents=[common])
    p.add_argument("path")

    p = sub.add_parser("mkdir", help="Create a directory.", parents=[common])
    p.add_argument("path")

    p = sub.add_parser("rmdir", help="Remove a directory.", parents=[common])
    p.add_argument("path")
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("copy", help="Copy a file or directory.", parents=[common])
    p.add_argument("source")
    p.add_argument("destination")

    p = sub.add_parser("move", help="Move a file or directory.", parents=[common])
    p.add_argument("source")
    p.add_argument("destination")

    p = sub.add_parser("delete", help="Delete a file.", parents=[common])
    p.add_argument("path")

    p = sub.add_parser("exists", help="Check if file/directory exists.", parents=[common])
    p.add_argument("path")

    p = sub.add_parser("disk", help="Show disk usage for a path.", parents=[common])
    p.add_argument("path", nargs="?", default=".")

    sub.add_parser("stats", help="Show session statistics.", parents=[common])

    p = sub.add_parser("config", help="Get or set configuration.", parents=[common])
    p.add_argument("action", choices=["get", "set", "path"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")

    p = sub.add_parser("sessions", help="Manage saved sessions.", parents=[common])
    p.add_argument("action", choices=["list", "save", "load"])
    p.add_argument("session_id", nargs="?")

    sub.add_parser("repl", help="Start an interactive REPL.", parents=[common])
    sub.add_parser("demo", help="Print a capability overview.", parents=[common])

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        demo()
        return 0

    override = {
        "backend": args.backend,
        "model": args.model,
        "n_gpu_layers": args.n_gpu_layers,
        "cpu_percent": args.cpu_percent,
    }
    cli = AgenticCLI(working_dir=args.working_dir, config=override)

    if args.command is None or args.command == "repl":
        return _repl(cli, getattr(args, "json", False))

    return _dispatch(cli, args)


if __name__ == "__main__":
    sys.exit(main())

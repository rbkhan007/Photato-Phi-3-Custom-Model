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
import platform
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
            print(f"error: {result['error']}")
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
    
    # ── Shell & Terminal ──
    elif cmd == "powershell":
        command = " ".join(args.command) if args.command else ""
        call = cli.execute_tool("run_powershell", command=command, timeout=args.timeout, working_dir=args.working_dir)
    elif cmd == "terminal":
        command = " ".join(args.command) if args.command else ""
        call = cli.execute_tool("run_terminal", command=command, timeout=args.timeout, shell=args.shell, working_dir=args.working_dir)
    elif cmd == "shells":
        call = cli.execute_tool("get_available_shells")
    
    # ── Windows Services ──
    elif cmd == "services":
        call = cli.execute_tool("list_services")
    elif cmd == "service":
        if args.action == "get":
            call = cli.execute_tool("get_service", name=args.name)
        elif args.action == "start":
            call = cli.execute_tool("start_service", name=args.name)
        elif args.action == "stop":
            call = cli.execute_tool("stop_service", name=args.name)
        elif args.action == "restart":
            call = cli.execute_tool("restart_service", name=args.name)
        else:
            call = cli.execute_tool("get_service", name=args.name)
    
    # ── Processes ──
    elif cmd == "kill":
        call = cli.execute_tool("kill_process", name_or_pid=args.name_or_pid)
    elif cmd == "run":
        call = cli.execute_tool("start_process", path=args.path, arguments=" ".join(args.args))
    
    # ── Registry ──
    elif cmd == "registry":
        call = cli.execute_tool("read_registry", key=args.key, value=args.value)
    
    # ── Event Log ──
    elif cmd == "events":
        call = cli.execute_tool("get_event_log", log_name=args.log, max_events=args.max, level=args.level)
    
    # ── Network ──
    elif cmd == "network":
        call = cli.execute_tool("get_network_config")
    elif cmd == "ping":
        call = cli.execute_tool("test_network", target=args.target)
    
    # ── Scheduled Tasks ──
    elif cmd == "tasks":
        call = cli.execute_tool("list_scheduled_tasks", folder=args.folder)
    
    # ── Windows System ──
    elif cmd == "winver":
        call = cli.execute_tool("get_windows_system_info")
    elif cmd == "drives":
        call = cli.execute_tool("get_disk_info")
    
    else:
        print(f"Unknown command: {cmd}")
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


def _repl(cli: AgenticCLI, as_json: bool, beginner: bool = False) -> int:
    """Interactive agentic bot loop with prompt_toolkit TUI.

    - Plain text is sent to the model (streamed when supported).
    - Tool commands (``list``, ``read``, ...) are parsed and dispatched.
    - Slash commands (``/help``, ``/status``, ...) control the session.
    - Set beginner=True for simplified commands and helpful tips.
    """
    # Beginner-friendly command aliases
    BEGINNER_ALIASES = {
        "files": "list",
        "dir": "list",
        "ls": "list",
        "view": "read",
        "cat": "read",
        "type": "read",
        "create": "write",
        "edit": "write",
        "make": "write",
        "find": "search",
        "grep": "search",
        "look": "search",
        "code": "run-code",
        "exec": "run-code",
        "run": "run-code",
        "execute": "exec",
        "cmd": "exec",
        "shell": "exec",
        "computer": "os",
        "system": "os",
        "info": "os",
        "programs": "processes",
        "apps": "processes",
        "running": "processes",
        "folder": "cwd",
        "path": "cwd",
        "location": "cwd",
        "space": "disk",
        "storage": "disk",
        "analyze": "analyze",
        "check": "analyze",
        "test": "analyze",
        "git": "git-status",
        "status": "git-status",
        "log": "git-log",
        "history": "git-log",
        "diff": "git-diff",
        "changes": "git-diff",
        # Shell & Terminal aliases
        "ps": "powershell",
        "pwsh": "powershell",
        "ps1": "powershell",
        "script": "powershell",
        "win-shell": "terminal",
        "command": "terminal",
        "svc": "services",
        "stop": "kill",
        "end": "kill",
        "terminate": "kill",
        "launch": "run",
        "execute": "exec",
        "cmd": "exec",
        "shell": "exec",
        "network": "network",
        "ip": "network",
        "wifi": "network",
        "net": "network",
        "reg": "registry",
        "date": "time",
    }

    try:
        from cli.tui import is_interactive_terminal, run_tui

        if is_interactive_terminal():
            rc = run_tui(cli)
            if rc == 0:
                return 0
    except ImportError:
        pass
    # Fallback to basic REPL if TUI unavailable or non-interactive
    _banner(cli, tui_mode=False, beginner=beginner)
    json_mode = as_json
    parser = _build_parser()
    commands = _command_names(parser)
    while True:
        try:
            line = input("\033[92myou> \033[0m").strip()
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
        if beginner and first_word in BEGINNER_ALIASES:
            tokens[0] = BEGINNER_ALIASES[first_word]
            first_word = tokens[0]
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
    
    # Beautiful user prompt
    print(f"\n\033[92m[you]\033[0m {line}")
    print()
    print("\033[94m[assistant]\033[0m ", end="", flush=True)
    
    try:
        for chunk in cli.chat_stream(line):
            print(chunk, end="", flush=True)
        print()
        print()
        
        # Show metrics subtly
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
                print(f"  \033[90m-> {' - '.join(bits)}\033[0m")
    except Exception as e:  # pragma: no cover - defensive
        print(f"\n\033[91m[error: {e}]\033[0m")


def _banner(cli: AgenticCLI, tui_mode: bool = False, beginner: bool = False) -> None:
    backend = cli.backend
    model = getattr(backend, "model_path", None) or getattr(backend, "model", cli.config.get("model"))
    cpu = cli.config.get("cpu_percent", 55.0)
    try:
        from optimization.cpu_throttle import recommended_threads

        budget = f"{cpu:.0f}% CPU / ~{recommended_threads(cpu)} threads"
    except Exception:
        budget = f"{cpu:.0f}% CPU"

    # Check capabilities
    caps = []
    if cli._rag_engine:
        caps.append("RAG")
    if cli._memory:
        caps.append("Memory")
    if cli._safety:
        caps.append("Safety")
    if cli._extended_thinker:
        caps.append("Thinking")
    caps_str = ", ".join(caps) if caps else "None"

    # Colors for terminal
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    mode = "TUI" if tui_mode else "REPL"
    print()
    print(f"  {BLUE}{BOLD}PHI-3 CUSTOM MODEL{RESET} {DIM}({mode} mode){RESET}")
    print(f"  {DIM}{'-' * 50}{RESET}")
    print(f"  Model: {GREEN}{model.split('/')[-1] if '/' in model else model}{RESET}")
    print(f"  Backend: {CYAN}{backend.name}{RESET}")
    print(f"  Capabilities: {YELLOW}{caps_str}{RESET}")
    print()
    print(f"  {BOLD}Welcome! Here's what you can do:{RESET}")
    print()
    print(f"  {GREEN}1.{RESET} {BOLD}Chat with AI{RESET} - just type a question!")
    print(f"     Example: {DIM}What is machine learning?{RESET}")
    print()
    print(f"  {GREEN}2.{RESET} {BOLD}Use commands{RESET} - type any of these:")
    print(f"     {CYAN}list{RESET} - see files  |  {CYAN}read <file>{RESET} - view file")
    print(f"     {CYAN}write <file>{RESET} - create/edit  |  {CYAN}run-code{RESET} - execute code")
    print(f"     {CYAN}os{RESET} - computer info  |  {CYAN}disk{RESET} - disk space")
    print()
    print(f"  {GREEN}3.{RESET} Type {YELLOW}/help{RESET} to see all commands")
    print(f"  {GREEN}4.{RESET} Type {YELLOW}/exit{RESET} to quit")
    print()
    print(f"  {DIM}{'-' * 50}{RESET}")
    print(f"  {BOLD}TIP:{RESET} Just type naturally - AI understands!")
    if beginner:
        print()
        print(f"  {YELLOW}BEGINNER MODE ACTIVE{RESET}")
        print(f"  {DIM}You can use simple commands:{RESET}")
        print(f"    {CYAN}files{RESET} = list files  |  {CYAN}view{RESET} = read file")
        print(f"    {CYAN}create{RESET} = write file  |  {CYAN}code{RESET} = run code")
        print(f"    {CYAN}computer{RESET} = system info  |  {CYAN}space{RESET} = disk space")
        print(f"    {CYAN}find{RESET} = search  |  {CYAN}check{RESET} = analyze")
    print()


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
        print()
        print("  AVAILABLE COMMANDS")
        print("  " + "=" * 50)
        print()
        print("  CHAT:")
        print("    Just type any message to chat with the AI")
        print("    Example: What is Python?")
        print()
        print("  FILES:")
        print("    list [path]          Show files in a folder")
        print("    read <file>          Read a file's contents")
        print("    write <file>         Create or edit a file")
        print("    search <query>       Find text in files")
        print("    analyze <file>       Analyze code quality")
        print()
        print("  CODE:")
        print("    run-code <code>      Run Python/JS/other code")
        print("    exec <command>       Run a system command")
        print()
        print("  SYSTEM:")
        print("    os                   Show computer info")
        print("    processes            Show running programs")
        print("    cwd                  Show current folder")
        print("    disk [path]          Show disk space")
        print("    env [name]           Show environment info")
        print()
        print("  GIT (for developers):")
        print("    git-status           Show git status")
        print("    git-log              Show git history")
        print("    git-diff             Show changes")
        print()
        print("  BEGINNER-FRIENDLY ALIASES:")
        print("    files, dir, ls       = list")
        print("    view, cat, type      = read")
        print("    create, edit, make   = write")
        print("    find, grep, look     = search")
        print("    code, run            = run-code")
        print("    execute, cmd, shell  = exec")
        print("    computer, system     = os")
        print("    programs, apps       = processes")
        print("    folder, location     = cwd")
        print("    space, storage       = disk")
        print("    check, test          = analyze")
        print("    date, now            = /time")
        print()
        print("  SESSION:")
        print("    /clear               Clear chat history")
        print("    /new                 Start fresh session")
        print("    /export              Save chat as file")
        print("    /search <query>      Search old conversations")
        print("    /time                Show current date and time")
        print()
        print("  SETTINGS:")
        print("    /status              Show current settings")
        print("    /model               Change AI model")
        print("    /backend             Change backend")
        print("    /help                Show this help")
        print("    /exit                Quit the program")
        print()
        print("  " + "=" * 50)
        print("  TIP: Just type naturally - AI understands!")
        print()
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
    if cmd == "time":
        from datetime import datetime
        import platform
        now = datetime.now()
        print()
        print("  CURRENT DATE & TIME")
        print("  " + "=" * 40)
        print(f"  Date:     {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})")
        print(f"  Time:     {now.strftime('%H:%M:%S')}")
        print(f"  Platform: {platform.system()} {platform.version()}")
        print("  " + "=" * 40)
        return None
    print(f"unknown command: /{cmd}  (try /help)")
    return None


def _command_names(parser: argparse.ArgumentParser) -> set:
    for action in parser._actions:
        if action.dest == "command":
            return set(action.choices)
    return set()


def _build_parser() -> argparse.ArgumentParser:
    __version__ = "0.2.0"

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="Print raw JSON output."
    )

    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Agentic CLI - execute real commands and chat with a local model.",
        parents=[common],
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose/debug output.")
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
    parser.add_argument(
        "--beginner",
        action="store_true",
        default=False,
        help="Enable beginner mode with simpler commands and tips.",
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
    p.add_argument("action", choices=["list", "save", "load", "search"])
    p.add_argument("session_id", nargs="?")

    sub.add_parser("health", help="Run health check on all components.", parents=[common])
    sub.add_parser("repl", help="Start an interactive REPL.", parents=[common])
    sub.add_parser("tui", help="Start the IndieCode full-screen TUI.", parents=[common])
    sub.add_parser("demo", help="Print a capability overview.", parents=[common])

    # ── Shell & Terminal Commands ──
    p = sub.add_parser("powershell", help="Execute a PowerShell command or script.", parents=[common])
    p.add_argument("command", nargs="*", help="PowerShell command to execute")
    p.add_argument("--timeout", type=int, default=60, help="Max execution time in seconds")
    p.add_argument("--working-dir", help="Working directory")

    p = sub.add_parser("terminal", help="Execute a terminal command.", parents=[common])
    p.add_argument("command", nargs="*", help="Command to execute")
    p.add_argument("--timeout", type=int, default=120, help="Max execution time in seconds")
    p.add_argument("--shell", choices=["powershell", "cmd", "bash", "auto"], default="auto", help="Shell to use")
    p.add_argument("--working-dir", help="Working directory")

    sub.add_parser("shells", help="Detect available shells on the system.", parents=[common])

    # ── Windows Service Commands ──
    sub.add_parser("services", help="List Windows services.", parents=[common])
    p = sub.add_parser("service", help="Manage a Windows service.", parents=[common])
    p.add_argument("action", choices=["get", "start", "stop", "restart"])
    p.add_argument("name", help="Service name")

    # ── Process Commands ──
    p = sub.add_parser("kill", help="Kill a process by name or PID.", parents=[common])
    p.add_argument("name_or_pid", help="Process name (e.g., notepad.exe) or PID")

    p = sub.add_parser("run", help="Start a new process.", parents=[common])
    p.add_argument("path", help="Path to executable")
    p.add_argument("args", nargs="*", help="Command-line arguments")

    # ── Windows Registry ──
    p = sub.add_parser("registry", help="Read a Windows Registry key.", parents=[common])
    p.add_argument("key", help="Registry path (e.g., HKLM:\\Software\\Microsoft)")
    p.add_argument("--value", help="Specific value name")

    # ── Event Log ──
    p = sub.add_parser("events", help="Read Windows Event Log.", parents=[common])
    p.add_argument("--log", default="System", help="Log name (System, Application, Security)")
    p.add_argument("--max", type=int, default=20, help="Maximum entries")
    p.add_argument("--level", choices=["Error", "Warning", "Information"], help="Filter by level")

    # ── Network ──
    sub.add_parser("network", help="Show network configuration.", parents=[common])
    p = sub.add_parser("ping", help="Test network connectivity.", parents=[common])
    p.add_argument("target", nargs="?", default="8.8.8.8", help="Hostname or IP to ping")

    # ── Scheduled Tasks ──
    p = sub.add_parser("tasks", help="List Windows Scheduled Tasks.", parents=[common])
    p.add_argument("--folder", default="\\", help="Task folder path")

    # ── Windows System ──
    sub.add_parser("winver", help="Show Windows version information.", parents=[common])
    sub.add_parser("drives", help="Show disk/drive information.", parents=[common])

    return parser


def _health_check(cli: AgenticCLI) -> None:
    """Run a comprehensive health check."""
    bar = "=" * 60
    print(bar)
    print("  HEALTH CHECK")
    print(bar)

    checks = []

    # System
    checks.append(("System", "OK", f"{cli._system_info.get('os')} {cli._system_info.get('arch')}"))

    # Backend
    try:
        name = cli.backend.name
        checks.append(("Backend", "OK", name))
    except Exception as e:
        checks.append(("Backend", "FAIL", str(e)[:50]))

    # RAG
    if cli._rag_engine:
        checks.append(("RAG Engine", "OK", "Initialized"))
    else:
        checks.append(("RAG Engine", "SKIP", "Not available"))

    # Memory
    if cli._memory:
        checks.append(("Memory", "OK", "Initialized"))
    else:
        checks.append(("Memory", "SKIP", "Not available"))

    # Safety
    if cli._safety:
        checks.append(("Safety Layer", "OK", "Initialized"))
    else:
        checks.append(("Safety Layer", "SKIP", "Not available"))

    # Extended Thinking
    if cli._extended_thinker:
        checks.append(("Extended Thinking", "OK", "Initialized"))
    else:
        checks.append(("Extended Thinking", "SKIP", "Not available"))

    # Tool Registry
    if cli._tool_registry is not None:
        checks.append(("Tool Registry", "OK", "Initialized"))
    else:
        checks.append(("Tool Registry", "SKIP", "Not available"))

    # Tools
    checks.append(("Tools", "OK", f"{len(cli.tools)} available"))

    # Config
    config_path = cli._paths()["config"]
    if config_path.exists():
        checks.append(("Config", "OK", str(config_path)))
    else:
        checks.append(("Config", "WARN", "No config file found"))

    # Sessions dir
    sessions_path = cli._paths()["sessions"]
    if sessions_path.exists():
        session_count = len(list(sessions_path.glob("*.json")))
        checks.append(("Sessions", "OK", f"{session_count} saved"))
    else:
        checks.append(("Sessions", "WARN", "No sessions directory"))

    # Print results
    passed = sum(1 for _, status, _ in checks if status == "OK")
    failed = sum(1 for _, status, _ in checks if status == "FAIL")
    warnings = sum(1 for _, status, _ in checks if status == "WARN")

    for name, status, detail in checks:
        icon = {"OK": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
        print(f"  {icon:<8} {name:<20} {detail}")

    print("-" * 60)
    print(f"  Result: {passed} passed, {failed} failed, {warnings} warnings")
    print(bar)


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        demo()
        return 0

    if args.command == "tui":
        from cli.tui import is_interactive_terminal, run_tui

        if not is_interactive_terminal():
            print("TUI requires an interactive terminal. Run 'python -m cli' in a terminal window.", file=sys.stderr)
            return 1
        cli = AgenticCLI(working_dir=args.working_dir, config={
            "backend": args.backend, "model": args.model,
            "n_gpu_layers": args.n_gpu_layers, "cpu_percent": args.cpu_percent,
        })
        return run_tui(cli)

    override = {
        "backend": args.backend,
        "model": args.model,
        "n_gpu_layers": args.n_gpu_layers,
        "cpu_percent": args.cpu_percent,
    }
    cli = AgenticCLI(working_dir=args.working_dir, config=override)

    if args.command is None or args.command == "repl":
        if args.command == "repl" or not getattr(args, "json", False):
            return _repl(cli, getattr(args, "json", False), beginner=getattr(args, "beginner", False))
        return _repl(cli, as_json=True)

    if args.command == "health":
        _health_check(cli)
        return 0

    return _dispatch(cli, args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

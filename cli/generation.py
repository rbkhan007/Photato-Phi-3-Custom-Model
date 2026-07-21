"""
Structured output generation and formatting for the TUI.

Provides:
- Syntax-highlighted code blocks
- Table/formatted data display
- Progressive token streaming with cleaner layout
- Collapsible sections
- JSON/YAML pretty-printing
"""

import json
import re
import sys
import shutil
from datetime import datetime
from typing import Any, Optional

try:
    TERM_WIDTH = shutil.get_terminal_size(fallback=(80, 24)).columns
except Exception:
    TERM_WIDTH = 80

try:
    stdout_enc = getattr(sys.stdout, "encoding", "").lower() if hasattr(sys.stdout, "encoding") else ""
    _USE_UNICODE = stdout_enc in ("utf-8", "utf8") or not stdout_enc
except Exception:
    _USE_UNICODE = False

if _USE_UNICODE:
    _C = {
        "vline": "\u2502",
        "hline": "\u2500",
        "tl": "\u250c",
        "tr": "\u2510",
        "bl": "\u2514",
        "br": "\u2518",
        "lm": "\u251c",
        "rm": "\u2524",
        "cross": "\u253c",
        "bullet": "\u2022",
        "arrow": "\u2192",
        "check": "\u2714",
        "cross_mark": "\u2716",
        "warn": "\u26a0",
        "collapsed": "\u25b6",
        "expanded": "\u25bc",
        "hbar_full": "\u2588",
        "hbar_empty": "\u2591",
        "ellipsis": "\u2026",
        "emdash": "\u2014",
        "empty_bullet": "\u25e6",
    }
else:
    _C = {
        "vline": "|",
        "hline": "-",
        "tl": "+",
        "tr": "+",
        "bl": "+",
        "br": "+",
        "lm": "+",
        "rm": "+",
        "cross": "+",
        "bullet": "*",
        "arrow": "->",
        "check": "+",
        "cross_mark": "x",
        "warn": "!",
        "collapsed": ">",
        "expanded": "v",
        "hbar_full": "#",
        "hbar_empty": ".",
        "ellipsis": "...",
        "emdash": "---",
        "empty_bullet": "o",
    }


class SyntaxHighlighter:
    KEYWORDS = {
        "python": r"\b(def|class|return|if|elif|else|for|while|import|from|as|try|except|finally|with|yield|lambda|pass|break|continue|and|or|not|in|is|None|True|False|raise|async|await|global|nonlocal|del|print|self|__init__)\b",
        "javascript": r"\b(function|const|let|var|return|if|else|for|while|do|switch|case|break|continue|new|this|class|import|export|default|from|async|await|try|catch|throw|typeof|instanceof|null|undefined|true|false|of|in)\b",
        "typescript": r"\b(function|const|let|var|return|if|else|for|while|do|switch|case|break|continue|new|this|class|import|export|default|from|async|await|try|catch|throw|typeof|instanceof|null|undefined|true|false|interface|type|enum|implements|extends|of|in|as|readonly|private|protected|public|static|abstract)\b",
        "bash": r"\b(if|then|elif|else|fi|for|while|do|done|case|esac|function|return|exit|export|source|local|echo|printf|read|set|unset|trap|exec|eval)\b",
        "sql": r"\b(SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|ALTER|DROP|INDEX|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AND|OR|NOT|IN|LIKE|BETWEEN|GROUP|BY|ORDER|HAVING|LIMIT|OFFSET|UNION|ALL|DISTINCT|AS|NULL|TRUE|FALSE|COUNT|SUM|AVG|MIN|MAX|EXISTS|CASE|WHEN|THEN|ELSE|END|PRIMARY|KEY|FOREIGN|REFERENCES|CASCADE)\b",
        "java": r"\b(public|private|protected|static|final|class|interface|enum|extends|implements|abstract|synchronized|volatile|transient|native|strictfp|if|else|for|while|do|switch|case|break|continue|return|new|this|super|try|catch|finally|throw|throws|import|package|void|int|long|double|float|boolean|char|byte|short|String|null|true|false|instanceof)\b",
        "cpp": r"\b(int|long|double|float|char|bool|void|auto|const|volatile|signed|unsigned|short|static|extern|register|mutable|class|struct|union|enum|template|typename|namespace|using|virtual|override|final|public|private|protected|if|else|for|while|do|switch|case|break|continue|return|new|delete|try|catch|throw|this|true|false|nullptr|include|define|pragma|ifdef|ifndef|endif|and|or|not)\b",
        "rust": r"\b(fn|let|mut|const|static|if|else|for|while|loop|match|return|break|continue|struct|enum|trait|impl|pub|use|mod|crate|super|self|as|in|where|type|dyn|async|await|unsafe|ref|move|true|false|Some|None|Ok|Err)\b",
    }

    STRING_PATTERNS = [
        (r'""".*?"""', "green"),
        (r"'''.*?'''", "green"),
        (r'"(?:[^"\\]|\\.)*"', "green"),
        (r"'(?:[^'\\]|\\.)*'", "green"),
        (r'`(?:[^`\\]|\\.)*`', "yellow"),
    ]

    COMMENT_PATTERNS = {
        "python": (r"#.*$", "dark_gray"),
        "javascript": (r"//.*$|/\*[\s\S]*?\*/", "dark_gray"),
        "typescript": (r"//.*$|/\*[\s\S]*?\*/", "dark_gray"),
        "bash": (r"#.*$", "dark_gray"),
        "java": (r"//.*$|/\*[\s\S]*?\*/", "dark_gray"),
        "cpp": (r"//.*$|/\*[\s\S]*?\*/", "dark_gray"),
        "rust": (r"//.*$|/\*[\s\S]*?\*/", "dark_gray"),
        "sql": (r"--.*$", "dark_gray"),
    }

    NUMBER = r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"

    ANSI = {
        "bold": "\033[1m",
        "reset": "\033[0m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "dark_gray": "\033[90m",
        "white": "\033[97m",
        "bg_blue": "\033[44m",
        "bg_dark": "\033[40m",
    }

    @classmethod
    def highlight(cls, code: str, language: str = "") -> str:
        lang = language.lower() if language else cls._detect_language(code)
        if not lang:
            return code

        patterns = []
        keywords = cls.KEYWORDS.get(lang, None)
        if keywords:
            patterns.append((keywords, cls._kw, "yellow"))

        comments = cls.COMMENT_PATTERNS.get(lang, None)
        if comments:
            patterns.append((comments[0], cls._decorate, "dark_gray"))

        patterns.append((cls.NUMBER, cls._decorate, "magenta"))

        for sp, color in cls.STRING_PATTERNS:
            patterns.append((sp, cls._decorate, color))

        for pattern, fn, color in patterns:
            try:
                code = re.sub(pattern, lambda m: fn(m, color), code)
            except Exception:
                pass

        return code

    @classmethod
    def _kw(cls, m, color):
        return f"{cls.ANSI['bold']}{cls.ANSI[color]}{m.group()}{cls.ANSI['reset']}"

    @classmethod
    def _decorate(cls, m, color):
        return f"{cls.ANSI[color]}{m.group()}{cls.ANSI['reset']}"

    @classmethod
    def _detect_language(cls, code: str) -> str:
        score = {}
        for lang, kw in cls.KEYWORDS.items():
            matches = re.findall(kw, code, re.IGNORECASE)
            if matches:
                score[lang] = len(matches)

        shebang = code.split("\n")[0].strip()
        if "python" in shebang.lower():
            score["python"] = score.get("python", 0) + 5
        elif "bash" in shebang.lower() or "sh" in shebang.lower():
            score["bash"] = score.get("bash", 0) + 5
        elif "node" in shebang.lower() or "javascript" in shebang.lower():
            score["javascript"] = score.get("javascript", 0) + 5

        if score:
            best = max(score, key=score.get)
            if score[best] >= 2:
                return best
        return ""


class CodeBlockFormatter:
    @staticmethod
    def format(text: str) -> str:
        def replace_block(m):
            lang = m.group(1) or ""
            code = m.group(2)
            highlighted = SyntaxHighlighter.highlight(code, lang)
            lang_label = f" {lang}" if lang else ""
            sep = f"\033[90m{_C['hline'] * (TERM_WIDTH - len(lang_label) - 6)}\033[0m"
            header = f"\033[90m{_C['vline']}{lang_label}\033[0m"
            return f"\n{sep}\n{header}\n{highlighted}\n{sep}\n"

        pattern = r"```(\w*)\n(.*?)```"
        return re.sub(pattern, replace_block, text, flags=re.DOTALL)

    @staticmethod
    def extract_blocks(text: str) -> list[dict]:
        blocks = []
        parts = re.split(r"(```\w*\n.*?```)", text, flags=re.DOTALL)
        for part in parts:
            m = re.match(r"```(\w*)\n(.*?)```", part, flags=re.DOTALL)
            if m:
                blocks.append({"type": "code", "language": m.group(1) or "text", "content": m.group(2)})
            elif part.strip():
                blocks.append({"type": "text", "content": part.strip()})
        return blocks if blocks else [{"type": "text", "content": text.strip()}]


class TableFormatter:
    @staticmethod
    def format_table(
        headers: list[str],
        rows: list[list[str]],
        title: str = "",
        align: Optional[list[str]] = None,
    ) -> str:
        if not headers and not rows:
            return ""
        if not rows:
            return "  ".join(f"\033[1m{h}\033[0m" for h in headers)

        col_count = max(len(headers), max((len(r) for r in rows), default=0))
        headers = headers + [""] * (col_count - len(headers))
        rows = [r + [""] * (col_count - len(r)) for r in rows]
        if align is None:
            align = ["left"] * col_count

        col_widths = [
            max(
                max((len(str(r[i])) for r in rows), default=0),
                len(str(headers[i])),
            )
            for i in range(col_count)
        ]
        total_w = sum(col_widths) + 3 * (col_count - 1) + 4
        if total_w > TERM_WIDTH:
            scale = (TERM_WIDTH - 4 - 3 * (col_count - 1)) / sum(col_widths)
            col_widths = [max(3, int(w * scale)) for w in col_widths]

        lines = []
        if title:
            lines.append(f"\n\033[1;96m  {title}\033[0m")
            lines.append("")

        def fmt_cell(text, width, a):
            text = str(text)
            if len(text) > width:
                text = text[: width - 1] + _C["ellipsis"]
            if a == "right":
                return text.rjust(width)
            elif a == "center":
                return text.center(width)
            return text.ljust(width)

        vl = _C["vline"]
        hl = _C["hline"]
        bl = _C["bl"]
        hdr_line = (
            f"\033[90m{vl}\033[0m "
            + f" \033[90m{vl}\033[0m ".join(
                f"\033[1m{fmt_cell(h, col_widths[i], 'left')}\033[0m" for i, h in enumerate(headers)
            )
            + f" \033[90m{vl}\033[0m"
        )
        lines.append(hdr_line)

        ruler = f"\033[90m{vl}{hl * (sum(col_widths) + 3 * col_count - 1)}\033[0m"
        lines.append(ruler)

        for row in rows:
            line = (
                f"\033[90m{vl}\033[0m "
                + f" \033[90m{vl}\033[0m ".join(
                    fmt_cell(row[i], col_widths[i], align[i]) for i in range(col_count)
                )
                + f" \033[90m{vl}\033[0m"
            )
            lines.append(line)

        bottom = f"\033[90m{bl}{hl * (sum(col_widths) + 3 * col_count - 1)}\033[0m"
        lines.append(bottom)

        return "\n".join(lines)

    @staticmethod
    def from_csv(csv_text: str, title: str = "") -> str:
        lines = [l.strip() for l in csv_text.strip().split("\n") if l.strip()]
        if not lines:
            return ""
        headers = [h.strip() for h in lines[0].split(",")]
        rows = [[c.strip() for c in l.split(",")] for l in lines[1:]]
        return TableFormatter.format_table(headers, rows, title)

    @staticmethod
    def from_dict_list(data: list[dict], title: str = "") -> str:
        if not data:
            return ""
        headers = list(data[0].keys())
        rows = [[str(d.get(h, "")) for h in headers] for d in data]
        return TableFormatter.format_table(headers, rows, title)


class CollapsibleSection:
    @staticmethod
    def section(title: str, content: str, collapsed: bool = False) -> str:
        icon = _C["collapsed"] if collapsed else _C["expanded"]
        sep_char = _C["hline"]
        max_w = TERM_WIDTH - 4
        avail = max_w - len(title) - 6
        sep = sep_char * max(avail, 3)
        header = f"\n\033[1;96m  {icon}  {title}  {sep}\033[0m"
        if collapsed:
            return header + f"\n  \033[90m[collapsed {_C['emdash']} expand to view]\033[0m\n"
        return header + "\n" + content + "\n"

    @staticmethod
    def accordion(items: list[tuple[str, str, bool]]) -> str:
        return "\n".join(CollapsibleSection.section(t, c, col) for t, c, col in items)


class JSONYAMLFormatter:
    @staticmethod
    def format_json(data: Any, indent: int = 2) -> str:
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return str(data)
        return JSONYAMLFormatter._highlight_json(formatted)

    @staticmethod
    def _highlight_json(text: str) -> str:
        a = SyntaxHighlighter.ANSI
        text = re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: f'{a["green"]}{m.group()}{a["reset"]}', text)
        text = re.sub(r"\b(true|false)\b", lambda m: f'{a["cyan"]}{m.group()}{a["reset"]}', text)
        text = re.sub(r"\b(null|None)\b", lambda m: f'{a["red"]}{m.group()}{a["reset"]}', text)
        text = re.sub(r"\b-?\d+\.?\d*(?:[eE][+-]?\d+)?\b", lambda m: f'{a["magenta"]}{m.group()}{a["reset"]}', text)
        return text

    @staticmethod
    def format_yaml(data: Any) -> str:
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            result = JSONYAMLFormatter._dict_to_yaml(parsed, 0)
        except Exception:
            return str(data)
        return result

    @staticmethod
    def _dict_to_yaml(obj, depth: int = 0) -> str:
        a = SyntaxHighlighter.ANSI
        prefix = "  " * depth
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_str = f'{a["blue"]}{k}{a["reset"]}:'
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{key_str}")
                    lines.append(JSONYAMLFormatter._dict_to_yaml(v, depth + 1))
                else:
                    val_str = JSONYAMLFormatter._yaml_value(v)
                    lines.append(f"{prefix}{key_str} {val_str}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{a['cyan']}-\033[0m")
                    sub = JSONYAMLFormatter._dict_to_yaml(item, depth + 1)
                    lines.append(re.sub(r"^(\s*)", r"\1  ", sub, flags=re.MULTILINE))
                else:
                    val_str = JSONYAMLFormatter._yaml_value(item)
                    lines.append(f"{prefix}{a['cyan']}-\033[0m {val_str}")
        else:
            lines.append(f"{prefix}{JSONYAMLFormatter._yaml_value(obj)}")
        return "\n".join(lines)

    @staticmethod
    def _yaml_value(v) -> str:
        a = SyntaxHighlighter.ANSI
        if v is None:
            return f'{a["red"]}null{a["reset"]}'
        if isinstance(v, bool):
            return f'{a["cyan"]}{str(v).lower()}{a["reset"]}'
        if isinstance(v, (int, float)):
            return f'{a["magenta"]}{v}{a["reset"]}'
        return f'{a["green"]}"{v}"{a["reset"]}'


class StreamFormatter:
    def __init__(self):
        self._buffer = ""
        self._line_buffer = ""
        self._code_block_depth = 0
        self._section_depth = 0
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines = []
        self._start_time = datetime.now()
        self._token_count = 0

    def add_token(self, token: str) -> str:
        self._buffer += token
        self._token_count += 1

        if self._in_code_block:
            self._code_lines.append(token)
            if "```" in token:
                self._in_code_block = False
                result = self._finalize_code_block()
                self._code_lines = []
                return result
            return ""

        if "```" in token:
            self._in_code_block = True
            self._code_lang = ""
            self._code_lines = [token]
            return ""

        return token

    def _finalize_code_block(self) -> str:
        full = "".join(self._code_lines)
        m = re.match(r"```(\w*)\n(.*)", full, flags=re.DOTALL)
        if m:
            lang = m.group(1) or ""
            code = m.group(2).rstrip("`\n")
            highlighted = SyntaxHighlighter.highlight(code, lang)
            lang_label = f" {lang}" if lang else ""
            sep = f"\033[90m{_C['hline'] * (TERM_WIDTH - len(lang_label) - 6)}\033[0m"
            return f"\n{sep}\033[90m{_C['vline']}{lang_label}\033[0m\n{highlighted}\n{sep}\n"
        return full

    def get_stats(self) -> dict:
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return {
            "tokens": self._token_count,
            "elapsed": elapsed,
            "tokens_per_second": round(self._token_count / elapsed, 1) if elapsed > 0 else 0,
        }

    def reset(self):
        self._buffer = ""
        self._line_buffer = ""
        self._code_block_depth = 0
        self._section_depth = 0
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines = []
        self._token_count = 0
        self._start_time = datetime.now()


class OutputFormatter:
    @staticmethod
    def info_box(title: str, content: str) -> str:
        hl = _C["hline"]
        tl = _C["tl"]
        bl = _C["bl"]
        top = f"\033[96m{tl} {title}\033[0m"
        bottom = f"\033[96m{bl}{hl * (TERM_WIDTH - 2)}\033[0m"
        return f"{top}\n{content}\n{bottom}"

    @staticmethod
    def success_box(message: str) -> str:
        return f"\033[92m{_C['check']} {message}\033[0m"

    @staticmethod
    def error_box(message: str) -> str:
        return f"\033[91m{_C['cross_mark']} {message}\033[0m"

    @staticmethod
    def warning_box(message: str) -> str:
        return f"\033[93m{_C['warn']} {message}\033[0m"

    @staticmethod
    def status_badge(text: str, color: str = "cyan") -> str:
        a = SyntaxHighlighter.ANSI
        c = a.get(color, a["cyan"])
        return f"{c}[{text}]\033[0m"

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 30) -> str:
        filled = int((current / total) * width) if total > 0 else 0
        bar = _C["hbar_full"] * filled + _C["hbar_empty"] * (width - filled)
        pct = (current / total) * 100 if total > 0 else 0
        return f"\033[96m{bar}\033[0m {pct:.0f}%"

    @staticmethod
    def divider(char: str = "", label: str = "") -> str:
        if not char:
            char = _C["hline"]
        if label:
            text = f" {label} "
            avail = TERM_WIDTH - len(text)
            side = char * max(avail // 2, 0)
            return f"\033[90m{side}{text}{side}\033[0m"
        return f"\033[90m{char * TERM_WIDTH}\033[0m"

    @staticmethod
    def header(text: str, level: int = 1) -> str:
        hl = _C["hline"]
        if level == 1:
            return f"\n\033[1;96m{hl * TERM_WIDTH}\033[0m\n\033[1;96m  {text}\033[0m\n\033[1;96m{hl * TERM_WIDTH}\033[0m\n"
        elif level == 2:
            return f"\n\033[1;94m{text}\033[0m\n\033[94m{hl * min(len(text) + 4, TERM_WIDTH)}\033[0m\n"
        return f"\n\033[1;93m  {_C['arrow']} {text}\033[0m\n"


def format_response(text: str) -> str:
    text = CodeBlockFormatter.format(text)
    text = _format_lists(text)
    text = _format_inline_code(text)
    text = _format_bold_italic(text)
    text = _format_links(text)
    return text


def _format_lists(text: str) -> str:
    lines = text.split("\n")
    result = []
    in_list = False
    for line in lines:
        ul = re.match(r"^(\s*)[\*\-]\s+(.*)", line)
        ol = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if ul:
            indent = ul.group(1)
            content = ul.group(2)
            bullet = f"\033[96m{_C['bullet']}\033[0m"
            result.append(f"{indent}{bullet} {content}")
            in_list = True
        elif ol:
            indent = ol.group(1)
            content = ol.group(2)
            bullet = f"\033[93m{_C['empty_bullet']}\033[0m"
            result.append(f"{indent}{bullet} {content}")
            in_list = True
        else:
            if in_list and line.strip() == "":
                result.append("")
                in_list = False
            elif in_list and not line.strip():
                pass
            else:
                result.append(line)
                in_list = False
    return "\n".join(result)


def _format_inline_code(text: str) -> str:
    return re.sub(r"`([^`]+)`", r"\033[93m\1\033[0m", text)


def _format_bold_italic(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\033[1m\1\033[0m", text)
    text = re.sub(r"\*(.+?)\*", r"\033[3m\1\033[0m", text)
    text = re.sub(r"__(.+?)__", r"\033[1m\1\033[0m", text)
    text = re.sub(r"_(.+?)_", r"\033[3m\1\033[0m", text)
    return text


def _format_links(text: str) -> str:
    return re.sub(
        r"(https?://[^\s]+)",
        r"\033[94m\1\033[0m",
        text,
    )

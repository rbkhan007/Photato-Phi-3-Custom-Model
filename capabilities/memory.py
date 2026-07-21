#!/usr/bin/env python3
"""
Conversation Memory and Context Management for Local LLMs.

Features:
- Multi-turn conversation tracking
- Context window management
- Memory summarization
- Sliding window memory
- Importance-based memory retention

Usage:
    from capabilities.memory import ConversationMemory

    memory = ConversationMemory(max_context_length=4096)
    memory.add_message("user", "Hello!")
    memory.add_message("assistant", "Hi there!")
    context = memory.get_context()
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """Conversation message."""
    role: str
    content: str
    timestamp: float = 0.0
    token_count: int = 0
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Memory configuration."""
    max_context_length: int = 4096
    max_messages: int = 50
    summary_threshold: int = 20
    sliding_window_size: int = 10
    importance_threshold: float = 0.3


class TokenCounter:
    """Simple token counter (approximate)."""

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Approximate token count.

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        return len(text) // 4

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Truncate text to approximate token count.

        Args:
            text: Input text
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."


class ConversationMemory:
    """
    Conversation memory with context management.

    Features:
    - Multi-turn tracking
    - Context window management
    - Memory summarization
    - Sliding window
    - Importance-based retention
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self.messages: list[Message] = []
        self.summaries: list[str] = []
        self.token_counter = TokenCounter()

    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ):
        """
        Add a message to memory.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            importance: Message importance (0-1)
            metadata: Additional metadata
        """
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=self.token_counter.count_tokens(content),
            importance=importance,
            metadata=metadata or {},
        )

        self.messages.append(message)

        # Check if we need to summarize
        if len(self.messages) > self.config.summary_threshold:
            self._summarize_old_messages()

        # Check if we need to trim
        if len(self.messages) > self.config.max_messages:
            self._trim_messages()

    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_summaries: bool = True,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        Get context for model input.

        Args:
            max_tokens: Maximum tokens for context
            include_summaries: Include old summaries
            system_prompt: System prompt to include

        Returns:
            List of message dicts
        """
        max_tokens = max_tokens or self.config.max_context_length
        context = []

        # Add system prompt
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})
            max_tokens -= self.token_counter.count_tokens(system_prompt)

        # Add summaries
        if include_summaries and self.summaries:
            summary_text = "\n".join(self.summaries)
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary_text}",
            })
            max_tokens -= self.token_counter.count_tokens(summary_text)

        # Add recent messages (most recent last)
        remaining_tokens = max_tokens
        for message in reversed(self.messages):
            msg_tokens = message.token_count
            if remaining_tokens - msg_tokens >= 0:
                context.insert(-1 if include_summaries else 0, {
                    "role": message.role,
                    "content": message.content,
                })
                remaining_tokens -= msg_tokens
            else:
                break

        return context

    def get_recent_messages(self, n: int = 10) -> list[Message]:
        """Get recent messages."""
        return self.messages[-n:]

    def search_messages(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Message]:
        """
        Search messages by content.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of matching messages
        """
        query_lower = query.lower()
        results = []

        for message in reversed(self.messages):
            if query_lower in message.content.lower():
                results.append(message)
                if len(results) >= max_results:
                    break

        return results

    def get_important_messages(
        self,
        min_importance: float = 0.7,
        max_results: int = 10,
    ) -> list[Message]:
        """
        Get important messages.

        Args:
            min_importance: Minimum importance score
            max_results: Maximum results

        Returns:
            List of important messages
        """
        important = [
            m for m in self.messages
            if m.importance >= min_importance
        ]
        return sorted(important, key=lambda m: m.importance, reverse=True)[:max_results]

    def _summarize_old_messages(self):
        """Summarize old messages."""
        # Split messages into old and recent
        old_messages = self.messages[:self.config.summary_threshold // 2]
        self.messages = self.messages[self.config.summary_threshold // 2:]

        # Create summary
        summary = self._create_summary(old_messages)
        if summary:
            self.summaries.append(summary)

    def _create_summary(self, messages: list[Message]) -> str:
        """
        Create summary of messages.

        Args:
            messages: Messages to summarize

        Returns:
            Summary string
        """
        if not messages:
            return ""

        # Simple extractive summary
        key_points = []
        for msg in messages:
            if msg.importance >= 0.5:
                # Extract first sentence
                sentences = msg.content.split(".")
                if sentences and sentences[0].strip():
                    key_points.append(f"{msg.role}: {sentences[0].strip()}")

        return " | ".join(key_points[:5]) if key_points else ""

    def _trim_messages(self):
        """Trim messages to max count."""
        # Keep most recent messages
        self.messages = self.messages[-self.config.max_messages:]

    def clear(self):
        """Clear all messages."""
        self.messages.clear()
        self.summaries.clear()

    def export_conversation(self) -> str:
        """Export conversation as JSON."""
        data = {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "importance": msg.importance,
                }
                for msg in self.messages
            ],
            "summaries": self.summaries,
        }
        return json.dumps(data, indent=2)

    def import_conversation(self, json_str: str):
        """Import conversation from JSON."""
        data = json.loads(json_str)
        self.messages = [
            Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", 0),
                importance=msg.get("importance", 0.5),
            )
            for msg in data.get("messages", [])
        ]
        self.summaries = data.get("summaries", [])

    def get_stats(self) -> dict:
        """Get memory statistics."""
        total_tokens = sum(m.token_count for m in self.messages)
        return {
            "total_messages": len(self.messages),
            "total_tokens": total_tokens,
            "summaries": len(self.summaries),
            "avg_importance": (
                sum(m.importance for m in self.messages) / len(self.messages)
                if self.messages else 0
            ),
        }


class SlidingWindowMemory:
    """Sliding window memory implementation."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: list[Message] = []

    def add(self, role: str, content: str):
        """Add message, keeping only window_size messages."""
        self.messages.append(Message(
            role=role,
            content=content,
            timestamp=time.time(),
        ))
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self) -> list[dict]:
        """Get context from window."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]


class ImportanceBasedMemory:
    """Memory that retains important messages."""

    def __init__(self, max_size: int = 50, min_importance: float = 0.3):
        self.max_size = max_size
        self.min_importance = min_importance
        self.messages: list[Message] = []

    def add(self, role: str, content: str, importance: float = 0.5):
        """Add message with importance score."""
        if importance < self.min_importance:
            return  # Skip low importance messages

        self.messages.append(Message(
            role=role,
            content=content,
            timestamp=time.time(),
            importance=importance,
        ))

        # Trim by importance if too many
        if len(self.messages) > self.max_size:
            self.messages.sort(key=lambda m: m.importance, reverse=True)
            self.messages = self.messages[:self.max_size]

    def get_context(self) -> list[dict]:
        """Get context sorted by timestamp."""
        sorted_messages = sorted(self.messages, key=lambda m: m.timestamp)
        return [
            {"role": m.role, "content": m.content}
            for m in sorted_messages
        ]


def main(argv=None):
    """CLI for conversation memory and context management."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.memory",
        description="Conversation memory and context management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--file", help="Conversation JSON file (loaded and/or written)")

    def msg_to_dict(m):
        return {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
            "token_count": m.token_count,
            "importance": m.importance,
            "metadata": m.metadata,
        }

    def load_memory(file_path):
        mem = ConversationMemory()
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    mem.import_conversation(f.read())
            except FileNotFoundError:
                pass
        return mem

    def save_memory(mem, file_path):
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(mem.export_conversation())

    p_add = sub.add_parser("add", parents=[parent], help="Add a message to the conversation")
    p_add.add_argument("--role", required=True)
    p_add.add_argument("--content", help="Message content (or @file, or stdin)")
    p_add.add_argument("--importance", type=float, default=0.5)
    p_add.add_argument("--output", help="Write conversation to this file")

    p_ctx = sub.add_parser("context", parents=[parent], help="Show model context")
    p_ctx.add_argument("--max-tokens", type=int, default=None)
    p_ctx.add_argument("--system-prompt")
    p_ctx.add_argument("--no-summaries", action="store_true")

    p_stats = sub.add_parser("stats", parents=[parent], help="Show memory statistics")

    p_search = sub.add_parser("search", parents=[parent], help="Search messages")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--max-results", type=int, default=5)

    p_imp = sub.add_parser("important", parents=[parent], help="Show important messages")
    p_imp.add_argument("--min-importance", type=float, default=0.7)
    p_imp.add_argument("--max-results", type=int, default=10)

    p_exp = sub.add_parser("export", parents=[parent], help="Export conversation as JSON")

    p_imp2 = sub.add_parser("import", help="Import and re-export a conversation JSON")
    p_imp2.add_argument("--input", required=True)
    p_imp2.add_argument("--output", help="Write normalized conversation to this file")

    args = parser.parse_args(argv)

    try:
        if args.command == "add":
            content = args.content
            if content is None:
                content = sys.stdin.read()
            elif content.startswith("@"):
                with open(content[1:], "r", encoding="utf-8") as f:
                    content = f.read()
            mem = load_memory(args.file)
            mem.add_message(args.role, content, importance=args.importance)
            save_memory(mem, args.output or args.file)
            print(json.dumps({
                "added": {
                    "role": args.role,
                    "content": content,
                    "importance": args.importance,
                },
                "stats": mem.get_stats(),
            }, indent=2, default=str))
        elif args.command == "context":
            mem = load_memory(args.file)
            ctx = mem.get_context(
                max_tokens=args.max_tokens,
                include_summaries=not args.no_summaries,
                system_prompt=args.system_prompt,
            )
            print(json.dumps(ctx, indent=2, default=str))
        elif args.command == "stats":
            mem = load_memory(args.file)
            print(json.dumps(mem.get_stats(), indent=2, default=str))
        elif args.command == "search":
            mem = load_memory(args.file)
            results = [msg_to_dict(m) for m in mem.search_messages(args.query, args.max_results)]
            print(json.dumps(results, indent=2, default=str))
        elif args.command == "important":
            mem = load_memory(args.file)
            results = [msg_to_dict(m) for m in mem.get_important_messages(args.min_importance, args.max_results)]
            print(json.dumps(results, indent=2, default=str))
        elif args.command == "export":
            mem = load_memory(args.file)
            print(mem.export_conversation())
        elif args.command == "import":
            with open(args.input, "r", encoding="utf-8") as f:
                data = f.read()
            mem = ConversationMemory()
            mem.import_conversation(data)
            out = mem.export_conversation()
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(out)
            print(out)
        return 0
    except (OSError, ValueError, KeyError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

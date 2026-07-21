#!/usr/bin/env python3
"""
Prompt Caching for Local LLMs.

Features:
- LRU cache for prompts
- Hash-based cache keys
- TTL-based expiration
- Cache statistics
- Similar prompt matching

Usage:
    from capabilities.prompt_cache import PromptCache

    cache = PromptCache()
    result = cache.get_or_compute("prompt hash", lambda: expensive_computation())
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = 0.0


class PromptCache:
    """
    LRU cache for prompts.

    Features:
    - LRU eviction
    - TTL-based expiration
    - Hash-based keys
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum cache size
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, **kwargs) -> str:
        """Make cache key from prompt."""
        content = prompt + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.misses += 1
                return None

            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.hits += 1
            return entry.value

        self.misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live
        """
        # Remove if exists
        if key in self.cache:
            del self.cache[key]

        # Evict if full
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # Add new entry
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: Optional[float] = None,
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            compute_fn: Function to compute value
            ttl: Time-to-live

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = compute_fn()
        self.set(key, value, ttl)
        return value

    def get_by_prompt(
        self,
        prompt: str,
        **kwargs,
    ) -> Optional[Any]:
        """Get value by prompt string."""
        key = self._make_key(prompt, **kwargs)
        return self.get(key)

    def set_by_prompt(
        self,
        prompt: str,
        value: Any,
        ttl: Optional[float] = None,
        **kwargs,
    ):
        """Set value by prompt string."""
        key = self._make_key(prompt, **kwargs)
        self.set(key, value, ttl)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry.timestamp > entry.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


class SimilarityCache:
    """Cache with similarity-based lookup."""

    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.9):
        self.max_size = max_size
        self.threshold = similarity_threshold
        self.entries: list[tuple[str, Any]] = []

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        if s1 == s2:
            return 1.0

        # Simple Jaccard similarity
        set1 = set(s1.split())
        set2 = set(s2.split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def get(self, prompt: str) -> Optional[Any]:
        """Get by similar prompt."""
        for cached_prompt, value in reversed(self.entries):
            if self._similarity(prompt, cached_prompt) >= self.threshold:
                return value
        return None

    def set(self, prompt: str, value: Any):
        """Set by prompt."""
        self.entries.append((prompt, value))
        if len(self.entries) > self.max_size:
            self.entries.pop(0)


class TTLCache:
    """TTL-based cache."""

    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self.cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value with TTL."""
        expiry = time.time() + (ttl or self.default_ttl)
        self.cache[key] = (value, expiry)

    def cleanup(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, (_, e) in self.cache.items() if now >= e]
        for k in expired:
            del self.cache[k]
        return len(expired)


def main(argv=None):
    """CLI for prompt caching (LRU + TTL)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.prompt_cache",
        description="Prompt caching (LRU + TTL) for local LLMs",
    )
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--store", help="Cache store JSON file (persisted between runs)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", parents=[parent], help="Set a key/value")
    p_set.add_argument("--key", required=True)
    p_set.add_argument("--value", required=True, help="Value (parsed as JSON if possible, else string)")
    p_set.add_argument("--ttl", type=float, default=None)

    p_get = sub.add_parser("get", parents=[parent], help="Get a value by key")
    p_get.add_argument("--key", required=True)

    p_sp = sub.add_parser("set-prompt", parents=[parent], help="Set by prompt string")
    sp_src = p_sp.add_mutually_exclusive_group(required=True)
    sp_src.add_argument("--prompt")
    sp_src.add_argument("--file")
    p_sp.add_argument("--value", required=True)
    p_sp.add_argument("--ttl", type=float, default=None)
    p_sp.add_argument("--kwargs", default="{}", help="JSON object of extra key components")

    p_gp = sub.add_parser("get-prompt", parents=[parent], help="Get by prompt string")
    gp_src = p_gp.add_mutually_exclusive_group(required=True)
    gp_src.add_argument("--prompt")
    gp_src.add_argument("--file")
    p_gp.add_argument("--kwargs", default="{}", help="JSON object of extra key components")

    p_inv = sub.add_parser("invalidate", parents=[parent], help="Invalidate a key")
    p_inv.add_argument("--key", required=True)

    p_clr = sub.add_parser("clear", parents=[parent], help="Clear the cache")

    p_st = sub.add_parser("stats", parents=[parent], help="Show cache statistics")

    args = parser.parse_args(argv)

    def parse_value(raw):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def load_cache(path):
        cache = PromptCache()
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, entry in data.get("cache", {}).items():
                    cache.cache[key] = CacheEntry(
                        key=key,
                        value=entry["value"],
                        timestamp=entry["timestamp"],
                        ttl=entry["ttl"],
                        access_count=entry.get("access_count", 0),
                        last_accessed=entry.get("last_accessed", 0.0),
                    )
                cache.hits = data.get("hits", 0)
                cache.misses = data.get("misses", 0)
            except FileNotFoundError:
                pass
        return cache

    def save_cache(cache, path):
        if not path:
            return
        data = {
            "cache": {
                k: {
                    "value": v.value,
                    "timestamp": v.timestamp,
                    "ttl": v.ttl,
                    "access_count": v.access_count,
                    "last_accessed": v.last_accessed,
                }
                for k, v in cache.cache.items()
            },
            "hits": cache.hits,
            "misses": cache.misses,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return value

    try:
        cache = load_cache(args.store)
        if args.command == "set":
            value = parse_value(args.value)
            cache.set(args.key, value, ttl=args.ttl)
            save_cache(cache, args.store)
            print(json.dumps({"key": args.key, "value": value, "stats": cache.get_stats()}, indent=2, default=str))
        elif args.command == "get":
            value = cache.get(args.key)
            print(json.dumps({"key": args.key, "value": value}, indent=2, default=str))
        elif args.command == "set-prompt":
            prompt = read_source(args.prompt, args.file)
            kwargs = json.loads(args.kwargs)
            value = parse_value(args.value)
            cache.set_by_prompt(prompt, value, ttl=args.ttl, **kwargs)
            save_cache(cache, args.store)
            print(json.dumps({"prompt": prompt, "value": value, "stats": cache.get_stats()}, indent=2, default=str))
        elif args.command == "get-prompt":
            prompt = read_source(args.prompt, args.file)
            kwargs = json.loads(args.kwargs)
            value = cache.get_by_prompt(prompt, **kwargs)
            print(json.dumps({"prompt": prompt, "value": value}, indent=2, default=str))
        elif args.command == "invalidate":
            ok = cache.invalidate(args.key)
            save_cache(cache, args.store)
            print(json.dumps({"key": args.key, "invalidated": ok}, indent=2))
        elif args.command == "clear":
            cache.clear()
            save_cache(cache, args.store)
            print(json.dumps({"cleared": True}, indent=2))
        elif args.command == "stats":
            print(json.dumps(cache.get_stats(), indent=2, default=str))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

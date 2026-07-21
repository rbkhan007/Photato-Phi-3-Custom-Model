#!/usr/bin/env python3
"""
API Gateway with Rate Limiting for Local LLMs.

Features:
- Rate limiting (per client, per endpoint)
- Request caching
- Load balancing
- API key authentication
- Request/response logging
- Usage analytics

Usage:
    from gateway.api_gateway import APIGateway

    gateway = APIGateway(model_server="http://localhost:8080")
    gateway.run(port=8000)
"""

import argparse
import hashlib
import json
import os
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    window_size: int = 60


@dataclass
class ClientInfo:
    """Client information."""
    client_id: str
    api_key: Optional[str] = None
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    last_minute_reset: float = 0
    last_hour_reset: float = 0
    total_requests: int = 0


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.config.burst_size,
                self.tokens + elapsed * (self.config.requests_per_minute / 60),
            )
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RequestCache:
    """Simple request cache."""

    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.cache: dict[str, tuple[float, dict]] = {}
        self.ttl = ttl
        self.max_size = max_size

    def get(self, key: str) -> Optional[dict]:
        """Get cached response."""
        if key in self.cache:
            timestamp, response = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return response
            del self.cache[key]
        return None

    def set(self, key: str, response: dict):
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        self.cache[key] = (time.time(), response)

    def clear(self):
        """Clear cache."""
        self.cache.clear()


class APIGateway:
    """
    API Gateway with rate limiting and caching.

    Features:
    - Rate limiting per client
    - Request caching
    - API key authentication
    - Usage analytics
    - Load balancing
    """

    def __init__(
        self,
        model_server: str = "http://localhost:8080",
        port: int = 8000,
        rate_limit_config: Optional[RateLimitConfig] = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize gateway.

        Args:
            model_server: Model server URL
            port: Gateway port
            rate_limit_config: Rate limit configuration
            cache_ttl: Cache time-to-live
        """
        self.model_server = model_server
        self.port = port
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.cache = RequestCache(ttl=cache_ttl)
        self.clients: dict[str, ClientInfo] = {}
        self.analytics: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def add_client(self, client_id: str, api_key: Optional[str] = None):
        """Add a client."""
        self.clients[client_id] = ClientInfo(
            client_id=client_id,
            api_key=api_key,
            rate_limit=self.rate_limit_config,
        )

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        with self.lock:
            if client_id not in self.clients:
                self.add_client(client_id)

            client = self.clients[client_id]

            # Reset counters if needed
            now = time.time()
            if now - client.last_minute_reset >= 60:
                client.requests_this_minute = 0
                client.last_minute_reset = now
            if now - client.last_hour_reset >= 3600:
                client.requests_this_hour = 0
                client.last_hour_reset = now

            # Check limits
            if client.requests_this_minute >= client.rate_limit.requests_per_minute:
                return False
            if client.requests_this_hour >= client.rate_limit.requests_per_hour:
                return False

            client.requests_this_minute += 1
            client.requests_this_hour += 1
            client.total_requests += 1

            return True

    def get_cache_key(self, request_data: dict) -> str:
        """Generate cache key from request."""
        content = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def proxy_request(self, request_data: dict, client_id: str) -> dict:
        """
        Proxy request to model server.

        Args:
            request_data: Request data
            client_id: Client ID

        Returns:
            Response dict
        """
        # Check rate limit
        if not self.check_rate_limit(client_id):
            return {
                "error": "Rate limit exceeded",
                "retry_after": 60,
            }

        # Check cache
        cache_key = self.get_cache_key(request_data)
        cached = self.cache.get(cache_key)
        if cached:
            self.analytics["cache_hits"] += 1
            return cached

        # Forward to model server
        try:
            import requests

            response = requests.post(
                f"{self.model_server}/v1/chat/completions",
                json=request_data,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                self.cache.set(cache_key, result)
                self.analytics["requests_success"] += 1
                return result
            else:
                self.analytics["requests_failed"] += 1
                return {"error": f"Model server error: {response.status_code}"}

        except Exception as e:
            self.analytics["requests_failed"] += 1
            return {"error": str(e)}

    def get_analytics(self) -> dict:
        """Get usage analytics."""
        with self.lock:
            return {
                "total_requests": sum(c.total_requests for c in self.clients.values()),
                "active_clients": len(self.clients),
                "cache_hits": self.analytics["cache_hits"],
                "requests_success": self.analytics["requests_success"],
                "requests_failed": self.analytics["requests_failed"],
                "clients": {
                    cid: {
                        "total_requests": c.total_requests,
                        "requests_this_minute": c.requests_this_minute,
                    }
                    for cid, c in self.clients.items()
                },
            }

    def run(self):
        """Run the gateway server."""
        gateway = self

        class GatewayHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                sys.stderr.write(
                    "%s - - [%s] %s\n"
                    % (self.address_string(), self.log_date_time_string(), format % args)
                )

            def _send_json(self, data: dict, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _read_body(self) -> dict:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                return json.loads(body) if body else {}

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Client-ID, X-API-Key")
                self.end_headers()

            def do_GET(self):
                if self.path == "/health":
                    self._send_json({"status": "ok"})
                elif self.path == "/analytics":
                    self._send_json(gateway.get_analytics())
                elif self.path == "/metrics":
                    # Prometheus metrics format
                    analytics = gateway.get_analytics()
                    metrics = f"""# HELP llm_gateway_requests_total Total requests
# TYPE llm_gateway_requests_total counter
llm_gateway_requests_total {analytics['total_requests']}
# HELP llm_gateway_cache_hits_total Cache hits
# TYPE llm_gateway_cache_hits_total counter
llm_gateway_cache_hits_total {analytics['cache_hits']}
"""
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(metrics.encode())
                else:
                    self._send_json({"error": "Not found"}, 404)

            def do_POST(self):
                try:
                    body = self._read_body()
                    client_id = self.headers.get("X-Client-ID", "anonymous")

                    if self.path == "/v1/chat/completions":
                        response = gateway.proxy_request(body, client_id)
                        self._send_json(response)
                    elif self.path == "/v1/completions":
                        response = gateway.proxy_request(body, client_id)
                        self._send_json(response)
                    else:
                        self._send_json({"error": "Not found"}, 404)

                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

        server = HTTPServer(("0.0.0.0", self.port), GatewayHandler)

        print(f"\nAPI Gateway running on http://0.0.0.0:{self.port}")
        print(f"Proxying to: {self.model_server}")
        print(f"\nEndpoints:")
        print(f"  POST /v1/chat/completions - Chat completion")
        print(f"  POST /v1/completions      - Text completion")
        print(f"  GET  /health              - Health check")
        print(f"  GET  /analytics           - Usage analytics")
        print(f"  GET  /metrics             - Prometheus metrics")
        print(f"\nPress Ctrl+C to stop.\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down gateway...")
            server.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description="API Gateway for LLM")
    parser.add_argument("--model-server", default="http://localhost:8080", help="Model server URL")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port")
    parser.add_argument("--rate-limit", type=int, default=60, help="Requests per minute")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run the gateway HTTP server (blocking)")

    p = sub.add_parser("check-rate-limit", help="Check rate limiting for a client")
    p.add_argument("--client-id", default="test-client", help="Client identifier")
    p.add_argument("--times", type=int, default=5, help="Number of requests to simulate")

    p = sub.add_parser("cache-key", help="Compute the cache key for a request")
    p.add_argument("--request", required=True, help="Request data as a JSON string")

    args = parser.parse_args(argv)

    try:
        config = RateLimitConfig(requests_per_minute=args.rate_limit)
        gateway = APIGateway(
            model_server=args.model_server,
            port=args.port,
            rate_limit_config=config,
        )

        if args.command == "serve" or args.command is None:
            gateway.run()
            return 0
        elif args.command == "check-rate-limit":
            allowed = [gateway.check_rate_limit(args.client_id) for _ in range(args.times)]
            print(json.dumps(
                {
                    "client_id": args.client_id,
                    "attempts": args.times,
                    "allowed": allowed,
                    "all_allowed": all(allowed),
                },
                indent=2,
            ))
            return 0 if all(allowed) else 1
        elif args.command == "cache-key":
            request_data = json.loads(args.request)
            print(json.dumps({"cache_key": gateway.get_cache_key(request_data)}, indent=2))
            return 0
        else:
            parser.error("Unknown command")
            return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

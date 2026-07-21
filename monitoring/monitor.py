#!/usr/bin/env python3
"""
Monitoring and Observability for Local LLMs.

Features:
- Request/response logging
- Performance metrics
- Error tracking
- Usage analytics
- Health checks
- Alerting

Usage:
    from monitoring.monitor import LLMMonitor

    monitor = LLMMonitor()
    monitor.log_request(request_data, response_data, latency)
"""

import argparse
import json
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RequestLog:
    """Request log entry."""
    timestamp: float
    request_id: str
    client_id: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    status: str
    error: Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Metrics snapshot."""
    timestamp: float
    total_requests: int
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_tokens: int
    tokens_per_second: float
    active_connections: int


class LLMMonitor:
    """
    Monitoring and observability for LLM deployments.

    Features:
    - Request/response logging
    - Performance metrics
    - Error tracking
    - Usage analytics
    - Health checks
    - Alerting
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        metrics_window: int = 300,
        alert_threshold: float = 0.1,
    ):
        """
        Initialize monitor.

        Args:
            log_dir: Log directory
            metrics_window: Metrics window in seconds
            alert_threshold: Error rate threshold for alerts
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_window = metrics_window
        self.alert_threshold = alert_threshold

        # Request logs
        self.request_logs: deque[RequestLog] = deque(maxlen=10000)
        self.metrics_history: deque[MetricsSnapshot] = deque(maxlen=1000)

        # Counters
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)

        # Latency tracking
        self.latencies: deque[float] = deque(maxlen=10000)

        # Error tracking
        self.errors: deque[dict] = deque(maxlen=1000)

        # Thread safety
        self.lock = threading.Lock()

        # Start metrics collection
        self._start_metrics_collection()

    def _start_metrics_collection(self):
        """Start background metrics collection."""
        def collect_metrics():
            while True:
                time.sleep(60)
                self._collect_metrics()

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

    def log_request(
        self,
        request_data: dict,
        response_data: dict,
        latency_ms: float,
        client_id: str = "anonymous",
    ):
        """
        Log a request.

        Args:
            request_data: Request data
            response_data: Response data
            latency_ms: Request latency in milliseconds
            client_id: Client ID
        """
        with self.lock:
            # Extract metrics
            prompt_tokens = request_data.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response_data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Create log entry
            log_entry = RequestLog(
                timestamp=time.time(),
                request_id=response_data.get("id", ""),
                client_id=client_id,
                endpoint=request_data.get("endpoint", "unknown"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                status="success" if "error" not in response_data else "error",
                error=response_data.get("error"),
            )

            self.request_logs.append(log_entry)
            self.latencies.append(latency_ms)

            # Update counters
            self.counters["total_requests"] += 1
            self.counters["total_tokens"] += total_tokens
            if log_entry.status == "error":
                self.counters["total_errors"] += 1
                self.errors.append({
                    "timestamp": time.time(),
                    "error": log_entry.error,
                    "client_id": client_id,
                })

    def log_error(self, error: str, context: Optional[dict] = None):
        """
        Log an error.

        Args:
            error: Error message
            context: Additional context
        """
        with self.lock:
            self.counters["total_errors"] += 1
            self.errors.append({
                "timestamp": time.time(),
                "error": error,
                "context": context,
            })

    def get_metrics(self) -> MetricsSnapshot:
        """
        Get current metrics.

        Returns:
            MetricsSnapshot
        """
        with self.lock:
            now = time.time()
            window_start = now - self.metrics_window

            # Filter recent logs
            recent_logs = [
                log for log in self.request_logs
                if log.timestamp >= window_start
            ]

            # Calculate metrics
            total_requests = len(recent_logs)
            if total_requests > 0:
                requests_per_second = total_requests / self.metrics_window
                avg_latency = sum(log.latency_ms for log in recent_logs) / total_requests

                # Calculate percentiles
                sorted_latencies = sorted(log.latency_ms for log in recent_logs)
                p95_index = int(total_requests * 0.95)
                p99_index = int(total_requests * 0.99)
                p95_latency = sorted_latencies[min(p95_index, total_requests - 1)]
                p99_latency = sorted_latencies[min(p99_index, total_requests - 1)]

                # Error rate
                errors = sum(1 for log in recent_logs if log.status == "error")
                error_rate = errors / total_requests

                # Tokens
                total_tokens = sum(log.total_tokens for log in recent_logs)
                tokens_per_second = total_tokens / self.metrics_window
            else:
                requests_per_second = 0
                avg_latency = 0
                p95_latency = 0
                p99_latency = 0
                error_rate = 0
                total_tokens = 0
                tokens_per_second = 0

            return MetricsSnapshot(
                timestamp=now,
                total_requests=self.counters["total_requests"],
                requests_per_second=requests_per_second,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                error_rate=error_rate,
                total_tokens=self.counters["total_tokens"],
                tokens_per_second=tokens_per_second,
                active_connections=self.gauges.get("active_connections", 0),
            )

    def _collect_metrics(self):
        """Collect and store metrics snapshot."""
        metrics = self.get_metrics()
        self.metrics_history.append(metrics)

        # Check for alerts
        if metrics.error_rate > self.alert_threshold:
            self._send_alert(f"High error rate: {metrics.error_rate:.2%}")

        # Save metrics to file
        self._save_metrics(metrics)

    def _send_alert(self, message: str):
        """Send alert notification."""
        print(f"\n[ALERT] {message}")
        # In production, send email, Slack, etc.

    def _save_metrics(self, metrics: MetricsSnapshot):
        """Save metrics to file."""
        timestamp = datetime.fromtimestamp(metrics.timestamp).strftime("%Y%m%d_%H%M%S")
        metrics_file = self.log_dir / f"metrics_{timestamp}.json"

        data = {
            "timestamp": metrics.timestamp,
            "total_requests": metrics.total_requests,
            "requests_per_second": metrics.requests_per_second,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "error_rate": metrics.error_rate,
            "total_tokens": metrics.total_tokens,
            "tokens_per_second": metrics.tokens_per_second,
        }

        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_health_check(self) -> dict:
        """
        Get health check status.

        Returns:
            Health check dict
        """
        metrics = self.get_metrics()

        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics": {
                "total_requests": metrics.total_requests,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
            },
        }

        # Check health conditions
        if metrics.error_rate > 0.5:
            health["status"] = "unhealthy"
            health["reason"] = "High error rate"
        elif metrics.avg_latency_ms > 10000:
            health["status"] = "degraded"
            health["reason"] = "High latency"

        return health

    def get_usage_report(self, hours: int = 24) -> dict:
        """
        Get usage report.

        Args:
            hours: Number of hours to report

        Returns:
            Usage report dict
        """
        with self.lock:
            cutoff = time.time() - (hours * 3600)
            recent_logs = [
                log for log in self.request_logs
                if log.timestamp >= cutoff
            ]

            # Group by client
            client_usage = defaultdict(lambda: {
                "requests": 0,
                "tokens": 0,
                "errors": 0,
            })

            for log in recent_logs:
                client_usage[log.client_id]["requests"] += 1
                client_usage[log.client_id]["tokens"] += log.total_tokens
                if log.status == "error":
                    client_usage[log.client_id]["errors"] += 1

            return {
                "period_hours": hours,
                "total_requests": len(recent_logs),
                "total_tokens": sum(log.total_tokens for log in recent_logs),
                "unique_clients": len(client_usage),
                "client_usage": dict(client_usage),
            }

    def export_logs(self, format: str = "json") -> str:
        """
        Export logs.

        Args:
            format: Export format (json, csv)

        Returns:
            Exported logs as string
        """
        with self.lock:
            if format == "json":
                logs = [
                    {
                        "timestamp": log.timestamp,
                        "request_id": log.request_id,
                        "client_id": log.client_id,
                        "endpoint": log.endpoint,
                        "prompt_tokens": log.prompt_tokens,
                        "completion_tokens": log.completion_tokens,
                        "total_tokens": log.total_tokens,
                        "latency_ms": log.latency_ms,
                        "status": log.status,
                        "error": log.error,
                    }
                    for log in self.request_logs
                ]
                return json.dumps(logs, indent=2)

            elif format == "csv":
                lines = ["timestamp,request_id,client_id,endpoint,prompt_tokens,completion_tokens,total_tokens,latency_ms,status,error"]
                for log in self.request_logs:
                    line = f"{log.timestamp},{log.request_id},{log.client_id},{log.endpoint},{log.prompt_tokens},{log.completion_tokens},{log.total_tokens},{log.latency_ms},{log.status},{log.error or ''}"
                    lines.append(line)
                return "\n".join(lines)

            else:
                raise ValueError(f"Unsupported format: {format}")


def main(argv=None):
    """Monitoring and observability operations from the command line."""
    parser = argparse.ArgumentParser(
        description="Monitoring and observability for local LLMs"
    )
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--metrics-window", type=int, default=300)
    parser.add_argument("--alert-threshold", type=float, default=0.1)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metrics", help="Print current metrics snapshot")
    sub.add_parser("health", help="Print health check status")

    p_usage = sub.add_parser("usage", help="Print usage report")
    p_usage.add_argument("--hours", type=int, default=24)

    p_export = sub.add_parser("export", help="Export request logs")
    p_export.add_argument("--format", default="json", choices=["json", "csv"])
    p_export.add_argument("--output", help="Write export to this path")

    p_sim = sub.add_parser("simulate", help="Log synthetic requests then show metrics")
    p_sim.add_argument("--count", type=int, default=10)

    args = parser.parse_args(argv)

    try:
        monitor = LLMMonitor(
            log_dir=args.log_dir,
            metrics_window=args.metrics_window,
            alert_threshold=args.alert_threshold,
        )

        if args.command == "metrics":
            print(json.dumps(monitor.get_metrics().__dict__, indent=2, default=str))

        elif args.command == "health":
            print(json.dumps(monitor.get_health_check(), indent=2, default=str))

        elif args.command == "usage":
            print(json.dumps(monitor.get_usage_report(args.hours), indent=2, default=str))

        elif args.command == "export":
            data = monitor.export_logs(args.format)
            if args.output:
                Path(args.output).write_text(data)
                print(f"Exported logs to {args.output}")
            else:
                print(data)

        elif args.command == "simulate":
            for i in range(args.count):
                request_data = {
                    "endpoint": "/v1/chat/completions",
                    "usage": {"prompt_tokens": 10},
                }
                response_data = {
                    "id": f"req_{i}",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 50},
                }
                monitor.log_request(request_data, response_data, latency_ms=100 + i * 10)
            print(json.dumps(monitor.get_metrics().__dict__, indent=2, default=str))

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

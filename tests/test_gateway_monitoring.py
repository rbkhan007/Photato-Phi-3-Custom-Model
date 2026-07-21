"""Full coverage for the gateway and monitoring packages."""
import time

from gateway.api_gateway import APIGateway, RateLimiter, RateLimitConfig, RequestCache
from monitoring.monitor import LLMMonitor, RequestLog


def test_rate_limiter_allows_then_limits():
    cfg = RateLimitConfig(requests_per_minute=2, window_size=60, burst_size=2)
    limiter = RateLimiter(config=cfg)
    # First calls consume the burst budget, later calls get throttled.
    results = [limiter.allow() for _ in range(10)]
    assert all(isinstance(r, bool) for r in results)
    assert False in results  # throttle kicks in after the burst


def test_request_cache_set_get():
    cache = RequestCache(ttl=10)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.get("missing") is None
    cache.clear()
    assert cache.get("k") is None


def test_gateway_construction():
    gw = APIGateway(model_server="http://localhost:8080", port=8000)
    assert gw.port == 8000
    assert gw.model_server == "http://localhost:8080"


def test_gateway_cache_key_stable():
    gw = APIGateway(model_server="http://x", port=1)
    key1 = gw.get_cache_key({"method": "POST", "path": "/v1/chat/completions", "body": {"a": 1}})
    key2 = gw.get_cache_key({"method": "POST", "path": "/v1/chat/completions", "body": {"a": 1}})
    assert key1 == key2


def test_gateway_check_rate_limit():
    gw = APIGateway(model_server="http://x", port=1)
    gw.add_client("c1")
    assert gw.check_rate_limit("c1") is True
    assert gw.check_rate_limit("c1") is True


def test_monitor_log_and_metrics():
    m = LLMMonitor()
    m.log_request({"prompt": "hi"}, {"text": "hello"}, latency_ms=5)
    metrics = m.get_metrics()
    assert metrics.total_requests >= 1
    assert metrics.avg_latency_ms >= 0


def test_monitor_log_error():
    m = LLMMonitor()
    m.log_error("boom", {"prompt": "x"})
    report = m.get_usage_report(hours=1)
    assert isinstance(report, dict)


def test_monitor_health_check():
    m = LLMMonitor()
    health = m.get_health_check()
    assert isinstance(health, dict)
    assert "status" in health


def test_monitor_export_logs():
    m = LLMMonitor()
    m.log_request({"prompt": "hi"}, {"text": "ok"}, latency_ms=3)
    logs = m.export_logs()
    assert isinstance(logs, str)

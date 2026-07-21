"""
Local model backends for the Agentic CLI.

Provides real inference against locally-running model servers using only the
Python standard library (no extra dependencies required):

- OllamaBackend:  talks to an Ollama server (default http://localhost:11434)
- OpenAICompatBackend:  talks to any OpenAI-compatible server such as the
  llama.cpp server or LM Studio (default http://localhost:8080/v1)
- EchoBackend:  offline fallback that requires no server; useful for testing
  and when no model server is available.

Use `get_backend(...)` / `auto_backend(...)` to obtain a backend instance.
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional


class BackendError(RuntimeError):
    """Raised when a model backend cannot fulfil a request."""


@dataclass
class ChatResult:
    """Result of a chat completion."""
    content: str
    model: str
    backend: str
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": True,
            "content": self.content,
            "model": self.model,
            "backend": self.backend,
            "usage": self.usage,
        }


def _http_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    """POST JSON and parse a JSON response using only stdlib."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise BackendError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise BackendError(f"Cannot reach {url}: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise BackendError(f"Invalid JSON from {url}: {body[:200]}") from e


def _http_get(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        raise BackendError(f"Cannot reach {url}: {e}") from e


class ModelBackend:
    """Abstract chat backend."""

    name = "base"

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        raise NotImplementedError

    def available(self) -> bool:
        """Return True if the backend is reachable/usable."""
        return True


class OllamaBackend(ModelBackend):
    """Chat via an Ollama server."""

    name = "ollama"

    def __init__(self, model: str = "phi3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def available(self) -> bool:
        try:
            _http_get(f"{self.host}/api/tags", timeout=3.0)
            return True
        except BackendError:
            return False

    def list_models(self) -> list[str]:
        data = _http_get(f"{self.host}/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = _http_json(f"{self.host}/api/chat", payload)
        content = (data.get("message") or {}).get("content", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }
        return ChatResult(content=content, model=self.model, backend=self.name, usage=usage)


class OpenAICompatBackend(ModelBackend):
    """Chat via any OpenAI-compatible server (llama.cpp, LM Studio, vLLM)."""

    name = "openai-compat"

    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:8080/v1", api_key: str = "sk-none"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def available(self) -> bool:
        try:
            _http_get(f"{self.base_url}/models", timeout=3.0)
            return True
        except BackendError:
            return False

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise BackendError(f"Cannot reach {self.base_url}: {e}") from e
        choice = (out.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "")
        return ChatResult(content=content, model=self.model, backend=self.name, usage=out.get("usage", {}))


class LlamaCppBackend(ModelBackend):
    """In-process inference via llama.cpp (llama_cpp).

    Fastest option: no HTTP server, no JSON serialization overhead.
    Loads the GGUF directly with CPU optimizations (mmap, mlock,
    physical-core threads) and task-aware sampling from ``inference.auto_tuner``.
    """

    name = "llamacpp"

    def __init__(
        self,
        model: Optional[str] = None,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_batch: int = 512,
        mlock: bool = True,
        n_gpu_layers: int = 0,
        cpu_percent: float = 55.0,
    ):
        # Resolve the GGUF path. ``model`` may be a direct path, a filename
        # in the project's notebooks/ dir, or a model name (falls back to the
        # default bundled GGUF).
        if model_path:
            self.model_path = model_path
        elif model and os.path.exists(model):
            self.model_path = model
        elif model and not os.path.sep in model and not model.endswith(".gguf"):
            candidate = os.path.join("notebooks", f"{model}.gguf")
            self.model_path = candidate if os.path.exists(candidate) else model
        else:
            self.model_path = model or os.path.join(
                "notebooks", "Phi-4-mini-instruct-Q4_K_M.gguf"
            )
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.mlock = mlock
        self.n_gpu_layers = n_gpu_layers
        self.cpu_percent = cpu_percent
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from inference.llama_engine import FastLlamaEngine

            self._engine = FastLlamaEngine(
                self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                mlock=self.mlock,
                n_gpu_layers=self.n_gpu_layers,
                cpu_percent=self.cpu_percent,
            )
        return self._engine

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return os.path.exists(self.model_path)
        except ImportError:
            return False

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> ChatResult:
        out = self.engine.generate(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return ChatResult(
            content=out["text"],
            model=os.path.basename(self.model_path),
            backend=self.name,
            usage={
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "tokens_per_second": round(out["tokens_per_second"], 2),
                "first_token_ms": round(out["first_token_ms"], 1),
            },
        )

    def stream_chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024, **kwargs):
        """Yield assistant reply chunks as they are generated (llama.cpp only)."""
        engine = self.engine
        prompt = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")[-800:]
        params = engine.tuner.get_params(prompt=prompt)
        stream = engine.model.create_chat_completion(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            max_tokens=max_tokens,
            stream=True,
        )
        for delta in stream:
            content = (delta.get("choices") or [{}])[0].get("delta", {}).get("content", "")
            if content:
                yield content


class EchoBackend(ModelBackend):
    """Offline fallback that echoes a helpful, deterministic response."""

    name = "echo"

    def __init__(self, model: str = "echo"):
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last = m.get("content", "")
                break
        content = (
            "[no model server available] "
            "Start Ollama (`ollama serve`) or a llama.cpp OpenAI server, "
            f"then retry. You said: {last!r}"
        )
        return ChatResult(content=content, model=self.model, backend=self.name)


_BACKENDS = {
    "ollama": OllamaBackend,
    "openai": OpenAICompatBackend,
    "openai-compat": OpenAICompatBackend,
    "llamacpp": LlamaCppBackend,
    "echo": EchoBackend,
}


def get_backend(kind: str, **kwargs) -> ModelBackend:
    """Construct a backend by name."""
    key = (kind or "echo").lower()
    if key not in _BACKENDS:
        raise BackendError(f"Unknown backend: {kind}. Options: {', '.join(sorted(set(_BACKENDS)))}")
    cls = _BACKENDS[key]
    valid = cls.__init__.__code__.co_varnames
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**filtered)


def auto_backend(model: Optional[str] = None, n_gpu_layers: int = 0) -> ModelBackend:
    """Pick the best available backend.

    Prefers the fast in-process llama.cpp engine (``llamacpp``), then an
    Ollama server, then any OpenAI-compatible server, falling back to
    ``EchoBackend``.
    """
    try:
        lc = LlamaCppBackend(model=model, n_gpu_layers=n_gpu_layers)
        if lc.available():
            return lc
    except Exception:
        pass
    ollama = OllamaBackend(model=model or "phi3")
    if ollama.available():
        return ollama
    oai = OpenAICompatBackend(model=model or "local-model")
    if oai.available():
        return oai
    return EchoBackend()

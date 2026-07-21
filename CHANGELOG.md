# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-07-21

### Added

#### Core Features
- **Real Local Inference** — In-process llama.cpp via `llama-cpp-python`
- **CPU Throttling** — Windows Job Object hard cap, priority lowering, thread budgeting
- **Auto Parameter Tuning** — Task-aware sampling presets
- **Streaming Responses** — Token-by-token streaming with real-time display

#### Agentic CLI
- **Full-Screen TUI** — prompt_toolkit-based interface with scrolling history, input bar, status bar
- **Rich REPL** — Fallback REPL for environments without console support
- **Multiple Backends** — `llamacpp` (in-process), `ollama`, `openai-compat`, `echo`
- **Session Persistence** — Save/load conversation history, tool calls, session metadata
- **Token Metrics** — Display tokens/sec, elapsed time, token count

#### RAG (Retrieval Augmented Generation)
- **Semantic Embeddings** — GGUF embedding models via llama.cpp (Qwen3-Embedding-0.6B)
- **pgvector Store** — PostgreSQL + pgvector for cosine similarity search
- **Memory Store** — In-memory vector store fallback
- **Document Ingestion** — Text and file ingestion with chunking

#### Claude-Compatible Tool Calling
- **Tool Registry** — Register, validate, and manage tools with JSON Schema
- **Tool Parser** — Extract tool calls from model output (JSON, XML, function patterns)
- **Tool Executor** — Sandboxed execution with timeout, retries, and error recovery
- **Claude API Format** — Full compatibility with Claude's `tool_use`/`tool_result` format

#### AI Capabilities
- **Extended Thinking** — Chain of thought reasoning
- **Conversation Memory** — Multi-turn tracking, context window management
- **Structured Output** — JSON mode, schema validation
- **Safety Layer** — Content filtering, jailbreak detection
- **Prompt Caching** — LRU cache, TTL expiration
- **Multi-modal** — Image understanding, description, OCR
- **J-Space** — Dynamic concept buffer for internal reasoning
- **J-Lens** — Layer-by-layer visualization of model's internal thoughts

#### Optimization
- **Inference Engine** — KV-cache, prefill optimization, early exit
- **Attention Optimization** — Flash Attention, GQA, sparse attention
- **Probability Sampling** — Top-k, top-p, min-p, Mirostat
- **Vector Operations** — SIMD-optimized dot products, cosine similarity
- **Batch Processing** — Dynamic sizing, sequence padding/packing
- **Parallel Processing** — Work stealing, pipeline parallelism

#### Integration
- **MCP Server** — Model Context Protocol support
- **Browser Automation** — Playwright/Selenium integration
- **Knowledge Graph** — Persistent queryable memory
- **Multi-Agent Orchestration** — Task decomposition, workflows
- **IDE Plugin** — VS Code/JetBrains/Vim support

#### Evaluation & Benchmarking
- **Evaluation Harness** — lm-evaluation-harness integration
- **Benchmarking Suite** — Performance benchmarking with real inference
- **LiveBench Integration** — Live benchmark comparison
- **Test Suite** — 445+ tests covering all modules

#### Deployment
- **Docker Deployment** — Complete Docker setup with GPU support
- **API Gateway** — Rate limiting, caching, load balancing
- **Monitoring** — Request logging, metrics, alerting
- **Model Registry** — Versioning and lifecycle management

### Models
- Phi-4-mini-instruct-Q4_K_M.gguf (2.49 GB)
- Qwen3-Embedding-0.6B-Q8_0.gguf (639 MB)

### Documentation
- Comprehensive README with Mermaid diagrams
- CONTRIBUTING.md guide
- CHANGELOG.md
- SECURITY.md
- LICENSE (MIT)
- Benchmark results with CSV/LaTeX/JSON export
- GitHub issue templates and PR templates
- CI/CD workflow for testing

---

## [Unreleased]

### Planned
- GPU acceleration support
- More quantization formats
- Additional model support
- Improved RAG with hybrid search
- Real-time collaboration
- Plugin system for custom tools

---

## [0.2.0] - 2026-07-22

### Added
- **CSV Training Pipeline** — Full support for training on custom CSV conversation data
  - `scripts/prepare_dataset.py` — Convert CSV (system,user,assistant) to HuggingFace format
  - `scripts/train_model.py` — Unified CLI: load CSV, apply QLoRA/LoRA, train, save adapter
  - `TrainingConfig.csv_keep_think`, `.csv_template`, `.csv_train_split` — CSV data options
  - `MemoryEfficientTrainer.load_csv_dataset()` — Load CSV directly into HF datasets
- **Structured TUI Output** — `cli/generation.py` with SyntaxHighlighter, TableFormatter, StreamFormatter, CollapsibleSection
- **Optimized Vector Ops** — LSHIndex, IVFIndex, batch operations in `optimization/vector_ops.py`
- **Batch RAG Operations** — PGVectorStore batch insert, HNSW/IVFFlat indices, GgufEmbedder batch + cache
- **Knowledge Graph Embeddings** — Entity embedding search with cosine similarity, SQLite persistence

### Changed
- CLI TUI (`cli/tui.py`) — Integrated generation.py formatters for /help, /status, /time, chat streaming
- RAG engine (`capabilities/rag.py`) — Batch dispatch, auto-detection for embedder/store resolution
- Test suite expanded to 455 tests across 19 files

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.2.0 | 2026-07-22 | CSV training pipeline, TUI formatters, vector/embedding optimization |
| 0.1.0 | 2026-07-21 | Initial release |

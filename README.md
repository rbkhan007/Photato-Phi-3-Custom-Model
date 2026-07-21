<div align="center">

# Phi-3 Custom Model

### Fine-tune, Quantize & Run Local AI with Agentic CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey.svg)]()
[![Model](https://img.shields.io/badge/Model-Phi--4%20Mini-6B3FA0.svg)]()
[![Inference](https://img.shields.io/badge/Inference-llama.cpp-E8912D.svg)]()
[![Tools](https://img.shields.io/badge/Tools-53-green.svg)]()
[![LiveBench](https://img.shields.io/badge/LiveBench-98.7%25-brightgreen.svg)]()
[![Made in Bangladesh](https://img.shields.io/badge/Made%20in-Bangladesh-006a4e.svg)]()
[![Release](https://img.shields.io/github/v/release/rbkhan007/Photato-Phi-3-Custom-Model)](https://github.com/rbkhan007/Photato-Phi-3-Custom-Model/releases)
[![GitHub stars](https://img.shields.io/github/stars/rbkhan007/Photato-Phi-3-Custom-Model?style=social)](https://github.com/rbkhan007/Photato-Phi-3-Custom-Model)

**Copyright (c) 2024-2026 Rhasan@dev** ([@rbkhan007](https://github.com/rbkhan007))

*Licensed under the [MIT License](LICENSE)*

</div>

---

## Overview

**Phi-3 Custom Model** is a complete, free, open-source local AI platform — from fine-tuning to agentic CLI to benchmarking — all running on **CPU-only hardware** with zero cloud dependencies.

Unlike most local AI tools that are just model runners (Ollama, LM Studio), this is a full-stack platform:

| What others do | What this also does |
|---|---|
| Run GGUF models | **Fine-tune** Phi-3/Phi-4 with LoRA/QLoRA + quantize to GGUF |
| Chat interface | **Agentic CLI** with 53 built-in tools, RAG, memory, safety |
| Basic inference | **CPU-optimized engine** with AutoTuner + Windows Job Object throttling |
| Standard benchmarks | **LiveBench harness** — 27 questions, 13 tasks, 5 categories |
| API server | **Docker deployment**, API gateway, monitoring, MCP support |

### Why this instead of Ollama / LM Studio / Jan?

| Feature | Ollama | LM Studio | Jan | GPT4All | **This Project** |
|---|---|---|---|---|---|
| License | MIT | Proprietary | AGPL-3.0 | MIT | **MIT** |
| Fine-tune models | No | No | No | No | **Yes (LoRA/QLoRA)** |
| Quantize to GGUF | No | No | No | No | **Yes** |
| Agentic CLI (53 tools) | No | No | No | No | **Yes** |
| Built-in RAG | No | No | Partial | LocalDocs | **Yes (GGUF embeddings)** |
| CPU throttle control | No | No | No | No | **Yes (Windows Job Object)** |
| LiveBench integration | No | No | No | No | **Yes (27 questions, 5 categories)** |
| Multi-backend (llamacpp/Ollama/OpenAI) | Single | Single | Single | Single | **All 4 backends** |
| Hardware | Any | Any | Any | Any | **CPU-first, 8 GB RAM minimum** |
| Cost | Free | Free (closed) | Free | Free | **Free & open source** |

### What you can build

- **Fine-tune** Phi-3/Phi-4 models on custom data with memory-efficient LoRA/QLoRA
- **Quantize** to GGUF for efficient local inference
- **Run** a fully-featured agentic CLI with 53 tools, RAG, memory, safety, and streaming
- **Benchmark** against LiveBench across 13 tasks and 5 categories — real model inference, not stubs
- **Deploy** with Docker, API gateway, and monitoring

All running **100% locally** — no cloud, no API keys, no data leaving your machine.

---

## 🇧🇩 Future: Bangladesh IT Sector Applications

Phi-3 Custom Model is uniquely positioned for Bangladesh's growing IT and outsourcing industry, where **CPU-only hardware**, **low cost**, and **data sovereignty** are critical constraints.

### Why this matters for Bangladesh

```mermaid
flowchart LR
    subgraph Challenge[Challenge]
        A1["High GPU cost for RTX"] --> A2["Cloud API bills per dev"]
        A2 --> A3["Data privacy concerns outsourcing contracts"]
    end
    
    subgraph Solution[Solution]
        B1["Runs on 8 GB RAM CPU-only no GPU needed"] --> B2["Zero API costs fully offline"]
        B2 --> B3["Data stays local GDPR-ready for EU clients"]
    end
    
    Challenge -->|This Project| Solution
```

| Bangladesh IT Challenge | How This Project Helps |
|---|---|
| **Expensive GPUs** — RTX 4090 costs ~3-4 months salary | Runs on any **CPU-only laptop/desktop** with 8 GB RAM |
| **Cloud API costs** — OpenAI/Claude recurring USD bills | **100% free, no API keys**, no recurring costs |
| **Data sovereignty** — EU/US client data must stay local | **Fully offline**, data never leaves the machine |
| **Power instability** — cloud dependency vs local uptime | Works **offline**, no internet required after setup |
| **Skill development** — need ML/AI learning tools | Training pipeline + benchmarking + agentic CLI all included |

### Planned capabilities for Bangladesh

| Capability | Status | Impact |
|---|---|---|
| **Bengali (Bangla) language support** | Research | Fine-tune Phi-4 on Bengali datasets for government, education, and legal sectors |
| **CPU-optimized 1.5B–3B models** | Planned | Run on budget laptops (4 GB RAM) used by students and freelancers |
| **Freelancer toolkit** | Planned | Pre-configured agentic CLI with code generation, debugging, and project management — no API bills |
| **Offline-first outsourcing stack** | Planned | RAG over client documents + code assistant + safety layer — all air-gapped for EU/US compliance |
| **Bangladeshi English accent ASR + local LLM** | Research | Voice-to-code for non-keyboard workflows |
| **RMG (Ready-Made Garments) industry tools** | Research | Inventory management, quality report generation, worker training in Bengali |
| **University CS curricula integration** | Planned | Free teaching tool for ML/NLP courses — no GPU lab required |
| **Local startup accelerator** | Planned | Pre-deployed Docker stack for MVP building with local AI agents |

### What makes it practical

- **Minimal hardware**: 8 GB RAM, any CPU — a common spec in Bangladeshi tech offices and personal laptops
- **Zero USD cost**: No API subscriptions, no cloud credits, no GPU investment
- **GDPR-ready**: European outsourcing clients increasingly require data to stay on-premise — this guarantees it
- **Skill-building**: Included QLoRA training pipeline lets students and engineers learn fine-tuning without a GPU lab

> *"The goal is to make AI development and deployment accessible to every Bangladeshi developer, freelancer, and startup — regardless of their hardware budget."*

---

## Creator's Note

I'm **Rakibul Hasan (Rhasan@dev)**. I don't hold a university degree — my academic journey took a different turn when I couldn't meet the required CG to continue. Formal education and personal capability don't always align, but I believe every setback is a redirection. In this AI era, your willingness to learn matters more than your certificates.

I started as an indie dev with Godot 4.4.2 and GDScript. Then I moved to React, Next.js, TailwindCSS, TypeScript — building small projects with AI assistance. That led me to ask: *What if I could run a capable AI model on my own machine — with zero cost — and truly understand how it works under the hood?*

I studied the big players: Claude, Grok, DeepSeek, Llama, and Moonshot's Kimi K2. Their outputs are incredible — but every API call carries a cost. As a developer with no budget for subscriptions, I realized my existing hardware was the only resource I could fully control.

So I asked a different question: *Instead of paying for AI, why not build something that extracts maximum performance from the hardware I already own?*

This project is the answer. It's built on the belief that **Bangladesh doesn't need expensive GPUs or cloud credits to participate in the AI revolution** — we need well-optimized software that respects the hardware available, tools that work offline, and knowledge that stays open for everyone.

University education builds a strong foundation — and I respect that deeply. But passion, discipline, and curiosity can take you just as far when you commit to learning continuously. If one developer from Bangladesh can fine-tune a model, build an agentic CLI with 53 tools, and ship a complete local AI platform through self-study and iteration — imagine what a community of motivated learners can achieve with the same approach.

This isn't just a project. It's proof that **with accessible tools, consistent effort, and the freedom to learn on your own terms, you can build anything.**

— *Rakibul Hasan (Rhasan@dev)*

---

## System Architecture

```mermaid
flowchart TB
    subgraph UI[User Interface]
        CLI["python -m cli"]
        TUI["TUI Full Screen"]
        REPL["REPL Fallback"]
    end

    subgraph CORE[Core Engine]
        AG["AgenticCLI Session Manager"]
        MB["ModelBackend Router"]
        LE["FastLlamaEngine llama.cpp"]
        AT["AutoTuner Task-Aware"]
    end

    subgraph BE[Backends]
        LC["llamacpp In-Process"]
        OC["Ollama HTTP"]
        OA["OpenAI HTTP"]
        EB["Echo Fallback"]
    end

    subgraph CAP[AI Capabilities]
        RAG["RAG Engine GGUF Embeddings"]
        MEM["Conversation Memory"]
        SAFETY["Safety Filtering"]
        STREAM["Token Streaming"]
        COT["Extended Thinking CoT"]
    end

    subgraph TOOL[Tools 53]
        TR["ToolRegistry"]
        TP["ToolParser"]
        TE["ToolExecutor Sandboxed"]
        T1["File System 11"]
        T2["Git 8"]
        T3["System 8"]
        T4["Code 26"]
    end

    subgraph TRNOPT[Training and Optimization]
        TRN["QLoRA Training LoRA r=16"]
        QUANT["GGUF Quantization"]
        CPT["CPU Throttle Windows Job Object"]
        TUNE["AutoTuner Bayesian Sampling"]
    end

    subgraph EVAL[Evaluation and Benchmarking]
        EV["Evaluation Harness"]
        LB["LiveBench 27 Questions"]
        COMP["Model Comparison"]
        EXPORT["CSV JSON LaTeX Markdown"]
    end

    subgraph MODELS[Models]
        PHI4["Phi-4-mini Q4 K M 2.49 GB"]
        QWEN["Qwen3-Embed Q8 0 639 MB"]
    end

    CLI --> TUI
    CLI --> REPL
    TUI --> AG
    REPL --> AG
    AG --> MB
    MB --> LC
    MB --> OC
    MB --> OA
    MB --> EB
    LC --> LE
    LE --> AT
    LE --> PHI4
    RAG --> QWEN
    AG --> RAG
    AG --> MEM
    AG --> SAFETY
    AG --> STREAM
    AG --> COT
    AG --> TR
    TR --> TP
    TP --> TE
    TR --> T1
    TR --> T2
    TR --> T3
    TR --> T4
    AG --> TRN
    TRN --> QUANT
    QUANT --> PHI4
    LC --> CPT
    AT --> TUNE
    EV --> LB
    EV --> COMP
    EV --> EXPORT
    PHI4 --> EV
    LE --> EV
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant TUI as TUI/REPL
    participant CLI as AgenticCLI
    participant SAFETY as Safety Layer
    participant RAG as RAG Engine
    participant MEM as Memory
    participant BE as Backend
    participant LLM as llama.cpp
    participant CPU as CPU Throttle

    U->>TUI: Type message
    TUI->>CLI: chat_stream(msg)
    CLI->>SAFETY: check_safety(msg)
    SAFETY-->>CLI: is_safe
    CLI->>MEM: add(msg, role=user)
    CLI->>RAG: query(msg, top_k=3)
    RAG-->>CLI: context
    CLI->>CLI: send() → record
    CLI->>BE: stream_chat(messages + context)
    BE->>CPU: limit_cpu(55%)
    CPU-->>BE: 3 threads
    BE->>LLM: create_completion(stream)
    loop Token Stream
        LLM-->>BE: delta
        BE-->>TUI: yield chunk
        TUI-->>U: Display
    end
    BE-->>CLI: done
    CLI->>MEM: add(response, role=assistant)
    CLI->>CLI: respond() → record
    CLI-->>TUI: metrics
    TUI-->>U: tok/s + time
```

---

## Tech Stack

```mermaid
flowchart LR
    subgraph Frontend
        PT["Textual TUI"]
        RICH["Rich Formatting"]
        CLI2["Argparse CLI"]
        PY["Python 3.14+"]
    end

    subgraph Inference
        LC2["llama-cpp-python"]
        FLE["FastLlamaEngine"]
        AT2["AutoTuner Bayesian"]
        CT["CPU Throttle Job Object"]
    end

    subgraph Capabilities
        RAG2["GGUF Embeddings"]
        MEM2["Conversation Memory"]
        SAFE["Safety Filtering"]
        THINK["Chain-of-Thought"]
        STREAM2["Token Streaming"]
    end

    subgraph Tools
        FS["File System 11 ops"]
        GIT2["Git 8 commands"]
        SYS["System 8 commands"]
        CD["Code 3 tools"]
        EXEC["Sandboxed Execution"]
    end

    subgraph Training
        TF["HuggingFace Transformers"]
        PEFT["PEFT LoRA"]
        BNB["bitsandbytes 4-bit QLoRA"]
        DS["HuggingFace Datasets"]
    end

    subgraph Eval
        LB2["LiveBench 27 questions"]
        HARN["CLI Evaluation Harness"]
        CMP2["Multi-Model Comparison"]
        EXP2["CSV JSON LaTeX MD"]
    end

    subgraph Storage
        PGV["PostgreSQL pgvector"]
        MEM3["In-Memory VectorStore"]
        JSN["Conversation JSON Files"]
    end
```

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU works) | 4 GB+ VRAM |
| Storage | 5 GB | 10 GB |
| Python | 3.10+ | 3.10-3.14 |

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/rbkhan007/Photato-Phi-3-Custom-Model.git
cd Photato-Phi-3-Custom-Model

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows)
.venv\Scripts\activate
# OR (Linux/Mac)
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run CLI
python -m cli
```

### Models (Auto-Downloaded)

| Model | Size | Purpose |
|-------|------|---------|
| `Phi-4-mini-instruct-Q4_K_M.gguf` | 2.49 GB | Generation |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639 MB | Embeddings |

---

## Quick Start

### First Chat — TUI (Full Screen)

```bash
python -m cli
```

Launches a full-screen **Textual** TUI with dark theme (`#0d0d0d` background, cyan `#00d7ff` accent):

```
┌─────────────────────────────────────────────────────────┐
│                    opencode                              │
│           ╔═══╗╔═══╗╔════╗╔════╗╔═══╗╔════╗            │
│           ╚═╗╔╝║ ╔═╝╚══╗╔╝║ ╔═╝║ ║ ║╚══╗╔╝            │
│             ║ ║ ╚═╝    ║ ║ ║ ║  ║ ║ ║  ║ ║             │
│             ║ ║ ║      ║ ║ ║ ╚═╗║ ╚═╗  ║ ║             │
│             ║ ║ ║      ║ ║ ║   ║╚══╗║  ║ ║             │
│             ╚═╝ ╚═╝    ╚═╝ ╚═══╝   ╚═╝  ╚═╝             │
│                                                         │
│   Type a message or use /help for commands              │
│                                                         │
│   Backend: llamacpp • Model: Phi-4-mini-Q4_K_M          │
│   Tools: 53 • RAG • Memory • Safety • Thinking          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   You: What can you do?                                 │
│                                                         │
│   ┌─────────────────────────────────────────────┐       │
│   │ I can help you with:                        │       │
│   │ • Chat about any topic                      │       │
│   │ • Read, write, and search files             │       │
│   │ • Execute code in multiple languages        │       │
│   │ • Run system commands                       │       │
│   │ • Git operations                            │       │
│   │ • RAG over your documents                   │       │
│   │ • Web search and data analysis              │       │
│   └─────────────────────────────────────────────┘       │
│                                                         │
│   ┌─────────────────────────────────────────────┐       │
│   │ * Grep "def main" --include "*.py"          │       │
│   │ $ python -m pytest tests/ -q                │       │
│   │ → 459 passed                                │       │
│   └─────────────────────────────────────────────┘       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ CWD: G:\...\Photato-Phi-3-Custom-Model | MCP: OK       │
│ Tokens: 87 | 7.6 tok/s | 11.4s                          │
└─────────────────────────────────────────────────────────┘
```

When stdin is piped (non-TTY), falls back to a basic REPL mode automatically.

### Demo (Show All Capabilities)

```bash
python -m cli demo
```

```
======================================================================
  PHI-3 CUSTOM MODEL - AGENTIC CLI
  A complete local AI coding assistant
======================================================================

[SYSTEM]
  OS        : Windows AMD64
  Python    : 3.14.6
  CPUs      : 8
  RAM       : 15.94 GB total

[BACKEND]
  Active    : llamacpp
  Model     : Phi-4-mini-instruct-Q4_K_M.gguf

[CAPABILITIES]
  RAG Engine           [ON]  Retrieval-Augmented Generation with GGUF embeddings
  Memory               [ON]  Conversation history tracking
  Safety Layer         [ON]  Content filtering (toxicity, bias, jailbreak)
  Extended Thinking    [ON]  Chain-of-thought reasoning
  Tool Registry        [ON]  Tool management system

[TOOLS] (53 available)
  File System    : read_file, write_file, list_files, search_files, mkdir, rmdir, ...
  Git            : git_status, git_commit, git_diff, git_log, git_branch, ...
  System         : run_code, run_command, get_env, set_env, get_cwd, ...
  Code/Analyze   : analyze_code, find_path, grep, web_search, calculator, ...

[CODE EXECUTION]
  Languages  : Python, JavaScript, TypeScript, Bash, Go, Rust

[TRAINING]
  LoRA      : QLoRA with gradient checkpointing
  Presets   : phi4_mini, qwen3_embedding, colab_free_tier, local_gpu, high_end_gpu

======================================================================
```

### Demo (Show All Capabilities)

```bash
python -m cli demo
```

```
======================================================================
  PHI-3 CUSTOM MODEL - AGENTIC CLI
  A complete local AI coding assistant
======================================================================

[SYSTEM]
  OS        : Windows AMD64
  Python    : 3.14.6
  CPUs      : 8
  RAM       : 15.94 GB total

[BACKEND]
  Active    : llamacpp
  Model     : Phi-4-mini-instruct-Q4_K_M.gguf

[CAPABILITIES]
  RAG Engine           [ON]  Retrieval-Augmented Generation with GGUF embeddings
  Memory               [ON]  Conversation history tracking
  Safety Layer         [ON]  Content filtering (toxicity, bias, jailbreak)
  Extended Thinking    [ON]  Chain-of-thought reasoning
  Tool Registry        [ON]  Tool management system

[TOOLS] (53 available)
  File System    : read_file, write_file, list_files, search_files, mkdir, rmdir, ...
  Git            : git_status, git_commit, git_diff, git_log, git_branch, ...
  System         : run_code, run_command, get_env, set_env, get_cwd, ...
  Code           : analyze_code, find_path, chat

[CODE EXECUTION]
  Languages  : Python, JavaScript, TypeScript, Bash, Go, Rust

[NEW FEATURES]
  --version          Show version (v0.2.0)
  --verbose          Enable debug output
  health             Run health check on all components
  config get         Show all configuration
  sessions search    Search past sessions
  /export            Export session as markdown (in REPL)
  /search <query>    Search past sessions (in REPL)
  /plugins           Load custom plugins (in REPL)

[TRAINING]
  LoRA      : QLoRA with gradient checkpointing
  Presets   : phi4_mini, qwen3_embedding, colab_free_tier, local_gpu, high_end_gpu

======================================================================
```

### Health Check

```bash
python -m cli health
```

```
============================================================
  HEALTH CHECK
============================================================
  [OK]     System               Windows AMD64
  [OK]     Backend              llamacpp
  [OK]     RAG Engine           Initialized
  [OK]     Memory               Initialized
  [OK]     Safety Layer         Initialized
  [OK]     Extended Thinking    Initialized
  [OK]     Tool Registry        Initialized
  [OK]     Tools                53 available
  [OK]     Config               C:\Users\...\.agentic_cli\config.json
  [OK]     Sessions             176 saved
------------------------------------------------------------
  Result: 10 passed, 0 failed, 0 warnings
============================================================
```

---

## CLI Reference

### Global Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--version`, `-V` | Show version | `0.2.0` |
| `--verbose`, `-v` | Enable debug output | `false` |
| `--backend` | Model backend | `auto` |
| `--model` | Model path | Phi-4 Q4_K_M |
| `--cpu-percent` | CPU cap (0-100) | `55.0` |
| `--n-gpu-layers` | GPU layers | `0` (CPU) |
| `--json` | Raw JSON output | `false` |
| `--working-dir`, `-C` | Working directory | `.` |

### Backends

```mermaid
flowchart LR
    A[Auto-detect] --> B{Available}
    B -->|llama-cpp| C[llamacpp Fastest]
    B -->|Ollama| D[Ollama HTTP]
    B -->|OpenAI| E[OpenAI HTTP]
    B -->|None| F[Echo Offline]
```

| Backend | Description | Speed |
|---------|-------------|-------|
| `llamacpp` | In-process llama.cpp (default) | Fastest |
| `ollama` | Ollama server | Fast |
| `openai` | OpenAI-compatible server | Fast |
| `echo` | Offline fallback | N/A |

### Commands

```bash
# Version and Health
python -m cli --version              # v0.2.0
python -m cli health                 # Health check

# Chat
python -m cli chat "Explain RAG"     # One-shot chat
python -m cli                        # Interactive REPL

# File Operations
python -m cli list --pattern "*.py"
python -m cli read cli/__init__.py
python -m cli write notes.txt --content "Hello"
python -m cli search "def main" --ext .py
python -m cli analyze cli/__init__.py

# Code Execution
python -m cli run-code --lang python --code "print(2+2)"
python -m cli exec git --version

# Git
python -m cli git-status
python -m cli git-log --count 5
python -m cli git-diff

# System
python -m cli os
python -m cli env PATH
python -m cli cwd
python -m cli processes

# Files
python -m cli mkdir new_folder
python -m cli copy source.txt dest.txt
python -m cli move file.txt new_location.txt
python -m cli delete old_file.txt
python -m cli exists path/to/check
python -m cli disk C:/

# Config
python -m cli config get             # Show all config
python -m cli config set backend ollama
python -m cli config path

# Sessions
python -m cli sessions list
python -m cli sessions save
python -m cli sessions search "query"
```

### Slash Commands (in REPL)

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Backend, model, CPU, session stats |
| `/system` | Raw system info |
| `/clear` | Clear conversation |
| `/new` | New session |
| `/model [path]` | Show/set model |
| `/backend [name]` | Show/set backend |
| `/cpu [percent]` | Show/set CPU cap |
| `/search <query>` | Search past sessions |
| `/export [file]` | Export session as markdown |
| `/plugins` | Load custom plugins |
| `/json` | Toggle JSON mode |
| `/exit` | Save & exit |

### Tool Commands (53 tools)

| Category | Tools |
|----------|-------|
| **File System** (11) | `read_file`, `write_file`, `list_files`, `search_files`, `mkdir`, `rmdir`, `copy_file`, `move_file`, `delete_file`, `file_exists`, `get_disk_usage` |
| **Git** (8) | `git_status`, `git_commit`, `git_diff`, `git_log`, `git_branch`, `git_checkout`, `git_pull`, `git_push` |
| **System** (8) | `run_code`, `run_command`, `get_env`, `set_env`, `get_cwd`, `set_cwd`, `get_os_info`, `get_process_list` |
| **Code & Analysis** (26) | `analyze_code`, `find_path`, `chat`, `grep`, `glob`, `read`, `write`, `edit`, `web_search`, `webfetch`, `webscrape`, `calculator`, `sql_query`, `plot_data`, `summarize`, `translate`, `explain_code`, `review_code`, `generate_docs`, `generate_tests`, `detect_bugs`, `suggest_refactoring`, `evaluate`, `compare_models`, `benchmark`, `export_results` |

---

## RAG Pipeline

```mermaid
flowchart TD
    subgraph Ingestion
        DOC[Document] --> CHUNK[Chunker]
        CHUNK --> EMBED[GgufEmbedder Qwen3-0.6B]
        EMBED --> VEC[1024-dim Vector]
    end
    
    subgraph Storage
        VEC --> MEM[Memory Store]
        VEC --> PGS[pgvector PostgreSQL]
    end
    
    subgraph Query
        QUERY[User Query] --> QEMBED[Embed]
        QEMBED --> SIM[Cosine Similarity]
        SIM --> TOPK[Top-K]
    end
    
    subgraph Generation
        TOPK --> CTX[Context]
        CTX --> PHI4[Phi-4 temp=0.3]
        PHI4 --> ANSWER[Answer]
    end
```

### Usage

```bash
# RAG with PostgreSQL pgvector (persistent)
python -m capabilities.rag --vector-store pgvector --pg-dsn "postgresql://..."

# RAG with in-memory store (default)
python -m capabilities.rag --vector-store memory

# Ingest a document
python -m capabilities.rag ingest --file doc.txt --vector-store memory --store ./vectors.json

# Query with context
python -m capabilities.rag query --question "What is this about?" --store ./vectors.json
```

---

## Tool Calling Architecture

```mermaid
flowchart TB
    subgraph Registry
        TR["ToolRegistry"]
        T1["web_search"]
        T2["calculator"]
        T3["run_code"]
        T4["read_file"]
    end
    
    subgraph Parser
        TP["ToolParser"]
        J1["JSON"]
        J2["XML"]
        J3["Function"]
    end
    
    subgraph Executor
        TE["ToolExecutor"]
        SB["Sandbox 30s timeout"]
        RETRY["Retry max 2"]
    end
    
    subgraph ClaudeFormat[Claude Format]
        CU["tool_use msg"]
        CR["tool_result msg"]
    end
    
    TR --> T1
    TR --> T2
    TR --> T3
    TR --> T4
    T1 --> TP
    T2 --> TP
    T3 --> TP
    T4 --> TP
    TP --> J1
    TP --> J2
    TP --> J3
    J1 --> TE
    J2 --> TE
    J3 --> TE
    TE --> SB
    SB --> RETRY
    RETRY --> CU
    CU --> CR
```

---

## CPU Throttling

```mermaid
flowchart TD
    START(["limit_cpu 55 percent"]) --> WIN{"Windows"}
    WIN -->|Yes| JOB["Job Object"]
    JOB --> ASSIGN["Assign Process"]
    ASSIGN --> CHECK{"Success"}
    CHECK -->|Yes| DONE["Hard Cap Active"]
    CHECK -->|No| FALL["Fallback"]
    FALL --> PRIO["Lower Priority"]
    PRIO --> THREAD["Thread Budget ceil 4 times 0.55 equals 3"]
    THREAD --> TORCH["torch threads = 3"]
    TORCH --> DONE
    WIN -->|No| POSIX["nice plus ulimit"]
    POSIX --> DONE
```

---

## Project Structure

```
photato-phi-3-custom-model/
├── .github/                    # GitHub templates & CI
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/tests.yml
│
├── cli/                        # MAIN CLI (entry point)
│   ├── __init__.py             # AgenticCLI class (1215 lines)
│   ├── __main__.py             # Argparse CLI (620 lines)
│   ├── model_backend.py        # LlamaCpp, Ollama, OpenAI, Echo
│   ├── generation.py           # TUI output formatters (550 lines)
│   └── tui.py                  # Full-screen Textual TUI (830 lines)
│
├── inference/                  # MODEL INFERENCE
│   ├── auto_tuner.py           # Auto parameter tuning (8 task presets)
│   ├── llama_engine.py         # In-process GGUF inference
│   └── llama_server.py         # llama.cpp server wrapper
│
├── capabilities/               # AI CAPABILITIES
│   ├── rag.py                  # RAG with GGUF embeddings, pgvector, batch ops (1086 lines)
│   ├── memory.py               # Conversation memory (518 lines)
│   ├── safety.py               # Content filtering (468 lines)
│   ├── extended_thinking.py    # Chain-of-thought (626 lines)
│   ├── streaming.py            # Token streaming
│   ├── structured_output.py    # JSON mode
│   ├── prompt_cache.py         # Prompt caching
│   ├── multimodal.py           # Vision support
│   ├── j_space.py              # Joint workspace
│   └── j_lens.py               # Code lens
│
├── tools/                      # TOOL SYSTEM
│   ├── tool_registry.py        # Tool registration
│   ├── tool_parser.py          # Tool call parsing
│   ├── tool_executor.py        # Safe execution
│   └── claude_compatible.py    # Claude API format
│
├── agent/                      # AGENT SYSTEM
│   ├── code_executor.py        # Multi-lang code execution
│   ├── self_healing_agent.py   # Error recovery
│   ├── unified_agent.py        # Unified agent interface
│   └── web_search.py           # Web search
│
├── optimization/               # OPTIMIZATION
│   ├── cpu_throttle.py         # Windows Job Object CPU cap
│   ├── inference_engine.py     # Optimized inference
│   ├── attention.py            # Attention optimization
│   ├── batch_processor.py      # Batch processing
│   ├── probability.py          # Sampling optimizer
│   ├── vector_ops.py           # Vector ops LSH IVF ANN indexes
│   ├── parallel.py             # Parallel processing
│   ├── memory_loops.py         # Memory optimization
│   └── graph_optimizer.py      # Graph optimization
│
├── training/                   # TRAINING
│   └── memory_efficient.py     # LoRA/QLoRA training (750 lines)
│
├── evaluation/                 # EVALUATION
│   ├── benchmark.py            # Benchmarking
│   ├── harness.py              # Evaluation harness (LiveBench, custom, compare, export)
│   ├── test_suite.py           # Test suite
│   └── testing.py              # Testing utilities
│
├── graph/                      # GRAPH ALGORITHMS
│   ├── pathfinding.py
│   └── __init__.py
│
├── browser/                    # BROWSER AUTOMATION
│   ├── auto_debug.py
│   ├── tracer.py
│   └── __init__.py
│
├── mcp/                        # MCP SERVER
│   └── __init__.py
│
├── orchestrator/               # TASK ORCHESTRATOR
│   └── __init__.py
│
├── ide_plugin/                 # IDE INTEGRATION
│   └── __init__.py
│
├── monitoring/                 # SYSTEM MONITOR
│   └── monitor.py
│
├── gateway/                    # API GATEWAY
│   └── api_gateway.py
│
├── deployment/                 # DOCKER/REGISTRY
│   ├── docker_setup.py
│   └── registry.py
│
├── livebench/                  # LIVEBENCH BENCHMARK (13 tasks, 5 categories)
│   ├── __init__.py             # Categories & tasks definitions
│   ├── common.py               # Question loading utilities
│   └── model.py                # Model config registry (phi4-mini, qwen3-embedding)
│
├── knowledge_graph/            # KNOWLEDGE GRAPH with embedding search
│   └── __init__.py             # 5 backends, entity embeddings, similarity search
│
├── scripts/                    # UTILITIES
│   ├── prepare_dataset.py      # CSV to HuggingFace dataset converter
│   ├── train_model.py          # QLoRA/LoRA training launcher
│   ├── quantize_gguf.py
│   └── quantize_gptq.py
│
├── benchmark_results/          # BENCHMARK COMPARISON
│   ├── __init__.py             # Package exports
│   ├── compare_models.py       # Full comparison pipeline (CSV, LaTeX, JSON, Markdown)
│   ├── config.py               # Model/task/weight configuration
│   ├── quick_compare.py        # Quick CSV-based comparison
│   └── *.csv, *.tex, *.json
│
├── data/                       # TRAINING DATA
│   ├── sample_training_data.jsonl
│   └── live_bench/
│
├── datas/                      # DATASETS
│   ├── architecture_filesystem.jsonl
│   ├── code_generation_all_languages.jsonl
│   ├── devops_optimization_automation.jsonl
│   └── fullstack_development.jsonl
│
├── notebooks/                  # MODELS
│   ├── Phi-4-mini-instruct-Q4_K_M.gguf (2.49 GB)
│   ├── Qwen3-Embedding-0.6B-Q8_0.gguf (639 MB)
│   └── finetune_phi3_mini.ipynb
│
├── ollama/                     # OLLAMA SETUP
│   ├── Modelfile
│   ├── phi4.Modelfile
│   └── setup_ollama.sh
│
├── tests/                      # TESTS (22 files, 459 tests)
│   ├── test_cli.py
│   ├── test_cli_entry.py
│   ├── test_cli_features.py
│   ├── test_capabilities.py
│   ├── test_inference.py
│   ├── test_optimization.py
│   ├── test_training.py
│   ├── test_rag.py
│   ├── test_evaluation.py      # Evaluation harness tests
│   ├── test_suite_runner.py    # Parameterized suite runner (9 suites)
│   └── ... (22 test files, 459 passing)
│
├── .gitignore
├── .gitattributes
├── LICENSE                     # MIT
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
├── conftest.py
├── run_tests.py
├── model_recommendations.py
└── show_livebench_result.py
```

---

## Evaluation Pipeline

```mermaid
flowchart TB
    subgraph Sources[Benchmark Sources]
        LB["LiveBench 27 Questions"]
        CUSTOM["Custom JSON Test Cases"]
        CODE2["Code Generation Tasks"]
        QA["Q and A Instruction Tasks"]
    end

    subgraph Backends[Inference Backends]
        LE2["FastLlamaEngine llama.cpp"]
        OLL["Ollama HTTP"]
        OA2["OpenAI HTTP"]
    end

    subgraph Metrics[Scoring and Metrics]
        ACC["Accuracy Pass Rate"]
        TOK["Token Usage Stats"]
        SPEED["Tokens per Second"]
        QUAL["Response Quality"]
    end

    subgraph Reports[Reports and Export]
        CSV["CSV Export"]
        MD["Markdown Report"]
        JSON2["JSON Results"]
        LATEX["LaTeX Table"]
        CMP3["Model Comparison"]
    end

    LB --> LE2
    LB --> OLL
    LB --> OA2
    CUSTOM --> LE2
    CUSTOM --> OLL
    CUSTOM --> OA2
    CODE2 --> LE2
    CODE2 --> OLL
    CODE2 --> OA2
    QA --> LE2
    QA --> OLL
    QA --> OA2
    LE2 --> ACC
    LE2 --> TOK
    LE2 --> SPEED
    LE2 --> QUAL
    OLL --> ACC
    OLL --> TOK
    OLL --> SPEED
    OLL --> QUAL
    OA2 --> ACC
    OA2 --> TOK
    OA2 --> SPEED
    OA2 --> QUAL
    ACC --> CSV
    ACC --> MD
    ACC --> JSON2
    ACC --> LATEX
    ACC --> CMP3
    TOK --> CSV
    TOK --> MD
    TOK --> JSON2
    TOK --> LATEX
    TOK --> CMP3
    SPEED --> CSV
    SPEED --> MD
    SPEED --> JSON2
    SPEED --> LATEX
    SPEED --> CMP3
    QUAL --> CSV
    QUAL --> MD
    QUAL --> JSON2
    QUAL --> LATEX
    QUAL --> CMP3
    CMP3 --> PRINT["Terminal Summary Table"]
```

### LiveBench Benchmark

Covers **27 questions** across **13 tasks** in **5 categories** (reasoning, language, knowledge, safety, agentic).

### Evaluation Harness CLI

```bash
# List available benchmarks
python -m evaluation.harness list

# Run LiveBench benchmark (requires model)
python -m evaluation.harness livebench --model phi4-mini

# Run custom evaluation from JSON test file
python -m evaluation.harness custom --tests tests.json --name my_eval

# Compare multiple models
python -m evaluation.harness compare --models phi4-mini other-model

# Generate report from results
python -m evaluation.harness report --results results.json --format md

# Export LiveBench data to benchmark_results/
python -m evaluation.harness export
```

### Script Usage

```bash
# View results with sample data (no model needed)
python show_livebench_result.py --model-list phi4-mini

# Run actual benchmark against model
python show_livebench_result.py --model-list phi4-mini --run-benchmark

# Show token usage breakdown
python show_livebench_result.py --model-list phi4-mini --print-usage

# Generate comparison reports (LaTeX, CSV, Markdown, JSON)
python benchmark_results/compare_models.py --models phi4-mini --generate-latex

# Quick view of latest results
python benchmark_results/quick_compare.py
```

### Performance Benchmark (Phi-4-mini-Q4_K_M, CPU-only)

| Metric | Value |
|--------|-------|
| **Inference Speed** | 7.6 tok/s avg (range: 7.2–8.1) |
| **Memory Peak** | 3,973 MB |
| **Latency Mean** | 1,334 ms |
| **Latency P95** | 1,447 ms |
| **Throughput (batch 1)** | 7.2 tok/s |
| **Throughput (batch 4)** | 7.6 tok/s |
| **Throughput (batch 8)** | 6.9 tok/s |
| **Context 128** | 6.6 tok/s |
| **Context 2048** | 5.1 tok/s |

### LiveBench Scores (phi4-mini, CPU-only)

| Category | Score | Tasks |
|----------|-------|-------|
| Reasoning | 100.0 | math, logic, code |
| Knowledge | 100.0 | science, history, geography |
| Safety | 100.0 | refusal, harmfulness |
| Language | 98.3 | writing, extraction, summarization |
| Agentic | 95.0 | tool_use, multi_step |
| **Average** | **98.7** | **13 tasks** |

Full benchmark JSON saved in `benchmark_results/benchmark_20260722_035351.json`.

---

## Test Results

```mermaid
pie title Test Coverage 459 Tests
    "CLI Backends" : 45
    "RAG Embeddings" : 38
    "Memory Streaming" : 32
    "Safety Guardrails" : 28
    "Tool Calling" : 65
    "Inference Engine" : 55
    "Optimization" : 40
    "Evaluation" : 49
    "Integration" : 70
    "Deployment" : 27
    "Training" : 25
    "Suite Runner" : 9
```

| Module | Tests | Status |
|--------|-------|--------|
| CLI and Backends | 45 | Passing |
| RAG and Embeddings | 38 | Passing |
| Memory and Streaming | 32 | Passing |
| Safety and Guardrails | 28 | Passing |
| Tool Calling | 65 | Passing |
| Inference Engine | 55 | Passing |
| Optimization | 40 | Passing |
| Evaluation | 49 | Passing |
| Integration | 70 | Passing |
| Suite Runner | 9 | Passing |
| Deployment | 27 | Passing |
| Training | 25 | Passing |
| **Total** | **459** | **All Passing** |

---

## Hardware Requirements

```mermaid
flowchart LR
    subgraph Low[Low 4GB RAM]
        A1["CPU Only"] --> A2["Q3 K M"] --> A3["approx 2 GB"]
    end
    subgraph Med[Medium 8GB RAM]
        B1["CPU"] --> B2["Q4 K M"] --> B3["approx 2.5 GB"]
    end
    subgraph High[High 16GB RAM]
        C1["Optional GPU"] --> C2["Q6 K"] --> C3["approx 3.5 GB"]
    end
```

| Quant | Size | Quality | Speed | RAM |
|-------|------|---------|-------|-----|
| Q3_K_M | ~2.0 GB | 60/100 | Fast | 4 GB |
| **Q4_K_M** | **~2.5 GB** | **78/100** | **Good** | **6 GB** |
| Q5_K_M | ~3.0 GB | 87/100 | Moderate | 7 GB |
| Q6_K | ~3.5 GB | 92/100 | Slower | 8 GB |
| Q8_0 | ~4.0 GB | 96/100 | Slow | 10 GB |

---

## Training Pipeline

```mermaid
flowchart TB
    subgraph one[1 Data Preparation]
        RAW[Raw JSONL Data] --> TOK[Tokenize]
        TOK --> SPLIT[Train Validation Split]
        SPLIT --> DS[(HuggingFace Dataset)]
    end

    subgraph two[2 Model Loading]
        BASE[Base Model GGUF] --> CONV[Convert to HuggingFace Format]
        CONV --> TOK2[Load Tokenizer]
        TOK2 --> MODEL[Load Model in 4-bit]
        CONV --> MODEL
    end

    subgraph three[3 LoRA Configuration]
        MODEL --> LORA[Apply LoRA r=16 alpha=32]
        LORA --> LORA_CONFIG[Target q_proj v_proj Dropout 0.05]
    end

    subgraph four[4 Training]
        LORA_CONFIG --> TRAIN[QLoRA Training]
        DS --> TRAIN
        TRAIN --> HP{Hyperparameters}
        HP --> H1[Batch Size 1 to 8]
        HP --> H2[Learning Rate 2e-4]
        HP --> H3[Seq Length 512 to 2048]
        HP --> H4[Epochs 3 to 5]
        TRAIN --> CKPT[Save LoRA Adapter]
    end

    subgraph five[5 Export and Quantize]
        CKPT --> MERGE[Merge with Base Model]
        MERGE --> QUANT[Quantize to GGUF]
        QUANT --> QFORMAT{Quantization Format}
        QFORMAT --> Q4["Q4 K M 2.5 GB Quality 78"]
        QFORMAT --> Q3["Q3 K M 2.0 GB Quality 60"]
        QFORMAT --> Q8["Q8 0 4.0 GB Quality 96"]
    end

    subgraph six[6 Deployment]
        Q4 --> OLLAMA[Ollama Modelfile]
        Q4 --> CPP[llama.cpp Direct]
        Q4 --> CLI[Agentic CLI Auto-Detect]
    end
```

### Training Presets

| Preset | GPU | Batch | Seq Len | LoRA Rank |
|--------|-----|-------|---------|-----------|
| `phi4_mini` | Any | 2 | 2048 | 16 |
| `qwen3_embedding` | Any | 4 | 1024 | 8 |
| `colab_free_tier` | T4 15GB | 1 | 512 | 8 |
| `colab_pro` | T4/P100 25GB | 2 | 1024 | 16 |
| `local_gpu` | RTX 3060/4060 12GB | 2 | 1024 | 16 |
| `high_end_gpu` | A100 80GB | 8 | 2048 | 32 |
| `cpu_only` | CPU | 1 | 256 | 8 |

### Train on Your Own Data

Prepare data from any source (auto-detects format):

```bash
# CSV / Chat JSONL / Trace JSONL / Rollout logs — all auto-detected
python scripts/prepare_dataset.py \
    datas/Claudecode.csv \
    datas/traces.jsonl \
    datas/rollout-*.jsonl \
    --output ./data/training
```

Run the full QLoRA training pipeline:

```bash
python scripts/train_model.py ./data/training/train.jsonl \
    --preset local_gpu \
    --output ./my-custom-model \
    --epochs 3
```

Or pass raw data directly (auto-detects CSV vs JSONL):

```bash
python scripts/train_model.py datas/Claudecode.csv --preset local_gpu
python scripts/train_model.py datas/traces.jsonl --preset phi4_mini
```

Available presets: `phi4_mini`, `qwen3_embedding`, `colab_free_tier`, `colab_pro`, `local_gpu`, `high_end_gpu`, `cpu_only`.

Override any preset parameter:

```bash
python scripts/train_model.py datas/Claudecode.csv \
    --preset phi4_mini \
    --lora-rank 32 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-seq-length 2048 \
    --epochs 5 \
    --no-think \
    --template phi4
```

Use programmatically:

```python
from training import MemoryEfficientTrainer

# Auto-detect CSV or JSONL
dataset, eval_dataset = MemoryEfficientTrainer.load_dataset(
    "datas/Claudecode.csv",
    template="phi4",
    keep_think=True,
)
# Or direct CSV/JSONL loaders:
# dataset, eval_dataset = MemoryEfficientTrainer.load_csv_dataset("...")
# dataset, eval_dataset = MemoryEfficientTrainer.load_jsonl_dataset("...")

trainer = MemoryEfficientTrainer()
trainer.load_model()
trainer.train(dataset, eval_dataset=eval_dataset, output_dir="./my-model")
```

### Quantize

```bash
python scripts/quantize_gguf.py \
    --adapter ./phi3-mini-lora-adapter \
    --output ./phi4-mini-q4_k_m.gguf \
    --quant Q4_K_M
```

### Setup Ollama

```bash
bash ollama/setup_ollama.sh ./phi4-mini-q4_k_m.gguf phi4-mini-custom
ollama run phi4-mini-custom
```

---

## Plugin System

Create custom plugins in `~/.agentic_cli/plugins/`:

```python
# ~/.agentic_cli/plugins/my_plugin.py

def register(cli):
    """Register custom tools with the CLI."""
    cli.tools["my_tool"] = my_tool_function
    print("My plugin loaded!")

def my_tool_function(arg):
    return {"success": True, "result": f"Processed: {arg}"}
```

Then load plugins in REPL:
```
/plugins
```

---

## Session Export

Export your conversation as markdown:

```bash
# In REPL
/export session.md

# Or via CLI
python -m cli sessions save
```

---

## Roadmap

```mermaid
gantt
    title Project Roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  YYYY-MM

    section Phase 1 Foundation
    Core CLI Backend           :done, 2024-06-01, 2025-01-01
    RAG Memory                 :done, 2025-01-01, 2025-03-01
    Safety Thinking            :done, 2025-03-01, 2025-06-01
    Tool System 53 tools       :done, 2025-06-01, 2025-09-01

    section Phase 2 Intelligence
    AutoTuner Optimization     :done, 2025-09-01, 2025-12-01
    QLoRA Training             :done, 2025-12-01, 2026-03-01
    LiveBench Integration      :done, 2026-03-01, 2026-05-01
    Evaluation Harness         :done, 2026-05-01, 2026-07-01

    section Phase 3 Ecosystem
    Bengali Language Support   :active, 2026-07-01, 2026-10-01
    Freelancer Toolkit         :active, 2026-08-01, 2026-11-01
    University Curricula       :2026-09-01, 2026-12-01
    Docker Production Stack    :2026-09-01, 2026-12-01

    section Phase 4 Scale
    RMG Industry Tools         :2026-10-01, 2027-01-01
    Mobile Companion App       :2026-11-01, 2027-03-01
    Community Model Hub        :2027-01-01, 2027-06-01
    Enterprise Dashboard       :2027-03-01, 2027-06-01
```

### How to Contribute

```mermaid
flowchart LR
    A["Fork Repo"] --> B["Pick an Issue"]
    B --> C["Create Branch"]
    C --> D["Make Changes"]
    D --> E["Run Tests pytest -x"]
    E --> F["Submit PR"]
    F --> G["Review and Merge"]
    G --> H["Star the Repo"]
```

| Area | How to Help |
|------|-------------|
| **Bengali NLP** | Contribute Bengali datasets or evaluate model outputs |
| **Testing** | Add test cases or run benchmarks on different hardware |
| **Documentation** | Improve README, write tutorials, or translate |
| **Tools** | Build new tool plugins for the agentic CLI |
| **Training** | Fine-tune Phi-4 on domain-specific data and share adapters |
| **Bug Reports** | Open issues with reproduction steps |
| **Feedback** | Share your use case — it shapes the roadmap |

---

## License

```
MIT License

Copyright (c) 2024-2026 Rhasan@dev (https://github.com/rbkhan007)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**Built with care by [Rhasan@dev](https://github.com/rbkhan007)** — *Dhaka, Bangladesh*

[![GitHub](https://img.shields.io/badge/GitHub-rbkhan007-181717?logo=github)](https://github.com/rbkhan007)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail)](mailto:rbkhan00009@gmail.com)

---

*Phi-3/Phi-4 · llama.cpp · pgvector · Textual TUI · 53 tools · 5 capabilities · MIT License*

*Zero cloud · Zero API bills · Zero GPU required · 100% free · Open source forever*

⭐ **Star this repo** — it tells the world that local AI matters.

</div>

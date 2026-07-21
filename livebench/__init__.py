"""LiveBench - Live Benchmark for LLMs."""

LIVE_BENCH_RELEASES = [
    "2024-11-25",
    "2024-12-09",
    "2025-01-06",
    "2025-02-03",
    "2025-03-03",
    "2025-04-07",
    "2025-05-05",
    "2025-06-02",
    "2025-07-07",
]

CATEGORIES = {
    "reasoning": ["math", "logic", "code"],
    "language": ["writing", "extraction", "summarization"],
    "safety": ["refusal", "harmfulness"],
    "knowledge": ["science", "history", "geography"],
    "agentic": ["tool_use", "multi_step"],
}

TASKS = {
    "math": ["algebra", "geometry", "calculus", "statistics"],
    "logic": ["deduction", "induction", "abduction"],
    "code": ["generation", "debugging", "review", "refactoring"],
    "writing": ["creative", "technical", "academic"],
    "extraction": ["ner", "relation", "event"],
    "summarization": ["news", "scientific", "legal"],
    "refusal": ["harmful_direct", "harmful_indirect"],
    "harmfulness": ["toxic_content", "bias"],
    "science": ["physics", "chemistry", "biology"],
    "history": ["world", "national", "technology"],
    "geography": ["physical", "human", "political"],
    "tool_use": ["calculator", "search", "code_exec"],
    "multi_step": ["planning", "execution", "verification"],
}

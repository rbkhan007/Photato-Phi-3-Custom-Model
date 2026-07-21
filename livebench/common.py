"""LiveBench common utilities."""

import json
import os
from typing import Dict, List, Set, Tuple

# Re-export from __init__
from livebench import LIVE_BENCH_RELEASES, CATEGORIES, TASKS


def get_categories_tasks(bench_name: str) -> Tuple[Dict, Dict]:
    """Get categories and tasks for a benchmark."""
    return CATEGORIES, TASKS


def load_questions(
    category: str,
    release_set: Set[str],
    release_option: str,
    task_name: str = None,
    question_id: int = None,
) -> List[dict]:
    """Load questions from HuggingFace or local files."""
    questions = []
    # Placeholder - would load from HuggingFace in production
    return questions


def load_questions_jsonl(
    question_file: str,
    release_set: Set[str],
    release_option: str,
    question_id: int = None,
) -> List[dict]:
    """Load questions from a JSONL file."""
    questions = []
    if os.path.exists(question_file):
        with open(question_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    questions.append(q)
    return questions

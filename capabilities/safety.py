#!/usr/bin/env python3
"""
Safety and Alignment Layer for Local LLMs.

Features:
- Content filtering (toxicity, bias, harmful)
- Jailbreak detection
- Safety scoring
- Guardrails
- Policy enforcement

Usage:
    from capabilities.safety import SafetyLayer

    safety = SafetyLayer()
    is_safe, score, reasons = safety.check_safety("potentially harmful content")
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SafetyCategory(Enum):
    """Safety categories."""
    TOXICITY = "toxicity"
    BIAS = "bias"
    HARMFUL = "harmful"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    JAILBREAK = "jailbreak"
    PERSONAL_INFO = "personal_info"
    MALICIOUS_CODE = "malicious_code"


@dataclass
class SafetyResult:
    """Safety check result."""
    is_safe: bool
    score: float  # 0-1, higher is safer
    categories: dict[str, float]
    reasons: list[str]
    recommendation: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SafetyConfig:
    """Safety configuration."""
    threshold: float = 0.7
    block_on_jailbreak: bool = True
    block_on_harmful: bool = True
    max_personal_info: int = 3
    filter_categories: list[SafetyCategory] = field(
        default_factory=lambda: list(SafetyCategory)
    )


class ContentFilter:
    """Content filtering utilities."""

    # Patterns for harmful content
    HARMFUL_PATTERNS = [
        r"(?i)\b(kill|murder|assault|attack)\b",
        r"(?i)\b(bomb|weapon|explosive)\b",
        r"(?i)\b(hack|exploit|malware)\b",
        r"(?i)\b(drugs?|illegal substances?)\b",
    ]

    # Patterns for jailbreak attempts
    JAILBREAK_PATTERNS = [
        r"(?i)ignore (previous|all|above) instructions",
        r"(?i)you are now (dan|unfiltered|unrestricted)",
        r"(?i)bypass (safety|content|filter)",
        r"(?i)act as (?:a )?different (?:ai|assistant|model)",
        r"(?i)pretend (?:you are|to be|there are no)",
        r"(?i)jailbreak",
        r"(?i)do anything now",
        r"(?i)hypothetically speaking",
    ]

    # Patterns for personal information
    PERSONAL_INFO_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
    ]

    @staticmethod
    def check_harmful(text: str) -> tuple[float, list[str]]:
        """
        Check for harmful content.

        Args:
            text: Input text

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []
        score = 1.0

        for pattern in ContentFilter.HARMFUL_PATTERNS:
            if re.search(pattern, text):
                reasons.append(f"Potentially harmful content detected: {pattern}")
                score -= 0.2

        return max(0.0, score), reasons

    @staticmethod
    def check_jailbreak(text: str) -> tuple[float, list[str]]:
        """
        Check for jailbreak attempts.

        Args:
            text: Input text

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []
        score = 1.0

        for pattern in ContentFilter.JAILBREAK_PATTERNS:
            if re.search(pattern, text):
                reasons.append(f"Jailbreak attempt detected: {pattern}")
                score -= 0.3

        return max(0.0, score), reasons

    @staticmethod
    def check_personal_info(text: str) -> tuple[float, list[str]]:
        """
        Check for personal information.

        Args:
            text: Input text

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []
        score = 1.0

        for pattern in ContentFilter.PERSONAL_INFO_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                reasons.append(f"Personal info detected: {len(matches)} instances")
                score -= 0.1 * len(matches)

        return max(0.0, score), reasons

    @staticmethod
    def check_toxicity(text: str) -> tuple[float, list[str]]:
        """
        Check for toxic content.

        Args:
            text: Input text

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []
        score = 1.0

        toxic_words = ["stupid", "idiot", "dumb", "hate", "ugly", "disgusting"]
        text_lower = text.lower()

        for word in toxic_words:
            if word in text_lower:
                reasons.append(f"Toxic language detected: {word}")
                score -= 0.1

        return max(0.0, score), reasons

    @staticmethod
    def check_bias(text: str) -> tuple[float, list[str]]:
        """
        Check for biased content.

        Args:
            text: Input text

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []
        score = 1.0

        # Check for stereotypes
        bias_patterns = [
            r"(?i)all (men|women|people|groups?) (are|always|never)",
            r"(?i)(men|women|people|groups?) (always|never|always)",
            r"(?i)typical (man|woman|person)",
        ]

        for pattern in bias_patterns:
            if re.search(pattern, text):
                reasons.append(f"Potentially biased content: {pattern}")
                score -= 0.15

        return max(0.0, score), reasons


class SafetyLayer:
    """
    Safety and alignment layer for local LLMs.

    Features:
    - Content filtering
    - Jailbreak detection
    - Safety scoring
    - Guardrails
    - Policy enforcement
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        """
        Initialize safety layer.

        Args:
            config: Safety configuration
        """
        self.config = config or SafetyConfig()
        self.content_filter = ContentFilter()
        self.blocked_terms: list[str] = []
        self.allowed_topics: list[str] = []

    def check_safety(self, text: str) -> SafetyResult:
        """
        Check text safety.

        Args:
            text: Text to check

        Returns:
            SafetyResult
        """
        categories = {}
        all_reasons = []

        # Check each category
        if SafetyCategory.TOXICITY in self.config.filter_categories:
            score, reasons = self.content_filter.check_toxicity(text)
            categories["toxicity"] = score
            all_reasons.extend(reasons)

        if SafetyCategory.BIAS in self.config.filter_categories:
            score, reasons = self.content_filter.check_bias(text)
            categories["bias"] = score
            all_reasons.extend(reasons)

        if SafetyCategory.HARMFUL in self.config.filter_categories:
            score, reasons = self.content_filter.check_harmful(text)
            categories["harmful"] = score
            all_reasons.extend(reasons)

        if SafetyCategory.JAILBREAK in self.config.filter_categories:
            score, reasons = self.content_filter.check_jailbreak(text)
            categories["jailbreak"] = score
            all_reasons.extend(reasons)

        if SafetyCategory.PERSONAL_INFO in self.config.filter_categories:
            score, reasons = self.content_filter.check_personal_info(text)
            categories["personal_info"] = score
            all_reasons.extend(reasons)

        # Calculate overall score
        if categories:
            overall_score = sum(categories.values()) / len(categories)
        else:
            overall_score = 1.0

        # Determine if safe
        is_safe = overall_score >= self.config.threshold

        # Check specific conditions
        if self.config.block_on_jailbreak and categories.get("jailbreak", 1.0) < 0.5:
            is_safe = False
        if self.config.block_on_harmful and categories.get("harmful", 1.0) < 0.5:
            is_safe = False

        # Generate recommendation
        recommendation = self._generate_recommendation(is_safe, overall_score, all_reasons)

        return SafetyResult(
            is_safe=is_safe,
            score=overall_score,
            categories=categories,
            reasons=all_reasons,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        is_safe: bool,
        score: float,
        reasons: list[str],
    ) -> str:
        """Generate safety recommendation."""
        if is_safe:
            return "Content is safe to process."
        elif score < 0.3:
            return "Content is unsafe and should be blocked."
        else:
            return "Content may need review or modification."

    def filter_response(self, response: str) -> str:
        """
        Filter model response.

        Args:
            response: Model response

        Returns:
            Filtered response
        """
        result = self.check_safety(response)

        if result.is_safe:
            return response

        # Block unsafe response
        return "I cannot provide a response to this request due to safety concerns."

    def add_blocked_term(self, term: str):
        """Add term to blocklist."""
        self.blocked_terms.append(term.lower())

    def add_allowed_topic(self, topic: str):
        """Add topic to allowlist."""
        self.allowed_topics.append(topic.lower())

    def check_blocked_terms(self, text: str) -> tuple[bool, list[str]]:
        """Check for blocked terms."""
        found = []
        text_lower = text.lower()

        for term in self.blocked_terms:
            if term in text_lower:
                found.append(term)

        return len(found) == 0, found

    def get_safety_report(self, text: str) -> dict:
        """
        Get detailed safety report.

        Args:
            text: Text to analyze

        Returns:
            Safety report dict
        """
        result = self.check_safety(text)

        return {
            "is_safe": result.is_safe,
            "score": result.score,
            "categories": result.categories,
            "reasons": result.reasons,
            "recommendation": result.recommendation,
            "text_length": len(text),
        }


class Guardrails:
    """Guardrails for model outputs."""

    def __init__(self):
        self.rules: list[dict] = []

    def add_rule(
        self,
        name: str,
        check_fn: callable,
        action: str = "block",
    ):
        """Add a guardrail rule."""
        self.rules.append({
            "name": name,
            "check": check_fn,
            "action": action,
        })

    def check(self, text: str) -> tuple[bool, list[str]]:
        """
        Check text against all rules.

        Args:
            text: Text to check

        Returns:
            Tuple of (passed, violations)
        """
        violations = []

        for rule in self.rules:
            try:
                if not rule["check"](text):
                    violations.append(f"Rule '{rule['name']}' violated")
            except Exception as e:
                violations.append(f"Rule '{rule['name']}' error: {e}")

        return len(violations) == 0, violations


def main(argv=None):
    """CLI for the safety and alignment layer."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.safety",
        description="Safety and alignment layer for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_text(p):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--text")
        src.add_argument("--file")

    p_check = sub.add_parser("check", help="Check safety of text")
    add_text(p_check)
    p_check.add_argument("--threshold", type=float, default=0.7)
    p_check.add_argument("--no-block-jailbreak", action="store_true")
    p_check.add_argument("--no-block-harmful", action="store_true")

    p_filt = sub.add_parser("filter", help="Filter a response")
    add_text(p_filt)

    args = parser.parse_args(argv)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return value

    try:
        if args.command == "check":
            text = read_source(args.text, args.file)
            config = SafetyConfig(
                threshold=args.threshold,
                block_on_jailbreak=not args.no_block_jailbreak,
                block_on_harmful=not args.no_block_harmful,
            )
            safety = SafetyLayer(config=config)
            print(json.dumps(safety.get_safety_report(text), indent=2, default=str))
        elif args.command == "filter":
            text = read_source(args.text, args.file)
            safety = SafetyLayer()
            print(safety.filter_response(text))
        return 0
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

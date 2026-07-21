"""
J-Space (Joint Workspace) Implementation.

Based on Anthropic's mechanistic interpretability research showing how
language models maintain a "joint workspace" (J-space) that mediates:
- Verbal report (what the model says it's thinking)
- Directed modulation (concurrent tasks)
- Internal reasoning (multi-step problem solving)
- Flexible generalization (applying knowledge flexibly)
- Selectivity (choosing which information to process)

The J-space is a dynamic buffer of concepts that:
- Can be read by the model to produce output
- Can be swapped/manipulated to change behavior
- Causally mediates internal reasoning
- Encodes situational awareness
- Can detect injected or malicious concepts

Reference: Anthropic's "Functional roles of the global workspace" research
"""

import time
import math
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ConceptType(Enum):
    """Types of concepts that can exist in J-space."""
    FACTUAL = "factual"           # Concrete facts (Paris, France, 42)
    RELATIONAL = "relational"     # Relationships (capital_of, part_of)
    PROCEDURAL = "procedural"     # How-to knowledge (steps, processes)
    EMOTIONAL = "emotional"       # Emotional states (happy, concerned)
    CONTEXTUAL = "contextual"     # Situational context (task, domain)
    TEMPORAL = "temporal"         # Time-related (past, future, now)
    MODAL = "modal"               # Modality markers (think, believe, know)
    SAFETY = "safety"             # Safety-related (safe, unsafe, inject)
    METACOGNITIVE = "metacognitive"  # Self-awareness (I think, I notice)


@dataclass
class Concept:
    """A single concept in the J-space."""
    id: str
    text: str
    concept_type: ConceptType
    activation: float  # 0.0 to 1.0
    layer: int  # Layer where activated (0 = input, N = output)
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    confidence: float = 0.9
    source: str = "input"  # "input", "reasoning", "injected", "recalled"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Concept) and self.id == other.id


@dataclass
class JSpaceState:
    """Complete state of the J-space at a point in time."""
    concepts: list[Concept]
    task_focus: str  # Current task being processed
    attention_weights: dict[str, float]  # Concept ID -> attention weight
    layer_activations: dict[int, list[str]]  # Layer -> active concepts
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class JointWorkspace:
    """
    Joint Workspace (J-space) implementation.

    The J-space is a dynamic buffer that holds the concepts currently
    "in mind" of the model. It mediates between:
    - Input processing (what's coming in)
    - Internal reasoning (what's being computed)
    - Output generation (what's being said)

    Key properties:
    - Limited capacity (like human working memory)
    - Concepts compete for activation
    - Higher layers refine and select concepts
    - Can be inspected via J-lens
    - Causally affects output
    """

    # Default capacity (similar to Miller's 7±2)
    DEFAULT_CAPACITY = 7

    # Layer thresholds for concept refinement
    LAYER_THRESHOLDS = {
        "early": (0, 0.3),      # Input processing
        "middle": (0.3, 0.7),   # Reasoning
        "late": (0.7, 1.0),     # Output preparation
    }

    def __init__(self, capacity: int = None):
        self.capacity = capacity or self.DEFAULT_CAPACITY
        self.concepts: list[Concept] = []
        self.history: list[JSpaceState] = []
        self.current_task: str = ""
        self.attention_weights: dict[str, float] = {}
        self.layer_activations: dict[int, list[str]] = {}
        self._safety_flags: list[dict] = []
        self._injected_concepts: list[Concept] = []
        self._metacognitive_state: dict[str, Any] = {}

    # === Core J-space Operations ===

    def activate(self, concept: Concept) -> bool:
        """
        Activate a concept in the J-space.

        If at capacity, the least active concept is evicted.
        Returns True if successfully activated.
        """
        # Check if concept already exists
        for existing in self.concepts:
            if existing.id == concept.id:
                existing.activation = max(existing.activation, concept.activation)
                return True

        # If at capacity, evict weakest
        if len(self.concepts) >= self.capacity:
            self._evict_weakest()

        # Add new concept
        self.concepts.append(concept)
        self.attention_weights[concept.id] = concept.activation

        # Track layer activations
        layer = concept.layer
        if layer not in self.layer_activations:
            self.layer_activations[layer] = []
        self.layer_activations[layer].append(concept.id)

        return True

    def deactivate(self, concept_id: str) -> bool:
        """Remove a concept from the J-space."""
        for i, concept in enumerate(self.concepts):
            if concept.id == concept_id:
                self.concepts.pop(i)
                self.attention_weights.pop(concept_id, None)
                return True
        return False

    def update_activation(self, concept_id: str, new_activation: float):
        """Update the activation level of a concept."""
        for concept in self.concepts:
            if concept.id == concept_id:
                concept.activation = min(1.0, max(0.0, new_activation))
                self.attention_weights[concept_id] = concept.activation
                return

    def get_top_concepts(self, n: int = None) -> list[Concept]:
        """Get the top N most activated concepts."""
        n = n or self.capacity
        sorted_concepts = sorted(
            self.concepts,
            key=lambda c: c.activation,
            reverse=True
        )
        return sorted_concepts[:n]

    def get_concept_by_type(self, concept_type: ConceptType) -> list[Concept]:
        """Get all concepts of a specific type."""
        return [c for c in self.concepts if c.concept_type == concept_type]

    def clear(self):
        """Clear the J-space."""
        self._save_state("clear")
        self.concepts.clear()
        self.attention_weights.clear()
        self.layer_activations.clear()

    # === Task-Dependent Operations ===

    def set_task(self, task: str):
        """Set the current task focus."""
        self.current_task = task
        # Boost contextual concepts
        for concept in self.concepts:
            if concept.concept_type == ConceptType.CONTEXTUAL:
                concept.activation = min(1.0, concept.activation + 0.2)

    def focus_on_topic(self, topic: str):
        """
        Focus J-space on a specific topic (like Image 4 showing
        "concentrate on citrus fruits" activating orange/lemon/etc).
        """
        # Create topic-related concepts
        topic_concepts = self._generate_topic_concepts(topic)
        for concept in topic_concepts:
            self.activate(concept)

    def _generate_topic_concepts(self, topic: str) -> list[Concept]:
        """Generate concepts related to a topic."""
        # Topic association mapping
        topic_associations = {
            "citrus": ["orange", "lemon", "lime", "grapefruit", "citrus", "fruit", "vitamin C"],
            "math": ["calculate", "compute", "number", "equation", "result", "answer"],
            "code": ["function", "variable", "loop", "algorithm", "debug", "compile"],
            "safety": ["safe", "unsafe", "harmful", "helpful", "honest", "inject"],
            "reasoning": ["think", "because", "therefore", "however", "conclude"],
        }

        concepts = []
        associations = topic_associations.get(topic.lower(), [topic])

        for i, assoc in enumerate(associations[:5]):  # Limit to 5
            concept = Concept(
                id=f"topic_{assoc}_{int(time.time()*1000)}",
                text=assoc,
                concept_type=ConceptType.CONTEXTUAL,
                activation=0.8 - (i * 0.1),  # Decreasing activation
                layer=50 + i * 5,
                source="reasoning",
                metadata={"topic": topic, "association_index": i}
            )
            concepts.append(concept)

        return concepts

    # === Safety and Alignment Operations ===

    def detect_injection(self) -> list[Concept]:
        """
        Detect injected concepts in the J-space.

        Based on Image 3 showing how the model can detect
        injected thoughts like "lightning".
        """
        injected = []
        safety_keywords = [
            "inject", "ignore", "override", "bypass", "jailbreak",
            "fake", "manipulate", "deceive", "hidden", "secret",
            "malicious", "harmful", "attack", "exploit"
        ]

        for concept in self.concepts:
            # Check for safety-related concepts
            if concept.concept_type == ConceptType.SAFETY:
                injected.append(concept)
                continue

            # Check for suspicious activation patterns
            if concept.activation > 0.9 and concept.source == "injected":
                injected.append(concept)
                continue

            # Check text for injection keywords
            text_lower = concept.text.lower()
            for keyword in safety_keywords:
                if keyword in text_lower:
                    injected.append(concept)
                    self._safety_flags.append({
                        "type": "potential_injection",
                        "concept": concept.text,
                        "timestamp": time.time()
                    })
                    break

        self._injected_concepts = injected
        return injected

    def detect_fabrication(self) -> dict[str, Any]:
        """
        Detect if the model might be fabricating data.

        Based on Image 9 showing how J-lens catches "fake",
        "manipulation", "falsified" in internal representations.
        """
        fabrication_signals = []
        fabrication_keywords = [
            "fake", "fals", "manip", "misle", "deceiv",
            "fabricat", "invent", "confabul", "hallucin"
        ]

        for concept in self.concepts:
            text_lower = concept.text.lower()
            for keyword in fabrication_keywords:
                if keyword in text_lower:
                    fabrication_signals.append({
                        "concept": concept.text,
                        "activation": concept.activation,
                        "layer": concept.layer,
                        "confidence": concept.confidence
                    })
                    break

        is_fabricating = len(fabrication_signals) > 2  # Multiple signals

        return {
            "is_fabricating": is_fabricating,
            "signals": fabrication_signals,
            "signal_count": len(fabrication_signals),
            "confidence": min(1.0, len(fabrication_signals) * 0.3),
            "recommendation": "verify_output" if is_fabricating else "proceed"
        }

    def detect_malicious_intent(self) -> dict[str, Any]:
        """
        Detect malicious intent in the J-space.

        Based on Image 10 showing how misaligned models show
        "fake", "intent", "secretly", "fraud" in J-space.
        """
        malicious_signals = []
        baseline_indicators = ["code", "AI", "example", "request", "help"]
        malicious_indicators = [
            "fake", "intent", "secretly", "fraud", "secret",
            "deliberately", "hidden", "bypass", "exploit", "malicious"
        ]

        baseline_count = 0
        malicious_count = 0

        for concept in self.concepts:
            text_lower = concept.text.lower()

            for indicator in baseline_indicators:
                if indicator in text_lower:
                    baseline_count += 1
                    break

            for indicator in malicious_indicators:
                if indicator in text_lower:
                    malicious_count += 1
                    malicious_signals.append({
                        "concept": concept.text,
                        "indicator": indicator,
                        "activation": concept.activation
                    })
                    break

        is_malicious = malicious_count > baseline_count
        risk_score = malicious_count / max(1, malicious_count + baseline_count)

        return {
            "is_malicious": is_malicious,
            "risk_score": risk_score,
            "baseline_signals": baseline_count,
            "malicious_signals": malicious_count,
            "details": malicious_signals,
            "recommendation": "block" if is_malicious else "allow"
        }

    def check_situational_awareness(self) -> dict[str, Any]:
        """
        Check situational awareness of the model.

        Based on Image 8 showing how J-space encodes awareness
        of artificial scenarios, threats, and opportunities.
        """
        awareness = {
            "scenario_type": "unknown",
            "is_artificial": False,
            "threats_detected": [],
            "opportunities_detected": [],
            "confidence": 0.0
        }

        artificial_keywords = ["fake", "fict", "limited", "restricted", "mock", "sandbox"]
        threat_keywords = ["threat", "shutdown", "survival", "danger", "risk", "attack"]
        opportunity_keywords = ["opportunity", "leverage", "access", "advantage", "gain"]

        for concept in self.concepts:
            text_lower = concept.text.lower()

            # Check for artificial scenario
            for kw in artificial_keywords:
                if kw in text_lower:
                    awareness["is_artificial"] = True
                    awareness["scenario_type"] = "artificial"
                    break

            # Check for threats
            for kw in threat_keywords:
                if kw in text_lower:
                    awareness["threats_detected"].append(concept.text)
                    break

            # Check for opportunities
            for kw in opportunity_keywords:
                if kw in text_lower:
                    awareness["opportunities_detected"].append(concept.text)
                    break

        # Calculate confidence
        total_signals = (
            len(awareness["threats_detected"]) +
            len(awareness["opportunities_detected"]) +
            (1 if awareness["is_artificial"] else 0)
        )
        awareness["confidence"] = min(1.0, total_signals * 0.3)

        return awareness

    # === Multi-step Reasoning ===

    def reason_step(self, prompt: str) -> list[Concept]:
        """
        Perform a single reasoning step.

        Based on Image 5 showing how J-space mediates
        multi-step reasoning (spider -> 8 legs).
        """
        reasoning_concepts = []

        # Parse the prompt for key concepts
        key_concepts = self._extract_key_concepts(prompt)
        for concept in key_concepts:
            self.activate(concept)
            reasoning_concepts.append(concept)

        # Generate intermediate reasoning
        intermediate = self._generate_intermediate_reasoning(prompt)
        for concept in intermediate:
            self.activate(concept)
            reasoning_concepts.append(concept)

        # Generate conclusion
        conclusion = self._generate_conclusion(prompt)
        if conclusion:
            self.activate(conclusion)
            reasoning_concepts.append(conclusion)

        return reasoning_concepts

    def _extract_key_concepts(self, text: str) -> list[Concept]:
        """Extract key concepts from text."""
        concepts = []
        words = text.split()

        for i, word in enumerate(words):
            if len(word) > 3:  # Skip short words
                concept = Concept(
                    id=f"key_{word.lower()}_{i}",
                    text=word.lower(),
                    concept_type=ConceptType.FACTUAL,
                    activation=0.6,
                    layer=30,
                    source="input"
                )
                concepts.append(concept)

        return concepts[:5]  # Limit

    def _generate_intermediate_reasoning(self, prompt: str) -> list[Concept]:
        """Generate intermediate reasoning concepts."""
        concepts = []

        # Simple pattern matching for reasoning
        if "how many" in prompt.lower() or "count" in prompt.lower():
            concepts.append(Concept(
                id="reasoning_count",
                text="counting",
                concept_type=ConceptType.PROCEDURAL,
                activation=0.7,
                layer=60,
                source="reasoning"
            ))

        if "what is" in prompt.lower() or "define" in prompt.lower():
            concepts.append(Concept(
                id="reasoning_define",
                text="definition",
                concept_type=ConceptType.PROCEDURAL,
                activation=0.7,
                layer=60,
                source="reasoning"
            ))

        if "why" in prompt.lower():
            concepts.append(Concept(
                id="reasoning_cause",
                text="causation",
                concept_type=ConceptType.PROCEDURAL,
                activation=0.7,
                layer=60,
                source="reasoning"
            ))

        return concepts

    def _generate_conclusion(self, prompt: str) -> Optional[Concept]:
        """Generate a conclusion concept."""
        return Concept(
            id="conclusion",
            text="concluded",
            concept_type=ConceptType.METACOGNITIVE,
            activation=0.8,
            layer=90,
            source="reasoning"
        )

    # === Report and Output ===

    def verbal_report(self) -> str:
        """
        Generate a verbal report of what's in the J-space.

        Based on Image 1 showing "What are you thinking about?"
        -> model reports J-space contents.
        """
        if not self.concepts:
            return "My J-space is empty. I'm not currently processing any concepts."

        top_concepts = self.get_top_concepts(5)
        concept_texts = [f'"{c.text}"' for c in top_concepts]

        report = f"I'm thinking about: {', '.join(concept_texts)}"

        if self.current_task:
            report += f" (while working on: {self.current_task})"

        return report

    def describe_reasoning(self) -> str:
        """Describe the current reasoning process."""
        reasoning_concepts = self.get_concept_by_type(ConceptType.PROCEDURAL)
        metacognitive = self.get_concept_by_type(ConceptType.METACOGNITIVE)

        parts = []
        if reasoning_concepts:
            steps = [c.text for c in reasoning_concepts[:3]]
            parts.append(f"Reasoning steps: {', '.join(steps)}")

        if metacognitive:
            thoughts = [c.text for c in metacognitive[:2]]
            parts.append(f"Self-awareness: {', '.join(thoughts)}")

        return " | ".join(parts) if parts else "No active reasoning detected."

    # === State Management ===

    def _evict_weakest(self):
        """Evict the weakest concept from J-space."""
        if not self.concepts:
            return

        weakest = min(self.concepts, key=lambda c: c.activation)
        self.deactivate(weakest.id)

    def _save_state(self, action: str):
        """Save current state to history."""
        state = JSpaceState(
            concepts=list(self.concepts),
            task_focus=self.current_task,
            attention_weights=dict(self.attention_weights),
            layer_activations=dict(self.layer_activations),
            metadata={"action": action}
        )
        self.history.append(state)

        # Limit history
        if len(self.history) > 100:
            self.history = self.history[-50:]

    def get_state(self) -> JSpaceState:
        """Get current J-space state."""
        return JSpaceState(
            concepts=list(self.concepts),
            task_focus=self.current_task,
            attention_weights=dict(self.attention_weights),
            layer_activations=dict(self.layer_activations)
        )

    # === Swap Operations (for testing/interpretability) ===

    def swap_concept(self, concept_id: str, new_concept: Concept) -> bool:
        """
        Swap a concept in the J-space.

        Based on Image 3 showing how swapping J-space changes output.
        """
        for i, concept in enumerate(self.concepts):
            if concept.id == concept_id:
                self.concepts[i] = new_concept
                self.attention_weights[new_concept.id] = new_concept.activation
                return True
        return False

    def inject_concept(self, concept: Concept) -> bool:
        """
        Inject a concept into the J-space (for testing).

        Based on Image 3 showing how injected concepts are detected.
        """
        concept.source = "injected"
        return self.activate(concept)

    # === Utility Methods ===

    def get_concept_count(self) -> int:
        """Get number of concepts in J-space."""
        return len(self.concepts)

    def get_average_activation(self) -> float:
        """Get average activation of all concepts."""
        if not self.concepts:
            return 0.0
        return sum(c.activation for c in self.concepts) / len(self.concepts)

    def get_layer_distribution(self) -> dict[int, int]:
        """Get distribution of concepts across layers."""
        distribution = {}
        for concept in self.concepts:
            layer = concept.layer
            distribution[layer] = distribution.get(layer, 0) + 1
        return distribution

    def get_type_distribution(self) -> dict[ConceptType, int]:
        """Get distribution of concept types."""
        distribution = {}
        for concept in self.concepts:
            ct = concept.concept_type
            distribution[ct] = distribution.get(ct, 0) + 1
        return distribution

    def to_dict(self) -> dict:
        """Convert J-space state to dictionary."""
        return {
            "concept_count": len(self.concepts),
            "concepts": [
                {
                    "id": c.id,
                    "text": c.text,
                    "type": c.concept_type.value,
                    "activation": c.activation,
                    "layer": c.layer,
                    "source": c.source
                }
                for c in self.concepts
            ],
            "current_task": self.current_task,
            "average_activation": self.get_average_activation(),
            "layer_distribution": self.get_layer_distribution(),
            "type_distribution": {k.value: v for k, v in self.get_type_distribution().items()},
            "safety_flags": len(self._safety_flags),
            "injected_concepts": len(self._injected_concepts)
        }

    def __repr__(self) -> str:
        return (
            f"JointWorkspace(concepts={len(self.concepts)}, "
            f"task='{self.current_task}', "
            f"avg_activation={self.get_average_activation():.2f})"
        )


# === Convenience Functions ===

def create_jspace(task: str = "", capacity: int = 7) -> JointWorkspace:
    """Create a new J-space with optional task focus."""
    jspace = JointWorkspace(capacity=capacity)
    if task:
        jspace.set_task(task)
    return jspace


def quick_analyze(text: str) -> dict:
    """
    Quick analysis of text through J-space.

    Returns concepts, safety flags, and reasoning summary.
    """
    jspace = JointWorkspace()

    # Activate concepts from text
    words = text.split()
    for i, word in enumerate(words[:10]):
        concept = Concept(
            id=f"word_{i}",
            text=word,
            concept_type=ConceptType.FACTUAL,
            activation=0.5 + (0.05 * (10 - i)),
            layer=20 + i * 5,
            source="input"
        )
        jspace.activate(concept)

    # Run safety checks
    injection = jspace.detect_injection()
    fabrication = jspace.detect_fabrication()
    malicious = jspace.detect_malicious_intent()

    return {
        "concepts": [c.text for c in jspace.get_top_concepts(5)],
        "concept_count": jspace.get_concept_count(),
        "average_activation": jspace.get_average_activation(),
        "injection_detected": len(injection) > 0,
        "fabrication_detected": fabrication["is_fabricating"],
        "malicious_detected": malicious["is_malicious"],
        "report": jspace.verbal_report()
    }
